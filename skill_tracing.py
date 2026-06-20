"""Export skill runs to LangSmith (and optionally OpenTelemetry) tracing.

The local ``skill_runs`` store is the source of truth for skill observability; this
module mirrors each recorded run (build / eval / refine / review) out to external
tracing so the skill-build loop shows up alongside the rest of your traces:

  * LangSmith — one run per skill operation, posted to a DEDICATED project
    (``LANGSMITH_SKILL_PROJECT``, default "BYO-WIKI Agent Skills") so it is isolated
    from the deepagents agent-layer project. The run's metrics (gate, rubric mean,
    deterministic ratio, triggering F1, tokens, latency, cost) are attached as
    metadata AND as numeric feedback so they chart in the LangSmith UI.
  * OpenTelemetry — an optional span per run, emitted when an OTLP endpoint and the
    OpenTelemetry SDK are present (e.g. exported to LangSmith's OTLP ingest).

Everything is best-effort: missing keys/SDKs or network errors degrade to a no-op
and never break a build. Enable with ``LANGSMITH_API_KEY`` plus ``SKILL_TRACING``
(or the existing ``LANGSMITH_TRACING``).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

_TRUTHY = {"1", "true", "yes", "on"}

# A dedicated project/"workspace" for the skill-generation scenario, kept separate
# from any LANGSMITH_PROJECT the agent layer traces into.
SKILL_PROJECT = os.environ.get("LANGSMITH_SKILL_PROJECT", "BYO-WIKI Agent Skills")

_client = None
_project_ready = False


def _tracing_toggle() -> bool:
    return (os.environ.get("SKILL_TRACING", "").lower() in _TRUTHY
            or os.environ.get("LANGSMITH_TRACING", "").lower() in _TRUTHY
            or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in _TRUTHY)


def langsmith_enabled() -> bool:
    """True when skill runs should be exported to LangSmith."""
    has_key = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    return has_key and _tracing_toggle()


def otel_enabled() -> bool:
    """True when an OTLP endpoint is configured (the SDK is checked lazily)."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")) and _tracing_toggle()


def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from langsmith import Client
        _client = Client()
    except Exception:  # noqa: BLE001
        _client = None
    return _client


def ensure_project() -> dict:
    """Create the dedicated LangSmith project if it does not exist (idempotent)."""
    global _project_ready
    if not langsmith_enabled():
        return {"ok": False, "reason": "LangSmith not configured (need LANGSMITH_API_KEY + SKILL_TRACING)"}
    client = get_client()
    if client is None:
        return {"ok": False, "reason": "langsmith SDK not installed"}
    try:
        try:
            client.read_project(project_name=SKILL_PROJECT)
            created = False
        except Exception:  # noqa: BLE001  (not found) -> create
            client.create_project(
                SKILL_PROJECT,
                description="BYO-WIKI agent-skill build / eval / refine / review runs "
                            "(skill auto-generation subsystem, layer 7).")
            created = True
        _project_ready = True
        return {"ok": True, "project": SKILL_PROJECT, "created": created}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "project": SKILL_PROJECT, "reason": str(exc)}


# --- payload building (pure; unit-tested) ------------------------------------
def build_run_record(run: dict) -> dict:
    """Shape a local skill-run dict into LangSmith run fields + numeric feedback."""
    kind = run.get("kind", "build")
    metrics = run.get("metrics") or {}
    end = datetime.now(timezone.utc)
    start = end - timedelta(milliseconds=int(run.get("duration_ms") or 0))
    inputs = {
        "kind": kind, "skill_name": run.get("skill_name"), "backend": run.get("provider"),
        "model": run.get("model"),
    }
    outputs = {
        "gate": run.get("gate"), "status": run.get("status"), "ok": run.get("ok", True),
        **{k: v for k, v in metrics.items() if v is not None},
    }
    if run.get("error"):
        outputs["error"] = run["error"]
    metadata = {
        "skill_id": run.get("skill_id"), "tokens": run.get("tokens"),
        "tools_used": run.get("tools_used"), "duration_ms": run.get("duration_ms"),
        "phases": run.get("phases"), "run_id_local": run.get("id"),
    }
    # Numeric feedback so the metrics chart in LangSmith.
    feedback = {}
    if run.get("gate") is not None:
        feedback["gate_pass"] = 1.0 if run.get("gate") == "accept" else 0.0
    for key in ("deterministic_ratio", "rubric_mean", "trigger_f1", "trigger_precision",
                "trigger_recall"):
        if isinstance(metrics.get(key), (int, float)):
            feedback[key] = float(metrics[key])
    if isinstance(run.get("tokens"), (int, float)):
        feedback["tokens"] = float(run["tokens"])
    if isinstance(run.get("duration_ms"), (int, float)):
        feedback["duration_ms"] = float(run["duration_ms"])
    return {
        "name": f"skill.{kind}",
        "run_type": "chain",
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata,
        "tags": ["byo-wiki", "agent-skill", kind, str(run.get("provider") or "")],
        "start_time": start,
        "end_time": end,
        "feedback": feedback,
    }


# Order the build phases appear under the parent run.
_PHASE_ORDER = (("understand", "chain"), ("analyze", "chain"), ("codeact", "llm"),
                ("deterministic", "chain"), ("rubric", "llm"), ("triggering", "chain"))


def build_tree_plan(run: dict, phase_detail: dict) -> dict:
    """Plan a parent run + one child run per phase (pure; unit-tested).

    Children are laid end-to-end inside the parent's time window so the trace shows
    a real timeline; each carries that phase's tokens / tool-calls / outputs."""
    base = build_run_record(run)
    parent = {
        "name": base["name"], "run_type": "chain", "inputs": base["inputs"],
        "outputs": base["outputs"], "metadata": base["metadata"], "tags": base["tags"],
        "start_time": base["start_time"], "end_time": base["end_time"],
        "feedback": base["feedback"],
    }
    cursor = base["start_time"]
    end_cap = base["end_time"]
    children = []
    for name, default_type in _PHASE_ORDER:
        d = phase_detail.get(name)
        if not d:
            continue
        ms = int(d.get("ms") or 1)
        cstart = min(cursor, end_cap)
        cend = min(cstart + timedelta(milliseconds=ms), end_cap)  # never exceed the parent window
        cursor = cend
        children.append({
            "name": f"skill.{name}",
            "run_type": d.get("run_type", default_type),
            "inputs": d.get("inputs") or {},
            "outputs": d.get("outputs") or {},
            "metadata": {k: d[k] for k in ("tokens", "tool_calls", "backend") if d.get(k) is not None},
            "start_time": cstart, "end_time": cend,
        })
    return {"parent": parent, "children": children}


# --- exporters ---------------------------------------------------------------
def _export_langsmith(rec: dict) -> bool:
    global _project_ready
    client = get_client()
    if client is None:
        return False
    if not _project_ready:
        ensure_project()
    run_id = uuid.uuid4()
    client.create_run(
        id=run_id, name=rec["name"], run_type=rec["run_type"], inputs=rec["inputs"],
        start_time=rec["start_time"], project_name=SKILL_PROJECT, tags=rec["tags"],
        extra={"metadata": rec["metadata"]})
    client.update_run(run_id, outputs=rec["outputs"], end_time=rec["end_time"])
    for key, score in (rec.get("feedback") or {}).items():
        try:
            client.create_feedback(run_id, key=key, score=score)
        except Exception:  # noqa: BLE001
            pass
    return True


_otel_tracer = None


def _otel():
    global _otel_tracer
    if _otel_tracer is not None:
        return _otel_tracer
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider = TracerProvider(resource=Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", "byo-wiki-skills")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        _otel_tracer = provider.get_tracer("byo-wiki.skills")
    except Exception:  # noqa: BLE001  (SDK/exporter not installed)
        _otel_tracer = None
    return _otel_tracer


def _export_otel(rec: dict) -> bool:
    tracer = _otel()
    if tracer is None:
        return False
    try:
        span = tracer.start_span(rec["name"])
        span.set_attribute("skill.kind", rec["inputs"].get("kind") or "")
        span.set_attribute("skill.backend", rec["inputs"].get("backend") or "")
        span.set_attribute("skill.gate", str(rec["outputs"].get("gate")))
        for key, score in (rec.get("feedback") or {}).items():
            span.set_attribute(f"skill.{key}", score)
        span.end()
        return True
    except Exception:  # noqa: BLE001
        return False


def export(run: dict) -> dict:
    """Mirror one local skill run to LangSmith and/or OTel. Best-effort, never raises."""
    out = {"langsmith": False, "otel": False}
    try:
        rec = build_run_record(run)
    except Exception:  # noqa: BLE001
        return out
    if langsmith_enabled():
        try:
            out["langsmith"] = _export_langsmith(rec)
        except Exception:  # noqa: BLE001
            out["langsmith"] = False
    if otel_enabled():
        try:
            out["otel"] = _export_otel(rec)
        except Exception:  # noqa: BLE001
            out["otel"] = False
    return out


def export_tree(run: dict, phase_detail: dict) -> dict:
    """Post a parent ``skill.build`` run with the per-phase calls as CHILD runs, so
    the trace is a full tree instead of one flat run. Falls back to a flat run on any
    error so a build is never lost. Best-effort; never raises."""
    out = {"langsmith": False, "otel": False, "tree": False}
    if not (langsmith_enabled() or otel_enabled()):
        return out
    if not langsmith_enabled():
        out["otel"] = _safe_otel(run)
        return out
    try:
        from langsmith.run_trees import RunTree
    except Exception:  # noqa: BLE001  (SDK missing) -> flat
        return export(run)
    if not _project_ready:
        ensure_project()
    try:
        plan = build_tree_plan(run, phase_detail)
        p = plan["parent"]
        parent = RunTree(name=p["name"], run_type="chain", inputs=p["inputs"],
                         project_name=SKILL_PROJECT, tags=p["tags"],
                         extra={"metadata": p["metadata"]}, start_time=p["start_time"])
        parent.post()
        for ch in plan["children"]:
            child = parent.create_child(name=ch["name"], run_type=ch["run_type"], inputs=ch["inputs"])
            child.start_time = ch["start_time"]
            if ch["metadata"]:
                child.extra = {"metadata": ch["metadata"]}
            child.post()
            child.end(outputs=ch["outputs"], end_time=ch["end_time"])
            child.patch()
        parent.end(outputs=p["outputs"], end_time=p["end_time"])
        parent.patch()
        client = get_client()
        if client is not None:
            for key, score in (p["feedback"] or {}).items():
                try:
                    client.create_feedback(parent.id, key=key, score=score)
                except Exception:  # noqa: BLE001
                    pass
        out.update({"langsmith": True, "tree": True, "children": len(plan["children"])})
    except Exception:  # noqa: BLE001  (RunTree quirks) -> don't lose the run
        return export(run)
    out["otel"] = _safe_otel(run)
    return out


def _safe_otel(run: dict) -> bool:
    if not otel_enabled():
        return False
    try:
        return _export_otel(build_run_record(run))
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    return {
        "langsmith_enabled": langsmith_enabled(),
        "otel_enabled": otel_enabled(),
        "project": SKILL_PROJECT,
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "otlp_endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    }
