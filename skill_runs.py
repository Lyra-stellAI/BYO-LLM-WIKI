"""Skill observability — a trace/metrics store for the skill-build loop.

Self-improvement is only credible if it is *measured*. This module is the
observability infrastructure for layer 7: every build, eval, refine, and
human-review is logged as a **run** with concrete metrics — per-phase timings,
token usage, the eval gate decision, deterministic pass-rate, rubric mean, and
triggering precision/recall. Aggregating those runs gives a **benchmark** readout
(pass rate, average latency, average/total tokens, gate distribution) so an author
can watch a skill — and the pipeline itself — improve across model updates and
iterations, in the spirit of Anthropic's skill-creator *benchmark mode* and
Hermes' *verify results / measure improvement* habit.

Stored as a single plain-JSON file (``data/skill_runs.json``), capped to the most
recent runs so it stays small and local-first like the other layers' stores.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_PATH = DATA_DIR / "skill_runs.json"

SCHEMA_VERSION = 1
# Keep the store bounded; older runs roll off (benchmark uses what remains).
MAX_RUNS = int(os.environ.get("SKILL_RUNS_MAX", "500"))

RUN_KINDS = {"build", "eval", "refine", "review", "test"}

_lock = RLock()
_cache: dict = {"mtime": None, "data": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict:
    return {"version": SCHEMA_VERSION, "created_at": _now(), "updated_at": _now(), "runs": []}


def _load() -> dict:
    if not RUNS_PATH.exists():
        return _empty()
    try:
        mtime = RUNS_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _cache["data"] is not None and _cache["mtime"] == mtime:
        return _cache["data"]
    try:
        data = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _empty()
    data.setdefault("version", SCHEMA_VERSION)
    data.setdefault("runs", [])
    _cache["data"], _cache["mtime"] = data, mtime
    return data


def _save(data: dict) -> None:
    data["updated_at"] = _now()
    if len(data["runs"]) > MAX_RUNS:
        data["runs"] = data["runs"][-MAX_RUNS:]
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RUNS_PATH)
    try:
        _cache["data"], _cache["mtime"] = data, RUNS_PATH.stat().st_mtime
    except OSError:
        _cache["data"], _cache["mtime"] = data, None


def record(*, kind: str, skill_id: str = "", skill_name: str = "", provider: str = "",
           model: str = "", gate: str | None = None, status: str | None = None,
           duration_ms: int | None = None, tokens: int | None = None,
           phases: dict | None = None, metrics: dict | None = None,
           tools_used: int | None = None, ok: bool = True, error: str = "",
           trace: bool = True) -> dict:
    """Append one observability run and return it. Never raises (best-effort).

    ``trace=False`` skips the flat LangSmith/OTel export — used when the caller will
    post a richer nested trace (a parent run with per-phase child runs) itself."""
    run = {
        "id": f"run_{uuid.uuid4().hex[:12]}",
        "kind": kind if kind in RUN_KINDS else "build",
        "skill_id": skill_id,
        "skill_name": skill_name,
        "provider": provider,
        "model": model,
        "gate": gate,
        "status": status,
        "duration_ms": duration_ms,
        "tokens": tokens,
        "tools_used": tools_used,
        "phases": phases or {},     # {phase: {ms, tokens}}
        "metrics": metrics or {},   # {deterministic_ratio, rubric_mean, trigger_precision, ...}
        "ok": bool(ok),
        "error": error or "",
        "at": _now(),
    }
    try:
        with _lock:
            data = _load()
            data["runs"].append(run)
            _save(data)
    except Exception:  # noqa: BLE001  (observability must never break a build)
        pass
    # Mirror to external tracing (LangSmith / OTel) outside the lock; best-effort.
    if trace:
        try:
            import skill_tracing
            skill_tracing.export(run)
        except Exception:  # noqa: BLE001
            pass
    return run


def list_runs(*, skill_id: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
    with _lock:
        runs = list(_load()["runs"])
    if skill_id:
        runs = [r for r in runs if r.get("skill_id") == skill_id]
    if kind:
        runs = [r for r in runs if r.get("kind") == kind]
    runs.sort(key=lambda r: r.get("at", ""), reverse=True)
    return runs[:max(1, limit)]


def get_run(run_id: str) -> dict | None:
    with _lock:
        for r in _load()["runs"]:
            if r.get("id") == run_id:
                return r
    return None


def _avg(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def benchmark(*, skill_id: str | None = None) -> dict:
    """Aggregate runs into a benchmark readout (the 'measure' of test/measure/refine)."""
    with _lock:
        runs = list(_load()["runs"])
    if skill_id:
        runs = [r for r in runs if r.get("skill_id") == skill_id]
    builds = [r for r in runs if r.get("kind") in ("build", "refine")]
    evals = [r for r in runs if r.get("kind") in ("build", "refine", "eval")]
    gates: dict[str, int] = {}
    for r in evals:
        g = r.get("gate")
        if g:
            gates[g] = gates.get(g, 0) + 1
    n_gated = sum(gates.values())
    passed = gates.get("accept", 0)
    return {
        "total_runs": len(runs),
        "builds": len(builds),
        "by_kind": {k: sum(1 for r in runs if r.get("kind") == k) for k in RUN_KINDS
                    if any(r.get("kind") == k for r in runs)},
        "gate_distribution": gates,
        # "pass rate" = share of evaluated drafts the gate accepted.
        "gate_pass_rate": round(passed / n_gated, 3) if n_gated else None,
        "avg_duration_ms": _avg([r.get("duration_ms") for r in builds]),
        "avg_tokens": _avg([r.get("tokens") for r in builds]),
        "total_tokens": int(sum(r.get("tokens") or 0 for r in runs)),
        "avg_rubric_mean": _avg([(r.get("metrics") or {}).get("rubric_mean") for r in evals]),
        "avg_deterministic_ratio": _avg([(r.get("metrics") or {}).get("deterministic_ratio") for r in evals]),
        "avg_trigger_f1": _avg([(r.get("metrics") or {}).get("trigger_f1") for r in evals]),
        "last_run_at": runs[-1].get("at") if runs else None,
    }


def clear() -> None:
    with _lock:
        _save(_empty())
