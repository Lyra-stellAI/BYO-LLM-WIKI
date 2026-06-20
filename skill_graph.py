"""LangGraph orchestration of the skill-build loop (incremental StateGraph).

The hand-rolled pipeline in ``skill_agent`` already works; this module wraps the
SAME phase functions as a LangGraph ``StateGraph`` to gain three things the linear
version cannot offer:

  * the **gate** becomes a conditional edge (reject → END; accept/review → review);
  * **human review** becomes a durable ``interrupt()`` — the build pauses, its state
    is checkpointed, and it resumes (even in a later process) when a decision
    arrives, instead of relying on a separate stored-state round-trip;
  * **refine** becomes a real cycle: a "revise" decision loops back to ``codeact``
    with the reviewer's notes, re-evaluates, and gates again.

Graph (entry → END)::

    gather ─▶ understand ─▶ analyze ─▶ codeact ─┐
          └▶ generate_cc ───────────────────────┴▶ evaluate ─▶(gate)
                                                              ├ reject ▶ END
                                                              └ else  ▶ human_review ─▶(decision)
                                                                          ├ accept ▶ finalize ▶ END
                                                                          ├ reject ▶ END
                                                                          └ revise ▶ prepare_revision ▶ codeact/generate_cc (cycle)

Nodes delegate to ``skill_agent`` / ``skill_eval`` / ``skill_library`` so behaviour
(and the observability + LangSmith tree export) matches the linear backend. State
is plain dicts/strings so it checkpoints cleanly. The checkpointer is SQLite by
default (``data/skill_graph.sqlite``), swappable to Postgres (e.g. your Supabase
``SUPABASE_DB_URL``) or in-memory via ``SKILL_GRAPH_CHECKPOINT``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

import skill_agent
import skill_eval
import skill_library as skills
import skill_runs

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
MAX_REVISIONS = int(os.environ.get("SKILL_GRAPH_MAX_REVISIONS", "4"))


class SkillState(TypedDict, total=False):
    # inputs
    chunk_ids: list
    text: str
    query: str
    tags: list
    where: str
    goal: str
    provider: str
    model: str
    backend: str
    use_tools: bool
    run_rubric: bool
    run_triggering: bool
    # working state
    bundle: dict
    understanding: dict
    spec: dict
    artifact: dict
    skill_id: str
    phases: dict
    tools_used: int
    report: dict
    gate: str
    status: str
    revision_note: str
    attempt: int
    wall_start: float
    # human-in-the-loop
    decision: str
    notes: str
    skill: dict
    done: bool


# --- nodes -------------------------------------------------------------------
def _node_gather(state: SkillState) -> dict:
    bundle = skill_agent.gather_context(
        chunk_ids=state.get("chunk_ids"), text=state.get("text"), query=state.get("query"),
        tags=state.get("tags"), where=state.get("where", "overall"))
    return {"bundle": bundle, "phases": {}, "attempt": state.get("attempt", 0),
            "wall_start": time.time()}


def _node_understand(state: SkillState) -> dict:
    chat, _rp, _rm = skill_agent._resolve_model(state.get("provider", "auto"), state.get("model"))
    meter = skill_agent._Meter()
    t0 = time.perf_counter()
    understanding = skill_agent.understand(state["bundle"], chat=chat, meter=meter)
    phases = dict(state.get("phases", {}))
    phases["understand"] = {"ms": int((time.perf_counter() - t0) * 1000), "tokens": meter.tokens}
    return {"understanding": understanding, "phases": phases}


def _node_analyze(state: SkillState) -> dict:
    chat, _rp, _rm = skill_agent._resolve_model(state.get("provider", "auto"), state.get("model"))
    meter = skill_agent._Meter()
    t0 = time.perf_counter()
    spec = skill_agent.analyze(state["bundle"], state.get("understanding", {}), chat=chat,
                               goal=state.get("goal", ""), meter=meter)
    phases = dict(state.get("phases", {}))
    phases["analyze"] = {"ms": int((time.perf_counter() - t0) * 1000), "tokens": meter.tokens}
    return {"spec": spec, "phases": phases}


def _persist_draft(artifact: dict, state: SkillState) -> dict:
    artifact = dict(artifact)
    artifact["provenance"] = state["bundle"]["provenance"]
    if state.get("revision_note"):
        artifact["revision_note"] = state["revision_note"]
    skill = skills.upsert_skill(artifact, where_id=state.get("skill_id"))
    if not skill:
        raise skill_agent.SkillError("Could not store the drafted skill.")
    skills.export_skill(skill["id"])
    return skill


def _node_codeact(state: SkillState) -> dict:
    chat, _rp, _rm = skill_agent._resolve_model(state.get("provider", "auto"), state.get("model"))
    meter = skill_agent._Meter()
    t0 = time.perf_counter()
    tools_used = 0
    tool_mode = bool(state.get("use_tools", True) and hasattr(chat, "bind_tools"))
    rn = state.get("revision_note", "")
    if tool_mode:
        artifact, tools_used = skill_agent.codeact_tooluse(state["bundle"], state["spec"],
                                                           chat=chat, revision_note=rn, meter=meter)
        if not (artifact.get("instructions") and artifact.get("name")):
            tool_mode = False
            artifact = skill_agent.codeact(state["bundle"], state["spec"], chat=chat,
                                           revision_note=rn, meter=meter)
    else:
        artifact = skill_agent.codeact(state["bundle"], state["spec"], chat=chat,
                                       revision_note=rn, meter=meter)
    if not (artifact.get("instructions") and artifact.get("name")):
        raise skill_agent.SkillError("The author phase did not produce a usable skill.")
    skill = _persist_draft(artifact, state)
    phases = dict(state.get("phases", {}))
    phases["codeact"] = {"ms": int((time.perf_counter() - t0) * 1000), "tokens": meter.tokens,
                         "tool_calls": tools_used, "tool_mode": tool_mode}
    return {"artifact": artifact, "skill_id": skill["id"], "tools_used": tools_used, "phases": phases}


def _node_generate_cc(state: SkillState) -> dict:
    import skill_claude_agent
    t0 = time.perf_counter()
    res = skill_claude_agent.generate(state["bundle"], goal=state.get("goal", ""),
                                      revision_note=state.get("revision_note", ""),
                                      model=state.get("model"), allow_tools=state.get("use_tools", True))
    ms = int((time.perf_counter() - t0) * 1000)
    meta = res["meta"]
    artifact = skill_agent._normalize_artifact(res["artifact"], {})
    if not (artifact.get("instructions") and artifact.get("name")):
        raise skill_agent.SkillError("Claude Code did not produce a usable skill.")
    skill = _persist_draft(artifact, state)
    phases = dict(state.get("phases", {}))
    phases["codeact"] = {"ms": ms, "tokens": meta.get("tokens"), "tool_calls": meta.get("num_turns"),
                         "backend": "claude_code"}
    return {"artifact": artifact, "skill_id": skill["id"], "tools_used": meta.get("num_turns") or 0,
            "phases": phases, "understanding": {"summary": meta.get("understanding")},
            "spec": {"name": artifact.get("name"), "rationale": meta.get("analysis")}}


def _node_evaluate(state: SkillState) -> dict:
    skill = skills.get_skill(state["skill_id"])
    backend = skill_agent._resolve_backend(state.get("backend"))
    t0 = time.perf_counter()
    report = skill_eval.run_eval(skill, provider="auto", model=None,
                                 run_rubric=state.get("run_rubric", True),
                                 run_triggering=state.get("run_triggering", True))
    phases = dict(state.get("phases", {}))
    phases["eval"] = {"ms": int((time.perf_counter() - t0) * 1000)}
    skill = skills.record_eval(state["skill_id"], report)

    kind = "build" if state.get("attempt", 0) == 0 else "refine"
    duration_ms = int((time.time() - state.get("wall_start", time.time())) * 1000)
    trig = report.get("triggering") or {}
    metrics = {"deterministic_ratio": report["deterministic"]["ratio"],
               "rubric_mean": report.get("rubric_mean"), "trigger_precision": trig.get("precision"),
               "trigger_recall": trig.get("recall"), "trigger_f1": trig.get("f1")}
    run = skill_runs.record(kind=kind, skill_id=state["skill_id"], skill_name=skill["name"],
                            provider=("claude_code" if backend == "claude_code" else "pipeline"),
                            model=state.get("model") or "", gate=report["gate"], status=skill["status"],
                            duration_ms=duration_ms, tokens=phases.get("codeact", {}).get("tokens"),
                            tools_used=state.get("tools_used", 0), phases=phases, metrics=metrics,
                            trace=False)
    try:
        import skill_tracing
        detail = skill_agent._phase_detail(phases, state.get("understanding", {}),
                                           state.get("spec", {}), state["artifact"], report, backend)
        skill_tracing.export_tree(run, detail)
    except Exception:  # noqa: BLE001
        pass
    skill_agent._remember_safe(
        f"{'Refined' if kind == 'refine' else 'Built'} agent skill '{skill['name']}' via "
        f"langgraph/{backend}; eval gate = {report['gate']}.",
        kind="observation", salience=2, origin="skill_graph", confidence="EXTRACTED",
        tags=["skill", kind])
    return {"report": report, "gate": report["gate"], "status": skill["status"], "skill": skill,
            "phases": phases}


def _node_human_review(state: SkillState) -> dict:
    from langgraph.types import interrupt
    decision = interrupt({
        "type": "skill_review", "skill_id": state["skill_id"], "gate": state.get("gate"),
        "skill": state.get("skill"), "eval": state.get("report"),
        "prompt": "Review this skill: respond with {decision: accept|reject|revise, score?, notes?}.",
    })
    if isinstance(decision, str):
        decision = {"decision": decision}
    d = (decision.get("decision") or "").strip().lower()
    updated = skills.record_human_review(
        state["skill_id"], decision=d, score=decision.get("score"),
        notes=decision.get("notes", ""), reviewer=decision.get("reviewer", "user"))
    if not updated:
        raise skill_agent.SkillError(f"Invalid review decision {d!r}.")
    skill_runs.record(kind="review", skill_id=state["skill_id"], skill_name=updated.get("name", ""),
                      status=updated["status"], gate=state.get("gate"),
                      metrics={"human_score": (updated.get("human") or {}).get("score"),
                               "aligned_with_gate": (updated.get("human") or {}).get("aligned_with_gate")})
    return {"decision": d, "notes": decision.get("notes", ""), "skill": updated,
            "status": updated["status"]}


def _node_prepare_revision(state: SkillState) -> dict:
    return {"revision_note": state.get("notes", ""), "attempt": state.get("attempt", 0) + 1}


def _node_finalize(state: SkillState) -> dict:
    skills.export_skill(state["skill_id"])
    skill = skills.get_skill(state["skill_id"])
    skill_agent._remember_safe(
        f"Accepted agent skill '{skill['name']}': {skill.get('description', '')}",
        kind="fact", salience=3, origin="skill_graph", confidence="USER", tags=["skill", "accepted"])
    return {"done": True, "status": skill["status"], "skill": skill}


# --- routers -----------------------------------------------------------------
def _route_backend(state: SkillState) -> str:
    return "generate_cc" if skill_agent._resolve_backend(state.get("backend")) == "claude_code" else "understand"


def _route_backend_revise(state: SkillState) -> str:
    return "generate_cc" if skill_agent._resolve_backend(state.get("backend")) == "claude_code" else "codeact"


def _route_gate(state: SkillState) -> str:
    return "reject" if state.get("gate") == "reject" else "review"


def _route_decision(state: SkillState) -> str:
    d = state.get("decision")
    if d == "accept":
        return "accept"
    if d == "revise":
        return "give_up" if state.get("attempt", 0) >= MAX_REVISIONS else "revise"
    return "reject"


# --- graph construction + checkpointer --------------------------------------
_graph_cache: dict = {"graph": None, "saver": None}


def _checkpointer():
    mode = os.environ.get("SKILL_GRAPH_CHECKPOINT", "sqlite").strip().lower()
    if mode == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    if mode == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            conn = os.environ.get("SKILL_GRAPH_DB_URL") or os.environ.get("SUPABASE_DB_URL")
            saver = PostgresSaver.from_conn_string(conn)
            if hasattr(saver, "__enter__"):
                saver = saver.__enter__()
            saver.setup()
            return saver
        except Exception:  # noqa: BLE001  (fall back to local sqlite)
            pass
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("SKILL_GRAPH_DB", str(DATA_DIR / "skill_graph.sqlite"))
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    try:
        saver.setup()
    except Exception:  # noqa: BLE001
        pass
    return saver


def _build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(SkillState)
    g.add_node("gather", _node_gather)
    g.add_node("understand", _node_understand)
    g.add_node("analyze", _node_analyze)
    g.add_node("codeact", _node_codeact)
    g.add_node("generate_cc", _node_generate_cc)
    g.add_node("evaluate", _node_evaluate)
    g.add_node("human_review", _node_human_review)
    g.add_node("prepare_revision", _node_prepare_revision)
    g.add_node("finalize", _node_finalize)

    g.set_entry_point("gather")
    g.add_conditional_edges("gather", _route_backend,
                            {"understand": "understand", "generate_cc": "generate_cc"})
    g.add_edge("understand", "analyze")
    g.add_edge("analyze", "codeact")
    g.add_edge("codeact", "evaluate")
    g.add_edge("generate_cc", "evaluate")
    g.add_conditional_edges("evaluate", _route_gate, {"review": "human_review", "reject": END})
    g.add_conditional_edges("human_review", _route_decision,
                            {"accept": "finalize", "revise": "prepare_revision",
                             "reject": END, "give_up": END})
    g.add_conditional_edges("prepare_revision", _route_backend_revise,
                            {"codeact": "codeact", "generate_cc": "generate_cc"})
    g.add_edge("finalize", END)
    return g


def get_graph():
    if _graph_cache["graph"] is None:
        saver = _checkpointer()
        _graph_cache["saver"] = saver
        _graph_cache["graph"] = _build_graph().compile(checkpointer=saver)
    return _graph_cache["graph"]


def reset() -> None:
    """Drop the cached compiled graph + checkpointer (used by tests)."""
    _graph_cache["graph"] = None
    _graph_cache["saver"] = None


# --- entrypoints -------------------------------------------------------------
def _summarize(thread_id: str, state: dict) -> dict:
    interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
    if interrupts:
        payload = getattr(interrupts[0], "value", interrupts[0])
        sid = payload.get("skill_id") if isinstance(payload, dict) else None
        if sid:  # link the paused build to its thread so the review queue can resume it
            try:
                skills.set_thread(sid, thread_id)
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "thread_id": thread_id, "awaiting_review": True,
                "status": "pending_review", "gate": payload.get("gate"),
                "skill": payload.get("skill"), "eval": payload.get("eval"),
                "interrupt": payload}
    return {"ok": True, "thread_id": thread_id, "awaiting_review": False, "done": True,
            "status": state.get("status"), "gate": state.get("gate"),
            "skill": state.get("skill"), "eval": state.get("report")}


def run_build(*, chunk_ids=None, text=None, query=None, tags=None, where="overall", goal="",
              provider="auto", model=None, backend=None, use_tools=True, run_rubric=True,
              run_triggering=True, thread_id: str | None = None) -> dict:
    """Start a checkpointed skill build. Runs to the human-review interrupt and
    returns ``{thread_id, awaiting_review, skill, eval, ...}``; resume with
    ``resume_review(thread_id, decision=…)``."""
    tid = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": tid}}
    inputs: dict[str, Any] = {
        "chunk_ids": chunk_ids, "text": text, "query": query, "tags": tags, "where": where,
        "goal": goal, "provider": provider, "model": model, "backend": backend,
        "use_tools": use_tools, "run_rubric": run_rubric, "run_triggering": run_triggering,
        "attempt": 0, "revision_note": "",
    }
    state = get_graph().invoke(inputs, config=config)
    return _summarize(tid, state)


def resume_review(thread_id: str, *, decision: str, score: float | None = None,
                  notes: str = "", reviewer: str = "user") -> dict:
    """Resume a paused build with a human decision (accept | reject | revise).

    ``revise`` loops back through codeact with ``notes`` and pauses again at the
    next review; ``accept`` finalizes; ``reject`` ends."""
    from langgraph.types import Command
    config = {"configurable": {"thread_id": thread_id}}
    state = get_graph().invoke(
        Command(resume={"decision": decision, "score": score, "notes": notes, "reviewer": reviewer}),
        config=config)
    return _summarize(thread_id, state)


def get_status(thread_id: str) -> dict:
    """Inspect a thread's current state (next node + whether it awaits review)."""
    snap = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    return {"thread_id": thread_id, "next": list(snap.next),
            "awaiting_review": "human_review" in snap.next,
            "skill_id": (snap.values or {}).get("skill_id"),
            "status": (snap.values or {}).get("status")}


def checkpoint_backend() -> str:
    return os.environ.get("SKILL_GRAPH_CHECKPOINT", "sqlite").strip().lower()
