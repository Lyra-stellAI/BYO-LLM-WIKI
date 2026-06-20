"""Skill-builder sub-agent system (layer 7) — turn selective context into a skill.

When the user feeds the library selective context (chunks they ingested, pasted
text, a saved query) and *asks for a skill*, this module runs a small pipeline of
specialized sub-agent roles that build, evaluate, and gate a new agent skill —
the same shape as the agent-skill build loop in OpenAI's *Evaluating skills*
guide, made explicit as named phases:

    gather  → collect the selected context + its provenance
    understand → the *Context Understander* reads the context and states what it is
    analyze    → the *Skill Analyst* designs the skill spec + success criteria
    codeact    → the *Skill Author* writes the skill (instructions, steps, tests)
    eval       → ``skill_eval`` grades it (deterministic checks + rubric panel)
    gate       → accept / reject / review; a passing gate routes to human review

The gate never finalizes acceptance on its own: a passing skill lands in
``pending_review`` and a human aligns on it (``review_skill``) before it becomes a
live, recallable skill in ``skill_library``. A human's *revise* sends notes back
into a fresh ``codeact`` pass (``rebuild_skill``), closing the loop.

Each phase is a focused LLM call (like ``rag``/``crossdoc``), not the heavier
deepagents harness, so the steps are explicit and inspectable; the phases reuse
the shared provider table so a single key lights them up.
"""

from __future__ import annotations

import json
import re
import time

import knowledge_graph as kg
import skill_eval
import skill_library as skills
import skill_runs
from providers import ProviderError, build_chat_model, skill_generator

try:
    import memory
except Exception:  # noqa: BLE001  (memory is optional; never block a build)
    memory = None

MAX_CONTEXT_CHARS = 12000
DIGEST_CHARS = 1800


class SkillError(RuntimeError):
    """Raised when a skill-build phase cannot run (missing deps/keys/context)."""


class _Meter:
    """Accumulates token usage across the phases of one build (observability)."""

    def __init__(self):
        self.tokens = 0

    def add(self, msg) -> None:
        try:
            from skill_runtime import usage_of
            self.tokens += usage_of(msg)
        except Exception:  # noqa: BLE001
            pass


# --- shared LLM plumbing -----------------------------------------------------
def _resolve_model(provider, model, *, temperature: float = 0.2, max_tokens: int = 2600):
    """Resolve and build the skill-generation chat model.

    Prefers the latest Claude (``providers.skill_generator``) so skills and their
    SKILL.md are authored by the strongest available model, honoring an explicit
    provider/model override."""
    rp, rm = skill_generator(provider, model)
    if not rp:
        raise SkillError(
            "No LLM provider is configured. Set one of ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / GEMINI_API_KEY / "
            "MISTRAL_API_KEY to build skills.")
    try:
        return build_chat_model(rp, rm, temperature=temperature, max_tokens=max_tokens), rp, rm
    except ProviderError as exc:
        raise SkillError(str(exc)) from exc


def _gen(model, prompt: str, meter: _Meter | None = None) -> str:
    msg = model.invoke(prompt)
    if meter is not None:
        meter.add(msg)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group())
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _memory_preamble(query: str, *, k: int = 5) -> str:
    if memory is None:
        return ""
    try:
        block, _ = memory.recall_block(query, k=k)
    except Exception:  # noqa: BLE001
        return ""
    return (block + "\n\n") if block else ""


def _remember_safe(text: str, **kw) -> None:
    if memory is None:
        return
    try:
        memory.remember(text, **kw)
    except Exception:  # noqa: BLE001
        pass


# --- phase 0: gather the selected context -----------------------------------
def _chunks_by_id(chunk_ids: list[str]) -> list[dict]:
    """Find chunk nodes by id across staging (current) and the integrated library."""
    wanted = set(chunk_ids)
    found: dict[str, dict] = {}
    for where in ("current", "overall"):
        g = kg.get_graph(where)
        for n in g["nodes"]:
            if n.get("type") == "chunk" and n["id"] in wanted and n["id"] not in found:
                found[n["id"]] = n
    return [found[c] for c in chunk_ids if c in found]


def gather_context(*, chunk_ids: list[str] | None = None, text: str | None = None,
                   query: str | None = None, tags: list[str] | None = None,
                   where: str = "overall", max_chunks: int = 24) -> dict:
    """Collect the user's selected context into a single bundle with provenance.

    Accepts any mix of: explicit ``chunk_ids`` (from staging or the library),
    raw ``text`` the user pasted, a ``query`` that pulls matching chunks, or
    ``tags`` to pull tagged chunks. Returns ``{context, provenance}`` where
    ``provenance`` records the chunk ids / source titles+urls / char count and a
    trimmed digest used later to ground the eval judges."""
    blocks: list[str] = []
    chunk_id_list: list[str] = []
    source_titles: list[str] = []
    source_urls: list[str] = []

    selected: list[dict] = []
    if chunk_ids:
        selected += _chunks_by_id(chunk_ids)
    if query:
        selected += kg.query(query, where=where, limit=max_chunks, types=["chunk"])
    if tags:
        tagset = {t.strip().lower() for t in tags if t and t.strip()}
        g = kg.get_graph(where)
        for n in g["nodes"]:
            if n.get("type") == "chunk" and tagset & {t.lower() for t in (n.get("tags") or [])}:
                selected.append(n)

    seen = set()
    for n in selected[:max_chunks]:
        cid = n.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        chunk_id_list.append(cid)
        title = n.get("source_title") or n.get("title") or ""
        url = n.get("source_url") or n.get("url") or ""
        if title and title not in source_titles:
            source_titles.append(title)
        if url and url not in source_urls:
            source_urls.append(url)
        head = f"[{cid}] {title}".strip()
        blocks.append(f"{head}\n{n.get('text', '')}".strip())

    if text and text.strip():
        blocks.append(f"[pasted-context]\n{text.strip()}")
        if "(pasted context)" not in source_titles:
            source_titles.append("(pasted context)")

    context = "\n\n---\n\n".join(b for b in blocks if b).strip()
    if not context:
        raise SkillError(
            "No context to build from. Provide chunk_ids, a query that matches "
            "ingested chunks, tags, or paste text.")
    context = context[:MAX_CONTEXT_CHARS]
    provenance = {
        "chunk_ids": chunk_id_list,
        "source_titles": source_titles,
        "source_urls": source_urls,
        "context_chars": len(context),
        "context_digest": context[:DIGEST_CHARS],
    }
    return {"context": context, "provenance": provenance}


# --- phase 1: understand -----------------------------------------------------
_UNDERSTAND_PROMPT = """You are the CONTEXT UNDERSTANDER, the first sub-agent in a skill-building
pipeline. Read the SELECTED CONTEXT below and state plainly what it is, before any
skill is designed. Do not invent facts beyond the context.

{memory}SELECTED CONTEXT:
{context}

Return ONLY JSON:
{{"domain": "<the subject area in a few words>",
"summary": "<2-4 sentence neutral summary of what this context covers>",
"repeatable_tasks": ["<a concrete, repeatable task an agent could do using this context>", ...],
"key_facts": ["<an essential fact/constraint from the context an agent would rely on>", ...],
"gaps": ["<something the context does NOT cover that a skill should not assume>", ...]}}"""


def understand(bundle: dict, *, chat, meter: _Meter | None = None) -> dict:
    prompt = _UNDERSTAND_PROMPT.format(
        memory=_memory_preamble(bundle["provenance"].get("context_digest", "")[:300]),
        context=bundle["context"])
    data = _parse_json(_gen(chat, prompt, meter))
    return {
        "domain": str(data.get("domain", "")).strip(),
        "summary": str(data.get("summary", "")).strip(),
        "repeatable_tasks": [str(x).strip() for x in (data.get("repeatable_tasks") or []) if str(x).strip()],
        "key_facts": [str(x).strip() for x in (data.get("key_facts") or []) if str(x).strip()],
        "gaps": [str(x).strip() for x in (data.get("gaps") or []) if str(x).strip()],
    }


# --- phase 2: analyze --------------------------------------------------------
_ANALYZE_PROMPT = """You are the SKILL ANALYST, the second sub-agent. Using the context and the
understanding below, design the spec for ONE high-value agent skill — a reusable
instruction set an agent can invoke to perform a repeatable task grounded in this
context. {goal}

Define success up front across four categories (from skill-evaluation best practice):
outcome (task completion), process (right steps/tools), style (conventions/clarity),
efficiency (no wasted work). Include positive trigger cases AND negative controls
(when the skill must NOT fire) so it can be evaluated.

UNDERSTANDING:
{understanding}

SELECTED CONTEXT:
{context}

Return ONLY JSON:
{{"name": "<short Title Case skill name, 2-5 words>",
"description": "<one sentence an agent reads to decide whether to invoke this skill>",
"rationale": "<why this skill is worth building from this context>",
"success_criteria": {{"outcome": "<...>", "process": "<...>", "style": "<...>", "efficiency": "<...>"}},
"triggers": ["<a situation where the skill SHOULD be used>", ...],
"anti_triggers": ["<a situation where the skill should NOT be used (negative control)>", ...],
"tools": ["<a tool/capability the skill expects, or 'none'>", ...],
"inputs": ["<what the user/agent must supply>", ...],
"risks": ["<a failure mode to guard against>", ...]}}"""


def analyze(bundle: dict, understanding: dict, *, chat, goal: str = "", meter: _Meter | None = None) -> dict:
    goal_line = (f"The user asked specifically for: \"{goal.strip()}\". Design that skill if the "
                 f"context supports it." if goal and goal.strip()
                 else "Pick the single most useful skill the context can support.")
    prompt = _ANALYZE_PROMPT.format(
        goal=goal_line,
        understanding=json.dumps(understanding, ensure_ascii=False),
        context=bundle["context"])
    data = _parse_json(_gen(chat, prompt, meter))
    crit = data.get("success_criteria") or {}
    return {
        "name": str(data.get("name", "")).strip(),
        "description": str(data.get("description", "")).strip(),
        "rationale": str(data.get("rationale", "")).strip(),
        "success_criteria": {k: crit.get(k, "") for k in ("outcome", "process", "style", "efficiency")},
        "triggers": [str(x).strip() for x in (data.get("triggers") or []) if str(x).strip()],
        "anti_triggers": [str(x).strip() for x in (data.get("anti_triggers") or []) if str(x).strip()],
        "tools": [str(x).strip() for x in (data.get("tools") or []) if str(x).strip()
                  and str(x).strip().lower() != "none"],
        "inputs": [str(x).strip() for x in (data.get("inputs") or []) if str(x).strip()],
        "risks": [str(x).strip() for x in (data.get("risks") or []) if str(x).strip()],
    }


# --- phase 3: codeact (author the skill) -------------------------------------
_CODEACT_PROMPT = """You are the SKILL AUTHOR (CodeAct), the third sub-agent. Write the actual,
ready-to-use agent skill from the spec and context below. The skill must be
self-contained, grounded ONLY in the context (no invented facts), and pass an
evaluation that checks for: a clear name + description, instructions with at least
2 explicit numbered steps, at least one trigger AND one anti-trigger, declared
tools, and a small test set with at least one positive and one negative case.
Do not leave placeholders (no TODO/TBD/<...>). {revision}

SKILL SPEC:
{spec}

SELECTED CONTEXT (ground everything here):
{context}

Return ONLY JSON:
{{"name": "<short Title Case name>",
"description": "<one-sentence invocation signal>",
"instructions": "<markdown: a short intro then numbered steps the agent follows; specific and grounded>",
"steps": ["<step 1>", "<step 2>", ...],
"triggers": ["<when to use>", ...],
"anti_triggers": ["<when NOT to use>", ...],
"tools": ["<expected tool/capability>", ...],
"success_criteria": {{"outcome": "<...>", "process": "<...>", "style": "<...>", "efficiency": "<...>"}},
"tests": [{{"prompt": "<example user request>", "should_trigger": true, "expect": "<what a good run does>"}},
{{"prompt": "<a request the skill should NOT handle>", "should_trigger": false, "expect": "<why it should defer>"}}]}}"""


def _normalize_artifact(data: dict, spec: dict) -> dict:
    crit = data.get("success_criteria") or spec.get("success_criteria") or {}
    tests = [t for t in (data.get("tests") or []) if isinstance(t, dict) and t.get("prompt")]
    return {
        "name": str(data.get("name") or spec.get("name", "")).strip(),
        "description": str(data.get("description") or spec.get("description", "")).strip(),
        "instructions": str(data.get("instructions", "")).strip(),
        "steps": [str(x).strip() for x in (data.get("steps") or []) if str(x).strip()],
        "triggers": [str(x).strip() for x in (data.get("triggers") or spec.get("triggers") or []) if str(x).strip()],
        "anti_triggers": [str(x).strip() for x in (data.get("anti_triggers") or spec.get("anti_triggers") or []) if str(x).strip()],
        "tools": [str(x).strip() for x in (data.get("tools") or spec.get("tools") or []) if str(x).strip()
                  and str(x).strip().lower() != "none"],
        "success_criteria": {k: crit.get(k, "") for k in ("outcome", "process", "style", "efficiency")},
        "tests": tests,
    }


def _revision_line(revision_note: str) -> str:
    return (f"A human reviewer asked for changes — address this in the rewrite: "
            f"\"{revision_note.strip()}\"." if revision_note and revision_note.strip() else "")


def codeact(bundle: dict, spec: dict, *, chat, revision_note: str = "", meter: _Meter | None = None) -> dict:
    prompt = _CODEACT_PROMPT.format(
        revision=_revision_line(revision_note),
        spec=json.dumps(spec, ensure_ascii=False),
        context=bundle["context"])
    return _normalize_artifact(_parse_json(_gen(chat, prompt, meter)), spec)


# The tool-using author: it may read context, recall memory, list existing skills,
# and—crucially—check its own draft (test → measure → refine) before finalizing.
_CODEACT_TOOLS_SYSTEM = """You are the SKILL AUTHOR (CodeAct) with tools. Write one ready-to-use agent
skill, grounded ONLY in the provided context. You have tools: read_context (pull
source passages), list_existing_skills (avoid duplicates), recall_memory (reuse
learnings), and check_draft (run the deterministic checks on a candidate).

Work iteratively: draft the skill, call check_draft on it, FIX every reported
failure, and only then return the final skill. The skill must have a clear name +
one-sentence description, instructions with >= 2 explicit numbered steps, at least
one trigger and one anti-trigger, declared tools, and a test set with at least one
positive (should_trigger true) and one negative (should_trigger false) case. No
placeholders (TODO/TBD/<...>).

When done, return ONLY the final skill as a JSON object with keys: name,
description, instructions, steps, triggers, anti_triggers, tools, success_criteria
{outcome, process, style, efficiency}, tests [{prompt, should_trigger, expect}]."""


def codeact_tooluse(bundle: dict, spec: dict, *, chat, revision_note: str = "",
                    meter: _Meter | None = None) -> tuple[dict, int]:
    """Author the skill with a tool-use loop (returns artifact + tool-call count)."""
    import skill_runtime
    user = (f"{_revision_line(revision_note)}\n\nSKILL SPEC:\n{json.dumps(spec, ensure_ascii=False)}"
            f"\n\nSELECTED CONTEXT (ground everything here):\n{bundle['context']}")
    loop = skill_runtime.run_tool_loop(
        chat, skill_runtime.builder_tools(bundle), _CODEACT_TOOLS_SYSTEM, user, max_iters=6)
    if meter is not None:
        meter.tokens += loop.get("tokens", 0)
    artifact = _normalize_artifact(_parse_json(loop.get("text", "")), spec)
    return artifact, loop.get("tool_calls", 0)


# --- orchestration -----------------------------------------------------------
def _timed(fn):
    """Run ``fn`` and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn()
    return out, int((time.perf_counter() - t0) * 1000)


def _build_from_bundle(bundle: dict, *, provider, model, goal: str = "",
                       revision_note: str = "", where_id: str | None = None,
                       run_rubric: bool = True, run_triggering: bool = True,
                       use_tools: bool = True, kind: str = "build",
                       judge_provider=None, judge_model=None) -> dict:
    chat, rp, rm = _resolve_model(provider, model)
    wall0 = time.perf_counter()
    phases: dict = {}

    mu = _Meter()
    understanding, ms = _timed(lambda: understand(bundle, chat=chat, meter=mu))
    phases["understand"] = {"ms": ms, "tokens": mu.tokens}

    ma = _Meter()
    spec, ms = _timed(lambda: analyze(bundle, understanding, chat=chat, goal=goal, meter=ma))
    phases["analyze"] = {"ms": ms, "tokens": ma.tokens}

    # codeact: a tool-using author (read context, recall, check its own draft) when
    # the model supports tool calls; otherwise a single grounded completion.
    mc = _Meter()
    tools_used = 0
    tool_mode = bool(use_tools and hasattr(chat, "bind_tools"))
    if tool_mode:
        (artifact, tools_used), ms = _timed(
            lambda: codeact_tooluse(bundle, spec, chat=chat, revision_note=revision_note, meter=mc))
        if not (artifact.get("instructions") and artifact.get("name")):
            # Tool loop produced nothing usable -> fall back to the one-shot author.
            tool_mode = False
            artifact, ms2 = _timed(lambda: codeact(bundle, spec, chat=chat,
                                                   revision_note=revision_note, meter=mc))
            ms += ms2
    else:
        artifact, ms = _timed(lambda: codeact(bundle, spec, chat=chat,
                                              revision_note=revision_note, meter=mc))
    phases["codeact"] = {"ms": ms, "tokens": mc.tokens, "tool_calls": tools_used,
                         "tool_mode": tool_mode}

    if not artifact.get("instructions") or not artifact.get("name"):
        skill_runs.record(kind=kind, provider=rp, model=rm, ok=False,
                          error="author produced no usable skill", phases=phases)
        raise SkillError("The author phase did not produce a usable skill (empty instructions/name).")

    artifact["provenance"] = bundle["provenance"]
    if revision_note:
        artifact["revision_note"] = revision_note
    skill = skills.upsert_skill(artifact, where_id=where_id)
    if not skill:
        raise SkillError("Could not store the drafted skill.")
    skill_id = skill["id"]
    skills.append_history(skill_id, kind, "drafted",
                          note=f"{rp}/{rm}{' +tools' if tool_mode else ''} · {spec.get('name', '')}")
    # Generate the SKILL.md on disk now (data/skills/<slug>/SKILL.md), so the
    # portable artifact exists from the draft, not only after acceptance.
    skills.export_skill(skill_id)

    # eval → gate → record (advances status to pending_review or rejected)
    report, eval_ms = _timed(lambda: skill_eval.run_eval(
        skill, provider=rp, model=rm, run_rubric=run_rubric, run_triggering=run_triggering,
        judge_provider=judge_provider, judge_model=judge_model))
    phases["eval"] = {"ms": eval_ms}
    skill = skills.record_eval(skill_id, report)

    duration_ms = int((time.perf_counter() - wall0) * 1000)
    total_tokens = mu.tokens + ma.tokens + mc.tokens
    trig = report.get("triggering") or {}
    metrics = {
        "deterministic_ratio": report["deterministic"]["ratio"],
        "rubric_mean": report.get("rubric_mean"),
        "trigger_precision": trig.get("precision"),
        "trigger_recall": trig.get("recall"),
        "trigger_f1": trig.get("f1"),
    }
    # Observability: log this build as a measured run (the 'measure' of the loop).
    skill_runs.record(kind=kind, skill_id=skill_id, skill_name=skill["name"],
                      provider=rp, model=rm, gate=report["gate"], status=skill["status"],
                      duration_ms=duration_ms, tokens=total_tokens, tools_used=tools_used,
                      phases=phases, metrics=metrics)

    # Memory partnership (Hermes step 4-6): record the experience as a durable note.
    _remember_safe(
        f"{'Refined' if kind == 'refine' else 'Built'} agent skill '{artifact['name']}' from "
        f"{len(bundle['provenance'].get('chunk_ids', []))} context chunk(s); eval gate = {report['gate']}.",
        kind="observation", salience=2, origin="skill_agent",
        confidence="EXTRACTED", tags=["skill", kind])

    return {
        "ok": True,
        "skill": skill,
        "skill_id": skill_id,
        "status": skill["status"],
        "gate": report["gate"],
        "phases": {"understand": understanding, "analyze": spec, "codeact": artifact},
        "observability": {"duration_ms": duration_ms, "tokens": total_tokens,
                          "tools_used": tools_used, "tool_mode": tool_mode, "timings": phases},
        "eval": report,
        "provider": rp,
        "model": rm,
    }


def build_skill(*, chunk_ids=None, text=None, query=None, tags=None, where="overall",
                goal: str = "", provider="auto", model=None, run_rubric: bool = True,
                run_triggering: bool = True, use_tools: bool = True,
                judge_provider=None, judge_model=None) -> dict:
    """Run the full pipeline: gather → understand → analyze → codeact → eval → gate.

    With ``use_tools`` (default) the author may read context, recall memory, and
    check its own draft before finalizing (test → measure → refine). The skill is
    drafted, evaluated, gated, and its SKILL.md written; on a passing gate it lands
    in ``pending_review`` for human alignment (it is NOT auto-accepted). Returns
    every phase's output, the eval report, and an observability summary."""
    bundle = gather_context(chunk_ids=chunk_ids, text=text, query=query, tags=tags, where=where)
    return _build_from_bundle(bundle, provider=provider, model=model, goal=goal,
                              run_rubric=run_rubric, run_triggering=run_triggering,
                              use_tools=use_tools, judge_provider=judge_provider,
                              judge_model=judge_model)


def rebuild_skill(skill_id: str, *, provider="auto", model=None, extra_guidance: str = "",
                  run_rubric: bool = True, run_triggering: bool = True, use_tools: bool = True,
                  judge_provider=None, judge_model=None, kind: str = "build") -> dict:
    """Re-run the pipeline for an existing skill, folding the human's revision notes
    (and any ``extra_guidance``) into a fresh ``codeact`` pass — the loop closer.

    Reuses the skill's recorded provenance so the rebuild stays grounded in the
    same context."""
    skill = skills.get_skill(skill_id)
    if not skill:
        raise SkillError(f"No skill with id {skill_id!r}.")
    prov = skill.get("provenance") or {}
    note = " ".join(filter(None, [(skill.get("human") or {}).get("notes", ""), extra_guidance])).strip()
    bundle = gather_context(chunk_ids=prov.get("chunk_ids") or None,
                            text=(prov.get("context_digest") if not prov.get("chunk_ids") else None),
                            where="overall")
    return _build_from_bundle(bundle, provider=provider, model=model,
                              revision_note=note, where_id=skill["id"],
                              run_rubric=run_rubric, run_triggering=run_triggering,
                              use_tools=use_tools, judge_provider=judge_provider,
                              judge_model=judge_model, kind=kind)


def _refine_guidance(skill: dict) -> str:
    """Turn a skill's last eval into concrete 'fix this' guidance (the refine input)."""
    ev = skill.get("eval") or {}
    parts = []
    failures = (ev.get("deterministic") or {}).get("failures") or []
    if failures:
        parts.append("Fix these failed checks: " + ", ".join(failures) + ".")
    trig = ev.get("triggering") or {}
    if trig.get("precision") is not None and trig["precision"] < 0.99:
        parts.append("Tighten the description/anti-triggers to reduce false positives "
                     f"(precision {trig['precision']}).")
    if trig.get("recall") is not None and trig["recall"] < 0.99:
        parts.append("Broaden the description/triggers so positive cases fire "
                     f"(recall {trig['recall']}).")
    rub = ev.get("rubric") or {}
    weakest = None
    if isinstance(rub.get("per_dimension"), dict) and rub["per_dimension"]:
        weakest = min(rub["per_dimension"], key=rub["per_dimension"].get)
        parts.append(f"Strengthen the weakest rubric dimension: {weakest}.")
    return " ".join(parts) or "Improve clarity, grounding, and step specificity."


def refine_skill(skill_id: str, *, provider="auto", model=None, run_rubric: bool = True,
                 run_triggering: bool = True, use_tools: bool = True) -> dict:
    """Self-improvement pass: read the skill's measured weaknesses (failed checks,
    triggering precision/recall, weakest rubric dimension) and rebuild it to address
    them — the test → measure → refine loop, logged as a ``refine`` run."""
    skill = skills.get_skill(skill_id)
    if not skill:
        raise SkillError(f"No skill with id {skill_id!r}.")
    guidance = _refine_guidance(skill)
    return rebuild_skill(skill_id, provider=provider, model=model, extra_guidance=guidance,
                         run_rubric=run_rubric, run_triggering=run_triggering,
                         use_tools=use_tools, kind="refine")


# --- human-in-the-loop alignment --------------------------------------------
def review_skill(skill_id: str, *, decision: str, score: float | None = None,
                 notes: str = "", reviewer: str = "user", rebuild_on_revise: bool = False,
                 provider="auto", model=None) -> dict:
    """Record a human's alignment decision (accept / reject / revise) on a skill.

    This is the authority the automated gate defers to: only ``accept`` makes a
    skill live in the library. ``revise`` records notes and (optionally) kicks off
    a rebuild that folds them in."""
    skill = skills.get_skill(skill_id)
    if not skill:
        raise SkillError(f"No skill with id {skill_id!r}.")
    updated = skills.record_human_review(skill_id, decision=decision, score=score,
                                          notes=notes, reviewer=reviewer)
    if not updated:
        raise SkillError(f"Invalid review decision {decision!r} (use accept | reject | revise).")

    result = {"ok": True, "skill": updated, "decision": decision.strip().lower(),
              "status": updated["status"]}
    skill_runs.record(kind="review", skill_id=skill_id, skill_name=updated.get("name", ""),
                      status=updated["status"], gate=(skill.get("eval") or {}).get("gate"),
                      metrics={"human_score": (updated.get("human") or {}).get("score"),
                               "aligned_with_gate": (updated.get("human") or {}).get("aligned_with_gate")})
    if updated["status"] == skills.ACCEPTED:
        skills.export_skill(skill_id)  # write/refresh SKILL.md to disk
        # Memory partnership: a durable fact so the agent auto-loads this skill later.
        _remember_safe(f"Accepted agent skill '{updated['name']}': {updated['description']}",
                       kind="fact", salience=3, origin="skill_review",
                       confidence="USER", tags=["skill", "accepted"])
    if result["decision"] == "revise" and rebuild_on_revise:
        try:
            result["rebuild"] = rebuild_skill(skill_id, provider=provider, model=model)
        except SkillError as exc:
            result["rebuild_error"] = str(exc)
    return result
