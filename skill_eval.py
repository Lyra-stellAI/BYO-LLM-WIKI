"""Skill evaluation — the eval + accept/reject gate of the skill-build loop.

Mirrors the two-layer approach from OpenAI's *Evaluating skills* guide
(https://developers.openai.com/blog/eval-skills): start with fast, explainable
**deterministic checks** that need no model, then layer **model-assisted rubric
scoring** for the qualitative requirements, and gate acceptance on the results.

1. Deterministic checks (``run_deterministic``) — pure Python, run offline.
   Cheap structural signals: does the skill have a name, a usable description,
   real instructions with explicit steps, positive *and* negative trigger cases,
   declared tools, a small test set, no leftover placeholders, and grounding in
   the context it was built from? These are the regression-style guardrails.

2. Model-assisted rubric (``run_rubric_panel``) — a cross-family **judge panel**
   (reused from ``providers.judge_panel``, the same anti-self-preference panel the
   RAG cross-doc eval uses) scores the skill on the four success-criteria
   categories from the guide — *outcome, process, style, efficiency* — plus
   *groundedness* against the source context, so "well-grounded" is checkable
   rather than guessed.

3. Gate (``decide_gate``) — combines the deterministic pass-rate and the rubric
   mean into one of ``accept`` / ``reject`` / ``review``. The gate is deliberately
   conservative: it never *finalizes* acceptance on its own. A passing gate routes
   the skill to ``pending_review`` for a human to align on (see ``skill_agent`` and
   ``skill_library.record_human_review``); only a human moves it to ``accepted``.
"""

from __future__ import annotations

import json
import os
import re

from providers import build_chat_model, judge_panel, resolve_judge, resolve_provider_model

# Gate thresholds (overridable via env for experimentation).
GATE_ACCEPT = float(os.environ.get("SKILL_GATE_ACCEPT", "0.70"))   # rubric mean to pass the bar
GATE_REJECT = float(os.environ.get("SKILL_GATE_REJECT", "0.40"))   # rubric mean below this -> reject
DET_PASS = float(os.environ.get("SKILL_DET_PASS", "0.80"))         # deterministic pass-ratio to pass

# Reasoning/thinking judges spend their budget on hidden reasoning; give the panel
# a larger token budget (as the cross-doc judge does) so structured output fits.
_JUDGE_MAX_TOKENS = 1200

_PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|lorem ipsum|your (?:skill|task|tool) here|placeholder)\b"
    r"|<[a-z_]{3,}>", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S")

# Checks that MUST pass for a skill to be acceptable at all.
_CRITICAL = {"has_name", "has_description", "has_instructions", "has_steps"}


# --- 1. deterministic checks -------------------------------------------------
def _step_count(skill: dict) -> int:
    steps = skill.get("steps") or []
    if steps:
        return len(steps)
    return len(_NUMBERED_RE.findall(skill.get("instructions", "")))


def run_deterministic(skill: dict) -> dict:
    """Fast, model-free structural checks. Returns a report with per-check rows.

    Each row is ``{key, passed, detail}``; the report aggregates a pass count and
    the list of failures. Safe to run with no API key (used by the offline tests)."""
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    instr = (skill.get("instructions") or "").strip()
    triggers = skill.get("triggers") or []
    anti = skill.get("anti_triggers") or []
    tools = skill.get("tools") or []
    tests = skill.get("tests") or []
    prov = skill.get("provenance") or {}
    steps = _step_count(skill)

    pos_tests = sum(1 for t in tests if isinstance(t, dict) and t.get("should_trigger"))
    neg_tests = sum(1 for t in tests if isinstance(t, dict) and t.get("should_trigger") is False)
    placeholder_hit = bool(_PLACEHOLDER_RE.search(desc + "\n" + instr))
    grounded = bool(prov.get("chunk_ids") or prov.get("source_titles") or prov.get("context_chars"))

    checks = [
        ("has_name", 2 <= len(name) <= 80, f"{len(name)} chars"),
        ("has_description", len(desc) >= 20 and len(desc.split()) >= 5,
         f"{len(desc)} chars / {len(desc.split())} words"),
        ("has_instructions", len(instr) >= 80, f"{len(instr)} chars"),
        ("has_steps", steps >= 2, f"{steps} steps"),
        ("has_triggers", len(triggers) >= 1, f"{len(triggers)} trigger(s)"),
        ("has_anti_triggers", len(anti) >= 1, f"{len(anti)} negative control(s)"),
        ("declares_tools", len(tools) >= 1, f"{len(tools)} tool(s)"),
        ("has_positive_and_negative_tests", pos_tests >= 1 and neg_tests >= 1,
         f"{pos_tests} positive / {neg_tests} negative"),
        ("no_placeholders", not placeholder_hit,
         "placeholder text found" if placeholder_hit else "clean"),
        ("grounded_in_context", grounded,
         "has provenance" if grounded else "no source context recorded"),
        ("within_budget", len(instr) <= 6000, f"{len(instr)} chars"),
    ]
    rows = [{"key": k, "passed": bool(p), "detail": d} for k, p, d in checks]
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    failures = [r["key"] for r in rows if not r["passed"]]
    critical_failed = [k for k in _CRITICAL if any(r["key"] == k and not r["passed"] for r in rows)]
    return {
        "checks": rows,
        "passed": passed,
        "total": total,
        "ratio": round(passed / total, 3) if total else 0.0,
        "failures": failures,
        "critical_failed": critical_failed,
    }


# --- 2. model-assisted rubric (judge panel) ----------------------------------
_RUBRIC_PROMPT = """You are grading a proposed AGENT SKILL — a reusable, named instruction
set an AI agent can invoke to perform a repeatable task. Grade it against the
SOURCE CONTEXT it was built from; do not rely on outside knowledge for groundedness.

Skill name: {name}
Skill description (the signal an agent uses to decide whether to invoke it):
{description}

When to use (triggers): {triggers}
When NOT to use (negative controls): {anti_triggers}
Declared tools: {tools}

Instructions:
{instructions}

Stated success criteria: {criteria}

Source context the skill must stay grounded in (excerpts):
{context}

Score each dimension from 0.0 to 1.0:
- outcome:      Would following these instructions actually accomplish the task the
                description promises? (task completion)
- process:      Are the steps correct, ordered, and do they invoke the right tools?
- style:        Is the skill clear, well-scoped, and conventional (good name +
                description, crisp instructions, sensible triggers/anti-triggers)?
- efficiency:   Does it avoid unnecessary steps, redundancy, and wasted tool calls?
- groundedness: Are the claims/steps supported by the source context above, with no
                invented facts? If the context is "(none provided)", judge internal
                consistency instead.

Return ONLY JSON:
{{"outcome": <float>, "process": <float>, "style": <float>, "efficiency": <float>,
"groundedness": <float>, "reason": "<one sentence>"}}"""

_RUBRIC_KEYS = ("outcome", "process", "style", "efficiency", "groundedness")


def _gen(model, prompt: str) -> str:
    msg = model.invoke(prompt)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _parse_scores(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for k in _RUBRIC_KEYS:
        try:
            out[k] = max(0.0, min(1.0, float(obj.get(k))))
        except (TypeError, ValueError):
            continue
    if not out:
        return None
    out["reason"] = str(obj.get("reason", ""))[:300]
    return out


def _context_digest(skill: dict, *, max_chars: int = 1600) -> str:
    prov = skill.get("provenance") or {}
    digest = prov.get("context_digest") or prov.get("context") or ""
    if not digest and prov.get("source_titles"):
        digest = "Sources: " + "; ".join(prov["source_titles"][:8])
    return (digest or "")[:max_chars]


def _judge_one(provider: str, model: str, skill: dict) -> dict:
    """Run one panel judge over the skill; returns per-dimension scores + meta."""
    crit = skill.get("success_criteria") or {}
    prompt = _RUBRIC_PROMPT.format(
        name=skill.get("name", ""),
        description=skill.get("description", "") or "(none)",
        triggers="; ".join(skill.get("triggers", []) or []) or "(none)",
        anti_triggers="; ".join(skill.get("anti_triggers", []) or []) or "(none)",
        tools=", ".join(skill.get("tools", []) or []) or "(none)",
        instructions=(skill.get("instructions", "") or "(none)")[:4000],
        criteria=json.dumps(crit, ensure_ascii=False) if crit else "(none)",
        context=_context_digest(skill) or "(none provided)",
    )
    try:
        judge = build_chat_model(provider, model, max_tokens=_JUDGE_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001
        return {"judge": f"{provider}/{model}", "error": str(exc), "scores": None}
    scores = None
    reason = ""
    for _ in range(2):  # one retry: flaky endpoints intermittently return empty
        try:
            raw = _gen(judge, prompt)
        except Exception as exc:  # noqa: BLE001
            reason = f"judge error: {type(exc).__name__}"
            continue
        parsed = _parse_scores(raw)
        if parsed:
            reason = parsed.pop("reason", "")
            scores = parsed
            break
    if scores is None:
        return {"judge": f"{provider}/{model}", "error": reason or "unparseable", "scores": None}
    overall = round(sum(scores.values()) / len(scores), 3)
    return {"judge": f"{provider}/{model}", "scores": scores, "overall": overall, "reason": reason}


def run_rubric_panel(skill: dict, *, provider: str = "auto", model: str | None = None,
                     judge_provider: str | None = None, judge_model: str | None = None,
                     max_judges: int = 3) -> dict | None:
    """Score a skill with a cross-family judge panel. Returns None when no provider
    is configured (so the caller can fall back to deterministic-only gating)."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        return None
    if judge_provider:
        jp, jm, _ = resolve_judge(rp, rm, judge_provider, judge_model)
        panel = [(jp, jm)]
    else:
        panel = judge_panel(rp, max_judges=max_judges) or [resolve_judge(rp, rm)[:2]]
    panel = panel[:max_judges]

    judges = [_judge_one(jp, jm, skill) for jp, jm in panel]
    valid = [j for j in judges if j.get("scores")]
    if not valid:
        return {"judges": judges, "panel": [f"{p}/{m}" for p, m in panel],
                "per_dimension": {}, "mean": None, "error": "no judge returned a score"}
    per_dim = {}
    for key in _RUBRIC_KEYS:
        vals = [j["scores"][key] for j in valid if key in j["scores"]]
        if vals:
            per_dim[key] = round(sum(vals) / len(vals), 3)
    overall = [j["overall"] for j in valid]
    mean = round(sum(overall) / len(overall), 3) if overall else None
    return {
        "judges": judges,
        "panel": [f"{p}/{m}" for p, m in panel],
        "per_dimension": per_dim,
        "per_judge_overall": {j["judge"]: j["overall"] for j in valid},
        "mean": mean,
    }


# --- 3. the accept/reject gate ----------------------------------------------
def decide_gate(deterministic: dict, rubric: dict | None) -> dict:
    """Combine deterministic + rubric signals into accept / reject / review.

    - Any failed *critical* deterministic check, or a deterministic pass-ratio
      below ``DET_PASS`` → ``reject``.
    - Otherwise, with a rubric: mean ≥ ``GATE_ACCEPT`` → ``accept``;
      mean < ``GATE_REJECT`` → ``reject``; in between → ``review``.
    - Without a rubric (no provider): a clean deterministic pass → ``review``
      (a human must judge quality), else ``reject``.
    A ``reject`` here is not final — the skill is kept and can be revised — but it
    never reaches ``pending_review`` automatically.
    """
    reasons = []
    det_ratio = deterministic.get("ratio", 0.0)
    crit_failed = deterministic.get("critical_failed") or []
    if crit_failed:
        reasons.append(f"critical checks failed: {', '.join(crit_failed)}")
        return {"gate": "reject", "reasons": reasons, "rubric_mean": (rubric or {}).get("mean"),
                "deterministic_ratio": det_ratio}
    if det_ratio < DET_PASS:
        reasons.append(f"deterministic pass-ratio {det_ratio} < {DET_PASS}")
        return {"gate": "reject", "reasons": reasons, "rubric_mean": (rubric or {}).get("mean"),
                "deterministic_ratio": det_ratio}

    mean = (rubric or {}).get("mean")
    if mean is None:
        reasons.append("deterministic checks passed; no rubric judge available — needs human review")
        return {"gate": "review", "reasons": reasons, "rubric_mean": None,
                "deterministic_ratio": det_ratio}
    if mean >= GATE_ACCEPT:
        reasons.append(f"rubric mean {mean} ≥ {GATE_ACCEPT} and deterministic clean")
        gate = "accept"
    elif mean < GATE_REJECT:
        reasons.append(f"rubric mean {mean} < {GATE_REJECT}")
        gate = "reject"
    else:
        reasons.append(f"rubric mean {mean} between {GATE_REJECT} and {GATE_ACCEPT} — borderline")
        gate = "review"
    return {"gate": gate, "reasons": reasons, "rubric_mean": mean, "deterministic_ratio": det_ratio}


def run_eval(skill: dict, *, provider: str = "auto", model: str | None = None,
             judge_provider: str | None = None, judge_model: str | None = None,
             run_rubric: bool = True, max_judges: int = 3) -> dict:
    """Full skill evaluation: deterministic checks + rubric panel + gate decision.

    Returns a report ready to attach with ``skill_library.record_eval``."""
    deterministic = run_deterministic(skill)
    rubric = None
    if run_rubric:
        rubric = run_rubric_panel(skill, provider=provider, model=model,
                                  judge_provider=judge_provider, judge_model=judge_model,
                                  max_judges=max_judges)
    decision = decide_gate(deterministic, rubric)
    return {
        "deterministic": deterministic,
        "rubric": rubric,
        "gate": decision["gate"],
        "gate_reasons": decision["reasons"],
        "rubric_mean": decision.get("rubric_mean"),
        "thresholds": {"accept": GATE_ACCEPT, "reject": GATE_REJECT, "det_pass": DET_PASS},
    }
