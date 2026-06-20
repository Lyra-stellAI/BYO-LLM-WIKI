"""Tool-use runtime for the skill author — let the generator act, not just write.

Anthropic's improved skill-creator and Hermes' loop both make the same point: a
good skill comes from an agent that *acts with tools and verifies*, not a single
blind completion. This module gives the skill author (the ``codeact`` phase) a
small set of tools and a native tool-calling loop so it can, while drafting:

  * ``read_context``        — pull the exact parts of the source context it needs
  * ``check_draft``         — run the deterministic checks on a candidate and see
                              what fails (test → measure → refine, in the loop)
  * ``list_existing_skills``— avoid duplicating a skill that already exists
  * ``recall_memory``       — reuse durable learnings (the memory↔skill partnership)

The loop uses LangChain's ``bind_tools`` so it works with any tool-capable provider
(Claude, GPT, …); it degrades gracefully — if the model never calls a tool it just
returns its final answer. Token usage is accumulated for the observability store.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import memory as _memory
import skill_eval
import skill_library as _skills


def usage_of(msg) -> int:
    """Best-effort total-token count from a LangChain message (0 if unknown)."""
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict) and um.get("total_tokens"):
        return int(um["total_tokens"])
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(tu, dict):
        if tu.get("total_tokens"):
            return int(tu["total_tokens"])
        return int(tu.get("input_tokens", 0)) + int(tu.get("output_tokens", 0))
    return 0


def _text(msg) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def builder_tools(bundle: dict) -> list:
    """Build the author's tools, closed over the gathered context ``bundle``."""
    context = bundle.get("context", "")
    provenance = bundle.get("provenance", {})

    @tool(parse_docstring=True)
    def read_context(query: str) -> str:
        """Search the SELECTED CONTEXT for passages relevant to a query.

        Use this to ground a step or claim in the source material before writing it.

        Args:
            query: Words describing what you are looking for in the context.
        """
        terms = [t for t in query.lower().split() if len(t) >= 3]
        hits = []
        for para in context.split("\n\n---\n\n"):
            low = para.lower()
            if any(t in low for t in terms) or not terms:
                hits.append(para[:600])
            if len(hits) >= 4:
                break
        return "\n\n".join(hits) if hits else "(no matching passage; ground only in the provided context)"

    @tool(parse_docstring=True)
    def check_draft(draft_json: str) -> str:
        """Run the deterministic skill checks on a CANDIDATE skill and report failures.

        Call this on your draft BEFORE finalizing, then fix whatever fails. The
        draft is a JSON object with name, description, instructions, steps,
        triggers, anti_triggers, tools, and tests (a positive and a negative case).

        Args:
            draft_json: The candidate skill as a JSON object string.
        """
        try:
            draft = json.loads(draft_json)
        except Exception:  # noqa: BLE001
            return json.dumps({"error": "draft_json was not valid JSON"})
        if not isinstance(draft, dict):
            return json.dumps({"error": "draft must be a JSON object"})
        draft.setdefault("provenance", provenance)  # so the grounding check can pass
        det = skill_eval.run_deterministic(draft)
        tips = {
            "has_steps": "Add at least 2 explicit numbered steps.",
            "has_anti_triggers": "Add at least one 'when NOT to use' negative control.",
            "has_positive_and_negative_tests": "Add one test with should_trigger true and one false.",
            "no_placeholders": "Remove TODO/TBD/<...> placeholders.",
            "has_description": "Write a fuller one-sentence description (>= 5 words).",
        }
        return json.dumps({
            "passed": det["passed"], "total": det["total"], "ratio": det["ratio"],
            "failures": det["failures"],
            "fix": [tips.get(f, f"address: {f}") for f in det["failures"]],
        }, ensure_ascii=False)

    @tool(parse_docstring=True)
    def list_existing_skills() -> str:
        """List skills already in the library, so you do not duplicate one. Returns
        each skill's name and description."""
        rows = _skills.list_skills(limit=50)
        return json.dumps([{"name": s["name"], "description": s.get("description", ""),
                            "status": s.get("status")} for s in rows], ensure_ascii=False)

    @tool(parse_docstring=True)
    def recall_memory(query: str) -> str:
        """Recall durable learnings (facts, preferences, gaps) relevant to the skill.

        Args:
            query: What to remember about (a topic or task).
        """
        try:
            rows = _memory.recall(query, k=5)
        except Exception:  # noqa: BLE001
            return "[]"
        return json.dumps([{"kind": r.get("kind"), "text": r.get("text")} for r in rows],
                          ensure_ascii=False)

    return [read_context, check_draft, list_existing_skills, recall_memory]


def run_tool_loop(chat, tools: list, system: str, user: str, *, max_iters: int = 6) -> dict:
    """Drive a native tool-calling loop and return {text, tokens, tool_calls, iters}.

    Binds ``tools`` to the chat model, then alternates model turns and tool
    execution until the model stops calling tools (or ``max_iters`` is hit)."""
    bound = chat.bind_tools(tools) if hasattr(chat, "bind_tools") else chat
    by_name = {t.name: t for t in tools}
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    tokens = 0
    calls = 0
    last_text = ""
    iters = 0
    for iters in range(1, max_iters + 1):
        ai = bound.invoke(messages)
        tokens += usage_of(ai)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            last_text = _text(ai)
            break
        for tc in tool_calls:
            calls += 1
            name = tc.get("name")
            args = tc.get("args") or {}
            t = by_name.get(name)
            try:
                result = t.invoke(args) if t else f"unknown tool: {name}"
            except Exception as exc:  # noqa: BLE001
                result = f"tool error: {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc.get("id", name)))
        last_text = _text(ai)  # keep the latest assistant text in case we hit the cap
    return {"text": last_text, "tokens": tokens, "tool_calls": calls, "iters": iters}
