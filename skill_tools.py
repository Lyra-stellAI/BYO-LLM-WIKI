"""LangChain tools that let the curating agent discover the library's agent skills.

Tool docstrings double as the descriptions the model sees. These are read-only:
they let the agent find and inspect *accepted* skills (the reusable competences in
layer 7) so it can apply one to a task, but authoring a skill is a deliberate,
human-gated pipeline (``skill_agent``), not something the agent does mid-answer.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

import skill_library as skills


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _brief(s: dict) -> dict:
    return {
        "id": s.get("id"), "name": s.get("name"), "description": s.get("description"),
        "status": s.get("status"), "tools": s.get("tools"),
        "triggers": s.get("triggers"), "anti_triggers": s.get("anti_triggers"),
    }


@tool(parse_docstring=True)
def skill_recall(query: str, limit: int = 4) -> str:
    """Find accepted agent skills whose purpose matches a task you are about to do.

    Call this when a request looks like a repeatable task — there may be a
    proven, human-approved skill that already knows how to do it. Returns the
    best-matching skills' names, descriptions, triggers and tools.

    Args:
        query: A short description of the task or situation you face.
        limit: Maximum number of skills to return.
    """
    rows = skills.recall(query, k=max(1, min(limit, 10)))
    return _dump([_brief(s) for s in rows])


@tool(parse_docstring=True)
def skill_get(name_or_id: str) -> str:
    """Get the full instructions of an agent skill (its steps, triggers, and tools).

    Use after skill_recall to read the skill you intend to follow.

    Args:
        name_or_id: The skill's name, slug, or id (e.g. "Draft Release Notes" or "skill_ab12...").
    """
    s = skills.get_skill(name_or_id)
    if not s:
        return _dump({"error": f"No skill matching {name_or_id!r}."})
    return _dump({
        "id": s.get("id"), "name": s.get("name"), "description": s.get("description"),
        "status": s.get("status"), "instructions": s.get("instructions"),
        "steps": s.get("steps"), "triggers": s.get("triggers"),
        "anti_triggers": s.get("anti_triggers"), "tools": s.get("tools"),
        "success_criteria": s.get("success_criteria"),
    })


@tool(parse_docstring=True)
def skill_list() -> str:
    """List the accepted, ready-to-use agent skills in the library (name + description)."""
    rows = skills.list_skills(status=skills.ACCEPTED, limit=50)
    return _dump([{"id": s["id"], "name": s["name"], "description": s["description"]} for s in rows])


_READ_TOOLS = [skill_recall, skill_get, skill_list]


def read_tools() -> list:
    """Read-only skill-discovery tools (safe for any agent mode)."""
    return list(_READ_TOOLS)


def all_tools() -> list:
    return list(_READ_TOOLS)
