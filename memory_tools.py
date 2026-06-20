"""LangChain tools that let the curating agent use the library's memory layer.

Tool docstrings double as the descriptions the model sees. Read tools let the
agent recall what the library already learned before answering or maintaining;
the write tool lets it record durable learnings, gaps, and corrections so the
library improves across sessions instead of forgetting between runs.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

import memory


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@tool(parse_docstring=True)
def memory_recall(query: str, kinds: str = "", limit: int = 6) -> str:
    """Recall what the library already learned that is relevant to a query.

    Call this BEFORE answering or maintaining so you build on prior sessions
    (known facts, durable answers, user preferences, recorded gaps, corrections)
    instead of starting from scratch or repeating work.

    Args:
        query: What you want to remember about (a question or topic).
        kinds: Optional comma-separated kinds to restrict to: fact, answer,
            preference, gap, correction, observation.
        limit: Maximum number of memories to return.
    """
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] or None
    rows = memory.recall(query, k=max(1, min(limit, 25)), kinds=kind_list)
    return _dump(rows)


@tool(parse_docstring=True)
def memory_write(text: str, kind: str = "fact", salience: int = 3, tags: str = "") -> str:
    """Record a durable memory so the library remembers it next session.

    Use for things worth carrying forward: a confirmed fact, a durable answer, a
    user goal/preference, a known gap to fill, or a correction. Near-duplicate
    memories are reinforced rather than duplicated, so it is safe to re-assert.

    Args:
        text: The memory content, as a self-contained sentence.
        kind: One of fact, answer, preference, gap, correction, observation.
        salience: Importance from 1 (minor) to 5 (central).
        tags: Optional comma-separated tags.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    rec = memory.remember(text, kind=kind, salience=salience, tags=tag_list,
                          confidence="EXTRACTED", origin="agent")
    return _dump({"ok": bool(rec), "memory": rec})


_READ_TOOLS = [memory_recall]
_WRITE_TOOLS = [memory_write]


def read_tools() -> list:
    """Recall-only (for read-only query mode)."""
    return list(_READ_TOOLS)


def all_tools() -> list:
    """Recall + write (for ingest / maintain modes)."""
    return list(_READ_TOOLS) + list(_WRITE_TOOLS)
