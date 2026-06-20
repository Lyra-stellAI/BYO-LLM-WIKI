"""BYO-WIKI as an MCP *server* — expose the library's tools to any MCP client.

Wraps the existing read functions (knowledge graph, RAG, memory, skills) — and one
gated build action — as MCP tools so Claude Desktop / Claude Code / Cursor (or any
MCP client, including BYO-WIKI's own ``mcp_tools``) can query and grow your personal
library. Run it over stdio:

    python mcp_server.py            # stdio (for Claude Desktop/Code config)
    python runner.py --mode mcp-serve

The tools are thin adapters over the same modules the web app and agents use, so
there's a single source of truth. Read tools are always safe; ``skill_build`` is a
*proposing* action — it drafts + evaluates a skill into ``pending_review`` and never
auto-accepts, so a human still gates it via the normal review queue.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

import knowledge_graph as kg
import memory
import skill_library

mcp = FastMCP("byo-wiki")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@mcp.tool()
def kg_search(query: str, limit: int = 10) -> str:
    """Search the knowledge graph for nodes (entities/chunks/topics/...) matching a query."""
    return _dump(kg.search_nodes(query, where="overall", limit=max(1, min(limit, 50))))


@mcp.tool()
def kg_get_entity(name_or_id: str) -> str:
    """Get a full entity profile: kind, aliases, summary, relations, and citing chunks."""
    ent = kg.get_entity(name_or_id, where="overall")
    return _dump(ent or {"error": f"No entity matching {name_or_id!r}."})


@mcp.tool()
def kg_stats() -> str:
    """Counts of every layer in the knowledge library."""
    return _dump(kg.stats())


@mcp.tool()
def rag_answer(question: str, k: int = 6) -> str:
    """Answer a question grounded in the contextual RAG library, with citations.

    Requires the vector library to be built and OPENAI_API_KEY for embeddings."""
    try:
        import rag
        return _dump(rag.answer(question, k=k))
    except Exception as exc:  # noqa: BLE001
        return _dump({"error": f"RAG unavailable: {exc}"})


@mcp.tool()
def memory_recall(query: str, limit: int = 6) -> str:
    """Recall durable cross-session memories (facts, answers, preferences, gaps)."""
    return _dump(memory.recall(query, k=max(1, min(limit, 25))))


@mcp.tool()
def skill_recall(query: str, limit: int = 5) -> str:
    """Find accepted, ready-to-use agent skills matching a task description."""
    return _dump(skill_library.recall(query, k=max(1, min(limit, 10))))


@mcp.tool()
def skill_list() -> str:
    """List accepted agent skills in the library (name + description)."""
    rows = skill_library.list_skills(status=skill_library.ACCEPTED, limit=50)
    return _dump([{"id": s["id"], "name": s["name"], "description": s["description"]} for s in rows])


@mcp.tool()
def skill_get(name_or_id: str) -> str:
    """Get a skill's full instructions/steps/triggers (after skill_recall)."""
    s = skill_library.get_skill(name_or_id)
    if not s:
        return _dump({"error": f"No skill matching {name_or_id!r}."})
    return _dump({k: s.get(k) for k in
                  ("id", "name", "description", "status", "instructions", "steps",
                   "triggers", "anti_triggers", "tools")})


@mcp.tool()
def skill_build(text: str = "", query: str = "", goal: str = "") -> str:
    """Draft + evaluate a NEW agent skill from context (KG query or pasted text).

    This is a *proposing* action: the skill is gated to pending_review and is NOT
    auto-accepted — a human approves it in the BYO-WIKI review queue. Returns the
    gate decision and the new skill id."""
    try:
        import skill_agent
        res = skill_agent.build_skill(text=text or None, query=query or None,
                                      goal=goal, run_rubric=False)
        s = res["skill"]
        return _dump({"ok": True, "skill_id": s["id"], "name": s["name"],
                      "status": s["status"], "gate": res["gate"],
                      "note": "Pending human review in BYO-WIKI."})
    except Exception as exc:  # noqa: BLE001
        return _dump({"error": str(exc)})


def main() -> None:
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
