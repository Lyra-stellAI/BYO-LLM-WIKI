"""MCP client — load external MCP tools and gate writes (BYO-WIKI as client).

Loads tools from the enabled MCP servers (``mcp_config``) via
``langchain-mcp-adapters`` so they become LangChain ``BaseTool`` objects that plug
into the curating agent and the skill-builder loop. Tools are split into:

  * READ tools  — handed to agents freely (``read_tools``).
  * WRITE tools — never auto-handed to a free-running agent; they run only through
    ``execute_write`` which is deny-by-default (``MCP_ALLOW_WRITES``) and requires an
    explicit human ``approved=True`` (the human-in-the-loop gate).

Everything degrades to a no-op when ``langchain-mcp-adapters`` isn't installed or no
servers are enabled, so the app stays local-first by default. The async MCP client
is bridged to BYO-WIKI's sync Flask/CLI via a small loop runner; loaded tools are
cached.
"""

from __future__ import annotations

import asyncio
import json
import threading

import mcp_config

_cache: dict = {"loaded": False, "pairs": []}  # pairs: list[(server, BaseTool)]


def available() -> bool:
    try:
        import langchain_mcp_adapters  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _run(coro):
    """Run a coroutine to completion from sync code (even inside a running loop)."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None and running.is_running():
        box: dict = {}
        def _worker():
            box["v"] = asyncio.run(coro)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if "err" in box:
            raise box["err"]
        return box.get("v")
    return asyncio.run(coro)


async def _aload(cfg: dict) -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(cfg)
    pairs = []
    for name in cfg:
        try:
            tools = await client.get_tools(server_name=name)
        except TypeError:  # older adapter: no per-server arg
            tools = await client.get_tools()
        except Exception:  # noqa: BLE001  (one bad server shouldn't kill the rest)
            tools = []
        for t in tools:
            pairs.append((name, t))
    return pairs


def _load(force: bool = False) -> list:
    if _cache["loaded"] and not force:
        return _cache["pairs"]
    cfg = mcp_config.client_config()
    if not available() or not cfg:
        _cache.update(loaded=True, pairs=[])
        return []
    try:
        pairs = _run(_aload(cfg))
    except Exception:  # noqa: BLE001
        pairs = []
    _cache.update(loaded=True, pairs=pairs)
    return pairs


def reset() -> None:
    _cache.update(loaded=False, pairs=[])


# --- read tools for agents ---------------------------------------------------
def read_tools() -> list:
    """Read-classified MCP tools — safe to give any agent. Best-effort/never raises."""
    try:
        return [t for (server, t) in _load() if not mcp_config.is_write_tool(server, t.name)]
    except Exception:  # noqa: BLE001
        return []


def tool_listing() -> list[dict]:
    """All loaded MCP tools with their server + write classification (for status)."""
    return [{"server": s, "tool": t.name, "description": (t.description or "")[:160],
             "write": mcp_config.is_write_tool(s, t.name)} for (s, t) in _load()]


# --- write tools (gated) -----------------------------------------------------
def _find(server: str, tool: str):
    for (s, t) in _load():
        if s == server and t.name == tool:
            return t
    return None


def call_tool(server: str, tool: str, args: dict) -> str:
    """Invoke a loaded MCP tool synchronously and return its text result."""
    t = _find(server, tool)
    if t is None:
        return json.dumps({"error": f"unknown MCP tool {server}/{tool}"})
    return str(_run(t.ainvoke(args or {})))


def execute_write(server: str, tool: str, args: dict, *, approved: bool = False) -> dict:
    """Run a WRITE MCP tool through the human gate.

    Deny-by-default: refuses unless ``MCP_ALLOW_WRITES`` is on AND ``approved`` is
    True. With ``approved=False`` it returns a preview + ``approval_required`` so a
    human (UI/CLI) can confirm before the mutation runs."""
    if not mcp_config.is_write_tool(server, tool):
        return {"ok": False, "error": f"{server}/{tool} is not a write tool (use call_tool)."}
    if not mcp_config.allow_writes():
        return {"ok": False, "error": "MCP writes are disabled. Set MCP_ALLOW_WRITES=1 to enable."}
    if not approved:
        return {"ok": False, "approval_required": True,
                "preview": {"server": server, "tool": tool, "args": args},
                "error": "Human approval required: re-send with approved=true."}
    try:
        return {"ok": True, "server": server, "tool": tool, "result": call_tool(server, tool, args)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def ingest_result(server: str, tool: str, args: dict, *, source_title: str = "",
                  chunk_size: int = 1200) -> dict:
    """Call a READ MCP tool and stage its output into the knowledge graph (staging).

    Turns external data (e.g. a Supabase ``list_tables`` / read-only ``execute_sql``,
    or a fetched page) into KG chunks the user can then Integrate. Refuses write tools."""
    if mcp_config.is_write_tool(server, tool):
        return {"ok": False, "error": "refusing to ingest from a write tool."}
    text = call_tool(server, tool, args)
    import knowledge_graph as kg
    import ingestion
    chunks = ingestion.chunk_text(text, chunk_size=chunk_size) or ([text] if text else [])
    if not chunks:
        return {"ok": False, "error": "MCP tool returned no content."}
    title = source_title or f"mcp:{server}/{tool}"
    total = len(chunks)
    for i, c in enumerate(chunks, 1):
        kg.add_chunk(c, source_title=(title if total == 1 else f"{title} [{i}/{total}]"),
                     tags=["mcp", server, tool])
    return {"ok": True, "chunks_staged": total, "source": title, "chars": len(text)}


def status() -> dict:
    out = mcp_config.status()
    out["available"] = available()
    if out["available"] and out.get("active"):
        try:
            out["tools"] = tool_listing()
        except Exception as exc:  # noqa: BLE001
            out["tools_error"] = str(exc)
    return out
