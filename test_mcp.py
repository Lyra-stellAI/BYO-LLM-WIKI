"""Tests for the MCP layer: server registry, write gating, and a local client↔server
loop (BYO-WIKI's own MCP server spawned over stdio and called by its MCP client).

The loop test needs `langchain-mcp-adapters` + `mcp` installed; it is skipped if not.
Runs offline — no external network — by talking to our own `mcp_server.py`.
"""

import importlib.util
import json
import os
import sys
import tempfile

os.environ.setdefault("KG_DATA_DIR", tempfile.mkdtemp(prefix="mcp_test_"))
os.environ.pop("OPENAI_API_KEY", None)

import knowledge_graph as kg  # noqa: E402
import mcp_config  # noqa: E402
import mcp_tools  # noqa: E402

_HAS_MCP = (importlib.util.find_spec("langchain_mcp_adapters") is not None
            and importlib.util.find_spec("mcp") is not None)


def _restore(saved):
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# --- registry + env resolution ----------------------------------------------
def test_enabled_and_spec_resolution():
    keys = ("MCP_ENABLED", "SUPABASE_ACCESS_TOKEN", "SUPABASE_PROJECT_REF")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["MCP_ENABLED"] = "supabase"
        os.environ.pop("SUPABASE_ACCESS_TOKEN", None)
        os.environ.pop("SUPABASE_PROJECT_REF", None)
        # missing required vars -> not configurable -> not in client_config
        assert mcp_config.client_config() == {}
        st = mcp_config.status()["servers"]["supabase"]
        assert st["enabled"] and not st["configurable"]
        assert set(st["missing_vars"]) == {"SUPABASE_ACCESS_TOKEN", "SUPABASE_PROJECT_REF"}
        # provide them -> spec resolves with the values substituted
        os.environ["SUPABASE_ACCESS_TOKEN"] = "demo-token"
        os.environ["SUPABASE_PROJECT_REF"] = "abc123"
        spec = mcp_config.server_spec("supabase")
        assert spec["transport"] == "streamable_http"
        assert "project_ref=abc123" in spec["url"] and "read_only=true" in spec["url"]
        assert spec["headers"]["Authorization"] == "Bearer demo-token"
        assert "supabase" in mcp_config.client_config()
    finally:
        _restore(saved)


def test_write_tool_classification():
    # supabase is read-only -> even execute_sql is not a write
    assert mcp_config.is_write_tool("supabase", "execute_sql") is False
    assert mcp_config.is_write_tool("supabase", "list_tables") is False
    # github is writable -> mutating verbs are writes, reads are not
    assert mcp_config.is_write_tool("github", "create_issue") is True
    assert mcp_config.is_write_tool("github", "merge_pull_request") is True
    assert mcp_config.is_write_tool("github", "list_issues") is False
    assert mcp_config.is_write_tool("github", "get_file_contents") is False


def test_supabase_writability_tracks_read_only():
    """The client gate must follow the server's read_only scope, not drift from it:
    opening the server for writes (SUPABASE_MCP_READ_ONLY=false) must re-arm the
    confirm-gate on mutating tools."""
    saved = {"SUPABASE_MCP_READ_ONLY": os.environ.get("SUPABASE_MCP_READ_ONLY")}
    try:
        os.environ["SUPABASE_MCP_READ_ONLY"] = "true"
        assert mcp_config.is_writable_server("supabase") is False
        assert mcp_config.is_write_tool("supabase", "execute_sql") is False
        os.environ["SUPABASE_MCP_READ_ONLY"] = "false"
        assert mcp_config.is_writable_server("supabase") is True
        assert mcp_config.is_write_tool("supabase", "execute_sql") is True
        assert mcp_config.is_write_tool("supabase", "apply_migration") is True
        # reads stay reads even when the server is writable
        assert mcp_config.is_write_tool("supabase", "list_tables") is False
        assert mcp_config.status()["servers"]["supabase"]["writable"] is True
    finally:
        _restore(saved)


# --- write gating ------------------------------------------------------------
def test_execute_write_is_gated():
    saved = {"MCP_ALLOW_WRITES": os.environ.get("MCP_ALLOW_WRITES")}
    try:
        os.environ.pop("MCP_ALLOW_WRITES", None)
        # a read tool routed to the write path is rejected
        r0 = mcp_tools.execute_write("github", "list_issues", {}, approved=True)
        assert r0["ok"] is False and "not a write tool" in r0["error"]
        # writes disabled by default
        r1 = mcp_tools.execute_write("github", "create_issue", {"title": "x"}, approved=True)
        assert r1["ok"] is False and "disabled" in r1["error"]
        # enabled but unapproved -> approval_required + preview (no execution)
        os.environ["MCP_ALLOW_WRITES"] = "1"
        r2 = mcp_tools.execute_write("github", "create_issue", {"title": "x"}, approved=False)
        assert r2.get("approval_required") is True and r2["preview"]["tool"] == "create_issue"
    finally:
        _restore(saved)


def test_status_without_servers_is_safe():
    saved = {"MCP_ENABLED": os.environ.get("MCP_ENABLED")}
    try:
        os.environ.pop("MCP_ENABLED", None)
        mcp_tools.reset()
        s = mcp_tools.status()
        assert s["active"] == [] and isinstance(s["available"], bool)
        assert mcp_tools.read_tools() == []
    finally:
        _restore(saved)


# --- local client <-> server loop (no external network) ----------------------
def test_local_client_server_loop():
    if not _HAS_MCP:
        print("  (skipped: langchain-mcp-adapters/mcp not installed)")
        return
    kg.clear("current")
    kg.clear("overall")
    kg.add_chunk("Deep agents plan, call tools, and write files to a workspace.",
                 source_title="Deep Agents")
    kg.integrate()

    repo = os.path.dirname(os.path.abspath(__file__))
    cfg = {"servers": {"wiki": {
        "transport": "stdio", "command": sys.executable,
        "args": [os.path.join(repo, "mcp_server.py")],
        # stdio servers get a clean env — pass our data dir + PATH through explicitly.
        "env": {"KG_DATA_DIR": "${KG_DATA_DIR}", "PATH": "${PATH}"},
        "writable": False}}}
    cfg_path = os.path.join(os.environ["KG_DATA_DIR"], "mcp.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    saved = {k: os.environ.get(k) for k in ("MCP_CONFIG", "MCP_ENABLED")}
    try:
        os.environ["MCP_CONFIG"] = cfg_path
        os.environ["MCP_ENABLED"] = "wiki"
        mcp_tools.reset()
        tools = mcp_tools.read_tools()
        names = {t.name for t in tools}
        assert "kg_search" in names and "skill_recall" in names, names
        # call our own MCP server's kg_search through the MCP client
        out = mcp_tools.call_tool("wiki", "kg_search", {"query": "deep agents"})
        assert "deep" in out.lower() or "entity" in out.lower(), out
    finally:
        _restore(saved)
        mcp_tools.reset()


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} mcp tests passed.")


if __name__ == "__main__":
    _run_all()
