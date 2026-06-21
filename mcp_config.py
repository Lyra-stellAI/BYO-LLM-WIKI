"""MCP server registry + config (BYO-WIKI as an MCP *client*).

Defines the external MCP servers the agents may connect to. Servers are **opt-in**
via ``MCP_ENABLED`` (comma-separated); with nothing set, no servers are enabled and
the app stays local-first. Specs reference env-var *names* (resolved at load time),
never values, so no secret lands in the repo. Mirrors ``providers.py``'s table.

A user can also drop an ``mcp.json`` (path via ``MCP_CONFIG``) to add/override
servers without touching code.

Tool classification: a tool is a *write* if its server is marked ``writable`` AND
the tool name matches a mutating verb. Read tools are safe for any agent; write
tools are gated (deny-by-default; ``MCP_ALLOW_WRITES`` + explicit human approval).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}

# Built-in catalog. Enable with e.g. MCP_ENABLED=supabase,fetch
DEFAULT_SERVERS: dict[str, dict] = {
    # Hosted Supabase MCP over HTTPS — reachable even where direct Postgres is
    # blocked. read_only + project-scoped + limited feature groups by default.
    "supabase": {
        "transport": "streamable_http",
        "url": ("https://mcp.supabase.com/mcp?project_ref=${SUPABASE_PROJECT_REF}"
                "&read_only=${SUPABASE_MCP_READ_ONLY:-true}"
                "&features=${SUPABASE_MCP_FEATURES:-database,docs}"),
        "headers": {"Authorization": "Bearer ${SUPABASE_ACCESS_TOKEN}"},
        "writable": False,  # fallback; real value tracks read_only= (is_writable_server)
        "requires": ["SUPABASE_ACCESS_TOKEN", "SUPABASE_PROJECT_REF"],
    },
    "github": {
        "transport": "streamable_http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer ${GITHUB_API_KEY}"},
        "writable": True,
        "requires": ["GITHUB_API_KEY"],
    },
    # Reference fetch server (stdio); needs `uvx` (uv) in the runtime.
    "fetch": {
        "transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"],
        "writable": False, "requires": [],
    },
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_WRITE_RE = re.compile(
    r"(?:^|_)(create|update|delete|drop|insert|apply|merge|deploy|write|push|"
    r"remove|edit|rebase|reset|restore|pause|execute_sql|run)(?:_|$)", re.I)


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in _TRUTHY


def allow_writes() -> bool:
    """Master switch for write MCP tools (still requires per-call human approval)."""
    return _truthy(os.environ.get("MCP_ALLOW_WRITES"))


def _catalog() -> dict[str, dict]:
    """Built-in servers merged with an optional MCP_CONFIG json file."""
    catalog = {k: dict(v) for k, v in DEFAULT_SERVERS.items()}
    path = os.environ.get("MCP_CONFIG")
    if path and Path(path).exists():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for name, spec in (data.get("servers") or data.get("mcpServers") or data).items():
                if isinstance(spec, dict):
                    catalog[name] = {**catalog.get(name, {}), **spec}
        except Exception:  # noqa: BLE001  (bad config -> ignore, log nothing secret)
            pass
    return catalog


def enabled() -> list[str]:
    raw = os.environ.get("MCP_ENABLED", "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _missing_vars(spec: dict) -> list[str]:
    return [v for v in spec.get("requires", []) if not os.environ.get(v)]


def _resolve(value):
    """Substitute ${VAR} / ${VAR:-default} in strings, dicts, and lists."""
    if isinstance(value, str):
        def repl(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def server_spec(name: str) -> dict | None:
    """Resolved connection spec for one server (for MultiServerMCPClient), or None
    if it is unknown or missing required env vars."""
    spec = _catalog().get(name)
    if not spec or _missing_vars(spec):
        return None
    out = {"transport": spec.get("transport", "stdio")}
    for key in ("url", "headers", "command", "args", "env"):
        if key in spec:
            out[key] = _resolve(spec[key])
    return out


def client_config() -> dict[str, dict]:
    """{name: spec} for every ENABLED + configurable server (input to the client)."""
    cfg = {}
    for name in enabled():
        spec = server_spec(name)
        if spec:
            cfg[name] = spec
    return cfg


def is_writable_server(name: str) -> bool:
    spec = _catalog().get(name)
    if not spec:
        return False
    # When an HTTP endpoint carries an explicit ``read_only`` scope, that scope is
    # the source of truth and overrides the static flag: a read-only server cannot
    # write (gating is moot) and a server opened for writes MUST be gated. This
    # keeps the client-side confirm-gate in lock-step with the server's own scope
    # (e.g. SUPABASE_MCP_READ_ONLY), so the two never drift apart.
    url = spec.get("url")
    if url:
        m = re.search(r"[?&]read_only=([^&]*)", _resolve(url))
        if m:
            return not _truthy(m.group(1))
    return bool(spec.get("writable"))


def is_write_tool(server: str, tool_name: str) -> bool:
    """A tool counts as a write if its server is writable AND the name is mutating."""
    if not is_writable_server(server):
        return False
    return bool(_WRITE_RE.search(tool_name or ""))


def status() -> dict:
    """Per-server config status (no secrets), for /api/mcp/status and the CLI."""
    cat = _catalog()
    en = set(enabled())
    servers = {}
    for name, spec in cat.items():
        missing = _missing_vars(spec)
        servers[name] = {
            "enabled": name in en,
            "configurable": not missing,
            "missing_vars": missing,
            "writable": is_writable_server(name),
            "transport": spec.get("transport", "stdio"),
        }
    return {
        "enabled": sorted(en),
        "active": sorted(client_config().keys()),
        "allow_writes": allow_writes(),
        "servers": servers,
    }
