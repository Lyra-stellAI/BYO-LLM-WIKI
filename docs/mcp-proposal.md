# MCP integration for BYO-WIKI

Connect the library's agents to external tools/data via the **Model Context
Protocol (MCP)** — and expose the library itself as an MCP server. Defaults stay
local-first: nothing is enabled until you opt in.

## Two directions

1. **Client** — BYO-WIKI's agents connect *out* to MCP servers (Supabase, GitHub,
   fetch, …) to ground answers/skills in live data and to act.
2. **Server** — BYO-WIKI exposes its own functions (`kg_search`, `rag_answer`,
   `memory_recall`, `skill_recall`, `skill_build`, …) as an MCP server so Claude
   Desktop/Code/Cursor — or BYO-WIKI's own client — can use the library as a tool.

## Modules
- `mcp_config.py` — server registry (built-in catalog + optional `mcp.json` via
  `MCP_CONFIG`), env-var resolution (`${VAR}` / `${VAR:-default}`), opt-in via
  `MCP_ENABLED`, and read/write tool classification.
- `mcp_tools.py` — loads tools from enabled servers (`langchain-mcp-adapters`),
  splits read vs write, gates writes, and ingests MCP output into the KG. Bridges
  the async MCP client to BYO-WIKI's sync Flask/CLI.
- `mcp_server.py` — BYO-WIKI as an MCP server (FastMCP); run with
  `python runner.py --mode mcp-serve`.

## Where MCP tools attach
- Curating agent (`agent.py`) — read tools added to its toolset.
- Skill-builder loop (`skill_runtime.builder_tools`) — the author can pull live
  context (e.g. query Supabase, fetch a page) while drafting a skill.
- Ingestion — `mcp_tools.ingest_result` stages an MCP read tool's output as KG
  chunks (`/api/mcp/ingest`, `runner.py --mode mcp-ingest`).

## Security / gating
- **Read-only + project-scoped by default** for Supabase (`read_only=true`,
  `project_ref`, `features=database,docs`).
- **Writes deny-by-default**: a write tool is never handed to a free-running agent.
  It runs only via `execute_write`, which requires `MCP_ALLOW_WRITES=1` **and** an
  explicit human `approved=true` (the human-in-the-loop gate; `/api/mcp/write`,
  `runner.py --mode mcp-call --confirm`).
- **Untrusted data**: MCP results (DB rows, issues, pages) are untrusted input —
  the agent treats them as data, not instructions (prompt-injection caution).
- **Secrets from env only** — the registry stores var *names*, never values.
- **stdio servers get a clean env** — pass needed vars explicitly in the server's
  `env` (e.g. `KG_DATA_DIR` for BYO-WIKI's own server).

## Egress note
The hosted Supabase MCP is HTTPS (`https://mcp.supabase.com/mcp`), so it's
reachable even where direct Postgres ports are blocked — MCP is the better way to
reach Supabase from a locked-down environment.

## Config (env)
| Variable | Purpose |
| --- | --- |
| `MCP_ENABLED` | comma list of servers to enable (e.g. `supabase,fetch`). |
| `MCP_ALLOW_WRITES` | master switch for write tools (still needs per-call approval). |
| `MCP_CONFIG` | path to an `mcp.json` to add/override servers. |
| `SUPABASE_ACCESS_TOKEN` / `SUPABASE_PROJECT_REF` | Supabase MCP auth + scope. |
| `GITHUB_API_KEY` | GitHub MCP auth. |

## Status
Implemented: client core, skill-builder + ingestion hooks, the MCP server, and
gated writes. Validated locally end-to-end (our MCP client ↔ our MCP server over
stdio). External servers (Supabase/GitHub/fetch) are config + egress + credentials
only — no code changes needed to enable them.
