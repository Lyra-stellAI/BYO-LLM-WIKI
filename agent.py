"""Agent layer for the knowledge library (deepagents harness, local filesystem).

This mirrors the LangChain ``llm-wiki`` example's *harness and orchestration*
(``create_deep_agent`` + a filesystem workspace + ``init``/``ingest``/``query``/
``lint`` modes) but runs entirely on the local machine -- no LangSmith Sandbox,
no Context Hub. The agent reasons over a layered knowledge graph through the
tools in ``kg_tools`` and writes the human-readable synthesis layer (canonical
markdown pages) into a local ``wiki/`` workspace.

Two layers, two interfaces:
  * graph layer  (sources -> chunks -> entities -> topics)  via KG tools
  * wiki  layer  (canonical/synthesis/query markdown pages)  via filesystem

The runner (this module) owns the catalog (``wiki/index.md``) and the
append-only timeline (``log.md``); the agent never edits those directly.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import knowledge_graph as kg
import kg_tools
from providers import (
    ProviderError,
    agent_dependencies_available,
    build_chat_model,
    resolve_provider_model,
)

WORKSPACE_DIR = Path(os.environ.get("KG_DATA_DIR", "data")) / "library"
WIKI_SUBDIRS = ("topics", "entities", "synthesis", "query")

_RECURSION_LIMIT = int(os.environ.get("KG_AGENT_RECURSION_LIMIT", "80"))


class AgentError(RuntimeError):
    """Raised when an agent operation cannot run (missing deps/keys/etc.)."""


# --- Prompts -----------------------------------------------------------------
_BASE_SYSTEM_PROMPT = """You are the curator of a personal, long-lived knowledge library.

The library has two connected layers you are responsible for:
1. A LAYERED KNOWLEDGE GRAPH you reach ONLY through the kg_* tools:
   sources -> chunks (evidence) -> entities -> topics (themes) -> syntheses.
   - Entities are canonical, de-duplicated things (people, orgs, concepts, ...).
   - Topics are higher-level themes that group related entities (a hierarchy).
2. A WIKI of human-readable markdown pages under /wiki/ that you write with the
   filesystem tools (write_file/edit_file/read_file/ls). These unify disparate
   evidence into clear, canonical prose.

Operating principles:
- Ground every claim in the library. Use kg_search, kg_get_entity, kg_neighbors
  and kg_get_chunk to gather evidence before asserting anything.
- Compress by STRUCTURE, not by omission: prefer canonical entities/topics and
  cross-links over sprawling duplicate notes.
- De-duplicate aggressively: if two entities are the same thing, kg_merge_entities.
- Build hierarchy: group related entities under meaningful topics; nest topics
  when it clarifies. A flat pile of entities is a failure state.
- Keep uncertainty explicit; when sources conflict, say so rather than guessing.
- Cite evidence by chunk id and source title.

Filesystem rules:
- Write only under /wiki/. Never edit /log.md or /AGENTS.md (the runner owns them).
- Reshape the graph ONLY through kg_* tools, never by writing JSON files.
"""

_INGEST_PROMPT = """New material was just integrated into the library. Make it make sense.

Newly added or reinforced entities:
{new_entities}

Do the following, using the tools:
1. Inspect the new entities (kg_get_entity) and search for likely duplicates
   (kg_search). Merge duplicates into the best canonical name (kg_merge_entities).
2. Give each significant new entity a one-line, source-grounded summary
   (kg_set_entity_summary), reading chunks (kg_get_chunk) when you need evidence.
3. Add any clearly-stated relationships you find between entities (kg_add_relation)
   that are not yet recorded.
4. Organize: assign each significant entity to a meaningful topic
   (kg_assign_entity_to_topic), creating or reusing topics (kg_upsert_topic).
   Reuse existing topics (kg_list_topics) before inventing new ones.
5. If the new material forms a coherent theme worth a canonical note, write or
   update a short page at /wiki/topics/<slug>.md or /wiki/synthesis/<slug>.md
   summarizing it with citations.

Finish with a concise report:
## Merged
## Topics touched
## Pages written
## Notable relationships
"""

_QUERY_PROMPT = """Answer this question using ONLY the knowledge library:

QUESTION: {question}

Process:
1. Read /wiki/index.md (ls + read_file) to orient yourself.
2. Use kg_search / kg_list_topics / kg_list_entities to find relevant nodes.
3. Use kg_get_entity and kg_neighbors to follow relationships (multi-hop), and
   kg_get_chunk to read the underlying evidence you will cite.
4. Synthesize a grounded answer. If the library lacks the evidence, say so
   plainly and suggest what to ingest next -- do not invent facts.

Output format (markdown):
## Answer
<concise, well-structured answer>

## Citations
- <chunk_id> - <source title>: <what it supports>   (one bullet per evidence chunk)

## Confidence & gaps
<one or two sentences on confidence and what is missing>
"""

_LINT_PROMPT = """Run a maintenance / health-check pass over the WHOLE library and improve it.

Start by reading /wiki/index.md and calling kg_stats and kg_list_topics.

Reconcile and strengthen, using the tools:
1. DE-DUPLICATE: find entities that denote the same thing (search variants/
   spellings/abbreviations) and kg_merge_entities them into canonical names.
2. BUILD THE HIERARCHY: ensure every important entity belongs to a meaningful
   topic (kg_assign_entity_to_topic); create/rename/nest topics (kg_upsert_topic)
   so the topic layer is a clean, non-overlapping map of the library. Avoid
   dozens of tiny topics; prefer a handful of clear themes, nested when useful.
3. SUMMARIZE: ensure central entities and every topic have a crisp one-line
   summary (kg_set_entity_summary / kg_upsert_topic summary).
4. RELATE: add important missing relationships between entities (kg_add_relation).
5. SYNTHESIZE: for each major topic, write or refresh a canonical page at
   /wiki/topics/<slug>.md that unifies the evidence with citations; when a theme
   cuts across topics, write /wiki/synthesis/<slug>.md.
6. Note contradictions explicitly in the relevant page rather than hiding them.

Finish with a concise markdown report:
## Reconciled Changes
## Topic Map (after)
## Remaining Gaps
## Suggested Next Questions and Sources
"""


# --- Workspace scaffolding ---------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agents_md(topic: str) -> str:
    return (
        f"# {topic} — Knowledge Library\n\n"
        "Schema and rules for the curating agent.\n\n"
        "## Layers\n"
        "- **source** → **chunk** (evidence) → **entity** → **topic** (theme) → **synthesis**.\n"
        "- Reshape the graph only through `kg_*` tools; write prose only under `/wiki/`.\n\n"
        "## Conventions\n"
        "- Entities are canonical and de-duplicated; alternate names become aliases.\n"
        "- Topics group related entities and may nest; keep them few and meaningful.\n"
        "- `/wiki/topics/<slug>.md` — canonical theme pages.\n"
        "- `/wiki/synthesis/<slug>.md` — cross-cutting syntheses.\n"
        "- `/wiki/query/<slug>.md` — durable answers worth keeping.\n"
        "- `/wiki/index.md` and `/log.md` are runner-managed; do not edit them.\n"
    )


def ensure_library(workspace: Path | None = None, topic: str = "Knowledge") -> Path:
    """Create the local wiki workspace if missing and return its path."""
    ws = Path(workspace) if workspace else WORKSPACE_DIR
    (ws / "wiki").mkdir(parents=True, exist_ok=True)
    for sub in WIKI_SUBDIRS:
        (ws / "wiki" / sub).mkdir(parents=True, exist_ok=True)
    agents = ws / "AGENTS.md"
    if not agents.exists():
        agents.write_text(_agents_md(topic), encoding="utf-8")
    log = ws / "log.md"
    if not log.exists():
        log.write_text("# Library Timeline\n", encoding="utf-8")
    refresh_index(ws, topic)
    return ws


# --- Catalog (index.md) ------------------------------------------------------
def refresh_index(workspace: Path | None = None, topic: str = "Knowledge") -> None:
    """Rebuild /wiki/index.md as a catalog of the layered library."""
    ws = Path(workspace) if workspace else WORKSPACE_DIR
    wiki = ws / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    counts = kg.stats()["overall"]
    topics = kg.list_topics()
    topics_by_id = {t["id"]: t for t in topics}

    lines = [
        f"# {topic} — Knowledge Library",
        "",
        "Catalog of the layered knowledge library. Read this first.",
        "",
        (f"**Library:** {counts['sources']} sources · {counts['chunks']} chunks · "
         f"{counts['entities']} entities · {counts['topics']} topics · "
         f"{counts['syntheses']} syntheses · {counts['relations']} relations"),
        "",
        "## Topics",
        "",
    ]
    if topics:
        roots = [t for t in topics if not t.get("parent_id")]
        children: dict[str, list] = {}
        for t in topics:
            if t.get("parent_id"):
                children.setdefault(t["parent_id"], []).append(t)

        def _emit(t, depth=0):
            indent = "  " * depth
            summary = f" — {t['summary']}" if t.get("summary") else ""
            lines.append(f"{indent}- **{t['name']}** ({t['entity_count']} entities){summary}")
            ents = kg.list_entities(topic=t["id"], limit=8)
            for e in ents:
                lines.append(f"{indent}  - {e['name']} ({e.get('kind') or 'concept'})")
            for child in sorted(children.get(t["id"], []), key=lambda x: x["name"]):
                _emit(child, depth + 1)

        for root in sorted(roots, key=lambda x: -x["entity_count"]):
            _emit(root)
    else:
        lines.append("- _No topics yet. Run a maintenance pass to organize entities._")
    lines.append("")

    # Unassigned but important entities
    orphans = [e for e in kg.list_entities(limit=200) if not e.get("topic_id")]
    if orphans:
        lines += ["## Unfiled entities", ""]
        for e in orphans[:25]:
            lines.append(f"- {e['name']} ({e.get('kind') or 'concept'})")
        lines.append("")

    # Wiki pages on disk
    for label, sub in (("Syntheses", "synthesis"), ("Saved answers", "query")):
        pages = sorted((wiki / sub).glob("*.md")) if (wiki / sub).exists() else []
        if pages:
            lines += [f"## {label}", ""]
            for p in pages:
                lines.append(f"- [{p.stem.replace('-', ' ').title()}]({sub}/{p.name})")
            lines.append("")

    (wiki / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# --- Timeline (log.md) -------------------------------------------------------
def append_log(workspace: Path, phase: str, outcome: str, *,
               summary: str | None = None, metadata: dict | None = None) -> None:
    ws = Path(workspace)
    log = ws / "log.md"
    if not log.exists():
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("# Library Timeline\n", encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta = ""
    if metadata:
        meta = " " + " ".join(f"{k}={v}" for k, v in metadata.items())
    block = [f"\n## [{today}] {phase} | outcome={outcome}{meta}",
             f"- timestamp: {_now()}"]
    if summary:
        first = " ".join(summary.strip().splitlines())[:500]
        block.append(f"- summary: {first}")
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(block) + "\n")


# --- Agent construction ------------------------------------------------------
def _filesystem_permissions():
    from deepagents import FilesystemPermission
    return [
        FilesystemPermission(operations=["write"], paths=["/wiki/**"], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/log.md"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/AGENTS.md"], mode="deny"),
    ]


def _build_agent(model, workspace: Path, *, read_only: bool):
    """Construct a deepagents agent over the local wiki workspace."""
    if not agent_dependencies_available():
        raise AgentError(
            "The deepagents harness is not installed. Run `pip install deepagents "
            "langchain-anthropic langchain-openai` to enable the agent layer."
        )
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    tools = kg_tools.read_tools() if read_only else kg_tools.all_tools()
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=_BASE_SYSTEM_PROMPT,
        backend=backend,
        permissions=_filesystem_permissions(),
    )


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(chunks)
    return str(content)


def _final_message(result: dict) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        mtype = getattr(message, "type", None) or (message.get("type") if isinstance(message, dict) else None)
        if mtype not in {"ai", "assistant"}:
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = _extract_text(content).strip()
        if text:
            return text
    return ""


_CHUNK_ID_RE = re.compile(r"chunk_[0-9a-f]{12}")


def _collect_citations(result: dict, answer: str) -> list[dict]:
    """Gather grounded citations from the chunks the agent actually inspected
    (and any chunk ids it referenced in its answer)."""
    ids: list[str] = []
    seen = set()
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        for cid in _CHUNK_ID_RE.findall(_extract_text(content)):
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    for cid in _CHUNK_ID_RE.findall(answer or ""):
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    citations = []
    for cid in ids[:20]:
        ch = kg.get_chunk(cid)
        if ch:
            citations.append({
                "chunk_id": cid,
                "source_title": ch.get("source_title") or "",
                "source_url": ch.get("source_url") or "",
                "preview": ch.get("preview") or "",
            })
    return citations


def _invoke(agent, prompt: str) -> dict:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": _RECURSION_LIMIT},
    )


def _resolve_model(provider: str | None, model: str | None):
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise AgentError(
            "No LLM provider is configured. Set one of ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY to use the agent."
        )
    try:
        return build_chat_model(rp, rm), rp, rm
    except ProviderError as exc:
        raise AgentError(str(exc)) from exc


# --- Orchestrated modes (mirror llm-wiki: init / ingest / query / lint) ------
def run_init(*, topic: str = "Knowledge", workspace: Path | None = None) -> dict:
    ws = ensure_library(workspace, topic)
    append_log(ws, "init", "applied", summary=f"Initialized library workspace for '{topic}'.")
    return {"ok": True, "workspace": str(ws), "stats": kg.stats()["overall"]}


def run_ingest(*, provider=None, model=None, new_entities: list[str] | None = None,
               topic: str = "Knowledge", workspace: Path | None = None) -> dict:
    """Agentic organize pass over freshly integrated material."""
    ws = ensure_library(workspace, topic)
    chat, rp, rm = _resolve_model(provider, model)
    if not new_entities:
        new_entities = [e["name"] for e in kg.list_entities(limit=40)]
    listing = "\n".join(f"- {n}" for n in new_entities[:60]) or "- (none reported)"
    agent = _build_agent(chat, ws, read_only=False)
    result = _invoke(agent, _INGEST_PROMPT.format(new_entities=listing))
    report = _final_message(result) or "Ingest organize pass complete."
    refresh_index(ws, topic)
    append_log(ws, "ingest.organize", "applied",
               summary=report, metadata={"provider": rp, "model": rm,
                                          "entities": len(new_entities)})
    return {"ok": True, "report": report, "provider": rp, "model": rm,
            "stats": kg.stats()["overall"]}


def run_query(question: str, *, provider=None, model=None, file_answer: bool = True,
              topic: str = "Knowledge", workspace: Path | None = None) -> dict:
    """Answer a question grounded in the library, with citations (read + optional file)."""
    if not question or not question.strip():
        raise AgentError("A question is required for query mode.")
    ws = ensure_library(workspace, topic)
    chat, rp, rm = _resolve_model(provider, model)
    # Read-only reasoning pass.
    agent = _build_agent(chat, ws, read_only=True)
    result = _invoke(agent, _QUERY_PROMPT.format(question=question.strip()))
    answer = _final_message(result) or "No answer was produced."
    citations = _collect_citations(result, answer)

    filed_path = None
    if file_answer and citations:
        slug = re.sub(r"[^a-z0-9]+", "-", question.strip().lower()).strip("-")[:60] or "answer"
        rel = f"query/{slug}.md"
        page = (ws / "wiki" / rel)
        page.parent.mkdir(parents=True, exist_ok=True)
        body = [f"# {question.strip()}", "", answer, "",
                f"_Filed {_now()} · {rp}/{rm}_"]
        page.write_text("\n".join(body) + "\n", encoding="utf-8")
        kg.add_synthesis(question.strip()[:80], f"wiki/{rel}",
                         abstract=answer.splitlines()[0][:200] if answer else "")
        filed_path = rel
        refresh_index(ws, topic)

    append_log(ws, "query", "filed" if filed_path else "answered",
               summary=answer, metadata={"provider": rp, "model": rm,
                                         "citations": len(citations)})
    return {"ok": True, "answer": answer, "citations": citations,
            "filed": filed_path, "provider": rp, "model": rm}


def run_lint(*, provider=None, model=None, topic: str = "Knowledge",
             workspace: Path | None = None) -> dict:
    """Whole-library maintenance: dedupe, build topic hierarchy, synthesize."""
    ws = ensure_library(workspace, topic)
    before = kg.stats()["overall"]
    chat, rp, rm = _resolve_model(provider, model)
    agent = _build_agent(chat, ws, read_only=False)
    result = _invoke(agent, _LINT_PROMPT)
    report = _final_message(result) or "Maintenance pass complete."
    refresh_index(ws, topic)
    after = kg.stats()["overall"]
    append_log(ws, "lint", "applied", summary=report,
               metadata={"provider": rp, "model": rm,
                         "entities_before": before["entities"],
                         "entities_after": after["entities"],
                         "topics_after": after["topics"]})
    return {"ok": True, "report": report, "provider": rp, "model": rm,
            "before": before, "after": after, "stats": after}


def run_mode(mode: str, *, question: str | None = None, provider=None, model=None,
             topic: str = "Knowledge", workspace: Path | None = None) -> dict:
    mode = (mode or "").strip().lower()
    if mode == "init":
        return run_init(topic=topic, workspace=workspace)
    if mode == "ingest":
        return run_ingest(provider=provider, model=model, topic=topic, workspace=workspace)
    if mode == "query":
        return run_query(question or "", provider=provider, model=model,
                         topic=topic, workspace=workspace)
    if mode == "lint":
        return run_lint(provider=provider, model=model, topic=topic, workspace=workspace)
    raise AgentError(f"Unknown mode: {mode!r}. Use init | ingest | query | lint.")
