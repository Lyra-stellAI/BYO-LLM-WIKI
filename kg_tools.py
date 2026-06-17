"""LangChain tools the agent uses to read and reshape the knowledge library.

Tool docstrings double as the descriptions the model sees, so they are written
for the agent's benefit. Read tools ground answers; write tools let the agent
canonicalize entities, build the topic hierarchy, and record relations.
All tools operate on the integrated ``overall`` library and return JSON text.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

import knowledge_graph as kg

_WHERE = "overall"


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# --- Read tools --------------------------------------------------------------
@tool(parse_docstring=True)
def kg_stats() -> str:
    """Get counts of every layer in the knowledge library.

    Returns counts of sources, chunks, entities, topics, syntheses and edges.
    Call this first to understand the size and shape of the library.
    """
    return _dump(kg.stats()["overall"])


@tool(parse_docstring=True)
def kg_search(query: str, types: str = "", limit: int = 20) -> str:
    """Search the library for nodes whose text/name matches a query.

    Args:
        query: Free-text query. Matches names, aliases, summaries and chunk text.
        types: Optional comma-separated node types to restrict to, e.g.
            "entity", "chunk", "topic", "source", "synthesis".
        limit: Maximum number of rows to return.
    """
    type_list = [t.strip() for t in types.split(",") if t.strip()] or None
    rows = kg.search_nodes(query, where=_WHERE, types=type_list, limit=max(1, min(limit, 50)))
    return _dump(rows)


@tool(parse_docstring=True)
def kg_get_entity(name_or_id: str) -> str:
    """Get a full entity profile: kind, aliases, summary, topic, typed relations, and the chunks that mention it.

    Args:
        name_or_id: Entity name, alias, or id (e.g. "Ada Lovelace" or "entity_ada-lovelace").
    """
    ent = kg.get_entity(name_or_id, where=_WHERE)
    if not ent:
        return _dump({"error": f"No entity matching {name_or_id!r}."})
    return _dump(ent)


@tool(parse_docstring=True)
def kg_get_chunk(chunk_id: str) -> str:
    """Get the full evidence text of a chunk and its source, for grounding/citation.

    Args:
        chunk_id: The chunk id (e.g. "chunk_ab12cd34ef56").
    """
    ch = kg.get_chunk(chunk_id, where=_WHERE)
    if not ch:
        return _dump({"error": f"No chunk with id {chunk_id!r}."})
    return _dump(ch)


@tool(parse_docstring=True)
def kg_neighbors(node_id: str, depth: int = 1) -> str:
    """Explore the subgraph around a node up to a given hop depth (for multi-hop reasoning).

    Args:
        node_id: The id of the node to expand from.
        depth: Number of hops to traverse (1 or 2 recommended).
    """
    sub = kg.neighbors(node_id, where=_WHERE, depth=max(1, min(depth, 3)))
    if not sub["node"]:
        return _dump({"error": f"No node with id {node_id!r}."})
    nodes = [{"id": n["id"], "type": n.get("type"), "name": n.get("name") or n.get("title") or n.get("preview", "")[:60]}
             for n in sub["nodes"]]
    edges = [{"from": e["from"], "to": e["to"], "kind": e.get("kind"), "label": e.get("label")}
             for e in sub["edges"]]
    return _dump({"center": node_id, "nodes": nodes[:60], "edges": edges[:80]})


@tool(parse_docstring=True)
def kg_list_entities(topic: str = "", kind: str = "", limit: int = 60) -> str:
    """List entities, most important first, optionally filtered by topic or kind.

    Args:
        topic: Optional topic name or id to filter by.
        kind: Optional entity kind (person, organization, place, concept, technology, method, event, product).
        limit: Maximum number of entities to return.
    """
    rows = kg.list_entities(where=_WHERE, topic=topic or None, kind=kind or None,
                            limit=max(1, min(limit, 200)))
    return _dump(rows)


@tool(parse_docstring=True)
def kg_list_topics() -> str:
    """List all topics (themes) with their parent and how many entities each contains."""
    return _dump(kg.list_topics(where=_WHERE))


# --- Write tools -------------------------------------------------------------
@tool(parse_docstring=True)
def kg_upsert_entity(name: str, kind: str = "concept", summary: str = "",
                     aliases: str = "", importance: int = 3) -> str:
    """Create an entity or reinforce/enrich an existing one (idempotent).

    Use this to add a canonical concept/person/etc. or to attach a one-line
    summary. Prefer reusing an existing canonical name over creating near-duplicates.

    Args:
        name: Canonical entity name (Title Case, 1-5 words).
        kind: One of person, organization, place, concept, technology, method, event, product.
        summary: Optional one-sentence description grounded in the library.
        aliases: Optional comma-separated alternative names/spellings.
        importance: Salience from 1 (minor) to 5 (central).
    """
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
    ent_id = kg.upsert_entity(name, kind, summary=summary, aliases=alias_list,
                              importance=importance, where=_WHERE)
    return _dump({"ok": bool(ent_id), "id": ent_id})


@tool(parse_docstring=True)
def kg_set_entity_summary(name_or_id: str, summary: str) -> str:
    """Set or replace the one-line canonical summary for an entity.

    Args:
        name_or_id: Entity name, alias, or id.
        summary: A concise, source-grounded description.
    """
    return _dump({"ok": kg.set_entity_summary(name_or_id, summary, where=_WHERE)})


@tool(parse_docstring=True)
def kg_add_relation(source: str, target: str, predicate: str, confidence: str = "EXTRACTED") -> str:
    """Add a typed, directed relationship between two entities.

    Args:
        source: Source entity name or id.
        target: Target entity name or id.
        predicate: Short active-voice phrase, e.g. "depends on", "created", "is part of".
        confidence: EXTRACTED (stated in sources), INFERRED, or AMBIGUOUS.
    """
    ok = kg.add_relation(source, target, predicate, confidence=confidence, where=_WHERE)
    return _dump({"ok": ok})


@tool(parse_docstring=True)
def kg_merge_entities(keep: str, drop: str) -> str:
    """Merge a duplicate entity into the canonical one, re-pointing all edges.

    Use when two entities are the same thing (e.g. "GPT-4" and "GPT 4"). The
    dropped name is preserved as an alias of the kept entity.

    Args:
        keep: The canonical entity to keep (name or id).
        drop: The duplicate entity to merge in and delete (name or id).
    """
    return _dump(kg.merge_entities(keep, drop, where=_WHERE))


@tool(parse_docstring=True)
def kg_upsert_topic(name: str, summary: str = "", parent: str = "") -> str:
    """Create or update a topic (theme) node, optionally nested under a parent topic.

    Topics are the hierarchical layer that groups related entities into
    coherent themes. Keep the count of topics small and meaningful.

    Args:
        name: Topic name (a short noun phrase).
        summary: Optional one-line description of what the topic covers.
        parent: Optional parent topic name/id to build a hierarchy.
    """
    tid = kg.upsert_topic(name, summary=summary, parent=parent or None, where=_WHERE)
    return _dump({"ok": bool(tid), "id": tid})


@tool(parse_docstring=True)
def kg_assign_entity_to_topic(entity: str, topic: str) -> str:
    """Assign an entity to a topic (creates the topic if it does not exist yet).

    Args:
        entity: Entity name or id.
        topic: Topic name or id.
    """
    return _dump({"ok": kg.assign_entity_to_topic(entity, topic, where=_WHERE)})


_READ_TOOLS = [
    kg_stats, kg_search, kg_get_entity, kg_get_chunk,
    kg_neighbors, kg_list_entities, kg_list_topics,
]
_WRITE_TOOLS = [
    kg_upsert_entity, kg_set_entity_summary, kg_add_relation,
    kg_merge_entities, kg_upsert_topic, kg_assign_entity_to_topic,
]


def read_tools() -> list:
    """Tools for read-only grounding (query mode)."""
    return list(_READ_TOOLS)


def all_tools() -> list:
    """Read + write tools (ingest / maintain / organize modes)."""
    return list(_READ_TOOLS) + list(_WRITE_TOOLS)
