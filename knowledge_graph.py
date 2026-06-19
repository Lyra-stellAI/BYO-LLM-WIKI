"""Multi-layer knowledge library store.

The graph is a layered property graph rather than a flat bag of entities.
Layers (a node's ``layer`` field), from concrete evidence up to synthesis:

    0  source     - a document / URL / file the knowledge came from
    1  chunk      - an immutable passage of evidence from a source
    2  entity     - a canonical, de-duplicated thing (person, org, concept...)
    3  topic       - a theme that groups related entities (hierarchical)
    4  synthesis   - an agent-written canonical note that unifies evidence

Edges are typed:

    from_source  chunk    -> source     provenance
    mentions     chunk    -> entity      evidence for an entity
    relation     entity   -> entity      semantic predicate (typed)
    belongs_to   entity   -> topic       roll-up into a theme (hierarchy)
    subtopic_of  topic    -> topic       topic hierarchy
    covers       synthesis-> entity|topic what a synthesis is about
    cites        synthesis-> chunk        grounding for a synthesis

Two stores are kept on disk: ``current.json`` (staging, where freshly added
chunks land) and ``overall.json`` (the integrated, multi-layer library).
The public functions used by the web app (``get_graph``, ``stats``,
``add_chunk``, ``integrate``, ``query``, ``remove_node``, ``clear``) keep
their original shape; the richer layer-aware helpers power the agent tools.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_PATH = DATA_DIR / "current.json"
OVERALL_PATH = DATA_DIR / "overall.json"

SCHEMA_VERSION = 2

# Node layers / types ---------------------------------------------------------
# source -> section (contextual summary) -> chunk -> entity -> topic -> synthesis
LAYER_OF = {"source": 0, "section": 1, "chunk": 2, "entity": 3, "topic": 4, "synthesis": 5}

ENTITY_KINDS = {
    "person", "organization", "place", "concept",
    "technology", "method", "event", "product",
}

_lock = RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_graph() -> dict:
    return {
        "version": SCHEMA_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "nodes": [],
        "edges": [],
    }


def _load(path: Path) -> dict:
    if not path.exists():
        return _empty_graph()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_graph()
    return _migrate(data)


def _migrate(data: dict) -> dict:
    """Bring older / partial graphs up to the current schema in memory."""
    data.setdefault("version", 1)
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    for node in data["nodes"]:
        ntype = node.get("type")
        if "layer" not in node and ntype in LAYER_OF:
            node["layer"] = LAYER_OF[ntype]
        if ntype == "entity":
            node.setdefault("aliases", [])
            node.setdefault("importance", 3)
            node.setdefault("summary", "")
    return data


def _save(path: Path, data: dict) -> None:
    data["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _slugify(name: str) -> str:
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-")[:60]


def _path_for(where: str) -> Path:
    return OVERALL_PATH if where == "overall" else CURRENT_PATH


def get_graph(where: str = "current") -> dict:
    with _lock:
        return _load(_path_for(where))


def _counts(g: dict) -> dict:
    nodes = g["nodes"]
    by_type: dict[str, int] = {}
    for n in nodes:
        by_type[n.get("type", "?")] = by_type.get(n.get("type", "?"), 0) + 1
    relations = sum(1 for e in g["edges"] if e.get("kind") == "relation")
    return {
        "sources": by_type.get("source", 0),
        "sections": by_type.get("section", 0),
        "chunks": by_type.get("chunk", 0),
        "entities": by_type.get("entity", 0),
        "topics": by_type.get("topic", 0),
        "syntheses": by_type.get("synthesis", 0),
        "edges": len(g["edges"]),
        "relations": relations,
    }


def stats() -> dict:
    with _lock:
        return {"current": _counts(_load(CURRENT_PATH)), "overall": _counts(_load(OVERALL_PATH))}


# Node types to include at each zoom level (coarse -> fine).
_GRANULARITY_TYPES = {
    "document": {"source", "topic", "synthesis"},
    "section": {"source", "section", "topic", "synthesis"},
    "chunk": {"source", "section", "chunk", "entity", "topic", "synthesis"},
}


def granularity_view(where: str = "overall", level: str = "section") -> dict:
    """Return the graph at a given zoom level.

    - ``document``: the high-level map (sources, topics, syntheses).
    - ``section``:  + contextual-summary sections under each source.
    - ``chunk``:    the full graph (down to chunks and entities).

    Edges are kept only between included nodes.
    """
    g = get_graph(where)
    types = _GRANULARITY_TYPES.get(level, _GRANULARITY_TYPES["chunk"])
    keep = {n["id"] for n in g["nodes"] if n.get("type") in types}
    nodes = [n for n in g["nodes"] if n["id"] in keep]
    edges = [e for e in g["edges"] if e["from"] in keep and e["to"] in keep]
    if level == "document":
        edges = edges + _shared_entity_edges(g)
    return {"level": level, "nodes": nodes, "edges": edges,
            "counts": _counts({"nodes": nodes, "edges": edges})}


# --- Staging -----------------------------------------------------------------
def add_chunk(text: str, *, source_url: str | None = None, source_title: str | None = None,
              tags: list[str] | None = None, note: str | None = None) -> dict:
    chunk = {
        "id": f"chunk_{uuid.uuid4().hex[:12]}",
        "type": "chunk",
        "layer": LAYER_OF["chunk"],
        "text": text,
        "preview": (text[:240] + "…") if len(text) > 240 else text,
        "source_url": source_url or "",
        "source_title": source_title or "",
        "tags": [t.strip() for t in (tags or []) if t and t.strip()],
        "note": note or "",
        "created_at": _now(),
        "integrated_at": None,
    }
    with _lock:
        g = _load(CURRENT_PATH)
        g["nodes"].append(chunk)
        _save(CURRENT_PATH, g)
    return chunk


def remove_node(node_id: str, where: str = "current") -> bool:
    path = _path_for(where)
    with _lock:
        g = _load(path)
        before = len(g["nodes"])
        g["nodes"] = [n for n in g["nodes"] if n["id"] != node_id]
        g["edges"] = [e for e in g["edges"] if e["from"] != node_id and e["to"] != node_id]
        _save(path, g)
        return before != len(g["nodes"])


def clear(where: str = "current") -> None:
    with _lock:
        _save(_path_for(where), _empty_graph())


# --- Heuristic fallback extraction (used when no LLM/agent is available) -----
STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Where", "When", "What", "Who",
    "How", "Why", "But", "And", "For", "With", "From", "Into", "Their", "Some", "Most",
    "Many", "More", "Less", "It", "Its", "An", "A", "Is", "Are", "Was", "Were", "Be",
    "Been", "Being", "Have", "Has", "Had", "Will", "Would", "Could", "Should", "May",
    "Might", "Must", "Shall", "Can", "Cannot", "In", "On", "At", "By", "To", "Of",
    "If", "As", "So", "Or", "Not", "Now", "Then", "Also", "Just", "Even", "Still",
    "Yet", "After", "Before", "While", "Because", "Since", "Until", "Round", "Part",
    "Step", "Section", "Chapter", "Figure", "Table", "Page", "Note", "Notes",
    "First", "Second", "Third", "Next", "Last", "Final", "One", "Two", "Three",
    "Working", "Building", "Getting", "Including", "According", "Based",
}

_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\n)([A-Z][a-zA-Z]+)\b")

LEADING_ARTICLES = {
    "The", "A", "An", "This", "That", "These", "Those", "By", "In", "On",
    "Of", "Our", "Their", "His", "Her", "Its",
}


def _heuristic_entities(text: str) -> list[dict]:
    """Fallback proper-noun phrase extractor. Prefers multi-word names,
    strips leading articles, and drops sentence-start singleton noise."""
    sentence_start_singles = set()
    for m in _SENTENCE_START.finditer(text):
        word = m.group(1)
        if " " not in word:
            sentence_start_singles.add(word)

    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4}\b", text)
    multi: dict[str, int] = {}
    single: dict[str, int] = {}
    for c in candidates:
        parts = c.split()
        while parts and parts[0] in LEADING_ARTICLES:
            parts = parts[1:]
        if not parts:
            continue
        if parts[0] in STOPWORDS:
            continue
        name = " ".join(parts)
        if " " in name:
            multi[name] = multi.get(name, 0) + 1
        else:
            single[name] = single.get(name, 0) + 1

    out: list[dict] = []
    seen: set[str] = set()
    for name, _ in sorted(multi.items(), key=lambda x: -x[1]):
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "kind": "concept", "confidence": "INFERRED"})
        if len(out) >= 12:
            break
    for name, count in sorted(single.items(), key=lambda x: -x[1]):
        key = name.lower()
        if key in seen or count < 2:
            continue
        if name in sentence_start_singles and count < 3:
            continue
        seen.add(key)
        out.append({"name": name, "kind": "concept", "confidence": "INFERRED"})
        if len(out) >= 18:
            break
    return out[:18]


def _maybe_subchunk(text: str, max_chars: int = 2500) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: list[str] = []
    buf = ""
    for p in paragraphs:
        cand = (buf + "\n\n" + p).strip() if buf else p
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                pieces.append(buf)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    pieces.append(p[i:i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        pieces.append(buf)
    return pieces or [text]


def _normalize_extraction(result) -> dict:
    if isinstance(result, dict):
        return {
            "entities": result.get("entities") or [],
            "relations": result.get("relations") or [],
        }
    if isinstance(result, list):
        return {"entities": result, "relations": []}
    return {"entities": [], "relations": []}


# --- Internal graph index helpers -------------------------------------------
def _entity_index(g: dict) -> dict[str, dict]:
    """Map id / slug / name / alias (lowercased) -> entity node."""
    index: dict[str, dict] = {}
    for n in g["nodes"]:
        if n.get("type") != "entity":
            continue
        index[n["id"]] = n
        index[n["id"].lower()] = n
        if n.get("name"):
            index[n["name"].strip().lower()] = n
        for alias in n.get("aliases", []):
            if alias:
                index.setdefault(alias.strip().lower(), n)
    return index


def _resolve_entity(g: dict, name_or_id: str) -> dict | None:
    if not name_or_id:
        return None
    idx = _entity_index(g)
    key = name_or_id.strip()
    return idx.get(key) or idx.get(key.lower()) or idx.get(f"entity_{_slugify(key)}")


def _node_by_id(g: dict, node_id: str) -> dict | None:
    for n in g["nodes"]:
        if n["id"] == node_id:
            return n
    return None


def _upsert_entity(g: dict, index: dict, name: str, kind: str, *,
                   summary: str = "", aliases=None, importance: int | None = None,
                   confidence: str = "EXTRACTED", summary_counts: dict | None = None) -> str | None:
    name = (name or "").strip()
    if not name:
        return None
    existing = index.get(name.lower())
    if existing is None:
        slug = _slugify(name)
        if not slug:
            return None
        ent_id = f"entity_{slug}"
        # slug collision with a different display name -> keep both via suffix
        node = _node_by_id(g, ent_id)
        if node is not None and node.get("name", "").lower() != name.lower():
            ent_id = f"entity_{slug}-{uuid.uuid4().hex[:4]}"
        node = {
            "id": ent_id,
            "type": "entity",
            "layer": LAYER_OF["entity"],
            "name": name,
            "kind": kind if kind in ENTITY_KINDS else "concept",
            "aliases": sorted({a.strip() for a in (aliases or []) if a and a.strip()}),
            "importance": importance if importance in (1, 2, 3, 4, 5) else 3,
            "mentions": 1,
            "summary": summary or "",
            "topic_id": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        g["nodes"].append(node)
        # update the in-memory index for subsequent calls in the same pass
        index[ent_id] = node
        index[name.lower()] = node
        for a in node["aliases"]:
            index.setdefault(a.lower(), node)
        if summary_counts is not None:
            summary_counts["entities_added"] = summary_counts.get("entities_added", 0) + 1
        return ent_id

    existing["mentions"] = existing.get("mentions", 1) + 1
    existing["updated_at"] = _now()
    if summary and len(summary) > len(existing.get("summary", "")):
        existing["summary"] = summary
    if kind in ENTITY_KINDS and existing.get("kind") in (None, "concept") and kind != "concept":
        existing["kind"] = kind
    if importance in (1, 2, 3, 4, 5):
        existing["importance"] = max(existing.get("importance", 3), importance)
    for a in (aliases or []):
        a = a.strip()
        if a and a.lower() != existing["name"].lower() and a not in existing.get("aliases", []):
            existing.setdefault("aliases", []).append(a)
            index.setdefault(a.lower(), existing)
    if summary_counts is not None:
        summary_counts["entities_reinforced"] = summary_counts.get("entities_reinforced", 0) + 1
    return existing["id"]


def _ensure_source(g: dict, title: str, url: str) -> str | None:
    title = (title or "").strip()
    url = (url or "").strip()
    if not title and not url:
        return None
    key = url or title
    for n in g["nodes"]:
        if n.get("type") == "source" and (n.get("url") == url and url or n.get("title") == title):
            return n["id"]
    src_id = f"source_{_slugify(key) or uuid.uuid4().hex[:8]}"
    if _node_by_id(g, src_id):
        src_id = f"source_{uuid.uuid4().hex[:10]}"
    g["nodes"].append({
        "id": src_id,
        "type": "source",
        "layer": LAYER_OF["source"],
        "title": title or url,
        "url": url,
        "created_at": _now(),
    })
    return src_id


def _add_edge(g: dict, frm: str, to: str, kind: str, label: str,
              *, confidence: str = "EXTRACTED", chunk_id: str | None = None) -> None:
    edge = {
        "id": f"e_{uuid.uuid4().hex[:10]}",
        "from": frm,
        "to": to,
        "kind": kind,
        "label": label,
        "confidence": confidence,
        "created_at": _now(),
    }
    if chunk_id:
        edge["chunk_id"] = chunk_id
    g["edges"].append(edge)


def _has_edge(g: dict, frm: str, to: str, kind: str) -> bool:
    return any(e["from"] == frm and e["to"] == to and e.get("kind") == kind for e in g["edges"])


def integrate(extract_fn=None) -> dict:
    """Move staged chunks into the overall library, building the layered graph.

    For each staged chunk we create/reuse a source node, link the chunk to it,
    extract entities + relations (via ``extract_fn`` if provided, else the
    heuristic fallback), canonicalize/de-duplicate entities, and wire edges.
    """
    summary = {
        "chunks_integrated": 0,
        "sources_added": 0,
        "entities_added": 0,
        "entities_reinforced": 0,
        "edges_added": 0,
        "relation_edges_added": 0,
        "errors": [],
    }
    with _lock:
        cur = _load(CURRENT_PATH)
        overall = _load(OVERALL_PATH)
        index = _entity_index(overall)
        existing_chunk_ids = {n["id"] for n in overall["nodes"] if n.get("type") == "chunk"}
        sources_before = sum(1 for n in overall["nodes"] if n.get("type") == "source")

        for chunk in cur["nodes"]:
            if chunk.get("type") != "chunk" or chunk["id"] in existing_chunk_ids:
                continue
            new_chunk = dict(chunk)
            new_chunk["integrated_at"] = _now()
            new_chunk.setdefault("layer", LAYER_OF["chunk"])
            src_id = _ensure_source(overall, chunk.get("source_title", ""), chunk.get("source_url", ""))
            if src_id:
                new_chunk["source_id"] = src_id
            overall["nodes"].append(new_chunk)
            summary["chunks_integrated"] += 1
            if src_id and not _has_edge(overall, new_chunk["id"], src_id, "from_source"):
                _add_edge(overall, new_chunk["id"], src_id, "from_source", "from source")
                summary["edges_added"] += 1

            all_entities: list[dict] = []
            all_relations: list[dict] = []
            for piece in _maybe_subchunk(chunk["text"]):
                extracted = None
                if extract_fn:
                    try:
                        extracted = _normalize_extraction(extract_fn(piece))
                    except Exception as e:  # noqa: BLE001
                        summary["errors"].append(str(e))
                if not extracted or (not extracted["entities"] and not extracted["relations"]):
                    extracted = {"entities": _heuristic_entities(piece), "relations": []}
                all_entities.extend(extracted["entities"])
                all_relations.extend(extracted["relations"])

            name_to_id: dict[str, str] = {}
            for ent in all_entities:
                if isinstance(ent, str):
                    ent = {"name": ent}
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                ent_id = _upsert_entity(
                    overall, index, name, ent.get("kind") or "concept",
                    summary=ent.get("summary") or "",
                    aliases=ent.get("aliases"),
                    importance=ent.get("importance"),
                    confidence=ent.get("confidence") or "EXTRACTED",
                    summary_counts=summary,
                )
                if not ent_id:
                    continue
                name_to_id[name.lower()] = ent_id
                if not _has_edge(overall, new_chunk["id"], ent_id, "mentions"):
                    _add_edge(overall, new_chunk["id"], ent_id, "mentions", "mentions",
                              confidence=ent.get("confidence") or "EXTRACTED",
                              chunk_id=new_chunk["id"])
                    summary["edges_added"] += 1

            for rel in all_relations:
                if not isinstance(rel, dict):
                    continue
                src_name = (rel.get("source") or "").strip()
                tgt_name = (rel.get("target") or "").strip()
                pred = (rel.get("predicate") or "related to").strip()
                if not src_name or not tgt_name:
                    continue
                src_eid = name_to_id.get(src_name.lower()) or _upsert_entity(
                    overall, index, src_name, "concept", confidence="INFERRED", summary_counts=summary)
                tgt_eid = name_to_id.get(tgt_name.lower()) or _upsert_entity(
                    overall, index, tgt_name, "concept", confidence="INFERRED", summary_counts=summary)
                if not src_eid or not tgt_eid or src_eid == tgt_eid:
                    continue
                name_to_id[src_name.lower()] = src_eid
                name_to_id[tgt_name.lower()] = tgt_eid
                if not any(e["from"] == src_eid and e["to"] == tgt_eid and e.get("label") == pred
                           for e in overall["edges"]):
                    _add_edge(overall, src_eid, tgt_eid, "relation", pred,
                              confidence=rel.get("confidence") or "EXTRACTED",
                              chunk_id=new_chunk["id"])
                    summary["relation_edges_added"] += 1
                    summary["edges_added"] += 1

        summary["sources_added"] = sum(1 for n in overall["nodes"] if n.get("type") == "source") - sources_before
        _save(OVERALL_PATH, overall)
        _save(CURRENT_PATH, _empty_graph())
    return summary


# --- Querying / retrieval (layer-aware) -------------------------------------
def query(text: str, where: str = "overall", limit: int = 30, *, types=None) -> list[dict]:
    g = get_graph(where)
    q = (text or "").lower().strip()
    matches = []
    for n in g["nodes"]:
        if types and n.get("type") not in types:
            continue
        if not q:
            matches.append(n)
            continue
        haystack = " ".join([
            n.get("text", ""), n.get("name", ""), n.get("preview", ""),
            n.get("summary", ""), n.get("title", ""),
            n.get("source_title", ""), " ".join(n.get("tags", []) or []), n.get("note", ""),
            " ".join(n.get("aliases", []) or []),
        ]).lower()
        if q in haystack:
            matches.append(n)
    return matches[:limit]


def search_nodes(query_text: str, *, where: str = "overall", layer=None,
                 types=None, limit: int = 25) -> list[dict]:
    """Compact search result rows for agent tools."""
    if isinstance(types, str):
        types = [types]
    rows = query(query_text, where=where, limit=500, types=types)
    out = []
    for n in rows:
        if layer is not None and n.get("layer") != layer:
            continue
        out.append({
            "id": n["id"],
            "type": n.get("type"),
            "layer": n.get("layer"),
            "name": n.get("name") or n.get("title") or n.get("preview", "")[:80],
            "kind": n.get("kind"),
            "mentions": n.get("mentions"),
        })
        if len(out) >= limit:
            break
    return out


def neighbors(node_id: str, where: str = "overall", depth: int = 1) -> dict:
    g = get_graph(where)
    by_id = {n["id"]: n for n in g["nodes"]}
    if node_id not in by_id:
        return {"node": None, "edges": [], "nodes": []}
    frontier = {node_id}
    seen = {node_id}
    collected_edges = []
    for _ in range(max(1, depth)):
        nxt = set()
        for e in g["edges"]:
            if e["from"] in frontier or e["to"] in frontier:
                collected_edges.append(e)
                for nid in (e["from"], e["to"]):
                    if nid not in seen:
                        seen.add(nid)
                        nxt.add(nid)
        frontier = nxt
        if not frontier:
            break
    # de-dup edges
    uniq = {e["id"]: e for e in collected_edges}
    return {
        "node": by_id.get(node_id),
        "nodes": [by_id[i] for i in seen if i in by_id],
        "edges": list(uniq.values()),
    }


def get_entity(name_or_id: str, where: str = "overall") -> dict | None:
    g = get_graph(where)
    ent = _resolve_entity(g, name_or_id)
    if not ent:
        return None
    by_id = {n["id"]: n for n in g["nodes"]}
    relations, mentioned_in = [], []
    for e in g["edges"]:
        if e.get("kind") == "relation" and e["from"] == ent["id"]:
            tgt = by_id.get(e["to"])
            if tgt:
                relations.append({"predicate": e.get("label"), "target": tgt.get("name"),
                                  "target_id": tgt["id"], "confidence": e.get("confidence")})
        elif e.get("kind") == "relation" and e["to"] == ent["id"]:
            src = by_id.get(e["from"])
            if src:
                relations.append({"predicate": e.get("label"), "source": src.get("name"),
                                  "source_id": src["id"], "confidence": e.get("confidence"),
                                  "inbound": True})
        elif e.get("kind") == "mentions" and e["to"] == ent["id"]:
            ch = by_id.get(e["from"])
            if ch:
                mentioned_in.append({"chunk_id": ch["id"], "preview": ch.get("preview", ""),
                                     "source_title": ch.get("source_title", ""),
                                     "source_url": ch.get("source_url", "")})
    topic = by_id.get(ent.get("topic_id")) if ent.get("topic_id") else None
    return {
        "id": ent["id"], "name": ent.get("name"), "kind": ent.get("kind"),
        "aliases": ent.get("aliases", []), "importance": ent.get("importance", 3),
        "mentions": ent.get("mentions", 1), "summary": ent.get("summary", ""),
        "topic": topic.get("name") if topic else None,
        "topic_id": ent.get("topic_id"),
        "relations": relations, "mentioned_in": mentioned_in[:20],
    }


def get_chunk(chunk_id: str, where: str = "overall") -> dict | None:
    g = get_graph(where)
    n = _node_by_id(g, chunk_id)
    if not n or n.get("type") != "chunk":
        return None
    return {
        "id": n["id"], "text": n.get("text", ""), "preview": n.get("preview", ""),
        "source_title": n.get("source_title", ""), "source_url": n.get("source_url", ""),
        "tags": n.get("tags", []), "note": n.get("note", ""),
    }


def list_entities(*, where: str = "overall", topic: str | None = None,
                  kind: str | None = None, limit: int = 100) -> list[dict]:
    g = get_graph(where)
    topic_id = None
    if topic:
        for n in g["nodes"]:
            if n.get("type") == "topic" and (n["id"] == topic or n.get("name", "").lower() == topic.lower()):
                topic_id = n["id"]
                break
    out = []
    for n in g["nodes"]:
        if n.get("type") != "entity":
            continue
        if kind and n.get("kind") != kind:
            continue
        if topic_id and n.get("topic_id") != topic_id:
            continue
        out.append({"id": n["id"], "name": n.get("name"), "kind": n.get("kind"),
                    "importance": n.get("importance", 3), "mentions": n.get("mentions", 1),
                    "topic_id": n.get("topic_id")})
    out.sort(key=lambda x: (-(x["importance"] or 0), -(x["mentions"] or 0)))
    return out[:limit]


def list_topics(where: str = "overall") -> list[dict]:
    g = get_graph(where)
    counts: dict[str, int] = {}
    for n in g["nodes"]:
        if n.get("type") == "entity" and n.get("topic_id"):
            counts[n["topic_id"]] = counts.get(n["topic_id"], 0) + 1
    out = []
    for n in g["nodes"]:
        if n.get("type") != "topic":
            continue
        out.append({"id": n["id"], "name": n.get("name"), "summary": n.get("summary", ""),
                    "parent_id": n.get("parent_id"), "entity_count": counts.get(n["id"], 0)})
    return out


# --- Mutations used by the agent layer --------------------------------------
def _mutate(where: str, fn):
    path = _path_for(where)
    with _lock:
        g = _load(path)
        result = fn(g)
        _save(path, g)
        return result


def upsert_entity(name: str, kind: str = "concept", *, summary: str = "",
                  aliases=None, importance: int | None = None, where: str = "overall") -> str | None:
    def _fn(g):
        return _upsert_entity(g, _entity_index(g), name, kind, summary=summary,
                              aliases=aliases, importance=importance)
    return _mutate(where, _fn)


def set_entity_summary(name_or_id: str, summary: str, where: str = "overall") -> bool:
    def _fn(g):
        ent = _resolve_entity(g, name_or_id)
        if not ent:
            return False
        ent["summary"] = summary
        ent["updated_at"] = _now()
        return True
    return _mutate(where, _fn)


def add_relation(source: str, target: str, predicate: str,
                 confidence: str = "EXTRACTED", where: str = "overall") -> bool:
    def _fn(g):
        idx = _entity_index(g)
        s = _resolve_entity(g, source) or _node_by_id(g, _upsert_entity(g, idx, source, "concept", confidence="INFERRED"))
        t = _resolve_entity(g, target) or _node_by_id(g, _upsert_entity(g, idx, target, "concept", confidence="INFERRED"))
        if not s or not t or s["id"] == t["id"]:
            return False
        if any(e["from"] == s["id"] and e["to"] == t["id"] and e.get("label") == predicate for e in g["edges"]):
            return False
        _add_edge(g, s["id"], t["id"], "relation", predicate, confidence=confidence)
        return True
    return _mutate(where, _fn)


def _merge_into(g: dict, keep: dict, drop: dict) -> int:
    """Re-point drop's edges to keep, fold aliases/mentions, remove drop. In-memory."""
    moved = 0
    for e in g["edges"]:
        if e["from"] == drop["id"]:
            e["from"] = keep["id"]; moved += 1
        if e["to"] == drop["id"]:
            e["to"] = keep["id"]; moved += 1
    aliases = set(keep.get("aliases", [])) | set(drop.get("aliases", []))
    if drop.get("name"):
        aliases.add(drop["name"])
    aliases.discard(keep.get("name", ""))
    keep["aliases"] = sorted(a for a in aliases if a)
    keep["mentions"] = keep.get("mentions", 1) + drop.get("mentions", 1)
    if len(drop.get("summary", "")) > len(keep.get("summary", "")):
        keep["summary"] = drop["summary"]
    keep["updated_at"] = _now()
    return moved


def merge_entities(keep: str, drop: str, where: str = "overall") -> dict:
    """Merge entity ``drop`` into ``keep``: re-point edges, fold aliases/mentions,
    and delete the duplicate node. The dropped name becomes an alias of keep."""
    def _fn(g):
        k = _resolve_entity(g, keep)
        d = _resolve_entity(g, drop)
        if not k or not d:
            return {"ok": False, "error": "entity not found", "keep": bool(k), "drop": bool(d)}
        if k["id"] == d["id"]:
            return {"ok": False, "error": "keep and drop are the same entity"}
        moved = _merge_into(g, k, d)
        g["edges"] = [e for e in g["edges"]
                      if not (e["from"] == k["id"] and e["to"] == k["id"] and e.get("kind") == "relation")]
        g["nodes"] = [n for n in g["nodes"] if n["id"] != d["id"]]
        return {"ok": True, "keep_id": k["id"], "dropped_id": d["id"], "edges_moved": moved}
    return _mutate(where, _fn)


def merge_duplicates(where: str = "overall") -> dict:
    """Batch-merge entities whose names normalize identically (e.g. 'GPT-4' / 'GPT 4').

    One load/save. Keeps the highest-mention member of each group as canonical and
    de-duplicates edges afterward. Grooming step before graph-RAG / topic building.
    """
    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (name or "").lower())

    def _fn(g):
        groups: dict[str, list] = {}
        for n in g["nodes"]:
            if n.get("type") == "entity":
                key = _norm(n.get("name", ""))
                if key:
                    groups.setdefault(key, []).append(n)
        merged, drop_ids = 0, set()
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda n: -n.get("mentions", 1))
            keep = members[0]
            for d in members[1:]:
                _merge_into(g, keep, d)
                drop_ids.add(d["id"])
                merged += 1
        if drop_ids:
            g["nodes"] = [n for n in g["nodes"] if n["id"] not in drop_ids]
            # de-duplicate edges (merging collapses parallel edges) + drop relation self-loops
            seen, kept = set(), []
            for e in g["edges"]:
                if e["from"] == e["to"] and e.get("kind") == "relation":
                    continue
                sig = (e["from"], e["to"], e.get("kind"), e.get("label"))
                if sig in seen:
                    continue
                seen.add(sig)
                kept.append(e)
            g["edges"] = kept
        entities = sum(1 for n in g["nodes"] if n.get("type") == "entity")
        return {"merged": merged, "entities_after": entities}
    return _mutate(where, _fn)


def entity_doc_frequency(where: str = "overall") -> tuple[dict, int]:
    """Return ({entity_name_lower: #distinct source docs mentioning it}, n_docs).

    Used to weight graph expansion by specificity (IDF): generic 'hub' entities
    (high document frequency) are downweighted vs. rare, discriminative ones.
    """
    g = get_graph(where)
    chunk_src = {n["id"]: n.get("source_id") for n in g["nodes"] if n.get("type") == "chunk"}
    ent_name = {n["id"]: (n.get("name") or "").lower() for n in g["nodes"] if n.get("type") == "entity"}
    ent_docs: dict[str, set] = {}
    for e in g["edges"]:
        if e.get("kind") == "mentions":
            src = chunk_src.get(e["from"])
            if src:
                ent_docs.setdefault(e["to"], set()).add(src)
    n_docs = sum(1 for n in g["nodes"] if n.get("type") == "source") or 1
    df = {}
    for eid, srcs in ent_docs.items():
        name = ent_name.get(eid)
        if name:
            df[name] = max(df.get(name, 0), len(srcs))
    return df, n_docs


def upsert_topic(name: str, *, summary: str = "", parent: str | None = None,
                 where: str = "overall") -> str | None:
    def _fn(g):
        for n in g["nodes"]:
            if n.get("type") == "topic" and n.get("name", "").lower() == name.strip().lower():
                if summary:
                    n["summary"] = summary
                topic_id = n["id"]
                break
        else:
            slug = _slugify(name)
            if not slug:
                return None
            topic_id = f"topic_{slug}"
            if _node_by_id(g, topic_id):
                topic_id = f"topic_{slug}-{uuid.uuid4().hex[:4]}"
            g["nodes"].append({
                "id": topic_id, "type": "topic", "layer": LAYER_OF["topic"],
                "name": name.strip(), "summary": summary, "parent_id": None,
                "created_at": _now(),
            })
        if parent:
            for n in g["nodes"]:
                if n.get("type") == "topic" and (n["id"] == parent or n.get("name", "").lower() == parent.lower()):
                    if n["id"] != topic_id:
                        topic_node = _node_by_id(g, topic_id)
                        topic_node["parent_id"] = n["id"]
                        if not _has_edge(g, topic_id, n["id"], "subtopic_of"):
                            _add_edge(g, topic_id, n["id"], "subtopic_of", "subtopic of")
                    break
        return topic_id
    return _mutate(where, _fn)


def assign_entity_to_topic(entity: str, topic: str, where: str = "overall") -> bool:
    def _fn(g):
        ent = _resolve_entity(g, entity)
        if not ent:
            return False
        topic_node = None
        for n in g["nodes"]:
            if n.get("type") == "topic" and (n["id"] == topic or n.get("name", "").lower() == topic.strip().lower()):
                topic_node = n
                break
        if topic_node is None:
            slug = _slugify(topic)
            if not slug:
                return False
            tid = f"topic_{slug}"
            topic_node = {"id": tid, "type": "topic", "layer": LAYER_OF["topic"],
                          "name": topic.strip(), "summary": "", "parent_id": None, "created_at": _now()}
            g["nodes"].append(topic_node)
        # remove stale belongs_to edges for this entity
        g["edges"] = [e for e in g["edges"]
                      if not (e["from"] == ent["id"] and e.get("kind") == "belongs_to")]
        ent["topic_id"] = topic_node["id"]
        ent["updated_at"] = _now()
        _add_edge(g, ent["id"], topic_node["id"], "belongs_to", "belongs to")
        return True
    return _mutate(where, _fn)


def add_synthesis(name: str, path: str, abstract: str = "", *, covers=None,
                  where: str = "overall") -> str:
    def _fn(g):
        slug = _slugify(name) or uuid.uuid4().hex[:8]
        sid = f"synthesis_{slug}"
        node = _node_by_id(g, sid)
        if node is None:
            node = {"id": sid, "type": "synthesis", "layer": LAYER_OF["synthesis"],
                    "name": name.strip(), "abstract": abstract, "path": path, "created_at": _now()}
            g["nodes"].append(node)
        else:
            node.update({"abstract": abstract or node.get("abstract", ""), "path": path,
                         "updated_at": _now()})
        for target in (covers or []):
            tgt = _resolve_entity(g, target)
            tid = tgt["id"] if tgt else None
            if not tid:
                for n in g["nodes"]:
                    if n.get("type") == "topic" and n.get("name", "").lower() == str(target).lower():
                        tid = n["id"]
                        break
            if tid and not _has_edge(g, sid, tid, "covers"):
                _add_edge(g, sid, tid, "covers", "covers")
        return sid
    return _mutate(where, _fn)


def add_rag_document(url: str, title: str, date: str, sections: list[dict],
                     chunks: list[dict], where: str = "overall") -> dict:
    """Build a hierarchical doc -> section (contextual summary) -> chunk subgraph.

    Re-ingesting the same URL replaces its prior nodes. ``sections`` items need
    ``id``/``title``/``summary``; ``chunks`` items need ``id``/``section_id``/
    ``text`` (and optionally ``preview``/``position``). Section and chunk ids are
    the same ids used by the vector store, so retrieval hits map onto the graph.
    """
    url = (url or "").strip()

    def _fn(g):
        # Drop any prior nodes/edges for this document (idempotent re-ingest).
        drop = {n["id"] for n in g["nodes"]
                if n.get("type") in ("source", "section", "chunk")
                and (n.get("url") == url or n.get("source_url") == url) and url}
        if drop:
            g["nodes"] = [n for n in g["nodes"] if n["id"] not in drop]
            g["edges"] = [e for e in g["edges"]
                          if e["from"] not in drop and e["to"] not in drop]

        src_id = f"source_{_slugify(url or title) or uuid.uuid4().hex[:8]}"
        if _node_by_id(g, src_id):
            src_id = f"source_{uuid.uuid4().hex[:10]}"
        g["nodes"].append({
            "id": src_id, "type": "source", "layer": LAYER_OF["source"],
            "title": title or url, "url": url, "date": date or "",
            "created_at": _now(),
        })

        for sec in sections:
            g["nodes"].append({
                "id": sec["id"], "type": "section", "layer": LAYER_OF["section"],
                "title": sec.get("title") or title, "name": sec.get("title") or title,
                "summary": sec.get("summary", ""), "url": url, "date": date or "",
                "source_id": src_id, "created_at": _now(),
            })
            _add_edge(g, sec["id"], src_id, "in_document", "in document")

        for ch in chunks:
            text = ch.get("text", "")
            g["nodes"].append({
                "id": ch["id"], "type": "chunk", "layer": LAYER_OF["chunk"],
                "text": text,
                "preview": ch.get("preview") or ((text[:240] + "…") if len(text) > 240 else text),
                "url": url, "source_url": url, "source_title": title,
                "source_id": src_id, "section_id": ch.get("section_id"),
                "date": date or "", "created_at": _now(), "integrated_at": _now(),
            })
            _add_edge(g, ch["id"], src_id, "from_source", "from source")
            if ch.get("section_id"):
                _add_edge(g, ch["id"], ch["section_id"], "in_section", "in section")
        return {"source_id": src_id, "sections": len(sections), "chunks": len(chunks)}

    return _mutate(where, _fn)


def add_extractions(items: list[dict], where: str = "overall") -> dict:
    """Batch-add entities / relations / mentions from extractions over existing chunks.

    ``items`` = [{"chunk_ids": [...], "entities": [...], "relations": [...]}]. One
    load/save for the whole batch. Mentions link each listed chunk -> entity so
    chunk-level graph-RAG (1-hop) reaches the entities.
    """
    summary = {"entities_added": 0, "entities_reinforced": 0,
               "relations_added": 0, "mentions_added": 0}

    def _fn(g):
        index = _entity_index(g)
        chunk_ids = {n["id"] for n in g["nodes"] if n.get("type") == "chunk"}
        existing_rel = {(e["from"], e["to"], e.get("label")) for e in g["edges"]
                        if e.get("kind") == "relation"}
        existing_men = {(e["from"], e["to"]) for e in g["edges"] if e.get("kind") == "mentions"}
        for it in items:
            cids = [c for c in it.get("chunk_ids", []) if c in chunk_ids]
            name_to_id: dict[str, str] = {}
            for ent in it.get("entities", []):
                if isinstance(ent, str):
                    ent = {"name": ent}
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                eid = _upsert_entity(g, index, name, ent.get("kind") or "concept",
                                     summary=ent.get("summary") or "", aliases=ent.get("aliases"),
                                     importance=ent.get("importance"), summary_counts=summary)
                if not eid:
                    continue
                name_to_id[name.lower()] = eid
                for cid in cids:
                    if (cid, eid) not in existing_men:
                        _add_edge(g, cid, eid, "mentions", "mentions",
                                  confidence=ent.get("confidence") or "EXTRACTED", chunk_id=cid)
                        existing_men.add((cid, eid))
                        summary["mentions_added"] += 1
            for rel in it.get("relations", []):
                if not isinstance(rel, dict):
                    continue
                s = (rel.get("source") or "").strip()
                t = (rel.get("target") or "").strip()
                pred = (rel.get("predicate") or "related to").strip()
                if not s or not t:
                    continue
                sid = name_to_id.get(s.lower()) or _upsert_entity(g, index, s, "concept",
                                                                  confidence="INFERRED", summary_counts=summary)
                tid = name_to_id.get(t.lower()) or _upsert_entity(g, index, t, "concept",
                                                                  confidence="INFERRED", summary_counts=summary)
                if not sid or not tid or sid == tid:
                    continue
                if (sid, tid, pred) not in existing_rel:
                    _add_edge(g, sid, tid, "relation", pred, confidence=rel.get("confidence") or "EXTRACTED")
                    existing_rel.add((sid, tid, pred))
                    summary["relations_added"] += 1
        return summary

    return _mutate(where, _fn)


def chunks_mentioning(names_or_ids, where: str = "overall", *, exclude_urls=None,
                      limit: int = 50) -> list[dict]:
    """Return chunk nodes that mention any of the given entities (by name/id),
    optionally excluding some source URLs. The non-embedding retrieval path for graph RAG."""
    g = get_graph(where)
    eids = set()
    for x in names_or_ids:
        ent = _resolve_entity(g, x)
        if ent:
            eids.add(ent["id"])
    if not eids:
        return []
    by_id = {n["id"]: n for n in g["nodes"]}
    exclude = set(exclude_urls or [])
    out, seen = [], set()
    for e in g["edges"]:
        if e.get("kind") == "mentions" and e["to"] in eids:
            ch = by_id.get(e["from"])
            if ch and ch["id"] not in seen and ch.get("url") not in exclude:
                seen.add(ch["id"])
                out.append(ch)
                if len(out) >= limit:
                    break
    return out


def chunk_entities(chunk_id: str, where: str = "overall") -> list[str]:
    """Names of entities a chunk mentions (1-hop)."""
    g = get_graph(where)
    by_id = {n["id"]: n for n in g["nodes"]}
    names = []
    for e in g["edges"]:
        if e.get("kind") == "mentions" and e["from"] == chunk_id:
            ent = by_id.get(e["to"])
            if ent and ent.get("name"):
                names.append(ent["name"])
    return names


def _shared_entity_edges(g: dict, min_shared: int = 2) -> list[dict]:
    """Connect sources that mention >= min_shared common entities (a document map)."""
    chunk_src = {n["id"]: n.get("source_id") for n in g["nodes"] if n.get("type") == "chunk"}
    ent_sources: dict[str, set] = {}
    for e in g["edges"]:
        if e.get("kind") == "mentions":
            src = chunk_src.get(e["from"])
            if src:
                ent_sources.setdefault(e["to"], set()).add(src)
    pair_counts: dict[tuple, int] = {}
    for srcs in ent_sources.values():
        ordered = sorted(s for s in srcs if s)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                key = (ordered[i], ordered[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1
    out = []
    for (a, b), c in pair_counts.items():
        if c >= min_shared:
            out.append({"id": f"shares_{a}_{b}"[:60], "from": a, "to": b,
                        "kind": "shares", "label": f"{c} shared", "weight": c})
    return out
