import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_PATH = DATA_DIR / "current.json"
OVERALL_PATH = DATA_DIR / "overall.json"

_lock = Lock()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _empty_graph() -> dict:
    return {
        "version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "nodes": [],
        "edges": [],
    }


def _load(path: Path) -> dict:
    if not path.exists():
        return _empty_graph()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_graph()


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


def stats() -> dict:
    with _lock:
        cur = _load(CURRENT_PATH)
        ov = _load(OVERALL_PATH)

    def counts(g):
        chunks = sum(1 for n in g["nodes"] if n.get("type") == "chunk")
        entities = sum(1 for n in g["nodes"] if n.get("type") == "entity")
        return {"chunks": chunks, "entities": entities, "edges": len(g["edges"])}

    return {"current": counts(cur), "overall": counts(ov)}


def add_chunk(text: str, *, source_url: str | None = None, source_title: str | None = None,
              tags: list[str] | None = None, note: str | None = None) -> dict:
    chunk = {
        "id": f"chunk_{uuid.uuid4().hex[:12]}",
        "type": "chunk",
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
    """Heuristic fallback: prefer multi-word proper-noun phrases; strip
    leading articles ("The Generator" → "Generator") and merge counts;
    drop sentence-start singletons that look like noise ("In", "Round").
    """
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
        first = parts[0]
        if first in STOPWORDS:
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
        if key in seen:
            continue
        if count < 2:
            continue
        if name in sentence_start_singles and count < 3:
            continue
        seen.add(key)
        out.append({"name": name, "kind": "concept", "confidence": "INFERRED"})
        if len(out) >= 18:
            break

    return out[:18]


def _maybe_subchunk(text: str, max_chars: int = 2500) -> list[str]:
    """Split a long chunk for extraction so we don't cap entities at 8 per huge text."""
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
    """Accept either the old list-of-entities or the new {entities, relations} shape."""
    if isinstance(result, dict):
        return {
            "entities": result.get("entities") or [],
            "relations": result.get("relations") or [],
        }
    if isinstance(result, list):
        return {"entities": result, "relations": []}
    return {"entities": [], "relations": []}


def _ensure_entity(overall: dict, existing: dict, name: str, kind: str,
                   confidence: str, summary: dict) -> str | None:
    name = (name or "").strip()
    slug = _slugify(name)
    if not slug:
        return None
    ent_id = f"entity_{slug}"
    if ent_id in existing:
        existing[ent_id]["mentions"] = existing[ent_id].get("mentions", 0) + 1
        existing[ent_id]["updated_at"] = _now()
        summary["entities_reinforced"] += 1
    else:
        node = {
            "id": ent_id,
            "type": "entity",
            "name": name,
            "kind": kind or "concept",
            "mentions": 1,
            "created_at": _now(),
        }
        overall["nodes"].append(node)
        existing[ent_id] = node
        summary["entities_added"] += 1
    return ent_id


def integrate(extract_fn=None) -> dict:
    summary = {
        "chunks_integrated": 0,
        "entities_added": 0,
        "entities_reinforced": 0,
        "edges_added": 0,
        "relation_edges_added": 0,
        "errors": [],
    }
    with _lock:
        cur = _load(CURRENT_PATH)
        overall = _load(OVERALL_PATH)
        existing_entities = {n["id"]: n for n in overall["nodes"] if n.get("type") == "entity"}
        existing_chunk_ids = {n["id"] for n in overall["nodes"] if n.get("type") == "chunk"}

        for chunk in cur["nodes"]:
            if chunk.get("type") != "chunk" or chunk["id"] in existing_chunk_ids:
                continue
            new_chunk = dict(chunk)
            new_chunk["integrated_at"] = _now()
            overall["nodes"].append(new_chunk)
            summary["chunks_integrated"] += 1

            pieces = _maybe_subchunk(chunk["text"])
            all_entities: list[dict] = []
            all_relations: list[dict] = []
            for piece in pieces:
                extracted = None
                if extract_fn:
                    try:
                        extracted = _normalize_extraction(extract_fn(piece))
                    except Exception as e:
                        summary["errors"].append(str(e))
                        extracted = None
                if not extracted or (not extracted["entities"] and not extracted["relations"]):
                    extracted = {"entities": _heuristic_entities(piece), "relations": []}
                all_entities.extend(extracted["entities"])
                all_relations.extend(extracted["relations"])

            name_to_id: dict[str, str] = {}
            for ent in all_entities:
                if isinstance(ent, str):
                    ent = {"name": ent}
                name = (ent.get("name") or "").strip()
                if not name or name.lower() in name_to_id:
                    if name:
                        existing_entities[name_to_id[name.lower()]]["mentions"] = (
                            existing_entities[name_to_id[name.lower()]].get("mentions", 1) + 1
                        )
                    continue
                ent_id = _ensure_entity(
                    overall, existing_entities, name,
                    ent.get("kind") or "concept",
                    ent.get("confidence") or "EXTRACTED",
                    summary,
                )
                if not ent_id:
                    continue
                name_to_id[name.lower()] = ent_id
                overall["edges"].append({
                    "id": f"e_{uuid.uuid4().hex[:10]}",
                    "from": new_chunk["id"],
                    "to": ent_id,
                    "label": "mentions",
                    "kind": "mentions",
                    "confidence": ent.get("confidence") or "EXTRACTED",
                    "created_at": _now(),
                })
                summary["edges_added"] += 1

            for rel in all_relations:
                if not isinstance(rel, dict):
                    continue
                src_name = (rel.get("source") or "").strip()
                tgt_name = (rel.get("target") or "").strip()
                pred = (rel.get("predicate") or "related to").strip()
                if not src_name or not tgt_name:
                    continue
                src_id = name_to_id.get(src_name.lower()) or _ensure_entity(
                    overall, existing_entities, src_name, "concept", "INFERRED", summary
                )
                tgt_id = name_to_id.get(tgt_name.lower()) or _ensure_entity(
                    overall, existing_entities, tgt_name, "concept", "INFERRED", summary
                )
                if not src_id or not tgt_id or src_id == tgt_id:
                    continue
                name_to_id[src_name.lower()] = src_id
                name_to_id[tgt_name.lower()] = tgt_id
                overall["edges"].append({
                    "id": f"e_{uuid.uuid4().hex[:10]}",
                    "from": src_id,
                    "to": tgt_id,
                    "label": pred,
                    "kind": "relation",
                    "confidence": rel.get("confidence") or "EXTRACTED",
                    "chunk_id": new_chunk["id"],
                    "created_at": _now(),
                })
                summary["relation_edges_added"] += 1
                summary["edges_added"] += 1

        _save(OVERALL_PATH, overall)
        _save(CURRENT_PATH, _empty_graph())
    return summary


def query(text: str, where: str = "overall", limit: int = 30) -> list[dict]:
    g = get_graph(where)
    q = (text or "").lower().strip()
    if not q:
        return g["nodes"][:limit]
    matches = []
    for n in g["nodes"]:
        haystack = " ".join([
            n.get("text", ""), n.get("name", ""), n.get("preview", ""),
            n.get("source_title", ""), " ".join(n.get("tags", [])), n.get("note", ""),
        ]).lower()
        if q in haystack:
            matches.append(n)
    return matches[:limit]
