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
        return json.loads(path.read_text())
    except Exception:
        return _empty_graph()


def _save(path: Path, data: dict) -> None:
    data["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
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
    "Might", "Must", "Shall", "Can", "Cannot",
}


def _heuristic_entities(text: str) -> list[dict]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)
    counts: dict[str, int] = {}
    for c in candidates:
        if c.split()[0] in STOPWORDS:
            continue
        counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:6]
    return [{"name": name, "kind": "concept", "confidence": "INFERRED"} for name, _ in top]


def integrate(extract_fn=None) -> dict:
    summary = {
        "chunks_integrated": 0,
        "entities_added": 0,
        "entities_reinforced": 0,
        "edges_added": 0,
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

            entities: list = []
            if extract_fn:
                try:
                    entities = extract_fn(chunk["text"]) or []
                except Exception as e:
                    summary["errors"].append(str(e))
                    entities = []
            if not entities:
                entities = _heuristic_entities(chunk["text"])

            for ent in entities:
                if isinstance(ent, str):
                    ent = {"name": ent, "kind": "concept", "confidence": "EXTRACTED"}
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                slug = _slugify(name)
                if not slug:
                    continue
                ent_id = f"entity_{slug}"
                kind = ent.get("kind") or "concept"
                confidence = ent.get("confidence") or "EXTRACTED"

                if ent_id in existing_entities:
                    existing_entities[ent_id]["mentions"] = existing_entities[ent_id].get("mentions", 0) + 1
                    existing_entities[ent_id]["updated_at"] = _now()
                    summary["entities_reinforced"] += 1
                else:
                    node = {
                        "id": ent_id,
                        "type": "entity",
                        "name": name,
                        "kind": kind,
                        "mentions": 1,
                        "created_at": _now(),
                    }
                    overall["nodes"].append(node)
                    existing_entities[ent_id] = node
                    summary["entities_added"] += 1

                overall["edges"].append({
                    "id": f"e_{uuid.uuid4().hex[:10]}",
                    "from": new_chunk["id"],
                    "to": ent_id,
                    "label": "mentions",
                    "confidence": confidence,
                    "created_at": _now(),
                })
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
