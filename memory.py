"""Memory layer — the library's long-lived, evolving memory (graph layer 6).

This is what makes the library *dynamic across sessions*. The knowledge graph
(sources → chunks → entities → topics → syntheses) and the vector index already
grow when you ingest documents; the memory layer adds the missing half:

  * RECALL    — every agent and RAG pass retrieves relevant memories and folds
                them into its context (semantic recall when an OpenAI key is set,
                a keyword fallback otherwise), so the library *remembers*.
  * WRITE-BACK — answering questions, running maintenance, and explicit user
                feedback store new or updated memories, so the library keeps
                *improving from its own use*, not only when documents are added.

Unlike a chunk (immutable evidence quoted from a source) a memory is something
the library *learned* or was *told*: a confirmed fact, a durable answer, a user
goal/preference, a known gap, or a correction. Memories reinforce with reuse
(``use_count`` / ``salience``) and can supersede one another (corrections),
mirroring how entities strengthen in the knowledge graph.

Storage is a single plain-JSON file (``data/memory.json``) to match the repo's
local-first design; per-record embeddings are stored inline so recall needs no
extra index files. No new heavy dependencies — just the existing OpenAI
embeddings (optional) and numpy.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import numpy as np

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH = DATA_DIR / "memory.json"

SCHEMA_VERSION = 1
LAYER = 6  # sits above synthesis (5) in the layered knowledge model

# Kinds of memory, from most evidential to most directive.
MEMORY_KINDS = {
    "fact",        # a learned / confirmed fact about the subject matter
    "answer",      # a durable answer to a question the library was asked
    "preference",  # a user goal, instruction, or standing preference
    "gap",         # a known gap: something the library could not answer yet
    "correction",  # a feedback correction (usually supersedes a prior memory)
    "observation", # a general observation made while curating the library
}

# Confidence levels, ranked so reinforcement can only strengthen a memory.
_CONF_RANK = {"AMBIGUOUS": 0, "INFERRED": 1, "EXTRACTED": 2, "CONFIRMED": 3, "USER": 4}

# Recall tuning (overridable via env for experimentation).
DEDUP_THRESHOLD = float(os.environ.get("KG_MEMORY_DEDUP_THRESHOLD", "0.92"))
_SEMANTIC_FLOOR = float(os.environ.get("KG_MEMORY_RECALL_FLOOR", "0.20"))
_KEYWORD_FLOOR = float(os.environ.get("KG_MEMORY_RECALL_FLOOR_KEYWORD", "0.05"))
_RECENCY_DAYS = 30.0

_lock = RLock()
# Small mtime-keyed cache so repeated recalls don't re-read/parse the JSON file.
_cache: dict = {"mtime": None, "data": None}


# --- low-level store ---------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict:
    return {"version": SCHEMA_VERSION, "created_at": _now(),
            "updated_at": _now(), "records": []}


def _load() -> dict:
    if not MEMORY_PATH.exists():
        return _empty()
    try:
        mtime = MEMORY_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _cache["data"] is not None and _cache["mtime"] == mtime:
        return _cache["data"]
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  (corrupt file -> start clean rather than crash)
        return _empty()
    data.setdefault("version", SCHEMA_VERSION)
    data.setdefault("records", [])
    _cache["data"], _cache["mtime"] = data, mtime
    return data


def _save(data: dict) -> None:
    data["updated_at"] = _now()
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MEMORY_PATH)
    try:
        _cache["data"], _cache["mtime"] = data, MEMORY_PATH.stat().st_mtime
    except OSError:
        _cache["data"], _cache["mtime"] = data, None


def _preview(text: str, n: int = 200) -> str:
    text = (text or "").strip()
    return (text[:n] + "…") if len(text) > n else text


def _conf_rank(conf: str) -> int:
    return _CONF_RANK.get((conf or "").upper(), 1)


# --- embeddings (optional, best-effort) --------------------------------------
def embeddings_on() -> bool:
    try:
        import embeddings as emb
        return emb.embeddings_available()
    except Exception:  # noqa: BLE001
        return False


def _embed(text: str) -> list[float] | None:
    """Embed a single string, or return None if embeddings are unavailable.

    Best-effort: any failure (no key, network, quota) degrades to the keyword
    path rather than raising, so memory never blocks an answer."""
    if not text or not embeddings_on():
        return None
    try:
        import embeddings as emb
        return emb.embed_query(text).astype(np.float32).tolist()
    except Exception:  # noqa: BLE001
        return None


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _keyword_sim(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    overlap = q & _tokens(text)
    return len(overlap) / len(q)


def _recency_bonus(rec: dict) -> float:
    stamp = rec.get("updated_at") or rec.get("created_at")
    if not stamp:
        return 0.0
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - when).total_seconds() / 86400.0
    return max(0.0, 1.0 - days / _RECENCY_DAYS)


# --- public shape ------------------------------------------------------------
_PUBLIC_FIELDS = (
    "id", "kind", "text", "preview", "summary", "salience", "confidence",
    "origin", "tags", "source_url", "use_count", "reinforced",
    "created_at", "updated_at", "last_used_at", "superseded_by",
)


def _public(rec: dict) -> dict:
    return {f: rec.get(f) for f in _PUBLIC_FIELDS}


def _active(data: dict) -> list[dict]:
    return [r for r in data["records"] if not r.get("superseded_by")]


def _by_id(data: dict, mem_id: str) -> dict | None:
    for r in data["records"]:
        if r.get("id") == mem_id:
            return r
    return None


# --- write -------------------------------------------------------------------
def _find_duplicate(data: dict, text: str, vec: list[float] | None, kind: str) -> dict | None:
    """Find an existing active memory that says the same thing (so we reinforce
    instead of piling up near-identical entries)."""
    best, best_sim = None, 0.0
    for r in _active(data):
        if vec is not None and r.get("embedding"):
            sim = _cosine(vec, r["embedding"])
            if sim >= DEDUP_THRESHOLD and sim > best_sim:
                best, best_sim = r, sim
        elif r.get("kind") == kind:
            sim = _keyword_sim(text, r.get("text", ""))
            # keyword path is stricter (both directions) to avoid false merges
            if sim >= 0.9 and _keyword_sim(r.get("text", ""), text) >= 0.9 and sim > best_sim:
                best, best_sim = r, sim
    return best


def _reinforce(rec: dict, *, salience: int | None, confidence: str | None,
               tags: list[str] | None, origin: str | None) -> None:
    rec["reinforced"] = rec.get("reinforced", 0) + 1
    rec["updated_at"] = _now()
    if salience in (1, 2, 3, 4, 5):
        rec["salience"] = max(rec.get("salience", 3), salience)
    if confidence and _conf_rank(confidence) > _conf_rank(rec.get("confidence", "INFERRED")):
        rec["confidence"] = confidence.upper()
    if origin:
        rec["origin"] = origin
    if tags:
        merged = list(dict.fromkeys([*rec.get("tags", []), *tags]))
        rec["tags"] = merged


def remember(text: str, *, kind: str = "fact", salience: int = 3,
             confidence: str = "INFERRED", origin: str = "user",
             tags: list[str] | None = None, source_url: str = "",
             summary: str = "", dedup: bool = True) -> dict | None:
    """Store a memory (or reinforce an existing near-identical one).

    Returns the (public) memory record, or None for empty text. Duplicate
    detection keeps the store compact: re-asserting a known memory bumps its
    salience / reinforced count instead of adding a twin."""
    text = (text or "").strip()
    if not text:
        return None
    kind = kind if kind in MEMORY_KINDS else "fact"
    salience = salience if salience in (1, 2, 3, 4, 5) else 3
    tags = [t.strip() for t in (tags or []) if t and t.strip()]
    vec = _embed(text)
    with _lock:
        data = _load()
        if dedup:
            dup = _find_duplicate(data, text, vec, kind)
            if dup is not None:
                _reinforce(dup, salience=salience, confidence=confidence,
                           tags=tags, origin=origin)
                _save(data)
                return _public(dup)
        rec = {
            "id": f"memory_{uuid.uuid4().hex[:12]}",
            "type": "memory",
            "layer": LAYER,
            "kind": kind,
            "text": text,
            "preview": _preview(text),
            "summary": summary or "",
            "salience": salience,
            "confidence": (confidence or "INFERRED").upper(),
            "origin": origin or "user",
            "tags": tags,
            "source_url": source_url or "",
            "use_count": 0,
            "reinforced": 0,
            "superseded_by": None,
            "embedding": vec,
            "created_at": _now(),
            "updated_at": _now(),
            "last_used_at": None,
        }
        data["records"].append(rec)
        _save(data)
        return _public(rec)


def bump_use(ids: list[str]) -> int:
    """Mark memories as actually used in an answer (reinforces by recency/use)."""
    if not ids:
        return 0
    wanted = set(ids)
    n = 0
    with _lock:
        data = _load()
        for r in data["records"]:
            if r.get("id") in wanted:
                r["use_count"] = r.get("use_count", 0) + 1
                r["last_used_at"] = _now()
                n += 1
        if n:
            _save(data)
    return n


def record_feedback(*, question: str = "", answer: str = "", rating: str = "",
                    correction: str = "", memory_id: str = "",
                    origin: str = "feedback") -> dict | None:
    """Fold user feedback back into memory.

    - ``correction`` text → a high-salience ``correction`` memory; if
      ``memory_id`` is given, that memory is superseded by the correction.
    - ``rating`` "up"/"down" on a ``memory_id`` reinforces or demotes it.
    - a "down" rating without a correction records a ``gap`` to fill.
    """
    rating = (rating or "").strip().lower()
    correction = (correction or "").strip()

    if correction:
        body = correction if not question else f"Q: {question}\nCorrected answer: {correction}"
        new = remember(body, kind="correction", salience=5, confidence="USER",
                       origin=origin, tags=["feedback"], dedup=False)
        if memory_id and new:
            with _lock:
                data = _load()
                old = _by_id(data, memory_id)
                if old is not None:
                    old["superseded_by"] = new["id"]
                    old["updated_at"] = _now()
                    _save(data)
        return new

    if memory_id and rating in ("up", "down"):
        with _lock:
            data = _load()
            rec = _by_id(data, memory_id)
            if rec is None:
                return None
            if rating == "up":
                rec["salience"] = min(5, rec.get("salience", 3) + 1)
                rec["confidence"] = "CONFIRMED" if _conf_rank("CONFIRMED") > \
                    _conf_rank(rec.get("confidence", "INFERRED")) else rec.get("confidence")
                rec["reinforced"] = rec.get("reinforced", 0) + 1
            else:
                rec["salience"] = max(1, rec.get("salience", 3) - 1)
            rec["updated_at"] = _now()
            _save(data)
            return _public(rec)

    if rating == "down" and question:
        return remember(f"Answer to '{question}' was rated poor — needs better "
                        f"sources or a correction.", kind="gap", salience=4,
                        confidence="USER", origin=origin, tags=["feedback"])
    return None


def forget(mem_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["records"])
        data["records"] = [r for r in data["records"] if r.get("id") != mem_id]
        # detach dangling supersede pointers
        for r in data["records"]:
            if r.get("superseded_by") == mem_id:
                r["superseded_by"] = None
        changed = len(data["records"]) != before
        if changed:
            _save(data)
        return changed


def clear() -> None:
    with _lock:
        _save(_empty())


# --- read --------------------------------------------------------------------
def recall(query: str, *, k: int = 6, kinds: list[str] | None = None,
           min_score: float | None = None) -> list[dict]:
    """Return up to ``k`` memories most relevant to ``query``.

    Scores by semantic similarity when embeddings are available, otherwise by
    keyword overlap, blended with a small salience and recency bonus. Read-only:
    call ``bump_use`` with the ids that actually informed an answer."""
    with _lock:
        data = _load()
        recs = _active(data)
    if kinds:
        kinds = set(kinds)
        recs = [r for r in recs if r.get("kind") in kinds]
    if not recs:
        return []
    qv = _embed(query) if query else None
    semantic = qv is not None
    floor = min_score if min_score is not None else (_SEMANTIC_FLOOR if semantic else _KEYWORD_FLOOR)

    scored = []
    for r in recs:
        if semantic and r.get("embedding"):
            sim = _cosine(qv, r["embedding"])
        else:
            sim = _keyword_sim(query, r.get("text", ""))
        final = sim + 0.04 * (r.get("salience", 3) - 3) + 0.02 * _recency_bonus(r)
        scored.append((final, sim, r))
    scored.sort(key=lambda x: -x[0])

    out = []
    for final, sim, r in scored[:k]:
        if sim < floor:
            continue
        row = _public(r)
        row["score"] = round(final, 4)
        row["similarity"] = round(sim, 4)
        out.append(row)
    return out


def format_memories(mems: list[dict]) -> str:
    """Render recalled memories as a compact context block for prompt injection."""
    if not mems:
        return ""
    lines = ["The library remembers (prior learnings from earlier sessions — use "
             "if relevant, and verify against current evidence):"]
    for m in mems:
        tag = f"[{m.get('kind', 'fact')}]"
        lines.append(f"- {tag} {m.get('text', '').strip()}")
    return "\n".join(lines)


def recall_block(query: str, *, k: int = 6, kinds: list[str] | None = None,
                 mark_used: bool = True) -> tuple[str, list[str]]:
    """Convenience for callers: recall, format, and (optionally) mark used.

    Returns ``(block_text, used_ids)``; ``block_text`` is "" when nothing matches."""
    mems = recall(query, k=k, kinds=kinds)
    if not mems:
        return "", []
    ids = [m["id"] for m in mems]
    if mark_used:
        bump_use(ids)
    return format_memories(mems), ids


def list_memories(*, kind: str | None = None, limit: int = 100,
                  include_superseded: bool = False) -> list[dict]:
    with _lock:
        data = _load()
        recs = data["records"] if include_superseded else _active(data)
    if kind:
        recs = [r for r in recs if r.get("kind") == kind]
    # salience desc, then most-recently-updated first
    recs = sorted(recs, key=lambda r: (r.get("salience", 3), r.get("updated_at", "")),
                  reverse=True)
    return [_public(r) for r in recs[:max(1, limit)]]


def get_memory(mem_id: str) -> dict | None:
    with _lock:
        data = _load()
        rec = _by_id(data, mem_id)
    return _public(rec) if rec else None


def stats() -> dict:
    with _lock:
        data = _load()
        recs = data["records"]
    active = [r for r in recs if not r.get("superseded_by")]
    by_kind: dict[str, int] = {}
    for r in active:
        by_kind[r.get("kind", "fact")] = by_kind.get(r.get("kind", "fact"), 0) + 1
    return {
        "total": len(active),
        "superseded": len(recs) - len(active),
        "by_kind": by_kind,
        "embedded": sum(1 for r in active if r.get("embedding")),
        "embeddings": embeddings_on(),
        "uses": sum(r.get("use_count", 0) for r in active),
    }
