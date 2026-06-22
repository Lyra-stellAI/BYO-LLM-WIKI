"""Canonical cached-content store: ingest once, reuse everywhere.

This module is the SINGLE SOURCE OF TRUTH for raw extracted content. A URL (or
pasted text / uploaded file) is fetched and parsed exactly once and stored as a
``CacheRecord``. The knowledge graph (``knowledge_graph.py``) and the vector
library (``vectorstore.py``) are DERIVED projections: each record tracks whether
it has been projected into the KG (``in_kg`` + ``kg_chunk_ids``) and into the
vector store (``vectorized`` + ``vector_doc_id``). The KG-convert and Q&A
vectorize flows read ``raw_text`` from here instead of re-fetching the network.

Storage is pluggable (``CACHED_STORE_BACKEND``):

  * ``local`` (default) - ``$KG_DATA_DIR/cached/items.json`` (the light index,
    no raw bodies inline) plus per-item sidecars ``cached/bodies/<id>.txt`` for
    the raw text. Reuses the proven atomic-save + RLock pattern from
    ``knowledge_graph.py`` so large pages never bloat the single index file.
  * ``supabase`` (opt-in) - one row per record in ``cached.cached_items`` over a
    psycopg connection pool (reusing the pgbouncer-safe kwargs from
    ``skill_graph.py``). Falls back to local on any connection failure, the same
    graceful contract as the skill-graph checkpointer.

Embeddings are never stored on the record in either backend - they stay a
derived projection in the vector store, keeping this layer small and
backend-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import nullcontext as _nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))

SCHEMA_VERSION = 1

# Projection-state fields preserved across a content-drift re-ingest (so a
# changed page keeps pointing at what it previously produced, and the stored
# *_projected_hash — the content hash at projection time — flags it as stale).
_FLAG_FIELDS = ("in_kg", "kg_chunk_ids", "kg_source_id", "kg_projected_hash",
                "vectorized", "vector_doc_id", "vector_projected_hash")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def make_record(*, kind: str, source_url: str = "", source_title: str = "",
                raw_text: str = "", date: str = "", tags=None, note: str = "",
                origin: str = "manual", backend: str = "local") -> dict:
    """Build a CacheRecord with a content-addressed id.

    ``url``-kind items dedupe on ``source_url`` (so the same page fetched twice
    is one record, despite ad/timestamp drift in the body); ``text``/``file``
    items dedupe on the content hash. ``content_hash`` is always the hash of the
    raw text, used for drift detection on re-ingest.
    """
    raw_text = raw_text or ""
    content_hash = _sha256(raw_text)
    dedupe_key = source_url if (kind == "url" and source_url) else content_hash
    rid = "cache_" + _sha256(dedupe_key)[:16]
    now = _now()
    return {
        "id": rid,
        "kind": kind,
        "source_url": (source_url or "").strip(),
        "source_title": (source_title or source_url or "Untitled").strip(),
        "date": (date or "").strip(),
        "raw_text": raw_text,
        "chars": len(raw_text),
        "content_hash": content_hash,
        "tags": [t.strip() for t in (tags or []) if t and t.strip()],
        "note": (note or "").strip(),
        "origin": origin,
        "created_at": now,
        "updated_at": now,
        "in_kg": False,
        "kg_chunk_ids": [],
        "kg_source_id": "",
        "kg_projected_hash": "",
        "vectorized": False,
        "vector_doc_id": "",
        "vector_projected_hash": "",
        "backend": backend,
    }


def kg_stale(rec: dict) -> bool:
    """True if the item is in the KG but its content changed since projection."""
    return bool(rec.get("in_kg")) and rec.get("kg_projected_hash", "") != rec.get("content_hash", "")


def vector_stale(rec: dict) -> bool:
    """True if the item is vectorized but its content changed since projection."""
    return bool(rec.get("vectorized")) and rec.get("vector_projected_hash", "") != rec.get("content_hash", "")


def url_cache_id(url: str) -> str:
    """The content-addressed id a url-kind record gets (dedupe key = the URL).
    Lets callers check existence by URL the same way ``ingest`` dedupes."""
    return "cache_" + _sha256((url or "").strip())[:16]


def projection_url(rec: dict) -> str:
    """The url used when projecting a record into the KG / vector store.

    Pasted-text and file records have no real URL, so they use a synthetic
    ``cache:<id>`` url. This keeps ``VectorStore.remove_doc`` and
    ``kg.add_rag_document`` (both keyed by url) idempotent for every kind.
    Strips first so the key matches the pipeline's (also stripped) projection
    url — otherwise a whitespace-only source_url would project and delete under
    different keys and orphan the vectors.
    """
    return (rec.get("source_url") or "").strip() or f"cache:{rec['id']}"


def _public(rec: dict, *, with_text: bool = False) -> dict:
    """A copy of a record for API responses (drops the bulky raw_text unless asked)."""
    out = {k: v for k, v in rec.items() if k != "raw_text"}
    if with_text:
        out["raw_text"] = rec.get("raw_text", "")
    return out


# --- Local backend -----------------------------------------------------------
class LocalCachedStore:
    """Local-first store: a light JSON index + per-item raw-text sidecars."""

    backend_name = "local"

    def __init__(self):
        self.dir = DATA_DIR / "cached"
        self.bodies = self.dir / "bodies"
        self.index_path = self.dir / "items.json"
        self.bodies.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    # persistence (atomic write + tolerant read, mirroring knowledge_graph.py) --
    def _load(self) -> dict:
        """Return the index as a flat ``{id: record}`` dict (records lack raw_text)."""
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return {r["id"]: r for r in data.get("records", []) if r.get("id")}

    def _save(self, by_id: dict) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "updated_at": _now(),
            "records": list(by_id.values()),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.index_path)

    def _body_path(self, rid: str) -> Path:
        return self.bodies / f"{rid}.txt"

    def _write_body(self, rid: str, raw_text: str) -> None:
        tmp = self._body_path(rid).with_suffix(".tmp")
        tmp.write_text(raw_text or "", encoding="utf-8")
        tmp.replace(self._body_path(rid))

    def _read_body(self, rid: str) -> str:
        p = self._body_path(rid)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # public API -------------------------------------------------------------
    def put(self, record: dict) -> dict:
        rid = record["id"]
        raw_text = record.get("raw_text", "")
        with self._lock:
            self._write_body(rid, raw_text)
            by_id = self._load()
            by_id[rid] = {k: v for k, v in record.items() if k != "raw_text"}
            self._save(by_id)
        return _public(record)

    def get(self, rid: str, *, with_text: bool = False) -> dict | None:
        with self._lock:
            rec = self._load().get(rid)
            if rec is None:
                return None
            rec = dict(rec)
            if with_text:
                rec["raw_text"] = self._read_body(rid)
            return rec

    def get_by_hash(self, content_hash: str) -> dict | None:
        with self._lock:
            for rec in self._load().values():
                if rec.get("content_hash") == content_hash:
                    return dict(rec)
        return None

    def list(self, *, in_kg: bool | None = None, vectorized: bool | None = None,
             q: str = "", limit: int = 0) -> list[dict]:
        with self._lock:
            recs = list(self._load().values())
        recs = _filter(recs, in_kg=in_kg, vectorized=vectorized, q=q)
        recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        if limit and limit > 0:
            recs = recs[:limit]
        return [_public(r) for r in recs]

    def delete(self, rid: str) -> bool:
        with self._lock:
            by_id = self._load()
            existed = rid in by_id
            by_id.pop(rid, None)
            self._save(by_id)
            p = self._body_path(rid)
            if p.exists():
                p.unlink()
        return existed

    def set_flags(self, rid: str, **flags) -> dict | None:
        with self._lock:
            by_id = self._load()
            rec = by_id.get(rid)
            if rec is None:
                return None
            for k, v in flags.items():
                rec[k] = v
            rec["updated_at"] = _now()
            self._save(by_id)
            return _public(rec)

    def stats(self) -> dict:
        with self._lock:
            recs = list(self._load().values())
        return {
            "backend": self.backend_name,
            "items": len(recs),
            "in_kg": sum(1 for r in recs if r.get("in_kg")),
            "vectorized": sum(1 for r in recs if r.get("vectorized")),
            "chars": sum(int(r.get("chars") or 0) for r in recs),
        }


def _filter(recs: list[dict], *, in_kg, vectorized, q) -> list[dict]:
    out = recs
    if in_kg is not None:
        out = [r for r in out if bool(r.get("in_kg")) == in_kg]
    if vectorized is not None:
        out = [r for r in out if bool(r.get("vectorized")) == vectorized]
    if q:
        ql = q.lower()
        out = [r for r in out
               if ql in (r.get("source_title") or "").lower()
               or ql in (r.get("source_url") or "").lower()]
    return out


# --- Supabase backend (opt-in, gated) ---------------------------------------
def _pg_conninfo() -> str | None:
    """Connection string for the cloud cache (own var wins, else Supabase's)."""
    return (os.environ.get("CACHED_STORE_DB_URL")
            or os.environ.get("SUPABASE_DB_URL") or "").strip() or None


def _pg_schema() -> str:
    s = os.environ.get("CACHED_STORE_PG_SCHEMA", "cached").strip()
    return re.sub(r"[^a-zA-Z0-9_]", "", s) or "cached"


class SupabaseCachedStore:
    """Cloud content backend over psycopg. Reuses the pgbouncer-safe pool kwargs
    from skill_graph.py. Raw text lives in the row; embeddings stay local."""

    backend_name = "supabase"

    def __init__(self):
        conninfo = _pg_conninfo()
        if not conninfo:
            raise RuntimeError("no CACHED_STORE_DB_URL / SUPABASE_DB_URL set")
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row

        self.schema = _pg_schema()
        self._table = f'"{self.schema}".cached_items'

        def _configure(conn):
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                cur.execute(f'SET search_path TO "{self.schema}", public')

        self.pool = ConnectionPool(
            conninfo=conninfo, min_size=1,
            max_size=int(os.environ.get("CACHED_STORE_PG_POOL", "4")),
            kwargs={"autocommit": True, "prepare_threshold": None,
                    "row_factory": dict_row, "connect_timeout": 10},
            configure=_configure, open=False,
        )
        try:
            self.pool.open(wait=True, timeout=15)
            self._setup()
        except Exception:
            try:
                self.pool.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def _setup(self) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id text PRIMARY KEY,
                    kind text,
                    source_url text,
                    source_title text,
                    date text,
                    raw_text text NOT NULL DEFAULT '',
                    chars int DEFAULT 0,
                    content_hash text,
                    tags jsonb DEFAULT '[]',
                    note text DEFAULT '',
                    origin text DEFAULT 'manual',
                    in_kg boolean DEFAULT false,
                    kg_chunk_ids jsonb DEFAULT '[]',
                    kg_source_id text DEFAULT '',
                    kg_projected_hash text DEFAULT '',
                    vectorized boolean DEFAULT false,
                    vector_doc_id text DEFAULT '',
                    vector_projected_hash text DEFAULT '',
                    backend text DEFAULT 'supabase',
                    created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now()
                )""")
            # Backfill the projected-hash columns on tables created before they existed.
            for col in ("kg_projected_hash", "vector_projected_hash"):
                cur.execute(f"ALTER TABLE {self._table} "
                            f"ADD COLUMN IF NOT EXISTS {col} text DEFAULT ''")
            cur.execute(f"CREATE INDEX IF NOT EXISTS cached_items_hash_idx "
                        f"ON {self._table} (content_hash)")

    @staticmethod
    def _row_to_record(row: dict, *, with_text: bool) -> dict:
        rec = {
            "id": row["id"], "kind": row.get("kind") or "url",
            "source_url": row.get("source_url") or "",
            "source_title": row.get("source_title") or "",
            "date": row.get("date") or "", "chars": int(row.get("chars") or 0),
            "content_hash": row.get("content_hash") or "",
            "tags": row.get("tags") or [], "note": row.get("note") or "",
            "origin": row.get("origin") or "manual",
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "in_kg": bool(row.get("in_kg")), "kg_chunk_ids": row.get("kg_chunk_ids") or [],
            "kg_source_id": row.get("kg_source_id") or "",
            "kg_projected_hash": row.get("kg_projected_hash") or "",
            "vectorized": bool(row.get("vectorized")),
            "vector_doc_id": row.get("vector_doc_id") or "",
            "vector_projected_hash": row.get("vector_projected_hash") or "",
            "backend": "supabase",
        }
        if with_text:
            rec["raw_text"] = row.get("raw_text") or ""
        return rec

    def put(self, record: dict) -> dict:
        from psycopg.types.json import Jsonb
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self._table}
                  (id, kind, source_url, source_title, date, raw_text, chars,
                   content_hash, tags, note, origin, in_kg, kg_chunk_ids,
                   kg_source_id, kg_projected_hash, vectorized, vector_doc_id,
                   vector_projected_hash, backend, created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now())
                ON CONFLICT (id) DO UPDATE SET
                  kind=EXCLUDED.kind, source_url=EXCLUDED.source_url,
                  source_title=EXCLUDED.source_title, date=EXCLUDED.date,
                  raw_text=EXCLUDED.raw_text, chars=EXCLUDED.chars,
                  content_hash=EXCLUDED.content_hash, tags=EXCLUDED.tags,
                  note=EXCLUDED.note, origin=EXCLUDED.origin, in_kg=EXCLUDED.in_kg,
                  kg_chunk_ids=EXCLUDED.kg_chunk_ids, kg_source_id=EXCLUDED.kg_source_id,
                  kg_projected_hash=EXCLUDED.kg_projected_hash,
                  vectorized=EXCLUDED.vectorized, vector_doc_id=EXCLUDED.vector_doc_id,
                  vector_projected_hash=EXCLUDED.vector_projected_hash,
                  updated_at=now()
            """, (record["id"], record.get("kind"), record.get("source_url"),
                  record.get("source_title"), record.get("date"),
                  record.get("raw_text", ""), record.get("chars", 0),
                  record.get("content_hash"), Jsonb(record.get("tags") or []),
                  record.get("note", ""), record.get("origin", "manual"),
                  bool(record.get("in_kg")), Jsonb(record.get("kg_chunk_ids") or []),
                  record.get("kg_source_id", ""), record.get("kg_projected_hash", ""),
                  bool(record.get("vectorized")), record.get("vector_doc_id", ""),
                  record.get("vector_projected_hash", ""),
                  record.get("backend", "supabase")))
        return _public(record)

    def get(self, rid: str, *, with_text: bool = False) -> dict | None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {self._table} WHERE id=%s", (rid,))
            row = cur.fetchone()
        return self._row_to_record(row, with_text=with_text) if row else None

    def get_by_hash(self, content_hash: str) -> dict | None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {self._table} WHERE content_hash=%s LIMIT 1",
                        (content_hash,))
            row = cur.fetchone()
        return self._row_to_record(row, with_text=False) if row else None

    def list(self, *, in_kg=None, vectorized=None, q="", limit=0) -> list[dict]:
        where, params = [], []
        if in_kg is not None:
            where.append("in_kg=%s"); params.append(in_kg)
        if vectorized is not None:
            where.append("vectorized=%s"); params.append(vectorized)
        if q:
            where.append("(source_title ILIKE %s OR source_url ILIKE %s)")
            params += [f"%{q}%", f"%{q}%"]
        sql = (f"SELECT id, kind, source_url, source_title, date, chars, content_hash, "
               f"tags, note, origin, in_kg, kg_chunk_ids, kg_source_id, kg_projected_hash, "
               f"vectorized, vector_doc_id, vector_projected_hash, created_at, updated_at "
               f"FROM {self._table}")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_record(r, with_text=False) for r in rows]

    def delete(self, rid: str) -> bool:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE id=%s", (rid,))
            return cur.rowcount > 0

    def set_flags(self, rid: str, **flags) -> dict | None:
        from psycopg.types.json import Jsonb
        allowed = {"in_kg", "kg_chunk_ids", "kg_source_id", "kg_projected_hash",
                   "vectorized", "vector_doc_id", "vector_projected_hash"}
        sets, params = [], []
        for k, v in flags.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=%s")
            params.append(Jsonb(v) if k == "kg_chunk_ids" else v)
        if not sets:
            return self.get(rid)
        params.append(rid)
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {self._table} SET {', '.join(sets)}, updated_at=now() "
                        f"WHERE id=%s", params)
        return self.get(rid)

    def stats(self) -> dict:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) AS items, "
                        f"count(*) FILTER (WHERE in_kg) AS in_kg, "
                        f"count(*) FILTER (WHERE vectorized) AS vectorized, "
                        f"coalesce(sum(chars),0) AS chars FROM {self._table}")
            row = cur.fetchone()
        return {"backend": self.backend_name, "items": int(row["items"]),
                "in_kg": int(row["in_kg"]), "vectorized": int(row["vectorized"]),
                "chars": int(row["chars"])}


# --- backend resolution + ingest --------------------------------------------
_STORE = None


def get_store():
    """Resolve the active backend once (CACHED_STORE_BACKEND). Supabase failures
    fall back to local with a loud stderr note, so a misconfigured DB never
    blocks the app (same contract as skill_graph._checkpointer)."""
    global _STORE
    if _STORE is None:
        backend = os.environ.get("CACHED_STORE_BACKEND", "local").strip().lower()
        if backend == "supabase":
            try:
                _STORE = SupabaseCachedStore()
            except Exception as exc:  # noqa: BLE001
                print(f"[cached_store] Supabase backend unavailable "
                      f"({type(exc).__name__}: {exc}); falling back to local.",
                      file=sys.stderr)
                _STORE = LocalCachedStore()
        else:
            _STORE = LocalCachedStore()
    return _STORE


def ingest(*, kind: str, source_url: str = "", source_title: str = "",
           raw_text: str = "", date: str = "", tags=None, note: str = "",
           origin: str = "manual") -> dict:
    """Idempotent ingest. Returns the stored record (raw_text omitted) with a
    transient ``created`` flag. Unchanged content (same id + content_hash) is a
    no-op reuse; changed content updates the body but PRESERVES projection flags
    so callers can detect drift (stored content_hash vs the hash of what was
    last projected)."""
    store = get_store()
    rec = make_record(kind=kind, source_url=source_url, source_title=source_title,
                      raw_text=raw_text, date=date, tags=tags, note=note,
                      origin=origin, backend=store.backend_name)
    # Make the read-merge-write atomic. The local backend's RLock is reentrant,
    # so get()/put() can re-acquire it; backends without a lock (Supabase) rely
    # on the row-level INSERT ... ON CONFLICT for atomicity.
    lock = getattr(store, "_lock", None)
    with (lock if lock is not None else _nullcontext()):
        existing = store.get(rec["id"])
        if existing is not None:
            if existing.get("content_hash") == rec["content_hash"]:
                out = _public(existing)
                out["created"] = False
                return out
            # Content changed: keep id + projection flags (drift), refresh body/hash.
            for f in _FLAG_FIELDS:
                rec[f] = existing.get(f, rec[f])
            rec["created_at"] = existing.get("created_at", rec["created_at"])
        out = store.put(rec)
    out["created"] = existing is None
    return out


def _vector_slug(url: str) -> str:
    """Mirror pipeline._slug without importing pipeline (avoids a heavy import)."""
    s = re.sub(r"^https?://", "", (url or "").lower())
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80] or "doc"


def backfill(*, overall_path: Path | None = None, vector_dir: Path | None = None) -> dict:
    """One-time, idempotent, NON-DESTRUCTIVE backfill of pre-existing content
    into the cache so it shows up (already projected) in the pickers.

    Reads ``overall.json`` (KG sources/chunks) and ``vectors/library/chunks.jsonl``,
    groups by url, and per url either creates a cache record (raw_text rebuilt
    best-effort from chunk text) or, if a record already exists, only updates its
    projection flags — never overwriting real extracted raw_text.
    """
    store = get_store()
    overall_path = overall_path or (DATA_DIR / "overall.json")
    vector_dir = vector_dir or (DATA_DIR / "vectors" / "library")

    agg: dict[str, dict] = {}

    def slot(url: str) -> dict:
        return agg.setdefault(url, {"title": "", "date": "", "kg": False, "vec": False,
                                    "kg_source_id": "", "kg_chunk_ids": [],
                                    "kg_texts": [], "vec_texts": []})

    if overall_path.exists():
        try:
            data = json.loads(overall_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
        for n in data.get("nodes", []):
            t = n.get("type")
            url = (n.get("url") or n.get("source_url") or "").strip()
            if not url:
                continue
            if t == "source":
                s = slot(url); s["kg"] = True
                s["title"] = s["title"] or (n.get("title") or "")
                s["date"] = s["date"] or (n.get("date") or "")
                s["kg_source_id"] = n.get("id") or s["kg_source_id"]
            elif t == "chunk":
                s = slot(url); s["kg"] = True
                if n.get("id"):
                    s["kg_chunk_ids"].append(n["id"])
                if n.get("text"):
                    s["kg_texts"].append(n["text"])
                s["title"] = s["title"] or (n.get("source_title") or n.get("title") or "")

    chunks_jsonl = vector_dir / "chunks.jsonl"
    if chunks_jsonl.exists():
        for line in chunks_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            url = (r.get("url") or "").strip()
            if not url:
                continue
            s = slot(url); s["vec"] = True
            s["title"] = s["title"] or (r.get("title") or "")
            s["date"] = s["date"] or (r.get("date") or "")
            if r.get("text"):
                s["vec_texts"].append(r["text"])

    created = updated = skipped = 0
    for url, s in agg.items():
        if not url or url.startswith("cache:"):
            skipped += 1
            continue
        rec = make_record(kind="url", source_url=url, source_title=s["title"] or url,
                          raw_text="", date=s["date"], origin="backfill",
                          backend=store.backend_name)
        flags = {}
        if s["kg"]:
            flags.update(in_kg=True, kg_chunk_ids=s["kg_chunk_ids"], kg_source_id=s["kg_source_id"])
        if s["vec"]:
            flags.update(vectorized=True, vector_doc_id=_vector_slug(url))
        existing = store.get(rec["id"])
        if existing is not None:
            # Non-destructive: only update flags; mark projections current against
            # the record's REAL (already-stored) content hash so it isn't flagged stale.
            eh = existing.get("content_hash", "")
            if s["kg"]:
                flags["kg_projected_hash"] = eh
            if s["vec"]:
                flags["vector_projected_hash"] = eh
            store.set_flags(rec["id"], **flags)
            updated += 1
            continue
        texts = s["kg_texts"] or s["vec_texts"]
        raw_text = "\n\n".join(t for t in texts if t)
        if not raw_text.strip():
            skipped += 1
            continue
        rec["raw_text"] = raw_text
        rec["chars"] = len(raw_text)
        rec["content_hash"] = _sha256(raw_text)   # recompute (make_record saw empty text)
        if s["kg"]:
            flags["kg_projected_hash"] = rec["content_hash"]
        if s["vec"]:
            flags["vector_projected_hash"] = rec["content_hash"]
        rec.update(flags)
        store.put(rec)
        created += 1
    return {"backend": store.backend_name, "created": created,
            "updated": updated, "skipped": skipped, "urls": len(agg)}
