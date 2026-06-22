"""Smoke tests for the cached content store (cached_store.py).

Runs fully offline: no API keys, no network, no server. A throwaway KG_DATA_DIR
is set BEFORE importing cached_store so the local backend reads/writes a clean
temp tree. Run directly (``python test_cached_store.py``) or under pytest.
"""

import json
import os
import shutil
import tempfile

# Isolate state BEFORE importing the module under test.
os.environ.setdefault("KG_DATA_DIR", tempfile.mkdtemp(prefix="cache_test_"))
os.environ.pop("CACHED_STORE_BACKEND", None)

import cached_store as cs  # noqa: E402


def setup_function(_=None):
    """Fresh local store per test: reset the singleton + wipe the cache dir."""
    cs._STORE = None
    d = cs.DATA_DIR / "cached"
    if d.exists():
        shutil.rmtree(d)
    cs.get_store()  # re-resolve (creates data/cached/bodies)


# --- record identity + hashing ----------------------------------------------
def test_make_record_url_dedup_key_is_url():
    a = cs.make_record(kind="url", source_url="https://x.com/a", raw_text="one")
    b = cs.make_record(kind="url", source_url="https://x.com/a", raw_text="TWO different")
    # url-kind dedupes on the URL, so different text -> SAME id.
    assert a["id"] == b["id"] == cs.url_cache_id("https://x.com/a")
    # content_hash still tracks the body (for drift detection).
    assert a["content_hash"] == cs._sha256("one")
    assert b["content_hash"] == cs._sha256("TWO different")
    assert a["content_hash"] != b["content_hash"]
    # defaults
    assert a["in_kg"] is False and a["vectorized"] is False
    assert a["kg_projected_hash"] == "" and a["vector_projected_hash"] == ""


def test_make_record_text_dedup_key_is_content():
    a = cs.make_record(kind="text", raw_text="same body")
    b = cs.make_record(kind="text", raw_text="same body")
    c = cs.make_record(kind="text", raw_text="other body")
    assert a["id"] == b["id"]          # identical text -> same id
    assert a["id"] != c["id"]          # different text -> different id
    assert a["source_url"] == ""


# --- ingest: dedup + drift ---------------------------------------------------
def test_ingest_created_then_reused():
    r1 = cs.ingest(kind="url", source_url="https://x.com/a", raw_text="hello", origin="read_page")
    assert r1["created"] is True
    r2 = cs.ingest(kind="url", source_url="https://x.com/a", raw_text="hello")
    assert r2["created"] is False      # unchanged content -> reuse
    assert cs.get_store().stats()["items"] == 1


def test_ingest_drift_preserves_projection_flags():
    r = cs.ingest(kind="url", source_url="https://x.com/a", raw_text="v1 content")
    cs.get_store().set_flags(r["id"], in_kg=True, kg_chunk_ids=["chunk_1", "chunk_2"],
                             kg_projected_hash=cs._sha256("v1 content"))
    # Re-ingest the SAME url with CHANGED content (drift).
    r2 = cs.ingest(kind="url", source_url="https://x.com/a", raw_text="v2 changed content")
    assert r2["created"] is False                       # same id
    rec = cs.get_store().get(r["id"], with_text=True)
    assert rec["raw_text"] == "v2 changed content"      # body refreshed
    assert rec["content_hash"] == cs._sha256("v2 changed content")
    assert rec["in_kg"] is True                         # projection flags preserved
    assert rec["kg_chunk_ids"] == ["chunk_1", "chunk_2"]
    assert rec["kg_projected_hash"] == cs._sha256("v1 content")  # old hash kept
    assert cs.kg_stale(rec) is True                     # -> flagged stale


# --- local backend CRUD ------------------------------------------------------
def test_put_get_with_text_and_sidecar():
    store = cs.get_store()
    rec = cs.make_record(kind="text", raw_text="body text here", source_title="T")
    store.put(rec)
    listed = store.get(rec["id"])
    assert "raw_text" not in listed                      # list/get omits body by default
    full = store.get(rec["id"], with_text=True)
    assert full["raw_text"] == "body text here"          # body read from sidecar
    assert store._body_path(rec["id"]).exists()


def test_get_by_hash():
    store = cs.get_store()
    rec = cs.make_record(kind="text", raw_text="findable body")
    store.put(rec)
    found = store.get_by_hash(cs._sha256("findable body"))
    assert found and found["id"] == rec["id"]
    assert store.get_by_hash(cs._sha256("nonexistent")) is None


def test_list_filters_and_limit():
    store = cs.get_store()
    a = cs.ingest(kind="url", source_url="https://x.com/alpha", source_title="Alpha", raw_text="a")
    b = cs.ingest(kind="url", source_url="https://x.com/beta", source_title="Beta", raw_text="b")
    store.set_flags(a["id"], in_kg=True)
    store.set_flags(b["id"], vectorized=True)
    assert {r["id"] for r in store.list(in_kg=True)} == {a["id"]}
    assert {r["id"] for r in store.list(in_kg=False)} == {b["id"]}
    assert {r["id"] for r in store.list(vectorized=True)} == {b["id"]}
    assert {r["id"] for r in store.list(q="alpha")} == {a["id"]}     # matches title/url
    assert len(store.list(limit=1)) == 1


def test_delete_removes_index_and_body():
    store = cs.get_store()
    rec = cs.make_record(kind="text", raw_text="to be deleted")
    store.put(rec)
    body = store._body_path(rec["id"])
    assert body.exists()
    assert store.delete(rec["id"]) is True
    assert store.get(rec["id"]) is None
    assert not body.exists()                              # sidecar cleaned up
    assert store.delete(rec["id"]) is False               # idempotent


def test_stats():
    store = cs.get_store()
    a = cs.ingest(kind="url", source_url="https://x.com/a", raw_text="aaaa")
    b = cs.ingest(kind="url", source_url="https://x.com/b", raw_text="bb")
    store.set_flags(a["id"], in_kg=True)
    st = store.stats()
    assert st["backend"] == "local" and st["items"] == 2
    assert st["in_kg"] == 1 and st["vectorized"] == 0
    assert st["chars"] == len("aaaa") + len("bb")


# --- projection url + staleness ---------------------------------------------
def test_projection_url_strips_and_falls_back():
    assert cs.projection_url({"id": "cache_x", "source_url": "https://a.com"}) == "https://a.com"
    assert cs.projection_url({"id": "cache_x", "source_url": "   "}) == "cache:cache_x"
    assert cs.projection_url({"id": "cache_x", "source_url": ""}) == "cache:cache_x"


def test_stale_helpers():
    base = {"content_hash": "h2"}
    assert cs.kg_stale({**base, "in_kg": True, "kg_projected_hash": "h1"}) is True
    assert cs.kg_stale({**base, "in_kg": True, "kg_projected_hash": "h2"}) is False
    assert cs.kg_stale({**base, "in_kg": False, "kg_projected_hash": ""}) is False
    assert cs.vector_stale({**base, "vectorized": True, "vector_projected_hash": "h1"}) is True
    assert cs.vector_stale({**base, "vectorized": True, "vector_projected_hash": "h2"}) is False


# --- backfill ----------------------------------------------------------------
def _write_synthetic_sources():
    """A KG overall.json (one integrated source+chunks) and a vector chunks.jsonl."""
    kg_url = "https://x.com/kg-doc"
    vec_url = "https://x.com/vec-doc"
    overall = {
        "nodes": [
            {"id": "source_kg", "type": "source", "url": kg_url, "title": "KG Doc", "date": "2025-01-01"},
            {"id": "chunk_kg_0", "type": "chunk", "url": kg_url, "text": "kg chunk zero"},
            {"id": "chunk_kg_1", "type": "chunk", "url": kg_url, "text": "kg chunk one"},
        ],
        "edges": [],
    }
    (cs.DATA_DIR / "overall.json").write_text(json.dumps(overall), encoding="utf-8")
    vdir = cs.DATA_DIR / "vectors" / "library"
    vdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "c0", "url": vec_url, "title": "Vec Doc", "date": "2025-02-02", "text": "vec chunk a"},
        {"id": "c1", "url": vec_url, "title": "Vec Doc", "text": "vec chunk b"},
    ]
    (vdir / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return kg_url, vec_url


def test_backfill_creates_records_with_correct_flags():
    kg_url, vec_url = _write_synthetic_sources()
    res = cs.backfill()
    assert res["created"] == 2 and res["urls"] == 2
    store = cs.get_store()

    kg_rec = store.get(cs.url_cache_id(kg_url), with_text=True)
    assert kg_rec["in_kg"] is True and kg_rec["kg_source_id"] == "source_kg"
    assert kg_rec["kg_chunk_ids"] == ["chunk_kg_0", "chunk_kg_1"]
    assert kg_rec["vectorized"] is False
    # content_hash recomputed from reconstructed text (NOT sha256 of empty string).
    assert kg_rec["content_hash"] == cs._sha256("kg chunk zero\n\nkg chunk one")
    assert kg_rec["content_hash"] != cs._sha256("")
    assert cs.kg_stale(kg_rec) is False                  # projected hash set -> not stale

    vec_rec = store.get(cs.url_cache_id(vec_url))
    assert vec_rec["vectorized"] is True and vec_rec["in_kg"] is False
    assert vec_rec["vector_doc_id"] == cs._vector_slug(vec_url)
    assert cs.vector_stale(vec_rec) is False


def test_backfill_is_non_destructive():
    kg_url, _ = _write_synthetic_sources()
    # Pre-existing record with REAL extracted text for the same url.
    pre = cs.ingest(kind="url", source_url=kg_url, source_title="KG Doc",
                    raw_text="the original, full extracted text", origin="read_page")
    res = cs.backfill()
    assert res["updated"] >= 1                            # existing record updated, not recreated
    rec = cs.get_store().get(pre["id"], with_text=True)
    assert rec["raw_text"] == "the original, full extracted text"  # body NOT clobbered
    assert rec["in_kg"] is True                           # flag merged in from backfill


# --- backend resolution ------------------------------------------------------
def test_supabase_backend_falls_back_to_local():
    cs._STORE = None
    os.environ["CACHED_STORE_BACKEND"] = "supabase"   # psycopg not installed in test env
    try:
        store = cs.get_store()
        assert store.backend_name == "local"          # graceful fallback
    finally:
        os.environ.pop("CACHED_STORE_BACKEND", None)
        cs._STORE = None


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        setup_function()
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} cached_store tests passed.")


if __name__ == "__main__":
    _run_all()
