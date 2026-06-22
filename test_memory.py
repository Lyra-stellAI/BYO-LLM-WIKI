"""Smoke tests for the memory layer (memory.py).

Runs fully offline: no API keys, no network. By unsetting OPENAI_API_KEY before
importing ``memory`` we exercise the keyword-recall + dedup fallback path, so the
test is deterministic everywhere. Run directly (``python test_memory.py``) or
under pytest.
"""

import json
import os
import tempfile

# Isolate state BEFORE importing memory: a throwaway data dir + no embeddings.
os.environ["KG_DATA_DIR"] = (os.environ.get("BYOWIKI_TEST_DATA_DIR")
                             or tempfile.mkdtemp(prefix="mem_test_"))
os.environ.pop("OPENAI_API_KEY", None)

import memory  # noqa: E402


def setup_function(_=None):
    memory.clear()


def test_add_and_keyword_recall():
    memory.clear()
    m = memory.remember("Ada Lovelace wrote the first algorithm.", kind="fact", salience=4)
    assert m and m["id"].startswith("memory_")
    assert m["kind"] == "fact" and m["salience"] == 4
    assert memory.embeddings_on() is False  # keyword path in this test env
    rows = memory.recall("who wrote the first algorithm")
    assert rows and any("Ada" in r["text"] for r in rows), rows
    assert "score" in rows[0] and "similarity" in rows[0]


def test_dedup_reinforces():
    memory.clear()
    memory.remember("HNSW is an approximate nearest neighbour index.", kind="fact")
    memory.remember("HNSW is an approximate nearest neighbour index.", kind="fact")
    active = memory.list_memories()
    assert len(active) == 1, active
    assert active[0]["reinforced"] >= 1


def test_stats_by_kind():
    memory.clear()
    memory.remember("A fact.", kind="fact")
    memory.remember("User prefers concise answers.", kind="preference")
    memory.remember("We can't answer X yet.", kind="gap")
    s = memory.stats()
    assert s["total"] == 3
    assert s["by_kind"].get("fact") == 1
    assert s["by_kind"].get("preference") == 1
    assert s["by_kind"].get("gap") == 1


def test_recall_kind_filter():
    memory.clear()
    memory.remember("Contextual retrieval improves precision.", kind="fact")
    memory.remember("User is researching agent memory.", kind="preference")
    prefs = memory.recall("agent memory research", kinds=["preference"])
    assert prefs and all(r["kind"] == "preference" for r in prefs)


def test_feedback_correction_supersedes():
    memory.clear()
    wrong = memory.remember("The capital is Sydney.", kind="fact", salience=3)
    new = memory.record_feedback(question="capital?", correction="The capital is Canberra.",
                                 memory_id=wrong["id"])
    assert new and new["kind"] == "correction" and new["confidence"] == "USER"
    active = memory.list_memories()
    ids = {m["id"] for m in active}
    assert wrong["id"] not in ids, "superseded memory should drop out of active list"
    assert new["id"] in ids
    assert memory.get_memory(wrong["id"])["superseded_by"] == new["id"]


def test_feedback_rating_adjusts_salience():
    memory.clear()
    m = memory.remember("Re-rankers lift context precision.", kind="fact", salience=3)
    up = memory.record_feedback(memory_id=m["id"], rating="up")
    assert up["salience"] == 4
    down = memory.record_feedback(memory_id=m["id"], rating="down")
    assert down["salience"] == 3


def test_bump_use_and_forget():
    memory.clear()
    m = memory.remember("Document-aware MMR spreads results across docs.", kind="fact")
    assert memory.bump_use([m["id"]]) == 1
    assert memory.get_memory(m["id"])["use_count"] == 1
    assert memory.forget(m["id"]) is True
    assert memory.get_memory(m["id"]) is None


def test_persistence_on_disk():
    memory.clear()
    memory.remember("Persisted across sessions.", kind="fact", salience=5)
    raw = json.loads(memory.MEMORY_PATH.read_text(encoding="utf-8"))
    assert raw["records"] and raw["records"][0]["text"] == "Persisted across sessions."
    # bypass the in-process cache to simulate a fresh process reading the file
    memory._cache["mtime"] = None
    memory._cache["data"] = None
    assert memory.stats()["total"] == 1


def test_backend_local_by_default():
    memory._STORE = None
    assert memory.get_store().backend_name == "local"
    assert memory.stats()["backend"] == "local"


def test_upsert_does_not_delete_absent_records():
    # Models two concurrent writers on a shared backend: instance A stores m1,
    # then a stale writer (that never saw m1) upserts only m2. m1 must survive —
    # the store must never delete rows absent from a caller's snapshot.
    memory.clear()
    m1 = memory.remember("memory from instance A", kind="fact")
    memory.get_store().upsert([{
        "id": "memory_stale_b", "type": "memory", "kind": "fact",
        "text": "memory from a stale instance B", "superseded_by": None,
        "salience": 3, "embedding": None,
        "created_at": memory._now(), "updated_at": memory._now()}])
    ids = {r["id"] for r in memory.list_memories()}
    assert m1["id"] in ids, "stale upsert must not delete another writer's row"
    assert "memory_stale_b" in ids


def test_forget_then_other_records_survive():
    memory.clear()
    a = memory.remember("keep me", kind="fact")
    b = memory.remember("forget me", kind="fact")
    assert memory.forget(b["id"]) is True
    ids = {r["id"] for r in memory.list_memories()}
    assert a["id"] in ids and b["id"] not in ids   # explicit delete is targeted


def test_supabase_backend_falls_back_to_local():
    memory._STORE = None
    os.environ["MEMORY_BACKEND"] = "supabase"   # psycopg not installed in test env
    os.environ["MEMORY_DB_URL"] = "postgresql://x:y@localhost:5432/z"
    try:
        assert memory.get_store().backend_name == "local"   # graceful fallback
        # store still works after falling back
        assert memory.remember("post-fallback memory", kind="fact") is not None
    finally:
        os.environ.pop("MEMORY_BACKEND", None)
        os.environ.pop("MEMORY_DB_URL", None)
        memory._STORE = None


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        setup_function()
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} memory tests passed.")


if __name__ == "__main__":
    _run_all()
