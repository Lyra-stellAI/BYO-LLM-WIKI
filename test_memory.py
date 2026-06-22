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
os.environ.setdefault("KG_DATA_DIR", tempfile.mkdtemp(prefix="mem_test_"))
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
