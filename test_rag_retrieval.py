"""Offline tests for hybrid retrieval (BM25 + dense + RRF), section
reconstruction, and the three-regime ICL/RAG router.

Fully offline: LLM generation, provider resolution, and embeddings are stubbed,
so no API keys or network are needed. Run directly or under pytest.
"""

import os
import tempfile

os.environ["KG_DATA_DIR"] = (os.environ.get("BYOWIKI_TEST_DATA_DIR")
                             or tempfile.mkdtemp(prefix="ragret_test_"))
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import numpy as np  # noqa: E402
import cached_store  # noqa: E402
import rag  # noqa: E402
from vectorstore import VectorStore, _BM25Index, _tokenize  # noqa: E402


# --- hybrid retrieval engine -------------------------------------------------
def test_bm25_ranks_exact_term_and_drops_zero():
    bm = _BM25Index([_tokenize(t) for t in
                     ["the cat sat on the mat", "transformer attention model", "a dog ran"]])
    idx, _ = bm.top("transformer attention", 5)
    assert list(idx)[0] == 1                  # exact-term doc ranks first
    assert 0 not in list(idx) and 2 not in list(idx)  # no-term docs dropped


def _toy_store(name="ragret_toy"):
    vs = VectorStore(name=name)
    vs.chunks = [
        {"id": "c0", "url": "u1", "title": "D1", "section_id": "s0", "position": 0,
         "text": "alpha beta", "contextual_summary": "s0"},
        {"id": "c1", "url": "u1", "title": "D1", "section_id": "s0", "position": 1,
         "text": "gamma RAREWORD", "contextual_summary": "s0"},
        {"id": "c2", "url": "u2", "title": "D2", "section_id": "s1", "position": 0,
         "text": "delta epsilon", "contextual_summary": "s1"},
    ]
    vs.chunk_emb = np.eye(3, 8, dtype=np.float32)
    vs.sections = [
        {"id": "s0", "url": "u1", "title": "D1", "section_title": "Sec0", "summary": "s0"},
        {"id": "s1", "url": "u2", "title": "D2", "section_title": "Sec1", "summary": "s1"},
    ]
    vs.section_emb = np.eye(2, 8, dtype=np.float32)
    return vs


def test_hybrid_surfaces_lexical_only_match():
    vs = _toy_store()
    q = np.zeros(8, dtype=np.float32); q[0] = 1.0   # dense favors c0
    hits = vs.search_hybrid(q, "RAREWORD", k=3)
    assert any("RAREWORD" in (h.get("text") or "") for h in hits)  # BM25 surfaced c1
    assert "rrf_score" in hits[0]


def test_hybrid_falls_back_to_dense_when_no_lexical_match():
    vs = _toy_store()
    q = np.zeros(8, dtype=np.float32); q[0] = 1.0
    hits = vs.search_hybrid(q, "zzz nonexistentterm", k=2)
    assert hits and hits[0]["id"] == "c0"


def test_section_full_text_and_ranking():
    vs = _toy_store()
    assert vs.section_full_text("s0") == "alpha beta\ngamma RAREWORD"
    q = np.zeros(8, dtype=np.float32); q[1] = 1.0   # row 1 == section s1
    ranked = vs.rank_sections(q, restrict_urls={"u2"})
    assert [r["id"] for r in ranked] == ["s1"]


# --- three-regime router (generation/embeddings stubbed) ---------------------
def _patch(**overrides):
    saved = {k: getattr(rag, k) for k in overrides}
    for k, v in overrides.items():
        setattr(rag, k, v)
    def restore():
        for k, v in saved.items():
            setattr(rag, k, v)
    return restore


def test_regime_no_focus_is_hybrid_rag():
    restore = _patch(
        resolve_provider_model=lambda p, m: ("anthropic", "claude-test"),
        retrieve=lambda *a, **k: [{"id": "c0", "title": "D", "url": "u", "text": "t",
                                   "preview": "p", "score": 0.9, "section_title": "s", "date": ""}],
        _answer_from_hits=lambda *a, **k: "RAG_ANSWER [1]",
    )
    try:
        res = rag.answer("q?", use_memory=False, write_back=False)
        assert res["regime"] == "rag"
        assert res["scope"]["passages"] == 1 and res["scope"]["hybrid"] is True
    finally:
        restore()


def test_regime_small_focus_is_icl_full_documents():
    item = cached_store.ingest(kind="text", source_title="Doc A",
                               raw_text="short body about cats")
    captured = {}
    restore = _patch(
        resolve_provider_model=lambda p, m: ("anthropic", "claude-test"),
        _icl_generate=lambda q, blocks, rp, rm, mem="": captured.update(blocks=blocks) or "ICL [1]",
    )
    try:
        res = rag.answer("q?", focus_ids=[item["id"]], use_memory=False, write_back=False)
        assert res["regime"] == "icl-documents"
        assert res["scope"]["documents"] == 1
        # the WHOLE document text is placed in context (no chunking)
        assert captured["blocks"][0]["text"] == "short body about cats"
    finally:
        restore()


def test_regime_large_focus_coarse_retrieves_sections():
    url = "https://ex.com/secdoc"
    vs = VectorStore(name="ragret_sections")
    vs.add(
        chunk_records=[
            {"id": "k0", "url": url, "title": "SecDoc", "section_id": "sa", "position": 0, "text": "section A body one"},
            {"id": "k1", "url": url, "title": "SecDoc", "section_id": "sa", "position": 1, "text": "section A body two"},
            {"id": "k2", "url": url, "title": "SecDoc", "section_id": "sb", "position": 0, "text": "section B body"},
        ],
        chunk_embeddings=np.eye(3, 4, dtype=np.float32),
        section_records=[
            {"id": "sa", "url": url, "title": "SecDoc", "section_title": "A", "summary": "a"},
            {"id": "sb", "url": url, "title": "SecDoc", "section_title": "B", "summary": "b"},
        ],
        section_embeddings=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
    )
    vs.persist()
    item = cached_store.ingest(kind="url", source_url=url, source_title="SecDoc",
                               raw_text="x" * 400)   # ~100 est-tokens
    # Mark FRESH-vectorized (projected hash matches current content) so the
    # section path trusts the vector index.
    cached_store.get_store().set_flags(item["id"], vectorized=True,
                                       vector_projected_hash=item["content_hash"])
    captured = {}
    saved_eq = rag.emb.embed_query
    rag.emb.embed_query = lambda text, model=None: np.array([1, 0, 0, 0], dtype=np.float32)
    restore = _patch(
        resolve_provider_model=lambda p, m: ("anthropic", "claude-test"),
        ICL_BUDGET_TOKENS=10,   # force over-budget -> section regime
        _icl_generate=lambda q, blocks, rp, rm, mem="": captured.update(blocks=blocks) or "ICL [1]",
    )
    try:
        res = rag.answer("q?", focus_ids=[item["id"]], vs_name="ragret_sections",
                         use_memory=False, write_back=False)
        assert res["regime"] == "icl-sections", res["regime"]
        # top-ranked section (sa) reconstructed to FULL text, fed to ICL
        assert any("section A body one" in b["text"] for b in captured["blocks"])
    finally:
        restore()
        rag.emb.embed_query = saved_eq


def test_regime_stale_vectorized_focus_falls_back_to_raw_text():
    """A doc re-ingested after vectorizing (vectorized but content changed) must
    NOT be served from its stale vector chunks — the section path falls back to
    its current raw_text so focused answers never cite outdated content."""
    url = "https://ex.com/stale"
    vs = VectorStore(name="ragret_stale")
    vs.add(
        chunk_records=[{"id": "z0", "url": url, "title": "Doc", "section_id": "zs",
                        "position": 0, "text": "OLD STALE SECTION CONTENT"}],
        chunk_embeddings=np.eye(1, 4, dtype=np.float32),
        section_records=[{"id": "zs", "url": url, "title": "Doc", "section_title": "Z", "summary": "old"}],
        section_embeddings=np.array([[1, 0, 0, 0]], dtype=np.float32),
    )
    vs.persist()
    item = cached_store.ingest(kind="url", source_url=url, source_title="Doc",
                               raw_text="NEW CURRENT CONTENT " + "x" * 400)
    # vectorized, but against an OLD hash -> vector_stale is True
    cached_store.get_store().set_flags(item["id"], vectorized=True,
                                       vector_projected_hash="OLDHASH_does_not_match")
    captured = {}
    saved_eq = rag.emb.embed_query
    rag.emb.embed_query = lambda text, model=None: np.array([1, 0, 0, 0], dtype=np.float32)
    restore = _patch(
        resolve_provider_model=lambda p, m: ("anthropic", "claude-test"),
        ICL_BUDGET_TOKENS=10,
        _icl_generate=lambda q, blocks, rp, rm, mem="": captured.update(blocks=blocks) or "ICL [1]",
    )
    try:
        res = rag.answer("q?", focus_ids=[item["id"]], vs_name="ragret_stale",
                         use_memory=False, write_back=False)
        body = " ".join(b["text"] for b in captured["blocks"])
        assert "NEW CURRENT CONTENT" in body              # current raw_text used
        assert "OLD STALE SECTION CONTENT" not in body    # stale vector chunk NOT used
        assert res["scope"].get("stale_fallback") == 1
    finally:
        restore()
        rag.emb.embed_query = saved_eq


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} rag-retrieval tests passed.")


if __name__ == "__main__":
    _run_all()
