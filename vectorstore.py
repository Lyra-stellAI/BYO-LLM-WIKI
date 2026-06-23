"""Two-layer vector store for hierarchical contextual retrieval.

Layers:
  * sections - coarse: one contextual-summary embedding per document section
  * chunks   - fine:   one embedding per chunk (context prepended before embedding)

Retrieval is hierarchical: a query matches section summaries first, then drills
into the chunks, blending the fine chunk score with the coarse parent-section
score. Uses an HNSW index (hnswlib) for the chunk layer when available, with a
numpy cosine fallback. Vectors are assumed L2-normalized (see embeddings.py),
so dot product == cosine similarity.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

_BASE = Path(os.environ.get("KG_DATA_DIR", "data")) / "vectors"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _try_hnsw():
    try:
        import hnswlib  # noqa: F401
        return hnswlib
    except Exception:  # noqa: BLE001
        return None


class _BM25Index:
    """Okapi BM25 over chunk texts — a lexical/sparse complement to the dense
    HNSW index. Catches exact terms, identifiers, acronyms, and rare tokens that
    embeddings blur. Pure numpy, no extra dependency."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in docs_tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: max(0.0, float(np.log((self.n - d + 0.5) / (d + 0.5) + 1.0)))
                    for t, d in df.items()}

    def top(self, query_text: str, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (chunk indices, scores) for the top-m BM25 matches (zero-score
        docs — no query term present — are dropped)."""
        if self.n == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        q = [t for t in set(_tokenize(query_text)) if t in self.idf]
        if not q:
            return np.array([], dtype=int), np.array([], dtype=float)
        scores = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            counts, dl = self.tf[i], self.doc_len[i]
            norm = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            s = 0.0
            for term in q:
                tf = counts.get(term, 0)
                if tf:
                    s += self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)
            scores[i] = s
        m = min(m, self.n)
        idx = np.argpartition(-scores, m - 1)[:m]
        idx = idx[np.argsort(-scores[idx])]
        keep = [int(i) for i in idx if scores[i] > 0.0]
        return np.array(keep, dtype=int), scores


class _ChunkIndex:
    """HNSW index over chunk vectors with a numpy fallback (same ranking)."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.n, self.dim = (embeddings.shape if embeddings.size else (0, 0))
        self._hnsw = None
        hnswlib = _try_hnsw()
        if hnswlib is not None and self.n > 0:
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.init_index(max_elements=self.n, ef_construction=200, M=16)
            idx.add_items(self.embeddings, np.arange(self.n))
            idx.set_ef(max(64, self.n if self.n < 64 else 64))
            self._hnsw = idx

    def top(self, q: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) of the top-m chunks for query q."""
        if self.n == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        m = min(m, self.n)
        if self._hnsw is not None:
            labels, dists = self._hnsw.knn_query(q, k=m)
            return labels[0], 1.0 - dists[0]  # cosine distance -> similarity
        sims = self.embeddings @ q
        idx = np.argpartition(-sims, m - 1)[:m]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]


class VectorStore:
    def __init__(self, name: str = "library", embed_model: str = "text-embedding-3-small"):
        self.name = name
        self.dir = _BASE / name
        self.embed_model = embed_model
        self.chunks: list[dict] = []
        self.sections: list[dict] = []
        self.chunk_emb = np.zeros((0, 0), dtype=np.float32)
        self.section_emb = np.zeros((0, 0), dtype=np.float32)
        self._index: _ChunkIndex | None = None
        self._bm25: _BM25Index | None = None

    # --- persistence --------------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps({
            "name": self.name, "embed_model": self.embed_model,
            "dim": int(self.chunk_emb.shape[1]) if self.chunk_emb.size else 0,
            "chunks": len(self.chunks), "sections": len(self.sections),
        }, indent=2), encoding="utf-8")
        _write_jsonl(self.dir / "chunks.jsonl", self.chunks)
        _write_jsonl(self.dir / "sections.jsonl", self.sections)
        np.save(self.dir / "chunks.npy", self.chunk_emb)
        np.save(self.dir / "sections.npy", self.section_emb)

    @classmethod
    def load(cls, name: str = "library") -> "VectorStore":
        vs = cls(name=name)
        if not vs.dir.exists():
            return vs
        meta_path = vs.dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vs.embed_model = meta.get("embed_model", vs.embed_model)
        vs.chunks = _read_jsonl(vs.dir / "chunks.jsonl")
        vs.sections = _read_jsonl(vs.dir / "sections.jsonl")
        vs.chunk_emb = _load_npy(vs.dir / "chunks.npy")
        vs.section_emb = _load_npy(vs.dir / "sections.npy")
        vs._index = None
        return vs

    # --- mutation -----------------------------------------------------------
    def add(self, chunk_records, chunk_embeddings, section_records, section_embeddings) -> None:
        """Append chunk + section records and their embeddings."""
        if chunk_records:
            self.chunks.extend(chunk_records)
            self.chunk_emb = _vstack(self.chunk_emb, np.asarray(chunk_embeddings, dtype=np.float32))
        if section_records:
            self.sections.extend(section_records)
            self.section_emb = _vstack(self.section_emb, np.asarray(section_embeddings, dtype=np.float32))
        self._index = None  # invalidate
        self._bm25 = None

    def remove_doc(self, url: str) -> int:
        """Drop all chunks/sections for a document URL (for re-ingest)."""
        keep_c = [i for i, r in enumerate(self.chunks) if r.get("url") != url]
        keep_s = [i for i, r in enumerate(self.sections) if r.get("url") != url]
        removed = len(self.chunks) - len(keep_c)
        if removed or len(keep_s) != len(self.sections):
            self.chunks = [self.chunks[i] for i in keep_c]
            self.chunk_emb = self.chunk_emb[keep_c] if self.chunk_emb.size else self.chunk_emb
            self.sections = [self.sections[i] for i in keep_s]
            self.section_emb = self.section_emb[keep_s] if self.section_emb.size else self.section_emb
            self._index = None
            self._bm25 = None
        return removed

    def doc_urls(self) -> set:
        return {r.get("url") for r in self.sections} | {r.get("url") for r in self.chunks}

    def stats(self) -> dict:
        docs = {r.get("url") for r in self.chunks}
        return {
            "documents": len([u for u in docs if u]),
            "sections": len(self.sections),
            "chunks": len(self.chunks),
            "dim": int(self.chunk_emb.shape[1]) if self.chunk_emb.size else 0,
            "index": "hnsw" if _try_hnsw() else "numpy",
            "embed_model": self.embed_model,
        }

    # --- retrieval ----------------------------------------------------------
    def _ensure_index(self):
        if self._index is None:
            self._index = _ChunkIndex(self.chunk_emb)
        return self._index

    def search(self, query_vec: np.ndarray, k: int = 8, *, n_sections: int = 5,
               section_weight: float = 0.35, restrict_to_sections: bool = False,
               mmr: bool = False, mmr_lambda: float = 0.5, per_doc_penalty: float = 0.4) -> list[dict]:
        """Hierarchical retrieve: rank sections, then blend chunk + parent-section
        scores. Returns chunk hits enriched with section context and scores.

        With ``mmr=True`` the top-k is selected by document-aware Maximal Marginal
        Relevance: each pick maximizes ``mmr_lambda*relevance - (1-mmr_lambda)*
        redundancy``, where redundancy is the max cosine similarity to already-picked
        chunks plus ``per_doc_penalty`` if the chunk's document is already
        represented. This spreads the top-k across documents (lifts multi-doc recall)."""
        if not self.chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)

        # Coarse layer: section summary similarities (few sections -> full compute).
        section_sim: dict[str, float] = {}
        top_section_ids: list[str] = []
        if self.sections and self.section_emb.size:
            s_sims = self.section_emb @ q
            section_sim = {self.sections[i]["id"]: float(s_sims[i]) for i in range(len(self.sections))}
            order = np.argsort(-s_sims)[:n_sections]
            top_section_ids = [self.sections[i]["id"] for i in order]

        # Fine layer: candidate chunks via HNSW (or numpy), then blend.
        idx = self._ensure_index()
        cand_n = max(k * 8, 48) if mmr else max(k * 6, 30)
        cand_idx, cand_sim = idx.top(q, cand_n)
        top_set = set(top_section_ids)
        cands = []  # (row_index, result_dict)
        for j, sim in zip(cand_idx.tolist(), cand_sim.tolist()):
            rec = self.chunks[j]
            sec_id = rec.get("section_id")
            sec_sim = section_sim.get(sec_id, 0.0)
            if restrict_to_sections and top_set and sec_id not in top_set:
                continue
            blended = float(sim) + section_weight * sec_sim
            cands.append((j, {
                "id": rec["id"], "url": rec.get("url"), "title": rec.get("title"),
                "date": rec.get("date"), "section_id": sec_id,
                "section_title": rec.get("section_title"),
                "contextual_summary": rec.get("contextual_summary"),
                "text": rec.get("text"),
                "chunk_score": round(float(sim), 4),
                "section_score": round(float(sec_sim), 4),
                "score": round(blended, 4),
            }))
        cands.sort(key=lambda c: -c[1]["score"])
        if not mmr:
            return [c[1] for c in cands[:k]]
        return self._mmr_select(cands, k, mmr_lambda, per_doc_penalty)

    def _mmr_select(self, cands: list, k: int, lam: float, per_doc_penalty: float) -> list[dict]:
        chosen_js: list[int] = []
        chosen_urls: set = set()
        out: list[dict] = []
        remaining = list(cands)
        while remaining and len(out) < k:
            best, best_score, best_pos = None, float("-inf"), -1
            for pos, (j, rec) in enumerate(remaining):
                rel = rec["score"]
                if chosen_js:
                    red = max(float(self.chunk_emb[j] @ self.chunk_emb[cj]) for cj in chosen_js)
                else:
                    red = 0.0
                if rec.get("url") in chosen_urls:
                    red += per_doc_penalty
                score = lam * rel - (1.0 - lam) * red
                if score > best_score:
                    best, best_score, best_pos = (j, rec), score, pos
            j, rec = best
            rec = dict(rec); rec["mmr_score"] = round(best_score, 4)
            out.append(rec); chosen_js.append(j); chosen_urls.add(rec.get("url"))
            remaining.pop(best_pos)
        return out

    # --- hybrid (dense + BM25) ----------------------------------------------
    def _ensure_bm25(self) -> _BM25Index:
        if self._bm25 is None:
            self._bm25 = _BM25Index([_tokenize(c.get("text", "")) for c in self.chunks])
        return self._bm25

    def _chunk_result(self, j: int, sim: float = 0.0, sec_sim: float = 0.0) -> dict:
        """Build a hit dict for chunk index j (used for BM25-only candidates)."""
        rec = self.chunks[j]
        return {
            "id": rec["id"], "url": rec.get("url"), "title": rec.get("title"),
            "date": rec.get("date"), "section_id": rec.get("section_id"),
            "section_title": rec.get("section_title"),
            "contextual_summary": rec.get("contextual_summary"),
            "text": rec.get("text"), "preview": rec.get("preview"),
            "chunk_score": round(float(sim), 4), "section_score": round(float(sec_sim), 4),
            "score": round(float(sim), 4),
        }

    def search_hybrid(self, query_vec: np.ndarray, query_text: str, k: int = 8, *,
                      n_sections: int = 5, section_weight: float = 0.35,
                      rrf_k: int = 60, cand: int | None = None) -> list[dict]:
        """Dense (hierarchical) + BM25 retrieval fused with Reciprocal Rank
        Fusion. RRF needs no score calibration between the two rankers — it
        combines by rank position, so a chunk that ranks well in *either* signal
        surfaces. Falls back to dense-only when BM25 has no matches."""
        if not self.chunks:
            return []
        cand = cand or max(k * 6, 30)
        dense = self.search(query_vec, k=cand, n_sections=n_sections,
                            section_weight=section_weight)
        bm25_idx, _ = self._ensure_bm25().top(query_text, cand)
        if len(bm25_idx) == 0:
            return dense[:k]
        by_id = {h["id"]: h for h in dense}
        dense_ids = [h["id"] for h in dense]
        bm25_ids = []
        for j in bm25_idx:
            cid = self.chunks[j]["id"]
            bm25_ids.append(cid)
            if cid not in by_id:
                by_id[cid] = self._chunk_result(int(j))
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda c: -fused[c])[:k]
        out = []
        for cid in ranked:
            h = dict(by_id[cid])
            h["rrf_score"] = round(fused[cid], 5)
            out.append(h)
        return out

    # --- section-level (for the coarse-retrieve -> ICL regime) --------------
    def section_full_text(self, section_id: str) -> str:
        """Reconstruct a section's full text from its chunks, in order."""
        chunks = [c for c in self.chunks if c.get("section_id") == section_id]
        chunks.sort(key=lambda c: c.get("position", 0))
        return "\n".join(c.get("text", "") for c in chunks).strip()

    def rank_sections(self, query_vec: np.ndarray, *, restrict_urls: set | None = None,
                      limit: int = 50) -> list[dict]:
        """Rank sections by query↔section-summary similarity (optionally limited
        to a set of document URLs). Returns section dicts with a ``score`` — the
        coarse step of coarse-retrieve-then-ICL."""
        if not self.sections or not self.section_emb.size:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        sims = self.section_emb @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            s = self.sections[int(i)]
            if restrict_urls is not None and s.get("url") not in restrict_urls:
                continue
            row = dict(s)
            row["score"] = round(float(sims[int(i)]), 4)
            out.append(row)
            if len(out) >= limit:
                break
        return out


# --- io helpers --------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 0), dtype=np.float32)
    return np.load(path).astype(np.float32)


def _vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    return np.vstack([a, b])
