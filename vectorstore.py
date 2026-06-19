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
from pathlib import Path

import numpy as np

_BASE = Path(os.environ.get("KG_DATA_DIR", "data")) / "vectors"


def _try_hnsw():
    try:
        import hnswlib  # noqa: F401
        return hnswlib
    except Exception:  # noqa: BLE001
        return None


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
               section_weight: float = 0.35, restrict_to_sections: bool = False) -> list[dict]:
        """Hierarchical retrieve: rank sections, then blend chunk + parent-section
        scores. Returns chunk hits enriched with section context and scores."""
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
        cand_idx, cand_sim = idx.top(q, max(k * 6, 30))
        top_set = set(top_section_ids)
        results = []
        for j, sim in zip(cand_idx.tolist(), cand_sim.tolist()):
            rec = self.chunks[j]
            sec_id = rec.get("section_id")
            sec_sim = section_sim.get(sec_id, 0.0)
            if restrict_to_sections and top_set and sec_id not in top_set:
                continue
            blended = float(sim) + section_weight * sec_sim
            results.append({
                "id": rec["id"], "url": rec.get("url"), "title": rec.get("title"),
                "date": rec.get("date"), "section_id": sec_id,
                "section_title": rec.get("section_title"),
                "contextual_summary": rec.get("contextual_summary"),
                "text": rec.get("text"),
                "chunk_score": round(float(sim), 4),
                "section_score": round(float(sec_sim), 4),
                "score": round(blended, 4),
            })
        results.sort(key=lambda r: -r["score"])
        return results[:k]


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
