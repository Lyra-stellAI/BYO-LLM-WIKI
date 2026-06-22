"""Embeddings for contextual retrieval.

Uses OpenAI's embedding API (Anthropic has no embeddings endpoint). Vectors are
L2-normalized so a dot product equals cosine similarity, which lets the HNSW
index and the numpy fallback share the same scoring.
"""

from __future__ import annotations

import os

import numpy as np

import config

DEFAULT_EMBED_MODEL = os.environ.get("KG_EMBED_MODEL", "text-embedding-3-small")
_BATCH = 96


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


def embeddings_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("openai package is not installed.") from exc
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EmbeddingError(
            "OPENAI_API_KEY is required for embeddings (used by the vector store)."
        )
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    return config.traced_openai(client)


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """Embed a list of texts. Returns an (n, dim) float32, L2-normalized array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    client = _client()
    model = model or DEFAULT_EMBED_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = [t if t.strip() else " " for t in texts[i:i + _BATCH]]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return _normalize(np.asarray(out, dtype=np.float32))


def embed_query(text: str, model: str | None = None) -> np.ndarray:
    """Embed a single query string. Returns a (dim,) L2-normalized vector."""
    mat = embed_texts([text or " "], model=model)
    return mat[0]
