"""
Vector store abstraction.

Production: FAISS (IndexFlatIP over normalized vectors == cosine similarity).
Fallback: NumpyVectorStore, a real (not mocked) brute-force cosine-similarity
search implemented in numpy, used automatically when the `faiss` package
isn't installed (e.g. this sandbox has no network to install it). Same
interface either way, so swapping backends requires no code changes upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    payload: dict


class NumpyVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids: List[str] = []
        self._payloads: List[dict] = []

    def add(self, ids: List[str], vectors: np.ndarray, payloads: List[dict]) -> None:
        assert vectors.shape[1] == self.dim
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        if len(self._ids) == 0:
            return []
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        v_norms = np.linalg.norm(self._vectors, axis=1)
        v_norms[v_norms == 0] = 1.0
        sims = (self._vectors @ q) / (v_norms * q_norm)
        top_idx = np.argsort(-sims)[:top_k]
        return [
            SearchResult(chunk_id=self._ids[i], score=float(sims[i]), payload=self._payloads[i])
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._ids)


class FaissVectorStore:
    """Thin wrapper around faiss.IndexFlatIP. Only constructed if faiss
    imports successfully; see get_vector_store()."""

    def __init__(self, dim: int, faiss_module):
        self.faiss = faiss_module
        self.dim = dim
        self.index = faiss_module.IndexFlatIP(dim)
        self._ids: List[str] = []
        self._payloads: List[dict] = []

    def add(self, ids: List[str], vectors, payloads: List[dict]) -> None:
        self.index.add(vectors.astype("float32"))
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector, top_k: int = 3) -> List[SearchResult]:
        import numpy as _np

        q = _np.asarray([query_vector], dtype="float32")
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(chunk_id=self._ids[idx], score=float(score), payload=self._payloads[idx])
            )
        return results

    def __len__(self) -> int:
        return len(self._ids)


def get_vector_store(dim: int):
    """Returns FaissVectorStore if faiss is installed, else NumpyVectorStore."""
    try:
        import faiss  # type: ignore

        return FaissVectorStore(dim, faiss)
    except Exception:
        return NumpyVectorStore(dim)
