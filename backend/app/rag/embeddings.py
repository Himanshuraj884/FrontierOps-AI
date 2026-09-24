"""
Embedding layer for the RAG pipeline.

Production usage: install `sentence-transformers` and use SentenceTransformerEmbedder
(all-MiniLM-L6-v2 or similar). This module auto-detects it.

Offline fallback: HashingEmbedder — a deterministic, dependency-free
bag-of-words hashing vectorizer with TF-IDF-style weighting, implemented in
pure numpy. It is not as semantically strong as a real sentence embedding
model, but it is a real, working vector representation (not a stub) so the
rest of the pipeline (retrieval, evaluation) can run end-to-end without
network access or GPU dependencies.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str) -> int:
    """Deterministic hash, unlike Python's built-in hash() which is
    randomized per-process (PYTHONHASHSEED) and would make embeddings —
    and therefore retrieval scores and evaluation metrics — non-reproducible
    across runs."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Deterministic hashing + TF-IDF-weighted bag-of-words embedder.

    Every call produces the same vector for the same text (needed for
    reproducible evaluation). Dimensionality is fixed; collisions are
    accepted as a standard hashing-trick tradeoff.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._doc_freq: Counter = Counter()
        self._num_docs: int = 0
        self._fitted = False

    def fit(self, corpus: List[str]) -> "HashingEmbedder":
        self._doc_freq = Counter()
        self._num_docs = len(corpus)
        for doc in corpus:
            seen = set(tokenize(doc))
            for tok in seen:
                self._doc_freq[tok] += 1
        self._fitted = True
        return self

    def _idf(self, token: str) -> float:
        if not self._fitted or self._num_docs == 0:
            return 1.0
        df = self._doc_freq.get(token, 0)
        return math.log((1 + self._num_docs) / (1 + df)) + 1.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec
        tf = Counter(tokens)
        for tok, count in tf.items():
            idx = _stable_hash(tok) % self.dim
            weight = (count / len(tokens)) * self._idf(tok)
            vec[idx] += weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


def get_embedder(dim: int = 512):
    """Returns the best available embedder. Tries sentence-transformers
    first (production quality); falls back to HashingEmbedder if the
    package isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _STWrapper:
            def fit(self, corpus):
                return self

            def encode(self, texts):
                return np.asarray(model.encode(texts, normalize_embeddings=True))

        return _STWrapper()
    except Exception:
        return HashingEmbedder(dim=dim)
