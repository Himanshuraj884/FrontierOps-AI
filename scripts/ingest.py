"""
Ingestion pipeline: reads data/documents/*.md, chunks them, embeds the
chunks, and builds a vector index in memory (returned, and optionally
pickled to disk for reuse by the API server).

Run: python scripts/ingest.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.chunking import chunk_document  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

DOCS_DIR = ROOT / "data" / "documents"
INDEX_PATH = ROOT / "data" / "index.pkl"


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        docs.append({"document_id": path.stem, "title": title, "text": text})
    return docs


def build_index(chunk_size_words: int = 120, overlap_words: int = 20, embed_dim: int = 512):
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(doc["document_id"], doc["title"], doc["text"], chunk_size_words, overlap_words)
        )

    embedder = get_embedder(dim=embed_dim)
    embedder.fit([c.text for c in all_chunks])

    vectors = embedder.encode([c.text for c in all_chunks])
    store = get_vector_store(dim=vectors.shape[1])
    store.add(
        ids=[c.chunk_id for c in all_chunks],
        vectors=vectors,
        payloads=[
            {
                "document_id": c.document_id,
                "title": c.title,
                "section": c.section,
                "text": c.text,
            }
            for c in all_chunks
        ],
    )
    return embedder, store, all_chunks


if __name__ == "__main__":
    embedder, store, chunks = build_index()
    print(f"Ingested {len(chunks)} chunks from {len(load_documents())} documents.")
    print(f"Vector store backend: {type(store).__name__}, dim={store.dim}, size={len(store)}")
    for c in chunks[:3]:
        print(f"  - {c.chunk_id} [{c.section}] {c.text[:70]}...")
