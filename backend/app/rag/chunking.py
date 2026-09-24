"""
Document chunking for the RAG ingestion pipeline.

Splits raw document text into overlapping word-based chunks and attaches
metadata (document_id, section, chunk_index) to each chunk. Chunk size and
overlap are configurable rather than hardcoded, per the project spec.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    section: Optional[str]
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def split_into_sections(raw_text: str) -> List[tuple]:
    """Split a markdown-style document into (section_title, section_text) pairs
    using '## Heading' as the section boundary. Falls back to a single
    untitled section if no headings are found."""
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return [(None, raw_text.strip())]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def chunk_text(
    text: str,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
) -> List[str]:
    """Word-based sliding-window chunking. chunk_size_words/overlap_words are
    an approximation of the ~500-800 token / 50-100 token overlap target in
    the spec (roughly 0.75 tokens per word for English prose)."""
    words = text.split()
    if not words:
        return []
    if chunk_size_words <= overlap_words:
        raise ValueError("chunk_size_words must be greater than overlap_words")

    chunks = []
    step = chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def chunk_document(
    document_id: str,
    title: str,
    raw_text: str,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
) -> List[Chunk]:
    """Full pipeline: extract sections, then chunk each section's text."""
    chunks: List[Chunk] = []
    idx = 0
    for section_title, section_text in split_into_sections(raw_text):
        for piece in chunk_text(section_text, chunk_size_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}::{idx}",
                    document_id=document_id,
                    title=title,
                    section=section_title,
                    text=piece,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks
