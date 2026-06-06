"""Chunking layer: split documents into retrieval-friendly segments."""
from __future__ import annotations

from src.chunking.text_splitter import Chunk, SentenceAwareChunker

__all__ = ["Chunk", "SentenceAwareChunker"]
