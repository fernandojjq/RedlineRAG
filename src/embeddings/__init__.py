"""Embedding layer: turn chunks into dense vector representations."""
from __future__ import annotations

from src.embeddings.embedder import TfidfEmbedder, EmbeddingMatrix

__all__ = ["TfidfEmbedder", "EmbeddingMatrix"]
