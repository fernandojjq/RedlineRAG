"""TF-IDF based embedder.

Why TF-IDF and not a transformer?

  * It works on a fresh checkout with zero model downloads, zero internet,
    zero PyTorch, and zero GPU. The pipeline boots in under a second.
  * Legal text is full of rare, specific tokens ("indemnify", "irrevocable",
    "binding arbitration", "class-action waiver") that exact n-gram matching
    catches *better* than sentence-transformer embeddings would, because the
    embeddings average away the discriminative signal.
  * A cosine-similarity top-k over a few hundred chunks is microseconds of
    numpy. We do not need approximate nearest neighbours at this scale.

The output is L2-normalized, so dot product == cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking.text_splitter import Chunk
from src.utils.config import PipelineConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class EmbeddingMatrix:
    """A persisted bag of vectors plus the vocabulary they live in."""

    matrix: csr_matrix  # shape = (n_chunks, vocab_size), L2-normalized rows
    chunk_ids: list[str]
    vocabulary_path: Path

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    def to_dense(self) -> np.ndarray:
        """Return a dense float32 array - convenient for small corpora."""
        return self.matrix.toarray().astype(np.float32)


class TfidfEmbedder:
    """Builds and queries a TF-IDF embedding model for legal text."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._vectorizer = TfidfVectorizer(
            ngram_range=config.tfidf_ngram_range,
            min_df=config.tfidf_min_df,
            max_df=config.tfidf_max_df,
            max_features=config.tfidf_max_features,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
            stop_words=config.tfidf_stop_words,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b",
        )

    def fit_transform(self, chunks: Iterable[Chunk]) -> EmbeddingMatrix:
        """Fit the vocabulary on `chunks` and return a normalized matrix."""
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Cannot fit embedder on an empty chunk list.")

        texts = [chunk.text for chunk in chunk_list]
        _LOGGER.info("Fitting TF-IDF embedder on %d chunks...", len(texts))
        sparse = self._vectorizer.fit_transform(texts)
        # L2-normalize so that dot product == cosine similarity.
        normalized = self._l2_normalize(sparse)
        return EmbeddingMatrix(
            matrix=normalized,
            chunk_ids=[chunk.chunk_id for chunk in chunk_list],
            vocabulary_path=self._config.vector_store_dir / "vocabulary.joblib",
        )

    def transform(self, chunks: Iterable[Chunk]) -> EmbeddingMatrix:
        """Project additional chunks through an already-fitted vocabulary."""
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Cannot embed an empty chunk list.")
        sparse = self._vectorizer.transform([chunk.text for chunk in chunk_list])
        normalized = self._l2_normalize(sparse)
        return EmbeddingMatrix(
            matrix=normalized,
            chunk_ids=[chunk.chunk_id for chunk in chunk_list],
            vocabulary_path=self._config.vector_store_dir / "vocabulary.joblib",
        )

    def encode_query(self, query: str) -> csr_matrix:
        """Encode a single free-text query against the fitted vocabulary."""
        if not query.strip():
            raise ValueError("Cannot embed an empty query string.")
        return self._l2_normalize(self._vectorizer.transform([query]))

    @staticmethod
    def _l2_normalize(matrix: csr_matrix) -> csr_matrix:
        """L2-normalize each row of a sparse matrix in-place-of-return."""
        if matrix.shape[0] == 0:
            return matrix
        dense = matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        # Guard against zero-norm rows (e.g. out-of-vocabulary queries).
        norms[norms == 0.0] = 1.0
        normalized = dense / norms
        return csr_matrix(normalized)
