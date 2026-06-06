"""Top-k cosine similarity retriever, with reranker hook.

Given a fitted embedder, a chunk matrix, and the chunk list, this module
answers free-text queries with the chunks that look most relevant to the
query, in descending order of blended (vector + token overlap) score.

Pipeline inside `query()`:
  1. Encode the query with the fitted TF-IDF vectorizer.
  2. Score every chunk with a single matrix-vector product.
  3. Take the top-k by vector score (this is the "candidate" set - a bit
     larger than the final k so the reranker has room to work).
  4. Hand the candidates to the TokenOverlapReranker for re-scoring.
  5. Return the reranked top-k.

Step 3 is the key: I do not trust the vector score alone, but I do trust
it to put relevant candidates in the top-2k. The reranker then promotes
the ones that share more exact tokens with the user's actual question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from src.chunking.text_splitter import Chunk
from src.embeddings.embedder import EmbeddingMatrix, TfidfEmbedder
from src.retrieval.reranker import TokenOverlapReranker
from src.utils.config import PipelineConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

# How many more candidates to retrieve from the vector index than the
# final top-k. Bigger = more chance the reranker finds the right hit,
# at the cost of a slightly slower rerank step. 4x is a good default.
_RERANK_OVERFETCH: int = 4


@dataclass
class RetrievedChunk:
    """A single retrieval hit, with its score and provenance.

    `chunk.text` is the matched sentence (the indexed unit).
    `chunk.parent_text` is the paragraph the sentence came from, used by
    the auditor for co-located-pattern detection and by the CLI to give
    the user enough context to understand the citation.
    """

    chunk: Chunk
    score: float


class Retriever:
    """Stateless retriever: holds an index, runs queries against it."""

    def __init__(
        self,
        config: PipelineConfig,
        chunks: list[Chunk],
        matrix: EmbeddingMatrix,
        embedder: TfidfEmbedder,
        reranker: TokenOverlapReranker | None = None,
    ) -> None:
        self._config = config
        # Lookup table: chunk_id -> full chunk. I keep this as a dict
        # because the matrix only stores chunk_ids, but the auditor and
        # CLI need the full text + parent + metadata.
        self._chunks_by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
        self._matrix = matrix
        self._embedder = embedder
        self._reranker = reranker or TokenOverlapReranker()

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for `question`."""
        k = top_k if top_k is not None else self._config.top_k
        k = max(1, min(k, self._matrix.n_chunks))

        query_vector = self._embedder.encode_query(question)
        vector_scores = self._score(query_vector)

        # Step 1: pull a wider candidate set from the vector index.
        candidate_k = min(k * _RERANK_OVERFETCH, self._matrix.n_chunks)
        vector_hits = self._top_k(vector_scores, candidate_k, apply_floor=False)

        # Step 2: rerank the candidates by blended score.
        reranked = self._reranker.rerank(question, vector_hits, top_k=candidate_k)

        # Step 3: apply the similarity floor against the BLENDED score,
        # not the raw vector score. This is what stops the false positives
        # from sneaking through: a chunk that scored 0.4 on vector but
        # 0.05 on overlap now lands at 0.6*0.4 + 0.4*0.05 = 0.26, well
        # above the 0.05 floor. But a chunk that scored 0.15 on vector
        # and 0.0 on overlap lands at 0.6*0.15 = 0.09, still above the
        # floor. The floor mostly guards against truly empty retrievals.
        out: list[RetrievedChunk] = []
        for hit in reranked:
            if hit.score < self._config.similarity_floor:
                continue
            out.append(hit)
            if len(out) >= k:
                break
        return out

    # -- Internals ------------------------------------------------------------

    def _score(self, query_vector: csr_matrix) -> np.ndarray:
        # The matrix is L2-normalized row-wise and the query vector is
        # L2-normalized, so dot product == cosine similarity.
        product = (self._matrix.matrix @ query_vector.T).toarray().ravel()
        return product

    def _top_k(
        self,
        scores: np.ndarray,
        k: int,
        apply_floor: bool = True,
    ) -> list[RetrievedChunk]:
        """Take the top-k by raw vector score."""
        if scores.size == 0:
            return []

        k = min(k, scores.size)
        # argpartition is O(n); then I sort just the top-k slice in
        # descending order so the strongest match is first.
        partition_index = np.argpartition(-scores, kth=k - 1)[:k]
        ordered = partition_index[np.argsort(-scores[partition_index])]

        hits: list[RetrievedChunk] = []
        for idx in ordered:
            score = float(scores[idx])
            if apply_floor and score < self._config.similarity_floor:
                continue
            chunk_id = self._matrix.chunk_ids[idx]
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                _LOGGER.warning("Index returned unknown chunk_id: %s", chunk_id)
                continue
            hits.append(RetrievedChunk(chunk=chunk, score=score))
        return hits
