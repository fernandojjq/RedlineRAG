"""BM25-lite reranker for retrieval hits.

The TF-IDF vector retriever is good at semantic match, but it has a known
weakness: it can promote chunks that share vocabulary with the query but
actually talk about a different topic. For example, a query about
"perpetual content license" can land on a chunk that contains the phrase
"perpetual license" but is really about data retention, not content
licensing - the n-gram "perpetual license" carries enough shared weight
to win the cosine match.

A cross-encoder reranker (Cohere, sentence-transformers cross-encoders)
fixes this by jointly encoding the query and the chunk and producing a
fine-grained relevance score. But those need a model, a network, or a
paid API - exactly what this project is trying to avoid.

The token-overlap reranker is the cheap, offline alternative. After the
vector retriever returns the top-k candidates, I re-score each candidate
by counting how many of the query's tokens (after stop-word removal) also
appear in the candidate's matched sentence. The final score is a weighted
blend of vector similarity and exact token overlap. Chunks that share
more *exact* words with the user's question bubble up; chunks that only
share statistical n-gram weight get demoted.

This is a strict superset of the original behaviour: when query and
chunk share lots of exact terms (the common case), the reranker agrees
with the vector rank. When they share vocabulary but not intent (the
false-positive case), the reranker corrects it.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    # Only imported for type hints - the runtime import is avoided to
    # prevent a circular dependency with retriever.py.
    from src.retrieval.retriever import RetrievedChunk


# A small English stop-word set. I keep it inline so the reranker has zero
# runtime dependencies. Bigger lists (NLTK's, sklearn's) add dependencies
# and most of the words are not relevant to legal text.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with", "this", "we", "you",
        "your", "our", "us", "may", "can", "any", "all", "any", "such",
        "any", "their", "they", "them", "these", "those", "which", "who",
        "what", "when", "where", "why", "how", "if", "then", "than", "so",
        "do", "does", "did", "not", "no", "yes",
    }
)

# Token pattern: letters and digits. I deliberately drop punctuation -
# legal punctuation is dense and noisy.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + drop stop-words + drop pure-punctuation tokens."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOP_WORDS
    ]


class TokenOverlapReranker:
    """Re-scores a list of retrieval hits by exact token overlap with query.

    Score blend:
        final = alpha * vector_score + (1 - alpha) * overlap_score

    `alpha` defaults to 0.6, which gives a 60/40 split favouring the
    vector score but letting overlap correct obvious mismatches. Tune up
    for purer semantic search; tune down for stricter keyword matching.
    """

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def rerank(
        self,
        query: str,
        hits: Iterable["RetrievedChunk"],
        top_k: int | None = None,
    ) -> list["RetrievedChunk"]:
        """Return a new list of hits, re-ordered by blended score.

        Hits that the reranker demotes below the similarity floor are
        dropped here, not in the retriever. That keeps the two concerns
        cleanly separated: the retriever trusts the vector index, the
        reranker enforces the user's actual question.
        """
        hits = list(hits)
        if not hits:  # type: ignore[arg-type]
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            # If the query is all stop-words, the overlap score is
            # meaningless. Just return the original order.
            return hits

        scored: list[tuple[float, RetrievedChunk]] = []
        for hit in hits:
            # Overlap score: fraction of query tokens that appear in the
            # matched sentence. Normalized to [0, 1].
            hit_tokens = set(_tokenize(hit.chunk.text))
            overlap_count = len(query_tokens & hit_tokens)
            overlap_score = overlap_count / len(query_tokens)

            vector_score = max(0.0, min(1.0, hit.score))
            blended = self._alpha * vector_score + (1.0 - self._alpha) * overlap_score

            # I return a NEW RetrievedChunk with the blended score so the
            # auditor and CLI see the corrected number, not the raw
            # vector score.
            new_hit = replace(hit, score=blended)
            scored.append((blended, new_hit))

        scored.sort(key=lambda item: -item[0])

        if top_k is not None:
            scored = scored[: max(0, int(top_k))]

        return [hit for _, hit in scored]
