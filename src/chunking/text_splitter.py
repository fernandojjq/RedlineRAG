"""Sentence-aware chunker for legal documents.

The old chunker (RecursiveChunker) collapsed all whitespace into single
spaces, which meant paragraph boundaries like "\\n\\n" disappeared, and
a 600-char chunk could mix clauses from two different paragraphs. That
made the auditor unreliable: pattern matching on the whole chunk would
fire on whichever clause appeared first, regardless of which clause the
user actually asked about.

The new chunker does the following:

  1. Split the document on "\\n\\n" first so paragraphs stay whole.
  2. Within each paragraph, split on sentence boundaries (". ", "? ", "! ").
  3. If a sentence is still too long, recursively split on commas/whitespace
     (still respecting the original paragraph boundary).
  4. Emit each sentence as a Chunk, but keep the parent paragraph attached
     via the `parent_text` field. The embedder indexes `text` (the sentence)
     so retrieval is fine-grained. The auditor evaluates `text` first
     (the "primary" match) and only looks at `parent_text` to find
     "co-located" matches in adjacent sentences.

This is a "late chunking" lite approach. We do not need a transformer
to get the benefits: sentence-level retrieval is plenty fine-grained for
clause-level risk detection, and keeping the parent paragraph attached
means the auditor has full context to decide whether a sibling clause
matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.ingestion.document_loader import LoadedDocument


# I split on these separators, in order. Paragraphs first, then sentences,
# then commas, then whitespace. The bottom of the hierarchy is a hard cut.
# Order matters: I want to keep clause text inside one paragraph together
# as long as possible.
_SENTENCE_SEPARATORS: tuple[str, ...] = (". ", "? ", "! ")
_FALLBACK_SEPARATORS: tuple[str, ...] = ("; ", ", ", " ")


@dataclass
class Chunk:
    """A retrieval-ready segment of text with provenance.

    `text` is what gets indexed and retrieved (fine-grained, usually one
    sentence). `parent_text` is the paragraph the sentence came from, used
    by the auditor to find co-located matches and by the CLI to give the
    user enough context to understand the finding.
    """

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    parent_text: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict)


# Pre-compiled patterns. I keep these simple - the chunker is small and I do
# not want a regex library explosion.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")  # blank line = new paragraph
_NON_BREAKING_SPACE = re.compile(r" ")
_WHITESPACE_RUN = re.compile(r"[ \t]+")  # only horizontal whitespace, NOT newlines


def _clean_paragraph(paragraph: str) -> str:
    """Tidy up a single paragraph: drop nbsp, collapse runs of spaces."""
    paragraph = _NON_BREAKING_SPACE.sub(" ", paragraph)
    paragraph = _WHITESPACE_RUN.sub(" ", paragraph)
    return paragraph.strip()


def _split_into_sentences(paragraph: str) -> list[str]:
    """Split a clean paragraph into sentences.

    This is a heuristic. Legal text is full of abbreviations ("U.S.A.",
    "e.g.", "Co.") that look like sentence boundaries. To keep things
    robust I just split on the common end-of-sentence punctuation followed
    by a space, and rely on the fact that over-splitting is safer than
    under-splitting: an auditor looking at a too-short "sentence" will
    just not find any pattern, which is the right answer for noise.
    """
    # First collapse ALL whitespace (including newlines) to single
    # spaces. The original file may have line breaks inside a sentence
    # (e.g. a 50-column PDF or a hand-formatted agreement), and the
    # auditor's regex patterns use literal whitespace. Without this
    # normalization, a chunk like "permanently\ndelete" would not
    # match the pattern "permanently delete".
    cleaned = re.sub(r"\s+", " ", paragraph).strip()
    if not cleaned:
        return []

    # I split greedily on sentence enders but I keep the punctuation
    # attached to the preceding sentence so the auditor can match phrases
    # like "binding arbitration." verbatim.
    pieces: list[str] = [cleaned]
    for separator in _SENTENCE_SEPARATORS:
        next_pieces: list[str] = []
        for piece in pieces:
            # I split on the separator and re-attach the punctuation to
            # the left side. So "A. B." becomes ["A.", "B."].
            chunks = piece.split(separator)
            for index, chunk in enumerate(chunks):
                if index < len(chunks) - 1:
                    next_pieces.append(chunk + separator.strip())
                else:
                    next_pieces.append(chunk)
        pieces = [p for p in next_pieces if p and p.strip()]

    # Last-resort: if any piece is still absurdly long, split on commas.
    refined: list[str] = []
    for piece in pieces:
        if len(piece) > 600:
            refined.extend(_split_long_piece(piece))
        else:
            refined.append(piece)
    return [p for p in refined if p and p.strip()]


def _split_long_piece(piece: str) -> list[str]:
    """Fallback splitter for a sentence that is still way too long."""
    out: list[str] = [piece]
    for separator in _FALLBACK_SEPARATORS:
        next_out: list[str] = []
        for p in out:
            if len(p) <= 500:
                next_out.append(p)
                continue
            parts = p.split(separator)
            for index, part in enumerate(parts):
                if index < len(parts) - 1:
                    next_out.append(part + separator.rstrip())
                else:
                    next_out.append(part)
        out = [p for p in next_out if p and p.strip()]
    return out


class SentenceAwareChunker:
    """Splits a document into sentence-level chunks with parent context."""

    def __init__(self) -> None:
        # No tunables right now. If we need them later (max sentence length
        # etc.) they can go here. Keeping the constructor parameterless for
        # now because the orchestrator instantiates this once.
        pass

    def chunk_documents(
        self, documents: Iterable[LoadedDocument]
    ) -> list[Chunk]:
        """Chunk every document into a flat list of sentence-level Chunks.

        Order is preserved: document order, then paragraph order within a
        document, then sentence order within a paragraph. The `position`
        field is a global index across all chunks, which makes the chunk_id
        easy to read in the CLI output.
        """
        out: list[Chunk] = []
        position = 0
        for document in documents:
            paragraphs = _PARAGRAPH_SPLIT.split(document.raw_text)
            for paragraph_index, raw_paragraph in enumerate(paragraphs):
                sentences = _split_into_sentences(raw_paragraph)
                if not sentences:
                    continue
                # The "parent text" is the joined paragraph, not the
                # original raw text - the cleaned version is what the user
                # sees in the report, so it has to be self-consistent.
                parent_text = " ".join(sentences)
                parent_id = f"{document.document_id}::para_{paragraph_index:04d}"
                for sentence_index, sentence in enumerate(sentences):
                    chunk_id = f"{parent_id}::sent_{sentence_index:04d}"
                    out.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document.document_id,
                            document_title=document.title,
                            text=sentence,
                            parent_text=parent_text,
                            position=position,
                            metadata={
                                **document.metadata,
                                "chunk_length": str(len(sentence)),
                                "parent_id": parent_id,
                                "sentence_index": str(sentence_index),
                            },
                        )
                    )
                    position += 1
        return out
