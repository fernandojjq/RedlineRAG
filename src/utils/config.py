"""Configuration models for the RedlineRAG pipeline.

Everything that the pipeline needs to know about paths, parsing, chunking,
and analysis lives here. Values are loaded from environment variables when
present, with sensible production defaults otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field, field_validator


# Project root is the parent of the `src/` directory. We resolve it once at
# import time so the rest of the pipeline never has to guess where it lives.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class PipelineConfig(BaseModel):
    """Tunable parameters for the RAG pipeline.

    Defaults are chosen to give good results on typical ToS / privacy policy
    documents (which tend to be dense legal text, 5k-30k words long).
    """

    # --- Storage paths ---------------------------------------------------------
    raw_documents_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "raw",
        description="Directory the user can drop real ToS files into.",
    )
    mock_documents_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "mock",
        description="Where auto-generated mock agreements are stored.",
    )
    vector_store_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "vector_store",
        description="Where the persistent index lives between runs.",
    )

    # --- Ingestion behaviour ---------------------------------------------------
    auto_generate_mocks: bool = Field(
        default=True,
        description=(
            "If the raw documents dir is empty, synthesize realistic mock ToS "
            "agreements with planted legal traps so the pipeline never fails "
            "on first run."
        ),
    )
    supported_extensions: tuple[str, ...] = Field(
        default=(".txt", ".md", ".pdf", ".docx"),
        description="File types the loader will try to ingest.",
    )

    # --- Chunking strategy -----------------------------------------------------
    # Why 600 / 80? ToS clauses often span a paragraph or two. A chunk of
    # ~600 characters gives the retriever enough context to understand a
    # clause (e.g. "binding arbitration") without dragging in unrelated
    # surrounding boilerplate. An 80-character overlap is enough to keep
    # a clause that straddles a chunk boundary semantically whole.
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)

    # --- Retrieval -------------------------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many candidate chunks to feed into the auditor per query.",
    )
    similarity_floor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Chunks with cosine similarity below this threshold are discarded "
            "to prevent the auditor from hallucinating on unrelated text."
        ),
    )

    # --- Embeddings ------------------------------------------------------------
    # We use a TF-IDF word-level n-gram embedding rather than a transformer
    # model. Two reasons:
    #   1) Zero model download -> the pipeline runs on a fresh machine in
    #      seconds, with no GPU, no internet, and no PyTorch.
    #   2) Legal text has very specific terminology ("binding arbitration",
    #      "class-action waiver", "irrevocable license"). Word-level n-grams
    #      capture those phrases verbatim, which is exactly what we need
    #      for clause-level risk detection. We keep stop-word filtering on
    #      so the embedder doesn't get distracted by common English.
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.9
    tfidf_max_features: int = 50_000
    tfidf_stop_words: str | tuple[str, ...] = "english"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if value >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({value}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )
        return value

    def ensure_directories(self) -> None:
        """Create the on-disk layout if it doesn't already exist."""
        for path in (self.raw_documents_dir, self.mock_documents_dir, self.vector_store_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> PipelineConfig:
    """Build a PipelineConfig and make sure its directories exist."""
    config = PipelineConfig()
    config.ensure_directories()
    return config
