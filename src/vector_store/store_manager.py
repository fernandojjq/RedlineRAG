"""Local, file-backed vector store.

No external database. No server. Everything lives in a single directory
on disk:

    data/vector_store/
        vocabulary.joblib    # the fitted TF-IDF vectorizer
        vectors.npz          # sparse chunk vectors
        chunks.jsonl         # chunk text + metadata, one per line
        manifest.json        # build metadata: when, how many chunks, etc.

The store is rebuilt on every full pipeline run. We deliberately do *not*
implement incremental update - the corpus is small, the rebuild is fast,
and a full rebuild gives us a clean, predictable state.

Schema note: the JSONL file holds a `parent_text` field for each chunk.
Older builds (before sentence-aware chunking) did not have it. The loader
is backwards-compatible: if `parent_text` is missing it falls back to the
sentence text, so the auditor still works on old stores. The CLI
displays a one-time warning so the user knows to rebuild for the full
parent-context experience.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix, save_npz, load_npz

from src.chunking.text_splitter import Chunk
from src.embeddings.embedder import EmbeddingMatrix, TfidfEmbedder
from src.utils.config import PipelineConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class VectorStoreManifest:
    """Metadata about a built vector store. Lives in manifest.json."""

    built_at: str
    chunk_count: int
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    # Bumped to 2 when we added parent_text. Bumped to 3 when we added
    # parent_id metadata. The loader uses this to detect old stores.
    schema_version: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class LocalVectorStore:
    """A tiny but real vector store that fits in a few hundred KB."""

    CURRENT_SCHEMA_VERSION: int = 3

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._store_dir = Path(config.vector_store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._vectors_path = self._store_dir / "vectors.npz"
        self._vocab_path = self._store_dir / "vocabulary.joblib"
        self._chunks_path = self._store_dir / "chunks.jsonl"
        self._manifest_path = self._store_dir / "manifest.json"

    # -- Construction ---------------------------------------------------------

    def build(
        self,
        chunks: list[Chunk],
        embedder: TfidfEmbedder,
    ) -> EmbeddingMatrix:
        """Fit the embedder on `chunks`, persist everything, return the matrix."""
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")

        matrix = embedder.fit_transform(chunks)

        # Persist the vectorizer (it owns the fitted vocabulary).
        joblib.dump(embedder._vectorizer, self._vocab_path)  # noqa: SLF001
        # Persist the normalized sparse matrix.
        save_npz(self._vectors_path, matrix.matrix)
        # Persist chunk text + metadata as JSONL so we can re-hydrate fast.
        # I include parent_text in every record so the auditor can do
        # co-located-pattern detection without re-chunking.
        with self._chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "text": chunk.text,
                    "parent_text": chunk.parent_text,
                    "position": chunk.position,
                    "metadata": chunk.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")

        manifest = VectorStoreManifest(
            built_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            chunk_count=len(chunks),
            embedding_model=f"tfidf-word{self._config.tfidf_ngram_range}",
            embedding_dim=int(matrix.matrix.shape[1]),
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            schema_version=self.CURRENT_SCHEMA_VERSION,
        )
        self._manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        _LOGGER.info(
            "Vector store built: %d chunks, %d-dim vocabulary (schema v%d) -> %s",
            manifest.chunk_count,
            manifest.embedding_dim,
            manifest.schema_version,
            self._store_dir,
        )
        return matrix

    # -- Inspection -----------------------------------------------------------

    def exists(self) -> bool:
        return (
            self._vectors_path.exists()
            and self._vocab_path.exists()
            and self._chunks_path.exists()
            and self._manifest_path.exists()
        )

    def load(self) -> tuple[EmbeddingMatrix, list[Chunk], TfidfEmbedder]:
        """Load a previously built store, plus the fitted embedder."""
        if not self.exists():
            raise FileNotFoundError(
                f"No built vector store at {self._store_dir}. "
                "Run the build step first."
            )

        embedder = TfidfEmbedder(self._config)
        embedder._vectorizer = joblib.load(self._vocab_path)  # noqa: SLF001

        matrix = load_npz(self._vectors_path)
        chunks, schema_version = self._load_chunks()

        # Re-validate: the chunk_ids in the matrix and the chunk_ids in the
        # JSONL must match exactly, otherwise the index is corrupted.
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Vector store is inconsistent: matrix has {matrix.shape[0]} "
                f"rows but chunks.jsonl has {len(chunks)} entries."
            )

        if schema_version < self.CURRENT_SCHEMA_VERSION:
            _LOGGER.warning(
                "Loaded store is schema v%d; current is v%d. Rebuild for the "
                "full parent-context experience: pass --rebuild on the next "
                "scan/ask call.",
                schema_version,
                self.CURRENT_SCHEMA_VERSION,
            )

        embedding_matrix = EmbeddingMatrix(
            matrix=matrix,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            vocabulary_path=self._vocab_path,
        )
        _LOGGER.info(
            "Loaded vector store: %d chunks, %d-dim vocabulary (schema v%d).",
            len(chunks),
            matrix.shape[1],
            schema_version,
        )
        return embedding_matrix, chunks, embedder

    def _load_chunks(self) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        schema_version = 1  # default for pre-versioning stores
        with self._chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                # Backwards-compat: old stores may not have parent_text.
                # Fall back to the indexed text so the auditor still works.
                text = payload["text"]
                parent_text = payload.get("parent_text") or text
                chunks.append(
                    Chunk(
                        chunk_id=payload["chunk_id"],
                        document_id=payload["document_id"],
                        document_title=payload["document_title"],
                        text=text,
                        parent_text=parent_text,
                        position=payload["position"],
                        metadata=payload.get("metadata", {}),
                    )
                )
                # Pick up the manifest's schema version if we encounter one
                # (the manifest is a separate file, but for paranoia we
                # also accept it inline in the JSONL).
                if "schema_version" in payload:
                    schema_version = max(schema_version, int(payload["schema_version"]))

        # Also read the manifest file for the canonical schema version.
        if self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if "schema_version" in manifest:
                    schema_version = max(schema_version, int(manifest["schema_version"]))
            except (json.JSONDecodeError, ValueError):
                pass
        return chunks, schema_version
