"""End-to-end RAG pipeline.

This module owns the lifecycle:

    1. Discover (or auto-generate) input documents.
    2. Chunk them with the sentence-aware chunker.
    3. Build (or load) the local vector store.
    4. Run a query through the retriever (with reranker).
    5. Feed hits to the risk auditor.
    6. Return a structured PipelineResult.

The pipeline is intentionally a single class with a small, linear API.
Keeping all the lifecycle in one place means the CLI is a thin wrapper
and tests can drive the pipeline without monkey-patching module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.chunking.text_splitter import Chunk, SentenceAwareChunker
from src.embeddings.embedder import EmbeddingMatrix, TfidfEmbedder
from src.generation.auditor import RiskAssessment, RiskAuditor
from src.ingestion.document_loader import LoadedDocument, discover_documents
from src.ingestion.mock_generator import MockTosGenerator
from src.retrieval.retriever import RetrievedChunk, Retriever
from src.retrieval.reranker import TokenOverlapReranker
from src.utils.config import PipelineConfig, load_config
from src.utils.logging_setup import get_logger
from src.vector_store.store_manager import LocalVectorStore

_LOGGER = get_logger(__name__)


@dataclass
class PipelineResult:
    """The output of one user query: hits + assessment + provenance."""

    question: str
    hits: list[RetrievedChunk]
    assessment: RiskAssessment
    documents_ingested: list[str] = field(default_factory=list)
    chunks_indexed: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "documents_ingested": self.documents_ingested,
            "chunks_indexed": self.chunks_indexed,
            "retrieval": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "document": hit.chunk.document_title,
                    "score": round(hit.score, 4),
                    "matched_text": " ".join(hit.chunk.text.split()),
                    "parent_text": " ".join(hit.chunk.parent_text.split())[:320],
                }
                for hit in self.hits
            ],
            "audit": self.assessment.to_dict(),
        }


class RagPipeline:
    """The full RAG pipeline, parameterized by a single PipelineConfig."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or load_config()
        # Sentence-aware chunker: one sentence = one indexed unit, with the
        # parent paragraph attached for co-located-pattern detection.
        self._chunker = SentenceAwareChunker()
        self._auditor = RiskAuditor()
        self._store = LocalVectorStore(self._config)
        # Reranker: token-overlap with the query, blended with vector score.
        # Lives at the orchestrator level so tests can swap it out.
        self._reranker = TokenOverlapReranker()
        self._documents: list[LoadedDocument] = []
        self._chunks: list[Chunk] = []
        self._matrix: EmbeddingMatrix | None = None
        self._embedder: TfidfEmbedder | None = None
        self._retriever: Retriever | None = None

    # -- Ingestion & indexing ------------------------------------------------

    def ingest(self) -> list[LoadedDocument]:
        """Discover documents; auto-generate mocks if the raw dir is empty."""
        self._config.ensure_directories()

        raw_documents = discover_documents(
            self._config.raw_documents_dir,
            self._config.supported_extensions,
        )
        documents: list[LoadedDocument] = list(raw_documents)

        if not documents and self._config.auto_generate_mocks:
            _LOGGER.info(
                "No input documents found in %s - generating mock ToS corpus.",
                self._config.raw_documents_dir,
            )
            generator = MockTosGenerator(self._config.raw_documents_dir)
            generated_paths = generator.generate()
            documents = discover_documents(
                self._config.raw_documents_dir,
                self._config.supported_extensions,
            )
            _LOGGER.info("Mock generator wrote %d file(s).", len(generated_paths))

        if not documents:
            raise RuntimeError(
                "No documents to ingest. Drop ToS files into "
                f"{self._config.raw_documents_dir} or set "
                "auto_generate_mocks=True in the config."
            )

        self._documents = documents
        return documents

    def index(self, force_rebuild: bool = False) -> int:
        """Build (or rebuild) the local vector store from the ingested docs."""
        if not self._documents:
            self.ingest()

        if self._store.exists() and not force_rebuild:
            _LOGGER.info("Loading existing vector store from disk.")
            matrix, chunks, embedder = self._store.load()
            self._matrix = matrix
            self._chunks = chunks
            self._embedder = embedder
        else:
            _LOGGER.info("Building new vector store (rebuild=%s).", force_rebuild)
            chunks = self._chunker.chunk_documents(self._documents)
            if not chunks:
                raise RuntimeError("Chunker produced zero chunks - input may be empty.")
            embedder = TfidfEmbedder(self._config)
            matrix = self._store.build(chunks, embedder)
            self._chunks = chunks
            self._embedder = embedder
            self._matrix = matrix

        self._retriever = Retriever(
            config=self._config,
            chunks=self._chunks,
            matrix=self._matrix,
            embedder=self._embedder,
            reranker=self._reranker,
        )
        return len(self._chunks)

    # -- Query ---------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> PipelineResult:
        """Run a single user query through the full pipeline."""
        if self._retriever is None or self._matrix is None:
            raise RuntimeError(
                "Pipeline is not indexed. Call RagPipeline.index() first."
            )
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        hits = self._retriever.query(question, top_k=top_k)
        assessment = self._auditor.audit(question, hits)
        return PipelineResult(
            question=question,
            hits=hits,
            assessment=assessment,
            documents_ingested=[doc.title for doc in self._documents],
            chunks_indexed=len(self._chunks),
        )

    # -- Introspection helpers ----------------------------------------------

    @property
    def indexed_chunks(self) -> int:
        return len(self._chunks)

    @property
    def ingested_documents(self) -> list[str]:
        return [doc.title for doc in self._documents]
