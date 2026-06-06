"""Ingestion layer: load documents and synthesize mocks when needed."""
from __future__ import annotations

from src.ingestion.document_loader import LoadedDocument, discover_documents
from src.ingestion.mock_generator import MockTosGenerator

__all__ = ["LoadedDocument", "discover_documents", "MockTosGenerator"]
