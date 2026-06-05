"""
MongoDocumentStore - repository for uploaded document metadata.

The full parsed text stays in backend memory during ingestion. MongoDB stores
only durable metadata here; searchable text lives per chunk in document_chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from services.shared.domain import Document, DocumentStatus
from services.shared.exceptions import StorageError

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection

logger = logging.getLogger(__name__)

_DEFAULT_DB_NAME = "dumb_ai"
_DEFAULT_COLLECTION = "documents"


class MongoDocumentStore:
    """MongoDB-backed repository for Document metadata."""

    def __init__(
        self,
        collection: "Collection[dict[str, Any]]",
        *,
        _client: "MongoClient[dict[str, Any]] | None" = None,
    ) -> None:
        self._col = collection
        self._client = _client
        self._owns_client = _client is not None

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        db_name: str = _DEFAULT_DB_NAME,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> "MongoDocumentStore":
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - import-time only
            raise StorageError(
                "pymongo is not installed - `pip install pymongo` is required "
                "to use MongoDocumentStore.from_uri()"
            ) from exc

        try:
            client: MongoClient[dict[str, Any]] = MongoClient(uri)
            collection = client[db_name][collection_name]
        except Exception as exc:
            raise StorageError(
                f"Failed to construct MongoClient for {uri!r}: {exc}"
            ) from exc

        logger.info("MongoDocumentStore: connected to %s.%s", db_name, collection_name)
        return cls(collection=collection, _client=client)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    def upsert(self, document: Document) -> None:
        """Persist metadata for *document* without storing Document.content."""
        try:
            self._col.replace_one(
                {"id": str(document.id)},
                self._document_to_doc(document),
                upsert=True,
            )
        except Exception as exc:
            raise StorageError(
                f"upsert for document {document.id} failed: {exc}"
            ) from exc

        logger.info(
            "MongoDocumentStore.upsert: stored document %s (%s)",
            document.filename,
            document.id,
        )

    def list_by_user(self, user_id: UUID | str) -> list[dict[str, Any]]:
        """
        Return document metadata for a user, newest first.

        The TUI only needs display fields, so this returns plain dicts instead
        of reconstructing Document objects with empty content.
        """
        try:
            cursor = self._col.find({"user_id": str(user_id)}).sort("uploaded_at", -1)
            docs = list(cursor)
        except Exception as exc:
            raise StorageError(f"list_by_user({user_id}) failed: {exc}") from exc

        return [self._doc_to_summary(doc) for doc in docs]

    @staticmethod
    def _document_to_doc(document: Document) -> dict[str, Any]:
        return {
            "id": str(document.id),
            "user_id": str(document.user_id),
            "filename": document.filename,
            "uploaded_at": document.uploaded_at,
            "status": document.status.value,
            "error_message": document.error_message,
            "schema_version": document.schema_version,
        }

    @staticmethod
    def _doc_to_summary(doc: dict[str, Any]) -> dict[str, Any]:
        uploaded_at = doc.get("uploaded_at")
        status = doc.get("status", DocumentStatus.UPLOADED.value)
        return {
            "document_id": str(doc.get("id", doc.get("_id", ""))),
            "filename": str(doc.get("filename", "")),
            "uploaded_at": uploaded_at.isoformat()
            if hasattr(uploaded_at, "isoformat")
            else str(uploaded_at or ""),
            "status": str(status),
            "error_message": doc.get("error_message"),
        }

    def __repr__(self) -> str:
        return f"MongoDocumentStore(collection={self._col!r})"
