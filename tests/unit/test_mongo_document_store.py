from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from services.db.mongo_document_store import MongoDocumentStore
from services.shared.domain import Document, DocumentStatus


def test_upsert_stores_metadata_without_content() -> None:
    collection = MagicMock()
    store = MongoDocumentStore(collection)
    document = Document(
        id=uuid4(),
        user_id=uuid4(),
        content="full parsed text should not be persisted",
        filename="notes.txt",
        uploaded_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        status=DocumentStatus.READY,
    )

    store.upsert(document)

    collection.replace_one.assert_called_once()
    filter_doc, replacement = collection.replace_one.call_args.args
    assert filter_doc == {"id": str(document.id)}
    assert replacement["id"] == str(document.id)
    assert replacement["user_id"] == str(document.user_id)
    assert replacement["filename"] == "notes.txt"
    assert replacement["status"] == "ready"
    assert "content" not in replacement


def test_list_by_user_returns_document_summaries() -> None:
    collection = MagicMock()
    cursor = MagicMock()
    uploaded_at = datetime(2026, 6, 5, 10, 15, tzinfo=timezone.utc)
    cursor.sort.return_value = [
        {
            "id": "doc-1",
            "filename": "networking.md",
            "uploaded_at": uploaded_at,
            "status": "ready",
            "error_message": None,
        }
    ]
    collection.find.return_value = cursor
    store = MongoDocumentStore(collection)

    result = store.list_by_user("user-1")

    collection.find.assert_called_once_with({"user_id": "user-1"})
    cursor.sort.assert_called_once_with("uploaded_at", -1)
    assert result == [
        {
            "document_id": "doc-1",
            "filename": "networking.md",
            "uploaded_at": uploaded_at.isoformat(),
            "status": "ready",
            "error_message": None,
        }
    ]
