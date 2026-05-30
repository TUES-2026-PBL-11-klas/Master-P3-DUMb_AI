"""
Unit tests for MongoUserStore

Uses a MagicMock Collection — no real MongoDB is required.

Covers:
  - find_by_username() returns None when no doc matches
  - find_by_username() maps a BSON doc back to UserAcc
  - find_by_username() returns None on empty username without hitting Mongo
  - find_by_username() wraps driver failures as StorageError
  - create() inserts the expected BSON shape and returns the UserAcc
  - create() rejects empty username / password_hash up-front
  - create() surfaces a DuplicateKeyError as a "username taken" StorageError
  - create() wraps other driver failures as StorageError with __cause__
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from services.db.mongo_user_store import MongoUserStore
from services.shared.domain import UserAcc
from services.shared.exceptions import StorageError


# A fake DuplicateKeyError that the store recognises by class name only —
# this avoids pulling pymongo into the test setup just to raise it.
class DuplicateKeyError(Exception):
    pass


def _user_doc(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid.uuid4()),
        "username": "alice",
        "password_hash": "scrypt$...$...",  # pragma: allowlist secret
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


# find_by_username


def test_find_by_username_returns_none_when_missing() -> None:
    col = MagicMock()
    col.find_one.return_value = None
    store = MongoUserStore(collection=col)

    assert store.find_by_username("ghost") is None
    col.find_one.assert_called_once_with({"username": "ghost"})


def test_find_by_username_maps_doc_to_useracc() -> None:
    col = MagicMock()
    user_id = uuid.uuid4()
    col.find_one.return_value = _user_doc(id=str(user_id), username="alice")
    store = MongoUserStore(collection=col)

    user = store.find_by_username("alice")

    assert isinstance(user, UserAcc)
    assert user.id == user_id
    assert user.username == "alice"
    assert user.password_hash == "scrypt$...$..."  # pragma: allowlist secret


def test_find_by_username_empty_skips_db() -> None:
    col = MagicMock()
    store = MongoUserStore(collection=col)

    assert store.find_by_username("") is None
    col.find_one.assert_not_called()


def test_find_by_username_wraps_driver_failure() -> None:
    col = MagicMock()
    col.find_one.side_effect = RuntimeError("connection lost")
    store = MongoUserStore(collection=col)

    with pytest.raises(StorageError) as excinfo:
        store.find_by_username("alice")
    assert "find_by_username" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# create


def test_create_inserts_expected_doc_and_returns_useracc() -> None:
    col = MagicMock()
    store = MongoUserStore(collection=col)

    user = store.create("alice", "scrypt$hash")

    assert isinstance(user, UserAcc)
    assert user.username == "alice"
    assert user.password_hash == "scrypt$hash"  # pragma: allowlist secret
    assert isinstance(user.id, uuid.UUID)
    assert user.created_at.tzinfo is not None  # tz-aware

    col.insert_one.assert_called_once()
    inserted = col.insert_one.call_args.args[0]
    assert inserted["username"] == "alice"
    assert inserted["password_hash"] == "scrypt$hash"  # pragma: allowlist secret
    assert inserted["id"] == str(user.id)
    assert inserted["created_at"] == user.created_at


def test_create_rejects_empty_username() -> None:
    col = MagicMock()
    store = MongoUserStore(collection=col)

    with pytest.raises(StorageError, match="username must be non-empty"):
        store.create("", "scrypt$hash")
    col.insert_one.assert_not_called()


def test_create_rejects_empty_password_hash() -> None:
    col = MagicMock()
    store = MongoUserStore(collection=col)

    with pytest.raises(StorageError, match="password_hash must be non-empty"):
        store.create("alice", "")
    col.insert_one.assert_not_called()


def test_create_duplicate_key_surfaces_as_taken() -> None:
    col = MagicMock()
    col.insert_one.side_effect = DuplicateKeyError("E11000 duplicate key")
    store = MongoUserStore(collection=col)

    with pytest.raises(StorageError, match="already taken"):
        store.create("alice", "scrypt$hash")


def test_create_other_failure_wraps_with_cause() -> None:
    col = MagicMock()
    col.insert_one.side_effect = RuntimeError("disk full")
    store = MongoUserStore(collection=col)

    with pytest.raises(StorageError) as excinfo:
        store.create("alice", "scrypt$hash")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
