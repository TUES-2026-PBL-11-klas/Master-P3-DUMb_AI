"""
Unit tests for the dummy_server auth helpers

Covers:
  - _hash_password / _verify_password round-trip
  - _verify_password returns False on wrong password
  - _verify_password returns False on a malformed stored string
  - Two hashes of the same password use different salts (no determinism)
  - _authenticate creates a brand-new account on first sight
  - _authenticate accepts a returning user with the right password
  - _authenticate rejects a wrong password with AuthError
  - _authenticate rejects empty username / password with AuthError
  - The _MemoryUserStore fallback raises StorageError on duplicate insert
"""

from __future__ import annotations

import pytest

from services.dummy_server import (
    _MemoryUserStore,
    _authenticate,
    _hash_password,
    _verify_password,
    set_user_store,
)
from services.shared.exceptions import AuthError, StorageError


@pytest.fixture(autouse=True)
def fresh_store() -> None:
    """Each test gets its own empty in-memory user store."""
    set_user_store(_MemoryUserStore())


# Password hashing


def test_hash_and_verify_round_trip() -> None:
    h = _hash_password("hunter2")
    assert _verify_password("hunter2", h) is True


def test_verify_rejects_wrong_password() -> None:
    h = _hash_password("hunter2")
    assert _verify_password("hunter3", h) is False


def test_verify_rejects_malformed_stored_string() -> None:
    assert _verify_password("hunter2", "not-a-real-hash") is False
    assert _verify_password("hunter2", "bcrypt$..$..$..$..") is False  # wrong scheme
    assert _verify_password("hunter2", "scrypt$a$b$c$xx$yy") is False  # bad ints


def test_two_hashes_of_same_password_differ() -> None:
    a = _hash_password("hunter2")
    b = _hash_password("hunter2")
    assert a != b  # salts are random
    assert _verify_password("hunter2", a)
    assert _verify_password("hunter2", b)


# _authenticate find-or-create flow


def test_authenticate_creates_new_account_on_first_sight() -> None:
    user, created = _authenticate("alice", "hunter2")
    assert created is True
    assert user.username == "alice"
    # Password is hashed in storage — never the raw value.
    assert user.password_hash != "hunter2"
    assert _verify_password("hunter2", user.password_hash)


def test_authenticate_accepts_returning_user() -> None:
    _authenticate("alice", "hunter2")  # create
    user, created = _authenticate("alice", "hunter2")  # sign-in
    assert created is False
    assert user.username == "alice"


def test_authenticate_rejects_wrong_password() -> None:
    _authenticate("alice", "hunter2")
    with pytest.raises(AuthError, match="incorrect password"):
        _authenticate("alice", "wrong")


def test_authenticate_rejects_empty_inputs() -> None:
    with pytest.raises(AuthError, match="username"):
        _authenticate("", "hunter2")
    with pytest.raises(AuthError, match="password"):
        _authenticate("alice", "")


# In-memory store fallback


def test_memory_store_duplicate_create_raises_storage_error() -> None:
    store = _MemoryUserStore()
    store.create("alice", "scrypt$x")
    with pytest.raises(StorageError, match="already taken"):
        store.create("alice", "scrypt$y")


def test_memory_store_find_by_username_returns_inserted_user() -> None:
    store = _MemoryUserStore()
    created = store.create("alice", "scrypt$x")
    found = store.find_by_username("alice")
    assert found is not None
    assert found.id == created.id
    assert found.username == "alice"