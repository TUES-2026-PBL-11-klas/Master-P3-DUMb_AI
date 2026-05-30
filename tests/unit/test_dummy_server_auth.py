"""
Unit tests for the dummy_server auth helpers

Covers:
  - _hash_password / _verify_password round-trip
  - _verify_password returns False on wrong password
  - _verify_password returns False on a malformed stored string
  - Two hashes of the same password use different salts (no determinism)
  - _signup creates a brand-new account
  - _signup rejects an existing username with AuthError("already taken")
  - _signup rejects empty username / password with AuthError
  - _login accepts a returning user with the right password
  - _login rejects a wrong password with AuthError("invalid credentials")
  - _login rejects an unknown user with the SAME message ("invalid
    credentials") — no enumeration leak
  - _login spends roughly the same time on unknown-user vs wrong-password
    (timing side-channel closed by always running scrypt)
  - _login rejects empty username / password with AuthError
  - The _MemoryUserStore fallback raises StorageError on duplicate insert
"""

from __future__ import annotations

import time

import pytest

from services.dummy_server import (
    _MemoryUserStore,
    _hash_password,
    _login,
    _signup,
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


# _signup


def test_signup_creates_new_account() -> None:
    user = _signup("alice", "hunter2")
    assert user.username == "alice"
    # Password is hashed in storage — never the raw value.
    assert user.password_hash != "hunter2"  # pragma: allowlist secret
    assert _verify_password("hunter2", user.password_hash)


def test_signup_rejects_existing_username() -> None:
    _signup("alice", "hunter2")
    with pytest.raises(AuthError, match="already taken"):
        _signup("alice", "different")


def test_signup_rejects_empty_inputs() -> None:
    with pytest.raises(AuthError, match="username"):
        _signup("", "hunter2")
    with pytest.raises(AuthError, match="password"):
        _signup("alice", "")


# _login


def test_login_accepts_returning_user() -> None:
    _signup("alice", "hunter2")
    user = _login("alice", "hunter2")
    assert user.username == "alice"


def test_login_rejects_wrong_password_with_generic_message() -> None:
    """Wrong-password and unknown-user must yield the SAME error message
    so the wire response cannot be used to enumerate accounts."""
    _signup("alice", "hunter2")
    with pytest.raises(AuthError, match="^invalid credentials$"):
        _login("alice", "wrong")


def test_login_rejects_unknown_user_with_generic_message() -> None:
    with pytest.raises(AuthError, match="^invalid credentials$"):
        _login("ghost", "hunter2")


def test_login_failure_messages_are_identical() -> None:
    """Belt-and-braces: the two failure paths emit byte-identical messages."""
    _signup("alice", "hunter2")
    with pytest.raises(AuthError) as wrong_pw:
        _login("alice", "wrong")
    with pytest.raises(AuthError) as unknown_user:
        _login("ghost", "hunter2")
    assert str(wrong_pw.value) == str(unknown_user.value)


def test_login_timing_is_comparable_for_unknown_user_and_wrong_password() -> None:
    """
    The unknown-user path must also pay the scrypt cost, otherwise an
    attacker can enumerate accounts via response time.

    We don't pin a tight bound (CI machines are noisy); we just assert
    the unknown-user path is within an order of magnitude of the
    wrong-password path. Without the dummy-hash fix the ratio is ~10000.
    """
    _signup("alice", "hunter2")

    # Warm up scrypt + lazy-init the dummy hash.
    try:
        _login("ghost", "x")
    except AuthError:
        pass

    def time_login(user: str, pw: str) -> float:
        start = time.perf_counter()
        try:
            _login(user, pw)
        except AuthError:
            pass
        return time.perf_counter() - start

    wrong_pw_time = min(time_login("alice", "wrong") for _ in range(3))
    unknown_time = min(time_login("ghost", "hunter2") for _ in range(3))

    # Generous bound. If the dummy-hash path is missing, unknown is microseconds
    # and wrong_pw is ~16 ms — ratio ~10000. Within 10x is plenty of headroom.
    ratio = max(wrong_pw_time, unknown_time) / max(
        min(wrong_pw_time, unknown_time), 1e-9
    )
    assert ratio < 10.0, (
        f"unknown-user path is {ratio:.1f}x off the wrong-password path — "
        f"the timing side-channel is open (unknown={unknown_time:.4f}s, "
        f"wrong_pw={wrong_pw_time:.4f}s)"
    )


def test_login_rejects_empty_inputs() -> None:
    with pytest.raises(AuthError, match="username"):
        _login("", "hunter2")
    with pytest.raises(AuthError, match="password"):
        _login("alice", "")


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
