"""
MongoUserStore — repository for UserAcc records backed by MongoDB.

Responsibilities:
  - Look up a user by username.
  - Insert a brand-new user with a hashed password.

Design pattern: Repository
    The class hides BSON ↔ dataclass translation from callers. Higher
    layers (the dummy server, future query/ingestion services) depend
    only on the UserAcc domain dataclass.

Construction:
    Mirrors MongoVectorStore — inject a pymongo Collection in tests,
    or call ``MongoUserStore.from_uri(...)`` in production.

Schema (matches infra/mongo/init_db.js):
    {
        "_id":           ObjectId,         # Mongo-managed surrogate
        "id":            "<uuid-str>",     # our domain UUID
        "username":      "<str>",          # unique index in init_db.js
        "password_hash": "<str>",
        "created_at":    ISODate,
    }

Password handling:
    This module never sees raw passwords. Hashing is the *caller's*
    responsibility — typically the auth handler in dummy_server. Passing
    a plaintext password into ``create()`` would silently store it as if
    it were a hash, which is exactly the kind of footgun we don't want
    here. See _hash_password() in dummy_server for the hashing scheme.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from services.shared.domain import UserAcc
from services.shared.exceptions import StorageError

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection

logger = logging.getLogger(__name__)

# Defaults — keep in lockstep with infra/mongo/init_db.js.
_DEFAULT_DB_NAME = "dumb_ai"
_DEFAULT_COLLECTION = "users"


class MongoUserStore:
    """
    MongoDB-backed repository for UserAcc.

    Attributes:
        _col:         pymongo Collection handle for the users collection.
        _client:      Optional MongoClient — set only when this store
                      opened its own connection via from_uri().
        _owns_client: True if this store is responsible for closing
                      ``_client``.
    """

    def __init__(
        self,
        collection: "Collection[dict[str, Any]]",
        *,
        _client: "MongoClient[dict[str, Any]] | None" = None,
    ) -> None:
        """
        Args:
            collection: pymongo Collection bound to the users collection.
                        Injecting the collection (rather than a URI)
                        keeps the class trivially testable with a
                        MagicMock or mongomock.
            _client:    Internal — set by from_uri() when the store opens
                        its own MongoClient. Not part of the public API.
        """
        self._col = collection
        self._client = _client
        self._owns_client = _client is not None

    # Construction helpers

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        db_name: str = _DEFAULT_DB_NAME,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> "MongoUserStore":
        """
        Open a MongoClient against *uri* and return a store bound to it.

        The returned store owns the client and will close it when
        ``close()`` is called.

        Raises:
            StorageError: if pymongo cannot be imported or the connection
                          parameters are malformed.
        """
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover — import-time only
            raise StorageError(
                "pymongo is not installed — `pip install pymongo` is required "
                "to use MongoUserStore.from_uri()"
            ) from exc

        try:
            client: MongoClient[dict[str, Any]] = MongoClient(uri)
            collection = client[db_name][collection_name]
        except Exception as exc:
            raise StorageError(
                f"Failed to construct MongoClient for {uri!r}: {exc}"
            ) from exc

        logger.info(
            "MongoUserStore: connected to %s.%s", db_name, collection_name,
        )
        return cls(collection=collection, _client=client)

    def close(self) -> None:
        """Close the underlying MongoClient, if this store owns one."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    # Repository interface

    def find_by_username(self, username: str) -> UserAcc | None:
        """
        Return the UserAcc with the given username, or None if not found.

        Raises:
            StorageError: if the underlying find_one call fails.
        """
        if not username:
            # Treat empty username as "no user" — never as a DB lookup,
            # since an empty {username: ""} query would still hit Mongo.
            return None
        try:
            doc = self._col.find_one({"username": username})
        except Exception as exc:
            raise StorageError(
                f"find_by_username({username!r}) failed: {exc}"
            ) from exc

        if doc is None:
            return None
        return self._doc_to_user(doc)

    def create(self, username: str, password_hash: str) -> UserAcc:
        """
        Insert a new user and return the resulting UserAcc.

        The caller is responsible for hashing the password — see the
        module docstring. This method never touches plaintext passwords.

        Raises:
            StorageError: if the username already exists (the unique
                          index in init_db.js fires) or the insert fails
                          for any other reason. The duplicate-key path is
                          flagged separately so the caller can decide
                          whether to surface it as "user already exists".
        """
        if not username:
            raise StorageError("username must be non-empty")
        if not password_hash:
            raise StorageError("password_hash must be non-empty")

        user = UserAcc(
            id=uuid4(),
            username=username,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc),
        )

        try:
            self._col.insert_one(self._user_to_doc(user))
        except Exception as exc:
            # We do not import DuplicateKeyError eagerly to keep tests
            # mock-friendly; check by class name instead so a bare
            # MagicMock-raised exception still surfaces as StorageError.
            if type(exc).__name__ == "DuplicateKeyError":
                raise StorageError(
                    f"username {username!r} is already taken"
                ) from exc
            raise StorageError(
                f"insert_one for user {username!r} failed: {exc}"
            ) from exc

        logger.info("MongoUserStore.create: inserted user %r", username)
        return user

    # Serialization helpers

    @staticmethod
    def _user_to_doc(user: UserAcc) -> dict[str, Any]:
        """Convert a UserAcc dataclass into a BSON-ready dict."""
        return {
            "id": str(user.id),
            "username": user.username,
            "password_hash": user.password_hash,
            "created_at": user.created_at,
        }

    @staticmethod
    def _doc_to_user(doc: dict[str, Any]) -> UserAcc:
        """Convert a BSON dict back into a UserAcc dataclass."""
        return UserAcc(
            id=UUID(doc["id"]),
            username=doc["username"],
            password_hash=doc["password_hash"],
            created_at=doc["created_at"],
        )

    def __repr__(self) -> str:
        return f"MongoUserStore(collection={self._col!r})"