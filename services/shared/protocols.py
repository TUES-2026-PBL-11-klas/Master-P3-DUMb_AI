from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from services.shared.domain import IngestionEvent

T_co = TypeVar("T_co", covariant=True)
T_inv = TypeVar("T_inv")


@runtime_checkable
class AIInterface(Protocol):
    """Structural interface for any embedding backend."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class DocumentParser(Protocol[T_co]):
    extensions: ClassVar[tuple[str, ...]]

    def parse(self, raw: bytes | bytearray, filename: str) -> T_co: ...


@runtime_checkable
class IngestionObserver(Protocol):
    def on_ingest(self, event: "IngestionEvent") -> None: ...


@runtime_checkable
class VectorStore(Protocol[T_inv]):
    def store(self, chunks: list[T_inv]) -> None: ...
    def search(self, vec: list[float], k: int) -> list[T_inv]: ...
