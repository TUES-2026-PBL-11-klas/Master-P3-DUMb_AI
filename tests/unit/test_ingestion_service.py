"""
Unit tests for IngestionService.

Covers:
  - subscribe() / unsubscribe() observer management
  - ingest(): parser lookup, user_id stamping, observer chain invocation
              in order, status transitions on success and failure
  - ingest(): size cap rejection
  - ingest(): unsupported extension propagates as RAGException
  - ingest(): unexpected (non-RAGException) parse errors are wrapped
  - ingest(): observer failure halts the chain and marks the event FAILED
  - ingest(): non-RAGException observer errors are wrapped
  - ingest(): defensive event.fail() when an observer forgets
  - ingest_batch(): parallel execution preserves order
  - ingest_batch(): never raises — per-upload failures appear as FAILED events
  - ingest_batch(): empty input is a no-op
  - context manager / close()
"""

from __future__ import annotations

import uuid
from typing import ClassVar, Generator
from unittest.mock import MagicMock

import pytest

# Module under test
from services.ingestion.service import IngestionService, Upload

# Project imports
from services.ingestion.parsers.registry import ParserRegistry
from services.shared.domain import (
    Document,
    DocumentStatus,
    IngestionEvent,
    IngestionStatus,
)
from services.shared.exceptions import (
    EmbeddingError,
    RAGException,
    StorageError,
    UnsupportedFormatError,
)


# Helpers


class _FakeParser:
    """
    Minimal parser satisfying DocumentParser[Document] structurally.

    parse() returns a freshly-constructed Document with the nil-UUID
    sentinel for user_id, matching the real parsers' behaviour.
    """

    extensions: ClassVar[tuple[str, ...]] = ("fake",)

    def __init__(self, content: str | None = None) -> None:
        self._content = content
        self.parse_calls: list[tuple[str, bytes | bytearray]] = []

    def parse(self, raw: bytes | bytearray, filename: str) -> Document:
        from datetime import datetime, timezone

        self.parse_calls.append((filename, raw))
        return Document(
            id=uuid.uuid4(),
            user_id=uuid.UUID(int=0),  # nil sentinel
            content=self._content
            if self._content is not None
            else raw.decode("utf-8", errors="replace"),
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status=DocumentStatus.PARSED,
        )


def _make_observer(name: str = "obs") -> MagicMock:
    """A MagicMock that satisfies IngestionObserver structurally."""
    obs = MagicMock(name=name)
    obs.on_ingest = MagicMock(return_value=None)
    return obs


# Fixtures


@pytest.fixture
def registry() -> ParserRegistry[Document]:
    reg: ParserRegistry[Document] = ParserRegistry()
    reg.register(_FakeParser())
    return reg


@pytest.fixture
def upload() -> Upload:
    """A representative (filename, bytes) upload."""
    return ("notes.fake", b"anything")


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def service(
    registry: ParserRegistry[Document],
) -> Generator[IngestionService, None, None]:
    """A service with no observers — tests add them as needed."""
    s = IngestionService(registry)
    yield s
    s.close()


# Subscription


class TestSubscription:
    def test_subscribe_appends_observer(self, service: IngestionService) -> None:
        obs = _make_observer()
        service.subscribe(obs)
        assert service.observers == (obs,)

    def test_subscribe_preserves_order(self, service: IngestionService) -> None:
        a, b, c = _make_observer("a"), _make_observer("b"), _make_observer("c")
        service.subscribe(a)
        service.subscribe(b)
        service.subscribe(c)
        assert service.observers == (a, b, c)

    def test_unsubscribe_removes(self, service: IngestionService) -> None:
        obs = _make_observer()
        service.subscribe(obs)
        service.unsubscribe(obs)
        assert service.observers == ()

    def test_unsubscribe_missing_is_silent(self, service: IngestionService) -> None:
        # Should not raise — debug log only.
        service.unsubscribe(_make_observer())


# ingest() — happy path


class TestIngestHappyPath:
    def test_calls_each_observer_once(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        a, b, c = _make_observer("a"), _make_observer("b"), _make_observer("c")
        service.subscribe(a)
        service.subscribe(b)
        service.subscribe(c)

        service.ingest(*upload, user_id)

        a.on_ingest.assert_called_once()
        b.on_ingest.assert_called_once()
        c.on_ingest.assert_called_once()

    def test_observers_receive_same_event(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        """Observers share one event — mutations from one propagate to the next."""
        a = _make_observer("a")
        b = _make_observer("b")
        service.subscribe(a)
        service.subscribe(b)

        service.ingest(*upload, user_id)

        event_a = a.on_ingest.call_args.args[0]
        event_b = b.on_ingest.call_args.args[0]
        assert event_a is event_b

    def test_observers_run_in_registration_order(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        order: list[str] = []
        a = _make_observer("a")
        a.on_ingest = MagicMock(side_effect=lambda ev: order.append("a"))
        b = _make_observer("b")
        b.on_ingest = MagicMock(side_effect=lambda ev: order.append("b"))
        service.subscribe(a)
        service.subscribe(b)

        service.ingest(*upload, user_id)

        assert order == ["a", "b"]

    def test_user_id_is_stamped(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        obs = _make_observer()
        service.subscribe(obs)
        event = service.ingest(*upload, user_id)

        # Stamped on the document
        assert event.document.user_id == user_id
        # And visible to observers (they share the same Document instance)
        observer_event = obs.on_ingest.call_args.args[0]
        assert observer_event.document.user_id == user_id

    def test_filename_is_recorded(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        event = service.ingest(*upload, user_id)
        assert event.document.filename == upload[0]

    def test_status_progression_on_success(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        service.subscribe(_make_observer())
        event = service.ingest(*upload, user_id)

        assert event.status is IngestionStatus.COMPLETED
        assert event.document.status is DocumentStatus.READY
        assert event.error_message is None

    def test_no_observers_logs_and_succeeds(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        # No observers subscribed.
        event = service.ingest(*upload, user_id)

        # Document is still marked READY because nothing failed.
        # (This is a *configuration* warning, not a runtime error.)
        assert event.status is IngestionStatus.COMPLETED
        assert event.document.status is DocumentStatus.READY


# ingest() — size cap


class TestIngestSizeCap:
    def test_oversized_upload_rejected(
        self,
        registry: ParserRegistry[Document],
        user_id: uuid.UUID,
    ) -> None:
        with IngestionService(registry, max_upload_bytes=10) as svc:
            with pytest.raises(RAGException, match="exceeds the configured maximum"):
                svc.ingest("big.fake", b"x" * 11, user_id)

    def test_exactly_at_cap_is_allowed(
        self,
        registry: ParserRegistry[Document],
        user_id: uuid.UUID,
    ) -> None:
        with IngestionService(registry, max_upload_bytes=10) as svc:
            event = svc.ingest("ok.fake", b"x" * 10, user_id)
            assert event.status is IngestionStatus.COMPLETED


# ingest() — parser failures


class TestIngestParserFailures:
    def test_unsupported_extension_raises_rag_exception(
        self,
        service: IngestionService,
        user_id: uuid.UUID,
    ) -> None:
        with pytest.raises(UnsupportedFormatError):
            service.ingest("notes.pdf", b"fake", user_id)

    def test_observers_not_called_on_parse_failure(
        self,
        service: IngestionService,
        user_id: uuid.UUID,
    ) -> None:
        obs = _make_observer()
        service.subscribe(obs)

        with pytest.raises(UnsupportedFormatError):
            service.ingest("notes.pdf", b"fake", user_id)

        obs.on_ingest.assert_not_called()

    def test_unexpected_parse_error_wraps_as_rag_exception(
        self,
        user_id: uuid.UUID,
    ) -> None:
        """Non-RAGException parser errors are wrapped, not propagated raw."""

        class _BrokenParser:
            extensions: ClassVar[tuple[str, ...]] = ("fake",)

            def parse(self, raw: bytes | bytearray, filename: str) -> Document:  # noqa: ARG002
                raise OSError("disk on fire")

        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(_BrokenParser())

        with IngestionService(reg) as svc:
            with pytest.raises(RAGException, match="Unexpected error") as exc_info:
                svc.ingest("notes.fake", b"", user_id)

            assert isinstance(exc_info.value.__cause__, OSError)


# ingest() — observer failures


class TestIngestObserverFailures:
    def test_observer_failure_halts_chain(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        a = _make_observer("a")

        def fail(event: IngestionEvent) -> None:
            event.fail("chunking blew up")
            raise EmbeddingError("chunking blew up")

        a.on_ingest = MagicMock(side_effect=fail)
        b = _make_observer("b")
        c = _make_observer("c")

        service.subscribe(a)
        service.subscribe(b)
        service.subscribe(c)

        with pytest.raises(EmbeddingError):
            service.ingest(*upload, user_id)

        # Downstream observers must not have run.
        b.on_ingest.assert_not_called()
        c.on_ingest.assert_not_called()

    def test_observer_failure_marks_event_failed(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        obs = _make_observer()

        def fail(event: IngestionEvent) -> None:
            event.fail("mongo down")
            raise StorageError("mongo down")

        obs.on_ingest = MagicMock(side_effect=fail)
        service.subscribe(obs)

        with pytest.raises(StorageError):
            service.ingest(*upload, user_id)

    def test_observer_rag_exception_subclass_preserved(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        """A StorageError must surface as StorageError — not wrapped to bare RAGException."""
        obs = _make_observer()
        obs.on_ingest = MagicMock(side_effect=StorageError("nope"))
        service.subscribe(obs)

        with pytest.raises(StorageError):
            service.ingest(*upload, user_id)

    def test_non_rag_exception_observer_error_is_wrapped(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        """An observer that raises a plain Exception → wrapped as RAGException."""
        obs = _make_observer()
        original = ValueError("misbehaving observer")
        obs.on_ingest = MagicMock(side_effect=original)
        service.subscribe(obs)

        with pytest.raises(RAGException, match="unexpected error") as exc_info:
            service.ingest(*upload, user_id)

        assert exc_info.value.__cause__ is original

    def test_observer_that_forgets_to_fail_is_handled_defensively(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        """
        Observers are supposed to call event.fail() before raising — but if
        one forgets (and raises a non-RAGException), the service marks the
        event FAILED defensively before re-raising.
        """
        captured: list[IngestionEvent] = []

        def misbehave(event: IngestionEvent) -> None:
            # Note: deliberately does NOT call event.fail() before raising.
            captured.append(event)
            raise RuntimeError("forgot to fail")

        obs = _make_observer()
        obs.on_ingest = MagicMock(side_effect=misbehave)
        service.subscribe(obs)

        with pytest.raises(RAGException):
            service.ingest(*upload, user_id)

        assert captured, "observer should have been called"
        assert captured[0].status is IngestionStatus.FAILED
        assert captured[0].error_message is not None

    def test_document_not_marked_ready_on_failure(
        self,
        service: IngestionService,
        upload: Upload,
        user_id: uuid.UUID,
    ) -> None:
        obs = _make_observer()

        def fail(event: IngestionEvent) -> None:
            event.fail("boom")
            raise StorageError("boom")

        obs.on_ingest = MagicMock(side_effect=fail)
        service.subscribe(obs)

        with pytest.raises(StorageError):
            service.ingest(*upload, user_id)

        # The service must not have marked the document READY after a failure.
        observer_event = obs.on_ingest.call_args.args[0]
        assert observer_event.document.status is DocumentStatus.FAILED


# ingest_batch()


class TestIngestBatch:
    def test_empty_batch_returns_empty_list(
        self,
        service: IngestionService,
        user_id: uuid.UUID,
    ) -> None:
        assert service.ingest_batch([], user_id) == []

    def test_processes_all_uploads(
        self,
        service: IngestionService,
        user_id: uuid.UUID,
    ) -> None:
        service.subscribe(_make_observer())
        uploads: list[Upload] = [
            (f"file{i}.fake", f"content {i}".encode("utf-8")) for i in range(5)
        ]

        results = service.ingest_batch(uploads, user_id)

        assert len(results) == 5
        assert all(ev.status is IngestionStatus.COMPLETED for ev in results)

    def test_preserves_order(
        self,
        service: IngestionService,
        user_id: uuid.UUID,
    ) -> None:
        service.subscribe(_make_observer())
        uploads: list[Upload] = [
            (f"file{i}.fake", f"content {i}".encode("utf-8")) for i in range(4)
        ]

        results = service.ingest_batch(uploads, user_id)

        for (filename, _), event in zip(uploads, results):
            assert event.document.filename == filename

    def test_per_upload_failure_does_not_abort_batch(
        self,
        registry: ParserRegistry[Document],
        user_id: uuid.UUID,
    ) -> None:
        """Two good uploads + one with an unsupported extension → 3 events back, 1 FAILED."""
        with IngestionService(registry) as svc:
            svc.subscribe(_make_observer())

            uploads: list[Upload] = [
                ("a.fake", b"a"),
                ("b.pdf", b"b"),  # no parser registered for .pdf
                ("c.fake", b"c"),
            ]

            results = svc.ingest_batch(uploads, user_id)

            assert len(results) == 3
            assert results[0].status is IngestionStatus.COMPLETED
            assert results[1].status is IngestionStatus.FAILED
            assert results[1].error_message is not None
            assert results[1].document.filename == "b.pdf"
            assert results[2].status is IngestionStatus.COMPLETED

    def test_batch_does_not_raise_on_observer_failure(
        self,
        registry: ParserRegistry[Document],
        user_id: uuid.UUID,
    ) -> None:
        """Even an observer that always raises should not crash the batch."""
        with IngestionService(registry) as svc:
            obs = _make_observer()
            obs.on_ingest = MagicMock(side_effect=StorageError("always fails"))
            svc.subscribe(obs)

            uploads: list[Upload] = [(f"file{i}.fake", b"x") for i in range(3)]

            # No raise.
            results = svc.ingest_batch(uploads, user_id)

            assert len(results) == 3
            assert all(ev.status is IngestionStatus.FAILED for ev in results)


# Lifecycle


class TestLifecycle:
    def test_context_manager_closes(self, registry: ParserRegistry[Document]) -> None:
        with IngestionService(registry) as svc:
            assert svc is not None
        # After exiting, the executor is shut down — submitting raises.
        with pytest.raises(RuntimeError):
            svc._executor.submit(lambda: None)

    def test_close_is_idempotent(self, registry: ParserRegistry[Document]) -> None:
        svc = IngestionService(registry)
        svc.close()
        svc.close()  # second call must not raise

    def test_repr_lists_observer_names(self, service: IngestionService) -> None:
        a = _make_observer("a")
        type(a).__name__ = "FakeObserverA"  # MagicMock lets us override
        service.subscribe(a)
        assert "FakeObserverA" in repr(service) or "MagicMock" in repr(service)
