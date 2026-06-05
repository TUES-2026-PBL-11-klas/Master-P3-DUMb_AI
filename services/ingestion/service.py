"""
IngestionService — Subject in the Observer pattern; orchestrator of the
ingestion pipeline.

Responsibilities:
  - Accept (filename, raw_bytes, user_id) from the WebSocket layer, look
    up the correct parser via ParserRegistry (Strategy pattern), and
    parse the bytes into a Document.
  - Enforce a hard upload size cap (defence in depth — the WebSocket
    handler enforces the same cap at the boundary).
  - Stamp the authenticated user_id onto the parsed Document (parsers
    leave user_id as the nil-UUID sentinel).
  - Build an IngestionEvent and notify every registered IngestionObserver
    in registration order. Observers mutate the event in-place — chunking,
    embedding, and storage each populate their slice of it.
  - Drive the document lifecycle: PARSED → CHUNKED → EMBEDDED → STORED → READY,
    or FAILED on any error.
  - Provide batch ingestion via ThreadPoolExecutor — this is the project's
    multithreading requirement.

Design patterns:
  - Observer: IngestionService is the Subject; ChunkingObserver,
    EmbeddingObserver, and StorageObserver are concrete observers.
    Subscription is open (subscribe()) so new pipeline steps (e.g. a
    logging observer) can be added at startup without touching this class.
  - Strategy (indirect): parser selection is delegated to ParserRegistry.
    This service never knows which parser is active.

Threading model:
  - A single document is processed sequentially through the observer chain
    on the calling thread (ingest()).
  - ingest_batch() submits each document to a ThreadPoolExecutor. Workers
    run the same sequential observer chain; the pool gives parallelism
    across documents, not across observers within a document.
  - This keeps observer implementations free of locking — they only ever
    see one event at a time per thread, and each event has its own
    document and chunk list.

Error policy:
  - Any exception from a parser or observer is logged and surfaced.
    The IngestionEvent is marked FAILED (observers already call
    event.fail() themselves before re-raising; the service does it
    defensively for parser errors and any uncaught exception).
  - Batch ingestion never raises — it returns one IngestionEvent per
    submitted upload, and failures are visible via event.status / event.error_message.
    This lets the caller process a folder without one bad file aborting the rest.

Network architecture note:
    This service receives bytes + filenames, never paths. The TUI reads
    files locally and sends them over the persistent socket; nothing in
    this module touches the filesystem.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING
from uuid import UUID

from services.shared.domain import (
    DocumentStatus,
    IngestionEvent,
    IngestionStatus,
)
from services.shared.exceptions import RAGException

if TYPE_CHECKING:
    from services.ingestion.parsers.registry import ParserRegistry
    from services.shared.domain import Document
    from services.shared.protocols import IngestionObserver

logger = logging.getLogger(__name__)

# Default hard upload size cap.
# ("Hard upload size cap (default: 5 MB per file)"). The WebSocket handler
# enforces the same cap at the boundary; this is defence in depth so the
# service is also safe to call from tests and direct in-process callers.
DEFAULT_MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024  # 5 MB


# Upload payload — (filename, raw_bytes). Used by ingest_batch().
Upload = tuple[str, bytes]


class IngestionService:
    """
    Subject in the Observer pattern; orchestrator of the ingestion pipeline.

    Attributes:
        _registry:        ParserRegistry[Document] — picks the parser per extension.
        _observers:       Ordered list of IngestionObserver instances. The chain
                          runs in registration order; the project's canonical chain
                          is [ChunkingObserver, EmbeddingObserver, StorageObserver].
        _executor:        ThreadPoolExecutor for ingest_batch(). Owned by the
                          service — close() shuts it down.
        _max_upload_bytes: Hard cap on incoming payload size (defence in depth).
    """

    def __init__(
        self,
        registry: "ParserRegistry[Document]",
        observers: "list[IngestionObserver] | None" = None,
        *,
        max_workers: int = 4,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        """
        Args:
            registry:         ParserRegistry pre-populated with the parsers the
                              service should know about. Injected — the service
                              never registers parsers itself, keeping the wiring
                              in one place (composition root).
            observers:        Initial list of observers, in execution order. May be
                              None or empty; observers can be added later via
                              subscribe(). The canonical order is chunk → embed →
                              store, but the service does not enforce that — it
                              simply runs them in the order given.
            max_workers:      Size of the thread pool used by ingest_batch().
                              Defaults to 4, matching the typical small-deployment
                              profile from the project reference.
            max_upload_bytes: Hard cap on a single upload's payload size. Defaults
                              to DEFAULT_MAX_UPLOAD_BYTES (5 MB). Exceeding the
                              cap raises RAGException — no parser is invoked.
        """
        self._registry = registry
        self._observers: list[IngestionObserver] = list(observers or [])
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ingest"
        )
        self._max_upload_bytes = max_upload_bytes

    # Subscription (Observer pattern API)

    def subscribe(self, observer: "IngestionObserver") -> None:
        """
        Register *observer* to receive future IngestionEvents.

        Observers are appended to the end of the chain. Adding a new
        step (e.g. a logging observer that runs last) requires only a
        subscribe() call — no changes to existing observers or to the
        service itself. This is the Open/Closed Principle in action.
        """
        self._observers.append(observer)

    def unsubscribe(self, observer: "IngestionObserver") -> None:
        """
        Remove *observer* from the chain. No-op if it was never subscribed.

        Primarily useful in tests, where a stubbed observer needs to be
        swapped out between cases.
        """
        try:
            self._observers.remove(observer)
        except ValueError:
            logger.debug(
                "IngestionService.unsubscribe: observer %r not registered",
                observer,
            )

    @property
    def observers(self) -> "tuple[IngestionObserver, ...]":
        """Read-only view of the current observer chain (for tests / diagnostics)."""
        return tuple(self._observers)

    @property
    def supported_extensions(self) -> list[str]:
        """
        Extensions accepted by the registered parsers.

        Delegates to the ParserRegistry so the WebSocket/socket layer can
        derive its upload allow-list from the live registry instead of
        hard-coding a set that could drift from the actual parsers.
        """
        return self._registry.supported_extensions

    # Single-document ingestion

    def ingest(
        self,
        filename: str,
        raw: bytes,
        user_id: UUID,
    ) -> IngestionEvent:
        """
        Parse a single upload and run it through the observer chain.

        Pipeline:
            1. Enforce the upload size cap.
            2. Resolve the parser via the registry using *filename*'s
               extension (Strategy lookup).
            3. parser.parse(raw, filename) → Document.
            4. Stamp user_id onto the document (parsers use a nil sentinel).
            5. Build an IngestionEvent and call _notify() to drive observers.
            6. On success, mark the document READY.

        Args:
            filename: Original filename supplied by the client. Used to
                      resolve the parser and stored on the resulting
                      Document. NOT a filesystem path.
            raw:      The raw file bytes as received from the client.
            user_id:  UUID of the authenticated user. Replaces the nil
                      sentinel set by the parser.

        Returns:
            The IngestionEvent after the chain has run. Inspect
            ``event.status`` to see whether the pipeline succeeded
            (COMPLETED) or failed (FAILED).

        Raises:
            RAGException: any unrecoverable error — payload too large,
                          parser lookup failure, parser exception, or
                          observer exception. The event (when one was
                          built) is marked FAILED before the exception
                          propagates, so callers always see a consistent
                          event-state-vs-exception pairing.
        """
        logger.info(
            "IngestionService: ingesting '%s' (%d bytes) for user %s",
            filename,
            len(raw),
            user_id,
        )

        # Step 1: enforce size cap before doing any real work.
        if len(raw) > self._max_upload_bytes:
            logger.exception(
                "IngestionService: upload '%s' is %d bytes — exceeds the "
                "configured maximum of %d bytes.",
                filename,
                len(raw),
                self._max_upload_bytes,
            )

            raise RAGException(
                f"Upload '{filename}' is {len(raw)} bytes — exceeds the "
                f"configured maximum of {self._max_upload_bytes} bytes."
            )

        # Steps 2 + 3: pick parser, parse the bytes.
        # If the registry has no parser for this extension, UnsupportedFormatError
        # propagates — it is a RAGException so the caller can catch it uniformly.
        try:
            parser = self._registry.get_for_filename(filename)
            document = parser.parse(raw, filename)
        except RAGException:
            # Parser-layer errors already carry a useful message; just re-raise.
            logger.exception("IngestionService: parsing failed for '%s'", filename)
            raise
        except Exception as exc:
            # Anything else (MemoryError, …) — wrap so callers can always
            # rely on RAGException as the umbrella type.
            logger.exception(
                "IngestionService: unexpected parse error for '%s'", filename
            )
            raise RAGException(
                f"Unexpected error while parsing '{filename}': {exc}"
            ) from exc

        # Step 4: stamp the real user_id over the nil sentinel.
        document.user_id = user_id

        # Step 5: build the event and run the chain.
        event = IngestionEvent(document=document)
        event.status = IngestionStatus.PARSED

        self._notify(event)

        # Step 6: if the chain finished without complaint, the document is READY.
        # Observers may have advanced event.status to STORED; mark COMPLETED here
        # so the difference between "pipeline finished" and "last observer wrote"
        # is visible.
        if event.status is not IngestionStatus.FAILED:
            event.status = IngestionStatus.COMPLETED
            document.mark_status(DocumentStatus.READY)
            logger.info(
                "IngestionService: ingestion complete for '%s' (doc_id=%s, %d chunks)",
                filename,
                document.id,
                len(event.chunks),
            )

        return event

    # Batch ingestion

    def ingest_batch(
        self,
        uploads: "list[Upload]",
        user_id: UUID,
    ) -> "list[IngestionEvent]":
        """
        Ingest many uploads in parallel via the internal ThreadPoolExecutor.

        Each upload runs through the full observer chain on a worker thread.
        Parallelism is *across* documents — a single document is still
        processed sequentially through chunk → embed → store. This is
        deliberate: observers are stateless w.r.t. each other, so the
        only safe and meaningful axis to parallelise on is the document.

        Unlike ingest(), this method *never raises*: per-upload failures
        are captured in the returned events (event.status == FAILED,
        event.error_message populated). Callers can iterate and decide
        which failures to retry, which to report, etc. This matches the
        TUI use case — a student uploading a folder shouldn't have the
        whole batch abort on one corrupt file.

        Args:
            uploads: List of (filename, raw_bytes) tuples to ingest.
            user_id: UUID of the authenticated user — applied to every
                     upload.

        Returns:
            One IngestionEvent per upload, in the same order as *uploads*.
            Failed events carry status=FAILED and a non-None error_message.
        """
        if not uploads:
            return []

        logger.info(
            "IngestionService: batch ingesting %d upload(s) for user %s",
            len(uploads),
            user_id,
        )

        # Submit all jobs first so they can run in parallel, then collect
        # results in the original order. We can't use executor.map() because
        # we need to swallow exceptions per-upload rather than let the first
        # failure short-circuit the iterator.
        futures: list[Future[IngestionEvent]] = [
            self._executor.submit(self._ingest_safe, filename, raw, user_id)
            for filename, raw in uploads
        ]

        results: list[IngestionEvent] = [f.result() for f in futures]

        # Summarise for the log — useful when a batch of 50 files has
        # 3 silent failures buried inside it.
        failed = sum(1 for ev in results if ev.status is IngestionStatus.FAILED)
        logger.info(
            "IngestionService: batch complete — %d succeeded, %d failed",
            len(results) - failed,
            failed,
        )

        return results

    def _ingest_safe(
        self,
        filename: str,
        raw: bytes,
        user_id: UUID,
    ) -> IngestionEvent:
        """
        ingest() wrapper that never raises — used by ingest_batch workers.

        Any exception is converted into a FAILED event. If the failure
        occurred before the event was constructed (e.g. parser lookup
        or size-cap rejection), we build a placeholder document-less
        event with no chunks, carrying just the error message.
        """
        try:
            return self.ingest(filename, raw, user_id)
        except Exception as exc:
            logger.error(
                "IngestionService: batch worker caught error for '%s': %s",
                filename,
                exc,
            )
            # We don't have a Document — synthesise a minimal failed event so
            # the caller can still see *which* upload failed and why.
            return self._synthetic_failure_event(filename, str(exc))

    @staticmethod
    def _synthetic_failure_event(filename: str, message: str) -> IngestionEvent:
        """
        Build a placeholder IngestionEvent for an upload that never produced
        a Document.

        The placeholder Document carries enough metadata (filename, FAILED
        status, error_message) for the caller to identify what went wrong.
        """
        # Lazy import — keeps the module importable in isolation.
        import os
        import uuid
        from datetime import datetime, timezone

        from services.shared.domain import Document

        placeholder = Document(
            id=uuid.uuid4(),
            user_id=uuid.UUID(int=0),
            content="",
            filename=os.path.basename(filename),
            uploaded_at=datetime.now(tz=timezone.utc),
            status=DocumentStatus.FAILED,
            error_message=message,
        )
        event = IngestionEvent(document=placeholder)
        event.fail(message)
        return event

    # Observer chain driver

    def _notify(self, event: IngestionEvent) -> None:
        """
        Run the observer chain on *event* in registration order.

        Each observer either:
          - mutates the event in-place and returns (success), or
          - calls event.fail(...) and raises (failure).

        If any observer raises, we stop the chain immediately — running
        the remaining observers on a half-built event is never useful
        (e.g. StorageObserver on chunks with no embeddings is itself an
        error; EmbeddingObserver on no chunks is a no-op at best).

        Observers are responsible for calling event.fail() *before*
        raising — the chunking, embedding, and storage observers all do.
        We add a defensive event.fail() here too, for safety against
        observers that might forget.
        """
        if not self._observers:
            logger.warning(
                "IngestionService: no observers registered — event for '%s' "
                "will not be chunked, embedded, or stored.",
                event.document.filename,
            )
            return

        for observer in self._observers:
            try:
                observer.on_ingest(event)
            except RAGException:
                # Observer already called event.fail() — re-raise as-is.
                # We do NOT wrap because the original RAGException subclass
                # (EmbeddingError, StorageError, …) carries useful type info.
                logger.error(
                    "IngestionService: observer %s failed for '%s'",
                    type(observer).__name__,
                    event.document.filename,
                )
                raise
            except Exception as exc:
                # An observer that raised a non-RAGException is a bug — the
                # observers in this project always wrap. Defensively mark
                # the event failed and convert to RAGException so callers
                # can rely on the umbrella type.
                message = (
                    f"Observer {type(observer).__name__} raised an "
                    f"unexpected error for '{event.document.filename}': {exc}"
                )
                logger.exception(message)
                if event.status is not IngestionStatus.FAILED:
                    event.fail(message)
                raise RAGException(message) from exc

    # Lifecycle

    def close(self) -> None:
        """
        Shut down the thread pool. Idempotent.

        After close(), ingest_batch() will raise RuntimeError on the
        executor. Call this on application shutdown.
        """
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "IngestionService":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __repr__(self) -> str:
        names = [type(o).__name__ for o in self._observers]
        return f"IngestionService(observers={names})"
