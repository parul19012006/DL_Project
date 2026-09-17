"""Batch ingestion as a tracked background job.

A batch of ~500 documents cannot be an ordinary synchronous request. Not
because of memory — that part is solved below — but because extracting,
chunking and embedding 500 documents takes minutes to hours, and every
proxy, load balancer and HTTP client between Express and this service
will give up long before it finishes. A 504 after forty minutes of real
work, with no way to find out what got indexed, is the worst possible
outcome.

So ``POST /ingest/batch`` accepts the batch, returns **202 Accepted**
with a job id immediately, and processes it in the background.
``GET /ingest/batch/{job_id}`` reports progress.

**How memory stays bounded.** Three rules, and all three are necessary:

1. **The request carries references, never contents.** A batch item is a
   ``document_id``, a ``tenant_id`` and a path. 500 of those is ~100 KB
   of JSON. There is no field in which 500 documents could arrive, which
   is a stronger guarantee than promising not to hold them.
2. **Documents are processed strictly one at a time.** The worker calls
   ``index_document`` per item and lets the extracted pages, blocks,
   chunks and vectors go out of scope before the next item starts. Peak
   memory is one document, not five hundred.
3. **Only a small result record is kept per item** — ids, counts, status
   and at most a couple of warning strings. Extracted text and chunks are
   never retained for reporting.

**What this job store is not.** It is in-memory and per-process. A
restart loses job *status*; it does not lose indexed data, because each
document is committed to the vector store as it completes and indexing
is idempotent — re-submitting a batch after a restart re-indexes onto the
same deterministic chunk ids rather than duplicating. A durable queue
(Celery, RQ, Arq) is the right answer for multiple replicas, and the
service functions are already framework-free so that swap is contained
to this module. Saying so plainly is better than a job store that
pretends to be durable.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.ingestion.errors import IngestionError
from app.logging_config import get_logger, get_request_id, set_request_id
from app.services import indexing

logger = get_logger(__name__)


class JobStatus:
    """Stable status codes. Express branches on these, not on prose."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    TERMINAL = (COMPLETED, FAILED)


#: At most this many warnings are stored per document. A batch of 500
#: damaged files must not turn the job record into the thing that
#: exhausts memory.
MAX_WARNINGS_PER_ITEM = 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ItemResult:
    """One document's outcome. Deliberately small and fixed-size."""

    document_id: str
    tenant_id: str
    status: str = "success"
    filename: str = ""
    chunks_indexed: int = 0
    pages: int = 0
    error_type: str = ""
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "filename": self.filename,
            "chunks_indexed": self.chunks_indexed,
            "pages": self.pages,
            "error_type": self.error_type or None,
            "error": self.error or None,
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
        }


@dataclass
class IngestJob:
    """A batch, its progress, and its per-document results."""

    job_id: str
    total: int
    request_id: str = ""
    status: str = JobStatus.QUEUED
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks_indexed: int = 0
    results: List[ItemResult] = field(default_factory=list)
    error: str = ""
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def finished(self) -> bool:
        return self.status in JobStatus.TERMINAL

    @property
    def progress(self) -> float:
        return round(self.processed / self.total, 4) if self.total else 1.0

    @property
    def duration_ms(self) -> int:
        if not self.started_at:
            return 0
        end = self.finished_at or _now()
        return int((end - self.started_at).total_seconds() * 1000)

    def to_dict(self, include_results: bool = True) -> Dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "chunks_indexed": self.chunks_indexed,
            "progress": self.progress,
            "error": self.error or None,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }
        if include_results:
            data["results"] = [r.to_dict() for r in self.results]
        return data


class JobRegistry:
    """In-memory job store, bounded so it cannot grow without limit."""

    def __init__(self, history_limit: int = 100) -> None:
        self._jobs: Dict[str, IngestJob] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self.history_limit = max(1, int(history_limit))

    def create(self, total: int, request_id: str = "") -> IngestJob:
        job = IngestJob(
            job_id=uuid.uuid4().hex, total=total, request_id=request_id
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._evict()
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.finished)

    def list(self, limit: int = 20) -> List[IngestJob]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order[-limit:])]

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._order.clear()

    def _evict(self) -> None:
        """Drop the oldest *finished* jobs past the limit.

        Running jobs are never evicted — losing the record of work still
        in progress would leave a caller polling an id that has ceased to
        exist while its documents are still being indexed.
        """
        while len(self._order) > self.history_limit:
            for position, job_id in enumerate(self._order):
                job = self._jobs.get(job_id)
                if job is None or job.finished:
                    self._order.pop(position)
                    self._jobs.pop(job_id, None)
                    break
            else:
                return  # every stored job is still running


_registry: Optional[JobRegistry] = None
_registry_lock = threading.Lock()


def get_job_registry(settings: Optional[Settings] = None) -> JobRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                settings = settings or get_settings()
                _registry = JobRegistry(settings.ingest_job_history)
    return _registry


def set_job_registry(registry: Optional[JobRegistry]) -> None:
    global _registry
    _registry = registry


# =====================================================================
# The worker
# =====================================================================


def run_batch(
    job: IngestJob,
    items: Sequence[Dict[str, Any]],
    settings: Optional[Settings] = None,
    index_one: Optional[Callable[..., Any]] = None,
) -> IngestJob:
    """Index a batch, one document at a time.

    ``index_one`` is injectable so the memory behaviour can be asserted
    in a test rather than asserted in a comment.

    A failure on one document never stops the batch: a damaged file in
    position 200 must not cost the 300 documents behind it. Each failure
    is recorded against its own id and the job continues.
    """
    settings = settings or get_settings()
    index_one = index_one or indexing.index_document

    # Carry the submitting request's id into the worker's log lines, so a
    # background job can be traced back to the call that started it.
    token = set_request_id(job.request_id or job.job_id[:16])
    job.status = JobStatus.RUNNING
    job.started_at = _now()

    logger.info(
        "Batch ingest job %s starting: %d document(s)", job.job_id, job.total
    )

    try:
        for item in items:
            _process_one(job, item, settings, index_one)
        job.status = JobStatus.COMPLETED
    except Exception as exc:  # pragma: no cover - the loop catches its own
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Batch ingest job %s failed", job.job_id)
    finally:
        job.finished_at = _now()
        from app.logging_config import reset_request_id

        reset_request_id(token)

    logger.info(
        "Batch ingest job %s %s: %d succeeded, %d failed, %d chunk(s) indexed "
        "(%d ms)",
        job.job_id,
        job.status,
        job.succeeded,
        job.failed,
        job.chunks_indexed,
        job.duration_ms,
    )
    return job


def _process_one(
    job: IngestJob,
    item: Dict[str, Any],
    settings: Settings,
    index_one: Callable[..., Any],
) -> None:
    """One document, start to finish, then everything about it released.

    Nothing derived from the document — pages, blocks, chunks, vectors —
    outlives this function. Only the small :class:`ItemResult` is kept.
    """
    started = time.perf_counter()
    document_id = str(item.get("document_id", ""))
    tenant_id = str(item.get("tenant_id", ""))

    try:
        result = index_one(settings=settings, **item)
        record = ItemResult(
            document_id=result.document_id or document_id,
            tenant_id=result.tenant_id or tenant_id,
            status=result.status,
            filename=result.filename,
            chunks_indexed=result.chunks_indexed,
            pages=result.pages,
            warnings=list(result.warnings)[:MAX_WARNINGS_PER_ITEM],
            duration_ms=result.duration_ms,
        )
        # "unchanged" is a success: the document is indexed and
        # current, which is what the caller asked for.
        if record.status in ("success", "unchanged"):
            job.succeeded += 1
            job.chunks_indexed += record.chunks_indexed
        else:
            job.failed += 1

    except IngestionError as exc:
        logger.warning(
            "Batch job %s: %s failed (%s)", job.job_id, document_id, exc.error_type
        )
        record = ItemResult(
            document_id=document_id,
            tenant_id=tenant_id,
            status="failed",
            error_type=exc.error_type,
            error=exc.message,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        job.failed += 1

    except Exception as exc:
        # A surprise on one document must not kill the batch.
        logger.exception("Batch job %s: unexpected error on %s", job.job_id, document_id)
        record = ItemResult(
            document_id=document_id,
            tenant_id=tenant_id,
            status="failed",
            error_type="internal_error",
            error=type(exc).__name__,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        job.failed += 1

    job.results.append(record)
    job.processed += 1


def submit_batch(
    items: Sequence[Dict[str, Any]],
    settings: Optional[Settings] = None,
    registry: Optional[JobRegistry] = None,
) -> IngestJob:
    """Register a job for a batch. The caller schedules :func:`run_batch`."""
    settings = settings or get_settings()
    registry = registry or get_job_registry(settings)
    return registry.create(total=len(items), request_id=get_request_id())


__all__ = [
    "IngestJob",
    "ItemResult",
    "JobStatus",
    "JobRegistry",
    "get_job_registry",
    "set_job_registry",
    "run_batch",
    "submit_batch",
    "MAX_WARNINGS_PER_ITEM",
]
