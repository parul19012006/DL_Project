"""Safe temporary-file handling.

Legal documents are confidential, so scratch copies get stricter
treatment than ``tempfile.NamedTemporaryFile`` gives by default:

* everything lives under the configured ``TEMP_DIR``, not the shared
  system temp directory where any local user can list filenames
* files are created ``0600`` and directories ``0700``, via ``mkstemp``
  (which never races with an attacker pre-creating the path)
* names are random, never derived from the uploaded filename, so a
  filename cannot leak through a directory listing or a log line
* cleanup runs in ``finally``, so it happens on success, on exception
  and on early return alike
* a cleanup failure is logged, never raised — it must not mask the real
  error that was already propagating

Uploads are streamed in bounded blocks and abandoned the moment they
exceed the size limit, so an oversized body is never fully buffered.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from app.config import Settings, get_settings
from app.ingestion.errors import FileTooLargeError
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Read/write in 1 MiB blocks — bounded memory regardless of file size.
STREAM_BLOCK_SIZE = 1024 * 1024


def _temp_root(settings: Optional[Settings] = None) -> Path:
    settings = settings or get_settings()
    root = settings.temp_dir_path
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:  # pragma: no cover - e.g. a mounted volume
        logger.debug("Could not tighten permissions on %s", root)
    return root


@contextmanager
def secure_temp_file(
    suffix: str = "", settings: Optional[Settings] = None
) -> Iterator[Path]:
    """Yield a private temp file path; always removed afterwards."""
    root = _temp_root(settings)
    handle, name = tempfile.mkstemp(suffix=suffix, dir=str(root))
    os.close(handle)
    path = Path(name)
    try:
        os.chmod(path, 0o600)
        yield path
    finally:
        _remove_quietly(path)


@contextmanager
def secure_temp_dir(settings: Optional[Settings] = None) -> Iterator[Path]:
    """Yield a private temp directory; always removed afterwards."""
    root = _temp_root(settings)
    path = Path(tempfile.mkdtemp(dir=str(root)))
    try:
        os.chmod(path, 0o700)
        yield path
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:  # pragma: no cover
            logger.warning("Could not remove temp directory %s", path)


def _remove_quietly(path: Path) -> None:
    """Delete ``path``, logging rather than raising on failure.

    Cleanup runs while another exception may be propagating; raising
    here would replace the real error with a misleading one.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover
        logger.warning("Could not remove temp file %s: %s", path, exc)


def stream_to_temp_file(
    source: BinaryIO,
    suffix: str = "",
    settings: Optional[Settings] = None,
) -> Path:
    """Copy ``source`` into a private temp file, enforcing the size cap.

    The caller owns the returned path and must delete it — use
    :func:`managed_upload` instead unless you need manual control.
    """
    settings = settings or get_settings()
    limit = settings.max_file_size_bytes
    root = _temp_root(settings)

    handle, name = tempfile.mkstemp(suffix=suffix, dir=str(root))
    path = Path(name)
    written = 0
    try:
        os.chmod(path, 0o600)
        with os.fdopen(handle, "wb") as out:
            while True:
                block = source.read(STREAM_BLOCK_SIZE)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    raise FileTooLargeError(
                        f"The upload exceeds the {settings.max_file_size_mb} "
                        "MB limit",
                        {"limit_mb": settings.max_file_size_mb},
                    )
                out.write(block)
    except BaseException:
        # Abandon the partial file on any failure, including cancellation.
        _remove_quietly(path)
        raise
    return path


def sweep_stale_temp_files(
    settings: Optional[Settings] = None, max_age_hours: Optional[float] = None
) -> int:
    """Delete leftovers from a previous run. Returns how many went.

    Every temp file this service creates is removed in a ``finally``, so
    under normal operation there is nothing to sweep. A ``SIGKILL``, an
    OOM kill or a container eviction skips that ``finally``, and the file
    stays — on a shared volume, for ever. Over a few hundred ingests of
    50 MB PDFs that is a disk-full incident with no obvious cause.

    Run once at startup, by age, and only inside ``TEMP_DIR``. The age
    bound is what makes it safe to run while the service is live: a file
    younger than the cut-off may belong to an in-flight request.

    Never raises. A service that will not start because it could not
    tidy up is worse than the untidiness.
    """
    settings = settings or get_settings()
    hours = (
        settings.temp_file_max_age_hours if max_age_hours is None else max_age_hours
    )
    if hours <= 0:
        return 0

    cutoff = time.time() - hours * 3600
    removed = 0

    try:
        root = _temp_root(settings)
        entries = list(root.iterdir())
    except Exception as exc:
        # An unreadable or uncreatable temp directory is a real problem,
        # but it is ingestion's problem to report when a request needs
        # the directory — not a reason for this tidy-up to raise.
        logger.warning("Could not scan the temp directory: %s", exc)
        return 0

    for entry in entries:
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
            removed += 1
        except OSError as exc:
            logger.debug("Could not remove stale temp entry %s: %s", entry, exc)

    if removed:
        logger.warning(
            "Removed %d stale temp file(s) older than %.1f h from %s. These "
            "are leftovers from a process that was killed mid-request.",
            removed,
            hours,
            root,
        )
    return removed


@contextmanager
def managed_upload(
    source: BinaryIO,
    suffix: str = "",
    settings: Optional[Settings] = None,
) -> Iterator[Path]:
    """Stream an upload to a private temp file and clean it up after."""
    path = stream_to_temp_file(source, suffix=suffix, settings=settings)
    try:
        yield path
    finally:
        _remove_quietly(path)


__all__ = [
    "STREAM_BLOCK_SIZE",
    "secure_temp_file",
    "secure_temp_dir",
    "stream_to_temp_file",
    "managed_upload",
    "sweep_stale_temp_files",
]
