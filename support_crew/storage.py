"""answers.txt persistence.

All writes go through `append_record`, which:
  * assigns every record a unique Record-ID,
  * takes an OS-level file lock so concurrent sessions can never interleave
    or corrupt records,
  * remembers the last record ID written on this thread, so the caller that
    triggered the crew run can verify *its* record landed (a plain
    file-size delta cannot distinguish between two concurrent sessions).
"""

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from . import config

logger = logging.getLogger(__name__)

RECORD_TEMPLATE = (
    "============================================================\n"
    "MULTI-AGENT CUSTOMER SUPPORT RESPONSE\n"
    "Record-ID: {record_id}\n"
    "============================================================\n"
    "\n"
    "Query:\n"
    "{query}\n"
    "\n"
    "------------------------------------------------------------\n"
    "Assistant Answer:\n"
    "{assistant_answer}\n"
    "\n"
    "------------------------------------------------------------\n"
    "Web Search Answer:\n"
    "{web_search_answer}\n"
    "\n"
    "============================================================\n"
    "\n"
)

# The Entry Agent's tool runs synchronously inside the same thread as the
# Streamlit session that kicked off the crew, so a thread-local is the right
# scope for "the record MY run just wrote".
_local = threading.local()


def _lock_for(path: Path) -> FileLock:
    """One sidecar .lock file per answers file (git-ignored)."""
    return FileLock(f"{path}.lock")


def _rotate_if_needed(path: Path, max_bytes: int) -> None:
    """Archive the answers file once it reaches the size cap.

    Rotation keeps the file (which accumulates user queries) from growing
    without bound. Must be called while holding the file lock. A cap of 0
    or less disables rotation.
    """
    if max_bytes <= 0 or not path.exists() or path.stat().st_size < max_bytes:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    if archive.exists():  # two rotations within one second
        archive = path.with_name(f"{path.stem}-{stamp}-{uuid.uuid4().hex[:8]}{path.suffix}")
    path.rename(archive)
    logger.info(
        "Rotated %s to %s (size cap %d bytes reached)",
        path.name, archive.name, max_bytes,
    )


def append_record(
    query: str,
    assistant_answer: str,
    web_search_answer: str,
    *,
    path: Path | None = None,
    max_bytes: int | None = None,
) -> str:
    """Append one support record under a file lock; return its Record-ID.

    Append mode preserves earlier records; UTF-8 is explicit. The generated
    Record-ID is also remembered thread-locally for save verification. When
    the file has reached the size cap it is rotated to a timestamped
    archive first.
    """
    path = path or config.ANSWERS_FILE
    max_bytes = config.ANSWERS_MAX_BYTES if max_bytes is None else max_bytes
    record_id = uuid.uuid4().hex
    record = RECORD_TEMPLATE.format(
        record_id=record_id,
        query=query.strip(),
        assistant_answer=assistant_answer.strip(),
        web_search_answer=web_search_answer.strip(),
    )
    with _lock_for(path):
        _rotate_if_needed(path, max_bytes)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record)
    _local.last_record_id = record_id
    logger.info("Appended support record %s to %s", record_id, path.name)
    return record_id


def reset_last_record_id() -> None:
    """Clear the thread-local marker before a new crew run."""
    _local.last_record_id = None


def get_last_record_id() -> str | None:
    """Record-ID written by this thread's most recent append, if any."""
    return getattr(_local, "last_record_id", None)


def record_exists(record_id: str | None, *, path: Path | None = None) -> bool:
    """True if a record with this ID is present in the answers file."""
    path = path or config.ANSWERS_FILE
    if not record_id or not path.exists():
        return False
    return f"Record-ID: {record_id}" in path.read_text(encoding="utf-8")
