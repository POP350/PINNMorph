"""Host-wide Linux API slots, independent from GPU worker concurrency."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import time


@contextmanager
def api_slot(base_url, model):
    limit = int(os.environ.get("PINNMORPH_LLM_CONCURRENCY", "0"))
    if limit <= 0 or os.name != "posix":
        yield
        return
    import fcntl
    # One limit across model aliases on the same endpoint; no credentials on disk.
    identity = hashlib.sha256(str(base_url).encode()).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / f"pinnmorph_llm_{os.getuid()}_{identity}"
    directory.mkdir(mode=0o700, exist_ok=True)
    handle = None
    while handle is None:
        for index in range(limit):
            candidate = (directory / f"slot_{index}").open("a+")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = candidate
                break
            except BlockingIOError:
                candidate.close()
        if handle is None:
            time.sleep(0.1)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
