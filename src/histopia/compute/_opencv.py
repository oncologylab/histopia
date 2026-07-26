"""Process-wide OpenCV execution controls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from threading import RLock

from histopia._validation import positive_int

_OPENCV_THREAD_LOCK = RLock()


@contextmanager
def opencv_thread_limit(thread_count: int | None) -> Iterator[int]:
    """Apply and restore an optional process-wide OpenCV thread limit."""

    requested = (
        None if thread_count is None else positive_int("opencv_threads", thread_count)
    )
    try:
        cv2 = import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV thread control requires the registration dependencies"
        ) from exc

    if requested is None:
        yield int(cv2.getNumThreads())
        return

    with _OPENCV_THREAD_LOCK:
        previous = int(cv2.getNumThreads())
        cv2.setNumThreads(requested)
        try:
            yield int(cv2.getNumThreads())
        finally:
            cv2.setNumThreads(previous)
