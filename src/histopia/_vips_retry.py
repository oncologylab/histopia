"""Bounded recovery for transient libvips/OpenSlide storage reads."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_TRANSIENT_ERROR_MARKERS = (
    "unable to write to memory",
    "operation not permitted",
    "resource temporarily unavailable",
    "temporarily unavailable",
    "input/output error",
    "i/o error",
    "stale file handle",
    "transport endpoint is not connected",
    "connection reset",
    "network is unreachable",
)


def is_transient_vips_error(error: BaseException) -> bool:
    """Return whether an exception resembles a recoverable remote-slide read."""

    message = str(error).casefold()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def retry_transient_vips_read(
    operation: Callable[[], _T],
    *,
    attempts: int = 3,
    initial_delay_seconds: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Retry a bounded libvips read while preserving the final exception.

    Each call to operation must build a fresh libvips graph or reopen the
    source slide. This matters for OpenSlide handles backed by SMB/NFS: a
    handle that observed a transient transport error is not reused.
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if initial_delay_seconds < 0:
        raise ValueError("initial_delay_seconds must be non-negative")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt + 1 >= attempts or not is_transient_vips_error(error):
                raise
            sleep(initial_delay_seconds * (2**attempt))
    raise AssertionError("unreachable")
