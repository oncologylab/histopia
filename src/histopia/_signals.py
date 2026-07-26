"""Process-signal handling shared by Histopia command-line entry points."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


@contextmanager
def graceful_sigterm() -> Iterator[None]:
    """Translate ``SIGTERM`` into a catchable, conventional process exit.

    Python's default ``SIGTERM`` action terminates immediately, bypassing
    ``finally`` blocks and workflow telemetry. Command-line entry points use
    this boundary so launchers such as QuPath can cancel a run while Histopia
    still records the active stage as interrupted and releases resources.
    """

    if (
        not hasattr(signal, "SIGTERM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    sigterm = signal.SIGTERM
    previous = signal.getsignal(sigterm)

    def terminate(signum: int, frame: FrameType | None) -> None:
        del frame
        raise SystemExit(128 + signum)

    signal.signal(sigterm, terminate)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)
