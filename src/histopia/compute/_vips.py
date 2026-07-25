"""Process-wide libvips execution controls."""

from __future__ import annotations

import os
import sys

from histopia._validation import positive_int


def configure_vips_threads(thread_count: int | None) -> None:
    """Set libvips' worker cap before pyvips initializes its native runtime."""

    if thread_count is None:
        return
    requested = str(positive_int("vips_threads", thread_count))
    current = os.environ.get("VIPS_CONCURRENCY")
    if "pyvips" in sys.modules and current != requested:
        raise RuntimeError(
            "vips_threads cannot change after pyvips is imported; "
            "start a new process or set VIPS_CONCURRENCY before import"
        )
    os.environ["VIPS_CONCURRENCY"] = requested
