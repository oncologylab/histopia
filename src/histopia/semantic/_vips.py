"""Process-wide libvips controls for semantic WSI extraction."""

from __future__ import annotations

import os
import sys


def configure_vips_threads(thread_count: int | None) -> None:
    """Set libvips' worker cap before pyvips is imported."""

    if thread_count is None:
        return
    requested = str(thread_count)
    current = os.environ.get("VIPS_CONCURRENCY")
    if "pyvips" in sys.modules and current != requested:
        raise RuntimeError(
            "vips_threads cannot change after pyvips is imported; "
            "start a new process or set VIPS_CONCURRENCY before import"
        )
    os.environ["VIPS_CONCURRENCY"] = requested
