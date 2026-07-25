"""Dependency-light compute-device discovery and selection."""

from histopia.compute._runtime import (
    ComputeDevice,
    inspect_compute,
    resolve_compute_device,
)
from histopia.compute._vips import configure_vips_threads

__all__ = [
    "ComputeDevice",
    "configure_vips_threads",
    "inspect_compute",
    "resolve_compute_device",
]
