"""Dependency-light compute-device discovery and selection."""

from histopia.compute._runtime import (
    ComputeDevice,
    inspect_compute,
    resolve_compute_device,
)

__all__ = ["ComputeDevice", "inspect_compute", "resolve_compute_device"]
