"""QuPath interoperability for registration and semantic results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from histopia.qupath._doctor import (
    QUPATH_WORKFLOW_API_VERSION,
    inspect_qupath_environment,
)

if TYPE_CHECKING:
    from histopia.qupath._export import export_qupath_bundle

__all__ = [
    "QUPATH_WORKFLOW_API_VERSION",
    "export_qupath_bundle",
    "inspect_qupath_environment",
]


def __getattr__(name: str) -> Any:
    if name == "export_qupath_bundle":
        from histopia.qupath._export import export_qupath_bundle

        return export_qupath_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
