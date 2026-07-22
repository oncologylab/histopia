"""Compatibility exports for visualization APIs formerly owned by registration."""

from histopia.visualization._viewer import (
    build_section_order_review,
    build_section_viewer,
)

__all__ = ["build_section_order_review", "build_section_viewer"]
