"""Interactive review visualization for Histopia workflows."""

from histopia.visualization._qc_showcase import export_registration_qc_showcase
from histopia.visualization._server import create_viewer_server, serve_viewer
from histopia.visualization._showcase import export_static_showcase
from histopia.visualization._viewer import (
    MAX_DISPLAY_LINKS,
    build_mask_review,
    build_section_order_review,
    build_section_viewer,
)

__all__ = [
    "MAX_DISPLAY_LINKS",
    "build_mask_review",
    "build_section_order_review",
    "build_section_viewer",
    "create_viewer_server",
    "export_registration_qc_showcase",
    "export_static_showcase",
    "serve_viewer",
]
