"""Interactive review visualization for Histopia workflows."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "MAX_DISPLAY_LINKS": ("histopia.visualization._viewer", "MAX_DISPLAY_LINKS"),
    "build_alignment_review": (
        "histopia.visualization._viewer",
        "build_alignment_review",
    ),
    "build_mask_review": (
        "histopia.visualization._viewer",
        "build_mask_review",
    ),
    "build_non_rigid_review": (
        "histopia.visualization._nonrigid_review",
        "build_non_rigid_review",
    ),
    "build_registration_cohort_review": (
        "histopia.visualization._review_portal",
        "build_registration_cohort_review",
    ),
    "build_registration_review": (
        "histopia.visualization._review_portal",
        "build_registration_review",
    ),
    "build_section_order_review": (
        "histopia.visualization._viewer",
        "build_section_order_review",
    ),
    "build_section_viewer": (
        "histopia.visualization._viewer",
        "build_section_viewer",
    ),
    "create_viewer_server": (
        "histopia.visualization._server",
        "create_viewer_server",
    ),
    "export_registration_qc_showcase": (
        "histopia.visualization._qc_showcase",
        "export_registration_qc_showcase",
    ),
    "export_static_showcase": (
        "histopia.visualization._showcase",
        "export_static_showcase",
    ),
    "serve_viewer": ("histopia.visualization._server", "serve_viewer"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "MAX_DISPLAY_LINKS",
    "build_alignment_review",
    "build_mask_review",
    "build_non_rigid_review",
    "build_registration_cohort_review",
    "build_registration_review",
    "build_section_order_review",
    "build_section_viewer",
    "create_viewer_server",
    "export_registration_qc_showcase",
    "export_static_showcase",
    "serve_viewer",
]
