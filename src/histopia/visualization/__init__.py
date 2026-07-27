"""Interactive review visualization for Histopia workflows."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "CohortWorkflowAudit": (
        "histopia.visualization._audit",
        "CohortWorkflowAudit",
    ),
    "RegistrationWorkflowAudit": (
        "histopia.visualization._audit",
        "RegistrationWorkflowAudit",
    ),
    "SemanticWorkflowAudit": (
        "histopia.visualization._audit",
        "SemanticWorkflowAudit",
    ),
    "StainWorkflowAudit": (
        "histopia.visualization._audit",
        "StainWorkflowAudit",
    ),
    "ViewerWorkflowAudit": (
        "histopia.visualization._audit",
        "ViewerWorkflowAudit",
    ),
    "load_registration_feedback": (
        "histopia.visualization._feedback",
        "load_registration_feedback",
    ),
    "registration_feedback_rows": (
        "histopia.visualization._feedback",
        "registration_feedback_rows",
    ),
    "summarize_registration_feedback": (
        "histopia.visualization._feedback",
        "summarize_registration_feedback",
    ),
    "WorkflowAudit": ("histopia.visualization._audit", "WorkflowAudit"),
    "audit_workflows": ("histopia.visualization._audit", "audit_workflows"),
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
    "build_stain_review": (
        "histopia.visualization._stain_review",
        "build_stain_review",
    ),
    "build_topology_review": (
        "histopia.visualization._topology_review",
        "build_topology_review",
    ),
    "build_workflow_review": (
        "histopia.visualization._review_portal",
        "build_workflow_review",
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
    "write_workflow_audit": (
        "histopia.visualization._audit",
        "write_workflow_audit",
    ),
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


__all__ = sorted(_PUBLIC_IMPORTS)
