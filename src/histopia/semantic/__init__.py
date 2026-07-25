"""Global semantic atlases for registered serial histology sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "PatchFeatures": ("histopia.semantic._features", "PatchFeatures"),
    "SemanticApproval": ("histopia.semantic._approval", "SemanticApproval"),
    "SemanticPreflight": ("histopia.semantic._preflight", "SemanticPreflight"),
    "SemanticAtlasConfig": ("histopia.semantic._config", "SemanticAtlasConfig"),
    "approve_semantic_result": (
        "histopia.semantic._approval",
        "approve_semantic_result",
    ),
    "fit_saved_features": ("histopia.semantic._pipeline", "fit_saved_features"),
    "fit_or_reuse_saved_features": (
        "histopia.semantic._pipeline",
        "fit_or_reuse_saved_features",
    ),
    "preflight_registration": (
        "histopia.semantic._preflight",
        "preflight_registration",
    ),
    "run_semantic_atlas": ("histopia.semantic._pipeline", "run_semantic_atlas"),
    "summarize_semantic_run": (
        "histopia.semantic._qc",
        "summarize_semantic_run",
    ),
    "validate_semantic_approval": (
        "histopia.semantic._approval",
        "validate_semantic_approval",
    ),
    "write_cohort_qc": ("histopia.semantic._qc", "write_cohort_qc"),
    "write_preflight": ("histopia.semantic._preflight", "write_preflight"),
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
    "PatchFeatures",
    "SemanticApproval",
    "SemanticPreflight",
    "SemanticAtlasConfig",
    "approve_semantic_result",
    "fit_or_reuse_saved_features",
    "fit_saved_features",
    "preflight_registration",
    "run_semantic_atlas",
    "summarize_semantic_run",
    "validate_semantic_approval",
    "write_cohort_qc",
    "write_preflight",
]
