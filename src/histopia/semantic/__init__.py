"""Global semantic atlases for registered serial histology sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "PatchFeatures": ("histopia.semantic._features", "PatchFeatures"),
    "SemanticPreflight": ("histopia.semantic._preflight", "SemanticPreflight"),
    "SemanticAtlasConfig": ("histopia.semantic._config", "SemanticAtlasConfig"),
    "fit_saved_features": ("histopia.semantic._pipeline", "fit_saved_features"),
    "preflight_registration": (
        "histopia.semantic._preflight",
        "preflight_registration",
    ),
    "run_semantic_atlas": ("histopia.semantic._pipeline", "run_semantic_atlas"),
    "summarize_semantic_run": (
        "histopia.semantic._qc",
        "summarize_semantic_run",
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
    "SemanticPreflight",
    "SemanticAtlasConfig",
    "fit_saved_features",
    "preflight_registration",
    "run_semantic_atlas",
    "summarize_semantic_run",
    "write_cohort_qc",
    "write_preflight",
]
