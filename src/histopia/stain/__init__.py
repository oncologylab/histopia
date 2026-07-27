"""Quantitative brightfield stain profiling for registered serial sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "SlideAssay": ("histopia.stain._assays", "SlideAssay"),
    "StainApproval": ("histopia.stain._approval", "StainApproval"),
    "StainFamily": ("histopia.stain._assays", "StainFamily"),
    "StainMap": ("histopia.stain._artifacts", "StainMap"),
    "StainModel": ("histopia.stain._model", "StainModel"),
    "StainPreflight": ("histopia.stain._preflight", "StainPreflight"),
    "StainQuantificationConfig": (
        "histopia.stain._config",
        "StainQuantificationConfig",
    ),
    "approve_stain_result": (
        "histopia.stain._approval",
        "approve_stain_result",
    ),
    "benchmark_stain_methods": (
        "histopia.stain._pipeline",
        "benchmark_stain_methods",
    ),
    "load_stain_config": ("histopia.stain._config", "load_stain_config"),
    "preflight_stain_run": (
        "histopia.stain._preflight",
        "preflight_stain_run",
    ),
    "run_stain_quantification": (
        "histopia.stain._pipeline",
        "run_stain_quantification",
    ),
    "summarize_stain_run": ("histopia.stain._qc", "summarize_stain_run"),
    "validate_stain_approval": (
        "histopia.stain._approval",
        "validate_stain_approval",
    ),
    "validate_stain_result": (
        "histopia.stain._result_validation",
        "validate_stain_result",
    ),
    "write_stain_cohort_qc": (
        "histopia.stain._qc",
        "write_stain_cohort_qc",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_IMPORTS)
