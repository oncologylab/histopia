"""Fingerprint-bound pathology annotations for serial histology sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "AnnotationClass": ("histopia.annotation._model", "AnnotationClass"),
    "AnnotationOntology": ("histopia.annotation._model", "AnnotationOntology"),
    "AnnotationStore": ("histopia.annotation._store", "AnnotationStore"),
    "default_pancreas_ontology": (
        "histopia.annotation._model",
        "default_pancreas_ontology",
    ),
    "registration_result_fingerprint": (
        "histopia.annotation._store",
        "registration_result_fingerprint",
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
