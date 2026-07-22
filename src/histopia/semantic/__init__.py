"""Global semantic atlases for registered serial histology sections."""

from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._features import PatchFeatures
from histopia.semantic._pipeline import fit_saved_features, run_semantic_atlas
from histopia.semantic._preflight import (
    SemanticPreflight,
    preflight_registration,
    write_preflight,
)

__all__ = [
    "PatchFeatures",
    "SemanticPreflight",
    "SemanticAtlasConfig",
    "fit_saved_features",
    "preflight_registration",
    "run_semantic_atlas",
    "write_preflight",
]
