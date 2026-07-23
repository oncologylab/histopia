"""End-to-end orchestration for semantic atlas stages."""

from __future__ import annotations

from pathlib import Path

from histopia.semantic._atlas import JointAtlas, fit_joint_atlas
from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._extract import extract_registration_features
from histopia.semantic._features import PatchEncoder, PatchFeatures
from histopia.semantic._result import write_atlas_result


def fit_saved_features(config: SemanticAtlasConfig) -> tuple[JointAtlas, Path]:
    """Fit and save an atlas from compact feature artifacts in section order."""

    paths = tuple(sorted((config.output_dir / "features").glob("*.npz")))
    if not paths:
        raise FileNotFoundError("no compact feature artifacts found")
    sections = tuple(PatchFeatures.load(path) for path in paths)
    atlas = fit_joint_atlas(
        sections,
        cluster_counts=config.cluster_counts,
        pca_components=config.pca_components,
        balanced_patch_cap=config.balanced_patch_cap,
        seed=config.seed,
        regularize=True,
        max_cross_section_distance_um=config.max_cross_section_distance_um,
    )
    result = write_atlas_result(
        atlas,
        sections,
        config.output_dir,
        primary_clusters=config.selected_clusters or atlas.selected_k,
    )
    return atlas, result


def run_semantic_atlas(
    config: SemanticAtlasConfig,
    encoder: PatchEncoder,
    *,
    overwrite_features: bool = False,
) -> Path:
    """Extract compact features, fit the global atlas, and request review."""

    extract_registration_features(config, encoder, overwrite=overwrite_features)
    _, result = fit_saved_features(config)
    return result
