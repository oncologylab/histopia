"""End-to-end orchestration for semantic atlas stages."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from histopia.semantic._atlas import JointAtlas, _sklearn_estimators, fit_joint_atlas
from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._extract import (
    extract_registration_features,
    feature_artifact_path,
)
from histopia.semantic._features import PatchEncoder, PatchFeatures
from histopia.semantic._performance import (
    elapsed_seconds,
    utc_timestamp,
    write_performance_stage,
)
from histopia.semantic._preflight import SemanticPreflight
from histopia.semantic._result import (
    _common_feature_provenance,
    write_atlas_result,
)


def fit_saved_features(config: SemanticAtlasConfig) -> tuple[JointAtlas, Path]:
    """Fit and save an atlas from compact feature artifacts in section order."""

    fit_started = time.perf_counter()
    performance: dict[str, object] = {
        "status": "running",
        "started_at": utc_timestamp(),
        "fit_threads": config.fit_threads,
        "elapsed_seconds": 0.0,
    }

    def checkpoint() -> None:
        performance["elapsed_seconds"] = elapsed_seconds(fit_started)
        write_performance_stage(config.output_dir, "fit", performance)

    checkpoint()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        performance["status"] = "failed"
        performance["failure_type"] = type(exc).__name__
        performance["completed_at"] = utc_timestamp()
        checkpoint()
        raise RuntimeError(
            "joint semantic atlas fitting requires the 'semantic' extra"
        ) from exc
    try:
        stage_started = time.perf_counter()
        sections = _load_saved_feature_sections(config)
        performance["feature_load_seconds"] = elapsed_seconds(stage_started)
        performance["slide_count"] = len(sections)
        performance["total_patches"] = sum(
            len(section.features) for section in sections
        )
        provenance = sections[0].provenance or {}
        if isinstance(provenance.get("preflight_fingerprint"), str):
            performance["preflight_fingerprint"] = provenance["preflight_fingerprint"]

        stage_started = time.perf_counter()
        # Load BLAS/OpenMP-backed estimators before threadpoolctl snapshots runtimes.
        _sklearn_estimators()
        performance["runtime_prepare_seconds"] = elapsed_seconds(stage_started)

        stage_started = time.perf_counter()
        with threadpool_limits(limits=config.fit_threads):
            atlas = fit_joint_atlas(
                sections,
                cluster_counts=config.cluster_counts,
                pca_components=config.pca_components,
                balanced_patch_cap=config.balanced_patch_cap,
                seed=config.seed,
                regularize=True,
                max_cross_section_distance_um=config.max_cross_section_distance_um,
            )
        performance["atlas_fit_seconds"] = elapsed_seconds(stage_started)

        stage_started = time.perf_counter()
        result = write_atlas_result(
            atlas,
            sections,
            config.output_dir,
            primary_clusters=config.selected_clusters or atlas.selected_k,
            fit_threads=config.fit_threads,
        )
        performance["artifact_write_seconds"] = elapsed_seconds(stage_started)
        performance["selected_k"] = atlas.selected_k
        performance["cluster_counts"] = list(config.cluster_counts)
        payload = json.loads(result.read_text())
        if isinstance(payload.get("fingerprint"), str):
            performance["semantic_result_fingerprint"] = payload["fingerprint"]
    except BaseException as exc:
        performance["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        performance["failure_type"] = type(exc).__name__
        performance["completed_at"] = utc_timestamp()
        checkpoint()
        raise
    performance["status"] = "completed"
    performance["completed_at"] = utc_timestamp()
    checkpoint()
    return atlas, result


def run_semantic_atlas(
    config: SemanticAtlasConfig,
    encoder: PatchEncoder,
    *,
    preflight: SemanticPreflight | None = None,
    overwrite_features: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Extract compact features, fit the global atlas, and request review."""

    extract_registration_features(
        config,
        encoder,
        preflight=preflight,
        overwrite=overwrite_features,
        progress=progress,
    )
    _, result = fit_saved_features(config)
    return result


def _load_saved_feature_sections(
    config: SemanticAtlasConfig,
) -> tuple[PatchFeatures, ...]:
    preflight_path = config.output_dir / "preflight.json"
    try:
        preflight = json.loads(preflight_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            "semantic preflight is missing; run preflight or extraction first"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError("semantic preflight is not valid JSON") from error
    if not isinstance(preflight, dict):
        raise ValueError("semantic preflight root must be an object")
    fingerprint = preflight.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("semantic preflight fingerprint is missing")
    raw_slides = preflight.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise ValueError("semantic preflight contains no slide order")
    if any(not isinstance(row, dict) for row in raw_slides):
        raise ValueError("semantic preflight slide rows must be objects")
    slides: tuple[dict[str, Any], ...] = tuple(raw_slides)
    names = tuple(str(row.get("slide_name", "")) for row in slides)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("semantic preflight slide names must be non-empty and unique")
    declared_count = preflight.get("slide_count")
    if declared_count is not None and declared_count != len(slides):
        raise ValueError("semantic preflight slide count is stale")

    feature_dir = config.output_dir / "features"
    paths = tuple(
        feature_artifact_path(feature_dir, order, name)
        for order, name in enumerate(names, start=1)
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"semantic feature artifact is missing: {missing[0]}")
    loaded: list[PatchFeatures] = []
    for path, slide in zip(paths, slides, strict=True):
        section = PatchFeatures.load(path)
        _validate_saved_feature_section(
            section,
            path,
            slide,
            fingerprint,
            config,
        )
        loaded.append(section)
    sections = tuple(loaded)
    if _common_feature_provenance(sections, config.output_dir) is None:
        raise ValueError("saved semantic features must contain provenance")
    return sections


def _validate_saved_feature_section(
    section: PatchFeatures,
    path: Path,
    preflight_slide: dict[str, Any],
    preflight_fingerprint: str,
    config: SemanticAtlasConfig,
) -> None:
    slide_name = str(preflight_slide["slide_name"])
    if section.slide_id != slide_name:
        raise ValueError(f"{path.name}: feature slide identity differs from preflight")
    provenance = section.provenance
    if not isinstance(provenance, dict):
        raise ValueError(f"{path.name}: feature provenance is missing")
    expected = {
        "preflight_fingerprint": preflight_fingerprint,
        "slide_name": slide_name,
        "analysis_mpp": config.analysis_mpp,
        "patch_size_px": config.patch_size_px,
        "min_tissue_fraction": config.min_tissue_fraction,
    }
    for key in ("source_sha256", "mask_sha256", "transform_sha256"):
        if key in preflight_slide:
            expected[key] = preflight_slide[key]
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"{path.name}: feature provenance differs for {key}")
    if section.analysis_mpp != config.analysis_mpp:
        raise ValueError(f"{path.name}: feature analysis MPP differs from config")
    if section.patch_size_px != config.patch_size_px:
        raise ValueError(f"{path.name}: feature patch size differs from config")
