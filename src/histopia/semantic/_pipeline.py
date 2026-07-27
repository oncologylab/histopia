"""End-to-end orchestration for semantic atlas stages."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from histopia.semantic._atlas import (
    JointAtlas,
    _bounded_regularization_workers,
    _sklearn_estimators,
    fit_joint_atlas,
)
from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._correspondence import CorrespondenceConfig
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
    _fit_config_payload,
    _package_version,
    validate_semantic_result,
    write_atlas_result,
)


def fit_saved_features(config: SemanticAtlasConfig) -> tuple[JointAtlas, Path]:
    """Fit and save a new atlas from compact features in section order."""

    atlas, result = _fit_saved_features(config, reuse_existing=False)
    if atlas is None:  # pragma: no cover - internal contract guard
        raise RuntimeError("semantic atlas fit unexpectedly returned no atlas")
    return atlas, result


def fit_or_reuse_saved_features(
    config: SemanticAtlasConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Return an exact cached result or fit one when its identity differs."""

    _, result = _fit_saved_features(config, reuse_existing=not overwrite)
    return result


def _fit_saved_features(
    config: SemanticAtlasConfig,
    *,
    reuse_existing: bool,
) -> tuple[JointAtlas | None, Path]:
    """Run the measured fit stage with optional strict result reuse."""

    fit_started = time.perf_counter()
    performance: dict[str, object] = {
        "status": "running",
        "started_at": utc_timestamp(),
        "fit_threads": config.fit_threads,
        "cache_requested": reuse_existing,
        "cache_hit": False,
        "elapsed_seconds": 0.0,
    }

    def checkpoint() -> None:
        performance["elapsed_seconds"] = elapsed_seconds(fit_started)
        write_performance_stage(config.output_dir, "fit", performance)

    checkpoint()
    try:
        stage_started = time.perf_counter()
        sections = _load_saved_feature_sections(config)
        performance["feature_load_seconds"] = elapsed_seconds(stage_started)
        performance["slide_count"] = len(sections)
        performance["total_patches"] = sum(
            len(section.features) for section in sections
        )
        performance["feature_storage_dtypes"] = sorted(
            {str(section.features.dtype) for section in sections}
        )
        performance["feature_working_dtype"] = "float32"
        performance["feature_working_copy_policy"] = (
            "direct-concatenate-bounded-in-place-normalize-v2"
        )
        performance["pca_training_copy_policy"] = "private-balanced-sample-reused-v1"
        provenance = sections[0].provenance or {}
        if isinstance(provenance.get("preflight_fingerprint"), str):
            performance["preflight_fingerprint"] = provenance["preflight_fingerprint"]

        if reuse_existing:
            stage_started = time.perf_counter()
            cached, cache_status = _matching_saved_atlas_result(config, sections)
            performance["cache_validation_seconds"] = elapsed_seconds(stage_started)
            performance["cache_status"] = cache_status
            if cached is not None:
                performance["cache_hit"] = True
                performance["atlas_fit_seconds"] = 0.0
                performance["artifact_write_seconds"] = 0.0
                performance["selected_k"] = cached["selected_k"]
                performance["cluster_counts"] = list(config.cluster_counts)
                performance["semantic_result_fingerprint"] = cached["fingerprint"]
                performance["status"] = "completed"
                performance["completed_at"] = utc_timestamp()
                checkpoint()
                return None, config.output_dir / "semantic_result.json"
        else:
            performance["cache_status"] = "disabled"

        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise RuntimeError(
                "joint semantic atlas fitting requires the 'semantic' extra"
            ) from exc

        stage_started = time.perf_counter()
        # Load BLAS/OpenMP-backed estimators before threadpoolctl snapshots runtimes.
        _sklearn_estimators()
        performance["runtime_prepare_seconds"] = elapsed_seconds(stage_started)

        stage_started = time.perf_counter()
        fit_phase_seconds: dict[str, float] = {}
        correspondence_workers = min(
            2,
            config.fit_threads,
            max(1, len(sections) - 1),
        )
        regularization_workers = _bounded_regularization_workers(
            config.fit_threads,
            len(config.cluster_counts),
        )
        performance["correspondence_workers"] = correspondence_workers
        performance["correspondence_descriptor_window_sections"] = min(
            len(sections),
            correspondence_workers + 1,
        )
        performance["regularization_workers"] = regularization_workers

        def record_fit_phase(name: str, seconds: float) -> None:
            fit_phase_seconds[name] = seconds
            performance["atlas_fit_phase_seconds"] = dict(fit_phase_seconds)

        with threadpool_limits(limits=config.fit_threads):
            atlas = fit_joint_atlas(
                sections,
                cluster_counts=config.cluster_counts,
                pca_components=config.pca_components,
                balanced_patch_cap=config.balanced_patch_cap,
                seed=config.seed,
                regularize=True,
                max_cross_section_distance_um=config.max_cross_section_distance_um,
                phase_callback=record_fit_phase,
                correspondence_workers=correspondence_workers,
                regularization_workers=regularization_workers,
            )
        performance["atlas_fit_seconds"] = elapsed_seconds(stage_started)

        stage_started = time.perf_counter()
        result = write_atlas_result(
            atlas,
            sections,
            config.output_dir,
            primary_clusters=config.selected_clusters or atlas.selected_k,
            fit_threads=config.fit_threads,
            requested_pca_components=config.pca_components,
            balanced_patch_cap=config.balanced_patch_cap,
            seed=config.seed,
            max_cross_section_distance_um=config.max_cross_section_distance_um,
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
    overwrite_fit: bool = False,
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
    return fit_or_reuse_saved_features(config, overwrite=overwrite_fit)


def _matching_saved_atlas_result(
    config: SemanticAtlasConfig,
    sections: tuple[PatchFeatures, ...],
) -> tuple[dict[str, object] | None, str]:
    """Validate whether the current sealed result is exact for this fit."""

    provenance = _common_feature_provenance(sections, config.output_dir)
    if provenance is None or provenance.get("feature_integrity") != "content-sha256-v1":
        return None, "features-not-content-sealed"
    try:
        payload = validate_semantic_result(config.output_dir)
    except FileNotFoundError:
        return None, "result-missing"
    except (OSError, TypeError, ValueError):
        return None, "result-invalid"

    expected_fit_config = _fit_config_payload(
        requested_pca_components=config.pca_components,
        balanced_patch_cap=config.balanced_patch_cap,
        seed=config.seed,
        max_cross_section_distance_um=config.max_cross_section_distance_um,
    )
    expected_runtime = {
        package: _package_version(package)
        for package in ("numpy", "scikit-learn", "scipy", "threadpoolctl")
    }
    expected_runtime["native_threads"] = config.fit_threads
    expected_correspondence = json.loads(
        json.dumps(
            asdict(
                CorrespondenceConfig(
                    patch_width_um=config.patch_size_px * config.analysis_mpp
                )
            )
        )
    )
    expected_components = min(
        config.pca_components,
        sections[0].features.shape[1],
        sum(
            min(len(section.features), config.balanced_patch_cap)
            for section in sections
        ),
    )
    slide_rows = payload.get("slides")
    slide_ids = (
        [row.get("id") for row in slide_rows]
        if isinstance(slide_rows, list)
        and all(isinstance(row, dict) for row in slide_rows)
        else None
    )
    selected_k = payload.get("selected_k")
    expected_primary = config.selected_clusters or selected_k
    checks = (
        (
            payload.get("feature_provenance") == provenance,
            "feature-identity-differs",
        ),
        (payload.get("fit_config") == expected_fit_config, "fit-config-differs"),
        (payload.get("fit_runtime") == expected_runtime, "fit-runtime-differs"),
        (
            payload.get("correspondence") == expected_correspondence,
            "correspondence-config-differs",
        ),
        (
            payload.get("cluster_counts") == list(config.cluster_counts),
            "cluster-counts-differ",
        ),
        (
            payload.get("pca_components") == expected_components,
            "pca-components-differ",
        ),
        (
            payload.get("feature_normalization") == "patch_l2_v2",
            "feature-normalization-differs",
        ),
        (
            isinstance(selected_k, int) and selected_k in config.cluster_counts,
            "selected-cluster-count-invalid",
        ),
        (
            payload.get("primary_clusters") == expected_primary,
            "primary-cluster-count-differs",
        ),
        (
            slide_ids == [section.slide_id for section in sections],
            "slide-order-differs",
        ),
    )
    for matches, status in checks:
        if not matches:
            return None, status
    return payload, "exact-result-reused"


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
