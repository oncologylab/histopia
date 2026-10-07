"""Resumable cohort fitting and continuous stain-map generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.compute import configure_vips_threads
from histopia.stain._adaptive import (
    AdaptiveBackgroundResult,
    CounterstainAdaptiveResult,
    apply_adaptive_background,
    apply_counterstain_conditioned_background,
    infer_adaptive_background,
    infer_counterstain_conditioned_background,
)
from histopia.stain._artifacts import AdaptiveStainMap, StainMap
from histopia.stain._assays import StainFamily
from histopia.stain._config import StainQuantificationConfig
from histopia.stain._io import AnalysisSlide, read_analysis_slide
from histopia.stain._model import (
    CandidateFit,
    StainModel,
    _rank_correlation,
    canonical_vectors,
    cohort_vector_template,
    counterstain_only_mask,
    fit_candidate,
    select_family_method,
    shrink_vectors,
    unmix_od,
)
from histopia.stain._od import (
    BackgroundModel,
    apply_shading_correction,
    estimate_background_model,
    rgb_to_od,
)
from histopia.stain._preflight import (
    StainPreflight,
    StainPreflightSlide,
    preflight_stain_run,
    write_stain_preflight,
)
from histopia.stain._result import write_stain_result

_FIT_ALGORITHM_VERSION = 4
_MAP_ALGORITHM_VERSION = 4
_NUMERIC_THREAD_LIMITER: object | None = None


def benchmark_stain_methods(
    config: StainQuantificationConfig,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit candidate vectors and select one method per stain family."""

    configure_vips_threads(config.vips_threads)
    preflight = preflight_stain_run(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_stain_preflight(preflight, config.output_dir / "preflight.json")
    registration = json.loads(
        (config.registration_run / "registration_result.json").read_text()
    )
    rows = {Path(str(row["path"])).name: row for row in registration["slides"]}
    fit_root = config.output_dir / "fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    fit_payloads: list[dict[str, object]] = []
    quantified = [
        slide
        for slide in preflight.slides
        if slide.assay.family is not StainFamily.CONTEXT_HE
        and slide.assay.analysis_included
    ]
    registration_order = {
        Path(str(row["path"])).name: order
        for order, row in enumerate(registration["slides"], start=1)
    }

    fit_items = list(enumerate(quantified, start=1))
    fit_tasks = [
        (
            config,
            preflight,
            slide,
            registration_order[slide.slide_name],
            rows[slide.slide_name]["transform"]["matrix"],
            overwrite,
        )
        for _, slide in fit_items
    ]
    if config.workers == 1:
        for (index, slide), task in zip(fit_items, fit_tasks, strict=True):
            _report_fit_progress(progress, index, len(quantified), slide)
            fit_payloads.append(_fit_slide_task(task))
    else:
        with _stain_process_pool(config) as executor:
            payloads = executor.map(_fit_slide_task, fit_tasks, chunksize=1)
            for (index, slide), payload in zip(
                fit_items,
                payloads,
                strict=True,
            ):
                _report_fit_progress(progress, index, len(quantified), slide)
                fit_payloads.append(payload)

    families: dict[str, dict[str, object]] = {}
    for family in (
        StainFamily.H_DAB,
        StainFamily.SIRIUS_RED,
        StainFamily.PAS,
        StainFamily.ALCIAN_BLUE,
    ):
        family_rows = [row for row in fit_payloads if row["family"] == family.value]
        if not family_rows:
            continue
        candidates = [
            [CandidateFit.from_json_dict(candidate) for candidate in row["candidates"]]
            for row in family_rows
        ]
        selected, metrics = select_family_method(
            candidates,
            minimum_target_rank=config.correction_rank_guard,
        )
        selected_vectors = [
            next(
                candidate.vectors
                for candidate in slide_candidates
                if candidate.method == selected
            )
            for slide_candidates in candidates
        ]
        template = cohort_vector_template(selected_vectors)
        families[family.value] = {
            "selected_method": selected,
            "method_metrics": metrics,
            "cohort_vectors": template.tolist(),
            "slide_count": len(family_rows),
        }
    benchmark_core = {
        "schema_version": 1,
        "algorithm_version": _FIT_ALGORITHM_VERSION,
        "preflight_fingerprint": preflight.fingerprint,
        "request": _benchmark_request(config),
        "families": families,
        "slides": [
            {
                "slide_id": row["slide_id"],
                "slide_order": row["slide_order"],
                "marker": row["marker"],
                "family": row["family"],
                "analysis_shape": row["analysis_shape"],
                "tissue_pixels": row["tissue_pixels"],
                "glass_pixels": row["glass_pixels"],
                "background": row["background"],
                "candidates": row["candidates"],
            }
            for row in fit_payloads
        ],
    }
    benchmark = {
        **benchmark_core,
        "fingerprint": _json_sha256(benchmark_core),
    }
    path = config.output_dir / "benchmark.json"
    write_json_atomic(path, benchmark)
    return path


def run_stain_quantification(
    config: StainQuantificationConfig,
    *,
    overwrite_fits: bool = False,
    overwrite_maps: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit or reuse cohort models, then generate sealed continuous maps."""

    started = time.perf_counter()
    benchmark_path = benchmark_stain_methods(
        config,
        overwrite=overwrite_fits,
        progress=progress,
    )
    benchmark = json.loads(benchmark_path.read_text())
    preflight = preflight_stain_run(config)
    if benchmark.get("preflight_fingerprint") != preflight.fingerprint:
        raise ValueError("stain benchmark belongs to a different preflight")
    fit_by_slide = {row["slide_id"]: row for row in benchmark.get("slides", [])}
    maps_root = config.output_dir / "maps"
    models_root = config.output_dir / "models"
    maps_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    slide_rows: list[dict[str, object]] = []
    map_items = list(enumerate(preflight.slides, start=1))
    map_tasks = [
        (
            config,
            preflight,
            slide,
            order,
            benchmark,
            fit_by_slide.get(slide.slide_name),
            overwrite_maps,
        )
        for order, slide in map_items
    ]
    if config.workers == 1:
        map_results = []
        for (order, slide), task in zip(map_items, map_tasks, strict=True):
            _report_map_progress(progress, order, len(preflight.slides), slide)
            map_results.append(_map_slide_task(task))
    else:
        map_results = []
        with _stain_process_pool(config) as executor:
            results = executor.map(_map_slide_task, map_tasks, chunksize=1)
            for (order, slide), result in zip(
                map_items,
                results,
                strict=True,
            ):
                _report_map_progress(progress, order, len(preflight.slides), slide)
                map_results.append(result)
    slide_rows.extend(row for row, _ in map_results)
    maps_written = sum(outcome == "written" for _, outcome in map_results)
    maps_reused = sum(outcome == "reused" for _, outcome in map_results)
    core = {
        "schema_version": 1,
        "measurement": {
            "quantity": "relative_chromogen_optical_density",
            "logarithm": "natural",
            "absolute_calibration": False,
            "cross_antibody_normalization": False,
            "source_space_measurement": True,
            "analysis_mpp": config.analysis_mpp,
        },
        "preflight_fingerprint": preflight.fingerprint,
        "registration_result_sha256": preflight.registration_result_sha256,
        "registration_approval_sha256": preflight.registration_approval_sha256,
        "preflight": "preflight.json",
        "benchmark": "benchmark.json",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "families": benchmark["families"],
        "runtime": {
            "workers": config.workers,
            "vips_threads": config.vips_threads,
        },
        "slides": slide_rows,
    }
    result_path = write_stain_result(config.output_dir, core)
    result = json.loads(result_path.read_text())
    write_json_atomic(
        config.output_dir / "stain_performance.json",
        {
            "schema_version": 1,
            "result_fingerprint": result["fingerprint"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "maps_written": maps_written,
            "maps_reused": maps_reused,
            "quantified_slides": maps_written + maps_reused,
            "workers": config.workers,
            "vips_threads": config.vips_threads,
            "compute_backend": "cpu",
        },
    )
    return result_path


def _fit_slide_task(
    task: tuple[
        StainQuantificationConfig,
        StainPreflight,
        StainPreflightSlide,
        int,
        object,
        bool,
    ],
) -> dict[str, object]:
    config, preflight, slide, slide_order, transform, overwrite = task
    fit_root = config.output_dir / "fits"
    path = _fit_artifact_path(fit_root, slide)
    provenance = _fit_provenance(config, preflight, slide)
    cached = (
        _load_matching_fit(path, provenance)
        if path.exists() and not overwrite
        else None
    )
    if cached is None:
        analysis = read_analysis_slide(
            config.registration_run,
            slide,
            analysis_mpp=config.analysis_mpp,
        )
        payload = _fit_slide(config, slide, analysis, provenance)
        write_json_atomic(path, payload)
    else:
        payload = cached
    result = dict(payload)
    result["slide_order"] = slide_order
    result["transform"] = transform
    return result


def _map_slide_task(
    task: tuple[
        StainQuantificationConfig,
        StainPreflight,
        StainPreflightSlide,
        int,
        dict[str, object],
        dict[str, object] | None,
        bool,
    ],
) -> tuple[dict[str, object], str]:
    config, preflight, slide, order, benchmark, fit_row, overwrite_maps = task
    if not slide.assay.analysis_included:
        return (
            {
                "id": slide.slide_name,
                "order": order,
                "marker": slide.assay.marker,
                "family": slide.assay.family.value,
                "batch_id": slide.assay.batch_id,
                "analysis_included": False,
                "exclusion_reason": slide.assay.exclusion_reason,
                "quantified": False,
                "map": None,
                "model": None,
                "qc": {"flags": ["analysis_excluded"]},
            },
            "excluded",
        )
    if slide.assay.family is StainFamily.CONTEXT_HE:
        return (
            {
                "id": slide.slide_name,
                "order": order,
                "marker": slide.assay.marker,
                "family": slide.assay.family.value,
                "batch_id": slide.assay.batch_id,
                "analysis_included": True,
                "exclusion_reason": None,
                "quantified": False,
                "map": None,
                "model": None,
                "qc": {"flags": ["context_only"]},
            },
            "context",
        )
    families = benchmark.get("families")
    if not isinstance(families, dict) or not isinstance(fit_row, dict):
        raise ValueError(f"{slide.slide_name}: benchmark fit is missing")
    family = families[slide.assay.family.value]
    if not isinstance(family, dict):
        raise ValueError(f"{slide.slide_name}: family benchmark is invalid")
    selected = str(family["selected_method"])
    candidate = next(
        CandidateFit.from_json_dict(value)
        for value in fit_row["candidates"]
        if value["method"] == selected
    )
    template = np.asarray(family["cohort_vectors"], dtype=float)
    corrected_vectors = shrink_vectors(
        candidate.vectors,
        template,
        config.vector_shrinkage,
    )
    background = BackgroundModel.from_json_dict(fit_row["background"])
    provenance = _map_provenance(
        config,
        preflight,
        slide,
        benchmark,
        selected,
        corrected_vectors,
    )
    maps_root = config.output_dir / "maps"
    adaptive_maps_root = config.output_dir / "adaptive_maps"
    models_root = config.output_dir / "models"
    map_path = maps_root / f"{order:03d}-{_safe_name(slide.slide_name)}.npz"
    model_path = models_root / f"{order:03d}-{_safe_name(slide.slide_name)}.json"
    adaptive_map_path = (
        adaptive_maps_root / f"{order:03d}-{_safe_name(slide.slide_name)}.npz"
    )
    cached = (
        _load_matching_map(map_path, provenance)
        if map_path.exists() and model_path.exists() and not overwrite_maps
        else None
    )
    if cached is None:
        analysis = read_analysis_slide(
            config.registration_run,
            slide,
            analysis_mpp=config.analysis_mpp,
        )
        model, stain_map = _quantify_slide(
            config,
            slide,
            analysis,
            background,
            selected,
            corrected_vectors,
            provenance,
        )
        stain_map.save(map_path)
        write_json_atomic(model_path, model.to_json_dict())
        outcome = "written"
    else:
        stain_map = cached
        model = StainModel.from_json_dict(json.loads(model_path.read_text()))
        outcome = "reused"
    adaptive_map = _counterstain_adaptive_artifact(
        stain_map,
        model,
        adaptive_map_path,
    )
    return (
        _result_slide_row(
            config.output_dir,
            order,
            slide,
            model,
            stain_map,
            map_path,
            model_path,
            adaptive_map=adaptive_map,
            adaptive_map_path=(adaptive_map_path if adaptive_map is not None else None),
        ),
        outcome,
    )


def _stain_process_pool(
    config: StainQuantificationConfig,
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_stain_worker,
        initargs=(config.vips_threads, config.workers),
    )


def _initialize_stain_worker(
    vips_threads: int | None,
    workers: int,
) -> None:
    """Bound native pools inside one spawned slide worker."""

    if vips_threads is not None:
        os.environ["VIPS_CONCURRENCY"] = str(max(1, math.ceil(vips_threads / workers)))
    try:
        from sklearn.decomposition import NMF
        from sklearn.mixture import GaussianMixture
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError(
            "parallel stain processing requires the 'stain' extra"
        ) from exc
    del NMF, GaussianMixture
    global _NUMERIC_THREAD_LIMITER
    _NUMERIC_THREAD_LIMITER = threadpool_limits(limits=1)


def _report_fit_progress(
    progress: Callable[[str], None] | None,
    index: int,
    total: int,
    slide: StainPreflightSlide,
) -> None:
    if progress is not None:
        progress(f"[fit {index}/{total}] {slide.assay.marker}: {slide.slide_name}")


def _report_map_progress(
    progress: Callable[[str], None] | None,
    order: int,
    total: int,
    slide: StainPreflightSlide,
) -> None:
    if progress is not None and slide.assay.family is not StainFamily.CONTEXT_HE:
        progress(f"[map {order}/{total}] {slide.assay.marker}: {slide.slide_name}")


def _fit_slide(
    config: StainQuantificationConfig,
    slide: StainPreflightSlide,
    analysis: AnalysisSlide,
    provenance: dict[str, object],
) -> dict[str, object]:
    background = estimate_background_model(
        analysis.rgb,
        analysis.tissue_mask,
        max_samples=config.white_sample_pixels,
        seed=config.seed,
    )
    corrected = apply_shading_correction(analysis.rgb, background)
    tissue_rgb = _sample_pixels(
        corrected,
        analysis.tissue_mask,
        config.sample_pixels,
        seed=config.seed,
    )
    glass_mask = _glass_mask(analysis.rgb, analysis.tissue_mask)
    glass_rgb = _sample_pixels(
        corrected,
        glass_mask,
        min(config.sample_pixels, config.white_sample_pixels),
        seed=config.seed + 1,
    )
    tissue_od = rgb_to_od(tissue_rgb, background.white_reference)
    glass_od = rgb_to_od(glass_rgb, background.white_reference)
    candidates = [
        fit_candidate(
            tissue_od,
            glass_od,
            slide.assay.family,
            method,
            seed=config.seed,
        )
        for method in config.methods
    ]
    return {
        "schema_version": 1,
        "provenance": provenance,
        "slide_id": slide.slide_name,
        "marker": slide.assay.marker,
        "family": slide.assay.family.value,
        "analysis_shape": list(analysis.rgb.shape[:2]),
        "tissue_pixels": int(np.count_nonzero(analysis.tissue_mask)),
        "glass_pixels": int(np.count_nonzero(glass_mask)),
        "background": background.to_json_dict(),
        "candidates": [candidate.to_json_dict() for candidate in candidates],
    }


def _quantify_slide(
    config: StainQuantificationConfig,
    slide: StainPreflightSlide,
    analysis: AnalysisSlide,
    background: BackgroundModel,
    method: str,
    corrected_vectors: np.ndarray,
    provenance: dict[str, object],
) -> tuple[StainModel, StainMap]:
    raw_vectors = canonical_vectors(slide.assay.family)
    corrected_rgb = apply_shading_correction(analysis.rgb, background)
    tissue_rgb = _sample_pixels(
        analysis.rgb,
        analysis.tissue_mask,
        config.sample_pixels,
        seed=config.seed + 2,
    )
    corrected_tissue_rgb = _sample_pixels(
        corrected_rgb,
        analysis.tissue_mask,
        config.sample_pixels,
        seed=config.seed + 2,
    )
    raw_sample, _ = unmix_od(
        rgb_to_od(tissue_rgb, background.white_reference),
        raw_vectors,
    )
    corrected_sample, _ = unmix_od(
        rgb_to_od(corrected_tissue_rgb, background.white_reference),
        corrected_vectors,
    )
    rank = _spearman(raw_sample[:, 1], corrected_sample[:, 1])
    counterstain_only = counterstain_only_mask(raw_sample)
    raw_counterstain_leakage = float(
        np.quantile(raw_sample[counterstain_only, 1], 0.95)
    )
    corrected_counterstain_leakage = float(
        np.quantile(corrected_sample[counterstain_only, 1], 0.95)
    )
    glass = _glass_mask(analysis.rgb, analysis.tissue_mask)
    raw_glass_rgb = _sample_pixels(
        analysis.rgb,
        glass,
        config.white_sample_pixels,
        seed=config.seed + 3,
    )
    corrected_glass_rgb = _sample_pixels(
        corrected_rgb,
        glass,
        config.white_sample_pixels,
        seed=config.seed + 3,
    )
    raw_glass, _ = unmix_od(
        rgb_to_od(raw_glass_rgb, background.white_reference),
        raw_vectors,
    )
    corrected_glass, _ = unmix_od(
        rgb_to_od(corrected_glass_rgb, background.white_reference),
        corrected_vectors,
    )
    raw_leakage = float(np.quantile(raw_glass[:, 1], 0.95))
    corrected_leakage = float(np.quantile(corrected_glass[:, 1], 0.95))
    accepted = bool(
        rank >= config.correction_rank_guard
        and corrected_leakage <= raw_leakage * 1.05 + 0.002
        and corrected_counterstain_leakage <= raw_counterstain_leakage * 1.05 + 0.002
        and background.after_spatial_cv <= background.before_spatial_cv * 1.05 + 1e-6
    )
    preliminary = StainModel(
        family=slide.assay.family,
        marker=slide.assay.marker,
        method=method,
        background=background,
        raw_vectors=raw_vectors,
        corrected_vectors=corrected_vectors,
        correction_accepted=accepted,
        correction_rank_correlation=rank,
        raw_glass_leakage=raw_leakage,
        corrected_glass_leakage=corrected_leakage,
        raw_counterstain_leakage=raw_counterstain_leakage,
        corrected_counterstain_leakage=corrected_counterstain_leakage,
        content_bbox_native_xywh=slide.content_bbox_xywh,
    )
    concentrations = preliminary.transform_rgb(analysis.rgb)
    selected_physical = (
        concentrations.corrected_target_od
        if preliminary.correction_accepted
        else concentrations.raw_target_od
    )
    tissue = analysis.tissue_mask
    residual = _mask_float(concentrations.reconstruction_residual, tissue)
    residual_scale = max(float(np.quantile(residual[tissue], 0.95)), 0.02)
    confidence = np.zeros(tissue.shape, dtype=np.float32)
    confidence[tissue] = np.exp(-residual[tissue] / residual_scale)
    if config.adaptive_background == "inferred_floor":
        adaptive = infer_adaptive_background(
            selected_physical,
            concentrations.counterstain_od,
            tissue,
            seed=config.seed,
        )
    elif config.adaptive_background == "counterstain_conditioned":
        adaptive = infer_counterstain_conditioned_background(
            selected_physical,
            concentrations.counterstain_od,
            tissue,
            confidence=confidence,
            seed=config.seed,
        )
    else:
        adaptive = None
    threshold, threshold_accepted = _positive_threshold(
        concentrations.corrected_target_od[analysis.tissue_mask],
        seed=config.seed,
    )
    model = StainModel(
        family=preliminary.family,
        marker=preliminary.marker,
        method=preliminary.method,
        background=preliminary.background,
        raw_vectors=preliminary.raw_vectors,
        corrected_vectors=preliminary.corrected_vectors,
        correction_accepted=preliminary.correction_accepted,
        correction_rank_correlation=preliminary.correction_rank_correlation,
        raw_glass_leakage=preliminary.raw_glass_leakage,
        corrected_glass_leakage=preliminary.corrected_glass_leakage,
        raw_counterstain_leakage=preliminary.raw_counterstain_leakage,
        corrected_counterstain_leakage=(preliminary.corrected_counterstain_leakage),
        content_bbox_native_xywh=preliminary.content_bbox_native_xywh,
        positive_threshold_od=threshold,
        threshold_accepted=threshold_accepted,
        adaptive_background=(adaptive.to_json_dict() if adaptive is not None else None),
    )
    raw_target = _mask_float(concentrations.raw_target_od, tissue)
    corrected_target = _mask_float(concentrations.corrected_target_od, tissue)
    counterstain = _mask_float(concentrations.counterstain_od, tissue)
    positive = (
        tissue & (corrected_target >= threshold)
        if threshold_accepted and threshold is not None
        else np.zeros(tissue.shape, dtype=bool)
    )
    stain_map = StainMap(
        slide_id=slide.slide_name,
        raw_target_od=raw_target,
        corrected_target_od=corrected_target,
        counterstain_od=counterstain,
        reconstruction_residual=residual,
        tissue_mask=tissue,
        confidence=confidence,
        positive_mask=positive,
        analysis_mpp=analysis.analysis_mpp,
        content_origin_native_xy=analysis.content_origin_native_xy,
        source_mpp_xy=analysis.source_mpp_xy,
        provenance=provenance,
    )
    return model, stain_map


def _result_slide_row(
    output_dir: Path,
    order: int,
    slide: StainPreflightSlide,
    model: StainModel,
    stain_map: StainMap,
    map_path: Path,
    model_path: Path,
    *,
    adaptive_map: AdaptiveStainMap | None = None,
    adaptive_map_path: Path | None = None,
) -> dict[str, object]:
    selected = (
        stain_map.corrected_target_od
        if model.correction_accepted
        else stain_map.raw_target_od
    )
    values = selected[stain_map.tissue_mask]
    quantile_levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = np.quantile(values, quantile_levels)
    maximum = max(float(quantiles[-1]), 1e-6)
    counts, edges = np.histogram(values, bins=64, range=(0.0, maximum))
    flags = []
    if not model.correction_accepted:
        flags.append("nuisance_correction_rejected")
    if not model.threshold_accepted:
        flags.append("positive_threshold_unstable")
    if model.background.fallback_used:
        flags.append("background_fallback")
    adaptive_payload = model.adaptive_background
    adaptive_method = (
        str(adaptive_payload.get("method", ""))
        if isinstance(adaptive_payload, dict)
        else ""
    )
    if adaptive_method == "counterstain-conditioned-v3":
        adaptive_result: AdaptiveBackgroundResult | CounterstainAdaptiveResult = (
            CounterstainAdaptiveResult.from_json_dict(adaptive_payload)
        )
        if adaptive_result.accepted and adaptive_map is None:
            raise ValueError("accepted counterstain correction has no derived map")
        adaptive_values = (
            np.asarray(adaptive_map.target_od)[stain_map.tissue_mask]
            if adaptive_map is not None
            else values
        )
    elif adaptive_payload is not None:
        adaptive_result = AdaptiveBackgroundResult.from_json_dict(adaptive_payload)
        adaptive_values = apply_adaptive_background(
            selected,
            stain_map.tissue_mask,
            adaptive_result,
        )[stain_map.tissue_mask]
    else:
        adaptive_result = None
        adaptive_values = values
    adaptive_quantiles = np.quantile(adaptive_values, quantile_levels)
    if adaptive_result is not None and not adaptive_result.accepted:
        flags.append("adaptive_background_rejected")
    row: dict[str, object] = {
        "id": slide.slide_name,
        "order": order,
        "marker": slide.assay.marker,
        "family": slide.assay.family.value,
        "batch_id": slide.assay.batch_id,
        "analysis_included": True,
        "exclusion_reason": None,
        "quantified": True,
        "map": map_path.relative_to(output_dir).as_posix(),
        "model": model_path.relative_to(output_dir).as_posix(),
        "map_fingerprint": stain_map.content_fingerprint,
        "quantiles": {
            str(level): float(value)
            for level, value in zip(quantile_levels, quantiles, strict=True)
        },
        "adaptive_quantiles": {
            str(level): float(value)
            for level, value in zip(quantile_levels, adaptive_quantiles, strict=True)
        },
        "histogram": {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
        },
        "positive_fraction": float(
            np.count_nonzero(stain_map.positive_mask)
            / max(np.count_nonzero(stain_map.tissue_mask), 1)
        ),
        "qc": {
            "correction_accepted": model.correction_accepted,
            "rank_correlation": model.correction_rank_correlation,
            "raw_glass_leakage": model.raw_glass_leakage,
            "corrected_glass_leakage": model.corrected_glass_leakage,
            "raw_counterstain_leakage": model.raw_counterstain_leakage,
            "corrected_counterstain_leakage": (model.corrected_counterstain_leakage),
            "background_spatial_cv_before": model.background.before_spatial_cv,
            "background_spatial_cv_after": model.background.after_spatial_cv,
            "threshold_accepted": model.threshold_accepted,
            "positive_threshold_od": model.positive_threshold_od,
            "adaptive_background": adaptive_payload,
            "median_reconstruction_residual": float(
                np.median(stain_map.reconstruction_residual[stain_map.tissue_mask])
            ),
            "flags": flags,
        },
    }
    if adaptive_map is not None and adaptive_map_path is not None:
        row["adaptive_map"] = adaptive_map_path.relative_to(output_dir).as_posix()
        row["adaptive_map_fingerprint"] = adaptive_map.content_fingerprint
    return row


def _counterstain_adaptive_artifact(
    stain_map: StainMap,
    model: StainModel,
    path: Path,
) -> AdaptiveStainMap | None:
    payload = model.adaptive_background
    if not isinstance(payload, dict) or payload.get("method") != (
        "counterstain-conditioned-v3"
    ):
        return None
    result = CounterstainAdaptiveResult.from_json_dict(payload)
    if not result.accepted:
        return None
    if path.is_file():
        existing = AdaptiveStainMap.load(path)
        if (
            existing.slide_id == stain_map.slide_id
            and existing.source_content_fingerprint == stain_map.content_fingerprint
            and existing.diagnostics == payload
        ):
            return existing
    selected = (
        stain_map.corrected_target_od
        if model.correction_accepted
        else stain_map.raw_target_od
    )
    target = apply_counterstain_conditioned_background(
        selected,
        stain_map.counterstain_od,
        stain_map.tissue_mask,
        result,
    )
    derived = AdaptiveStainMap(
        slide_id=stain_map.slide_id,
        target_od=target,
        tissue_mask=stain_map.tissue_mask,
        analysis_mpp=stain_map.analysis_mpp,
        content_origin_native_xy=stain_map.content_origin_native_xy,
        source_mpp_xy=stain_map.source_mpp_xy,
        source_content_fingerprint=str(stain_map.content_fingerprint),
        method=result.method,
        diagnostics=payload,
    )
    derived.save(path)
    return derived


def _positive_threshold(
    values: np.ndarray,
    *,
    seed: int,
) -> tuple[float | None, bool]:
    signal = np.asarray(values, dtype=np.float64)
    signal = signal[np.isfinite(signal) & (signal >= 0)]
    if len(signal) < 256 or float(np.ptp(signal)) <= 1e-8:
        return None, False
    rng = np.random.default_rng(seed)
    if len(signal) > 200_000:
        signal = signal[rng.choice(len(signal), size=200_000, replace=False)]
    otsu = _otsu_threshold(signal)
    lower = signal[signal <= np.quantile(signal, 0.60)]
    median = float(np.median(lower))
    mad = float(np.median(np.abs(lower - median)))
    robust = median + 3.0 * 1.4826 * mad
    mixture, separation = _mixture_threshold(signal, seed=seed)
    candidates = np.asarray([otsu, robust, mixture], dtype=float)
    threshold = float(np.median(candidates))
    spread = float(np.max(candidates) - np.min(candidates))
    accepted = bool(
        separation >= 1.0
        and spread <= max(0.08, 0.75 * max(threshold, 1e-6))
        and 0 < threshold < float(np.quantile(signal, 0.995))
    )
    return threshold, accepted


def _otsu_threshold(values: np.ndarray) -> float:
    low, high = np.quantile(values, [0.001, 0.999])
    if high <= low:
        return float(low)
    counts, edges = np.histogram(values, bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weights_left = np.cumsum(counts)
    weights_right = np.cumsum(counts[::-1])[::-1]
    means_left = np.cumsum(counts * centers) / np.maximum(weights_left, 1)
    means_right = np.cumsum((counts * centers)[::-1])[::-1] / np.maximum(
        weights_right, 1
    )
    between = (
        weights_left[:-1] * weights_right[1:] * (means_left[:-1] - means_right[1:]) ** 2
    )
    return float(centers[int(np.argmax(between))])


def _mixture_threshold(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError("stain thresholding requires the 'stain' extra") from exc
    transformed = np.log1p(values)[:, None]
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=seed,
        reg_covar=1e-6,
        n_init=3,
    ).fit(transformed)
    order = np.argsort(model.means_[:, 0])
    low, high = model.means_[order, 0]
    variances = model.covariances_[order, 0, 0]
    threshold = float(np.expm1((low + high) / 2))
    separation = float(
        abs(high - low) / max(math.sqrt(float(np.mean(variances))), 1e-8)
    )
    return threshold, separation


def _sample_pixels(
    rgb: np.ndarray,
    mask: np.ndarray,
    maximum: int,
    *,
    seed: int,
) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("pixel sampling mask is empty")
    if len(coordinates) > maximum:
        rng = np.random.default_rng(seed)
        coordinates = coordinates[
            rng.choice(len(coordinates), size=maximum, replace=False)
        ]
    return rgb[coordinates[:, 0], coordinates[:, 1]]


def _glass_mask(rgb: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    brightness = np.asarray(rgb, dtype=np.uint8).mean(axis=2)
    chroma = np.asarray(rgb, dtype=np.uint8).max(axis=2) - np.asarray(
        rgb, dtype=np.uint8
    ).min(axis=2)
    glass = (~tissue) & (brightness >= 150) & (chroma <= 100)
    if np.count_nonzero(glass) < 64:
        glass = ~tissue
    if np.count_nonzero(glass) < 64:
        cutoff = float(np.quantile(brightness, 0.95))
        glass = brightness >= cutoff
    return glass


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _rank_correlation(left, right)


def _mask_float(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    output[~mask] = 0.0
    return output


def _fit_provenance(
    config: StainQuantificationConfig,
    preflight: StainPreflight,
    slide: StainPreflightSlide,
) -> dict[str, object]:
    return {
        "algorithm_version": _FIT_ALGORITHM_VERSION,
        "preflight_fingerprint": preflight.fingerprint,
        "slide_id": slide.slide_name,
        "source_sha256": slide.source_sha256,
        "mask_sha256": slide.mask_sha256,
        "family": slide.assay.family.value,
        "analysis_mpp": config.analysis_mpp,
        "methods": list(config.methods),
        "sample_pixels": config.sample_pixels,
        "white_sample_pixels": config.white_sample_pixels,
        "seed": config.seed,
    }


def _map_provenance(
    config: StainQuantificationConfig,
    preflight: StainPreflight,
    slide: StainPreflightSlide,
    benchmark: dict[str, object],
    selected_method: str,
    corrected_vectors: np.ndarray,
) -> dict[str, object]:
    return {
        "algorithm_version": _MAP_ALGORITHM_VERSION,
        "preflight_fingerprint": preflight.fingerprint,
        "benchmark_fingerprint": benchmark["fingerprint"],
        "slide_id": slide.slide_name,
        "source_sha256": slide.source_sha256,
        "mask_sha256": slide.mask_sha256,
        "transform_sha256": slide.transform_sha256,
        "analysis_mpp": config.analysis_mpp,
        "selected_method": selected_method,
        "corrected_vectors": np.asarray(corrected_vectors).tolist(),
        "vector_shrinkage": config.vector_shrinkage,
        "correction_rank_guard": config.correction_rank_guard,
        "adaptive_background": config.adaptive_background,
    }


def _benchmark_request(config: StainQuantificationConfig) -> dict[str, object]:
    return {
        "analysis_mpp": config.analysis_mpp,
        "methods": list(config.methods),
        "sample_pixels": config.sample_pixels,
        "white_sample_pixels": config.white_sample_pixels,
        "vector_shrinkage": config.vector_shrinkage,
        "adaptive_background": config.adaptive_background,
        "seed": config.seed,
    }


def _load_matching_fit(
    path: Path,
    provenance: dict[str, object],
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") == 1
            and _equivalent_fit_provenance(payload.get("provenance"), provenance)
            and isinstance(payload.get("candidates"), list)
        ):
            for candidate in payload["candidates"]:
                CandidateFit.from_json_dict(candidate)
            BackgroundModel.from_json_dict(payload["background"])
            return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _equivalent_fit_provenance(
    observed: object,
    expected: dict[str, object],
) -> bool:
    """Permit fit reuse across campaigns only when exact fit inputs match."""

    if not isinstance(observed, dict):
        return False
    ignored = {"preflight_fingerprint"}
    return {key: value for key, value in observed.items() if key not in ignored} == {
        key: value for key, value in expected.items() if key not in ignored
    }


def _load_matching_map(
    path: Path,
    provenance: dict[str, object],
) -> StainMap | None:
    try:
        artifact = StainMap.load(path)
    except (OSError, ValueError):
        return None
    return artifact if artifact.provenance == provenance else None


def _fit_artifact_path(root: Path, slide: StainPreflightSlide) -> Path:
    return root / f"{_safe_name(slide.slide_name)}.json"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "section"


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
