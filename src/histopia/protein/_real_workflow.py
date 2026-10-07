"""End-to-end real-data protein demo workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def run_real_protein_workflow(
    config,
    *,
    registration_run: Path,
    cell_run: Path,
    stain_run: Path,
    semantic_run: Path,
    model_cache_dir: Path,
    mouse_id: str,
    device: str = "auto",
    batch_size: int = 64,
    max_training_cells_per_section: int = 12_000,
    feature_sections: tuple[str, ...] | None = None,
    features_only: bool = False,
    source_cache_dir: Path | None = None,
) -> Path:
    """Build neutral UNI2-h features, benchmark repeated stains, and predict."""

    from histopia.protein._cell_features import (
        CellTokenFeatures,
        VipsPatchReader,
        extract_cell_token_features,
        label_centroids_from_tiff,
        neutralize_hdab_morphology,
        validate_cell_token_feature_binding,
    )
    from histopia.protein._cli import (
        _benchmark,
        _fit_model,
        _region_centered_blend,
        _semantic_baseline,
    )
    from histopia.protein._manifest import write_protein_result
    from histopia.protein._real_data import (
        aggregate_map_to_sampled_labels,
        compartment_sampled_labels,
        map_native_to_reference,
        sampled_label_expected_analysis_pixels,
        sampled_label_geometry,
        target_sections,
        validate_real_run_bindings,
        validated_adaptive_target_measurement,
    )
    from histopia.protein._result import CellExpressionTable, ProteinPredictions
    from histopia.protein._selection import evaluate_protein_promotion
    from histopia.protein._transfer import morphospatial_features
    from histopia.semantic._uni2h import Uni2hEncoder
    from histopia.stain._artifacts import AdaptiveStainMap, StainMap

    bindings = validate_real_run_bindings(
        registration_run,
        cell_run,
        stain_run,
        semantic_run,
        verify_artifacts=not features_only,
    )
    sections = (
        ()
        if features_only
        else target_sections(
            bindings.stain,
            config.target.target_id,
            require_threshold=config.target.binary_enabled,
            minimum_sections=1,
        )
    )
    output = config.output_dir
    neutral_dir = output / "cell_token_features"
    geometry_dir = output / "native_geometry_v2"
    prediction_dir = output / "predictions"
    for directory in (neutral_dir, geometry_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)
    selected_k = int(bindings.semantic["selected_k"])
    region_count = selected_k
    encoder = None
    stain_by_id = {row["id"]: row for row in bindings.stain["slides"]}
    semantic_by_id = {row["id"]: row for row in bindings.semantic["slides"]}
    cell_row_by_section = {str(row["section"]): row for row in bindings.cells["slides"]}
    cell_artifacts = bindings.cells.get("artifacts")
    if not isinstance(cell_artifacts, dict):
        raise ValueError("cell result artifact manifest is missing")
    validated_cell_labels: set[str] = set()

    def validate_cell_label(cell_row: dict[str, object]) -> Path:
        """Validate one immutable label artifact immediately before use."""

        from histopia.cells._result import validate_cell_artifact

        relative = str(cell_row["labels"])
        if relative not in validated_cell_labels:
            expected = cell_artifacts.get(relative)
            if not isinstance(expected, str):
                raise ValueError("cell label artifact binding is missing")
            validate_cell_artifact(Path(cell_run) / relative, expected)
            validated_cell_labels.add(relative)
        return Path(cell_run) / relative

    requested_features = (
        set(cell_row_by_section) if feature_sections is None else set(feature_sections)
    )
    unknown_features = requested_features - set(cell_row_by_section)
    if unknown_features:
        raise ValueError(
            "unknown feature sections: " + ", ".join(sorted(unknown_features))
        )

    preflight_path = Path(cell_run) / str(bindings.cells["preflight"])
    cell_preflight = json.loads(preflight_path.read_text())
    preflight_by_slide = {
        str(row["slide_name"]): row for row in cell_preflight["slides"]
    }

    # Native 0.5 µm/px, target-free spatial UNI2-h tokens for every cell.
    neutral_paths: dict[str, Path] = {}
    for cell_row in bindings.cells["slides"]:
        section = str(cell_row["section"])
        if section not in requested_features:
            continue
        path = neutral_dir / f"{section}.npz"
        neutral_paths[section] = path
        preflight_row = preflight_by_slide[str(cell_row["slide"])]
        if path.is_file():
            validate_cell_token_feature_binding(
                path,
                slide_id=str(cell_row["slide"]),
                cell_result_fingerprint=str(bindings.cells["fingerprint"]),
                source_identity=str(preflight_row["source_identity"]),
                mask_sha256=str(preflight_row["mask_sha256"]),
            )
            continue
        if encoder is None:
            encoder = Uni2hEncoder.from_cache(
                model_cache_dir, device=device, local_only=True
            )
        source_path = Path(str(preflight_row["source_path"]))
        if source_cache_dir is not None:
            cached_source = Path(source_cache_dir) / source_path.name
            if not cached_source.is_file():
                raise ValueError(f"cached source slide is missing: {source_path.name}")
            from histopia.cells._preflight import _file_identity

            if _file_identity(cached_source) != preflight_row["source_identity"]:
                raise ValueError(
                    f"cached source slide identity is stale: {source_path.name}"
                )
            source_path = cached_source
        label_ids, label_xy, _areas = label_centroids_from_tiff(
            validate_cell_label(cell_row)
        )
        bbox = tuple(int(value) for value in preflight_row["content_bbox_xywh"])
        from PIL import Image

        tissue_mask = (
            np.asarray(Image.open(preflight_row["mask_path"]).convert("L")) > 0
        )
        x0, y0, width, height = bbox
        native_to_mask = np.asarray(
            [
                [tissue_mask.shape[1] / width, 0, -x0 * tissue_mask.shape[1] / width],
                [0, tissue_mask.shape[0] / height, -y0 * tissue_mask.shape[0] / height],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        extract_cell_token_features(
            slide_id=str(cell_row["slide"]),
            content_bbox_native_xywh=bbox,
            source_mpp_xy=tuple(float(v) for v in preflight_row["mpp_xy"]),
            tissue_mask=tissue_mask,
            native_to_mask=native_to_mask,
            label_ids=label_ids,
            cell_native_xy=label_xy + np.asarray([x0, y0]),
            reader=VipsPatchReader(source_path),
            neutralizer=neutralize_hdab_morphology,
            encoder=encoder,
            batch_size=batch_size,
            provenance={
                "source_identity": preflight_row["source_identity"],
                "mask_sha256": preflight_row["mask_sha256"],
                "cell_result_fingerprint": bindings.cells["fingerprint"],
                "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
            },
        ).save(path)

    if features_only:
        return neutral_dir

    registration_slides = bindings.registration["slides"]
    reference_id = bindings.registration["reference_slide"]
    registration_by_id = {row.get("path", row.get("aligned_to")): row for row in []}
    del registration_by_id
    registration_by_slide = {
        Path(str(row["path"])).name: row for row in registration_slides
    }
    reference_slide = next(
        row for row in registration_slides if str(row["path"]) == reference_id
    )

    geometry: dict[str, dict[str, np.ndarray]] = {}
    measurements: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    measurement_views: set[str] = set()
    for cell_row in bindings.cells["slides"]:
        section = str(cell_row["section"])
        cache = geometry_dir / f"{section}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                present, native_xy, reference_xy, counts = (
                    data["present"],
                    data["native_xy"],
                    data["reference_um_xy"],
                    data["counts"],
                )
            sampled = None
        else:
            token_artifact = CellTokenFeatures.load(neutral_paths[section])
            present = token_artifact.label_ids
            native_xy = token_artifact.native_xy
            area_ids, _area_xy, areas = label_centroids_from_tiff(
                validate_cell_label(cell_row)
            )
            if not np.array_equal(area_ids, present):
                raise ValueError("native cell geometry differs from token artifact")
            counts = areas
            registration_row = registration_by_slide[Path(cell_row["slide"]).name]
            reference_xy = map_native_to_reference(
                native_xy, registration_row, reference_slide
            )
            np.savez_compressed(
                cache,
                present=present,
                native_xy=native_xy,
                reference_um_xy=reference_xy,
                counts=counts,
            )
            sampled = None
        geometry[section] = {
            "present": present,
            "native_xy": native_xy,
            "reference_um_xy": reference_xy,
            "counts": counts,
        }
        if section in sections:
            if sampled is None:
                sampled, _p, _xy, _counts = sampled_label_geometry(
                    validate_cell_label(cell_row),
                    subifd=2,
                    cell_count=int(cell_row["cell_count"]),
                )
            stain_row = stain_by_id[cell_row["slide"]]
            stain_map = StainMap.load(Path(stain_run) / str(stain_row["map"]))
            adaptive_map = (
                AdaptiveStainMap.load(Path(stain_run) / str(stain_row["adaptive_map"]))
                if isinstance(stain_row.get("adaptive_map"), str)
                else None
            )
            measurement = validated_adaptive_target_measurement(
                stain_map,
                stain_row,
                adaptive_stain_map=adaptive_map,
            )
            measurement_views.add(measurement.measurement_view)
            sampled = compartment_sampled_labels(
                sampled,
                config.target.compartment,
                source_mpp_xy=stain_map.source_mpp_xy,
                pyramid_scale=4.0,
            )
            mean, effective, fraction = aggregate_map_to_sampled_labels(
                sampled,
                measurement.target_od,
                measurement.tissue_mask,
                measurement.positive_mask,
                source_mpp_xy=stain_map.source_mpp_xy,
                analysis_mpp=float(stain_map.analysis_mpp),
                content_origin_native_xy=stain_map.content_origin_native_xy,
                pyramid_scale=4.0,
                cell_count=int(np.max(present, initial=0)),
                statistic=config.target.measurement_statistic,
            )
            expected = sampled_label_expected_analysis_pixels(
                sampled,
                cell_count=int(np.max(present, initial=0)),
                source_mpp_xy=stain_map.source_mpp_xy,
                pyramid_scale=4.0,
                analysis_mpp=float(stain_map.analysis_mpp),
            )
            coverage = np.divide(
                effective,
                expected,
                out=np.zeros_like(effective),
                where=expected > 0,
            )
            measurements[section] = (mean, effective, fraction, coverage)

    training_parts = []
    training_sample_complete = True
    rng = np.random.default_rng(config.seed)
    for section in sections:
        values = geometry[section]
        token_artifact = CellTokenFeatures.load(neutral_paths[section])
        if not np.array_equal(token_artifact.label_ids, values["present"]):
            raise ValueError("cell token labels differ from native geometry")
        feature = _standardize_patch_features(token_artifact.features)
        supported = token_artifact.supported
        regions = _semantic_regions_for_cells(
            Path(semantic_run),
            semantic_by_id,
            cell_row_by_section=cell_row_by_section,
            section=section,
            cell_native_xy=values["native_xy"],
            region_count=region_count,
        )
        shape, neighborhood = _cell_context_features(values, config.section_spacing_um)
        xyz = np.column_stack(
            (
                values["reference_um_xy"],
                np.full(
                    len(values["present"]), int(section) * config.section_spacing_um
                ),
            )
        )
        feature = morphospatial_features(
            feature,
            xyz,
            shape_features=shape,
            neighborhood_features=neighborhood,
            coordinate_scale_um=config.coordinate_scale_um,
        )
        mean, effective, fraction, coverage = measurements[section]
        ids = values["present"].astype(np.int64)
        eligible = np.flatnonzero(
            supported
            & (effective[ids] >= config.minimum_effective_pixels)
            & (coverage[ids] >= config.minimum_coverage)
        )
        if len(eligible) > max_training_cells_per_section:
            training_sample_complete = False
            eligible = np.sort(
                rng.choice(eligible, max_training_cells_per_section, replace=False)
            )
        binary = np.full(len(eligible), -1, np.int8)
        stain_qc = dict(
            stain_by_id[str(cell_row_by_section[section]["slide"])].get("qc", {})
        )
        if config.target.binary_enabled and stain_qc.get("threshold_accepted") is True:
            binary[fraction[ids[eligible]] <= config.negative_area_fraction] = 0
            binary[fraction[ids[eligible]] >= config.positive_area_fraction] = 1
        training_parts.append((section, eligible, feature, supported, regions, binary))

    raw_training_od = np.concatenate(
        [measurements[s][0][geometry[s]["present"][i]] for s, i, *_ in training_parts]
    ).astype(np.float32)
    training_sections = np.concatenate(
        [np.full(len(i), s) for s, i, *_ in training_parts]
    )
    training_reference_xy = np.concatenate(
        [geometry[s]["reference_um_xy"][i] for s, i, *_ in training_parts]
    )
    training_regions = np.concatenate(
        [regions[i] for _s, i, _f, _sup, regions, _b in training_parts]
    )
    training_support = np.concatenate(
        [support[i] for _s, i, _f, support, _regions, _binary in training_parts]
    )
    # Adaptive correction is the validated physical measurement.  Protein
    # outcomes must not be rescaled again across sections: doing so changes the
    # ground truth and makes the displayed OD differ from the fitted outcome.
    calibration = None
    calibrated_training_od = raw_training_od
    if len(measurement_views) != 1:
        raise ValueError("protein target measurement views differ between sections")
    measurement_view = measurement_views.pop()

    table = CellExpressionTable(
        target_id=config.target.target_id,
        label_ids=np.concatenate(
            [geometry[s]["present"][i] for s, i, *_ in training_parts]
        ),
        section_ids=training_sections,
        mouse_ids=np.concatenate(
            [np.full(len(i), mouse_id) for _s, i, *_ in training_parts]
        ),
        native_xy=np.concatenate(
            [geometry[s]["native_xy"][i] for s, i, *_ in training_parts]
        ),
        reference_um_xy=training_reference_xy,
        features=np.concatenate([feature[i] for _s, i, feature, *_ in training_parts]),
        measured_od=calibrated_training_od,
        binary_label=np.concatenate([binary for *_head, binary in training_parts]),
        measurement_coverage=np.concatenate(
            [
                np.clip(
                    measurements[s][3][geometry[s]["present"][i]],
                    0,
                    1,
                )
                for s, i, *_ in training_parts
            ]
        ),
        semantic_region=training_regions,
        semantic_support=training_support,
        provenance={
            "registration_result_sha256": bindings.cells["registration_result_sha256"],
            "cell_result_fingerprint": bindings.cells["fingerprint"],
            "stain_result_fingerprint": bindings.stain["fingerprint"],
            "semantic_result_fingerprint": bindings.semantic["fingerprint"],
            "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
            "measurement_view": measurement_view,
            "measurement_compartment": config.target.compartment,
            "measurement_statistic": config.target.measurement_statistic,
            "coverage_definition": "tissue_supported_compartment_area_fraction",
            "target_sections": list(sections),
            "od_calibration": {
                "method": "identity",
                "reason": "adaptive_target_od_is_the_validated_measurement",
            },
        },
    )
    table_path = table.save(output / "cell_expression_table.npz")
    benchmark_metrics = _benchmark(config, table, group="section")
    benchmark_winner = str(benchmark_metrics["selected_candidate"])
    winner_metrics = dict(benchmark_metrics["candidates"][benchmark_winner])
    promotion = evaluate_protein_promotion(winner_metrics)
    train = np.arange(len(table.label_ids), dtype=np.int64)
    fit_attention = bool(
        benchmark_winner == "cross_attention" and training_sample_complete
    )
    deploy_attention = bool(fit_attention and promotion.accepted)
    oof_by_section: dict[str, dict[int, float]] = {}
    if fit_attention:
        from histopia.protein._attention import fit_portable_cross_attention
        from histopia.protein._deep import out_of_fold_deep_predictions

        attention_model = fit_portable_cross_attention(
            table.features,
            table.measured_od,
            train,
            seed=config.seed,
        )
        attention_model = replace(
            attention_model,
            provenance={
                **attention_model.provenance,
                "table_fingerprint": table.fingerprint,
                "semantic_postprocessor": "region-median-centered-blend-v1",
                "semantic_blend_weight": config.semantic_blend_weight,
                "od_calibration_fingerprint": (
                    calibration.fingerprint if calibration is not None else None
                ),
            },
            fingerprint=None,
        )
        model_path = attention_model.save(output / "protein_attention_model.npz")
        model_fingerprint = str(attention_model.fingerprint)
        model_target_id = config.target.target_id
        model_assay_domain = config.target.assay_domain
        metrics = {**benchmark_metrics, **winner_metrics}
        metrics["deployed_candidate"] = "cross_attention"
        xyz = np.column_stack(
            (
                table.reference_um_xy,
                np.asarray(table.section_ids, dtype=np.float64)
                * config.section_spacing_um,
            )
        )
        oof = out_of_fold_deep_predictions(
            "cross_attention",
            table.features,
            table.measured_od,
            table.section_ids,
            reference_um_xyz=xyz,
            seed=config.seed,
        )
        for section in sections:
            selected = np.asarray(table.section_ids) == section
            test = np.flatnonzero(selected)
            train_fold = np.flatnonzero(~selected)
            semantic_oof, _probability = _semantic_baseline(table, train_fold, test)
            blended_oof = _region_centered_blend(
                oof[selected],
                semantic_oof,
                np.asarray(table.semantic_region)[selected],
                weight=config.semantic_blend_weight,
            )
            oof_by_section[section] = dict(
                zip(
                    np.asarray(table.label_ids)[selected].astype(int).tolist(),
                    blended_oof.astype(float).tolist(),
                    strict=True,
                )
            )
    else:
        model = _fit_model(config, table, train)
        model_path = model.save(output / "protein_model.npz")
        model_fingerprint = str(model.fingerprint)
        model_target_id = model.target_id
        model_assay_domain = model.assay_domain
        metrics = _deployed_candidate_metrics(
            benchmark_metrics,
            deployed_candidate="hurdle_mlp",
        )

    rows = []
    for cell_row in bindings.cells["slides"]:
        section = str(cell_row["section"])
        values = geometry[section]
        token_artifact = CellTokenFeatures.load(neutral_paths[section])
        features = _standardize_patch_features(token_artifact.features)
        supported = token_artifact.supported
        regions = _semantic_regions_for_cells(
            Path(semantic_run),
            semantic_by_id,
            cell_row_by_section=cell_row_by_section,
            section=section,
            cell_native_xy=values["native_xy"],
            region_count=region_count,
        )
        held_out = np.asarray(table.section_ids) != section
        semantic_reference = _semantic_reference_for_regions(
            table,
            regions,
            train=(np.flatnonzero(held_out) if section in sections else None),
        )
        shape, neighborhood = _cell_context_features(values, config.section_spacing_um)
        xyz = np.column_stack(
            (
                values["reference_um_xy"],
                np.full(
                    len(values["present"]), int(section) * config.section_spacing_um
                ),
            )
        )
        features = morphospatial_features(
            features,
            xyz,
            shape_features=shape,
            neighborhood_features=neighborhood,
            coordinate_scale_um=config.coordinate_scale_um,
        )
        present = values["present"].astype(np.int64)
        count = len(present)
        ids = present.astype(np.uint32)
        probability = np.full(count, np.nan, np.float32)
        relative = np.zeros(count, np.float32)
        od = np.zeros(count, np.float32)
        uncertainty = np.ones(count, np.float32)
        for start in range(0, len(present), 20_000):
            stop = min(start + 20_000, len(present))
            if fit_attention:
                if section in oof_by_section:
                    pred = semantic_reference[start:stop]
                    unc = np.ones(stop - start, dtype=np.float32)
                else:
                    pred, unc = attention_model.predict_accelerated(
                        features[start:stop], device=device
                    )
                prob = None
                reference = np.sort(np.asarray(table.measured_od, dtype=np.float32))
                rel = np.searchsorted(reference, pred, side="right") / max(
                    len(reference), 1
                )
            else:
                prob, rel, pred, unc = model.predict(features[start:stop])
            target = np.arange(start, stop)
            if prob is not None:
                probability[target] = prob
            relative[target], od[target], uncertainty[target] = rel, pred, unc
        if fit_attention and section not in oof_by_section:
            od[:] = _region_centered_blend(
                od,
                semantic_reference,
                regions,
                weight=config.semantic_blend_weight,
            )
        if section in oof_by_section:
            label_to_index = {int(label): index for index, label in enumerate(ids)}
            for label, value in oof_by_section[section].items():
                od[label_to_index[label]] = value
        if fit_attention:
            reference = np.sort(np.asarray(table.measured_od, dtype=np.float32))
            relative[:] = np.searchsorted(reference, od, side="right") / max(
                len(reference), 1
            )
        measured = np.full(count, np.nan, np.float32)
        if section in measurements:
            raw_measured = measurements[section][0][present]
            accepted_measurement = (
                measurements[section][1][present] >= config.minimum_effective_pixels
            ) & (measurements[section][3][present] >= config.minimum_coverage)
            accepted_values = (
                calibration.for_section(section).apply(raw_measured)
                if calibration is not None
                else raw_measured
            )
            measured[accepted_measurement] = accepted_values[accepted_measurement]
        support_full = supported
        prediction = ProteinPredictions(
            target_id=model_target_id,
            model_fingerprint=model_fingerprint,
            label_ids=ids,
            section_ids=np.full(count, section),
            expression_probability=probability,
            relative_expression=relative,
            predicted_od_reference=od,
            measured_od=measured,
            uncertainty=uncertainty,
            supported=support_full,
            provenance={
                "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
                "feature_source_section": section,
                "target_od_scope": "reference-scale prediction; not a new measurement",
                "measurement_scale": "registered_repeated_stain_reference_od",
                "od_calibration_fingerprint": (
                    calibration.fingerprint if calibration is not None else None
                ),
                "evaluation_role": (
                    "out-of-fold" if section in oof_by_section else "transfer"
                ),
                "semantic_blend_weight": config.semantic_blend_weight,
                "table_fingerprint": table.fingerprint,
            },
        )
        path = prediction.save(prediction_dir / f"{section}.npz")
        rows.append(
            {
                "section": section,
                "predictions": path.relative_to(output).as_posix(),
                "prediction_fingerprint": prediction.fingerprint,
                "cells": int(cell_row["cell_count"]),
                "measured_cells": int(np.isfinite(measured).sum()),
                "evaluation_role": (
                    "out-of-fold"
                    if section in oof_by_section
                    else (
                        "fit-visible"
                        if section in measurements
                        else "unmeasured-transfer"
                    )
                ),
            }
        )
    version_slug = (
        "counterstain-v3" if measurement_view.endswith("-v3") else "adaptive-v1"
    )
    return write_protein_result(
        output,
        {
            "schema_version": 3,
            "model_id": (
                f"{model_target_id}-{metrics['deployed_candidate']}-"
                f"{mouse_id}-{version_slug}"
            ),
            "model_label": (
                f"{model_target_id.upper()} · "
                f"{str(metrics['deployed_candidate']).replace('_', ' ')} · "
                f"trained {mouse_id}"
            ),
            "architecture": metrics["deployed_candidate"],
            "model_version": (
                "counterstain-conditioned-v3"
                if measurement_view.endswith("-v3")
                else "adaptive-v1"
            ),
            "training_cohorts": [mouse_id],
            "measurement_view": measurement_view,
            "measurement_compartment": config.target.compartment,
            "measurement_statistic": config.target.measurement_statistic,
            "target_id": model_target_id,
            "assay_domain": model_assay_domain,
            "model": model_path.name,
            "slides": rows,
            "model_fingerprint": model_fingerprint,
            "table_fingerprint": table.fingerprint,
            "training_table": table_path.name,
            "registration_result_sha256": bindings.cells["registration_result_sha256"],
            "cell_result_fingerprint": bindings.cells["fingerprint"],
            "metrics": metrics,
            "binary_enabled": config.target.binary_enabled,
            "status": "exploratory",
            "model_candidates": list(config.model_candidates),
            "candidate_promotion": {
                "candidate": benchmark_winner,
                "accepted": promotion.accepted,
                "reasons": list(promotion.reasons),
                "full_training_cells": training_sample_complete,
                "deployed": deploy_attention,
                "policy": "held-out-accuracy-baseline-parity-and-od-bias-v1",
            },
            "prediction_scope": "assay-relative percentile; not measured OD",
            "feature_resolution": {
                "source_mpp": 0.5,
                "crop_pixels": 224,
                "crop_stride_pixels": 112,
                "token_lattice": [16, 16],
                "token_spacing_um": 7.0,
                "cell_assignment": "bilinear_hann_blended_spatial_tokens",
            },
            "dense_display_mpp": 4.0,
            "semantic_role": "soft_region_prior_not_cell_type_bound",
            "semantic_blend_weight": config.semantic_blend_weight,
            "od_calibration": None,
        },
    )


def _deployed_candidate_metrics(
    benchmark: dict[str, object],
    *,
    deployed_candidate: str,
) -> dict[str, object]:
    """Report metrics for the model that produced the portable predictions."""

    candidates = benchmark.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("protein benchmark candidate metrics are missing")
    deployed = candidates.get(deployed_candidate)
    if not isinstance(deployed, dict) or deployed.get("available") is not True:
        raise ValueError("deployed protein candidate has no valid benchmark")
    benchmark_winner = benchmark.get("selected_candidate")
    metrics = {**benchmark, **deployed}
    metrics["benchmark_selected_candidate"] = benchmark_winner
    metrics["benchmark_selection_metric"] = benchmark.get("selection_metric")
    metrics["selected_candidate"] = deployed_candidate
    metrics["deployed_candidate"] = deployed_candidate
    metrics["selection_metric"] = "portable_deployed_candidate"
    if benchmark_winner != deployed_candidate:
        metrics["deployment_note"] = (
            f"{benchmark_winner} won candidate benchmarking but is not the "
            "portable deployed artifact; displayed predictions and top-level "
            f"metrics use {deployed_candidate}"
        )
    return metrics


def _section_feature_path(root: Path, section: str) -> Path:
    matches = tuple((root / "features").glob(f"{section}-*.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one semantic feature artifact for section {section}"
        )
    return matches[0]


def _semantic_regions_for_cells(
    semantic_run: Path,
    semantic_by_id: dict[str, dict[str, object]],
    cell_row_by_section: dict[str, dict[str, object]],
    *,
    section: str,
    cell_native_xy: np.ndarray,
    region_count: int,
) -> np.ndarray:
    """Assign the coarse semantic prior as a gate, never as a hard cell type."""

    from histopia.protein._real_data import pool_neutral_features
    from histopia.semantic._features import PatchFeatures

    cell_row = cell_row_by_section[section]
    patch = PatchFeatures.load(_section_feature_path(semantic_run, section))
    semantic_row = semantic_by_id[str(cell_row["slide"])]
    label_path = semantic_run / str(semantic_row["labels"][str(region_count)])
    with np.load(label_path, allow_pickle=False) as labels:
        patch_regions = np.asarray(labels["joint_labels"], dtype=np.int16)
        label_grid_rc = np.asarray(labels["grid_rc"], dtype=np.int64)
    retained_patch_xy = _align_semantic_patch_rows(
        patch.grid_rc,
        patch.native_xy,
        patch.grid_shape,
        label_grid_rc,
        patch_regions,
        region_count=region_count,
    )
    _discarded, _support, regions = pool_neutral_features(
        retained_patch_xy,
        np.zeros((len(retained_patch_xy), 1), dtype=np.float32),
        patch_regions,
        cell_native_xy,
        region_count=region_count,
    )
    return regions


def _align_semantic_patch_rows(
    patch_grid_rc: np.ndarray,
    patch_native_xy: np.ndarray,
    patch_grid_shape: tuple[int, int],
    label_grid_rc: np.ndarray,
    patch_regions: np.ndarray,
    *,
    region_count: int,
) -> np.ndarray:
    """Match filtered semantic-label rows back to their source patch centers.

    Semantic atlas fitting may remove detached, single-section tissue
    components before clustering.  Its label artifacts therefore contain a
    coordinate-keyed subset of the source UNI2-h patch table.  Protein table
    assembly must join that subset by ``grid_rc`` instead of assuming both
    artifacts have identical row counts.
    """

    patch_grid = np.asarray(patch_grid_rc, dtype=np.int64)
    label_grid = np.asarray(label_grid_rc, dtype=np.int64)
    regions = np.asarray(patch_regions, dtype=np.int16)
    if patch_grid.ndim != 2 or patch_grid.shape[1] != 2:
        raise ValueError("semantic feature grid must have shape (patches, 2)")
    if label_grid.ndim != 2 or label_grid.shape[1] != 2:
        raise ValueError("semantic label grid must have shape (patches, 2)")
    if regions.shape != (len(label_grid),):
        raise ValueError("semantic labels and label-grid rows do not align")
    if np.any(regions < 0) or np.any(regions >= region_count):
        raise ValueError("semantic label is outside the configured region count")
    rows, columns = map(int, patch_grid_shape)
    for name, grid in (("feature", patch_grid), ("label", label_grid)):
        if len(grid) and (
            np.any(grid[:, 0] < 0)
            or np.any(grid[:, 0] >= rows)
            or np.any(grid[:, 1] < 0)
            or np.any(grid[:, 1] >= columns)
        ):
            raise ValueError(f"semantic {name} grid coordinate is out of bounds")
    patch_linear = patch_grid[:, 0] * columns + patch_grid[:, 1]
    label_linear = label_grid[:, 0] * columns + label_grid[:, 1]
    patch_order = np.argsort(patch_linear, kind="stable")
    sorted_patch = patch_linear[patch_order]
    if len(sorted_patch) > 1 and np.any(sorted_patch[1:] == sorted_patch[:-1]):
        raise ValueError("semantic feature grid coordinates are not unique")
    if len(label_linear) > 1 and len(np.unique(label_linear)) != len(label_linear):
        raise ValueError("semantic label grid coordinates are not unique")
    positions = np.searchsorted(sorted_patch, label_linear)
    valid = positions < len(sorted_patch)
    if np.any(valid):
        valid[valid] &= sorted_patch[positions[valid]] == label_linear[valid]
    if not np.all(valid):
        raise ValueError("semantic label grid is not a subset of feature patches")
    indices = patch_order[positions]
    native_xy = np.asarray(patch_native_xy, dtype=np.float64)
    if native_xy.shape != (len(patch_grid), 2):
        raise ValueError("semantic native patch centers do not align with the grid")
    return native_xy[indices]


def _standardize_patch_features(features: np.ndarray) -> np.ndarray:
    """Remove section-wide UNI2-h location/scale without using target outcomes."""

    values = np.asarray(features, dtype=np.float32)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (values - center) / scale


def _cell_context_features(
    geometry: dict[str, np.ndarray], section_spacing_um: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return stain-free sampled area and local registered-space density.

    Sampled label area is a coarse shape proxy.  Density is based on the eighth
    registered-space neighbor and therefore does not encode a semantic class.
    """

    from scipy.spatial import cKDTree

    present = np.asarray(geometry["present"], dtype=np.int64)
    counts = np.asarray(geometry["counts"], dtype=np.float64)
    if counts.shape != present.shape:
        raise ValueError("native cell areas must align with present label IDs")
    shape = np.log1p(counts)[:, None]
    xy = np.asarray(geometry["reference_um_xy"], dtype=np.float64)
    if len(xy) < 2:
        return shape.astype(np.float32), np.zeros((len(xy), 1), np.float32)
    count = min(9, len(xy))
    distance, _index = cKDTree(xy).query(xy, k=count)
    radius = distance[:, -1] if distance.ndim == 2 else distance
    density = (count - 1) / (np.pi * np.maximum(radius, section_spacing_um) ** 2)
    return shape.astype(np.float32), np.log1p(density)[:, None].astype(np.float32)


def _semantic_reference_for_regions(
    table, regions: np.ndarray, *, train: np.ndarray | None = None
) -> np.ndarray:
    """Return a coarse training-only region prior for conservative shrinkage."""

    training_regions = np.asarray(table.semantic_region, dtype=np.int16)
    measured = np.asarray(table.measured_od, dtype=np.float64)
    selected = np.ones(len(measured), dtype=bool)
    if train is not None:
        selected[:] = False
        selected[np.asarray(train, dtype=np.int64)] = True
    valid = np.isfinite(measured) & selected
    global_od = float(np.median(measured[valid]))
    output = np.full(len(regions), global_od, dtype=np.float32)
    for region in np.unique(regions):
        region_values = measured[valid & (training_regions == region)]
        if region_values.size:
            output[np.asarray(regions) == region] = np.median(region_values)
    return output
