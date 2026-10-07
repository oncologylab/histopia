"""Bounded-memory assembly of real cell/stain/semantic protein runs.

The adapter deliberately keeps the target chromogen out of the predictor.  UNI2-h
sees only a fixed-palette rendering of the stain model's counterstain channel.
Target OD is sampled separately, under the validated tissue mask, and is used only
as an outcome on sections carrying the requested marker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RealRunBindings:
    """Validated, exact-run inputs for one mouse."""

    registration: dict[str, object]
    cells: dict[str, object]
    stain: dict[str, object]
    semantic: dict[str, object]


@dataclass(frozen=True, slots=True)
class AdaptiveTargetMeasurement:
    """One sealed adaptive-corrected target-OD outcome at 4 microns/px."""

    target_od: np.ndarray
    tissue_mask: np.ndarray
    positive_mask: np.ndarray
    floor_od: float | None
    positive_threshold_od: float | None
    measurement_view: str
    fingerprint: str


def validated_adaptive_target_measurement(
    stain_map: object,
    stain_row: dict[str, object],
    *,
    adaptive_stain_map: object | None = None,
    expected_analysis_mpp: float = 4.0,
) -> AdaptiveTargetMeasurement:
    """Return the only target-OD view permitted for protein outcomes.

    The physical stain artifact deliberately retains its corrected target-OD
    array unchanged.  This selector applies the separately validated adaptive
    tissue floor before any cell aggregation or rendering.  Raw-OD and
    correction/adaptive fallbacks are refused so training labels and displayed
    ground truth cannot silently diverge.
    """

    qc = stain_row.get("qc")
    if not isinstance(qc, dict) or qc.get("correction_accepted") is not True:
        raise ValueError("protein target requires accepted nuisance correction")
    adaptive = qc.get("adaptive_background")
    if not isinstance(adaptive, dict) or adaptive.get("accepted") is not True:
        raise ValueError("protein target requires accepted adaptive background")
    method = adaptive.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("adaptive background method is missing")
    analysis_mpp = float(stain_map.analysis_mpp)
    if not np.isfinite(analysis_mpp) or not np.isclose(
        analysis_mpp, expected_analysis_mpp, rtol=0, atol=1e-9
    ):
        raise ValueError(
            f"protein target requires {expected_analysis_mpp:g} microns/px"
        )
    source_fingerprint = getattr(stain_map, "content_fingerprint", None)
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        source_fingerprint = getattr(stain_map, "fingerprint", None)
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("stain map content fingerprint is missing")

    if method == "counterstain-conditioned-v3":
        if adaptive_stain_map is None:
            raise ValueError("counterstain-conditioned target map is missing")
        values = np.asarray(adaptive_stain_map.target_od, dtype=np.float32)
        tissue = np.asarray(adaptive_stain_map.tissue_mask, dtype=bool)
        if (
            getattr(adaptive_stain_map, "method", None) != method
            or getattr(adaptive_stain_map, "diagnostics", None) != adaptive
            or getattr(adaptive_stain_map, "source_content_fingerprint", None)
            != source_fingerprint
            or getattr(adaptive_stain_map, "slide_id", None)
            != getattr(stain_map, "slide_id", None)
            or not np.isclose(
                float(adaptive_stain_map.analysis_mpp),
                analysis_mpp,
                rtol=0,
                atol=1e-9,
            )
            or tuple(adaptive_stain_map.content_origin_native_xy)
            != tuple(stain_map.content_origin_native_xy)
            or tuple(adaptive_stain_map.source_mpp_xy) != tuple(stain_map.source_mpp_xy)
        ):
            raise ValueError("counterstain-conditioned target map binding differs")
        expected_fingerprint = stain_row.get("adaptive_map_fingerprint")
        observed_fingerprint = getattr(adaptive_stain_map, "content_fingerprint", None)
        if (
            not isinstance(expected_fingerprint, str)
            or expected_fingerprint != observed_fingerprint
        ):
            raise ValueError("counterstain-conditioned target digest differs")
        if values.ndim != 2 or values.shape != tissue.shape:
            raise ValueError("adaptive target OD and tissue support do not align")
        if not np.array_equal(tissue, np.asarray(stain_map.tissue_mask, dtype=bool)):
            raise ValueError("counterstain-conditioned tissue support differs")
        if np.any(~np.isfinite(values[tissue])) or np.any(values[tissue] < 0):
            raise ValueError("adaptive target OD contains invalid tissue values")
        positive = np.zeros(tissue.shape, dtype=bool)
        measurement_view = "tissue-masked-counterstain-conditioned-target-od-4um-v3"
        digest_payload = {
            "schema_version": 1,
            "measurement_view": measurement_view,
            "source_content_fingerprint": source_fingerprint,
            "adaptive_content_fingerprint": observed_fingerprint,
            "analysis_mpp": analysis_mpp,
            "adaptive_method": method,
            "positive_threshold_od": None,
        }
        fingerprint = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        values.setflags(write=False)
        tissue.setflags(write=False)
        positive.setflags(write=False)
        return AdaptiveTargetMeasurement(
            target_od=values,
            tissue_mask=tissue,
            positive_mask=positive,
            floor_od=None,
            positive_threshold_od=None,
            measurement_view=measurement_view,
            fingerprint=fingerprint,
        )

    floor_value = adaptive.get("floor_od")
    if isinstance(floor_value, bool) or not isinstance(floor_value, (int, float)):
        raise ValueError("adaptive background floor is missing")
    floor = float(floor_value)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("adaptive background floor must be finite and nonnegative")

    corrected = np.asarray(stain_map.corrected_target_od, dtype=np.float32)
    tissue = np.asarray(stain_map.tissue_mask, dtype=bool)
    if corrected.ndim != 2 or corrected.shape != tissue.shape:
        raise ValueError("adaptive target OD and tissue support do not align")
    if np.any(~np.isfinite(corrected[tissue])):
        raise ValueError("adaptive target OD contains non-finite tissue values")
    values = np.zeros(corrected.shape, dtype=np.float32)
    values[tissue] = np.maximum(corrected[tissue] - floor, 0)

    threshold: float | None = None
    positive = np.zeros(tissue.shape, dtype=bool)
    threshold_value = qc.get("positive_threshold_od")
    if qc.get("threshold_accepted") is True:
        if isinstance(threshold_value, bool) or not isinstance(
            threshold_value, (int, float)
        ):
            raise ValueError("accepted positive threshold is missing")
        physical_threshold = float(threshold_value)
        if not np.isfinite(physical_threshold) or physical_threshold < 0:
            raise ValueError("positive threshold must be finite and nonnegative")
        threshold = max(physical_threshold - floor, 0.0)
        positive = tissue & (values >= threshold)

    measurement_view = "tissue-masked-adaptive-corrected-target-od-4um-v1"
    digest_payload = {
        "schema_version": 1,
        "measurement_view": measurement_view,
        "source_content_fingerprint": source_fingerprint,
        "map_artifact_digest": stain_row.get("map_artifact_digest"),
        "analysis_mpp": analysis_mpp,
        "adaptive_method": method,
        "adaptive_floor_od": floor,
        "positive_threshold_od": threshold,
    }
    fingerprint = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values.setflags(write=False)
    tissue.setflags(write=False)
    positive.setflags(write=False)
    return AdaptiveTargetMeasurement(
        target_od=values,
        tissue_mask=tissue,
        positive_mask=positive,
        floor_od=floor,
        positive_threshold_od=threshold,
        measurement_view=measurement_view,
        fingerprint=fingerprint,
    )


def validate_real_run_bindings(
    registration_run: Path | str,
    cell_run: Path | str,
    stain_run: Path | str,
    semantic_run: Path | str,
    *,
    verify_artifacts: bool = True,
) -> RealRunBindings:
    """Require every input to name the same immutable registration result.

    Read-only batch registries may defer large artifact digests until the exact
    label or stain map is opened. Direct validation remains strict by default.
    """

    from histopia.cells._result import (
        validate_cell_artifact,
        validate_cell_result,
        validate_cell_result_index,
    )
    from histopia.semantic._registration_binding import (
        validate_semantic_registration_binding,
    )
    from histopia.semantic._result_validation import (
        validate_semantic_result,
        validate_semantic_result_index,
    )
    from histopia.stain._result_validation import (
        validate_stain_result,
        validate_stain_result_index,
    )

    registration_root = Path(registration_run)
    registration = json.loads(
        (registration_root / "registration_result.json").read_text()
    )
    digest = _sha256_file(registration_root / "registration_result.json")
    cells = (
        validate_cell_result(cell_run)
        if verify_artifacts
        else validate_cell_result_index(cell_run)
    )
    stain = (
        validate_stain_result(stain_run)
        if verify_artifacts
        else validate_stain_result_index(stain_run)
    )
    semantic = (
        validate_semantic_result(semantic_run)
        if verify_artifacts
        else validate_semantic_result_index(semantic_run)
    )
    validate_semantic_registration_binding(
        registration_root, semantic_run, semantic_payload=semantic
    )
    for name, payload in (("cell", cells), ("stain", stain)):
        if payload.get("registration_result_sha256") != digest:
            raise ValueError(f"{name} run is not bound to the requested registration")
    cell_ids = [row["slide"] for row in cells["slides"]]
    stain_ids = [row["id"] for row in stain["slides"]]
    semantic_ids = [row["id"] for row in semantic["slides"]]
    preflight_relative = cells.get("preflight")
    artifacts = cells.get("artifacts")
    if (
        not isinstance(preflight_relative, str)
        or not isinstance(artifacts, dict)
        or not isinstance(artifacts.get(preflight_relative), str)
    ):
        raise ValueError("cell preflight binding is missing")
    preflight_path = Path(cell_run) / preflight_relative
    validate_cell_artifact(preflight_path, artifacts[preflight_relative])
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("registration_result_sha256") != digest:
        raise ValueError("cell preflight is not bound to the requested registration")
    _validate_analysis_slide_order(
        cell_ids,
        stain_ids,
        semantic_ids,
        preflight.get("excluded_slides", []),
    )
    return RealRunBindings(registration, cells, stain, semantic)


def _validate_analysis_slide_order(
    cell_ids: list[object],
    stain_ids: list[object],
    semantic_ids: list[object],
    excluded_slides: object,
) -> None:
    """Match analysis slides while honoring sealed owner exclusions.

    Stain and semantic results must still describe the complete registration in
    exactly the same order.  A cell run may omit only slides explicitly named
    by its fingerprinted preflight manifest; silently missing or reordered
    slides remain invalid.
    """

    if (
        any(not isinstance(value, str) or not value for value in cell_ids)
        or any(not isinstance(value, str) or not value for value in stain_ids)
        or any(not isinstance(value, str) or not value for value in semantic_ids)
    ):
        raise ValueError("bound slide IDs must be non-empty strings")
    if stain_ids != semantic_ids:
        raise ValueError("stain and semantic slide order differs")
    if len(set(stain_ids)) != len(stain_ids) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("bound slide IDs must be unique")
    if not isinstance(excluded_slides, list):
        raise ValueError("cell preflight exclusions must be a list")
    excluded: list[str] = []
    for row in excluded_slides:
        if not isinstance(row, dict):
            raise ValueError("cell preflight exclusion rows must be objects")
        slide_id = row.get("slide_id")
        reason = row.get("reason")
        if (
            not isinstance(slide_id, str)
            or not slide_id
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("cell preflight exclusions require a slide and reason")
        excluded.append(slide_id)
    if len(set(excluded)) != len(excluded):
        raise ValueError("cell preflight exclusions must be unique")
    unknown = set(excluded) - set(stain_ids)
    if unknown:
        raise ValueError("cell preflight excludes an unknown registration slide")
    excluded_set = set(excluded)
    expected = [slide_id for slide_id in stain_ids if slide_id not in excluded_set]
    if cell_ids != expected:
        raise ValueError("cell analysis slide order differs from sealed exclusions")


def target_sections(
    stain: dict[str, object],
    target_id: str,
    *,
    require_threshold: bool = True,
    minimum_sections: int = 2,
) -> tuple[str, ...]:
    """Return QC-accepted sections for continuous or binary prediction.

    ``minimum_sections=0`` supports external-transfer cohorts without accepted
    target truth; callers remain responsible for requiring measurements in
    every declared training cohort.
    """

    from histopia.protein._config import normalize_target_id

    wanted = normalize_target_id(target_id)
    if minimum_sections < 0:
        raise ValueError("minimum target sections must be non-negative")
    selected: list[str] = []
    for row in stain["slides"]:
        marker = row.get("marker")
        qc = row.get("qc", {})
        if (
            isinstance(marker, str)
            and normalize_target_id(marker) == wanted
            and row.get("quantified") is True
            and isinstance(row.get("map"), str)
            and qc.get("correction_accepted") is True
            and isinstance(qc.get("adaptive_background"), dict)
            and qc["adaptive_background"].get("accepted") is True
            and (not require_threshold or qc.get("threshold_accepted") is True)
        ):
            selected.append(f"{int(row['order']):03d}")
    if len(selected) < minimum_sections:
        count = "two" if minimum_sections == 2 else str(minimum_sections)
        raise ValueError(
            "real protein benchmarking requires "
            f"{count} accepted target section"
            f"{'s' if minimum_sections != 1 else ''}"
        )
    return tuple(selected)


def neutral_patch_images(
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    native_xy: np.ndarray,
    *,
    content_origin_native_xy: np.ndarray,
    source_mpp_xy: np.ndarray,
    analysis_mpp: float,
    patch_width_um: float = 112.0,
    output_px: int = 224,
) -> np.ndarray:
    """Render counterstain-only patches at semantic patch centers."""

    from PIL import Image

    from histopia.protein._features import render_neutral_morphology

    od = np.asarray(counterstain_od, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    centers = (
        (np.asarray(native_xy, dtype=np.float64) - content_origin_native_xy)
        * np.asarray(source_mpp_xy, dtype=np.float64)
        / float(analysis_mpp)
    )
    rendered = render_neutral_morphology(od, tissue)
    width = max(1, int(round(patch_width_um / analysis_mpp)))
    half = width / 2.0
    output = np.empty((len(centers), output_px, output_px, 3), dtype=np.uint8)
    for index, (x, y) in enumerate(centers):
        x0, y0 = int(np.floor(x - half)), int(np.floor(y - half))
        x1, y1 = x0 + width, y0 + width
        patch = np.full((width, width, 3), 255, dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(rendered.shape[1], x1), min(rendered.shape[0], y1)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = rendered[sy0:sy1, sx0:sx1]
        output[index] = np.asarray(
            Image.fromarray(patch).resize(
                (output_px, output_px), Image.Resampling.BILINEAR
            )
        )
    return output


def sampled_label_geometry(
    label_path: Path | str, *, subifd: int = 2, cell_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-sample native labels and return sampled cell geometry.

    Generic TIFF pyramid levels average pixels and are invalid for categorical
    cell IDs. ``subifd`` selects an equivalent power-of-two reduction only.
    """

    try:
        import pyvips
    except ImportError as error:
        raise RuntimeError("real protein assembly requires the 'wsi' extra") from error
    source = pyvips.Image.new_from_file(str(label_path), access="sequential")
    scale = float(2**subifd)
    image = source.resize(1.0 / scale, kernel="nearest")
    labels = np.frombuffer(image.write_to_memory(), dtype=np.uint32).reshape(
        image.height, image.width
    )
    positive = labels > 0
    ids = labels[positive].astype(np.int64)
    if ids.size and len(np.unique(ids)) > cell_count:
        raise ValueError("sampled labels exceed the declared distinct cell count")
    yy, xx = np.nonzero(positive)
    length = int(ids.max(initial=0)) + 1
    counts = np.bincount(ids, minlength=length)
    xsum = np.bincount(ids, weights=xx + 0.5, minlength=length)
    ysum = np.bincount(ids, weights=yy + 0.5, minlength=length)
    present = np.flatnonzero(counts[1:] > 0) + 1
    xy = (
        np.column_stack(
            (xsum[present] / counts[present], ysum[present] / counts[present])
        )
        * scale
    )
    return labels, present.astype(np.uint32), xy.astype(np.float64), counts


def compartment_sampled_labels(
    labels: np.ndarray,
    compartment: str,
    *,
    source_mpp_xy: np.ndarray,
    pyramid_scale: float,
    boundary_width_um: float = 4.0,
) -> np.ndarray:
    """Select a whole-cell or inward membrane band on categorical labels.

    The operation stays on the nearest-neighbor sampled label grid.  A 4-µm
    inward band matches the validated stain-map pixel size and avoids implying
    membrane detail finer than the measurement itself.
    """

    values = np.asarray(labels)
    if compartment == "whole_cell":
        return values
    if compartment != "inner_boundary":
        raise ValueError("unsupported protein measurement compartment")
    spacing = float(np.mean(np.asarray(source_mpp_xy, dtype=np.float64))) * float(
        pyramid_scale
    )
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("sampled label spacing must be finite and positive")
    if not np.isfinite(boundary_width_um) or boundary_width_um <= 0:
        raise ValueError("boundary width must be finite and positive")
    from histopia.protein._measurements import inner_boundary_mask

    return inner_boundary_mask(
        values,
        width_px=max(1, int(round(boundary_width_um / spacing))),
    )


def sampled_label_expected_analysis_pixels(
    labels: np.ndarray,
    *,
    cell_count: int,
    source_mpp_xy: np.ndarray,
    pyramid_scale: float,
    analysis_mpp: float,
) -> np.ndarray:
    """Return each sampled compartment's expected area in analysis pixels."""

    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("sampled labels must be a two-dimensional integer array")
    if cell_count < 0 or np.any((values < 0) | (values > cell_count)):
        raise ValueError("sampled labels exceed the declared cell count")
    mpp = np.asarray(source_mpp_xy, dtype=np.float64)
    if mpp.shape != (2,) or not np.all(np.isfinite(mpp)) or np.any(mpp <= 0):
        raise ValueError("source microns-per-pixel must contain two positive values")
    if (
        not np.isfinite(pyramid_scale)
        or pyramid_scale <= 0
        or not np.isfinite(analysis_mpp)
        or analysis_mpp <= 0
    ):
        raise ValueError("pyramid scale and analysis mpp must be positive")
    counts = np.bincount(values.ravel(), minlength=cell_count + 1).astype(np.float64)
    sampled_area_um2 = float(np.prod(mpp * pyramid_scale))
    return (counts * sampled_area_um2 / analysis_mpp**2).astype(np.float32)


def aggregate_map_to_sampled_labels(
    sampled_labels: np.ndarray,
    target_od: np.ndarray,
    tissue_mask: np.ndarray,
    positive_mask: np.ndarray,
    *,
    source_mpp_xy: np.ndarray,
    analysis_mpp: float,
    content_origin_native_xy: np.ndarray,
    pyramid_scale: float,
    cell_count: int,
    supersample: int = 2,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractionally aggregate 4-µm outcome pixels to sampled cell labels.

    Cell-label images use content-box-local coordinates, whereas stain-map
    metadata records that box's absolute native origin.  Map samples are first
    expressed in absolute native coordinates and then translated back into the
    label image's local frame.  Keeping both transforms explicit prevents a
    non-zero OpenSlide bounds origin from shifting every measurement.
    """

    od = np.asarray(target_od, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    positive = np.asarray(positive_mask, dtype=bool)
    if od.shape != tissue.shape or od.shape != positive.shape:
        raise ValueError("stain map arrays must share a shape")
    if supersample < 1:
        raise ValueError("supersample must be positive")
    if statistic not in {"mean", "q90"}:
        raise ValueError("protein measurement statistic must be mean or q90")
    origin = np.asarray(content_origin_native_xy, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("content origin must contain two finite coordinates")
    counts = np.zeros(cell_count + 1, dtype=np.float64)
    sums = np.zeros(cell_count + 1, dtype=np.float64)
    positives = np.zeros(cell_count + 1, dtype=np.float64)
    quantile_ids: list[np.ndarray] = []
    quantile_values: list[np.ndarray] = []
    weight = 1.0 / supersample**2
    for sy in range(supersample):
        y_native = (
            origin[1]
            + (np.arange(od.shape[0], dtype=np.float64) + (sy + 0.5) / supersample)
            * analysis_mpp
            / source_mpp_xy[1]
        )
        yy = np.clip(
            ((y_native - origin[1]) / pyramid_scale).astype(int),
            0,
            sampled_labels.shape[0] - 1,
        )
        for sx in range(supersample):
            x_native = (
                origin[0]
                + (np.arange(od.shape[1], dtype=np.float64) + (sx + 0.5) / supersample)
                * analysis_mpp
                / source_mpp_xy[0]
            )
            xx = np.clip(
                ((x_native - origin[0]) / pyramid_scale).astype(int),
                0,
                sampled_labels.shape[1] - 1,
            )
            mapped = sampled_labels[np.ix_(yy, xx)]
            valid = tissue & (mapped > 0) & np.isfinite(od)
            ids = mapped[valid].astype(np.int64)
            if np.any(ids > cell_count):
                raise ValueError("sampled labels exceed the declared cell count")
            counts += np.bincount(ids, minlength=cell_count + 1) * weight
            sums += (
                np.bincount(ids, weights=od[valid], minlength=cell_count + 1) * weight
            )
            positives += (
                np.bincount(
                    ids,
                    weights=positive[valid].astype(float),
                    minlength=cell_count + 1,
                )
                * weight
            )
            if statistic == "q90":
                quantile_ids.append(ids)
                quantile_values.append(np.asarray(od[valid], dtype=np.float32))
    aggregated = np.divide(
        sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
    )
    if statistic == "q90" and quantile_ids:
        ids = np.concatenate(quantile_ids)
        values = np.concatenate(quantile_values)
        order = np.lexsort((values, ids))
        ids = ids[order]
        values = values[order]
        unique, starts, group_counts = np.unique(
            ids, return_index=True, return_counts=True
        )
        offsets = np.ceil(0.90 * group_counts).astype(np.int64) - 1
        aggregated[unique] = values[starts + offsets]
    fraction = np.divide(
        positives, counts, out=np.zeros_like(positives), where=counts > 0
    )
    return (
        aggregated.astype(np.float32),
        counts.astype(np.float32),
        fraction.astype(np.float32),
    )


def map_native_to_reference(
    native_xy: np.ndarray,
    slide: dict[str, object],
    reference_slide: dict[str, object],
) -> np.ndarray:
    """Map native cell centers to reference micrometres."""

    from histopia.semantic._features import map_native_to_reference_um

    geometry = slide["geometry"]
    reference_geometry = reference_slide["geometry"]
    native_to_thumbnail = np.linalg.inv(
        np.asarray(geometry["thumbnail_to_native"], dtype=np.float64)
    )
    return map_native_to_reference_um(
        native_xy,
        native_to_thumbnail=native_to_thumbnail,
        moving_to_reference_thumbnail=np.asarray(slide["transform"]["matrix"]),
        reference_thumbnail_to_native=np.asarray(
            reference_geometry["thumbnail_to_native"]
        ),
        reference_mpp_xy=tuple(reference_geometry["mpp_xy"]),
    )


def pool_neutral_features(
    patch_xy: np.ndarray,
    patch_features: np.ndarray,
    patch_regions: np.ndarray,
    cell_xy: np.ndarray,
    *,
    region_count: int,
    maximum_distance: float = 168.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool stain-neutral UNI2-h morphology; return semantics for auditing only.

    ``region_count`` is retained for API compatibility and validation, but a
    semantic one-hot vector is deliberately not appended to model features.
    """

    from scipy.spatial import cKDTree

    cells = np.asarray(cell_xy, dtype=np.float64)
    patches = np.asarray(patch_xy, dtype=np.float64)
    if region_count < 1:
        raise ValueError("region_count must be positive")
    if not len(patches):
        return (
            np.zeros((len(cells), patch_features.shape[1]), np.float32),
            np.zeros(len(cells), bool),
            np.full(len(cells), -1, np.int16),
        )
    distance, index = cKDTree(patches).query(cells, k=1)
    supported = np.asarray(distance <= maximum_distance)
    regions = np.asarray(patch_regions, dtype=np.int16)[index]
    output = np.asarray(patch_features[index], np.float32).copy()
    output[~supported] = 0
    return output, supported, regions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
