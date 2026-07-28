"""Continuous tissue-envelope and semantic-region reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic
from histopia.topology._interpolate import (
    _signed_distance,
    interpolate_pair,
    smooth_displacement_field,
)
from histopia.topology._model import ObservedSection

_FLOW_CONFIDENCE_GATE = 0.45
_VIEWER_FACE_TARGET = 200_000


@dataclass(frozen=True, slots=True)
class RegisteredMaskStack:
    """Approved tissue masks resampled into one reference physical grid."""

    masks: np.ndarray
    origin_um_xy: tuple[float, float]
    spacing_um: float
    provenance: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class DenseVolume:
    """Numerically sampled envelope and semantic fields."""

    envelope_sdf: np.ndarray
    membership: np.ndarray
    uncertainty: np.ndarray
    z_positions_um: np.ndarray
    observed_section_indices: np.ndarray
    segments: np.ndarray

    @property
    def support(self) -> np.ndarray:
        return self.envelope_sdf >= 0


def load_registered_mask_stack(
    registration_run: Path | str,
    slide_ids: tuple[str, ...],
    *,
    target_spacing_um: float,
    max_xy_dim_px: int,
    require_review: bool = True,
) -> RegisteredMaskStack:
    """Warp approved masks into a bounded reference-physical grid."""

    try:
        from PIL import Image
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise RuntimeError(
            "registered mask reconstruction requires the 'topology' extra and Pillow"
        ) from exc

    root = Path(registration_run).expanduser().resolve()
    payload = json.loads((root / "registration_result.json").read_text())
    rows = payload.get("slides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("registration result contains no slides")
    by_id = {
        Path(str(row.get("path", ""))).name: row
        for row in rows
        if isinstance(row, dict)
    }
    if set(slide_ids) - set(by_id):
        missing = sorted(set(slide_ids) - set(by_id))
        raise ValueError(f"registration result is missing topology slides: {missing}")
    reference = next(
        (row for row in rows if isinstance(row, dict) and row.get("is_reference")),
        None,
    )
    if reference is None:
        raise ValueError("registration result has no reference slide")
    reference_to_physical = _matrix(
        reference.get("geometry", {}).get("thumbnail_to_physical"),
        "reference thumbnail-to-physical transform",
    )

    sources: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    provenance: list[dict[str, object]] = []
    bounds: list[np.ndarray] = []
    for slide_id in slide_ids:
        row = by_id[slide_id]
        mask_payload = row.get("mask")
        review = row.get("mask_review")
        mask_accepted = isinstance(mask_payload, dict) and bool(
            mask_payload.get("accepted")
        )
        review_approved = isinstance(review, dict) and (
            review.get("status") in {"auto_pass", "override_pass"}
            or bool(review.get("approved"))
        )
        if not mask_accepted or (require_review and not review_approved):
            raise ValueError(f"{slide_id}: tissue mask is not approved")
        if not isinstance(review, dict):
            review = {}
        source = root / "processed" / f"{Path(slide_id).stem}.mask.png"
        if not source.is_file():
            raise FileNotFoundError(f"{slide_id}: approved tissue mask is missing")
        with Image.open(source) as image:
            mask = np.asarray(image.convert("L")) > 0
        expected_shape = tuple(
            int(value) for value in row["geometry"]["thumbnail_shape"]
        )
        if mask.shape != expected_shape:
            raise ValueError(f"{slide_id}: tissue mask shape differs from geometry")
        if not np.any(mask):
            raise ValueError(f"{slide_id}: approved tissue mask is empty")
        source_to_physical = reference_to_physical @ _matrix(
            row.get("transform", {}).get("matrix"),
            f"{slide_id} transform",
        )
        rr, cc = np.nonzero(mask)
        corners = np.asarray(
            [
                [cc.min(), rr.min(), 1.0],
                [cc.max() + 1, rr.min(), 1.0],
                [cc.min(), rr.max() + 1, 1.0],
                [cc.max() + 1, rr.max() + 1, 1.0],
            ]
        )
        mapped = (source_to_physical @ corners.T).T
        bounds.append(mapped[:, :2] / mapped[:, 2, None])
        sources.append(mask)
        transforms.append(source_to_physical)
        provenance.append(
            {
                "slide_id": slide_id,
                "mask_artifact": source.relative_to(root).as_posix(),
                "mask_sha256": _sha256_file(source),
                "mask_method": mask_payload.get("method"),
                "mask_review_status": review.get("status"),
            }
        )

    points = np.concatenate(bounds)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    spacing_um = max(
        float(target_spacing_um),
        float(np.max(extent)) / max(max_xy_dim_px - 4, 1),
    )
    origin = minimum - spacing_um
    width, height = np.ceil((maximum - origin) / spacing_um).astype(int) + 2
    shape = (int(height), int(width))
    if max(shape) > max_xy_dim_px:
        spacing_um *= max(shape) / max_xy_dim_px
        origin = minimum - spacing_um
        width, height = np.ceil((maximum - origin) / spacing_um).astype(int) + 2
        shape = (int(height), int(width))

    grid_rows, grid_cols = np.indices(shape, dtype=np.float64)
    physical = np.stack(
        (
            origin[0] + grid_cols.ravel() * spacing_um,
            origin[1] + grid_rows.ravel() * spacing_um,
            np.ones(grid_rows.size),
        )
    )
    resampled: list[np.ndarray] = []
    for mask, source_to_physical in zip(sources, transforms, strict=True):
        source_xy = np.linalg.inv(source_to_physical) @ physical
        source_xy /= source_xy[2]
        sampled = map_coordinates(
            mask.astype(np.uint8),
            [source_xy[1], source_xy[0]],
            order=0,
            mode="constant",
            cval=0,
        )
        resampled.append(sampled.reshape(shape).astype(bool))
    return RegisteredMaskStack(
        masks=np.stack(resampled),
        origin_um_xy=(float(origin[0]), float(origin[1])),
        spacing_um=float(spacing_um),
        provenance=tuple(provenance),
    )


def benchmark_envelope_methods(
    masks: np.ndarray,
    sections: tuple[ObservedSection, ...],
    z_positions_um: tuple[float, ...],
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> dict[str, object]:
    """Select an interpolation method with leave-one-section-out evidence."""

    if len(masks) < 3:
        raise ValueError("envelope benchmarking requires at least three masks")
    if len(masks) != len(sections) or len(masks) != len(z_positions_um):
        raise ValueError("envelope benchmark inputs must have equal section counts")
    from scipy.ndimage import distance_transform_edt

    sdfs = np.stack(
        [_signed_distance(mask, distance_transform_edt) for mask in masks]
    ).astype(np.float32)
    methods = ("linear_sdf", "flow_sdf", "pchip_sdf")
    cases: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    for index in range(1, len(masks) - 1):
        source = index - 1
        target = index + 1
        fraction = (z_positions_um[index] - z_positions_um[source]) / (
            z_positions_um[target] - z_positions_um[source]
        )
        predictions = {
            "linear_sdf": ((1.0 - fraction) * sdfs[source] + fraction * sdfs[target])
            >= 0,
            "flow_sdf": _flow_sdf(
                sdfs[source],
                sdfs[target],
                sections[source],
                sections[target],
                fraction=fraction,
                origin_um_xy=origin_um_xy,
                spacing_um=spacing_um,
            )[0]
            >= 0,
            "pchip_sdf": _heldout_pchip_sdf(
                sdfs,
                z_positions_um,
                hidden=index,
            )
            >= 0,
        }
        for method, predicted in predictions.items():
            cases[method].append(_mask_metrics(predicted, masks[index]))

    candidates: list[dict[str, object]] = []
    for method in methods:
        rows = cases[method]
        dice = np.asarray([row["tissue_dice"] for row in rows])
        boundary = np.asarray([row["boundary_f1"] for row in rows])
        candidates.append(
            {
                "method": method,
                "case_count": len(rows),
                "median_tissue_dice": float(np.median(dice)),
                "tenth_percentile_tissue_dice": float(np.percentile(dice, 10)),
                "median_boundary_f1": float(np.median(boundary)),
                "cases": rows,
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["median_tissue_dice"]),
            float(row["tenth_percentile_tissue_dice"]),
            float(row["median_boundary_f1"]),
        ),
        reverse=True,
    )
    baseline = next(row for row in candidates if row["method"] == "linear_sdf")
    eligible = [
        row
        for row in candidates
        if row["method"] == "linear_sdf"
        or (
            float(row["median_tissue_dice"])
            >= float(baseline["median_tissue_dice"]) + 0.002
        )
        or (
            float(row["median_boundary_f1"])
            >= float(baseline["median_boundary_f1"]) + 0.02
            and float(row["median_tissue_dice"])
            >= float(baseline["median_tissue_dice"]) - 0.002
        )
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["median_tissue_dice"]),
            float(row["median_boundary_f1"]),
        ),
    )
    passed = (
        float(selected["median_tissue_dice"]) >= 0.90
        and float(selected["tenth_percentile_tissue_dice"]) >= 0.80
        and float(selected["median_boundary_f1"]) >= 0.75
    )
    return {
        "method": "leave_one_section_out_registered_mask",
        "selected_method": selected["method"],
        "status": "passed" if passed else "failed",
        "gates": {
            "median_tissue_dice": 0.90,
            "tenth_percentile_tissue_dice": 0.80,
            "median_boundary_f1": 0.75,
        },
        "candidates": candidates,
    }


def reconstruct_dense_volume(
    masks: np.ndarray,
    sections: tuple[ObservedSection, ...],
    links: tuple[dict[str, np.ndarray], ...],
    z_positions_um: tuple[float, ...],
    intervals: tuple[int, ...],
    segments: tuple[int, ...],
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    section_thickness_um: float,
    samples_per_interval: int,
    envelope_method: str,
) -> DenseVolume:
    """Reconstruct dense z samples without inventing biological sections."""

    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if len(masks) != len(sections) or len(links) != len(sections) - 1:
        raise ValueError("dense reconstruction requires aligned masks and links")
    observed_sdfs = np.stack(
        [_signed_distance(mask, distance_transform_edt) for mask in masks]
    ).astype(np.float32)
    dense_z: list[float] = []
    dense_sdf: list[np.ndarray] = []
    dense_membership: list[np.ndarray] = []
    dense_uncertainty: list[np.ndarray] = []
    dense_observed: list[int] = []
    dense_segments: list[int] = []

    pchip_fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if envelope_method == "pchip_sdf":
        for segment in sorted(set(segments)):
            indices = np.asarray(
                [index for index, value in enumerate(segments) if value == segment]
            )
            if len(indices) >= 3:
                pchip_fields[segment] = (
                    indices,
                    observed_sdfs[indices],
                )

    def append_observed(index: int) -> None:
        dense_z.append(float(z_positions_um[index]))
        dense_sdf.append(observed_sdfs[index])
        dense_membership.append(sections[index].membership)
        dense_uncertainty.append(np.zeros(masks.shape[1:], dtype=np.float32))
        dense_observed.append(index)
        dense_segments.append(segments[index])

    append_observed(0)
    for index, interval_count in enumerate(intervals):
        subdivisions = max(1, interval_count * samples_per_interval)
        source_sdf = observed_sdfs[index]
        target_sdf = observed_sdfs[index + 1]
        link = links[index]
        field, field_confidence = _flow_field(
            sections[index],
            sections[index + 1],
            link,
            origin_um_xy=origin_um_xy,
            spacing_um=spacing_um,
        )
        for step in range(1, subdivisions):
            fraction = step / subdivisions
            z_um = z_positions_um[index] + fraction * (
                z_positions_um[index + 1] - z_positions_um[index]
            )
            if envelope_method == "flow_sdf" and field is not None:
                sdf, confidence = _warp_sdf_pair(
                    source_sdf,
                    target_sdf,
                    field,
                    fraction,
                    field_confidence,
                )
            elif (
                envelope_method == "pchip_sdf"
                and segments[index] == segments[index + 1]
                and segments[index] in pchip_fields
            ):
                section_indices, segment_sdfs = pchip_fields[segments[index]]
                sdf = _evaluate_pchip(
                    segment_sdfs,
                    tuple(z_positions_um[item] for item in section_indices),
                    z_um,
                )
                confidence = np.ones_like(sdf, dtype=np.float32)
            else:
                sdf = (1.0 - fraction) * source_sdf + fraction * target_sdf
                confidence = np.ones_like(sdf, dtype=np.float32)
            semantic_plane = interpolate_pair(
                sections[index],
                sections[index + 1],
                fraction=fraction,
                z_um=z_um,
                segment=segments[index],
                source_section=index,
                target_section=index + 1,
                displacement_rc=(
                    field
                    if field is not None
                    else np.zeros((2, *masks.shape[1:]), dtype=np.float32)
                ),
                flow_confidence=confidence,
            )
            support = sdf >= 0
            membership = np.asarray(semantic_plane.membership, dtype=np.float32)
            membership[:, ~support] = 0
            dense_z.append(float(z_um))
            dense_sdf.append(np.asarray(sdf, dtype=np.float32))
            dense_membership.append(membership)
            dense_uncertainty.append(
                np.clip(
                    semantic_plane.uncertainty
                    + 0.25 * (1.0 - np.asarray(confidence, dtype=np.float32)),
                    0,
                    1,
                )
            )
            dense_observed.append(-1)
            dense_segments.append(segments[index])
        append_observed(index + 1)

    sdf_stack = np.stack(dense_sdf).astype(np.float32)
    membership = np.stack(dense_membership).astype(np.float32)
    for plane_index in range(len(membership)):
        for class_index in range(membership.shape[1]):
            membership[plane_index, class_index] = gaussian_filter(
                membership[plane_index, class_index],
                sigma=0.6,
                mode="nearest",
            )
        total = membership[plane_index].sum(axis=0)
        np.divide(
            membership[plane_index],
            np.maximum(total, np.finfo(np.float32).eps),
            out=membership[plane_index],
            where=total > np.finfo(np.float32).eps,
        )
        membership[plane_index, :, sdf_stack[plane_index] < 0] = 0

    expected_step = section_thickness_um / samples_per_interval
    differences = np.diff(np.asarray(dense_z))
    if len(differences) and not np.allclose(
        differences,
        expected_step,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError("dense reconstruction z samples are not uniformly spaced")
    return DenseVolume(
        envelope_sdf=sdf_stack,
        membership=membership,
        uncertainty=np.stack(dense_uncertainty).astype(np.float32),
        z_positions_um=np.asarray(dense_z, dtype=np.float64),
        observed_section_indices=np.asarray(dense_observed, dtype=np.int32),
        segments=np.asarray(dense_segments, dtype=np.int16),
    )


def write_dense_volume(
    root: Path,
    volume: DenseVolume,
) -> dict[str, object]:
    """Persist the compact scientific dense-field artifact."""

    from histopia.topology._pipeline import _savez_atomic

    path = root / "volume" / "dense-fields.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    _savez_atomic(
        path,
        envelope_sdf=volume.envelope_sdf.astype(np.float16),
        membership=volume.membership.astype(np.float16),
        uncertainty=volume.uncertainty.astype(np.float16),
        z_positions_um=volume.z_positions_um,
        observed_section_indices=volume.observed_section_indices,
        segments=volume.segments,
    )
    return {
        "artifact": path.relative_to(root).as_posix(),
        "shape_zrc": list(volume.envelope_sdf.shape),
        "numerical_sample_count": int(len(volume.z_positions_um)),
        "observed_sample_count": int(
            np.count_nonzero(volume.observed_section_indices >= 0)
        ),
    }


def write_connected_meshes(
    root: Path,
    volume: DenseVolume,
    *,
    palette: tuple[str, ...],
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    z_spacing_um: float,
    source_patch_width_um: float,
    section_thickness_um: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
]:
    """Write one envelope, selected semantic regions, and uncertainty surface."""

    from scipy.ndimage import gaussian_filter

    support = volume.support
    z_sigma = max(1.0, section_thickness_um / z_spacing_um * 1.25)
    display_envelope = gaussian_filter(
        volume.envelope_sdf,
        sigma=(z_sigma, 0.55, 0.55),
        mode="nearest",
    )
    envelope = _write_scalar_mesh(
        root,
        display_envelope,
        stem="envelope",
        role="envelope",
        color="#d7dde5",
        level=0.0,
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
        z_origin_um=float(volume.z_positions_um[0]),
        z_spacing_um=z_spacing_um,
    )
    envelope["viewer_regularization"] = {
        "z_sigma_um": z_sigma * z_spacing_um,
        "xy_sigma_um": 0.55 * spacing_um,
        "changes_measurement": False,
    }
    scientific_labels = np.argmax(volume.membership, axis=1).astype(np.int16)
    scientific_membership = volume.membership.sum(axis=1) > 0
    scientific_labels[~support | ~scientific_membership] = -1
    display_membership = gaussian_filter(
        volume.membership,
        sigma=(1.1 * z_sigma, 0, 0.8, 0.8),
        mode="nearest",
    )
    display_support = display_envelope >= 0
    regions: list[dict[str, object]] = []
    classes: list[dict[str, object]] = []
    voxel_volume_um3 = spacing_um**2 * z_spacing_um
    minimum_component_volume = 4.0 * source_patch_width_um**2 * section_thickness_um
    observed_dense = np.flatnonzero(volume.observed_section_indices >= 0)
    for class_index, color in enumerate(palette):
        scientific = scientific_labels == class_index
        display = (display_membership[:, class_index] >= 0.35) & display_support
        displayed, before, after = filter_persistent_components(
            display,
            observed_dense_indices=observed_dense,
            voxel_volume_um3=voxel_volume_um3,
            minimum_component_volume_um3=minimum_component_volume,
        )
        if np.any(displayed):
            display_field = display_membership[:, class_index].copy()
            display_field[~displayed] = 0
            region = _write_scalar_mesh(
                root,
                display_field,
                stem=f"region-{class_index:02d}",
                role="semantic_region",
                color=color,
                level=0.35,
                origin_um_xy=origin_um_xy,
                spacing_um=spacing_um,
                z_origin_um=float(volume.z_positions_um[0]),
                z_spacing_um=z_spacing_um,
            )
            region["class_index"] = class_index
            region["component_count_before_filter"] = before
            region["component_count_after_filter"] = after
            region["viewer_regularization"] = {
                "z_sigma_um": 1.1 * z_sigma * z_spacing_um,
                "xy_sigma_um": 0.8 * spacing_um,
                "changes_measurement": False,
            }
            regions.append(region)
        classes.append(
            {
                "class_index": class_index,
                "color": color,
                "estimated_volume_mm3": (
                    float(np.count_nonzero(scientific)) * voxel_volume_um3 / 1e9
                ),
                "viewer_component_count_before_filter": before,
                "viewer_component_count_after_filter": after,
                "viewer_filter_changes_measurement": False,
            }
        )
    uncertain = support & (volume.uncertainty >= 0.65)
    uncertainty_mesh = (
        _write_scalar_mesh(
            root,
            uncertain.astype(np.float32),
            stem="uncertainty",
            role="uncertainty",
            color="#ffb454",
            level=0.5,
            origin_um_xy=origin_um_xy,
            spacing_um=spacing_um,
            z_origin_um=float(volume.z_positions_um[0]),
            z_spacing_um=z_spacing_um,
        )
        if np.any(uncertain) and np.any(~uncertain)
        else None
    )
    return envelope, regions, classes, uncertainty_mesh


def filter_persistent_components(
    occupancy: np.ndarray,
    *,
    observed_dense_indices: np.ndarray,
    voxel_volume_um3: float,
    minimum_component_volume_um3: float,
) -> tuple[np.ndarray, int, int]:
    """Suppress viewer speckles while retaining scientific occupancy unchanged."""

    from scipy.ndimage import label

    components, count = label(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    keep = np.zeros(count + 1, dtype=bool)
    for component in range(1, count + 1):
        selected = components == component
        observed_hits = sum(
            bool(np.any(selected[index]))
            for index in observed_dense_indices
            if 0 <= index < len(selected)
        )
        volume_um3 = float(np.count_nonzero(selected)) * voxel_volume_um3
        keep[component] = (
            observed_hits >= 2 or volume_um3 >= minimum_component_volume_um3
        )
    filtered = keep[components]
    return filtered, int(count), int(np.count_nonzero(keep))


def _write_scalar_mesh(
    root: Path,
    field: np.ndarray,
    *,
    stem: str,
    role: str,
    color: str,
    level: float,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    z_origin_um: float,
    z_spacing_um: float,
) -> dict[str, object]:
    from skimage.measure import marching_cubes, mesh_surface_area

    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    padded = np.pad(np.asarray(field, dtype=np.float32), 1, mode="constant")
    vertices_zyx, faces, _, _ = marching_cubes(
        padded,
        level=level,
        spacing=(z_spacing_um, spacing_um, spacing_um),
        allow_degenerate=False,
    )
    vertices = np.column_stack(
        [
            origin_um_xy[0] - spacing_um + vertices_zyx[:, 2],
            origin_um_xy[1] - spacing_um + vertices_zyx[:, 1],
            z_origin_um - z_spacing_um + vertices_zyx[:, 0],
        ]
    ).astype(np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    full_path = mesh_dir / f"{stem}.npz"
    viewer_path = mesh_dir / f"{stem}.bin"
    from histopia.topology._pipeline import _savez_atomic

    _savez_atomic(full_path, vertices_um_xyz=vertices, faces=faces)
    viewer_vertices, viewer_faces, downsample = _viewer_mesh(
        field,
        level=level,
        vertices=vertices,
        faces=faces,
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
        z_origin_um=z_origin_um,
        z_spacing_um=z_spacing_um,
    )
    _write_mesh_binary(viewer_path, viewer_vertices, viewer_faces)
    return {
        "role": role,
        "color": color,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "viewer_face_count": len(viewer_faces),
        "viewer_downsample_xy": downsample,
        "surface_area_mm2": float(mesh_surface_area(vertices, faces)) / 1e6,
        "artifact": full_path.relative_to(root).as_posix(),
        "viewer_asset": viewer_path.relative_to(root).as_posix(),
    }


def _viewer_mesh(
    field: np.ndarray,
    *,
    level: float,
    vertices: np.ndarray,
    faces: np.ndarray,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    z_origin_um: float,
    z_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(faces) <= _VIEWER_FACE_TARGET:
        return vertices, faces, 1
    from skimage.measure import marching_cubes

    initial = max(
        2,
        int(math.ceil(math.sqrt(len(faces) / _VIEWER_FACE_TARGET))),
    )
    for factor in range(initial, 9):
        reduced = np.asarray(field)[:, ::factor, ::factor]
        padded = np.pad(reduced.astype(np.float32), 1, mode="constant")
        vertices_zyx, viewer_faces, _, _ = marching_cubes(
            padded,
            level=level,
            spacing=(z_spacing_um, spacing_um * factor, spacing_um * factor),
            allow_degenerate=False,
        )
        if len(viewer_faces) <= _VIEWER_FACE_TARGET or factor == 8:
            break
    viewer_spacing = spacing_um * factor
    viewer_vertices = np.column_stack(
        [
            origin_um_xy[0] - viewer_spacing + vertices_zyx[:, 2],
            origin_um_xy[1] - viewer_spacing + vertices_zyx[:, 1],
            z_origin_um - z_spacing_um + vertices_zyx[:, 0],
        ]
    ).astype(np.float32)
    return viewer_vertices, np.asarray(viewer_faces, dtype=np.uint32), factor


def _flow_sdf(
    source_sdf: np.ndarray,
    target_sdf: np.ndarray,
    source: ObservedSection,
    target: ObservedSection,
    *,
    fraction: float,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    from histopia.topology._benchmark import _semantic_endpoint_links

    source_xy, target_xy, confidence, _, _ = _semantic_endpoint_links(
        source,
        target,
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
    )
    accepted = confidence >= _FLOW_CONFIDENCE_GATE
    if np.count_nonzero(accepted) < 8:
        return (
            (1.0 - fraction) * source_sdf + fraction * target_sdf,
            np.zeros(source_sdf.shape, dtype=np.float32),
        )
    field, field_confidence = smooth_displacement_field(
        source_sdf.shape,
        source_xy=source_xy[accepted],
        target_xy=target_xy[accepted],
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
    )
    return _warp_sdf_pair(
        source_sdf,
        target_sdf,
        field,
        fraction,
        field_confidence,
    )


def _flow_field(
    source: ObservedSection,
    target: ObservedSection,
    link: dict[str, np.ndarray],
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    confidence = np.asarray(link["confidence"], dtype=np.float32)
    accepted = confidence >= _FLOW_CONFIDENCE_GATE
    if np.count_nonzero(accepted) < 8:
        return None, np.zeros(source.support.shape, dtype=np.float32)
    field, field_confidence = smooth_displacement_field(
        source.support.shape,
        source_xy=np.asarray(link["source_um_xy"])[accepted],
        target_xy=np.asarray(link["target_um_xy"])[accepted],
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
    )
    return field, field_confidence


def _warp_sdf_pair(
    source_sdf: np.ndarray,
    target_sdf: np.ndarray,
    field: np.ndarray,
    fraction: float,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import map_coordinates

    rows, cols = np.indices(source_sdf.shape, dtype=np.float32)
    source_coords = np.asarray([rows - fraction * field[0], cols - fraction * field[1]])
    target_coords = np.asarray(
        [
            rows + (1.0 - fraction) * field[0],
            cols + (1.0 - fraction) * field[1],
        ]
    )
    source_warped = map_coordinates(
        source_sdf,
        source_coords,
        order=1,
        mode="constant",
        cval=float(source_sdf.min()),
    )
    target_warped = map_coordinates(
        target_sdf,
        target_coords,
        order=1,
        mode="constant",
        cval=float(target_sdf.min()),
    )
    return (
        ((1.0 - fraction) * source_warped + fraction * target_warped).astype(
            np.float32
        ),
        np.asarray(confidence, dtype=np.float32),
    )


def _heldout_pchip_sdf(
    sdfs: np.ndarray,
    z_positions_um: tuple[float, ...],
    *,
    hidden: int,
) -> np.ndarray:
    available = [index for index in range(len(sdfs)) if index != hidden]
    available.sort(key=lambda index: abs(index - hidden))
    selected = np.asarray(sorted(available[:4]))
    return _evaluate_pchip(
        sdfs[selected],
        tuple(z_positions_um[index] for index in selected),
        z_positions_um[hidden],
    )


def _evaluate_pchip(
    fields: np.ndarray,
    z_positions_um: tuple[float, ...],
    z_um: float,
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    output = np.empty(fields.shape[1:], dtype=np.float32)
    for start in range(0, fields.shape[1], 32):
        stop = min(fields.shape[1], start + 32)
        interpolator = PchipInterpolator(
            np.asarray(z_positions_um, dtype=float),
            fields[:, start:stop],
            axis=0,
            extrapolate=False,
        )
        output[start:stop] = interpolator(z_um).astype(np.float32)
    return output


def _mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    from scipy.ndimage import binary_dilation, binary_erosion

    from histopia.topology._benchmark import _dice

    predicted_boundary = predicted ^ binary_erosion(predicted)
    truth_boundary = truth ^ binary_erosion(truth)
    tolerance = np.ones((5, 5), dtype=bool)
    precision = (
        float(
            np.mean(
                binary_dilation(truth_boundary, structure=tolerance)[predicted_boundary]
            )
        )
        if np.any(predicted_boundary)
        else 0.0
    )
    recall = (
        float(
            np.mean(
                binary_dilation(predicted_boundary, structure=tolerance)[truth_boundary]
            )
        )
        if np.any(truth_boundary)
        else 0.0
    )
    boundary_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else float(not np.any(predicted_boundary) and not np.any(truth_boundary))
    )

    return {
        "tissue_dice": _dice(predicted, truth),
        "boundary_f1": boundary_f1,
    }


def _matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return matrix


def _write_mesh_binary(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Path:
    def write(stream) -> None:
        stream.write(b"HTM1")
        stream.write(struct.pack("<II", len(vertices), len(faces)))
        stream.write(np.asarray(vertices, dtype="<f4").tobytes())
        stream.write(np.asarray(faces, dtype="<u4").tobytes())

    return write_binary_atomic(path, write)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
