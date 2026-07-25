"""Export validated Histopia results for import by the QuPath extension."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from histopia.semantic._approval import (
    SemanticApproval,
    validate_semantic_approval,
)
from histopia.semantic._result_validation import validate_semantic_result

if TYPE_CHECKING:
    from histopia.registration._approval import RegistrationApproval

_SEMANTIC_GEOMETRIES = ("regions", "tiles")
_SEMANTIC_GEOMETRY_VERSIONS = {
    "regions": "regions-v1",
    "tiles": "tiles-v1",
}


def export_qupath_bundle(
    registration_run: Path | str,
    output_dir: Path | str,
    *,
    semantic_run: Path | str | None = None,
    clusters: int | None = None,
    semantic_geometry: str = "regions",
) -> Path:
    """Export transforms and optional semantic annotations for QuPath.

    Semantic polygons use each source slide's native pixel coordinates, so
    they can be imported directly into the corresponding original image.
    """

    if semantic_geometry not in _SEMANTIC_GEOMETRIES:
        choices = ", ".join(_SEMANTIC_GEOMETRIES)
        raise ValueError(f"semantic_geometry must be one of: {choices}")
    registration_run = Path(registration_run).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    registration_path = registration_run / "registration_result.json"
    registration = json.loads(registration_path.read_text())
    slides = registration.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("registration result contains no slides")
    annotation_dir = output_dir / "annotations"
    semantic_payload = None
    semantic_root = None
    semantic_approval: SemanticApproval | None = None
    registration_approval: RegistrationApproval | None = None
    registration_approval_sha256: str | None = None
    semantic_preflight_fingerprint: str | None = None
    palette: list[str] = []
    if semantic_run is not None:
        semantic_root = Path(semantic_run).expanduser().resolve()
        semantic_payload = validate_semantic_result(semantic_root)
        semantic_approval = validate_semantic_approval(semantic_root)
        (
            semantic_preflight_fingerprint,
            registration_approval,
            registration_approval_sha256,
        ) = _validate_registration_binding(
            registration_path,
            registration,
            semantic_root,
            semantic_payload,
        )
        available = tuple(int(value) for value in semantic_payload["cluster_counts"])
        selected = int(
            semantic_payload.get("selected_k", semantic_payload["primary_clusters"])
        )
        clusters = selected if clusters is None else clusters
        if clusters not in available:
            raise ValueError(f"K={clusters} is unavailable; choose one of {available}")
        palette = [str(value) for value in semantic_payload["palette"]]
        if not palette:
            raise ValueError("semantic palette is empty")
        annotation_dir = annotation_dir / (
            f"{semantic_payload['fingerprint']}-k{clusters}-"
            f"{_SEMANTIC_GEOMETRY_VERSIONS[semantic_geometry]}"
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            (
                registration_approval,
                registration_approval_sha256,
            ) = _validate_registration_approval(registration_run)
        except FileNotFoundError as error:
            raise ValueError(
                "QuPath registration export requires a sealed registration approval"
            ) from error
        output_dir.mkdir(parents=True, exist_ok=True)

    semantic_by_id = (
        {str(row["id"]): row for row in semantic_payload["slides"]}
        if semantic_payload is not None
        else {}
    )
    slide_rows: list[dict[str, object]] = []
    for order, slide in enumerate(slides, start=1):
        source = Path(str(slide["path"])).expanduser().resolve()
        slide_id = source.name
        geometry = slide.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"registration geometry is missing for {slide_id}")
        row: dict[str, object] = {
            "order": order,
            "id": slide_id,
            "source_uri": source.as_uri(),
            "is_reference": bool(slide.get("is_reference")),
            "aligned_to": (
                Path(str(slide["aligned_to"])).name if slide.get("aligned_to") else None
            ),
            "thumbnail_transform": slide["transform"]["matrix"],
            "thumbnail_geometry": geometry,
            "alignment_metrics": slide.get("alignment_metrics", {}),
        }
        full_warp = slide.get("full_resolution_warp")
        if isinstance(full_warp, dict) and full_warp.get("output_path"):
            row["registered_image_uri"] = (
                Path(str(full_warp["output_path"])).expanduser().resolve().as_uri()
            )
        if semantic_payload is not None and semantic_root is not None:
            semantic_row = semantic_by_id.get(slide_id)
            if semantic_row is None:
                raise ValueError(f"semantic labels are missing for {slide_id}")
            label_path = semantic_root / semantic_row["labels"][str(clusters)]
            relative = (
                annotation_dir.relative_to(output_dir)
                / f"{order:03d}-{_safe_name(source.stem)}.geojson"
            )
            summary = _write_semantic_geojson(
                output_dir / relative,
                slide_id=slide_id,
                label_path=label_path,
                geometry=geometry,
                clusters=int(clusters),
                palette=palette,
                semantic_geometry=semantic_geometry,
            )
            row["semantic_annotations"] = relative.as_posix()
            row["semantic_annotations_sha256"] = _file_sha256(output_dir / relative)
            row["semantic_annotations_bytes"] = (output_dir / relative).stat().st_size
            row["semantic_annotation_classes"] = summary["class_count"]
            row["semantic_annotation_regions"] = summary["region_count"]
            row["semantic_patch_count"] = summary["patch_count"]
        slide_rows.append(row)

    registration_sha256 = _file_sha256(registration_path)
    if registration_approval is not None:
        if registration_approval_sha256 is None:
            raise RuntimeError("registration approval digest is missing")
        if (
            registration_sha256 != registration_approval.registration_result_sha256
            or _file_sha256(registration_run / "registration_approval.json")
            != registration_approval_sha256
        ):
            raise ValueError("registration approval changed during QuPath export")
    manifest = {
        "schema_version": 4 if registration_approval is not None else 3,
        "format": "histopia-qupath-bundle",
        "coordinate_conventions": {
            "semantic_annotations": "source_native_pixels",
            "thumbnail_transform": "moving_thumbnail_to_reference_thumbnail",
            "point_order": "x_y",
        },
        "registration_sha256": registration_sha256,
        "semantic_fingerprint": (
            semantic_payload.get("fingerprint")
            if semantic_payload is not None
            else None
        ),
        "semantic_preflight_fingerprint": semantic_preflight_fingerprint,
        "semantic_approval": (
            {
                "fingerprint": semantic_approval.fingerprint,
                "reviewer": semantic_approval.reviewer,
                "reviewed_at": semantic_approval.reviewed_at,
            }
            if semantic_approval is not None
            else None
        ),
        "semantic_clusters": clusters if semantic_payload is not None else None,
        "semantic_geometry": (
            semantic_geometry if semantic_payload is not None else None
        ),
        "semantic_geometry_version": (
            _SEMANTIC_GEOMETRY_VERSIONS[semantic_geometry]
            if semantic_payload is not None
            else None
        ),
        "slides": slide_rows,
    }
    if registration_approval is not None:
        manifest["registration_approval"] = {
            "approval_sha256": registration_approval_sha256,
            "registration_result_sha256": (
                registration_approval.registration_result_sha256
            ),
            "order_fingerprint": registration_approval.order_fingerprint,
            "reviewer": registration_approval.reviewer,
            "reviewed_at": registration_approval.reviewed_at,
        }
    path = output_dir / "histopia-qupath.json"
    _write_json_atomic(path, manifest, compact=False)
    return path


def _validate_registration_binding(
    registration_path: Path,
    registration: dict[str, object],
    semantic_root: Path,
    semantic_payload: dict[str, object],
) -> tuple[str, RegistrationApproval | None, str | None]:
    preflight_path = semantic_root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if not isinstance(preflight, dict):
        raise ValueError("semantic preflight root must be an object")
    schema = preflight.get("schema_version")
    if schema not in {1, 2, 3}:
        raise ValueError("semantic preflight schema is unsupported")
    fingerprint = preflight.get("fingerprint")
    provenance = semantic_payload.get("feature_provenance")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(provenance, dict)
        or provenance.get("preflight_fingerprint") != fingerprint
    ):
        raise ValueError("semantic preflight fingerprint is stale")
    core = {
        "schema_version": schema,
        "registration_result_sha256": preflight.get("registration_result_sha256"),
        "order_review_fingerprint": preflight.get("order_review_fingerprint"),
        "reference_slide": preflight.get("reference_slide"),
        "slides": _portable_preflight_slides(preflight.get("slides")),
    }
    if schema == 3:
        core["registration_approval_sha256"] = preflight.get(
            "registration_approval_sha256"
        )
    if _json_sha256(core) != fingerprint:
        raise ValueError("semantic preflight record is stale")
    if core["registration_result_sha256"] != _file_sha256(registration_path):
        raise ValueError("semantic atlas belongs to a different registration result")
    slides = registration.get("slides")
    if not isinstance(slides, list):
        raise ValueError("registration result contains no slides")
    registration_ids = [
        Path(str(row.get("path", ""))).name for row in slides if isinstance(row, dict)
    ]
    preflight_ids = [str(row.get("slide_name", "")) for row in core["slides"]]
    semantic_ids = [
        str(row.get("id", ""))
        for row in semantic_payload.get("slides", [])
        if isinstance(row, dict)
    ]
    if (
        len(registration_ids) != len(slides)
        or any(not value for value in registration_ids)
        or len(set(registration_ids)) != len(registration_ids)
        or registration_ids != preflight_ids
        or registration_ids != semantic_ids
    ):
        raise ValueError("semantic and registration slide order differs")
    references = [
        slide_id
        for slide_id, row in zip(registration_ids, slides, strict=True)
        if row.get("is_reference")
    ]
    if references != [preflight.get("reference_slide")]:
        raise ValueError("semantic and registration references differ")
    approval = None
    approval_sha256 = None
    if schema == 3:
        expected_approval = core["registration_approval_sha256"]
        if not isinstance(expected_approval, str) or not expected_approval:
            raise ValueError("semantic preflight registration approval is stale")
        approval, approval_sha256 = _validate_registration_approval(
            registration_path.parent
        )
        if expected_approval != approval_sha256:
            raise ValueError("semantic preflight registration approval is stale")
        if approval.registration_result_sha256 != core["registration_result_sha256"]:
            raise ValueError("registration approval differs from semantic preflight")
    return fingerprint, approval, approval_sha256


def _validate_registration_approval(
    run_dir: Path,
) -> tuple[RegistrationApproval, str]:
    from histopia.registration._approval import validate_registration_approval

    path = run_dir / "registration_approval.json"
    before = _file_sha256(path)
    approval = validate_registration_approval(run_dir)
    after = _file_sha256(path)
    if before != after:
        raise ValueError("registration approval changed during validation")
    return approval, after


def _portable_preflight_slides(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("semantic preflight contains no slides")
    portable: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("semantic preflight slides must be objects")
        portable.append(
            {key: item for key, item in row.items() if key != "source_path"}
        )
    return portable


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_semantic_geojson(
    path: Path,
    *,
    slide_id: str,
    label_path: Path,
    geometry: dict[str, Any],
    clusters: int,
    palette: list[str],
    semantic_geometry: str,
) -> dict[str, int]:
    with np.load(label_path, allow_pickle=False) as labels_data:
        raw_labels = np.asarray(labels_data["labels"])
        raw_grid_rc = np.asarray(labels_data["grid_rc"])
        raw_grid_shape = np.asarray(labels_data["grid_shape"])
        patch_size_px = np.asarray(labels_data["patch_size_px"])
        analysis_mpp = np.asarray(labels_data["analysis_mpp"])
    if (
        raw_labels.ndim != 1
        or not np.issubdtype(raw_labels.dtype, np.integer)
        or raw_grid_rc.ndim != 2
        or raw_grid_rc.shape[1] != 2
        or not np.issubdtype(raw_grid_rc.dtype, np.integer)
        or raw_grid_shape.shape != (2,)
        or not np.issubdtype(raw_grid_shape.dtype, np.integer)
        or patch_size_px.shape != ()
        or analysis_mpp.shape != ()
    ):
        raise ValueError(f"semantic label grid is invalid for {slide_id}")
    labels = raw_labels.astype(np.int64, copy=False)
    grid_rc = raw_grid_rc.astype(np.int64, copy=False)
    grid_shape = tuple(int(value) for value in raw_grid_shape)
    patch_size = float(patch_size_px)
    analysis_scale = float(analysis_mpp)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= clusters):
        raise ValueError(f"semantic labels are outside K={clusters} for {slide_id}")
    if (
        len(labels) != len(grid_rc)
        or not len(labels)
        or len(np.unique(grid_rc, axis=0)) != len(grid_rc)
        or min(grid_shape) <= 0
        or np.any(grid_rc < 0)
        or np.any(grid_rc[:, 0] >= grid_shape[0])
        or np.any(grid_rc[:, 1] >= grid_shape[1])
        or not math.isfinite(patch_size)
        or not patch_size.is_integer()
        or patch_size <= 0
        or not math.isfinite(analysis_scale)
        or analysis_scale <= 0
    ):
        raise ValueError(f"semantic label grid is invalid for {slide_id}")
    mpp = geometry.get("mpp_xy")
    native_shape = geometry.get("native_shape")
    content_bbox = geometry.get("content_bbox_xywh")
    if (
        not isinstance(mpp, list)
        or len(mpp) != 2
        or not isinstance(native_shape, list)
        or len(native_shape) != 2
        or not isinstance(content_bbox, list)
        or len(content_bbox) != 4
    ):
        raise ValueError(f"calibrated native geometry is required for {slide_id}")
    mpp_xy = tuple(float(value) for value in mpp)
    native_height, native_width = (int(value) for value in native_shape)
    content_x, content_y, content_width, content_height = (
        int(value) for value in content_bbox
    )
    if (
        not np.all(np.isfinite(mpp_xy))
        or min(mpp_xy) <= 0
        or min(native_height, native_width, content_width, content_height) <= 0
        or content_x < 0
        or content_y < 0
        or content_x + content_width > native_width
        or content_y + content_height > native_height
    ):
        raise ValueError(f"calibrated native geometry is required for {slide_id}")
    patch_um = patch_size * analysis_scale
    native_patch_width = max(1, round(patch_um / mpp_xy[0]))
    native_patch_height = max(1, round(patch_um / mpp_xy[1]))
    expected_grid_shape = (
        content_height // native_patch_height,
        content_width // native_patch_width,
    )
    if grid_shape != expected_grid_shape:
        raise ValueError(
            f"semantic label grid differs from registration geometry for {slide_id}"
        )
    native_xy = np.column_stack(
        (
            content_x + grid_rc[:, 1] * native_patch_width + native_patch_width / 2,
            content_y + grid_rc[:, 0] * native_patch_height + native_patch_height / 2,
        )
    )
    half_width = native_patch_width / 2
    half_height = native_patch_height / 2
    features_json = []
    region_count = 0
    for label in range(clusters):
        selected = labels == label
        points = native_xy[selected]
        if not len(points):
            continue
        rectangles = (
            _coalesce_patch_rectangles(
                grid_rc[selected],
                points,
                half_width=half_width,
                half_height=half_height,
                native_width=native_width,
                native_height=native_height,
            )
            if semantic_geometry == "regions"
            else _tile_rectangles(
                points,
                half_width=half_width,
                half_height=half_height,
                native_width=native_width,
                native_height=native_height,
            )
        )
        polygons = [_rectangle_polygon(rectangle) for rectangle in rectangles]
        region_count += len(rectangles)
        color = _hex_color(palette[label % len(palette)])
        features_json.append(
            {
                "type": "Feature",
                "id": f"histopia-k{clusters}-class-{label + 1}",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": polygons,
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": f"Histopia K{clusters} / Region {label + 1}",
                        "color": color,
                    },
                    "histopia": {
                        "cluster": label,
                        "clusters": clusters,
                        "patch_count": int(len(points)),
                        "region_count": len(rectangles),
                        "geometry": semantic_geometry,
                        "slide_id": slide_id,
                    },
                },
            }
        )
    _write_json_atomic(
        path,
        {
            "type": "FeatureCollection",
            "histopia_schema_version": 2,
            "histopia": {
                "slide_id": slide_id,
                "clusters": clusters,
                "geometry": semantic_geometry,
                "geometry_version": _SEMANTIC_GEOMETRY_VERSIONS[semantic_geometry],
                "patch_count": int(len(labels)),
                "region_count": region_count,
            },
            "features": features_json,
        },
        compact=True,
    )
    return {
        "class_count": len(features_json),
        "patch_count": int(len(labels)),
        "region_count": region_count,
    }


def _tile_rectangles(
    native_xy: np.ndarray,
    *,
    half_width: float,
    half_height: float,
    native_width: int,
    native_height: int,
) -> list[tuple[float, float, float, float]]:
    return [
        _bounded_rectangle(
            float(x),
            float(y),
            float(x),
            float(y),
            half_width=half_width,
            half_height=half_height,
            native_width=native_width,
            native_height=native_height,
        )
        for x, y in native_xy
    ]


def _coalesce_patch_rectangles(
    grid_rc: np.ndarray,
    native_xy: np.ndarray,
    *,
    half_width: float,
    half_height: float,
    native_width: int,
    native_height: int,
) -> list[tuple[float, float, float, float]]:
    """Merge equal-label horizontal runs across identical adjacent rows."""

    rows: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for (row, col), (x, y) in zip(grid_rc, native_xy, strict=True):
        rows[int(row)].append((int(col), float(x), float(y)))
    active: dict[tuple[int, int], tuple[int, tuple[float, float, float, float]]] = {}
    completed: list[tuple[float, float, float, float]] = []
    previous_row: int | None = None
    for row in sorted(rows):
        runs = _row_runs(rows[row])
        if previous_row is None or row != previous_row + 1:
            completed.extend(rectangle for _, rectangle in active.values())
            active = {}
        next_active: dict[
            tuple[int, int], tuple[int, tuple[float, float, float, float]]
        ] = {}
        for start_col, end_col, start_x, end_x, y in runs:
            key = (start_col, end_col)
            rectangle = _bounded_rectangle(
                start_x,
                y,
                end_x,
                y,
                half_width=half_width,
                half_height=half_height,
                native_width=native_width,
                native_height=native_height,
            )
            prior = active.pop(key, None)
            if prior is not None and _same_horizontal_bounds(prior[1], rectangle):
                left, top, right, _ = prior[1]
                rectangle = (left, top, right, rectangle[3])
            next_active[key] = (row, rectangle)
        completed.extend(rectangle for _, rectangle in active.values())
        active = next_active
        previous_row = row
    completed.extend(rectangle for _, rectangle in active.values())
    return completed


def _row_runs(
    cells: list[tuple[int, float, float]],
) -> list[tuple[int, int, float, float, float]]:
    ordered = sorted(cells)
    runs: list[tuple[int, int, float, float, float]] = []
    start_col, start_x, y = ordered[0]
    end_col, end_x = start_col, start_x
    for col, x, cell_y in ordered[1:]:
        if col == end_col + 1 and np.isclose(cell_y, y, rtol=0, atol=1e-6):
            end_col, end_x = col, x
            continue
        runs.append((start_col, end_col, start_x, end_x, y))
        start_col, end_col = col, col
        start_x, end_x, y = x, x, cell_y
    runs.append((start_col, end_col, start_x, end_x, y))
    return runs


def _bounded_rectangle(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    *,
    half_width: float,
    half_height: float,
    native_width: int,
    native_height: int,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, start_x - half_width),
        max(0.0, start_y - half_height),
        min(float(native_width), end_x + half_width),
        min(float(native_height), end_y + half_height),
    )


def _same_horizontal_bounds(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return bool(
        np.isclose(first[0], second[0], rtol=0, atol=1e-6)
        and np.isclose(first[2], second[2], rtol=0, atol=1e-6)
    )


def _rectangle_polygon(
    rectangle: tuple[float, float, float, float],
) -> list[list[list[float | int]]]:
    left, top, right, bottom = (_json_coordinate(value) for value in rectangle)
    return [
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
            [left, top],
        ]
    ]


def _json_coordinate(value: float) -> float | int:
    rounded = round(value)
    return rounded if np.isclose(value, rounded, rtol=0, atol=1e-9) else value


def _hex_color(value: str) -> list[int]:
    match = value.removeprefix("#")
    if len(match) != 6:
        raise ValueError(f"invalid semantic palette color: {value!r}")
    try:
        return [int(match[index : index + 2], 16) for index in (0, 2, 4)]
    except ValueError as exc:
        raise ValueError(f"invalid semantic palette color: {value!r}") from exc


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(
    path: Path,
    payload: dict[str, object],
    *,
    compact: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if compact:
            text = json.dumps(payload, separators=(",", ":")) + "\n"
        else:
            text = json.dumps(payload, indent=2) + "\n"
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
