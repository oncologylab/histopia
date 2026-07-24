"""Export validated Histopia results for import by the QuPath extension."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from histopia.semantic._result_validation import validate_semantic_result

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
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = output_dir / "annotations"
    semantic_payload = None
    semantic_root = None
    feature_paths: dict[str, Path] = {}
    palette: list[str] = []
    if semantic_run is not None:
        semantic_root = Path(semantic_run).expanduser().resolve()
        semantic_payload = validate_semantic_result(semantic_root)
        available = tuple(int(value) for value in semantic_payload["cluster_counts"])
        selected = int(
            semantic_payload.get("selected_k", semantic_payload["primary_clusters"])
        )
        clusters = selected if clusters is None else clusters
        if clusters not in available:
            raise ValueError(f"K={clusters} is unavailable; choose one of {available}")
        feature_paths = _index_feature_paths(semantic_root / "features")
        palette = [str(value) for value in semantic_payload["palette"]]
        if not palette:
            raise ValueError("semantic palette is empty")
        annotation_dir = annotation_dir / (
            f"{semantic_payload['fingerprint']}-k{clusters}-"
            f"{_SEMANTIC_GEOMETRY_VERSIONS[semantic_geometry]}"
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)

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
            feature_path = feature_paths.get(slide_id)
            if semantic_row is None or feature_path is None:
                raise ValueError(f"semantic features are missing for {slide_id}")
            label_path = semantic_root / semantic_row["labels"][str(clusters)]
            relative = (
                annotation_dir.relative_to(output_dir)
                / f"{order:03d}-{_safe_name(source.stem)}.geojson"
            )
            summary = _write_semantic_geojson(
                output_dir / relative,
                slide_id=slide_id,
                feature_path=feature_path,
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

    manifest = {
        "schema_version": 2,
        "format": "histopia-qupath-bundle",
        "coordinate_conventions": {
            "semantic_annotations": "source_native_pixels",
            "thumbnail_transform": "moving_thumbnail_to_reference_thumbnail",
            "point_order": "x_y",
        },
        "registration_sha256": _file_sha256(registration_path),
        "semantic_fingerprint": (
            semantic_payload.get("fingerprint")
            if semantic_payload is not None
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
    path = output_dir / "histopia-qupath.json"
    _write_json_atomic(path, manifest, compact=False)
    return path


def _index_feature_paths(feature_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(feature_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                slide_id = str(data["slide_id"].item())
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"invalid semantic feature artifact: {path.name}") from exc
        if slide_id in indexed:
            raise ValueError(f"duplicate semantic features for {slide_id}")
        indexed[slide_id] = path
    return indexed


def _write_semantic_geojson(
    path: Path,
    *,
    slide_id: str,
    feature_path: Path,
    label_path: Path,
    geometry: dict[str, Any],
    clusters: int,
    palette: list[str],
    semantic_geometry: str,
) -> dict[str, int]:
    with np.load(feature_path, allow_pickle=False) as features:
        native_xy = np.asarray(features["native_xy"], dtype=np.float64)
        feature_grid_rc = np.asarray(features["grid_rc"], dtype=np.int32)
        feature_slide_id = str(features["slide_id"].item())
    with np.load(label_path, allow_pickle=False) as labels_data:
        labels = np.asarray(labels_data["labels"], dtype=np.int32)
        label_grid_rc = np.asarray(labels_data["grid_rc"], dtype=np.int32)
        patch_um = float(labels_data["patch_size_px"]) * float(
            labels_data["analysis_mpp"]
        )
    if (
        feature_slide_id != slide_id
        or len(native_xy) != len(labels)
        or feature_grid_rc.shape != label_grid_rc.shape
        or not np.array_equal(feature_grid_rc, label_grid_rc)
    ):
        raise ValueError(f"semantic coordinates do not match labels for {slide_id}")
    if (
        native_xy.ndim != 2
        or native_xy.shape[1] != 2
        or not np.all(np.isfinite(native_xy))
        or feature_grid_rc.ndim != 2
        or feature_grid_rc.shape[1] != 2
        or len(np.unique(feature_grid_rc, axis=0)) != len(feature_grid_rc)
    ):
        raise ValueError(f"semantic coordinates are invalid for {slide_id}")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= clusters):
        raise ValueError(f"semantic labels are outside K={clusters} for {slide_id}")
    mpp = geometry.get("mpp_xy")
    native_shape = geometry.get("native_shape")
    if (
        not isinstance(mpp, list)
        or len(mpp) != 2
        or not isinstance(native_shape, list)
        or len(native_shape) != 2
    ):
        raise ValueError(f"calibrated native geometry is required for {slide_id}")
    half_width = max(1, round(patch_um / float(mpp[0]))) / 2
    half_height = max(1, round(patch_um / float(mpp[1]))) / 2
    native_height, native_width = (int(value) for value in native_shape)
    features_json = []
    region_count = 0
    for label in range(clusters):
        selected = labels == label
        points = native_xy[selected]
        if not len(points):
            continue
        rectangles = (
            _coalesce_patch_rectangles(
                feature_grid_rc[selected],
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
