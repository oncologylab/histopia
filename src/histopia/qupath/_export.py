"""Export validated Histopia results for import by the QuPath extension."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from histopia.semantic._result import validate_semantic_result


def export_qupath_bundle(
    registration_run: Path | str,
    output_dir: Path | str,
    *,
    semantic_run: Path | str | None = None,
    clusters: int | None = None,
) -> Path:
    """Export transforms and optional semantic annotations for QuPath.

    Semantic polygons use each source slide's native pixel coordinates, so
    they can be imported directly into the corresponding original image.
    """

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
                Path("annotations") / f"{order:03d}-{_safe_name(source.stem)}.geojson"
            )
            _write_semantic_geojson(
                output_dir / relative,
                slide_id=slide_id,
                feature_path=feature_path,
                label_path=label_path,
                geometry=geometry,
                clusters=int(clusters),
                palette=palette,
            )
            row["semantic_annotations"] = relative.as_posix()
        slide_rows.append(row)

    manifest = {
        "schema_version": 1,
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
) -> None:
    with np.load(feature_path, allow_pickle=False) as features:
        native_xy = np.asarray(features["native_xy"], dtype=np.float64)
        feature_slide_id = str(features["slide_id"].item())
    with np.load(label_path, allow_pickle=False) as labels_data:
        labels = np.asarray(labels_data["labels"], dtype=np.int32)
        patch_um = float(labels_data["patch_size_px"]) * float(
            labels_data["analysis_mpp"]
        )
    if feature_slide_id != slide_id or len(native_xy) != len(labels):
        raise ValueError(f"semantic coordinates do not match labels for {slide_id}")
    mpp = geometry.get("mpp_xy")
    native_shape = geometry.get("native_shape")
    if (
        not isinstance(mpp, list)
        or len(mpp) != 2
        or not isinstance(native_shape, list)
        or len(native_shape) != 2
    ):
        raise ValueError(f"calibrated native geometry is required for {slide_id}")
    half_width = patch_um / float(mpp[0]) / 2
    half_height = patch_um / float(mpp[1]) / 2
    native_height, native_width = (int(value) for value in native_shape)
    features_json = []
    for label in range(clusters):
        points = native_xy[labels == label]
        if not len(points):
            continue
        polygons = [
            [
                [
                    [max(0.0, x - half_width), max(0.0, y - half_height)],
                    [
                        min(float(native_width), x + half_width),
                        max(0.0, y - half_height),
                    ],
                    [
                        min(float(native_width), x + half_width),
                        min(float(native_height), y + half_height),
                    ],
                    [
                        max(0.0, x - half_width),
                        min(float(native_height), y + half_height),
                    ],
                    [max(0.0, x - half_width), max(0.0, y - half_height)],
                ]
            ]
            for x, y in points
        ]
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
                        "slide_id": slide_id,
                    },
                },
            }
        )
    _write_json_atomic(
        path,
        {
            "type": "FeatureCollection",
            "histopia_schema_version": 1,
            "features": features_json,
        },
        compact=True,
    )


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
