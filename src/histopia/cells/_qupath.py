"""Bounded QuPath detection export for reviewed cell-label regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.cells._result import validate_cell_result


def export_qupath_cell_roi(
    run_dir: Path | str,
    section: str,
    roi_xywh: tuple[int, int, int, int],
    output: Path | str,
    *,
    simplify_tolerance_px: float = 1.0,
    max_cells: int = 50_000,
) -> Path:
    """Export cells intersecting one native-pixel ROI as QuPath detections."""

    if max_cells <= 0:
        raise ValueError("max_cells must be positive")
    if simplify_tolerance_px < 0:
        raise ValueError("simplify_tolerance_px must be nonnegative")
    x, y, width, height = roi_xywh
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI must be a positive native-pixel rectangle")
    root = Path(run_dir)
    result = validate_cell_result(root)
    row = next(
        (
            value
            for value in result["slides"]
            if isinstance(value, dict) and value.get("section") == section
        ),
        None,
    )
    if row is None:
        raise ValueError(f"unknown cell-result section: {section}")
    bbox = row.get("content_bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("cell result has no native content-box geometry")
    crop_x, crop_y, crop_width, crop_height = (int(value) for value in bbox)
    left = max(x, crop_x)
    top = max(y, crop_y)
    right = min(x + width, crop_x + crop_width)
    bottom = min(y + height, crop_y + crop_height)
    if right <= left or bottom <= top:
        raise ValueError("ROI does not intersect the segmented tissue content box")
    labels = _read_label_crop(
        root / str(row["labels"]),
        left - crop_x,
        top - crop_y,
        right - left,
        bottom - top,
    )
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    if len(label_ids) > max_cells:
        raise ValueError(
            f"ROI contains {len(label_ids):,} cells; reduce the ROI below "
            f"the {max_cells:,}-cell import limit"
        )
    features = _cell_features(
        labels,
        label_ids,
        offset_xy=(left, top),
        section=section,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    return write_json_atomic(
        output,
        {
            "type": "FeatureCollection",
            "histopia_schema_version": 1,
            "histopia": {
                "workflow": "cells",
                "cell_result_fingerprint": result["fingerprint"],
                "section": section,
                "roi_xywh": [x, y, width, height],
                "cell_count": len(features),
            },
            "features": features,
        },
        indent=None,
        separators=(",", ":"),
    )


def _cell_features(
    labels: np.ndarray,
    label_ids: np.ndarray,
    *,
    offset_xy: tuple[int, int],
    section: str,
    simplify_tolerance_px: float,
) -> list[dict[str, object]]:
    from scipy import ndimage as ndi
    from skimage.measure import approximate_polygon, find_contours

    output = []
    objects = ndi.find_objects(labels)
    offset_x, offset_y = offset_xy
    for label_id in label_ids:
        bbox = objects[int(label_id) - 1]
        if bbox is None:
            continue
        region = labels[bbox] == label_id
        contours = find_contours(np.pad(region, 1), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len) - 1
        if simplify_tolerance_px > 0:
            contour = approximate_polygon(contour, tolerance=simplify_tolerance_px)
        if len(contour) < 3:
            continue
        y0, x0 = bbox[0].start, bbox[1].start
        ring = [
            [
                round(float(offset_x + x0 + point[1]), 3),
                round(float(offset_y + y0 + point[0]), 3),
            ]
            for point in contour
        ]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        output.append(
            {
                "type": "Feature",
                "id": f"histopia-cell-{section}-{int(label_id)}",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType": "detection",
                    "classification": {
                        "name": "Histopia cell",
                        "color": [230, 45, 35],
                    },
                    "histopia": {
                        "section": section,
                        "label": int(label_id),
                        "clipped_to_roi": bool(
                            bbox[0].start == 0
                            or bbox[1].start == 0
                            or bbox[0].stop == labels.shape[0]
                            or bbox[1].stop == labels.shape[1]
                        ),
                    },
                },
            }
        )
    return output


def _read_label_crop(path: Path, x: int, y: int, width: int, height: int) -> np.ndarray:
    try:
        import pyvips
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "QuPath cell export requires the 'cells' dependencies"
        ) from error
    image: Any = pyvips.Image.new_from_file(str(path), access="random")
    tile = image.crop(x, y, width, height)
    if tile.bands > 1:
        tile = tile[0]
    if tile.format != "uint":
        tile = tile.cast("uint")
    return np.frombuffer(tile.write_to_memory(), dtype=np.uint32).reshape(height, width)
