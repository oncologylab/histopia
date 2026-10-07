"""Exact-support composition for visually reviewed dense-cell recovery.

A whole-section post-filter can change otherwise accepted labels outside a
small recovery region.  This module keeps the validated source canvas
immutable and admits only candidate instances that are wholly contained in an
explicit support mask, do not overlap any source label, and satisfy the source
minimum-area rule.  It therefore turns a reviewed recovery into a strictly
additive, spatially bounded operation rather than another global refilter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION = 96
BOUNDED_ADDITIVE_RECOVERY_METHOD_PROFILE = (
    "combined-containment-cpsam-stardist-bounded-addition-v3"
)
BOUNDED_ADDITIVE_RECOVERY_SCOPE = "exact-support-new-instance-addition-v1"
_RECOVERY_TILE_KEY = re.compile(r"^r[0-9]{4}_c[0-9]{4}_y[0-9]+_x[0-9]+\.npz$")


@dataclass(frozen=True, slots=True)
class BoundedAdditiveSelection:
    """Deterministic decision for candidate IDs represented by count arrays."""

    selected_candidate_ids: tuple[int, ...]
    selected_areas_px: tuple[int, ...]
    candidate_instance_count: int
    rejected_below_minimum_count: int
    rejected_outside_support_count: int
    rejected_source_overlap_count: int

    @property
    def selected_instance_count(self) -> int:
        """Return the number of admitted candidate instances."""

        return len(self.selected_candidate_ids)

    @property
    def added_pixels(self) -> int:
        """Return the exact number of pixels admitted into the source canvas."""

        return sum(self.selected_areas_px)

    @property
    def selected_ids_sha256(self) -> str:
        """Seal the ordered candidate IDs without exposing a large JSON list."""

        values = np.asarray(self.selected_candidate_ids, dtype="<u8")
        return hashlib.sha256(values.tobytes()).hexdigest()


def select_bounded_additive_instances(
    total_pixels: np.ndarray,
    support_pixels: np.ndarray,
    source_overlap_pixels: np.ndarray,
    novel_support_pixels: np.ndarray,
    *,
    minimum_area_px: int,
) -> BoundedAdditiveSelection:
    """Select wholly new candidate instances using blockwise count summaries.

    Arrays are indexed by candidate label ID.  ID zero is background.  A
    candidate enters the decision set only when it contributes at least one
    previously unlabeled pixel inside the explicit support.  Rejection classes
    are mutually exclusive and evaluated in this order: below minimum area,
    outside support, then source overlap.
    """

    arrays = tuple(
        _nonnegative_counts(value, name)
        for value, name in (
            (total_pixels, "total_pixels"),
            (support_pixels, "support_pixels"),
            (source_overlap_pixels, "source_overlap_pixels"),
            (novel_support_pixels, "novel_support_pixels"),
        )
    )
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("bounded-additive count arrays must have identical shapes")
    if not isinstance(minimum_area_px, int) or isinstance(minimum_area_px, bool):
        raise TypeError("minimum_area_px must be an integer")
    if minimum_area_px <= 0:
        raise ValueError("minimum_area_px must be positive")

    total, support, overlap, novel = arrays
    if np.any(support > total) or np.any(overlap > total) or np.any(novel > support):
        raise ValueError("bounded-additive count summaries are inconsistent")
    candidate = novel > 0
    candidate[0] = False
    below = candidate & (total < minimum_area_px)
    outside = candidate & ~below & (support != total)
    overlapping = candidate & ~below & ~outside & (overlap > 0)
    selected = candidate & ~below & ~outside & ~overlapping
    ids = np.flatnonzero(selected).astype(np.int64, copy=False)
    areas = total[ids].astype(np.int64, copy=False)
    return BoundedAdditiveSelection(
        selected_candidate_ids=tuple(int(value) for value in ids),
        selected_areas_px=tuple(int(value) for value in areas),
        candidate_instance_count=int(np.count_nonzero(candidate)),
        rejected_below_minimum_count=int(np.count_nonzero(below)),
        rejected_outside_support_count=int(np.count_nonzero(outside)),
        rejected_source_overlap_count=int(np.count_nonzero(overlapping)),
    )


def bounded_additive_label_lookup(
    selection: BoundedAdditiveSelection,
    *,
    maximum_candidate_id: int,
    first_output_id: int,
) -> np.ndarray:
    """Map admitted candidate IDs to deterministic, collision-free uint32 IDs."""

    for value, name in (
        (maximum_candidate_id, "maximum_candidate_id"),
        (first_output_id, "first_output_id"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if maximum_candidate_id < 0 or first_output_id <= 0:
        raise ValueError("bounded-additive label bounds are invalid")
    ids = np.asarray(selection.selected_candidate_ids, dtype=np.int64)
    if ids.size and (ids[0] <= 0 or ids[-1] > maximum_candidate_id):
        raise ValueError("selected candidate ID is outside the lookup range")
    if ids.size and np.any(ids[1:] <= ids[:-1]):
        raise ValueError("selected candidate IDs must be strictly increasing")
    last = first_output_id + len(ids) - 1
    if last > np.iinfo(np.uint32).max:
        raise OverflowError("bounded-additive output IDs exceed uint32")
    lookup = np.zeros(maximum_candidate_id + 1, dtype=np.uint32)
    if ids.size:
        lookup[ids] = np.arange(
            first_output_id,
            first_output_id + len(ids),
            dtype=np.uint32,
        )
    return lookup


def apply_bounded_additions(
    source_labels: np.ndarray,
    candidate_labels: np.ndarray,
    support_mask: np.ndarray,
    candidate_to_output: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return a source-preserving block with only mapped support additions."""

    source = np.asarray(source_labels)
    candidate = np.asarray(candidate_labels)
    support = np.asarray(support_mask, dtype=bool)
    lookup = np.asarray(candidate_to_output)
    if source.shape != candidate.shape or source.shape != support.shape:
        raise ValueError("bounded-additive block geometry differs")
    if source.ndim != 2:
        raise ValueError("bounded-additive blocks must be two-dimensional")
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < 0):
        raise ValueError("source labels must be nonnegative integers")
    if not np.issubdtype(candidate.dtype, np.integer) or np.any(candidate < 0):
        raise ValueError("candidate labels must be nonnegative integers")
    if lookup.ndim != 1 or not np.issubdtype(lookup.dtype, np.integer):
        raise ValueError("candidate lookup must be a one-dimensional integer array")
    maximum = int(candidate.max(initial=0))
    if maximum >= len(lookup):
        raise ValueError("candidate label exceeds the lookup range")
    mapped = lookup[candidate]
    selected = mapped > 0
    if np.any(selected & ~support):
        raise ValueError("selected candidate extends outside bounded support")
    if np.any(selected & (source > 0)):
        raise ValueError("selected candidate overlaps a source instance")
    output = np.asarray(source, dtype=np.uint32).copy()
    output[selected] = mapped[selected].astype(np.uint32, copy=False)
    return output, int(np.count_nonzero(selected))


def bounded_additive_recovery_parameters(*, minimum_area_px: int) -> dict[str, object]:
    """Return path-free immutable parameters for the exact additive policy."""

    if not isinstance(minimum_area_px, int) or isinstance(minimum_area_px, bool):
        raise TypeError("minimum_area_px must be an integer")
    if minimum_area_px <= 0:
        raise ValueError("minimum_area_px must be positive")
    return {
        "schema_version": 1,
        "scope": BOUNDED_ADDITIVE_RECOVERY_SCOPE,
        "minimum_area_px": minimum_area_px,
        "candidate_rule": "novel-pixel-inside-explicit-support-v1",
        "containment_rule": "all-candidate-pixels-inside-support-v1",
        "source_overlap_rule": "zero-source-overlap-pixels-v1",
        "label_assignment": "ascending-candidate-id-to-new-uint32-id-v1",
        "source_mutation": "none",
    }


def validate_bounded_additive_recovery_provenance(
    value: object,
    subset_source: object,
    *,
    section: object,
    source_identity: object,
) -> dict[str, object]:
    """Validate the complete path-free seal for a bounded additive result."""

    recovery = _mapping(value, "bounded-additive recovery")
    if set(recovery) != {
        "schema_version",
        "scope",
        "parameters",
        "source",
        "candidate",
        "section",
        "source_identity",
        "support",
        "selection",
    }:
        raise ValueError("bounded-additive recovery fields are stale")
    if (
        recovery.get("schema_version") != 1
        or recovery.get("scope") != BOUNDED_ADDITIVE_RECOVERY_SCOPE
        or recovery.get("parameters")
        != bounded_additive_recovery_parameters(minimum_area_px=15)
        or recovery.get("section") != section
        or recovery.get("source_identity") != source_identity
    ):
        raise ValueError("bounded-additive recovery provenance is stale")

    source = _mapping(recovery.get("source"), "bounded-additive source")
    candidate = _mapping(recovery.get("candidate"), "bounded-additive candidate")
    source_fields = {
        "result_fingerprint",
        "algorithm_version",
        "profile_fingerprint",
        "preflight_fingerprint",
        "section_fingerprint",
        "labels_sha256",
        "qc_sha256",
    }
    candidate_fields = {
        *source_fields,
        "recovery_manifest_fingerprint",
    }
    if (
        set(source) != source_fields
        or source.get("algorithm_version") != 75
        or any(
            not _is_sha256(source.get(key))
            for key in source_fields - {"algorithm_version"}
        )
        or set(candidate) != candidate_fields
        or candidate.get("algorithm_version") != 95
        or any(
            not _is_sha256(candidate.get(key))
            for key in candidate_fields - {"algorithm_version"}
        )
    ):
        raise ValueError("bounded-additive source or candidate seal is stale")

    support = _mapping(recovery.get("support"), "bounded-additive support")
    tiles = support.get("tiles")
    if (
        set(support)
        != {
            "kind",
            "tiles",
            "fingerprint",
            "union_bbox_xywh",
            "union_pixels",
        }
        or support.get("kind") != "exact-recovery-tile-union-v1"
        or not isinstance(tiles, list)
        or not tiles
        or any(not isinstance(tile, dict) for tile in tiles)
    ):
        raise ValueError("bounded-additive support seal is stale")
    normalized_tiles: list[dict[str, object]] = []
    for raw_tile in tiles:
        assert isinstance(raw_tile, dict)
        if set(raw_tile) != {
            "tile",
            "x",
            "y",
            "width",
            "height",
            "recovery_fingerprint",
            "file_sha256",
        }:
            raise ValueError("bounded-additive support tile fields are stale")
        key = raw_tile.get("tile")
        coordinates = tuple(raw_tile.get(name) for name in ("x", "y"))
        dimensions = tuple(raw_tile.get(name) for name in ("width", "height"))
        if (
            not isinstance(key, str)
            or _RECOVERY_TILE_KEY.fullmatch(key) is None
            or any(not _nonnegative_int(value) for value in coordinates)
            or any(not _positive_int(value) for value in dimensions)
            or not _is_sha256(raw_tile.get("recovery_fingerprint"))
            or not _is_sha256(raw_tile.get("file_sha256"))
        ):
            raise ValueError("bounded-additive support tile seal is stale")
        normalized_tiles.append(dict(raw_tile))
    keys = [str(tile["tile"]) for tile in normalized_tiles]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("bounded-additive support tile order is stale")
    support_core = {
        "kind": "exact-recovery-tile-union-v1",
        "tiles": normalized_tiles,
    }
    expected_bbox, expected_area = bounded_support_geometry(normalized_tiles)
    if (
        support.get("fingerprint") != _json_sha256(support_core)
        or support.get("union_bbox_xywh") != expected_bbox
        or support.get("union_pixels") != expected_area
    ):
        raise ValueError("bounded-additive support geometry is stale")

    selection = _mapping(recovery.get("selection"), "bounded-additive selection")
    expected_selection_fields = {
        "candidate_instance_count",
        "selected_instance_count",
        "rejected_below_minimum_count",
        "rejected_outside_support_count",
        "rejected_source_overlap_count",
        "selected_ids_sha256",
        "added_pixels",
        "removed_pixels",
        "changed_outside_support_pixels",
        "first_output_id",
        "last_output_id",
    }
    counts = tuple(
        selection.get(key)
        for key in (
            "candidate_instance_count",
            "selected_instance_count",
            "rejected_below_minimum_count",
            "rejected_outside_support_count",
            "rejected_source_overlap_count",
        )
    )
    if (
        set(selection) != expected_selection_fields
        or any(not _nonnegative_int(value) for value in counts)
        or int(counts[0]) <= 0
        or int(counts[1]) <= 0
        or sum(int(value) for value in counts[1:]) != int(counts[0])
        or not _is_sha256(selection.get("selected_ids_sha256"))
        or not _positive_int(selection.get("added_pixels"))
        or int(selection["added_pixels"]) < 15 * int(counts[1])
        or selection.get("removed_pixels") != 0
        or selection.get("changed_outside_support_pixels") != 0
        or not _positive_int(selection.get("first_output_id"))
        or selection.get("last_output_id")
        != int(selection["first_output_id"]) + int(counts[1]) - 1
    ):
        raise ValueError("bounded-additive selection seal is stale")

    subset = _mapping(subset_source, "bounded-additive subset source")
    if (
        set(subset)
        != {
            "scope",
            "source_result_fingerprint",
            "source_preflight_fingerprint",
            "source_profile_fingerprint",
            "source_section_fingerprint",
            "source_labels_sha256",
        }
        or subset.get("scope") != BOUNDED_ADDITIVE_RECOVERY_SCOPE
        or subset.get("source_result_fingerprint") != source.get("result_fingerprint")
        or subset.get("source_preflight_fingerprint")
        != source.get("preflight_fingerprint")
        or subset.get("source_profile_fingerprint") != source.get("profile_fingerprint")
        or subset.get("source_section_fingerprint") != source.get("section_fingerprint")
        or subset.get("source_labels_sha256") != source.get("labels_sha256")
    ):
        raise ValueError("bounded-additive source binding is stale")
    return dict(recovery)


def validate_bounded_additive_recovery_qc(
    qc: Mapping[str, object],
    recovery: Mapping[str, object],
) -> None:
    """Bind result QC counters to the exact additive selection seal."""

    selection = _mapping(recovery.get("selection"), "bounded-additive selection")
    source = _mapping(recovery.get("source"), "bounded-additive source")
    selected = int(selection["selected_instance_count"])
    added = int(selection["added_pixels"])
    cell_count = qc.get("cell_count")
    foreground = qc.get("foreground_pixels")
    if (
        qc.get("bounded_additive_recovery") != recovery
        or qc.get("tiles_inferred") != 0
        or not _positive_int(qc.get("tiles_reused_from_prior_run"))
        or qc.get("filter_upgrade_from_algorithm_version") != 75
        or qc.get("filter_upgrade_source_labels_sha256") != source.get("labels_sha256")
        or qc.get("bounded_additive_candidate_instances")
        != selection.get("candidate_instance_count")
        or qc.get("bounded_additive_instances_added") != selected
        or qc.get("bounded_additive_pixels_added") != added
        or qc.get("bounded_additive_pixels_removed") != 0
        or qc.get("bounded_additive_changed_outside_support_pixels") != 0
        or not _positive_int(cell_count)
        or int(cell_count) <= selected
        or not _positive_int(foreground)
        or int(foreground) <= added
    ):
        raise ValueError("bounded-additive QC provenance is stale")


def bounded_support_geometry(
    tiles: list[dict[str, object]],
) -> tuple[list[int], int]:
    """Return exact rectangle-union bbox and area without raster allocation."""

    if not tiles:
        raise ValueError("bounded-additive support contains no tiles")
    rectangles = [
        (
            int(tile["x"]),
            int(tile["y"]),
            int(tile["x"]) + int(tile["width"]),
            int(tile["y"]) + int(tile["height"]),
        )
        for tile in tiles
    ]
    xs = sorted(
        {value for rectangle in rectangles for value in (rectangle[0], rectangle[2])}
    )
    area = 0
    for left, right in zip(xs, xs[1:], strict=False):
        intervals = sorted(
            (top, bottom)
            for x0, top, x1, bottom in rectangles
            if x0 < right and x1 > left
        )
        covered = 0
        if intervals:
            start, stop = intervals[0]
            for top, bottom in intervals[1:]:
                if top > stop:
                    covered += stop - start
                    start, stop = top, bottom
                else:
                    stop = max(stop, bottom)
            covered += stop - start
        area += (right - left) * covered
    x0 = min(rectangle[0] for rectangle in rectangles)
    y0 = min(rectangle[1] for rectangle in rectangles)
    x1 = max(rectangle[2] for rectangle in rectangles)
    y1 = max(rectangle[3] for rectangle in rectangles)
    return [x0, y0, x1 - x0, y1 - y0], area


def _nonnegative_counts(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not array.size or np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative counts")
    return array.astype(np.uint64, copy=False)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
    return _nonnegative_int(value) and int(value) > 0


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
