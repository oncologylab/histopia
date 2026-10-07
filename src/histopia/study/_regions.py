"""Immutable connected tissue regions and boundary-overlap expression summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia.protein._vector_transfer import _array_hash, _ids
from histopia.study._manifest import fingerprint


@dataclass(frozen=True)
class TissueRegions:
    labels: np.ndarray  # background 0, connected region IDs 1..N
    regions: tuple[dict[str, object], ...]
    fingerprint: str
    pixel_size_um: float
    origin_um_xy: tuple[float, float]
    coordinate_units: str = "um"
    pixel_size_um_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class RegionAssignments:
    cell_ids: np.ndarray
    region_ids: np.ndarray  # empty string for ambiguous or unsupported cells
    status: np.ndarray
    dominant_fraction: np.ndarray
    region_fingerprint: str
    fingerprint: str


def sample_region_labels(
    regions: TissueRegions, native_xy, native_to_grid
) -> np.ndarray:
    """Sample categorical regions through a homogeneous native-XY → grid transform.

    Grid coordinates denote pixel edges; floor selects the containing pixel.
    Out-of-grid points remain 0. Nonfinite or singular mappings are rejected.
    """
    xy, matrix = np.asarray(native_xy, float), np.asarray(native_to_grid, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError("finite native XY points are required")
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or abs(np.linalg.det(matrix)) < 1e-15
    ):
        raise ValueError("native_to_grid must be an invertible finite 3x3 transform")
    homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("transform maps a point to infinity")
    grid = homogeneous[:, :2] / homogeneous[:, 2, None]
    h, w = regions.labels.shape
    inside = (grid[:, 0] >= 0) & (grid[:, 1] >= 0) & (grid[:, 0] < w) & (grid[:, 1] < h)
    result = np.zeros(len(xy), np.int32)
    c, r = np.floor(grid[inside]).astype(np.int64).T
    result[inside] = regions.labels[r, c]
    return result


def connected_tissue_regions(
    semantic_labels,
    *,
    semantic_fingerprint: str,
    approval_fingerprint: str,
    section_id: str,
    pixel_size_um: float,
    origin_um_xy=(0.0, 0.0),
    pixel_size_um_xy=None,
) -> TissueRegions:
    """Split each semantic class into independent 4-connected components.

    Class labels are zero-based, with -1 for unsupported/background pixels.
    ``pixel_size_um_xy`` preserves rectangular physical grid pixels when given;
    its X spacing must equal ``pixel_size_um``. Existing square-grid identities
    are unchanged. Diagonal contact does not merge regions. IDs bind the exact approved
    semantic raster, section, scale and component membership, not class names.
    """
    from scipy.ndimage import label

    classes = np.asarray(semantic_labels)
    if classes.ndim != 2 or classes.dtype.kind not in "iu" or np.any(classes < -1):
        raise ValueError("semantic labels must be a 2D integer raster, background -1")
    if not all((semantic_fingerprint, approval_fingerprint, section_id)):
        raise ValueError(
            "regions require exact semantic, approval and section bindings"
        )
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    spacing = np.asarray(
        pixel_size_um_xy if pixel_size_um_xy is not None else [pixel_size_um] * 2,
        dtype=float,
    )
    if (
        spacing.shape != (2,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
        or spacing[0] != pixel_size_um
    ):
        raise ValueError("XY pixel spacing must be positive and retain the X scale")
    origin = np.asarray(origin_um_xy, float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must be finite XY coordinates")
    binding = fingerprint(
        {
            "algorithm": "connected-tissue-regions-4-v1",
            "semantic": semantic_fingerprint,
            "approval": approval_fingerprint,
            "section": section_id,
            "classes": _array_hash(classes),
            "pixel_size_um": pixel_size_um,
            "origin_um_xy": origin.tolist(),
            **(
                {"pixel_size_um_xy": spacing.tolist()}
                if spacing[0] != spacing[1]
                else {}
            ),
        }
    )
    labels = np.zeros(classes.shape, np.int32)
    regions = []
    for semantic_class in np.unique(classes[classes >= 0]):
        components, count = label(classes == semantic_class)
        for component in range(1, count + 1):
            rr, cc = np.nonzero(components == component)
            index = len(regions) + 1
            labels[rr, cc] = index
            region_id = (
                "r-"
                + fingerprint(
                    {
                        "atlas": binding,
                        "class": int(semantic_class),
                        "pixels": _array_hash(np.column_stack((rr, cc))),
                    }
                )[:24]
            )
            regions.append(
                {
                    "region_id": region_id,
                    "index": index,
                    "semantic_class": int(semantic_class),
                    "section_id": section_id,
                    "pixels": len(rr),
                    "area_um2": len(rr) * float(np.prod(spacing)),
                    "bbox_rc": [
                        int(rr.min()),
                        int(cc.min()),
                        int(rr.max()) + 1,
                        int(cc.max()) + 1,
                    ],
                    "annotation": "unclassified",
                    "annotation_status": "pending_lab_review",
                }
            )
    return TissueRegions(
        labels,
        tuple(regions),
        binding,
        float(pixel_size_um),
        tuple(origin),
        pixel_size_um_xy=tuple(spacing),
    )


def assign_cells_by_overlap(
    cell_labels,
    region_labels,
    regions: TissueRegions,
    *,
    cell_fingerprint: str,
    minimum_fraction: float = 0.5,
) -> RegionAssignments:
    """Assign native cell instances by dominant boundary overlap on one grid.

    ``region_labels`` must already be sampled into the native cell raster frame
    by a validated transform (nearest-neighbor labels, never linear mixing).
    Every boundary pixel counts in the denominator, including unsupported
    tissue. A strict majority is required. Ties and low coverage are explicit.
    This routine is suitable for bounded fields; whole-slide callers can merge
    per-cell overlap counts using :func:`assign_cells_from_overlap`.
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    cells, raster = np.asarray(cell_labels), np.asarray(region_labels)
    if (
        cells.ndim != 2
        or raster.shape != cells.shape
        or cells.dtype.kind not in "iu"
        or raster.dtype.kind not in "iu"
        or np.any(cells < 0)
        or np.any(raster < 0)
    ):
        raise ValueError("aligned nonnegative integer cell/region rasters are required")
    if np.any(raster > len(regions.regions)):
        raise ValueError("raster contains unknown region indices")
    # Four-connected instance perimeter, including raster-edge background.
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    boundary = (cells > 0) & (
        (minimum_filter(cells, footprint=cross, mode="constant", cval=0) != cells)
        | (maximum_filter(cells, footprint=cross, mode="constant", cval=0) != cells)
    )
    pairs, counts = np.unique(
        np.column_stack((cells[boundary], raster[boundary])), axis=0, return_counts=True
    )
    cell_ids = np.unique(cells[cells > 0])
    overlap = np.zeros((len(cell_ids), len(regions.regions) + 1), np.int64)
    if len(pairs):
        overlap[np.searchsorted(cell_ids, pairs[:, 0]), pairs[:, 1]] = counts
    return assign_cells_from_overlap(
        cell_ids,
        overlap,
        regions,
        cell_fingerprint=cell_fingerprint,
        minimum_fraction=minimum_fraction,
    )


def assign_cells_from_overlap(
    cell_ids,
    boundary_counts,
    regions: TissueRegions,
    *,
    cell_fingerprint: str,
    minimum_fraction: float = 0.5,
) -> RegionAssignments:
    """Consume complete per-cell perimeter counts; column 0 is unsupported.

    Stripe/WSI readers must use a one-pixel halo and count each core pixel once.
    """
    from scipy.sparse import csr_matrix, issparse

    ids = _ids(cell_ids, "cell IDs")
    counts = (
        boundary_counts.tocsr(copy=True)
        if issparse(boundary_counts)
        else csr_matrix(np.asarray(boundary_counts))
    )
    counts.sum_duplicates()
    counts.sort_indices()
    if (
        counts.shape != (len(ids), len(regions.regions) + 1)
        or counts.dtype.kind not in "iu"
        or np.any(counts.data < 0)
    ):
        raise ValueError("boundary counts must align with cells and regions")
    if not 0.5 <= minimum_fraction < 1 or not cell_fingerprint:
        raise ValueError("a bound cell result and majority threshold are required")
    assigned = np.full(len(ids), "", dtype="U26")
    status = np.full(len(ids), "unsupported", dtype="U16")
    fraction = np.zeros(len(ids))
    if regions.regions:
        known = counts[:, 1:]
        total = np.asarray(counts.sum(axis=1)).ravel()
        best = np.asarray(known.argmax(axis=1)).ravel()
        largest = known.max(axis=1).toarray().ravel()
        fraction = np.divide(largest, total, out=np.zeros(len(ids)), where=total > 0)
        coo = known.tocoo()
        tied = (
            np.bincount(
                coo.row, weights=coo.data == largest[coo.row], minlength=len(ids)
            )
            > 1
        )
        accepted = (fraction > minimum_fraction) & ~tied
        status[largest > 0] = "ambiguous"
        status[accepted] = "assigned"
        names = np.array([r["region_id"] for r in regions.regions])
        assigned[accepted] = names[best[accepted]]
    binding = fingerprint(
        {
            "regions": regions.fingerprint,
            "cells": cell_fingerprint,
            "ids": ids.tolist(),
            "counts": [
                _array_hash(a) for a in (counts.indptr, counts.indices, counts.data)
            ],
            "minimum_fraction": minimum_fraction,
        }
    )
    return RegionAssignments(
        ids, assigned, status, fraction, regions.fingerprint, binding
    )


def region_expression_summary(
    regions: TissueRegions,
    assignments: RegionAssignments,
    cell_ids,
    protein_ids,
    values,
    support,
    *,
    evidence_kind: str,
    expression_fingerprint: str,
    positivity_thresholds: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Summarize each region/protein; exclude ambiguous cells without guessing.

    Mean, median, population SD and IQR use finite supported cells only. Coverage
    divides supported cells by all unambiguous cells in that region. Positivity
    is calculated only for a validated, fingerprinted threshold. Evidence kinds
    are never pooled. Empty statistics are missing, not zero.
    """
    ids = _ids(cell_ids, "cell IDs")
    proteins = _ids(protein_ids, "protein IDs")
    if not np.array_equal(ids, assignments.cell_ids):
        raise ValueError("expression cell ordering differs from region assignments")
    if assignments.region_fingerprint != regions.fingerprint:
        raise ValueError("assignments refer to another semantic region version")
    if evidence_kind not in {"measured", "directly_predicted", "transferred"}:
        raise ValueError("unknown expression evidence kind")
    array, valid = np.asarray(values, float), np.asarray(support)
    if (
        array.shape != (len(ids), len(proteins))
        or valid.shape != array.shape
        or valid.dtype != bool
    ):
        raise ValueError(
            "values and boolean support must align with cells and proteins"
        )
    if np.any(np.isinf(array)) or not expression_fingerprint:
        raise ValueError("values must be finite or missing and fingerprinted")
    valid = valid & np.isfinite(array)
    thresholds = positivity_thresholds or {}
    for protein, threshold in thresholds.items():
        if (
            protein not in proteins
            or threshold.get("validated") is not True
            or not threshold.get("fingerprint")
            or not np.isfinite(threshold.get("value", np.nan))
        ):
            raise ValueError(
                "positivity requires a validated protein-specific threshold"
            )
    rows = []
    for region in regions.regions:
        member = assignments.region_ids == region["region_id"]
        n = int(member.sum())
        for col, protein in enumerate(proteins):
            selected = array[member & valid[:, col], col]
            count = len(selected)
            threshold = thresholds.get(protein)
            rows.append(
                {
                    "region_id": region["region_id"],
                    "section_id": region["section_id"],
                    "semantic_class": region["semantic_class"],
                    "protein_id": str(protein),
                    "evidence_kind": evidence_kind,
                    "cell_count": n,
                    "supported_count": count,
                    "supported_coverage": count / n if n else None,
                    "mean": float(np.mean(selected)) if count else None,
                    "median": float(np.median(selected)) if count else None,
                    "dispersion_sd": float(np.std(selected)) if count else None,
                    "dispersion_iqr": float(
                        np.subtract(*np.quantile(selected, [0.75, 0.25]))
                    )
                    if count
                    else None,
                    "positive_fraction": float(np.mean(selected >= threshold["value"]))
                    if count and threshold
                    else None,
                    "positivity_threshold_fingerprint": threshold["fingerprint"]
                    if threshold
                    else None,
                    "region_fingerprint": regions.fingerprint,
                    "assignment_fingerprint": assignments.fingerprint,
                    "expression_fingerprint": expression_fingerprint,
                }
            )
    return rows


def write_summary_tables(
    rows: list[dict[str, object]], path: Path | str, *, parquet=False
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("summary table is empty")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet export requires the 'study' extra") from exc
        pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))
    return path
