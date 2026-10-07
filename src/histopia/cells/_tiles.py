"""Deterministic tile geometry and overlap-aware instance stitching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True, slots=True)
class CellTile:
    """One row-major tile in content-box coordinates."""

    row: int
    column: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def key(self) -> str:
        return f"r{self.row:04d}_c{self.column:04d}_y{self.y0}_x{self.x0}"


def make_tiles(
    height: int, width: int, *, tile_size: int, overlap: int
) -> tuple[CellTile, ...]:
    """Create an end-aligned deterministic row-major tile grid."""

    if height <= 0 or width <= 0:
        raise ValueError("tile extent must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    return tuple(
        CellTile(
            row, column, y, min(y + tile_size, height), x, min(x + tile_size, width)
        )
        for row, y in enumerate(ys)
        for column, x in enumerate(xs)
    )


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return starts that cover the extent and align the last tile to its end."""

    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def mutual_best_mapping(
    local: np.ndarray,
    existing: np.ndarray,
    threshold: float,
) -> dict[int, int]:
    """Match overlapping instances only when local/global best choices agree."""

    overlap = (local > 0) & (existing > 0)
    if not np.any(overlap):
        return {}
    pairs = np.stack([local[overlap], existing[overlap]], axis=1)
    unique_pairs, intersections = np.unique(pairs, axis=0, return_counts=True)
    local_area = np.bincount(local.ravel())
    existing_ids, existing_counts = np.unique(
        existing[existing > 0], return_counts=True
    )
    existing_area = dict(
        zip(existing_ids.tolist(), existing_counts.tolist(), strict=True)
    )
    local_best: dict[int, tuple[int, float, int]] = {}
    global_best: dict[int, tuple[int, float, int]] = {}
    for pair, intersection in zip(unique_pairs, intersections, strict=True):
        local_id, global_id = (int(pair[0]), int(pair[1]))
        denominator = min(local_area[local_id], existing_area[global_id])
        score = float(intersection) / float(denominator) if denominator else 0.0
        local_candidate = (global_id, score, int(intersection))
        global_candidate = (local_id, score, int(intersection))
        if _better_candidate(local_candidate, local_best.get(local_id)):
            local_best[local_id] = local_candidate
        if _better_candidate(global_candidate, global_best.get(global_id)):
            global_best[global_id] = global_candidate
    return {
        local_id: global_id
        for local_id, (global_id, score, _intersection) in local_best.items()
        if score >= threshold
        and global_id in global_best
        and global_best[global_id][0] == local_id
    }


def center_weight(shape: tuple[int, int]) -> np.ndarray:
    """Favor tile centers when overlapping predictions disagree."""

    height, width = shape
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
    return np.minimum(y[:, None], x[None, :]).astype(np.uint16)


class DiskLabelCanvas:
    """Disk-backed global label and weight canvases for one WSI content box."""

    def __init__(self, root: Path, shape: tuple[int, int], *, resume: bool) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        pixel_count = int(np.prod(shape, dtype=np.int64))
        expected_label_bytes = pixel_count * np.dtype(np.uint32).itemsize
        expected_weight_bytes = pixel_count * np.dtype(np.uint16).itemsize
        resumable = (
            resume
            and self.label_path.is_file()
            and self.weight_path.is_file()
            and self.label_path.stat().st_size == expected_label_bytes
            and self.weight_path.stat().st_size == expected_weight_bytes
        )
        mode = "r+" if resumable else "w+"
        self.labels = np.memmap(
            self.label_path, dtype=np.uint32, mode=mode, shape=shape
        )
        self.weights = np.memmap(
            self.weight_path, dtype=np.uint16, mode=mode, shape=shape
        )
        if mode == "w+":
            self.next_label = 0
        else:
            self.next_label = int(self.labels.max())

    @property
    def label_path(self) -> Path:
        return self.root / "labels.uint32.memmap"

    @property
    def weight_path(self) -> Path:
        return self.root / "weights.uint16.memmap"

    def merge(
        self,
        tile: CellTile,
        local_mask: np.ndarray,
        *,
        match_ios: float,
        sanitize: bool = True,
    ) -> None:
        from histopia.cells._algorithms import clean_mask

        local = (
            clean_mask(local_mask)
            if sanitize
            else np.asarray(local_mask, dtype=np.int32)
        )
        expected = (tile.y1 - tile.y0, tile.x1 - tile.x0)
        if local.shape != expected:
            raise ValueError(f"tile mask shape {local.shape} does not match {expected}")
        canvas = self.labels[tile.y0 : tile.y1, tile.x0 : tile.x1]
        weights = self.weights[tile.y0 : tile.y1, tile.x0 : tile.x1]
        mapping = mutual_best_mapping(local, canvas, match_ios)
        local_ids = np.unique(local)
        lookup = np.zeros(int(local_ids[-1]) + 1, dtype=np.uint32)
        for local_id in local_ids:
            if local_id == 0:
                continue
            global_id = mapping.get(int(local_id))
            if global_id is None:
                self.next_label += 1
                global_id = self.next_label
            lookup[local_id] = global_id
        remapped = lookup[local]
        tile_weight = center_weight(local.shape)
        foreground = remapped > 0
        replace = foreground & (
            (canvas == 0) | (canvas == remapped) | (tile_weight > weights)
        )
        canvas[replace] = remapped[replace]
        weights[replace] = tile_weight[replace]

    def flush(self) -> None:
        self.labels.flush()
        self.weights.flush()


def _better_candidate(
    candidate: tuple[int, float, int],
    current: tuple[int, float, int] | None,
) -> bool:
    if current is None:
        return True
    return (candidate[1], candidate[2], -candidate[0]) > (
        current[1],
        current[2],
        -current[0],
    )
