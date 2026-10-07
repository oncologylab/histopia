"""Point-annotation metrics and deterministic cell-method benchmark summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.cells._cellpose import (
    load_cellpose_runtime,
    runtime_provenance,
    segment_tile,
)
from histopia.cells._config import CellSegmentationConfig


def read_centroid_tsv(path: Path | str) -> np.ndarray:
    """Read QuPath-style centroid TSV columns containing x/y coordinates."""

    source = Path(path)
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows:
        return np.zeros((0, 2), dtype=float)
    header = [
        value.strip()
        .lower()
        .replace("\N{MICRO SIGN}", "u")
        .replace("\N{GREEK SMALL LETTER MU}", "u")
        for value in rows[0]
    ]
    x_index = _coordinate_column(
        header, ("x", "centroid x", "centroid x um", "centroid x px")
    )
    y_index = _coordinate_column(
        header, ("y", "centroid y", "centroid y um", "centroid y px")
    )
    start = 1
    if x_index is None or y_index is None:
        x_index, y_index, start = 0, 1, 0
    coordinates: list[tuple[float, float]] = []
    for row in rows[start:]:
        if len(row) <= max(x_index, y_index):
            continue
        try:
            coordinates.append((float(row[x_index]), float(row[y_index])))
        except ValueError:
            continue
    return np.asarray(coordinates, dtype=float).reshape(-1, 2)


def centroid_instance_metrics(
    mask: np.ndarray,
    centroids_xy: np.ndarray,
) -> dict[str, float | int]:
    """Score point coverage and instance burden without inventing true negatives."""

    labels = np.asarray(mask)
    points = np.asarray(centroids_xy, dtype=float)
    if labels.ndim != 2:
        raise ValueError("cell mask must be two-dimensional")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centroids must have shape (N, 2)")
    predicted = np.unique(labels)
    predicted_count = int(np.count_nonzero(predicted))
    if points.size:
        xs = np.rint(points[:, 0]).astype(np.int64)
        ys = np.rint(points[:, 1]).astype(np.int64)
        valid = (xs >= 0) & (xs < labels.shape[1]) & (ys >= 0) & (ys < labels.shape[0])
        hits = np.zeros(points.shape[0], dtype=np.int64)
        hits[valid] = labels[ys[valid], xs[valid]].astype(np.int64)
    else:
        hits = np.zeros(0, dtype=np.int64)
    positive_hits = hits[hits > 0]
    hit_labels, hit_counts = np.unique(positive_hits, return_counts=True)
    true_positive = int(hit_labels.size)
    false_negative = int(np.count_nonzero(hits == 0) + np.sum(hit_counts - 1))
    false_positive = max(0, predicted_count - true_positive)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "centroid_count": int(points.shape[0]),
        "predicted_cell_count": predicted_count,
        "hit_instances": true_positive,
        "missed_or_merged_centroids": false_negative,
        "unmatched_instances": false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_benchmark(rows: list[dict[str, object]]) -> dict[str, object]:
    """Rank profiles by protein-macro F1, then pooled F1 and runtime."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["profile"])].append(row)
    profiles = []
    for profile, values in sorted(grouped.items()):
        proteins: dict[str, list[float]] = defaultdict(list)
        for value in values:
            proteins[str(value["protein"])].append(float(value["f1"]))
        macro = float(np.mean([np.mean(scores) for scores in proteins.values()]))
        pooled = float(np.mean([float(value["f1"]) for value in values]))
        elapsed = float(
            sum(float(value.get("elapsed_seconds", 0.0)) for value in values)
        )
        profiles.append(
            {
                "profile": profile,
                "tile_count": len(values),
                "protein_count": len(proteins),
                "protein_macro_f1": macro,
                "tile_mean_f1": pooled,
                "elapsed_seconds": elapsed,
            }
        )
    profiles.sort(
        key=lambda row: (
            -float(row["protein_macro_f1"]),
            -float(row["tile_mean_f1"]),
            float(row["elapsed_seconds"]),
            str(row["profile"]),
        )
    )
    return {
        "schema_version": 1,
        "metric": "protein_macro_centroid_instance_f1",
        "selected_profile": profiles[0]["profile"] if profiles else None,
        "profiles": profiles,
        "rows": rows,
    }


def write_benchmark_summary(rows: list[dict[str, object]], output: Path | str) -> Path:
    """Write a portable benchmark summary."""

    return write_json_atomic(output, summarize_benchmark(rows))


def benchmark_cell_methods(
    image_dir: Path | str,
    annotation_dir: Path | str,
    output_dir: Path | str,
    *,
    model_cache: Path | str,
    models: tuple[str, ...] = ("cpsam", "cpsam_v2"),
    methods: tuple[str, ...] = ("direct", "combined", "containment"),
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Benchmark explicit Cellpose profiles on every matched annotated tile."""

    import tifffile

    images = _annotated_images(Path(image_dir), Path(annotation_dir))
    if not images:
        raise ValueError("benchmark contains no image/centroid pairs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prior_elapsed = _prior_elapsed(output / "benchmark.json")
    rows: list[dict[str, object]] = []
    model_rows: dict[str, dict[str, object]] = {}
    total = len(images) * len(models) * len(methods)
    completed = 0
    for model_name in models:
        base = CellSegmentationConfig(
            registration_run=Path("."),
            output_dir=output,
            model=model_name,
            method="direct",
            device=device,
            model_cache=Path(model_cache),
        )
        runtime = load_cellpose_runtime(base)
        model_rows[model_name] = runtime_provenance(runtime)
        for method in methods:
            config = CellSegmentationConfig(
                registration_run=Path("."),
                output_dir=output,
                model=model_name,
                method=method,
                device=device,
                model_cache=Path(model_cache),
            )
            profile = f"{model_name}:{method}:legacy-v1"
            for image_path, points_path, protein in images:
                completed += 1
                if progress is not None:
                    progress(f"[{completed}/{total}] {profile} {image_path.name}")
                image = np.asarray(tifffile.imread(image_path))
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=2)
                image = np.asarray(image[..., :3], dtype=np.uint8)
                fingerprint = _benchmark_fingerprint(
                    image_path, points_path, runtime.weight_sha256, profile, config
                )
                cache = (
                    output
                    / "masks"
                    / profile.replace(":", "__")
                    / (image_path.stem + ".npz")
                )
                cached = _load_benchmark_mask(cache, fingerprint)
                started = time.perf_counter()
                if cached is None:
                    mask = segment_tile(image, runtime, config)
                    elapsed = time.perf_counter() - started
                    _write_benchmark_mask(cache, mask, fingerprint, elapsed)
                else:
                    mask, cached_elapsed = cached
                    elapsed = (
                        cached_elapsed
                        if cached_elapsed is not None
                        else prior_elapsed.get((profile, image_path.name), 0.0)
                    )
                metrics = centroid_instance_metrics(
                    mask, read_centroid_tsv(points_path)
                )
                row: dict[str, object] = {
                    "profile": profile,
                    "model": model_name,
                    "method": method,
                    "protein": protein,
                    "tile": image_path.name,
                    "elapsed_seconds": round(elapsed, 6),
                    **metrics,
                }
                rows.append(row)
                _write_benchmark_overlay(
                    image,
                    mask,
                    output
                    / "overlays"
                    / profile.replace(":", "__")
                    / protein
                    / f"{image_path.stem}.jpg",
                )
    summary = summarize_benchmark(rows)
    summary["models"] = model_rows
    summary["tile_count"] = len(images)
    return write_json_atomic(output / "benchmark.json", summary)


def _annotated_images(
    image_dir: Path,
    annotation_dir: Path,
) -> list[tuple[Path, Path, str]]:
    annotations: dict[str, list[Path]] = defaultdict(list)
    for path in annotation_dir.rglob("*.tsv"):
        key = path.name.removesuffix("-points.tsv")
        annotations[key].append(path)
    output = []
    for image in sorted(image_dir.rglob("*.tif*")):
        candidates = list(
            dict.fromkeys(
                [
                    *annotations.get(image.name, []),
                    *annotations.get(image.stem, []),
                ]
            )
        )
        if len(candidates) == 1:
            output.append((image, candidates[0], image.parent.name))
    return output


def _benchmark_fingerprint(
    image: Path,
    points: Path,
    weight_sha256: str,
    profile: str,
    config: CellSegmentationConfig,
) -> str:
    payload = {
        "image": _path_identity(image),
        "points": hashlib.sha256(points.read_bytes()).hexdigest(),
        "weight": weight_sha256,
        "profile": profile,
        "parameters": {
            "first_cellprob": config.first_cellprob,
            "first_diameter": config.first_diameter,
            "second_cellprob": config.second_cellprob,
            "second_diameter": config.second_diameter,
            "merge_threshold": config.merge_threshold,
            "min_size": config.min_size,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _path_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_benchmark_mask(
    path: Path, fingerprint: str
) -> tuple[np.ndarray, float | None] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["fingerprint"].item()) != fingerprint:
                return None
            elapsed = (
                float(payload["elapsed_seconds"].item())
                if "elapsed_seconds" in payload
                else None
            )
            return np.asarray(payload["mask"], dtype=np.int32), elapsed
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def _write_benchmark_mask(
    path: Path,
    mask: np.ndarray,
    fingerprint: str,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(
            temporary,
            mask=mask,
            fingerprint=np.asarray(fingerprint),
            elapsed_seconds=np.asarray(elapsed_seconds),
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prior_elapsed(path: Path) -> dict[tuple[str, str], float]:
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(rows, list):
        return {}
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = row.get("profile")
        tile = row.get("tile")
        elapsed = row.get("elapsed_seconds")
        if (
            isinstance(profile, str)
            and isinstance(tile, str)
            and isinstance(elapsed, (int, float))
        ):
            output[(profile, tile)] = float(elapsed)
    return output


def _write_benchmark_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    from PIL import Image
    from scipy import ndimage as ndi

    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:] |= mask[1:] != mask[:-1]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    overlay = image.copy()
    overlay[ndi.binary_dilation(boundary, iterations=1)] = (20, 20, 20)
    overlay[boundary] = (255, 30, 20)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path, quality=92)


def _coordinate_column(header: list[str], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in header:
            return header.index(name)
    for index, value in enumerate(header):
        if any(name in value for name in names[1:]):
            return index
    return None


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0
