import numpy as np

from histopia.cells import centroid_instance_metrics, summarize_benchmark
from histopia.cells._benchmark import (
    _annotated_images,
    _load_benchmark_mask,
    _write_benchmark_mask,
    read_centroid_tsv,
)


def test_centroid_metrics_count_misses_merges_and_unmatched_instances() -> None:
    mask = np.zeros((8, 10), dtype=np.int32)
    mask[1:4, 1:4] = 1
    mask[4:7, 6:9] = 2
    points = np.array([[2, 2], [3, 2], [0, 7]], dtype=float)

    metrics = centroid_instance_metrics(mask, points)

    assert metrics["hit_instances"] == 1
    assert metrics["missed_or_merged_centroids"] == 2
    assert metrics["unmatched_instances"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1 / 3


def test_benchmark_selection_uses_protein_macro_f1() -> None:
    rows = [
        {"profile": "a", "protein": "p1", "f1": 1.0},
        {"profile": "a", "protein": "p2", "f1": 0.4},
        {"profile": "b", "protein": "p1", "f1": 0.6},
        {"profile": "b", "protein": "p2", "f1": 0.9},
    ]

    summary = summarize_benchmark(rows)

    assert summary["selected_profile"] == "b"
    assert "tn" not in summary["metric"]


def test_out_of_bounds_centroids_are_misses_not_border_hits() -> None:
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[0, 0] = 1

    metrics = centroid_instance_metrics(mask, np.array([[-2, -2]], dtype=float))

    assert metrics["hit_instances"] == 0
    assert metrics["missed_or_merged_centroids"] == 1


def test_benchmark_discovers_stem_matched_centroid_files(tmp_path) -> None:
    images = tmp_path / "images" / "pERK"
    annotations = tmp_path / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    image = images / "tile.tiff"
    points = annotations / "tile-points.tsv"
    image.write_bytes(b"image")
    points.write_text("Centroid X um\tCentroid Y um\n2\t3\n")

    assert _annotated_images(tmp_path / "images", annotations) == [
        (image, points, "pERK")
    ]
    assert read_centroid_tsv(points).tolist() == [[2.0, 3.0]]


def test_benchmark_mask_cache_preserves_runtime(tmp_path) -> None:
    path = tmp_path / "mask.npz"
    mask = np.array([[0, 1], [2, 0]], dtype=np.int32)

    _write_benchmark_mask(path, mask, "fingerprint", 1.25)

    cached = _load_benchmark_mask(path, "fingerprint")
    assert cached is not None
    cached_mask, elapsed = cached
    assert np.array_equal(cached_mask, mask)
    assert elapsed == 1.25
