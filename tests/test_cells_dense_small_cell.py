import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD,
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD_PROFILE,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD_PROFILE,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_ALGORITHM_VERSION,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD_PROFILE,
    DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSION,
    DENSE_SMALL_CELL_RECOVERY_METHOD,
    DENSE_SMALL_CELL_RECOVERY_METHOD_PROFILE,
    dense_small_cell_adjacency_recovery_parameters,
    dense_small_cell_cpsam_context_adjacency_recovery_parameters,
    dense_small_cell_cpsam_context_recovery_parameters,
    dense_small_cell_recovery_manifest_profile,
    dense_small_cell_recovery_parameters,
    load_dense_small_cell_recovery_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _recovery_fixture(
    tmp_path: Path,
    *,
    dimensions_in_evidence: bool = False,
    adjacency: bool = False,
    cpsam_context: bool = False,
    cpsam_context_adjacency: bool = False,
) -> tuple[Path, Path, dict[str, str]]:
    if sum((adjacency, cpsam_context, cpsam_context_adjacency)) > 1:
        raise ValueError("fixture recovery methods are mutually exclusive")
    source = tmp_path / "source"
    (source / ".cell-cache" / "015").mkdir(parents=True)
    (source / "qc").mkdir()
    (source / "labels").mkdir()
    labels_path = source / "labels" / "015.cells.tiff"
    labels_path.write_bytes(b"sealed-labels")
    qc_path = source / "qc" / "015.json"
    qc_path.write_text(json.dumps({"labels_sha256": _sha256(labels_path)}))
    key = "r0001_c0002_y64_x128.npz"
    source_cache = source / ".cell-cache" / "015" / key
    source_fingerprint = "a" * 64
    height, width = (512, 512) if cpsam_context or cpsam_context_adjacency else (8, 9)
    source_mask = np.zeros((height, width), dtype=np.int32)
    source_mask[2:5, 2:5] = 1
    np.savez_compressed(
        source_cache,
        mask=source_mask,
        fingerprint=np.asarray(source_fingerprint),
    )
    source_cache_sha256 = _sha256(source_cache)

    recovery = tmp_path / "recovery"
    tile_dir = recovery / "tiles" / "015"
    tile_dir.mkdir(parents=True)
    mask = np.zeros((height, width), dtype=np.int32)
    mask[2:5, 2:5] = 1
    if cpsam_context or cpsam_context_adjacency:
        mask[10:14, 10:12] = 2
    else:
        mask[3:7, 6:8] = 2
    model = (
        {
            "cellpose_version": "4.2.1",
            "weight_name": "cpsam_v2",
            "weight_sha256": (
                "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
            ),
            "device": {
                "requested": "cuda:0",
                "resolved": "cuda:0",
                "backend": "cuda",
                "accelerator_name": "Synthetic GPU",
            },
            "name": "cpsam_v2",
            "method": "containment",
            "first_cellprob": -2.5,
            "first_diameter": 26,
            "second_cellprob": -1.75,
            "second_diameter": 15,
            "flow_threshold": 0.0,
            "merge_threshold": 0.1,
            "minimum_size_px": 15,
        }
        if cpsam_context or cpsam_context_adjacency
        else {"name": "synthetic", "files": {"weights": "b" * 64}}
    )
    parameters = (
        dense_small_cell_cpsam_context_adjacency_recovery_parameters()
        if cpsam_context_adjacency
        else (
            dense_small_cell_adjacency_recovery_parameters()
            if adjacency
            else (
                dense_small_cell_cpsam_context_recovery_parameters()
                if cpsam_context
                else dense_small_cell_recovery_parameters()
            )
        )
    )
    recovery_core = {
        "schema_version": 1,
        "source_identity": "c" * 64,
        "section": "015",
        "tile": key,
        "x": 128,
        "y": 64,
        "source_cache_sha256": source_cache_sha256,
        "source_tile_fingerprint": source_fingerprint,
        "model": model,
        "parameters": parameters,
    }
    recovery_fingerprint = _json_sha256(recovery_core)
    tile_path = tile_dir / key
    np.savez_compressed(
        tile_path,
        mask=mask,
        fingerprint=np.asarray(recovery_fingerprint),
        source_cache_sha256=np.asarray(source_cache_sha256),
    )
    areas = np.bincount(mask.ravel())[1:]
    areas = areas[areas > 0]
    row: dict[str, object] = {
        **recovery_core,
        "recovery_fingerprint": recovery_fingerprint,
        "file": str(tile_path.relative_to(recovery)),
        "file_sha256": _sha256(tile_path),
        "nucleus_count": 2,
        "source_cell_count": 1,
        "added_cell_count": 1,
        "cell_count": 2,
        "foreground_fraction": float(np.mean(mask > 0)),
        "area_px_quantiles": [
            round(float(value), 3) for value in np.quantile(areas, [0.05, 0.5, 0.95])
        ],
        "screen_evidence": (
            {
                **(
                    {
                        "tile": key,
                        "x": 128,
                        "y": 64,
                        "width": width,
                        "height": height,
                    }
                    if cpsam_context_adjacency
                    else {}
                ),
                "selection_role": "strict_seed",
                "candidate_gap_bin_count": 1,
                "selected_gap_bin_count": 1,
                "selected_gap_bin_fraction": (
                    1.0 / 64.0 if cpsam_context_adjacency else 1.0
                ),
                "selected_gap_bins": [[0, 0]],
                "global_nuclear_fraction": 0.9,
            }
            if adjacency or cpsam_context_adjacency
            else (
                {
                    "tile": key,
                    "x": 128,
                    "y": 64,
                    "width": width,
                    "height": height,
                    "global_nuclear_fraction": 0.9,
                    "gap_bin_count": 8,
                    "gap_bin_fraction": 0.125,
                }
                if cpsam_context
                else {
                    "gap_bin_count": 8,
                    **({"width": 9, "height": 8} if dimensions_in_evidence else {}),
                }
            )
        ),
    }
    if not dimensions_in_evidence:
        row.update(width=width, height=height)
    manifest_core = {
        "schema_version": 1,
        "method": (
            DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD
            if cpsam_context_adjacency
            else (
                DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD
                if adjacency
                else (
                    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD
                    if cpsam_context
                    else DENSE_SMALL_CELL_RECOVERY_METHOD
                )
            )
        ),
        "source_preflight_fingerprint": "d" * 64,
        "source_section": "015",
        "source_identity": "c" * 64,
        "source_section_qc_sha256": _sha256(qc_path),
        "source_labels_sha256": _sha256(labels_path),
        "screen_report_sha256": "e" * 64,
        "model": model,
        "parameters": parameters,
        "tiles": [row],
    }
    manifest = {
        **manifest_core,
        "fingerprint": _json_sha256(manifest_core),
        "elapsed_seconds": 1.25,
    }
    manifest_path = recovery / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return source, manifest_path, {"tile": str(tile_path), "cache": str(source_cache)}


def test_dense_small_cell_manifest_validates_and_is_path_free(tmp_path: Path) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path)

    manifest = load_dense_small_cell_recovery_manifest(
        manifest_path,
        source_run=source,
        expected_preflight_fingerprint="d" * 64,
        expected_section="015",
        expected_source_identity="c" * 64,
    )

    assert tuple(manifest.tiles) == ("r0001_c0002_y64_x128.npz",)
    assert manifest.tiles["r0001_c0002_y64_x128.npz"].load_mask().max() == 2
    request = manifest.request_payload()
    assert request["manifest_fingerprint"] == manifest.fingerprint
    assert request["method"] == DENSE_SMALL_CELL_RECOVERY_METHOD
    assert str(tmp_path) not in json.dumps(request)


def test_dense_small_cell_manifest_accepts_screen_evidence_dimensions(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path, dimensions_in_evidence=True)

    manifest = load_dense_small_cell_recovery_manifest(
        manifest_path,
        source_run=source,
        expected_preflight_fingerprint="d" * 64,
        expected_section="015",
        expected_source_identity="c" * 64,
    )

    assert manifest.tiles["r0001_c0002_y64_x128.npz"].shape == (8, 9)


def test_dense_small_cell_manifest_accepts_seed_connected_adjacency_scope(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path, adjacency=True)

    manifest = load_dense_small_cell_recovery_manifest(
        manifest_path,
        source_run=source,
        expected_preflight_fingerprint="d" * 64,
        expected_section="015",
        expected_source_identity="c" * 64,
    )

    assert manifest.method == DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD
    assert (
        manifest.parameters["recovery_scope"] == "selected-seed-connected-gap-bins-only"
    )
    assert dense_small_cell_recovery_manifest_profile(manifest_path) == (
        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD_PROFILE,
    )


def test_dense_small_cell_manifest_profile_preserves_v1_compatibility(
    tmp_path: Path,
) -> None:
    _source, manifest_path, _ = _recovery_fixture(tmp_path)

    assert dense_small_cell_recovery_manifest_profile(manifest_path) == (
        DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_RECOVERY_METHOD_PROFILE,
    )


def test_dense_small_cell_manifest_accepts_cellpose_only_expanded_context(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path, cpsam_context=True)

    manifest = load_dense_small_cell_recovery_manifest(
        manifest_path,
        source_run=source,
        expected_preflight_fingerprint="d" * 64,
        expected_section="015",
        expected_source_identity="c" * 64,
    )

    assert manifest.method == DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD
    assert dense_small_cell_recovery_manifest_profile(manifest_path) == (
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD_PROFILE,
    )


def test_dense_small_cell_manifest_accepts_seed_connected_cpsam_context(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(
        tmp_path,
        cpsam_context_adjacency=True,
    )

    manifest = load_dense_small_cell_recovery_manifest(
        manifest_path,
        source_run=source,
        expected_preflight_fingerprint="d" * 64,
        expected_section="015",
        expected_source_identity="c" * 64,
    )

    assert manifest.method == DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD
    assert manifest.parameters["recovery_scope"] == (
        "selected-seed-connected-gap-bins-only"
    )
    assert dense_small_cell_recovery_manifest_profile(manifest_path) == (
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD_PROFILE,
    )


def test_dense_small_cell_manifest_rejects_stale_cpsam_context_adjacency_geometry(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(
        tmp_path,
        cpsam_context_adjacency=True,
    )
    payload = json.loads(manifest_path.read_text())
    payload["tiles"][0]["screen_evidence"]["x"] += 64
    core = dict(payload)
    core.pop("fingerprint")
    core.pop("elapsed_seconds")
    payload["fingerprint"] = _json_sha256(core)
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="adjacency geometry"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )


def test_dense_small_cell_manifest_rejects_stale_cpsam_context_screen(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path, cpsam_context=True)
    payload = json.loads(manifest_path.read_text())
    payload["tiles"][0]["screen_evidence"]["global_nuclear_fraction"] = 0.2
    core = dict(payload)
    core.pop("fingerprint")
    core.pop("elapsed_seconds")
    payload["fingerprint"] = _json_sha256(core)
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="expanded-context screen"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )


def test_dense_small_cell_manifest_profile_rejects_resealed_diagnostic_method(
    tmp_path: Path,
) -> None:
    _source, manifest_path, _ = _recovery_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["method"] = "stardist-he-scale2-bounded-territory-v1"
    core = dict(payload)
    core.pop("fingerprint")
    core.pop("elapsed_seconds")
    payload["fingerprint"] = _json_sha256(core)
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="not promotable"):
        dense_small_cell_recovery_manifest_profile(manifest_path)


def test_dense_small_cell_manifest_rejects_stale_adjacency_scope(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path, adjacency=True)
    payload = json.loads(manifest_path.read_text())
    payload["tiles"][0]["screen_evidence"]["selected_gap_bins"] = [[0, 0], [0, 0]]
    payload["tiles"][0]["screen_evidence"]["selected_gap_bin_count"] = 2
    core = dict(payload)
    core.pop("fingerprint")
    core.pop("elapsed_seconds")
    payload["fingerprint"] = _json_sha256(core)
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="adjacency evidence"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )


def test_dense_small_cell_manifest_rejects_modified_recovery_tile(
    tmp_path: Path,
) -> None:
    source, manifest_path, paths = _recovery_fixture(tmp_path)
    Path(paths["tile"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="tile changed"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )


def test_dense_small_cell_manifest_rejects_modified_source_cache(
    tmp_path: Path,
) -> None:
    source, manifest_path, paths = _recovery_fixture(tmp_path)
    Path(paths["cache"]).write_bytes(b"stale")

    with pytest.raises(ValueError, match="source cache changed"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )


def test_dense_small_cell_manifest_rejects_changed_additive_parameters(
    tmp_path: Path,
) -> None:
    source, manifest_path, _ = _recovery_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["parameters"]["recovery_context_bins"] = 2
    core = dict(payload)
    core.pop("fingerprint")
    core.pop("elapsed_seconds")
    payload["fingerprint"] = _json_sha256(core)
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="additive parameters"):
        load_dense_small_cell_recovery_manifest(
            manifest_path,
            source_run=source,
            expected_preflight_fingerprint="d" * 64,
            expected_section="015",
            expected_source_identity="c" * 64,
        )
