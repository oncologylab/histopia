"""New-input recovery must retain input identity as well as exact geometry."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

import numpy as np
import pytest

from histopia.cells._reviewed_addition import _json_sha256
from histopia.cells._reviewed_input_addition import (
    FIXED_HDAB_PROJECTION,
    REVIEWED_INPUT_ADDITION_ACTION,
    REVIEWED_INPUT_ADDITION_SCOPE,
    validate_reviewed_input_addition_manifest,
)
from histopia.cells._reviewed_input_addition_compose import (
    _array_sha256,
    _load_verified_input_mask,
)
from histopia.protein._cell_features import neutralize_hdab_morphology


def _make_input_archive(tmp_path, raw, rgb, *, x=0, index=0):
    """Small saved inference fixture; no neural model or external data."""
    parameters = {
        "model": "cpsam_v2",
        "method": "containment",
        "first_diameter": 26,
        "second_diameter": 15,
        "first_cellprob": -2.5,
        "second_cellprob": -1.75,
        "flow_threshold": 0.0,
        "merge_threshold": 0.1,
        "min_size": 15,
        "inference_batch_size": 32,
        "multiscale_nuclear_support": True,
        "nuclear_minimum_optical_density": 0.12,
        "nuclear_minimum_pixels": 3,
        "nuclear_minimum_fraction": 0.005,
    }
    identity = {
        "schema_version": 1,
        "rgb_shape": list(rgb.shape),
        "rgb_sha256": _array_sha256(neutralize_hdab_morphology(rgb, (0, 0, 1, 1))),
        "inference_binding": {
            "model_weight_sha256": "f" * 64,
            "algorithm_sha256": "1" * 64,
            "adapter_sha256": "2" * 64,
            "cellpose_version": "4.2.1",
            "torch_version": "test-fixture-no-inference",
            "parameters": parameters,
        },
    }
    archive = tmp_path / f"new-input-{index}.npz"
    np.savez_compressed(
        archive,
        mask=raw,
        mask_sha256=np.asarray(_array_sha256(raw)),
        identity_json=np.asarray(
            json.dumps(identity, sort_keys=True, separators=(",", ":"))
        ),
        inference_seconds=np.asarray(0.0),
    )
    tile = {
        "x": x,
        "y": 0,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "mask_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "mask_array_sha256": _array_sha256(raw),
        "mask_fingerprint": _json_sha256(identity),
        "source_rgb_sha256": _array_sha256(rgb),
        "input_identity": identity,
        "generation_plan_sha256": "3" * 64,
        "generation_worker_sha256": "4" * 64,
        "projection": {
            **copy.deepcopy(FIXED_HDAB_PROJECTION),
            "implementation_sha256": hashlib.sha256(
                inspect.getsource(neutralize_hdab_morphology).encode()
            ).hexdigest(),
        },
    }
    return archive, tile


def _manifest(tmp_path):
    from test_cells_reviewed_multi_addition import _manifest as original_manifest

    manifest = original_manifest()
    manifest["scope"] = REVIEWED_INPUT_ADDITION_SCOPE
    for i, patch in enumerate(manifest["sections"][0]["patches"]):
        original = patch.pop("raw_tile")
        raw = np.zeros((100, 100), np.int32)
        raw[10:15, 10:15] = 4
        rgb = np.full((100, 100, 3), [130, 110, 160], np.uint8)
        _, tile = _make_input_archive(tmp_path, raw, rgb, x=original["x"], index=i)
        patch["input_tile"] = tile
        patch["input_receipt_sha256"] = "5" * 64
        patch["selection"]["instance_action"] = REVIEWED_INPUT_ADDITION_ACTION
    return manifest


def test_new_input_manifest_is_explicit_and_cannot_pass_as_original_cache(tmp_path):
    from histopia.cells._reviewed_multi_addition import (
        validate_reviewed_multi_addition_manifest,
    )

    manifest = _manifest(tmp_path)
    assert len(validate_reviewed_input_addition_manifest(manifest)["patches"]) == 2
    with pytest.raises(ValueError):
        validate_reviewed_multi_addition_manifest(manifest)
    disguised = copy.deepcopy(manifest)
    disguised["scope"] = "post-inference-reviewed-disjoint-cache-addition-v1"
    with pytest.raises(ValueError):
        validate_reviewed_multi_addition_manifest(disguised)


@pytest.mark.parametrize(
    "mutation",
    ["palette", "diameter", "weights", "shape", "source", "overlap", "review", "ids"],
)
def test_new_input_manifest_rejects_changed_science_identity_and_selection(
    tmp_path, mutation
):
    manifest = _manifest(tmp_path)
    patch = manifest["sections"][0]["patches"][1]
    tile = patch["input_tile"]
    if mutation == "palette":
        tile["projection"]["purple_rgb"][0] += 1
    elif mutation == "diameter":
        tile["input_identity"]["inference_binding"]["parameters"]["first_diameter"] = 27
        tile["mask_fingerprint"] = _json_sha256(tile["input_identity"])
    elif mutation == "weights":
        tile["input_identity"]["inference_binding"]["model_weight_sha256"] = "a" * 64
        tile["mask_fingerprint"] = _json_sha256(tile["input_identity"])
    elif mutation == "shape":
        tile["width"] = 101
    elif mutation == "source":
        patch["source_identity"] = "a" * 64
    elif mutation == "overlap":
        tile["x"] = 99
        patch["selection"]["support_geometry"]["bbox_xywh"][0] = 99
    elif mutation == "review":
        patch["review"]["decision"] = "pending"
    else:
        patch["selection"].update(first_output_id=101, last_output_id=102)
    with pytest.raises(ValueError):
        validate_reviewed_input_addition_manifest(manifest)


@pytest.mark.parametrize(
    "mutation", ["native_rgb", "projected_rgb", "mask_pixels", "identity", "archive"]
)
def test_actual_saved_mask_rejects_pixel_and_archive_changes(tmp_path, mutation):
    raw = np.zeros((16, 16), np.int32)
    raw[4:8, 4:8] = 1
    rgb = np.full((16, 16, 3), [130, 110, 160], np.uint8)
    archive, tile = _make_input_archive(tmp_path, raw, rgb)
    np.testing.assert_array_equal(_load_verified_input_mask(archive, tile, rgb), raw)
    if mutation == "native_rgb":
        rgb[0, 0, 0] += 1
    elif mutation == "projected_rgb":
        tile["input_identity"]["rgb_sha256"] = "a" * 64
    elif mutation == "mask_pixels":
        with np.load(archive, allow_pickle=False) as p:
            payload = {k: p[k] for k in p.files}
        payload["mask"][0, 0] = 1
        np.savez_compressed(archive, **payload)
        tile["mask_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    elif mutation == "identity":
        tile["input_identity"]["inference_binding"]["torch_version"] = "changed"
    else:
        archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        _load_verified_input_mask(archive, tile, rgb)
