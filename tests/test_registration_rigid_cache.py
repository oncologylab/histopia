from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from histopia.registration._rigid import RigidTransformResult
from histopia.registration._rigid_cache import (
    load_rigid_pair_cache,
    rigid_crop_fingerprint,
    rigid_pair_fingerprint,
    write_rigid_pair_cache,
)


def test_rigid_pair_cache_round_trips_exact_transforms(tmp_path: Path) -> None:
    full = _transform(translation=(4.5, -2.25), method="full")
    crop = _transform(translation=(7.0, 1.5), method="crop")
    fingerprint = rigid_pair_fingerprint("fixed", "moving", {"method": "feature"})

    path = write_rigid_pair_cache(tmp_path, fingerprint, full, crop)
    loaded = load_rigid_pair_cache(tmp_path, fingerprint)

    assert path == tmp_path / f"{fingerprint}.json"
    assert loaded is not None
    loaded_full, loaded_crop = loaded
    assert np.array_equal(loaded_full.matrix, full.matrix)
    assert loaded_full.to_json_dict() == full.to_json_dict()
    assert loaded_crop.to_json_dict() == crop.to_json_dict()
    assert not tuple(tmp_path.glob(f".{fingerprint}.json.*.tmp"))


def test_rigid_pair_cache_rejects_stale_and_corrupt_records(tmp_path: Path) -> None:
    fingerprint = rigid_pair_fingerprint("fixed", "moving", {})
    path = write_rigid_pair_cache(
        tmp_path,
        fingerprint,
        _transform(),
        _transform(),
    )

    assert load_rigid_pair_cache(tmp_path, "stale") is None
    payload = json.loads(path.read_text())
    payload["payload"]["full_transform"]["matrix"][0][2] = 999
    path.write_text(json.dumps(payload))

    assert load_rigid_pair_cache(tmp_path, fingerprint) is None


def test_rigid_pair_cache_rejects_symlink_entry(tmp_path: Path) -> None:
    fingerprint = rigid_pair_fingerprint("fixed", "moving", {})
    target = tmp_path / "target.json"
    write_rigid_pair_cache(
        tmp_path,
        fingerprint,
        _transform(),
        _transform(),
    ).replace(target)
    (tmp_path / f"{fingerprint}.json").symlink_to(target)

    assert load_rigid_pair_cache(tmp_path, fingerprint) is None


def test_rigid_pair_fingerprint_binds_direction_pixels_masks_and_settings(
    tmp_path: Path,
) -> None:
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.ones((4, 5), dtype=bool)
    fixed = rigid_crop_fingerprint(
        tmp_path / "fixed.ndpi",
        image,
        mask,
        np.array([2.0, 3.0]),
        1.5,
    )
    changed_mask = mask.copy()
    changed_mask[0, 0] = False
    changed = rigid_crop_fingerprint(
        tmp_path / "fixed.ndpi",
        image,
        changed_mask,
        np.array([2.0, 3.0]),
        1.5,
    )
    moving = rigid_crop_fingerprint(
        tmp_path / "moving.ndpi",
        image,
        mask,
        np.array([2.0, 3.0]),
        1.5,
    )

    baseline = rigid_pair_fingerprint(fixed, moving, {"refine": True})

    assert changed != fixed
    assert rigid_pair_fingerprint(moving, fixed, {"refine": True}) != baseline
    assert rigid_pair_fingerprint(fixed, moving, {"refine": False}) != baseline


def _transform(
    *,
    translation: tuple[float, float] = (0.0, 0.0),
    method: str = "phase",
) -> RigidTransformResult:
    matrix = np.eye(3, dtype=float)
    matrix[:2, 2] = translation
    return RigidTransformResult(
        matrix=matrix,
        method=method,
        match_count=12,
        inlier_count=10,
        warnings=["review"],
    )
