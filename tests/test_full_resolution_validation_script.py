import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_kpf_full_resolution.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "histopia_validate_kpf_full_resolution",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_mouse = _MODULE.audit_mouse


@pytest.mark.integration
def test_full_resolution_audit_rejects_one_bad_slide_without_median_masking(
    tmp_path: Path,
) -> None:
    pyvips = pytest.importorskip("pyvips")
    mouse = "test-mouse"
    registration_root = tmp_path / "runs"
    full_resolution_root = tmp_path / "registered"
    run = registration_root / mouse
    output = full_resolution_root / mouse
    processed = run / "processed"
    alignment = run / "qc" / "alignment"
    processed.mkdir(parents=True)
    alignment.mkdir(parents=True)
    output.mkdir(parents=True)

    expected = np.full((24, 32, 3), 255, dtype=np.uint8)
    expected[6:18, 8:24] = [120, 80, 60]
    bad = np.full_like(expected, 255)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[6:18, 8:24] = 255
    for stem in ("reference", "moving"):
        Image.fromarray(expected).save(processed / f"{stem}.thumbnail.png")
        Image.fromarray(mask).save(processed / f"{stem}.mask.png")
    Image.fromarray(expected).save(alignment / "moving.warped.png")

    source = tmp_path / "source"
    source.mkdir()
    reference_path = source / "reference.tiff"
    moving_path = source / "moving.tiff"
    _write_tiff(pyvips, reference_path, expected)
    _write_tiff(pyvips, moving_path, expected)
    reference_output = output / "reference.registered.tiff"
    moving_output = output / "moving.registered.tiff"
    _write_tiff(pyvips, reference_output, expected)
    _write_tiff(pyvips, moving_output, bad)
    identity = np.eye(3).tolist()
    (run / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(reference_path),
                "slides": [
                    {
                        "path": str(reference_path),
                        "is_reference": True,
                        "transform": {"matrix": identity},
                    },
                    {
                        "path": str(moving_path),
                        "is_reference": False,
                        "transform": {"matrix": identity},
                    },
                ],
            }
        )
    )
    (output / "full_resolution_warps.json").write_text(
        json.dumps(
            [
                _summary_row(reference_output),
                _summary_row(moving_output),
            ]
        )
    )

    report = audit_mouse(
        mouse,
        registration_root,
        full_resolution_root,
        32,
        15.0,
        25.0,
    )

    assert report["provenance_records"] == 2
    assert report["rejected_slides"] == ["moving"]
    assert report["median_thumbnail_mae"] < report["maximum_thumbnail_mae"]

    selected = audit_mouse(
        mouse,
        registration_root,
        full_resolution_root,
        32,
        15.0,
        30.0,
        slide_names=("reference.tiff",),
    )

    assert selected["expected_files"] == 1
    assert selected["output_files"] == 1
    assert selected["warp_records"] == 1
    assert selected["aggregate_thresholds_applied"] is False
    assert selected["rejected_slides"] == []
    with pytest.raises(ValueError, match="not present"):
        audit_mouse(
            mouse,
            registration_root,
            full_resolution_root,
            32,
            15.0,
            30.0,
            slide_names=("missing",),
        )


def _write_tiff(pyvips: object, path: Path, array: np.ndarray) -> None:
    pyvips.Image.new_from_memory(
        array.tobytes(),
        array.shape[1],
        array.shape[0],
        3,
        "uchar",
    ).tiffsave(str(path), compression="lzw")


def _summary_row(path: Path) -> dict[str, object]:
    return {
        "output_path": str(path),
        "reference_shape": [24, 32],
        "provenance": {"schema_version": 1},
    }
