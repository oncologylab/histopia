from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from histopia.registration._cli import _config_from_mapping
from histopia.registration._config import (
    BrightfieldMaskConfig,
    MaskRefinementConfig,
    NonRigidRefinementConfig,
    RegistrationConfig,
)


def test_registration_config_normalizes_paths_and_defaults(tmp_path: Path) -> None:
    config = RegistrationConfig(
        tmp_path / "input",
        tmp_path / "output",
        automatic_mask_snapshot_path=tmp_path / "snapshot.json",
    )

    assert config.crop_mode == "reference"
    assert config.automatic_mask_snapshot_path == tmp_path / "snapshot.json"
    assert config.thumbnail_workers == 1
    assert config.mask_workers == 1
    assert config.ordering_workers == 1
    assert config.preprocessing_cache is True
    assert config.alignment_cache is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("reference_policy", "first", "reference_policy"),
        ("section_order_strategy", "automatic", "section_order_strategy"),
        ("crop_mode", "none", "crop_mode"),
        ("rigid_method", "sift", "rigid_method"),
        ("align_strategy", "pairwise", "align_strategy"),
        ("wsi_compression", "zstd", "wsi_compression"),
    ),
)
def test_registration_config_rejects_unknown_modes(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RegistrationConfig(
            tmp_path / "input",
            tmp_path / "output",
            **{field: value},
        )


def test_registration_config_requires_explicit_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference_slide is required"):
        RegistrationConfig(
            tmp_path / "input",
            tmp_path / "output",
            reference_policy="explicit",
        )
    with pytest.raises(ValueError, match="section_order_path is required"):
        RegistrationConfig(
            tmp_path / "input",
            tmp_path / "output",
            section_order_strategy="manifest",
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"mode": "threshold"}, ValueError, "mask mode"),
        ({"allow_full_fallback": 1}, TypeError, "boolean"),
        ({"min_foreground_fraction": -0.1}, ValueError, "between 0 and 1"),
        ({"max_foreground_fraction": float("nan")}, ValueError, "finite"),
        (
            {"min_foreground_fraction": 0.8, "max_foreground_fraction": 0.2},
            ValueError,
            "must not exceed",
        ),
        ({"min_object_area_px": 0}, ValueError, "must be positive"),
        ({"close_radius_px": -1}, ValueError, "non-negative"),
        ({"open_radius_px": 1.5}, TypeError, "integer"),
    ),
)
def test_brightfield_mask_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        BrightfieldMaskConfig(**kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            MaskRefinementConfig,
            {"min_dice_improvement": 1.1},
            "between 0 and 1",
        ),
        (
            MaskRefinementConfig,
            {"max_relative_anisotropy": float("inf")},
            "finite",
        ),
        (
            NonRigidRefinementConfig,
            {"min_similarity_improvement": -0.1},
            "between 0 and 2",
        ),
        (
            NonRigidRefinementConfig,
            {"max_mask_dice_loss": 1.1},
            "between 0 and 1",
        ),
        (
            NonRigidRefinementConfig,
            {"min_jacobian_p01": 1.1},
            "between 0 and 1",
        ),
        (
            NonRigidRefinementConfig,
            {"max_jacobian_p99": 0.9},
            "at least 1",
        ),
    ),
)
def test_refinement_configs_reject_invalid_acceptance_gates(
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("max_processed_image_dim_px", 0, ValueError),
        ("thumbnail_workers", True, TypeError),
        ("mask_workers", 1.5, TypeError),
        ("ordering_workers", 0, ValueError),
        ("wsi_jpeg_quality", 101, ValueError),
        ("wsi_tile_size", -1, ValueError),
        ("write_warped_images", 1, TypeError),
        ("alignment_cache", 1, TypeError),
        ("input_slides", "slide.ndpi", TypeError),
    ),
)
def test_registration_config_rejects_invalid_execution_controls(
    tmp_path: Path,
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        RegistrationConfig(
            tmp_path / "input",
            tmp_path / "output",
            **{field: value},
        )


def test_registration_mapping_parser_does_not_mutate_input(tmp_path: Path) -> None:
    payload = {
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
        "mask": {"mode": "auto_tissue"},
        "refinement": {"enabled": True},
        "non_rigid_refinement": {"enabled": False},
    }
    original = deepcopy(payload)

    config = _config_from_mapping(payload)

    assert config.mask.mode == "auto_tissue"
    assert payload == original


def test_registration_mapping_parses_alignment_cache(tmp_path: Path) -> None:
    config = _config_from_mapping(
        {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(tmp_path / "output"),
            "alignment_cache": False,
        }
    )

    assert config.alignment_cache is False
