"""Compose reviewed new-input candidates after exact native RGB verification."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from histopia.cells._bounded_additive_recovery import select_bounded_additive_instances
from histopia.cells._pipeline import _import_pyvips, _tile_tissue_mask
from histopia.cells._reviewed_addition import _json_sha256, _validate_selected_geometry
from histopia.cells._reviewed_addition_compose import (
    _object,
    _read_label_crop,
    _reviewed_label_subset,
    _sha256_file,
)
from histopia.cells._reviewed_input_addition import (
    REVIEWED_INPUT_ADDITION_ACTION,
    ReviewedInputPatch,
    validate_input_receipt,
)
from histopia.cells._tiles import CellTile


def compose_reviewed_input_addition(
    base_run: Path | str,
    section: str,
    patches: tuple[ReviewedInputPatch, ...],
    output_dir: Path | str,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Reuse fixed H-DAB input masks, preserving every original cell ID/pixel.

    This is a bounded reviewed addition, not replacement of the source image or
    segmentation. Native RGB and the projected RGB are read/recreated solely to
    verify the saved inference identity. No neural model is loaded or executed.
    The fixed H-DAB projection is an IHC hypothesis, not a stain-independent
    estimate of biological hematoxylin. Off-assay images require another profile.
    """
    from histopia.cells._reviewed_multi_addition_compose import (
        _compose_reviewed_addition,
    )

    return _compose_reviewed_addition(
        base_run,
        section,
        patches,
        output_dir,
        expected_model="cpsam_v2",
        progress=progress,
        new_input=True,
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True
    )
    return hashlib.sha256(header.encode() + array.tobytes()).hexdigest()


def _load_verified_input_mask(
    archive: Path, tile: dict[str, Any], rgb: np.ndarray
) -> np.ndarray:
    """Reject changed archives, labels, native pixels, projections or identities."""
    from histopia.protein._cell_features import neutralize_hdab_morphology

    if _sha256_file(archive) != tile["mask_archive_sha256"]:
        raise ValueError("new-input mask archive changed")
    if rgb.dtype != np.uint8 or rgb.shape != (tile["height"], tile["width"], 3):
        raise ValueError("native input RGB geometry differs")
    if _array_sha256(rgb) != tile["source_rgb_sha256"]:
        raise ValueError("native RGB differs from reviewed input")
    implementation = hashlib.sha256(
        inspect.getsource(neutralize_hdab_morphology).encode()
    ).hexdigest()
    if implementation != tile["projection"]["implementation_sha256"]:
        raise ValueError("projection implementation differs from reviewed receipt")
    projected = neutralize_hdab_morphology(rgb, (0, 0, tile["width"], tile["height"]))
    if _array_sha256(projected) != tile["input_identity"]["rgb_sha256"]:
        raise ValueError("projected RGB differs from saved inference")
    with np.load(archive, allow_pickle=False) as payload:
        identity_json = str(payload["identity_json"].item())
        expected_json = json.dumps(
            tile["input_identity"], sort_keys=True, separators=(",", ":")
        )
        if identity_json != expected_json:
            raise ValueError("saved new-input inference identity changed")
        raw = payload["mask"]
        if (
            raw.ndim != 2
            or raw.shape != rgb.shape[:2]
            or raw.dtype != np.int32
            or np.any(raw < 0)
        ):
            raise ValueError("saved new-input mask geometry is invalid")
        if (
            str(payload["mask_sha256"].item()) != tile["mask_array_sha256"]
            or _array_sha256(raw) != tile["mask_array_sha256"]
        ):
            raise ValueError("saved new-input mask pixels changed")
        seconds = float(payload["inference_seconds"].item())
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("saved new-input inference time is invalid")
    return raw


def _prepare_input_patch(
    patch: ReviewedInputPatch,
    base: dict[str, Any],
    row: dict[str, Any],
    slide: dict[str, Any],
    source: Any,
    tissue: np.ndarray,
    labels_sha: str,
    qc_sha: str,
) -> dict[str, Any]:
    receipt_path = Path(patch.input_receipt).expanduser().resolve()
    if _sha256_file(receipt_path) != patch.input_receipt_sha256:
        raise ValueError("reviewed input receipt changed")
    receipt = validate_input_receipt(_object(receipt_path, "reviewed input receipt"))
    expected = {
        "source_result_fingerprint": base["fingerprint"],
        "source_preflight_fingerprint": base["preflight_fingerprint"],
        "source_identity": row["source_identity"],
        "section": row["section"],
        "source_labels_sha256": labels_sha,
        "source_qc_sha256": qc_sha,
    }
    if any(receipt[k] != value for k, value in expected.items()):
        raise ValueError("reviewed new-input source binding differs")
    tile = receipt["input_tile"]
    binding = tile["input_identity"]["inference_binding"]
    if (
        binding["model_weight_sha256"] != base["model"]["weight_sha256"]
        or binding["cellpose_version"] != base["model"]["cellpose_version"]
    ):
        raise ValueError("reviewed new-input model differs from source")
    x, y, w, h = (tile[k] for k in ("x", "y", "width", "height"))
    height, width = row["native_shape"]
    sx, sy, sw, sh = patch.support_bbox_xywh
    if (
        sx < x
        or sy < y
        or sx + sw > x + w
        or sy + sh > y + h
        or x + w > width
        or y + h > height
    ):
        raise ValueError("reviewed new-input support leaves native geometry")
    from histopia._vips_image import normalize_vips_rgb_uchar

    native = normalize_vips_rgb_uchar(
        _import_pyvips().Image.new_from_file(slide["source_path"], access="random")
    )
    x0, y0 = slide["content_bbox_xywh"][:2]
    rgb = np.frombuffer(
        native.crop(x0 + x, y0 + y, w, h).write_to_memory(), np.uint8
    ).reshape(h, w, 3)
    raw = _load_verified_input_mask(Path(patch.mask_archive), tile, rgb)
    before = _read_label_crop(source, x, y, w, h)
    support = np.zeros((h, w), bool)
    support[sy - y : sy - y + sh, sx - x : sx - x + sw] = True
    length = int(raw.max()) + 1
    if length > raw.size + 1:
        raise ValueError("new-input label IDs are not a compact local mask")
    total = np.bincount(raw.ravel(), minlength=length)
    inside = np.bincount(raw[support].ravel(), minlength=length)
    overlap = np.bincount(raw[before > 0].ravel(), minlength=length)
    novel = np.bincount(raw[support & (before == 0)].ravel(), minlength=length)
    eligible = select_bounded_additive_instances(
        total, inside, overlap, novel, minimum_area_px=15
    )
    selected = _reviewed_label_subset(
        patch.selected_tile_label_ids, eligible.selected_candidate_ids
    )
    restricted = np.zeros_like(novel)
    restricted[list(selected)] = novel[list(selected)]
    selection = select_bounded_additive_instances(
        total, inside, overlap, restricted, minimum_area_px=15
    )
    tile_tissue = _tile_tissue_mask(
        tissue,
        CellTile(row=0, column=0, x0=x, x1=x + w, y0=y, y1=y + h),
        content_shape=(height, width),
    )
    if np.any(np.isin(raw, selected) & ~tile_tissue):
        raise ValueError("reviewed new-input instances leave accepted tissue")
    evidence = [Path(p).expanduser().resolve() for p in patch.evidence_paths]
    if not evidence or any(not p.is_file() for p in evidence):
        raise ValueError("reviewed new-input evidence is missing")
    record = {
        **{
            k: receipt[k]
            for k in (
                "section",
                "source_identity",
                "source_labels_sha256",
                "source_qc_sha256",
            )
        },
        "input_tile": tile,
        "input_receipt_sha256": patch.input_receipt_sha256,
        "selected_tile_label_ids": list(selected),
        "selected_tile_label_ids_sha256": _json_sha256(list(selected)),
        "selection": {
            "instance_action": REVIEWED_INPUT_ADDITION_ACTION,
            "native_coordinate_space": "source_wsi_pixels",
            "support_geometry": {"bbox_xywh": list(patch.support_bbox_xywh)},
            "containment_rule": "all-selected-instance-pixels-inside-support-v1",
            "source_overlap_rule": "zero-source-overlap-pixels-v1",
            "minimum_area_px": 15,
            "selected_instance_count": len(selected),
            "selected_pixels": selection.added_pixels,
            "first_output_id": 1,
            "last_output_id": len(selected),
            "changed_source_pixels": 0,
            "changed_outside_support_pixels": 0,
        },
        "review": {
            "decision": "approved",
            "reviewer": patch.reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "evidence_sha256s": [_sha256_file(p) for p in evidence],
            "notes": patch.notes,
        },
    }
    _validate_selected_geometry(
        record, tile, instance_action=REVIEWED_INPUT_ADDITION_ACTION
    )
    return {"raw": raw, "support": support, "selection": selection, "record": record}
