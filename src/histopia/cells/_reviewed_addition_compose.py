"""Compose a source-preserving cell recovery from one reviewed raw tile."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from histopia.cells._bounded_additive_recovery import (
    apply_bounded_additions,
    bounded_additive_label_lookup,
    select_bounded_additive_instances,
)
from histopia.cells._pipeline import (
    _copy_labels_to_canvas,
    _disk_label_qc,
    _import_pyvips,
    _tile_tissue_mask,
    _write_pyramidal_labels,
)
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_SHA256,
    validate_cell_promotion_candidate,
)
from histopia.cells._result import validate_cell_result, write_cell_result
from histopia.cells._reviewed_addition import (
    REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
    REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
    REVIEWED_CACHE_ADDITION_METHOD_PROFILE,
    REVIEWED_CACHE_ADDITION_SCOPE,
    REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION,
    reviewed_cache_addition_qc_evidence,
    validate_reviewed_cache_addition_manifest,
)
from histopia.cells._tiles import CellTile

_TILE_KEY = re.compile(r"^r[0-9]{4}_c[0-9]{4}_y(?P<y>[0-9]+)_x(?P<x>[0-9]+)\.npz$")


def compose_reviewed_cache_addition(
    base_run: Path | str,
    section: str,
    raw_tile: str,
    support_bbox_xywh: tuple[int, int, int, int],
    output_dir: Path | str,
    *,
    evidence_paths: tuple[Path | str, ...],
    reviewer: str,
    notes: str,
    minimum_area_px: int = 15,
    selected_tile_label_ids: tuple[int, ...] | None = None,
    expected_model: str = "cpsam",
) -> Path:
    """Add complete, nonoverlapping raw-tile instances inside one reviewed ROI.

    The accepted source label canvas is copied exactly.  Candidate instances
    must be wholly contained in ``support_bbox_xywh``, have no pixel overlap
    with any accepted source label, and satisfy ``minimum_area_px``.  All input
    and evidence artifacts are fingerprinted before the output is written.
    ``selected_tile_label_ids`` can restrict the geometrically eligible set
    after reviewing individual raw instances. It can never admit an instance
    that fails containment, minimum area, or source-overlap checks. The exact
    selected IDs and review evidence remain sealed in the existing manifest.
    A CPSAM-v2 source requires an explicit ``expected_model="cpsam_v2"``;
    all containment-profile and artifact checks still apply to both models.
    """

    if len(section) != 3 or not section.isdigit():
        raise ValueError("section must use three digits")
    if expected_model not in {"cpsam", "cpsam_v2"}:
        raise ValueError("reviewed cache addition requires a CPSAM source model")
    if minimum_area_px != 15:
        raise ValueError("reviewed cache addition requires the validated 15 px floor")
    if not reviewer.strip() or len(reviewer) > 128:
        raise ValueError("reviewer is invalid")
    if len(notes) > 4096:
        raise ValueError("review notes are too long")
    if (
        len(support_bbox_xywh) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in support_bbox_xywh
        )
        or any(value < 0 for value in support_bbox_xywh[:2])
        or any(value <= 0 for value in support_bbox_xywh[2:])
    ):
        raise ValueError("support bbox is invalid")
    match = _TILE_KEY.fullmatch(raw_tile)
    if match is None or Path(raw_tile).name != raw_tile:
        raise ValueError("raw tile key is invalid")

    source_root = Path(base_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"reviewed cache output already exists: {destination}")
    base = validate_cell_promotion_candidate(source_root, expected_model=expected_model)
    if (
        base.get("algorithm_version")
        != REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION
    ):
        raise ValueError("reviewed cache addition requires an algorithm-75 source")
    row = _single_section(base, section, "source result")
    preflight = _object(source_root / str(base["preflight"]), "source preflight")
    preflight_row = _single_section(preflight, section, "source preflight")
    for result_key, preflight_key in (
        ("source_identity", "source_identity"),
        ("native_shape", "native_shape"),
        ("content_bbox_xywh", "content_bbox_xywh"),
        ("mpp_xy", "mpp_xy"),
    ):
        if row.get(result_key) != preflight_row.get(preflight_key):
            raise ValueError(f"source identity differs at {result_key}")

    artifacts = _mapping(base.get("artifacts"), "source artifacts")
    labels_relative = _relative(row.get("labels"), "source labels")
    qc_relative = _relative(row.get("qc"), "source QC")
    source_labels_path = source_root / labels_relative
    source_qc_path = source_root / qc_relative
    source_labels_sha256 = _sha(artifacts.get(labels_relative), "source labels")
    source_qc_sha256 = _sha(artifacts.get(qc_relative), "source QC")
    if (
        _sha256_file(source_labels_path) != source_labels_sha256
        or _sha256_file(source_qc_path) != source_qc_sha256
    ):
        raise ValueError("source artifacts changed after sealing")
    source_qc = _object(source_qc_path, "source QC")
    if source_qc.get("labels_sha256") != source_labels_sha256:
        raise ValueError("source QC label seal is stale")

    cache_path = source_root / ".cell-cache" / section / raw_tile
    if not cache_path.is_file():
        raise ValueError("reviewed raw cache tile is unavailable")
    cache_sha256 = _sha256_file(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        raw_mask = np.asarray(payload["mask"], dtype=np.int32)
        tile_fingerprint = str(payload["fingerprint"].item())
    if raw_mask.ndim != 2 or not raw_mask.size or not _is_sha256(tile_fingerprint):
        raise ValueError("reviewed raw cache payload is invalid")
    tile_x, tile_y = int(match.group("x")), int(match.group("y"))
    tile_height, tile_width = raw_mask.shape
    support_x, support_y, support_width, support_height = support_bbox_xywh
    if (
        support_x < tile_x
        or support_y < tile_y
        or support_x + support_width > tile_x + tile_width
        or support_y + support_height > tile_y + tile_height
    ):
        raise ValueError("reviewed support must be wholly inside the raw tile")

    native_shape = row.get("native_shape")
    if (
        not isinstance(native_shape, list)
        or len(native_shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in native_shape)
    ):
        raise ValueError("source native geometry is invalid")
    native_height, native_width = native_shape
    if tile_x + tile_width > native_width or tile_y + tile_height > native_height:
        raise ValueError("reviewed raw tile exceeds native geometry")

    pyvips = _import_pyvips()
    source_image = pyvips.Image.new_from_file(str(source_labels_path), access="random")
    source_tile = _read_label_crop(
        source_image,
        tile_x,
        tile_y,
        tile_width,
        tile_height,
    )
    support = np.zeros(raw_mask.shape, dtype=bool)
    local_x = support_x - tile_x
    local_y = support_y - tile_y
    support[
        local_y : local_y + support_height,
        local_x : local_x + support_width,
    ] = True
    maximum_raw = int(raw_mask.max(initial=0))
    total = np.bincount(raw_mask.ravel(), minlength=maximum_raw + 1)
    inside = np.bincount(raw_mask[support].ravel(), minlength=maximum_raw + 1)
    overlap = np.bincount(raw_mask[source_tile > 0].ravel(), minlength=maximum_raw + 1)
    novel = np.bincount(
        raw_mask[support & (source_tile == 0)].ravel(),
        minlength=maximum_raw + 1,
    )
    selection = select_bounded_additive_instances(
        total,
        inside,
        overlap,
        novel,
        minimum_area_px=minimum_area_px,
    )
    if selected_tile_label_ids is not None:
        selected = _reviewed_label_subset(
            selected_tile_label_ids, selection.selected_candidate_ids
        )
        reviewed_novel = np.zeros_like(novel)
        reviewed_novel[list(selected)] = novel[list(selected)]
        selection = select_bounded_additive_instances(
            total, inside, overlap, reviewed_novel, minimum_area_px=minimum_area_px
        )
    if selection.selected_instance_count <= 0:
        raise ValueError("reviewed cache addition selected no new instances")
    # Check the exact nearest-neighbour tissue geometry before copying a WSI.
    # Raw containment masks can cross small holes in the accepted tissue mask.
    tissue_mask = (
        np.asarray(Image.open(str(preflight_row["mask_path"])).convert("L")) > 0
    )
    tile_tissue = _tile_tissue_mask(
        tissue_mask,
        CellTile(
            row=0,
            column=0,
            x0=tile_x,
            x1=tile_x + tile_width,
            y0=tile_y,
            y1=tile_y + tile_height,
        ),
        content_shape=(native_height, native_width),
    )
    selected_mask = np.isin(raw_mask, selection.selected_candidate_ids)
    if np.any(selected_mask & ~tile_tissue):
        raise ValueError("reviewed raw labels leave the accepted tissue mask")

    evidence = tuple(Path(path).expanduser().resolve() for path in evidence_paths)
    if not evidence or any(not path.is_file() for path in evidence):
        raise ValueError("reviewed cache evidence is missing")
    evidence_sha256s = [_sha256_file(path) for path in evidence]
    if len(evidence_sha256s) != len(set(evidence_sha256s)):
        raise ValueError("reviewed cache evidence must be unique")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    canvas = None
    try:
        (temporary / "labels").mkdir()
        (temporary / "qc").mkdir()
        shutil.copy2(source_root / str(base["preflight"]), temporary / "preflight.json")
        raw_path = temporary / "labels" / f".{section}.reviewed-cache.raw"
        canvas = np.memmap(
            raw_path,
            dtype=np.uint32,
            mode="w+",
            shape=(native_height, native_width),
        )
        source_maximum = _copy_labels_to_canvas(
            source_labels_path,
            canvas,
            block_size=1024,
        )
        first_output_id = source_maximum + 1
        lookup = bounded_additive_label_lookup(
            selection,
            maximum_candidate_id=maximum_raw,
            first_output_id=first_output_id,
        )
        view = canvas[
            tile_y : tile_y + tile_height,
            tile_x : tile_x + tile_width,
        ]
        updated, added_pixels = apply_bounded_additions(
            view,
            raw_mask,
            support,
            lookup,
        )
        view[:] = updated
        if added_pixels != selection.added_pixels:
            raise RuntimeError("reviewed cache addition pixel count changed")
        canvas.flush()

        labels_output_relative = f"labels/{section}.cells.tiff"
        labels_output = temporary / labels_output_relative
        _write_pyramidal_labels(
            raw_path,
            labels_output,
            width=native_width,
            height=native_height,
        )
        labels_output_sha256 = _sha256_file(labels_output)
        tissue_mask = (
            np.asarray(Image.open(str(preflight_row["mask_path"])).convert("L")) > 0
        )
        maximum_output = source_maximum + selection.selected_instance_count
        label_qc = _disk_label_qc(
            canvas,
            maximum_output,
            tissue_mask=tissue_mask,
            content_shape=(native_height, native_width),
            min_size=minimum_area_px,
        )
        if label_qc["cell_count"] != int(source_qc["cell_count"]) + (
            selection.selected_instance_count
        ):
            raise RuntimeError("reviewed cache addition cell count changed")
        if label_qc["foreground_pixels"] != int(source_qc["foreground_pixels"]) + (
            selection.added_pixels
        ):
            raise RuntimeError("reviewed cache addition foreground count changed")
        if label_qc["outside_tissue_pixels_final"] != 0:
            raise RuntimeError(
                "reviewed cache additions leave the accepted tissue mask"
            )

        selected_ids = list(selection.selected_candidate_ids)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "scope": REVIEWED_CACHE_ADDITION_SCOPE,
            "source_algorithm_version": (
                REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION
            ),
            "source_result_fingerprint": base["fingerprint"],
            "source_preflight_fingerprint": base["preflight_fingerprint"],
            "sections": [
                {
                    "section": section,
                    "source_identity": row["source_identity"],
                    "source_labels_sha256": source_labels_sha256,
                    "source_qc_sha256": source_qc_sha256,
                    "raw_tile": {
                        "tile": raw_tile,
                        "x": tile_x,
                        "y": tile_y,
                        "width": tile_width,
                        "height": tile_height,
                        "source_cache_sha256": cache_sha256,
                        "source_tile_fingerprint": tile_fingerprint,
                    },
                    "selected_tile_label_ids": selected_ids,
                    "selected_tile_label_ids_sha256": _json_sha256(selected_ids),
                    "selection": {
                        "instance_action": REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
                        "native_coordinate_space": "source_wsi_pixels",
                        "support_geometry": {
                            "bbox_xywh": list(support_bbox_xywh),
                        },
                        "containment_rule": (
                            "all-selected-instance-pixels-inside-support-v1"
                        ),
                        "source_overlap_rule": "zero-source-overlap-pixels-v1",
                        "minimum_area_px": minimum_area_px,
                        "selected_instance_count": selection.selected_instance_count,
                        "selected_pixels": selection.added_pixels,
                        "first_output_id": first_output_id,
                        "last_output_id": maximum_output,
                        "changed_source_pixels": 0,
                        "changed_outside_support_pixels": 0,
                    },
                    "review": {
                        "decision": "approved",
                        "reviewer": reviewer,
                        "reviewed_at": datetime.now().astimezone().isoformat(),
                        "evidence_sha256s": evidence_sha256s,
                        "notes": notes,
                    },
                }
            ],
        }
        validate_reviewed_cache_addition_manifest(
            manifest,
            expected_sections=(section,),
        )
        request = json.loads(json.dumps(base["request"]))
        request["sections"] = [section]
        request["method_reference"] = {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": REVIEWED_CACHE_ADDITION_METHOD_PROFILE,
        }
        request["reviewed_cache_addition"] = manifest
        model = json.loads(json.dumps(base["model"]))
        profile_fingerprint = _json_sha256(
            {
                "algorithm_version": REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
                "preflight_fingerprint": base["preflight_fingerprint"],
                "model": model,
                "request": request,
            }
        )
        section_fingerprint = _json_sha256(
            {
                "checkpoint_schema_version": 1,
                "preflight": base["preflight_fingerprint"],
                "section": section,
                "source": row["source_identity"],
                "mask": preflight_row["mask_sha256"],
                "profile": profile_fingerprint,
            }
        )
        reused = int(source_qc["tiles_inferred"]) + int(source_qc["tiles_reused"])
        qc = json.loads(json.dumps(source_qc))
        qc.update(label_qc)
        qc.update(
            {
                "algorithm_version": REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
                "method_profile": REVIEWED_CACHE_ADDITION_METHOD_PROFILE,
                "profile_fingerprint": profile_fingerprint,
                "section_fingerprint": section_fingerprint,
                "labels_sha256": labels_output_sha256,
                "tiles_inferred": 0,
                "tiles_reused": reused,
                "tiles_reused_from_prior_run": reused,
                "filter_upgrade_from_algorithm_version": (
                    REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION
                ),
                "filter_upgrade_source_labels_sha256": source_labels_sha256,
                "reviewed_cache_addition": reviewed_cache_addition_qc_evidence(
                    manifest,
                    section,
                ),
                "reviewed_cache_addition_instances_added": (
                    selection.selected_instance_count
                ),
                "reviewed_cache_addition_pixels_added": selection.added_pixels,
                "reviewed_cache_addition_pixels_removed": 0,
                "reviewed_cache_addition_changed_source_pixels": 0,
                "reviewed_cache_addition_changed_outside_support_pixels": 0,
            }
        )
        instance_evidence = _mapping(
            qc.get("instance_evidence"), "source instance evidence"
        )
        qc["instance_evidence"] = {
            **instance_evidence,
            "reviewed_cache_addition": qc["reviewed_cache_addition"],
        }
        qc_output_relative = f"qc/{section}.json"
        (temporary / qc_output_relative).write_text(
            json.dumps(qc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_row = json.loads(json.dumps(row))
        output_row.update(
            labels=labels_output_relative,
            qc=qc_output_relative,
            cell_count=label_qc["cell_count"],
        )
        result_path = write_cell_result(
            temporary,
            {
                "schema_version": 1,
                "algorithm_version": REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
                "coordinate_space": base["coordinate_space"],
                "preflight": "preflight.json",
                "preflight_fingerprint": base["preflight_fingerprint"],
                "registration_result_sha256": base["registration_result_sha256"],
                "registration_approval_sha256": base.get(
                    "registration_approval_sha256"
                ),
                "model": model,
                "request": request,
                "profile_fingerprint": profile_fingerprint,
                "subset_source": {
                    "scope": REVIEWED_CACHE_ADDITION_SCOPE,
                    "source_result_fingerprint": base["fingerprint"],
                    "source_preflight_fingerprint": base["preflight_fingerprint"],
                    "source_profile_fingerprint": base["profile_fingerprint"],
                    "source_section_fingerprint": source_qc["section_fingerprint"],
                    "source_labels_sha256": source_labels_sha256,
                },
                "slides": [output_row],
            },
        )
        validate_cell_result(temporary)
        validate_cell_promotion_candidate(
            temporary,
            expected_model=expected_model,
            require_complete_preflight=False,
        )
        # SMB cannot rename a directory containing a delete-pending open map.
        # Release the map and cached image handles before removing its raw file.
        canvas._mmap.close()
        cache_limit = pyvips.cache_get_max()
        pyvips.cache_set_max(0)
        try:
            raw_path.unlink(missing_ok=True)
            os.replace(temporary, destination)
        finally:
            pyvips.cache_set_max(cache_limit)
        return destination / result_path.name
    except BaseException as error:
        if canvas is not None and not canvas._mmap.closed:
            canvas._mmap.close()
        # Retain unfinished work and sealed artifacts for explicit inspection.
        # A storage-finalization failure must not destroy completed composition.
        try:
            (temporary / "composition_failure.json").write_text(
                json.dumps({"error_type": type(error).__name__, "message": str(error)})
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        raise


def _read_label_crop(
    image: Any,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    crop = image.crop(x, y, width, height)
    return np.frombuffer(crop.write_to_memory(), dtype=np.uint32).reshape(
        height,
        width,
    )


def _reviewed_label_subset(
    requested: tuple[int, ...], eligible: tuple[int, ...]
) -> tuple[int, ...]:
    """Validate an explicit review restriction without relaxing geometry."""
    if (
        not isinstance(requested, tuple)
        or not requested
        or any(type(value) is not int or value <= 0 for value in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("reviewed raw label IDs must be unique positive integers")
    if not set(requested).issubset(eligible):
        raise ValueError("reviewed raw labels fail the unchanged geometry gates")
    return tuple(sorted(requested))


def _single_section(
    value: dict[str, object],
    section: str,
    label: str,
) -> dict[str, Any]:
    rows = value.get("slides")
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("section") == section]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(f"{label} does not contain section {section} exactly once")
    return matches[0]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _object(path: Path, label: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text()), label)


def _relative(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ValueError(f"{label} path is invalid")
    return Path(value).as_posix()


def _sha(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} digest is invalid")
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
