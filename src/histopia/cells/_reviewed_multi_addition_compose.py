"""Compose disjoint reviewed raw-cache recoveries in one whole-slide pass."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
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
from histopia.cells._reviewed_addition import REVIEWED_CACHE_ADDITION_INSTANCE_ACTION
from histopia.cells._reviewed_addition_compose import (
    _TILE_KEY,
    _is_sha256,
    _json_sha256,
    _object,
    _read_label_crop,
    _reviewed_label_subset,
    _sha256_file,
    _single_section,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION as VERSION,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_METHOD_PROFILE as PROFILE,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_SCOPE as SCOPE,
)
from histopia.cells._reviewed_multi_addition import (
    ReviewedCachePatch,
    require_disjoint_supports,
    reviewed_multi_addition_qc_evidence,
    validate_reviewed_multi_addition_manifest,
)
from histopia.cells._tiles import CellTile


def compose_reviewed_multi_cache_addition(
    base_run: Path | str,
    section: str,
    patches: tuple[ReviewedCachePatch, ...],
    output_dir: Path | str,
    *,
    expected_model: str = "cpsam",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Add complete reviewed raw instances without altering any source pixel.

    Each patch is tied to a raw containment tile from one sealed algorithm-75
    source. Patches must have disjoint native support rectangles, selected IDs
    must satisfy the unchanged 15-pixel floor and zero source overlap, and all
    selected pixels must be within the accepted tissue mask. The source is
    copied and the pyramidal label image is written only once. Completed input
    inference is never repeated. Temporary work is preserved on any failure.
    """
    return _compose_reviewed_addition(
        base_run,
        section,
        patches,
        output_dir,
        expected_model=expected_model,
        progress=progress,
        new_input=False,
    )


def _compose_reviewed_addition(
    base_run: Path | str,
    section: str,
    patches: tuple[Any, ...],
    output_dir: Path | str,
    *,
    expected_model: str,
    progress: Callable[[dict[str, Any]], None] | None,
    new_input: bool,
) -> Path:
    version, profile, scope = VERSION, PROFILE, SCOPE
    manifest_key = "reviewed_multi_cache_addition"
    patch_type: type = ReviewedCachePatch
    validate_manifest = validate_reviewed_multi_addition_manifest
    qc_evidence = reviewed_multi_addition_qc_evidence
    if new_input:
        from histopia.cells._reviewed_input_addition import (
            REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION,
            REVIEWED_INPUT_ADDITION_KEY,
            REVIEWED_INPUT_ADDITION_METHOD_PROFILE,
            REVIEWED_INPUT_ADDITION_SCOPE,
            ReviewedInputPatch,
            reviewed_input_addition_qc_evidence,
            validate_reviewed_input_addition_manifest,
        )

        version = REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION
        profile = REVIEWED_INPUT_ADDITION_METHOD_PROFILE
        scope = REVIEWED_INPUT_ADDITION_SCOPE
        manifest_key = REVIEWED_INPUT_ADDITION_KEY
        patch_type = ReviewedInputPatch
        validate_manifest = validate_reviewed_input_addition_manifest
        qc_evidence = reviewed_input_addition_qc_evidence
    if len(section) != 3 or not section.isdigit():
        raise ValueError("section must use three digits")
    if expected_model not in {"cpsam", "cpsam_v2"}:
        raise ValueError("reviewed recovery requires CPSAM containment")
    if not isinstance(patches, tuple) or any(
        not isinstance(p, patch_type) for p in patches
    ):
        raise ValueError("reviewed patches must be an immutable tuple")
    require_disjoint_supports(tuple(p.support_bbox_xywh for p in patches))
    root = Path(base_run).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("preserve existing reviewed multi-addition output")
    base = validate_cell_promotion_candidate(root, expected_model=expected_model)
    if base.get("algorithm_version") != 75:
        raise ValueError("reviewed multi-addition requires an algorithm-75 source")
    row = _single_section(base, section, "source result")
    preflight = _object(root / str(base["preflight"]), "source preflight")
    slide = _single_section(preflight, section, "source preflight")
    for key in ("source_identity", "native_shape", "content_bbox_xywh", "mpp_xy"):
        if row.get(key) != slide.get(key):
            raise ValueError(f"source preflight differs at {key}")
    labels_path = root / row["labels"]
    qc_path = root / row["qc"]
    labels_sha = base["artifacts"][row["labels"]]
    qc_sha = base["artifacts"][row["qc"]]
    if _sha256_file(labels_path) != labels_sha or _sha256_file(qc_path) != qc_sha:
        raise ValueError("source label or QC artifact changed")
    source_qc = _object(qc_path, "source QC")
    if source_qc.get("labels_sha256") != labels_sha:
        raise ValueError("source QC seal is stale")
    height, width = row["native_shape"]
    pyvips = _import_pyvips()
    source = pyvips.Image.new_from_file(str(labels_path), access="random")
    if source.width != width or source.height != height or source.format != "uint":
        raise ValueError("source label image geometry differs")
    tissue_path = Path(slide["mask_path"])
    if _sha256_file(tissue_path) != slide["mask_sha256"]:
        raise ValueError("accepted tissue mask changed")
    tissue = np.asarray(Image.open(str(tissue_path)).convert("L")) > 0
    if new_input:
        from histopia.cells._reviewed_input_addition_compose import _prepare_input_patch

        prepared = [
            _prepare_input_patch(
                patch, base, row, slide, source, tissue, labels_sha, qc_sha
            )
            for patch in patches
        ]
    else:
        prepared = [
            _prepare_patch(
                root,
                section,
                patch,
                source,
                tissue,
                (height, width),
                row["source_identity"],
                labels_sha,
                qc_sha,
            )
            for patch in patches
        ]
    total_instances = sum(p["selection"].selected_instance_count for p in prepared)
    total_pixels = sum(p["selection"].added_pixels for p in prepared)
    if new_input:
        from histopia.cells._reviewed_input_addition import (
            validate_input_binding_to_source,
        )

        provisional = json.loads(json.dumps([p["record"] for p in prepared]))
        next_id = 1
        for record in provisional:
            count = record["selection"]["selected_instance_count"]
            record["selection"].update(
                first_output_id=next_id, last_output_id=next_id + count - 1
            )
            next_id += count
        validate_input_binding_to_source(
            {
                "schema_version": 1,
                "scope": scope,
                "source_algorithm_version": 75,
                "source_result_fingerprint": base["fingerprint"],
                "source_preflight_fingerprint": base["preflight_fingerprint"],
                "sections": [
                    {
                        "section": section,
                        "source_identity": row["source_identity"],
                        "source_labels_sha256": labels_sha,
                        "source_qc_sha256": qc_sha,
                        "patches": provisional,
                    }
                ],
            },
            base["model"],
            base["request"],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    canvas = None

    def report(phase: str) -> None:
        if progress:
            progress(
                {
                    "phase": phase,
                    "temporary_dir": str(temporary),
                    "selected_instances": total_instances,
                    "selected_pixels": total_pixels,
                }
            )

    try:
        (temporary / "labels").mkdir()
        (temporary / "qc").mkdir()
        shutil.copy2(root / str(base["preflight"]), temporary / "preflight.json")
        raw_path = temporary / "labels" / f".{section}.reviewed-multi.raw"
        canvas = np.memmap(raw_path, dtype=np.uint32, mode="w+", shape=(height, width))
        report("copying_exact_source_labels")
        maximum = _copy_labels_to_canvas(labels_path, canvas, block_size=1024)
        first_id = maximum + 1
        manifest_patches = []
        report("applying_disjoint_whole_instances")
        for item in prepared:
            raw, support, selection = item["raw"], item["support"], item["selection"]
            record = item["record"]
            tile = record["input_tile" if new_input else "raw_tile"]
            x, y = tile["x"], tile["y"]
            lookup = bounded_additive_label_lookup(
                selection, maximum_candidate_id=int(raw.max()), first_output_id=first_id
            )
            view = canvas[y : y + raw.shape[0], x : x + raw.shape[1]]
            updated, added = apply_bounded_additions(view, raw, support, lookup)
            if added != selection.added_pixels:
                raise RuntimeError("reviewed patch pixel count changed")
            view[:] = updated
            record["selection"].update(
                first_output_id=first_id,
                last_output_id=first_id + selection.selected_instance_count - 1,
            )
            first_id += selection.selected_instance_count
            manifest_patches.append(record)
        canvas.flush()
        manifest = {
            "schema_version": 1,
            "scope": scope,
            "source_algorithm_version": 75,
            "source_result_fingerprint": base["fingerprint"],
            "source_preflight_fingerprint": base["preflight_fingerprint"],
            "sections": [
                {
                    "section": section,
                    "source_identity": row["source_identity"],
                    "source_labels_sha256": labels_sha,
                    "source_qc_sha256": qc_sha,
                    "patches": manifest_patches,
                }
            ],
        }
        validate_manifest(manifest)
        labels_relative = f"labels/{section}.cells.tiff"
        report("writing_native_pyramidal_labels")
        _write_pyramidal_labels(
            raw_path, temporary / labels_relative, width=width, height=height
        )
        report("checking_full_canvas_geometry")
        label_qc = _disk_label_qc(
            canvas,
            first_id - 1,
            tissue_mask=tissue,
            content_shape=(height, width),
            min_size=15,
        )
        if (
            label_qc["cell_count"] != int(source_qc["cell_count"]) + total_instances
            or label_qc["foreground_pixels"]
            != int(source_qc["foreground_pixels"]) + total_pixels
            or label_qc["outside_tissue_pixels_final"] != 0
        ):
            raise RuntimeError("reviewed multi-addition full-canvas QC differs")
        request = json.loads(json.dumps(base["request"]))
        request["sections"] = [section]
        request["method_reference"] = {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": profile,
        }
        request[manifest_key] = manifest
        model = json.loads(json.dumps(base["model"]))
        profile_fp = _json_sha256(
            {
                "algorithm_version": version,
                "preflight_fingerprint": base["preflight_fingerprint"],
                "model": model,
                "request": request,
            }
        )
        section_fp = _json_sha256(
            {
                "checkpoint_schema_version": 1,
                "preflight": base["preflight_fingerprint"],
                "section": section,
                "source": row["source_identity"],
                "mask": slide["mask_sha256"],
                "profile": profile_fp,
            }
        )
        reused = int(source_qc["tiles_inferred"]) + int(source_qc["tiles_reused"])
        qc = json.loads(json.dumps(source_qc))
        qc.update(label_qc)
        qc.update(
            {
                "algorithm_version": version,
                "method_profile": profile,
                "profile_fingerprint": profile_fp,
                "section_fingerprint": section_fp,
                "labels_sha256": _sha256_file(temporary / labels_relative),
                "tiles_inferred": 0,
                "tiles_reused": reused,
                "tiles_reused_from_prior_run": reused,
                "filter_upgrade_from_algorithm_version": 75,
                "filter_upgrade_source_labels_sha256": labels_sha,
                manifest_key: qc_evidence(manifest),
                f"{manifest_key}_instances_added": total_instances,
                f"{manifest_key}_pixels_added": total_pixels,
                f"{manifest_key}_pixels_removed": 0,
                f"{manifest_key}_changed_source_pixels": 0,
                f"{manifest_key}_changed_outside_support_pixels": 0,
            }
        )
        if new_input:
            qc["new_input_mask_archives_reused"] = len(
                {p["input_tile"]["mask_fingerprint"] for p in manifest_patches}
            )
        qc["instance_evidence"] = {
            **qc["instance_evidence"],
            manifest_key: qc[manifest_key],
        }
        qc_relative = f"qc/{section}.json"
        (temporary / qc_relative).write_text(
            json.dumps(qc, indent=2, sort_keys=True) + "\n"
        )
        output_row = json.loads(json.dumps(row))
        output_row.update(
            labels=labels_relative, qc=qc_relative, cell_count=label_qc["cell_count"]
        )
        report("sealing_and_validating_result")
        path = write_cell_result(
            temporary,
            {
                "schema_version": 1,
                "algorithm_version": version,
                "coordinate_space": base["coordinate_space"],
                "preflight": "preflight.json",
                "preflight_fingerprint": base["preflight_fingerprint"],
                "registration_result_sha256": base["registration_result_sha256"],
                "registration_approval_sha256": base.get(
                    "registration_approval_sha256"
                ),
                "model": model,
                "request": request,
                "profile_fingerprint": profile_fp,
                "subset_source": {
                    "scope": scope,
                    "source_result_fingerprint": base["fingerprint"],
                    "source_preflight_fingerprint": base["preflight_fingerprint"],
                    "source_profile_fingerprint": base["profile_fingerprint"],
                    "source_section_fingerprint": source_qc["section_fingerprint"],
                    "source_labels_sha256": labels_sha,
                },
                "slides": [output_row],
            },
        )
        validate_cell_result(temporary)
        validate_cell_promotion_candidate(
            temporary, expected_model=expected_model, require_complete_preflight=False
        )
        canvas._mmap.close()
        cache_limit = pyvips.cache_get_max()
        pyvips.cache_set_max(0)
        try:
            raw_path.unlink()
            os.replace(temporary, output)
        finally:
            pyvips.cache_set_max(cache_limit)
        return output / path.name
    except BaseException as error:
        if canvas is not None and not canvas._mmap.closed:
            canvas._mmap.close()
        try:
            (temporary / "composition_failure.json").write_text(
                json.dumps({"error_type": type(error).__name__, "message": str(error)})
                + "\n"
            )
        except OSError:
            pass
        raise


def _prepare_patch(
    root: Path,
    section: str,
    patch: ReviewedCachePatch,
    source: Any,
    tissue: np.ndarray,
    shape: tuple[int, int],
    source_identity: str,
    labels_sha: str,
    qc_sha: str,
) -> dict[str, Any]:
    match = _TILE_KEY.fullmatch(patch.raw_tile)
    if match is None or Path(patch.raw_tile).name != patch.raw_tile:
        raise ValueError("invalid reviewed raw tile key")
    path = root / ".cell-cache" / section / patch.raw_tile
    cache_sha = _sha256_file(path)
    with np.load(path, allow_pickle=False) as payload:
        raw = payload["mask"]
        tile_fp = str(payload["fingerprint"].item())
    if (
        raw.ndim != 2
        or not raw.size
        or raw.dtype.kind not in "iu"
        or np.any(raw < 0)
        or int(raw.max()) > np.iinfo(np.int32).max
        or not _is_sha256(tile_fp)
    ):
        raise ValueError("invalid reviewed raw mask payload")
    raw = raw.astype(np.int32, copy=False)
    x, y = int(match["x"]), int(match["y"])
    h, w = raw.shape
    sx, sy, sw, sh = patch.support_bbox_xywh
    if (
        sx < x
        or sy < y
        or sx + sw > x + w
        or sy + sh > y + h
        or x + w > shape[1]
        or y + h > shape[0]
    ):
        raise ValueError("reviewed support or raw tile leaves native geometry")
    before = _read_label_crop(source, x, y, w, h)
    support = np.zeros(raw.shape, bool)
    support[sy - y : sy - y + sh, sx - x : sx - x + sw] = True
    length = int(raw.max()) + 1
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
        content_shape=shape,
    )
    if np.any(np.isin(raw, selected) & ~tile_tissue):
        raise ValueError("reviewed instances leave accepted tissue")
    evidence = [Path(p).expanduser().resolve() for p in patch.evidence_paths]
    if not evidence or any(not p.is_file() for p in evidence):
        raise ValueError("reviewed patch evidence is missing")
    hashes = [_sha256_file(p) for p in evidence]
    record = {
        "section": section,
        "source_identity": source_identity,
        "source_labels_sha256": labels_sha,
        "source_qc_sha256": qc_sha,
        "raw_tile": {
            "tile": patch.raw_tile,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "source_cache_sha256": cache_sha,
            "source_tile_fingerprint": tile_fp,
        },
        "selected_tile_label_ids": list(selected),
        "selected_tile_label_ids_sha256": _json_sha256(list(selected)),
        "selection": {
            "instance_action": REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
            "native_coordinate_space": "source_wsi_pixels",
            "support_geometry": {"bbox_xywh": list(patch.support_bbox_xywh)},
            "containment_rule": "all-selected-instance-pixels-inside-support-v1",
            "source_overlap_rule": "zero-source-overlap-pixels-v1",
            "minimum_area_px": 15,
            "selected_instance_count": selection.selected_instance_count,
            "selected_pixels": selection.added_pixels,
            "first_output_id": 1,
            "last_output_id": selection.selected_instance_count,
            "changed_source_pixels": 0,
            "changed_outside_support_pixels": 0,
        },
        "review": {
            "decision": "approved",
            "reviewer": patch.reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "evidence_sha256s": hashes,
            "notes": patch.notes,
        },
    }
    # Validate all review metadata before allocating a whole-slide canvas.
    from histopia.cells._reviewed_addition import _validate_section

    _validate_section(record)
    return {"raw": raw, "support": support, "selection": selection, "record": record}
