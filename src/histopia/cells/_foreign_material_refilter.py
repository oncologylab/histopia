"""Fingerprint-bound removal of saturated-cyan foreign-material labels."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.cells._chromatic_refilter import (
    _grow,
    _is_sha256,
    _json_sha256,
    _label_crop,
    _object,
    _open_images,
    _progress,
    _rgb_crop,
    _section_fingerprint,
    _sha256_file,
    _unique_rows,
    _write_filtered_raw,
)
from histopia.cells._foreign_material import (
    SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
    SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE,
    SATURATED_CYAN_FOREIGN_MATERIAL_POLICY,
    SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE,
    saturated_cyan_foreign_material_core,
    saturated_cyan_foreign_material_gate,
    saturated_cyan_foreign_material_instances,
)
from histopia.cells._pipeline import _write_pyramidal_labels
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_SHA256,
    validate_cell_promotion_candidate,
)
from histopia.cells._result import write_cell_result

_BLOCK_SIZE = 1024
_SOURCE_ALGORITHM_VERSION = 75


def refilter_saturated_cyan_foreign_material(
    source_run: Path | str,
    output_dir: Path | str,
    sections: Iterable[str],
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Remove labels supported by the exact saturated-cyan native-pixel gate."""

    source_root = Path(source_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == source_root:
        raise ValueError("foreign-material refilter output must differ from source")
    if destination.exists():
        raise FileExistsError(
            f"foreign-material refilter output already exists: {destination}"
        )
    requested = _validated_sections(sections)
    source = validate_cell_promotion_candidate(
        source_root,
        require_complete_preflight=False,
    )
    if source.get("algorithm_version") != _SOURCE_ALGORITHM_VERSION:
        raise ValueError(
            "saturated-cyan refilter requires an exact validated v75 source"
        )
    source_rows = _unique_rows(source.get("slides"), "section", "source result")
    missing = sorted(set(requested) - set(source_rows))
    if missing:
        raise ValueError(
            "foreign-material refilter source omits sections: " + ", ".join(missing)
        )
    ordered_sections = tuple(
        str(row["section"])
        for row in source["slides"]  # type: ignore[index]
        if str(row["section"]) in requested
    )
    preflight_relative = source.get("preflight")
    if not isinstance(preflight_relative, str):
        raise ValueError("foreign-material refilter source preflight is missing")
    source_preflight_path = source_root / preflight_relative
    preflight = _object(source_preflight_path, "source preflight")
    preflight_rows = _unique_rows(preflight.get("slides"), "section", "preflight")
    if not set(requested).issubset(preflight_rows):
        raise ValueError(
            "foreign-material refilter sections differ from the source preflight"
        )
    source_request = source.get("request")
    source_model = source.get("model")
    if not isinstance(source_request, dict) or not isinstance(source_model, dict):
        raise ValueError("foreign-material refilter source provenance is invalid")

    request = json.loads(json.dumps(source_request))
    reference = request.get("method_reference")
    if not isinstance(reference, dict):
        raise ValueError("foreign-material refilter method reference is invalid")
    reference.update(
        {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE,
        }
    )
    gate = saturated_cyan_foreign_material_gate()
    request["saturated_cyan_foreign_material_gate"] = {
        "schema_version": 1,
        "sections": [
            {"section": section, "gate": gate} for section in ordered_sections
        ],
    }
    filter_upgrade = {
        "source_algorithm_version": _SOURCE_ALGORITHM_VERSION,
        "source_result_fingerprint": source["fingerprint"],
        "scope": SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE,
    }
    profile_fingerprint = _json_sha256(
        {
            "algorithm_version": (SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION),
            "preflight_fingerprint": source["preflight_fingerprint"],
            "model": source_model,
            "request": request,
            "filter_upgrade": filter_upgrade,
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copy2(source_preflight_path, temporary / "preflight.json")
        (temporary / "labels").mkdir()
        (temporary / "qc").mkdir()
        output_rows = [
            _refilter_section(
                source_root,
                temporary,
                source,
                source_rows[section],
                preflight_rows[section],
                profile_fingerprint=profile_fingerprint,
                progress=progress,
            )
            for section in ordered_sections
        ]
        core = {
            "schema_version": 1,
            "algorithm_version": SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
            "coordinate_space": source.get("coordinate_space"),
            "preflight": "preflight.json",
            "preflight_fingerprint": source["preflight_fingerprint"],
            "registration_result_sha256": source["registration_result_sha256"],
            "registration_approval_sha256": source.get("registration_approval_sha256"),
            "model": source_model,
            "request": request,
            "profile_fingerprint": profile_fingerprint,
            "filter_upgrade": filter_upgrade,
            "slides": output_rows,
        }
        result_path = write_cell_result(temporary, core)
        validate_cell_promotion_candidate(
            temporary,
            require_complete_preflight=False,
        )
        os.replace(temporary, destination)
        return destination / result_path.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _refilter_section(
    source_root: Path,
    output_root: Path,
    source_result: dict[str, object],
    source_row: dict[str, object],
    preflight_row: dict[str, object],
    *,
    profile_fingerprint: str,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    section = str(source_row["section"])
    _require_section_identity(source_row, preflight_row)
    source_label_relative = source_row.get("labels")
    source_qc_relative = source_row.get("qc")
    artifacts = source_result.get("artifacts")
    if (
        not isinstance(source_label_relative, str)
        or not isinstance(source_qc_relative, str)
        or not isinstance(artifacts, dict)
    ):
        raise ValueError(f"section {section}: source artifacts are invalid")
    source_label_sha256 = artifacts.get(source_label_relative)
    if not _is_sha256(source_label_sha256):
        raise ValueError(f"section {section}: source label digest is invalid")
    source_qc = _object(source_root / source_qc_relative, "source section QC")
    if source_qc.get("labels_sha256") != source_label_sha256:
        raise ValueError(f"section {section}: source QC label digest is stale")

    source_path = preflight_row.get("source_path")
    bbox = preflight_row.get("content_bbox_xywh")
    if (
        not isinstance(source_path, str)
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(value, int) for value in bbox)
    ):
        raise ValueError(f"section {section}: source WSI geometry is invalid")
    crop_x, crop_y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"section {section}: source content box is invalid")
    label_image, source_content = _open_images(
        source_root / source_label_relative,
        Path(source_path),
        crop_x=crop_x,
        crop_y=crop_y,
        width=width,
        height=height,
    )
    areas, cyan_pixels = _accumulate_evidence(
        label_image,
        source_content,
        width=width,
        height=height,
        expected_cells=source_qc.get("cell_count"),
        progress=progress,
        section=section,
    )
    selected = saturated_cyan_foreign_material_instances(areas, cyan_pixels)
    selected_count = int(np.count_nonzero(selected))
    selected_pixels = int(areas[selected].sum())
    if selected_count == 0 or selected_pixels == 0:
        raise ValueError(
            f"section {section}: saturated-cyan gate selected no instances"
        )
    _progress(
        progress,
        f"[{section}] removing {selected_count} labels / {selected_pixels} pixels",
    )

    label_relative = Path("labels") / f"{section}.cells.tiff"
    raw_path = output_root / f".{section}.cells.raw"
    try:
        _write_filtered_raw(
            label_image,
            raw_path,
            selected,
            width=width,
            height=height,
            progress=progress,
            section=section,
        )
        _write_pyramidal_labels(
            raw_path,
            output_root / label_relative,
            width=width,
            height=height,
        )
    finally:
        raw_path.unlink(missing_ok=True)

    final_areas = areas.copy()
    final_areas[selected] = 0
    foreground = final_areas[1:]
    foreground = foreground[foreground > 0]
    if foreground.size == 0:
        raise ValueError(f"section {section}: cyan gate removed every cell")
    quantiles = np.quantile(foreground, [0.05, 0.50, 0.95]).tolist()
    labels_sha256 = _sha256_file(output_root / label_relative)
    gate = saturated_cyan_foreign_material_gate()
    evidence = source_qc.get("instance_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"section {section}: source instance evidence is invalid")
    policy = SATURATED_CYAN_FOREIGN_MATERIAL_POLICY
    candidate = (
        (areas > 0)
        & (cyan_pixels >= policy.minimum_instance_pixels)
        & (
            cyan_pixels
            >= np.ceil(
                areas.astype(np.float64) * policy.minimum_instance_fraction
            ).astype(np.uint64)
        )
    )
    qc = json.loads(json.dumps(source_qc))
    qc.update(
        {
            "cell_count": int(foreground.size),
            "foreground_pixels": int(foreground.sum()),
            "area_px_quantiles": [round(float(value), 3) for value in quantiles],
            "checkpoint_schema_version": 1,
            "algorithm_version": SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
            "method_profile": SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE,
            "profile_fingerprint": profile_fingerprint,
            "section_fingerprint": _section_fingerprint(
                source_result,
                preflight_row,
                profile_fingerprint,
            ),
            "labels_sha256": labels_sha256,
            "tiles_inferred": 0,
            "tiles_reused": int(source_qc["tiles_total"])
            - int(source_qc["tiles_skipped_outside_tissue"]),
            "tiles_reused_from_prior_run": 0,
            "filter_upgrade_from_algorithm_version": _SOURCE_ALGORITHM_VERSION,
            "filter_upgrade_source_labels_sha256": source_label_sha256,
            "saturated_cyan_foreign_material_candidate_instances": int(
                np.count_nonzero(candidate)
            ),
            "saturated_cyan_foreign_material_instances_removed": selected_count,
            "saturated_cyan_foreign_material_pixels_removed": selected_pixels,
            "saturated_cyan_foreign_material_support_pixels": int(
                cyan_pixels[selected].sum()
            ),
            "instance_evidence": {
                **evidence,
                "saturated_cyan_foreign_material_gate": gate,
            },
        }
    )
    qc_relative = Path("qc") / f"{section}.json"
    write_json_atomic(output_root / qc_relative, qc)
    return {
        key: source_row[key]
        for key in (
            "section",
            "slide",
            "source_identity",
            "is_reference",
            "mpp_xy",
            "native_shape",
            "content_bbox_xywh",
            "transform_sha256",
        )
    } | {
        "labels": label_relative.as_posix(),
        "qc": qc_relative.as_posix(),
        "cell_count": int(foreground.size),
    }


def _accumulate_evidence(
    labels: Any,
    source: Any,
    *,
    width: int,
    height: int,
    expected_cells: object,
    progress: Callable[[str], None] | None,
    section: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(expected_cells, int) or isinstance(expected_cells, bool):
        raise ValueError(f"section {section}: source cell count is invalid")
    size = max(expected_cells + 1, 2)
    areas = np.zeros(size, dtype=np.uint64)
    cyan_pixels = np.zeros(size, dtype=np.uint64)
    blocks = 0
    total_blocks = ((height + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * (
        (width + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    )
    for y0 in range(0, height, _BLOCK_SIZE):
        block_height = min(_BLOCK_SIZE, height - y0)
        for x0 in range(0, width, _BLOCK_SIZE):
            block_width = min(_BLOCK_SIZE, width - x0)
            label_block = _label_crop(labels, x0, y0, block_width, block_height)
            maximum = int(label_block.max(initial=0))
            if maximum >= len(areas):
                new_size = maximum + 1
                areas = _grow(areas, new_size)
                cyan_pixels = _grow(cyan_pixels, new_size)
            rgb = _rgb_crop(source, x0, y0, block_width, block_height)
            core = saturated_cyan_foreign_material_core(rgb)
            flat = label_block.ravel()
            counts = np.bincount(flat, minlength=len(areas))
            areas += counts[: len(areas)].astype(np.uint64)
            cyan_counts = np.bincount(flat[core.ravel()], minlength=len(areas))
            cyan_pixels += cyan_counts[: len(areas)].astype(np.uint64)
            blocks += 1
            if blocks % 64 == 0 or blocks == total_blocks:
                _progress(
                    progress,
                    f"[{section}] evidence block {blocks}/{total_blocks}",
                )
    observed_cells = int(np.count_nonzero(areas[1:]))
    if observed_cells != expected_cells:
        raise ValueError(
            f"section {section}: source cell count differs from its label pyramid"
        )
    return areas, cyan_pixels


def _validated_sections(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        section = str(value)
        if len(section) != 3 or not section.isdigit() or section in output:
            raise ValueError("foreign-material sections must be unique three-digit IDs")
        output.append(section)
    if not output:
        raise ValueError("at least one foreign-material section is required")
    return tuple(output)


def _require_section_identity(
    result_row: dict[str, object], preflight_row: dict[str, object]
) -> None:
    pairs = (
        ("section", "section"),
        ("slide", "slide_name"),
        ("source_identity", "source_identity"),
        ("is_reference", "is_reference"),
        ("mpp_xy", "mpp_xy"),
        ("native_shape", "native_shape"),
        ("content_bbox_xywh", "content_bbox_xywh"),
        ("transform_sha256", "transform_sha256"),
    )
    if any(result_row.get(left) != preflight_row.get(right) for left, right in pairs):
        raise ValueError(
            "foreign-material refilter section identity differs from preflight"
        )
