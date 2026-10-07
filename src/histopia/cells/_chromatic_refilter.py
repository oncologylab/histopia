"""Fingerprint-bound postfiltering of chromatic pseudo-cell microclusters."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.cells._chromatic import (
    CHROMATIC_MICROCLUSTER_BLUE_RATIO,
    CHROMATIC_MICROCLUSTER_POLICIES,
    blue_counterstain_core,
    chromatic_microcluster_gate,
    chromatic_microcluster_run_spec,
    clustered_chromatic_microobject_instances,
)
from histopia.cells._pipeline import _write_pyramidal_labels
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_SHA256,
    validate_cell_promotion_candidate,
)
from histopia.cells._result import write_cell_result

_BLOCK_SIZE = 1024


def refilter_chromatic_microclusters(
    source_run: Path | str,
    output_dir: Path | str,
    section_profiles: Mapping[str, str],
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Remove whole labels selected by exact validated chromatic profiles.

    This operation never invokes Cellpose. It validates and hashes the source
    result first, accumulates exact native-pixel evidence from the source WSI
    and sealed label pyramid, and snapshots a new immutable result. The source
    result and label digest remain explicit in both result and section QC.
    """

    source_root = Path(source_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == source_root:
        raise ValueError("chromatic refilter output must differ from its source")
    if destination.exists():
        raise FileExistsError(
            f"chromatic refilter output already exists: {destination}"
        )
    profiles = _validated_section_profiles(section_profiles)
    source = validate_cell_promotion_candidate(
        source_root,
        require_complete_preflight=False,
    )
    run_spec = chromatic_microcluster_run_spec(profiles.values())
    if source.get("algorithm_version") != run_spec.source_algorithm_version:
        raise ValueError(
            "chromatic refilter requires an exact validated "
            f"v{run_spec.source_algorithm_version} source"
        )
    source_rows = _unique_rows(source.get("slides"), "section", "source result")
    missing = sorted(set(profiles) - set(source_rows))
    if missing:
        raise ValueError(
            "chromatic refilter source omits sections: " + ", ".join(missing)
        )
    preflight_relative = source.get("preflight")
    if not isinstance(preflight_relative, str):
        raise ValueError("chromatic refilter source preflight is missing")
    source_preflight_path = source_root / preflight_relative
    preflight = _object(source_preflight_path, "source preflight")
    preflight_rows = _unique_rows(preflight.get("slides"), "section", "preflight")
    if not set(profiles).issubset(preflight_rows):
        raise ValueError("chromatic refilter sections differ from the source preflight")
    source_request = source.get("request")
    source_model = source.get("model")
    if not isinstance(source_request, dict) or not isinstance(source_model, dict):
        raise ValueError("chromatic refilter source method provenance is invalid")

    request = json.loads(json.dumps(source_request))
    reference = request.get("method_reference")
    if not isinstance(reference, dict):
        raise ValueError("chromatic refilter source method reference is invalid")
    reference.update(
        {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": run_spec.method_profile,
        }
    )
    request["method_reference"] = reference
    request["chromatic_microcluster_gate"] = {
        "schema_version": 1,
        "sections": [
            {
                "section": section,
                "gate": chromatic_microcluster_gate(profile),
            }
            for section, profile in profiles.items()
        ],
    }
    filter_upgrade = {
        "source_algorithm_version": source["algorithm_version"],
        "source_result_fingerprint": source["fingerprint"],
        "scope": run_spec.scope,
    }
    profile_fingerprint = _json_sha256(
        {
            "algorithm_version": run_spec.algorithm_version,
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
        output_rows: list[dict[str, object]] = []
        for section, profile in profiles.items():
            _progress(progress, f"[{section}] accumulating sealed label evidence")
            source_row = source_rows[section]
            preflight_row = preflight_rows[section]
            output_rows.append(
                _refilter_section(
                    source_root,
                    temporary,
                    source,
                    source_row,
                    preflight_row,
                    profile=profile,
                    profile_fingerprint=profile_fingerprint,
                    algorithm_version=run_spec.algorithm_version,
                    method_profile=run_spec.method_profile,
                    progress=progress,
                )
            )
        core = {
            "schema_version": 1,
            "algorithm_version": run_spec.algorithm_version,
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
    profile: str,
    profile_fingerprint: str,
    algorithm_version: int,
    method_profile: str,
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
    label_path = source_root / source_label_relative
    label_image, source_content = _open_images(
        label_path,
        Path(source_path),
        crop_x=crop_x,
        crop_y=crop_y,
        width=width,
        height=height,
    )
    areas, x_sums, y_sums, counterstain = _accumulate_evidence(
        label_image,
        source_content,
        width=width,
        height=height,
        expected_cells=source_qc.get("cell_count"),
        progress=progress,
        section=section,
    )
    policy = CHROMATIC_MICROCLUSTER_POLICIES[profile]
    selected = clustered_chromatic_microobject_instances(
        areas,
        x_sums,
        y_sums,
        counterstain,
        policy=policy,
    )
    selected_count = int(np.count_nonzero(selected))
    selected_pixels = int(areas[selected].sum())
    if selected_count == 0 or selected_pixels == 0:
        raise ValueError(
            f"section {section}: validated chromatic gate selected no instances"
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
        raise ValueError(f"section {section}: chromatic gate removed every cell")
    quantiles = np.quantile(foreground, [0.05, 0.50, 0.95]).tolist()
    labels_sha256 = _sha256_file(output_root / label_relative)
    gate = chromatic_microcluster_gate(profile)
    evidence = source_qc.get("instance_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"section {section}: source instance evidence is invalid")
    qc = json.loads(json.dumps(source_qc))
    qc.update(
        {
            "cell_count": int(foreground.size),
            "foreground_pixels": int(foreground.sum()),
            "area_px_quantiles": [round(float(value), 3) for value in quantiles],
            "checkpoint_schema_version": 1,
            "algorithm_version": algorithm_version,
            "method_profile": method_profile,
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
            "filter_upgrade_from_algorithm_version": source_result["algorithm_version"],
            "filter_upgrade_source_labels_sha256": source_label_sha256,
            "chromatic_microcluster_candidate_instances": int(
                np.count_nonzero(
                    (areas > 0)
                    & (areas <= policy.maximum_instance_area_pixels)
                    & (
                        counterstain / np.maximum(areas, 1)
                        < policy.minimum_counterstain_fraction
                    )
                )
            ),
            "chromatic_microcluster_instances_removed": selected_count,
            "chromatic_microcluster_pixels_removed": selected_pixels,
            "instance_evidence": {
                **evidence,
                "chromatic_microcluster_gate": gate,
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


def _open_images(
    labels: Path,
    source: Path,
    *,
    crop_x: int,
    crop_y: int,
    width: int,
    height: int,
) -> tuple[Any, Any]:
    import pyvips

    label_image = pyvips.Image.new_from_file(str(labels), access="random")
    if (
        label_image.width != width
        or label_image.height != height
        or label_image.bands != 1
        or label_image.format != "uint"
    ):
        raise ValueError("chromatic refilter label geometry is invalid")
    source_image = normalize_vips_rgb_uchar(
        pyvips.Image.new_from_file(str(source), access="random")
    )
    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + width > source_image.width
        or crop_y + height > source_image.height
    ):
        raise ValueError("chromatic refilter source content box leaves the WSI")
    return label_image, source_image.crop(crop_x, crop_y, width, height)


def _accumulate_evidence(
    labels: Any,
    source: Any,
    *,
    width: int,
    height: int,
    expected_cells: object,
    progress: Callable[[str], None] | None,
    section: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(expected_cells, int) or isinstance(expected_cells, bool):
        raise ValueError(f"section {section}: source cell count is invalid")
    size = max(expected_cells + 1, 2)
    areas = np.zeros(size, dtype=np.uint64)
    x_sums = np.zeros(size, dtype=np.float64)
    y_sums = np.zeros(size, dtype=np.float64)
    counterstain = np.zeros(size, dtype=np.uint64)
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
                x_sums = _grow(x_sums, new_size)
                y_sums = _grow(y_sums, new_size)
                counterstain = _grow(counterstain, new_size)
            rgb = _rgb_crop(source, x0, y0, block_width, block_height)
            blue = blue_counterstain_core(
                rgb,
                minimum_blue_ratio=CHROMATIC_MICROCLUSTER_BLUE_RATIO,
            )
            flat = label_block.ravel()
            counts = np.bincount(flat, minlength=len(areas))
            areas += counts[: len(areas)].astype(np.uint64)
            columns = np.broadcast_to(
                np.arange(x0, x0 + block_width, dtype=np.float64),
                (block_height, block_width),
            )
            rows = np.broadcast_to(
                np.arange(y0, y0 + block_height, dtype=np.float64)[:, None],
                (block_height, block_width),
            )
            x_sums += np.bincount(
                flat,
                weights=columns.ravel(),
                minlength=len(areas),
            )[: len(areas)]
            y_sums += np.bincount(
                flat,
                weights=rows.ravel(),
                minlength=len(areas),
            )[: len(areas)]
            blue_counts = np.bincount(flat[blue.ravel()], minlength=len(areas))
            counterstain += blue_counts[: len(areas)].astype(np.uint64)
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
    return areas, x_sums, y_sums, counterstain


def _write_filtered_raw(
    labels: Any,
    output: Path,
    selected: np.ndarray,
    *,
    width: int,
    height: int,
    progress: Callable[[str], None] | None,
    section: str,
) -> None:
    canvas = np.memmap(output, mode="w+", dtype=np.uint32, shape=(height, width))
    blocks = 0
    total_blocks = ((height + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * (
        (width + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    )
    for y0 in range(0, height, _BLOCK_SIZE):
        block_height = min(_BLOCK_SIZE, height - y0)
        for x0 in range(0, width, _BLOCK_SIZE):
            block_width = min(_BLOCK_SIZE, width - x0)
            block = _label_crop(labels, x0, y0, block_width, block_height)
            if block.size and int(block.max(initial=0)) >= len(selected):
                raise ValueError("chromatic refilter label exceeds evidence lookup")
            filtered = block.copy()
            filtered[selected[block]] = 0
            canvas[y0 : y0 + block_height, x0 : x0 + block_width] = filtered
            blocks += 1
            if blocks % 128 == 0 or blocks == total_blocks:
                _progress(progress, f"[{section}] output block {blocks}/{total_blocks}")
    canvas.flush()
    del canvas


def _label_crop(image: Any, x: int, y: int, width: int, height: int) -> np.ndarray:
    crop = image.crop(x, y, width, height)
    return np.frombuffer(crop.write_to_memory(), dtype=np.uint32).reshape(height, width)


def _rgb_crop(image: Any, x: int, y: int, width: int, height: int) -> np.ndarray:
    crop = image.crop(x, y, width, height)
    return np.frombuffer(crop.write_to_memory(), dtype=np.uint8).reshape(
        height, width, 3
    )


def _grow(values: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=values.dtype)
    output[: len(values)] = values
    return output


def _validated_section_profiles(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("at least one chromatic section profile is required")
    output: dict[str, str] = {}
    for raw_section, raw_profile in values.items():
        section = str(raw_section)
        profile = str(raw_profile)
        if len(section) != 3 or not section.isdigit() or section in output:
            raise ValueError("chromatic sections must be unique three-digit IDs")
        if profile not in CHROMATIC_MICROCLUSTER_POLICIES:
            raise ValueError(f"unknown chromatic microcluster profile: {profile}")
        output[section] = profile
    return output


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
        raise ValueError("chromatic refilter section identity differs from preflight")


def _section_fingerprint(
    source_result: dict[str, object],
    preflight_row: dict[str, object],
    profile_fingerprint: str,
) -> str:
    return _json_sha256(
        {
            "checkpoint_schema_version": 1,
            "preflight": source_result["preflight_fingerprint"],
            "section": preflight_row["section"],
            "source": preflight_row["source_identity"],
            "mask": preflight_row["mask_sha256"],
            "profile": profile_fingerprint,
        }
    )


def _unique_rows(value: object, key: str, label: str) -> dict[str, dict[str, object]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise ValueError(f"chromatic refilter {label} rows are invalid")
    output: dict[str, dict[str, object]] = {}
    for row in value:
        assert isinstance(row, dict)
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in output:
            raise ValueError(f"chromatic refilter {label} identifiers are invalid")
        output[identifier] = row
    return output


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"chromatic refilter {label} is unavailable") from error
    if not isinstance(value, dict):
        raise ValueError(f"chromatic refilter {label} must be an object")
    return value


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


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
