"""Fingerprint-bound execution of exact reviewed whole-label exclusions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
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
    _progress,
    _section_fingerprint,
    _sha256_file,
    _unique_rows,
)
from histopia.cells._pipeline import _write_pyramidal_labels
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_SHA256,
    validate_cell_promotion_candidate,
)
from histopia.cells._result import write_cell_result
from histopia.cells._reviewed_addition import (
    validate_reviewed_cache_addition_manifest,
)
from histopia.cells._reviewed_exclusion import (
    REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
    REVIEWED_ADDITION_EXCLUSION_METHOD_PROFILE,
    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_METHOD_PROFILE,
    REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
    REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    reviewed_artifact_exclusion_manifest_sha256,
    reviewed_artifact_exclusion_qc_evidence,
    reviewed_artifact_exclusion_section_sha256,
    validate_reviewed_artifact_exclusion_manifest,
)

_BLOCK_SIZE = 1024


def refilter_reviewed_artifact_exclusions(
    source_run: Path | str,
    output_dir: Path | str,
    spec: Path | str,
    *,
    expected_model: str = "cpsam",
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Remove exactly the source-bound whole-label IDs in an approved spec."""

    if expected_model not in {"cpsam", "cpsam_v2"}:
        raise ValueError("reviewed-exclusion source model is unsupported")
    source_root = Path(source_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    spec_path = Path(spec).expanduser().resolve()
    if destination == source_root:
        raise ValueError("reviewed-exclusion output must differ from source")
    if destination.exists():
        raise FileExistsError(
            f"reviewed-exclusion output already exists: {destination}"
        )
    manifest = _object(spec_path, "reviewed exclusion spec")
    manifest_rows = validate_reviewed_artifact_exclusion_manifest(manifest)
    source = validate_cell_promotion_candidate(
        source_root,
        expected_model=expected_model,
        require_complete_preflight=False,
    )
    source_algorithm_version = source.get("algorithm_version")
    if source_algorithm_version == REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION:
        output_algorithm_version = REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION
        output_method_profile = REVIEWED_ARTIFACT_EXCLUSION_METHOD_PROFILE
    elif (
        source_algorithm_version == REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
    ):
        output_algorithm_version = REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
        output_method_profile = REVIEWED_ADDITION_EXCLUSION_METHOD_PROFILE
    else:
        raise ValueError(
            "reviewed-exclusion source must be algorithm 75 or reviewed-addition "
            "algorithm 97"
        )
    if (
        manifest.get("source_algorithm_version") != source.get("algorithm_version")
        or manifest.get("source_result_fingerprint") != source.get("fingerprint")
        or manifest.get("source_preflight_fingerprint")
        != source.get("preflight_fingerprint")
    ):
        raise ValueError("reviewed-exclusion spec differs from its sealed source")
    source_rows = _unique_rows(source.get("slides"), "section", "source result")
    missing = sorted(set(manifest_rows) - set(source_rows))
    if missing:
        raise ValueError(
            "reviewed-exclusion source omits sections: " + ", ".join(missing)
        )
    ordered_sections = tuple(
        str(row["section"])
        for row in source["slides"]  # type: ignore[index]
        if str(row["section"]) in manifest_rows
    )
    validate_reviewed_artifact_exclusion_manifest(
        manifest,
        expected_sections=ordered_sections,
    )

    preflight_relative = source.get("preflight")
    if not isinstance(preflight_relative, str):
        raise ValueError("reviewed-exclusion source preflight is missing")
    source_preflight_path = source_root / preflight_relative
    preflight = _object(source_preflight_path, "source preflight")
    preflight_rows = _unique_rows(preflight.get("slides"), "section", "preflight")
    if not set(manifest_rows).issubset(preflight_rows):
        raise ValueError("reviewed-exclusion sections differ from source preflight")
    source_request = source.get("request")
    source_model = source.get("model")
    if not isinstance(source_request, dict) or not isinstance(source_model, dict):
        raise ValueError("reviewed-exclusion source method provenance is invalid")
    if source_algorithm_version == REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION:
        addition_rows = validate_reviewed_cache_addition_manifest(
            source_request.get("reviewed_cache_addition"),
            expected_sections=ordered_sections,
        )
        for section in ordered_sections:
            selection = addition_rows[section].get("selection")
            first_output_id = (
                selection.get("first_output_id")
                if isinstance(selection, dict)
                else None
            )
            excluded_ids = manifest_rows[section].get("label_ids")
            if (
                not isinstance(first_output_id, int)
                or isinstance(first_output_id, bool)
                or not isinstance(excluded_ids, list)
                or any(int(label_id) >= first_output_id for label_id in excluded_ids)
            ):
                raise ValueError(
                    f"section {section}: chained reviewed exclusion must preserve "
                    "every reviewed addition"
                )

    request = json.loads(json.dumps(source_request))
    reference = request.get("method_reference")
    if not isinstance(reference, dict):
        raise ValueError("reviewed-exclusion source method reference is invalid")
    reference.update(
        {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": output_method_profile,
        }
    )
    request["method_reference"] = reference
    request["reviewed_artifact_exclusion"] = manifest
    filter_upgrade = {
        "source_algorithm_version": source["algorithm_version"],
        "source_result_fingerprint": source["fingerprint"],
        "scope": REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
    }
    profile_fingerprint = _json_sha256(
        {
            "algorithm_version": output_algorithm_version,
            "preflight_fingerprint": source["preflight_fingerprint"],
            "model": source_model,
            "request": request,
            "filter_upgrade": filter_upgrade,
        }
    )
    manifest_sha256 = reviewed_artifact_exclusion_manifest_sha256(manifest)

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
                manifest,
                manifest_rows[section],
                profile_fingerprint=profile_fingerprint,
                manifest_sha256=manifest_sha256,
                output_algorithm_version=output_algorithm_version,
                output_method_profile=output_method_profile,
                source_algorithm_version=int(source_algorithm_version),
                progress=progress,
            )
            for section in ordered_sections
        ]
        result_path = write_cell_result(
            temporary,
            {
                "schema_version": 1,
                "algorithm_version": output_algorithm_version,
                "coordinate_space": source.get("coordinate_space"),
                "preflight": "preflight.json",
                "preflight_fingerprint": source["preflight_fingerprint"],
                "registration_result_sha256": source["registration_result_sha256"],
                "registration_approval_sha256": source.get(
                    "registration_approval_sha256"
                ),
                "model": source_model,
                "request": request,
                "profile_fingerprint": profile_fingerprint,
                "filter_upgrade": filter_upgrade,
                **(
                    {"subset_source": json.loads(json.dumps(source["subset_source"]))}
                    if output_algorithm_version
                    == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
                    else {}
                ),
                "slides": output_rows,
            },
        )
        validate_cell_promotion_candidate(
            temporary,
            expected_model=expected_model,
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
    manifest: dict[str, object],
    manifest_row: dict[str, object],
    *,
    profile_fingerprint: str,
    manifest_sha256: str,
    output_algorithm_version: int,
    output_method_profile: str,
    source_algorithm_version: int,
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
    if (
        manifest_row.get("source_identity") != source_row.get("source_identity")
        or manifest_row.get("source_labels_sha256") != source_label_sha256
    ):
        raise ValueError(f"section {section}: reviewed source identity is stale")
    source_qc = _object(source_root / source_qc_relative, "source section QC")
    if source_qc.get("labels_sha256") != source_label_sha256:
        raise ValueError(f"section {section}: source QC label digest is stale")
    bbox = preflight_row.get("content_bbox_xywh")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(value, int) for value in bbox)
        or bbox[2] <= 0
        or bbox[3] <= 0
    ):
        raise ValueError(f"section {section}: source content box is invalid")
    _, _, width, height = bbox
    label_image = _open_label(
        source_root / source_label_relative,
        width=width,
        height=height,
    )
    label_ids = np.asarray(manifest_row["label_ids"], dtype=np.uint32)
    label_relative = Path("labels") / f"{section}.cells.tiff"
    raw_path = output_root / f".{section}.cells.raw"
    try:
        areas = _write_filtered_raw_and_measure(
            label_image,
            raw_path,
            label_ids,
            width=width,
            height=height,
            expected_cells=source_qc.get("cell_count"),
            progress=progress,
            section=section,
        )
        selected_pixels = int(areas[label_ids].sum())
        if selected_pixels <= 0:
            raise ValueError(f"section {section}: reviewed labels contain no pixels")
        _write_pyramidal_labels(
            raw_path,
            output_root / label_relative,
            width=width,
            height=height,
        )
    finally:
        raw_path.unlink(missing_ok=True)

    final_areas = areas.copy()
    final_areas[label_ids] = 0
    foreground = final_areas[1:]
    foreground = foreground[foreground > 0]
    if foreground.size == 0:
        raise ValueError(f"section {section}: reviewed exclusion removed every cell")
    quantiles = np.quantile(foreground, [0.05, 0.50, 0.95]).tolist()
    labels_sha256 = _sha256_file(output_root / label_relative)
    evidence = source_qc.get("instance_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"section {section}: source instance evidence is invalid")
    section_sha256 = reviewed_artifact_exclusion_section_sha256(manifest_row)
    if manifest_sha256 != reviewed_artifact_exclusion_manifest_sha256(manifest):
        raise ValueError("reviewed exclusion manifest changed during filtering")
    qc = json.loads(json.dumps(source_qc))
    qc.update(
        {
            "cell_count": int(foreground.size),
            "foreground_pixels": int(foreground.sum()),
            "area_px_quantiles": [round(float(value), 3) for value in quantiles],
            "checkpoint_schema_version": 1,
            "algorithm_version": output_algorithm_version,
            "method_profile": output_method_profile,
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
            "tiles_reused_from_prior_run": (
                int(source_qc["tiles_reused_from_prior_run"])
                if source_algorithm_version
                == REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
                else 0
            ),
            "filter_upgrade_from_algorithm_version": (source_algorithm_version),
            "filter_upgrade_source_labels_sha256": source_label_sha256,
            "reviewed_artifact_candidate_instances": int(label_ids.size),
            "reviewed_artifact_instances_removed": int(label_ids.size),
            "reviewed_artifact_pixels_removed": selected_pixels,
            "reviewed_artifact_manifest_sha256": manifest_sha256,
            "reviewed_artifact_section_sha256": section_sha256,
            "reviewed_artifact_label_ids_sha256": manifest_row["label_ids_sha256"],
            "instance_evidence": {
                **evidence,
                "reviewed_artifact_exclusion": (
                    reviewed_artifact_exclusion_qc_evidence(manifest, section)
                ),
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


def _write_filtered_raw_and_measure(
    labels: Any,
    output: Path,
    label_ids: np.ndarray,
    *,
    width: int,
    height: int,
    expected_cells: object,
    progress: Callable[[str], None] | None,
    section: str,
) -> np.ndarray:
    if not isinstance(expected_cells, int) or isinstance(expected_cells, bool):
        raise ValueError(f"section {section}: source cell count is invalid")
    lookup_size = max(expected_cells + 1, int(label_ids.max()) + 1, 2)
    selected = np.zeros(lookup_size, dtype=bool)
    selected[label_ids] = True
    areas = np.zeros(lookup_size, dtype=np.uint64)
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
            maximum = int(block.max(initial=0))
            if maximum >= len(areas):
                areas = _grow(areas, maximum + 1)
                selected = _grow(selected, maximum + 1)
                selected[label_ids] = True
            counts = np.bincount(block.ravel(), minlength=len(areas))
            areas += counts[: len(areas)].astype(np.uint64)
            filtered = block.copy()
            filtered[selected[block]] = 0
            canvas[y0 : y0 + block_height, x0 : x0 + block_width] = filtered
            blocks += 1
            if blocks % 128 == 0 or blocks == total_blocks:
                _progress(
                    progress,
                    f"[{section}] reviewed output block {blocks}/{total_blocks}",
                )
    canvas.flush()
    del canvas
    observed_cells = int(np.count_nonzero(areas[1:]))
    if observed_cells != expected_cells:
        raise ValueError(
            f"section {section}: source cell count differs from label pyramid"
        )
    missing = label_ids[areas[label_ids] == 0]
    if missing.size:
        preview = ", ".join(str(int(value)) for value in missing[:10])
        raise ValueError(
            f"section {section}: reviewed labels are absent from source: {preview}"
        )
    return areas


def _open_label(path: Path, *, width: int, height: int) -> Any:
    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="random")
    if (
        image.width != width
        or image.height != height
        or image.bands != 1
        or image.format != "uint"
    ):
        raise ValueError("reviewed-exclusion label geometry is invalid")
    return image


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
        raise ValueError("reviewed-exclusion section identity differs from preflight")
