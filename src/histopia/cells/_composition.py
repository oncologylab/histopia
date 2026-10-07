"""Compose a complete cell result from independently validated sections.

Serial-section cohorts occasionally need a bounded rescue for one stain or
artifact class.  Recomputing every accepted slide with the rescue parameters is
both wasteful and scientifically undesirable.  This module snapshots the
accepted artifacts section by section while retaining the exact source result,
profile, preflight, label, and QC digests for every selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from histopia.cells._bounded_additive_recovery import (
    BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION,
)
from histopia.cells._chromatic import CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS
from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS,
)
from histopia.cells._foreign_material import (
    SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
)
from histopia.cells._neutral_precipitate import (
    NEUTRAL_PRECIPITATE_ALGORITHM_VERSION,
    validate_neutral_precipitate_source,
)
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_SHA256,
    CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION,
    CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
    validate_cell_promotion_candidate,
)
from histopia.cells._result import write_cell_result
from histopia.cells._reviewed_addition import (
    REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
)
from histopia.cells._reviewed_exclusion import (
    REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
)
from histopia.cells._reviewed_input_addition import (
    REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION,
)

CELL_COMPOSITE_ALGORITHM_VERSION = 85
CELL_COMPOSITE_METHOD_PROFILE = "validated-section-composite-v1"

_RESULT_IDENTITY_KEYS = (
    "section",
    "slide",
    "source_identity",
    "is_reference",
    "mpp_xy",
    "native_shape",
    "content_bbox_xywh",
    "transform_sha256",
)
_PREFLIGHT_IDENTITY_KEYS = (
    "section",
    "slide_name",
    "source_identity",
    "mask_sha256",
    "transform_sha256",
    "native_shape",
    "content_bbox_xywh",
    "thumbnail_shape",
    "mpp_xy",
    "is_reference",
)


def compose_cell_result(
    base_run: Path | str,
    output_dir: Path | str,
    replacements: Mapping[str, Path | str],
    *,
    expected_model: str = "cpsam",
) -> Path:
    """Snapshot a complete validated run with selected validated replacements.

    ``base_run`` must cover its complete preflight.  Each replacement source
    may cover either a complete preflight or an explicitly bounded subset, but
    it must pass the same scientific promotion gate and contain the requested
    section exactly once.  Label pyramids are hard-linked when the filesystem
    permits it; atomic source replacement therefore leaves the snapshot inode
    intact while avoiding duplicate multi-gigabyte storage.
    """

    base_root = Path(base_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == base_root or any(
        destination == Path(value).expanduser().resolve()
        for value in replacements.values()
    ):
        raise ValueError("composite output must differ from every source run")
    if not replacements:
        raise ValueError("at least one section replacement is required")
    normalized_replacements = {
        _section_id(section): Path(run).expanduser().resolve()
        for section, run in replacements.items()
    }
    if len(normalized_replacements) != len(replacements):
        raise ValueError("replacement sections must be unique")
    if destination.exists():
        raise FileExistsError(f"composite output already exists: {destination}")

    base = validate_cell_promotion_candidate(base_root, expected_model=expected_model)
    base_preflight = _object(base_root / str(base["preflight"]), "base preflight")
    base_rows = _rows(base, "slides", "base result")
    base_by_section = _unique_rows(base_rows, "section", "base result")
    base_preflight_by_section = _unique_rows(
        _rows(base_preflight, "slides", "base preflight"),
        "section",
        "base preflight",
    )
    unknown = sorted(set(normalized_replacements) - set(base_by_section))
    if unknown:
        raise ValueError(
            "replacement sections are not in the base run: " + ", ".join(unknown)
        )

    source_cache: dict[
        Path, tuple[dict[str, object], dict[str, dict[str, object]]]
    ] = {}
    selected: dict[str, tuple[Path, dict[str, object], dict[str, object]]] = {}
    for section, source_root in normalized_replacements.items():
        if source_root not in source_cache:
            source = validate_cell_promotion_candidate(
                source_root,
                expected_model=expected_model,
                require_complete_preflight=False,
            )
            source_rows = _unique_rows(
                _rows(source, "slides", "replacement result"),
                "section",
                "replacement result",
            )
            source_cache[source_root] = (source, source_rows)
        source, source_rows = source_cache[source_root]
        if section not in source_rows:
            raise ValueError(f"replacement run does not contain section {section}")
        _validate_registration_binding(base, source, section=section)
        source_preflight = _object(
            source_root / str(source["preflight"]), "replacement preflight"
        )
        source_preflight_rows = _unique_rows(
            _rows(source_preflight, "slides", "replacement preflight"),
            "section",
            "replacement preflight",
        )
        if section not in source_preflight_rows:
            raise ValueError(f"replacement preflight omits section {section}")
        _require_same_identity(
            base_by_section[section],
            source_rows[section],
            keys=_RESULT_IDENTITY_KEYS,
            label=f"section {section} result",
        )
        _require_same_identity(
            base_preflight_by_section[section],
            source_preflight_rows[section],
            keys=_PREFLIGHT_IDENTITY_KEYS,
            label=f"section {section} preflight",
        )
        if (
            source.get("algorithm_version")
            in DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS
        ):
            _validate_dense_recovery_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
            )
        if (
            source.get("algorithm_version")
            == BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION
        ):
            _validate_bounded_additive_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
            )
        if source.get("algorithm_version") == REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION:
            _validate_reviewed_cache_addition_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
                manifest_key="reviewed_multi_cache_addition",
            )
        if source.get("algorithm_version") == REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION:
            _validate_reviewed_cache_addition_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
                manifest_key="reviewed_input_addition",
            )
        if source.get("algorithm_version") in {
            REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
            REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
        }:
            _validate_reviewed_cache_addition_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
            )
        if source.get("algorithm_version") == NEUTRAL_PRECIPITATE_ALGORITHM_VERSION:
            _validate_neutral_precipitate_source_binding(
                base_root,
                base,
                base_by_section[section],
                source,
                section=section,
            )
        selected[section] = (source_root, source, source_rows[section])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copy2(base_root / str(base["preflight"]), temporary / "preflight.json")
        (temporary / "labels").mkdir()
        (temporary / "qc").mkdir()
        output_rows: list[dict[str, object]] = []
        section_sources: list[dict[str, object]] = []
        for base_row in base_rows:
            section = str(base_row["section"])
            source_root, source, source_row = selected.get(
                section, (base_root, base, base_row)
            )
            source_labels = source_root / str(source_row["labels"])
            source_qc = source_root / str(source_row["qc"])
            labels_relative = Path("labels") / f"{section}.cells.tiff"
            qc_relative = Path("qc") / f"{section}.json"
            _snapshot_label(source_labels, temporary / labels_relative)
            shutil.copy2(source_qc, temporary / qc_relative)
            qc = _object(source_qc, f"section {section} QC")
            source_artifacts = source.get("artifacts")
            if not isinstance(source_artifacts, dict):
                raise ValueError(
                    f"section {section}: source artifact manifest is invalid"
                )
            labels_sha = source_artifacts.get(str(source_row["labels"]))
            qc_sha = source_artifacts.get(str(source_row["qc"]))
            if not _is_sha256(labels_sha) or not _is_sha256(qc_sha):
                raise ValueError(
                    f"section {section}: source artifact digests are missing"
                )
            output_row = dict(source_row)
            output_row["labels"] = labels_relative.as_posix()
            output_row["qc"] = qc_relative.as_posix()
            output_rows.append(output_row)
            source_request = source.get("request")
            source_model = source.get("model")
            if not isinstance(source_request, dict) or not isinstance(
                source_model, dict
            ):
                raise ValueError(
                    f"section {section}: source method provenance is missing"
                )
            method_reference = source_request.get("method_reference")
            if not isinstance(method_reference, dict):
                raise ValueError(
                    f"section {section}: source method reference is missing"
                )
            entry: dict[str, object] = {
                "section": section,
                "replacement": section in selected,
                "source_result_fingerprint": source["fingerprint"],
                "source_preflight_fingerprint": source["preflight_fingerprint"],
                "source_algorithm_version": source["algorithm_version"],
                "source_method_profile": method_reference["profile"],
                "source_profile_fingerprint": source["profile_fingerprint"],
                "source_request_sha256": _json_sha256(source_request),
                "source_model_sha256": _json_sha256(source_model),
                "model_weight_name": source_model.get("weight_name"),
                "model_weight_sha256": source_model.get("weight_sha256"),
                "source_section_fingerprint": qc.get("section_fingerprint"),
                "source_labels_sha256": labels_sha,
                "source_qc_sha256": qc_sha,
            }
            if source.get("algorithm_version") == 83:
                entry["focus_artifact_gate"] = source_request.get("focus_artifact_gate")
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                in CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS
            ):
                gate_payload = source_request.get("chromatic_microcluster_gate")
                gates = (
                    gate_payload.get("sections")
                    if isinstance(gate_payload, dict)
                    else None
                )
                matching = (
                    [
                        gate
                        for gate in gates
                        if isinstance(gate, dict) and gate.get("section") == section
                    ]
                    if isinstance(gates, list)
                    else []
                )
                if len(matching) != 1:
                    raise ValueError(
                        f"section {section}: chromatic gate provenance is missing"
                    )
                entry["chromatic_microcluster_gate"] = matching[0].get("gate")
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                == SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION
            ):
                gate_payload = source_request.get(
                    "saturated_cyan_foreign_material_gate"
                )
                gates = (
                    gate_payload.get("sections")
                    if isinstance(gate_payload, dict)
                    else None
                )
                matching = (
                    [
                        gate
                        for gate in gates
                        if isinstance(gate, dict) and gate.get("section") == section
                    ]
                    if isinstance(gates, list)
                    else []
                )
                if len(matching) != 1:
                    raise ValueError(
                        f"section {section}: foreign-material gate provenance "
                        "is missing"
                    )
                entry["saturated_cyan_foreign_material_gate"] = matching[0].get("gate")
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                == REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION
            ):
                entry["reviewed_multi_cache_addition"] = source_request.get(
                    "reviewed_multi_cache_addition"
                )
                entry["subset_source"] = source.get("subset_source")
            if (
                source.get("algorithm_version")
                == REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION
            ):
                entry["reviewed_input_addition"] = source_request.get(
                    "reviewed_input_addition"
                )
                entry["subset_source"] = source.get("subset_source")
            if source.get("algorithm_version") in {
                REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
                REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
            }:
                entry["reviewed_artifact_exclusion"] = json.loads(
                    json.dumps(source_request.get("reviewed_artifact_exclusion"))
                )
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                == CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION
            ):
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION
            ):
                isolated_debris = source_request.get("isolated_debris_gate")
                entry["satellite_gate"] = (
                    isolated_debris.get("satellite_gate")
                    if isinstance(isolated_debris, dict)
                    else None
                )
                entry["filter_upgrade"] = source.get("filter_upgrade")
            if (
                source.get("algorithm_version")
                in DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS
            ):
                entry["dense_small_cell_recovery"] = source_request.get(
                    "dense_small_cell_recovery"
                )
                entry["subset_source"] = source.get("subset_source")
            if (
                source.get("algorithm_version")
                == BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION
            ):
                entry["bounded_additive_recovery"] = source_request.get(
                    "bounded_additive_recovery"
                )
                entry["subset_source"] = source.get("subset_source")
            if source.get("algorithm_version") in {
                REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
                REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
            }:
                entry["reviewed_cache_addition"] = source_request.get(
                    "reviewed_cache_addition"
                )
                entry["subset_source"] = source.get("subset_source")
            if source.get("algorithm_version") == NEUTRAL_PRECIPITATE_ALGORITHM_VERSION:
                entry["subset_source"] = source.get("subset_source")
            section_sources.append(entry)

        base_request = base.get("request")
        base_model = base.get("model")
        if not isinstance(base_request, dict) or not isinstance(base_model, dict):
            raise ValueError("base method provenance is missing")
        request = {
            "analysis_selection": base_request.get("analysis_selection"),
            "model": base_request.get("model"),
            "method": base_request.get("method"),
            "method_reference": {
                "name": CELL_METHOD_REFERENCE_NAME,
                "sha256": CELL_METHOD_REFERENCE_SHA256,
                "profile": CELL_COMPOSITE_METHOD_PROFILE,
            },
            "tissue_constraint": base_request.get("tissue_constraint"),
            "composition": "per-section-validated-method-selection-v1",
        }
        composition = {
            "schema_version": 1,
            "scope": "validated-section-composite-v1",
            "base_result_fingerprint": base["fingerprint"],
            "section_sources": section_sources,
        }
        profile_fingerprint = _json_sha256(
            {
                "algorithm_version": CELL_COMPOSITE_ALGORITHM_VERSION,
                "preflight_fingerprint": base["preflight_fingerprint"],
                "model": base_model,
                "request": request,
                "composition": composition,
            }
        )
        result_path = write_cell_result(
            temporary,
            {
                "schema_version": 1,
                "algorithm_version": CELL_COMPOSITE_ALGORITHM_VERSION,
                "coordinate_space": base.get("coordinate_space"),
                "preflight": "preflight.json",
                "preflight_fingerprint": base["preflight_fingerprint"],
                "registration_result_sha256": base["registration_result_sha256"],
                "registration_approval_sha256": base.get(
                    "registration_approval_sha256"
                ),
                "model": base_model,
                "request": request,
                "profile_fingerprint": profile_fingerprint,
                "composition": composition,
                "slides": output_rows,
            },
        )
        validate_cell_promotion_candidate(temporary, expected_model=expected_model)
        os.replace(temporary, destination)
        return destination / result_path.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _snapshot_label(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validate_registration_binding(
    base: dict[str, object], source: dict[str, object], *, section: str
) -> None:
    for key in ("registration_result_sha256", "registration_approval_sha256"):
        if source.get(key) != base.get(key):
            raise ValueError(f"section {section}: replacement {key} differs from base")
    if source.get("coordinate_space") != base.get("coordinate_space"):
        raise ValueError(f"section {section}: replacement coordinate space differs")


def _validate_dense_recovery_source_binding(
    base_root: Path,
    base: dict[str, object],
    base_row: dict[str, object],
    replacement: dict[str, object],
    *,
    section: str,
) -> None:
    """Require a dense recovery to derive from the exact selected base labels."""

    subset_source = replacement.get("subset_source")
    base_artifacts = base.get("artifacts")
    labels_relative = base_row.get("labels")
    qc_relative = base_row.get("qc")
    if (
        not isinstance(subset_source, dict)
        or not isinstance(base_artifacts, dict)
        or not isinstance(labels_relative, str)
        or not isinstance(qc_relative, str)
    ):
        raise ValueError(
            f"section {section}: dense-recovery base provenance is missing"
        )
    base_qc = _object(base_root / qc_relative, f"section {section} base QC")
    if (
        subset_source.get("source_labels_sha256") != base_artifacts.get(labels_relative)
        or subset_source.get("source_labels_sha256") != base_qc.get("labels_sha256")
        or subset_source.get("source_section_fingerprint")
        != base_qc.get("section_fingerprint")
    ):
        raise ValueError(
            f"section {section}: dense recovery was not derived from the selected "
            "base labels"
        )

    seals: list[dict[str, object]] = [
        {
            "source_result_fingerprint": base.get("fingerprint"),
            "source_preflight_fingerprint": base.get("preflight_fingerprint"),
            "source_profile_fingerprint": base.get("profile_fingerprint"),
        }
    ]
    composition = base.get("composition")
    section_sources = (
        composition.get("section_sources") if isinstance(composition, dict) else None
    )
    if isinstance(section_sources, list):
        seals.extend(
            entry
            for entry in section_sources
            if isinstance(entry, dict) and entry.get("section") == section
        )
    sealed_source = subset_source.get("source_result_fingerprint")
    active_preflight = subset_source.get("source_preflight_fingerprint")
    active_profile = subset_source.get("source_profile_fingerprint")
    if not any(
        (
            _is_sha256(sealed_source)
            and sealed_source == seal.get("source_result_fingerprint")
        )
        or (
            active_preflight == seal.get("source_preflight_fingerprint")
            and active_profile == seal.get("source_profile_fingerprint")
        )
        for seal in seals
    ):
        raise ValueError(
            f"section {section}: dense-recovery source seal differs from base"
        )


def _validate_bounded_additive_source_binding(
    base_root: Path,
    base: dict[str, object],
    base_row: dict[str, object],
    replacement: dict[str, object],
    *,
    section: str,
) -> None:
    """Require a bounded addition to preserve the exact selected base labels."""

    request = replacement.get("request")
    recovery = (
        request.get("bounded_additive_recovery") if isinstance(request, dict) else None
    )
    source = recovery.get("source") if isinstance(recovery, dict) else None
    subset = replacement.get("subset_source")
    artifacts = base.get("artifacts")
    labels_relative = base_row.get("labels")
    qc_relative = base_row.get("qc")
    if (
        not isinstance(source, dict)
        or not isinstance(subset, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(labels_relative, str)
        or not isinstance(qc_relative, str)
    ):
        raise ValueError(
            f"section {section}: bounded-additive base provenance is missing"
        )
    base_qc = _object(base_root / qc_relative, f"section {section} base QC")
    if (
        source.get("result_fingerprint") != base.get("fingerprint")
        or source.get("algorithm_version") != base.get("algorithm_version")
        or source.get("profile_fingerprint") != base.get("profile_fingerprint")
        or source.get("preflight_fingerprint") != base.get("preflight_fingerprint")
        or source.get("section_fingerprint") != base_qc.get("section_fingerprint")
        or source.get("labels_sha256") != artifacts.get(labels_relative)
        or source.get("qc_sha256") != artifacts.get(qc_relative)
        or subset.get("source_result_fingerprint") != base.get("fingerprint")
        or subset.get("source_section_fingerprint")
        != base_qc.get("section_fingerprint")
        or subset.get("source_labels_sha256") != artifacts.get(labels_relative)
    ):
        raise ValueError(
            f"section {section}: bounded-additive source binding differs from base"
        )


def _validate_reviewed_cache_addition_source_binding(
    base_root: Path,
    base: dict[str, object],
    base_row: dict[str, object],
    replacement: dict[str, object],
    *,
    section: str,
    manifest_key: str = "reviewed_cache_addition",
) -> None:
    """Require a reviewed raw-cache addition to preserve its exact base."""

    request = replacement.get("request")
    manifest = request.get(manifest_key) if isinstance(request, dict) else None
    subset = replacement.get("subset_source")
    artifacts = base.get("artifacts")
    labels_relative = base_row.get("labels")
    qc_relative = base_row.get("qc")
    rows = manifest.get("sections") if isinstance(manifest, dict) else None
    matching = (
        [row for row in rows if isinstance(row, dict) and row.get("section") == section]
        if isinstance(rows, list)
        else []
    )
    if (
        not isinstance(manifest, dict)
        or not isinstance(subset, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(labels_relative, str)
        or not isinstance(qc_relative, str)
        or len(matching) != 1
    ):
        raise ValueError(
            f"section {section}: reviewed-cache base provenance is missing"
        )
    base_qc = _object(base_root / qc_relative, f"section {section} base QC")
    manifest_row = matching[0]
    if (
        manifest.get("source_result_fingerprint") != base.get("fingerprint")
        or manifest.get("source_preflight_fingerprint")
        != base.get("preflight_fingerprint")
        or manifest_row.get("source_labels_sha256") != artifacts.get(labels_relative)
        or manifest_row.get("source_qc_sha256") != artifacts.get(qc_relative)
        or subset.get("source_result_fingerprint") != base.get("fingerprint")
        or subset.get("source_preflight_fingerprint")
        != base.get("preflight_fingerprint")
        or subset.get("source_profile_fingerprint") != base.get("profile_fingerprint")
        or subset.get("source_section_fingerprint")
        != base_qc.get("section_fingerprint")
        or subset.get("source_labels_sha256") != artifacts.get(labels_relative)
    ):
        raise ValueError(
            f"section {section}: reviewed-cache source binding differs from base"
        )


def _validate_neutral_precipitate_source_binding(
    base_root: Path,
    base: dict[str, object],
    base_row: dict[str, object],
    replacement: dict[str, object],
    *,
    section: str,
) -> None:
    """Require the neutral-only refilter to derive from selected base labels."""

    subset = validate_neutral_precipitate_source(replacement.get("subset_source"))
    artifacts = base.get("artifacts")
    labels_relative = base_row.get("labels")
    qc_relative = base_row.get("qc")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(labels_relative, str)
        or not isinstance(qc_relative, str)
    ):
        raise ValueError(
            f"section {section}: neutral-precipitate base provenance is missing"
        )
    base_qc = _object(base_root / qc_relative, f"section {section} base QC")
    if (
        subset.get("source_result_fingerprint") != base.get("fingerprint")
        or subset.get("source_section_fingerprint")
        != base_qc.get("section_fingerprint")
        or subset.get("source_labels_sha256") != artifacts.get(labels_relative)
        or subset.get("source_labels_sha256") != base_qc.get("labels_sha256")
    ):
        raise ValueError(
            f"section {section}: neutral-precipitate source binding differs from base"
        )


def _require_same_identity(
    expected: dict[str, object],
    observed: dict[str, object],
    *,
    keys: tuple[str, ...],
    label: str,
) -> None:
    missing = [key for key in keys if key not in expected or key not in observed]
    if missing:
        raise ValueError(f"{label} is missing identity fields: {', '.join(missing)}")
    if any(expected[key] != observed[key] for key in keys):
        raise ValueError(f"{label} geometry or source identity differs from base")


def _rows(payload: dict[str, object], key: str, label: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise ValueError(f"{label} {key} are invalid")
    return value  # type: ignore[return-value]


def _unique_rows(
    rows: list[dict[str, object]], key: str, label: str
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise ValueError(f"{label} {key} values must be unique strings")
        output[value] = row
    return output


def _object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _section_id(value: object) -> str:
    section = str(value)
    if not section.isdigit() or len(section) != 3:
        raise ValueError("replacement section must be a three-digit identifier")
    return section


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
