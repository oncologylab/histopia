"""Study bindings independent of the frozen protein-study evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from histopia._atomic import write_json_atomic

_ROLES = {"development", "test", "protected", "unassigned"}
_FEATURE_BINDINGS = {
    "source_sha256",
    "original_scan_id",
    "cell_ids_sha256",
    "mask_fingerprint",
    "registration_fingerprint",
    "model_fingerprint",
    "modality",
    "preprocessing",
    "normalization",
    "analysis_mpp",
    "crop_size_px",
    "software_version",
}


def fingerprint(value: object) -> str:
    """Hash strict canonical JSON, including all scientific controls."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_identity(bindings: Mapping[str, object]) -> str:
    """Refuse incomplete archive reuse, including unspecified normalization.

    A new preprocessing policy necessarily creates a different identity. A
    legacy archive tagged H–DAB must not be relabelled as an H&E RGB archive.
    """
    missing = _FEATURE_BINDINGS - bindings.keys()
    if missing or any(bindings.get(key) in (None, "", {}) for key in _FEATURE_BINDINGS):
        raise ValueError(f"incomplete feature bindings: {sorted(missing)}")
    return fingerprint(dict(bindings))


def validate_fit_scope(
    manifest: Mapping[str, object],
    fitting_mice: Sequence[str],
    *,
    evaluation_mice: Sequence[str] = (),
) -> None:
    """Use for model fitting, preprocessing, calibration and threshold fitting."""
    roles = {row["mouse_id"]: row["role"] for row in manifest["mice"]}
    if not fitting_mice or len(set(fitting_mice)) != len(fitting_mice):
        raise ValueError("fitting mice must be nonempty and unique")
    if any(roles.get(mouse) != "development" for mouse in fitting_mice):
        raise ValueError("only development mice may fit models or calibration")
    if set(fitting_mice) & set(evaluation_mice):
        raise ValueError("fitting and evaluation mice overlap")
    if any(mouse not in roles for mouse in evaluation_mice):
        raise ValueError("unknown evaluation mouse")
    if any(roles[mouse] in {"protected", "unassigned"} for mouse in evaluation_mice):
        raise ValueError("new methods cannot evaluate protected or unassigned mice")


def validate_study_manifest(payload: dict[str, object]) -> dict[str, object]:
    """Validate cross-organ mouse roles, derivative identity and evidence state.

    Unknown exposure is conservatively ineligible for an untouched test role.
    Physical spacing is never inferred from section filename order.
    """
    if payload.get("schema_version") != 1 or not payload.get("study_id"):
        raise ValueError("study requires schema_version 1 and study_id")
    mice = payload.get("mice")
    slides = payload.get("slides")
    if not isinstance(mice, list) or not isinstance(slides, list):
        raise ValueError("mice and slides must be lists")
    roles = {}
    for row in mice:
        mouse = row.get("mouse_id")
        if not isinstance(mouse, str) or not mouse or mouse in roles:
            raise ValueError("mouse identities must be nonempty and unique")
        if row.get("role") not in _ROLES:
            raise ValueError("invalid mouse role")
        if row["role"] == "test" and (
            row.get("previously_exposed") is not False
            or not row.get("exposure_evidence")
        ):
            raise ValueError("untouched test mice require verified exposure history")
        roles[mouse] = row["role"]
    ids = set()
    for row in slides:
        slide = row.get("slide_id")
        if not isinstance(slide, str) or not slide or slide in ids:
            raise ValueError("slide identities must be nonempty and unique")
        ids.add(slide)
        if row.get("mouse_id") not in roles or not row.get("organ"):
            raise ValueError("slides require a known mouse and organ")
        if row.get("role", roles[row["mouse_id"]]) != roles[row["mouse_id"]]:
            raise ValueError("every organ from a mouse must share one split")
        if not row.get("original_scan_id"):
            raise ValueError("every slide must identify its original scan")
        if row.get("is_derivative") is True and not row.get("derivative_of"):
            raise ValueError("converted images require derivative_of")
        spacing = row.get("z_spacing_um")
        spacing_kind = row.get("z_spacing_kind", "unknown")
        if spacing_kind not in {"unknown", "assumed", "physical"}:
            raise ValueError("invalid Z spacing kind")
        if spacing_kind == "unknown":
            if spacing is not None:
                raise ValueError("unknown spacing must remain missing")
        elif (
            not isinstance(spacing, (float, int))
            or isinstance(spacing, bool)
            or not math.isfinite(spacing)
            or spacing <= 0
        ):
            raise ValueError("specified Z spacing must be positive and finite")
        if spacing_kind == "physical" and not row.get("spacing_evidence"):
            raise ValueError("physical spacing requires source evidence")
    for model in payload.get("models", []):
        validate_fit_scope(
            payload,
            model["fitting_mice"],
            evaluation_mice=model.get("evaluation_mice", []),
        )
    for old in payload.get("frozen_studies", []):
        if old.get("study_id") == payload["study_id"] or not old.get("sha256"):
            raise ValueError("new study must retain separate frozen-study bindings")
    for evidence in payload.get("evidence", []):
        if evidence.get("status") not in {
            "approved",
            "exploratory",
            "pending",
            "rejected",
        }:
            raise ValueError("invalid evidence status")
        if evidence["status"] == "approved" and (
            not evidence.get("approval_fingerprint") or not evidence.get("scope")
        ):
            raise ValueError("approved evidence requires a bound approval and scope")
    core = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected = fingerprint(core)
    if payload.get("fingerprint", expected) != expected:
        raise ValueError("study fingerprint mismatch")
    return {**core, "fingerprint": expected}


def write_study_manifest(path: Path | str, payload: dict[str, object]) -> Path:
    """Freeze a new manifest; refuse to replace an existing different version."""
    path = Path(path)
    sealed = validate_study_manifest(payload)
    if path.exists() and load_study_manifest(path) != sealed:
        raise ValueError("study is frozen; choose a new version and output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, sealed)
    return path


def load_study_manifest(path: Path | str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if "fingerprint" not in payload:
        raise ValueError("study manifest is not sealed")
    return validate_study_manifest(payload)
