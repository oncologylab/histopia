"""Sealed protein-run manifests and fingerprint-bound approval state."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from histopia._atomic import write_json_atomic

_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ADAPTIVE_MEASUREMENT_VIEW = "tissue-masked-adaptive-corrected-target-od-4um-v1"
_HARMONIZED_MEASUREMENT_VIEW = (
    "tissue-masked-harmonized-adaptive-corrected-target-od-4um-v2"
)
_COUNTERSTAIN_MEASUREMENT_VIEW = (
    "tissue-masked-counterstain-conditioned-target-od-4um-v3"
)
_ADAPTIVE_MEASUREMENT_VIEWS = frozenset(
    {
        _ADAPTIVE_MEASUREMENT_VIEW,
        _HARMONIZED_MEASUREMENT_VIEW,
        _COUNTERSTAIN_MEASUREMENT_VIEW,
    }
)
_MORPHOLOGY_TRANSFER_METHODS = frozenset(
    {
        "calibrated-neural-plus-training-only-morphology-od-transfer-v1",
        "neural-plus-training-only-morphology-od-transfer-v2",
    }
)
_REGISTERED_RESIDUAL_METHOD = "confidence-weighted-registered-residual-transfer-v5"


def write_protein_result(root: Path | str, core: dict[str, object]) -> Path:
    """Seal declared artifacts and write an unapproved result manifest."""

    directory = Path(root)
    if core.get("schema_version") == 4:
        _validate_v4_metadata(core)
    payload = _seal(directory, core)
    path = write_json_atomic(directory / "protein_result.json", payload)
    approval = directory / "protein_approval.json"
    current = _current_approval(approval, str(payload["fingerprint"]))
    write_json_atomic(approval, current)
    return path


def validate_protein_result(
    root: Path | str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    """Validate schema, result fingerprint, and every portable artifact."""

    loaded = validate_protein_result_index(root, payload)
    directory = Path(root)
    references = _references(directory, loaded)
    artifacts = loaded["artifacts"]
    assert isinstance(artifacts, dict)
    for relative, path in references.items():
        if artifacts[relative] != _sha256_file(path):
            raise ValueError(f"protein artifact is missing or stale: {relative}")
    return loaded


def validate_protein_result_index(
    root: Path | str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    """Validate a sealed result index without eagerly reading large artifacts.

    This is intended for metadata catalogs that can contain many whole-study
    models.  Every reference must exist and the result fingerprint must bind
    its declared artifact digests.  Consumers must still call
    :func:`validate_protein_artifact` before opening an individual artifact.
    """

    directory = Path(root)
    loaded = (
        json.loads((directory / "protein_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") not in {1, 2, 3, 4}:
        raise ValueError("protein result must use schema version 1, 2, 3, or 4")
    if loaded.get("schema_version") == 4:
        _validate_v4_metadata(loaded)
    references = _references(directory, loaded)
    artifacts = loaded.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(references):
        raise ValueError("protein artifact manifest is incomplete or stale")
    for relative, path in references.items():
        digest = artifacts[relative]
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"protein artifact is missing or stale: {relative}")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if loaded.get("fingerprint") != _fingerprint(core):
        raise ValueError("protein result fingerprint is stale")
    return loaded


def validate_protein_artifact(path: Path | str, expected_sha256: str) -> Path:
    """Validate one lazily opened artifact against its sealed SHA-256 digest."""

    artifact = Path(path)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not artifact.is_file()
        or _sha256_file(artifact) != expected_sha256
    ):
        raise ValueError(f"protein artifact is missing or stale: {artifact.name}")
    return artifact


def approve_protein_result(
    root: Path | str,
    *,
    reviewer: str,
    accepted: bool,
    reasons: tuple[str, ...] = (),
) -> Path:
    """Write explicit approval bound to the current sealed result."""

    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    result = validate_protein_result(root)
    payload = {
        "schema_version": 1,
        "fingerprint": result["fingerprint"],
        "accepted": bool(accepted),
        "reviewer": reviewer.strip(),
        "reasons": list(reasons),
    }
    return write_json_atomic(Path(root) / "protein_approval.json", payload)


def validate_protein_approval(root: Path | str) -> dict[str, object]:
    """Require an accepted approval for the current result fingerprint."""

    result = validate_protein_result(root)
    payload = json.loads((Path(root) / "protein_approval.json").read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("fingerprint") != result["fingerprint"]
        or payload.get("accepted") is not True
    ):
        raise ValueError("protein result is not approved for its current fingerprint")
    return payload


def reassess_protein_result(root: Path | str) -> Path:
    """Reapply the current study promotion policy without changing artifacts."""

    result = validate_protein_result(root)
    if result.get("schema_version") != 4:
        raise ValueError("promotion reassessment requires a schema-v4 study result")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("protein study result has no validation metrics")
    from histopia.protein._selection import evaluate_protein_promotion

    decision = evaluate_protein_promotion(metrics)
    core = {
        key: value
        for key, value in result.items()
        if key not in {"artifacts", "fingerprint"}
    }
    previous = core.get("candidate_promotion")
    promotion = dict(previous) if isinstance(previous, dict) else {}
    promotion.update(
        {
            "accepted": decision.accepted,
            "reasons": list(decision.reasons),
            "policy": "held-out-accuracy-baseline-parity-and-od-bias-v1",
        }
    )
    core["status"] = "promoted" if decision.accepted else "candidate"
    core["candidate_promotion"] = promotion
    return write_protein_result(root, core)


def _seal(root: Path, core: dict[str, object]) -> dict[str, object]:
    sealed = dict(core)
    references = _references(root, sealed)
    sealed["artifacts"] = {
        relative: _sha256_file(path) for relative, path in sorted(references.items())
    }
    return {**sealed, "fingerprint": _fingerprint(sealed)}


def _references(root: Path, payload: dict[str, object]) -> dict[str, Path]:
    values: list[object] = [payload.get("model")]
    if payload.get("schema_version") == 4:
        values.append(payload.get("training_table"))
        model_artifacts = payload.get("model_artifacts", [])
        if not isinstance(model_artifacts, list):
            raise ValueError("protein model_artifacts must be a list")
        values.extend(model_artifacts)
    if payload.get("measurement_audit") is not None:
        values.append(payload.get("measurement_audit"))
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("protein result must contain one or more slide predictions")
    for row in slides:
        if not isinstance(row, dict):
            raise ValueError("protein result slide rows must be objects")
        values.append(row.get("predictions"))
    references: dict[str, Path] = {}
    resolved_root = root.resolve()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("protein result artifact paths must be non-empty strings")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("protein artifacts must stay inside their run")
        resolved = (resolved_root / relative).resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("protein artifacts must stay inside their run")
        references[relative.as_posix()] = resolved
    if len(references) != len(values):
        raise ValueError("protein result artifacts must be unique")
    return references


def _validate_v4_metadata(payload: dict[str, object]) -> None:
    """Validate path-free study/model identity before artifact sealing."""

    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("schema-v4 protein result model_id is invalid")
    for key in ("target_id", "architecture", "model_version"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"schema-v4 protein result {key} is required")
    implementation = payload.get("implementation")
    if implementation is not None and (
        payload.get("architecture") != "extra_trees"
        or not isinstance(implementation, dict)
        or set(implementation) != {"serialization", "scikit_learn", "joblib", "numpy"}
        or implementation.get("serialization") != "joblib-pickle-exact-runtime-v1"
        or any(
            not isinstance(implementation.get(key), str) or not implementation.get(key)
            for key in ("scikit_learn", "joblib", "numpy")
        )
    ):
        raise ValueError("schema-v4 protein runtime implementation is invalid")
    prediction_protocol = payload.get("prediction_protocol", "leave-one-mouse-out")
    if prediction_protocol not in {
        "leave-one-mouse-out",
        "training-visible",
    }:
        raise ValueError("schema-v4 protein prediction protocol is invalid")
    measurement_view = payload.get("measurement_view")
    if measurement_view not in _ADAPTIVE_MEASUREMENT_VIEWS:
        raise ValueError("schema-v4 protein result requires strict adaptive target OD")
    if measurement_view == _HARMONIZED_MEASUREMENT_VIEW:
        if payload.get("measurement_source_view") != _ADAPTIVE_MEASUREMENT_VIEW:
            raise ValueError(
                "harmonized protein result must declare its strict adaptive source"
            )
        from histopia.protein._study_workflow import (
            _validate_study_od_calibration,
        )

        _validate_study_od_calibration(payload.get("od_calibration"))
    elif measurement_view == _COUNTERSTAIN_MEASUREMENT_VIEW:
        if (
            payload.get("measurement_source_view") != _COUNTERSTAIN_MEASUREMENT_VIEW
            or payload.get("od_calibration") is not None
        ):
            raise ValueError(
                "counterstain-conditioned protein result must preserve "
                "its unharmonized measurement"
            )
    statistic = payload.get("measurement_statistic", "mean")
    if statistic not in {"mean", "q90"}:
        raise ValueError("schema-v4 protein measurement statistic is invalid")
    cohorts = payload.get("training_cohorts")
    if (
        not isinstance(cohorts, list)
        or not cohorts
        or any(not isinstance(value, str) or not value for value in cohorts)
        or len(set(cohorts)) != len(cohorts)
    ):
        raise ValueError("schema-v4 protein training cohorts are invalid")
    bindings = payload.get("cohort_bindings")
    if not isinstance(bindings, dict) or not set(cohorts).issubset(bindings):
        raise ValueError("schema-v4 protein cohort bindings are incomplete")
    required = {
        "registration_result_sha256",
        "cell_result_fingerprint",
        "stain_result_fingerprint",
        "semantic_result_fingerprint",
    }
    for cohort, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != required:
            raise ValueError(f"protein cohort binding is incomplete: {cohort}")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in binding.values()
        ):
            raise ValueError(f"protein cohort binding digest is invalid: {cohort}")
    model_reuse = payload.get("model_reuse")
    if model_reuse is not None and (
        not isinstance(model_reuse, dict)
        or set(model_reuse) != {"scope", "source_result_fingerprint"}
        or model_reuse.get("scope") != "identical-training-scope-v1"
        or not isinstance(model_reuse.get("source_result_fingerprint"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(model_reuse.get("source_result_fingerprint")),
        )
        is None
    ):
        raise ValueError("schema-v4 protein model reuse provenance is invalid")
    refinement = payload.get("postfit_refinement")
    refinement_keys = {
        "schema_version",
        "method",
        "weight",
        "neighbors",
        "evidence_sha256",
        "base_model_fingerprint",
        "base_model_sha256",
        "development_cohorts",
        "confirmation_cohorts",
        "confirmation_gate",
        "evidence_source_result_fingerprints",
    }
    if refinement is not None:
        if (
            not isinstance(refinement, dict)
            or set(refinement) != refinement_keys
            or payload.get("architecture")
            not in {"dual_bank_attention", "graph_transformer", "multi_tower"}
            or refinement.get("schema_version") != 1
            or refinement.get("method") not in _MORPHOLOGY_TRANSFER_METHODS
            or not isinstance(refinement.get("weight"), (int, float))
            or isinstance(refinement.get("weight"), bool)
            or not 0 < float(refinement["weight"]) <= 1
            or not isinstance(refinement.get("neighbors"), int)
            or isinstance(refinement.get("neighbors"), bool)
            or int(refinement["neighbors"]) < 1
            or refinement.get("confirmation_gate")
            != "accepted-once-after-development-freeze"
        ):
            raise ValueError("schema-v4 protein post-fit refinement is invalid")
        for key in (
            "evidence_sha256",
            "base_model_fingerprint",
            "base_model_sha256",
        ):
            if (
                not isinstance(refinement.get(key), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(refinement.get(key))) is None
            ):
                raise ValueError("schema-v4 protein post-fit refinement is invalid")
        for key in (
            "development_cohorts",
            "confirmation_cohorts",
            "evidence_source_result_fingerprints",
        ):
            values = refinement.get(key)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("schema-v4 protein post-fit refinement is invalid")
        if set(refinement["development_cohorts"]) & set(
            refinement["confirmation_cohorts"]
        ):
            raise ValueError("schema-v4 protein refinement cohorts overlap")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in refinement["evidence_source_result_fingerprints"]
        ):
            raise ValueError("schema-v4 protein post-fit refinement is invalid")
    _validate_registered_residual_refinement(payload)
    slides = payload.get("slides")
    if not isinstance(slides, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("cohort"), str)
        or not row.get("cohort")
        for row in slides
    ):
        raise ValueError("schema-v4 protein slide cohorts are required")
    slide_cohorts = {str(row["cohort"]) for row in slides}
    prediction_cohorts = payload.get("prediction_cohorts")
    if prediction_cohorts is None:
        expected_slide_cohorts = set(bindings)
    elif (
        not isinstance(prediction_cohorts, list)
        or not prediction_cohorts
        or any(not isinstance(value, str) or not value for value in prediction_cohorts)
        or len(set(prediction_cohorts)) != len(prediction_cohorts)
        or not set(prediction_cohorts).issubset(bindings)
    ):
        raise ValueError("schema-v4 protein prediction cohorts are invalid")
    else:
        expected_slide_cohorts = set(prediction_cohorts)
    if slide_cohorts != expected_slide_cohorts:
        raise ValueError("schema-v4 protein slide cohorts differ from bindings")


def _validate_registered_residual_refinement(payload: dict[str, object]) -> None:
    """Validate a frozen, evidence-bound serial-section residual transfer."""

    refinement = payload.get("registered_residual_refinement")
    if refinement is None:
        return
    expected_keys = {
        "schema_version",
        "method",
        "base_model_fingerprint",
        "base_result_fingerprint",
        "frozen_design_fingerprint",
        "development_cohorts",
        "confirmation_cohorts",
        "anchor_cohorts",
        "anchor_sections",
        "anchor_bank_artifact",
        "evidence_artifacts",
        "quantitative_gate",
        "visual_gate",
        "unsupported_behavior",
    }
    if (
        not isinstance(refinement, dict)
        or set(refinement) != expected_keys
        or refinement.get("schema_version") != 1
        or refinement.get("method") != _REGISTERED_RESIDUAL_METHOD
        or payload.get("architecture")
        not in {"dual_bank_attention", "graph_transformer", "multi_tower"}
        or payload.get("prediction_protocol") != "leave-one-mouse-out"
        or refinement.get("quantitative_gate") != "passed-once-after-design-freeze"
        or refinement.get("visual_gate") != "accepted-cell-resolved-native-audit"
        or refinement.get("unsupported_behavior")
        != "exact-frozen-inductive-baseline-fallback"
    ):
        raise ValueError("schema-v4 registered residual refinement is invalid")

    for key in (
        "base_model_fingerprint",
        "base_result_fingerprint",
        "frozen_design_fingerprint",
    ):
        value = refinement.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("schema-v4 registered residual refinement is invalid")

    cohort_sets: dict[str, set[str]] = {}
    for key in ("development_cohorts", "confirmation_cohorts", "anchor_cohorts"):
        values = refinement.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("schema-v4 registered residual refinement is invalid")
        cohort_sets[key] = set(values)
    if cohort_sets["development_cohorts"] & cohort_sets["confirmation_cohorts"]:
        raise ValueError("schema-v4 registered residual cohorts overlap")
    if not cohort_sets["anchor_cohorts"].issubset(cohort_sets["confirmation_cohorts"]):
        raise ValueError("schema-v4 registered residual anchors are not confirmed")

    anchor_sections = refinement.get("anchor_sections")
    if (
        not isinstance(anchor_sections, dict)
        or set(anchor_sections) != cohort_sets["anchor_cohorts"]
    ):
        raise ValueError("schema-v4 registered residual anchor sections are invalid")
    for sections in anchor_sections.values():
        if (
            not isinstance(sections, list)
            or not sections
            or any(not isinstance(section, str) or not section for section in sections)
            or len(set(sections)) != len(sections)
        ):
            raise ValueError(
                "schema-v4 registered residual anchor sections are invalid"
            )

    model_artifacts = payload.get("model_artifacts")
    if not isinstance(model_artifacts, list):
        raise ValueError("schema-v4 registered residual artifacts are invalid")
    declared = set(model_artifacts)
    anchor_artifact = refinement.get("anchor_bank_artifact")
    evidence = refinement.get("evidence_artifacts")
    if (
        not isinstance(anchor_artifact, str)
        or anchor_artifact not in declared
        or not isinstance(evidence, dict)
        or set(evidence)
        != {"frozen_design", "quantitative_confirmation", "visual_acceptance"}
        or any(
            not isinstance(value, str) or value not in declared
            for value in evidence.values()
        )
        or len({anchor_artifact, *evidence.values()}) != 4
    ):
        raise ValueError("schema-v4 registered residual artifacts are invalid")
    for relative in (anchor_artifact, *evidence.values()):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("schema-v4 registered residual artifacts are invalid")


def _current_approval(path: Path, fingerprint: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = None
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("fingerprint") == fingerprint
    ):
        return payload
    return {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "accepted": False,
        "reviewer": None,
        "reasons": ["explicit approval required"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
