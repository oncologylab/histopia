"""Integrity binding between a semantic atlas and its registration run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia.registration._approval import (
    RegistrationApproval,
    validate_registration_approval,
)
from histopia.semantic._result_validation import validate_semantic_result


@dataclass(frozen=True, slots=True)
class SemanticRegistrationBinding:
    """Validated identity shared by one semantic atlas and registration run."""

    preflight_schema_version: int
    preflight_fingerprint: str
    registration_result_sha256: str
    registration_approval: RegistrationApproval | None
    registration_approval_sha256: str | None

    @property
    def approval_bound(self) -> bool:
        """Return whether the preflight seals a final registration approval."""

        return self.preflight_schema_version == 3

    def to_json_dict(self) -> dict[str, object]:
        """Return portable provenance suitable for manifests and audit logs."""

        return {
            "preflight_schema_version": self.preflight_schema_version,
            "preflight_fingerprint": self.preflight_fingerprint,
            "registration_result_sha256": self.registration_result_sha256,
            "registration_approval_sha256": self.registration_approval_sha256,
            "approval_bound": self.approval_bound,
        }


def validate_semantic_registration_binding(
    registration_run: Path | str,
    semantic_run: Path | str,
    *,
    registration_payload: dict[str, object] | None = None,
    semantic_payload: dict[str, object] | None = None,
) -> SemanticRegistrationBinding:
    """Verify that an atlas belongs to the exact supplied registration run.

    Preflight schemas 1 and 2 bind the exact registration result and section
    order. Schema 3 additionally binds the final registration approval.
    Optional payloads are compared with the current files to detect stale
    caller reads; semantic artifacts are fully validated in either case.
    """

    registration_root = Path(registration_run).expanduser().resolve()
    semantic_root = Path(semantic_run).expanduser().resolve()
    registration_path = registration_root / "registration_result.json"
    semantic_result_path = semantic_root / "semantic_result.json"
    preflight_path = semantic_root / "preflight.json"

    registration_bytes = registration_path.read_bytes()
    registration_sha256 = hashlib.sha256(registration_bytes).hexdigest()
    loaded_registration = _json_object(registration_bytes, "registration result")
    if registration_payload is not None and registration_payload != loaded_registration:
        raise ValueError("registration result changed before semantic validation")
    semantic_result_bytes = semantic_result_path.read_bytes()
    semantic_result_sha256 = hashlib.sha256(semantic_result_bytes).hexdigest()
    current_semantic = _json_object(semantic_result_bytes, "semantic result")
    if semantic_payload is not None and semantic_payload != current_semantic:
        raise ValueError("semantic result changed before registration validation")
    loaded_semantic = validate_semantic_result(
        semantic_root,
        payload=current_semantic,
    )
    preflight_bytes = preflight_path.read_bytes()
    preflight_sha256 = hashlib.sha256(preflight_bytes).hexdigest()
    preflight = _json_object(preflight_bytes, "semantic preflight")

    schema = preflight.get("schema_version")
    if (
        not isinstance(schema, int)
        or isinstance(schema, bool)
        or schema not in {1, 2, 3}
    ):
        raise ValueError("semantic preflight schema is unsupported")
    fingerprint = preflight.get("fingerprint")
    provenance = loaded_semantic.get("feature_provenance")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(provenance, dict)
        or provenance.get("preflight_fingerprint") != fingerprint
    ):
        raise ValueError("semantic preflight fingerprint is stale")

    portable_slides = _portable_preflight_slides(preflight.get("slides"))
    core: dict[str, object] = {
        "schema_version": schema,
        "registration_result_sha256": preflight.get("registration_result_sha256"),
        "order_review_fingerprint": preflight.get("order_review_fingerprint"),
        "reference_slide": preflight.get("reference_slide"),
        "slides": portable_slides,
    }
    if schema == 3:
        core["registration_approval_sha256"] = preflight.get(
            "registration_approval_sha256"
        )
    if _json_sha256(core) != fingerprint:
        raise ValueError("semantic preflight record is stale")
    if core["registration_result_sha256"] != registration_sha256:
        raise ValueError("semantic atlas belongs to a different registration result")

    registration_ids, references = _registration_slide_identity(loaded_registration)
    preflight_ids = [str(row.get("slide_name", "")) for row in portable_slides]
    semantic_rows = loaded_semantic.get("slides")
    if not isinstance(semantic_rows, list):
        raise ValueError("semantic result contains no slides")
    semantic_ids = [
        str(row.get("id", "")) for row in semantic_rows if isinstance(row, dict)
    ]
    if (
        len(semantic_ids) != len(semantic_rows)
        or any(not value for value in semantic_ids)
        or len(set(semantic_ids)) != len(semantic_ids)
        or registration_ids != preflight_ids
        or registration_ids != semantic_ids
    ):
        raise ValueError("semantic and registration slide order differs")
    if references != [preflight.get("reference_slide")]:
        raise ValueError("semantic and registration references differ")

    approval: RegistrationApproval | None = None
    approval_sha256: str | None = None
    if schema == 3:
        expected_approval = core["registration_approval_sha256"]
        if not isinstance(expected_approval, str) or not expected_approval:
            raise ValueError("semantic preflight registration approval is stale")
        approval, approval_sha256 = _validated_registration_approval(registration_root)
        if expected_approval != approval_sha256:
            raise ValueError("semantic preflight registration approval is stale")
        if approval.registration_result_sha256 != registration_sha256:
            raise ValueError("registration approval differs from semantic preflight")

    if _file_sha256(registration_path) != registration_sha256:
        raise ValueError("registration result changed during semantic validation")
    if _file_sha256(preflight_path) != preflight_sha256:
        raise ValueError("semantic preflight changed during validation")
    if _file_sha256(semantic_result_path) != semantic_result_sha256:
        raise ValueError("semantic result changed during registration validation")
    if (
        approval_sha256 is not None
        and _file_sha256(registration_root / "registration_approval.json")
        != approval_sha256
    ):
        raise ValueError("registration approval changed during semantic validation")

    return SemanticRegistrationBinding(
        preflight_schema_version=schema,
        preflight_fingerprint=fingerprint,
        registration_result_sha256=registration_sha256,
        registration_approval=approval,
        registration_approval_sha256=approval_sha256,
    )


def _registration_slide_identity(
    registration: dict[str, object],
) -> tuple[list[str], list[str]]:
    slides = registration.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("registration result contains no slides")
    registration_ids = [
        Path(str(row.get("path", ""))).name for row in slides if isinstance(row, dict)
    ]
    if (
        len(registration_ids) != len(slides)
        or any(not value for value in registration_ids)
        or len(set(registration_ids)) != len(registration_ids)
    ):
        raise ValueError("registration slide names must be non-empty and unique")
    references = [
        slide_id
        for slide_id, row in zip(registration_ids, slides, strict=True)
        if row.get("is_reference")
    ]
    if len(references) != 1:
        raise ValueError("registration must contain exactly one reference slide")
    return registration_ids, references


def _validated_registration_approval(
    run_dir: Path,
) -> tuple[RegistrationApproval, str]:
    path = run_dir / "registration_approval.json"
    before = _file_sha256(path)
    approval = validate_registration_approval(run_dir)
    after = _file_sha256(path)
    if before != after:
        raise ValueError("registration approval changed during validation")
    return approval, after


def _portable_preflight_slides(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("semantic preflight contains no slides")
    portable: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("semantic preflight slides must be objects")
        portable.append(
            {key: item for key, item in row.items() if key != "source_path"}
        )
    return portable


def _json_object(payload: bytes, name: str) -> dict[str, Any]:
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} root must be an object")
    return loaded


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
