"""Portable integrity audit for registration, semantic, and viewer workflows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from histopia._atomic import write_json_atomic

WorkflowStatus = Literal["approved", "review_required", "incomplete", "invalid"]
RegistrationStatus = WorkflowStatus
SemanticStatus = Literal[
    "approved",
    "review_required",
    "not_requested",
    "incomplete",
    "invalid",
]
ViewerStatus = Literal["current", "not_requested", "incomplete", "invalid"]


@dataclass(frozen=True, slots=True)
class RegistrationWorkflowAudit:
    """Portable validation state for one registration result."""

    status: RegistrationStatus
    slide_count: int | None = None
    result_sha256: str | None = None
    order_fingerprint: str | None = None
    issue: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "slide_count": self.slide_count,
            "result_sha256": self.result_sha256,
            "order_fingerprint": self.order_fingerprint,
            "issue": self.issue,
        }


@dataclass(frozen=True, slots=True)
class SemanticWorkflowAudit:
    """Portable validation state for one semantic result."""

    status: SemanticStatus
    slide_count: int | None = None
    fingerprint: str | None = None
    registration_binding: Literal[
        "approval_bound",
        "legacy_unsealed",
        "not_requested",
        "unavailable",
    ] = "unavailable"
    issue: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "slide_count": self.slide_count,
            "fingerprint": self.fingerprint,
            "registration_binding": self.registration_binding,
            "issue": self.issue,
        }


@dataclass(frozen=True, slots=True)
class ViewerWorkflowAudit:
    """Portable validation state for one cohort in a generated viewer."""

    status: ViewerStatus
    issue: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {"status": self.status, "issue": self.issue}


@dataclass(frozen=True, slots=True)
class CohortWorkflowAudit:
    """Exact workflow state for one named cohort."""

    cohort_id: str
    status: WorkflowStatus
    registration: RegistrationWorkflowAudit
    semantic: SemanticWorkflowAudit
    viewer: ViewerWorkflowAudit

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.cohort_id,
            "status": self.status,
            "registration": self.registration.to_json_dict(),
            "semantic": self.semantic.to_json_dict(),
            "viewer": self.viewer.to_json_dict(),
        }


@dataclass(frozen=True, slots=True)
class WorkflowAudit:
    """Path-free batch workflow audit suitable for CI and review logs."""

    cohorts: tuple[CohortWorkflowAudit, ...]
    viewer_unmapped_ids: tuple[str, ...] = ()

    @property
    def status(self) -> WorkflowStatus:
        return _overall_status(tuple(cohort.status for cohort in self.cohorts))

    @property
    def exit_code(self) -> int:
        """Return 0 for approved, 2 for review gates, and 1 for defects."""

        if self.status == "approved":
            return 0
        if self.status == "review_required":
            return 2
        return 1

    def to_json_dict(self) -> dict[str, object]:
        counts = {
            status: sum(cohort.status == status for cohort in self.cohorts)
            for status in ("approved", "review_required", "incomplete", "invalid")
        }
        return {
            "schema_version": 1,
            "status": self.status,
            "summary": {
                "cohort_count": len(self.cohorts),
                **counts,
                "viewer_unmapped_count": len(self.viewer_unmapped_ids),
            },
            "viewer_unmapped_ids": list(self.viewer_unmapped_ids),
            "cohorts": [cohort.to_json_dict() for cohort in self.cohorts],
        }


@dataclass(frozen=True, slots=True)
class _RegistrationIdentity:
    slide_count: int
    result_sha256: str


@dataclass(frozen=True, slots=True)
class _SemanticIdentity:
    slide_count: int
    fingerprint: str
    binding_payload: dict[str, object]
    review_approved: bool


@dataclass(frozen=True, slots=True)
class _ViewerManifest:
    mice: dict[str, dict[str, object]]
    issue: str | None = None


def audit_workflows(
    registration_runs: dict[str, Path | str],
    *,
    semantic_runs: dict[str, Path | str] | None = None,
    viewer_manifest: Path | str | None = None,
) -> WorkflowAudit:
    """Validate exact workflow state without publishing local filesystem paths."""

    if not registration_runs:
        raise ValueError("at least one registration run is required")
    _validate_cohort_ids(registration_runs)
    semantic_runs = semantic_runs or {}
    _validate_cohort_ids(semantic_runs)
    unknown_semantics = sorted(set(semantic_runs) - set(registration_runs))
    if unknown_semantics:
        raise ValueError(
            "semantic runs have no matching registration: "
            + ", ".join(unknown_semantics)
        )

    viewer = (
        _load_viewer_manifest(Path(viewer_manifest))
        if viewer_manifest is not None
        else None
    )
    cohorts: list[CohortWorkflowAudit] = []
    for cohort_id in sorted(registration_runs):
        registration_root = Path(registration_runs[cohort_id])
        registration, registration_identity = _audit_registration(registration_root)
        semantic_root = (
            Path(semantic_runs[cohort_id]) if cohort_id in semantic_runs else None
        )
        semantic, semantic_identity = _audit_semantic(
            registration_root,
            semantic_root,
        )
        viewer_state = _audit_viewer(
            cohort_id,
            viewer,
            registration,
            registration_identity,
            semantic,
            semantic_identity,
        )
        status = _overall_status(
            (
                registration.status,
                semantic.status,
                viewer_state.status,
            )
        )
        cohorts.append(
            CohortWorkflowAudit(
                cohort_id=cohort_id,
                status=status,
                registration=registration,
                semantic=semantic,
                viewer=viewer_state,
            )
        )

    unmapped = (
        tuple(sorted(set(viewer.mice) - set(registration_runs)))
        if viewer is not None and viewer.issue is None
        else ()
    )
    return WorkflowAudit(tuple(cohorts), unmapped)


def write_workflow_audit(
    report: WorkflowAudit,
    output: Path | str,
) -> Path:
    """Atomically write one portable workflow audit."""

    return write_json_atomic(
        output,
        report.to_json_dict(),
        sort_keys=True,
    )


def _audit_registration(
    root: Path,
) -> tuple[RegistrationWorkflowAudit, _RegistrationIdentity | None]:
    result_path = root / "registration_result.json"
    try:
        result_bytes = result_path.read_bytes()
    except FileNotFoundError:
        return (
            RegistrationWorkflowAudit(
                status="incomplete",
                issue="registration_result_missing",
            ),
            None,
        )
    except OSError:
        return (
            RegistrationWorkflowAudit(
                status="invalid",
                issue="registration_result_unreadable",
            ),
            None,
        )
    try:
        payload = json.loads(result_bytes)
        slide_count = _validate_registration_result(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return (
            RegistrationWorkflowAudit(
                status="invalid",
                issue="registration_result_invalid",
            ),
            None,
        )
    identity = _RegistrationIdentity(
        slide_count=slide_count,
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
    )
    approval_path = root / "registration_approval.json"
    if not approval_path.is_file():
        return (
            RegistrationWorkflowAudit(
                status="review_required",
                slide_count=slide_count,
                result_sha256=identity.result_sha256,
                issue="registration_approval_required",
            ),
            identity,
        )
    try:
        from histopia.registration._approval import validate_registration_approval

        approval = validate_registration_approval(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return (
            RegistrationWorkflowAudit(
                status="invalid",
                slide_count=slide_count,
                result_sha256=identity.result_sha256,
                issue="registration_approval_invalid",
            ),
            identity,
        )
    return (
        RegistrationWorkflowAudit(
            status="approved",
            slide_count=slide_count,
            result_sha256=identity.result_sha256,
            order_fingerprint=approval.order_fingerprint,
        ),
        identity,
    )


def _audit_semantic(
    registration_root: Path,
    semantic_root: Path | None,
) -> tuple[SemanticWorkflowAudit, _SemanticIdentity | None]:
    if semantic_root is None:
        return (
            SemanticWorkflowAudit(
                status="not_requested",
                registration_binding="not_requested",
            ),
            None,
        )
    result_path = semantic_root / "semantic_result.json"
    if not result_path.is_file():
        return (
            SemanticWorkflowAudit(
                status="incomplete",
                issue="semantic_result_missing",
            ),
            None,
        )
    try:
        from histopia.semantic._registration_binding import (
            validate_semantic_registration_binding,
        )
        from histopia.semantic._result_validation import validate_semantic_result

        payload = validate_semantic_result(semantic_root)
        binding = validate_semantic_registration_binding(
            registration_root,
            semantic_root,
            semantic_payload=payload,
        )
        fingerprint = _required_string(payload, "fingerprint")
        slide_count = _semantic_slide_count(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return (
            SemanticWorkflowAudit(
                status="invalid",
                issue="semantic_result_or_binding_invalid",
            ),
            None,
        )
    binding_name: Literal["approval_bound", "legacy_unsealed"] = (
        "approval_bound" if binding.approval_bound else "legacy_unsealed"
    )
    review_path = semantic_root / "semantic_review.json"
    try:
        review = json.loads(review_path.read_text())
    except FileNotFoundError:
        return (
            SemanticWorkflowAudit(
                status="incomplete",
                slide_count=slide_count,
                fingerprint=fingerprint,
                registration_binding=binding_name,
                issue="semantic_review_missing",
            ),
            None,
        )
    except (OSError, json.JSONDecodeError):
        return (
            SemanticWorkflowAudit(
                status="invalid",
                slide_count=slide_count,
                fingerprint=fingerprint,
                registration_binding=binding_name,
                issue="semantic_review_invalid",
            ),
            None,
        )
    if (
        not isinstance(review, dict)
        or review.get("schema_version") != 3
        or review.get("fingerprint") != fingerprint
        or not isinstance(review.get("approved"), bool)
    ):
        return (
            SemanticWorkflowAudit(
                status="invalid",
                slide_count=slide_count,
                fingerprint=fingerprint,
                registration_binding=binding_name,
                issue="semantic_review_invalid",
            ),
            None,
        )
    approved = bool(review["approved"])
    if approved:
        try:
            from histopia.semantic._approval import validate_semantic_approval

            validate_semantic_approval(semantic_root)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return (
                SemanticWorkflowAudit(
                    status="invalid",
                    slide_count=slide_count,
                    fingerprint=fingerprint,
                    registration_binding=binding_name,
                    issue="semantic_approval_invalid",
                ),
                None,
            )
    identity = _SemanticIdentity(
        slide_count=slide_count,
        fingerprint=fingerprint,
        binding_payload=binding.to_json_dict(),
        review_approved=approved,
    )
    if not binding.approval_bound:
        return (
            SemanticWorkflowAudit(
                status="review_required",
                slide_count=slide_count,
                fingerprint=fingerprint,
                registration_binding=binding_name,
                issue="semantic_registration_approval_binding_required",
            ),
            identity,
        )
    if not approved:
        return (
            SemanticWorkflowAudit(
                status="review_required",
                slide_count=slide_count,
                fingerprint=fingerprint,
                registration_binding=binding_name,
                issue="semantic_approval_required",
            ),
            identity,
        )
    return (
        SemanticWorkflowAudit(
            status="approved",
            slide_count=slide_count,
            fingerprint=fingerprint,
            registration_binding=binding_name,
        ),
        identity,
    )


def _audit_viewer(
    cohort_id: str,
    viewer: _ViewerManifest | None,
    registration: RegistrationWorkflowAudit,
    registration_identity: _RegistrationIdentity | None,
    semantic: SemanticWorkflowAudit,
    semantic_identity: _SemanticIdentity | None,
) -> ViewerWorkflowAudit:
    if viewer is None:
        return ViewerWorkflowAudit(status="not_requested")
    if viewer.issue is not None:
        return ViewerWorkflowAudit(status="invalid", issue=viewer.issue)
    mouse = viewer.mice.get(cohort_id)
    if mouse is None:
        return ViewerWorkflowAudit(status="incomplete", issue="viewer_cohort_missing")
    issues: list[str] = []
    slides = mouse.get("slides")
    if (
        not isinstance(slides, list)
        or registration_identity is None
        or len(slides) != registration_identity.slide_count
    ):
        issues.append("viewer_registration_slide_count_mismatch")
    viewer_approval = mouse.get("registration_approval")
    if registration.status == "approved":
        if (
            not isinstance(viewer_approval, dict)
            or viewer_approval.get("approved") is not True
            or viewer_approval.get("registration_result_sha256")
            != registration.result_sha256
            or viewer_approval.get("order_fingerprint")
            != registration.order_fingerprint
        ):
            issues.append("viewer_registration_approval_mismatch")
    elif viewer_approval is not None:
        issues.append("viewer_registration_approval_mismatch")

    viewer_semantic = mouse.get("semantic")
    if semantic.status == "not_requested":
        pass
    elif semantic_identity is None:
        issues.append("viewer_semantic_not_verifiable")
    elif not isinstance(viewer_semantic, dict):
        issues.append("viewer_semantic_missing")
    else:
        if viewer_semantic.get("fingerprint") != semantic_identity.fingerprint:
            issues.append("viewer_semantic_fingerprint_mismatch")
        if (
            viewer_semantic.get("registration_binding")
            != semantic_identity.binding_payload
        ):
            issues.append("viewer_semantic_binding_mismatch")
        review = viewer_semantic.get("review")
        if (
            not isinstance(review, dict)
            or review.get("approved") is not semantic_identity.review_approved
            or review.get("fingerprint_matches") is not True
        ):
            issues.append("viewer_semantic_review_mismatch")
    if issues:
        return ViewerWorkflowAudit(status="invalid", issue="+".join(sorted(issues)))
    return ViewerWorkflowAudit(status="current")


def _load_viewer_manifest(path: Path) -> _ViewerManifest:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return _ViewerManifest({}, "viewer_manifest_missing")
    except (OSError, json.JSONDecodeError):
        return _ViewerManifest({}, "viewer_manifest_invalid")
    if not isinstance(payload, dict):
        return _ViewerManifest({}, "viewer_manifest_invalid")
    rows = payload.get("mice")
    if payload.get("schema_version") != 1 or not isinstance(rows, list):
        return _ViewerManifest({}, "viewer_manifest_invalid")
    mice: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            return _ViewerManifest({}, "viewer_manifest_invalid")
        cohort_id = row.get("id")
        if not isinstance(cohort_id, str) or not cohort_id or cohort_id in mice:
            return _ViewerManifest({}, "viewer_manifest_invalid")
        mice[cohort_id] = row
    return _ViewerManifest(mice)


def _validate_registration_result(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("registration result root must be an object")
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("registration result contains no slides")
    names: list[str] = []
    references: list[str] = []
    for row in slides:
        if not isinstance(row, dict):
            raise ValueError("registration slides must be objects")
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("registration slide path is invalid")
        name = Path(path).name
        if not name or name in names:
            raise ValueError("registration slide names must be unique")
        names.append(name)
        is_reference = row.get("is_reference")
        if not isinstance(is_reference, bool):
            raise ValueError("registration reference state is invalid")
        if is_reference:
            references.append(name)
        _validate_transform(row.get("transform"))
    reference = payload.get("reference_slide")
    if (
        len(references) != 1
        or not isinstance(reference, str)
        or Path(reference).name != references[0]
    ):
        raise ValueError("registration reference is invalid")
    return len(slides)


def _validate_transform(value: object) -> None:
    matrix = value.get("matrix") if isinstance(value, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("registration transform is invalid")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("registration transform is invalid")
        for item in row:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise ValueError("registration transform is invalid")


def _semantic_slide_count(payload: dict[str, object]) -> int:
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("semantic result contains no slides")
    return len(slides)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _validate_cohort_ids(runs: dict[str, Path | str]) -> None:
    invalid = sorted(
        str(cohort_id)
        for cohort_id in runs
        if not isinstance(cohort_id, str)
        or not cohort_id
        or cohort_id in {".", ".."}
        or "/" in cohort_id
        or "\\" in cohort_id
    )
    if invalid:
        raise ValueError("invalid cohort IDs: " + ", ".join(invalid))


def _overall_status(statuses: tuple[str, ...]) -> WorkflowStatus:
    relevant = set(statuses) - {"not_requested", "current"}
    if "invalid" in relevant:
        return "invalid"
    if "incomplete" in relevant:
        return "incomplete"
    if "review_required" in relevant:
        return "review_required"
    return "approved"
