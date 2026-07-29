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
StainStatus = Literal[
    "approved",
    "review_required",
    "not_requested",
    "incomplete",
    "invalid",
]
TopologyStatus = Literal[
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
class StainWorkflowAudit:
    """Portable validation state for one quantitative stain result."""

    status: StainStatus
    slide_count: int | None = None
    fingerprint: str | None = None
    approved_families: tuple[str, ...] = ()
    pending_families: tuple[str, ...] = ()
    issue: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "slide_count": self.slide_count,
            "fingerprint": self.fingerprint,
            "approved_families": list(self.approved_families),
            "pending_families": list(self.pending_families),
            "issue": self.issue,
        }


@dataclass(frozen=True, slots=True)
class TopologyWorkflowAudit:
    """Portable validation state for one reconstructed semantic topology."""

    status: TopologyStatus
    fingerprint: str | None = None
    observed_section_count: int | None = None
    semantic_binding: Literal[
        "approval_bound",
        "legacy_unsealed",
        "not_requested",
        "unavailable",
    ] = "unavailable"
    issue: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fingerprint": self.fingerprint,
            "observed_section_count": self.observed_section_count,
            "semantic_binding": self.semantic_binding,
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
    stain: StainWorkflowAudit
    topology: TopologyWorkflowAudit
    viewer: ViewerWorkflowAudit

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.cohort_id,
            "status": self.status,
            "registration": self.registration.to_json_dict(),
            "semantic": self.semantic.to_json_dict(),
            "stain": self.stain.to_json_dict(),
            "topology": self.topology.to_json_dict(),
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
            "schema_version": 3,
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
    result_sha256: str
    slide_ids: tuple[str, ...]
    selected_k: int
    binding_payload: dict[str, object]
    review_approved: bool


@dataclass(frozen=True, slots=True)
class _StainIdentity:
    slide_count: int
    fingerprint: str
    approved_families: tuple[str, ...]
    pending_families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ViewerManifest:
    mice: dict[str, dict[str, object]]
    issue: str | None = None


def audit_workflows(
    registration_runs: dict[str, Path | str],
    *,
    semantic_runs: dict[str, Path | str] | None = None,
    stain_runs: dict[str, Path | str] | None = None,
    topology_runs: dict[str, Path | str] | None = None,
    viewer_manifest: Path | str | None = None,
) -> WorkflowAudit:
    """Validate exact workflow state without publishing local filesystem paths."""

    if not registration_runs:
        raise ValueError("at least one registration run is required")
    _validate_cohort_ids(registration_runs)
    semantic_runs = semantic_runs or {}
    stain_runs = stain_runs or {}
    topology_runs = topology_runs or {}
    _validate_cohort_ids(semantic_runs)
    _validate_cohort_ids(stain_runs)
    _validate_cohort_ids(topology_runs)
    unknown_semantics = sorted(set(semantic_runs) - set(registration_runs))
    if unknown_semantics:
        raise ValueError(
            "semantic runs have no matching registration: "
            + ", ".join(unknown_semantics)
        )
    unknown_stains = sorted(set(stain_runs) - set(registration_runs))
    if unknown_stains:
        raise ValueError(
            "stain runs have no matching registration: " + ", ".join(unknown_stains)
        )
    unknown_topologies = sorted(set(topology_runs) - set(registration_runs))
    if unknown_topologies:
        raise ValueError(
            "topology runs have no matching registration: "
            + ", ".join(unknown_topologies)
        )
    missing_semantics = sorted(set(topology_runs) - set(semantic_runs))
    if missing_semantics:
        raise ValueError(
            "topology runs have no matching semantic run: "
            + ", ".join(missing_semantics)
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
        stain_root = Path(stain_runs[cohort_id]) if cohort_id in stain_runs else None
        stain, stain_identity = _audit_stain(
            registration_identity,
            stain_root,
        )
        topology_root = (
            Path(topology_runs[cohort_id]) if cohort_id in topology_runs else None
        )
        topology = _audit_topology(
            registration_identity,
            semantic_identity,
            topology_root,
        )
        viewer_state = _audit_viewer(
            cohort_id,
            viewer,
            registration,
            registration_identity,
            semantic,
            semantic_identity,
            stain,
            stain_identity,
        )
        status = _overall_status(
            (
                registration.status,
                semantic.status,
                stain.status,
                topology.status,
                viewer_state.status,
            )
        )
        cohorts.append(
            CohortWorkflowAudit(
                cohort_id=cohort_id,
                status=status,
                registration=registration,
                semantic=semantic,
                stain=stain,
                topology=topology,
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

        result_bytes = result_path.read_bytes()
        result_sha256 = hashlib.sha256(result_bytes).hexdigest()
        payload = validate_semantic_result(semantic_root)
        binding = validate_semantic_registration_binding(
            registration_root,
            semantic_root,
            semantic_payload=payload,
        )
        fingerprint = _required_string(payload, "fingerprint")
        slide_count = _semantic_slide_count(payload)
        if hashlib.sha256(result_path.read_bytes()).hexdigest() != result_sha256:
            raise ValueError("semantic result changed during audit")
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
        result_sha256=result_sha256,
        slide_ids=_semantic_slide_ids(payload),
        selected_k=_semantic_selected_k(payload),
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


def _audit_stain(
    registration_identity: _RegistrationIdentity | None,
    stain_root: Path | None,
) -> tuple[StainWorkflowAudit, _StainIdentity | None]:
    if stain_root is None:
        return StainWorkflowAudit(status="not_requested"), None
    result_path = stain_root / "stain_result.json"
    if not result_path.is_file():
        return (
            StainWorkflowAudit(
                status="incomplete",
                issue="stain_result_missing",
            ),
            None,
        )
    try:
        from histopia.stain._approval import stain_review_status
        from histopia.stain._result_validation import validate_stain_result

        payload = validate_stain_result(stain_root)
        fingerprint = _required_string(payload, "fingerprint")
        slides = payload.get("slides")
        if not isinstance(slides, list) or not slides:
            raise ValueError("stain result contains no slides")
        if (
            registration_identity is None
            or payload.get("registration_result_sha256")
            != registration_identity.result_sha256
            or len(slides) != registration_identity.slide_count
        ):
            raise ValueError("stain registration binding is stale")
        review = stain_review_status(stain_root, payload)
        approved = _string_tuple(review.get("approved_families"))
        pending = _string_tuple(review.get("pending_families"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return (
            StainWorkflowAudit(
                status="invalid",
                issue="stain_result_or_review_invalid",
            ),
            None,
        )
    identity = _StainIdentity(
        slide_count=len(slides),
        fingerprint=fingerprint,
        approved_families=approved,
        pending_families=pending,
    )
    if pending:
        return (
            StainWorkflowAudit(
                status="review_required",
                slide_count=identity.slide_count,
                fingerprint=fingerprint,
                approved_families=approved,
                pending_families=pending,
                issue="stain_approval_required",
            ),
            identity,
        )
    return (
        StainWorkflowAudit(
            status="approved",
            slide_count=identity.slide_count,
            fingerprint=fingerprint,
            approved_families=approved,
        ),
        identity,
    )


def _audit_topology(
    registration_identity: _RegistrationIdentity | None,
    semantic_identity: _SemanticIdentity | None,
    topology_root: Path | None,
) -> TopologyWorkflowAudit:
    if topology_root is None:
        return TopologyWorkflowAudit(
            status="not_requested",
            semantic_binding="not_requested",
        )
    result_path = topology_root / "topology_result.json"
    if not result_path.is_file():
        return TopologyWorkflowAudit(
            status="incomplete",
            issue="topology_result_missing",
        )
    try:
        from histopia.topology._result import validate_topology_result

        result = validate_topology_result(topology_root)
        fingerprint = _required_string(result, "fingerprint")
        preflight_name = _required_string(result, "preflight")
        preflight_path = _safe_relative_artifact(topology_root, preflight_name)
        preflight = json.loads(preflight_path.read_text())
        if not isinstance(preflight, dict):
            raise ValueError("topology preflight must be an object")
        _validate_fingerprinted_payload(preflight, "topology preflight")
        observed_count = _positive_int(result.get("observed_section_count"))
        selected_k = _positive_int(result.get("selected_k"))
        slide_ids = _string_tuple(preflight.get("slide_ids"))
        if (
            registration_identity is None
            or semantic_identity is None
            or preflight.get("registration_result_sha256")
            != registration_identity.result_sha256
            or preflight.get("semantic_result_sha256")
            != semantic_identity.result_sha256
            or preflight.get("semantic_fingerprint") != semantic_identity.fingerprint
            or preflight.get("selected_k") != semantic_identity.selected_k
            or selected_k != semantic_identity.selected_k
            or slide_ids != semantic_identity.slide_ids
            or observed_count != semantic_identity.slide_count
            or result.get("preflight_fingerprint") != preflight.get("fingerprint")
            or result.get("registration_result_sha256")
            != registration_identity.result_sha256
            or result.get("semantic_fingerprint") != semantic_identity.fingerprint
        ):
            raise ValueError("topology source binding is stale")
        approval_bound = _validate_topology_approval_snapshot(preflight)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return TopologyWorkflowAudit(
            status="invalid",
            issue="topology_result_or_binding_invalid",
        )

    binding_name: Literal["approval_bound", "legacy_unsealed"] = (
        "approval_bound" if approval_bound else "legacy_unsealed"
    )
    if not approval_bound:
        return TopologyWorkflowAudit(
            status="review_required",
            fingerprint=fingerprint,
            observed_section_count=observed_count,
            semantic_binding=binding_name,
            issue="topology_approval_bound_rebuild_required",
        )
    try:
        from histopia.topology._approval import validate_topology_approval

        validate_topology_approval(topology_root)
    except FileNotFoundError:
        return TopologyWorkflowAudit(
            status="review_required",
            fingerprint=fingerprint,
            observed_section_count=observed_count,
            semantic_binding=binding_name,
            issue="topology_approval_required",
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        review_path = topology_root / "topology_review.json"
        try:
            review = json.loads(review_path.read_text())
        except (OSError, json.JSONDecodeError):
            review = None
        if (
            isinstance(review, dict)
            and review.get("schema_version") == 1
            and review.get("approved") is False
            and review.get("fingerprint") == fingerprint
        ):
            return TopologyWorkflowAudit(
                status="review_required",
                fingerprint=fingerprint,
                observed_section_count=observed_count,
                semantic_binding=binding_name,
                issue="topology_approval_required",
            )
        return TopologyWorkflowAudit(
            status="invalid",
            fingerprint=fingerprint,
            observed_section_count=observed_count,
            semantic_binding=binding_name,
            issue="topology_approval_invalid",
        )
    return TopologyWorkflowAudit(
        status="approved",
        fingerprint=fingerprint,
        observed_section_count=observed_count,
        semantic_binding=binding_name,
    )


def _audit_viewer(
    cohort_id: str,
    viewer: _ViewerManifest | None,
    registration: RegistrationWorkflowAudit,
    registration_identity: _RegistrationIdentity | None,
    semantic: SemanticWorkflowAudit,
    semantic_identity: _SemanticIdentity | None,
    stain: StainWorkflowAudit,
    stain_identity: _StainIdentity | None,
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
    viewer_stain = mouse.get("stain")
    if stain.status == "not_requested":
        pass
    elif stain_identity is None:
        issues.append("viewer_stain_not_verifiable")
    elif not stain_identity.approved_families:
        if viewer_stain is not None:
            issues.append("viewer_unapproved_stain_published")
    elif not isinstance(viewer_stain, dict):
        issues.append("viewer_stain_missing")
    else:
        if viewer_stain.get("fingerprint") != stain_identity.fingerprint:
            issues.append("viewer_stain_fingerprint_mismatch")
        review = viewer_stain.get("review")
        if (
            not isinstance(review, dict)
            or _string_tuple(review.get("approved_families"))
            != stain_identity.approved_families
            or _string_tuple(review.get("pending_families"))
            != stain_identity.pending_families
        ):
            issues.append("viewer_stain_review_mismatch")
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


def _semantic_slide_ids(payload: dict[str, object]) -> tuple[str, ...]:
    slides = payload.get("slides")
    if not isinstance(slides, list):
        raise ValueError("semantic result contains no slides")
    values: list[str] = []
    for row in slides:
        value = row.get("id") if isinstance(row, dict) else None
        if not isinstance(value, str) or not value:
            raise ValueError("semantic slide IDs are invalid")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("semantic slide IDs must be unique")
    return tuple(values)


def _semantic_selected_k(payload: dict[str, object]) -> int:
    value = payload.get("selected_k", payload.get("primary_clusters"))
    return _positive_int(value)


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("value must be a positive integer")
    return value


def _safe_relative_artifact(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must stay inside the run")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("artifact path must stay inside the run")
    return resolved


def _validate_fingerprinted_payload(payload: dict[str, object], name: str) -> None:
    fingerprint = payload.get("fingerprint")
    core = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != expected:
        raise ValueError(f"{name} fingerprint is stale")


def _validate_topology_approval_snapshot(preflight: dict[str, object]) -> bool:
    approval = preflight.get("semantic_approval")
    if approval is None:
        return False
    if not isinstance(approval, dict):
        raise ValueError("topology semantic approval snapshot is invalid")
    reviewer = approval.get("semantic_reviewer")
    if (
        approval.get("semantic_fingerprint") != preflight.get("semantic_fingerprint")
        or approval.get("registration_result_sha256")
        != preflight.get("registration_result_sha256")
        or not isinstance(reviewer, str)
        or not reviewer.strip()
    ):
        raise ValueError("topology semantic approval snapshot is stale")
    return True


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("expected a list of non-empty strings")
    return tuple(value)


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
