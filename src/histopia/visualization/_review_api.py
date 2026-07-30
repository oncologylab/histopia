"""Web decisions for prepared Histopia review artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from histopia.topology._feedback import TopologyFeedbackStore
from histopia.visualization._feedback import RegistrationFeedbackStore

_COHORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_STAGES = ("mask", "order", "registration", "semantic", "topology", "stain")


@dataclass(frozen=True, slots=True)
class ReviewRuns:
    """Filesystem locations for one cohort's reviewable workflows."""

    registration: Path
    semantic: Path | None = None
    topology: Path | None = None
    stain: Path | None = None
    registered_wsi: Path | None = None


class ReviewDecisionService:
    """Apply fingerprint-bound approvals to an explicit local run registry."""

    def __init__(
        self,
        cohorts: dict[str, ReviewRuns],
        *,
        feedback_store: RegistrationFeedbackStore | None = None,
        topology_feedback_store: TopologyFeedbackStore | None = None,
    ) -> None:
        if not cohorts:
            raise ValueError("review registry must contain at least one cohort")
        self._cohorts = dict(sorted(cohorts.items()))
        self._feedback_store = feedback_store
        self._topology_feedback_store = topology_feedback_store

    @classmethod
    def from_file(cls, path: Path | str) -> ReviewDecisionService:
        """Load a local registry; paths are never returned by the web API."""

        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text())
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("review registry must use schema version 1")
        raw_cohorts = payload.get("cohorts")
        if not isinstance(raw_cohorts, dict) or not raw_cohorts:
            raise ValueError("review registry cohorts must be a non-empty object")
        cohorts: dict[str, ReviewRuns] = {}
        for cohort, raw in raw_cohorts.items():
            if not isinstance(cohort, str) or not _COHORT_RE.fullmatch(cohort):
                raise ValueError(f"invalid review cohort name: {cohort!r}")
            if not isinstance(raw, dict):
                raise ValueError(f"review cohort {cohort} must be an object")
            registration = _configured_path(
                raw,
                "registration",
                config_path.parent,
                required=True,
            )
            assert registration is not None
            cohorts[cohort] = ReviewRuns(
                registration=registration,
                semantic=_configured_path(
                    raw,
                    "semantic",
                    config_path.parent,
                    required=False,
                ),
                topology=_configured_path(
                    raw,
                    "topology",
                    config_path.parent,
                    required=False,
                ),
                stain=_configured_path(
                    raw,
                    "stain",
                    config_path.parent,
                    required=False,
                ),
                registered_wsi=_configured_path(
                    raw,
                    "registered_wsi",
                    config_path.parent,
                    required=False,
                ),
            )
        feedback_dir = payload.get("feedback_dir")
        if feedback_dir is not None and (
            not isinstance(feedback_dir, str) or not feedback_dir.strip()
        ):
            raise ValueError("review registry feedback_dir must be a path")
        feedback_path = None
        if isinstance(feedback_dir, str):
            raw_feedback = Path(feedback_dir).expanduser()
            feedback_path = (
                (config_path.parent / raw_feedback).resolve()
                if not raw_feedback.is_absolute()
                else raw_feedback.resolve()
            )
        return cls(
            cohorts,
            feedback_store=(
                RegistrationFeedbackStore(feedback_path)
                if feedback_path is not None
                else None
            ),
            topology_feedback_store=(
                TopologyFeedbackStore(feedback_path)
                if feedback_path is not None
                else None
            ),
        )

    def status(self) -> dict[str, object]:
        """Return path-free, live approval state for every configured cohort."""

        return {
            "schema_version": 1,
            "stages": list(_STAGES),
            "feedback_configured": self._feedback_store is not None,
            "cohorts": [
                self._cohort_status(name, runs) for name, runs in self._cohorts.items()
            ],
        }

    def wsi_runs(self) -> dict[str, tuple[Path, Path]]:
        """Return private WSI bindings for server construction only."""

        return {
            cohort: (runs.registration, runs.registered_wsi)
            for cohort, runs in self._cohorts.items()
            if runs.registered_wsi is not None
        }

    def approve(self, request: dict[str, object]) -> dict[str, object]:
        """Validate and apply one exact scientific approval."""

        cohort = _required_text(request, "cohort")
        stage = _required_text(request, "stage")
        reviewer = _required_text(request, "reviewer")
        notes = _required_text(request, "notes")
        try:
            runs = self._cohorts[cohort]
        except KeyError as error:
            raise ValueError(f"unknown review cohort: {cohort}") from error
        if stage not in _STAGES:
            raise ValueError(f"unknown review stage: {stage}")

        if stage == "mask":
            self._require_registration_feedback(cohort, "mask", runs)
            from histopia.registration import approve_mask_review

            approve_mask_review(runs.registration, reviewer=reviewer, notes=notes)
        elif stage == "order":
            self._require_registration_feedback(cohort, "order", runs)
            from histopia.registration import approve_section_order

            approve_section_order(runs.registration, reviewer=reviewer, notes=notes)
        elif stage == "registration":
            self._require_registration_feedback(cohort, "alignment", runs)
            from histopia.registration import approve_registration_run

            approve_registration_run(
                runs.registration,
                reviewer=reviewer,
                notes=notes,
            )
        elif stage == "semantic":
            if runs.semantic is None:
                raise ValueError(f"cohort {cohort} has no semantic review")
            from histopia.semantic import approve_semantic_result

            approve_semantic_result(
                runs.semantic,
                registration_run=runs.registration,
                reviewer=reviewer,
                notes=notes,
            )
        elif stage == "topology":
            if runs.topology is None:
                raise ValueError(f"cohort {cohort} has no topology review")
            if runs.semantic is None:
                raise ValueError(f"cohort {cohort} has no semantic review")
            if self._topology_feedback_store is not None:
                self._topology_feedback_store.require_accepted(
                    cohort=cohort,
                    topology_run=runs.topology,
                )
            _validate_topology_for_approval(
                runs.registration,
                runs.semantic,
                runs.topology,
            )
            from histopia.topology import approve_topology_result

            approve_topology_result(
                runs.topology,
                reviewer=reviewer,
                notes=notes,
            )
        else:
            if runs.stain is None:
                raise ValueError(f"cohort {cohort} has no stain review")
            stain_status = _stain_status(runs.registration, runs.stain)
            if (
                stain_status.get("available") is not True
                or stain_status.get("invalid") is True
            ):
                raise ValueError(
                    "stain approval requires a valid result bound to the current "
                    "registration"
                )
            _validate_stain_for_approval(runs.registration, runs.stain)
            families = request.get("families")
            if (
                not isinstance(families, list)
                or not families
                or any(not isinstance(item, str) or not item for item in families)
            ):
                raise ValueError("stain approval requires at least one family")
            from histopia.stain import approve_stain_result

            approve_stain_result(
                runs.stain,
                reviewer=reviewer,
                notes=notes,
                families=families,
            )
        return self._cohort_status(cohort, runs)

    def feedback(self, cohort: str, stage: str) -> dict[str, object]:
        """Return current per-slide registration feedback."""

        runs = self._required_cohort(cohort)
        if stage == "topology":
            if runs.topology is None:
                raise ValueError(f"cohort {cohort} has no topology review")
            return self._required_topology_feedback_store().review(
                cohort=cohort,
                topology_run=runs.topology,
            )
        store = self._required_feedback_store()
        return store.review(
            cohort=cohort,
            stage=stage,
            registration_run=runs.registration,
        )

    def save_feedback(self, request: dict[str, object]) -> dict[str, object]:
        """Persist one fingerprint-bound per-slide review record."""

        cohort = _required_text(request, "cohort")
        runs = self._required_cohort(cohort)
        if request.get("stage") == "topology":
            if runs.topology is None:
                raise ValueError(f"cohort {cohort} has no topology review")
            return self._required_topology_feedback_store().save(
                request,
                topology_run=runs.topology,
            )
        store = self._required_feedback_store()
        return store.save(request, registration_run=runs.registration)

    def feedback_summary(self) -> dict[str, object]:
        """Return aggregate issue frequencies for method improvement."""

        return {
            "schema_version": 1,
            "registration": (
                self._feedback_store.summary()
                if self._feedback_store is not None
                else None
            ),
            "topology": (
                self._topology_feedback_store.summary()
                if self._topology_feedback_store is not None
                else None
            ),
        }

    def _required_cohort(self, cohort: str) -> ReviewRuns:
        try:
            return self._cohorts[cohort]
        except KeyError as error:
            raise ValueError(f"unknown review cohort: {cohort}") from error

    def _required_feedback_store(self) -> RegistrationFeedbackStore:
        if self._feedback_store is None:
            raise ValueError("registration feedback storage is not configured")
        return self._feedback_store

    def _required_topology_feedback_store(self) -> TopologyFeedbackStore:
        if self._topology_feedback_store is None:
            raise ValueError("topology feedback storage is not configured")
        return self._topology_feedback_store

    def _require_registration_feedback(
        self,
        cohort: str,
        stage: str,
        runs: ReviewRuns,
    ) -> None:
        if self._feedback_store is None:
            return
        self._feedback_store.require_accepted(
            cohort=cohort,
            stage=stage,
            registration_run=runs.registration,
        )

    def _cohort_status(self, cohort: str, runs: ReviewRuns) -> dict[str, object]:
        return {
            "id": cohort,
            "stages": {
                "mask": _mask_status(runs.registration),
                "order": _order_status(runs.registration),
                "registration": _registration_status(runs.registration),
                "semantic": _semantic_status(runs.registration, runs.semantic),
                "topology": _topology_status(
                    runs.registration,
                    runs.semantic,
                    runs.topology,
                ),
                "stain": _stain_status(runs.registration, runs.stain),
            },
        }


def _configured_path(
    row: dict[str, object],
    key: str,
    base: Path,
    *,
    required: bool,
) -> Path | None:
    raw = row.get(key)
    if raw is None and not required:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"review registry {key} path is missing")
    path = Path(raw).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must not be blank")
    return value.strip()


def _mask_status(run: Path) -> dict[str, object]:
    path = run / "mask_review.json"
    if not path.is_file():
        return {"available": False, "approved": False}
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("slides")
        approved = (
            isinstance(rows, list)
            and bool(rows)
            and all(
                isinstance(row, dict)
                and row.get("status") in {"auto_pass", "override_pass"}
                for row in rows
            )
        )
        return {"available": True, "approved": approved}
    except (OSError, json.JSONDecodeError):
        return {"available": True, "approved": False, "invalid": True}


def _order_status(run: Path) -> dict[str, object]:
    path = run / "section_order_review.json"
    if not path.is_file():
        return {"available": False, "approved": False}
    try:
        payload = json.loads(path.read_text())
        return {
            "available": True,
            "approved": payload.get("approved") is True,
            "pending_update": (run / "section_order_review.pending.json").is_file(),
        }
    except (OSError, json.JSONDecodeError):
        return {"available": True, "approved": False, "invalid": True}


def _registration_status(run: Path) -> dict[str, object]:
    if not (run / "registration_result.json").is_file():
        return {"available": False, "approved": False}
    try:
        from histopia.registration import validate_registration_approval

        validate_registration_approval(run)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {"available": True, "approved": False}
    return {"available": True, "approved": True}


def _semantic_status(
    registration_run: Path,
    run: Path | None,
) -> dict[str, object]:
    if run is None:
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
        }
    result_path = run / "semantic_result.json"
    if not result_path.is_file():
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
        }
    try:
        result = _json_object(result_path)
        _require_current_json_fingerprint(result, "semantic result")
        registration_bytes = (
            registration_run / "registration_result.json"
        ).read_bytes()
        preflight = _json_object(run / "preflight.json")
        fingerprint = preflight.get("fingerprint")
        provenance = result.get("feature_provenance")
        if (
            not isinstance(fingerprint, str)
            or not fingerprint
            or not isinstance(provenance, dict)
            or provenance.get("preflight_fingerprint") != fingerprint
            or preflight.get("registration_result_sha256")
            != hashlib.sha256(registration_bytes).hexdigest()
        ):
            raise ValueError("semantic registration binding is stale")
        if preflight.get("schema_version") != 3:
            return {
                "available": True,
                "approved": False,
                "approval_ready": False,
                "issue": "semantic_registration_approval_binding_required",
            }
        approval_path = registration_run / "registration_approval.json"
        if (
            preflight.get("registration_approval_sha256") != _sha256_file(approval_path)
            or not _registration_status(registration_run)["approved"]
        ):
            raise ValueError("semantic registration approval binding is stale")
        review = _json_object(run / "semantic_review.json")
        semantic_fingerprint = result.get("fingerprint")
        if (
            review.get("schema_version") != 3
            or review.get("fingerprint") != semantic_fingerprint
            or not isinstance(review.get("approved"), bool)
        ):
            raise ValueError("semantic review is stale")
        approved = review["approved"] is True
        if approved and (
            not isinstance(review.get("reviewer"), str)
            or not str(review["reviewer"]).strip()
            or not isinstance(review.get("notes"), str)
            or not str(review["notes"]).strip()
        ):
            raise ValueError("semantic approval metadata is invalid")
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {
            "available": True,
            "approved": False,
            "approval_ready": False,
            "invalid": True,
            "issue": "semantic_result_binding_or_approval_invalid",
        }
    return {
        "available": True,
        "approved": approved,
        "approval_ready": not approved,
        "issue": None if approved else "semantic_approval_required",
    }


def _stain_status(
    registration_run: Path,
    run: Path | None,
) -> dict[str, object]:
    if run is None:
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
            "families": [],
        }
    result_path = run / "stain_result.json"
    if not result_path.is_file():
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
            "families": [],
        }
    try:
        result = _json_object(result_path)
        _require_current_json_fingerprint(result, "stain result")
        registration_bytes = (
            registration_run / "registration_result.json"
        ).read_bytes()
        registration = json.loads(registration_bytes)
        registration_slides = (
            registration.get("slides") if isinstance(registration, dict) else None
        )
        stain_slides = result.get("slides")
        if (
            not isinstance(registration_slides, list)
            or not registration_slides
            or not isinstance(stain_slides, list)
            or len(stain_slides) != len(registration_slides)
            or result.get("registration_result_sha256")
            != hashlib.sha256(registration_bytes).hexdigest()
        ):
            raise ValueError("stain registration binding is stale")
        raw_families = result.get("families")
        if not isinstance(raw_families, dict) or not raw_families:
            raise ValueError("stain result has no quantified families")
        family_names = tuple(sorted(str(family) for family in raw_families))
        approved = _lightweight_stain_approvals(run, result, family_names)
        pending = tuple(
            family for family in family_names if family not in set(approved)
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {
            "available": True,
            "approved": False,
            "approval_ready": False,
            "families": [],
            "invalid": True,
            "issue": "stain_result_binding_or_approval_invalid",
        }
    return {
        "available": True,
        "approved": not pending,
        "approval_ready": bool(pending),
        "issue": None if not pending else "stain_approval_required",
        "families": [
            {"id": family, "approved": family in approved} for family in family_names
        ],
    }


def _topology_status(
    registration_run: Path,
    semantic_run: Path | None,
    run: Path | None,
) -> dict[str, object]:
    if run is None:
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
        }
    result_path = run / "topology_result.json"
    if not result_path.is_file():
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
        }
    try:
        result = _json_object(result_path)
        _require_current_json_fingerprint(result, "topology result")
        preflight_name = result.get("preflight")
        if not isinstance(preflight_name, str) or not preflight_name:
            raise ValueError("topology result preflight is missing")
        preflight = _json_object(run / preflight_name)
        _require_current_json_fingerprint(preflight, "topology preflight")
        if (
            result.get("preflight_fingerprint") != preflight.get("fingerprint")
            or result.get("registration_result_sha256")
            != preflight.get("registration_result_sha256")
            or result.get("semantic_fingerprint")
            != preflight.get("semantic_fingerprint")
        ):
            raise ValueError("topology source binding is stale")
        snapshot = preflight.get("semantic_approval")
        snapshot_ready = (
            isinstance(snapshot, dict)
            and snapshot.get("semantic_fingerprint")
            == preflight.get("semantic_fingerprint")
            and snapshot.get("registration_result_sha256")
            == preflight.get("registration_result_sha256")
            and isinstance(snapshot.get("semantic_reviewer"), str)
            and bool(str(snapshot["semantic_reviewer"]).strip())
        )
        if not snapshot_ready:
            return {
                "available": True,
                "approved": False,
                "approval_ready": False,
                "issue": "approval_bound_rebuild_required",
            }
        if semantic_run is None:
            raise ValueError("topology semantic source is not configured")
        semantic_state = _semantic_status(registration_run, semantic_run)
        if semantic_state.get("approved") is not True:
            raise ValueError("topology semantic approval is not current")
        registration_bytes = (
            registration_run / "registration_result.json"
        ).read_bytes()
        semantic_bytes = (semantic_run / "semantic_result.json").read_bytes()
        semantic_result = json.loads(semantic_bytes)
        semantic_review = _json_object(semantic_run / "semantic_review.json")
        if (
            not isinstance(semantic_result, dict)
            or preflight.get("registration_result_sha256")
            != hashlib.sha256(registration_bytes).hexdigest()
            or preflight.get("semantic_result_sha256")
            != hashlib.sha256(semantic_bytes).hexdigest()
            or preflight.get("semantic_fingerprint")
            != semantic_result.get("fingerprint")
            or snapshot.get("semantic_reviewer") != semantic_review.get("reviewer")
        ):
            raise ValueError("topology current source binding is stale")
        review = _json_object(run / "topology_review.json")
        if (
            review.get("schema_version") != 1
            or review.get("fingerprint") != result.get("fingerprint")
            or not isinstance(review.get("approved"), bool)
        ):
            raise ValueError("topology review is stale")
        approved = review["approved"] is True
        if approved and (
            not isinstance(review.get("reviewer"), str)
            or not str(review["reviewer"]).strip()
            or not isinstance(review.get("notes"), str)
            or not str(review["notes"]).strip()
        ):
            raise ValueError("topology approval metadata is invalid")
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {
            "available": True,
            "approved": False,
            "approval_ready": False,
            "invalid": True,
            "issue": "topology_result_or_approval_invalid",
        }
    return {
        "available": True,
        "approved": approved,
        "approval_ready": not approved,
        "issue": None if approved else "topology_approval_required",
    }


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _require_current_json_fingerprint(
    payload: dict[str, object],
    name: str,
) -> None:
    fingerprint = payload.get("fingerprint")
    core = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != expected:
        raise ValueError(f"{name} fingerprint is stale")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lightweight_stain_approvals(
    run: Path,
    result: dict[str, object],
    families: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        review = _json_object(run / "stain_review.json")
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return ()
    if review.get("fingerprint") != result.get("fingerprint"):
        return ()
    if review.get("schema_version") == 1:
        if review.get("approved") is not True:
            return ()
        _require_review_metadata(review, "stain approval")
        return families
    if review.get("schema_version") != 2:
        return ()
    rows = review.get("families")
    if not isinstance(rows, dict):
        return ()
    approved: list[str] = []
    for family in families:
        row = rows.get(family)
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        _require_review_metadata(row, f"stain approval for {family}")
        approved.append(family)
    return tuple(approved)


def _require_review_metadata(payload: dict[str, object], name: str) -> None:
    if (
        not isinstance(payload.get("reviewer"), str)
        or not str(payload["reviewer"]).strip()
        or not isinstance(payload.get("reviewed_at"), str)
        or not str(payload["reviewed_at"]).strip()
        or not isinstance(payload.get("notes"), str)
        or not str(payload["notes"]).strip()
    ):
        raise ValueError(f"{name} metadata is invalid")


def _validate_stain_for_approval(
    registration_run: Path,
    stain_run: Path,
) -> None:
    from histopia.stain import validate_stain_result

    result = validate_stain_result(stain_run)
    registration_bytes = (registration_run / "registration_result.json").read_bytes()
    registration = json.loads(registration_bytes)
    registration_slides = (
        registration.get("slides") if isinstance(registration, dict) else None
    )
    stain_slides = result.get("slides")
    if (
        not isinstance(registration_slides, list)
        or not registration_slides
        or not isinstance(stain_slides, list)
        or len(stain_slides) != len(registration_slides)
        or result.get("registration_result_sha256")
        != hashlib.sha256(registration_bytes).hexdigest()
    ):
        raise ValueError(
            "stain approval requires a result bound to the current registration"
        )


def _validate_topology_for_approval(
    registration_run: Path,
    semantic_run: Path,
    topology_run: Path,
) -> None:
    from histopia.visualization._audit import audit_workflows

    report = audit_workflows(
        {"current": registration_run},
        semantic_runs={"current": semantic_run},
        topology_runs={"current": topology_run},
    )
    cohort = report.cohorts[0]
    if cohort.registration.status != "approved":
        raise ValueError("topology approval requires the current registration approval")
    if cohort.semantic.status != "approved":
        raise ValueError("topology approval requires the current semantic approval")
    if cohort.topology.status not in {"approved", "review_required"} or (
        cohort.topology.status == "review_required"
        and cohort.topology.issue != "topology_approval_required"
    ):
        issue = cohort.topology.issue or cohort.topology.status
        raise ValueError(
            "topology approval requires a result bound to the current approved "
            f"registration and semantic atlas ({issue})"
        )
