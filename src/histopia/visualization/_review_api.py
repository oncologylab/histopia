"""Web decisions for prepared Histopia review artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from histopia.topology._feedback import TopologyFeedbackStore
from histopia.visualization._cell_scope import CellSectionScope
from histopia.visualization._feedback import RegistrationFeedbackStore
from histopia.visualization._provisional_feedback import ProvisionalFeedbackStore

if TYPE_CHECKING:
    from histopia.annotation import AnnotationStore

_COHORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_STAGES = (
    "mask",
    "order",
    "registration",
    "semantic",
    "topology",
    "stain",
    "cells",
    "protein",
)


@dataclass(frozen=True, slots=True)
class ReviewRuns:
    """Filesystem locations for one cohort's reviewable workflows."""

    registration: Path
    semantic: Path | None = None
    topology: Path | None = None
    stain: Path | None = None
    cells: Path | None = None
    cell_geometry: Path | None = None
    protein: Path | None = None
    protein_models: dict[str, Path] = field(default_factory=dict)
    protein_model_stains: dict[str, Path] = field(default_factory=dict)
    annotations: Path | None = None
    registered_wsi: Path | None = None
    cell_section_scope: CellSectionScope | None = None


class ReviewDecisionService:
    """Apply fingerprint-bound approvals to an explicit local run registry."""

    def __init__(
        self,
        cohorts: dict[str, ReviewRuns],
        *,
        feedback_store: RegistrationFeedbackStore | None = None,
        topology_feedback_store: TopologyFeedbackStore | None = None,
        provisional_feedback_store: ProvisionalFeedbackStore | None = None,
    ) -> None:
        if not cohorts:
            raise ValueError("review registry must contain at least one cohort")
        self._cohorts = dict(sorted(cohorts.items()))
        self._feedback_store = feedback_store
        self._topology_feedback_store = topology_feedback_store
        self._provisional_feedback_store = provisional_feedback_store
        self._annotation_stores: dict[str, AnnotationStore] = {}

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
        shared_protein_models, shared_protein_stains = _configured_protein_models(
            payload,
            config_path.parent,
            key="shared_protein_models",
        )
        if shared_protein_stains:
            raise ValueError(
                "review registry shared_protein_models cannot declare a "
                "cohort-specific stain"
            )
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
            protein_models, protein_model_stains = _configured_protein_models(
                raw,
                config_path.parent,
            )
            shared_stain_value = raw.get("shared_protein_stain")
            if shared_stain_value is not None:
                if not shared_protein_models:
                    raise ValueError(
                        "review registry shared_protein_stain requires "
                        "shared_protein_models"
                    )
                shared_stain = _resolved_configured_path(
                    shared_stain_value,
                    f"cohorts.{cohort}.shared_protein_stain",
                    config_path.parent,
                )
                for model_id, shared_run in shared_protein_models.items():
                    local_run = protein_models.get(model_id)
                    if local_run is not None and local_run != shared_run:
                        raise ValueError(
                            f"review registry protein model {model_id!r} "
                            "has conflicting local and shared runs"
                        )
                protein_models = {
                    **shared_protein_models,
                    **protein_models,
                }
                protein_model_stains = {
                    **{model_id: shared_stain for model_id in shared_protein_models},
                    **protein_model_stains,
                }
            raw_scope = raw.get("cell_section_scope")
            if raw_scope is not None and raw.get("cells") is None:
                raise ValueError("cell scope has no matching cell run")
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
                cells=_configured_path(
                    raw,
                    "cells",
                    config_path.parent,
                    required=False,
                ),
                cell_geometry=_configured_path(
                    raw,
                    "cell_geometry",
                    config_path.parent,
                    required=False,
                ),
                protein=_configured_path(
                    raw,
                    "protein",
                    config_path.parent,
                    required=False,
                ),
                protein_models=protein_models,
                cell_section_scope=(
                    CellSectionScope.from_dict(raw_scope)
                    if raw_scope is not None
                    else None
                ),
                protein_model_stains=protein_model_stains,
                annotations=_configured_path(
                    raw,
                    "annotations",
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
            provisional_feedback_store=(
                ProvisionalFeedbackStore(feedback_path / "provisional")
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
            "provisional_feedback_configured": (
                self._provisional_feedback_store is not None
            ),
            "cohorts": [
                self._cohort_status(name, runs) for name, runs in self._cohorts.items()
            ],
        }

    def wsi_runs(self) -> dict[str, tuple[Path, Path | None]]:
        """Return private WSI bindings for server construction only."""

        return {
            cohort: (runs.registration, runs.registered_wsi)
            for cohort, runs in self._cohorts.items()
            if (
                runs.registered_wsi is not None
                or runs.cells is not None
                or runs.annotations is not None
                or runs.stain is not None
                or runs.protein is not None
                or runs.protein_models
            )
        }

    def cell_runs(self) -> dict[str, Path]:
        """Return private validated cell-run bindings for tile rendering."""

        return {
            cohort: runs.cells
            for cohort, runs in self._cohorts.items()
            if runs.cells is not None
        }

    def cell_section_scopes(self) -> dict[str, CellSectionScope]:
        """Return exact cell-layer display scopes without granting approval."""
        return {
            cohort: runs.cell_section_scope
            for cohort, runs in self._cohorts.items()
            if runs.cell_section_scope is not None
        }

    def cell_geometry_runs(self) -> dict[str, Path]:
        """Return private multiscale cell-geometry caches for static builds."""

        return {
            cohort: runs.cell_geometry
            for cohort, runs in self._cohorts.items()
            if runs.cell_geometry is not None
        }

    def stain_runs(self) -> dict[str, Path]:
        """Return private validated stain-run bindings for tile rendering."""

        return {
            cohort: runs.stain
            for cohort, runs in self._cohorts.items()
            if runs.stain is not None
        }

    def protein_runs(self) -> dict[str, Path]:
        """Return private validated protein-run bindings for tile rendering."""

        return {
            cohort: runs.protein
            for cohort, runs in self._cohorts.items()
            if runs.protein is not None
        }

    def protein_model_runs(self) -> dict[str, dict[str, Path]]:
        """Return private model-scoped protein bindings for tile rendering."""

        return {
            cohort: dict(sorted(runs.protein_models.items()))
            for cohort, runs in self._cohorts.items()
            if runs.protein_models
        }

    def protein_model_stain_runs(self) -> dict[str, dict[str, Path]]:
        """Return exact stain provenance overrides for configured models."""

        return {
            cohort: dict(sorted(runs.protein_model_stains.items()))
            for cohort, runs in self._cohorts.items()
            if runs.protein_model_stains
        }

    def configured_runs(self) -> dict[str, ReviewRuns]:
        """Return a copy of the local registry for local audit/build commands."""

        return dict(self._cohorts)

    def annotation_catalog(self, cohort: str) -> dict[str, object]:
        """Return path-free ontology and annotation revision metadata."""

        catalog = self._annotation_store(cohort).catalog()
        return {**catalog, "cohort": cohort}

    def annotation_section(self, cohort: str, section: str) -> dict[str, object]:
        """Return one current section annotation collection."""

        return self._annotation_store(cohort).read_section(section)

    def save_annotation_section(
        self,
        request: dict[str, object],
    ) -> dict[str, object]:
        """Persist one optimistic, revision-bound annotation update."""

        cohort = _required_text(request, "cohort")
        section = _required_text(request, "section")
        reviewer = _required_text(request, "reviewer")
        expected_revision = request.get("expected_revision")
        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision, int
        ):
            raise TypeError("annotation expected_revision must be an integer")
        feature_collection = request.get("feature_collection")
        notes = request.get("notes", "")
        if not isinstance(notes, str):
            raise TypeError("annotation notes must be text")
        return self._annotation_store(cohort, validate_current=True).save_section(
            section,
            feature_collection,
            reviewer=reviewer,
            expected_revision=expected_revision,
            notes=notes,
        )

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
        elif stage == "stain":
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
        else:
            if runs.cells is None:
                raise ValueError(f"cohort {cohort} has no cell review")
            if runs.cell_section_scope is not None:
                raise ValueError("selected cell sections cannot approve a whole run")
            from histopia.cells import approve_cell_result

            approve_cell_result(runs.cells)
        return self._cohort_status(cohort, runs)

    def review_cell_section(self, request: dict[str, object]) -> dict[str, object]:
        """Persist one section-level cell-boundary decision."""

        cohort = _required_text(request, "cohort")
        section = _required_text(request, "section")
        reviewer = _required_text(request, "reviewer")
        runs = self._required_cohort(cohort)
        if runs.cells is None:
            raise ValueError(f"cohort {cohort} has no cell review")
        self._require_cell_section_in_scope(runs, section)
        accepted = request.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("cell review accepted must be a boolean")
        notes = request.get("notes", "")
        issues = request.get("issues", [])
        if not isinstance(notes, str):
            raise ValueError("cell review notes must be text")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) for issue in issues
        ):
            raise ValueError("cell review issues must be a list of text labels")
        from histopia.cells import review_cell_section

        return review_cell_section(
            runs.cells,
            section,
            accepted=accepted,
            reviewer=reviewer,
            notes=notes,
            issues=tuple(issues),
        )

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
            "provisional": (
                self._provisional_feedback_store.summary()
                if self._provisional_feedback_store is not None
                else None
            ),
        }

    def provisional_feedback(self, cohort: str, stage: str) -> dict[str, object]:
        """Return current stain or cell observations without approving artifacts."""

        runs = self._required_cohort(cohort)
        run = (
            runs.stain if stage == "stain" else runs.cells if stage == "cells" else None
        )
        if run is None:
            raise ValueError(f"cohort {cohort} has no {stage} review")
        return self._required_provisional_feedback_store().review(
            cohort=cohort,
            stage=stage,
            run=run,
        )

    def save_provisional_feedback(
        self, request: dict[str, object]
    ) -> dict[str, object]:
        """Persist one explicitly provisional stain or cell observation."""

        cohort = _required_text(request, "cohort")
        stage = _required_text(request, "stage")
        runs = self._required_cohort(cohort)
        run = (
            runs.stain if stage == "stain" else runs.cells if stage == "cells" else None
        )
        if run is None:
            raise ValueError(f"cohort {cohort} has no {stage} review")
        if stage == "cells" and runs.cell_section_scope is not None:
            self._require_cell_section_in_scope(
                runs, _required_text(request, "slide_id")
            )
        return self._required_provisional_feedback_store().save(request, run=run)

    @staticmethod
    def _require_cell_section_in_scope(runs: ReviewRuns, section: str) -> None:
        """Bind a scoped review write to the exact displayed cell artifact."""

        scope = runs.cell_section_scope
        if scope is None:
            return
        if section not in scope.labels_by_section:
            raise ValueError("cell section is outside the selected review scope")
        from histopia.cells._result import validate_cell_result_index

        assert runs.cells is not None
        scope.select(validate_cell_result_index(runs.cells))

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

    def _required_provisional_feedback_store(self) -> ProvisionalFeedbackStore:
        if self._provisional_feedback_store is None:
            raise ValueError("provisional feedback storage is not configured")
        return self._provisional_feedback_store

    def _annotation_store(
        self,
        cohort: str,
        *,
        validate_current: bool = False,
    ) -> AnnotationStore:
        if not validate_current:
            try:
                return self._annotation_stores[cohort]
            except KeyError:
                pass
        runs = self._required_cohort(cohort)
        if runs.annotations is None:
            raise ValueError(f"cohort {cohort} has no annotation review")
        if runs.semantic is None:
            raise ValueError(f"cohort {cohort} has no semantic review")
        from histopia.annotation import AnnotationStore

        store = AnnotationStore.from_runs(
            runs.annotations,
            registration_run=runs.registration,
            semantic_run=runs.semantic,
        )
        self._annotation_stores[cohort] = store
        return store

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
                "cells": _cell_status(runs.registration, runs.cells),
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


def _configured_protein_models(
    row: dict[str, object],
    base: Path,
    *,
    key: str = "protein_models",
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Parse legacy paths and provenance-bound protein model descriptors."""

    raw = row.get(key)
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"review registry {key} must be a non-empty object")
    runs: dict[str, Path] = {}
    stains: dict[str, Path] = {}
    for model_id, value in raw.items():
        if not isinstance(model_id, str) or not _COHORT_RE.fullmatch(model_id):
            raise ValueError(f"invalid protein model name: {model_id!r}")
        run_value: object = value
        stain_value: object | None = None
        if isinstance(value, dict):
            unknown = set(value) - {"run", "stain"}
            if unknown:
                raise ValueError(
                    f"review registry {key}.{model_id} has unknown fields: "
                    + ", ".join(sorted(unknown))
                )
            run_value = value.get("run")
            stain_value = value.get("stain")
        runs[model_id] = _resolved_configured_path(
            run_value,
            f"{key}.{model_id}.run" if isinstance(value, dict) else f"{key}.{model_id}",
            base,
        )
        if isinstance(value, dict) and "stain" in value:
            stains[model_id] = _resolved_configured_path(
                stain_value,
                f"{key}.{model_id}.stain",
                base,
            )
    return dict(sorted(runs.items())), dict(sorted(stains.items()))


def _resolved_configured_path(value: object, label: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review registry {label} path is missing")
    path = Path(value).expanduser()
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


def _cell_status(
    registration_run: Path,
    run: Path | None,
) -> dict[str, object]:
    if run is None:
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
            "sections": [],
        }
    result_path = run / "cell_result.json"
    if not result_path.is_file():
        return {
            "available": False,
            "approved": False,
            "approval_ready": False,
            "sections": [],
        }
    try:
        result = _json_object(result_path)
        _require_current_json_fingerprint(result, "cell result")
        registration_sha = hashlib.sha256(
            (registration_run / "registration_result.json").read_bytes()
        ).hexdigest()
        if result.get("registration_result_sha256") != registration_sha:
            raise ValueError("cell registration binding is stale")
        slides = result.get("slides")
        if not isinstance(slides, list) or not slides:
            raise ValueError("cell result contains no sections")
        review = _json_object(run / "cell_review.json")
        if (
            review.get("schema_version") != 1
            or review.get("fingerprint") != result.get("fingerprint")
            or not isinstance(review.get("sections"), dict)
        ):
            raise ValueError("cell review is stale")
        review_rows = review["sections"]
        section_rows = []
        for slide in slides:
            if not isinstance(slide, dict) or not isinstance(slide.get("section"), str):
                raise ValueError("cell result section row is invalid")
            section = str(slide["section"])
            decision = review_rows.get(section)
            accepted = isinstance(decision, dict) and decision.get("accepted") is True
            section_rows.append(
                {
                    "id": section,
                    "slide": str(slide.get("slide", "")),
                    "cell_count": int(slide.get("cell_count", 0)),
                    "accepted": accepted,
                    "reviewer": (
                        decision.get("reviewer") if isinstance(decision, dict) else None
                    ),
                    "reviewed_at": (
                        decision.get("reviewed_at")
                        if isinstance(decision, dict)
                        else None
                    ),
                    "notes": (
                        str(decision.get("notes", ""))
                        if isinstance(decision, dict)
                        else ""
                    ),
                    "issues": (
                        list(decision.get("issues", []))
                        if isinstance(decision, dict)
                        and isinstance(decision.get("issues"), list)
                        else []
                    ),
                }
            )
        approved = all(row["accepted"] for row in section_rows)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "available": True,
            "approved": False,
            "approval_ready": False,
            "sections": [],
            "invalid": True,
            "issue": "cell_result_binding_or_review_invalid",
        }
    return {
        "available": True,
        "approved": approved,
        "approval_ready": not approved,
        "issue": None if approved else "cell_section_review_required",
        "sections": section_rows,
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
