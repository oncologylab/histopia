"""Persistent pair-level review feedback for topology reconstruction."""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from histopia._atomic import write_json_atomic
from histopia.topology._result import validate_topology_result

TOPOLOGY_FEEDBACK_LABELS = (
    "wrong_gap_count",
    "unsupported_interpolation",
    "surface_fragmentation",
    "class_discontinuity",
    "excess_uncertainty",
    "wrong_z_spacing",
    "other",
)
_DECISIONS = frozenset({"accept", "hold", "reject"})


class TopologyFeedbackStore:
    """Append fingerprint-bound decisions for every reconstructed transition."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def review(self, *, cohort: str, topology_run: Path) -> dict[str, object]:
        evidence = topology_feedback_evidence(topology_run)
        payload = self._load(cohort, str(evidence["fingerprint"]))
        latest = _latest(payload["records"])
        return {
            **evidence,
            "cohort": cohort,
            "labels": list(TOPOLOGY_FEEDBACK_LABELS),
            "feedback": latest,
            "summary": _summary(latest),
        }

    def save(
        self,
        request: dict[str, object],
        *,
        topology_run: Path,
    ) -> dict[str, object]:
        cohort = _required_text(request, "cohort")
        if request.get("stage") != "topology":
            raise ValueError("topology feedback stage must be topology")
        evidence = topology_feedback_evidence(topology_run)
        fingerprint = _required_text(request, "fingerprint")
        if fingerprint != evidence["fingerprint"]:
            raise ValueError("topology evidence changed; reload before saving feedback")
        pair_id = _required_text(request, "slide_id")
        pairs = {str(row["id"]): row for row in evidence["slides"]}
        if pair_id not in pairs:
            raise ValueError("topology pair is not part of the current evidence")
        decision = _required_text(request, "decision")
        if decision not in _DECISIONS:
            raise ValueError("feedback decision must be accept, hold, or reject")
        labels = request.get("labels", [])
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
            or not set(labels).issubset(TOPOLOGY_FEEDBACK_LABELS)
        ):
            raise ValueError("topology feedback contains invalid issue labels")
        comment = request.get("comment", "")
        if not isinstance(comment, str) or len(comment) > 4_000:
            raise ValueError("comment must be text no longer than 4000 characters")
        reviewer = _required_text(request, "reviewer")
        if decision == "accept" and labels:
            raise ValueError("accepted feedback cannot contain issue labels")
        if decision != "accept" and not labels and not comment.strip():
            raise ValueError("hold or reject feedback requires an issue or comment")
        suggested_intervals = request.get("suggested_intervals")
        if suggested_intervals is not None and (
            isinstance(suggested_intervals, bool)
            or not isinstance(suggested_intervals, int)
            or not 1 <= suggested_intervals <= 4
        ):
            raise ValueError("suggested intervals must be an integer from 1 to 4")

        with self._lock:
            payload = self._load(cohort, fingerprint)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            record = {
                "record_id": uuid4().hex,
                "slide_id": pair_id,
                "slide_order": int(pairs[pair_id]["order"]),
                "decision": decision,
                "labels": labels,
                "comment": comment.strip(),
                "reviewer": reviewer,
                "reviewed_at": timestamp,
            }
            if suggested_intervals is not None:
                record["suggested_intervals"] = suggested_intervals
            payload["records"].append(record)
            payload["updated_at"] = timestamp
            write_json_atomic(self._path(cohort, fingerprint), payload)
        return self.review(cohort=cohort, topology_run=topology_run)

    def require_accepted(self, *, cohort: str, topology_run: Path) -> None:
        review = self.review(cohort=cohort, topology_run=topology_run)
        latest = review["feedback"]
        expected = [str(row["id"]) for row in review["slides"]]
        missing = [pair for pair in expected if pair not in latest]
        unresolved = [
            pair
            for pair in expected
            if pair in latest and latest[pair]["decision"] != "accept"
        ]
        if missing or unresolved:
            raise ValueError(
                "topology feedback is incomplete: "
                f"{len(missing)} unreviewed, {len(unresolved)} unresolved"
            )

    def summary(self) -> dict[str, object]:
        decisions: Counter[str] = Counter()
        issues: Counter[str] = Counter()
        pairs = 0
        for path in sorted(self.root.glob("*/topology/*.json")):
            payload = json.loads(path.read_text())
            for record in _latest(payload.get("records")).values():
                pairs += 1
                decisions[str(record["decision"])] += 1
                issues.update(str(value) for value in record.get("labels", []))
        return {
            "schema_version": 1,
            "reviewed_pairs": pairs,
            "by_decision": dict(sorted(decisions.items())),
            "by_issue": dict(sorted(issues.items())),
        }

    def _path(self, cohort: str, fingerprint: str) -> Path:
        return self.root / cohort / "topology" / f"{fingerprint}.json"

    def _load(self, cohort: str, fingerprint: str) -> dict[str, object]:
        path = self._path(cohort, fingerprint)
        if not path.is_file():
            return {
                "schema_version": 1,
                "cohort": cohort,
                "stage": "topology",
                "artifact_fingerprint": fingerprint,
                "updated_at": None,
                "records": [],
            }
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("cohort") != cohort
            or payload.get("stage") != "topology"
            or payload.get("artifact_fingerprint") != fingerprint
            or not isinstance(payload.get("records"), list)
        ):
            raise ValueError("stored topology feedback is invalid")
        return payload


def topology_feedback_evidence(topology_run: Path | str) -> dict[str, object]:
    payload = validate_topology_result(topology_run)
    decisions = payload.get("gap_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("topology result contains no pair decisions")
    pairs = []
    for order, row in enumerate(decisions, start=1):
        source = int(row["source_section"])
        target = int(row["target_section"])
        pairs.append(
            {
                "id": f"{source:03d}-{target:03d}",
                "order": order,
                "source_section": source,
                "target_section": target,
                "status": row["status"],
                "intervals": int(row["intervals"]),
            }
        )
    return {
        "schema_version": 1,
        "stage": "topology",
        "fingerprint": payload["fingerprint"],
        "slides": pairs,
    }


def _latest(records: object) -> dict[str, dict[str, object]]:
    if not isinstance(records, list):
        raise ValueError("topology feedback records must be a list")
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("slide_id"), str):
            raise ValueError("topology feedback contains an invalid record")
        latest[str(record["slide_id"])] = record
    return latest


def _summary(records: dict[str, dict[str, object]]) -> dict[str, object]:
    decisions = Counter(str(record["decision"]) for record in records.values())
    return {
        "reviewed": len(records),
        "by_decision": dict(sorted(decisions.items())),
    }


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must not be blank")
    return value.strip()
