"""Portable, target-free combiners for independently trained protein models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic

_DIGEST = re.compile(r"[0-9a-f]{64}")
_METHOD = "query-quantile-tail-switch-v1"
_ROLES = ("base", "tail", "graph")


@dataclass(frozen=True, slots=True)
class TailSwitchResult:
    """Combined OD, uncertainty, and query-derived switching diagnostics."""

    predicted_od: np.ndarray
    uncertainty: np.ndarray
    low_switched: np.ndarray
    high_switched: np.ndarray
    lower_threshold: np.ndarray
    upper_threshold: np.ndarray


@dataclass(frozen=True, slots=True)
class TargetFreeTailSwitch:
    """Frozen query-only correction for compressed protein-expression tails.

    The component bundle bindings identify three independently trained models:
    the primary model, a tail-aware variant, and a graph variant. Switching is
    based only on their predictions for the query cohort. No measured protein
    outcome, semantic label, or protected-confirmation value is consumed.
    """

    component_bundle_fingerprints: tuple[str, str, str]
    component_bundle_sha256: tuple[str, str, str]
    low_quantile: float = 0.25
    high_quantile: float = 0.90
    high_blend: float = 0.40
    signal_base_weight: float = 0.50
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name, values in (
            ("component_bundle_fingerprints", self.component_bundle_fingerprints),
            ("component_bundle_sha256", self.component_bundle_sha256),
        ):
            if len(values) != len(_ROLES) or any(
                not isinstance(value, str) or _DIGEST.fullmatch(value) is None
                for value in values
            ):
                raise ValueError(f"{name} must contain three SHA-256 digests")
        if len(set(self.component_bundle_fingerprints)) != len(_ROLES):
            raise ValueError("tail-switch component bundles must be distinct")
        for name, value in (
            ("low_quantile", self.low_quantile),
            ("high_quantile", self.high_quantile),
            ("high_blend", self.high_blend),
            ("signal_base_weight", self.signal_base_weight),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.low_quantile < self.high_quantile <= 1:
            raise ValueError("tail-switch quantiles must be ordered within [0, 1]")
        if not 0 <= self.high_blend <= 1:
            raise ValueError("tail-switch high blend must be within [0, 1]")
        if not 0 <= self.signal_base_weight <= 1:
            raise ValueError("tail-switch signal weight must be within [0, 1]")
        expected = _protocol_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("tail-switch protocol fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def combine(
        self,
        base_od: np.ndarray,
        tail_od: np.ndarray,
        graph_od: np.ndarray,
        *,
        base_uncertainty: np.ndarray | None = None,
        tail_uncertainty: np.ndarray | None = None,
        graph_uncertainty: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
    ) -> TailSwitchResult:
        """Combine aligned predictions using quantiles derived per query group."""

        members = tuple(
            _prediction_vector(name, values)
            for name, values in zip(
                _ROLES,
                (base_od, tail_od, graph_od),
                strict=True,
            )
        )
        count = len(members[0])
        if any(len(values) != count for values in members[1:]):
            raise ValueError("tail-switch component predictions must align")
        uncertainty_members = tuple(
            _uncertainty_vector(name, values, count)
            for name, values in zip(
                _ROLES,
                (base_uncertainty, tail_uncertainty, graph_uncertainty),
                strict=True,
            )
        )
        groups = _group_vector(group_ids, count)
        base, tail, graph = members
        base_unc, tail_unc, graph_unc = uncertainty_members
        valid = np.logical_and.reduce([np.isfinite(values) for values in members])
        signal = (
            self.signal_base_weight * base + (1.0 - self.signal_base_weight) * graph
        )
        predicted = base.copy()
        uncertainty = base_unc.copy()
        low = np.zeros(count, dtype=bool)
        high = np.zeros(count, dtype=bool)
        lower_threshold = np.full(count, np.nan, dtype=np.float64)
        upper_threshold = np.full(count, np.nan, dtype=np.float64)
        for group in np.unique(groups):
            selected = (groups == group) & valid
            if not np.any(selected):
                continue
            lower, upper = np.quantile(
                signal[selected],
                (self.low_quantile, self.high_quantile),
            )
            lower_threshold[selected] = lower
            upper_threshold[selected] = upper
            group_low = selected & (signal <= lower)
            group_high = selected & (signal >= upper)
            low[group_low] = True
            high[group_high] = True

            choose_tail = group_low & (tail < base)
            tied_low = group_low & (tail == base)
            predicted[choose_tail] = tail[choose_tail]
            uncertainty[choose_tail] = tail_unc[choose_tail]
            uncertainty[tied_low] = np.maximum(base_unc[tied_low], tail_unc[tied_low])

            member_od = np.stack(
                (base[group_high], tail[group_high], graph[group_high])
            )
            member_unc = np.stack(
                (base_unc[group_high], tail_unc[group_high], graph_unc[group_high])
            )
            target = np.max(member_od, axis=0)
            target_uncertainty = np.max(
                np.where(member_od == target[None, :], member_unc, -np.inf),
                axis=0,
            )
            current = predicted[group_high].copy()
            current_uncertainty = uncertainty[group_high].copy()
            blend = self.high_blend
            predicted[group_high] = current + blend * (target - current)
            uncertainty[group_high] = np.sqrt(
                np.maximum(
                    (1.0 - blend) ** 2 * current_uncertainty**2
                    + blend**2 * target_uncertainty**2
                    + blend * (1.0 - blend) * (target - current) ** 2,
                    0.0,
                )
            )
        return TailSwitchResult(
            predicted_od=predicted.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            low_switched=low,
            high_switched=high,
            lower_threshold=lower_threshold.astype(np.float32),
            upper_threshold=upper_threshold.astype(np.float32),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a path-free portable protocol payload."""

        return {
            "schema_version": 1,
            "method": _METHOD,
            "component_roles": list(_ROLES),
            "component_bundle_fingerprints": list(self.component_bundle_fingerprints),
            "component_bundle_sha256": list(self.component_bundle_sha256),
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "high_blend": self.high_blend,
            "signal_base_weight": self.signal_base_weight,
            "outcome_free_gating": True,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: Path | str) -> Path:
        """Atomically save the portable protocol as JSON."""

        return write_json_atomic(path, self.to_dict(), sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> TargetFreeTailSwitch:
        """Load and validate a portable protocol."""

        payload = json.loads(Path(path).read_text())
        expected_keys = {
            "schema_version",
            "method",
            "component_roles",
            "component_bundle_fingerprints",
            "component_bundle_sha256",
            "low_quantile",
            "high_quantile",
            "high_blend",
            "signal_base_weight",
            "outcome_free_gating",
            "fingerprint",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_keys
            or payload.get("schema_version") != 1
            or payload.get("method") != _METHOD
            or payload.get("component_roles") != list(_ROLES)
            or payload.get("outcome_free_gating") is not True
        ):
            raise ValueError("unsupported target-free tail-switch protocol")
        return cls(
            component_bundle_fingerprints=tuple(
                payload["component_bundle_fingerprints"]
            ),
            component_bundle_sha256=tuple(payload["component_bundle_sha256"]),
            low_quantile=float(payload["low_quantile"]),
            high_quantile=float(payload["high_quantile"]),
            high_blend=float(payload["high_blend"]),
            signal_base_weight=float(payload["signal_base_weight"]),
            fingerprint=str(payload["fingerprint"]),
        )


def _prediction_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or np.any(vector[np.isfinite(vector)] < 0):
        raise ValueError(f"{name} predictions must be a nonnegative vector")
    return vector


def _uncertainty_vector(
    name: str,
    values: np.ndarray | None,
    count: int,
) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (count,) or np.any(~np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError(f"{name} uncertainty must be a finite nonnegative vector")
    return vector


def _group_vector(values: np.ndarray | None, count: int) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.int8)
    groups = np.asarray(values)
    if groups.shape != (count,):
        raise ValueError("tail-switch query groups must align with predictions")
    return groups


def _protocol_fingerprint(protocol: TargetFreeTailSwitch) -> str:
    payload = {
        "schema_version": 1,
        "method": _METHOD,
        "component_roles": list(_ROLES),
        "component_bundle_fingerprints": list(protocol.component_bundle_fingerprints),
        "component_bundle_sha256": list(protocol.component_bundle_sha256),
        "low_quantile": protocol.low_quantile,
        "high_quantile": protocol.high_quantile,
        "high_blend": protocol.high_blend,
        "signal_base_weight": protocol.signal_base_weight,
        "outcome_free_gating": True,
    }
    digest = hashlib.sha256(b"histopia-target-free-tail-switch-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
