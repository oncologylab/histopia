"""Portable, fingerprinted protein training and prediction tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic


@dataclass(frozen=True, slots=True)
class CellExpressionTable:
    """One row per native cell with neutral features and optional measured OD."""

    target_id: str
    label_ids: np.ndarray
    section_ids: np.ndarray
    mouse_ids: np.ndarray
    native_xy: np.ndarray
    reference_um_xy: np.ndarray
    features: np.ndarray
    measured_od: np.ndarray
    binary_label: np.ndarray
    measurement_coverage: np.ndarray
    semantic_region: np.ndarray
    semantic_support: np.ndarray
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        count = len(np.asarray(self.label_ids))
        vectors = (
            self.section_ids,
            self.mouse_ids,
            self.measured_od,
            self.binary_label,
            self.measurement_coverage,
            self.semantic_region,
            self.semantic_support,
        )
        if any(np.asarray(value).shape != (count,) for value in vectors):
            raise ValueError("cell expression vectors must contain one value per cell")
        for name, value in (
            ("native_xy", self.native_xy),
            ("reference_um_xy", self.reference_um_xy),
        ):
            if np.asarray(value).shape != (count, 2):
                raise ValueError(f"{name} must have shape (cells, 2)")
        if np.asarray(self.features).ndim != 2 or len(self.features) != count:
            raise ValueError("features must have shape (cells, features)")
        if not np.issubdtype(np.asarray(self.label_ids).dtype, np.integer):
            raise TypeError("label_ids must be integers")
        if np.any(np.asarray(self.label_ids) <= 0):
            raise ValueError("label_ids must be positive")
        for value in (self.native_xy, self.reference_um_xy, self.features):
            if not np.all(np.isfinite(value)):
                raise ValueError("cell coordinates and features must be finite")
        coverage = np.asarray(self.measurement_coverage, dtype=float)
        if np.any((coverage < 0) | (coverage > 1)):
            raise ValueError("measurement coverage must be between zero and one")
        labels = np.asarray(self.binary_label)
        if np.any(~np.isin(labels, (-1, 0, 1))):
            raise ValueError("binary labels must use -1, 0, or 1")
        expected = _table_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("cell expression fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        arrays = _table_arrays(self)
        metadata = {
            "schema_version": 1,
            "target_id": self.target_id,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **arrays,
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> CellExpressionTable:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported cell expression table schema")
            return cls(
                target_id=metadata["target_id"],
                provenance=metadata["provenance"],
                fingerprint=metadata["fingerprint"],
                **{name: data[name] for name in _TABLE_ARRAY_NAMES},
            )


def extend_cell_expression_table(
    base: CellExpressionTable,
    addition: CellExpressionTable,
    *,
    addition_cohorts: Sequence[str] | None = None,
) -> CellExpressionTable:
    """Append disjoint outcome cohorts without changing any base-table row.

    This operation supports immutable model evaluation on newly completed mice.
    The historical table remains the exact source of every training row, while
    the second table contributes only explicitly new cohorts.  Resampled rows
    for cohorts already present in ``base`` are intentionally ignored.

    Both tables must bind the same target, feature schema, measurement view,
    compartment, statistic, and other scientific scope.  Existing cohort
    bindings must agree exactly, new cell identities must be unique, and the
    resulting provenance records both immutable parent fingerprints.  OD values
    are copied as stored; this function performs no quantification, calibration,
    or harmonization.
    """

    if base.target_id != addition.target_id:
        raise ValueError("cell expression table targets differ")
    base_arrays = _table_arrays(base)
    addition_arrays = _table_arrays(addition)
    if base_arrays["features"].shape[1] != addition_arrays["features"].shape[1]:
        raise ValueError("cell expression table feature dimensions differ")

    base_provenance = _validated_table_provenance(base.provenance)
    addition_provenance = _validated_table_provenance(addition.provenance)
    dynamic = {"cohort_bindings", "target_sections", "table_extension"}
    base_scope = {
        key: value for key, value in base_provenance.items() if key not in dynamic
    }
    addition_scope = {
        key: value for key, value in addition_provenance.items() if key not in dynamic
    }
    if base_scope != addition_scope:
        raise ValueError("cell expression table scientific scopes differ")

    base_bindings = _validated_cohort_bindings(base_provenance)
    candidate_bindings = _validated_cohort_bindings(addition_provenance)
    for cohort in set(base_bindings) & set(candidate_bindings):
        if base_bindings[cohort] != candidate_bindings[cohort]:
            raise ValueError(f"cell expression cohort binding differs: {cohort}")

    if addition_cohorts is None:
        selected_cohorts = tuple(sorted(set(candidate_bindings) - set(base_bindings)))
    else:
        selected_cohorts = tuple(str(value) for value in addition_cohorts)
        if (
            not selected_cohorts
            or any(not value for value in selected_cohorts)
            or len(set(selected_cohorts)) != len(selected_cohorts)
        ):
            raise ValueError("addition cohorts must be non-empty and unique")
    if not selected_cohorts:
        raise ValueError("cell expression extension contains no new cohorts")
    if set(selected_cohorts) & set(base_bindings):
        raise ValueError("cell expression extension cohorts must be new")
    missing = set(selected_cohorts) - set(candidate_bindings)
    if missing:
        raise ValueError(
            "cell expression extension cohort binding is missing: "
            + ", ".join(sorted(missing))
        )

    base_mice = np.asarray(base_arrays["mouse_ids"], dtype=str)
    addition_mice = np.asarray(addition_arrays["mouse_ids"], dtype=str)
    if set(base_mice) - set(base_bindings):
        raise ValueError("base table rows lack cohort bindings")
    selected = np.flatnonzero(np.isin(addition_mice, selected_cohorts))
    if not len(selected) or set(addition_mice[selected]) != set(selected_cohorts):
        raise ValueError("cell expression extension cohort rows are incomplete")

    base_sections = _validated_target_sections(base_provenance)
    selected_sections = tuple(
        row
        for row in _validated_target_sections(addition_provenance)
        if row["cohort"] in selected_cohorts
    )
    expected_base_sections = set(
        zip(
            base_mice.tolist(),
            np.asarray(base_arrays["section_ids"], dtype=str),
            strict=True,
        )
    )
    expected_added_sections = set(
        zip(
            addition_mice[selected].tolist(),
            np.asarray(addition_arrays["section_ids"], dtype=str)[selected],
            strict=True,
        )
    )
    if _target_section_keys(base_sections) != expected_base_sections:
        raise ValueError("base table target-section provenance is incomplete")
    if _target_section_keys(selected_sections) != expected_added_sections:
        raise ValueError("extension target-section provenance is incomplete")

    _validate_unique_cell_identities(base_arrays)
    _validate_unique_cell_identities(
        {name: value[selected] for name, value in addition_arrays.items()}
    )
    merged_bindings = {
        **base_bindings,
        **{cohort: candidate_bindings[cohort] for cohort in selected_cohorts},
    }
    provenance = {
        **base_scope,
        "cohort_bindings": {
            key: merged_bindings[key] for key in sorted(merged_bindings)
        },
        "target_sections": [
            *base_sections,
            *sorted(
                selected_sections,
                key=lambda row: (str(row["cohort"]), str(row["section"])),
            ),
        ],
        "table_extension": {
            "protocol": "exact-base-plus-disjoint-outcome-cohorts-v1",
            "base_table_fingerprint": base.fingerprint,
            "addition_table_fingerprint": addition.fingerprint,
            "addition_cohorts": list(selected_cohorts),
        },
    }
    combined = {
        name: np.concatenate((base_arrays[name], addition_arrays[name][selected]))
        for name in _TABLE_ARRAY_NAMES
    }
    return CellExpressionTable(
        target_id=base.target_id,
        provenance=provenance,
        **combined,
    )


@dataclass(frozen=True, slots=True)
class ProteinPredictions:
    """Per-cell target predictions with explicit support and uncertainty."""

    target_id: str
    model_fingerprint: str
    label_ids: np.ndarray
    section_ids: np.ndarray
    expression_probability: np.ndarray
    relative_expression: np.ndarray
    predicted_od_reference: np.ndarray
    measured_od: np.ndarray
    uncertainty: np.ndarray
    supported: np.ndarray
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        count = len(np.asarray(self.label_ids))
        for value in (
            self.section_ids,
            self.expression_probability,
            self.relative_expression,
            self.predicted_od_reference,
            self.measured_od,
            self.uncertainty,
            self.supported,
        ):
            if np.asarray(value).shape != (count,):
                raise ValueError("protein prediction arrays must align")
        probability = np.asarray(self.expression_probability, dtype=float)
        relative = np.asarray(self.relative_expression, dtype=float)
        if np.any(np.isfinite(probability) & ((probability < 0) | (probability > 1))):
            raise ValueError(
                "expression probabilities must be null or between zero and one"
            )
        if np.any((relative < 0) | (relative > 1)):
            raise ValueError("relative expression must be between zero and one")
        if np.any(np.asarray(self.predicted_od_reference) < 0) or np.any(
            np.asarray(self.uncertainty) < 0
        ):
            raise ValueError("predicted OD and uncertainty must be nonnegative")
        expected = _prediction_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("protein prediction fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        arrays = _prediction_arrays(self)
        metadata = {
            "schema_version": 1,
            "target_id": self.target_id,
            "model_fingerprint": self.model_fingerprint,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **arrays,
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> ProteinPredictions:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported protein prediction schema")
            return cls(
                target_id=metadata["target_id"],
                model_fingerprint=metadata["model_fingerprint"],
                provenance=metadata["provenance"],
                fingerprint=metadata["fingerprint"],
                **{name: data[name] for name in _PREDICTION_ARRAY_NAMES},
            )


_TABLE_ARRAY_NAMES = (
    "label_ids",
    "section_ids",
    "mouse_ids",
    "native_xy",
    "reference_um_xy",
    "features",
    "measured_od",
    "binary_label",
    "measurement_coverage",
    "semantic_region",
    "semantic_support",
)
_PREDICTION_ARRAY_NAMES = (
    "label_ids",
    "section_ids",
    "expression_probability",
    "relative_expression",
    "predicted_od_reference",
    "measured_od",
    "uncertainty",
    "supported",
)


def _validated_table_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise ValueError("cell expression table provenance is malformed")
    # Round-trip through canonical JSON to detach mutable nested values and to
    # reject non-portable objects before constructing the extended artifact.
    try:
        cloned = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError("cell expression table provenance is not portable") from error
    if not isinstance(cloned, dict):  # pragma: no cover - guarded above
        raise ValueError("cell expression table provenance is malformed")
    return cloned


def _validated_cohort_bindings(
    provenance: dict[str, object],
) -> dict[str, object]:
    raw = provenance.get("cohort_bindings")
    if (
        not isinstance(raw, dict)
        or not raw
        or any(not isinstance(key, str) or not key for key in raw)
        or any(not isinstance(value, dict) or not value for value in raw.values())
    ):
        raise ValueError("cell expression table cohort bindings are malformed")
    return dict(raw)


def _validated_target_sections(
    provenance: dict[str, object],
) -> tuple[dict[str, object], ...]:
    raw = provenance.get("target_sections")
    if not isinstance(raw, list) or not raw:
        raise ValueError("cell expression table target sections are malformed")
    output: list[dict[str, object]] = []
    keys: set[tuple[str, str]] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("cell expression table target sections are malformed")
        row = dict(value)
        cohort = row.get("cohort")
        section = row.get("section")
        if (
            not isinstance(cohort, str)
            or not cohort
            or not isinstance(section, str)
            or not section
        ):
            raise ValueError("cell expression table target sections are malformed")
        key = (cohort, section)
        if key in keys:
            raise ValueError("cell expression table target sections are duplicated")
        keys.add(key)
        output.append(row)
    return tuple(output)


def _target_section_keys(
    rows: Sequence[dict[str, object]],
) -> set[tuple[str, str]]:
    return {(str(row["cohort"]), str(row["section"])) for row in rows}


def _validate_unique_cell_identities(arrays: dict[str, np.ndarray]) -> None:
    identities = np.rec.fromarrays(
        (
            np.asarray(arrays["mouse_ids"], dtype=str),
            np.asarray(arrays["section_ids"], dtype=str),
            np.asarray(arrays["label_ids"], dtype=np.uint32),
        ),
        names=("mouse", "section", "label"),
    )
    if len(np.unique(identities)) != len(identities):
        raise ValueError("cell expression table contains duplicate cell identities")


def _table_arrays(table: CellExpressionTable) -> dict[str, np.ndarray]:
    dtypes = {
        "label_ids": np.uint32,
        "section_ids": np.str_,
        "mouse_ids": np.str_,
        "native_xy": np.float64,
        "reference_um_xy": np.float64,
        "features": np.float32,
        "measured_od": np.float32,
        "binary_label": np.int8,
        "measurement_coverage": np.float32,
        "semantic_region": np.int16,
        "semantic_support": np.bool_,
    }
    return {
        name: np.asarray(getattr(table, name), dtype=dtypes[name])
        for name in _TABLE_ARRAY_NAMES
    }


def _prediction_arrays(result: ProteinPredictions) -> dict[str, np.ndarray]:
    dtypes = {
        "label_ids": np.uint32,
        "section_ids": np.str_,
        "expression_probability": np.float32,
        "relative_expression": np.float32,
        "predicted_od_reference": np.float32,
        "measured_od": np.float32,
        "uncertainty": np.float32,
        "supported": np.bool_,
    }
    return {
        name: np.asarray(getattr(result, name), dtype=dtypes[name])
        for name in _PREDICTION_ARRAY_NAMES
    }


def _table_fingerprint(table: CellExpressionTable) -> str:
    return _fingerprint(
        b"histopia-cell-expression-v1\0",
        {"target_id": table.target_id, "provenance": table.provenance},
        _table_arrays(table),
    )


def _prediction_fingerprint(result: ProteinPredictions) -> str:
    return _fingerprint(
        b"histopia-protein-predictions-v1\0",
        {
            "target_id": result.target_id,
            "model_fingerprint": result.model_fingerprint,
            "provenance": result.provenance,
        },
        _prediction_arrays(result),
    )


def _fingerprint(
    prefix: bytes, metadata: dict[str, object], arrays: dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(prefix)
    digest.update(_canonical_json(metadata).encode())
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
