"""Broad class probabilities, blinded review selection and supported interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histopia.protein._vector_transfer import _array_hash, _ids
from histopia.study._manifest import fingerprint

BROAD_CLASSES = ("epithelial", "stromal", "immune", "vascular", "uncertain")
CLASS_COLORS = ("#d73027", "#1a9850", "#4575b4", "#984ea3", "#7f8c8d")
QC_CATEGORIES = ("artifact", "necrosis")


@dataclass(frozen=True)
class CellLabelProbabilities:
    cell_ids: np.ndarray
    probabilities: np.ndarray
    support: np.ndarray
    upstream_fingerprint: str
    method: str
    evidence_kind: str = "predicted"
    classes: tuple[str, ...] = BROAD_CLASSES

    def __post_init__(self):
        ids = _ids(self.cell_ids, "cell IDs")
        p, support = np.asarray(self.probabilities, float), np.asarray(self.support)
        if len(set(self.classes)) != len(self.classes) or not self.classes:
            raise ValueError("class IDs must be unique and nonempty")
        if (
            p.shape != (len(ids), len(self.classes))
            or support.shape != ids.shape
            or support.dtype != bool
        ):
            raise ValueError("class probabilities and support must align with cell IDs")
        if (
            not np.all(np.isfinite(p[support]))
            or np.any(p[support] < 0)
            or not np.allclose(p[support].sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError("supported probability rows must be finite and sum to one")
        if not np.all(np.isnan(p[~support])):
            raise ValueError("unsupported label probabilities must remain missing")
        if self.evidence_kind not in {"predicted", "interpolated"}:
            raise ValueError("model output cannot be marked as reference labels")
        if not self.upstream_fingerprint or not self.method:
            raise ValueError("probabilities require upstream provenance and method")
        for name, array in (
            ("cell_ids", ids),
            ("probabilities", p),
            ("support", support),
        ):
            immutable = np.array(array, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "ids": self.cell_ids.tolist(),
                "classes": self.classes,
                "p": _array_hash(self.probabilities),
                "method": self.method,
                "upstream": self.upstream_fingerprint,
                "kind": self.evidence_kind,
            }
        )


def interpolate_label_probabilities(
    labels: CellLabelProbabilities, correspondence
) -> CellLabelProbabilities:
    """Interpolate through an already tissue-gated correspondence, retaining NaNs."""
    if not np.array_equal(labels.cell_ids, correspondence.source_cell_ids):
        raise ValueError("label IDs differ from correspondence source ordering")
    from histopia.protein._vector_transfer import transfer_protein_vectors

    result = transfer_protein_vectors(
        correspondence,
        labels.cell_ids,
        labels.classes,
        labels.probabilities,
        source_fingerprint=labels.fingerprint,
    )
    support = result.support.all(axis=1)
    values = np.where(support[:, None], result.values, np.nan)
    return CellLabelProbabilities(
        correspondence.target_cell_ids,
        values,
        support,
        result.fingerprint,
        "registered-probability-interpolation-v1",
        "interpolated",
        labels.classes,
    )


def annotation_review_sample(
    labels: CellLabelProbabilities,
    mouse_ids,
    roles: dict[str, str],
    *,
    per_class: int = 200,
    minimum_mice: int = 4,
    seed: int = 0,
    blinded: bool = False,
) -> dict[str, object]:
    """Balance development suggestions; blind evaluation uses uniform sampling.

    Evaluation selection never stratifies on model suggestions. No reference
    label is filled in automatically; insufficient mice/classes are reported.
    """
    mice = np.asarray(mouse_ids, str)
    if mice.shape != labels.cell_ids.shape or any(m not in roles for m in mice):
        raise ValueError("known mouse IDs must align with labels")
    if per_class < 1 or minimum_mice < 1:
        raise ValueError("annotation budgets must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    coverage = {}
    if blinded:
        eligible = np.flatnonzero(
            np.isin(mice, [m for m, r in roles.items() if r == "test"])
        )
        chosen = rng.permutation(eligible)[: per_class * len(labels.classes)]
        for i in chosen:
            rows.append(
                {
                    "cell_id": labels.cell_ids[i],
                    "mouse_id": mice[i],
                    "reference_label": "",
                }
            )
    else:
        prediction = np.argmax(np.nan_to_num(labels.probabilities, nan=-1), axis=1)
        for k, name in enumerate(labels.classes):
            eligible = np.flatnonzero(
                labels.support
                & (prediction == k)
                & np.isin(mice, [m for m, r in roles.items() if r == "development"])
            )
            queues = [
                list(rng.permutation(eligible[mice[eligible] == m]))
                for m in sorted(set(mice[eligible]))
            ]
            chosen = []
            while queues and len(chosen) < per_class:
                for queue in queues:
                    if queue and len(chosen) < per_class:
                        chosen.append(queue.pop())
                queues = [queue for queue in queues if queue]
            actual_mice = len(set(mice[chosen]))
            coverage[name] = {
                "cells": len(chosen),
                "mice": actual_mice,
                "budget_met": len(chosen) == per_class and actual_mice >= minimum_mice,
            }
            for i in chosen:
                rows.append(
                    {
                        "cell_id": labels.cell_ids[i],
                        "mouse_id": mice[i],
                        "suggested_class": name,
                        "reference_label": "",
                    }
                )
    return {
        "blinded": blinded,
        "seed": seed,
        "rows": rows,
        "coverage": coverage,
        "reference_status": "pending_independent_lab_review",
    }


def evaluate_cell_labels(
    labels: CellLabelProbabilities, reference: dict[str, object], *, abstain_below=0.6
):
    """Report accuracy, Brier score and class support against reviewed labels."""
    if (
        reference.get("independent_lab_review") is not True
        or reference.get("blinded") is not True
        or not reference.get("review_fingerprint")
    ):
        raise ValueError(
            "accuracy claims require independent blinded lab-reviewed labels"
        )
    ids = _ids(reference["cell_ids"], "reference IDs")
    if not np.array_equal(ids, labels.cell_ids):
        raise ValueError("reference ordering differs from probabilities")
    y = np.asarray(reference["class_ids"], str)
    if y.shape != ids.shape or any(name not in labels.classes for name in y):
        raise ValueError("unknown reference class")
    if not 0 <= abstain_below <= 1:
        raise ValueError("abstention threshold must be in [0, 1]")
    p = labels.probabilities
    called = labels.support & (np.nan_to_num(p, nan=0).max(axis=1) >= abstain_below)
    predicted = np.asarray(labels.classes)[np.nan_to_num(p, nan=-1).argmax(axis=1)]
    actual = np.array([labels.classes.index(name) for name in y])
    onehot = np.eye(len(labels.classes))[actual]
    return {
        "called_cells": int(called.sum()),
        "reference_cells": len(y),
        "abstention_fraction": float(1 - called.mean()) if len(y) else None,
        "accuracy": float(np.mean(predicted[called] == y[called]))
        if called.any()
        else None,
        "brier_score": float(
            np.mean(np.sum((p[labels.support] - onehot[labels.support]) ** 2, axis=1))
        )
        if labels.support.any()
        else None,
        "per_class": {
            name: {
                "reference_count": int(np.sum(y == name)),
                "called_count": int(np.sum(called & (y == name))),
                "accuracy": float(np.mean(predicted[called & (y == name)] == name))
                if np.any(called & (y == name))
                else None,
            }
            for name in labels.classes
        },
    }
