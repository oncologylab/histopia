"""Optional Astir comparator for a verified, sufficiently supported marker panel."""

from __future__ import annotations

import numpy as np

from histopia.protein._vector_transfer import _ids
from histopia.study._labels import BROAD_CLASSES, CellLabelProbabilities
from histopia.study._manifest import fingerprint, validate_fit_scope


def marker_prior_coverage(protein_ids, priors: dict[str, list[str]]) -> dict[str, dict]:
    """Report unavailable broad classes instead of manufacturing missing markers."""
    proteins = set(_ids(protein_ids, "protein IDs"))
    if any(name not in BROAD_CLASSES[:-1] for name in priors):
        raise ValueError("Astir priors must use supported broad class names")
    return {
        name: {
            "available": sorted(set(priors.get(name, [])) & proteins),
            "missing": sorted(set(priors.get(name, [])) - proteins),
            "supported": bool(priors.get(name)) and set(priors[name]) <= proteins,
        }
        for name in BROAD_CLASSES[:-1]
    }


def fit_astir_comparator(
    study,
    cell_ids,
    mouse_ids,
    protein_ids,
    values,
    support,
    *,
    priors: dict[str, list[str]],
    prior_fingerprint: str,
    input_fingerprint: str,
    evidence_kind: str = "measured",
    seed: int = 0,
    max_epochs: int = 50,
) -> tuple[object, CellLabelProbabilities, dict]:
    """Fit a development-only protein comparator, retaining unsupported rows.

    Caller must provide curated marker priors appropriate to species and assay.
    This development fit never supplies reference labels or test accuracy.
    Transferred profiles remain a distinct exploratory input, if explicitly
    requested. Astir's additional ``Other`` class maps to ``uncertain``.
    """
    ids, proteins = _ids(cell_ids, "cell IDs"), _ids(protein_ids, "protein IDs")
    mice, data, mask = (
        np.asarray(mouse_ids, str),
        np.asarray(values, float),
        np.asarray(support),
    )
    if (
        mice.shape != ids.shape
        or data.shape != (len(ids), len(proteins))
        or mask.shape != data.shape
        or mask.dtype != bool
    ):
        raise ValueError("Astir cells, mice, proteins and support must align")
    validate_fit_scope(study, sorted(set(mice)))
    if (
        not prior_fingerprint
        or not input_fingerprint
        or evidence_kind not in {"measured", "directly_predicted", "transferred"}
    ):
        raise ValueError("curated priors and input evidence bindings are required")
    coverage = marker_prior_coverage(proteins, priors)
    supported_classes = [name for name, row in coverage.items() if row["supported"]]
    if len(supported_classes) < 2:
        raise ValueError(
            "Astir needs at least two broad classes with complete marker priors"
        )
    used = sorted({p for name in supported_classes for p in priors[name]})
    columns = [list(proteins).index(p) for p in used]
    eligible = np.all(mask[:, columns] & np.isfinite(data[:, columns]), axis=1)
    if eligible.sum() < 2 or np.any(data[eligible][:, columns] < 0):
        raise ValueError(
            "Astir requires nonnegative supported multi-marker cell profiles"
        )
    try:
        import pandas as pd
        from astir import Astir
    except ImportError as exc:
        raise RuntimeError(
            "Astir comparator requires the optional 'astir' package"
        ) from exc
    frame = pd.DataFrame(data[eligible][:, columns], index=ids[eligible], columns=used)
    model = Astir(
        frame,
        {"cell_types": {name: priors[name] for name in supported_classes}},
        random_seed=seed,
    )
    model.fit_type(max_epochs=max_epochs)
    probabilities = model.get_celltype_probabilities()
    if set(probabilities.index.astype(str)) != set(ids[eligible]):
        raise ValueError("Astir returned different cell identities")
    probabilities = probabilities.loc[ids[eligible]].rename(
        columns={"Other": "uncertain"}
    )
    if set(probabilities.columns) - set(BROAD_CLASSES):
        raise ValueError("Astir returned unknown classes")
    result = np.full((len(ids), len(BROAD_CLASSES)), np.nan)
    result[eligible] = probabilities.reindex(
        columns=BROAD_CLASSES, fill_value=0
    ).to_numpy()
    binding = fingerprint(
        {
            "input": input_fingerprint,
            "priors": prior_fingerprint,
            "marker_priors": priors,
            "mice": sorted(set(mice)),
            "seed": seed,
            "max_epochs": max_epochs,
            "evidence_kind": evidence_kind,
        }
    )
    return (
        model,
        CellLabelProbabilities(
            ids, result, eligible, binding, "astir-protein-comparator"
        ),
        coverage,
    )
