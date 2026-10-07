"""Compact, hash-bound array artifacts for vector, label and neighborhood results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic, write_json_atomic
from histopia.study._manifest import file_sha256, fingerprint


def write_analysis_arrays(
    path: Path | str, arrays: dict[str, np.ndarray], *, metadata: dict[str, object]
) -> Path:
    """Write a portable NPZ and strict manifest, rejecting object/pickle arrays."""
    target = Path(path)
    if (
        target.suffix != ".npz"
        or not arrays
        or not metadata.get("upstream_fingerprints")
    ):
        raise ValueError("NPZ output, arrays and upstream fingerprints are required")
    if metadata.get("coordinate_units") != "um":
        raise ValueError("analysis arrays require explicit um coordinates")
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("object arrays are not portable")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_binary_atomic(target, lambda stream: np.savez_compressed(stream, **arrays))
    core = {
        "schema_version": 1,
        "artifact": target.name,
        "sha256": file_sha256(target),
        "arrays": {
            k: {"shape": list(np.asarray(a).shape), "dtype": np.asarray(a).dtype.str}
            for k, a in arrays.items()
        },
        "metadata": metadata,
    }
    write_json_atomic(
        target.with_suffix(".json"), {**core, "fingerprint": fingerprint(core)}
    )
    return target


def load_analysis_arrays(path: Path | str):
    target = Path(path)
    record = json.loads(target.with_suffix(".json").read_text())
    core = {k: v for k, v in record.items() if k != "fingerprint"}
    if (
        record.get("fingerprint") != fingerprint(core)
        or record.get("artifact") != target.name
        or record.get("sha256") != file_sha256(target)
    ):
        raise ValueError("analysis array binding changed")
    with np.load(target, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if set(arrays) != set(record["arrays"]):
        raise ValueError("analysis array schema changed")
    return arrays, record["metadata"]
