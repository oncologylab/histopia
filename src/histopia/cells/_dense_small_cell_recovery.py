"""Validation of fingerprint-bound dense-small-cell recovery tiles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

_WHOLE_TILE_DIAGNOSTIC_METHOD = "stardist-he-scale2-bounded-territory-v1"
DENSE_SMALL_CELL_RECOVERY_METHOD = "cpsam-stardist-dense-gap-completion-v1"
DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSION = 94
DENSE_SMALL_CELL_RECOVERY_METHOD_PROFILE = (
    "combined-containment-cpsam-stardist-dense-recovery-v1"
)
DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD = (
    "cpsam-stardist-adjacency-dense-gap-completion-v2"
)
DENSE_SMALL_CELL_ADJACENCY_RECOVERY_ALGORITHM_VERSION = 95
DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD_PROFILE = (
    "combined-containment-cpsam-stardist-dense-recovery-v2"
)
DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD = (
    "cpsam-expanded-context-dense-gap-completion-v1"
)
DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_ALGORITHM_VERSION = 99
DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD_PROFILE = (
    "combined-containment-cpsam-expanded-context-dense-recovery-v1"
)
DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD = (
    "cpsam-expanded-context-adjacency-dense-gap-completion-v2"
)
DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_ALGORITHM_VERSION = 100
DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD_PROFILE = (
    "combined-containment-cpsam-expanded-context-dense-recovery-v2"
)
_PRODUCTION_SPECS = {
    DENSE_SMALL_CELL_RECOVERY_METHOD: (
        DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_RECOVERY_METHOD_PROFILE,
    ),
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD: (
        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD_PROFILE,
    ),
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD: (
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD_PROFILE,
    ),
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD: (
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_ALGORITHM_VERSION,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD_PROFILE,
    ),
}
DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS = frozenset(
    version for version, _profile in _PRODUCTION_SPECS.values()
)
DENSE_SMALL_CELL_RECOVERY_PROMOTION_PROFILES = frozenset(_PRODUCTION_SPECS.values())
_SUPPORTED_METHODS = {
    _WHOLE_TILE_DIAGNOSTIC_METHOD,
    DENSE_SMALL_CELL_RECOVERY_METHOD,
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD,
}
_ADDITIVE_PARAMETERS = {
    "schema_version": 2,
    "screening_window_size_px": 1024,
    "screening_bin_size_px": 64,
    "nuclear_minimum_optical_density": 0.12,
    "minimum_global_nuclear_fraction": 0.80,
    "minimum_bin_nuclear_fraction": 0.80,
    "maximum_bin_instance_fraction": 0.40,
    "minimum_gap_bins": 8,
    "recovery_context_bins": 1,
    "maximum_intersection_over_smaller": 0.10,
    "stardist_input_scale": 2.0,
    "stardist_normalize_low_percentile": 1.0,
    "stardist_normalize_high_percentile": 99.8,
    "stardist_probability_threshold": 0.20,
    "stardist_nms_threshold": 0.40,
    "minimum_native_nucleus_pixels": 4,
    "maximum_cell_growth_pixels": 4.0,
}
_ADJACENCY_ADDITIVE_PARAMETERS = {
    **_ADDITIVE_PARAMETERS,
    "schema_version": 3,
    "continuation_policy": "global-gap-component-connected-to-strict-seed-v1",
    "continuation_connectivity": 8,
    "search_policy": "iterative-one-tile-neighborhood-to-fixed-point-v1",
    "recovery_scope": "selected-seed-connected-gap-bins-only",
}
_CPSAM_CONTEXT_ADDITIVE_PARAMETERS = {
    "schema_version": 1,
    "context_pixels_each_side": 256,
    "recovery_bin_size_px": 64,
    "recovery_context_bins": 1,
    "maximum_intersection_over_smaller": 0.10,
    "proposal": "same-cpsam-v2-containment-profile-on-expanded-native-crop",
}
_CPSAM_CONTEXT_ADJACENCY_ADDITIVE_PARAMETERS = {
    **_CPSAM_CONTEXT_ADDITIVE_PARAMETERS,
    "schema_version": 2,
    "continuation_policy": "global-gap-component-connected-to-strict-seed-v1",
    "continuation_connectivity": 8,
    "search_policy": "iterative-one-tile-neighborhood-to-fixed-point-v1",
    "recovery_scope": "selected-seed-connected-gap-bins-only",
}
_ADDITIVE_METHODS = {
    DENSE_SMALL_CELL_RECOVERY_METHOD,
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD,
    DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD,
}
_CACHE_COORDINATES = re.compile(r"_y(?P<y>[0-9]+)_x(?P<x>[0-9]+)\.npz$")


@dataclass(frozen=True, slots=True)
class DenseSmallCellRecoveryTile:
    """One validated replacement for an exact native CPSAM cache tile."""

    key: str
    path: Path
    file_sha256: str
    source_cache_path: Path
    source_cache_sha256: str
    source_tile_fingerprint: str
    recovery_fingerprint: str
    shape: tuple[int, int]

    def load_mask(self) -> np.ndarray:
        """Reload and revalidate a recovery mask immediately before use."""

        if _sha256_file(self.source_cache_path) != self.source_cache_sha256:
            raise ValueError(f"dense recovery source cache changed: {self.key}")
        if _sha256_file(self.path) != self.file_sha256:
            raise ValueError(f"dense recovery tile changed: {self.key}")
        with np.load(self.path, allow_pickle=False) as payload:
            mask = np.asarray(payload["mask"], dtype=np.int32)
            fingerprint = str(payload["fingerprint"].item())
            source_sha256 = str(payload["source_cache_sha256"].item())
        if (
            mask.shape != self.shape
            or fingerprint != self.recovery_fingerprint
            or source_sha256 != self.source_cache_sha256
            or np.any(mask < 0)
        ):
            raise ValueError(f"dense recovery tile payload is stale: {self.key}")
        return mask


@dataclass(frozen=True, slots=True)
class DenseSmallCellRecoveryManifest:
    """Validated, path-safe recovery manifest for one source section."""

    fingerprint: str
    method: str
    section: str
    source_preflight_fingerprint: str
    source_identity: str
    source_section_qc_sha256: str
    source_labels_sha256: str
    model: Mapping[str, object]
    parameters: Mapping[str, object]
    tiles: Mapping[str, DenseSmallCellRecoveryTile]

    def request_payload(self) -> dict[str, object]:
        """Return path-free provenance suitable for a result fingerprint."""

        return {
            "schema_version": 1,
            "method": self.method,
            "manifest_fingerprint": self.fingerprint,
            "source_preflight_fingerprint": self.source_preflight_fingerprint,
            "source_section": self.section,
            "source_identity": self.source_identity,
            "source_section_qc_sha256": self.source_section_qc_sha256,
            "source_labels_sha256": self.source_labels_sha256,
            "model": dict(self.model),
            "parameters": dict(self.parameters),
            "tiles": [
                {
                    "tile": tile.key,
                    "source_cache_sha256": tile.source_cache_sha256,
                    "source_tile_fingerprint": tile.source_tile_fingerprint,
                    "recovery_fingerprint": tile.recovery_fingerprint,
                    "file_sha256": tile.file_sha256,
                    "shape": list(tile.shape),
                }
                for tile in self.tiles.values()
            ],
        }


def load_dense_small_cell_recovery_manifest(
    path: Path | str,
    *,
    source_run: Path | str,
    expected_preflight_fingerprint: str,
    expected_section: str,
    expected_source_identity: str,
) -> DenseSmallCellRecoveryManifest:
    """Validate a generated recovery manifest and every bound tile artifact."""

    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    root = manifest_path.parent.resolve()
    source_root = Path(source_run).expanduser().resolve()
    payload = _object(manifest_path, "dense recovery manifest")
    fingerprint = payload.get("fingerprint")
    if not _is_sha256(fingerprint):
        raise ValueError("dense recovery manifest fingerprint is invalid")
    core = dict(payload)
    core.pop("fingerprint", None)
    core.pop("elapsed_seconds", None)
    if _json_sha256(core) != fingerprint:
        raise ValueError("dense recovery manifest fingerprint is stale")
    method = payload.get("method")
    if (
        payload.get("schema_version") != 1
        or not isinstance(method, str)
        or method not in _SUPPORTED_METHODS
    ):
        raise ValueError("dense recovery manifest method is unsupported")
    section = payload.get("source_section")
    source_identity = payload.get("source_identity")
    preflight_fingerprint = payload.get("source_preflight_fingerprint")
    if (
        section != expected_section
        or source_identity != expected_source_identity
        or preflight_fingerprint != expected_preflight_fingerprint
    ):
        raise ValueError("dense recovery manifest is bound to another source section")
    source_qc_sha256 = payload.get("source_section_qc_sha256")
    source_labels_sha256 = payload.get("source_labels_sha256")
    if not _is_sha256(source_qc_sha256) or not _is_sha256(source_labels_sha256):
        raise ValueError("dense recovery source artifact digests are invalid")
    source_qc = source_root / "qc" / f"{section}.json"
    source_labels = source_root / "labels" / f"{section}.cells.tiff"
    if (
        _sha256_file(source_qc) != source_qc_sha256
        or _sha256_file(source_labels) != source_labels_sha256
    ):
        raise ValueError("dense recovery source section artifacts changed")
    model = payload.get("model")
    parameters = payload.get("parameters")
    rows = payload.get("tiles")
    if not isinstance(model, dict) or not isinstance(parameters, dict):
        raise ValueError("dense recovery model or parameters are invalid")
    expected_parameters = {
        DENSE_SMALL_CELL_RECOVERY_METHOD: dense_small_cell_recovery_parameters(),
        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD: (
            dense_small_cell_adjacency_recovery_parameters()
        ),
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD: (
            dense_small_cell_cpsam_context_recovery_parameters()
        ),
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD: (
            dense_small_cell_cpsam_context_adjacency_recovery_parameters()
        ),
    }.get(method)
    if expected_parameters is not None and parameters != expected_parameters:
        raise ValueError("dense recovery additive parameters are unsupported")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dense recovery manifest contains no tiles")

    tiles: dict[str, DenseSmallCellRecoveryTile] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "dense recovery tile")
        key = row.get("tile")
        relative = row.get("file")
        if (
            not isinstance(key, str)
            or Path(key).name != key
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
        ):
            raise ValueError("dense recovery tile path is unsafe")
        if key in tiles:
            raise ValueError(f"dense recovery tile is duplicated: {key}")
        match = _CACHE_COORDINATES.search(key)
        if match is None:
            raise ValueError(f"dense recovery tile coordinates are malformed: {key}")
        x = _nonnegative_int(row.get("x"), "dense recovery tile x")
        y = _nonnegative_int(row.get("y"), "dense recovery tile y")
        screen_evidence = _mapping(
            row.get("screen_evidence"), "dense recovery tile screen evidence"
        )
        width = _positive_int(
            row.get("width", screen_evidence.get("width")),
            "dense recovery tile width",
        )
        height = _positive_int(
            row.get("height", screen_evidence.get("height")),
            "dense recovery tile height",
        )
        if x != int(match.group("x")) or y != int(match.group("y")):
            raise ValueError(f"dense recovery tile coordinates changed: {key}")
        if (
            row.get("section") != section
            or row.get("source_identity") != source_identity
        ):
            raise ValueError(f"dense recovery tile source binding changed: {key}")
        if row.get("model") != model or row.get("parameters") != parameters:
            raise ValueError(f"dense recovery tile method binding changed: {key}")
        file_sha256 = row.get("file_sha256")
        source_cache_sha256 = row.get("source_cache_sha256")
        source_tile_fingerprint = row.get("source_tile_fingerprint")
        recovery_fingerprint = row.get("recovery_fingerprint")
        if not all(
            _is_sha256(value)
            for value in (
                file_sha256,
                source_cache_sha256,
                source_tile_fingerprint,
                recovery_fingerprint,
            )
        ):
            raise ValueError(f"dense recovery tile digests are invalid: {key}")
        tile_path = (root / relative).resolve()
        if root not in tile_path.parents or not tile_path.is_file():
            raise ValueError(f"dense recovery tile path escaped its manifest: {key}")
        source_cache_path = source_root / ".cell-cache" / str(section) / key
        if _sha256_file(source_cache_path) != source_cache_sha256:
            raise ValueError(f"dense recovery source cache changed: {key}")
        with np.load(source_cache_path, allow_pickle=False) as source_payload:
            source_mask = np.asarray(source_payload["mask"])
            cached_fingerprint = str(source_payload["fingerprint"].item())
        if (
            source_mask.shape != (height, width)
            or cached_fingerprint != source_tile_fingerprint
        ):
            raise ValueError(f"dense recovery source tile payload changed: {key}")
        recovery_core = {
            name: row.get(name)
            for name in (
                "schema_version",
                "source_identity",
                "section",
                "tile",
                "x",
                "y",
                "source_cache_sha256",
                "source_tile_fingerprint",
                "model",
                "parameters",
            )
        }
        if _json_sha256(recovery_core) != recovery_fingerprint:
            raise ValueError(f"dense recovery tile fingerprint is stale: {key}")
        tile = DenseSmallCellRecoveryTile(
            key=key,
            path=tile_path,
            file_sha256=str(file_sha256),
            source_cache_path=source_cache_path,
            source_cache_sha256=str(source_cache_sha256),
            source_tile_fingerprint=str(source_tile_fingerprint),
            recovery_fingerprint=str(recovery_fingerprint),
            shape=(height, width),
        )
        recovered = tile.load_mask()
        _validate_recorded_statistics(
            row,
            recovered,
            key,
            source_mask=source_mask,
            additive=method in _ADDITIVE_METHODS,
        )
        if method == DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD:
            _validate_adjacency_screen_evidence(
                screen_evidence,
                width=width,
                height=height,
                key=key,
            )
        if method == DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD:
            _validate_cpsam_context_screen_evidence(
                screen_evidence,
                x=x,
                y=y,
                width=width,
                height=height,
                key=key,
            )
        if method == DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD:
            _validate_cpsam_context_adjacency_screen_evidence(
                screen_evidence,
                x=x,
                y=y,
                width=width,
                height=height,
                key=key,
            )
        tiles[key] = tile
    if method in {
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD,
    }:
        validate_dense_small_cell_recovery_model(method, model)
    return DenseSmallCellRecoveryManifest(
        fingerprint=str(fingerprint),
        method=str(method),
        section=str(section),
        source_preflight_fingerprint=str(preflight_fingerprint),
        source_identity=str(source_identity),
        source_section_qc_sha256=str(source_qc_sha256),
        source_labels_sha256=str(source_labels_sha256),
        model=MappingProxyType(dict(model)),
        parameters=MappingProxyType(dict(parameters)),
        tiles=MappingProxyType(tiles),
    )


def dense_small_cell_recovery_manifest_profile(
    path: Path | str,
) -> tuple[int, str]:
    """Return the fingerprint-validated production profile for a manifest.

    Profile selection happens before the full source-bound manifest validation
    because it is part of the result fingerprint.  Requiring the manifest seal
    here prevents a changed method string from silently selecting another
    algorithm version.
    """

    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    payload = _object(manifest_path, "dense recovery manifest")
    fingerprint = payload.get("fingerprint")
    if not _is_sha256(fingerprint):
        raise ValueError("dense recovery manifest fingerprint is invalid")
    core = dict(payload)
    core.pop("fingerprint", None)
    core.pop("elapsed_seconds", None)
    if _json_sha256(core) != fingerprint:
        raise ValueError("dense recovery manifest fingerprint is stale")
    if payload.get("schema_version") != 1:
        raise ValueError("dense recovery manifest method is unsupported")
    return dense_small_cell_recovery_profile(payload.get("method"))


def dense_small_cell_recovery_profile(method: object) -> tuple[int, str]:
    """Return the exact algorithm/profile pair for a production method."""

    if not isinstance(method, str) or method not in _PRODUCTION_SPECS:
        raise ValueError("dense recovery manifest method is not promotable")
    return _PRODUCTION_SPECS[method]


def dense_small_cell_recovery_parameters_for_method(
    method: object,
) -> dict[str, object]:
    """Return the immutable parameter payload coupled to a production method."""

    dense_small_cell_recovery_profile(method)
    if method == DENSE_SMALL_CELL_RECOVERY_METHOD:
        return dense_small_cell_recovery_parameters()
    if method == DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD:
        return dense_small_cell_adjacency_recovery_parameters()
    if method == DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD:
        return dense_small_cell_cpsam_context_recovery_parameters()
    return dense_small_cell_cpsam_context_adjacency_recovery_parameters()


def _validate_recorded_statistics(
    row: Mapping[str, object],
    mask: np.ndarray,
    key: str,
    *,
    source_mask: np.ndarray,
    additive: bool,
) -> None:
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if not ids.size:
        raise ValueError(f"dense recovery tile contains no cells: {key}")
    areas = np.bincount(mask.ravel())[ids]
    recorded_quantiles = row.get("area_px_quantiles")
    expected_quantiles = [
        round(float(value), 3) for value in np.quantile(areas, [0.05, 0.5, 0.95])
    ]
    nucleus_count = row.get("nucleus_count")
    if (
        row.get("cell_count") != int(ids.size)
        or not isinstance(nucleus_count, int)
        or isinstance(nucleus_count, bool)
        or nucleus_count <= 0
        or row.get("foreground_fraction") != float(np.mean(mask > 0))
        or recorded_quantiles != expected_quantiles
    ):
        raise ValueError(f"dense recovery tile statistics are stale: {key}")
    if additive:
        source_ids = np.unique(source_mask)
        source_count = int(np.count_nonzero(source_ids > 0))
        if (
            row.get("source_cell_count") != source_count
            or row.get("added_cell_count") != int(ids.size) - source_count
            or int(ids.size) <= source_count
            or np.any((source_mask > 0) & (mask == 0))
        ):
            raise ValueError(f"dense recovery additive statistics are stale: {key}")


def dense_small_cell_recovery_parameters() -> dict[str, object]:
    """Return the exact immutable additive recovery parameter payload."""

    return dict(_ADDITIVE_PARAMETERS)


def dense_small_cell_adjacency_recovery_parameters() -> dict[str, object]:
    """Return the exact seed-connected adjacency recovery parameter payload."""

    return dict(_ADJACENCY_ADDITIVE_PARAMETERS)


def dense_small_cell_cpsam_context_recovery_parameters() -> dict[str, object]:
    """Return the immutable Cellpose-only expanded-context recovery policy."""

    return dict(_CPSAM_CONTEXT_ADDITIVE_PARAMETERS)


def dense_small_cell_cpsam_context_adjacency_recovery_parameters() -> dict[str, object]:
    """Return the immutable seed-connected expanded-context CPSAM policy."""

    return dict(_CPSAM_CONTEXT_ADJACENCY_ADDITIVE_PARAMETERS)


def validate_dense_small_cell_recovery_model(
    method: object,
    value: object,
) -> None:
    """Validate the neural-model identity appropriate to a recovery method."""

    model = _mapping(value, "dense recovery model")
    if method in {
        DENSE_SMALL_CELL_CPSAM_CONTEXT_RECOVERY_METHOD,
        DENSE_SMALL_CELL_CPSAM_CONTEXT_ADJACENCY_RECOVERY_METHOD,
    }:
        device = _mapping(model.get("device"), "CPSAM recovery device")
        expected = {
            "cellpose_version": "4.2.1",
            "weight_name": "cpsam_v2",
            "weight_sha256": (
                "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
            ),
            "name": "cpsam_v2",
            "method": "containment",
            "first_cellprob": -2.5,
            "first_diameter": 26,
            "second_cellprob": -1.75,
            "second_diameter": 15,
            "flow_threshold": 0.0,
            "merge_threshold": 0.1,
            "minimum_size_px": 15,
        }
        if (
            any(
                model.get(key) != expected_value
                for key, expected_value in expected.items()
            )
            or device.get("backend") != "cuda"
            or device.get("resolved") != "cuda:0"
            or not isinstance(device.get("accelerator_name"), str)
            or not str(device["accelerator_name"]).strip()
        ):
            raise ValueError("CPSAM expanded-context model provenance is stale")
        return

    files = model.get("files")
    if (
        model.get("name") not in {"2D_versatile_he", "stardist-2D_versatile_he"}
        or model.get("stardist_version") != "0.9.1"
        or not isinstance(model.get("tensorflow_version"), str)
        or not model["tensorflow_version"]
        or not isinstance(files, dict)
        or set(files) != {"config.json", "thresholds.json", "weights_best.h5"}
        or any(not _is_sha256(digest) for digest in files.values())
    ):
        raise ValueError("StarDist dense-recovery model provenance is stale")


def _validate_cpsam_context_screen_evidence(
    evidence: Mapping[str, object],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    key: str,
) -> None:
    """Require the exact dense-gap screen that selected each CPSAM tile."""

    gap_count = evidence.get("gap_bin_count")
    nuclear_fraction = evidence.get("global_nuclear_fraction")
    gap_fraction = evidence.get("gap_bin_fraction")
    bin_size = int(_CPSAM_CONTEXT_ADDITIVE_PARAMETERS["recovery_bin_size_px"])
    bins = ((height + bin_size - 1) // bin_size) * ((width + bin_size - 1) // bin_size)
    if (
        evidence.get("tile") != key
        or evidence.get("x") != x
        or evidence.get("y") != y
        or evidence.get("width") != width
        or evidence.get("height") != height
        or not isinstance(gap_count, int)
        or isinstance(gap_count, bool)
        or gap_count < 8
        or gap_count > bins
        or not isinstance(nuclear_fraction, (int, float))
        or isinstance(nuclear_fraction, bool)
        or not 0.80 <= float(nuclear_fraction) <= 1.0
        or not isinstance(gap_fraction, (int, float))
        or isinstance(gap_fraction, bool)
        or float(gap_fraction) != round(gap_count / bins, 9)
    ):
        raise ValueError(f"CPSAM expanded-context screen evidence is stale: {key}")


def _validate_cpsam_context_adjacency_screen_evidence(
    evidence: Mapping[str, object],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    key: str,
) -> None:
    """Require an exact seed-connected bin scope for an expanded CPSAM tile."""

    if (
        evidence.get("tile") != key
        or evidence.get("x") != x
        or evidence.get("y") != y
        or evidence.get("width") != width
        or evidence.get("height") != height
    ):
        raise ValueError(f"CPSAM expanded-context adjacency geometry is stale: {key}")
    _validate_adjacency_screen_evidence(
        evidence,
        width=width,
        height=height,
        key=key,
    )


def _validate_adjacency_screen_evidence(
    evidence: Mapping[str, object],
    *,
    width: int,
    height: int,
    key: str,
) -> None:
    """Validate the exact local bin scope selected by the global screen."""

    role = evidence.get("selection_role")
    candidate_count = evidence.get("candidate_gap_bin_count")
    selected_count = evidence.get("selected_gap_bin_count")
    selected_fraction = evidence.get("selected_gap_bin_fraction")
    global_nuclear_fraction = evidence.get("global_nuclear_fraction")
    encoded = evidence.get("selected_gap_bins")
    if (
        role not in {"strict_seed", "connected_continuation"}
        or not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count <= 0
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count <= 0
        or int(selected_count) > int(candidate_count)
        or not isinstance(selected_fraction, (int, float))
        or isinstance(selected_fraction, bool)
        or not np.isfinite(selected_fraction)
        or not isinstance(global_nuclear_fraction, (int, float))
        or isinstance(global_nuclear_fraction, bool)
        or not np.isfinite(global_nuclear_fraction)
        or not 0 <= float(global_nuclear_fraction) <= 1
        or not isinstance(encoded, list)
        or len(encoded) != selected_count
    ):
        raise ValueError(f"dense recovery adjacency evidence is invalid: {key}")
    bin_size = int(_ADJACENCY_ADDITIVE_PARAMETERS["screening_bin_size_px"])
    rows = (height + bin_size - 1) // bin_size
    columns = (width + bin_size - 1) // bin_size
    if int(candidate_count) > rows * columns or float(selected_fraction) != (
        int(selected_count) / (rows * columns)
    ):
        raise ValueError(f"dense recovery adjacency bin statistics are stale: {key}")
    if role == "strict_seed" and float(global_nuclear_fraction) < float(
        _ADJACENCY_ADDITIVE_PARAMETERS["minimum_global_nuclear_fraction"]
    ):
        raise ValueError(f"dense recovery strict seed is no longer eligible: {key}")
    coordinates: set[tuple[int, int]] = set()
    for value in encoded:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                not isinstance(index, int) or isinstance(index, bool) for index in value
            )
        ):
            raise ValueError(f"dense recovery adjacency bin is malformed: {key}")
        row, column = value
        if not 0 <= row < rows or not 0 <= column < columns:
            raise ValueError(f"dense recovery adjacency bin is out of bounds: {key}")
        coordinates.add((row, column))
    if len(coordinates) != len(encoded):
        raise ValueError(f"dense recovery adjacency bins are duplicated: {key}")


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except (FileNotFoundError, OSError) as error:
        message = f"required recovery artifact is unreadable: {path.name}"
        raise ValueError(message) from error
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
