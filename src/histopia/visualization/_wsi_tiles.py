"""Approval-bound, on-demand tiles for local whole-slide review."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from io import BytesIO
from pathlib import Path
from typing import Any

from histopia.registration._approval import validate_registration_approval
from histopia.registration._errors import OptionalDependencyError
from histopia.visualization._cell_scope import CellSectionScope

_SECTION_RE = re.compile(r"[0-9]{3,6}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_STAIN_LAYERS = frozenset(
    {
        "stain_raw",
        "stain_corrected",
        "stain_adaptive",
        "stain_adaptive_map",
        "stain_adaptive_v3",
        "stain_adaptive_v3_map",
        "stain_output",
        "stain_output_map",
        "stain_counterstain",
        "stain_residual",
        "stain_support",
    }
)
_PROTEIN_LAYERS = frozenset(
    {
        "protein_predicted",
        "protein_predicted_contrast",
        "protein_dense",
        "protein_dense_contrast",
        "protein_probability",
        "protein_uncertainty",
        "protein_measured",
        "protein_measured_contrast",
        "protein_residual",
    }
)
_PROTEIN_OBSERVED_LAYERS = frozenset(
    {"protein_observed_target", "protein_nearest_observed_target"}
)
_PROTEIN_SUPPORT_LAYERS = frozenset({"protein_tissue_support"})
_PNG_LAYERS = frozenset(
    {
        "mask",
        "cells",
        *_STAIN_LAYERS,
        *_PROTEIN_LAYERS,
        *_PROTEIN_OBSERVED_LAYERS,
        *_PROTEIN_SUPPORT_LAYERS,
    }
)
_LAYERS = frozenset({"raw", "registered", *_PNG_LAYERS})


def _select_protein_current_layer(
    layers: Mapping[str, WsiLayer],
    *,
    counterstain_v3: bool,
) -> tuple[str, str]:
    """Select honest current-section context for an unmeasured target.

    Quantified OD remains preferred when it exists. A raw native slide is the
    scientifically useful fallback: it shows the section's actual stain without
    presenting it as target-protein truth. Neutral tissue support is retained
    only for registries that genuinely lack native imagery.
    """

    if counterstain_v3 and "stain_adaptive_v3_map" in layers:
        return (
            "stain_adaptive_v3_map",
            "current_marker_counterstain_conditioned_od_4um",
        )
    if "stain_adaptive_map" in layers:
        return "stain_adaptive_map", "current_marker_adaptive_od_4um"
    if "raw" in layers:
        return "raw", "native_histology_context_only"
    if "protein_tissue_support" in layers:
        return "protein_tissue_support", "no_quantified_stain"
    if "mask" in layers:
        return "mask", "no_quantified_stain"
    raise ValueError("Protein review section has no current-slide context layer")


def _cached_wsi_layer(
    cache: dict[tuple[object, ...], WsiLayer] | None,
    key: tuple[object, ...],
    builder: Callable[[], WsiLayer],
) -> WsiLayer:
    """Reuse immutable base-layer geometry during one registry build."""

    if cache is None:
        return builder()
    loaded = cache.get(key)
    if loaded is None:
        loaded = builder()
        cache[key] = loaded
    return loaded


@dataclass(frozen=True, slots=True)
class WsiLevel:
    """One native pyramid level, ordered from smallest to largest."""

    width: int
    height: int
    source_level: int
    source_kind: str = "page"


@dataclass(frozen=True, slots=True)
class WsiLayer:
    """One explicitly registered image source."""

    name: str
    path: Path
    digest: str
    levels: tuple[WsiLevel, ...]
    tile_size: int
    microns_per_pixel: float | None
    mask_path: Path | None = None
    crop_bbox_xywh: tuple[int, int, int, int] | None = None
    source_shape: tuple[int, int] | None = None
    analysis_mpp: float | None = None
    display_max_od: float | None = None
    section_display_max_od: float | None = None
    map_kind: str | None = None
    content_origin_native_xy: tuple[int, int] | None = None
    render_mode: str | None = None
    selected_source: str | None = None
    stain_family: str | None = None
    adaptive_floor_od: float | None = None
    adaptive_method: str | None = None
    stain_artifact_kind: str = "physical"
    label_path: Path | None = None
    protein_target: str | None = None
    protein_model_fingerprint: str | None = None
    prediction_section: str | None = None
    prediction_artifact_digest: str | None = None
    prediction_fingerprint: str | None = None
    label_artifact_digest: str | None = None
    stain_artifact_digest: str | None = None
    comparison_source_section: str | None = None
    comparison_source_label: str | None = None
    comparison_target_section: str | None = None
    target_map_to_source_map: tuple[tuple[float, float, float], ...] | None = None
    calibration_source_knots: tuple[float, ...] | None = None
    calibration_reference_knots: tuple[float, ...] | None = None
    calibration_fingerprint: str | None = None
    focus_bbox_fraction: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class WsiSection:
    """Path-free public identity plus private image sources for one section."""

    cohort: str
    section: str
    slide: str
    label: str
    reference: bool
    layers: dict[str, WsiLayer]
    protein_comparison: dict[str, object] | None = None
    focus_bbox_fraction: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class _StainMapSpec:
    width: int
    height: int
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True, slots=True)
class _StainMapHeader:
    slide_id: str
    shape: tuple[int, int]
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float]
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class _AdaptiveStainMapHeader:
    slide_id: str
    shape: tuple[int, int]
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float]
    source_content_fingerprint: str
    method: str
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ProteinOverview:
    """Bounded cell-resolved raster used only below resolvable cell scale."""

    values: object
    supported: object


def _protein_target_label(
    target_id: str,
    declared_label: object | None = None,
) -> str:
    if (
        isinstance(declared_label, str)
        and declared_label.strip()
        and len(declared_label.strip()) <= 80
        and all(ord(character) >= 32 for character in declared_label.strip())
    ):
        return declared_label.strip()
    return {
        "yap": "YAP",
        "ecad": "E-Cad",
        "ck19": "CK19",
        "ki67": "Ki67",
        "perk": "pERK",
        "ncad": "N-Cad",
        "cjun": "cJun",
    }.get(target_id.lower(), target_id.upper())


def _normalized_protein_model_label(
    label: str,
    target_id: str,
    target_label: str | None = None,
) -> str:
    display = _protein_target_label(target_id, target_label)
    raw_prefix = target_id.upper()
    return display + label[len(raw_prefix) :] if label.startswith(raw_prefix) else label


def _protein_model_summary(
    model_id: str,
    path: Path,
    *,
    validated_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a path-free identity for one sealed model-scoped result."""

    from histopia.protein._manifest import validate_protein_result

    root = Path(path).expanduser().resolve()
    result = (
        validate_protein_result(root) if validated_result is None else validated_result
    )
    declared = result.get("model_id")
    if declared is not None and declared != model_id:
        raise ValueError(
            f"configured protein model {model_id!r} differs from its sealed model ID"
        )
    metrics = result.get("metrics")
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    architecture = result.get("architecture") or metrics.get("deployed_candidate")
    if not isinstance(architecture, str) or not architecture:
        architecture = "legacy"
    training_cohorts = result.get("training_cohorts")
    if not isinstance(training_cohorts, list) or not all(
        isinstance(value, str) and value for value in training_cohorts
    ):
        training_cohorts = []
    try:
        approval = json.loads((root / "protein_approval.json").read_text())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        approval = {}
    approved = bool(
        approval.get("accepted") is True
        and approval.get("fingerprint") == result.get("fingerprint")
    )
    target_id = str(result.get("target_id", ""))
    target_label = _protein_target_label(target_id, result.get("target_label"))
    declared_label = str(
        result.get("model_label")
        or f"{target_label} · {architecture.replace('_', ' ')}"
    )
    return {
        "id": model_id,
        "label": _normalized_protein_model_label(
            declared_label,
            target_id,
            target_label,
        ),
        "target_id": target_id,
        "target_label": target_label,
        "architecture": architecture,
        "training_cohorts": training_cohorts,
        "version": result.get("model_version") or result.get("schema_version"),
        "prediction_protocol": result.get("prediction_protocol", "leave-one-mouse-out"),
        "feature_schema_id": result.get("feature_schema_id"),
        "model_fingerprint": result.get("model_fingerprint"),
        "result_fingerprint": result.get("fingerprint"),
        "measurement_view": result.get("measurement_view"),
        "measurement_statistic": result.get("measurement_statistic", "mean"),
        "metrics": metrics,
        "approved": approved,
        "promoted": bool(result.get("status") == "promoted" and approved),
        "private_path": str(root),
    }


def _select_default_protein_model(
    summaries: list[dict[str, object]],
) -> str | None:
    """Prefer reviewed models with the strongest held-out spatial agreement."""

    if not summaries:
        return None

    def key(row: dict[str, object]) -> tuple[object, ...]:
        metrics = row.get("metrics")
        values = metrics if isinstance(metrics, dict) else {}

        def descending(name: str) -> float:
            value = values.get(name)
            return (
                -float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                else math.inf
            )

        return (
            not bool(row.get("promoted")),
            not bool(row.get("approved")),
            descending("spearman_64um"),
            descending("spearman"),
            str(row.get("id", "")),
        )

    return str(min(summaries, key=key)["id"])


def _select_recommended_protein_models(
    summaries: list[dict[str, object]],
) -> dict[str, dict[str, str]]:
    """Choose approved measured-fit and unmeasured-transfer defaults."""

    output: dict[str, dict[str, str]] = {}
    targets = sorted({str(row.get("target_id", "")) for row in summaries})
    for target in targets:
        rows = [
            row
            for row in summaries
            if row.get("target_id") == target and bool(row.get("approved"))
        ]
        transfer = [
            row for row in rows if row.get("prediction_protocol") != "training-visible"
        ]
        measured = [
            row for row in rows if row.get("prediction_protocol") == "training-visible"
        ]
        transfer_id = _select_default_protein_model(transfer)
        measured_id = _select_default_protein_model(measured or transfer)
        selected = {
            role: model_id
            for role, model_id in (
                ("measured", measured_id),
                ("unmeasured", transfer_id),
            )
            if model_id is not None
        }
        if selected:
            output[target] = selected
    return output


class WsiTileService:
    """Serve immutable tiles from an explicit, approval-bound cohort registry."""

    def __init__(
        self,
        sections: dict[tuple[str, str], WsiSection],
        *,
        protein_sections: dict[tuple[str, str, str], WsiSection] | None = None,
        protein_models: dict[tuple[str, str], dict[str, object]] | None = None,
        default_protein_models: dict[str, str] | None = None,
        recommended_protein_models: dict[str, dict[str, dict[str, str]]] | None = None,
        max_concurrent_tiles: int = 24,
    ) -> None:
        if max_concurrent_tiles <= 0:
            raise ValueError("max_concurrent_tiles must be positive")
        self._sections = dict(sections)
        self._protein_sections = dict(protein_sections or {})
        self._protein_models = dict(protein_models or {})
        self._default_protein_models = dict(default_protein_models or {})
        self._recommended_protein_models = dict(recommended_protein_models or {})
        self._capacity = threading.BoundedSemaphore(max_concurrent_tiles)
        self._stain_maps: OrderedDict[Path, object] = OrderedDict()
        self._stain_cache_lock = threading.Lock()
        self._validated_cell_artifacts: set[tuple[Path, str]] = set()
        self._cell_artifact_lock = threading.Lock()
        self._protein_predictions: OrderedDict[Path, object] = OrderedDict()
        self._protein_overviews: OrderedDict[tuple[Path, str], _ProteinOverview] = (
            OrderedDict()
        )
        self._protein_label_overviews: OrderedDict[
            tuple[Path, tuple[int, int], float], object
        ] = OrderedDict()
        self._protein_cache_lock = threading.Lock()

    @classmethod
    def from_runs(
        cls,
        runs: dict[str, tuple[Path, Path | None]],
        *,
        cell_runs: dict[str, Path] | None = None,
        cell_section_scopes: dict[str, CellSectionScope] | None = None,
        stain_runs: dict[str, Path] | None = None,
        protein_runs: dict[str, Path] | None = None,
        protein_model_runs: dict[str, dict[str, Path]] | None = None,
        protein_model_stain_runs: dict[str, dict[str, Path]] | None = None,
        max_concurrent_tiles: int = 24,
    ) -> WsiTileService:
        """Build native catalogs from explicitly configured approved runs."""

        sections: dict[tuple[str, str], WsiSection] = {}
        cell_runs = cell_runs or {}
        cell_section_scopes = cell_section_scopes or {}
        if set(cell_section_scopes) - set(cell_runs):
            raise ValueError("cell scope has no matching cell run")
        stain_runs = stain_runs or {}
        protein_runs = protein_runs or {}
        protein_model_runs = protein_model_runs or {}
        protein_model_stain_runs = protein_model_stain_runs or {}
        unknown = (
            set(cell_runs)
            | set(stain_runs)
            | set(protein_runs)
            | set(protein_model_runs)
            | set(protein_model_stain_runs)
        ) - set(runs)
        if unknown:
            raise ValueError(
                "tile inputs have no matching registration: "
                + ", ".join(sorted(unknown))
            )
        protein_sections: dict[tuple[str, str, str], WsiSection] = {}
        protein_models: dict[tuple[str, str], dict[str, object]] = {}
        default_protein_models: dict[str, str] = {}
        recommended_protein_models: dict[str, dict[str, dict[str, str]]] = {}
        validated_protein_results: dict[Path, dict[str, object]] = {}
        validated_cell_results: dict[Path, dict[str, Any]] = {}
        validated_stain_maps: dict[
            tuple[Path, Path, str],
            tuple[dict[str, tuple[dict[str, Any], _StainMapSpec]], float],
        ] = {}
        base_layer_cache: dict[tuple[object, ...], WsiLayer] = {}

        def validated_protein_result(path: Path) -> dict[str, object]:
            from histopia.protein._manifest import validate_protein_result_index

            root = Path(path).expanduser().resolve()
            if root not in validated_protein_results:
                validated_protein_results[root] = validate_protein_result_index(root)
            return validated_protein_results[root]

        for cohort, (registration, registered_wsi) in sorted(runs.items()):
            scoped = protein_model_runs.get(cohort, {})
            if not isinstance(scoped, dict):
                raise TypeError("protein model runs must be grouped by model ID")
            scoped_stains = protein_model_stain_runs.get(cohort, {})
            if not isinstance(scoped_stains, dict):
                raise TypeError("protein model stain runs must be grouped by model ID")
            for model_id in scoped:
                if not isinstance(model_id, str) or not _MODEL_RE.fullmatch(model_id):
                    raise ValueError(f"invalid protein model ID: {model_id!r}")
            unknown_stain_models = set(scoped_stains) - set(scoped)
            if unknown_stain_models:
                raise ValueError(
                    "protein model stain inputs have no matching model: "
                    + ", ".join(sorted(unknown_stain_models))
                )
            legacy = protein_runs.get(cohort)
            summaries = []
            scoped_results: dict[str, dict[str, object]] = {}
            for model_id, path in sorted(scoped.items()):
                result = validated_protein_result(path)
                scoped_results[model_id] = result
                summaries.append(
                    _protein_model_summary(
                        model_id,
                        path,
                        validated_result=result,
                    )
                )
            summaries.sort(
                key=lambda row: (
                    not bool(row.get("promoted")),
                    not bool(row.get("approved")),
                    str(row["id"]),
                )
            )
            seen_fingerprints: set[str] = set()
            deduplicated: list[dict[str, object]] = []
            for summary in summaries:
                fingerprint = str(summary["result_fingerprint"])
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                deduplicated.append(summary)
            scoped = {
                str(summary["id"]): Path(str(summary["private_path"]))
                for summary in deduplicated
            }
            for summary in deduplicated:
                summary.pop("private_path", None)
            default_model = _select_default_protein_model(deduplicated)
            recommended_protein_models[cohort] = _select_recommended_protein_models(
                deduplicated
            )
            default_run = scoped.get(default_model) if default_model else legacy
            if default_run is None:
                default_run = legacy
            default_stain_run = (
                scoped_stains.get(default_model, stain_runs.get(cohort))
                if default_model is not None
                else stain_runs.get(cohort)
            )
            default_result = (
                scoped_results.get(default_model)
                if default_model is not None
                else validated_protein_result(legacy)
                if legacy is not None
                else None
            )
            loaded_default = _load_cohort_sections(
                cohort,
                registration,
                registered_wsi,
                cell_runs.get(cohort),
                default_stain_run,
                default_run,
                protein_result=default_result,
                cell_result_cache=validated_cell_results,
                cell_section_scope=cell_section_scopes.get(cohort),
                stain_map_cache=validated_stain_maps,
                base_layer_cache=base_layer_cache,
            )
            for section in loaded_default:
                key = (cohort, section.section)
                if key in sections:
                    raise ValueError(
                        f"duplicate WSI section: {cohort}/{section.section}"
                    )
                sections[key] = section
            if default_model is not None:
                default_protein_models[cohort] = default_model
            default_by_section = {row.section: row for row in loaded_default}
            for summary in deduplicated:
                model_id = str(summary["id"])
                model_run = scoped[model_id]
                model_stain_run = scoped_stains.get(
                    model_id,
                    stain_runs.get(cohort),
                )
                loaded = (
                    loaded_default
                    if model_run == default_run and model_stain_run == default_stain_run
                    else _load_cohort_sections(
                        cohort,
                        registration,
                        registered_wsi,
                        cell_runs.get(cohort),
                        model_stain_run,
                        model_run,
                        protein_result=scoped_results[model_id],
                        cell_result_cache=validated_cell_results,
                        cell_section_scope=cell_section_scopes.get(cohort),
                        stain_map_cache=validated_stain_maps,
                        base_layer_cache=base_layer_cache,
                    )
                )
                available_sections: list[str] = []
                for section in loaded:
                    if section.section not in default_by_section:
                        raise ValueError("protein model section is not registered")
                    protein_sections[(cohort, model_id, section.section)] = section
                    available_sections.append(section.section)
                public_summary = {
                    **summary,
                    "sections": available_sections,
                }
                protein_models[(cohort, model_id)] = public_summary
        if not sections:
            raise ValueError("WSI registry contains no exported sections")
        return cls(
            sections,
            protein_sections=protein_sections,
            protein_models=protein_models,
            default_protein_models=default_protein_models,
            recommended_protein_models=recommended_protein_models,
            max_concurrent_tiles=max_concurrent_tiles,
        )

    def metadata(
        self,
        cohort: str,
        section: str,
        protein_model: str | None = None,
    ) -> dict[str, object]:
        """Return path-free metadata for one section."""

        item = self._section(cohort, section, protein_model=protein_model)
        payload: dict[str, object] = {
            "schema_version": 1,
            "cohort": item.cohort,
            "section": item.section,
            "slide": item.slide,
            "label": item.label,
            "reference": item.reference,
            "layers": {
                name: _public_layer_metadata(name, layer)
                for name, layer in sorted(item.layers.items())
            },
        }
        if item.focus_bbox_fraction is not None:
            x, y, width, height = item.focus_bbox_fraction
            payload["focus_bbox"] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "coordinate_space": "normalized_native_content_bbox",
            }
        if item.protein_comparison is not None:
            payload["protein_comparison"] = item.protein_comparison
        models = self._public_protein_models(cohort)
        if models:
            active = protein_model or self._default_protein_models.get(cohort)
            payload["protein_models"] = models
            payload["protein_model_id"] = active
            payload["recommended_protein_models"] = (
                self._recommended_protein_models.get(cohort, {})
            )
        return payload

    def section(self, cohort: str, section: str) -> WsiSection:
        """Return an explicitly configured section for trusted local exporters."""

        return self._section(cohort, section)

    def sections(self, cohort: str) -> tuple[str, ...]:
        """Return sorted configured section IDs for one cohort."""

        return tuple(
            section
            for item_cohort, section in sorted(self._sections)
            if item_cohort == cohort
        )

    def catalog(self, cohort: str) -> dict[str, object]:
        """Return path-free available-section metadata for one cohort."""

        rows = [
            {
                "section": section,
                "slide": item.slide,
                "label": item.label,
                "reference": item.reference,
                "layers": sorted(item.layers),
            }
            for (item_cohort, section), item in sorted(self._sections.items())
            if item_cohort == cohort
        ]
        if not rows:
            raise FileNotFoundError("unknown WSI cohort")
        payload: dict[str, object] = {
            "schema_version": 1,
            "cohort": cohort,
            "sections": rows,
        }
        models = self._public_protein_models(cohort)
        if models:
            payload["protein_models"] = models
            payload["default_protein_model_id"] = self._default_protein_models.get(
                cohort
            )
            payload["recommended_protein_models"] = (
                self._recommended_protein_models.get(cohort, {})
            )
        return payload

    def render_tile(
        self,
        cohort: str,
        section: str,
        layer_name: str,
        digest: str,
        level: int,
        x: int,
        y: int,
        *,
        protein_model: str | None = None,
    ) -> tuple[bytes, str, str]:
        """Render one bounded tile and return bytes, media type, and ETag."""

        item = self._section(cohort, section, protein_model=protein_model)
        if layer_name not in _LAYERS or layer_name not in item.layers:
            raise FileNotFoundError("unknown WSI layer")
        layer = item.layers[layer_name]
        if not _DIGEST_RE.fullmatch(digest) or digest != layer.digest:
            raise FileNotFoundError("stale WSI layer")
        if level < 0 or level >= len(layer.levels) or x < 0 or y < 0:
            raise FileNotFoundError("invalid WSI tile coordinates")
        dimensions = layer.levels[level]
        columns = math.ceil(dimensions.width / layer.tile_size)
        rows = math.ceil(dimensions.height / layer.tile_size)
        if x >= columns or y >= rows:
            raise FileNotFoundError("invalid WSI tile coordinates")
        if not self._capacity.acquire(blocking=False):
            raise WsiTileCapacityError("WSI tile capacity is full")
        try:
            if layer_name in _STAIN_LAYERS:
                with self._stain_cache_lock:
                    stain_map = self._load_stain_map(layer)
                payload = _render_stain_layer_tile(
                    layer,
                    level,
                    x,
                    y,
                    stain_map,
                )
            elif layer_name in _PROTEIN_OBSERVED_LAYERS:
                with self._stain_cache_lock:
                    stain_map = self._load_stain_map(layer)
                payload = _render_observed_target_tile(
                    layer,
                    level,
                    x,
                    y,
                    stain_map,
                )
            elif layer_name in _PROTEIN_LAYERS:
                self._validate_cell_labels(layer)
                with self._protein_cache_lock:
                    predictions = self._load_protein_predictions(layer)
                    overview = (
                        self._load_protein_overview(layer, predictions)
                        if layer.mask_path is not None and dimensions.width < 2_048
                        else None
                    )
                payload = _render_protein_layer_tile(
                    layer,
                    level,
                    x,
                    y,
                    predictions,
                    overview=overview,
                )
            else:
                if layer_name == "cells":
                    self._validate_cell_labels(layer)
                payload = _render_layer_tile(layer, level, x, y)
        finally:
            self._capacity.release()
        media_type = "image/png" if layer_name in _PNG_LAYERS else "image/jpeg"
        etag = f'"{digest}-{level}-{x}-{y}"'
        return payload, media_type, etag

    def _load_stain_map(self, layer: WsiLayer) -> object:
        """Load a sealed map while retaining current and nearest slides."""

        key = layer.path
        cached = self._stain_maps.get(key)
        if cached is not None:
            self._stain_maps.move_to_end(key)
            return cached
        from histopia.stain._result_validation import validate_stain_artifact

        if layer.stain_artifact_digest is None:
            raise ValueError("stain layer has no sealed artifact digest")
        validate_stain_artifact(layer.path, layer.stain_artifact_digest)
        from histopia.stain._artifacts import AdaptiveStainMap, StainMap

        loaded = (
            AdaptiveStainMap.load(layer.path)
            if layer.stain_artifact_kind == "adaptive"
            else StainMap.load(layer.path)
        )
        while len(self._stain_maps) >= 2:
            self._stain_maps.popitem(last=False)
        self._stain_maps[key] = loaded
        return loaded

    def _validate_cell_labels(self, layer: WsiLayer) -> None:
        """Hash a large label image once, immediately before first rendering."""

        path = layer.label_path
        digest = layer.label_artifact_digest
        if path is None or digest is None:
            raise ValueError("cell-resolved layer has no sealed label artifact")
        key = (path, digest)
        with self._cell_artifact_lock:
            if key in self._validated_cell_artifacts:
                return
            from histopia.cells._result import validate_cell_artifact

            validate_cell_artifact(path, digest)
            self._validated_cell_artifacts.add(key)

    def _load_protein_predictions(self, layer: WsiLayer) -> object:
        """Load one sealed per-section prediction while retaining one slide."""

        key = layer.path
        cached = self._protein_predictions.get(key)
        if cached is not None:
            self._protein_predictions.move_to_end(key)
            return cached
        from histopia.protein._manifest import validate_protein_artifact
        from histopia.protein._result import ProteinPredictions

        if layer.prediction_artifact_digest is None:
            raise ValueError("protein layer has no sealed artifact digest")
        validate_protein_artifact(layer.path, layer.prediction_artifact_digest)
        loaded = ProteinPredictions.load(layer.path)
        if (
            loaded.target_id != layer.protein_target
            or loaded.model_fingerprint != layer.protein_model_fingerprint
            or set(str(value) for value in loaded.section_ids)
            != {layer.prediction_section}
            or loaded.fingerprint != layer.prediction_fingerprint
        ):
            raise ValueError("protein prediction identity does not match its layer")
        while len(self._protein_predictions) >= 4:
            self._protein_predictions.popitem(last=False)
        self._protein_predictions[key] = loaded
        return loaded

    def _load_protein_overview(
        self, layer: WsiLayer, predictions: object
    ) -> _ProteinOverview:
        """Build one bounded spatial overview without pooling cells by tile."""

        key = (layer.path, layer.digest)
        cached = self._protein_overviews.get(key)
        if cached is not None:
            self._protein_overviews.move_to_end(key)
            return cached
        label_overview = self._load_protein_label_overview(layer)
        loaded = _build_protein_overview(
            layer,
            predictions,
            label_overview=label_overview,
        )
        while len(self._protein_overviews) >= 4:
            self._protein_overviews.popitem(last=False)
        self._protein_overviews[key] = loaded
        return loaded

    def _load_protein_label_overview(self, layer: WsiLayer) -> object:
        """Reuse one bounded categorical label raster across protein panes."""

        if (
            layer.label_path is None
            or layer.source_shape is None
            or layer.microns_per_pixel is None
        ):
            raise ValueError("protein overview geometry is incomplete")
        key = (
            layer.label_path,
            layer.source_shape,
            float(layer.microns_per_pixel),
        )
        cached = self._protein_label_overviews.get(key)
        if cached is not None:
            self._protein_label_overviews.move_to_end(key)
            return cached
        loaded = _build_protein_label_overview(layer)
        while len(self._protein_label_overviews) >= 2:
            self._protein_label_overviews.popitem(last=False)
        self._protein_label_overviews[key] = loaded
        return loaded

    def _section(
        self,
        cohort: str,
        section: str,
        *,
        protein_model: str | None = None,
    ) -> WsiSection:
        if not _SECTION_RE.fullmatch(section):
            raise FileNotFoundError("unknown WSI section")
        try:
            if protein_model is not None:
                if not _MODEL_RE.fullmatch(protein_model):
                    raise FileNotFoundError("unknown protein model")
                return self._protein_sections[(cohort, protein_model, section)]
            return self._sections[(cohort, section)]
        except KeyError as error:
            raise FileNotFoundError("unknown WSI section") from error

    def _public_protein_models(self, cohort: str) -> list[dict[str, object]]:
        return [
            dict(summary)
            for (item_cohort, _model_id), summary in sorted(
                self._protein_models.items()
            )
            if item_cohort == cohort
        ]


class WsiTileCapacityError(RuntimeError):
    """Raised when bounded native tile rendering has no free worker slot."""


def _public_layer_metadata(name: str, layer: WsiLayer) -> dict[str, object]:
    """Return additive, path-free metadata for one immutable tile layer."""

    payload: dict[str, object] = {
        "digest": layer.digest,
        "tile_size": layer.tile_size,
        "width": layer.levels[-1].width,
        "height": layer.levels[-1].height,
        "levels": [
            {"width": level.width, "height": level.height} for level in layer.levels
        ],
        "microns_per_pixel": layer.microns_per_pixel,
        "format": "png" if name in _PNG_LAYERS else "jpg",
    }
    if name in _STAIN_LAYERS:
        payload.update(
            {
                "analysis_mpp": layer.analysis_mpp,
                "display_max_od": layer.display_max_od,
                "content_origin_native_xy": list(
                    layer.content_origin_native_xy or (0, 0)
                ),
                "coordinate_space": "native_content_bbox",
                "render_mode": layer.render_mode,
                "selected_source": layer.selected_source,
                "stain_family": layer.stain_family,
                "adaptive_floor_od": layer.adaptive_floor_od,
                "adaptive_method": layer.adaptive_method,
                "correction_version": (
                    "v3"
                    if layer.stain_artifact_kind == "adaptive"
                    else "v2"
                    if layer.adaptive_floor_od is not None
                    else None
                ),
            }
        )
    elif name in _PROTEIN_SUPPORT_LAYERS:
        payload.update(
            {
                "coordinate_space": "native_content_bbox",
                "measurement_scope": "tissue_support_only",
                "render_mode": "neutral_stain_free_support",
            }
        )
    elif name in _PROTEIN_LAYERS:
        measurement_scopes = {
            "protein_predicted": "predicted_target_od_per_native_cell",
            "protein_predicted_contrast": (
                "predicted_target_od_per_native_cell_contrast_view"
            ),
            "protein_dense": "experimental_4um_cell_raster",
            "protein_dense_contrast": ("experimental_4um_cell_raster_contrast_view"),
            "protein_probability": ("predicted_expression_probability_per_native_cell"),
            "protein_uncertainty": ("predicted_target_od_uncertainty_per_native_cell"),
            "protein_measured": (
                "observed_target_od_aggregated_per_native_cell_from_4um"
            ),
            "protein_measured_contrast": (
                "observed_target_od_aggregated_per_native_cell_from_4um_contrast_view"
            ),
            "protein_residual": "absolute_target_od_error_per_native_cell",
        }
        payload.update(
            {
                "coordinate_space": "native_content_bbox",
                "measurement_scope": measurement_scopes[name],
                "analysis_mpp": layer.analysis_mpp,
                "display_max": layer.display_max_od,
                "section_display_max": layer.section_display_max_od,
                "target_id": layer.protein_target,
                "model_fingerprint": layer.protein_model_fingerprint,
                "render_mode": layer.map_kind,
                "overview_mode": "tissue_summary_below_cell_resolution",
            }
        )
    elif name in _PROTEIN_OBSERVED_LAYERS:
        payload.update(
            {
                "coordinate_space": "target_native_content_bbox",
                "measurement_scope": (
                    "tissue_masked_counterstain_conditioned_target_od_4um"
                    if layer.stain_artifact_kind == "adaptive"
                    else "tissue_masked_harmonized_adaptive_corrected_target_od_4um"
                    if layer.adaptive_floor_od is not None
                    and layer.calibration_fingerprint is not None
                    else "tissue_masked_adaptive_corrected_target_od_4um"
                    if layer.adaptive_floor_od is not None
                    else "observed_target_od_4um"
                ),
                "analysis_mpp": layer.analysis_mpp,
                "display_max": layer.display_max_od,
                "target_id": layer.protein_target,
                "render_mode": layer.render_mode,
                "source_section": layer.comparison_source_section,
                "source_label": layer.comparison_source_label,
                "registered_to_section": layer.comparison_target_section,
                "calibration_fingerprint": layer.calibration_fingerprint,
                "adaptive_floor_od": layer.adaptive_floor_od,
                "adaptive_method": layer.adaptive_method,
                "ground_truth": name == "protein_observed_target",
            }
        )
    return payload


def _load_cohort_sections(
    cohort: str,
    registration: Path,
    registered_wsi: Path | None,
    cell_run: Path | None = None,
    stain_run: Path | None = None,
    protein_run: Path | None = None,
    *,
    protein_result: dict[str, object] | None = None,
    cell_result_cache: dict[Path, dict[str, Any]] | None = None,
    cell_section_scope: CellSectionScope | None = None,
    stain_map_cache: dict[
        tuple[Path, Path, str],
        tuple[dict[str, tuple[dict[str, Any], _StainMapSpec]], float],
    ]
    | None = None,
    base_layer_cache: dict[tuple[object, ...], WsiLayer] | None = None,
) -> tuple[WsiSection, ...]:
    approval = validate_registration_approval(registration)
    result_path = registration / "registration_result.json"
    result = json.loads(result_path.read_text())
    slides = result.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError(f"registration contains no slides: {registration}")
    if any(not isinstance(row, dict) for row in slides):
        raise ValueError("registration slides must contain objects")

    by_stem: dict[str, dict[str, Any]] = {}
    if registered_wsi is not None:
        summary_path = registered_wsi / "full_resolution_warps.json"
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, list):
            raise ValueError("full-resolution warp summary must contain a list")
        root = registered_wsi.expanduser().resolve()
        for row in summary:
            if not isinstance(row, dict):
                raise ValueError("full-resolution warp summary rows must be objects")
            output_value = row.get("output_path")
            provenance = row.get("provenance")
            if not isinstance(output_value, str) or not isinstance(provenance, dict):
                raise ValueError("full-resolution warp provenance is incomplete")
            output = Path(output_value).expanduser().resolve()
            if output.parent != root or not output.is_file():
                raise ValueError(
                    "registered WSI output is outside its configured directory"
                )
            if (
                provenance.get("registration_result_sha256")
                != approval.registration_result_sha256
            ):
                raise ValueError(
                    "registered WSI is not bound to the approved registration"
                )
            export_digest = provenance.get("export_fingerprint")
            if not isinstance(export_digest, str) or not _DIGEST_RE.fullmatch(
                export_digest
            ):
                raise ValueError("registered WSI export fingerprint is invalid")
            stem = output.name.removesuffix(".registered.tiff")
            if stem in by_stem:
                raise ValueError(f"duplicate registered WSI output stem: {stem}")
            by_stem[stem] = row

    cells_by_slide: dict[str, dict[str, Any]] = {}
    cell_artifacts: dict[str, str] = {}
    cell_result_fingerprint: str | None = None
    if cell_run is not None:
        from histopia.cells._result import validate_cell_result_index

        cell_key = Path(cell_run).expanduser().resolve()
        cell_result = (
            cell_result_cache.get(cell_key) if cell_result_cache is not None else None
        )
        if cell_result is None:
            cell_result = validate_cell_result_index(cell_key)
            if cell_result_cache is not None:
                cell_result_cache[cell_key] = cell_result
        cell_result_fingerprint = str(cell_result.get("fingerprint", ""))
        if (
            cell_result.get("registration_result_sha256")
            != approval.registration_result_sha256
        ):
            raise ValueError("cell result is not bound to the approved registration")
        raw_artifacts = cell_result.get("artifacts")
        if not isinstance(raw_artifacts, dict):
            raise ValueError("cell result has no artifact manifest")
        cell_artifacts = {str(key): str(value) for key, value in raw_artifacts.items()}
        cell_slides = (
            cell_section_scope.select(cell_result)
            if cell_section_scope is not None
            else cell_result.get("slides", [])
        )
        for row in cell_slides:
            if not isinstance(row, dict) or not isinstance(row.get("slide"), str):
                raise ValueError("cell result slide rows are invalid")
            cells_by_slide[str(row["slide"])] = row

    stain_by_slide: dict[str, tuple[dict[str, Any], _StainMapSpec]] = {}
    stain_display_max = 0.0
    if stain_run is not None:
        stain_key = (
            Path(registration).expanduser().resolve(),
            Path(stain_run).expanduser().resolve(),
            approval.registration_result_sha256,
        )
        cached_stain = (
            stain_map_cache.get(stain_key) if stain_map_cache is not None else None
        )
        if cached_stain is None:
            cached_stain = _validated_stain_maps(
                registration,
                result,
                approval.registration_result_sha256,
                stain_run,
            )
            if stain_map_cache is not None:
                stain_map_cache[stain_key] = cached_stain
        stain_by_slide, stain_display_max = cached_stain

    protein_by_section: dict[str, dict[str, Any]] = {}
    protein_artifacts: dict[str, str] = {}
    protein_metadata: dict[str, Any] = {}
    if protein_run is not None:
        if cell_run is None or cell_result_fingerprint is None:
            raise ValueError("protein tiles require a matching cell run")
        (
            protein_by_section,
            protein_artifacts,
            protein_metadata,
        ) = _validated_protein_predictions(
            protein_run,
            approval.registration_result_sha256,
            cell_result_fingerprint,
            cohort=cohort,
            validated_result=protein_result,
        )

    slides_by_section = {
        f"{order:03d}": slide for order, slide in enumerate(slides, start=1)
    }
    stain_by_section = {
        f"{order:03d}": stain_by_slide[Path(str(slide.get("path", ""))).name]
        for order, slide in enumerate(slides, start=1)
        if Path(str(slide.get("path", ""))).name in stain_by_slide
    }
    calibrations = dict(protein_metadata.get("od_calibration_by_section", {}))
    observed_target_sections: tuple[str, ...] = ()
    measurement_view = protein_metadata.get("measurement_view")
    legacy_adaptive_measurement = measurement_view in {
        "tissue-masked-adaptive-corrected-target-od-4um-v1",
        "tissue-masked-harmonized-adaptive-corrected-target-od-4um-v2",
    }
    counterstain_v3_measurement = measurement_view == (
        "tissue-masked-counterstain-conditioned-target-od-4um-v3"
    )
    strict_adaptive_measurement = bool(
        legacy_adaptive_measurement or counterstain_v3_measurement
    )
    harmonized_adaptive_measurement = measurement_view == (
        "tissue-masked-harmonized-adaptive-corrected-target-od-4um-v2"
    )
    reference_slide: dict[str, Any] | None = None
    if calibrations or strict_adaptive_measurement:
        if stain_run is None:
            raise ValueError("observed protein comparisons require a stain run")
        reference_rows = [slide for slide in slides if slide.get("is_reference")]
        if len(reference_rows) != 1:
            raise ValueError("protein comparison requires one registration reference")
        reference_slide = reference_rows[0]
        from histopia.protein._config import normalize_target_id

        target_id = normalize_target_id(str(protein_metadata["target_id"]))
        target_section_values = (
            tuple(calibrations)
            if calibrations
            else tuple(
                section
                for section, row in protein_by_section.items()
                if bool(row.get("has_measured"))
            )
        )
        for section in target_section_values:
            calibration = calibrations.get(section)
            protein_row = protein_by_section.get(section)
            stain_entry = stain_by_section.get(section)
            if (
                protein_row is None
                or not bool(protein_row.get("has_measured"))
                or stain_entry is None
            ):
                raise ValueError(
                    "protein OD calibration has no matching measured stain section"
                )
            stain_row, _spec = stain_entry
            marker = stain_row.get("marker")
            correction = dict(stain_row.get("qc", {})).get("correction_accepted")
            if (
                not isinstance(marker, str)
                or normalize_target_id(marker) != target_id
                or correction is not True
                or (calibrations and not isinstance(calibration, dict))
            ):
                raise ValueError(
                    "protein measurement is not bound to an accepted target stain"
                )
            stain_qc = dict(stain_row.get("qc", {}))
            current_adaptive = dict(stain_qc.get("adaptive_background") or {})
            adaptive = (
                dict(stain_qc.get("legacy_adaptive_background") or {})
                if legacy_adaptive_measurement
                and current_adaptive.get("method") == "counterstain-conditioned-v3"
                else current_adaptive
            )
            invalid_legacy = bool(
                legacy_adaptive_measurement
                and (
                    adaptive.get("accepted") is not True
                    or not isinstance(adaptive.get("floor_od"), (int, float))
                    or float(adaptive["floor_od"]) < 0
                )
            )
            invalid_v3 = bool(
                counterstain_v3_measurement
                and (
                    adaptive.get("accepted") is not True
                    or adaptive.get("method") != "counterstain-conditioned-v3"
                    or not isinstance(stain_row.get("adaptive_map"), str)
                    or not isinstance(
                        stain_row.get("adaptive_map_artifact_digest"), str
                    )
                )
            )
            if strict_adaptive_measurement and (
                invalid_legacy
                or invalid_v3
                or not math.isclose(_spec.analysis_mpp, 4.0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "protein measurement requires accepted adaptive target OD "
                    "at 4 um/px"
                )
        observed_target_sections = tuple(
            sorted(target_section_values, key=lambda value: int(value))
        )

    sections: list[WsiSection] = []
    for order, slide in enumerate(slides, start=1):
        section_id = f"{order:03d}"
        source = Path(str(slide.get("path", ""))).expanduser().resolve()
        row = by_stem.get(source.stem)
        cell_row = cells_by_slide.get(source.name)
        stain_entry = stain_by_slide.get(source.name)
        protein_entry = protein_by_section.get(section_id)
        raw_digest = _file_identity_digest(source)
        mask_path = registration / "processed" / f"{source.stem}.mask.png"
        protein_focus_bbox = (
            _mask_focus_bbox_fraction(mask_path)
            if protein_entry is not None and mask_path.is_file()
            else None
        )
        geometry = slide.get("geometry")
        section_label = _marker_label(source.stem)
        protein_comparison: dict[str, object] | None = None
        content_bbox = _geometry_content_bbox(geometry)
        layers = {
            "raw": _cached_wsi_layer(
                base_layer_cache,
                ("image", "raw", source, raw_digest, content_bbox),
                partial(
                    _image_layer,
                    "raw",
                    source,
                    raw_digest,
                    crop_bbox_xywh=content_bbox,
                ),
            ),
        }
        if stain_entry is not None:
            native_shape = _geometry_native_shape(geometry)
            expected_source_shape = (native_shape[1], native_shape[0])
            if layers["raw"].source_shape != expected_source_shape:
                raise ValueError(
                    "native source WSI dimensions do not match registration geometry"
                )
        if row is not None:
            registered = Path(str(row["output_path"])).expanduser().resolve()
            export_digest = str(row["provenance"]["export_fingerprint"])
            layers["registered"] = _cached_wsi_layer(
                base_layer_cache,
                ("image", "registered", registered, export_digest),
                lambda registered=registered, export_digest=export_digest: _image_layer(
                    "registered", registered, export_digest
                ),
            )
        if mask_path.is_file():
            raw_layer = layers["raw"]
            layers["mask"] = _cached_wsi_layer(
                base_layer_cache,
                ("mask", mask_path, raw_layer.digest),
                lambda mask_path=mask_path, raw_layer=raw_layer: _mask_layer(
                    mask_path, raw_layer
                ),
            )
        if cell_row is not None:
            relative = str(cell_row.get("labels", ""))
            digest = cell_artifacts.get(relative)
            if digest is None or not _DIGEST_RE.fullmatch(digest):
                raise ValueError("cell label artifact digest is missing")
            label_path = (cell_run / relative).resolve()
            raw_layer = layers["raw"]
            layers["cells"] = _cached_wsi_layer(
                base_layer_cache,
                ("cells", label_path, digest, raw_layer.digest),
                lambda label_path=label_path, digest=digest, raw_layer=raw_layer: (
                    _cell_layer(label_path, digest, raw_layer)
                ),
            )
            if protein_entry is not None:
                prediction_relative = str(protein_entry["predictions"])
                prediction_digest = protein_artifacts[prediction_relative]
                prediction_path = (protein_run / prediction_relative).resolve()
                for protein_layer, value_kind in (
                    ("protein_predicted", "predicted_od_reference"),
                    ("protein_predicted_contrast", "predicted_od_reference"),
                    ("protein_dense", "predicted_od_reference"),
                    ("protein_dense_contrast", "predicted_od_reference"),
                    ("protein_probability", "expression_probability"),
                    ("protein_uncertainty", "uncertainty"),
                    ("protein_measured", "measured_od"),
                    ("protein_measured_contrast", "measured_od"),
                    ("protein_residual", "residual"),
                ):
                    if value_kind in {"measured_od", "residual"} and not bool(
                        protein_entry.get("has_measured")
                    ):
                        continue
                    if value_kind == "expression_probability" and not bool(
                        protein_entry.get("has_probability")
                    ):
                        continue
                    layers[protein_layer] = _protein_layer(
                        protein_layer,
                        prediction_path,
                        prediction_digest,
                        label_path,
                        digest,
                        layers["cells"],
                        value_kind=value_kind,
                        section=section_id,
                        target_id=protein_metadata["target_id"],
                        model_fingerprint=protein_metadata["model_fingerprint"],
                        prediction_fingerprint=str(
                            protein_entry["prediction_fingerprint"]
                        ),
                        display_max=float(
                            protein_entry.get("section_display_max", {}).get(
                                value_kind,
                                protein_metadata["display_max"][value_kind],
                            )
                            if protein_layer.endswith("_contrast")
                            else protein_metadata["display_max"][value_kind]
                        ),
                        section_display_max=float(
                            protein_entry.get("section_display_max", {}).get(
                                value_kind,
                                protein_metadata["display_max"][value_kind],
                            )
                        ),
                        analysis_mpp=(
                            4.0 if protein_layer.startswith("protein_dense") else None
                        ),
                        mask_path=(
                            layers["mask"].mask_path if "mask" in layers else None
                        ),
                        focus_bbox_fraction=protein_focus_bbox,
                    )
                if "mask" in layers:
                    layers["protein_tissue_support"] = _protein_tissue_support_layer(
                        layers["mask"]
                    )
        elif protein_entry is not None:
            raise ValueError("protein predictions have no matching cell labels")
        if stain_entry is not None:
            stain_row, stain_spec = stain_entry
            map_relative = str(stain_row["map"])
            map_digest = str(stain_row["map_artifact_digest"])
            map_path = (stain_run / map_relative).resolve()
            correction_accepted = bool(
                dict(stain_row.get("qc", {})).get("correction_accepted")
            )
            layers["stain_raw"] = _stain_layer(
                "stain_raw",
                map_path,
                map_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind="raw_target_od",
                render_mode="overlay",
                selected_source="raw",
                stain_family=str(stain_row.get("family", "h-dab")),
            )
            output_kind = (
                "corrected_target_od" if correction_accepted else "raw_target_od"
            )
            output_source = "corrected" if correction_accepted else "raw"
            family_name = str(stain_row.get("family", "h-dab"))
            stain_qc = dict(stain_row.get("qc", {}))
            adaptive = dict(stain_qc.get("adaptive_background") or {})
            adaptive_method = str(adaptive.get("method", ""))
            counterstain_v3 = bool(
                adaptive.get("accepted") is True
                and adaptive_method == "counterstain-conditioned-v3"
                and isinstance(stain_row.get("adaptive_map"), str)
                and isinstance(stain_row.get("adaptive_map_artifact_digest"), str)
            )
            legacy_adaptive = (
                dict(stain_qc.get("legacy_adaptive_background") or {})
                if adaptive_method == "counterstain-conditioned-v3"
                else adaptive
            )
            adaptive_floor = (
                float(legacy_adaptive["floor_od"])
                if legacy_adaptive.get("accepted") is True
                and isinstance(legacy_adaptive.get("floor_od"), (int, float))
                else None
            )
            output_overlay = _stain_layer(
                "stain_corrected",
                map_path,
                map_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind=output_kind,
                render_mode="overlay",
                selected_source=output_source,
                stain_family=family_name,
            )
            layers["stain_corrected"] = output_overlay
            if adaptive_floor is not None:
                layers["stain_adaptive"] = _stain_layer(
                    "stain_adaptive",
                    map_path,
                    map_digest,
                    stain_spec,
                    display_max_od=stain_display_max,
                    map_kind=output_kind,
                    render_mode="overlay",
                    selected_source=f"adaptive_{output_source}",
                    stain_family=family_name,
                    adaptive_floor_od=adaptive_floor,
                    adaptive_method=str(legacy_adaptive.get("method", "")),
                )
                layers["stain_adaptive_map"] = _stain_layer(
                    "stain_adaptive_map",
                    map_path,
                    map_digest,
                    stain_spec,
                    display_max_od=stain_display_max,
                    map_kind=output_kind,
                    render_mode="opaque_tissue",
                    selected_source=f"adaptive_{output_source}",
                    stain_family=family_name,
                    adaptive_floor_od=adaptive_floor,
                    adaptive_method=str(legacy_adaptive.get("method", "")),
                )
            if counterstain_v3:
                v3_path = (
                    (stain_run / str(stain_row["adaptive_map"])).expanduser().resolve()
                )
                v3_digest = str(stain_row["adaptive_map_artifact_digest"])
                for layer_name, render_mode in (
                    ("stain_adaptive_v3", "overlay"),
                    ("stain_adaptive_v3_map", "opaque_tissue"),
                ):
                    layers[layer_name] = _stain_layer(
                        layer_name,
                        v3_path,
                        v3_digest,
                        stain_spec,
                        display_max_od=stain_display_max,
                        map_kind="target_od",
                        render_mode=render_mode,
                        selected_source="counterstain_conditioned_target_od",
                        stain_family=family_name,
                        adaptive_method=adaptive_method,
                        stain_artifact_kind="adaptive",
                    )
                output_path = v3_path
                output_digest = v3_digest
                output_map_kind = "target_od"
                output_selected_source = "counterstain_conditioned_target_od"
                output_floor = None
                output_artifact_kind = "adaptive"
            else:
                output_path = map_path
                output_digest = map_digest
                output_map_kind = output_kind
                output_selected_source = (
                    f"adaptive_{output_source}"
                    if adaptive_floor is not None
                    else output_source
                )
                output_floor = adaptive_floor
                output_artifact_kind = "physical"
            layers["stain_output"] = _stain_layer(
                "stain_output",
                output_path,
                output_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind=output_map_kind,
                render_mode="overlay",
                selected_source=output_selected_source,
                stain_family=family_name,
                adaptive_floor_od=output_floor,
                adaptive_method=(
                    adaptive_method
                    if counterstain_v3
                    else str(legacy_adaptive.get("method", "")) or None
                ),
                stain_artifact_kind=output_artifact_kind,
            )
            layers["stain_output_map"] = _stain_layer(
                "stain_output_map",
                output_path,
                output_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind=output_map_kind,
                render_mode="opaque_tissue",
                selected_source=output_selected_source,
                stain_family=family_name,
                adaptive_floor_od=output_floor,
                adaptive_method=(
                    adaptive_method
                    if counterstain_v3
                    else str(legacy_adaptive.get("method", "")) or None
                ),
                stain_artifact_kind=output_artifact_kind,
            )
            layers["stain_counterstain"] = _stain_layer(
                "stain_counterstain",
                map_path,
                map_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind="counterstain_od",
                render_mode="overlay",
                selected_source="counterstain",
                stain_family=family_name,
            )
            layers["stain_residual"] = _stain_layer(
                "stain_residual",
                map_path,
                map_digest,
                stain_spec,
                display_max_od=stain_display_max,
                map_kind="reconstruction_residual",
                render_mode="overlay",
                selected_source="residual",
                stain_family=family_name,
            )
            layers["stain_support"] = _stain_layer(
                "stain_support",
                map_path,
                map_digest,
                stain_spec,
                display_max_od=1.0,
                map_kind="tissue_mask",
                render_mode="support",
                selected_source="tissue_mask",
                stain_family=family_name,
            )
        if protein_entry is not None and observed_target_sections:
            if reference_slide is None or stain_run is None:
                raise AssertionError("validated comparison inputs are missing")
            measured = section_id in observed_target_sections
            source_section = (
                section_id
                if measured
                else min(
                    observed_target_sections,
                    key=lambda value: (
                        abs(int(value) - order),
                        int(value),
                    ),
                )
            )
            source_stain_row, source_stain_spec = stain_by_section[source_section]
            source_slide = slides_by_section[source_section]
            source_slide_path = Path(str(source_slide.get("path", "")))
            source_label = _marker_label(source_slide_path.stem)
            source_qc = dict(source_stain_row.get("qc", {}))
            source_current_adaptive = dict(source_qc.get("adaptive_background") or {})
            source_measurement_adaptive = (
                dict(source_qc.get("legacy_adaptive_background") or {})
                if legacy_adaptive_measurement
                and source_current_adaptive.get("method")
                == "counterstain-conditioned-v3"
                else source_current_adaptive
            )
            if counterstain_v3_measurement:
                source_map_path = (
                    (stain_run / str(source_stain_row["adaptive_map"]))
                    .expanduser()
                    .resolve()
                )
                source_map_digest = str(
                    source_stain_row["adaptive_map_artifact_digest"]
                )
                source_artifact_kind = "adaptive"
                source_adaptive_floor = None
            else:
                source_map_path = (
                    (stain_run / str(source_stain_row["map"])).expanduser().resolve()
                )
                source_map_digest = str(source_stain_row["map_artifact_digest"])
                source_artifact_kind = "physical"
                source_adaptive_floor = (
                    float(source_measurement_adaptive["floor_od"])
                    if legacy_adaptive_measurement
                    else None
                )
            observed_name = (
                "protein_observed_target"
                if measured
                else "protein_nearest_observed_target"
            )
            layers[observed_name] = _observed_target_layer(
                observed_name,
                source_map_path,
                source_map_digest,
                source_stain_spec,
                source_slide=source_slide,
                target_slide=slide,
                reference_slide=reference_slide,
                source_section=source_section,
                source_label=source_label,
                target_section=section_id,
                target_id=str(protein_metadata["target_id"]),
                calibration=(
                    dict(calibrations[source_section]) if calibrations else None
                ),
                calibration_fingerprint=(
                    str(protein_metadata["od_calibration_fingerprint"])
                    if calibrations
                    else None
                ),
                adaptive_floor_od=source_adaptive_floor,
                adaptive_method=str(source_measurement_adaptive.get("method", ""))
                or None,
                stain_artifact_kind=source_artifact_kind,
                display_max_od=float(protein_metadata["quantitative_display_max_od"]),
                registration_result_sha256=approval.registration_result_sha256,
                target_mask_path=mask_path,
            )
            if measured:
                current_layer = observed_name
                current_scope = (
                    "counterstain_conditioned_target_od_4um"
                    if counterstain_v3_measurement
                    else "harmonized_adaptive_target_od_4um"
                    if harmonized_adaptive_measurement
                    else "adaptive_target_od_4um"
                    if strict_adaptive_measurement
                    else "calibrated_target_od_4um"
                )
                comparison_layer = "protein_residual"
                comparison_label = "Absolute per-cell residual"
            else:
                if strict_adaptive_measurement:
                    current_layer, current_scope = _select_protein_current_layer(
                        layers,
                        counterstain_v3=counterstain_v3_measurement,
                    )
                else:
                    current_layer = (
                        "stain_adaptive_map"
                        if stain_entry is not None and "stain_adaptive_map" in layers
                        else "stain_output_map"
                        if stain_entry is not None
                        else "raw"
                    )
                    current_scope = (
                        "current_marker_adaptive_od_4um"
                        if current_layer == "stain_adaptive_map"
                        else "current_marker_od_4um"
                        if stain_entry is not None
                        else "native_histology_context_only"
                    )
                comparison_layer = observed_name
                comparison_label = (
                    "Nearest observed "
                    + _protein_target_label(
                        str(protein_metadata["target_id"]),
                        protein_metadata.get("target_label"),
                    )
                    + " · "
                    f"section {source_section} registered here"
                )
            protein_comparison = {
                "schema_version": 2 if strict_adaptive_measurement else 1,
                "mode": "measured-target" if measured else "unmeasured-transfer",
                "target_id": str(protein_metadata["target_id"]),
                "target_label": _protein_target_label(
                    str(protein_metadata["target_id"]),
                    protein_metadata.get("target_label"),
                ),
                "predicted_layer": "protein_predicted",
                "current_layer": current_layer,
                "current_label": section_label,
                "current_scope": current_scope,
                "comparison_layer": comparison_layer,
                "comparison_label": comparison_label,
                "ground_truth_available": measured,
                "nearest_target_section": None if measured else source_section,
                "nearest_distance_sections": (
                    0 if measured else abs(int(source_section) - order)
                ),
                "measurement_view": protein_metadata.get("measurement_view"),
                "measurement_statistic": protein_metadata.get(
                    "measurement_statistic", "mean"
                ),
            }
        elif protein_entry is not None:
            if strict_adaptive_measurement:
                current_layer, current_scope = _select_protein_current_layer(
                    layers,
                    counterstain_v3=counterstain_v3_measurement,
                )
            else:
                current_layer = (
                    "stain_adaptive_map"
                    if stain_entry is not None and "stain_adaptive_map" in layers
                    else "stain_output_map"
                    if stain_entry is not None
                    else "raw"
                )
                current_scope = (
                    "current_marker_adaptive_od_4um"
                    if current_layer == "stain_adaptive_map"
                    else "current_marker_od_4um"
                    if stain_entry is not None
                    else "native_histology_context_only"
                )
            protein_comparison = {
                "schema_version": 3,
                "mode": "external-no-local-truth",
                "target_id": str(protein_metadata["target_id"]),
                "target_label": _protein_target_label(
                    str(protein_metadata["target_id"]),
                    protein_metadata.get("target_label"),
                ),
                "predicted_layer": "protein_predicted",
                "current_layer": current_layer,
                "current_label": section_label,
                "current_scope": current_scope,
                "comparison_layer": "protein_uncertainty",
                "comparison_label": "Prediction uncertainty",
                "ground_truth_available": False,
                "nearest_target_section": None,
                "nearest_distance_sections": None,
                "measurement_view": protein_metadata.get("measurement_view"),
                "measurement_statistic": protein_metadata.get(
                    "measurement_statistic", "mean"
                ),
            }
        sections.append(
            WsiSection(
                cohort=cohort,
                section=section_id,
                slide=source.name,
                label=section_label,
                reference=bool(slide.get("is_reference")),
                layers=layers,
                protein_comparison=protein_comparison,
                focus_bbox_fraction=protein_focus_bbox,
            )
        )
    return tuple(sections)


def _mask_focus_bbox_fraction(
    mask_path: Path,
    *,
    padding_fraction: float = 0.08,
) -> tuple[float, float, float, float] | None:
    """Return a padded tissue bbox in path-free normalized native coordinates."""

    import numpy as np
    from PIL import Image

    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    image_height, image_width = mask.shape
    x0 = int(columns.min())
    y0 = int(rows.min())
    x1 = int(columns.max()) + 1
    y1 = int(rows.max()) + 1
    pad_x = max(2, round((x1 - x0) * padding_fraction))
    pad_y = max(2, round((y1 - y0) * padding_fraction))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(image_width, x1 + pad_x)
    y1 = min(image_height, y1 + pad_y)
    return (
        x0 / image_width,
        y0 / image_height,
        (x1 - x0) / image_width,
        (y1 - y0) / image_height,
    )


def _validated_protein_predictions(
    protein_run: Path,
    registration_result_sha256: str,
    cell_result_fingerprint: str,
    *,
    cohort: str | None = None,
    validated_result: dict[str, object] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Validate exact upstream bindings and path-free per-section artifacts."""

    from histopia.protein._manifest import validate_protein_result_index

    root = protein_run.expanduser().resolve()
    result = (
        validate_protein_result_index(root)
        if validated_result is None
        else validated_result
    )
    binding = result
    if result.get("schema_version") == 4:
        cohort_bindings = result.get("cohort_bindings")
        if not isinstance(cohort_bindings, dict) or cohort not in cohort_bindings:
            raise ValueError("protein result has no binding for this cohort")
        raw_binding = cohort_bindings[cohort]
        if not isinstance(raw_binding, dict):
            raise ValueError("protein cohort binding is invalid")
        binding = raw_binding
    if binding.get("registration_result_sha256") != registration_result_sha256:
        raise ValueError("protein result is not bound to the approved registration")
    if binding.get("cell_result_fingerprint") != cell_result_fingerprint:
        raise ValueError("protein result is not bound to the configured cell result")
    target_id = result.get("target_id")
    model_fingerprint = result.get("model_fingerprint")
    if not isinstance(target_id, str) or not isinstance(model_fingerprint, str):
        raise ValueError("protein result target/model identity is incomplete")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("protein result artifact manifest is missing")
    rows: dict[str, dict[str, Any]] = {}
    declared_scale = result.get("target_global_display_max_od", 1.0)
    if (
        isinstance(declared_scale, bool)
        or not isinstance(declared_scale, (int, float))
        or not math.isfinite(float(declared_scale))
        or float(declared_scale) <= 0
    ):
        raise ValueError("protein target-global display scale is invalid")
    quantitative_max = float(declared_scale)
    raw_slides = result.get("slides")
    if not isinstance(raw_slides, list):
        raise ValueError("protein result slides are missing")
    for raw in raw_slides:
        if not isinstance(raw, dict):
            raise ValueError("protein result slide rows must be objects")
        row_cohort = raw.get("cohort")
        if result.get("schema_version") == 4 and row_cohort != cohort:
            continue
        section = str(raw.get("section", ""))
        relative = str(raw.get("predictions", ""))
        if not _SECTION_RE.fullmatch(section) or relative not in artifacts:
            raise ValueError("protein prediction section or artifact is invalid")
        prediction_fingerprint = raw.get("prediction_fingerprint")
        cells = raw.get("cells")
        measured_cells = raw.get("measured_cells", 0)
        if (
            not isinstance(prediction_fingerprint, str)
            or not _DIGEST_RE.fullmatch(prediction_fingerprint)
            or isinstance(cells, bool)
            or not isinstance(cells, int)
            or cells <= 0
            or isinstance(measured_cells, bool)
            or not isinstance(measured_cells, int)
            or not 0 <= measured_cells <= cells
        ):
            raise ValueError("protein prediction summary is invalid")
        if section in rows:
            raise ValueError(f"duplicate protein prediction section: {section}")
        has_probability = raw.get("has_probability")
        if has_probability is None:
            has_probability = bool(result.get("binary_enabled", False))
        section_scale = raw.get("section_display_max_od", quantitative_max)
        if (
            isinstance(section_scale, bool)
            or not isinstance(section_scale, (int, float))
            or not math.isfinite(float(section_scale))
            or float(section_scale) <= 0
        ):
            raise ValueError("protein section display scale is invalid")
        rows[section] = {
            **raw,
            "has_measured": measured_cells > 0,
            "has_probability": bool(has_probability),
            "section_display_max": {
                "predicted_od_reference": float(section_scale),
                "measured_od": float(section_scale),
            },
        }
    display_max = {
        "relative_expression": 1.0,
        "expression_probability": 1.0,
        "predicted_od_reference": quantitative_max,
        "measured_od": quantitative_max,
        "uncertainty": quantitative_max,
        "residual": quantitative_max,
    }
    # Measured and predicted OD must use one numerical color scale.  Separate
    # normalization made a weak prediction look deceptively similar to truth.
    display_max["predicted_od_reference"] = quantitative_max
    display_max["measured_od"] = quantitative_max
    calibration_by_section, calibration_fingerprint = _validated_od_calibration(
        result.get("od_calibration"), cohort=cohort
    )
    return (
        rows,
        {str(key): str(value) for key, value in artifacts.items()},
        {
            "target_id": target_id,
            "target_label": _protein_target_label(
                target_id,
                result.get("target_label"),
            ),
            "model_fingerprint": model_fingerprint,
            "model_id": result.get("model_id"),
            "architecture": result.get("architecture"),
            "training_cohorts": result.get("training_cohorts", []),
            "measurement_view": result.get("measurement_view"),
            "measurement_statistic": result.get("measurement_statistic", "mean"),
            "display_max": display_max,
            "quantitative_display_max_od": quantitative_max,
            "od_calibration_by_section": calibration_by_section,
            "od_calibration_fingerprint": calibration_fingerprint,
        },
    )


def _validated_od_calibration(
    raw: object,
    *,
    cohort: str | None = None,
) -> tuple[dict[str, dict[str, object]], str | None]:
    """Validate an optional sealed repeated-stain OD calibration."""

    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        raise ValueError("protein OD calibration must be an object")
    schema_method = (raw.get("schema_version"), raw.get("method"))
    if schema_method not in {
        (1, "registered-bin-monotonic-v1"),
        (2, "equal-section-adaptive-quantile-v2"),
    }:
        raise ValueError("protein OD calibration schema is unsupported")
    fingerprint = raw.get("fingerprint")
    sections = raw.get("sections")
    if (
        not isinstance(fingerprint, str)
        or not _DIGEST_RE.fullmatch(fingerprint)
        or not isinstance(sections, list)
        or not sections
    ):
        raise ValueError("protein OD calibration identity is incomplete")
    core = {key: value for key, value in raw.items() if key != "fingerprint"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != fingerprint:
        raise ValueError("protein OD calibration fingerprint does not match")
    from histopia.protein._calibration import SectionOdCalibration

    cohort_aware = schema_method[0] == 2
    if cohort_aware and not cohort:
        raise ValueError("cohort-aware protein OD calibration requires a cohort")
    validated: dict[str, dict[str, object]] = {}
    identities: set[tuple[str | None, str]] = set()
    for value in sections:
        if not isinstance(value, dict):
            raise ValueError("protein OD calibration sections must be objects")
        section = str(value.get("section", ""))
        row_cohort = value.get("cohort") if cohort_aware else None
        identity = (
            str(row_cohort) if isinstance(row_cohort, str) else None,
            section,
        )
        if (
            not _SECTION_RE.fullmatch(section)
            or (cohort_aware and (not isinstance(row_cohort, str) or not row_cohort))
            or identity in identities
        ):
            raise ValueError("protein OD calibration section is invalid")
        identities.add(identity)
        import numpy as np

        calibration = SectionOdCalibration(
            section=section,
            source_knots=np.asarray(value.get("source_knots"), dtype=np.float64),
            reference_knots=np.asarray(value.get("reference_knots"), dtype=np.float64),
            anchor_count=int(value.get("anchor_count", -1)),
            status=str(value.get("status", "")),
            scale=float(value.get("scale", float("nan"))),
        )
        if not cohort_aware or row_cohort == cohort:
            if section in validated:
                raise ValueError("protein OD calibration section is duplicated")
            validated[section] = calibration.as_dict()
    if cohort_aware and not validated:
        raise ValueError("protein OD calibration has no sections for this cohort")
    return validated, fingerprint


def _validated_stain_maps(
    registration: Path,
    registration_result: dict[str, object],
    registration_result_sha256: str,
    stain_run: Path,
) -> tuple[dict[str, tuple[dict[str, Any], _StainMapSpec]], float]:
    """Validate sealed maps against their exact registration and geometry."""

    from histopia.visualization._stain_viewer import load_stain_viewer_run

    root = stain_run.expanduser().resolve()
    viewer_run = load_stain_viewer_run(
        registration,
        registration_result,
        root,
        verify_artifacts=False,
    )
    if (
        viewer_run.payload.get("registration_result_sha256")
        != registration_result_sha256
    ):
        raise ValueError("stain result is not bound to the approved registration")
    approval_digest = viewer_run.payload.get("registration_approval_sha256")
    approval_path = registration / "registration_approval.json"
    if approval_digest != _sha256_file(approval_path):
        raise ValueError("stain result is not bound to the exact registration approval")
    measurement = viewer_run.payload.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("stain result measurement metadata is missing")
    analysis_mpp = _positive_float(measurement.get("analysis_mpp"), "analysis MPP")
    artifacts = viewer_run.payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("stain result artifact manifest is missing")
    registration_rows = registration_result.get("slides")
    stain_rows = viewer_run.payload.get("slides")
    if not isinstance(registration_rows, list) or not isinstance(stain_rows, list):
        raise ValueError("stain and registration slide rows are required")

    validated: dict[str, tuple[dict[str, Any], _StainMapSpec]] = {}
    for order, (registration_row, raw_stain_row) in enumerate(
        zip(registration_rows, stain_rows, strict=True),
        start=1,
    ):
        if not isinstance(registration_row, dict) or not isinstance(
            raw_stain_row, dict
        ):
            raise ValueError("stain and registration slide rows must be objects")
        if raw_stain_row.get("order") != order:
            raise ValueError("stain result slide order metadata does not match")
        if raw_stain_row.get("quantified") is not True:
            continue
        slide_name = Path(str(registration_row.get("path", ""))).name
        relative = raw_stain_row.get("map")
        if not isinstance(relative, str) or relative not in artifacts:
            raise ValueError("stain map artifact digest is missing")
        artifact_digest = artifacts[relative]
        if not isinstance(artifact_digest, str) or not _DIGEST_RE.fullmatch(
            artifact_digest
        ):
            raise ValueError("stain map artifact digest is invalid")
        map_path = (root / relative).resolve()
        if not map_path.is_relative_to(root):
            raise ValueError("stain map must stay inside its configured run")
        stain_map = _stain_map_header(map_path)
        geometry = registration_row.get("geometry")
        bbox = _required_geometry_tuple(geometry, "content_bbox_xywh", integer=True)
        native_shape = _required_geometry_tuple(geometry, "native_shape", integer=True)
        mpp_xy = _required_geometry_tuple(geometry, "mpp_xy", integer=False)
        x, y, width, height = (int(value) for value in bbox)
        native_height, native_width = (int(value) for value in native_shape)
        if x + width > native_width or y + height > native_height:
            raise ValueError("stain content bounding box is outside source geometry")
        expected_width = max(1, round(width * mpp_xy[0] / analysis_mpp))
        expected_height = max(1, round(height * mpp_xy[1] / analysis_mpp))
        if stain_map.slide_id != slide_name:
            raise ValueError("stain map slide identity does not match registration")
        if stain_map.content_fingerprint != raw_stain_row.get("map_fingerprint"):
            raise ValueError("stain map content fingerprint does not match result")
        if not math.isclose(stain_map.analysis_mpp, analysis_mpp, abs_tol=1e-9):
            raise ValueError("stain map physical resolution does not match result")
        if stain_map.content_origin_native_xy != (x, y):
            raise ValueError("stain map origin does not match registration geometry")
        if any(
            not math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(stain_map.source_mpp_xy, mpp_xy, strict=True)
        ):
            raise ValueError("stain map source resolution does not match registration")
        if stain_map.shape != (expected_height, expected_width):
            raise ValueError("stain map dimensions do not match source geometry")
        row = dict(raw_stain_row)
        row["map_artifact_digest"] = artifact_digest
        adaptive_relative = raw_stain_row.get("adaptive_map")
        adaptive_qc = dict(raw_stain_row.get("qc", {})).get("adaptive_background")
        expects_v3 = bool(
            isinstance(adaptive_qc, dict)
            and adaptive_qc.get("accepted") is True
            and adaptive_qc.get("method") == "counterstain-conditioned-v3"
        )
        if adaptive_relative is not None:
            if (
                not isinstance(adaptive_relative, str)
                or adaptive_relative not in artifacts
            ):
                raise ValueError("adaptive stain map artifact digest is missing")
            adaptive_digest = artifacts[adaptive_relative]
            if not isinstance(adaptive_digest, str) or not _DIGEST_RE.fullmatch(
                adaptive_digest
            ):
                raise ValueError("adaptive stain map artifact digest is invalid")
            adaptive_path = (root / adaptive_relative).resolve()
            if not adaptive_path.is_relative_to(root):
                raise ValueError(
                    "adaptive stain map must stay inside its configured run"
                )
            adaptive_map = _adaptive_stain_map_header(adaptive_path)
            if (
                adaptive_map.slide_id != stain_map.slide_id
                or adaptive_map.shape != stain_map.shape
                or not math.isclose(
                    adaptive_map.analysis_mpp, stain_map.analysis_mpp, abs_tol=1e-9
                )
                or adaptive_map.content_origin_native_xy
                != stain_map.content_origin_native_xy
                or any(
                    not math.isclose(actual, expected, abs_tol=1e-9)
                    for actual, expected in zip(
                        adaptive_map.source_mpp_xy,
                        stain_map.source_mpp_xy,
                        strict=True,
                    )
                )
                or adaptive_map.source_content_fingerprint
                != stain_map.content_fingerprint
                or adaptive_map.content_fingerprint
                != raw_stain_row.get("adaptive_map_fingerprint")
                or not isinstance(adaptive_qc, dict)
                or adaptive_map.method != adaptive_qc.get("method")
            ):
                raise ValueError("adaptive stain map binding or geometry differs")
            row["adaptive_map_artifact_digest"] = adaptive_digest
        elif expects_v3:
            raise ValueError("accepted counterstain correction has no adaptive map")
        validated[slide_name] = (
            row,
            _StainMapSpec(
                width=expected_width,
                height=expected_height,
                analysis_mpp=stain_map.analysis_mpp,
                content_origin_native_xy=stain_map.content_origin_native_xy,
                source_mpp_xy=stain_map.source_mpp_xy,
            ),
        )
    if not validated:
        raise ValueError("stain result contains no quantified maps")
    return validated, viewer_run.display_max_od


def _adaptive_stain_map_header(path: Path) -> _AdaptiveStainMapHeader:
    """Read a sealed derived target map without inflating its image arrays."""

    import zipfile

    import numpy as np

    arrays = {"target_od", "tissue_mask"}
    metadata = {
        "schema_version",
        "slide_id",
        "analysis_mpp",
        "content_origin_native_xy",
        "source_mpp_xy",
        "source_content_fingerprint",
        "method",
        "diagnostics_json",
        "content_fingerprint",
    }
    expected = {f"{name}.npy" for name in arrays | metadata}
    try:
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != expected:
                raise ValueError("adaptive stain archive fields are incomplete")
            target_header = _npy_array_header(archive, "target_od", np)
            tissue_header = _npy_array_header(archive, "tissue_mask", np)
        if target_header[0] != tissue_header[0] or len(target_header[0]) != 2:
            raise ValueError("adaptive stain map arrays do not align")
        if target_header[1] != np.dtype("float32") or tissue_header[1] != np.dtype(
            "uint8"
        ):
            raise ValueError("adaptive stain map array dtype is invalid")
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported adaptive stain map schema")
            diagnostics = json.loads(str(data["diagnostics_json"]))
            header = _AdaptiveStainMapHeader(
                slide_id=str(data["slide_id"]),
                shape=tuple(int(value) for value in target_header[0]),
                analysis_mpp=float(data["analysis_mpp"]),
                content_origin_native_xy=tuple(
                    int(value) for value in data["content_origin_native_xy"]
                ),
                source_mpp_xy=tuple(float(value) for value in data["source_mpp_xy"]),
                source_content_fingerprint=str(data["source_content_fingerprint"]),
                method=str(data["method"]),
                content_fingerprint=str(data["content_fingerprint"]),
            )
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"adaptive stain map archive is invalid: {path.name}") from exc
    if (
        not header.slide_id
        or header.analysis_mpp <= 0
        or len(header.content_origin_native_xy) != 2
        or len(header.source_mpp_xy) != 2
        or any(value <= 0 or not math.isfinite(value) for value in header.source_mpp_xy)
        or not _DIGEST_RE.fullmatch(header.source_content_fingerprint)
        or not _DIGEST_RE.fullmatch(header.content_fingerprint)
        or not isinstance(diagnostics, dict)
        or diagnostics.get("method") != header.method
    ):
        raise ValueError("adaptive stain map metadata is invalid")
    return header


def _stain_map_header(path: Path) -> _StainMapHeader:
    """Read sealed map metadata and NPY headers without inflating image arrays."""

    import zipfile

    import numpy as np

    continuous = {
        "raw_target_od",
        "corrected_target_od",
        "counterstain_od",
        "reconstruction_residual",
        "confidence",
    }
    binary = {"tissue_mask", "positive_mask"}
    metadata = {
        "schema_version",
        "slide_id",
        "analysis_mpp",
        "content_origin_native_xy",
        "source_mpp_xy",
        "provenance_json",
        "fingerprint",
        "content_fingerprint",
    }
    expected = {f"{name}.npy" for name in continuous | binary | metadata}
    try:
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != expected:
                raise ValueError("stain map archive fields are incomplete")
            headers = {
                name: _npy_array_header(archive, name, np)
                for name in continuous | binary
            }
        shapes = {shape for shape, _dtype in headers.values()}
        if len(shapes) != 1:
            raise ValueError("stain map arrays do not share one shape")
        shape = shapes.pop()
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("stain map array shape is invalid")
        if any(headers[name][1] != np.dtype("float32") for name in continuous):
            raise ValueError("continuous stain map array dtype is invalid")
        if any(headers[name][1] != np.dtype("uint8") for name in binary):
            raise ValueError("binary stain map array dtype is invalid")
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported stain map schema")
            slide_id = str(data["slide_id"])
            analysis_mpp = float(data["analysis_mpp"])
            origin = tuple(int(value) for value in data["content_origin_native_xy"])
            source_mpp = tuple(float(value) for value in data["source_mpp_xy"])
            content_fingerprint = str(data["content_fingerprint"])
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"stain map archive is invalid: {path.name}") from exc
    if (
        not slide_id
        or not math.isfinite(analysis_mpp)
        or analysis_mpp <= 0
        or len(origin) != 2
        or len(source_mpp) != 2
        or any(not math.isfinite(value) or value <= 0 for value in source_mpp)
        or not _DIGEST_RE.fullmatch(content_fingerprint)
    ):
        raise ValueError("stain map metadata is invalid")
    return _StainMapHeader(
        slide_id=slide_id,
        shape=(int(shape[0]), int(shape[1])),
        analysis_mpp=analysis_mpp,
        content_origin_native_xy=(origin[0], origin[1]),
        source_mpp_xy=(source_mpp[0], source_mpp[1]),
        content_fingerprint=content_fingerprint,
    )


def _npy_array_header(
    archive: object,
    name: str,
    np: Any,
) -> tuple[tuple[int, ...], Any]:
    with archive.open(f"{name}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version: {version}")
    return tuple(int(value) for value in shape), dtype


def _required_geometry_tuple(
    geometry: object,
    name: str,
    *,
    integer: bool,
) -> tuple[float, ...]:
    if not isinstance(geometry, dict):
        raise ValueError("registration source geometry is missing")
    value = geometry.get(name)
    if not isinstance(value, list) or len(value) not in {2, 4}:
        raise ValueError(f"registration geometry {name} is invalid")
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError(f"registration geometry {name} is invalid")
    elif any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) <= 0
        for item in value
    ):
        raise ValueError(f"registration geometry {name} is invalid")
    return tuple(float(item) for item in value)


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _stain_layer(
    name: str,
    path: Path,
    artifact_digest: str,
    stain_map: _StainMapSpec,
    *,
    display_max_od: float,
    map_kind: str,
    render_mode: str = "overlay",
    selected_source: str | None = None,
    stain_family: str | None = None,
    adaptive_floor_od: float | None = None,
    adaptive_method: str | None = None,
    stain_artifact_kind: str = "physical",
) -> WsiLayer:
    width, height = stain_map.width, stain_map.height
    levels = _virtual_dzi_levels(width, height)
    digest_payload = {
        "artifact": artifact_digest,
        "display_max_od": display_max_od,
        "layer": name,
        "map_kind": map_kind,
        "render_mode": render_mode,
        "selected_source": selected_source,
        "stain_family": stain_family,
        "adaptive_floor_od": adaptive_floor_od,
        "adaptive_method": adaptive_method,
        "stain_artifact_kind": stain_artifact_kind,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return WsiLayer(
        name=name,
        path=path,
        digest=digest,
        levels=levels,
        tile_size=512,
        microns_per_pixel=stain_map.analysis_mpp,
        analysis_mpp=stain_map.analysis_mpp,
        display_max_od=display_max_od,
        map_kind=map_kind,
        content_origin_native_xy=stain_map.content_origin_native_xy,
        source_shape=(width, height),
        render_mode=render_mode,
        selected_source=selected_source,
        stain_family=stain_family,
        adaptive_floor_od=adaptive_floor_od,
        adaptive_method=adaptive_method,
        stain_artifact_kind=stain_artifact_kind,
        stain_artifact_digest=artifact_digest,
    )


def _virtual_dzi_levels(width: int, height: int) -> tuple[WsiLevel, ...]:
    max_level = (
        math.ceil(math.log2(max(width, height))) if max(width, height) > 1 else 0
    )
    return tuple(
        WsiLevel(
            width=max(1, math.ceil(width / 2 ** (max_level - level))),
            height=max(1, math.ceil(height / 2 ** (max_level - level))),
            source_level=max_level - level,
            source_kind="stain",
        )
        for level in range(max_level + 1)
    )


def _image_layer(
    name: str,
    path: Path,
    digest: str,
    *,
    crop_bbox_xywh: tuple[int, int, int, int] | None = None,
) -> WsiLayer:
    source_levels, microns_per_pixel = _discover_levels(path)
    full = source_levels[-1]
    source_shape = (full.width, full.height)
    target_width, target_height = full.width, full.height
    if crop_bbox_xywh is not None:
        _, _, crop_width, crop_height = crop_bbox_xywh
        target_width, target_height = crop_width, crop_height
    levels = _canonical_image_levels(
        target_width,
        target_height,
        source_levels,
        source_shape=source_shape,
    )
    return WsiLayer(
        name=name,
        path=path,
        digest=digest,
        levels=levels,
        tile_size=512,
        microns_per_pixel=microns_per_pixel,
        crop_bbox_xywh=crop_bbox_xywh,
        source_shape=source_shape,
    )


def _mask_layer(
    mask_path: Path,
    raw: WsiLayer,
) -> WsiLayer:
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    return WsiLayer(
        name="mask",
        path=raw.path,
        digest=digest,
        levels=raw.levels,
        tile_size=raw.tile_size,
        microns_per_pixel=raw.microns_per_pixel,
        mask_path=mask_path,
    )


def _protein_tissue_support_layer(mask: WsiLayer) -> WsiLayer:
    """Describe a neutral tissue silhouette without implying a stain value."""

    if mask.mask_path is None:
        raise ValueError("protein tissue support requires a tissue mask")
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "mask": mask.digest,
                "palette": "neutral-warm-tissue-support-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return WsiLayer(
        name="protein_tissue_support",
        path=mask.path,
        digest=digest,
        levels=mask.levels,
        tile_size=mask.tile_size,
        microns_per_pixel=mask.microns_per_pixel,
        mask_path=mask.mask_path,
        render_mode="neutral_stain_free_support",
    )


def _cell_layer(path: Path, digest: str, raw: WsiLayer) -> WsiLayer:
    source_levels, _mpp = _discover_levels(path)
    full = source_levels[-1]
    raw_full = raw.levels[-1]
    if (full.width, full.height) != (raw_full.width, raw_full.height):
        raise ValueError("cell labels do not match the native content-box geometry")
    return WsiLayer(
        name="cells",
        path=path,
        digest=digest,
        levels=_canonical_image_levels(
            full.width,
            full.height,
            source_levels,
            source_shape=(full.width, full.height),
        ),
        tile_size=raw.tile_size,
        microns_per_pixel=raw.microns_per_pixel,
        source_shape=(full.width, full.height),
        label_path=path,
        label_artifact_digest=digest,
    )


def _protein_layer(
    name: str,
    prediction_path: Path,
    prediction_digest: str,
    label_path: Path,
    label_digest: str,
    cells: WsiLayer,
    *,
    value_kind: str,
    section: str,
    target_id: str,
    model_fingerprint: str,
    prediction_fingerprint: str | None = None,
    display_max: float = 1.0,
    section_display_max: float | None = None,
    analysis_mpp: float | None = None,
    mask_path: Path | None = None,
    focus_bbox_fraction: tuple[float, float, float, float] | None = None,
) -> WsiLayer:
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 2,
                "render_palette": "expression-dab-diagnostics-v3-opaque-support",
                "layer": name,
                "predictions": prediction_digest,
                "labels": label_digest,
                "value_kind": value_kind,
                "section": section,
                "target_id": target_id,
                "model_fingerprint": model_fingerprint,
                "display_max": display_max,
                "section_display_max": section_display_max,
                "focus_bbox_fraction": focus_bbox_fraction,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    levels = cells.levels
    microns_per_pixel = cells.microns_per_pixel
    if analysis_mpp is not None:
        if microns_per_pixel is None:
            raise ValueError("dense protein layers require native pixel size")
        full = cells.levels[-1]
        levels = _virtual_dzi_levels(
            max(1, math.ceil(full.width * microns_per_pixel / analysis_mpp)),
            max(1, math.ceil(full.height * microns_per_pixel / analysis_mpp)),
        )
        microns_per_pixel = analysis_mpp
    return WsiLayer(
        name=name,
        path=prediction_path,
        digest=digest,
        levels=levels,
        tile_size=cells.tile_size,
        microns_per_pixel=microns_per_pixel,
        source_shape=cells.source_shape,
        map_kind=value_kind,
        analysis_mpp=analysis_mpp,
        display_max_od=display_max,
        section_display_max_od=section_display_max,
        mask_path=mask_path,
        label_path=label_path,
        label_artifact_digest=label_digest,
        protein_target=target_id,
        protein_model_fingerprint=model_fingerprint,
        prediction_section=section,
        prediction_artifact_digest=prediction_digest,
        prediction_fingerprint=prediction_fingerprint,
        focus_bbox_fraction=focus_bbox_fraction,
    )


def _observed_target_layer(
    name: str,
    map_path: Path,
    map_artifact_digest: str,
    source_map: _StainMapSpec,
    *,
    source_slide: dict[str, object],
    target_slide: dict[str, object],
    reference_slide: dict[str, object],
    source_section: str,
    source_label: str,
    target_section: str,
    target_id: str,
    calibration: dict[str, object] | None,
    calibration_fingerprint: str | None,
    adaptive_floor_od: float | None,
    adaptive_method: str | None,
    stain_artifact_kind: str,
    display_max_od: float,
    registration_result_sha256: str,
    target_mask_path: Path,
) -> WsiLayer:
    """Describe validated observed OD in a target section's native geometry."""

    import numpy as np

    if name not in _PROTEIN_OBSERVED_LAYERS:
        raise ValueError("unknown observed protein layer")
    if not target_mask_path.is_file():
        raise ValueError("observed protein comparison requires target tissue support")
    target_geometry = target_slide.get("geometry")
    target_bbox = _required_geometry_tuple(
        target_geometry, "content_bbox_xywh", integer=True
    )
    target_mpp = _required_geometry_tuple(target_geometry, "mpp_xy", integer=False)
    origin_x, origin_y, content_width, content_height = target_bbox
    analysis_mpp = source_map.analysis_mpp
    width = max(1, round(content_width * target_mpp[0] / analysis_mpp))
    height = max(1, round(content_height * target_mpp[1] / analysis_mpp))

    target_index_to_native = np.asarray(
        [
            [
                analysis_mpp / target_mpp[0],
                0.0,
                origin_x + analysis_mpp / (2.0 * target_mpp[0]),
            ],
            [
                0.0,
                analysis_mpp / target_mpp[1],
                origin_y + analysis_mpp / (2.0 * target_mpp[1]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_native_to_index = np.asarray(
        [
            [
                source_map.source_mpp_xy[0] / analysis_mpp,
                0.0,
                -source_map.content_origin_native_xy[0]
                * source_map.source_mpp_xy[0]
                / analysis_mpp
                - 0.5,
            ],
            [
                0.0,
                source_map.source_mpp_xy[1] / analysis_mpp,
                -source_map.content_origin_native_xy[1]
                * source_map.source_mpp_xy[1]
                / analysis_mpp
                - 0.5,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_native_to_reference = _native_to_reference_um_matrix(
        source_slide, reference_slide
    )
    target_native_to_reference = _native_to_reference_um_matrix(
        target_slide, reference_slide
    )
    matrix = (
        source_native_to_index
        @ np.linalg.inv(source_native_to_reference)
        @ target_native_to_reference
        @ target_index_to_native
    )
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("observed protein registration matrix is invalid")

    source_knots = (
        tuple(float(value) for value in calibration["source_knots"])
        if calibration is not None
        else (0.0, 1.0)
    )
    reference_knots = (
        tuple(float(value) for value in calibration["reference_knots"])
        if calibration is not None
        else (0.0, 1.0)
    )
    digest_payload = {
        "schema_version": 1,
        "render_palette": (
            "harmonized-adaptive-observed-dab-v2"
            if adaptive_floor_od is not None and calibration_fingerprint is not None
            else "adaptive-observed-dab-v1"
            if adaptive_floor_od is not None
            else "calibrated-observed-dab-v1"
        ),
        "layer": name,
        "map_artifact": map_artifact_digest,
        "registration_result": registration_result_sha256,
        "target_mask": _sha256_file(target_mask_path),
        "source_section": source_section,
        "target_section": target_section,
        "target_id": target_id,
        "analysis_mpp": analysis_mpp,
        "display_max_od": display_max_od,
        "target_size": [width, height],
        "target_map_to_source_map": matrix.tolist(),
        "calibration_fingerprint": calibration_fingerprint,
        "calibration_source_knots": source_knots,
        "calibration_reference_knots": reference_knots,
        "adaptive_floor_od": adaptive_floor_od,
        "adaptive_method": adaptive_method,
        "stain_artifact_kind": stain_artifact_kind,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return WsiLayer(
        name=name,
        path=map_path,
        digest=digest,
        levels=_virtual_dzi_levels(width, height),
        tile_size=512,
        microns_per_pixel=analysis_mpp,
        mask_path=target_mask_path,
        source_shape=(width, height),
        analysis_mpp=analysis_mpp,
        display_max_od=display_max_od,
        map_kind=(
            "target_od"
            if stain_artifact_kind == "adaptive"
            else "harmonized_adaptive_corrected_target_od"
            if adaptive_floor_od is not None and calibration_fingerprint is not None
            else "adaptive_corrected_target_od"
            if adaptive_floor_od is not None
            else "corrected_target_od"
        ),
        content_origin_native_xy=(int(origin_x), int(origin_y)),
        render_mode=(
            "same_section_observed"
            if source_section == target_section
            else "registered_nearest_observed"
        ),
        stain_family="h-dab",
        adaptive_floor_od=adaptive_floor_od,
        adaptive_method=adaptive_method,
        stain_artifact_kind=stain_artifact_kind,
        stain_artifact_digest=map_artifact_digest,
        protein_target=target_id,
        comparison_source_section=source_section,
        comparison_source_label=source_label,
        comparison_target_section=target_section,
        target_map_to_source_map=tuple(
            tuple(float(value) for value in row) for row in matrix
        ),
        calibration_source_knots=source_knots,
        calibration_reference_knots=reference_knots,
        calibration_fingerprint=calibration_fingerprint,
    )


def _native_to_reference_um_matrix(
    slide: dict[str, object], reference_slide: dict[str, object]
) -> object:
    """Return the approved source-native to reference-micrometre transform."""

    import numpy as np

    geometry = slide.get("geometry")
    reference_geometry = reference_slide.get("geometry")
    if not isinstance(geometry, dict) or not isinstance(reference_geometry, dict):
        raise ValueError("registration geometry is missing")
    transform = slide.get("transform")
    if not isinstance(transform, dict):
        raise ValueError("registration transform is missing")
    thumbnail_to_native = np.asarray(
        geometry.get("thumbnail_to_native"), dtype=np.float64
    )
    moving_to_reference = np.asarray(transform.get("matrix"), dtype=np.float64)
    reference_thumbnail_to_native = np.asarray(
        reference_geometry.get("thumbnail_to_native"), dtype=np.float64
    )
    reference_mpp = _required_geometry_tuple(
        reference_geometry, "mpp_xy", integer=False
    )
    matrices = (
        thumbnail_to_native,
        moving_to_reference,
        reference_thumbnail_to_native,
    )
    if any(matrix.shape != (3, 3) for matrix in matrices) or any(
        not np.all(np.isfinite(matrix)) for matrix in matrices
    ):
        raise ValueError("registration coordinate matrices are invalid")
    return (
        np.diag([reference_mpp[0], reference_mpp[1], 1.0])
        @ reference_thumbnail_to_native
        @ moving_to_reference
        @ np.linalg.inv(thumbnail_to_native)
    )


def _canonical_image_levels(
    width: int,
    height: int,
    source_levels: tuple[WsiLevel, ...],
    *,
    source_shape: tuple[int, int],
) -> tuple[WsiLevel, ...]:
    """Map canonical Deep Zoom geometry to the closest non-blurring source levels."""

    targets = _virtual_dzi_levels(width, height)
    source_width, source_height = source_shape
    mapped: list[WsiLevel] = []
    for target in targets:
        target_scale = max(target.width / width, target.height / height)
        eligible = [
            source
            for source in source_levels
            if min(
                source.width / source_width,
                source.height / source_height,
            )
            >= target_scale - 1e-12
        ]
        selected = min(
            eligible or [source_levels[-1]],
            key=lambda source: (source.width * source.height, source.source_level),
        )
        mapped.append(
            WsiLevel(
                width=target.width,
                height=target.height,
                source_level=selected.source_level,
                source_kind=selected.source_kind,
            )
        )
    return tuple(mapped)


@lru_cache(maxsize=1_024)
def _discover_levels(path: Path) -> tuple[tuple[WsiLevel, ...], float | None]:
    """Discover immutable image geometry once per server process and path."""

    pyvips = _import_pyvips()
    image = pyvips.Image.new_from_file(str(path), access="random")
    full_to_small: list[tuple[int, int, int, str]] = []
    level_count = _integer_field(image, "openslide.level-count")
    if level_count is not None and level_count > 0:
        for source_level in range(level_count):
            width = _integer_field(image, f"openslide.level[{source_level}].width")
            height = _integer_field(image, f"openslide.level[{source_level}].height")
            if width is None or height is None:
                level_image = pyvips.Image.new_from_file(
                    str(path),
                    level=source_level,
                    access="random",
                )
                width, height = level_image.width, level_image.height
            full_to_small.append((width, height, source_level, "level"))
    else:
        page_count = _integer_field(image, "n-pages") or 1
        subifd_count = _integer_field(image, "n-subifds") or 0
        if subifd_count > 0:
            full_to_small.append((image.width, image.height, 0, "page"))
            for source_level in range(subifd_count):
                level_image = pyvips.Image.new_from_file(
                    str(path),
                    subifd=source_level,
                    access="random",
                )
                full_to_small.append(
                    (
                        level_image.width,
                        level_image.height,
                        source_level,
                        "subifd",
                    )
                )
        else:
            for source_level in range(page_count):
                level_image = pyvips.Image.new_from_file(
                    str(path),
                    page=source_level,
                    n=1,
                    access="random",
                )
                full_to_small.append(
                    (level_image.width, level_image.height, source_level, "page")
                )
    unique: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for row in full_to_small:
        if row[:2] not in seen:
            unique.append(row)
            seen.add(row[:2])
    unique.sort(key=lambda row: row[0] * row[1])
    levels = tuple(
        WsiLevel(
            width=width,
            height=height,
            source_level=source_level,
            source_kind=source_kind,
        )
        for width, height, source_level, source_kind in unique
    )
    if not levels:
        raise ValueError(f"WSI contains no readable pyramid levels: {path}")
    xres = float(image.xres) if image.xres and image.xres > 0 else 0.0
    microns_per_pixel = 1000.0 / xres if xres > 0 else None
    return levels, microns_per_pixel


def _render_layer_tile(layer: WsiLayer, level: int, x: int, y: int) -> bytes:
    pyvips = _import_pyvips()
    target = layer.levels[level]
    if (
        layer.name == "cells"
        and layer.levels[-1].width > 10_000
        and target.width < 2_048
    ):
        return _transparent_tile(layer, target, x, y)
    if layer.mask_path is not None:
        image = pyvips.Image.new_from_file(str(layer.mask_path), access="random")
    elif layer.name == "cells":
        # Generic TIFF pyramids average categorical IDs. Derive every review
        # level from native labels with nearest-neighbor sampling instead.
        image = pyvips.Image.new_from_file(str(layer.path), access="random")
    else:
        loader_options = {"access": "random"}
        if target.source_kind == "level":
            loader_options["level"] = target.source_level
        elif target.source_kind == "subifd":
            loader_options["subifd"] = target.source_level
        else:
            loader_options["page"] = target.source_level
            loader_options["n"] = 1
        image = pyvips.Image.new_from_file(str(layer.path), **loader_options)
    if layer.crop_bbox_xywh is not None:
        if layer.source_shape is None:
            raise ValueError("cropped WSI layer has no source shape")
        crop_x, crop_y, _, _ = layer.crop_bbox_xywh
        source_width, source_height = layer.source_shape
        crop_left = int(round(crop_x * image.width / source_width))
        crop_top = int(round(crop_y * image.height / source_height))
        crop_width = max(
            1,
            int(round(layer.crop_bbox_xywh[2] * image.width / source_width)),
        )
        crop_height = max(
            1,
            int(round(layer.crop_bbox_xywh[3] * image.height / source_height)),
        )
        crop_width = min(crop_width, image.width - crop_left)
        crop_height = min(crop_height, image.height - crop_top)
        image = image.crop(crop_left, crop_top, crop_width, crop_height)
    if (image.width, image.height) != (target.width, target.height):
        kernel = (
            "nearest"
            if layer.mask_path is not None or layer.name == "cells"
            else "linear"
        )
        image = image.resize(
            target.width / image.width,
            vscale=target.height / image.height,
            kernel=kernel,
        )
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    tile = image.crop(left, top, width, height)
    if layer.name == "cells":
        return _render_cell_boundaries(tile)
    if layer.name == "protein_tissue_support":
        return _render_protein_tissue_support(tile)
    if layer.mask_path is not None:
        if tile.bands > 1:
            tile = tile[0]
        if tile.format != "uchar":
            tile = tile.cast("uchar")
        return bytes(tile.pngsave_buffer(compression=3, strip=True))
    tile = _normalize_rgb_uchar(tile)
    return bytes(tile.jpegsave_buffer(Q=90, strip=True, optimize_coding=True))


def _render_protein_tissue_support(tile: object) -> bytes:
    """Render mask support in a quiet neutral palette for stain-only review."""

    import numpy as np
    from PIL import Image

    if tile.bands > 1:
        tile = tile[0]
    if tile.format != "uchar":
        tile = tile.cast("uchar")
    tissue = (
        np.frombuffer(tile.write_to_memory(), dtype=np.uint8).reshape(
            tile.height, tile.width
        )
        > 0
    )
    rgba = np.zeros((*tissue.shape, 4), dtype=np.uint8)
    rgba[tissue] = (226, 218, 204, 255)
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def _render_stain_layer_tile(
    layer: WsiLayer,
    level: int,
    x: int,
    y: int,
    stain_map: object,
) -> bytes:
    """Render one transparent OD tile from an existing analysis map."""

    import numpy as np
    from PIL import Image

    from histopia.visualization._stain_viewer import _heatmap_rgba

    target = layer.levels[level]
    if layer.map_kind == "tissue_mask":
        values = np.asarray(stain_map.tissue_mask, dtype=np.float32)
    else:
        values = np.asarray(getattr(stain_map, str(layer.map_kind)), dtype=np.float32)
    tissue = np.asarray(stain_map.tissue_mask, dtype=bool)
    target_size = (target.width, target.height)
    if target_size != (values.shape[1], values.shape[0]):
        values = np.asarray(
            Image.fromarray(values, mode="F").resize(
                target_size,
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        tissue = (
            np.asarray(
                Image.fromarray(tissue.astype(np.uint8) * 255).resize(
                    target_size,
                    resample=Image.Resampling.NEAREST,
                )
            )
            > 127
        )
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    tile_values = values[top : top + height, left : left + width]
    tile_tissue = tissue[top : top + height, left : left + width]
    if layer.render_mode == "support":
        rgba = np.zeros((*tile_tissue.shape, 4), dtype=np.uint8)
        rgba[tile_tissue] = (38, 142, 91, 190)
    else:
        if layer.adaptive_floor_od is not None:
            tile_values = np.maximum(tile_values - layer.adaptive_floor_od, 0)
        rgba = _heatmap_rgba(
            tile_values,
            tile_tissue,
            float(layer.display_max_od),
            opaque_tissue=layer.render_mode == "opaque_tissue",
            family=layer.stain_family or "h-dab",
        )
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(
        stream,
        format="PNG",
        compress_level=3,
    )
    return stream.getvalue()


def _render_observed_target_tile(
    layer: WsiLayer,
    level: int,
    x: int,
    y: int,
    stain_map: object,
) -> bytes:
    """Warp sealed validated target OD at its measured 4-micron resolution."""

    import numpy as np
    from PIL import Image

    from histopia.visualization._stain_viewer import _heatmap_rgba

    matrix_value = layer.target_map_to_source_map
    source_knots = layer.calibration_source_knots
    reference_knots = layer.calibration_reference_knots
    if matrix_value is None or source_knots is None or reference_knots is None:
        raise ValueError("observed protein layer calibration is incomplete")
    matrix = np.asarray(matrix_value, dtype=np.float64)
    source_values = np.asarray(
        stain_map.target_od
        if layer.stain_artifact_kind == "adaptive"
        else stain_map.corrected_target_od,
        dtype=np.float32,
    )
    source_tissue = np.asarray(stain_map.tissue_mask, dtype=bool)
    if source_values.shape != source_tissue.shape:
        raise ValueError("observed stain values and tissue support do not align")
    if layer.adaptive_floor_od is not None:
        source_values = np.where(
            source_tissue,
            np.maximum(source_values - layer.adaptive_floor_od, 0.0),
            0.0,
        ).astype(np.float32)

    target = layer.levels[level]
    full = layer.levels[-1]
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    full_x = (
        left + np.arange(width, dtype=np.float64) + 0.5
    ) * full.width / target.width - 0.5
    full_y = (
        top + np.arange(height, dtype=np.float64) + 0.5
    ) * full.height / target.height - 0.5
    xx, yy = np.meshgrid(full_x, full_y)
    denominator = matrix[2, 0] * xx + matrix[2, 1] * yy + matrix[2, 2]
    finite = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    source_x = np.full(xx.shape, -1.0, dtype=np.float64)
    source_y = np.full(yy.shape, -1.0, dtype=np.float64)
    source_x[finite] = (
        matrix[0, 0] * xx[finite] + matrix[0, 1] * yy[finite] + matrix[0, 2]
    ) / denominator[finite]
    source_y[finite] = (
        matrix[1, 0] * xx[finite] + matrix[1, 1] * yy[finite] + matrix[1, 2]
    ) / denominator[finite]
    sampled, source_support = _sample_calibrated_od(
        source_values,
        source_tissue,
        source_x,
        source_y,
        source_knots=source_knots,
        reference_knots=reference_knots,
    )
    target_support = np.asarray(
        _read_protein_overview_mask(layer, level, x, y), dtype=bool
    )
    support = source_support & target_support & np.isfinite(sampled)
    values = np.where(support, sampled, 0.0).astype(np.float32)
    rgba = _heatmap_rgba(
        values,
        support,
        float(layer.display_max_od or 1.0),
        opaque_tissue=True,
        family="h-dab",
    )
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def _sample_calibrated_od(
    values: object,
    tissue: object,
    x: object,
    y: object,
    *,
    source_knots: tuple[float, ...],
    reference_knots: tuple[float, ...],
) -> tuple[object, object]:
    """Bilinearly warp a calibrated field and nearest-sample its support."""

    import numpy as np

    source = np.asarray(values, dtype=np.float32)
    source_tissue = np.asarray(tissue, dtype=bool)
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    inside = (
        np.isfinite(xx)
        & np.isfinite(yy)
        & (xx >= 0)
        & (yy >= 0)
        & (xx <= source.shape[1] - 1)
        & (yy <= source.shape[0] - 1)
    )
    x0 = np.clip(np.floor(xx).astype(np.int64), 0, source.shape[1] - 1)
    y0 = np.clip(np.floor(yy).astype(np.int64), 0, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    dx = xx - x0
    dy = yy - y0
    q00 = _apply_od_calibration(source[y0, x0], source_knots, reference_knots)
    q10 = _apply_od_calibration(source[y0, x1], source_knots, reference_knots)
    q01 = _apply_od_calibration(source[y1, x0], source_knots, reference_knots)
    q11 = _apply_od_calibration(source[y1, x1], source_knots, reference_knots)
    sampled = (
        q00 * (1.0 - dx) * (1.0 - dy)
        + q10 * dx * (1.0 - dy)
        + q01 * (1.0 - dx) * dy
        + q11 * dx * dy
    )
    nearest_x = np.clip(np.floor(xx + 0.5).astype(np.int64), 0, source.shape[1] - 1)
    nearest_y = np.clip(np.floor(yy + 0.5).astype(np.int64), 0, source.shape[0] - 1)
    support = inside & source_tissue[nearest_y, nearest_x]
    return np.where(inside, sampled, np.nan).astype(np.float32), support


def _apply_od_calibration(
    values: object,
    source_knots: tuple[float, ...],
    reference_knots: tuple[float, ...],
) -> object:
    """Apply the sealed zero-anchored monotonic calibration with tail slope."""

    import numpy as np

    raw = np.asarray(values, dtype=np.float64)
    source = np.asarray(source_knots, dtype=np.float64)
    reference = np.asarray(reference_knots, dtype=np.float64)
    mapped = np.interp(raw, source, reference)
    high = raw > source[-1]
    if np.any(high):
        tail = max(
            (reference[-1] - reference[-2]) / (source[-1] - source[-2]),
            0.0,
        )
        mapped[high] = reference[-1] + tail * (raw[high] - source[-1])
    return np.maximum(mapped, 0.0)


def _protein_cell_values(
    layer: WsiLayer, predictions: object
) -> tuple[object, object, object, str]:
    """Return prediction IDs, normalized display values, validity, and kind."""

    import numpy as np

    prediction_labels = np.asarray(predictions.label_ids, dtype=np.int64)
    supported = np.asarray(predictions.supported, dtype=bool)
    kind = str(layer.map_kind)
    if kind == "residual":
        measured = np.asarray(predictions.measured_od, dtype=np.float64)
        predicted = np.asarray(predictions.predicted_od_reference, dtype=np.float64)
        values = np.abs(measured - predicted)
    else:
        values = np.asarray(getattr(predictions, kind), dtype=np.float64)
    valid = supported & np.isfinite(values)
    if kind not in {"relative_expression", "expression_probability"}:
        display_max = float(layer.display_max_od or 1.0)
        values = values / max(display_max, 1e-8)
    return prediction_labels, np.clip(values, 0.0, 1.0), valid, kind


def _build_protein_overview(
    layer: WsiLayer,
    predictions: object,
    *,
    label_overview: object | None = None,
) -> _ProteinOverview:
    """Rasterize real cell values into a bounded, non-tile-pooled overview."""

    import numpy as np

    labels = np.asarray(
        _build_protein_label_overview(layer)
        if label_overview is None
        else label_overview,
        dtype=np.uint32,
    )
    if labels.ndim != 2:
        raise ValueError("protein label overview must be two-dimensional")
    prediction_labels, values, valid, _kind = _protein_cell_values(layer, predictions)
    maximum = max(int(labels.max(initial=0)), int(prediction_labels.max(initial=0)))
    lookup = np.zeros(maximum + 1, dtype=np.float32)
    support_lookup = np.zeros(maximum + 1, dtype=bool)
    lookup[prediction_labels[valid]] = values[valid]
    support_lookup[prediction_labels[valid]] = True
    return _ProteinOverview(
        values=lookup[labels],
        supported=support_lookup[labels] & (labels > 0),
    )


def _build_protein_label_overview(layer: WsiLayer) -> object:
    """Downsample native categorical labels once for all low-zoom panes."""

    import numpy as np

    if (
        layer.label_path is None
        or layer.source_shape is None
        or layer.microns_per_pixel is None
    ):
        raise ValueError("protein overview geometry is incomplete")
    source_width, source_height = layer.source_shape
    analysis_mpp = 4.0
    physical_width = max(
        1, math.ceil(source_width * layer.microns_per_pixel / analysis_mpp)
    )
    physical_height = max(
        1, math.ceil(source_height * layer.microns_per_pixel / analysis_mpp)
    )
    bounded_scale = min(1.0, 2_048 / max(physical_width, physical_height))
    width = max(1, round(physical_width * bounded_scale))
    height = max(1, round(physical_height * bounded_scale))
    pyvips = _import_pyvips()
    image = pyvips.Image.new_from_file(str(layer.label_path), access="sequential")
    if image.bands > 1:
        image = image[0]
    focus = layer.focus_bbox_fraction
    if focus is None:
        image = image.resize(
            width / image.width,
            vscale=height / image.height,
            kernel="nearest",
        )
        if (image.width, image.height) != (width, height):
            image = image.crop(0, 0, min(image.width, width), min(image.height, height))
            image = image.embed(0, 0, width, height, extend="black")
    else:
        focus_x, focus_y, focus_width, focus_height = focus
        source_left = max(0, math.floor(focus_x * image.width))
        source_top = max(0, math.floor(focus_y * image.height))
        source_right = min(
            image.width, math.ceil((focus_x + focus_width) * image.width)
        )
        source_bottom = min(
            image.height, math.ceil((focus_y + focus_height) * image.height)
        )
        target_left = max(0, math.floor(focus_x * width))
        target_top = max(0, math.floor(focus_y * height))
        target_right = min(width, math.ceil((focus_x + focus_width) * width))
        target_bottom = min(height, math.ceil((focus_y + focus_height) * height))
        source_crop_width = max(1, source_right - source_left)
        source_crop_height = max(1, source_bottom - source_top)
        target_width = max(1, target_right - target_left)
        target_height = max(1, target_bottom - target_top)
        image = image.crop(
            source_left,
            source_top,
            source_crop_width,
            source_crop_height,
        ).resize(
            target_width / source_crop_width,
            vscale=target_height / source_crop_height,
            kernel="nearest",
        )
        if (image.width, image.height) != (target_width, target_height):
            image = image.crop(
                0,
                0,
                min(image.width, target_width),
                min(image.height, target_height),
            ).embed(0, 0, target_width, target_height, extend="black")
        image = image.embed(
            target_left,
            target_top,
            width,
            height,
            extend="black",
        )
    if image.format != "uint":
        image = image.cast("uint")
    labels = np.frombuffer(image.write_to_memory(), dtype=np.uint32).reshape(
        height, width
    )
    return labels


def _render_protein_layer_tile(
    layer: WsiLayer,
    level: int,
    x: int,
    y: int,
    predictions: object,
    *,
    overview: _ProteinOverview | None = None,
) -> bytes:
    """Render one transparent, per-native-cell protein prediction tile."""

    import numpy as np
    from PIL import Image

    prediction_labels, values, valid, kind = _protein_cell_values(layer, predictions)
    target = layer.levels[level]
    if layer.mask_path is not None and target.width < 2_048:
        if overview is None:
            tissue = _read_protein_overview_mask(layer, level, x, y)
            summary = float(np.median(values[valid])) if np.any(valid) else 0.0
            tile_values = np.full(tissue.shape, summary, dtype=np.float32)
            tile_support = tissue
        else:
            tile_values, tile_support = _read_spatial_protein_overview(
                layer, level, x, y, overview
            )
        rgba = _protein_heatmap_rgba(tile_values, tile_support, kind=kind)
        stream = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=3)
        return stream.getvalue()
    labels = _read_protein_label_tile(layer, level, x, y)
    maximum = max(int(labels.max(initial=0)), int(prediction_labels.max(initial=0)))
    lookup = np.zeros(maximum + 1, dtype=np.float32)
    support_lookup = np.zeros(maximum + 1, dtype=bool)
    lookup[prediction_labels[valid]] = values[valid]
    support_lookup[prediction_labels[valid]] = True
    tile_values = lookup[labels]
    tile_support = support_lookup[labels] & (labels > 0)
    rgba = _protein_heatmap_rgba(tile_values, tile_support, kind=kind)
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def _read_spatial_protein_overview(
    layer: WsiLayer,
    level: int,
    x: int,
    y: int,
    overview: _ProteinOverview,
) -> tuple[object, object]:
    """Read one seam-free tile from a bounded cell-resolved overview."""

    import numpy as np
    from PIL import Image

    target = layer.levels[level]
    values = np.asarray(overview.values, dtype=np.float32)
    supported = np.asarray(overview.supported, dtype=bool)
    target_size = (target.width, target.height)
    if target_size != (values.shape[1], values.shape[0]):
        values = np.asarray(
            Image.fromarray(values, mode="F").resize(
                target_size, resample=Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        supported = (
            np.asarray(
                Image.fromarray(supported.astype(np.uint8) * 255).resize(
                    target_size, resample=Image.Resampling.NEAREST
                )
            )
            > 127
        )
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    return (
        values[top : top + height, left : left + width],
        supported[top : top + height, left : left + width],
    )


def _read_protein_overview_mask(layer: WsiLayer, level: int, x: int, y: int) -> object:
    """Read a cheap tissue silhouette below resolvable native cell scale."""

    import numpy as np
    from PIL import Image

    if layer.mask_path is None:
        raise ValueError("protein overview has no tissue mask")
    target = layer.levels[level]
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    with Image.open(layer.mask_path) as source:
        mask = source.convert("L").resize(
            (target.width, target.height), resample=resampling
        )
        tile = mask.crop((left, top, left + width, top + height))
    return np.asarray(tile) > 0


def _read_protein_label_tile(layer: WsiLayer, level: int, x: int, y: int) -> object:
    import numpy as np

    if layer.label_path is None:
        raise ValueError("protein layer has no cell-label source")
    pyvips = _import_pyvips()
    target = layer.levels[level]
    # Never consume averaged categorical sub-IFDs.
    image = pyvips.Image.new_from_file(str(layer.label_path), access="random")
    if image.bands > 1:
        image = image[0]
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    scale_x = image.width / target.width
    scale_y = image.height / target.height
    source_left = min(image.width - 1, math.floor(left * scale_x))
    source_top = min(image.height - 1, math.floor(top * scale_y))
    source_right = min(image.width, math.ceil((left + width) * scale_x))
    source_bottom = min(image.height, math.ceil((top + height) * scale_y))
    tile = image.crop(
        source_left,
        source_top,
        max(1, source_right - source_left),
        max(1, source_bottom - source_top),
    )
    if (tile.width, tile.height) != (width, height):
        tile = tile.resize(
            width / tile.width,
            vscale=height / tile.height,
            kernel="nearest",
        )
        if (tile.width, tile.height) != (width, height):
            tile = tile.crop(0, 0, min(tile.width, width), min(tile.height, height))
            tile = tile.embed(0, 0, width, height, extend="copy")
    if tile.format != "uint":
        tile = tile.cast("uint")
    return np.frombuffer(tile.write_to_memory(), dtype=np.uint32).reshape(
        tile.height, tile.width
    )


def _transparent_tile(layer: WsiLayer, target: WsiLevel, x: int, y: int) -> bytes:
    """Return a cheap categorical overview tile below useful cell resolution."""

    from io import BytesIO

    import numpy as np
    from PIL import Image

    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    stream = BytesIO()
    Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode="RGBA").save(
        stream, format="PNG", compress_level=3
    )
    return stream.getvalue()


def _protein_heatmap_rgba(
    values: object, supported: object, *, kind: str = "relative_expression"
) -> object:
    import numpy as np

    normalized = np.asarray(values, dtype=np.float32)
    mask = np.asarray(supported, dtype=bool)
    rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)
    if kind == "uncertainty":
        low = np.asarray([245.0, 247.0, 250.0])
        middle = np.asarray([121.0, 149.0, 180.0])
        high = np.asarray([48.0, 63.0, 96.0])
    elif kind == "residual":
        low = np.asarray([255.0, 247.0, 242.0])
        middle = np.asarray([226.0, 122.0, 92.0])
        high = np.asarray([142.0, 29.0, 29.0])
    else:
        # DAB-like white/tan/brown for measured and predicted expression.
        low = np.asarray([250.0, 246.0, 235.0])
        middle = np.asarray([190.0, 133.0, 72.0])
        high = np.asarray([92.0, 48.0, 20.0])
    first = np.minimum(normalized * 2.0, 1.0)[..., None]
    second = np.maximum(normalized * 2.0 - 1.0, 0.0)[..., None]
    rgb = low * (1.0 - first) + middle * first
    rgb = rgb * (1.0 - second) + high * second
    rgba[..., :3] = np.rint(rgb).astype(np.uint8)
    # Expression magnitude is already encoded by the calibrated color scale.
    # Opacity encodes support only so stain-only panes do not double-attenuate
    # low values or become invisible against the white review background.
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba


def _render_cell_boundaries(tile) -> bytes:  # type: ignore[no-untyped-def]
    from io import BytesIO

    import numpy as np
    from PIL import Image

    if tile.bands > 1:
        tile = tile[0]
    if tile.format != "uint":
        tile = tile.cast("uint")
    labels = np.frombuffer(tile.write_to_memory(), dtype=np.uint32).reshape(
        tile.height, tile.width
    )
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (
        (labels[1:, :] > 0) | (labels[:-1, :] > 0)
    )
    boundary[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (
        (labels[:, 1:] > 0) | (labels[:, :-1] > 0)
    )
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    rgba[boundary] = (255, 36, 24, 235)
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def _normalize_rgb_uchar(image):  # type: ignore[no-untyped-def]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    elif image.bands == 2:
        image = image[0].bandjoin([image[0], image[0]])
    elif image.bands > 3:
        image = image[:3]
    if image.format != "uchar":
        image = image.cast("uchar")
    return image


def _integer_field(image, name: str) -> int | None:  # type: ignore[no-untyped-def]
    try:
        value = image.get(name)
    except Exception:  # libvips exposes loader-specific metadata dynamically.
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_identity_digest(path: Path) -> str:
    stat = path.stat()
    payload = {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_content_bbox(
    geometry: object,
) -> tuple[int, int, int, int] | None:
    if not isinstance(geometry, dict):
        return None
    value = geometry.get("content_bbox_xywh")
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or item < 0 for item in value)
        or value[2] < 1
        or value[3] < 1
    ):
        return None
    return tuple(value)


def _geometry_native_shape(geometry: object) -> tuple[int, int]:
    if not isinstance(geometry, dict):
        raise ValueError("registration source geometry is missing")
    value = geometry.get("native_shape")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 1 for item in value)
    ):
        raise ValueError("registration native source shape is invalid")
    return value[0], value[1]


def _marker_label(stem: str) -> str:
    match = re.search(r"panc[_-](.+?)(?:-\[|$)", stem, flags=re.IGNORECASE)
    label = match.group(1) if match else stem
    canonical = re.sub(r"[^a-z0-9]", "", label.lower())
    return {
        "yap": "YAP",
        "ecad": "E-Cad",
        "ecadherin": "E-Cad",
        "ck19": "CK19",
        "ki67": "Ki67",
        "perk": "pERK",
        "ncad": "N-Cad",
        "ncadherin": "N-Cad",
        "cjun": "cJun",
    }.get(canonical, label)


def _import_pyvips():
    try:
        import pyvips
    except (ImportError, OSError) as error:
        raise OptionalDependencyError("pyvips", "wsi") from error
    return pyvips
