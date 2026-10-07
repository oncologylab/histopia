"""Fingerprint-safe derivation of adaptive outputs from validated OD arrays."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.stain._adaptive import (
    infer_adaptive_background,
    infer_counterstain_conditioned_background,
)
from histopia.stain._artifacts import StainMap
from histopia.stain._assays import StainFamily
from histopia.stain._config import StainQuantificationConfig
from histopia.stain._model import StainModel
from histopia.stain._pipeline import (
    _counterstain_adaptive_artifact,
    _json_sha256,
    _result_slide_row,
)
from histopia.stain._preflight import preflight_stain_run, write_stain_preflight
from histopia.stain._result import write_stain_result
from histopia.stain._result_validation import validate_stain_result


def derive_adaptive_stain_run(
    config: StainQuantificationConfig,
    source_run: Path | str,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Add gated adaptive metadata without recomputing physical stain OD.

    The source and destination must have identical quantitative selection and
    exact per-slide source, mask, transform, geometry, and physical resolution.
    Raw/corrected arrays are copied bit-for-bit after loading.  A changed assay
    selection must instead be fully refit because it can change cohort vectors.
    """

    if config.adaptive_background not in {
        "inferred_floor",
        "counterstain_conditioned",
    }:
        raise ValueError(
            "adaptive derivation requires inferred_floor or "
            "counterstain_conditioned configuration"
        )
    source_root = Path(source_run).expanduser().resolve()
    source = validate_stain_result(source_root)
    preflight = preflight_stain_run(config)
    source_rows = source.get("slides")
    if not isinstance(source_rows, list) or len(source_rows) != preflight.slide_count:
        raise ValueError("source stain result does not match destination slide count")
    source_by_id = {
        str(row.get("id")): row for row in source_rows if isinstance(row, dict)
    }
    new_quantified = {
        slide.slide_name
        for slide in preflight.slides
        if slide.assay.analysis_included
        and slide.assay.family is not StainFamily.CONTEXT_HE
    }
    old_quantified = {
        str(row.get("id"))
        for row in source_rows
        if isinstance(row, dict) and row.get("quantified") is True
    }
    if new_quantified != old_quantified:
        raise ValueError(
            "adaptive derivation cannot change quantitative slide selection; refit"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_stain_preflight(preflight, config.output_dir / "preflight.json")
    benchmark = _rewrite_benchmark(
        source_root / str(source["benchmark"]),
        config.output_dir / "benchmark.json",
        preflight.fingerprint,
    )
    rows: list[dict[str, object]] = []
    for order, slide in enumerate(preflight.slides, start=1):
        old_row = source_by_id.get(slide.slide_name)
        if old_row is None or old_row.get("order") != order:
            raise ValueError("source stain slide order differs from destination")
        if slide.assay.family is StainFamily.CONTEXT_HE:
            rows.append(_nonquantified_row(slide, order, "context_only"))
            continue
        if not slide.assay.analysis_included:
            rows.append(_nonquantified_row(slide, order, "analysis_excluded"))
            continue
        if progress is not None:
            progress(f"[derive {order}/{preflight.slide_count}] {slide.slide_name}")
        rows.append(
            _derive_slide(
                config,
                source_root,
                old_row,
                slide,
                order,
                preflight.fingerprint,
            )
        )

    measurement = dict(source.get("measurement", {}))
    measurement["analysis_mpp"] = config.analysis_mpp
    core = {
        "schema_version": 1,
        "measurement": measurement,
        "preflight_fingerprint": preflight.fingerprint,
        "registration_result_sha256": preflight.registration_result_sha256,
        "registration_approval_sha256": preflight.registration_approval_sha256,
        "preflight": "preflight.json",
        "benchmark": "benchmark.json",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "families": benchmark["families"],
        "runtime": {
            "workers": 1,
            "vips_threads": 0,
            "derived_without_od_recomputation": True,
        },
        "slides": rows,
    }
    return write_stain_result(config.output_dir, core)


def _derive_slide(
    config: StainQuantificationConfig,
    source_root: Path,
    old_row: dict[str, object],
    slide: object,
    order: int,
    preflight_fingerprint: str,
) -> dict[str, object]:
    old_map_path = source_root / str(old_row["map"])
    old_model_path = source_root / str(old_row["model"])
    old_map = StainMap.load(old_map_path)
    model = StainModel.from_json_dict(json.loads(old_model_path.read_text()))
    _validate_exact_inputs(old_map, slide, config.analysis_mpp)
    selected = (
        old_map.corrected_target_od
        if model.correction_accepted
        else old_map.raw_target_od
    )
    if config.adaptive_background == "counterstain_conditioned":
        adaptive = infer_counterstain_conditioned_background(
            selected,
            old_map.counterstain_od,
            old_map.tissue_mask,
            confidence=old_map.confidence,
            seed=config.seed,
        )
    else:
        adaptive = infer_adaptive_background(
            selected,
            old_map.counterstain_od,
            old_map.tissue_mask,
            seed=config.seed,
        )
    new_model = replace(model, adaptive_background=adaptive.to_json_dict())
    provenance = dict(old_map.provenance)
    provenance.update(
        {
            "preflight_fingerprint": preflight_fingerprint,
            "adaptive_background": config.adaptive_background,
            "derived_from_content_fingerprint": old_map.content_fingerprint,
        }
    )
    new_map = replace(
        old_map,
        provenance=provenance,
        fingerprint=None,
        content_fingerprint=None,
    )
    map_relative = Path(str(old_row["map"]))
    model_relative = Path(str(old_row["model"]))
    adaptive_map_relative = Path("adaptive_maps") / map_relative.name
    map_path = config.output_dir / map_relative
    model_path = config.output_dir / model_relative
    adaptive_map_path = config.output_dir / adaptive_map_relative
    new_map.save(map_path)
    write_json_atomic(model_path, new_model.to_json_dict())
    for name in ("raw_target_od", "corrected_target_od"):
        if not np.array_equal(getattr(old_map, name), getattr(new_map, name)):
            raise RuntimeError(f"adaptive derivation changed physical array: {name}")
    adaptive_map = _counterstain_adaptive_artifact(
        new_map,
        new_model,
        adaptive_map_path,
    )
    row = _result_slide_row(
        config.output_dir,
        order,
        slide,
        new_model,
        new_map,
        map_path,
        model_path,
        adaptive_map=adaptive_map,
        adaptive_map_path=(adaptive_map_path if adaptive_map is not None else None),
    )
    legacy_adaptive = model.adaptive_background
    if (
        config.adaptive_background == "counterstain_conditioned"
        and isinstance(legacy_adaptive, dict)
        and legacy_adaptive.get("method") == "inferred_floor-v2"
    ):
        qc = row.get("qc")
        if isinstance(qc, dict):
            qc["legacy_adaptive_background"] = dict(legacy_adaptive)
        legacy_quantiles = old_row.get("adaptive_quantiles")
        if isinstance(legacy_quantiles, dict):
            row["legacy_adaptive_quantiles"] = dict(legacy_quantiles)
    return row


def _validate_exact_inputs(stain_map: StainMap, slide: object, mpp: float) -> None:
    provenance = stain_map.provenance
    checks = {
        "slide_id": slide.slide_name,
        "source_sha256": slide.source_sha256,
        "mask_sha256": slide.mask_sha256,
        "transform_sha256": slide.transform_sha256,
    }
    for key, expected in checks.items():
        if provenance.get(key) != expected:
            raise ValueError(f"source stain map has stale {key}")
    if not np.isclose(stain_map.analysis_mpp, mpp, atol=1e-9):
        raise ValueError("source stain map physical resolution differs")
    if stain_map.content_origin_native_xy != tuple(slide.content_bbox_xywh[:2]):
        raise ValueError("source stain map content origin differs")
    if stain_map.source_mpp_xy != tuple(slide.mpp_xy):
        raise ValueError("source stain map source geometry differs")


def _rewrite_benchmark(source: Path, destination: Path, fingerprint: str) -> dict:
    payload = json.loads(source.read_text())
    core = {key: value for key, value in payload.items() if key != "fingerprint"}
    core["preflight_fingerprint"] = fingerprint
    rewritten = {**core, "fingerprint": _json_sha256(core)}
    write_json_atomic(destination, rewritten)
    return rewritten


def _nonquantified_row(slide: object, order: int, flag: str) -> dict[str, object]:
    return {
        "id": slide.slide_name,
        "order": order,
        "marker": slide.assay.marker,
        "family": slide.assay.family.value,
        "batch_id": slide.assay.batch_id,
        "analysis_included": slide.assay.analysis_included,
        "exclusion_reason": slide.assay.exclusion_reason,
        "quantified": False,
        "map": None,
        "model": None,
        "qc": {"flags": [flag]},
    }
