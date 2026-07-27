"""Decision-focused review portal for quantitative stain viewer assets."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from histopia._atomic import write_json_atomic, write_text_atomic

_ASSET_PACKAGE = "histopia.visualization._stain_review_assets"


def build_stain_review(
    viewer_dir: Path | str,
    output_dir: Path | str,
    *,
    mice: Sequence[str] | None = None,
    issues: Mapping[str, Mapping[str, str]] | None = None,
) -> Path:
    """Build a static, fingerprint-bound stain decision portal.

    The portal reads only generated viewer assets. Draft slide decisions remain
    in browser storage and do not alter fingerprint-bound scientific approval.
    """

    viewer_dir = Path(viewer_dir).resolve()
    output_dir = Path(output_dir).resolve()
    payload = _load_viewer_manifest(viewer_dir)
    if isinstance(mice, str):
        raise TypeError("stain review mice must be a sequence of mouse IDs")
    selected = list(mice) if mice is not None else None
    if selected is not None and (not selected or len(selected) != len(set(selected))):
        raise ValueError("stain review mice must be unique and non-empty")
    issue_rows = _validated_issues(issues or {})
    available = {
        str(mouse["id"]): mouse
        for mouse in payload["mice"]
        if isinstance(mouse, dict) and isinstance(mouse.get("stain"), dict)
    }
    names = selected or list(available)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"viewer has no stain result for: {', '.join(missing)}")

    asset_base = _relative_href(output_dir, viewer_dir)
    mouse_rows = [
        _review_mouse(
            available[name],
            viewer_dir,
            asset_base,
            issue_rows.get(name, {}),
        )
        for name in names
    ]
    manifest = {
        "schema_version": 1,
        "scope": {
            "decision": "continuous_relative_target_optical_density",
            "analysis_mpp": sorted(
                {float(mouse["measurement"]["analysis_mpp"]) for mouse in mouse_rows}
            ),
            "excluded": [
                "binary_positive_negative_calls",
                "cross_antibody_normalization",
                "absolute_concentration",
                "single_cell_expression",
            ],
        },
        "viewer_href": f"{asset_base}index.html",
        "mice": mouse_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "manifest.json", manifest)
    encoded = json.dumps(manifest, separators=(",", ":"))
    write_text_atomic(
        output_dir / "manifest-data.js",
        f"globalThis.HISTOPIA_STAIN_REVIEW={encoded};\n",
    )
    for name in ("index.html", "stain-review.css", "stain-review.js"):
        write_text_atomic(
            output_dir / name,
            files(_ASSET_PACKAGE).joinpath(name).read_text(encoding="utf-8"),
        )
    return output_dir / "index.html"


def load_stain_review_issues(path: Path | str) -> dict[str, dict[str, str]]:
    """Load optional slide notes keyed by mouse then slide ID or order."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("stain review issues must be a JSON object")
    return _validated_issues(payload)


def _validated_issues(
    payload: Mapping[object, object],
) -> dict[str, dict[str, str]]:
    issues: dict[str, dict[str, str]] = {}
    for mouse_id, rows in payload.items():
        if not isinstance(mouse_id, str) or not isinstance(rows, Mapping):
            raise ValueError("stain review issues must map mouse IDs to objects")
        parsed: dict[str, str] = {}
        for slide, note in rows.items():
            if (
                not isinstance(slide, str)
                or not isinstance(note, str)
                or not note.strip()
            ):
                raise ValueError("stain review issue keys and notes must be strings")
            parsed[slide] = note.strip()
        issues[mouse_id] = parsed
    return issues


def _load_viewer_manifest(viewer_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((viewer_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("viewer manifest is missing or invalid") from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("mice"), list)
        or not payload["mice"]
    ):
        raise ValueError("viewer manifest contains no mice")
    return payload


def _review_mouse(
    mouse: dict[str, Any],
    viewer_dir: Path,
    asset_base: str,
    issues: Mapping[str, str],
) -> dict[str, object]:
    mouse_id = str(mouse["id"])
    stain = mouse["stain"]
    slides = [
        row
        for row in mouse.get("slides", [])
        if isinstance(row, dict)
        and isinstance(row.get("stain"), dict)
        and row["stain"].get("quantified") is True
    ]
    if not slides:
        raise ValueError(f"{mouse_id} has no quantified stain slides")
    review_slides = [
        _review_slide(mouse_id, row, viewer_dir, asset_base, issues) for row in slides
    ]
    _assign_priorities(review_slides)
    review_slides.sort(key=lambda row: int(row["order"]))
    families = stain.get("families")
    if not isinstance(families, dict):
        raise ValueError(f"{mouse_id} stain families are missing")
    family_rows = {}
    for name, row in families.items():
        if not isinstance(row, dict):
            raise ValueError(f"{mouse_id} stain family {name} is invalid")
        family_slides = [
            slide for slide in review_slides if slide["family"] == str(name)
        ]
        slide_count = _integer(row.get("slide_count"), "family slide count")
        if slide_count != len(family_slides):
            raise ValueError(f"{mouse_id} stain family {name} slide count differs")
        family_rows[str(name)] = {
            "selected_method": str(row.get("selected_method", "")),
            "slide_count": slide_count,
            "correction_accepted": sum(
                bool(slide["qc"]["correction_accepted"]) for slide in family_slides
            ),
            "threshold_accepted": sum(
                bool(slide["qc"]["threshold_accepted"]) for slide in family_slides
            ),
        }
    measurement = stain.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError(f"{mouse_id} stain measurement is missing")
    required = sum(bool(row["priority"]["required"]) for row in review_slides)
    blockers = sum(bool(row["priority"]["blocking"]) for row in review_slides)
    correction_count = sum(
        bool(row["qc"]["correction_accepted"]) for row in review_slides
    )
    threshold_count = sum(
        bool(row["qc"]["threshold_accepted"]) for row in review_slides
    )
    review = stain.get("review")
    fingerprint = str(stain.get("fingerprint", "")).strip()
    if not fingerprint:
        raise ValueError(f"{mouse_id} stain fingerprint is missing")
    display_max_od = _finite(stain.get("display_max_od"), "display_max_od")
    if display_max_od <= 0:
        raise ValueError("display_max_od must be positive")
    analysis_mpp = _finite(measurement.get("analysis_mpp"), "analysis_mpp")
    if analysis_mpp <= 0:
        raise ValueError("analysis_mpp must be positive")
    return {
        "id": mouse_id,
        "fingerprint": fingerprint,
        "display_max_od": display_max_od,
        "measurement": {
            "quantity": str(measurement.get("quantity", "")),
            "analysis_mpp": analysis_mpp,
        },
        "review": review if isinstance(review, dict) else {},
        "summary": {
            "quantified_slides": len(review_slides),
            "required_slides": required,
            "blocking_issues": blockers,
            "correction_accepted": correction_count,
            "threshold_accepted": threshold_count,
        },
        "families": family_rows,
        "slides": review_slides,
    }


def _review_slide(
    mouse_id: str,
    slide: dict[str, Any],
    viewer_dir: Path,
    asset_base: str,
    issues: Mapping[str, str],
) -> dict[str, object]:
    stain = slide["stain"]
    order = _integer(slide.get("order"), "slide order")
    slide_id = str(slide.get("id", "")).strip()
    if not slide_id:
        raise ValueError(f"{mouse_id} slide {order} has no ID")
    qc = stain.get("qc")
    if not isinstance(qc, dict):
        raise ValueError(f"{mouse_id} slide {order} has no stain QC")
    asset_paths = {
        "histology": slide.get("texture"),
        "raw": _nested(stain, "textures", "raw"),
        "corrected": _nested(stain, "textures", "corrected"),
        "raw_overlay": _nested(stain, "overlay_textures", "raw"),
        "corrected_overlay": _nested(stain, "overlay_textures", "corrected"),
    }
    assets = {
        name: _asset_href(viewer_dir, asset_base, path, mouse_id, order)
        for name, path in asset_paths.items()
    }
    issue = issues.get(slide_id) or issues.get(str(order))
    return {
        "id": slide_id,
        "order": order,
        "label": str(slide.get("label") or stain.get("marker") or slide_id),
        "family": str(stain.get("family", "")),
        "assets": assets,
        "qc": {
            "correction_accepted": _boolean(
                qc.get("correction_accepted"),
                "correction_accepted",
            ),
            "threshold_accepted": _boolean(
                qc.get("threshold_accepted"),
                "threshold_accepted",
            ),
            "rank_correlation": _finite(
                qc.get("rank_correlation"),
                "rank_correlation",
            ),
            "raw_glass_leakage": _finite(
                qc.get("raw_glass_leakage"),
                "raw_glass_leakage",
            ),
            "corrected_glass_leakage": _finite(
                qc.get("corrected_glass_leakage"),
                "corrected_glass_leakage",
            ),
            "background_cv_before": _finite(
                qc.get("background_spatial_cv_before"),
                "background_spatial_cv_before",
            ),
            "background_cv_after": _finite(
                qc.get("background_spatial_cv_after"),
                "background_spatial_cv_after",
            ),
            "reconstruction_residual": _finite(
                qc.get("median_reconstruction_residual"),
                "median_reconstruction_residual",
            ),
        },
        "quantiles": {
            key: _finite(value, f"quantile {key}")
            for key, value in _mapping(
                stain.get("quantiles"),
                "stain quantiles",
            ).items()
            if key in {"0.5", "0.9", "0.95", "0.99"}
        },
        "known_issue": issue,
        "priority": {
            "required": False,
            "blocking": bool(issue),
            "score": 100 if issue else 0,
            "reasons": ["known upstream issue"] if issue else [],
        },
    }


def _assign_priorities(slides: list[dict[str, object]]) -> None:
    if not slides:
        return
    for slide in slides:
        qc = slide["qc"]
        priority = slide["priority"]
        if not qc["correction_accepted"]:
            _priority(priority, 80, "correction rejected")
        if qc["rank_correlation"] < 0.98:
            _priority(priority, 70, "rank guard failed")
        if qc["corrected_glass_leakage"] > qc["raw_glass_leakage"] + 1e-8:
            _priority(priority, 60, "glass leakage increased")

    _mark_extremes(
        slides,
        "corrected_glass_leakage",
        reverse=True,
        reason="highest corrected glass leakage",
    )
    _mark_extremes(
        slides,
        "reconstruction_residual",
        reverse=True,
        reason="highest reconstruction residual",
    )
    _mark_extremes(
        slides,
        "rank_correlation",
        reverse=False,
        reason="lowest rank preservation",
    )
    families = sorted({str(slide["family"]) for slide in slides})
    for family in families:
        candidates = [slide for slide in slides if slide["family"] == family]
        representative = max(
            candidates,
            key=lambda row: (
                int(row["priority"]["score"]),
                float(row["qc"]["corrected_glass_leakage"]),
                -int(row["order"]),
            ),
        )
        _priority(representative["priority"], 5, f"{family} representative")

    for slide in slides:
        priority = slide["priority"]
        priority["required"] = bool(priority["reasons"])


def _mark_extremes(
    slides: list[dict[str, object]],
    metric: str,
    *,
    reverse: bool,
    reason: str,
) -> None:
    count = min(2, len(slides))
    ranked = sorted(
        slides,
        key=lambda row: (float(row["qc"][metric]), -int(row["order"])),
        reverse=reverse,
    )
    for row in ranked[:count]:
        _priority(row["priority"], 20, reason)


def _priority(priority: dict[str, object], score: int, reason: str) -> None:
    priority["score"] = int(priority["score"]) + score
    reasons = priority["reasons"]
    if reason not in reasons:
        reasons.append(reason)


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _asset_href(
    viewer_dir: Path,
    asset_base: str,
    value: object,
    mouse_id: str,
    order: int,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{mouse_id} slide {order} is missing a review asset")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{mouse_id} slide {order} has an unsafe review asset")
    path = viewer_dir.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"{mouse_id} slide {order} review asset does not exist")
    return quote(f"{asset_base}{relative.as_posix()}", safe="/._-")


def _relative_href(output_dir: Path, viewer_dir: Path) -> str:
    relative = Path(os.path.relpath(viewer_dir, output_dir)).as_posix()
    return f"{quote(relative, safe='/._-').rstrip('/')}/"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
