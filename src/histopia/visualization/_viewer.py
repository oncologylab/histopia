# ruff: noqa: E501
"""Generate a static Three.js viewer for registered section stacks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from histopia.registration._approval import validate_registration_approval
from histopia.registration._errors import OptionalDependencyError
from histopia.registration._io import (
    overlay_mask,
    warp_mask_thumbnail,
    warp_rgb_thumbnail,
)
from histopia.semantic._result import validate_semantic_result

THREE_VERSION = "0.170.0"
MAX_DISPLAY_LINKS = 500
VIEWER_MOUSE_CACHE_VERSION = 1
_VIEWER_QC_FIELDS = frozenset(
    {
        "fingerprint",
        "selected_k",
        "slide_count",
        "patch_count",
        "median_tissue_fraction",
        "accepted_topology_links",
        "median_topology_confidence",
        "topology_coverage",
        "zero_link_pairs",
        "selected_k_stability",
        "selected_k_score",
        "minimum_cluster_fraction",
        "batch_correction_accepted",
        "raw_slide_variance_fraction",
        "corrected_slide_variance_fraction",
        "raw_slide_prediction_accuracy",
        "corrected_slide_prediction_accuracy",
        "unsupported_sections",
        "review_approved",
        "flags",
    }
)


def build_section_viewer(
    runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    provisional_mice: set[str] | None = None,
    semantic_runs: dict[str, Path | str] | None = None,
    cohort_qc: Path | str | None = None,
) -> Path:
    """Build a browser viewer from completed registration run directories."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    old_asset_cache = _load_asset_cache(output_dir / ".histopia-asset-cache.json")
    old_mouse_cache = _load_mouse_cache(output_dir / ".histopia-mouse-cache.json")
    old_mice = _load_previous_mice(output_dir / "manifest.json")
    new_asset_cache: dict[str, dict[str, object]] = {}
    new_mouse_cache: dict[str, dict[str, object]] = {}
    cache_stats = {"reused": 0, "encoded": 0}
    mouse_stats = {"reused": 0, "rendered": 0}
    provisional_mice = provisional_mice or set()
    semantic_runs = semantic_runs or {}
    cohort_rows = _load_cohort_rows(cohort_qc)
    mouse_payloads: list[dict[str, object]] = []

    for mouse_id, run_value in sorted(runs.items()):
        run_dir = Path(run_value)
        payload = json.loads((run_dir / "registration_result.json").read_text())
        semantic_dir = (
            Path(semantic_runs[mouse_id]) if mouse_id in semantic_runs else None
        )
        semantic_payload = (
            validate_semantic_result(semantic_dir) if semantic_dir is not None else None
        )
        semantic_review = (
            _semantic_review_payload(semantic_dir, semantic_payload)
            if semantic_dir is not None and semantic_payload is not None
            else None
        )
        semantic_qc = cohort_rows.get(mouse_id)
        if (
            cohort_qc is not None
            and semantic_payload is not None
            and semantic_qc is None
        ):
            raise ValueError(f"cohort QC is missing mouse {mouse_id}")
        if semantic_qc is not None and semantic_payload is not None:
            if semantic_qc.get("fingerprint") != semantic_payload.get("fingerprint"):
                raise ValueError(f"cohort QC fingerprint does not match {mouse_id}")
        registration_approval = _registration_approval_payload(run_dir)
        mouse_fingerprint = _viewer_mouse_fingerprint(
            run_dir,
            payload,
            semantic_payload=semantic_payload,
            semantic_qc=semantic_qc,
        )
        previous_mouse = old_mice.get(mouse_id)
        previous_cache = old_mouse_cache.get(mouse_id)
        if (
            previous_mouse is not None
            and previous_cache is not None
            and previous_cache.get("fingerprint") == mouse_fingerprint
            and _reuse_mouse_assets(
                previous_mouse,
                previous_cache,
                output_dir,
                old_asset_cache,
                new_asset_cache,
                cache_stats,
            )
        ):
            previous_mouse["provisional_order"] = mouse_id in provisional_mice
            previous_mouse["registration_approval"] = registration_approval
            if previous_mouse.get("semantic") is not None:
                previous_mouse["semantic"]["review"] = semantic_review
                previous_mouse["semantic"]["qc"] = semantic_qc
            mouse_payloads.append(previous_mouse)
            new_mouse_cache[mouse_id] = previous_cache
            mouse_stats["reused"] += 1
            continue
        mouse_stats["rendered"] += 1
        semantic_slides = (
            {row["id"]: row for row in semantic_payload["slides"]}
            if semantic_payload is not None
            else {}
        )
        cluster_count = (
            int(
                semantic_payload.get("selected_k", semantic_payload["primary_clusters"])
            )
            if semantic_payload is not None
            else None
        )
        cluster_counts = (
            [
                int(value)
                for value in semantic_payload.get("cluster_counts", [cluster_count])
            ]
            if semantic_payload is not None
            else []
        )
        palette = semantic_payload["palette"] if semantic_payload is not None else []
        reference_path = Path(payload["reference_slide"])
        reference_row = next(row for row in payload["slides"] if row["is_reference"])
        reference_image = _read_rgb(
            run_dir / "processed" / f"{reference_path.stem}.thumbnail.png"
        )
        mouse_assets = assets_dir / _safe_name(mouse_id)
        topology_links = (
            _viewer_topology_pairs(
                semantic_dir,
                semantic_payload,
                reference_row["geometry"],
                reference_image.shape[:2],
            )
            if semantic_payload is not None
            else []
        )
        mouse_assets.mkdir(parents=True, exist_ok=True)
        topology_url = None
        if semantic_payload is not None:
            topology_name = "topology.json"
            _write_json_atomic(
                mouse_assets / topology_name,
                {"schema_version": 1, "links": topology_links},
                compact=True,
            )
            topology_url = f"assets/{_safe_name(mouse_id)}/{topology_name}"
        slides: list[dict[str, object]] = []
        for order, slide in enumerate(payload["slides"], start=1):
            source_path = Path(slide["path"])
            source = _read_rgb(
                run_dir / "processed" / f"{source_path.stem}.thumbnail.png"
            )
            mask = _read_mask(run_dir / "processed" / f"{source_path.stem}.mask.png")
            matrix = np.asarray(slide["transform"]["matrix"], dtype=float)
            registered = warp_rgb_thumbnail(source, matrix, reference_image.shape[:2])
            registered_mask = warp_mask_thumbnail(
                mask,
                matrix,
                reference_image.shape[:2],
            )
            rgba = np.dstack([registered, (registered_mask * 255).astype(np.uint8)])
            filename = f"{order:03d}-{_safe_name(source_path.stem)}.webp"
            _write_cached_webp(
                Image,
                rgba,
                mouse_assets / filename,
                output_dir=output_dir,
                options={"lossless": False, "quality": 88, "method": 6},
                old_cache=old_asset_cache,
                new_cache=new_asset_cache,
                stats=cache_stats,
            )
            slide_payload = {
                "id": source_path.name,
                "label": _marker_label(source_path.stem),
                "order": order,
                "texture": f"assets/{_safe_name(mouse_id)}/{filename}",
                "reference": bool(slide["is_reference"]),
            }
            if semantic_payload is not None:
                semantic_row = semantic_slides.get(source_path.name)
                if semantic_row is None:
                    raise ValueError(f"semantic result is missing {source_path.name}")
                semantic_textures: dict[str, str] = {}
                selected_rgba = None
                for count in cluster_counts:
                    labels_path = semantic_dir / semantic_row["labels"][str(count)]
                    semantic_rgba = _semantic_rgba(
                        labels_path,
                        palette,
                        reference_row["geometry"],
                        registered_mask,
                    )
                    semantic_name = (
                        f"{order:03d}-{_safe_name(source_path.stem)}"
                        f"-k{count}-semantic.webp"
                    )
                    _write_cached_webp(
                        Image,
                        semantic_rgba,
                        mouse_assets / semantic_name,
                        output_dir=output_dir,
                        options={"lossless": True, "method": 6},
                        old_cache=old_asset_cache,
                        new_cache=new_asset_cache,
                        stats=cache_stats,
                    )
                    semantic_textures[str(count)] = (
                        f"assets/{_safe_name(mouse_id)}/{semantic_name}"
                    )
                    if count == cluster_count:
                        selected_rgba = semantic_rgba
                if selected_rgba is None:
                    raise ValueError("selected K is missing from semantic labels")
                blend_name = f"{order:03d}-{_safe_name(source_path.stem)}-blend.webp"
                blended = _blend_semantic(registered, registered_mask, selected_rgba)
                _write_cached_webp(
                    Image,
                    blended,
                    mouse_assets / blend_name,
                    output_dir=output_dir,
                    options={"lossless": False, "quality": 90, "method": 6},
                    old_cache=old_asset_cache,
                    new_cache=new_asset_cache,
                    stats=cache_stats,
                )
                slide_payload["semantic_textures"] = semantic_textures
                slide_payload["semantic_texture"] = semantic_textures[
                    str(cluster_count)
                ]
                slide_payload["blend_texture"] = (
                    f"assets/{_safe_name(mouse_id)}/{blend_name}"
                )
            slides.append(slide_payload)
        mouse_payload = {
            "id": mouse_id,
            "provisional_order": mouse_id in provisional_mice,
            "registration_approval": registration_approval,
            "width": int(reference_image.shape[1]),
            "height": int(reference_image.shape[0]),
            "slides": slides,
            "semantic": (
                {
                    "cluster_count": cluster_count,
                    "selected_k": cluster_count,
                    "cluster_counts": cluster_counts,
                    "palette": palette,
                    "batch_correction": semantic_payload.get("batch_correction"),
                    "k_selection": semantic_payload.get("k_selection"),
                    "fingerprint": semantic_payload.get("fingerprint"),
                    "review": semantic_review,
                    "qc": semantic_qc,
                    "links_url": topology_url,
                    "link_pair_count": len(topology_links),
                }
                if semantic_payload is not None
                else None
            ),
        }
        mouse_payloads.append(mouse_payload)
        new_mouse_cache[mouse_id] = {
            "fingerprint": mouse_fingerprint,
            "outputs": _mouse_output_hashes(mouse_payload, output_dir),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "mice": mouse_payloads}, indent=2) + "\n"
    )
    (output_dir / "index.html").write_text(
        _INDEX_HTML.replace("__THREE__", THREE_VERSION)
    )
    (output_dir / "viewer.js").write_text(
        _VIEWER_JS.replace("__THREE__", THREE_VERSION)
    )
    (output_dir / "styles.css").write_text(_STYLES_CSS)
    _write_json_atomic(
        output_dir / ".histopia-asset-cache.json",
        {"schema_version": 1, "assets": new_asset_cache},
    )
    _write_json_atomic(
        output_dir / ".histopia-mouse-cache.json",
        {
            "schema_version": VIEWER_MOUSE_CACHE_VERSION,
            "mice": new_mouse_cache,
        },
    )
    _write_json_atomic(
        output_dir / "build-report.json",
        {
            "schema_version": 1,
            "mouse_count": len(mouse_payloads),
            "slide_count": sum(len(mouse["slides"]) for mouse in mouse_payloads),
            "assets_reused": cache_stats["reused"],
            "assets_encoded": cache_stats["encoded"],
            "mice_reused": mouse_stats["reused"],
            "mice_rendered": mouse_stats["rendered"],
            "elapsed_seconds": round(time.perf_counter() - build_started, 3),
        },
    )
    return output_dir / "index.html"


def build_mask_review(
    registration_run: Path | str,
    output_dir: Path | str,
) -> Path:
    """Build a fixed-viewport audit of accepted masks for one registration run."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    registration_run = Path(registration_run)
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    payload = _mask_review_source_payload(registration_run)
    rows = []
    digest = hashlib.sha256(b"histopia-mask-review-v1")
    for order, slide in enumerate(payload.get("slides", []), start=1):
        source = Path(str(slide["path"]))
        image = _read_rgb(
            registration_run / "processed" / f"{source.stem}.thumbnail.png"
        )
        mask_path = registration_run / "processed" / f"{source.stem}.mask.png"
        mask_bytes = mask_path.read_bytes()
        digest.update(source.name.encode())
        digest.update(np.ascontiguousarray(image).tobytes())
        digest.update(mask_bytes)
        mask = _read_mask(mask_path)
        overlay = overlay_mask(image, mask)
        filename = f"{order:03d}-{_safe_name(source.stem)}.webp"
        Image.fromarray(overlay).save(
            assets_dir / filename,
            "WEBP",
            lossless=False,
            quality=88,
            method=6,
        )
        mask_data = slide.get("mask", {})
        review = slide.get("mask_review") or {}
        review_status = str(review.get("status", "pending"))
        rows.append(
            {
                "order": order,
                "slide": source.name,
                "label": _marker_label(source.stem),
                "texture": f"assets/{filename}",
                "method": str(mask_data.get("method", "unknown")),
                "foreground_fraction": float(
                    mask_data.get("metrics", {}).get("foreground_fraction", mask.mean())
                ),
                "approved": bool(review.get("approved"))
                or review_status in {"auto_pass", "override_pass"},
                "warning_count": len(mask_data.get("warnings", [])),
            }
        )
    if not rows:
        raise ValueError("registration result contains no slides")
    manifest = {
        "schema_version": 1,
        "fingerprint": digest.hexdigest(),
        "approved": all(row["approved"] for row in rows),
        "slides": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "index.html").write_text(_MASK_REVIEW_HTML)
    (output_dir / "mask-review.js").write_text(_MASK_REVIEW_JS)
    (output_dir / "mask-review.css").write_text(_ORDER_REVIEW_CSS)
    return output_dir / "index.html"


def _mask_review_source_payload(registration_run: Path) -> dict[str, object]:
    result_path = registration_run / "registration_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("registration result must be a JSON object")
        return payload

    review_path = registration_run / "mask_review.json"
    payload = json.loads(review_path.read_text())
    review_rows = payload.get("slides") if isinstance(payload, dict) else None
    if not isinstance(review_rows, list):
        raise ValueError("mask review must contain a slides list")
    slides: list[dict[str, object]] = []
    for review in review_rows:
        if not isinstance(review, dict):
            raise ValueError("mask review slides must contain objects")
        name = review.get("slide")
        if not isinstance(name, str) or not name:
            raise ValueError("mask review contains an invalid slide")
        slides.append(
            {
                "path": name,
                "mask": {
                    "method": review.get("method", "unknown"),
                    "metrics": {},
                    "warnings": [],
                },
                "mask_review": review,
            }
        )
    return {"slides": slides}


def _load_asset_cache(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, dict):
        return {}
    return {
        str(name): entry
        for name, entry in assets.items()
        if isinstance(name, str) and isinstance(entry, dict)
    }


def _load_mouse_cache(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != VIEWER_MOUSE_CACHE_VERSION
    ):
        return {}
    mice = payload.get("mice")
    if not isinstance(mice, dict):
        return {}
    return {
        str(mouse_id): entry
        for mouse_id, entry in mice.items()
        if isinstance(mouse_id, str) and isinstance(entry, dict)
    }


def _load_previous_mice(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    mice = payload.get("mice") if isinstance(payload, dict) else None
    if not isinstance(mice, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for mouse in mice:
        if not isinstance(mouse, dict):
            continue
        mouse_id = mouse.get("id")
        if isinstance(mouse_id, str) and mouse_id not in result:
            result[mouse_id] = mouse
    return result


def _viewer_mouse_fingerprint(
    run_dir: Path,
    registration: dict[str, object],
    *,
    semantic_payload: dict[str, object] | None,
    semantic_qc: dict[str, object] | None,
) -> str:
    slides = []
    for slide in registration.get("slides", []):
        source = Path(str(slide["path"]))
        review = slide.get("mask_review")
        review_fingerprint = (
            review.get("thumbnail_sha256") if isinstance(review, dict) else None
        )
        if not isinstance(review_fingerprint, str) or not review_fingerprint:
            stem = source.stem
            review_fingerprint = hashlib.sha256(
                (
                    _file_sha256(run_dir / "processed" / f"{stem}.thumbnail.png")
                    + _file_sha256(run_dir / "processed" / f"{stem}.mask.png")
                ).encode()
            ).hexdigest()
        slides.append(
            {
                "source": source.name,
                "is_reference": bool(slide.get("is_reference")),
                "transform": slide.get("transform"),
                "geometry": slide.get("geometry"),
                "thumbnail_fingerprint": review_fingerprint,
            }
        )
    core = {
        "version": VIEWER_MOUSE_CACHE_VERSION,
        "reference_slide": Path(str(registration["reference_slide"])).name,
        "slides": slides,
        "semantic_fingerprint": (
            semantic_payload.get("fingerprint")
            if semantic_payload is not None
            else None
        ),
        "semantic_qc": semantic_qc,
    }
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _registration_approval_payload(run_dir: Path) -> dict[str, object] | None:
    if not (run_dir / "registration_approval.json").is_file():
        return None
    approval = validate_registration_approval(run_dir)
    return {
        "approved": True,
        "reviewer": approval.reviewer,
        "reviewed_at": approval.reviewed_at,
        "order_fingerprint": approval.order_fingerprint,
        "registration_result_sha256": approval.registration_result_sha256,
    }


def _reuse_mouse_assets(
    mouse: dict[str, object],
    mouse_cache: dict[str, object],
    output_dir: Path,
    old_cache: dict[str, dict[str, object]],
    new_cache: dict[str, dict[str, object]],
    stats: dict[str, int],
) -> bool:
    paths = _mouse_asset_paths(mouse)
    outputs = mouse_cache.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(paths):
        return False
    verified: dict[str, dict[str, object]] = {}
    for relative in paths:
        path = output_dir / relative
        if (
            not path.is_file()
            or not isinstance(outputs.get(relative), str)
            or _file_sha256(path) != outputs[relative]
        ):
            return False
        if path.suffix == ".webp":
            previous = old_cache.get(relative)
            if previous is None or previous.get("output_sha256") != outputs[relative]:
                return False
            verified[relative] = previous
    new_cache.update(verified)
    stats["reused"] += len(verified)
    return True


def _mouse_output_hashes(
    mouse: dict[str, object],
    output_dir: Path,
) -> dict[str, str]:
    return {
        relative: _file_sha256(output_dir / relative)
        for relative in _mouse_asset_paths(mouse)
    }


def _mouse_asset_paths(mouse: dict[str, object]) -> tuple[str, ...]:
    paths: set[str] = set()
    slides = mouse.get("slides")
    if not isinstance(slides, list):
        return ()
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        for key in ("texture", "semantic_texture", "blend_texture"):
            value = slide.get(key)
            if isinstance(value, str):
                paths.add(value)
        semantic_textures = slide.get("semantic_textures")
        if isinstance(semantic_textures, dict):
            paths.update(
                value for value in semantic_textures.values() if isinstance(value, str)
            )
    semantic = mouse.get("semantic")
    if isinstance(semantic, dict) and isinstance(semantic.get("links_url"), str):
        paths.add(semantic["links_url"])
    return tuple(sorted(paths))


def _write_cached_webp(
    image_module,
    image: np.ndarray,
    path: Path,
    *,
    output_dir: Path,
    options: dict[str, object],
    old_cache: dict[str, dict[str, object]],
    new_cache: dict[str, dict[str, object]],
    stats: dict[str, int],
) -> None:
    """Reuse an exact WEBP asset or encode and checksum a replacement."""

    array = np.ascontiguousarray(image)
    relative = path.relative_to(output_dir).as_posix()
    digest = hashlib.sha256()
    digest.update(b"histopia-viewer-webp-v1")
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    digest.update(json.dumps(options, sort_keys=True, separators=(",", ":")).encode())
    input_sha256 = digest.hexdigest()
    previous = old_cache.get(relative, {})
    if (
        path.is_file()
        and previous.get("input_sha256") == input_sha256
        and previous.get("output_sha256") == _file_sha256(path)
    ):
        new_cache[relative] = previous
        stats["reused"] += 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image_module.fromarray(array).save(path, "WEBP", **options)
    new_cache[relative] = {
        "input_sha256": input_sha256,
        "output_sha256": _file_sha256(path),
    }
    stats["encoded"] += 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(
    path: Path,
    payload: dict[str, object],
    *,
    compact: bool = False,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        text = (
            json.dumps(payload, separators=(",", ":"))
            if compact
            else json.dumps(payload, indent=2)
        )
        temporary.write_text(text + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cohort_rows(
    cohort_qc: Path | str | None,
) -> dict[str, dict[str, object]]:
    if cohort_qc is None:
        return {}
    payload = json.loads(Path(cohort_qc).read_text())
    rows: dict[str, dict[str, object]] = {}
    for raw_row in payload.get("mice", []):
        mouse_id = str(raw_row.get("mouse_id", ""))
        if not mouse_id:
            raise ValueError("cohort QC rows must contain a mouse ID")
        if mouse_id in rows:
            raise ValueError(f"cohort QC contains duplicate mouse {mouse_id}")
        rows[mouse_id] = _public_qc_row(raw_row)
    return rows


def _public_qc_row(raw_row: dict[str, object]) -> dict[str, object]:
    row = {key: raw_row[key] for key in _VIEWER_QC_FIELDS if key in raw_row}
    fingerprint = row.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("cohort QC fingerprint must be a SHA256 digest")
    for key in (
        "selected_k",
        "slide_count",
        "patch_count",
        "accepted_topology_links",
        "zero_link_pairs",
    ):
        value = row.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"cohort QC {key} must be an integer")
    numeric_fields = _VIEWER_QC_FIELDS - {
        "fingerprint",
        "selected_k",
        "slide_count",
        "patch_count",
        "accepted_topology_links",
        "zero_link_pairs",
        "batch_correction_accepted",
        "review_approved",
        "unsupported_sections",
        "flags",
    }
    for key in numeric_fields:
        value = row.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"cohort QC {key} must be finite numeric data")
    for key in ("batch_correction_accepted", "review_approved"):
        value = row.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"cohort QC {key} must be boolean")
    unsupported = row.get("unsupported_sections", [])
    if not isinstance(unsupported, (list, tuple)) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in unsupported
    ):
        raise ValueError("cohort QC unsupported_sections must be nonnegative integers")
    row["unsupported_sections"] = list(unsupported)
    flags = row.get("flags", [])
    if not isinstance(flags, (list, tuple)) or any(
        not isinstance(flag, str) or re.fullmatch(r"[a-z0-9_]+", flag) is None
        for flag in flags
    ):
        raise ValueError("invalid cohort QC flag")
    row["flags"] = list(flags)
    return row


def _semantic_review_payload(
    semantic_dir: Path,
    semantic_payload: dict[str, object],
) -> dict[str, object]:
    review_path = semantic_dir / "semantic_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    matches = review.get("fingerprint") == semantic_payload.get("fingerprint")
    return {
        "approved": bool(review.get("approved")) and matches,
        "fingerprint_matches": matches,
    }


def _semantic_rgba(
    labels_path: Path,
    palette: list[str],
    reference_geometry: dict[str, object],
    registered_mask: np.ndarray,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    with np.load(labels_path, allow_pickle=False) as data:
        labels = data["labels"]
        points_um = data["reference_um_xy"]
        patch_um = float(data["patch_size_px"]) * float(data["analysis_mpp"])
    mpp_x, mpp_y = (float(value) for value in reference_geometry["mpp_xy"])
    x, y, native_width, native_height = (
        float(value) for value in reference_geometry["content_bbox_xywh"]
    )
    thumb_height, thumb_width = registered_mask.shape
    points_px = np.column_stack(
        [
            (points_um[:, 0] / mpp_x - x) * thumb_width / native_width,
            (points_um[:, 1] / mpp_y - y) * thumb_height / native_height,
        ]
    )
    half_width = patch_um / mpp_x * thumb_width / native_width / 2
    half_height = patch_um / mpp_y * thumb_height / native_height / 2
    canvas = Image.new("RGBA", (thumb_width, thumb_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for label, (center_x, center_y) in zip(labels, points_px, strict=True):
        color = palette[int(label) % len(palette)]
        draw.rectangle(
            (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            ),
            fill=color + "dc",
        )
    rgba = np.asarray(canvas).copy()
    rgba[..., 3] = np.where(registered_mask, rgba[..., 3], 0)
    return rgba


def _viewer_topology_pairs(
    semantic_dir: Path,
    semantic_payload: dict[str, object],
    reference_geometry: dict[str, object],
    reference_shape: tuple[int, int],
    *,
    max_links: int = MAX_DISPLAY_LINKS,
) -> list[dict[str, object]]:
    """Convert top-confidence correspondence pairs to compact viewer coordinates."""

    height, width = reference_shape
    mpp_x, mpp_y = (float(value) for value in reference_geometry["mpp_xy"])
    x, y, native_width, native_height = (
        float(value) for value in reference_geometry["content_bbox_xywh"]
    )
    scale = 320.0 / max(width, height)

    def plane_xy(points_um: np.ndarray) -> np.ndarray:
        px = (points_um[:, 0] / mpp_x - x) * width / native_width
        py = (points_um[:, 1] / mpp_y - y) * height / native_height
        return np.column_stack([(px - width / 2) * scale, (height / 2 - py) * scale])

    pairs: list[dict[str, object]] = []
    for row in semantic_payload.get("topology_pairs", []):
        with np.load(semantic_dir / row["artifact"], allow_pickle=False) as data:
            confidence = np.asarray(data["confidence"], dtype=float)
            selected = np.argsort(-confidence, kind="stable")[:max_links]
            source_xy = plane_xy(data["source_um_xy"][selected])
            target_xy = plane_xy(data["target_um_xy"][selected])
        pairs.append(
            {
                "source_section": int(row["source_section"]),
                "target_section": int(row["target_section"]),
                "accepted_links": int(row["accepted_links"]),
                "displayed_links": int(len(selected)),
                "source_xy": np.round(source_xy, 3).tolist(),
                "target_xy": np.round(target_xy, 3).tolist(),
                "confidence": np.round(confidence[selected], 4).tolist(),
            }
        )
    return pairs


def _blend_semantic(
    registered: np.ndarray,
    registered_mask: np.ndarray,
    semantic_rgba: np.ndarray,
) -> np.ndarray:
    alpha = semantic_rgba[..., 3:4].astype(np.float32) / 255 * 0.55
    rgb = registered.astype(np.float32) * (1 - alpha) + semantic_rgba[..., :3] * alpha
    return np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            (registered_mask * 255).astype(np.uint8),
        ]
    )


def build_section_order_review(
    proposal_path: Path | str,
    processed_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    """Build a non-scrolling review grid for a fingerprinted order proposal."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    proposal_path = Path(proposal_path)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(proposal_path.read_text())
    slides = payload.get("slides", [])
    if not isinstance(slides, list) or not slides:
        raise ValueError("section order proposal contains no slides")

    review_slides: list[dict[str, object]] = []
    for row in slides:
        slide_name = str(row["slide"])
        stem = Path(slide_name).stem
        image = _read_rgb(processed_dir / f"{stem}.thumbnail.png")
        mask = _read_mask(processed_dir / f"{stem}.mask.png")
        turns = int(row.get("quarter_turns_ccw", 0)) % 4
        image = np.rot90(image, turns).copy()
        mask = np.rot90(mask, turns).copy()
        rgba = _tissue_review_crop(image, mask)
        filename = f"{int(row['order']):03d}-{_safe_name(stem)}.webp"
        Image.fromarray(rgba).save(
            assets_dir / filename,
            "WEBP",
            lossless=False,
            quality=86,
            method=6,
        )
        review_slides.append(
            {
                **row,
                "label": _marker_label(stem),
                "texture": f"assets/{filename}",
            }
        )

    review_payload = {
        "schema_version": 1,
        "approved": bool(payload.get("approved")),
        "fingerprint": str(payload.get("fingerprint", "")),
        "objective": payload.get("objective"),
        "runner_up_objective": payload.get("runner_up_objective"),
        "confidence_margin": payload.get("confidence_margin"),
        "physically_calibrated": bool(payload.get("physically_calibrated")),
        "slides": review_slides,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(review_payload, indent=2) + "\n"
    )
    (output_dir / "index.html").write_text(_ORDER_REVIEW_HTML)
    (output_dir / "order-review.js").write_text(_ORDER_REVIEW_JS)
    (output_dir / "order-review.css").write_text(_ORDER_REVIEW_CSS)
    return output_dir / "index.html"


def _tissue_review_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop a review image around accepted tissue while retaining context."""

    rows, cols = np.nonzero(mask)
    if not rows.size:
        raise ValueError("section order review masks must contain tissue")
    height, width = mask.shape
    padding = max(4, int(round(max(height, width) * 0.03)))
    top = max(0, int(rows.min()) - padding)
    bottom = min(height, int(rows.max()) + padding + 1)
    left = max(0, int(cols.min()) - padding)
    right = min(width, int(cols.max()) + padding + 1)
    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    return np.dstack([cropped_image, np.where(cropped_mask, 255, 32).astype(np.uint8)])


def _read_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "section"


def _marker_label(stem: str) -> str:
    match = re.search(r"panc[_-](.+?)(?:-\[|$)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia Section Stack</title>
  <link rel="icon" href="data:">
  <link rel="stylesheet" href="styles.css">
  <script type="importmap">
    {"imports": {
      "three": "https://unpkg.com/three@__THREE__/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@__THREE__/examples/jsm/"
    }}
  </script>
</head>
<body>
  <main>
    <aside>
      <h1>Histopia</h1>
      <label>Mouse<select id="mouse"></select></label>
      <p id="order-status"></p>
      <div id="mode" class="segmented" aria-label="Texture mode">
        <button data-mode="histology" class="active">Histology</button>
        <button data-mode="blend">Blend</button>
        <button data-mode="semantic">Semantic</button>
      </div>
      <div id="legend"></div>
      <label>Regions<select id="clusters" disabled></select></label>
      <p id="qc"></p>
      <label>Adjacent pair<select id="link-pair" disabled></select></label>
      <label class="check"><input id="show-links" type="checkbox" checked>Show topology links</label>
      <label>Spacing<input id="spacing" type="range" min="2" max="80" value="24"></label>
      <label>Opacity<input id="opacity" type="range" min="0.05" max="1" step="0.05" value="0.72"></label>
      <div class="slide-navigation" aria-label="Slide navigation">
        <button id="previous-slide" title="Previous slide" aria-label="Previous slide">←</button>
        <output id="slide-focus" aria-live="polite">All slides</output>
        <button id="next-slide" title="Next slide" aria-label="Next slide">→</button>
      </div>
      <div class="visibility-commands">
        <button id="select-all">Select all</button>
        <button id="deselect-all">Deselect all</button>
      </div>
      <div class="commands">
        <button id="reset" title="Reset camera">Reset view</button>
        <button id="export" title="Export section order">Export order</button>
      </div>
      <ol id="sections"></ol>
    </aside>
    <section id="viewport" aria-label="Interactive registered section stack"></section>
  </main>
  <script type="module" src="viewer.js"></script>
</body>
</html>
"""

_ORDER_REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia Section Order Review</title>
  <link rel="stylesheet" href="order-review.css">
</head>
<body>
  <header>
    <strong>Histopia section order</strong>
    <span id="status"></span>
    <span id="score"></span>
    <code id="fingerprint"></code>
  </header>
  <main id="slides"></main>
  <script type="module" src="order-review.js"></script>
</body>
</html>
"""

_MASK_REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia Tissue Mask Review</title>
  <link rel="stylesheet" href="mask-review.css">
</head>
<body>
  <header>
    <strong>Histopia tissue masks</strong>
    <span id="status"></span>
    <span id="summary"></span>
    <code id="fingerprint"></code>
  </header>
  <main id="slides"></main>
  <script type="module" src="mask-review.js"></script>
</body>
</html>
"""

_MASK_REVIEW_JS = """const data = await (await fetch('manifest.json')).json();
const slides = document.querySelector('#slides');
const rowCount = innerWidth >= 2400
  ? (data.slides.length <= 18 ? 2 : 3)
  : (data.slides.length <= 18 ? 3 : 4);
slides.style.setProperty('--rows', rowCount);
slides.style.setProperty('--columns', Math.ceil(data.slides.length / rowCount));
document.querySelector('#status').textContent =
  data.approved ? 'Approved masks' : 'Approval required';
document.querySelector('#summary').textContent =
  `${data.slides.length} sections | ` +
  `${data.slides.reduce((sum, slide) => sum + slide.warning_count, 0)} warnings`;
document.querySelector('#fingerprint').textContent = data.fingerprint.slice(0, 16);
for (const slide of data.slides) {
  const card = document.createElement('article');
  if (slide.approved) card.classList.add('fixed');
  const image = document.createElement('img');
  image.src = slide.texture;
  image.alt = `Accepted tissue mask for ${slide.slide}`;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = `${String(slide.order).padStart(2, '0')} ${slide.label}`;
  const metrics = document.createElement('div');
  metrics.className = 'metrics';
  metrics.textContent =
    `${slide.method} | tissue ${(100 * slide.foreground_fraction).toFixed(1)}%`;
  card.append(image, label, metrics);
  slides.append(card);
}
"""

_ORDER_REVIEW_JS = """const data = await (await fetch('manifest.json')).json();
const slides = document.querySelector('#slides');
const rowCount = innerWidth >= 2400
  ? (data.slides.length <= 18 ? 2 : 3)
  : (data.slides.length <= 18 ? 3 : 4);
slides.style.setProperty('--rows', rowCount);
slides.style.setProperty('--columns', Math.ceil(data.slides.length / rowCount));
document.querySelector('#status').textContent =
  `${data.approved ? 'Approved' : 'Approval required'} | ` +
  `${data.physically_calibrated ? 'physical scale' : 'pixel scale'}`;
document.querySelector('#score').textContent =
  `cost ${Number(data.objective).toFixed(4)} | margin ` +
  `${Number(data.confidence_margin).toFixed(4)}`;
document.querySelector('#fingerprint').textContent =
  data.fingerprint ? data.fingerprint.slice(0, 16) : '';
for (const slide of data.slides) {
  const card = document.createElement('article');
  if (slide.fixed) card.classList.add('fixed');
  const image = document.createElement('img');
  image.src = slide.texture;
  image.alt = slide.slide;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = `${String(slide.order).padStart(2, '0')} ${slide.label}`;
  const metrics = document.createElement('div');
  metrics.className = 'metrics';
  const distance = slide.distance_from_previous == null
    ? 'anchor'
    : `d ${Number(slide.distance_from_previous).toFixed(3)}`;
  const area = slide.physical_tissue_area_um2 == null
    ? ''
    : ` | ${(Number(slide.physical_tissue_area_um2) / 1e6).toFixed(1)} mm2`;
  metrics.textContent = `${slide.fixed ? 'fixed | ' : ''}${distance}${area}`;
  card.append(image, label, metrics);
  slides.append(card);
}
"""

_ORDER_REVIEW_CSS = """*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden}
body{background:#151719;color:#f3f4f5;font:13px Arial,sans-serif}
header{height:46px;display:flex;align-items:center;gap:16px;padding:7px 12px;border-bottom:1px solid #45494d}
header strong{font-size:16px}header code{margin-left:auto;color:#aeb7bf}
main{height:calc(100vh - 46px);display:grid;grid-template-columns:repeat(var(--columns),minmax(0,1fr));grid-template-rows:repeat(var(--rows),minmax(0,1fr));gap:4px;padding:4px}
article{position:relative;min-width:0;min-height:0;background:#f4f4f2;border:1px solid #555;overflow:hidden}
article.fixed{border:3px solid #e0b84b}img{display:block;width:100%;height:calc(100% - 34px);object-fit:contain;background:white}
.label,.metrics{height:17px;padding:1px 5px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:#16191b}
.label{font-weight:700}.metrics{font-size:11px;color:#4b5156}
@media(max-width:600px){
  header{height:32px;gap:8px;padding:4px 7px;font-size:10px;white-space:nowrap}
  header strong{font-size:12px}header #score,header #summary,header code{display:none}
  main{height:calc(100vh - 32px);gap:3px;padding:3px}
  img{height:calc(100% - 32px)}
  .label,.metrics{height:16px;padding:1px 4px}.metrics{font-size:10px}
}
"""

_VIEWER_JS = """import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const manifest = await (await fetch('manifest.json')).json();
const viewport = document.querySelector('#viewport');
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0xf4f5f3);
viewport.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const group = new THREE.Group();
scene.add(group);
const loader = new THREE.TextureLoader();
let current;
let currentMode = 'histology';
let currentK = null;
let linkObject = null;
let loadGeneration = 0;
let textureGeneration = 0;
let focusedSlideIndex = null;
function disposeTexture(texture) { if (texture) texture.dispose(); }

function resize() {
  const box = viewport.getBoundingClientRect();
  renderer.setSize(box.width, box.height, true);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}
function resetCamera() {
  const visibleMeshes = group.children.filter(child => child.visible);
  if (!visibleMeshes.length) return;
  group.updateMatrixWorld(true);
  const bounds = new THREE.Box3();
  visibleMeshes.forEach(mesh => bounds.expandByObject(mesh));
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const limitingFov = Math.min(verticalFov, horizontalFov);
  const distance = sphere.radius / Math.sin(limitingFov / 2) * 1.12;
  const direction = new THREE.Vector3(0, -4, 3).normalize();
  camera.position.copy(sphere.center).addScaledVector(direction, distance);
  controls.target.copy(sphere.center);
  camera.near = Math.max(0.01, sphere.radius / 10000);
  camera.far = distance + sphere.radius * 20;
  controls.minDistance = Math.max(sphere.radius * 0.12, camera.near * 10);
  controls.maxDistance = distance * 8;
  camera.updateProjectionMatrix();
  controls.update();
}
function orderedSlides() {
  return [...document.querySelectorAll('#sections li')].map(li =>
    current.slides.find(slide => slide.id === li.dataset.id));
}
function clearLinks() {
  if (!linkObject) return;
  scene.remove(linkObject);
  linkObject.geometry.dispose();
  linkObject.material.dispose();
  linkObject = null;
}
function rebuildLinks() {
  clearLinks();
  if (!current?.semantic?.links?.length || !document.querySelector('#show-links').checked) return;
  const pair = current.semantic.links[Number(document.querySelector('#link-pair').value) || 0];
  if (!pair) return;
  if (!current.slides[pair.source_section].mesh.visible ||
      !current.slides[pair.target_section].mesh.visible) return;
  const sourceZ = current.slides[pair.source_section].mesh.position.z;
  const targetZ = current.slides[pair.target_section].mesh.position.z;
  const positions = [];
  pair.source_xy.forEach((source, index) => {
    const target = pair.target_xy[index];
    positions.push(source[0], source[1], sourceZ, target[0], target[1], targetZ);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x007f8b,
    transparent: true,
    opacity: 0.55,
    depthTest: false,
  });
  linkObject = new THREE.LineSegments(geometry, material);
  linkObject.renderOrder = 10;
  scene.add(linkObject);
}

function layout() {
  const spacing = Number(document.querySelector('#spacing').value);
  const opacity = Number(document.querySelector('#opacity').value);
  orderedSlides().forEach((slide, index, all) => {
    slide.mesh.position.z = (index - (all.length - 1) / 2) * spacing;
    slide.mesh.material.opacity = opacity;
  });
  rebuildLinks();
}
function updateSlideFocus() {
  const slides = orderedSlides();
  const checked = document.querySelectorAll('#sections input:checked').length;
  document.querySelector('#slide-focus').textContent = focusedSlideIndex == null
    ? `${checked} selected`
    : `${focusedSlideIndex + 1} / ${slides.length}`;
}
function setSlideVisibility(predicate) {
  const slides = orderedSlides();
  document.querySelectorAll('#sections li').forEach((item, index) => {
    const visible = predicate(index, slides[index]);
    item.querySelector('input').checked = visible;
    slides[index].mesh.visible = visible;
  });
  updateSlideFocus();
  rebuildLinks();
}
function focusSlide(index) {
  const slides = orderedSlides();
  if (!slides.length) return;
  focusedSlideIndex = (index + slides.length) % slides.length;
  setSlideVisibility(itemIndex => itemIndex === focusedSlideIndex);
  resetCamera();
}
function stepSlide(offset) {
  const slides = orderedSlides();
  if (!slides.length) return;
  const start = focusedSlideIndex == null
    ? (offset > 0 ? -1 : 0)
    : focusedSlideIndex;
  focusSlide(start + offset);
}
function buildList() {
  const list = document.querySelector('#sections');
  list.replaceChildren();
  current.slides.forEach(slide => {
    const item = document.createElement('li');
    item.dataset.id = slide.id;
    item.draggable = true;
    const toggle = document.createElement('input');
    toggle.type = 'checkbox'; toggle.checked = true;
    toggle.addEventListener('change', () => {
      focusedSlideIndex = null;
      slide.mesh.visible = toggle.checked;
      updateSlideFocus();
      rebuildLinks();
    });
    const text = document.createElement('span');
    text.textContent = `${slide.label}${slide.reference ? ' (reference)' : ''}`;
    text.title = 'Show only this slide';
    text.addEventListener('click', () =>
      focusSlide([...list.children].indexOf(item)));
    item.append(toggle, text);
    item.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', slide.id));
    item.addEventListener('dragover', event => event.preventDefault());
    item.addEventListener('drop', event => {
      event.preventDefault();
      const dragged = list.querySelector(`[data-id="${CSS.escape(event.dataTransfer.getData('text/plain'))}"]`);
      if (dragged && dragged !== item) list.insertBefore(dragged, item);
      layout();
    });
    list.append(item);
  });
  focusedSlideIndex = null;
  updateSlideFocus();
}
function textureUrl(slide, mode, clusterCount = currentK) {
  if (mode === 'semantic') return slide.semantic_textures[String(clusterCount)];
  if (mode === 'blend') return slide.blend_texture;
  return slide.texture;
}
function updateModeControls() {
  const available = Boolean(current?.semantic);
  document.querySelectorAll('#mode button').forEach(button => {
    button.disabled = button.dataset.mode !== 'histology' && !available;
    button.classList.toggle('active', button.dataset.mode === currentMode);
  });
  const legend = document.querySelector('#legend');
  legend.replaceChildren();
  if (available && currentMode !== 'histology') {
    current.semantic.palette.slice(0, currentK).forEach((color, index) => {
      const item = document.createElement('span');
      item.innerHTML = `<i style="background:${color}"></i>Region ${index + 1}`;
      legend.append(item);
    });
  }
  const qc = document.querySelector('#qc');
  const review = current?.semantic?.review;
  const cohort = current?.semantic?.qc;
  const batch = current?.semantic?.batch_correction;
  const metric = current?.semantic?.k_selection?.find(row => row.k === currentK);
  const details = [];
  if (review) {
    details.push(review.approved
      ? 'Approved'
      : (review.fingerprint_matches ? 'Approval required' : 'Review fingerprint mismatch'));
  }
  if (cohort) {
    const flags = cohort.flags || [];
    details.push(flags.length ? `QC: ${flags.join(', ')}` : 'QC: pass');
    details.push(`topology coverage ${(100 * Number(cohort.topology_coverage)).toFixed(1)}%`);
  }
  if (batch && metric) {
    const raw = Number(batch.raw.slide_variance_fraction).toFixed(4);
    const proposed = Number(batch.corrected.slide_variance_fraction).toFixed(4);
    details.push(`batch ${batch.accepted ? 'accepted' : 'rejected'}: ${raw} to ${proposed}`);
    details.push(`K ${currentK} score ${Number(metric.composite_score).toFixed(3)}`);
  }
  qc.textContent = details.join(' | ');
}
async function setMode(mode, force = false) {
  if (!current || (mode !== 'histology' && !current.semantic) || (!force && mode === currentMode)) return;
  const mouse = current;
  const generation = ++textureGeneration;
  const textures = await Promise.all(mouse.slides.map(async slide => {
    const texture = await loader.loadAsync(textureUrl(slide, mode));
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }));
  if (generation !== textureGeneration || current !== mouse) {
    textures.forEach(disposeTexture);
    return;
  }
  currentMode = mode;
  mouse.slides.forEach((slide, index) => {
    const previous = slide.mesh.material.map;
    slide.mesh.material.map = textures[index];
    slide.mesh.material.needsUpdate = true;
    disposeTexture(previous);
  });
  updateModeControls();
}
async function loadTopologyLinks(mouse) {
  if (!mouse.semantic) return [];
  if (mouse.semantic.links) return mouse.semantic.links;
  if (!mouse.semantic.links_url) return [];
  const response = await fetch(mouse.semantic.links_url);
  if (!response.ok) throw new Error(`Topology request failed: ${response.status}`);
  const payload = await response.json();
  if (payload.schema_version !== 1 || !Array.isArray(payload.links))
    throw new Error('Invalid topology payload');
  return payload.links;
}
async function loadMouse(mouse) {
  const generation = ++loadGeneration;
  ++textureGeneration;
  const progress = document.querySelector('#slide-focus');
  progress.textContent = `Loading ${mouse.id}...`;
  viewport.setAttribute('aria-busy', 'true');
  const requestedMode =
    mouse.semantic || currentMode === 'histology' ? currentMode : 'histology';
  const requestedK = mouse.semantic?.selected_k ?? null;
  let loadedTextures = 0;
  const [textures, topologyLinks] = await Promise.all([
    Promise.all(mouse.slides.map(async slide => {
      const texture = await loader.loadAsync(
        textureUrl(slide, requestedMode, requestedK));
      texture.colorSpace = THREE.SRGBColorSpace;
      loadedTextures += 1;
      if (generation === loadGeneration)
        progress.textContent = `Loading ${loadedTextures} / ${mouse.slides.length}`;
      return texture;
    })),
    loadTopologyLinks(mouse),
  ]);
  if (generation !== loadGeneration) {
    textures.forEach(disposeTexture);
    return;
  }
  group.children.forEach(mesh => {
    disposeTexture(mesh.material.map); mesh.material.dispose(); mesh.geometry.dispose();
  });
  group.clear(); current = mouse; currentMode = requestedMode;
  if (mouse.semantic) mouse.semantic.links = topologyLinks;
  currentK = mouse.semantic?.selected_k ?? null;
  const clusterSelect = document.querySelector('#clusters');
  clusterSelect.replaceChildren();
  (mouse.semantic?.cluster_counts || []).forEach(k =>
    clusterSelect.add(new Option(`K = ${k}${k === currentK ? ' (selected)' : ''}`, k)));
  clusterSelect.value = currentK ?? '';
  clusterSelect.disabled = !mouse.semantic;
  if (!mouse.semantic) currentMode = 'histology';
  const approval = mouse.registration_approval;
  document.querySelector('#order-status').textContent = approval
    ? `Approved registration · ${approval.reviewed_at.slice(0, 10)}`
    : (mouse.provisional_order ? 'Provisional section order' : 'Confirmed section order');
  const pairSelect = document.querySelector('#link-pair');
  pairSelect.replaceChildren();
  (mouse.semantic?.links || []).forEach((pair, index) => {
    const left = mouse.slides[pair.source_section].label;
    const right = mouse.slides[pair.target_section].label;
    pairSelect.add(new Option(`${left} to ${right} (${pair.displayed_links})`, index));
  });
  const linksAvailable = Boolean((mouse.semantic?.links || []).length);
  pairSelect.disabled = !linksAvailable;
  const showLinks = document.querySelector('#show-links');
  showLinks.disabled = !linksAvailable;
  showLinks.checked = linksAvailable;
  const scale = 320 / Math.max(mouse.width, mouse.height);
  mouse.slides.forEach((slide, index) => {
    const texture = textures[index];
    const material = new THREE.MeshBasicMaterial({map: texture, transparent: true, side: THREE.DoubleSide, depthWrite: false});
    slide.mesh = new THREE.Mesh(new THREE.PlaneGeometry(mouse.width * scale, mouse.height * scale), material);
    group.add(slide.mesh);
  });
  buildList(); layout(); updateModeControls(); resetCamera();
  viewport.setAttribute('aria-busy', 'false');
}
function reportLoadError(error) {
  document.querySelector('#slide-focus').textContent = 'Load failed';
  viewport.setAttribute('aria-busy', 'false');
  console.error(error);
}
const select = document.querySelector('#mouse');
manifest.mice.forEach(mouse => select.add(new Option(mouse.id, mouse.id)));
select.addEventListener('change', () => {
  loadMouse(manifest.mice.find(mouse => mouse.id === select.value))
    .catch(reportLoadError);
});
document.querySelector('#spacing').addEventListener('input', layout);
document.querySelector('#opacity').addEventListener('input', layout);
document.querySelector('#clusters').addEventListener('change', async event => {
  currentK = Number(event.target.value);
  await setMode('semantic', true);
});
document.querySelector('#reset').addEventListener('click', resetCamera);
document.querySelector('#previous-slide').addEventListener('click', () => stepSlide(-1));
document.querySelector('#next-slide').addEventListener('click', () => stepSlide(1));
document.querySelector('#select-all').addEventListener('click', () => {
  focusedSlideIndex = null;
  setSlideVisibility(() => true);
});
document.querySelector('#deselect-all').addEventListener('click', () => {
  focusedSlideIndex = null;
  setSlideVisibility(() => false);
});
document.querySelector('#show-links').addEventListener('change', rebuildLinks);
document.querySelector('#link-pair').addEventListener('change', rebuildLinks);
document.querySelector('#export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({mouse: current.id, slides: orderedSlides().map((s, i) => ({slide: s.id, order: i + 1}))}, null, 2)], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${current.id}-section-order.json`; link.click(); URL.revokeObjectURL(link.href);
});
document.querySelectorAll('#mode button').forEach(button =>
  button.addEventListener('click', () => setMode(button.dataset.mode)));
new ResizeObserver(resize).observe(viewport); resize(); resetCamera();
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
try { await loadMouse(manifest.mice[0]); } catch (error) { reportLoadError(error); }
animate();
"""

_STYLES_CSS = """*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}body{font-family:Arial,sans-serif;color:#202426;background:#f4f5f3}main{display:grid;grid-template-columns:300px minmax(0,1fr);width:100%;height:100%;overflow:hidden}aside{min-width:0;min-height:0;padding:18px;border-right:1px solid #c9ceca;background:#fff;overflow-y:auto;overflow-x:hidden}h1{font-size:22px;margin:0 0 18px}label{display:grid;gap:6px;font-size:13px;margin:14px 0}select,input{width:100%}.commands,.segmented,.visibility-commands{display:flex;gap:8px;margin:16px 0}button{border:1px solid #88918b;background:#fff;padding:7px 10px;border-radius:4px;cursor:pointer}.segmented{gap:0}.segmented button{flex:1;border-radius:0;margin-left:-1px}.segmented button:first-child{margin-left:0;border-radius:4px 0 0 4px}.segmented button:last-child{border-radius:0 4px 4px 0}.segmented button.active{background:#202426;color:#fff}.segmented button:disabled{color:#a7aca8;cursor:default}.slide-navigation{display:grid;grid-template-columns:36px minmax(0,1fr) 36px;align-items:center;gap:8px;margin:16px 0}.slide-navigation button{width:36px;height:32px;padding:0;font-size:18px}.slide-navigation output{text-align:center;font-size:12px;white-space:nowrap}.visibility-commands button{flex:1}#legend{display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:11px}#legend span{display:flex;align-items:center;gap:5px}#legend i{display:block;width:12px;height:12px;border:1px solid #555}#order-status{font-size:12px;color:#8a4f12}ol{padding:0;list-style:none}li{display:grid;grid-template-columns:20px minmax(0,1fr);align-items:center;min-height:32px;border-bottom:1px solid #eceeec;font-size:12px;cursor:grab}li span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer}li input{width:14px}#viewport{position:relative;min-width:0;min-height:0;width:100%;height:100%;overflow:hidden}canvas{display:block;width:100%!important;height:100%!important}@media(max-width:720px){main{grid-template-columns:1fr;grid-template-rows:250px minmax(0,1fr)}aside{border-right:0;border-bottom:1px solid #c9ceca}#sections{display:none}}"""
