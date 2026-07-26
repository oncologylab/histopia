# ruff: noqa: E501
"""Fixed-viewport review for provisional non-rigid validation runs."""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from histopia._atomic import write_text_atomic
from histopia.registration._errors import OptionalDependencyError

_PANE_COUNT = 6
_PANE_GAP = 8
_PANE_TOP = 86
_EXPORTED_PANES = (
    ("affine", 1),
    ("candidate", 2),
    ("checker", 4),
    ("magnitude", 5),
)


def build_non_rigid_review(
    source_run: Path | str,
    output_dir: Path | str,
    *,
    workers: int = 1,
) -> Path:
    """Build a path-free browser for one provisional dense-field audit.

    ``source_run`` may be a standalone validation bundle containing
    ``summary.json`` or a Histopia registration run containing
    ``registration_result.json``.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("non-rigid review workers must be a positive integer")
    source = Path(source_run)
    output = Path(output_dir)
    summary, qc_dir = _load_review_source(source)
    if summary.get("status") != "provisional_validation":
        raise ValueError("non-rigid review requires a provisional validation summary")
    reference_shape = summary.get("reference_shape")
    if (
        not isinstance(reference_shape, list)
        or len(reference_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in reference_shape
        )
    ):
        raise ValueError("non-rigid validation reference_shape is invalid")
    pane_height, pane_width = reference_shape
    rows = summary.get("slides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("non-rigid validation summary has no slides")

    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()

    normalized = [
        _normalize_slide(row, qc_dir, pane_width, pane_height) for row in rows
    ]
    reference_asset = assets / "reference.webp"
    _encode_reference(
        Path(normalized[0]["_contact_path"]),
        reference_asset,
        pane_width,
        pane_height,
    )

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        return _encode_slide_assets(
            row,
            assets,
            pane_width,
            pane_height,
        )

    if workers == 1:
        manifest_rows = [encode(row) for row in normalized]
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="histopia-nonrigid-review",
        ) as executor:
            manifest_rows = list(executor.map(encode, normalized))

    manifest = {
        "schema_version": 1,
        "status": "provisional_validation",
        "mouse_id": str(summary.get("mouse_id", "unknown")),
        "reference_slide": Path(str(summary.get("reference_slide", "reference"))).name,
        "reference_asset": "assets/reference.webp",
        "slide_count": len(manifest_rows),
        "accepted_count": sum(bool(row["accepted"]) for row in manifest_rows),
        "slides": manifest_rows,
    }
    rendered = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    write_text_atomic(output / "manifest.json", rendered)
    write_text_atomic(
        output / "manifest-data.js",
        "window.HISTOPIA_NONRIGID_REVIEW = "
        + json.dumps(manifest, separators=(",", ":"), ensure_ascii=True)
        + ";\n",
    )
    write_text_atomic(output / "index.html", _HTML)
    write_text_atomic(output / "nonrigid-review.css", _CSS)
    write_text_atomic(output / "nonrigid-review.js", _JS)
    return output / "index.html"


def _normalize_slide(
    value: object,
    qc_dir: Path,
    pane_width: int,
    pane_height: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("non-rigid validation slide must be an object")
    slide = value.get("slide")
    order = value.get("order")
    accepted = value.get("accepted")
    if not isinstance(slide, str) or not slide:
        raise ValueError("non-rigid validation slide name is invalid")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"non-rigid validation order is invalid: {slide}")
    if not isinstance(accepted, bool):
        raise ValueError(f"non-rigid validation acceptance is invalid: {slide}")
    contact = qc_dir / f"{Path(slide).stem}.contact.png"
    if not contact.is_file():
        raise FileNotFoundError(contact)
    _validate_contact(contact, pane_width, pane_height)
    sparse = value.get("sparse_feature_validation")
    if sparse is not None and not isinstance(sparse, dict):
        raise ValueError(f"sparse feature validation is invalid: {slide}")
    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or any(
        not isinstance(warning, str) for warning in warnings
    ):
        raise ValueError(f"non-rigid validation warnings are invalid: {slide}")
    return {
        "_contact_path": str(contact),
        "order": order,
        "slide": Path(slide).name,
        "label": str(value.get("label", Path(slide).stem)),
        "accepted": accepted,
        "method": str(value.get("method", "unknown")),
        "initial_similarity": _optional_number(value.get("initial_similarity")),
        "final_similarity": _optional_number(value.get("final_similarity")),
        "initial_mask_dice": _optional_number(value.get("initial_mask_dice")),
        "final_mask_dice": _optional_number(value.get("final_mask_dice")),
        "jacobian_p01": _optional_number(value.get("jacobian_p01")),
        "jacobian_p99": _optional_number(value.get("jacobian_p99")),
        "displacement_p95": _optional_number(value.get("displacement_p95")),
        "inverse_consistency_p95": _optional_number(
            value.get("inverse_consistency_p95")
        ),
        "warnings": warnings,
        "sparse_feature_validation": _normalize_sparse(sparse),
    }


def _normalize_sparse(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or any(
        not isinstance(warning, str) for warning in warnings
    ):
        raise ValueError("sparse feature validation warnings are invalid")
    return {
        "status": str(value.get("status", "unknown")),
        "detector": str(value.get("detector", "unknown")),
        "mutual_matches": _optional_integer(value.get("mutual_matches")),
        "coherent_matches": _optional_integer(value.get("coherent_matches")),
        "initial_median_residual_px": _optional_number(
            value.get("initial_median_residual_px")
        ),
        "final_median_residual_px": _optional_number(
            value.get("final_median_residual_px")
        ),
        "initial_p95_residual_px": _optional_number(
            value.get("initial_p95_residual_px")
        ),
        "final_p95_residual_px": _optional_number(value.get("final_p95_residual_px")),
        "improved_fraction": _optional_number(value.get("improved_fraction")),
        "warnings": warnings,
    }


def _encode_reference(
    contact_path: Path,
    output_path: Path,
    pane_width: int,
    pane_height: int,
) -> None:
    Image = _import_image()
    with Image.open(contact_path) as contact:
        reference = contact.convert("RGB").crop(
            (0, _PANE_TOP, pane_width, _PANE_TOP + pane_height)
        )
        reference.save(output_path, "WEBP", quality=88, method=4)


def _encode_slide_assets(
    row: dict[str, Any],
    assets: Path,
    pane_width: int,
    pane_height: int,
) -> dict[str, Any]:
    Image = _import_image()
    safe = f"{row['order']:03d}-{_safe_stem(Path(row['slide']).stem)}"
    encoded: dict[str, str] = {}
    with Image.open(row["_contact_path"]) as contact:
        rgb = contact.convert("RGB")
        for pane_name, pane_index in _EXPORTED_PANES:
            left = pane_index * (pane_width + _PANE_GAP)
            crop = rgb.crop(
                (
                    left,
                    _PANE_TOP,
                    left + pane_width,
                    _PANE_TOP + pane_height,
                )
            )
            name = f"{safe}-{pane_name}.webp"
            crop.save(assets / name, "WEBP", quality=88, method=4)
            encoded[pane_name] = f"assets/{name}"
    public = {key: value for key, value in row.items() if not key.startswith("_")}
    public["assets"] = encoded
    return public


def _validate_contact(path: Path, pane_width: int, pane_height: int) -> None:
    Image = _import_image()
    with Image.open(path) as image:
        minimum_width = _PANE_COUNT * pane_width + (_PANE_COUNT - 1) * _PANE_GAP
        minimum_height = _PANE_TOP + pane_height
        if image.width < minimum_width or image.height < minimum_height:
            raise ValueError(
                f"non-rigid QC contact dimensions are invalid: {path.name}"
            )


def _import_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc
    return Image


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_review_source(source: Path) -> tuple[dict[str, Any], Path]:
    summary_path = source / "summary.json"
    if summary_path.is_file():
        return _load_object(summary_path), source / "qc"
    result_path = source / "registration_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"expected summary.json or registration_result.json under {source}"
        )
    result = _load_object(result_path)
    reference = result.get("reference_slide")
    rows = result.get("slides")
    if not isinstance(reference, str) or not reference:
        raise ValueError("registration result reference_slide is invalid")
    if not isinstance(rows, list) or not rows:
        raise ValueError("registration result has no slides")
    reference_thumbnail = source / "processed" / f"{Path(reference).stem}.thumbnail.png"
    if not reference_thumbnail.is_file():
        raise FileNotFoundError(reference_thumbnail)
    Image = _import_image()
    with Image.open(reference_thumbnail) as image:
        reference_shape = [image.height, image.width]
    slides: list[dict[str, Any]] = []
    for order, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("registration result slide must be an object")
        if row.get("is_reference") is True:
            continue
        transform = row.get("non_rigid_transform")
        if not isinstance(transform, dict):
            raise ValueError("registration result has no non-rigid candidate")
        slide = row.get("path")
        if not isinstance(slide, str) or not slide:
            raise ValueError("registration result slide path is invalid")
        slides.append(
            {
                "order": order,
                "slide": Path(slide).name,
                "label": Path(slide).stem,
                **transform,
            }
        )
    if not slides:
        raise ValueError("registration result has no non-reference slides")
    return (
        {
            "status": "provisional_validation",
            "mouse_id": source.name,
            "reference_slide": Path(reference).name,
            "reference_shape": reference_shape,
            "slides": slides,
        },
        source / "qc" / "non_rigid",
    )


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-rigid metric must be numeric or null")
    return float(value)


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("non-rigid count must be an integer or null")
    return value


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "slide"


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Histopia non-rigid review</title>
  <link rel="stylesheet" href="nonrigid-review.css">
</head>
<body>
  <main class="app">
    <header class="topbar">
      <div class="brand">Histopia <span>non-rigid validation</span></div>
      <div class="provisional">PROVISIONAL &middot; AFFINE BASELINE RETAINED</div>
      <div class="counter" id="counter"></div>
    </header>
    <div class="workspace">
      <aside class="rail">
        <div class="rail-head">
          <div class="mouse" id="mouse"></div>
          <div class="segmented" id="filters">
            <button type="button" data-filter="all" class="active">All</button>
            <button type="button" data-filter="accepted">Accepted</button>
            <button type="button" data-filter="rejected">Rejected</button>
          </div>
        </div>
        <div class="slide-list" id="slide-list"></div>
      </aside>
      <section class="content">
        <div class="section-head">
          <button class="icon-button" id="previous" type="button" title="Previous slide" aria-label="Previous slide">&lsaquo;</button>
          <div class="section-title">
            <strong id="slide-title"></strong>
            <span id="slide-method"></span>
          </div>
          <div class="status" id="status"></div>
          <button class="icon-button" id="next" type="button" title="Next slide" aria-label="Next slide">&rsaquo;</button>
        </div>
        <div class="primary">
          <figure>
            <figcaption>Reference</figcaption>
            <img id="reference" alt="Reference section">
          </figure>
          <figure>
            <figcaption>Affine baseline</figcaption>
            <img id="affine" alt="Affine registered moving section">
          </figure>
          <figure>
            <figcaption>Dense candidate</figcaption>
            <img id="candidate" alt="Candidate dense refinement">
          </figure>
        </div>
        <div class="diagnostics">
          <figure>
            <figcaption>Reference / candidate checkerboard</figcaption>
            <img id="checker" alt="Reference and candidate checkerboard">
          </figure>
          <figure>
            <figcaption>Candidate displacement magnitude</figcaption>
            <img id="magnitude" alt="Dense displacement magnitude">
          </figure>
          <section class="metrics" aria-label="Registration metrics">
            <dl id="metrics"></dl>
            <div class="warnings" id="warnings"></div>
          </section>
        </div>
      </section>
    </div>
  </main>
  <dialog id="zoom">
    <button class="icon-button close" type="button" title="Close" aria-label="Close">&times;</button>
    <img alt="Expanded scientific image">
  </dialog>
  <script src="manifest-data.js"></script>
  <script src="nonrigid-review.js"></script>
</body>
</html>
"""


_CSS = """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #202427;
  background: #f4f5f3;
  font-synthesis: none;
}
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
button { font: inherit; letter-spacing: 0; }
.app { height: 100%; display: grid; grid-template-rows: 48px minmax(0, 1fr); }
.topbar {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 0 14px;
  border-bottom: 1px solid #c9ceca;
  background: #fff;
}
.brand { font-weight: 750; font-size: 15px; }
.brand span { color: #626a66; font-weight: 520; margin-left: 6px; }
.provisional {
  color: #8b3a2f;
  background: #f7e8e3;
  border: 1px solid #dfb3a8;
  border-radius: 4px;
  padding: 4px 9px;
  font-size: 11px;
  font-weight: 750;
}
.counter { justify-self: end; color: #626a66; font-size: 12px; }
.workspace { min-height: 0; display: grid; grid-template-columns: 286px minmax(0, 1fr); }
.rail {
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  border-right: 1px solid #c9ceca;
  background: #eceeeb;
}
.rail-head { padding: 11px; border-bottom: 1px solid #c9ceca; }
.mouse { font-size: 13px; font-weight: 750; margin-bottom: 9px; }
.segmented { display: grid; grid-template-columns: repeat(3, 1fr); }
.segmented button {
  border: 1px solid #aeb5b0;
  border-right: 0;
  background: #fff;
  color: #505753;
  padding: 6px 4px;
  font-size: 11px;
  cursor: pointer;
}
.segmented button:first-child { border-radius: 4px 0 0 4px; }
.segmented button:last-child { border-right: 1px solid #aeb5b0; border-radius: 0 4px 4px 0; }
.segmented button.active { background: #263b3b; color: #fff; border-color: #263b3b; }
.slide-list { min-height: 0; overflow: auto; padding: 5px; }
.slide-row {
  width: 100%;
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 9px;
  gap: 6px;
  align-items: center;
  border: 0;
  border-bottom: 1px solid #d3d7d3;
  background: transparent;
  color: #303532;
  padding: 8px 6px;
  text-align: left;
  cursor: pointer;
}
.slide-row:hover { background: #fff; }
.slide-row.active { background: #fff; box-shadow: inset 3px 0 #147a78; }
.slide-row .order { color: #6d7470; font-variant-numeric: tabular-nums; font-size: 11px; }
.slide-row .name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; }
.slide-row .dot { width: 7px; height: 7px; border-radius: 50%; background: #a94636; }
.slide-row .dot.accepted { background: #2f7d4c; }
.content {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: 42px minmax(0, 1.15fr) minmax(210px, .85fr);
  background: #f7f8f6;
}
.section-head {
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr) auto 32px;
  gap: 8px;
  align-items: center;
  padding: 5px 10px;
  border-bottom: 1px solid #d3d7d3;
  background: #fff;
}
.icon-button {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 1px solid #b7bdb8;
  border-radius: 4px;
  background: #fff;
  color: #303532;
  font-size: 21px;
  line-height: 1;
  cursor: pointer;
}
.icon-button:hover { border-color: #147a78; color: #147a78; }
.icon-button:disabled { opacity: .35; cursor: default; }
.section-title { min-width: 0; display: flex; align-items: baseline; gap: 8px; }
.section-title strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.section-title span { color: #737a76; font-size: 10px; white-space: nowrap; }
.status { border-radius: 4px; padding: 4px 7px; font-size: 10px; font-weight: 750; }
.status.accepted { color: #205c37; background: #e2f1e6; border: 1px solid #a9ceb4; }
.status.rejected { color: #8b3126; background: #f8e5e1; border: 1px solid #dfaaa1; }
.primary { min-height: 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-bottom: 1px solid #c9ceca; }
figure { min-width: 0; min-height: 0; margin: 0; display: grid; grid-template-rows: 26px minmax(0, 1fr); background: #fff; border-right: 1px solid #d3d7d3; }
figure:last-child { border-right: 0; }
figcaption { display: flex; align-items: center; padding: 0 9px; color: #555d58; background: #f0f2ef; border-bottom: 1px solid #d3d7d3; font-size: 10px; font-weight: 700; }
figure img { width: 100%; height: 100%; min-height: 0; object-fit: contain; cursor: zoom-in; }
.diagnostics { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 330px; }
.metrics { min-height: 0; overflow: auto; padding: 9px 11px; background: #fff; }
.metrics dl { margin: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 5px 10px; font-size: 10px; }
.metrics dt { color: #66706a; }
.metrics dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 700; }
.warnings { margin-top: 9px; padding-top: 8px; border-top: 1px solid #d3d7d3; color: #8b3126; font-size: 10px; line-height: 1.4; }
dialog { width: calc(100vw - 28px); height: calc(100vh - 28px); max-width: none; max-height: none; padding: 0; border: 1px solid #7d8580; background: #fff; }
dialog::backdrop { background: rgba(24, 28, 26, .72); }
dialog img { width: 100%; height: 100%; object-fit: contain; }
dialog .close { position: absolute; top: 8px; right: 8px; z-index: 2; }
@media (max-width: 900px) {
  .topbar { grid-template-columns: 1fr auto; }
  .provisional { display: none; }
  .workspace { grid-template-columns: 210px minmax(0, 1fr); }
  .diagnostics { grid-template-columns: 1fr 1fr; }
  .metrics { display: none; }
  .section-title span { display: none; }
}
@media (max-width: 700px) {
  .workspace {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: 126px minmax(0, 1fr);
  }
  .rail {
    grid-template-rows: 58px minmax(0, 1fr);
    border-right: 0;
    border-bottom: 1px solid #c9ceca;
  }
  .rail-head {
    display: grid;
    grid-template-columns: auto minmax(190px, 1fr);
    gap: 10px;
    align-items: center;
    padding: 8px;
  }
  .mouse { margin: 0; white-space: nowrap; }
  .slide-list { display: flex; overflow-x: auto; overflow-y: hidden; }
  .slide-row {
    flex: 0 0 176px;
    height: 56px;
    border-right: 1px solid #d3d7d3;
    border-bottom: 0;
  }
  .slide-row.active { box-shadow: inset 0 -3px #147a78; }
  .content {
    grid-template-rows: 44px minmax(0, 1.15fr) minmax(160px, .85fr);
  }
  .section-head { padding: 5px 7px; }
}
"""


_JS = """
(() => {
  'use strict';
  const data = window.HISTOPIA_NONRIGID_REVIEW;
  const list = document.getElementById('slide-list');
  const filters = document.getElementById('filters');
  const zoom = document.getElementById('zoom');
  const zoomImage = zoom.querySelector('img');
  let filter = 'all';
  let selectedOrder = data.slides[0].order;
  let visible = [];

  const number = (value, digits = 3) => value == null ? 'n/a' : Number(value).toFixed(digits);
  const percent = value => value == null ? 'n/a' : `${(Number(value) * 100).toFixed(1)}%`;
  const rowByOrder = order => data.slides.find(row => row.order === order);

  function metric(label, value) {
    return `<dt>${label}</dt><dd>${value}</dd>`;
  }

  function renderList() {
    visible = data.slides.filter(row => filter === 'all' || (filter === 'accepted') === row.accepted);
    if (!visible.some(row => row.order === selectedOrder)) selectedOrder = visible[0]?.order;
    list.replaceChildren(...visible.map(row => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `slide-row${row.order === selectedOrder ? ' active' : ''}`;
      button.title = row.slide;
      const order = document.createElement('span');
      order.className = 'order';
      order.textContent = String(row.order).padStart(2, '0');
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = row.slide;
      const dot = document.createElement('span');
      dot.className = `dot ${row.accepted ? 'accepted' : ''}`;
      button.replaceChildren(order, name, dot);
      button.addEventListener('click', () => {
        selectedOrder = row.order;
        renderList();
        renderSlide();
      });
      return button;
    }));
    renderSlide();
  }

  function renderSlide() {
    const row = rowByOrder(selectedOrder);
    if (!row) return;
    document.getElementById('mouse').textContent = `Mouse ${data.mouse_id} \u00b7 ${data.slide_count} candidates`;
    const candidatePosition = data.slides.findIndex(item => item.order === row.order) + 1;
    document.getElementById('counter').textContent = `${candidatePosition} / ${data.slide_count} \u00b7 section ${row.order}`;
    document.getElementById('slide-title').textContent = row.slide;
    document.getElementById('slide-method').textContent = row.method;
    const status = document.getElementById('status');
    status.className = `status ${row.accepted ? 'accepted' : 'rejected'}`;
    status.textContent = row.accepted ? 'ACCEPTED CANDIDATE' : 'REJECTED \u00b7 AFFINE RETAINED';
    document.getElementById('reference').src = data.reference_asset;
    for (const name of ['affine', 'candidate', 'checker', 'magnitude']) {
      document.getElementById(name).src = row.assets[name];
    }
    const sparse = row.sparse_feature_validation;
    document.getElementById('metrics').innerHTML = [
      metric('Structural similarity', `${number(row.initial_similarity, 4)} \u2192 ${number(row.final_similarity, 4)}`),
      metric('Tissue-mask Dice', `${number(row.initial_mask_dice, 4)} \u2192 ${number(row.final_mask_dice, 4)}`),
      metric('Jacobian p01 / p99', `${number(row.jacobian_p01)} / ${number(row.jacobian_p99)}`),
      metric('Displacement p95', `${number(row.displacement_p95, 2)} px`),
      metric('Inverse consistency p95', `${number(row.inverse_consistency_p95, 2)} px`),
      metric('Sparse coherent matches', sparse?.coherent_matches ?? 'n/a'),
      metric('Sparse median residual', sparse?.status === 'available' ? `${number(sparse.initial_median_residual_px, 2)} \u2192 ${number(sparse.final_median_residual_px, 2)} px` : 'unavailable'),
      metric('Sparse p95 residual', sparse?.status === 'available' ? `${number(sparse.initial_p95_residual_px, 2)} \u2192 ${number(sparse.final_p95_residual_px, 2)} px` : 'unavailable'),
      metric('Sparse matches improved', sparse?.status === 'available' ? percent(sparse.improved_fraction) : 'n/a')
    ].join('');
    const warnings = [
      ...row.warnings,
      ...(sparse?.warnings ?? []).map(warning => `Sparse: ${warning}`)
    ];
    document.getElementById('warnings').textContent = warnings.length ? warnings.join(' \u00b7 ') : 'No acceptance warnings';
    const position = visible.findIndex(item => item.order === row.order);
    document.getElementById('previous').disabled = position <= 0;
    document.getElementById('next').disabled = position < 0 || position >= visible.length - 1;
  }

  function move(offset) {
    const position = visible.findIndex(row => row.order === selectedOrder);
    const target = visible[position + offset];
    if (!target) return;
    selectedOrder = target.order;
    renderList();
    list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  }

  filters.addEventListener('click', event => {
    const button = event.target.closest('button[data-filter]');
    if (!button) return;
    filter = button.dataset.filter;
    filters.querySelectorAll('button').forEach(item => item.classList.toggle('active', item === button));
    renderList();
  });
  document.getElementById('previous').addEventListener('click', () => move(-1));
  document.getElementById('next').addEventListener('click', () => move(1));
  document.addEventListener('keydown', event => {
    if (zoom.open) return;
    if (event.key === 'ArrowLeft') move(-1);
    if (event.key === 'ArrowRight') move(1);
  });
  document.querySelectorAll('figure img').forEach(image => image.addEventListener('click', () => {
    zoomImage.src = image.src;
    zoomImage.alt = image.alt;
    zoom.showModal();
  }));
  zoom.querySelector('.close').addEventListener('click', () => zoom.close());
  zoom.addEventListener('click', event => {
    if (event.target === zoom) zoom.close();
  });
  renderList();
})();
"""
