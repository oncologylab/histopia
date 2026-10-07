"""Static, presentation-friendly methods page for the workflow review hub."""

from __future__ import annotations

import json
from collections.abc import Iterable
from html import escape
from pathlib import Path

from histopia.visualization._review_theme import themed_review_css

_STAGE_LABELS = {
    "registration": "registration",
    "atlas": "3D atlas",
    "stain": "stain OD",
    "topology": "topology",
    "cells": "cell boundaries",
    "protein": "protein prediction",
    "protein-atlas": "cellular protein atlas",
    "annotations": "annotations",
}


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def build_methods_review(
    output_dir: Path | str,
    *,
    cohorts: Iterable[str] = (),
    stages: Iterable[str] = (),
) -> Path:
    """Write the path-free scientific methods companion for a review hub."""

    output_dir = Path(output_dir)
    cohort_ids = _ordered_unique(cohorts)
    stage_ids = _ordered_unique(stages)
    manifest = {
        "schema_version": 1,
        "method_revision": "histopia-review-methods-v2",
        "cohorts": list(cohort_ids),
        "workflow_stages": list(stage_ids),
    }
    cohort_text = ", ".join(escape(value) for value in cohort_ids) or "not declared"
    stage_text = (
        ", ".join(
            escape(_STAGE_LABELS.get(value, value.replace("_", " ")))
            for value in stage_ids
        )
        or "methods only"
    )
    html = _METHODS_HTML.replace("{{COHORTS}}", cohort_text).replace(
        "{{STAGES}}", stage_text
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "index.html").write_text(html)
    (output_dir / "methods-review.css").write_text(themed_review_css(_METHODS_CSS))
    return output_dir / "index.html"


_METHODS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Histopia methods and interpretation</title>
  <link rel="stylesheet" href="methods-review.css">
</head>
<body>
  <header class="methods-header">
    <div>
      <p class="eyebrow">Histopia methods</p>
      <h1>From native whole-slide images to measured and inferred spatial outputs</h1>
      <p class="lead">A concise scientific contract for interpreting every pane in
        this review workspace.</p>
    </div>
    <dl class="build-scope">
      <div><dt>Cohorts in this build</dt><dd>{{COHORTS}}</dd></div>
      <div><dt>Available evidence</dt><dd>{{STAGES}}</dd></div>
    </dl>
  </header>

  <nav class="methods-toc" aria-label="Methods sections">
    <a href="#contract">Reading contract</a>
    <a href="#registration">Registration</a>
    <a href="#semantic">Semantic atlas</a>
    <a href="#stain">Stain OD</a>
    <a href="#cells">Cells</a>
    <a href="#protein">Protein prediction</a>
    <a href="#topology">Topology</a>
    <a href="#annotations">Review and annotations</a>
  </nav>

  <main>
    <section id="contract" class="hero-card">
      <div>
        <p class="section-label">How to read the website</p>
        <h2>Measured evidence and model inference are never interchangeable</h2>
        <p>Histopia keeps source-native histology, registered coordinates,
          quantitative optical-density maps, cell instances, and inferred protein
          values as separate, fingerprint-bound artifacts. Registration links
          sections spatially; it does not create a new biochemical measurement.</p>
      </div>
      <div class="evidence-key" aria-label="Evidence classes">
        <article class="native"><strong>Native evidence</strong><span>Scanner pixels,
          tissue masks, and native cell geometry.</span></article>
        <article class="measured"><strong>Measured</strong><span>Continuous target OD
          sampled and validated at 4 µm/px.</span></article>
        <article class="inferred"><strong>Inferred</strong><span>Semantic regions,
          continuous topology, and per-cell protein predictions.</span></article>
      </div>
      <p class="contract-note"><strong>Hard tissue constraint.</strong> Quantitative
        stain extraction is performed only under the accepted registration tissue
        mask. Retained cell instances are also clipped to that mask and must pass
        independent optical-density, nuclear-support, and debris checks. Pixels
        outside tissue are required to remain zero in sealed outputs.</p>
    </section>

    <section class="pipeline" aria-labelledby="pipeline-title">
      <p class="section-label">Analysis sequence</p>
      <h2 id="pipeline-title">One provenance chain, multiple review scales</h2>
      <ol>
        <li><b>1</b><span><strong>Native WSI</strong>Scanner geometry and content
          bounds remain the coordinate source of truth.</span></li>
        <li><b>2</b><span><strong>Tissue and registration</strong>Reviewed masks,
          section order, and guarded transforms define cross-slide geometry.</span></li>
        <li><b>3</b><span><strong>Morphology atlas</strong>UNI2-h patch features are
          fit jointly across the serial stack.</span></li>
        <li><b>4</b><span><strong>Measurements and cells</strong>Tissue-only OD and
          native cell instances are computed independently.</span></li>
        <li><b>5</b><span><strong>Spatial inference</strong>Per-cell target models and
          connected topology use sealed upstream artifacts.</span></li>
        <li><b>6</b><span><strong>Review</strong>Visual decisions are bound to exact
          fingerprints; a draft comment is not scientific approval.</span></li>
      </ol>
    </section>

    <section id="registration" class="method-card">
      <div class="method-number">01</div>
      <div class="method-copy">
        <p class="section-label">Registration and tissue support</p>
        <h2>Serial sections are linked without replacing their native pixels</h2>
        <p>Whole-slide metadata and scanner content bounds are validated first.
          Tissue masks are reviewed before use. Physical order comes from reviewed
          anchors or an explicitly provisional morphology proposal. Rigid candidates
          use tissue shape and image features; the hybrid strategy compares direct
          reference alignment with serial-neighbor composition. Guarded affine
          refinement uses tissue-mask signed-distance fields and is accepted only
          when overlap improves within scale and anisotropy limits.</p>
        <p>Transforms map native positions into a common reference space for stack
          comparison, semantic correspondence, and 3D display. Source-native WSI
          remains the evidence shown at high zoom.</p>
      </div>
      <aside><strong>Interpretation boundary</strong>Similarity order is not a
        measured z-axis. Registered rendering may interpolate pixels, so source-space
        measurements and native-coordinate annotation remain separate.</aside>
    </section>

    <section id="semantic" class="method-card">
      <div class="method-number">02</div>
      <div class="method-copy">
        <p class="section-label">Global serial-section semantic atlas</p>
        <h2>Morphology is learned jointly across the complete section stack</h2>
        <p>Accepted tissue is sampled by default as non-overlapping 224-pixel patches
          at 0.5 µm/px. Each patch receives a frozen UNI2-h representation plus
          native and registered coordinates. L2-normalized embeddings enter one
          section-balanced PCA and one global MiniBatchKMeans space, so region IDs
          have a common meaning across slides. Reciprocal adjacent-section matches
          estimate deformation-aware correspondence. A section-level additive batch
          correction is retained only when anchor distance and slide-attributable
          variation improve while within-slide neighborhoods are preserved.</p>
      </div>
      <aside><strong>Lower-bound context</strong>Semantic regions are coarse,
        unsupervised morphology territories. They can mix multiple cell types and
        states; they are context and a reporting stratum, not cell-type truth.</aside>
    </section>

    <section id="stain" class="method-card">
      <div class="method-number">03</div>
      <div class="method-copy">
        <p class="section-label">Quantitative brightfield stain profiling</p>
        <h2>Continuous target optical density is measured at 4 µm/px</h2>
        <p>Within the accepted tissue mask, Histopia estimates slide-specific glass
          white and illumination, benchmarks fixed and adaptive stain vectors within
          each assay family, and shrinks selected vectors toward a robust cohort
          template. The sealed result retains raw target OD, conservative corrected
          OD, counterstain, residual, confidence, and tissue support. Correction is
          accepted only when rank preservation and glass, counterstain-leakage, and
          spatial-background guards pass.</p>
        <p>An optional tissue-only adaptive layer subtracts a stable broad target-OD
          floor or bounded counterstain-conditioned nuisance envelope. It must retain
          strong signal, preserve ranks, and stay within suppression limits. If a
          guard fails, the physical raw/corrected map remains the explicit fallback.</p>
      </div>
      <aside><strong>Measurement boundary</strong>The OD layer does not imply detail
        finer than 4 µm/px, absolute concentration, cross-antibody comparability, or
        automatic cell expression. Palette color changes display only—not numeric
        OD.</aside>
    </section>

    <section id="cells" class="method-card">
      <div class="method-number">04</div>
      <div class="method-copy">
        <p class="section-label">Native whole-slide cell boundaries</p>
        <h2>Two-scale CPSAM instances are stitched and debris-guarded</h2>
        <p>The production profile preserves the benchmarked CPSAM Combined passes
          at cell-probability/diameter settings −2.5/26 and −1.75/15, both with flow
          threshold 0 and minimum size 15, followed by containment merging at an
          intersection-over-smaller threshold of 0.1. Overlapping native WSI tiles
          are stitched by deterministic mutual-best matches.</p>
        <p>The accepted tissue mask is a hard outer boundary, not sufficient cell
          evidence. Retained instances must also pass local stain support, adaptive
          hematoxylin-like nuclear support, and guards for glass, folds, detached
          fragments, sparse pale debris, and dead-cell aggregates. Each section is
          sealed as a native-resolution uint32 label image with complete tile and QC
          accounting.</p>
      </div>
      <aside><strong>Validation boundary</strong>Centroid benchmarks test detection
        and containment, not contour truth. High-resolution visual boundary review is
        still required for scientific approval.</aside>
    </section>

    <section id="protein" class="method-card feature-card">
      <div class="method-number">05</div>
      <div class="method-copy">
        <p class="section-label">Stain-invariant per-cell protein prediction</p>
        <h2>One antibody target is transferred at a time between serial sections</h2>
        <p>Training outcomes come only from accepted tissue-masked target OD at
          4 µm/px. Fractional OD-pixel/cell overlap yields one measured outcome per
          eligible native cell without inventing pixel-scale stain detail. Predictors
          combine a high-resolution, target-chromogen-free UNI2-h vector per cell,
          exact boundary shape, hematoxylin-only texture, neighborhoods at
          16/32/64/128 µm, and registered x/y/z Fourier position. Marker name, source
          RGB target stain, target OD, and section order are prohibited features.
          Coarse semantic regions remain a weak contextual prior.</p>
        <p>Candidate families include tree and hurdle controls, multiscale towers,
          cross-attention, inductive dual-bank attention, graph transformers, and
          shared-trunk multi-target regularization. Evaluation uses spatial blocks,
          held-out sections, and held-out mice. Training-visible variants are labeled
          capacity diagnostics and cannot masquerade as transfer validation.</p>
        <p>Architecture selection is repeated independently for each antibody under
          one fingerprinted feature table, split, cell budget, epoch budget, and seed.
          Held-out 64-µm aggregate Spearman is ranked first, followed by cell-level
          Spearman, with fold stability, OD bias, MAE, and semantic-baseline parity as
          guards. Shared-trunk fits retain a distinct antibody head and exclude the
          held-out mouse from every primary and auxiliary target table. The browser
          recommends a role-specific model only after fingerprint-bound review: an
          approved measured-fit capacity view where that antibody was observed or an
          approved held-out transfer elsewhere. If neither passes, it says no approved
          model and leaves exploratory selection explicit.</p>
        <p>Evidence is tiered per antibody even though every result uses the same
          open-ended registry. Repeated assays can receive held-out-mouse metrics. A
          single-mouse target remains a labelled capacity diagnostic plus unvalidated
          exploratory transfer and cannot be promoted as a generalizing model. A
          rejected corrected OD map is never admitted as training truth.</p>
      </div>
      <aside><strong>What the three panes mean</strong>On a measured target section:
        prediction, observed target OD, and absolute error. Elsewhere: inferred target,
        the section's actual antibody when available, and the nearest registered
        observed target as context. No residual is shown without target truth.</aside>
    </section>

    <section id="topology" class="method-card">
      <div class="method-number">06</div>
      <div class="method-copy">
        <p class="section-label">Continuous semantic topology</p>
        <h2>Reviewed tissue masks define the envelope; semantic patches define
          fields</h2>
        <p>The connected anatomical envelope is reconstructed from registration
          tissue masks, while selected-K semantic fields are propagated separately.
          Linear signed-distance, guarded correspondence-flow, and shape-preserving
          interpolation candidates are compared with held-out mask tests. Numerical
          samples between sections create a continuous field for visualization and
          measurement; they are not synthesized histology sections. Display meshes
          are derived from sealed dense fields and do not replace them.</p>
      </div>
      <aside><strong>Z provenance</strong>Physical thickness or positions must be
        supplied. When morphology-based gap calibration is weak, Histopia abstains
        and labels z spacing uniformly assumed. Volume and surface area are then model
        estimates, not direct physical measurements.</aside>
    </section>

    <section id="annotations" class="method-card">
      <div class="method-number">07</div>
      <div class="method-copy">
        <p class="section-label">Annotations, provenance, and review</p>
        <h2>Every editable or promotable result is bound to exact upstream bytes</h2>
        <p>Pathology polygons are revisioned GeoJSON in source-native pixel
          coordinates and are bound to the registration and semantic fingerprints.
          Stain, cell, semantic, topology, and protein artifacts similarly seal their
          input identities, geometry, configuration, and content digests. Immutable
          tile URLs prevent stale browser data from silently replacing current data.
          Review drafts support triage, notes, holds, and rejections; formal approval
          remains a separate fingerprint-validated action.</p>
      </div>
      <aside><strong>Review boundary</strong>Presence on this website means the
        artifact is available for review. It does not by itself mean that the result,
        biological interpretation, or model has been scientifically approved.</aside>
    </section>

    <section class="references" aria-labelledby="references-title">
      <p class="section-label">Method context</p>
      <h2 id="references-title">Related methodological inspiration</h2>
      <p>Histopia independently adapts hierarchical morphology and cross-section
        ideas from <a href="https://www.nature.com/articles/s41587-023-02019-9"
        target="_blank" rel="noreferrer">iStar</a> and
        <a href="https://www.nature.com/articles/s41592-025-02770-8"
        target="_blank" rel="noreferrer">iSCALE</a> to a serial-section,
        per-native-cell protein setting. It does not reproduce their spatial
        transcriptomics tasks or treat predicted protein as a measured assay.</p>
    </section>
  </main>
  <footer>Histopia · Histology Spatial Topology for Omics Profiling and
    Inter-section Alignment</footer>
</body>
</html>
"""


_METHODS_CSS = """
:root{font-family:Arial,system-ui,sans-serif;color:#111827;background:#f3f6fa}
*{box-sizing:border-box}
html,body{width:100%;min-width:0;margin:0;overflow-x:hidden}
body{background:#f3f6fa;color:#111827}
.methods-header{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,.7fr);
gap:28px;align-items:end;padding:30px max(24px,calc((100vw - 1240px)/2));
border-bottom:1px solid #dbe3ec;background:#fff}
.eyebrow,.section-label{margin:0 0 7px;color:#1d4ed8;font-size:11px;
font-weight:800;letter-spacing:.09em;text-transform:uppercase}
h1{max-width:900px;margin:0;font-size:clamp(25px,3vw,42px);line-height:1.08}
.lead{max-width:760px;margin:12px 0 0;color:#4b5563;font-size:16px;line-height:1.5}
.build-scope{display:grid;gap:8px;margin:0;padding:14px;border:1px solid #dbe3ec;
border-radius:10px;background:#f9fafb}
.build-scope div{display:grid;grid-template-columns:130px minmax(0,1fr);gap:10px}
.build-scope dt{color:#64748b;font-size:11px;font-weight:800;text-transform:uppercase}
.build-scope dd{margin:0;font-size:13px;font-weight:700;overflow-wrap:anywhere}
.methods-toc{position:sticky;z-index:3;top:0;display:flex;gap:4px;width:100%;
padding:8px max(20px,calc((100vw - 1240px)/2));overflow-x:auto;
border-bottom:1px solid #cfd8e3;background:rgba(255,255,255,.97)}
.methods-toc a{flex:0 0 auto;padding:7px 10px;border-radius:7px;color:#334155;
font-size:12px;font-weight:700;text-decoration:none;white-space:nowrap}
.methods-toc a:hover{color:#1d4ed8;background:#eaf2ff}
main{display:grid;gap:16px;width:min(1240px,calc(100% - 40px));margin:22px auto 38px}
.hero-card,.pipeline,.method-card,.references{border:1px solid #dbe3ec;
border-radius:10px;background:#fff;box-shadow:0 8px 24px rgba(15,23,42,.04)}
.hero-card{display:grid;grid-template-columns:minmax(0,1fr) minmax(350px,.8fr);
gap:24px;padding:24px}
h2{margin:0 0 10px;font-size:20px;line-height:1.25}
p{line-height:1.56}
.hero-card p,.method-copy p,.references p{margin:0;color:#374151}
.evidence-key{display:grid;gap:8px}
.evidence-key article{display:grid;grid-template-columns:110px minmax(0,1fr);gap:10px;
align-items:start;padding:11px 12px;border:1px solid #dbe3ec;border-left-width:5px;
border-radius:8px;background:#f9fafb}
.evidence-key .native{border-left-color:#2563eb}
.evidence-key .measured{border-left-color:#17835b}
.evidence-key .inferred{border-left-color:#b56a00}
.evidence-key span{color:#4b5563;font-size:12px;line-height:1.45}
.contract-note{grid-column:1/-1;padding:12px 14px!important;border-radius:8px;
background:#eef6ff;color:#1e3a5f!important}
.pipeline{padding:22px 24px}
.pipeline ol{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;
margin:16px 0 0;padding:0;list-style:none}
.pipeline li{position:relative;display:grid;grid-template-columns:25px minmax(0,1fr);
gap:8px;min-width:0;padding:12px;border:1px solid #dbe3ec;border-radius:8px;
background:#f9fafb}
.pipeline b{display:grid;place-items:center;width:25px;height:25px;border-radius:50%;
background:#1d4ed8;color:#fff;font-size:12px}
.pipeline span{color:#4b5563;font-size:11px;line-height:1.4}
.pipeline strong{display:block;margin-bottom:3px;color:#111827;font-size:12px}
.method-card{display:grid;grid-template-columns:54px minmax(0,1fr) 280px;gap:18px;
align-items:start;padding:22px 24px;scroll-margin-top:58px}
.method-number{display:grid;place-items:center;width:44px;height:44px;border-radius:9px;
background:#0f172a;color:#fff;font-size:14px;font-weight:800}
.method-copy p+p{margin-top:10px}
.method-card aside{padding:13px 14px;border:1px solid #e7c98d;border-radius:8px;
background:#fff8e7;color:#5f4515;font-size:12px;line-height:1.5}
.method-card aside strong{display:block;margin-bottom:5px;color:#714b00}
.feature-card{border-color:#a9c7f7}
.feature-card .method-number{background:#1d4ed8}
.references{padding:22px 24px}
.references a{color:#1d4ed8;font-weight:700}
footer{padding:18px 24px;border-top:1px solid #dbe3ec;background:#fff;
color:#64748b;font-size:12px;text-align:center}
@media(max-width:1050px){
 .methods-header{grid-template-columns:1fr;gap:18px}
 .pipeline ol{grid-template-columns:repeat(3,minmax(0,1fr))}
 .method-card{grid-template-columns:48px minmax(0,1fr)}
 .method-card aside{grid-column:2}
}
@media(max-width:680px){
 .methods-header{padding:22px 16px}
 .build-scope div{grid-template-columns:1fr;gap:2px}
 .methods-toc{padding:7px 10px}
 main{width:calc(100% - 20px);margin:12px auto 26px;gap:10px}
 .hero-card{grid-template-columns:1fr;gap:16px;padding:17px}
 .contract-note{grid-column:1}
 .evidence-key article{grid-template-columns:90px minmax(0,1fr)}
 .pipeline{padding:17px}
 .pipeline ol{grid-template-columns:1fr}
 .method-card{grid-template-columns:1fr;gap:10px;padding:17px;scroll-margin-top:55px}
 .method-number{width:38px;height:38px}
 .method-card aside{grid-column:1}
 .references{padding:17px}
}
@media print{
 .methods-toc{display:none}
 .methods-header,main{width:100%;padding-inline:0}
 .method-card,.hero-card,.pipeline,.references{box-shadow:none;break-inside:avoid}
}
"""
