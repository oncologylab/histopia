# Histopia Methods Overview

Histopia is a serial-section histology workflow that keeps native image
evidence, registered geometry, quantitative stain measurements, cell
instances, semantic regions, topology reconstructions, and protein predictions
as distinct provenance-bound artifacts. This separation is central to the
method: an inferred value is never silently substituted for a measured assay,
and registered resampling is not used to create source-space measurements.

This document provides the concise end-to-end method used by the scientific
review website. Detailed operational and algorithmic contracts are available
in the linked workflow documents.

## Coordinate and Measurement Contract

| Quantity | Coordinate or sampling space | Interpretation |
|---|---|---|
| Histology | Source-native scanner pixels | Highest-resolution image evidence |
| Tissue mask | Registration thumbnail plus native projection | Accepted support for downstream computation |
| Cell instances | Source-native scanner pixels | One integer label per retained native cell |
| Stain target OD | Source content bounding box at 4 µm/px | Continuous relative optical density |
| Semantic patches | Native patches with registered physical coordinates | Coarse morphology context shared across sections |
| Protein prediction | One value per native cell | Inferred target OD summary, not a measured stain |
| Topology | Registered x/y and supplied or qualified z geometry | Continuous model of tissue and semantic fields |
| Annotations | Source-native scanner pixels | Revisioned human pathology polygons |

The accepted tissue mask is a hard scientific constraint. Quantitative stain
extraction occurs only under that mask. Cell instances are clipped to it and
must pass additional stain-support, nuclear-support, and debris checks. Sealed
continuous maps and label images are required to be zero outside accepted
tissue.

## Registration and Section Order

Whole-slide metadata, scanner content bounds, physical calibration, and slide
identity are validated before image processing. Tissue masks are proposed from
brightfield thumbnails and must be visually reviewed. Physical order comes
from an explicit manifest when available. A morphology-based proposal may fill
unassigned positions, but it remains provisional and is not interpreted as a
measured z-axis.

Rigid alignment candidates combine tissue-shape moments and image features.
The hybrid strategy compares direct-to-reference alignment with serial-neighbor
composition and retains the better tissue overlap. Guarded affine refinement
uses signed-distance fields from tissue masks rather than target stain
intensity. It is accepted only when tissue Dice improves and relative scale and
anisotropy remain within configured bounds. The approved transform links
native positions to a common reference canvas, while native WSI pixels remain
the source of high-resolution review and quantitative measurements.

See [registration.md](registration.md) for configuration, review gates,
non-rigid diagnostics, and full-resolution export.

## Global Serial-Section Semantic Atlas

Only patches with sufficient accepted-tissue coverage are encoded. The default
sampling uses non-overlapping 224-pixel fields at 0.5 µm/px. Each patch stores a
frozen UNI2-h representation together with source-grid, native-pixel, and
registered-reference coordinates.

Histopia fits a single morphology atlas across the complete section stack. It
L2-normalizes embeddings, fits a section-balanced PCA, and clusters the shared
space with MiniBatchKMeans. Reciprocal deformation-aware correspondences
between adjacent sections estimate a smooth local relationship without
warping accepted image evidence. A section-level additive batch correction is
accepted only when anchor distance and slide-attributable variance improve
while within-slide neighborhoods are preserved. Topology regularization then
encourages cross-section consistency while retaining observed feature support.

Semantic regions are a lower-bound contextual representation. A region may
contain several cell types and states; it must not be interpreted as single-cell
identity or as a primary protein predictor.

See [semantic_atlas.md](semantic_atlas.md) for extraction, model provenance,
batch-correction guards, K selection, and validation.

## Tissue-Masked Brightfield Stain Quantification

Stain measurements are made in each source section at the configured physical
sampling resolution, currently 4 µm/px for the validated review data.
Registration transforms link sections for comparison but do not define the
measurement.

Within accepted tissue, Histopia:

1. Estimates a slide-specific glass-white reference and low-order illumination
   field using non-tissue pixels.
2. Benchmarks fixed, legacy, Macenko, and nonnegative-matrix-factorization
   stain vectors within each assay family.
3. Selects and cohort-shrinks a vector using reconstruction error, glass
   leakage, counterstain-only target leakage, prior drift, convergence, and
   bootstrap stability.
4. Retains raw target OD, conservative corrected target OD, counterstain,
   residual, confidence, and tissue-support arrays.
5. Accepts nuisance correction only when signal ranks are preserved and glass,
   counterstain leakage, and spatial background do not worsen.

An optional adaptive output removes a broad tissue-level target-OD floor or a
bounded counterstain-conditioned nuisance envelope. It must pass support,
bootstrap stability, rank-preservation, signal-retention, and suppression
guards. A failed proposal falls back explicitly to the physical raw or
corrected map; it is never hidden as a successful correction.

Color is a display encoding only. The OD result does not establish absolute
concentration, cross-antibody normalization, clinical positivity, or detail
finer than 4 µm/px.

See [stain_quantification.md](stain_quantification.md) for assay-specific
models, correction guards, artifacts, and approval.

## Native Whole-Slide Cell Boundaries

The production cell profile is a whole-slide adaptation of the benchmarked
CPSAM Combined-containment method. It uses two Cellpose passes with
cell-probability/diameter pairs −2.5/26 and −1.75/15, flow threshold 0, and
minimum size 15. Instances are merged at an intersection-over-smaller
containment threshold of 0.1. Native WSI tiles overlap and are joined through
deterministic mutual-best instance matches.

Tissue membership alone is not accepted as cell evidence. Retained instances
must pass local optical-density support, hematoxylin-like nuclear support, and
guards targeting glass, dark neutral folds, detached fragments, sparse pale
debris, and dead-cell aggregates. The final artifact is a native-resolution
uint32 label pyramid with complete tile accounting, zero-outside-tissue QC, and
content hashes.

Centroid benchmarks measure detection and containment, not contour truth.
High-resolution boundary inspection remains necessary before scientific
approval.

See [cell_segmentation.md](cell_segmentation.md) for exact guards, thresholds,
resumability, and review requirements.

## Stain-Invariant Per-Cell Protein Prediction

One antibody target is modeled at a time. Training outcomes originate only
from an accepted tissue-masked target-OD map. Fractional overlap between
4 µm/px OD pixels and native cell compartments is aggregated to one eligible
cell outcome without inventing native-resolution stain detail.

The current multiscale feature schema combines:

- one high-resolution, target-chromogen-free UNI2-h representation per cell;
- exact boundary morphology;
- hematoxylin-only texture;
- neighborhood summaries at 16, 32, 64, and 128 µm; and
- registered x/y/z Fourier position.

The target chromogen, target OD, marker name, raw source-stain RGB summaries,
and section order are prohibited predictors. Semantic regions provide weak
context and reporting strata rather than hard cell-type gates.

Candidate models include ExtraTrees and hurdle controls, portable multiscale
feature towers, cross-attention, inductive dual-bank attention, inductive graph
transformers, and shared-trunk multi-antibody regularization. Same-slide
validation uses complete spatial blocks with an exclusion buffer. Transfer is
evaluated by held-out section and held-out mouse, with 64-µm aggregate Spearman
as the primary ranking metric followed by cell-level Spearman. Training-visible
fits are capacity diagnostics and are never presented as unbiased transfer
performance.

Model choice is made independently for each antibody with a fingerprinted,
fixed-budget architecture sweep. The primary score is held-out-mouse spatial
rank agreement after 64-µm aggregation, followed by cell-level rank agreement;
fold stability, OD bias, MAE, and semantic-baseline parity are validation
guards. In the shared-trunk candidate, each antibody retains its own output
head. The held-out mouse is removed from all primary and auxiliary antibody
tables, preventing cross-target multi-task training from leaking that mouse's
measurements into its transfer evaluation. Clone-specific or chemistry-specific
assays remain separate targets unless equivalence is explicitly demonstrated.

Each selected architecture produces two clearly labelled views. The
leave-one-mouse-out result is the transfer estimate used on sections without
that target stain. A separate training-visible result is a reconstruction
upper bound used only on measured target sections. Website recommendation is
section-aware, and results are added through a target-agnostic registry so a
larger antibody panel does not alter the artifact or review schema. Incomplete
or invalid production runs are excluded atomically rather than exposed as
partially available models.

Evidence remains target-specific inside that shared registry. Targets repeated
across mice receive held-out-mouse metrics; a target observed in only one mouse
is explicitly limited to a training-visible capacity diagnostic plus
unvalidated exploratory transfer. If no accepted corrected continuous OD map
exists, that slide is not admitted as training truth. Thus panel size never
overrides stain QC or converts a single-reference experiment into a validated
cross-mouse claim.

On a section with target truth, the reviewer shows predicted target OD,
observed target OD, and absolute cell error. On an unmeasured section, it shows
the per-cell target prediction, the section's actual antibody map when
available, and the nearest registered observed target section as spatial
context. It never draws a residual when target truth is absent.

The design independently adapts hierarchical morphology and cross-section
ideas from [iStar](https://www.nature.com/articles/s41587-023-02019-9),
*Inferring super-resolution tissue architecture by integrating spatial
transcriptomics with histology*, and
[iSCALE](https://www.nature.com/articles/s41592-025-02770-8), *Scaling up
spatial transcriptomics for large-sized tissues*. Histopia does not reproduce
their transcriptomics tasks and never treats predicted protein as a measured
assay.

See [protein_prediction.md](protein_prediction.md) for feature leakage rules,
model protocols, harmonization, promotion gates, and interactive review.

## Cellular Protein Atlas

The browser atlas joins the sealed cell geometry and selected
leave-one-mouse-out prediction for each antibody without refitting either
artifact. One morphology-aware glyph is drawn per cell using its registered
centroid and native-boundary area, eccentricity, and orientation. Whole-volume
views use a deterministic bounded sample from every section; the complete cell
set is loaded for a selected section or its immediate neighbours on demand.

Three.js renders the 3D stack, linked orthogonal projections, isolated section
view, anatomical envelope, and cutaway entirely in the client browser. Protein
arrays are fingerprinted static binary chunks composited in a Web Worker. The
site has no CDN or live API dependency and retains a fingerprinted static
preview when WebGL is unavailable.

Measured, predicted, and residual values remain visually and semantically
distinct. Observed and residual controls are enabled only where target truth
exists; a training-visible reconstruction is never substituted into the
held-out transfer volume. Promoted targets are visible by default, while
failed or insufficiently validated targets require an explicit exploratory
opt-in. X/Y are registered coordinates, Z provenance is labelled, and observed
OD remains a 4-micrometre-per-pixel measurement aggregated per cell.

See [cellular_protein_atlas.md](cellular_protein_atlas.md) for the static data
contract, build interface, and review checklist.

## Continuous Semantic Topology

The outer anatomical envelope is reconstructed from reviewed registration
masks, not semantic patches. Selected-K semantic probabilities define separate
interior fields. Histopia compares linear signed-distance, guarded
correspondence-flow, and shape-preserving interpolation using held-out mask
tests. Numerical samples between observed sections form a continuous field;
they are not inferred histology sections.

Physical section thickness or z positions must be supplied. If a z manifest is
unavailable, morphology-based gap calibration is evaluated with held-out
endpoints. When calibration is inadequate, Histopia abstains from inserting
virtual gaps and labels spacing as uniformly assumed. Under inferred or
assumed z geometry, volume and surface area are model estimates rather than
direct physical measurements.

See [topology.md](topology.md) for reconstruction candidates, metrics, surfaces,
z provenance, and review.

## Annotations, Review, and Provenance

Pathology annotations are GeoJSON Polygon or MultiPolygon features in
source-native pixel coordinates. Each revision is bound to the exact
registration-result bytes and semantic-result fingerprint. Optimistic revision
checks prevent one browser session from overwriting a newer revision.

Every workflow result seals upstream identities, geometry, scientific
configuration, and content digests. Immutable tile URLs prevent stale browser
data from silently replacing the reviewed artifact. Browser notes and draft
accept/hold/reject states are triage evidence; formal approval is a separate
fingerprint-validated operation.

Presence in the review website therefore means that an artifact is available
for inspection. It does not, by itself, mean that the method, result, or
biological interpretation has been scientifically approved.

See [pathology_features_and_annotations.md](pathology_features_and_annotations.md)
for annotation storage and spatial feature definitions.
