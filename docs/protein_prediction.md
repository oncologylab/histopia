# Stain-invariant per-cell protein prediction

Histopia's protein workflow learns one biological target at a time from cells
with validated stain measurements, then applies that target model to cells on
other serial sections. Predictions are inferences: they are never substituted
for measured target optical density.

The implementation independently adapts the hierarchical morphology and
multi-section ideas described by iStar and iSCALE. Raw-stain embeddings are
permitted only as leakage controls.

## Scientific contract

- Cell geometry and boundaries remain in native source coordinates.
- Stain outcomes originate only from an accepted tissue-masked derived
  `4 µm/px` OD map. Counterstain-conditioned v3 outcomes are consumed directly
  from their sealed sidecar without applying another floor or rescaling them.
- Fractional OD-pixel/cell overlaps are aggregated without inventing
  native-resolution stain detail.
- Morphology input is a fixed synthetic rendering of the counterstain or
  hematoxylin channel under the accepted tissue mask.
- Coarse semantic regions are a lower-bound control and reporting stratum, not
  primary predictors; they can contain several cell states and types.
- The v3 feature schema combines a high-resolution stain-neutral UNI2-h vector
  per cell with exact boundary shape, hematoxylin-only texture, neighborhood
  summaries at 16/32/64/128 µm, and registered x/y/z Fourier position. The
  target chromogen is explicitly excluded.
- Source stain, marker name, raw RGB summaries, target OD, and section order are
  prohibited model features.

Familiar spellings normalize to stable target IDs (`E-Cad`/`Ecad` to `ecad`,
`N-Cad`/`Ncad` to `ncad`, and `c-Jun`/`cJun` to `cjun`) while retaining exact
assay-domain provenance. Different antibody clones or detection chemistries
remain separate assay domains unless an explicit equivalence validation pools
them. The study and review registries are target-agnostic, so adding a protein
adds a configuration, table, and sealed model entry rather than a new schema.

## Prepared cell table

`CellExpressionTable` is a sealed NPZ artifact with one row per native cell. It
contains portable cell and section identifiers, native/reference coordinates,
stain-neutral features, optional measured OD, optional validated binary labels,
measurement coverage, semantic support, and upstream fingerprints.

Use `aggregate_cell_measurements()` to summarize fractional overlaps. If a
validated pixel threshold exists, the default binary policy is positive at
`>=25%` positive compartment area, negative at `<=5%`, and unclassified between
those limits. A cell needs at least two effective OD pixels and 70% compartment
coverage to become a training measurement. Targets without an accepted
threshold remain continuous-only.

The target configuration declares the cell statistic. `mean` is appropriate
for broad membrane/cytoplasmic signal; `q90` preserves focal nuclear peaks
without claiming pixel-level prediction. Cross-section OD harmonization is an
explicit configuration choice. Setting `od_harmonization = "none"` keeps the
validated per-section OD unchanged and is preferred when v3 background
correction already defines the scientific measurement.

## Model and evaluation

Candidate selection compares ExtraTrees and hurdle-MLP controls with portable
multiscale feature towers, cross-attention, inductive dual-bank attention,
inductive graph transformers, and a shared-trunk/multi-antibody regularizer.
Relational training graphs contain training cells only; held-out morphology
cannot enter message passing. Their two banks are target-free morphology and
registered physical XYZ; neighboring target OD is used only to estimate
uncertainty, never as an input to the prediction mean. The primary metric is
held-out 64-µm aggregate
Spearman, followed by cell-level Spearman. A benchmark winner is recorded
separately from the deployed model until its portable artifact passes
validation. Stable binary labels add a balanced classification head. Portable
neural candidates use JSON metadata and NPZ arrays. ExtraTrees research
candidates retain a version-fingerprinted joblib artifact; serving uses sealed
NPZ predictions and never unpickles that estimator.

Same-slide validation uses complete `1,024 µm` spatial blocks with a `112 µm`
exclusion buffer. Section and mouse generalization use
`leave_one_group_out()` with section and mouse identifiers respectively. A
production model may be refitted on all validated mice only after frozen
evaluations pass.

Newly completed mice enter first as external holdouts. Their target outcomes
are excluded from fitting and architecture selection, and repeated-assay
harmonization derives its reference curve only from the declared training
mice. The frozen all-training candidate is then evaluated on the new mouse in
a separate `frozen_external_mouse_holdout` metrics block. Once those results
influence feature engineering or model choice, that mouse is no longer called
untouched; it can join the next fingerprinted training round while a later
mouse becomes the holdout.

Post-fit morphology transfer follows a separate, write-once two-stage gate.
Candidate blends use optical density from a fitted training-only bank indexed
only by the stain-neutral cell-morphology feature group; query outcomes never
enter that bank and exact self matches are excluded. Relational models carry
the bank in their portable artifact. Other portable architectures reconstruct
the same bank only from the fingerprinted feature table rows belonging to the
declared training cohorts, and bind that table digest and cohort list into the
protocol. After acceptance, those exact morphology keys and training-only OD
outcomes are sealed into the portable model; leave-one-mouse-out artifacts get
a separate bank with that fold's mouse excluded. Blend weights and neighbor
counts are selected on declared
development mice, sealed, and then opened once on disjoint confirmation mice.
A refinement is accepted only when MAE improves in every confirmation mouse
without a material mean Spearman loss. The source-model digest, cohort roles,
candidate grid, selected controls, and all baseline/refined metrics are
retained as immutable evidence. This allows new mice to improve calibration
without retroactively relabeling a tuned cohort as an untouched validation
set.

Architecture selection is target-specific. A fingerprinted sweep evaluates
every candidate with the same feature table, held-out-mouse folds, cell budget,
epoch budget, and random seed. The primary ranking statistic is held-out
64-µm aggregate Spearman; cell-level Spearman, fold stability, mean/median OD
bias, MAE, and performance relative to the semantic-region baseline are
retained as guards. This avoids forcing proteins with different spatial
patterns into one model family and keeps the selected benchmark distinct from
the subsequently sealed production fit.

The shared-multitask candidate has one target-free morphology/geometry trunk
and a separate regression head for each antibody table. For a held-out mouse,
that mouse is removed from the primary target and from every auxiliary target
before fitting. The portable production artifact retains only the primary
target head needed for inference, plus the exact auxiliary target IDs and table
fingerprints used during training. It therefore gains regularization from
other antibodies without changing the one-target-per-result scientific
contract or allowing an observed target OD to become an input feature.

The multi-mouse fitter also offers an explicitly diagnostic
`training-visible` protocol. Its final all-training selected model is used
on measured training sections to establish a reconstruction/capacity upper
bound and on other sections of the same mouse for within-mouse transfer. Those
fit metrics are never eligible for promotion and the result retains its nested
leave-one-mouse-out reference metrics. This permits a strong measured-section
comparison without presenting data leakage as generalization.

```bash
python -m pip install -e ".[protein]"
histopia-protein preflight protein.toml cells-perk.npz
histopia-protein prepare protein.toml cells-perk.npz
histopia-protein benchmark protein.toml protein-run/cell_expression_table.npz
histopia-protein fit protein.toml protein-run/cell_expression_table.npz
histopia-protein approve protein-run --reviewer reviewer-id --binary

# Multi-mouse GPU capacity diagnostic; not a promotion evaluation.
histopia-protein study-fit protein.toml study.json table.npz output-run \
  --geometry-cache geometry-cache --architecture cross_attention \
  --prediction-protocol training-visible --device cuda

# Target-specific shared-trunk fit. Repeat --auxiliary-table for each distinct
# antibody table; the held-out mouse is excluded from every table in each fold.
histopia-protein study-fit target.toml study.json target-table.npz output-run \
  --geometry-cache geometry-cache --architecture shared_multitask \
  --prediction-protocol leave-one-mouse-out --device cuda \
  --auxiliary-table auxiliary-a.npz \
  --auxiliary-table auxiliary-b.npz

# Derive the labelled measured-fit view from an exact transfer fit without
# repeating any model optimization.
histopia-protein study-fit protein.toml study.json table.npz measured-fit-run \
  --geometry-cache geometry-cache --architecture multi_tower \
  --prediction-protocol training-visible --device cuda \
  --reuse-models-from transfer-run

# Rebuild a target table with the reusable multiscale per-cell schema.
histopia-protein study-table protein.toml study.json table-v3.npz \
  --geometry-cache geometry-cache \
  --feature-schema native-hdab-neutral-cell-multiscale-v3 \
  --harmonization-reference-cohort 4312 \
  --harmonization-reference-cohort 4630 \
  --harmonization-reference-cohort 6180

# Fit only the established mice; any other measured mouse in study.json is
# evaluated by the frozen final fit and reported as an external holdout. Limit
# whole-slide streaming to that new mouse so completed training-mouse rasters
# are not regenerated; this does not narrow fitting or table-level evaluation.
histopia-protein study-fit protein.toml study.json table-v3.npz holdout-run \
  --geometry-cache geometry-cache --architecture multi_tower --device cuda \
  --prediction-cohort 6034 \
  --training-cohort 4312 --training-cohort 4630 --training-cohort 6180

# Several hosts can claim disjoint fingerprinted tasks from one shared sweep.
histopia-protein sweep-create sweep-spec.json sweep --hours 8
histopia-protein sweep-worker sweep/sweep.json --worker-id gpu-worker-1
histopia-protein sweep-status sweep/sweep.json
histopia-protein sweep-select sweep-part-a/sweep.json sweep-part-b/sweep.json
```

Results remain unavailable as promoted models until explicit approval. Poor
models retain their failure reasons and do not receive a production overlay.

For repeated-stain real data, `run-real` validates the registration, cell,
stain, and semantic fingerprints before doing any work. It renders only the
counterstain channel in a fixed palette for UNI2-h, assigns each 4 µm OD pixel
fractionally with a 2×2 subpixel grid under the tissue mask, benchmarks every
target section as a held-out section, and streams predictions one section at a
time:

```bash
histopia-protein run-real protein.toml \
  --registration-run registration-run --cell-run cell-run \
  --stain-run stain-run --semantic-run semantic-run \
  --model-cache-dir external-model-cache --mouse-id mouse-id
```

Generated categorical TIFF sub-pyramids are never treated as cell labels:
review and assembly derive reduced levels from the native label plane with
nearest-neighbor sampling. At overview zoom, cell overlays are transparent and
appear once the image reaches useful cellular detail.

## Interactive review

Configure a protein run beside its exact registration and cell runs, rebuild
the workflow review, and open `?view=protein`. Three expression-first panes show
predicted target OD, observed target OD, and absolute cell error on measured
target sections. On an unmeasured section they instead show the per-cell target
prediction, that section's actual antibody map (when quantified), and the
nearest registered observed target section. A residual is never shown where no
target truth exists. If the current section has no quantitative stain map, a
neutral tissue silhouette is shown instead of inventing a stain value. Histology
and cell-boundary layers are deliberately absent from this expression-focused
page. Protein tiles use immutable digests, bounded prediction/label-overview
caches, and an explicit loading state until the first tiles are drawn.

Separate target, architecture, and model selectors may contain several targets
and variants for the same mouse and section. “Recommended for section” is
available only for a fingerprint-approved role-specific model: a measured-fit
capacity view on a section with target truth or an approved held-out transfer
elsewhere. When neither has passed review, the control instead says “No approved
model for section.” Exploratory comparisons remain selectable but cannot silently
become the default ahead of a reviewed result.

The registry and public manifest are open-ended: a new antibody contributes a
display label, assay-domain identity, sealed transfer result, and optional
measured-fit diagnostic under the existing schema. Clone-specific assays such
as MYC and MYC(CST) remain separately selectable unless an explicit assay
equivalence study justifies pooling them. Model publication is atomic, so an
incomplete run cannot appear as a partially populated target in the browser.

Evidence is tiered independently for every target. Repeated assays in two or
more mice can support leave-one-mouse-out evaluation. A target measured in only
one mouse can still provide a labelled training-visible capacity diagnostic
and exploratory predictions on the complete section stack, but it has no
cross-mouse validation and cannot be promoted as a generalizing model. Targets
without an accepted corrected continuous OD map are excluded rather than
silently trained against a raw or rejected stain result. These rules let the
same registry host a broad antibody panel without making unequal evidence look
equivalent.

Review labels distinguish measured `4 µm/px` OD from per-native-cell prediction.
On sections without target truth, accuracy is not claimed; the overlay is marked
as inferred, and the nearest observed target is explicitly labeled as a spatial
reference rather than ground truth. Per-cell uncertainty remains sealed in the
prediction artifact for quantitative QC.
## Repeated-assay OD harmonization

The multi-mouse study first subtracts each stain run's accepted adaptive
background floor under its tissue mask. Repeated target sections are then
mapped onto an equal-section reference distribution with one bounded,
zero-anchored monotonic transform per cohort and section. Section identifiers
are never used without their cohort because the same number recurs in different
mice. Every knot, support count, scale, upstream measurement fingerprint, and a
digest of the complete calibration is sealed into the table and result.

This is technical repeated-assay normalization, not recovery of an absolute OD
scale across biologically different specimens. It cannot create a local
expression pattern or change within-section cell ordering. Held-out Spearman
metrics therefore measure spatial/rank transfer on the harmonized assay scale;
they do not establish absolute cross-mouse concentration accuracy. The original
adaptive corrected maps remain the immutable measurement sources.

The earlier registered-neighborhood calibration remains available for
within-cohort repeated stains with adequate matched spatial support. It falls
back to identity when those anchors are insufficient.

Review tiles use one shared OD range for observed and predicted expression by
default. The optional per-section contrast mode has separately fingerprinted
tile URLs and is display-only; it must not be used for quantitative comparison.

Deep candidates are evaluated by leave-one-section-out prediction. Promotion
requires cell and 64 µm rank correlation, MAE, per-fold semantic-baseline
parity, and mean/median OD bias gates. A benchmark winner is not automatically
made live; the sealed result records the promotion decision so deployment can
remain atomic.
