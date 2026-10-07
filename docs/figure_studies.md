# Evidence-bound figure studies

`histopia.study` provides a separate study identity for workflow-plus-results
figures. It does not modify a frozen protein study, promote predictions, approve
cell labels, or convert embedding clusters into validated biological niches.
The package import and manifest validation use no image, GPU or numerical
dependencies. Install `histopia[study]` for analysis, SVG/PDF/PNG figures and
CSV/Parquet exports; WSI readers and Astir remain separate optional dependencies.

## Freeze inputs and mouse roles

Use `write_study_manifest(path, payload)` to seal a schema-1 JSON manifest.
Required top-level fields are `study_id`, `schema_version`, `mice` and `slides`.
Each mouse has a unique `mouse_id` and one role: `development`, `test`,
`protected` or `unassigned`. Each slide identifies its mouse, organ and original
scan. Converted images also record `is_derivative` and `derivative_of`.
`section_order` must come from reviewed ordering evidence, not scan counters.

An untouched `test` role requires `previously_exposed: false` and explicit
`exposure_evidence`. Every organ from a mouse must have the same role. Keep
previously protected mice in their existing study and evaluation order. An
unassigned exposure history does not become an untouched test subject.
`validate_fit_scope` guards model fitting, calibration, positivity thresholds,
and learned preprocessing. New-method evaluation rejects protected/unassigned
mice and overlapping fitting/evaluation mice. Held-out development-mouse
evaluation is possible, but must not be described as an untouched external test.

Z spacing is either unknown (`null`), assumed, or physical. Physical spacing
requires `spacing_evidence`; assumptions must be explicit in the manifest and
captions. A displayed stack with exaggerated assumed depth is not a measured
3D reconstruction.

Record previous frozen specifications and decisions in `frozen_studies` with
their original IDs and SHA-256 hashes. Rewriting an existing study manifest with
different contents fails; create another version. Analysis artifacts bind the
frozen manifest rather than updating it with each new result.

```bash
python -m histopia.study validate study-manifest.json
python -m histopia.study figure study-manifest.json figure-spec.json figure-output
```

`attach_study_figure(figure_dir, review_dir)` adds a versioned tab to an existing
workflow website after verifying export hashes and region/study bindings.
`build_workflow_review(..., study_figure_dir=...)` includes it during a rebuild.
The copied files contain only figure and region-view exports. The original
workflow tabs and scientific approval states are retained.

## H&E feature identity and legacy baselines

`prepare_he_rgb` implements `he-native-rgb-v1`: retain the original uint8 H&E
RGB values, followed by the encoder's own checkpoint transform. It fits no
color normalization. It is distinct from the legacy fixed H–DAB basis,
hematoxylin extraction, patchwise percentile scaling and purple/white rendering.
A legacy H&E archive with H–DAB preprocessing must retain that actual identity.

`feature_identity` requires source, original scan, ordered cell IDs, mask,
registration, model, modality, preprocessing, normalization, scale, crop size
and software bindings. Every scientific change invalidates reuse. A finite,
repeatable bounded feature extraction is a technical check; it does not qualify
whole-slide extraction or establish protein-prediction accuracy. Compare
compatible legacy models before fitting new models to a changed feature domain.

## Protein vectors and evaluation

`histopia.protein.match_registered_cells` accepts section-qualified cell IDs,
registered XYZ positions in micrometres, morphology and boolean tissue support.
It has no target-protein input. Combined and spatial controls search XY;
the morphology control searches source-standardized embeddings. All retain
the same hard physical radius and supplied tissue-support constraints.
The morphology control is therefore a morphology ranking control within
eligible registered support, not unrestricted cross-tissue matching.

`transfer_protein_vectors` reuses that graph for all protein columns. It rejects
changed cell ordering, retains per-protein support counts and leaves unsupported
entries as NaN. Missing source markers do not change matching. Source/target IDs,
source indices, weights, physical distances and correspondence scores remain
available. Serial-section correspondence never establishes same-cell identity.

Transfer uncertainty combines weighted between-anchor variance and source
variance; unknown source uncertainty stays unknown. Correspondence confidence
is a geometric/morphological score, not a calibrated identity probability.

`evaluate_protein_vectors` keeps `direct_he_prediction` and
`he_prediction_then_section_transfer` separate from spatial/morphology controls.
It reports supported cell error, equal-weight physical-bin error, spatial error
rows and coverage for each mouse/protein. Its default bin width is 64 µm.
It records the original measurement resolution and does not resample OD maps.
These diagnostics supplement the existing protein promotion guardrails and
native visual QC; they cannot approve a result.

## Tissue-region identity and summaries

`connected_tissue_regions` splits each approved semantic class into independent
four-connected components. Semantic class labels are zero-based; `-1` means
background/unsupported tissue. Output region indices start at 1 with background
0. Disconnected areas of the same class have separate immutable IDs. IDs include
the semantic result, approval, section, coordinate frame and component pixels.

`sample_region_labels` maps native XY points through a validated homogeneous
transform into categorical grid pixels. Out-of-grid samples remain unsupported.
`assign_cells_by_overlap` uses native instance perimeter pixels, including
unsupported pixels in each denominator. A strict majority is required; ties
and low-coverage cases are ambiguous. For whole slides,
`assign_cells_from_overlap` accepts sparse per-cell perimeter counts from
streamed stripes. Use a one-pixel halo and count each core pixel exactly once.
Do not substitute centroid assignment or averaged TIFF pyramid labels.

`region_expression_summary` exports long-form region-by-protein rows: cell
count, supported count/coverage, mean, median, population SD and IQR. Positivity
is omitted unless a validated protein-specific threshold and its fingerprint
are supplied. Ambiguous assignments are excluded, and their number must be
reported with the summary scope. Measured, directly predicted and transferred
summaries use separate calls and remain separate in the viewer. Statistical
support does not increase the source measurement's spatial resolution.

`histopia.visualization.build_region_view` creates an offline-capable SVG
explorer with boundary-only rendering, one-region selection, previous/next
navigation, optional dimmed tissue, separate evidence menus and SVG export.
The selected immutable ID persists in `?region=...`. The outline, annotation,
table and export metadata use that same ID. All assets are relative so port
proxies work; a static outline and JSON fallback remain available without
JavaScript. The caller must align the optional background to the exact raster.

## Broad cell annotation and interpolation

`CellLabelProbabilities` uses epithelial, stromal, immune, vascular and uncertain
classes by default. Artifact and necrosis are QC categories, separate from
biological classes. Unsupported probability rows remain NaN; supported rows
sum to one. Model outputs cannot be marked as reference labels.

`marker_prior_coverage` reports actual available classes for a curated marker
panel. `fit_astir_comparator` optionally calls
[Astir](https://github.com/camlab-bioml/astir), using development mice and at
least two fully supported class priors. Its output remains a prediction. A
single measured marker per serial section is not a simultaneous measured
multi-marker panel. Transferred profiles must retain their exploratory status.
Astir is imported only when requested; it is not installed with the study extra.

`annotation_review_sample` balances model-assisted development suggestions
across mice, targeting 200 cells per supported class and at least four mice.
It reports shortages. Blinded evaluation selection is uniform over eligible
test cells, never stratified by model suggestions, and omits suggestions.
All reference-label fields start blank. `evaluate_cell_labels` requires a bound,
independent, blinded lab review and reports accuracy, Brier score, abstention
and per-class support. Morphology/protein/combined outputs must be evaluated
against the same independent reference subset.

`interpolate_label_probabilities` uses only a supported correspondence graph.
It retains complete probability distributions and missing regions, and marks
outputs as interpolated. It does not synthesize an observed cell at an
intervening position or claim that inferred tissue was sampled.

## Neighborhoods and external references

`physical_neighborhoods` excludes the center cell and crosses neither mouse
nor tissue support. Within-section analysis uses XY. Adjacent-section analysis
uses XYZ, explicit spacing and supported registered tissue components; local
semantic region IDs alone cannot identify cross-section tissue components.
Run 32, 64 and 128 µm, with 64 µm primary. Feature-specific counts preserve
missing markers and class probabilities.

`within_tissue_permutation` permutes whole profiles and their support masks
within mouse/section/tissue strata, retaining cross-marker covariance.
`niche_stability` compares clustering assignments on shared cell IDs using ARI.
These are analysis primitives, not an automatic biological validation or an
automatic choice of niche number. Keep embedding-defined regions distinct
from cell-neighborhood clustering; validate with independent biology rather
than the same features that defined the clusters.

`hpa_ihc_inventory` reads antibody-specific normal-human IHC records from the
[HPA single-entry XML format](https://www.proteinatlas.org/about/download).
It preserves gene, antibody, organ, specimen, image and metadata hashes.
Species and phospho/total equivalence require curated mappings. HPA explicitly
states that [its IHC specimens do not correspond to TCGA specimens](https://www.proteinatlas.org/ENSG00000173110-HSPA6/cancer).
Use TCGA as a separate H&E generalization cohort.

## Figure and operational outputs

The figure specification defines panels A–F with a status, caption, optional
assets and a visible reason for pending/rejected results. Each raster asset
needs its SHA-256, evidence ID, label and optional physical width for a scale
bar. Approved panels can use only approved, scoped evidence. SVG/PDF retain
editable labels and vector layout; micrographs remain embedded rasters. The
builder writes presentation PNG, captions, path-free provenance and a matching
web view. Keep generated figures and scientific artifacts outside the repository.

`write_analysis_arrays` exports compact NPZ with a matching hash-bound JSON
manifest. Object/pickle arrays are rejected. `write_summary_tables` writes CSV
and optional Parquet. Include cell/region IDs, units, support and upstream
fingerprints in all downstream exports.

`AssignmentLedger` uses exclusive shared-storage files for artifact ownership,
worker claims and terminal evidence. Claims record host, boot ID, process ID
and process start ticks. It never expires a claim by age. A worker must own
the claim to finish it, and previous attempts remain available for audit.
Keep this ledger separate from existing scientific repair/evaluation queues.
