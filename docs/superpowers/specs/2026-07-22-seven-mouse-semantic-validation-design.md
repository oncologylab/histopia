# Seven-Mouse Semantic Validation Design

## Scope

Phase one validates Histopia's semantic-atlas workflow on the seven KPF mice
that already have current Histopia registration results: 3528, 4257, 4312,
4577, 4630, 5996, and 5997. The campaign includes straightforward cases,
large morphology changes, difficult staining or registration cases, and all
earlier-tested mice. Registration of the remaining locally staged mice is a
separate phase after these seven results are approved.

Every mouse is processed from its current Histopia registration. Legacy
UNI2-H embeddings are excluded even when present because their masks,
orientation, order, scale, or registration provenance may differ.

## Input Preflight

Preflight creates a machine-readable manifest for each mouse before feature
extraction. It verifies:

- one registration result and review state;
- a complete, unique slide inventory;
- readable registered images and tissue masks;
- accepted section order and recorded quarter-turn orientation;
- finite transforms and positive physical pixel scale;
- matching image, mask, registration, and order fingerprints;
- no material blank-space or debris leakage in semantic patch support.

A failed preflight blocks only that mouse. The failure report identifies the
slide and underlying mask, registration, or metadata defect. Remediation must
change the source artifact and its fingerprint; the semantic stage must not
carry ad hoc per-slide exclusions or inherited overrides.

## Uniform UNI2-H Extraction

All seven mice use one local, fingerprinted UNI2-H checkpoint and a shared
extraction configuration. Features are extracted afresh from the current
registered images at a low-resolution analysis scale using the same patch
size, tissue-fraction threshold, color normalization, precision, and package
constraints. Extraction is resumable at slide boundaries. A cached slide is
reused only when all input and configuration fingerprints match exactly.

Each feature artifact records slide identity, registration fingerprint,
checkpoint fingerprint, analysis microns per pixel, patch geometry, tissue
coverage, feature dimensions, software versions, and completion state.
Partial files are written atomically and are never treated as valid caches.

## Per-Mouse Global Atlas

Each mouse is fitted independently as one global serial-section atlas. It does
not propagate labels through a pairwise clustering chain. Independent global
clusterings are fitted for every K from 5 through 15, and the built-in
topology-aware score selects a recommended K while retaining every K for
review.

Adjacent sections are linked with deformation-aware correspondences using
registered physical coordinates, feature similarity, local displacement-field
consistency, reciprocal matching, and neighborhood support. Complete accepted
links remain in external result artifacts. The browser displays at most the
500 highest-confidence links for the selected adjacent pair.

Slide batch correction is a guarded proposal. It is accepted only when it
improves anchor agreement and slide-effect diagnostics without violating
within-slide neighborhood preservation. Raw, legacy mean-centering, proposed,
and accepted diagnostics are retained. Unsupported sections remain unchanged.

## Quantitative Validation

Per-slide QC includes tissue area and coverage, accepted patch count, blank or
debris risk, feature norm summaries, registration overlap, and topology-link
support. Per-mouse QC includes slide completeness, K metrics, selected K,
cluster-size balance, seed stability, within-section coherence, cross-section
continuity, topology confidence and coverage, unsupported sections, and all
batch-correction diagnostics.

A cohort summary compares the seven mice under the same definitions and flags
outliers relative to both fixed scientific thresholds and cohort distributions.
Automatic flags do not constitute rejection by themselves; they direct visual
review. A result remains unapproved until the user reviews its semantic layers,
K alternatives, batch metrics, and topology links.

## Viewer And Review

The package-owned visualization module builds one stable `/histopia/` viewer
with a mouse selector. Changing mouse, K, section pair, semantic visibility,
or topology visibility does not require a page reload. Refresh restores each
mouse's accepted section order. Generated images, feature arrays, topology
arrays, checkpoints, and review outputs remain outside Git.

Each mouse has a fingerprinted semantic review record initialized with
`approved: false`. Approval applies only to the exact registration, extraction,
configuration, and semantic-result fingerprints. Any rerun invalidates prior
approval. The cohort is complete only when all seven current fingerprints are
approved.

## Execution And Failure Handling

Execution proceeds in gates: preflight all mice, extract all valid mice,
fit each complete mouse, build cohort QC, deploy the viewer, and request review.
Independent mice may run concurrently when GPU memory and I/O permit, but one
GPU extraction process owns the checkpoint at a time. A failure is recorded
per mouse and does not corrupt or erase completed outputs from other mice.

The pipeline is restartable and never modifies raw WSI data. Source data,
checkpoints, local paths, credentials, generated artifacts, `AGENTS.md`, and
`PLANS.md` remain untracked. Only typed reusable code, deterministic synthetic
tests, public documentation, and path-neutral examples may be pushed.

## Acceptance Criteria

- All seven preflight manifests pass against current registration artifacts.
- Every registered slide has a fresh, fingerprint-valid UNI2-H feature file.
- Every mouse contains K=5 through K=15 results and a recommended K.
- Batch correction and topology diagnostics are complete and auditable.
- Cohort QC contains all seven mice under identical metric definitions.
- The stable public viewer exposes all seven mice without viewport overflow or
  browser errors at 1920x1080 and 3840x2160.
- Package tests, Ruff checks, and reproducibility checks pass.
- The user approves each of the seven exact semantic-result fingerprints.
- Sanitized code and documentation are pushed without local operational or
  scientific data artifacts.
