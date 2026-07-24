# Global Serial-Section Semantic Atlas

Histopia builds one unsupervised morphology atlas across an accepted registered
section stack. It does not independently cluster each slide and then attempt to
rename the clusters. Every source slide contributes to one normalized PCA and
MiniBatchKMeans space, which gives region labels a single global meaning.

Histopia L2-normalizes each patch embedding and fits one section-balanced PCA
to bootstrap deformation-aware correspondences between adjacent sections.
Those reciprocal links estimate a smooth local displacement field without
warping accepted image pixels. A confidence-weighted additive batch correction
is proposed from the links, but is accepted only when anchor distance and
slide-attributable variance improve while within-slide neighbourhoods are
preserved. Section offsets are not removed before this guarded correction, so
the reported raw and corrected batch diagnostics remain meaningful.

## Data Model

- Source WSI patches are sampled at 0.5 micrometres per pixel using 224 by 224
  non-overlapping patches by default.
- Only patches with sufficient coverage in the accepted registration tissue
  mask are encoded.
- Each patch stores one float16 UNI2-h vector plus source-grid, native-pixel,
  and registered-reference micrometre coordinates in a compressed NPZ file.
- Model weights, source slides, compact features, and generated results remain
  outside the package repository.

## Workflow

Create a configuration based on `examples/semantic_atlas_config.toml`, then:

```bash
histopia-semantic cache-model --cache-dir /external/model/cache
histopia-semantic extract --config semantic-atlas.toml
histopia-semantic fit --config semantic-atlas.toml
```

`cache-model` requires prior acceptance of the upstream gated model terms and
authenticated Hugging Face access. Subsequent extraction defaults to local-only
model loading. `histopia-semantic run` combines extraction and fitting.

Set `device = "auto"` to prefer CUDA, then Apple MPS, then CPU. Explicit
`"cuda"`, `"cuda:N"`, `"mps"`, and `"cpu"` values fail clearly when the
requested backend is unavailable. Use `histopia-semantic doctor` to inspect
the resolved device and accelerator memory before extraction. CUDA extraction
uses bfloat16 autocast and recursively reduces a batch after an out-of-memory
error; CPU execution remains available for portability and small validation
runs.

By default, independent five-seed fits are evaluated for K=5 through K=15.
Selection balances silhouette, seed stability, within-section coherence, and
accepted cross-section continuity, rejects tiny clusters, and prefers smaller
K when scores are effectively tied. Four-neighbour patch edges and accepted
adjacent-section correspondences provide conservative topology regularization.
Regularized labels are accepted only when adjacency does not worsen, at most 25
percent of labels change, and registered centroid distance does not worsen by
more than 10 percent.

## Review And Viewer

Every fit writes `semantic_result.json`, per-slide label grids,
`atlas_model.npz`, and `semantic_review.json`. A new result is unapproved and
fingerprinted. The fingerprint binds the model, every label grid, every
topology artifact, and the exact preflight slide order; stale or incomplete
artifacts are rejected before QC or viewer generation. Scientific
interpretation should wait until semantic overlays and sensitivity fits have
been reviewed.

Add an atlas to the section viewer with:

```bash
histopia-register \
  --viewer-run sample=/path/to/registration-run \
  --viewer-semantic-run sample=/path/to/semantic-run \
  --viewer-output-dir /path/to/viewer
```

For a multi-sample review, write one portable cohort report and pass it to the
stable viewer build:

```bash
histopia-semantic cohort-qc \
  --run sample-a=/path/to/semantic-a \
  --run sample-b=/path/to/semantic-b \
  --output /path/to/cohort-qc.json

histopia-visualize build /path/to/viewer-root \
  --run sample-a=/path/to/registration-a \
  --run sample-b=/path/to/registration-b \
  --semantic-run sample-a=/path/to/semantic-a \
  --semantic-run sample-b=/path/to/semantic-b \
  --cohort-qc /path/to/cohort-qc.json
```

The canonical `histopia.visualization` viewer exposes Histology, Blend, and
Semantic modes, selectable K, quantitative batch and K diagnostics, and one
selected adjacent-pair topology overlay. Cohort builds also expose compact QC
flags and exact-fingerprint review status. The viewer loads only the active
texture set, disposes replaced GPU textures, and displays at most the 500
highest-confidence links while preserving complete correspondences in result
artifacts. Browser checks are available through the `browser-test` optional
dependency and verify desktop layout, WebGL output, assets, and rapid sample
switching.

Viewer builds checksum their generated WEBP assets and reuse exact matches.
`build-report.json` records elapsed time and encoded/reused asset counts for
each build. A changed image, transform, mask, label grid, palette, or encoder
setting produces different rendered pixels and replaces only the affected
asset.
