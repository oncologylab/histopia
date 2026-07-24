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

Validate the exact backend intended for a run, then optionally override only
the machine-level controls without editing the saved scientific configuration:

```bash
histopia-semantic doctor --device cuda:0
histopia-semantic extract --config semantic-atlas.toml --device cuda:0 \
  --batch-size 128 --patch-workers 4 --vips-threads 8
```

The effective overrides are included in feature provenance and cache identity.
Use `--device cpu` for portable validation or when GPU memory is unavailable.

Set `patch_workers` above one to prefetch complete WSI batches concurrently
before they are consumed in order by the encoder. Result order and feature
fingerprints remain deterministic, and strip geometry is unchanged across
worker counts. Each worker can hold one decoded RGB batch and invoke native
libvips, so `1` is the portable default; benchmark `2` or `4` with the intended
storage and batch size.

For regular source grids, the built-in pyvips reader coalesces adjacent patches
into bounded row strips and prefetches one batch while the accelerator encodes
the current batch. This avoids repeated WSI tile decoding without loading a
whole slide into memory. Reader and extraction-method versions are part of
feature provenance, so a changed sampling implementation invalidates stale
caches.

Feature provenance also records inference batch size, resolved device,
precision, accelerator identity, and relevant package versions. Switching
between CPU and GPU, changing batch size, or changing the numerical runtime
therefore creates a distinct cache identity.

The WSI provenance includes pyvips and native libvips versions. Result
provenance separately records the NumPy, SciPy, and scikit-learn versions used
for PCA, correspondence correction, K optimization, and regularization.

Set `vips_threads` to cap libvips' native process-wide worker pool separately
from `patch_workers`. The setting is applied before pyvips is imported and
therefore cannot be changed later in the same process. Leave it unset to use
libvips' adaptive default.

On the validated server, a representative 57,600 by 50,944 NDPI with 9,213
accepted patches took 9.98, 7.48, and 6.91 seconds with one, two, and four
patch workers. All feature and coordinate hashes were identical. Vectorized
mask-grid coverage reduced its 49,533-patch selection stage from 3.33 seconds
to 0.018 seconds with every fraction identical.

The tested 40 GiB A100 runtime used 3.51 GiB at batch 64 and 6.34 GiB at batch
256. Batch 256 reached about 35 patches per second and is a useful starting
point on that class of GPU; keep 64 for portable configurations and benchmark
the target hardware. The real CPU path is supported but substantially slower:
the same model required about 5.8 GiB peak process memory and 2.7 seconds for a
single-patch validation inference.

CLI extraction reports each cached, started, and completed slide, including
patch count and elapsed time. Feature files are committed atomically, so an
interrupted campaign resumes only exact, provenance-valid completed slides.

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

Preflight also requires every registration mask to be accepted and backed by
an approved mask-review record. Its portable slide provenance records the
effective processed-mask checksum, mask method, and review status. This binds
semantic patch selection to the cleaned mask actually used for registration,
including reviewed overrides.

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
