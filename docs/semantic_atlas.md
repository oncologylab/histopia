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
preserved. Because each proposal adds one constant vector per section,
within-section Euclidean distances and nearest-neighbour identities are
invariant by construction. Section offsets are not removed before this guarded
correction, so the reported raw and corrected batch diagnostics remain
meaningful.

## Data Model

- Source WSI patches are sampled at 0.5 micrometres per pixel using 224 by 224
  non-overlapping patches by default.
- Only patches with sufficient coverage in the accepted registration tissue
  mask are encoded.
- Each patch stores one float16 UNI2-h vector plus source-grid, native-pixel,
  and registered-reference micrometre coordinates in a compressed NPZ file.
- Schema-3 feature artifacts carry a canonical SHA-256 seal over slide
  identity, extraction provenance, metadata, and every stored array.
- Model weights, source slides, compact features, and generated results remain
  outside the package repository.

## Workflow

Create a configuration based on `examples/semantic_atlas_config.toml`, then:

```bash
histopia-semantic cache-model --cache-dir /external/model/cache
histopia-semantic extract --config semantic-atlas.toml
histopia-semantic fit --config semantic-atlas.toml
```

Semantic configuration is validated before model or WSI loading. Physical
scales and topology distances must be finite and positive; patch, batch,
worker, PCA, sampling, seed, and cluster controls must be integers in their
documented ranges. Cluster counts are never silently rounded.

`cache-model` requires prior acceptance of the upstream gated model terms and
authenticated Hugging Face access. Subsequent extraction defaults to local-only
model loading. `histopia-semantic run` combines extraction and fitting.
Both `fit` and `run` reuse an existing atlas only after validating every result
artifact digest and matching content-sealed features, slide order, clustering,
PCA, sampling, topology, dependency versions, native thread count, and the
semantic algorithm revision. Historical unsealed results are readable but are
never eligible for fit reuse. Use `--overwrite-fit` when an intentional
recomputation is required.

Set `device = "auto"` to prefer CUDA, then Apple MPS, then CPU. Explicit
`"cuda"`, `"cuda:N"`, `"mps"`, and `"cpu"` values fail clearly when the
requested backend is unavailable. Use `histopia-semantic doctor` to inspect
the resolved device and accelerator memory before extraction. CUDA extraction
uses native bfloat16 autocast when the selected GPU supports it and float16
autocast otherwise, then recursively reduces a batch after an out-of-memory
error. CPU and Apple MPS extraction use float32. The resolved precision is
recorded in feature provenance. The device applies to UNI2-h inference, not
global atlas fitting. PCA, correspondence, guarded batch correction, K
optimization, and topology regularization use the CPU implementation so their
validated algorithms and numerical identity do not depend on an accelerator.

Validate the exact backend intended for a run, then optionally override only
the machine-level controls without editing the saved scientific configuration:

```bash
histopia-semantic doctor --device cuda:0
histopia-semantic extract --config semantic-atlas.toml --device cuda:0 \
  --batch-size 128 --patch-workers 4 --vips-threads 8
histopia-semantic fit --config semantic-atlas.toml --fit-threads 4
histopia-semantic fit --config semantic-atlas.toml --fit-threads 4 \
  --overwrite-fit
```

Device and batch-size overrides are included in feature provenance and cache
identity because they can change numerical inference. `patch_workers` and
`vips_threads` only control throughput and intentionally do not invalidate
scientifically equivalent completed features. Retain the effective command or
generated QuPath config when benchmarking those controls. Use `--device cpu`
for portable validation or when GPU memory is unavailable.

Set `patch_workers` above one to prefetch complete WSI batches concurrently
before they are consumed in order by the encoder. Result order and feature
fingerprints remain deterministic, and strip geometry is unchanged across
worker counts. Each worker can hold one decoded RGB batch and invoke native
libvips, so `1` is the portable default; benchmark `2` or `4` with the intended
storage and batch size.

For regular source grids, the built-in pyvips reader coalesces adjacent patches
into bounded row strips and adds one whole horizontal patch of source context
when available. The context keeps interpolation identical when a row is split
across different inference batches. Histopia prefetches the next strip batch
while the accelerator encodes the current batch, avoiding repeated WSI tile
decoding without loading a whole slide into memory. Reader and
extraction-method versions are part of feature provenance, so a changed
sampling implementation invalidates stale caches.

Feature provenance also records inference batch size, resolved device,
precision, accelerator identity, and relevant package versions. Switching
between CPU and GPU, changing batch size, or changing the numerical runtime
therefore creates a distinct cache identity.

Feature cache reuse additionally requires a valid schema-3 content seal.
Historical schema-1 and schema-2 files remain readable for fitting, but they
are intentionally treated as cache misses during extraction because they
cannot prove that stored feature or coordinate arrays are unchanged.
On a validated 18-section corpus with 17,482 patches, checking the seals added
about 40 milliseconds to a cached local load (0.318 versus 0.358 seconds).

The model fingerprint binds to the exact cached Hugging Face commit. Histopia
passes that commit explicitly to timm for both model configuration and weights,
including when authenticated model downloads are allowed.

The WSI provenance includes pyvips and native libvips versions. Result
provenance separately records the NumPy, SciPy, and scikit-learn versions used
for PCA, correspondence correction, K optimization, and regularization.

Set `vips_threads` to cap libvips' native process-wide worker pool separately
from `patch_workers`. The setting is applied before pyvips is imported and
therefore cannot be changed later in the same process. Leave it unset to use
libvips' adaptive default.

`fit_threads` independently caps native BLAS and OpenMP pools during PCA,
batch correction, global clustering, K selection, and topology
regularization. It defaults to four and is recorded in sealed result runtime
provenance. Keeping this value explicit avoids machine-dependent
oversubscription and makes fitting performance reproducible without changing
feature artifacts. The observational `semantic_performance.json` report records
`cache_hit`, the validation time, and why a candidate result was not reusable;
an exact hit reports zero atlas-fit and artifact-write time.

On a validated 23-section atlas with 76,499 patches, one, four, eight, 16, and
32 fit threads took 95.7, 71.7, 73.3, 73.6, and 106.2 seconds, respectively.
The four- and 32-thread fits selected the same K, produced exactly identical
labels for every K from 5 through 15, and retained every adjacent-section
topology pair exactly. Four threads reduced runtime by 33% relative to the
machine's unbounded 32-thread native pools.

With the same 23 sections and 76,499 patches stored as content-sealed
features, an isolated cold fit took 75.82 seconds and 2.30 GiB peak RSS. A
second process validated and reused the exact result in 1.88 seconds with
301 MiB peak RSS: 40.3 times faster with about 87% lower peak memory. Of the
1.51 seconds measured inside Histopia, feature loading used 1.44 seconds and
validation of the result plus every referenced artifact used 0.07 seconds.

A separate cold-process end-to-end refit, including artifact writing, fell
from 109.0 to 78.4 seconds after loading native estimator runtimes before
applying the cap. Average CPU use fell from 11.1 to 2.1 cores, while all
841,489 K-specific labels and all 59,919 topology links remained exact.

Because guarded batch correction applies one additive vector to every patch in
a section, its within-section KNN preservation is exactly one and does not
require rebuilding nearest-neighbour indexes. On a validated 21-section,
57,365-patch corpus, using that invariant reduced the batch-correction stage
from 5.63 to 1.88 seconds and the complete in-memory fit from 46.49 to 42.96
seconds. A separate 16-section, 80,307-patch diagnostic benchmark fell from
7.06 to 1.30 seconds. Both before/after atlas objects retained identical
selected K values and byte-exact aggregate digests over every field and array.

On the validated server, a representative 57,600 by 50,944 NDPI with 9,213
accepted patches took 9.98, 7.48, and 6.91 seconds with one, two, and four
patch workers. All feature and coordinate hashes were identical. Vectorized
mask-grid coverage reduced its 49,533-patch selection stage from 3.33 seconds
to 0.018 seconds with every fraction identical.

In a reader benchmark using a lightweight deterministic summary encoder, a
second real 30,720 by 29,440 H&E NDPI with 6,379 accepted patches and two patch
workers took 6.11-6.67, 3.91-4.30, 3.23-3.35, 3.46-3.52, and 4.05-4.09
seconds with native libvips caps of 1, 2, 4, 8, and 16 threads, respectively;
the adaptive default took 4.39-4.40 seconds. Every feature, coordinate, and
coverage hash was identical. Four threads was the stable optimum on that host,
but Histopia leaves the cap unset because storage, libvips, and host
concurrency determine the optimum. At four libvips threads, feature batch sizes
32, 64, and 128 took 3.71, 3.15, and 2.86 seconds with an identical feature
hash, confirming that strip interpolation no longer depends on batch
boundaries. These timings isolate WSI reading and preprocessing; the separate
UNI2-h measurements below include model inference.

Standard 224-pixel RGB patches are normalized as one tensor batch. On the
validated A100 runtime, this reduced 64-patch preprocessing from 62.4 to
14.9 milliseconds and complete preprocessing plus inference from 0.319 to
0.190 seconds. Normalized tensors and all 64 embeddings were bit-for-bit
identical to the per-image transform path, which remains the fallback for
nonstandard input dimensions.

The tested 40 GiB A100 runtime used 3.51 GiB at batch 64 and 6.34 GiB at batch
256. Batch 256 reached about 35 patches per second and is a useful starting
point on that class of GPU; keep 64 for portable configurations and benchmark
the target hardware. The real CPU path is supported but substantially slower:
the same model required about 5.8 GiB peak process memory and 2.7 seconds for a
single-patch validation inference.

A four-cohort validation campaign processed 80 approved serial sections and
729,452 accepted UNI2-h patches. Every adjacent section pair retained accepted
topology links, with cohort topology coverage from 82.4 to 91.0 percent. The
batch-correction guard accepted all four proposals and reduced between-section
centroid variance from 0.194-0.323 before correction to 0.0020-0.0053 after
correction. The largest cohort contained 24 sections and 524,317 patches; its
end-to-end extraction, atlas fit, and artifact generation took 38.7 minutes on
a 40 GiB A100 host. These are observational validation results, not portable
performance guarantees, and the semantic review records remained unapproved
until visual review.

CLI extraction reports each cached, started, and completed slide, including
patch count and elapsed time. Feature files are committed atomically, so an
interrupted campaign resumes only exact, provenance-valid completed slides.
Fitting reads only the deterministic artifact path for every slide in
`preflight.json`, in that recorded order. It validates slide identity,
source/mask/transform checksums, extraction scale, and common model provenance
before starting PCA or clustering; unrelated stale NPZ files are never admitted
to the atlas. New results record the ordered content seals for all schema-3
features; a mixed sealed/unsealed campaign is rejected.

Preflight also validates the complete native scanner geometry: positive native
and thumbnail dimensions, in-bounds content coordinates, and positive finite
MPP. Registration, native WSI export, and semantic patch extraction share this
raw, non-auto-oriented coordinate contract.

By default, independent five-seed fits are evaluated for K=5 through K=15.
Selection balances silhouette, seed stability, within-section coherence, and
accepted cross-section continuity, rejects tiny clusters, and prefers smaller
K when scores are effectively tied. Four-neighbour patch edges and accepted
adjacent-section correspondences provide conservative topology regularization.
Regularized labels are accepted only when adjacency does not worsen, at most 25
percent of labels change, and registered centroid distance does not worsen by
more than 10 percent.

Reciprocal candidate ranking uses bounded vector operations within each source
patch while retaining deterministic target tie-breaking and sparse memory. On
a 23-section atlas with 76,499 patches, one complete 22-pair correspondence
pass fell from 30.26 to 22.89 seconds. A profiled full fit fell from 135.15 to
117.23 seconds. The semantic JSON, review fingerprint, atlas model, all 22
topology artifacts, and all 189 stored arrays were exactly unchanged. When a
batch-correction proposal is rejected, Histopia also retains the original
topology graph because the accepted feature matrix is unchanged; accepted
corrections still trigger a complete correspondence rebuild.

Candidate rows with equal neighbourhood sizes are additionally ranked in
bounded matrix batches. Each source window contains at most 1,024 patches and
each scoring batch targets at most 8,192 candidate edges; one source
neighbourhood remains indivisible. This limits descriptor gathers while
preserving the original floating-point operation order and sequential target
tie-breaking. Each section's context descriptor is now computed once per atlas
pass, and one target spatial index is reused across the three coarse-to-fine
search radii. On a 16-section, 80,307-patch atlas, this preparation reuse
reduced a controlled fit from 64.825 to 62.546 seconds (3.64 percent), while
all 191 in-memory scientific arrays and metadata retained the same SHA-256
digest. Earlier bounded-ranking work reduced the same corpus's complete cold
fit from 82.13 to 66.01 seconds and the atlas phase from 76.47 to 60.94
seconds, with slightly lower peak memory. On a larger 24-section,
524,317-patch stress atlas, the complete measured fit fell from 566.23 to
553.24 seconds and the atlas phase from 540.99 to 529.50 seconds. The selected
K, result fingerprints, and all 194 and 290 scientific artifacts,
respectively, remained byte-for-byte identical.

## Review And Viewer

Every fit writes `semantic_result.json`, per-slide label grids,
`atlas_model.npz`, and `semantic_review.json`. Model, label, topology, result,
and review files use atomic per-file replacement, so cancellation cannot expose
a partially written artifact at its final path. A new or changed result is
unapproved and fingerprinted. An exact deterministic rerun retains an existing
approval only after the regenerated result and every sealed artifact validate
against the same fingerprint. The fingerprint binds the model, every label
grid, every topology artifact, and the exact preflight slide order; stale or
incomplete artifacts are rejected before QC or viewer generation. Scientific
interpretation should wait until semantic overlays and sensitivity fits have
been reviewed.

`semantic_performance.json` is a separate atomic observational record. During
extraction it reports running/completed/failed state, effective device and
worker controls, per-slide cache status, patch counts, elapsed time, and patch
throughput. Fitting adds feature-load, native-runtime preparation, atlas-fit,
and artifact-write durations plus the result fingerprint. A fresh extraction
clears stale fit timing. This file is deliberately excluded from semantic
result artifacts, fingerprints, and approval because elapsed time is
machine-dependent and has no scientific meaning.

Approve the exact reviewed fingerprint with:

```bash
histopia-semantic approve \
  --run /path/to/semantic-run \
  --reviewer "Reviewer name" \
  --review-notes "Reviewed semantic, blend, K sensitivity, and topology views."
```

Approval revalidates every sealed result artifact before atomically updating
`semantic_review.json`. QuPath export rejects missing, unapproved, or stale
review records.

Preflight also requires every registration mask to be accepted and backed by
an approved mask-review record. Its portable slide provenance records the
effective processed-mask checksum, mask method, and review status. This binds
semantic patch selection to the cleaned mask actually used for registration,
including reviewed overrides.

New schema-3 preflights additionally require the final
`registration_approval.json` seal and include its exact SHA-256 in the
preflight fingerprint. Semantic extraction therefore cannot begin from a
registration whose masks and order were reviewed but whose completed alignment
was never finally approved. Historical schema-1 and schema-2 preflights remain
readable for fitting and approved-result export.

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
  --cohort-qc /path/to/cohort-qc.json \
  --workers 4
```

The canonical `histopia.visualization` viewer exposes Histology, Blend, and
Semantic modes, selectable K, quantitative batch and K diagnostics, and one
selected adjacent-pair topology overlay. Cohort builds also expose compact QC
flags and exact-fingerprint review status. The viewer loads only the active
texture set, disposes replaced GPU textures, and displays at most the 500
highest-confidence links while preserving complete correspondences in result
artifacts. Browser checks are available through the `browser-test` optional
dependency and verify desktop layout, WebGL output, assets, and rapid sample
switching. Texture changes and sample loads keep the viewport busy until a
rendered frame is available. If the browser loses its WebGL context, the viewer
restores the light canvas background, repaints the current stack, and clears
the busy state only after rendering resumes.

Viewer builds checksum their generated WEBP assets and reuse exact matches.
`build-report.json` records elapsed time and encoded/reused asset counts for
each build. A changed image, transform, mask, label grid, palette, or encoder
setting produces different rendered pixels and replaces only the affected
asset. Viewer rasterization and WEBP encoding are CPU-backed. `--workers`
bounds a persistent encoder pool and defaults to one, while a deterministic
producer queue retains at most twice that many pending images. Worker counts
do not change rendered bytes or cache ordering. `build-report.json` records
`compute_backend = "cpu"` and `peak_pending_assets` so execution and memory
controls are explicit.

On a clean 134-section, five-sample build containing 1,742 assets, this bounded
pipeline plus vectorized semantic patch rasterization reduced four-worker wall
time from 227.48 to 177.79 seconds (21.8%). All 1,756 generated non-report
files remained byte-identical; an unchanged warm build continued to reuse
every asset in 0.67 seconds.
