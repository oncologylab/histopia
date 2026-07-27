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
  and registered-reference micrometre coordinates in a portable NPZ file.
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
error. Standard CUDA inputs are transferred as compact uint8 tensors before
float32 conversion and normalization on the selected GPU. CPU and Apple MPS
extraction retain host float32 preprocessing. The resolved precision and input
pipeline are recorded in feature provenance. The device applies to UNI2-h
inference, not global atlas fitting. PCA, correspondence, guarded batch
correction, K optimization, and topology regularization use the CPU
implementation so their validated algorithms and numerical identity do not
depend on an accelerator.

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

New schema-3 feature files use deterministic ZIP-stored NPZ members. Existing
deflate-compressed NPZ artifacts remain fully readable and cache-valid. On a
representative 20,994-patch artifact, ZIP storage reduced serialization from
2.44 to 0.06-0.08 seconds and validation loading from 0.37 to 0.09 seconds.
The file grew from 59.99 to 65.43 MB (9.1 percent); deflate level 1 retained
nearly the same size but still required about 2.16 seconds to write.

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

`fit_threads` independently bounds CPU work during global fitting. PCA, batch
correction, global clustering, and K selection cap their native BLAS and OpenMP
pools at this value. Adjacent-section correspondence uses one pair worker when
the budget is one and at most two pair workers otherwise; each matcher receives
one native math thread to avoid nested oversubscription. Context descriptors
are prepared in consecutive windows and retained for at most the active pair
workers plus their shared boundary section. Each section descriptor is still
computed exactly once, but a large cohort no longer materializes descriptors
for every section simultaneously. Independent topology regularization jobs use
at most `fit_threads` workers and are additionally capped at eight to bound
simultaneous probability matrices. It defaults to four and is recorded in
sealed result runtime provenance. Keeping this value explicit avoids
machine-dependent oversubscription and makes fitting performance reproducible
without changing feature artifacts.

The observational `semantic_performance.json` report records `cache_hit`, the
validation time, and why a candidate result was not reusable; an exact hit
reports zero atlas-fit and artifact-write time. A computed fit also records
`correspondence_workers`, `correspondence_descriptor_window_sections`,
`regularization_workers`, and an `atlas_fit_phase_seconds` object covering
feature preparation, PCA fit and projection, initial and corrected
correspondence and graph construction, guarded batch correction, K selection,
and label regularization.

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

Phase-level profiling on the same 23-section, 76,499-patch atlas showed that
the initial and corrected correspondence passes used 40.41 of 62.75 atlas-fit
seconds. Running independent adjacent pairs with the bounded two-worker policy
reduced correspondence to 29.31 seconds and the complete measured workflow
from 68.34 to 52.10 seconds. The semantic fingerprint, model, result and review
records, and all 22 topology artifacts were byte-identical. On this host, four
and eight pair workers were slower than two, so Histopia deliberately caps
pair concurrency instead of equating it with the full native thread limit.
On a larger 15-section atlas with 333,739 patches, the same policy reduced
atlas fitting from 242.35 to 181.84 seconds and the complete workflow from
257.84 to 196.88 seconds. All 182 non-observational output files retained the
same aggregate digest and semantic fingerprint.

A controlled replay of that 333,739-patch atlas compared all-section
descriptor retention with the bounded three-section window used by two pair
workers. Peak RSS fell from 7,769,896 KiB to 6,656,900 KiB (14.3 percent).
Wall time remained effectively flat at 173.29 versus 174.17 seconds, and all
198 non-observational files were byte-identical. The smaller 76,499-patch
atlas remained at 2.30 GiB peak RSS because later fitting stages, rather than
descriptor retention, set its process peak.

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

For standard CUDA batches, transferring the uint8 tensor before conversion and
normalization reduces host-to-device input traffic by four times. On a
15-section, 333,739-patch A100 campaign at batch 256, this reduced extraction
from 1,007.63 to 887.72 seconds (11.9 percent). Sustained sections increased
from roughly 331-342 to 381-383 patches per second. All stored feature,
coordinate, and tissue-fraction arrays were exact across all 15 sections, and a
downstream refit produced the same selected K, diagnostics, and 180
byte-identical model, label, and topology artifacts. GPU sampling reached
91-100 percent SM utilization through most inference intervals and used about
8.5 GiB of device memory.

The tested 40 GiB A100 runtime also used 3.51 GiB at batch 64 in an isolated
inference benchmark. Batch 256 is a useful starting point on that class of GPU;
keep 64 for portable configurations and benchmark the target hardware. The
real CPU path is supported but substantially slower: a current single-patch
validation used about 5.8 GiB peak process memory and 0.94 seconds for warm
inference, excluding model loading.

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

Winner and runner-up scores are extracted once per equal-neighbourhood matrix
batch rather than sorting every source row independently. Target updates remain
sequential, including deterministic lowest-index tie-breaking. On the
23-section, 76,499-patch atlas, three controlled correspondence passes improved
from a 14.368-second median to 13.766 seconds (4.19 percent), with the same
60,265 accepted links and exact aggregate array digest. Its complete measured
workflow fell from 52.10 to 49.63 seconds. On the 15-section,
333,739-patch stress atlas, both correspondence stages fell from 125.90 to
119.50 seconds and the complete workflow from 196.88 to 189.82 seconds, while
selected K, peak memory, and all 182 non-observational artifacts remained
unchanged.

Each requested K is regularized independently after global cluster selection.
Histopia executes these jobs with a bounded pool controlled by `fit_threads`;
result collection remains in requested-K order. On the 15-section,
333,739-patch stress atlas, controlled artifact replay reduced the median
regularization stage from 22.50 seconds with one worker to 12.82 seconds with
two, 7.51 seconds with four, and 5.84 seconds with eight. All labels and
acceptance guards were exact at every setting. A complete four-worker refit
reduced the phase from 21.32 to 7.27 seconds and retained all 182
non-observational files byte for byte at 7,709,400 KiB peak RSS. On the
23-section, 76,499-patch atlas, a paired complete refit fell from 56.02 to
52.23 seconds; regularization fell from 4.49 to 1.43 seconds, peak RSS differed
by only 260 KiB, and all 278 non-observational files were byte-identical.

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
throughput. Fitting adds feature-load, native-runtime preparation, phase-level
atlas-fit, and artifact-write durations, effective correspondence workers, and
the result fingerprint. A fresh extraction clears stale fit timing. This file
is deliberately excluded from semantic result artifacts, fingerprints, and
approval because elapsed time is machine-dependent and has no scientific
meaning.

Approve the exact reviewed fingerprint with:

```bash
histopia-semantic approve \
  --run /path/to/semantic-run \
  --registration-run /path/to/registration-run \
  --reviewer "Reviewer name" \
  --review-notes "Reviewed semantic, blend, K sensitivity, and topology views."
```

Approval revalidates every sealed result artifact and verifies that the atlas
belongs to the exact current registration result and registration approval
before atomically updating `semantic_review.json`. QuPath export rejects
missing, unapproved, stale, or registration-mismatched review records.

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

Viewer and QuPath export use the same registration-binding validator. It
requires the exact registration-result bytes, section order, reference, and
semantic preflight fingerprint recorded during extraction; schema-3 runs also
require the exact final registration approval. A later transform edit is
therefore rejected even when all slide filenames still match. The viewer
records the validated binding in its manifest and rechecks it after asset
generation before publishing a new manifest.

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

Validate the exact registration seals, semantic bindings, mutable review
records, and generated viewer manifest as one batch:

```bash
histopia-visualize audit \
  --run sample-a=/path/to/registration-a \
  --run sample-b=/path/to/registration-b \
  --semantic-run sample-a=/path/to/semantic-a \
  --semantic-run sample-b=/path/to/semantic-b \
  --viewer-manifest /path/to/viewer-root/histopia/manifest.json \
  --output /path/to/workflow-audit.json
```

`audit` emits the same portable JSON to standard output and, when requested,
atomically writes it to `--output`. It never includes source or run-directory
paths. Exit code `0` means every requested scientific stage is approved and
the viewer is current. Exit code `2` means all inspected artifacts are valid
but at least one human review or current registration-approval binding remains
required. Exit code `1` means a requested artifact is missing, malformed,
stale, mismatched, or otherwise unverifiable.

The audit treats semantic preflight schemas 1 and 2 as `legacy_unsealed`, even
when their semantic review record is approved. A production `approved` result
requires schema 3, which binds the semantic features to the exact final
registration approval. Viewer entries must match the current registration
result digest, order fingerprint, semantic fingerprint, registration binding,
review state, and section count.

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
asset. Mouse-level cache identity includes the current processed thumbnail and
mask bytes, not only their review metadata, so post-run file replacement
cannot reuse stale rendered textures. Every semantic atlas and registration
binding is fully validated before rendering and again before publication, so
an artifact changed during generation fails the build. Each binding validation
is the authoritative integrity pass; the builder does not redundantly hash the
same atlas immediately before it. Viewer rasterization and WEBP encoding are
CPU-backed. `--workers`
bounds a persistent encoder pool and defaults to one, while a deterministic
producer queue retains at most twice that many pending images. Worker counts
do not change rendered bytes or cache ordering. `build-report.json` records
`compute_backend = "cpu"`, `peak_pending_assets`, the mouse-cache schema, and
the lossless and lossy WEBP effort levels so execution, memory, and encoder
controls are explicit. Registered and blended review textures use effort 5 at
their existing quality settings. Semantic label textures remain lossless at
effort 6. The mouse-cache schema is bumped whenever this encoding or
source-identity contract changes, preventing stale mouse-level reuse.

For one section, every K layer in a sealed semantic result shares the same
patch grid. The builder therefore resolves patch-to-display-pixel ownership
once and reuses that raster for the remaining K layers.
`semantic_rasters_built` and `semantic_rasters_reused` make this work visible
in the build report.

On a complete 16-cohort build with 401 sections, 11 K layers per section, and
5,213 WEBPs, four workers reduced cold build time from 196.28 to 178.86
seconds. The builder created 401 patch rasters and reused them 4,010 times.
All 5,213 image assets, the manifest, and browser runtime remained
byte-identical to the corresponding current outputs. An unchanged warm build
reused all 16 mice and 5,213 assets in a 2.99-second median while retaining
both semantic integrity passes.

On a paired clean 134-section, five-sample build containing 1,742 WEBPs, the
four-worker effort-5 build took 65.84 seconds internally and 66.13 seconds wall
time, compared with 180.23 and 180.47 seconds for the published effort-6
implementation. Peak resident memory fell from 352.5 to 284.5 MiB. All 1,474
lossless semantic textures, topology payloads, manifests, and runtime files
remained byte-identical, and every lossy texture retained its alpha mask
exactly. Registered and blended textures grew by 1.66% and 1.67%; their
old-versus-new decoded PSNR values were 44.91 and 46.33 dB. A browser audit
loaded all five samples, all three modes, every K control, and mobile, 1080p,
and 4K layouts without blank canvases, failed requests, or console errors. An
unchanged warm build reused all five mice and 1,742 WEBPs in 2.58 seconds.
