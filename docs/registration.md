# Registration Development

Histopia registration is being built around robust brightfield/IHC tissue
masking rather than using full-image masks as the default behavior.

## Installation

Install the full registration development stack:

```bash
python -m pip install -e ".[dev,registration,wsi]" \
    -c constraints/registration-repro.txt
```

The WSI loader uses `pyvips`, which also requires native `libvips`. See
`docs/dependency_management.md`.

All `SlideGeometry` coordinates use the raw scanner pixel frame. Histopia does
not apply EXIF/TIFF auto-orientation during native warping or semantic patch
reads. Reviewed quarter-turn decisions are converted back into that source
frame before transforms are saved.

## Current Scope

Implemented now:

- brightfield/IHC tissue-mask candidates with artifact, frame, and QC scoring
- image-aware inset scanner-frame cleanup gated by selected exterior foreground
- adjacent-section recurrence and image-evidence checks for detached mask fragments
- fail-closed mask review manifests and exact-shape binary overrides
- strict WSI discovery that excludes label photos and generated artifacts
- scanner-content geometry for SCN thumbnails and native-coordinate warping
- content-scale-aware pyramid fallback when a WSI thumbnail level is corrupt
- consistent brightfield channel normalization, including grayscale-plus-alpha
  images composited onto white for masking, warping, and semantic extraction
- explicit full-mask mode for legacy reproduction only
- feature and mask-moment rigid thumbnail registration
- conservative affine tissue-mask refinement with transform plausibility gates
- direct, serial, and hybrid transform selection
- per-slide Dice, coverage, area-ratio, and pass/review/fail QC
- lazy full-resolution affine and accepted non-rigid WSI warping
- tiled pyramidal TIFF output with atomic, resumable writes
- opt-in tissue-supported dense refinement with similarity, Dice, Jacobian, and
  forward/backward consistency gates
- KPF raw/reference manifest generation
- CLI entry point for manifests and config-driven thumbnail registration
- static Three.js section-stack viewer generation

Not implemented yet:

- validated OME-XML metadata and OME-TIFF output
- landmark-based cell-level registration validation
- cell-level correspondence and 3D reconstruction

## External Validation Data

Keep validation slides outside the repository and pass their dataset folders to
the manifest command:

```bash
histopia-register --manifest /path/to/validation-data/mouse-1
histopia-register --manifest /path/to/validation-data/mouse-2
```

Generated registration outputs should go to scratch space, for example:

```text
/tmp/histopia-registration-runs/<mouse>/<timestamp>/
```

Do not write generated masks, warped images, or QC images into the KPF source
data tree.

## Example Config

```toml
input_dir = "/path/to/validation-data/mouse-1/raw_wsi"
output_dir = "/tmp/histopia-registration-runs/4577/test-run"
reference_policy = "best_connected"
max_processed_image_dim_px = 1200
crop_mode = "reference"
rigid_method = "feature"
align_strategy = "hybrid"
non_rigid = false
thumbnail_workers = 4
mask_workers = 4
ordering_workers = 4
rigid_workers = 4
# Optional native OpenCV threads used inside each worker.
# opencv_threads = 16
qc_workers = 2
write_processed_images = true
alignment_qc_mode = "review"
write_warped_images = false
registered_output_dir = "/tmp/histopia-registration-runs/4577/registered"
wsi_compression = "jpeg"
wsi_jpeg_quality = 95
wsi_tile_size = 512
# vips_threads = 8
mask_override_dir = "/path/to/reviews/mask_overrides"
automatic_mask_snapshot_path = "/path/to/reviews/automatic_masks/snapshot.json"
require_approved_masks = true
require_approved_order = true

[mask]
mode = "auto_tissue"
allow_full_fallback = false

[refinement]
enabled = true
max_dim_px = 500
min_dice_improvement = 0.01
max_relative_scale_change = 0.35
max_relative_anisotropy = 1.30

[non_rigid_refinement]
enabled = false
max_displacement_fraction = 0.03
smoothing_sigma_px = 12.0
support_dilation_fraction = 0.03
min_similarity_improvement = 0.01
max_mask_dice_loss = 0.01
min_jacobian_p01 = 0.25
max_jacobian_p99 = 4.0
max_inverse_consistency_fraction = 0.02
```

Configuration is validated before slide discovery. Registration modes and WSI
compression must use documented values; canvas cropping is either `reference`
or `overlap`. Worker counts, dimensions, and morphology sizes must be integers
in their documented ranges. Mask fractions and affine/non-rigid acceptance
gates must be finite and physically meaningful, so malformed configs fail
before expensive WSI decoding.

External workflow UIs such as the Histopia QuPath extension can supply
`input_slides` as an exact ordered list of absolute paths. This supports a
QuPath project cohort whose source slides are stored in different directories.
Histopia validates every path and rejects duplicate files, unsupported formats,
derived images, duplicate filenames, and byte-identical slide content before
reading slide pixels. Exact-content preflight groups files by size and compares
small beginning/middle/end samples before full SHA-256 hashing, so files with
unique sizes do not incur a full-file read. The list order is the initial
physical order; configured similarity ordering may then propose a reviewed
replacement.

`mode = "full"` is available only for legacy reproduction and debugging. The
default production path should use `auto_tissue`.

`reference_policy = "best_connected"` chooses a central, well-connected anchor.
`align_strategy = "hybrid"` evaluates direct-reference alignment and
serial-neighbor composition, then keeps the transform with better final
tissue-mask overlap. Physical section order should come from a manifest;
similarity order is provisional and must not be interpreted as a measured
z-axis.

For semi-automatic ordering, set
`section_order_strategy = "anchored_similarity"` and provide a CSV or JSON
manifest whose positive one-based positions are fixed anchors. Unassigned
slides are proposed only for the remaining slots using registration support,
physical tissue area when slide calibration is available, and mask topology.
When no anchor manifest is supplied, an explicit registration reference is
fixed at position 1. Supply a manifest to place one or more anchors elsewhere.
The proposal records adjacent distances, physical areas, a runner-up margin,
the largest internal-cavity fraction for each slide, a graded cavity-continuity
summary, a calibrated-area continuity diagnostic, and a fingerprint. The area
diagnostic marks strong jumps or reversals for review but does not alter the
proposed order, because similarity ordering is not a measured physical z-axis.
Pairwise cavity distance is continuous after a small noise floor, so nearly
identical sections on opposite sides of a review threshold cannot receive a
categorical penalty. Substantial cavities seed continuity blocks, neighboring
weaker cavities extend them, and a single borderline section may bridge a
block. Multiple separated blocks are marked for human review. Set
`require_approved_order = true` to stop before registration until the exact
fingerprint is approved.

Quarter-turn proposals produced by `orient_section_group(...).to_json_dict()`
can be passed directly as `section_orientation_path`. The loader also accepts
the explicit `{"slides": [{"slide": ..., "quarter_turns_ccw": ...}]}` form.
The approved order fingerprint includes these turns, so changing an orientation
invalidates order approval.

Build a fixed-height visual review from the generated proposal and processed
images:

```python
from histopia.visualization import build_section_order_review

build_section_order_review(
    "run/section_order_review.json",
    "run/processed",
    "order-review",
)
```

The equivalent CLI supports bounded parallel encoding:

```bash
histopia-visualize order-review \
    run/section_order_review.json \
    run/processed \
    order-review \
    --workers 4
```

Review cards are cropped around accepted tissue for morphology comparison.
Physical tissue area remains a separate displayed measurement. Changing masks,
cavity topology, anchors, pairwise distances, or the proposed sequence
invalidates approval.

Order-review WebPs use stable per-slide names and a local checksum cache.
Changing only the proposed sequence reuses unchanged image assets; changing
thumbnail pixels, mask pixels, quarter-turn orientation, or encoder settings
re-encodes the affected slide. Output checksums are verified before reuse, so
a missing or modified WebP is regenerated. The cache is performance metadata
and never changes or bypasses the order fingerprint. On a representative
24-slide review, a cold build took 30.48 seconds, an exact warm build took
1.78 seconds, and a reorder-only build took 1.77 seconds while preserving all
24 WebP checksums. `workers = 1` remains the memory-conservative library and
CLI default; increase it only after measuring cold-build memory on
representative slides. Worker count does not alter cache keys, manifests, or
encoded bytes. On the same cold review, one, two, four, and eight workers took
30.77, 15.68, 8.16, and 4.66 seconds with peak RSS of approximately 125, 176,
273, and 448 MiB. Four workers is a balanced server setting; eight favors
throughput when memory is available.

Build a fixed-viewport audit of every accepted tissue mask before approval:

```bash
histopia-visualize mask-review \
    /path/to/registration-run \
    /path/to/mask-review
```

The audit uses full thumbnails rather than tissue crops so scanner frames,
debris, and excluded peripheral tissue remain visible. It records the exact
mask fingerprint and does not mark a cohort approved.

Pairwise morphology distances are cached under the registration output
directory because an all-pairs comparison is expensive for long stacks. The
cache is reused only when the ordered slide set, reviewed mask pixels, physical
geometry, quarter-turn orientation, rigid method, refinement settings, and
distance algorithm/version and weights match exactly. A stale, incomplete, or
checksum-invalid cache is ignored and rebuilt; it never bypasses order
fingerprint approval.

The deterministic anchored beam-search proposal has a separate atomic cache
bound to the exact distance-matrix bytes, anchors, physical areas, input-mask
fingerprints, orientations, cavity metrics, beam width, and algorithm version.
Its retained slide order and runner-up score carry a content checksum; on
reuse, Histopia reconstructs objective, adjacent-distance, fixed-position, and
proposal-fingerprint fields from current inputs. A stale, malformed,
symlinked, or checksum-invalid proposal is recomputed without affecting the
distance cache or an exact human approval.

The v3 ordering implementation prepares immutable rigid features once per
slide and reuses them across every pair. On an 11-slide validation cohort at
1200 processed pixels, this reduced a cold distance build from 85.16 seconds
and 1.29 GB peak RSS to 5.27 seconds and 0.70 GB on a 32-vCPU AMD EPYC
environment. The old and new distance matrices were element-for-element
identical, so this optimization does not alter ordering results.

The same run-scoped feature set is reused for automatic reference scoring,
direct-reference matching, and serial matching. On a 23-slide approved
brightfield stack, hybrid alignment feature detections fell from 88 to 23 and
the measured alignment stage fell from 9.43 to 7.23 seconds. All 22
non-reference matrices, methods, match and inlier counts, warnings, and parent
links were exactly unchanged. Parallel preparation uses additional transient
memory, so use a smaller worker count when memory is the limiting resource.
Before an explicit-reference warm run, Histopia now validates and preloads the
exact required reference, serial, or hybrid transform set. Feature detection
is skipped only when every pair is current; one missing or corrupt pair
restores eager per-slide preparation before any transform is recomputed. On a
24-slide hybrid validation stack, this preflight and the proposal cache
reduced exact-warm pipeline time from 8.78 to 6.02 seconds and process wall
time from 9.43 to 6.52 seconds. The section-order review, registration result,
rigid transforms, and 93-bundle QC manifest remained byte-identical.

Set `thumbnail_workers` above one to decode independent WSI thumbnails in
parallel. This usually shortens startup for multi-slide cohorts, but each
worker temporarily holds another decoded WSI region. Output ordering and image
values are unchanged. Start with `2` or `4` and measure peak memory.

Set `ordering_workers` above one to evaluate independent slide pairs in
parallel on CPU and to prepare per-slide rigid features concurrently. Results
are assigned in deterministic order and the worker count does not change the
scientific fingerprint. Start conservatively because each worker also invokes
native OpenCV routines and holds image crops; `1` is the portable default.

Set `rigid_workers` above one to estimate independent reference and adjacent
slide transforms through a bounded ordered CPU pool. Serial transform
composition remains deterministic after all pair estimates complete. Exact
pair-cache reads, writes, hit counts, and result ordering are thread-safe.
`1` remains the portable default; `4` is a balanced server setting, while
larger values retain more crop and OpenCV working memory.

On a 24-slide, 1200-pixel hybrid registration benchmark, four rigid workers
reduced reusable feature preparation plus 45 unique pair estimates from 8.84
to 2.84 seconds. In a disposable full pipeline run that forced every alignment
to recompute, rigid alignment fell from 9.32 to 3.25 seconds and total runtime
from 14.27 to 5.45 seconds. The complete `registration_result.json` remained
byte-identical. A separate cold cache-enabled run produced exactly 45 cache
entries, 45 misses, and one duplicate-pair hit with the same scientific
digest.

Prepared slide features also retain the exact center, principal-axis angle,
and scale already needed by the tissue-mask fallback. A slide is therefore
scanned once during bounded feature preparation instead of again for every
fallback pair. On the same forced-recomputation workload, this reduced
one-worker rigid alignment from 9.32 to 5.84 seconds and total runtime from
14.26 to 8.13 seconds. With four workers, rigid alignment fell from 3.25 to
2.94 seconds and total runtime from 5.43 to 5.14 seconds. A profile exercised
32 fallback estimates while evaluating mask properties exactly 24 times, once
per slide. One- and four-worker results were byte-identical and matched the
pre-optimization result after normalizing only the output directory.

`opencv_threads` optionally caps OpenCV's process-wide native pool while a
registration is active. Histopia records the requested and effective values,
then restores the caller's prior OpenCV setting even if registration raises.
Leave it unset to preserve OpenCV's host-specific default. This inner pool and
`rigid_workers` are independent, so benchmark them together instead of
multiplying both to the processor count.

On the same 24-slide workload with four rigid workers, three runs at 16 OpenCV
threads had median total/rigid times of 5.11/2.96 seconds and used 737% average
CPU. Three runs at this host's 32-thread default took 4.90/2.76 seconds and
used 1,018% average CPU. Six complete results were byte-identical. Sixteen is
therefore a useful shared-server setting on this 32-vCPU host: it trades about
4% wall time for about 28% less CPU pressure, while leaving the default unset
retains maximum measured throughput.

Set `mask_workers` above one to create per-slide mask candidate sets in
parallel on CPU. Cohort-aware ranking, pale-tissue recovery, component
consensus, frame cleanup, and artifact encoding also use bounded ordered maps,
with a full cohort barrier between every scientifically dependent phase.
Independent group-cache entries are compressed, loaded, and verified through
the same ordered worker pool.
Worker count does not change mask pixels, review JSON, or rendered artifact
bytes. Each worker holds several thumbnail-sized arrays, so `1` remains the
memory-conservative default; `4` is a balanced server setting and `8` should
be treated as a high-throughput ceiling until the target host is benchmarked.
On a 20-slide, 1200-pixel cold independent-mask benchmark on a 32-vCPU AMD EPYC
host, one, two, four, eight, and 16 workers took 48.39, 25.00, 13.59, 8.21, and
8.27 seconds, with peak RSS of 0.54, 0.66, 0.83, 1.19, and 2.02 GB. Sixteen
workers therefore doubled peak memory without improving elapsed time.

Candidate-independent brightness, optical-density, saturation, background, and
blank-glass maps are computed once per slide. Connected-component filters use
label bounds instead of repeatedly scanning full label images. On the same
four-worker workload these exact-output changes reduced independent candidate
generation from 43.68 to 13.38 seconds (69.4%). Complete result digests,
including every candidate mask and QC field, remained identical across 140
real slides from seven cohorts, so the scientific cache schema did not change.
Mask scoring also reuses each canonical morphology metric set, and group
consensus reuses target-specific peer translations across its broad, direct,
and adjacent support radii.

Moderately undercovered sections can recover open, pale tissue only when
strong cohort support, attachment to an existing tissue component, containment
within that component's convex envelope, pervasive fine-scale texture, and
material mask growth all agree. Detached-object recovery remains a separate,
more conservative path. These checks preserve pale tissue while rejecting
smooth scanner-frame whitespace, narrow glass gaps, and unsupported debris.

Group refinement additionally reuses independent candidate metrics,
target-specific dominant-object centroids, and candidate-independent
augmentation geometry. On a representative 20-slide stack, these
exact-output changes reduced the four-worker group phase from 9.57 to 7.70
seconds (19.6%). Complete group-refined results matched the validated
pre-change cache for all 182 slides across nine real cohorts.

Mask artifact rendering prepares image-only background and tissue colors once
per slide and computes the same four-connected boundary with direct array
operations. For 420 PNG artifacts from a 20-slide stack, sequential rendering
fell from 69.83 to 51.17 seconds and four-worker rendering fell from 18.84 to
13.44 seconds. Eight workers reached 7.73 seconds, while isolated peak RSS
rose from 568 MiB with one worker to 778 MiB with four and 1.01 GiB with
eight. Every generated PNG remained byte-identical to the baseline, including
all candidate masks and overlays.

Set `qc_workers` above one to render independent pair-crop, registered-view,
non-rigid, and primary-review bundles concurrently. Bundle filenames, pixels,
checksums, cache manifests, registration results, and approval fingerprints are
independent of this execution control. Each active alignment renderer can hold
several processed-image arrays plus a wide contact sheet, so `1` is the
memory-conservative default; measure peak RSS before using `2` or `4` for
1200-pixel cohorts. On a 24-slide cached scientific run with all 93 QC bundles
removed, four workers reduced total wall time from 269.78 to 77.49 seconds
(3.48x) while peak RSS rose from 1.36 to 1.58 GB. All 300 retained PNGs, the
QC manifest, and the canonical registration result were byte-identical.

`alignment_qc_mode` separates scientific registration from optional diagnostic
image volume. The production default, `review`, writes one primary panel per
slide and any non-rigid acceptance panel, which is sufficient for the
interactive registration viewer and routine review. `full` additionally writes
pair-crop, full-thumbnail, and crop-frame warped, blend, checkerboard, and
contact diagnostics for algorithm debugging. `none` skips post-mask alignment
images while retaining processed thumbnails, masks, transforms, metrics, and
full-resolution export support. The mode is recorded in performance telemetry
but excluded from registration results and approval fingerprints. When an
existing run is reduced from `full` to `review` or `none`, Histopia removes
only diagnostic files recorded in its checksum manifest, retains untracked
files, and reports the pruned file and byte counts in performance telemetry.
On a representative 24-slide brightfield run, `review` reduced cold
registration-plus-QC time from 69.83 to 10.97 seconds (84.3%) and retained
artifact volume from 1.278 GB to 437 MB (65.8%). The canonical registration
result and all 24 primary review panels were byte-identical. Reducing the
existing `full` run in place took 5.15 seconds and removed 276
manifest-tracked diagnostics totaling 840.5 MB without changing untracked
reviewer files.

`preprocessing_cache = true` is the default. Histopia reuses decoded
thumbnails, independent mask candidates, group-refined masks, and rendered
mask-review artifacts only when their source metadata, pixel data,
configuration, physical calibration, and algorithm schema match. Missing or
corrupt entries are regenerated. Set it to `false` for an intentionally cold
run; review approval fingerprints are enforced independently of this cache.
Every rendered mask-artifact bundle also records the exact relative path,
byte size, and SHA-256 of each thumbnail, mask, overlay, and candidate image.
Missing, changed, truncated, symlinked, or escaping outputs regenerate only
the affected slide bundle. Upgrading an older filename-only manifest rebuilds
each bundle once. On a 24-slide validation stack with 504 retained PNGs, a
fully warm checksum pass took 0.43 seconds; deliberately truncating one PNG
rebuilt one slide in 5.43 seconds, reused the other 23, and restored every
baseline output hash.

`alignment_cache = true` independently enables exact directed rigid-pair
reuse. Each entry binds oriented crop pixels, reviewed masks, crop
offsets/scales, pair direction, rigid and affine-refinement settings, algorithm
schema, and OpenCV/NumPy versions. The cached full and crop-space transforms
carry a content checksum and are loaded only when every bound input matches;
stale, malformed, or corrupted entries are recomputed atomically. Cache
hit/miss and actual-computation counts appear in
`registration_performance.json`.
Within every run, identical directed pairs also use a thread-safe single-flight
memory cache even when persistent alignment caching is disabled. This prevents
the reference/serial overlap in hybrid alignment from running the same OpenCV
fit twice. Telemetry reports these as `rigid_pair_memory_hits`; on the
24-slide benchmark, computations fell from 46 to the 45 unique pairs with an
exactly unchanged registration result.

The same setting enables checksum-validated registration QC reuse. Alignment,
crop, non-rigid, and labeled review bundles bind their exact render inputs and
are reused only when every expected file has the recorded path, size, and
SHA-256. Missing, changed, truncated, or symlinked outputs regenerate only
their affected bundle. Hybrid alignment no longer renders a direct-reference
pair diagnostic that the serial diagnostic immediately overwrote; the retained
pair-crop output is unchanged in purpose. On a 24-slide, 1200-pixel validation
stack, initial current-code QC population took 264.97 seconds. An exact warm
rerun verified 46 rigid-pair entries and 93 QC bundles, rendered nothing, and
completed in 8.49 seconds, a 31.2x speedup with an identical canonical
registration-result SHA-256.

Set `mask_override_dir` when manual corrections are required and
`require_approved_masks = true` for production runs. Keep the generated
`mask_review.json` under the run directory so the final approval can seal it.
Changed thumbnail pixels or geometry invalidate the saved approval fingerprint.
Candidate overlays and binary masks are written under `qc/mask_candidates/`
for adjudication.

Mask, order, and registered-stack review builders keep checksum-validated WebP
caches inside their generated output directories. Reopening unchanged QC
verifies each input fingerprint and output checksum before reuse. Changed
masks, thumbnails, transforms, encoding settings, missing files, or corrupted
assets regenerate only the affected images. `--workers` controls cold mask,
order, and alignment rendering; warm reuse remains checksum-bound regardless
of worker count. On a 51-slide four-cohort review, an eight-worker cold build
fell from 25.22 to 12.75 seconds and an unchanged warm rebuild fell from 16.32
to 1.12 seconds; every mask-review WebP remained byte-identical.

Combine several staged runs under one fixed-viewport review endpoint:

```bash
histopia-visualize registration-cohort-review candidate-review/ \
  --run cohort-a=/path/to/cohort-a/run \
  --run cohort-b=/path/to/cohort-b/run \
  --workers 4
```

The outer portal reports each cohort's slide count and per-stage approval
state. Generation rejects a cohort when its prepared mask, order, or alignment
stages disagree on slide count. Existing stage arrays remain in the manifest
for compatibility, while `stage_summary` records the validated count and
approval state. On an eight-cohort, 131-section review, a four-worker cold
build took 50.84 seconds at 541 MiB peak RSS. The exact warm build took 2.53
seconds at 177 MiB, reused all 131 mask assets, and preserved all 387
non-observational files byte-for-byte.

For a strict production run, keep the review manifests in the registration
run directory and advance the workflow explicitly:

```bash
# 1. Prepare masks. This returns a review_required JSON status.
histopia-register --config registration.toml --staged
histopia-visualize registration-review run/ run-review/ --workers 4

# 2. After reviewing every mask, record the exact mask fingerprints.
histopia-register \
  --approve-masks run/ \
  --reviewer "Reviewer name" \
  --review-notes "Every tissue mask visually reviewed."

# 3. Re-run to prepare morphology-aware order, then review and approve it.
histopia-register --config registration.toml --staged
histopia-visualize registration-review run/ run-review/ --workers 4
histopia-register \
  --approve-order run/ \
  --reviewer "Reviewer name" \
  --review-notes "Section morphology and physical order visually reviewed."

# 4. Re-run to compute registration.
histopia-register --config registration.toml --staged
```

`--staged` changes only CLI handling of an expected review gate: it returns
exit code zero with `status = "review_required"`. Direct library calls and CLI
runs without `--staged` still raise `RegistrationApprovalRequired`. Mask and
order approvals are written atomically and survive a rerun only when their
exact fingerprints remain current. Prepared mask/order manifests, final
registration JSON, validation reports, and incremental full-resolution warp
summaries also use atomic replacement, preventing cancellation from leaving a
truncated final-path artifact. The combined review portal supports a mask-only
first stage and adds the order tab after the proposal exists.

Each invocation also atomically checkpoints `registration_performance.json`.
It records safe worker and algorithm controls, total elapsed time, and durations
for discovery, thumbnail loading, masks, ordering, rigid alignment, optional
refinement, QC rendering, full-resolution export, and result writing. A normal
mask or order pause is recorded as `review_required`; cancellation is
`interrupted`, and an actual exception is `failed`. The command-line boundary
translates launcher `SIGTERM` into a graceful exit so the active stage is
checkpointed before the process stops. This file is observational:
it is excluded from the registration result and approval fingerprints, so
timing differences cannot invalidate or alter scientific results.
Review generation uses legacy artifact discovery only when this telemetry file
is absent. If a current telemetry file exists but is unreadable or invalid,
review generation fails closed instead of exposing potentially stale mask,
order, or alignment artifacts.
Stable viewer generation follows the same rule for final approvals. An absent
`registration_approval.json` remains an explicit review gate. If the file
exists but its schema, artifact digests, order fingerprint, slide count, mask
states, reviewer, or timestamp is invalid, viewer generation fails rather than
silently presenting the run as merely unapproved. Unsealed runs are labeled
`Registration approval required` in the 3D viewer.
Mask telemetry additionally separates independent candidate extraction, group
refinement, review resolution, artifact encoding, and rendered/reused slide
counts. Ordering telemetry distinguishes distance/proposal cache hits and
proposal-search time; rigid telemetry records feature-preparation time and
slide count plus the number of exact pair transforms preloaded before
alignment.

Registration currently uses CPU implementations in NumPy, SciPy, and OpenCV;
it does not expose a GPU selector or silently move registration work to CUDA.
The performance record reports `compute_backend = "cpu"` alongside the worker,
effective OpenCV, and libvips controls. GPU, CPU, and Apple MPS selection is
available for the separate UNI2-h feature-extraction stage.

After reviewing the completed mask, order, and registration views, seal the
exact artifacts without recomputing unchanged transforms:

```bash
histopia-register \
  --approve-run /path/to/completed-run \
  --reviewer "Reviewer name" \
  --review-notes "Masks, order, and registration visually reviewed."
```

The command refuses mismatched slide sets, reordered results, changed mask
fingerprints, rejected masks, and missing overrides. It updates review metadata
with atomic per-file replacement, then writes `registration_approval.json`
last with SHA-256 digests for the registration result, mask review, and order
review. Any later artifact change invalidates that approval.

Use `automatic_mask_snapshot_path` when a complete set of automatically
generated masks has already passed visual review. The JSON snapshot must use
schema version 1 and contain exactly one row per input slide:

```json
{
  "schema_version": 1,
  "slides": [
    {
      "slide": "section-001.ndpi",
      "mask": "section-001.mask.png",
      "sha256": "<sha256-of-the-encoded-mask-file>"
    }
  ]
}
```

Mask paths are relative to the snapshot. Histopia rejects missing or extra
slides, hash mismatches, and masks whose pixel dimensions differ from the
current processing thumbnail. The snapshot records reviewed automatic output;
manual corrections still belong in `mask_override_dir`.

Affine refinement uses signed distance fields from tissue masks, not stain
intensity. A candidate is accepted only if it improves tissue Dice and stays
within the configured relative scale and anisotropy limits.

The generated `registration_result.json` contains mask and alignment metrics.
`validation_report.md` applies the acceptance thresholds documented in
`docs/kpf_registration_validation.md`.

## Full-Resolution Export

Set `write_warped_images = true` to export during registration, or apply an
already validated run without repeating registration:

```bash
histopia-register \
    --warp-run /tmp/histopia-registration-runs/4630/qc-1200-hybrid \
    --registered-output-dir /tmp/histopia-full-resolution-runs/4630 \
    --vips-threads 8 \
    --warp-crop-mode reference
```

Repeat `--warp-slide` with an exact source filename or stem to export a reviewed
subset without rewriting unaffected TIFFs:

```bash
histopia-register \
    --warp-run /tmp/histopia-registration-runs/mouse-1 \
    --registered-output-dir /tmp/histopia-full-resolution-runs/mouse-1 \
    --warp-slide section-001.ndpi \
    --warp-slide section-004
```

The command is resumable by default. Histopia reconstructs native coordinates
from the scanner content bounds saved during registration. Each completed file
is written atomically and recorded in `full_resolution_warps.json` with a
fingerprint of the registration result, source/reference file identities,
transform, non-rigid displacement, crop, writer settings, and output identity.
An existing TIFF is reused only when that complete request still matches and
the file remains readable with the expected canvas. Outputs created by an
older summary schema, changed inputs, or changed settings require
`--overwrite`.

`vips_threads` and `--vips-threads` bound libvips' process-wide native worker
pool. The setting controls throughput only and is excluded from scientific
fingerprints. It must be applied before pyvips initializes, so use a fresh
process when changing it. Leave it unset for libvips' adaptive default, or
benchmark explicit values against the intended scanner format, storage, and
host memory.

On the validated 17,280 x 17,664 4630 SCN export, explicit caps of 1, 2, 4, 8,
and 16 completed in 21.15, 14.40, 8.46, 5.17, and 5.25 seconds, respectively.
Peak RSS was 472, 492, 518, 598, and 839 MiB. All five pyramidal TIFFs were
byte-identical to the previously reviewed correction. Eight threads was the
measured throughput point on that host; sixteen added memory without reducing
wall time. This benchmark is guidance rather than a portable default.

`reference` is the safe crop default and preserves the entire reference
canvas. `overlap` reproduces a legacy-style common valid rectangle, but can
remove reference anatomy when a cohort contains partial sections. Pyramidal
output currently requires JPEG compression, the path validated against the KPF
slides. Files are named `*.registered.tiff`: Histopia does not claim OME-TIFF
until OME-XML metadata is implemented and independently validated.

## Section-Stack Viewer

```bash
histopia-register \
  --viewer-run mouse-1=/path/to/run-1 \
  --viewer-run mouse-2=/path/to/run-2 \
  --provisional-mouse mouse-2 \
  --viewer-workers 4 \
  --viewer-output-dir /path/to/viewer
```

Serve the output directory over HTTP. Browser module imports do not work
reliably when opening `index.html` directly from the filesystem.

Stable viewer builds include only fingerprint-approved registration results,
approved semantic results, and approved stain families. Build one separate,
fixed-viewport review endpoint for pending work:

```bash
histopia-visualize review /path/to/review \
  --run mouse-1=/path/to/run-1 \
  --run mouse-2=/path/to/run-2 \
  --semantic-run mouse-1=/path/to/semantic-1 \
  --stain-run mouse-1=/path/to/stain-1 \
  --workers 4
```

The hub orders pending registration cohorts first and exposes only the workflow
tabs that have prepared artifacts. It does not grant scientific approval.

Repeated builds maintain checksum-verified asset and mouse caches. An unchanged
mouse is reused only when its ordered transforms, geometry, reviewed thumbnail
fingerprints, semantic fingerprint, and cohort QC all match and every
referenced output still matches its saved checksum. The build report separates
reused/rendered mice and reused/encoded assets so incremental performance is
auditable. Semantic artifacts and their registration binding are fully
verified before generation and rechecked before publication, including on a
warm cache hit. `--viewer-workers` bounds concurrent WebP encoders. The default
of one is memory-conservative; four is a balanced server setting. Encoded
bytes, manifest data, and cache order are deterministic across worker counts.
Registered and blended review textures use libwebp effort 5; lossless semantic
textures retain effort 6. `build-report.json` records both settings and the
mouse-cache schema that invalidates outputs when this contract changes.

On a paired 15-section semantic-atlas benchmark producing 195 WebPs, reducing
only the lossy encoder effort cut one-worker wall time from 35.05 to 12.46
seconds and four-worker wall time from 16.16 to 5.40 seconds. All 165 lossless
semantic WebPs remained byte-identical, every alpha mask remained exact, and
registered and blended payload sizes increased by about 1.0%.

On a larger five-cohort benchmark with 134 sections and 1,742 WebPs, the
four-worker build fell from 180.23 to 65.84 seconds internally and from 180.47
to 66.13 seconds wall time. Peak resident memory fell from 352.5 to 284.5 MiB.
A complete browser audit passed sample switching, Histology, Blend, and
Semantic modes, K switching, and mobile, 1080p, and 4K layouts. An exact warm
build reused all five mouse payloads and all 1,742 WebPs in 2.58 seconds
internally. Four remains the balanced cold-build setting; exact cache reuse
remains the preferred path for repeated review.

On the complete 16-cohort, 401-section viewer, eliminating duplicate integrity
work reduced a three-run warm median from 5.99 to 2.99 seconds. All 5,213
assets were checksum-verified and reused, both semantic validation passes
remained active, and the production manifest was byte-identical.

## Non-Rigid Refinement

Non-rigid refinement is opt-in. Set `non_rigid = true` or
`non_rigid_refinement.enabled = true`. The stored flow maps reference
thumbnail coordinates to the affine-warped moving image. It is accepted only
when structural similarity improves, tissue Dice does not regress beyond the
configured tolerance, Jacobian percentiles stay bounded, and independently
estimated forward/reverse flows are consistent. Rejected fields are identity
and are not applied to WSI output.

Each candidate is also evaluated against mutual ORB correspondences detected
after affine alignment. DIS does not consume these sparse features, so their
before/after median and p95 residuals provide a held-out algorithmic check on
the dense field. They are diagnostics rather than an acceptance gate and must
not be described as anatomical landmarks or cell-level ground truth. A
rejected candidate remains available in memory long enough to render labeled
QC, while the serialized and applied displacement remains identity.

Build a fixed-viewport provisional review from either a non-rigid registration
run or a standalone validation bundle:

```bash
histopia-visualize non-rigid-review \
    /path/to/non-rigid-run \
    /path/to/non-rigid-review \
    --workers 4
```

The browser compares the reference, affine baseline, dense candidate,
checkerboard, displacement magnitude, acceptance checks, and sparse
correspondence diagnostics. Its output is a review artifact only; generation
does not approve or promote a dense field.

Export only accepted fields to a separate native validation tree with
`--accepted-non-rigid-only`. This avoids replacing the validated affine
baseline while non-rigid landmark validation is still pending.

The original OncoSpatial manual-sorted IHC workflow explicitly configured
VALIS with `non_rigid_registrar_cls=None`. Use Histopia's affine workflow when
reproducing that analysis; enabling dense refinement changes the method.

Approved section-order artifacts are immutable during registration. If a new
proposal conflicts with an approved fingerprint, Histopia preserves the
approved file byte-for-byte and writes the candidate to
`section_order_review.pending.json` when working in the same run directory. If
the approved review was supplied from another directory, the candidate is
written to the new run's canonical `section_order_review.json`. Explicit
section-order approval promotes a same-run pending proposal to the canonical
path.

Compare a completed KPF run to existing historical registered outputs:

```bash
histopia-register \
    --compare-kpf-run /tmp/histopia-registration-runs/4577/example \
    --mouse-dir /path/to/validation-data/mouse-1
```

This writes normalized tissue-crop comparison panels under
`historical_reference_qc/`.
