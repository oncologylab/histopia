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

## Current Scope

Implemented now:

- brightfield/IHC tissue-mask candidates with artifact, frame, and QC scoring
- image-aware inset scanner-frame cleanup gated by selected exterior foreground
- fail-closed mask review manifests and exact-shape binary overrides
- strict WSI discovery that excludes label photos and generated artifacts
- scanner-content geometry for SCN thumbnails and native-coordinate warping
- content-scale-aware pyramid fallback when a WSI thumbnail level is corrupt
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
write_processed_images = true
write_warped_images = false
registered_output_dir = "/tmp/histopia-registration-runs/4577/registered"
wsi_compression = "jpeg"
wsi_jpeg_quality = 95
wsi_tile_size = 512
mask_review_path = "/path/to/reviews/mask_review.json"
mask_override_dir = "/path/to/reviews/mask_overrides"
automatic_mask_snapshot_path = "/path/to/reviews/automatic_masks/snapshot.json"
require_approved_masks = true

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
The proposal records adjacent distances, physical areas, a runner-up margin,
the largest internal-cavity fraction for each slide, a graded cavity-continuity
summary, and a fingerprint. Pairwise cavity distance is continuous after a
small noise floor, so nearly identical sections on opposite sides of a review
threshold cannot receive a categorical penalty. Substantial cavities seed
continuity blocks, neighboring weaker cavities extend them, and a single
borderline section may bridge a block. Multiple separated blocks are marked
for human review. Set `require_approved_order = true` to stop before
registration until the exact fingerprint is approved.

Quarter-turn proposals produced by `orient_section_group(...).to_json_dict()`
can be passed directly as `section_orientation_path`. The loader also accepts
the explicit `{"slides": [{"slide": ..., "quarter_turns_ccw": ...}]}` form.
The approved order fingerprint includes these turns, so changing an orientation
invalidates order approval.

Build a fixed-height visual review from the generated proposal and processed
images:

```python
from histopia.registration import build_section_order_review

build_section_order_review(
    "run/section_order_review.json",
    "run/processed",
    "order-review",
)
```

Review cards are cropped around accepted tissue for morphology comparison.
Physical tissue area remains a separate displayed measurement. Changing masks,
cavity topology, anchors, pairwise distances, or the proposed sequence
invalidates approval.

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

Set `thumbnail_workers` above one to decode independent WSI thumbnails in
parallel. This usually shortens startup for multi-slide cohorts, but each
worker temporarily holds another decoded WSI region. Output ordering and image
values are unchanged. Start with `2` or `4` and measure peak memory.

Set `ordering_workers` above one to evaluate independent slide pairs in
parallel on CPU. Results are assigned in deterministic pair order and the
worker count does not change the scientific fingerprint. Start conservatively
because each worker also invokes native OpenCV routines and holds image crops;
`1` is the portable default.

Set `mask_workers` above one to create per-slide mask candidate sets in
parallel on CPU. Group consensus still runs after all independent masks are
complete, and worker count does not change mask pixels. Each worker holds
several thumbnail-sized arrays, so `1` remains the memory-conservative default;
benchmark `2` or `4` on representative cohorts before increasing it further.

Set `mask_review_path`, `mask_override_dir`, and
`require_approved_masks = true` for production runs. Changed thumbnail pixels
or geometry invalidate the saved approval fingerprint. Candidate overlays and
binary masks are written under `qc/mask_candidates/` for adjudication.

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
    --warp-crop-mode reference
```

The command is resumable by default. Existing outputs are checked against the
requested canvas; use `--overwrite` to replace them. Each completed file is
written atomically and recorded in `full_resolution_warps.json`.

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
  --viewer-output-dir /path/to/viewer
```

Serve the output directory over HTTP. Browser module imports do not work
reliably when opening `index.html` directly from the filesystem.

Repeated builds maintain checksum-verified asset and mouse caches. An unchanged
mouse is reused only when its ordered transforms, geometry, reviewed thumbnail
fingerprints, semantic fingerprint, and cohort QC all match and every
referenced output still matches its saved checksum. The build report separates
reused/rendered mice and reused/encoded assets so incremental performance is
auditable.

## Non-Rigid Refinement

Non-rigid refinement is opt-in. Set `non_rigid = true` or
`non_rigid_refinement.enabled = true`. The stored flow maps reference
thumbnail coordinates to the affine-warped moving image. It is accepted only
when structural similarity improves, tissue Dice does not regress beyond the
configured tolerance, Jacobian percentiles stay bounded, and independently
estimated forward/reverse flows are consistent. Rejected fields are identity
and are not applied to WSI output.

Export only accepted fields to a separate native validation tree with
`--accepted-non-rigid-only`. This avoids replacing the validated affine
baseline while non-rigid landmark validation is still pending.

The original OncoSpatial manual-sorted IHC workflow explicitly configured
VALIS with `non_rigid_registrar_cls=None`. Use Histopia's affine workflow when
reproducing that analysis; enabling dense refinement changes the method.

Compare a completed KPF run to existing historical registered outputs:

```bash
histopia-register \
    --compare-kpf-run /tmp/histopia-registration-runs/4577/example \
    --mouse-dir /path/to/validation-data/mouse-1
```

This writes normalized tissue-crop comparison panels under
`historical_reference_qc/`.
