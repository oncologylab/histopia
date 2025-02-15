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
- explicit full-mask mode and recorded full-mask fallback
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
reference_slide = "[#429] Yi_#4577__panc_HE.ndpi"
max_processed_image_dim_px = 1200
crop_mode = "overlap"
rigid_method = "feature"
align_strategy = "hybrid"
non_rigid = false
write_processed_images = true
write_warped_images = false
registered_output_dir = "/tmp/histopia-registration-runs/4577/registered"
wsi_compression = "jpeg"
wsi_jpeg_quality = 95
wsi_tile_size = 512

[mask]
mode = "auto_tissue"
allow_full_fallback = true

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

For current KPF validation, leaving `reference_slide` unset uses the first
natural-order section and has produced better thumbnail-level alignment than
forcing HE for every mouse. `align_strategy = "hybrid"` evaluates direct
reference alignment and serial-neighbor composition, then keeps the transform
with better final tissue-mask overlap.

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
