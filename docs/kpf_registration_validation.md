# KPF Registration Validation Notes

This document records the current thumbnail-level validation status for the
staged KPF example data.

## Current Best Local Run

Run profile:

- input root: external KPF validation dataset (not tracked)
- output root: `/tmp/histopia-registration-runs`
- processing size: `max_processed_image_dim_px = 1200`
- reference strategy: best-connected central anchor
- order strategy: recorded physical order for 4257, 4577, and 4630;
  provisional morphology-similarity order for 5997
- alignment strategy: `hybrid`
- mask mode: `auto_tissue`
- full-mask fallback: disabled
- mask review: 105 visually reviewed; 84 automatic passes and 21 reviewed
  overrides
- affine refinement: signed-distance ECC with conservative plausibility gates

Current results:

| Mouse | Slides | Reference | Pass | Review | Fail | Full masks | Median Dice |
|---|---:|---|---:|---:|---:|---:|---:|
| 4257 | 38 | `[#220] Yi_#4257_panc_p-p38.ndpi` | 37 | 0 | 0 | 0 | 0.894 |
| 4577 | 25 | `[#206] Yi_#4577_panc_PAS.ndpi` | 24 | 0 | 0 | 0 | 0.873 |
| 4630 | 24 | `[#234] Yi_#4630_panc_Yap.ndpi` | 22 | 1 | 0 | 0 | 0.964 |
| 5997 | 18 | `[#475] Yi_#5997_panc_SMA.ndpi` | 16 | 1 | 0 | 0 | 0.878 |

Reference sections are listed separately and are not counted as pass slides.
All 101 non-reference sections completed without a hard registration failure.

Historical-reference comparison:

| Mouse | Slides | Median direct tissue Dice | Median moment-aligned tissue Dice | Main outliers |
|---|---:|---:|---:|---|
| 4257 | 38 | 0.720 | 0.630 | Casp3, pS6, CK18 |
| 4577 | 25 | 0.591 | 0.523 | Yap, Sox2, Lamp1 |
| 4630 | 24 | 0.937 | 0.917 | PAS, GFP, Sox5 |
| 5997 | 18 | 0.731 | 0.907 | Tm, pERK, E-Cad |

These comparison metrics use normalized tissue crops against the historical
`registered/` OME-TIFFs. The visual panels are the primary evidence because
the historical outputs and Histopia thumbnail outputs can have different
coordinate frames and scanner canvas conventions.

## Acceptance Criteria

The thumbnail rigid stage is accepted for coarse section placement when every
non-reference slide avoids `fail`, no full-image mask is used, and each review
case is visually adjudicated.

- `fail`: Dice below 0.25, reference coverage below 0.15, moving coverage below
  0.25, or transform score below 10.
- `review`: Dice below 0.55, reference coverage below 0.35, moving coverage
  below 0.50, or a full-image mask was used.
- `pass`: none of the fail or review conditions apply.

The remaining low-coverage cases are 4630 pERK and 5997 pERK. Their panels
show partial sections placed in plausible reference locations.
The missing reference anatomy is not present in those source slides, so these
are coverage limitations rather than transform failures.

## Full-Resolution Validation

All 105 affine results were applied to the original NDPI/SCN files and read
back from the dedicated registration output tree under
`/media/volume/data/histopia`. The source data tree was not modified.

| Mouse | Files | Reference canvas (H x W) | Pyramid levels | Native/thumb mask Dice | Native/thumb MAE |
|---|---:|---:|---:|---:|---:|
| 4257 | 38 | 43,264 x 51,840 | 8 | 0.968 | 3.62 |
| 4577 | 25 | 47,360 x 59,520 | 8 | 0.977 | 2.68 |
| 4630 | 24 | 17,664 x 17,280 | 7 | 0.998 | 2.63 |
| 5997 | 18 | 19,456 x 19,200 | 7 | 0.981 | 3.21 |

Every TIFF had the expected reference dimensions, three `uchar` RGB bands,
readable pyramid pages, and no unfinished temporary file. There were no
missing, unexpected, or malformed outputs. `Native/thumb` values compare a
thumbnail rendered from the native pyramid with the previously accepted
thumbnail-space warp. MAE is measured on the 0-255 RGB scale.

Visual readback included representative aligned sections, all four rigid review
cases, and rescued 4257 pAMPKa. Native output preserved the same placement and
partial-coverage decisions as thumbnail validation.

Reproduce this audit with:

```bash
python scripts/validate_kpf_full_resolution.py \
    --registration-root /media/volume/data/histopia/registration/KPF/runs \
    --full-resolution-root /media/volume/data/histopia/registration/KPF/registered \
    --output /media/volume/data/histopia/registration/KPF/registered/audit.json
```

## Non-Rigid Validation

The active calls in the legacy
`utils_image_registration_manual_sorted.py` workflow all passed
`non_rigid_registrar_cls=None` to VALIS. The original IHC analysis therefore
used rigid/affine registration only. Histopia's dense refinement is a new,
opt-in experiment and is not part of the legacy-reproduction requirement.

The optional tissue-supported DIS refinement was run on all 101 non-reference
slides. Acceptance is deliberately conservative.

| Mouse | Accepted | Rejected | Median similarity gain | Median Dice change | Median inverse residual (px) |
|---|---:|---:|---:|---:|---:|
| 4257 | 18 | 19 | 0.168 | 0.032 | 12.42 |
| 4577 | 6 | 18 | 0.168 | 0.034 | 18.15 |
| 4630 | 14 | 9 | 0.140 | 0.016 | 16.28 |
| 5997 | 4 | 13 | 0.089 | 0.022 | 15.65 |

Forty-two fields passed all similarity, Dice, Jacobian, displacement, and
forward/backward consistency checks. Rejected fields remain affine. A final
accepted 4630 E-Cad field was composed into the original 15,360 x 12,800 WSI;
its six pyramid levels were readable, pyramid-to-level-0 MAE was 0.74/255, and
visual alignment showed restrained local correction.

All 42 accepted fields were then exported from the original WSI data to a
separate native validation tree. Every output had the reference canvas, three
RGB bands, the expected pyramid depth, accepted-field provenance, and no
unfinished temporary file.

| Mouse | Native fields | Pyramid levels | Native/expected mask Dice | Native/expected MAE |
|---|---:|---:|---:|---:|
| 4257 | 18 | 8 | 0.936 | 4.66 |
| 4577 | 6 | 8 | 0.963 | 3.78 |
| 4630 | 14 | 6 | 0.994 | 4.41 |
| 5997 | 4 | 7 | 0.993 | 2.74 |

The native-composition audit requires median mask Dice of at least 0.90 and
median RGB MAE no greater than 5/255. The minimum per-slide mask Dice was 0.854
for pale 4257 pAMPKa; its RGB MAE was 6.62/255 and direct visual comparison
showed the same anatomy and local correction. This lower mask agreement is
reported as sensitivity of binary tissue thresholding after JPEG pyramid
encoding, not evidence for a geometric failure.

Non-rigid output remains opt-in. These metrics demonstrate numerical safety
and reproducibility, but they do not replace independent landmarks or
cell-level ground truth.

Audit saved decisions and flow files with:

```bash
python scripts/validate_kpf_nonrigid.py \
    --non-rigid-root /tmp/histopia-nonrigid-runs \
    --output /tmp/histopia-nonrigid-runs/audit.json

python scripts/validate_kpf_nonrigid_full_resolution.py \
    --non-rigid-root /tmp/histopia-nonrigid-runs \
    --full-resolution-root /tmp/histopia-full-resolution-nonrigid \
    --output /tmp/histopia-full-resolution-nonrigid/audit.json
```

## Visual QC Findings

- Brightfield tissue masking is substantially improved over the original
  full-mask workaround. The current run used zero full-mask fallbacks.
- The new `edge_texture` mask candidate rescued pale IHC slides such as
  `4257` pAMPKa, where optical-density masking only selected artifacts.
- Frame-shaped scanner artifacts and tiny remote debris no longer determine
  tissue crop bounds.
- Tissue-crop normalization is required. Registering whole-slide thumbnails can
  fail when raw slides have very different blank scanner canvas sizes.
- First natural-order reference is currently better than forcing HE for all
  mice. HE slides can have large blank canvases or different tissue extent.
- Hybrid alignment prevents serial drift by choosing direct-reference alignment
  unless serial-neighbor composition improves final tissue-mask overlap. In the
  current runs, serial was selected for `4257` Lamp1 and direct-reference was
  selected for the other inspected difficult cases.
- Historical-reference comparison is strongest for `4630`; 4257, 4577, and
  5997 improve over the pre-refinement direct-Dice baselines.
- The current stage is scientifically usable for coarse section placement,
  ordering, coverage assessment, and initialization of later registration.
  It is not validated for cell-level correspondence or quantitative transfer
  of marker measurements between sections.

## Remaining Work

- Add curated anatomical landmarks and cell-level annotations for independent
  non-rigid validation.
- Add validated OME-XML metadata before offering OME-TIFF output.
- Add landmark or contour annotations for independent geometric validation;
  historical output Dice is useful but is not ground truth.
- Add a gated local integration test that reads the external KPF data and
  enforces the acceptance criteria above in CI-capable data infrastructure.
