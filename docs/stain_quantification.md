# Quantitative Brightfield Stain Profiling

Histopia measures continuous chromogen deposition in the source space of each
accepted registered section. The primary quantity is relative optical density
(OD), calculated with the natural logarithm after estimating a slide-specific
glass white reference. Registration transforms are used to link and display
the maps; registered resampling is not used to create the measurement.

This workflow does not provide absolute concentration, cross-antibody
normalization, cell-level expression, or clinical interpretation. Comparing
the same marker across controlled batches requires experimental controls and a
study-specific normalization model beyond this workflow.

## Supported Assays

- Hematoxylin-DAB immunohistochemistry
- Sirius Red
- PAS
- Alcian Blue
- H&E as non-quantified morphological context

Known special-stain names are inferred conservatively. Other brightfield
markers use `default_family`, normally `h-dab`. Use an exact assay manifest
when filenames are ambiguous or when acquisition batches must be recorded.

## Method

At the configured physical resolution, Histopia:

1. Validates source WSI, accepted tissue mask, physical geometry, transform,
   slide order, and registration-result identity.
2. Estimates robust glass color and a low-order illumination field outside the
   accepted tissue mask.
3. Benchmarks fixed, legacy, Macenko, and nonnegative matrix-factorization
   stain vectors within each assay family.
4. Selects one family method from reconstruction error, glass leakage, prior
   drift, and bootstrap stability. Adaptive candidates are eligible only when
   their main and bootstrap optimizations converge for every family slide and
   their target signal preserves the fixed-baseline pixel ranking on at least
   90% of family slides. Selected slide vectors are then shrunk toward a robust
   cohort template.
5. Proposes background correction but accepts it only when rank correlation is
   at least the configured guard and glass leakage does not worsen.
6. Writes continuous raw target OD, corrected target OD, counterstain,
   reconstruction residual, confidence, and tissue support maps.

The raw map uses the versioned fixed vector on the uncorrected source image.
The corrected map uses the selected, cohort-shrunk vector and illumination
correction only when the rank, glass-leakage, and background-spatial-variation
guards all pass. If any guard fails, the corrected map is the raw map. This
makes the raw/corrected comparison a conservative audit of the complete
nuisance-correction proposal.

Automatic positivity is secondary. Otsu, a robust low-mode estimate, and a
two-component mixture must agree and show sufficient separation. If that gate
fails, Histopia records `positive_threshold_unstable` and writes no positive
pixels; the continuous OD map remains available.

## Installation

```bash
python -m pip install -e ".[stain]"
histopia-stain doctor
```

`pyvips` also requires a compatible native libvips installation. For exact
validation versions:

```bash
python -m pip install -e ".[dev,stain]" \
  -c constraints/stain-repro.txt
```

## Workflow

Start with `examples/stain_config.toml`:

```bash
histopia-stain preflight --config stain.toml
histopia-stain benchmark --config stain.toml
histopia-stain run --config stain.toml
```

Runs are resumable. Fits and maps are reused only when their source, accepted
mask, transform, assay, physical resolution, methods, and scientific controls
match. `stain_result.json` seals every referenced model and map. Runtime timing
is intentionally excluded from scientific identity, so an unchanged cached
rerun has the same fingerprint. `stain_performance.json` records cache use and
elapsed time separately. Sections are processed one at a time to keep WSI
memory bounded when `workers = 1`. Increasing `workers` fits or maps independent
sections concurrently while preserving result order; peak memory scales with
that value. Parallel sections use spawned processes with one BLAS/OpenMP thread
each, avoiding nested numerical thread pools. `vips_threads` is a total native
libvips budget divided across those processes. Budget both controls against
available CPU, I/O bandwidth, and memory. Independent cohorts can also be run
as separate processes.

Each quantified section also writes a reusable `StainModel`. Its
`transform_native_tile()` method accepts exact native-resolution RGB tiles and
their source-slide bounding boxes, allowing later cell-level workflows to use
the fitted white reference, illumination field, vectors, and correction guard
without loading or recomputing the full WSI.

Review the continuous maps and QC before recording approval:

```bash
histopia-stain approve --run /path/to/stain-output \
  --reviewer "Reviewer name" \
  --review-notes "Continuous maps, residuals, and correction QC inspected."
```

Summarize several validated runs:

```bash
histopia-stain cohort-qc \
  --run sample-a=/path/to/stain-a \
  --run sample-b=/path/to/stain-b \
  --output /path/to/stain-cohort-qc.json
```

## 3D Viewer

Bind a stain result to the exact registration run during viewer generation:

```bash
histopia-visualize build /path/to/viewer-root \
  --run sample=/path/to/registration-run \
  --stain-run sample=/path/to/stain-output \
  --workers 4
```

The viewer exposes raw and corrected signal-only layers, histology overlays,
fixed per-mouse OD scales, correction and approval status, and a linked ROI
probe across visible sections. Probe values come from bounded registered grids;
full analysis arrays and source WSI are never loaded into the browser. H&E
sections remain visible as context but are excluded from quantitative probe
rows.

Color is a display encoding, not a second normalization. Use the numeric OD
summary and QC flags for interpretation. The viewer clips display and probe
grids at the largest slide-level 99th-percentile value within each mouse while
the sealed source-space maps retain their complete continuous values.

## Decision Review

Build a cohort review portal from an existing generated viewer:

```bash
histopia-visualize stain-review /path/to/viewer-root/histopia \
  /path/to/viewer-root/stain-review \
  --mouse sample-a \
  --mouse sample-b
```

The portal ranks a bounded review set from correction rejection, rank-guard
failure, increased candidate glass leakage, high corrected leakage, high
reconstruction residual, and assay-family coverage. It presents registered
histology, raw OD overlay, final output overlay, and final output OD with linked
zoom. When a correction proposal fails, the final panels are explicitly
identified as the raw fallback.

Slide checks, notes, and draft accept/hold/reject decisions are stored in the
browser under the exact stain fingerprint. They are review aids and do not
write or imply scientific approval. Export the draft as JSON, then use
`histopia-stain approve` only after the continuous-OD evidence is acceptable.
Binary positivity, cross-antibody normalization, absolute concentration, and
cell-level expression remain outside this approval scope.

Known upstream issues can be displayed from an optional JSON file:

```json
{
  "sample-a": {
    "12": "Upstream tissue support requires correction."
  }
}
```

Pass it with `--issues /path/to/issues.json`. A slide can be keyed by its
integer order encoded as a string or by its exact slide ID.
