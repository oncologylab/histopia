# Cellular Protein Atlas

The cellular protein atlas packages registered cell geometry and sealed
protein-prediction artifacts as a self-contained interactive website. It is a
visualization and delivery layer: it does not refit a model, recompute stain
quantification, or convert predicted values into measurements.

## Scientific Contract

Each displayed body represents one accepted native-resolution cell instance.
Its section-plane footprint is an angular vector sampled directly from the
exterior of that cell's accepted label mask: 16 vertices for on-demand detail
and 8 vertices for the bounded whole-volume overview. This is a compact
boundary approximation, not an ellipse reconstructed from area or
eccentricity. Cell identity and protein expression use the same geometry;
only the per-cell color changes.

The atlas deliberately preserves three distinct scopes:

- X/Y are accepted native cell boundaries mapped through the exact sealed
  section-registration transform into reference physical coordinates.
- Z is the supplied section geometry. If physical gap calibration fails, the
  atlas labels Z as uniformly assumed and offers explicitly labelled display
  exaggerations without changing any measurement.
- A rounded body is inferred for 3D display from each detected XY footprint.
  Its full depth is capped at the nominal section thickness. It is not a
  cross-section cell track or a measured Z boundary. Section and top views
  retain the detected XY footprint.
- Observed target OD is the adaptive corrected tissue-masked map sampled at
  4 micrometres per pixel and aggregated over each cell. The body does not
  imply a native-resolution or per-pixel protein measurement.

One leave-one-mouse-out model is selected for each antibody and applied across
the complete volume. A training-visible reconstruction is never substituted
on measured sections. This keeps one prediction protocol across every section
and prevents a visual discontinuity from being mistaken for biology. Observed
and residual views are enabled only where the selected target has measurement
support. Residual is absolute per-cell error in the target's display scale.

Internal model-promotion, calibration, and diagnostic states remain in the
path-free manifest for provenance and downstream audit, but are deliberately
absent from the presentation interface. The marker selector and active-color
legend show protein names only. Presets are named for a specific biological
comparison and list their constituent markers directly. Multi-antibody colors
are normalized within antibody; color intensity must not be compared as
absolute abundance between antibodies.

The presentation opens on the protein layer with dominant-marker composition.
Predicted and observed protein views share the same atlas-global,
within-antibody display transform. A fixed 25% post-normalization focus cutoff
suppresses weak diffuse signal, making regional patterns easier to compare
without changing stored predictions or observed OD values. The cutoff is
encoded in the URL and remains adjustable from 0% to 50%.

## Browser Architecture

The application uses locally vendored Three.js and OrbitControls. Rendering,
camera movement, orthogonal projections, section isolation, cutaways, protein
compositing, and value thresholds all run in the client browser. A Web Worker
loads and composites quantized protein arrays without blocking interaction.
No CDN, tile API, model server, or Python process is required after export.

The `HCPA1` kind-2 geometry format stores quantized boundary radii, within-bin
angles, native centroids, and the exact native-to-reference affine transform.
The decoder retains read compatibility with legacy kind-1 morphology assets,
but new builds emit boundary-derived kind-2 geometry only. A
bounded deterministic overview provides responsive whole-volume rendering;
the complete cell set for the selected section, or its immediate neighbours,
is loaded on demand. Browser caching reuses immutable assets across camera and
display changes. A static fingerprinted preview remains visible when WebGL is
unavailable or the context cannot be restored.

The generated manifest contains no source paths. Every file is inventoried by
SHA-256, symbolic links are rejected during showcase packaging, and an atomic
directory publish prevents a partial atlas from replacing the previous build.
The default atlas budget is 650 MiB and the containing showcase budget is
900 MiB.

## Build

```bash
histopia-visualize protein-atlas /path/to/new/atlas \
    --run sample-a=/path/to/registration-a \
    --cell-run sample-a=/path/to/cells-a \
    --cell-geometry sample-a=/path/to/geometry-cache-a \
    --protein-model sample-a:yap=/path/to/yap-transfer-run \
    --topology-run sample-a=/path/to/topology-a \
    --workers 2
```

Repeat every scoped input for each cohort and antibody. The builder validates
the exact registration digest, cell-result fingerprint, label-artifact digest,
content box, section transform, section and label order, cell count, and
geometry provenance before publishing. A deterministic sample must pass the
boundary overlap, area, and centroid quality gates. Geometry assets can be
reused only when their source fingerprint and file digest still match;
protein-model changes therefore do not require re-extracting unchanged cell
boundaries.

To include the result in a static showcase:

```bash
histopia-visualize showcase \
    /path/to/generated/viewer \
    /path/to/new/showcase \
    --mouse sample-a \
    --protein-atlas /path/to/new/atlas
```

The atlas can also be hosted directly from any static HTTP server. Query
parameters preserve the mouse, section, protein channels, 3D/orthogonal/section
mode, expression source, composition rule, and labelled Z display scale.

## Review Checklist

- Confirm the whole volume appears without an initial click and switching mice
  does not retain stale geometry or protein values.
- Inspect the full-detail section view to confirm irregular and elongated
  footprints follow accepted cell boundaries rather than repeated ellipses,
  patches, or tile boundaries.
- Confirm the Cell identity and Proteins layers retain identical footprints,
  and that selection outlines follow the selected boundary vector.
- Compare predicted, observed, and residual views only on measured target
  sections; verify those controls are disabled elsewhere.
- Confirm model-promotion and diagnostic status words do not appear in the
  presentation controls, legend, preset names, tooltips, or URL.
- Confirm every preset names a concrete biological comparison and lists its
  constituent markers.
- Check section alignment in top, front, side, and orthogonal views and retain
  the assumed-Z warning when calibration is unavailable.
- Test WebGL context loss, the static fallback, mobile controls, 1080p, and 4K
  layouts without page overflow.
