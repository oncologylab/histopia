# Pathology Features and Annotations

Histopia provides three small, composable additions for pathology-guided
analysis of a registered semantic atlas:

- a supervised linear probe over frozen patch embeddings;
- interpretable 2D, adjacent-section, and 3D spatial feature spectra; and
- revisioned native-slide polygon annotations in the scientific review app.

Install the CPU analysis profile:

```bash
pip install "histopia[semantic,topology]"
```

## Supervised Semantic Probe

`fit_semantic_probe` L2-normalizes embeddings, fits PCA on training data only,
then fits a regularized logistic-regression probe. A validation split is used
only for scalar temperature calibration. The saved NPZ model contains NumPy
arrays, class names, configuration, diagnostics, and a SHA-256 fingerprint; it
does not pickle a scikit-learn estimator.

```python
from histopia.semantic import SemanticProbeConfig, fit_semantic_probe

fit = fit_semantic_probe(
    train_features,
    train_labels,
    validation_features=validation_features,
    validation_labels=validation_labels,
    config=SemanticProbeConfig(pca_components=64, regularization_c=1.0),
)
fit.model.save("semantic_probe.npz")
probabilities = fit.model.predict_proba(test_features)
```

Splits must be made at the animal or patient level before calling the probe.
Patch-level random splitting leaks slide and subject identity and is not a
valid estimate of generalization.

Histopia can import STAMP-compatible HDF5 feature files containing `feats` and
`coords` datasets. Pixel-to-reference geometry is intentionally explicit:

```python
from histopia.semantic import load_stamp_features

table = load_stamp_features("STAMP_features.h5")
patches = table.to_patch_features(
    slide_id="section-001",
    native_patch_size_px=224,
    native_mpp=0.5,
    native_to_reference_um=transform_points,
)
```

## Spatial Feature Spectrum

`extract_spatial_feature_spectrum` accepts semantic label images using `-1`
for background. It reports physical-area composition, local entropy, connected
region graphs, class-pair interactions, adjacent-section continuity, and
volumetric connectivity/surface features. Physical x/y spacing and z positions
are required inputs rather than inferred from array indices.

```python
from histopia.topology import extract_spatial_feature_spectrum

spectrum = extract_spatial_feature_spectrum(
    label_stack,
    spacing_um_xy=(112.0, 112.0),
    z_positions_um=z_positions_um,
    class_names=("epithelial", "stroma", "immune"),
)
rows = spectrum.rows()
```

These measurements describe the supplied semantic fields. They do not turn
interpolated z samples into observed histology and do not establish clinical
validity.

## Pathology Annotations

Annotations are GeoJSON Polygon or MultiPolygon features in
`source_native_px` coordinates. Every revision is bound to the exact bytes of
`registration_result.json` and the validated semantic-result fingerprint.
Current collections live under `sections/`; immutable revisions live under
`history/`. Stale browser revisions and stale workflow bindings fail closed.

Add an annotation directory to the local review registry:

```json
{
  "schema_version": 1,
  "cohorts": {
    "sample": {
      "registration": "/path/to/registration-run",
      "semantic": "/path/to/semantic-run",
      "annotations": "/path/to/annotation-store"
    }
  }
}
```

Generate the stable review hub with an annotation tab:

```bash
histopia-visualize review /path/to/site/review \
  --run sample=/path/to/registration-run \
  --semantic-run sample=/path/to/semantic-run \
  --annotation-run sample=/path/to/annotation-store
```

Serve with the existing review configuration. The editor uses full-resolution
OpenSeadragon tiles, adjacent-section context, class/confidence controls,
freehand polygons, undo/redo, and optimistic revision saves. Editing is
restricted to the raw layer because annotation coordinates are native pixels.
The registered layer is comparison-only until a validated coordinate transform
is exposed to the editor.

## Method Provenance

The supervised-probe and spatial-spectrum work was informed by the published
[PathPrism Cancer Cell article](https://www.sciencedirect.com/science/article/pii/S153561082600259X)
and its [research repository](https://github.com/KatherLab/PathPrism). Histopia
implements these general scientific ideas independently against its own serial
section data model and APIs. No PathPrism source was copied or translated into
Histopia. PathPrism is distributed under
[CC BY-NC 4.0](https://github.com/KatherLab/PathPrism/blob/main/LICENSE), which
is not vendored into Histopia's BSD-licensed package.

Outcome modeling, survival prediction, MSI prediction, treatment-response
models, LLM interpretation, and synthetic virtual slides are not part of this
module. They require appropriate cohort labels, leakage-safe evaluation, and
separate scientific validation.
