# QuPath Integration

Histopia keeps GPU and Python image-analysis dependencies outside QuPath's JVM.
The `histopia-qupath` command writes a validated interchange bundle for the
companion Histopia QuPath extension.

```bash
pip install "histopia[qupath]"

histopia-qupath \
  --registration-run /path/to/registration-run \
  --semantic-run /path/to/semantic-run \
  --clusters 7 \
  --output /path/to/qupath-bundle
```

The bundle contains:

- `histopia-qupath.json`, with source image URIs, section order, registration
  QC, thumbnail-coordinate transforms, geometry, and result fingerprints
- one GeoJSON file per source slide when a semantic run is supplied
- semantic regions classified and colored consistently for the selected K

Semantic annotations use original source-WSI pixel coordinates and can be
imported directly into the matching QuPath image. Registration matrices remain
explicitly labeled as moving-thumbnail to reference-thumbnail transforms; they
must not be applied as native-pixel transforms.

The exporter validates the complete semantic result and checks that label rows
match the corresponding extracted source coordinates. Selecting an unavailable
K, missing a slide, or using uncalibrated geometry fails before a partial
bundle is presented as complete.

The companion extension is maintained separately at
[`oncologylab/qupath-extension-histopia`](https://github.com/oncologylab/qupath-extension-histopia).
This follows QuPath's recommended extension layout and keeps its Java/Gradle
licensing and release lifecycle separate from the Python package.

QuPath documents GeoJSON as its preferred annotation interchange format:
[Exporting annotations](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html).
