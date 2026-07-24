# QuPath Integration

Histopia keeps GPU and Python image-analysis dependencies outside QuPath's JVM.
The companion extension makes QuPath the workflow front end: users select
slides from an open QuPath project, configure registration and semantic
analysis, launch and cancel jobs, review registration QC, and import semantic
regions. Python remains a child process so WSI, GPU, and model dependencies do
not enter QuPath's JVM.

```bash
pip install "histopia[qupath]"

histopia-qupath \
  --registration-run /path/to/registration-run \
  --semantic-run /path/to/semantic-run \
  --clusters 7 \
  --output /path/to/qupath-bundle
```

`--semantic-geometry regions` is the default. It losslessly coalesces
horizontally and vertically adjacent patches of the same class into maximal
rectangles. This preserves source-pixel coverage while reducing GeoJSON size
and QuPath geometry overhead. Use `--semantic-geometry tiles` to retain one
rectangle per source patch for low-level audits.

The bundle contains:

- `histopia-qupath.json`, with source image URIs, section order, registration
  QC, thumbnail-coordinate transforms, geometry, and result fingerprints
- one full-fingerprint and geometry-version-scoped GeoJSON file per source
  slide when a semantic run is supplied
- semantic regions classified and colored consistently for the selected K
- SHA-256, byte size, class count, coalesced-region count, and source-patch
  count for each annotation artifact

Semantic annotations use original source-WSI pixel coordinates and can be
imported directly into the matching QuPath image. Registration matrices remain
explicitly labeled as moving-thumbnail to reference-thumbnail transforms; they
must not be applied as native-pixel transforms.

The exporter validates the complete semantic result and requires exact equality
between each label grid and its corresponding extracted source grid. It uses
the same rounded native-pixel patch dimensions as feature extraction. Selecting
an unavailable K, missing a slide, changing grid rows, or using uncalibrated
geometry fails before a new manifest is presented as complete. Fingerprinted
annotation directories keep an older manifest internally consistent while a
new export is being written.

## QuPath Extension

Download the
[latest companion extension release](https://github.com/oncologylab/qupath-extension-histopia/releases/latest),
verify the accompanying SHA-256 checksum, and drag the JAR onto QuPath 0.7.
Restart QuPath, then open
**Extensions > Histopia > Open Histopia tools**.

The primary **Project workflow** tab supports:

- exact multi-selection from local WSI entries in the open QuPath project
- QuPath project order, morphology-only sorting, or morphology sorting with a
  selected reference fixed at position 1
- automatic or explicit registration reference selection
- registration resolution and worker controls
- semantic device, K range, batch-size, patch-reader, and model-cache controls
- live process output and descendant-process cancellation
- one self-contained browser portal that opens at the mask-only preparation
  stage, then adds section order and registered-stack QC when available
- separate fingerprint-bound mask and order approvals, followed by final
  sealing of the registered result
- direct semantic execution from the approved registration workspace

The extension writes runtime-only configs and an exact slide-selection
manifest under `<workspace>/.histopia`. **Open registration QC** generates the
review portal there and opens its local `index.html`; it does not start a
server or make external requests. Selected slides may come from different
directories, but each must have a unique filename and a single local NDPI,
SCN, SVS, TIFF, or OME-TIFF source URI.

The project workflow is deliberately staged:

1. **Run registration** prepares masks and stops at the mask gate.
2. Review the mask tab, enter reviewer metadata, and choose **Approve masks**.
3. **Run registration** prepares morphology-aware order and stops at the order
   gate.
4. Review both tabs and choose **Approve order**.
5. **Run registration** computes alignment; review it and choose **Seal
   reviewed run**.
6. Run the semantic atlas. The extension refuses semantic execution before the
   registration seal exists and its artifact checksums, order fingerprint,
   slide count, embedded mask statuses, reviewer, and timestamp still match.

The same button is used for each computational stage because preprocessing and
pairwise-distance caches make unchanged work resumable. Review-required stages
are reported as successful structured statuses rather than failed processes.
Repeated **Open registration QC** calls also reuse exact mask, order, and
registered-stack assets after validating both their input fingerprints and
stored output checksums.

The **Run analysis** tab retains advanced config-file execution. The **Export
and import** tab supports:

- loading all available K values from a semantic result, defaulting to the
  atlas-selected K
- exporting the schema-2 bundle and importing the matching open slide
- optionally replacing existing Histopia annotations rather than duplicating
  them

The extension verifies each schema-2 GeoJSON checksum before import. It invokes
the Python package as a child process; GPU, WSI, and model dependencies remain
in the Python environment rather than QuPath's JVM. Configure and test that
environment independently with `histopia-semantic doctor`.

Source code and release history are maintained separately at
[`oncologylab/qupath-extension-histopia`](https://github.com/oncologylab/qupath-extension-histopia).
This follows QuPath's recommended extension layout and keeps its Java/Gradle
licensing and release lifecycle separate from the Python package.

QuPath documents GeoJSON as its preferred annotation interchange format:
[Exporting annotations](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html).
