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

The exporter validates the complete semantic result, requires exact semantic
approval, binds the atlas preflight to the selected registration-result
SHA-256, and reconstructs native tile bounds from each result-sealed label
grid, calibrated patch scale, and preflight-bound registration content
geometry. Mutable compact feature files are not trusted during export. It uses
the same rounded native-pixel patch dimensions as feature extraction.
Selecting an unavailable K, pairing a different registration, missing a slide,
changing grid rows, or using uncalibrated geometry fails before a new manifest
is presented as complete.
Fingerprinted annotation directories keep an older manifest internally
consistent while a new export is being written.

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
- semantic device including explicit `cuda:N`, K range, batch-size,
  patch-reader, optional native libvips thread cap, and model-cache controls
- an in-panel compute check that reports the resolved Python/Torch backend and
  accelerator before extraction
- a conservative automatic registration worker count capped at four, with an
  editable override for measured host-specific tuning
- live process output, review-note redaction, and complete process-tree
  cancellation with bounded force escalation
- one self-contained browser portal that opens at the mask-only preparation
  stage, then adds section order and registered-stack QC when available
- separate fingerprint-bound mask and order approvals, followed by final
  sealing of the registered result
- direct semantic execution from the approved registration workspace
- local semantic, blend, K-sensitivity, and topology review followed by
  fingerprint-bound semantic approval

The extension writes runtime-only configs and an exact slide-selection
manifest under `<workspace>/.histopia`. **Open registration QC** generates the
review portal there and opens its local `index.html`; it does not start a
server or make external requests. **Open semantic QC** starts an
ephemeral loopback-only server on `127.0.0.1` so the WebGL viewer's modules and
assets load correctly; it is replaced the next time semantic QC is opened.
Selected slides may come from different directories, but each must have a
unique filename and a single local NDPI, SCN, SVS, TIFF, or OME-TIFF source
URI.

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
   It also requires the current QuPath slide selection to equal the sealed
   registration cohort. Every approval action is similarly bound to the
   prepared selection manifest.
7. Choose **Open semantic QC** and review histology, blend, semantic K choices,
   batch diagnostics, and adjacent-section topology.
8. Enter review metadata and choose **Approve semantic**. Approval revalidates
   the complete semantic artifact seal and records the exact fingerprint.
9. Export the approved atlas and import its regions into matching open slides.

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
- exporting approval-bound schema-4 bundles while retaining schema-3
  compatibility, and importing the matching open slide
- optionally replacing existing Histopia annotations rather than duplicating
  them

Schema-3 exports bind the semantic approval, semantic preflight, and exact
registration-result SHA-256. New approval-bound workflows emit schema 4, which
also records the final registration-approval SHA-256, matching registration
result digest, order fingerprint, reviewer, and timestamp. Registration-only
exports require this final seal. The extension verifies these schema-4 fields,
annotation checksums and byte sizes, rejects paths or symlinks outside the
bundle, and retains import compatibility with schema 1-3. It invokes the
Python package as a child process; GPU, WSI, and model dependencies remain in
the Python environment rather than QuPath's JVM. Configure and test that
environment independently with `histopia-semantic doctor`.

Source code and release history are maintained separately at
[`oncologylab/qupath-extension-histopia`](https://github.com/oncologylab/qupath-extension-histopia).
This follows QuPath's recommended extension layout and keeps its Java/Gradle
licensing and release lifecycle separate from the Python package.

QuPath documents GeoJSON as its preferred annotation interchange format:
[Exporting annotations](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html).
