# QuPath Integration

Histopia keeps GPU and Python image-analysis dependencies outside QuPath's JVM.
The companion extension makes QuPath the workflow front end: users select
slides from an open QuPath project, configure registration and semantic
analysis, launch and cancel jobs, review registration QC, and import semantic
regions. Python remains a child process so WSI, GPU, and model dependencies do
not enter QuPath's JVM.

```bash
# Full registration, UNI2-h semantic, WSI, and interchange workflows:
python -m pip install \
  "histopia[registration,wsi,uni2h,qupath] @ git+https://github.com/oncologylab/histopia.git@main"

# Lightweight interchange export only:
python -m pip install \
  "histopia[qupath] @ git+https://github.com/oncologylab/histopia.git@main"

histopia-qupath \
  --registration-run /path/to/registration-run \
  --semantic-run /path/to/semantic-run \
  --clusters 7 \
  --output /path/to/qupath-bundle

# Validate the exact environment used by the extension:
histopia-qupath --doctor --workflow full --device auto --require-api 1
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
The browser viewer and QuPath exporter share this fail-closed binding
validator, including exact section order and reference checks and the final
registration-approval digest for schema-3 preflights.
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
- registration resolution and bounded thumbnail, mask, ordering, and QC
  worker controls
- semantic device including explicit `cuda:N`, K range, batch-size,
  patch-reader, optional native libvips thread cap, bounded global-fit threads,
  and model-cache controls
- an in-panel environment check that reports the Histopia workflow API,
  dependency and libvips versions, resolved Python/Torch backend, and
  accelerator
- automatic workflow-specific preflight before registration, semantic
  analysis, and export, so an incomplete selected Python environment fails
  before analysis begins
- conservative automatic preprocessing and QC worker counts capped at four,
  with independent editable controls so faster mask preparation does not
  over-parallelize memory-heavier registration diagnostics
- unbuffered live process output, review-note redaction, and complete
  process-tree cancellation with bounded force escalation; POSIX termination
  is translated into a graceful exit so registration and semantic stage
  telemetry records `interrupted` instead of remaining `running`
- one self-contained browser portal that opens at the mask-only preparation
  stage, then adds section order and registered-stack QC when available
- stage-artifact cohort checks that hide downstream QC left by an earlier
  workspace run and reject mismatched review or approval actions
- separate fingerprint-bound mask and order approvals, followed by final
  sealing of the registered result
- direct semantic execution from the approved registration workspace
- local semantic, blend, K-sensitivity, and topology review followed by
  semantic approval bound to the exact current registration result and seal

The extension writes runtime-only configs and an exact slide-selection
manifest under `<workspace>/.histopia`. **Open registration QC** generates the
review portal there and opens its local `index.html`; it does not start a
server or make external requests. **Open semantic QC** starts an
ephemeral loopback-only server on `127.0.0.1` so the WebGL viewer's modules and
assets load correctly; it is replaced the next time semantic QC is opened.
Reopening the tools refreshes a newly opened project while preserving the
selected cohort and reference for the same project. Missing or malformed local
project entries are reported and skipped without preventing usable WSI entries
from loading. Controls are locked while a child process is active. Semantic
launch verifies the existing selection manifest and registration seal before
atomically updating only the semantic runtime config, leaving reviewed
registration provenance untouched.
Reusing a workspace with a changed project selection prunes the current mask
manifest to the selected cohort. Downstream order and alignment artifacts from
an earlier run are not offered for review, approval, or semantic analysis
unless their exact slide cohort matches the current staged workflow.

The generated JSON/TOML contracts can also be validated without loading
OpenCV, libvips, PyTorch, or other workflow dependencies:

```python
from histopia.registration import load_registration_config
from histopia.semantic import load_semantic_config

registration = load_registration_config("registration-config.json")
semantic = load_semantic_config("semantic-config.json")
```

The extension's build executes these public loaders against its generated
configs using a pinned Histopia revision. This cross-project check catches
configuration drift independently of the Java command-construction tests and
the Python workflow tests.

The QuPath **Device** control selects the backend for UNI2-h feature extraction.
The global semantic atlas then uses the validated CPU implementation;
**Fit threads** bounds its native BLAS and OpenMP pools. An idle GPU during the
fit stage is therefore expected and does not indicate a fallback or failure.

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
   registration cohort. Every review and approval action is bound to both the
   prepared selection manifest and the actual current stage artifacts.
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
environment independently with:

```bash
histopia-qupath --doctor --workflow full --device auto --require-api 1
```

Source code and release history are maintained separately at
[`oncologylab/qupath-extension-histopia`](https://github.com/oncologylab/qupath-extension-histopia).
This follows QuPath's recommended extension layout and keeps its Java/Gradle
licensing and release lifecycle separate from the Python package.

QuPath documents GeoJSON as its preferred annotation interchange format:
[Exporting annotations](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html).
