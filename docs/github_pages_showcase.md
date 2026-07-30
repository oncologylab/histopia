# Static GitHub Pages Showcase

Histopia can export one or more fingerprint-approved viewer mice as a static
site:

```bash
histopia-visualize showcase \
    /path/to/generated/viewer/histopia \
    /path/to/new/showcase \
    --mouse sample-a \
    --mouse sample-b
```

An approved sparse native-resolution subset can be embedded in the same
artifact:

```bash
histopia-visualize showcase \
    /path/to/generated/viewer/histopia \
    /path/to/new/showcase \
    --mouse sample-a \
    --review-config /path/to/review-server.json \
    --wsi-section sample-a=001 \
    --wsi-section sample-a=009 \
    --wsi-section sample-a=017 \
    --max-bytes 943718400
```

The WSI exporter verifies the sealed registration and full-resolution export
fingerprints, omits all-white tiles, inventories every retained tile, and
fails before exceeding the configured byte budget. The default budget is
900 MiB. Only explicitly selected sections are embedded; other sections keep
their compact overview textures.

Repeat `--mouse` in the desired browser order. The exporter copies only the
selected mice, rejects duplicate or local absolute paths, refuses unapproved
semantic or stain results, and writes:

- the browser entry point, JavaScript, CSS, and selected static textures;
- pinned Three.js and OpenSeadragon runtimes with their licenses, so the
  viewer has no CDN or third-party runtime request;
- a selected-cohort `manifest.json`;
- `.nojekyll` for static hosting; and
- `showcase.json`, which records semantic and stain fingerprints plus the
  SHA-256 digest of every inventoried file.

The current public artifact contains 16 review-stage registration stacks
spanning 401 serial sections. Nine carry current fingerprint-bound
registration approvals; seven remain explicit review-gated demonstrations.
Seven stacks also contain fingerprint-reviewed semantic atlases created before
final registration-approval binding was introduced. They remain demonstration
artifacts rather than current schema-3 production approvals. The viewer
permits specimen switching, slide-by-slide navigation, select-all/deselect-all
visibility, histology/semantic/blended rendering, K=5 through K=15 exploration,
and adjacent-section topology links.

The `mouse` query parameter creates a shareable link to a specimen, for example
`?mouse=sample-a`. Changing the specimen updates the URL without reloading the
viewer. An unknown specimen falls back to the first available stack.

The viewer runtime is pinned to Three.js 0.170.0. Histopia verifies the
packaged runtime checksums during every build, records the version in
`build-report.json`, and includes the runtime files in the static artifact
inventory.

The native-resolution focus viewer uses locally vendored OpenSeadragon 6.0.2.
It supports pan, wheel or pinch zoom, section stepping, fit-to-tissue, mask
overlay, and registered-section comparison. Analytical overlays retain their
native patch or map resolution instead of being presented as scanner-resolution
measurements.

Rendering is demand-driven. The viewer redraws while sections load and while
the camera is moving, then stops requesting animation frames when the scene is
idle. Browser tests verify that the canvas remains populated after rendering
stops, which reduces background CPU and GPU use without changing scientific
textures. Texture and specimen transitions remain busy until a rendered frame
is presented. Browser WebGL context loss triggers a bounded repaint of the
current stack over the stable light viewport.

## Registration QC Portal

The same release contains a separate `/qc/` portal for workflow diagnostics:

```bash
histopia-visualize qc-showcase \
    /path/to/generated/viewer/histopia \
    /path/to/new/showcase/qc \
    --mouse sample-a \
    --mouse sample-b
```

The portal contains only selected review artifacts and registered histology
textures. It provides tissue-mask, orientation/order, and interactive 3D
registration views. Semantic textures, raw slides, source paths, and unrelated
specimens are excluded. Tissue-mask review is required. A legacy cohort that
predates a formal orientation/order artifact remains exportable, but that stage
is visibly disabled rather than reconstructed from incomplete provenance. The
exporter rejects missing mask reviews, unsafe texture paths, local absolute
paths, and non-empty output directories.

The current public portal covers the same 16 registration stacks and 401 serial
sections as the atlas. Tissue-mask evidence is available for all 16 cohorts;
13 cohorts also have a formal orientation/order review. For the three legacy
cohorts without that record, the orientation/order control is disabled. Nine
stacks have completed registration approval; the other seven remain review
evidence only. A visible workflow stage does not itself confer scientific
approval.

QC portals accept shareable `mouse` and `stage` query parameters, for example
`?mouse=sample-a&stage=order`. Changing the selected specimen or review stage
updates the URL without reloading the portal. Unknown specimens and unavailable
stages fall back to available review evidence.

Mask and order reviews may be exported before a cohort is promoted into the
source viewer manifest. The portal then disables 3D registration for that
cohort, preserving review-before-promotion. Use a separately generated
provisional viewer as the source when all three review stages should be
available without changing the accepted main viewer.

Generated textures and manifests are not tracked in the source repository. The
approved showcase is packaged as a versioned GitHub Release asset. The Pages
workflow downloads the exact release URL, verifies the archive SHA-256, checks
that it contains no symbolic links and both local viewer runtimes, and deploys
it through the GitHub Pages artifact workflow.

To publish a new exact cohort:

1. Export and browser-test every approved mouse in the selected cohort.
2. Create a deterministic compressed archive of the exported directory.
3. Upload it under a new `pages-demo-<cohort>-v<version>` release tag.
4. Update the archive name, URL, and SHA-256 in
   `.github/workflows/pages.yml`.
5. Run the test suite and deploy the workflow from the default branch.

The repository README links to the stable Pages URL. A GitHub README cannot
embed executable JavaScript, so the interactive WebGL application is hosted by
GitHub Pages rather than embedded directly in README content.
