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

Repeat `--mouse` in the desired browser order. The exporter copies only the
selected mice, rejects duplicate or local absolute paths, refuses unapproved
semantic results, and writes:

- the browser entry point, JavaScript, CSS, and selected static textures;
- a pinned Three.js runtime and its license, so the viewer has no CDN or
  third-party runtime request;
- a selected-cohort `manifest.json`;
- `.nojekyll` for static hosting; and
- `showcase.json`, which records each semantic fingerprint and the SHA-256
  digest of every inventoried file.

The current public artifact contains 16 reviewed registration stacks spanning
401 serial sections. Seven stacks also contain fingerprint-approved semantic
atlases. The viewer permits specimen switching, slide-by-slide navigation,
select-all/deselect-all visibility, histology/semantic/blended rendering, K=5
through K=15 exploration, and adjacent-section topology links.

The viewer runtime is pinned to Three.js 0.170.0. Histopia verifies the
packaged runtime checksums during every build, records the version in
`build-report.json`, and includes the runtime files in the static artifact
inventory.

Rendering is demand-driven. The viewer redraws while sections load and while
the camera is moving, then stops requesting animation frames when the scene is
idle. Browser tests verify that the canvas remains populated after rendering
stops, which reduces background CPU and GPU use without changing scientific
textures.

## Registration QC Portal

The same release contains a separate `/qc/` portal for approved workflow
diagnostics. The current portal presents the mask/orientation, section-order,
and 3D registration evidence for four recently sealed cohorts:

```bash
histopia-visualize qc-showcase \
    /path/to/generated/viewer/histopia \
    /path/to/new/showcase/qc \
    --mouse sample-a \
    --mouse sample-b
```

The portal contains only selected review artifacts and registered histology
textures. It provides mask/orientation, section-order, and interactive 3D
registration views. Semantic textures, raw slides, source paths, and unrelated
specimens are excluded. The exporter rejects missing reviews, unsafe texture
paths, local absolute paths, and non-empty output directories.

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
