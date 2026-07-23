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
- a selected-cohort `manifest.json`;
- `.nojekyll` for static hosting; and
- `showcase.json`, which records each semantic fingerprint and the SHA-256
  digest of every inventoried file.

The current public artifact contains seven approved mice and 187 serial
sections. The viewer permits specimen switching, slide-by-slide navigation,
select-all/deselect-all visibility, histology/semantic/blended rendering, K=5
through K=15 exploration, and adjacent-section topology links.

## Registration QC Portal

The same release may contain a separate `/qc/` portal for approved workflow
diagnostics:

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

Generated textures and manifests are not tracked in the source repository. The
approved showcase is packaged as a versioned GitHub Release asset. The Pages
workflow downloads the exact release URL, verifies the archive SHA-256, checks
that it contains no symbolic links, and deploys it through the GitHub Pages
artifact workflow.

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
