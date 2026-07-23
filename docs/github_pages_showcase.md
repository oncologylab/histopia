# Static GitHub Pages Showcase

Histopia can export one fingerprint-approved viewer mouse as a static site:

```bash
histopia-visualize showcase \
    /path/to/generated/viewer/histopia \
    /path/to/new/showcase \
    --mouse sample
```

The exporter copies only the selected mouse, rejects local absolute paths,
refuses unapproved semantic results, and writes:

- the browser entry point, JavaScript, CSS, and selected static textures;
- a single-mouse `manifest.json`;
- `.nojekyll` for static hosting; and
- `showcase.json`, which records the semantic fingerprint and SHA-256 digest
  of every deployed file.

Generated textures and manifests are not tracked in the source repository. The
approved showcase is packaged as a versioned GitHub Release asset. The Pages
workflow downloads the exact release URL, verifies the archive SHA-256, checks
that it contains no symbolic links, and deploys it through the GitHub Pages
artifact workflow.

To publish a new exact atlas:

1. Export and browser-test the approved mouse.
2. Create a deterministic compressed archive of the exported directory.
3. Upload it under a new `pages-demo-<mouse>-v<version>` release tag.
4. Update the archive name, URL, and SHA-256 in
   `.github/workflows/pages.yml`.
5. Run the test suite and deploy the workflow from the default branch.

The repository README links to the stable Pages URL. A GitHub README cannot
embed executable JavaScript, so the interactive WebGL application is hosted by
GitHub Pages rather than embedded directly in README content.
