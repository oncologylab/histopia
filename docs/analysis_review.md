# Analysis review

The review shell organizes published evidence into five workspaces. It reuses
the original scientific viewers and does not grant approvals or reconstruction
eligibility.

| Workspace | Evidence and tools |
| --- | --- |
| Registration & 3D | Eligible observed stacks, semantic stacks and section connections, alignment, continuous volumes |
| Stain measurement | Native stain measurements, public counterstain extraction, measurement QC |
| Cells & annotation | Whole-slide and selected-section boundaries, annotations, cell features |
| Protein prediction | Predictions, section transfer, cell maps, cell atlas, performance |
| Spatial analysis | Tissue regions, neighborhoods, stability |

Data & metadata, Figures, and Review history remain available as reference views.
The metadata view preserves curated tables separately from scan-file counts.
Previous protein diagnostics are opt-in; they retain their original identities.

`build_analysis_review` consumes already gated catalogs and native viewer
manifests. It checks catalog fingerprints and explicit stage mappings before
emitting `workspace-index.json`, its script fallback, and a navigation coverage
manifest. Missing reconstruction bindings cannot become 3D viewer entries.
Public counterstain benchmarks remain separate 2D evidence. Assumed Z spacing is
labelled on the corresponding artifact; it is never promoted to measured depth.

The URL retains analysis, mode, source, organ, specimen, result, and applicable
section, protein, field, region, or inventory filters. Switching analysis opens
available evidence and identifies any specimen change. An explicit unavailable
result bookmark preserves its identity and offers alternatives. Legacy tool
bookmarks resolve to their canonical workspace, including the narrower section
scope of selected-cell reviews.

The Semantic stack mode opens the original histology, semantic-region and stain
viewer with its layer, adjacent-pair and connection controls visible. It retains
the selected specimen when switching from 3D stacks. Existing `view=atlas`
bookmarks and direct atlas links remain supported. The spacing slider controls
the display; section connections do not establish same-cell identity.

Continuous volume starts with all available semantic classes visible. Its class
and display controls, and Cell atlas's protein selectors, open alongside the
image on desktop. All three 3D viewers have a visible Controls button. On mobile,
the volume and cell-atlas drawers open on demand so they do not cover the image
on entry. Display controls are independent of scientific review forms.

Scientific review forms are opened on demand. Catalog notes are local drafts,
bound to the result fingerprint. Pipeline approvals use the existing native
review API and its explicit submission flow. Navigation never submits reviews.

Publishers should build the shell alongside filtered catalogs, reconcile the
coverage manifest against prior publication, inspect the staged views, and
switch both using the same release pointer. Presentation asset fingerprints
invalidate browser asset caches when the shell changes. Native viewers and
scientific artifacts retain their own fingerprints.

Focused tests cover gate preservation, complete stage mapping, distinct cell
scopes, legacy routes, proxy prefixes, direct image loading, browser history,
section restoration, mobile navigation, empty publications, and absence of
automatic writes. Static fallback links retain specimen identity when
JavaScript is disabled.
