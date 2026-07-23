# Histopia

**Histology Spatial Topology for Omics Profiling and Inter-section Alignment**

Histopia is a computational research-software package for serial-section
histology and proteomic image analysis. The long-term goal is to support
inter-section image alignment, spatial topology reconstruction, protein and
marker intensity profiling, and later 3D reconstruction across tissue sections.

**[Launch the interactive 3D semantic atlas](https://oncologylab.github.io/histopia/)**
| **[Registration QC showcase](https://oncologylab.github.io/histopia/qc/)**
| [PyPI](https://pypi.org/project/histopia/)

## Installation

```bash
pip install histopia
```

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Current Status

Histopia is in early development. Registration now includes brightfield/IHC
tissue-mask QC, hybrid serial/reference rigid alignment, conservative affine
mask refinement, per-slide acceptance metrics, resumable full-resolution WSI
warping, and opt-in acceptance-gated dense refinement. The current KPF
validation supports coarse section placement, QC, and pyramidal registered TIFF
export. OME metadata and cell-level correspondence remain under development.

## Registration Development

The registration module is intentionally small and dependency-light at import
time. Install optional dependencies for active registration work:

```bash
python -m pip install -e ".[dev,registration,wsi]"
```

For reproducible local validation, use the pinned constraints file:

```bash
python -m pip install -e ".[dev,registration,wsi]" \
    -c constraints/registration-repro.txt
```

Build a registration manifest without modifying source data:

```bash
histopia-register --manifest /path/to/registration-dataset
```

See `docs/registration.md` for the current API and validation workflow, and
`docs/dependency_management.md` for install profiles.
Current KPF validation notes are in `docs/kpf_registration_validation.md`.

## Serial-Section Semantic Atlas

Histopia can extract compact UNI2-h patch features from unregistered source
slides, map them through accepted registration transforms, and fit one global
semantic atlas across the section stack. Install the light analysis layer with
`.[semantic]` or the GPU extraction stack with `.[uni2h]`.

Model weights remain external and subject to their own access and license
terms. See `docs/semantic_atlas.md` for the staged, review-gated workflow.

## Visualization

Interactive review generation and serving are available from the canonical
`histopia.visualization` module and the `histopia-visualize` command. A viewer
root always exposes its current build at the stable `/histopia/` endpoint:

```bash
histopia-visualize build /path/to/viewer-root \
    --run sample=/path/to/registration-run \
    --semantic-run sample=/path/to/semantic-run
histopia-visualize serve /path/to/viewer-root --port 8765
```

The semantic topology overlay displays at most 500 high-confidence links for
one adjacent pair. Complete correspondence arrays remain in semantic result
artifacts.

The public showcase contains seven fingerprint-approved mouse atlases spanning
187 serial sections. It supports specimen switching, slide-by-slide navigation,
select-all/deselect-all visibility, histology, semantic, and blended textures,
selectable K values, and adjacent-section topology links. See
`docs/github_pages_showcase.md` for the artifact and deployment model.

A separate static registration QC portal presents reviewed tissue masks and
orientation, H&E-anchored section order, and an interactive histology-only 3D
registration stack without publishing raw slides or local source paths.

## License

The license is pending. See `LICENSE` for the current placeholder.
