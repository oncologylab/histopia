# Histopia

**Histology Spatial Topology for Omics Profiling and Inter-section Alignment**

[![Tests](https://github.com/oncologylab/histopia/actions/workflows/tests.yml/badge.svg)](https://github.com/oncologylab/histopia/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/histopia.svg)](https://pypi.org/project/histopia/)
[![Python](https://img.shields.io/pypi/pyversions/histopia.svg)](https://pypi.org/project/histopia/)
[![QuPath](https://img.shields.io/github/v/release/oncologylab/qupath-extension-histopia?label=QuPath)](https://github.com/oncologylab/qupath-extension-histopia/releases/latest)

Histopia is computational research software for serial-section histology and
proteomic image analysis. It provides reviewed inter-section alignment,
group-aware tissue masking, global morphology segmentation, spatial topology,
and interactive 3D reconstruction across tissue sections.

**[Launch the interactive 3D semantic atlas](https://oncologylab.github.io/histopia/)**
| **[Registration QC showcase](https://oncologylab.github.io/histopia/qc/)**
| [QuPath extension](https://github.com/oncologylab/qupath-extension-histopia/releases/latest)
| [PyPI](https://pypi.org/project/histopia/)

## Installation

```bash
pip install "histopia @ git+https://github.com/oncologylab/histopia.git@main"
```

Install only the workflow dependencies you need:

```bash
pip install "histopia[registration,wsi] @ git+https://github.com/oncologylab/histopia.git@main"
pip install "histopia[semantic] @ git+https://github.com/oncologylab/histopia.git@main"
pip install "histopia[topology] @ git+https://github.com/oncologylab/histopia.git@main"
pip install "histopia[stain] @ git+https://github.com/oncologylab/histopia.git@main"
pip install "histopia[uni2h] @ git+https://github.com/oncologylab/histopia.git@main"
pip install "histopia[qupath] @ git+https://github.com/oncologylab/histopia.git@main"
```

Exact validation environments use the checked-in constraint files described in
[dependency management](docs/dependency_management.md).

## Workflows

- **Registration:** group-aware tissue masks, reviewed section orientation and
  order, hybrid serial/reference affine alignment, QC, and resumable WSI export.
- **Semantic atlas:** globally fitted UNI2-h morphology regions, guarded slide
  correction, automatic K evaluation, and cross-section topology.
- **Semantic topology:** registered-mask tissue envelopes, confidence-core and
  exhaustive selected-K semantic surfaces, held-out reconstruction validation,
  explicit physical z spacing, and uncertainty-aware connected-volume review.
- **Stain profiling:** source-space relative optical density for H-DAB,
  Sirius Red, PAS, and Alcian Blue with guarded background correction.
- **Visualization:** interactive 3D histology, semantic, and stain stacks with
  quantitative QC, linked ROI probes, and adjacent-section correspondence
  links.
- **QuPath:** extension-launched registration and semantic jobs, fail-closed
  workflow integrity audits, compact checksummed GeoJSON regions, dynamic K
  selection, and native WSI coordinates.
  Install the [QuPath 0.7 extension release](https://github.com/oncologylab/qupath-extension-histopia/releases/latest).

Start with [registration](docs/registration.md),
[semantic atlas](docs/semantic_atlas.md),
[semantic topology](docs/topology.md),
[stain quantification](docs/stain_quantification.md), or
[QuPath integration](docs/qupath.md). Installation profiles and reproducible
constraints are documented in
[dependency management](docs/dependency_management.md).

## Compute

UNI2-h extraction supports `device = "auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`,
or `"mps"` without importing Torch at base-package import time. Inspect the
active machine before starting a long extraction:

```bash
histopia-semantic doctor --device auto
histopia-semantic doctor --device cuda:0
histopia-qupath --doctor --workflow full --device auto --require-api 2
```

Long extraction jobs can override compute settings without editing their
scientific configuration:

```bash
histopia-semantic extract --config atlas.toml --device cuda:0 \
  --batch-size 128 --patch-workers 4 --vips-threads 8
histopia-semantic fit --config atlas.toml --fit-threads 4
# Force a new fit after benchmarking or algorithm development:
histopia-semantic fit --config atlas.toml --fit-threads 4 --overwrite-fit
```

Registration exposes bounded thumbnail, mask, ordering, rigid-pair, QC, and
native libvips worker controls; semantic extraction and fitting expose device
and thread controls. Registration ordering and viewer assets use exact,
checksummed caches. Semantic fitting reuses only a fully sealed result whose
feature contents, slide order, scientific controls, algorithm revision, and
numerical runtime all match. Any change to reviewed masks, geometry,
orientation, transforms, semantic labels, or encoding settings invalidates the
affected cache.

Audit a completed cohort before publishing or consuming it downstream:

```bash
histopia-visualize audit \
  --run sample=/path/to/registration-run \
  --semantic-run sample=/path/to/semantic-run \
  --stain-run sample=/path/to/stain-run \
  --viewer-manifest /path/to/viewer/manifest.json
```

The path-free report distinguishes approved results, explicit review gates,
missing stages, and integrity failures, including family-scoped stain approval.
Stable viewer builds publish approved results only. Use
`histopia-visualize review` to generate a separate review hub containing
pending registration, 3D, semantic, and stain evidence.

## Development

```bash
git clone https://github.com/oncologylab/histopia.git
cd histopia
python -m pip install -e ".[dev,registration,semantic,wsi]"
python -m pytest
python -m ruff check .
```

Histopia remains research software. Registration and semantic results are
fingerprinted and explicitly review-gated; current validation does not establish
clinical use, cell-level correspondence, or final OME metadata conformance.

## License

The license is pending. See `LICENSE` for the current placeholder.
