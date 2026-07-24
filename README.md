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
pip install histopia
```

Install only the workflow dependencies you need:

```bash
pip install "histopia[registration,wsi]"  # brightfield WSI registration
pip install "histopia[semantic]"          # atlas fitting from saved features
pip install "histopia[uni2h]"             # CPU/GPU UNI2-h extraction
pip install "histopia[uni2h-repro]"       # exact validated UNI2-h runtime
pip install "histopia[qupath]"            # QuPath extension bridge
```

## Workflows

- **Registration:** group-aware tissue masks, reviewed section orientation and
  order, hybrid serial/reference affine alignment, QC, and resumable WSI export.
- **Semantic atlas:** globally fitted UNI2-h morphology regions, guarded slide
  correction, automatic K evaluation, and cross-section topology.
- **Visualization:** interactive 3D histology/semantic stacks with quantitative
  QC and adjacent-section correspondence links.
- **QuPath:** extension-launched registration and semantic jobs, compact
  checksummed GeoJSON regions, dynamic K selection, and native WSI coordinates.
  Install the [QuPath 0.7 extension release](https://github.com/oncologylab/qupath-extension-histopia/releases/latest).

Start with [registration](docs/registration.md),
[semantic atlas](docs/semantic_atlas.md), or
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
```

Long extraction jobs can override compute settings without editing their
scientific configuration:

```bash
histopia-semantic extract --config atlas.toml --device cuda:0 \
  --batch-size 128 --patch-workers 4 --vips-threads 8
```

Registration ordering and viewer assets use exact, checksummed caches. Any
change to reviewed masks, geometry, orientation, transforms, semantic labels,
or encoding settings invalidates the affected cache.

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
