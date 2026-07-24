# Histopia

**Histology Spatial Topology for Omics Profiling and Inter-section Alignment**

[![Tests](https://github.com/oncologylab/histopia/actions/workflows/tests.yml/badge.svg)](https://github.com/oncologylab/histopia/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/histopia.svg)](https://pypi.org/project/histopia/)
[![Python](https://img.shields.io/pypi/pyversions/histopia.svg)](https://pypi.org/project/histopia/)

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

Install only the workflow dependencies you need:

```bash
pip install "histopia[registration,wsi]"  # brightfield WSI registration
pip install "histopia[semantic]"          # atlas fitting from saved features
pip install "histopia[uni2h]"             # CPU/GPU UNI2-h extraction
pip install "histopia[qupath]"             # QuPath interchange export
```

## Workflows

- **Registration:** group-aware tissue masks, reviewed section orientation and
  order, hybrid serial/reference affine alignment, QC, and resumable WSI export.
- **Semantic atlas:** globally fitted UNI2-h morphology regions, guarded slide
  correction, automatic K evaluation, and cross-section topology.
- **Visualization:** interactive 3D histology/semantic stacks with quantitative
  QC and adjacent-section correspondence links.
- **QuPath:** validated registration manifests and semantic GeoJSON annotations
  in native WSI coordinates.

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
histopia-semantic doctor
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
