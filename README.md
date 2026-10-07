# Histopia

**Histology Spatial Topology for Omics Profiling and Inter-section Alignment**

[![Tests](https://github.com/oncologylab/histopia/actions/workflows/tests.yml/badge.svg)](https://github.com/oncologylab/histopia/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/histopia.svg)](https://pypi.org/project/histopia/)

Histopia provides reusable tools for serial-section alignment, cell segmentation,
stain quantification, protein prediction, and spatial tissue reconstruction.

[Documentation](docs/README.md) · [Interactive demo](https://oncologylab.github.io/histopia/) · [QuPath extension](https://github.com/oncologylab/qupath-extension-histopia/releases/latest)

## Install

The base package is lightweight. Install dependencies for the workflows you need:

```bash
pip install "histopia[registration,wsi]"
```

For the latest source version:

```bash
git clone https://github.com/oncologylab/histopia.git
cd histopia
pip install -e ".[registration,wsi]"
```

Other optional profiles include `stain`, `cells`, `semantic`, `uni2h`, `protein`,
and `topology`. See [installation details](docs/dependency_management.md).

## Workflows

| Workflow | Guide |
| --- | --- |
| Tissue masking and serial-section alignment | [Registration](docs/registration.md) |
| Chromogenic stain quantification | [Stain quantification](docs/stain_quantification.md) |
| Cell boundaries and segmentation merging | [Cell segmentation](docs/cell_segmentation.md) |
| Histology embeddings and tissue regions | [Semantic atlas](docs/semantic_atlas.md) |
| Protein-expression models | [Protein prediction](docs/protein_prediction.md) |
| Spatial topology and reconstruction | [Topology](docs/topology.md) |
| Per-cell protein visualization | [Cellular protein atlas](docs/cellular_protein_atlas.md) |

Start with the [example configurations](examples/README.md). Each command supports
`--help`, for example `histopia-register --help`.

## Development

```bash
pip install -e ".[dev,registration,semantic,topology,stain,wsi]"
ruff check .
pytest
```

Reusable code lives in `src/histopia`, with small fixtures and tests in `tests`.
Datasets, checkpoints, generated reports, and local research operations stay
outside version control. Heavy image and model dependencies are loaded only
when needed.

Histopia is research software under active development. Registration,
prediction, and reconstructed or interpolated tissue require task-specific
validation; a displayed result does not establish biological accuracy.

## License

[BSD 3-Clause](LICENSE).
