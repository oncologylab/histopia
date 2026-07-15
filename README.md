# Histopia

**Histology Spatial Topology for Omics Profiling and Inter-section Alignment**

Histopia is a computational research-software package for serial-section
histology and proteomic image analysis. The long-term goal is to support
inter-section image alignment, spatial topology reconstruction, protein and
marker intensity profiling, and later 3D reconstruction across tissue sections.

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

## License

The license is pending. See `LICENSE` for the current placeholder.
