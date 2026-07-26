# Dependency Management

Histopia keeps the base package lightweight. Heavy scientific and whole-slide
image dependencies are optional so collaborators can install only the workflows
they need.

## Install Profiles

Base package:

```bash
python -m pip install \
    "histopia @ git+https://github.com/oncologylab/histopia.git@main"
```

The base install keeps workflow namespaces lazy. Package imports and every
command's `--help` remain available without NumPy or a workflow extra; executing
scientific work still requires the matching profile below.

Local development:

```bash
python -m pip install -e ".[dev]"
```

Registration algorithms on standard images:

```bash
python -m pip install -e ".[registration]"
```

Registration uses NumPy, SciPy, and OpenCV on CPU. Its bounded mask, ordering,
thumbnail, and QC worker controls plus the optional libvips thread cap govern
throughput; there is no registration GPU selector. Accelerator selection is
reserved for workflows, such as UNI2-h extraction, that have a validated
PyTorch backend.

Whole-slide registration development:

```bash
python -m pip install -e ".[registration,wsi]"
```

Semantic atlas fitting from existing compact features:

```bash
python -m pip install -e ".[semantic]"
```

This profile fits the global atlas on CPU and does not install PyTorch. Use
`fit_threads` to bound its native BLAS and OpenMP pools.

UNI2-h extraction from source whole-slide images:

```bash
python -m pip install -e ".[uni2h]" \
    -c constraints/semantic-repro.txt
```

The `uni2h` profile adds PyTorch and the WSI stack for CPU, CUDA, or Apple MPS
feature extraction. The selected accelerator does not replace the CPU atlas
fit.

The exact tested UNI2-h runtime is also available directly:

```bash
python -m pip install -e ".[uni2h-repro]"
```

Reproducible KPF validation environment:

```bash
python -m pip install -e ".[dev,registration,wsi]" \
    -c constraints/registration-repro.txt
```

The `registration-repro` extra pins the same package versions directly:

```bash
python -m pip install -e ".[dev,registration-repro]"
```

On Python 3.10, the exact profiles also pin the conditional `tomli` parser
used to read TOML configuration. CI installs the complete reproducible CPU
registration and atlas-fitting profiles, verifies every installed version,
runs the registration and interchange QuPath doctors, and checks dependency
consistency.

Full reproducible registration, WSI, UNI2-h, and QuPath workflow:

```bash
python -m pip install -e ".[registration-repro,uni2h-repro,qupath]"
histopia-qupath --doctor --workflow full --device auto --require-api 1
```

The QuPath doctor checks only the selected workflow's imports. It loads
libvips before the accelerator stack, reports exact dependency and compute
versions, and rejects an extension that requires a newer workflow API. Use
`--workflow registration`, `semantic`, or `interchange` to validate a smaller
installation.

## System Dependencies

`pyvips` requires the native `libvips` library. The local validation environment
used:

```text
libvips 8.15.1
```

On Ubuntu-like systems, install it with:

```bash
sudo apt-get install libvips libvips-tools
```

Confirm availability with:

```bash
vips --version
python -c "import pyvips; print(pyvips.version(0), pyvips.version(1), pyvips.version(2))"
```

Install `pyvips` and `libvips` from one coherent environment. In particular,
do not reuse a locally built `pyvips` wheel across Conda and system Python
environments: such a wheel can retain an absolute runtime library path and load
an incompatible native dependency. Conda users should install both packages
from `conda-forge`. For a system-Python environment with an already installed
system `libvips`, rebuild the binding in that environment when necessary:

```bash
python -m pip install --no-cache-dir --force-reinstall \
    --no-binary=pyvips "pyvips>=2.2,<4"
```

Run the import check above before starting a WSI workflow. A native loader
failure can terminate Python before Histopia can report a normal exception.

## Reproducibility Policy

- Keep runtime dependencies in optional extras unless needed at import time.
- Use lower and upper bounds for normal workflow extras.
- Keep exact `*-repro` extras synchronized with their checked-in constraint
  files, including conditional Python dependencies. The test suite rejects
  version drift between those two interfaces.
- Use `constraints/registration-repro.txt` for exact validation reruns.
- Use `constraints/semantic-repro.txt` for the tested semantic analysis and
  GPU extraction stack. Validation used Python 3.10, an NVIDIA A100, and the
  PyTorch CUDA 13.0 wheel; use the equivalent platform wheel when CUDA 13.0 is
  unavailable.
- The `uni2h-repro` extra mirrors that constraint file. The normal `uni2h`
  extra retains bounded ranges for portable CPU, CUDA, and Apple MPS installs.
- Do not commit virtual environments, raw slides, generated masks, warped
  images, or registration output directories.
- Record `histopia-register` config files and `registration_result.json` files
  with validation reports, but keep large image artifacts outside Git.

## Package Release

Version tags beginning with `v` use PyPI trusted publishing from
`.github/workflows/publish.yml`. The PyPI project must authorize the
`oncologylab/histopia` repository, that workflow filename, and the `pypi`
GitHub environment once. No API token is stored in the repository or GitHub
Actions configuration. Showcase tags do not run the package-publishing job.
