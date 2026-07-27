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
Integrity auditing of existing registration, semantic, and viewer artifacts is
also available from the base install because it uses JSON parsing and streamed
SHA-256 validation rather than image or model runtimes:

```bash
histopia-visualize audit --run sample=/path/to/registration-run
```

Local development:

```bash
python -m pip install -e ".[dev]"
```

Registration algorithms on standard images:

```bash
python -m pip install -e ".[registration]"
```

Registration uses NumPy, SciPy, and OpenCV on CPU. Its bounded mask, ordering,
rigid-pair, and QC worker controls plus optional process-restored OpenCV and
libvips thread caps govern throughput; there is no registration GPU selector.
Accelerator selection is reserved for workflows, such as UNI2-h extraction,
that have a validated PyTorch backend.

Whole-slide registration development:

```bash
python -m pip install -e ".[registration,wsi]"
```

Semantic atlas fitting from existing compact features:

```bash
python -m pip install -e ".[semantic]"
```

This profile fits the global atlas on CPU and does not install PyTorch. Use
`fit_threads` to bound its independent fit tasks and native BLAS/OpenMP pools.

Semantic topology reconstruction from an approved atlas:

```bash
python -m pip install -e ".[topology]" \
    -c constraints/topology-repro.txt
histopia-topology doctor
```

This CPU profile operates on compact selected-K fields and does not install
PyTorch or WSI readers.

Quantitative brightfield stain profiling:

```bash
python -m pip install -e ".[stain]"
histopia-stain doctor
```

The stain profile installs the WSI, numerical fitting, and registered-viewer
dependencies. Quantification runs on CPU at a configured physical resolution;
the optional UNI2-h workflow is not used to alter measured OD.

Use the exact tested stain runtime through its constraint file:

```bash
python -m pip install -e ".[dev,stain]" \
    -c constraints/stain-repro.txt
```

UNI2-h extraction from source whole-slide images:

```bash
python -m pip install -e ".[uni2h]" \
    -c constraints/semantic-repro.txt
```

The `uni2h` profile adds PyTorch and the WSI stack for CPU, CUDA, or Apple MPS
feature extraction. The selected accelerator does not replace the CPU atlas
fit.

Reproducible KPF validation environment:

```bash
python -m pip install -e ".[dev,registration,wsi]" \
    -c constraints/registration-repro.txt
```

On Python 3.10, the constraints also pin the conditional `tomli` parser used to
read TOML configuration. CI installs the complete reproducible CPU registration
and atlas-fitting profiles, verifies every installed version, runs the
registration and interchange QuPath doctors, and checks dependency consistency.

Full reproducible registration, topology, stain, UNI2-h, and QuPath workflow:

```bash
python -m pip install -e \
    ".[registration,semantic,topology,stain,wsi,uni2h,qupath]" \
    -c constraints/registration-repro.txt \
    -c constraints/semantic-repro.txt \
    -c constraints/topology-repro.txt \
    -c constraints/stain-repro.txt
histopia-qupath --doctor --workflow full --device auto --require-api 1
```

The QuPath doctor checks only the selected workflow's imports, validates their
installed versions against Histopia's supported ranges, and loads libvips
before the accelerator stack. It reports exact dependency and compute versions
and rejects an extension that requires a newer workflow API. Use
`--workflow registration`, `semantic`, `topology`, or `interchange` to validate
a smaller installation.

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
an incompatible native dependency. Histopia currently constrains the normal
WSI profiles to pyvips 2.x and pins 2.2.3 in reproducible profiles: a clean
Python 3.12 API-mode build of pyvips 3.1.1 crashed while initializing the
supported libvips 8.15.1 runtime, whereas 2.2.3 passed the native image and
QuPath doctor probes. Conda users should install both packages from
`conda-forge`. For a system-Python environment with an already installed
system `libvips`, rebuild the supported binding in that environment when
necessary:

```bash
python -m pip install --no-cache-dir --force-reinstall \
    --no-binary=pyvips "pyvips>=2.2,<3"
```

Run the import check above before starting a WSI workflow. A native loader
failure can terminate Python before Histopia can report a normal exception.

## Reproducibility Policy

- Keep runtime dependencies in optional extras unless needed at import time.
- Use lower and upper bounds for normal workflow extras.
- Keep exact versions in checked-in constraint files rather than duplicating
  them in package extras.
- Use `constraints/registration-repro.txt` for exact validation reruns.
- Use `constraints/semantic-repro.txt` for the tested semantic analysis and
  GPU extraction stack. Validation used Python 3.10, an NVIDIA A100, and the
  PyTorch CUDA 13.0 wheel; use the equivalent platform wheel when CUDA 13.0 is
  unavailable.
- Use `constraints/stain-repro.txt` for quantitative brightfield validation.
- Use `constraints/topology-repro.txt` for selected-K 3D reconstruction and
  surface extraction.
- The normal `uni2h` extra retains bounded ranges for portable CPU, CUDA, and
  Apple MPS installs.
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
