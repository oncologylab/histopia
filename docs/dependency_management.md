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

Local development:

```bash
python -m pip install -e ".[dev]"
```

Registration algorithms on standard images:

```bash
python -m pip install -e ".[registration]"
```

Whole-slide registration development:

```bash
python -m pip install -e ".[registration,wsi]"
```

Semantic atlas fitting from existing compact features:

```bash
python -m pip install -e ".[semantic]"
```

UNI2-h extraction from source whole-slide images:

```bash
python -m pip install -e ".[uni2h]" \
    -c constraints/semantic-repro.txt
```

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
```

## Reproducibility Policy

- Keep runtime dependencies in optional extras unless needed at import time.
- Use lower and upper bounds for normal workflow extras.
- Keep exact `*-repro` extras synchronized with their checked-in constraint
  files. The test suite rejects version drift between those two interfaces.
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
