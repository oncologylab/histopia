# Dependency Management

Histopia keeps the base package lightweight. Heavy scientific and whole-slide
image dependencies are optional so collaborators can install only the workflows
they need.

## Install Profiles

Base package:

```bash
python -m pip install histopia
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
- Use `constraints/registration-repro.txt` for exact validation reruns.
- Do not commit virtual environments, raw slides, generated masks, warped
  images, or registration output directories.
- Record `histopia-register` config files and `registration_result.json` files
  with validation reports, but keep large image artifacts outside Git.
