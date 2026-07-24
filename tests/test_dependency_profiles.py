from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def _pins(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        package, separator, version = requirement.partition("==")
        if separator:
            pins[package.lower().replace("_", "-")] = version.split(";", 1)[0].strip()
    return pins


def _constraint_pins(path: Path) -> dict[str, str]:
    requirements = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _pins(requirements)


def test_reproducible_extras_match_constraint_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    registration = _constraint_pins(ROOT / "constraints/registration-repro.txt")
    semantic = _constraint_pins(ROOT / "constraints/semantic-repro.txt")
    registration.pop("tomli")
    semantic.pop("tomli")

    assert _pins(extras["registration-repro"]) == registration
    assert _pins(extras["uni2h-repro"]) == semantic
    assert _pins(extras["semantic-repro"]) == {
        package: semantic[package]
        for package in ("numpy", "pillow", "scikit-learn", "scipy")
    }
