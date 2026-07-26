from __future__ import annotations

import os
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

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

    assert _pins(extras["registration-repro"]) == registration
    assert _pins(extras["uni2h-repro"]) == semantic
    assert _pins(extras["semantic-repro"]) == {
        package: semantic[package]
        for package in (
            "numpy",
            "pillow",
            "scikit-learn",
            "scipy",
            "threadpoolctl",
            "tomli",
        )
    }


@pytest.mark.skipif(
    os.environ.get("HISTOPIA_VERIFY_REPRO") != "1",
    reason="requires an exact reproducible-profile installation",
)
def test_installed_cpu_reproducible_profiles_match_exact_pins() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    expected = _pins(extras["registration-repro"] + extras["semantic-repro"])

    assert {package: distribution_version(package) for package in expected} == expected
