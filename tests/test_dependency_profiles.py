from __future__ import annotations

import os
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from histopia.qupath._doctor import _MODULE_REQUIREMENTS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def _pins(
    requirements: list[str],
    *,
    active_only: bool = False,
) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if (
            active_only
            and requirement.marker is not None
            and not requirement.marker.evaluate()
        ):
            continue
        exact_versions = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        ]
        if len(exact_versions) == 1:
            pins[canonicalize_name(requirement.name)] = exact_versions[0]
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
    stain = _constraint_pins(ROOT / "constraints/stain-repro.txt")

    assert _pins(extras["registration-repro"]) == registration
    assert _pins(extras["uni2h-repro"]) == semantic
    assert _pins(extras["stain-repro"]) == stain
    assert _pins(extras["semantic-repro"]) == {
        package: semantic[package]
        for package in (
            "numpy",
            "packaging",
            "pillow",
            "scikit-learn",
            "scipy",
            "threadpoolctl",
            "tomli",
        )
    }


def test_qupath_doctor_ranges_match_normal_workflow_extras() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    expected: dict[str, str] = {}
    for extra in ("registration", "wsi", "uni2h", "qupath"):
        for requirement_text in extras[extra]:
            requirement = Requirement(requirement_text)
            name = canonicalize_name(requirement.name)
            specifier = str(requirement.specifier)
            if name in expected:
                assert expected[name] == specifier
            expected[name] = specifier

    observed = {
        canonicalize_name(Requirement(requirement_text).name): str(
            Requirement(requirement_text).specifier
        )
        for _, requirement_text in _MODULE_REQUIREMENTS
    }

    assert observed == expected


@pytest.mark.skipif(
    os.environ.get("HISTOPIA_VERIFY_REPRO") != "1",
    reason="requires an exact reproducible-profile installation",
)
def test_installed_cpu_reproducible_profiles_match_exact_pins() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    expected = _pins(
        extras["registration-repro"] + extras["semantic-repro"] + extras["stain-repro"],
        active_only=True,
    )

    assert {package: distribution_version(package) for package in expected} == expected
