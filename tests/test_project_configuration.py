from pathlib import Path

import pytest

from histopia.registration._cli import _config_from_mapping

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_ci_installs_dependencies_exercised_by_semantic_tests() -> None:
    workflow = Path(".github/workflows/tests.yml").read_text()

    assert '".[dev,registration,semantic,wsi]"' in workflow
    assert '".[browser-test,registration,semantic,wsi]"' in workflow
    assert "python -m pytest -m browser" in workflow


def test_browser_test_extra_contains_its_test_runner() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text())

    assert any(
        requirement.startswith("pytest")
        for requirement in metadata["project"]["optional-dependencies"]["browser-test"]
    )


def test_uni2h_repro_extra_matches_checked_in_constraints() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    requirements = set(metadata["project"]["optional-dependencies"]["uni2h-repro"])
    constrained = {
        line
        for line in Path("constraints/semantic-repro.txt").read_text().splitlines()
        if line and not line.startswith("#") and "tomli" not in line
    }

    assert requirements == constrained


def test_pages_workflow_uses_fingerprinted_release_artifact() -> None:
    workflow = Path(".github/workflows/pages.yml").read_text()

    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "SHOWCASE_SHA256:" in workflow
    assert "sha256sum --check" in workflow
    assert "find _site -type l" in workflow


def test_readme_links_to_interactive_pages_showcase() -> None:
    readme = Path("README.md").read_text()

    assert "https://oncologylab.github.io/histopia/" in readme
    assert (
        "https://github.com/oncologylab/qupath-extension-histopia/releases/latest"
        in readme
    )


def test_registration_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown registration config keys: typo"):
        _config_from_mapping(
            {
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(tmp_path / "output"),
                "typo": True,
            }
        )
