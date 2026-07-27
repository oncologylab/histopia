from pathlib import Path

import pytest

from histopia.registration import registration_config_from_mapping

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_ci_installs_dependencies_exercised_by_semantic_tests() -> None:
    workflow = Path(".github/workflows/tests.yml").read_text()

    assert '".[dev,registration,semantic,stain,wsi]"' in workflow
    assert '".[browser-test,registration,semantic,stain,wsi]"' in workflow
    assert '".[dev,registration,semantic,stain,wsi,qupath]"' in workflow
    assert "HISTOPIA_VERIFY_REPRO_CONSTRAINTS" in workflow
    assert "-c constraints/registration-repro.txt" in workflow
    assert "-c constraints/semantic-repro.txt" in workflow
    assert "-c constraints/stain-repro.txt" in workflow
    assert "python -m pytest -m browser" in workflow


def test_browser_test_extra_contains_its_test_runner() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text())

    assert any(
        requirement.startswith("pytest")
        for requirement in metadata["project"]["optional-dependencies"]["browser-test"]
    )


def test_exact_versions_are_not_duplicated_in_extras() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]

    assert not any(name.endswith("-repro") for name in extras)


def test_pages_workflow_uses_fingerprinted_release_artifact() -> None:
    workflow = Path(".github/workflows/pages.yml").read_text()

    assert "actions/configure-pages@v6" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "actions/deploy-pages@v5" in workflow
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
        registration_config_from_mapping(
            {
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(tmp_path / "output"),
                "typo": True,
            }
        )


def test_registration_config_accepts_exact_external_slide_selection(
    tmp_path: Path,
) -> None:
    slides = (tmp_path / "second.ndpi", tmp_path / "first.scn")

    config = registration_config_from_mapping(
        {
            "input_dir": str(tmp_path),
            "input_slides": [str(path) for path in slides],
            "output_dir": str(tmp_path / "output"),
        }
    )

    assert config.input_slides == slides
