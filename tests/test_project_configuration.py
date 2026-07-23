from pathlib import Path

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
