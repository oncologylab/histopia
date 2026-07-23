from pathlib import Path


def test_ci_installs_dependencies_exercised_by_semantic_tests() -> None:
    workflow = Path(".github/workflows/tests.yml").read_text()

    assert '".[dev,registration,semantic,wsi]"' in workflow
    assert '".[browser-test,registration,semantic,wsi]"' in workflow
    assert "python -m pytest -m browser" in workflow
