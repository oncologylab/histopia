from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        ("histopia.registration._cli", ["--help"]),
        ("histopia.semantic._cli", ["--help"]),
        ("histopia.visualization._cli", ["--help"]),
        ("histopia.qupath._cli", ["--help"]),
    ],
)
def test_cli_help_does_not_import_optional_numpy(
    module: str,
    arguments: list[str],
) -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise ModuleNotFoundError("NumPy is intentionally unavailable")
                return None

        sys.meta_path.insert(0, BlockNumpy())
        from {module} import main

        try:
            main({arguments!r})
        except SystemExit as error:
            if error.code != 0:
                raise
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_config_loaders_do_not_import_optional_dependencies() -> None:
    script = textwrap.dedent(
        """
        import sys

        from histopia.registration import load_registration_config
        from histopia.semantic import load_semantic_config

        assert callable(load_registration_config)
        assert callable(load_semantic_config)
        assert "numpy" not in sys.modules
        assert "cv2" not in sys.modules
        assert "torch" not in sys.modules
        assert "pyvips" not in sys.modules
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
