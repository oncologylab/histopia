"""Versioned environment checks for the Histopia QuPath extension."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from signal import Signals
from subprocess import CompletedProcess, run
from typing import Any, Final

from histopia import __version__

QUPATH_WORKFLOW_API_VERSION: Final[int] = 1
QUPATH_WORKFLOWS: Final[tuple[str, ...]] = (
    "registration",
    "semantic",
    "interchange",
    "full",
)

_MODULE_DISTRIBUTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("numpy", "numpy"),
    ("cv2", "opencv-contrib-python-headless"),
    ("scipy", "scipy"),
    ("PIL", "Pillow"),
    # Load libvips before accelerator frameworks to avoid native-library conflicts.
    ("pyvips", "pyvips"),
    ("tifffile", "tifffile"),
    ("sklearn", "scikit-learn"),
    ("threadpoolctl", "threadpoolctl"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("huggingface_hub", "huggingface-hub"),
)
_WORKFLOW_MODULES: Final[dict[str, frozenset[str]]] = {
    "registration": frozenset(("numpy", "cv2", "scipy", "PIL", "pyvips", "tifffile")),
    "semantic": frozenset(
        (
            "numpy",
            "scipy",
            "PIL",
            "pyvips",
            "tifffile",
            "sklearn",
            "threadpoolctl",
            "torch",
            "torchvision",
            "timm",
            "huggingface_hub",
        )
    ),
    "interchange": frozenset(("numpy",)),
}
_WORKFLOW_MODULES["full"] = frozenset().union(*_WORKFLOW_MODULES.values())
_INSTALL_PROFILES: Final[dict[str, str]] = {
    "registration": "histopia[registration,wsi]",
    "semantic": "histopia[uni2h]",
    "interchange": "histopia[qupath]",
    "full": "histopia[registration,wsi,uni2h,qupath]",
}


def inspect_qupath_environment(
    workflow: str = "full",
    *,
    device: str = "auto",
    required_api: int = QUPATH_WORKFLOW_API_VERSION,
    importer: Callable[[str], Any] = import_module,
    compute_inspector: Callable[..., dict[str, object]] | None = None,
    version_resolver: Callable[[str], str] = distribution_version,
    native_probe: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Validate one QuPath workflow and return its versioned capability report."""

    normalized_workflow = workflow.strip().lower()
    if normalized_workflow not in QUPATH_WORKFLOWS:
        choices = ", ".join(QUPATH_WORKFLOWS)
        raise ValueError(f"workflow must be one of: {choices}")
    if required_api <= 0:
        raise ValueError("required QuPath workflow API must be positive")
    if required_api > QUPATH_WORKFLOW_API_VERSION:
        raise RuntimeError(
            "Histopia's QuPath workflow API is too old: "
            f"extension requires {required_api}, installed package provides "
            f"{QUPATH_WORKFLOW_API_VERSION}. Upgrade Histopia in the selected "
            "Python environment."
        )

    required_modules = _WORKFLOW_MODULES[normalized_workflow]
    if "pyvips" in required_modules:
        if native_probe is not None:
            native_probe()
        elif importer is import_module:
            _probe_native_pyvips()
    imported: dict[str, Any] = {}
    dependencies: dict[str, dict[str, str]] = {}
    for module_name, distribution_name in _MODULE_DISTRIBUTIONS:
        if module_name not in required_modules:
            continue
        try:
            module = importer(module_name)
        except Exception as error:
            detail = str(error).strip() or type(error).__name__
            raise RuntimeError(
                f"Histopia {normalized_workflow} preflight failed while importing "
                f"{module_name!r}: {detail}. Install or repair "
                f"{_INSTALL_PROFILES[normalized_workflow]} in the selected Python "
                "environment."
            ) from error
        imported[module_name] = module
        dependencies[module_name] = {
            "distribution": distribution_name,
            "version": _dependency_version(
                module, distribution_name, version_resolver=version_resolver
            ),
        }

    compute = None
    if normalized_workflow in {"semantic", "full"}:
        if compute_inspector is None:
            from histopia.compute import inspect_compute

            compute_inspector = inspect_compute
        compute = compute_inspector(device, torch_module=imported["torch"])

    pyvips = imported.get("pyvips")
    return {
        "schema_version": 1,
        "status": "ok",
        "histopia_version": __version__,
        "qupath_workflow_api_version": QUPATH_WORKFLOW_API_VERSION,
        "workflow": normalized_workflow,
        "python": {
            "executable": sys.executable,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "capabilities": {
            "registration_api_version": 1,
            "semantic_atlas_api_version": 1,
            "qupath_interchange_schema_version": 4,
            "native_vips_thread_control_version": 1,
        },
        "dependencies": dependencies,
        "libvips_version": _libvips_version(pyvips) if pyvips is not None else None,
        "compute": compute,
    }


def _dependency_version(
    module: Any,
    distribution_name: str,
    *,
    version_resolver: Callable[[str], str],
) -> str:
    try:
        return str(version_resolver(distribution_name))
    except PackageNotFoundError:
        value = getattr(module, "__version__", None)
        return str(value) if value is not None else "unknown"


def _libvips_version(pyvips: Any) -> str:
    return ".".join(str(pyvips.version(index)) for index in range(3))


def _probe_native_pyvips(
    *,
    runner: Callable[..., CompletedProcess[str]] = run,
) -> None:
    """Import pyvips out of process so a native loader crash is recoverable."""

    completed = runner(
        [
            sys.executable,
            "-c",
            (
                "import pyvips; "
                "print('.'.join(str(pyvips.version(i)) for i in range(3)))"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode == 0:
        return
    if completed.returncode < 0:
        try:
            reason = f"signal {Signals(-completed.returncode).name}"
        except ValueError:
            reason = f"signal {-completed.returncode}"
    else:
        reason = f"exit code {completed.returncode}"
    raise RuntimeError(
        f"Histopia native pyvips probe failed with {reason}. The selected Python "
        "environment may be resolving incompatible pyvips and libvips libraries. "
        "For Conda, install both from conda-forge. For system Python, install "
        "system libvips and rebuild the binding with python -m pip install "
        "--no-cache-dir --force-reinstall --no-binary=pyvips 'pyvips>=2.2,<3'."
    )
