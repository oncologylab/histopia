from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from histopia.qupath import _cli
from histopia.qupath._doctor import (
    QUPATH_WORKFLOW_API_VERSION,
    _probe_native_pyvips,
    inspect_qupath_environment,
)


class _FakeVips(SimpleNamespace):
    def version(self, index: int) -> int:
        return (8, 16, 1)[index]


def _fake_modules() -> dict[str, object]:
    modules = {
        name: SimpleNamespace(__version__="test")
        for name in (
            "numpy",
            "cv2",
            "scipy",
            "PIL",
            "tifffile",
            "sklearn",
            "threadpoolctl",
            "torch",
            "torchvision",
            "timm",
            "huggingface_hub",
        )
    }
    modules["pyvips"] = _FakeVips(__version__="test")
    return modules


def test_registration_doctor_checks_only_registration_dependencies_in_safe_order():
    modules = _fake_modules()
    imported: list[str] = []

    def importer(name: str) -> object:
        imported.append(name)
        return modules[name]

    report = inspect_qupath_environment(
        "registration",
        importer=importer,
        version_resolver=lambda name: f"{name}-version",
    )

    assert imported == ["numpy", "cv2", "scipy", "PIL", "pyvips", "tifffile"]
    assert report["workflow"] == "registration"
    assert report["compute"] is None
    assert report["libvips_version"] == "8.16.1"


def test_full_doctor_loads_libvips_before_torch_and_reports_compute():
    modules = _fake_modules()
    imported: list[str] = []
    compute_calls: list[tuple[str, object]] = []

    def importer(name: str) -> object:
        imported.append(name)
        return modules[name]

    def inspect_compute(device: str, *, torch_module: object) -> dict[str, object]:
        compute_calls.append((device, torch_module))
        return {"selected": device}

    report = inspect_qupath_environment(
        "full",
        device="cuda:2",
        importer=importer,
        compute_inspector=inspect_compute,
        version_resolver=lambda name: "1",
    )

    assert imported.index("pyvips") < imported.index("torch")
    assert compute_calls == [("cuda:2", modules["torch"])]
    assert report["compute"] == {"selected": "cuda:2"}
    assert report["qupath_workflow_api_version"] == QUPATH_WORKFLOW_API_VERSION


def test_doctor_reports_missing_dependency_with_install_profile():
    def importer(name: str) -> object:
        if name == "cv2":
            raise ImportError("missing shared object")
        return _fake_modules()[name]

    with pytest.raises(RuntimeError, match=r"cv2.*histopia\[registration,wsi\]"):
        inspect_qupath_environment(
            "registration",
            importer=importer,
            version_resolver=lambda name: "1",
        )


def test_doctor_rejects_newer_required_api():
    with pytest.raises(RuntimeError, match="Upgrade Histopia"):
        inspect_qupath_environment(
            "interchange",
            required_api=QUPATH_WORKFLOW_API_VERSION + 1,
        )


def test_native_pyvips_probe_turns_process_crash_into_repair_error():
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=-11, stdout="", stderr="")

    with pytest.raises(RuntimeError, match=r"SIGSEGV.*--no-binary=pyvips"):
        _probe_native_pyvips(runner=runner)


def test_qupath_cli_prints_doctor_json(monkeypatch, capsys):
    monkeypatch.setattr(
        _cli,
        "inspect_qupath_environment",
        lambda workflow, **kwargs: {
            "status": "ok",
            "workflow": workflow,
            "device": kwargs["device"],
        },
    )

    assert (
        _cli.main(
            [
                "--doctor",
                "--workflow",
                "semantic",
                "--device",
                "cpu",
                "--require-api",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "workflow": "semantic",
        "device": "cpu",
    }


def test_qupath_cli_requires_export_paths():
    with pytest.raises(SystemExit, match="2"):
        _cli.main([])
