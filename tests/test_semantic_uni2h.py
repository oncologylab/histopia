from __future__ import annotations

from histopia.semantic._uni2h import _preload_wsi_backend


def test_uni2h_preloads_wsi_backend_before_gpu_stack() -> None:
    imported: list[str] = []

    _preload_wsi_backend(imported.append)

    assert imported == ["pyvips"]
