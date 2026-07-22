from __future__ import annotations

from histopia.semantic._uni2h import _cached_model_revision, _preload_wsi_backend


def test_uni2h_preloads_wsi_backend_before_gpu_stack() -> None:
    imported: list[str] = []

    _preload_wsi_backend(imported.append)

    assert imported == ["pyvips"]


def test_cached_model_revision_reads_pinned_snapshot(tmp_path) -> None:
    model = tmp_path / "models--MahmoodLab--UNI2-h"
    (model / "refs").mkdir(parents=True)
    (model / "snapshots" / "abc123").mkdir(parents=True)
    (model / "refs" / "main").write_text("abc123\n")

    assert _cached_model_revision(tmp_path) == "abc123"
