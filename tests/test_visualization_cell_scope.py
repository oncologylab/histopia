import copy
import json
from types import SimpleNamespace

import pytest

from histopia.visualization import CellSectionScope, _wsi_tiles, build_cell_review
from histopia.visualization._review_api import ReviewDecisionService, ReviewRuns


def result():
    return {
        "fingerprint": "a" * 64,
        "registration_result_sha256": "b" * 64,
        "slides": [
            {"section": s, "slide": f"slide{s}.ndpi", "labels": f"{s}.tiff"}
            for s in ("001", "002")
        ],
        "artifacts": {"001.tiff": "c" * 64, "002.tiff": "d" * 64},
    }


def scope():
    return CellSectionScope("a" * 64, {"002": "d" * 64})


@pytest.mark.parametrize("mutation", ["fingerprint", "labels", "unknown", "duplicate"])
def test_scope_rejects_stale_or_ambiguous_source(mutation):
    source = result()
    if mutation == "fingerprint":
        source["fingerprint"] = "e" * 64
    elif mutation == "labels":
        source["artifacts"]["002.tiff"] = "e" * 64
    elif mutation == "unknown":
        source["slides"].pop()
    else:
        source["slides"].append(copy.deepcopy(source["slides"][1]))
    with pytest.raises(ValueError):
        scope().select(source)


def test_scope_is_immutable_and_preserves_source_order():
    labels = {"002": "d" * 64, "001": "c" * 64}
    selected = CellSectionScope("a" * 64, labels)
    labels.clear()
    assert [s["section"] for s in selected.select(result())] == ["001", "002"]
    with pytest.raises(TypeError):
        selected.labels_by_section["003"] = "e" * 64
    with pytest.raises(ValueError):
        CellSectionScope("a" * 64, {})


@pytest.mark.parametrize("operation", ["approve", "section", "provisional"])
def test_scoped_review_rejects_writes_outside_scope(tmp_path, operation):
    service = ReviewDecisionService(
        {"mouse": ReviewRuns(tmp_path, cells=tmp_path, cell_section_scope=scope())}
    )
    request = {
        "cohort": "mouse",
        "stage": "cells",
        "reviewer": "reviewer",
        "notes": "reviewed",
        "section": "001",
        "slide_id": "001",
        "accepted": True,
    }
    method = {
        "approve": service.approve,
        "section": service.review_cell_section,
        "provisional": service.save_provisional_feedback,
    }[operation]
    with pytest.raises(ValueError, match="whole run|outside.*scope"):
        method(request)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stale", [False, True])
def test_scoped_review_validates_selected_source_before_write(
    tmp_path, monkeypatch, stale
):
    source = result()
    if stale:
        source["fingerprint"] = "e" * 64
    monkeypatch.setattr(
        "histopia.cells._result.validate_cell_result_index", lambda _: source
    )
    writes = []
    monkeypatch.setattr(
        "histopia.cells.review_cell_section",
        lambda *args, **kwargs: writes.append((args, kwargs)) or {"saved": True},
    )
    service = ReviewDecisionService(
        {"mouse": ReviewRuns(tmp_path, cells=tmp_path, cell_section_scope=scope())}
    )
    request = {
        "cohort": "mouse",
        "section": "002",
        "reviewer": "reviewer",
        "accepted": True,
    }
    if stale:
        with pytest.raises(ValueError, match="fingerprint differs"):
            service.review_cell_section(request)
        assert not writes
    else:
        assert service.review_cell_section(request) == {"saved": True}
        assert len(writes) == 1


def test_static_review_contains_only_selected_sections(tmp_path):
    run = tmp_path / "cells"
    run.mkdir()
    (run / "cell_result.json").write_text(json.dumps(result()))
    index = build_cell_review(
        {"mouse": run}, tmp_path / "site", cell_section_scopes={"mouse": scope()}
    )
    row = json.loads((index.parent / "manifest.json").read_text())["cohorts"][0]
    assert [s["id"] for s in row["sections"]] == ["002"]
    assert row["selected_section_scope"] is True
    assert "selected sections" in (index.parent / "cell-review.js").read_text()
    with pytest.raises(ValueError, match="no matching"):
        build_cell_review(
            {"mouse": run},
            tmp_path / "bad",
            cell_section_scopes={"other": scope()},
        )


def test_registry_loads_scope_and_rejects_scope_without_cells(tmp_path):
    config = tmp_path / "registry.json"
    row = {
        "registration": "registration",
        "cells": "cells",
        "cell_section_scope": {
            "schema_version": 1,
            "source_fingerprint": "a" * 64,
            "labels_by_section": {"002": "d" * 64},
        },
    }
    config.write_text(json.dumps({"schema_version": 1, "cohorts": {"mouse": row}}))
    loaded = ReviewDecisionService.from_file(config).cell_section_scopes()["mouse"]
    assert loaded.select(result())[0]["section"] == "002"
    del row["cells"]
    config.write_text(json.dumps({"schema_version": 1, "cohorts": {"mouse": row}}))
    with pytest.raises(ValueError, match="no matching"):
        ReviewDecisionService.from_file(config)


def test_tile_service_omits_excluded_cell_layer(tmp_path, monkeypatch):
    registration = tmp_path / "registration"
    registration.mkdir()
    slides = []
    for section in ("001", "002"):
        path = tmp_path / f"slide{section}.ndpi"
        path.write_bytes(b"synthetic")
        slides.append({"path": str(path), "is_reference": section == "001"})
    (registration / "registration_result.json").write_text(
        json.dumps({"slides": slides})
    )
    monkeypatch.setattr(
        _wsi_tiles,
        "validate_registration_approval",
        lambda _: SimpleNamespace(registration_result_sha256="b" * 64),
    )
    monkeypatch.setattr(
        "histopia.cells._result.validate_cell_result_index", lambda _: result()
    )

    def image_layer(name, path, digest, **kwargs):
        return _wsi_tiles.WsiLayer(
            name, path, digest, (_wsi_tiles.WsiLevel(40, 24, 0),), 512, 0.5
        )

    monkeypatch.setattr(_wsi_tiles, "_image_layer", image_layer)
    monkeypatch.setattr(
        _wsi_tiles,
        "_cell_layer",
        lambda path, digest, raw: image_layer("cells", path, digest),
    )
    service = _wsi_tiles.WsiTileService.from_runs(
        {"mouse": (registration, None)},
        cell_runs={"mouse": tmp_path / "cells"},
        cell_section_scopes={"mouse": scope()},
    )
    assert "raw" in service.metadata("mouse", "001")["layers"]
    assert "cells" not in service.metadata("mouse", "001")["layers"]
    assert "cells" in service.metadata("mouse", "002")["layers"]
    with pytest.raises(FileNotFoundError):
        service.render_tile("mouse", "001", "cells", "c" * 64, 0, 0, 0)
