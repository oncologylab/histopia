from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.cells._result import write_cell_result
from histopia.visualization._provisional_feedback import ProvisionalFeedbackStore


def test_provisional_feedback_is_fingerprint_bound_append_only_and_path_free(
    tmp_path: Path,
) -> None:
    run = _cell_run(tmp_path)
    fingerprint = json.loads((run / "cell_result.json").read_text())["fingerprint"]
    store = ProvisionalFeedbackStore(tmp_path / "private-feedback")
    request: dict[str, object] = {
        "cohort": "mouse",
        "stage": "cells",
        "fingerprint": fingerprint,
        "slide_id": "001",
        "decision": "hold",
        "labels": ["debris_or_necrosis"],
        "checks": {"nuclear_support": False},
        "comment": "Debris is being outlined as cells.",
        "reviewer": "ai-provisional",
    }

    first = store.save(request, run=run)
    request["decision"] = "accept"
    request["labels"] = []
    request["comment"] = "Improved after rerun."
    second = store.save(request, run=run)

    assert first["provisional"] is True
    assert second["feedback"]["001"]["decision"] == "accept"
    assert second["summary"] == {"reviewed": 1, "accept": 1, "hold": 0, "reject": 0}
    stored = next((tmp_path / "private-feedback").glob("*/*/*.json"))
    assert len(json.loads(stored.read_text())["records"]) == 2
    assert str(tmp_path) not in json.dumps(second)
    assert store.summary()["reviewed_slides"] == 1


def test_provisional_feedback_rejects_stale_or_invalid_observations(
    tmp_path: Path,
) -> None:
    run = _cell_run(tmp_path)
    store = ProvisionalFeedbackStore(tmp_path / "feedback")
    base: dict[str, object] = {
        "cohort": "mouse",
        "stage": "cells",
        "fingerprint": "0" * 64,
        "slide_id": "001",
        "decision": "reject",
        "labels": [],
        "checks": {},
        "comment": "",
        "reviewer": "reviewer",
    }

    with pytest.raises(ValueError, match="artifact changed"):
        store.save(base, run=run)
    base["fingerprint"] = json.loads((run / "cell_result.json").read_text())[
        "fingerprint"
    ]
    base["labels"] = ["not-a-label"]
    with pytest.raises(ValueError, match="labels"):
        store.save(base, run=run)


def _cell_run(root: Path) -> Path:
    run = root / "private-cell-run"
    (run / "labels").mkdir(parents=True)
    (run / "qc").mkdir()
    (run / "preflight.json").write_text("{}\n")
    (run / "labels/001.cells.tiff").write_bytes(b"labels")
    (run / "qc/001.json").write_text("{}\n")
    write_cell_result(
        run,
        {
            "schema_version": 1,
            "registration_result_sha256": "a" * 64,
            "preflight": "preflight.json",
            "slides": [
                {
                    "section": "001",
                    "slide": "sample.ndpi",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    return run
