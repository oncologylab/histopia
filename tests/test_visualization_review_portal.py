from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.visualization import _review_portal


def test_registration_review_builds_path_free_fixed_viewport_portal(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "registration"
    output = tmp_path / "review"

    def build_mask(
        registration_run: Path,
        destination: Path,
        *,
        workers: int,
    ) -> Path:
        assert registration_run == run
        assert workers == 3
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "approved": False,
                    "fingerprint": "mask-fingerprint",
                    "slides": [{}, {}],
                }
            )
        )
        return destination / "index.html"

    def build_order(
        proposal: Path,
        processed: Path,
        destination: Path,
        *,
        workers: int,
    ) -> Path:
        assert proposal == run / "section_order_review.json"
        assert processed == run / "processed"
        assert workers == 3
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "fingerprint": "order-fingerprint",
                    "physical_area_continuity": {
                        "review_recommended": True,
                    },
                    "slides": [{}, {}],
                }
            )
        )
        return destination / "index.html"

    monkeypatch.setattr(_review_portal, "build_mask_review", build_mask)
    monkeypatch.setattr(_review_portal, "build_section_order_review", build_order)
    run.mkdir()
    (run / "section_order_review.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"slide": "HE.ndpi"},
                    {"slide": "CK19.ndpi"},
                ]
            }
        )
    )

    index = _review_portal.build_registration_review(
        run,
        output,
        workers=3,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert index == output / "index.html"
    assert manifest["mask"] == {
        "approved": False,
        "fingerprint": "mask-fingerprint",
        "slide_count": 2,
        "href": "mask/index.html",
    }
    assert manifest["order"]["approved"] is True
    assert manifest["order"]["review_recommended"] is True
    assert str(tmp_path) not in (output / "index.html").read_text()
    assert (output / "manifest-data.js").is_file()
    assert "manifest-data.js" in (output / "index.html").read_text()
    assert "overflow:hidden" in (output / "registration-review.css").read_text()
    assert "stage" in (output / "registration-review.js").read_text()


def test_registration_review_supports_mask_only_preparation(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "registration"
    output = tmp_path / "review"

    def build_mask(
        registration_run: Path,
        destination: Path,
        *,
        workers: int,
    ) -> Path:
        assert registration_run == run
        assert workers == 1
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "approved": False,
                    "fingerprint": "mask-only",
                    "slides": [{}],
                }
            )
        )
        return destination / "index.html"

    def reject_order(*args, **kwargs) -> Path:
        raise AssertionError("order review must not run before order preparation")

    monkeypatch.setattr(_review_portal, "build_mask_review", build_mask)
    monkeypatch.setattr(_review_portal, "build_section_order_review", reject_order)

    _review_portal.build_registration_review(run, output)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mask"]["slide_count"] == 1
    assert "order" not in manifest
    script = (output / "registration-review.js").read_text()
    assert "button.hidden" in script


def test_registration_review_hides_stale_downstream_stages_at_new_mask_gate(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "registration"
    output = tmp_path / "review"
    run.mkdir()
    current_names = ("HE.ndpi", "CK19.ndpi")
    (run / "mask_review.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [{"slide": name} for name in current_names],
            }
        )
    )
    (run / "section_order_review.json").write_text("not current JSON")
    (run / "registration_result.json").write_text("not current JSON")
    (run / "registration_performance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "registration",
                "observational_only": True,
                "status": "review_required",
                "review_stage": "masks",
                "stages": {"mask_review": {"status": "review_required"}},
            }
        )
    )

    def build_mask(
        registration_run: Path,
        destination: Path,
        *,
        workers: int,
    ) -> Path:
        assert registration_run == run
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "approved": False,
                    "fingerprint": "current-mask",
                    "slides": [{}, {}],
                }
            )
        )
        return destination / "index.html"

    def reject_stale(*args, **kwargs) -> Path:
        raise AssertionError("stale downstream review must remain hidden")

    monkeypatch.setattr(_review_portal, "build_mask_review", build_mask)
    monkeypatch.setattr(_review_portal, "build_section_order_review", reject_stale)
    monkeypatch.setattr(_review_portal, "build_alignment_review", reject_stale)

    _review_portal.build_registration_review(run, output)

    assert json.loads((output / "manifest.json").read_text()) == {
        "schema_version": 1,
        "mask": {
            "approved": False,
            "fingerprint": "current-mask",
            "slide_count": 2,
            "href": "mask/index.html",
        },
    }


def test_registration_review_refuses_stale_artifacts_when_telemetry_is_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = tmp_path / "registration"
    run.mkdir()
    (run / "mask_review.json").write_text(
        json.dumps({"slides": [{"slide": "stale.ndpi"}]})
    )
    (run / "registration_performance.json").write_text("truncated")

    def reject_stale(*args, **kwargs) -> Path:
        raise AssertionError("stale review must not be built")

    monkeypatch.setattr(_review_portal, "build_mask_review", reject_stale)

    with pytest.raises(ValueError, match="refusing stale review stages"):
        _review_portal.build_registration_review(run, tmp_path / "review")


def test_registration_review_rejects_mask_from_unreached_latest_stage(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "registration"
    run.mkdir()
    (run / "mask_review.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [{"slide": "stale.ndpi"}],
            }
        )
    )
    (run / "registration_performance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "registration",
                "observational_only": True,
                "status": "failed",
                "stages": {"thumbnail_load": {"status": "failed"}},
            }
        )
    )

    def reject_stale(*args, **kwargs) -> Path:
        raise AssertionError("stale mask review must not be built")

    monkeypatch.setattr(_review_portal, "build_mask_review", reject_stale)

    with pytest.raises(ValueError, match="has not prepared mask review"):
        _review_portal.build_registration_review(run, tmp_path / "review")


def test_registration_cohort_review_builds_one_path_free_entrypoint(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "review"
    runs = {"4845": tmp_path / "run-4845", "8471": tmp_path / "run-8471"}

    def build(run: Path, destination: Path, *, workers: int) -> Path:
        assert run in runs.values()
        assert workers == 3
        slide_count = 17 if run == runs["4845"] else 12
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "mask": {
                        "approved": True,
                        "slide_count": slide_count,
                    },
                    "order": {
                        "approved": False,
                        "slide_count": slide_count,
                        "review_recommended": run == runs["8471"],
                    },
                }
            )
        )
        (destination / "index.html").write_text("<p>review</p>")
        return destination / "index.html"

    monkeypatch.setattr(_review_portal, "build_registration_review", build)

    index = _review_portal.build_registration_cohort_review(
        runs,
        output,
        workers=3,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert index == output / "index.html"
    assert [row["id"] for row in manifest["reviews"]] == ["4845", "8471"]
    assert manifest["reviews"][0]["stages"] == ["mask", "order"]
    assert manifest["reviews"][0]["slide_count"] == 17
    assert manifest["reviews"][0]["stage_summary"] == {
        "mask": {"approved": True, "slide_count": 17},
        "order": {
            "approved": False,
            "slide_count": 17,
            "review_recommended": False,
        },
    }
    assert manifest["reviews"][1]["stage_summary"]["order"] == {
        "approved": False,
        "slide_count": 12,
        "review_recommended": True,
    }
    assert str(tmp_path) not in index.read_text()
    assert "overflow:hidden" in (output / "cohort-review.css").read_text()
    script = (output / "cohort-review.js").read_text()
    assert "review required" in script
    assert "status.title" in script


def test_registration_cohort_review_rejects_inconsistent_stage_counts(
    tmp_path: Path, monkeypatch
) -> None:
    def build(run: Path, destination: Path, *, workers: int) -> Path:
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "mask": {"approved": True, "slide_count": 17},
                    "order": {"approved": False, "slide_count": 16},
                }
            )
        )
        (destination / "index.html").write_text("<p>review</p>")
        return destination / "index.html"

    monkeypatch.setattr(_review_portal, "build_registration_review", build)

    with pytest.raises(ValueError, match="stage slide counts differ"):
        _review_portal.build_registration_cohort_review(
            {"4845": tmp_path / "run"},
            tmp_path / "review",
        )


def test_registration_cohort_review_rejects_unsafe_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid registration review name"):
        _review_portal.build_registration_cohort_review(
            {"../escape": tmp_path / "run"},
            tmp_path / "review",
        )


def test_workflow_review_builds_one_fixed_stage_hub(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text("{}")

    def registration_review(runs, output, *, workers):
        calls.append(("registration", (runs, output, workers)))
        output.mkdir(parents=True)
        (output / "index.html").write_text("registration")
        return output / "index.html"

    def atlas(runs, output, **kwargs):
        calls.append(("atlas", (runs, output, kwargs)))
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "mice": []})
        )
        (output / "index.html").write_text("atlas")
        return output / "index.html"

    def stain(viewer, output, *, mice):
        calls.append(("stain", (viewer, output, mice)))
        output.mkdir(parents=True)
        (output / "index.html").write_text("stain")
        return output / "index.html"

    monkeypatch.setattr(
        _review_portal,
        "build_registration_cohort_review",
        registration_review,
    )
    monkeypatch.setattr(_review_portal, "build_section_viewer", atlas)
    monkeypatch.setattr(
        "histopia.visualization._stain_review.build_stain_review",
        stain,
    )

    output = tmp_path / "review"
    index = _review_portal.build_workflow_review(
        {"mouse": registration},
        output,
        semantic_runs={"mouse": tmp_path / "semantic"},
        stain_runs={"mouse": tmp_path / "stain"},
        workers=3,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert index == output / "index.html"
    assert [tab["id"] for tab in manifest["tabs"]] == [
        "registration",
        "atlas",
        "stain",
        "decisions",
    ]
    assert (output / "decisions" / "index.html").is_file()
    assert (
        "/api/reviews/approve"
        in (output / "decisions" / "review-decisions.js").read_text()
    )
    assert calls[1][1][2]["require_approvals"] is False
    assert "overflow:hidden" in (output / "workflow-review.css").read_text()
    assert str(tmp_path) not in (output / "manifest-data.js").read_text()


@pytest.mark.browser
def test_registration_cohort_review_shows_full_mobile_approval_status(
    tmp_path: Path, monkeypatch
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    runs = {"4845": tmp_path / "run-4845", "8471": tmp_path / "run-8471"}

    def build(run: Path, destination: Path, *, workers: int) -> Path:
        slide_count = 17 if run == runs["4845"] else 12
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "mask": {"approved": True, "slide_count": slide_count},
                    "order": {
                        "approved": False,
                        "slide_count": slide_count,
                        "review_recommended": run == runs["8471"],
                    },
                }
            )
        )
        (destination / "index.html").write_text("<p>review</p>")
        return destination / "index.html"

    monkeypatch.setattr(_review_portal, "build_registration_review", build)
    index = _review_portal.build_registration_cohort_review(
        runs,
        tmp_path / "review",
    )

    errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("requestfailed", lambda request: errors.append(request.url))
        page.goto(f"{index.as_uri()}?cohort=8471", wait_until="load")
        page.wait_for_function(
            "() => document.querySelector('#status').textContent.includes('12 slides')"
        )
        assert page.locator("#status").inner_text() == (
            "12 slides · masks approved · order review required (continuity flag)"
        )
        status_width = page.locator("#status").evaluate(
            "(element) => [element.clientWidth, element.scrollWidth]"
        )
        assert status_width[1] <= status_width[0] + 1
        dimensions = page.evaluate(
            """() => ({
              x: document.documentElement.scrollWidth > innerWidth,
              y: document.documentElement.scrollHeight > innerHeight,
              bodyY: document.body.scrollHeight > document.body.clientHeight,
              mainY: document.querySelector('main').scrollHeight >
                document.querySelector('main').clientHeight,
              iframeDisplay: getComputedStyle(
                document.querySelector('iframe')).display,
            })"""
        )
        assert not dimensions["x"]
        assert not dimensions["y"]
        assert not dimensions["bodyY"]
        assert not dimensions["mainY"]
        assert dimensions["iframeDisplay"] == "block"
        browser.close()
    assert errors == []


@pytest.mark.browser
def test_registration_review_opens_directly_without_server(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    run = tmp_path / "registration"
    processed = run / "processed"
    processed.mkdir(parents=True)
    slides = []
    order_slides = []
    for index, name in enumerate(("HE.ndpi", "CK19.ndpi"), start=1):
        image = np.full((30, 40, 3), 235, dtype=np.uint8)
        image[5:26, 7 + index : 31 + index] = (125, 75, 90)
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5:26, 7 + index : 31 + index] = 255
        stem = Path(name).stem
        Image.fromarray(image).save(processed / f"{stem}.thumbnail.png")
        Image.fromarray(mask).save(processed / f"{stem}.mask.png")
        slides.append(
            {
                "path": str(tmp_path / name),
                "is_reference": index == 1,
                "mask": {
                    "method": "object_aware_fusion",
                    "metrics": {"foreground_fraction": float(mask.mean() / 255)},
                    "warnings": [],
                },
                "mask_review": {"status": "pending"},
                "transform": {"matrix": np.eye(3).tolist()},
                "alignment_metrics": (
                    {"dice": 1.0, "coverage": 1.0, "status": "reference"}
                    if index == 1
                    else {
                        "dice": 0.9,
                        "coverage": None,
                        "status": "pass",
                    }
                ),
            }
        )
        order_slides.append(
            {
                "order": index,
                "slide": name,
                "fixed": index == 1,
                "distance_from_previous": None if index == 1 else 0.1,
                "physical_tissue_area_um2": 2_000_000.0,
            }
        )
    (run / "registration_result.json").write_text(
        json.dumps({"reference_slide": slides[0]["path"], "slides": slides})
    )
    (run / "section_order_review.json").write_text(
        json.dumps(
            {
                "approved": False,
                "fingerprint": "order-fingerprint",
                "objective": 0.1,
                "confidence_margin": 0.2,
                "physically_calibrated": True,
                "slides": order_slides,
            }
        )
    )
    index = _review_portal.build_registration_review(
        run,
        tmp_path / "review",
    )

    errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("requestfailed", lambda request: errors.append(request.url))
        page.goto(index.as_uri(), wait_until="load")
        page.wait_for_function(
            "() => document.querySelector('#status').textContent.includes('2 slides')"
        )
        assert page.frame_locator("#review").locator("article").count() == 2
        page.get_by_role("button", name="Section order").click()
        page.frame_locator("#review").locator("article").first.wait_for()
        assert page.frame_locator("#review").locator("article").count() == 2
        page.get_by_role("button", name="Registered stack").click()
        page.frame_locator("#review").locator("article").first.wait_for()
        assert page.frame_locator("#review").locator("article").count() == 2
        alignment = page.frame_locator("#review")
        assert alignment.locator("#summary").inner_text().endswith("median Dice 0.900")
        metrics = alignment.locator("article .metrics").all_inner_texts()
        assert metrics == ["reference", "Dice 0.900 | pass"]
        assert "0.000" not in " ".join(metrics)
        dimensions = page.evaluate(
            """() => ({
              x: document.documentElement.scrollWidth > innerWidth,
              y: document.documentElement.scrollHeight > innerHeight,
              bodyY: document.body.scrollHeight > document.body.clientHeight,
              mainY: document.querySelector('main').scrollHeight >
                document.querySelector('main').clientHeight,
              iframeDisplay: getComputedStyle(
                document.querySelector('iframe')).display,
            })"""
        )
        assert not dimensions["x"]
        assert not dimensions["y"]
        assert not dimensions["bodyY"]
        assert not dimensions["mainY"]
        assert dimensions["iframeDisplay"] == "block"
        browser.close()
    assert errors == []
