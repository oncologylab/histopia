from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.visualization import (
    build_mask_review,
    load_registration_feedback,
    registration_feedback_rows,
    summarize_registration_feedback,
)
from histopia.visualization._feedback import (
    RegistrationFeedbackStore,
    registration_feedback_evidence,
)
from histopia.visualization._server import create_viewer_server


def _mask_run(tmp_path: Path) -> Path:
    run = tmp_path / "mouse"
    processed = run / "processed"
    processed.mkdir(parents=True)
    slides = []
    for name in ("HE.ndpi", "CK19.ndpi"):
        stem = Path(name).stem
        (processed / f"{stem}.thumbnail.png").write_bytes(f"image-{name}".encode())
        (processed / f"{stem}.mask.png").write_bytes(f"mask-{name}".encode())
        slides.append(
            {
                "slide": name,
                "thumbnail_sha256": f"review-{name}",
                "status": "pending",
            }
        )
    (run / "mask_review.json").write_text(
        json.dumps({"schema_version": 2, "slides": slides})
    )
    return run


def test_feedback_store_persists_history_and_summarizes_latest_record(
    tmp_path: Path,
) -> None:
    run = _mask_run(tmp_path)
    store = RegistrationFeedbackStore(tmp_path / "feedback")
    evidence = registration_feedback_evidence(run, "mask")
    request: dict[str, object] = {
        "cohort": "mouse",
        "stage": "mask",
        "fingerprint": evidence["fingerprint"],
        "slide_id": "HE.ndpi",
        "decision": "reject",
        "labels": ["extra_debris"],
        "comment": "Detached material is not shared across sections.",
        "reviewer": "Reviewer",
    }

    first = store.save(request, registration_run=run)
    request.update({"decision": "accept", "labels": [], "comment": "Corrected."})
    second = store.save(request, registration_run=run)

    assert first["feedback"]["HE.ndpi"]["decision"] == "reject"
    assert second["feedback"]["HE.ndpi"]["decision"] == "accept"
    payloads = load_registration_feedback(tmp_path / "feedback")
    assert len(payloads) == 1
    assert len(payloads[0]["records"]) == 2
    rows = registration_feedback_rows(tmp_path / "feedback")
    assert len(rows) == 1
    assert rows[0]["decision"] == "accept"
    assert rows[0]["artifact_fingerprint"] == evidence["fingerprint"]
    assert summarize_registration_feedback(tmp_path / "feedback") == {
        "schema_version": 1,
        "reviewed_slides": 1,
        "by_stage": {"mask": 1},
        "by_decision": {"accept": 1},
        "by_issue": {},
        "by_cohort": {"mouse": 1},
    }
    with pytest.raises(ValueError, match="1 unreviewed"):
        store.require_accepted(
            cohort="mouse",
            stage="mask",
            registration_run=run,
        )


def test_feedback_store_rejects_stale_or_uninformative_concern(
    tmp_path: Path,
) -> None:
    run = _mask_run(tmp_path)
    store = RegistrationFeedbackStore(tmp_path / "feedback")
    evidence = registration_feedback_evidence(run, "mask")
    base: dict[str, object] = {
        "cohort": "mouse",
        "stage": "mask",
        "fingerprint": evidence["fingerprint"],
        "slide_id": "HE.ndpi",
        "reviewer": "Reviewer",
        "comment": "",
        "labels": [],
    }

    with pytest.raises(ValueError, match="requires an issue or comment"):
        store.save({**base, "decision": "hold"}, registration_run=run)
    with pytest.raises(ValueError, match="evidence changed"):
        store.save(
            {**base, "decision": "accept", "fingerprint": "stale"},
            registration_run=run,
        )
    with pytest.raises(ValueError, match="invalid or duplicate"):
        store.save(
            {
                **base,
                "decision": "reject",
                "labels": ["not-a-mask-issue"],
            },
            registration_run=run,
        )


def test_order_feedback_accepts_bounded_corrections(tmp_path: Path) -> None:
    run = tmp_path / "mouse"
    run.mkdir()
    (run / "section_order_review.json").write_text(
        json.dumps(
            {
                "fingerprint": "order-fingerprint",
                "slides": [
                    {"slide": "HE.ndpi", "order": 1},
                    {"slide": "CK19.ndpi", "order": 2},
                ],
            }
        )
    )
    store = RegistrationFeedbackStore(tmp_path / "feedback")

    result = store.save(
        {
            "cohort": "mouse",
            "stage": "order",
            "fingerprint": "order-fingerprint",
            "slide_id": "CK19.ndpi",
            "decision": "hold",
            "labels": ["wrong_position", "wrong_orientation"],
            "comment": "Move before H&E and rotate.",
            "reviewer": "Reviewer",
            "suggested_order": 1,
            "suggested_quarter_turns_ccw": 2,
        },
        registration_run=run,
    )

    record = result["feedback"]["CK19.ndpi"]
    assert record["suggested_order"] == 1
    assert record["suggested_quarter_turns_ccw"] == 2


@pytest.mark.browser
def test_mask_review_browser_persists_per_slide_feedback(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    run = tmp_path / "mouse"
    processed = run / "processed"
    processed.mkdir(parents=True)
    image = np.full((80, 100, 3), 245, dtype=np.uint8)
    image[20:65, 20:80] = (150, 90, 75)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:65, 20:80] = 255
    Image.fromarray(image).save(processed / "HE.thumbnail.png")
    Image.fromarray(mask).save(processed / "HE.mask.png")
    (run / "mask_review.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [
                    {
                        "slide": "HE.ndpi",
                        "thumbnail_sha256": "prepared-review",
                        "status": "pending",
                        "method": "object_aware_fusion",
                    }
                ],
            }
        )
    )
    site = tmp_path / "site"
    stable = site / "histopia"
    stable.mkdir(parents=True)
    (stable / "index.html").write_text("stable")
    build_mask_review(run, site / "review")
    feedback_dir = tmp_path / "feedback"
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feedback_dir": str(feedback_dir),
                "cohorts": {"mouse": {"registration": str(run)}},
            }
        )
    )
    token = "a-secure-test-review-token"
    server = create_viewer_server(
        site,
        bind="127.0.0.1",
        port=0,
        review_config=config,
        review_token=token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(
                f"http://127.0.0.1:{server.server_port}/review/",
                wait_until="networkidle",
            )
            page.locator("#feedback-key").fill(token)
            page.locator("#feedback-connect").click()
            page.get_by_text("0/1 reviewed").wait_for()
            assert page.locator("#feedback-reviewer").input_value() == "Web reviewer"
            page.locator("#feedback-reviewer").fill("")
            reject = page.locator("[data-feedback-decision='reject']")
            reject.click()
            assert (
                reject.evaluate(
                    "(element) => getComputedStyle(element).backgroundColor"
                )
                == "rgb(161, 46, 42)"
            )
            page.get_by_text(
                "Reject selected. Enter a reviewer and save this slide review."
            ).wait_for()
            page.locator("#feedback-labels").get_by_text("Extra debris").click()
            page.locator("#feedback-save").click()
            page.get_by_text("Enter a reviewer name before saving.").wait_for()
            assert page.locator("#feedback-reviewer").evaluate(
                "(element) => element === document.activeElement"
            )
            page.locator("#feedback-reviewer").fill("Reviewer")
            page.locator("#feedback-comment").fill("Detached artifact.")
            page.locator("#feedback-save").click()
            page.get_by_text("Slide review saved").wait_for()
            assert page.get_by_text("1/1 reviewed").is_visible()
            assert page.locator("article.feedback-reject").count() == 1
            page.locator("[data-feedback-decision='accept']").click()
            page.get_by_text("Accepted and saved").wait_for()
            assert page.get_by_text("1/1 reviewed").is_visible()
            assert page.locator("article.feedback-accept").count() == 1
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    payloads = load_registration_feedback(feedback_dir)
    assert payloads[0]["records"][0]["labels"] == ["extra_debris"]
    assert payloads[0]["records"][1]["decision"] == "accept"
    assert payloads[0]["records"][1]["labels"] == []
