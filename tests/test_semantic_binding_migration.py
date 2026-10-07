from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import histopia.semantic._binding_migration as migration_module
from histopia.semantic._binding_migration import (
    rebind_semantic_to_registration_approval,
)
from histopia.semantic._preflight import (
    SemanticPreflight,
    SemanticPreflightSlide,
)
from histopia.semantic._result_validation import (
    _seal_semantic_result,
    validate_semantic_result,
)


def test_rebind_semantic_requires_exact_scientific_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic, current = _write_legacy_semantic(tmp_path)
    monkeypatch.setattr(
        migration_module,
        "preflight_registration",
        lambda _run: current,
    )

    migrated = rebind_semantic_to_registration_approval(
        tmp_path / "registration",
        semantic,
    )

    assert migrated.changed is True
    assert migrated.slide_count == 1
    result = validate_semantic_result(semantic)
    assert result["feature_provenance"]["preflight_fingerprint"] == current.fingerprint
    assert result["fingerprint"] == migrated.current_semantic_fingerprint
    review = json.loads((semantic / "semantic_review.json").read_text())
    assert review == {
        "schema_version": 3,
        "approved": False,
        "fingerprint": migrated.current_semantic_fingerprint,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
    }
    assert list(semantic.glob("preflight.superseded-*.json"))
    assert list(semantic.glob("semantic_result.superseded-*.json"))
    record = json.loads(next(semantic.glob("binding_migration-*.json")).read_text())
    assert record["semantic_artifacts_recomputed"] is False
    assert record["review_reset"] is True


def test_rebind_semantic_rejects_changed_transform_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic, current = _write_legacy_semantic(tmp_path)
    original_preflight = (semantic / "preflight.json").read_bytes()
    original_result = (semantic / "semantic_result.json").read_bytes()
    changed_slide = replace(current.slides[0], transform_sha256="9" * 64)
    changed = _replace_preflight_slides(current, (changed_slide,))
    monkeypatch.setattr(
        migration_module,
        "preflight_registration",
        lambda _run: changed,
    )

    with pytest.raises(ValueError, match="scientific identity changed.*transform"):
        rebind_semantic_to_registration_approval(
            tmp_path / "registration",
            semantic,
        )

    assert (semantic / "preflight.json").read_bytes() == original_preflight
    assert (semantic / "semantic_result.json").read_bytes() == original_result
    assert not list(semantic.glob("*.superseded-*.json"))


def test_rebind_semantic_rejects_stale_legacy_preflight(
    tmp_path: Path,
) -> None:
    semantic, _current = _write_legacy_semantic(tmp_path)
    preflight_path = semantic / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["fingerprint"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight))

    with pytest.raises(ValueError, match="preflight fingerprint is stale"):
        rebind_semantic_to_registration_approval(
            tmp_path / "registration",
            semantic,
        )


def _write_legacy_semantic(root: Path) -> tuple[Path, SemanticPreflight]:
    semantic = root / "semantic"
    (semantic / "labels").mkdir(parents=True)
    (semantic / "atlas_model.npz").write_bytes(b"model")
    (semantic / "labels" / "001.npz").write_bytes(b"labels")
    slide = SemanticPreflightSlide(
        slide_name="section.ndpi",
        source_path=str(root / "section.ndpi"),
        source_sha256="1" * 64,
        thumbnail_sha256="2" * 64,
        mask_sha256="3" * 64,
        mask_method="automatic",
        mask_review_status="auto_pass",
        transform_sha256="4" * 64,
        thumbnail_shape=(100, 120),
        mpp_xy=(0.5, 0.5),
        is_reference=True,
    )
    old_slide = {
        key: value
        for key, value in asdict(slide).items()
        if key not in {"mask_method", "mask_review_status"}
    }
    old_core = {
        "schema_version": 1,
        "registration_result_sha256": "5" * 64,
        "order_review_fingerprint": "6" * 64,
        "reference_slide": slide.slide_name,
        "slides": [
            {key: value for key, value in old_slide.items() if key != "source_path"}
        ],
    }
    old_fingerprint = _sha256_json(old_core)
    (semantic / "preflight.json").write_text(
        json.dumps(
            {
                **old_core,
                "registration_run": str(root / "registration"),
                "slides": [old_slide],
                "slide_count": 1,
                "fingerprint": old_fingerprint,
            }
        )
    )
    result = _seal_semantic_result(
        semantic,
        {
            "schema_version": 3,
            "primary_clusters": 2,
            "cluster_counts": [2],
            "selected_k": 2,
            "palette": ["#000000", "#ffffff"],
            "feature_provenance": {
                "preflight_fingerprint": old_fingerprint,
                "expected_slide_ids": [slide.slide_name],
            },
            "model": "atlas_model.npz",
            "slides": [{"id": slide.slide_name, "labels": {"2": "labels/001.npz"}}],
            "topology_pairs": [],
        },
    )
    (semantic / "semantic_result.json").write_text(json.dumps(result))
    (semantic / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": True,
                "fingerprint": result["fingerprint"],
                "reviewer": "Legacy reviewer",
                "reviewed_at": "2026-07-01T00:00:00+00:00",
                "notes": "Reviewed legacy atlas.",
            }
        )
    )
    current = _replace_preflight_slides(
        SemanticPreflight(
            schema_version=3,
            registration_run=str(root / "registration"),
            registration_result_sha256="7" * 64,
            registration_approval_sha256="8" * 64,
            order_review_fingerprint="9" * 64,
            reference_slide=slide.slide_name,
            slides=(slide,),
            fingerprint="0" * 64,
        ),
        (slide,),
    )
    return semantic, current


def _replace_preflight_slides(
    preflight: SemanticPreflight,
    slides: tuple[SemanticPreflightSlide, ...],
) -> SemanticPreflight:
    portable = []
    for slide in slides:
        row = asdict(slide)
        row.pop("source_path")
        portable.append(row)
    core = {
        "schema_version": preflight.schema_version,
        "registration_result_sha256": preflight.registration_result_sha256,
        "registration_approval_sha256": preflight.registration_approval_sha256,
        "order_review_fingerprint": preflight.order_review_fingerprint,
        "reference_slide": preflight.reference_slide,
        "slides": portable,
    }
    return replace(preflight, slides=slides, fingerprint=_sha256_json(core))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
