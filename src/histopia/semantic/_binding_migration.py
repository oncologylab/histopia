"""Fail-closed migration of legacy semantic registration bindings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.semantic._preflight import (
    SemanticPreflight,
    preflight_registration,
)
from histopia.semantic._registration_binding import (
    validate_semantic_registration_binding,
)
from histopia.semantic._result_validation import (
    _fingerprint_core,
    validate_semantic_result,
    validate_semantic_result_index,
)

_SCIENTIFIC_SLIDE_FIELDS = (
    "slide_name",
    "source_sha256",
    "thumbnail_sha256",
    "mask_sha256",
    "transform_sha256",
    "thumbnail_shape",
    "mpp_xy",
    "is_reference",
)


@dataclass(frozen=True, slots=True)
class SemanticBindingMigration:
    """Audit summary for one metadata-only semantic binding migration."""

    semantic_run: Path
    slide_count: int
    previous_preflight_fingerprint: str
    current_preflight_fingerprint: str
    previous_semantic_fingerprint: str
    current_semantic_fingerprint: str
    changed: bool


def rebind_semantic_to_registration_approval(
    registration_run: Path | str,
    semantic_run: Path | str,
) -> SemanticBindingMigration:
    """Bind an unchanged legacy atlas to a newly sealed registration approval.

    This migration is intentionally narrower than recomputation. It is allowed
    only when the old semantic preflight is internally valid and every field
    that can affect extracted features or physical placement is identical to a
    fresh preflight of the approved registration. Source slides, thumbnails,
    masks, transforms, dimensions, MPP, order, and reference identity are all
    compared exactly. Semantic arrays are never recomputed or modified.

    The semantic result receives a new metadata fingerprint and its review is
    reset to pending. Superseded JSON records are retained for audit and
    recovery. Any scientific input change fails closed and requires normal
    feature extraction and atlas fitting.
    """

    registration_root = Path(registration_run).expanduser().resolve()
    semantic_root = Path(semantic_run).expanduser().resolve()
    preflight_path = semantic_root / "preflight.json"
    result_path = semantic_root / "semantic_result.json"
    review_path = semantic_root / "semantic_review.json"

    old_preflight_text = preflight_path.read_text(encoding="utf-8")
    old_result_text = result_path.read_text(encoding="utf-8")
    old_review_text = review_path.read_text(encoding="utf-8")
    old_preflight = _json_object(old_preflight_text, "semantic preflight")
    old_result = validate_semantic_result(semantic_root)
    old_review = _json_object(old_review_text, "semantic review")
    _validate_preflight_fingerprint(old_preflight)

    old_preflight_fingerprint = _required_digest(
        old_preflight,
        "fingerprint",
        "semantic preflight",
    )
    old_semantic_fingerprint = _required_digest(
        old_result,
        "fingerprint",
        "semantic result",
    )
    provenance = old_result.get("feature_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("preflight_fingerprint") != old_preflight_fingerprint
    ):
        raise ValueError("semantic result preflight fingerprint is stale")

    try:
        existing = validate_semantic_registration_binding(
            registration_root,
            semantic_root,
            semantic_payload=old_result,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        existing = None
    if existing is not None and existing.approval_bound:
        return SemanticBindingMigration(
            semantic_run=semantic_root,
            slide_count=len(_preflight_slides(old_preflight)),
            previous_preflight_fingerprint=old_preflight_fingerprint,
            current_preflight_fingerprint=old_preflight_fingerprint,
            previous_semantic_fingerprint=old_semantic_fingerprint,
            current_semantic_fingerprint=old_semantic_fingerprint,
            changed=False,
        )
    if old_preflight.get("schema_version") not in {1, 2}:
        raise ValueError(
            "only self-consistent schema-1/2 semantic preflights can be migrated"
        )

    current = preflight_registration(registration_root)
    current_payload = _preflight_payload(current)
    _validate_preflight_fingerprint(current_payload)
    old_slides = _preflight_slides(old_preflight)
    current_slides = _preflight_slides(current_payload)
    _require_unchanged_scientific_identity(
        old_preflight,
        old_slides,
        current_payload,
        current_slides,
    )
    _require_semantic_slide_identity(old_result, current_slides)

    new_preflight_fingerprint = _required_digest(
        current_payload,
        "fingerprint",
        "current semantic preflight",
    )
    new_result = dict(old_result)
    new_provenance = dict(provenance)
    new_provenance["preflight_fingerprint"] = new_preflight_fingerprint
    new_result["feature_provenance"] = new_provenance
    new_result.pop("fingerprint", None)
    new_result["fingerprint"] = _fingerprint_core(new_result)
    validate_semantic_result_index(semantic_root, new_result)
    new_semantic_fingerprint = _required_digest(
        new_result,
        "fingerprint",
        "migrated semantic result",
    )
    new_review = {
        "schema_version": 3,
        "approved": False,
        "fingerprint": new_semantic_fingerprint,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
    }

    _write_backup_once(
        semantic_root / f"preflight.superseded-{old_preflight_fingerprint[:12]}.json",
        old_preflight_text,
    )
    _write_backup_once(
        semantic_root
        / f"semantic_result.superseded-{old_semantic_fingerprint[:12]}.json",
        old_result_text,
    )
    _write_backup_once(
        semantic_root
        / f"semantic_review.superseded-{old_semantic_fingerprint[:12]}.json",
        old_review_text,
    )

    try:
        write_json_atomic(preflight_path, current_payload)
        write_json_atomic(result_path, new_result)
        write_json_atomic(review_path, new_review)
        validate_semantic_result(semantic_root)
    except BaseException:
        write_text_atomic(preflight_path, old_preflight_text)
        write_text_atomic(result_path, old_result_text)
        write_text_atomic(review_path, old_review_text)
        raise

    migration = {
        "schema_version": 1,
        "algorithm": "exact-scientific-identity-registration-rebind-v1",
        "migrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slide_count": len(current_slides),
        "scientific_identity_fields": list(_SCIENTIFIC_SLIDE_FIELDS),
        "semantic_artifacts_recomputed": False,
        "review_reset": True,
        "previous_preflight_fingerprint": old_preflight_fingerprint,
        "current_preflight_fingerprint": new_preflight_fingerprint,
        "previous_semantic_fingerprint": old_semantic_fingerprint,
        "current_semantic_fingerprint": new_semantic_fingerprint,
        "previous_review": {
            "approved": old_review.get("approved"),
            "reviewer": old_review.get("reviewer"),
            "reviewed_at": old_review.get("reviewed_at"),
        },
    }
    write_json_atomic(
        semantic_root / f"binding_migration-{new_preflight_fingerprint[:12]}.json",
        migration,
    )
    return SemanticBindingMigration(
        semantic_run=semantic_root,
        slide_count=len(current_slides),
        previous_preflight_fingerprint=old_preflight_fingerprint,
        current_preflight_fingerprint=new_preflight_fingerprint,
        previous_semantic_fingerprint=old_semantic_fingerprint,
        current_semantic_fingerprint=new_semantic_fingerprint,
        changed=True,
    )


def _preflight_payload(preflight: SemanticPreflight) -> dict[str, object]:
    payload = asdict(preflight)
    payload["slides"] = [asdict(slide) for slide in preflight.slides]
    payload["slide_count"] = preflight.slide_count
    return payload


def _require_unchanged_scientific_identity(
    old_preflight: dict[str, object],
    old_slides: list[dict[str, object]],
    current_preflight: dict[str, object],
    current_slides: list[dict[str, object]],
) -> None:
    if old_preflight.get("reference_slide") != current_preflight.get("reference_slide"):
        raise ValueError("semantic registration reference slide changed")
    if len(old_slides) != len(current_slides):
        raise ValueError("semantic registration slide count changed")
    for index, (old, current) in enumerate(
        zip(old_slides, current_slides, strict=True),
        start=1,
    ):
        for field in _SCIENTIFIC_SLIDE_FIELDS:
            if field not in old:
                raise ValueError(
                    f"legacy semantic preflight lacks {field} for slide {index}"
                )
            if _normalized(old[field]) != _normalized(current.get(field)):
                name = str(old.get("slide_name") or index)
                raise ValueError(
                    f"semantic scientific identity changed for {name}: {field}"
                )


def _require_semantic_slide_identity(
    result: dict[str, object],
    current_slides: list[dict[str, object]],
) -> None:
    expected = [str(row.get("slide_name", "")) for row in current_slides]
    rows = result.get("slides")
    if not isinstance(rows, list):
        raise ValueError("semantic result contains no slide records")
    actual = [str(row.get("id", "")) for row in rows if isinstance(row, dict)]
    provenance = result.get("feature_provenance")
    provenance_ids = (
        provenance.get("expected_slide_ids") if isinstance(provenance, dict) else None
    )
    if actual != expected or provenance_ids != expected:
        raise ValueError(
            "semantic result slide order differs from current registration"
        )


def _validate_preflight_fingerprint(payload: dict[str, object]) -> None:
    schema = payload.get("schema_version")
    if schema not in {1, 2, 3}:
        raise ValueError("semantic preflight schema is unsupported")
    slides = _preflight_slides(payload)
    portable = [
        {key: value for key, value in row.items() if key != "source_path"}
        for row in slides
    ]
    core: dict[str, object] = {
        "schema_version": schema,
        "registration_result_sha256": payload.get("registration_result_sha256"),
        "order_review_fingerprint": payload.get("order_review_fingerprint"),
        "reference_slide": payload.get("reference_slide"),
        "slides": portable,
    }
    if schema == 3:
        core["registration_approval_sha256"] = payload.get(
            "registration_approval_sha256"
        )
    expected = _required_digest(payload, "fingerprint", "semantic preflight")
    if _sha256_json(core) != expected:
        raise ValueError("semantic preflight fingerprint is stale")


def _preflight_slides(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload.get("slides")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("semantic preflight contains no valid slides")
    return rows


def _required_digest(
    payload: dict[str, object],
    key: str,
    label: str,
) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} {key} is invalid")
    return value


def _normalized(value: object) -> object:
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_object(text: str, label: str) -> dict[str, object]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _write_backup_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"semantic migration backup differs: {path.name}")
        return
    write_text_atomic(path, text)
