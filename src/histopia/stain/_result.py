"""Writing sealed stain results and fingerprint-bound review state."""

from __future__ import annotations

import json
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.stain._result_validation import _seal_stain_result


def write_stain_result(
    output_dir: Path,
    core: dict[str, object],
) -> Path:
    """Seal artifacts, write the result, and preserve only current review state."""

    payload = _seal_stain_result(output_dir, core)
    path = output_dir / "stain_result.json"
    write_json_atomic(path, payload)
    review_path = output_dir / "stain_review.json"
    review = _current_review(review_path, str(payload["fingerprint"]))
    write_json_atomic(review_path, review)
    return path


def _current_review(path: Path, fingerprint: str) -> dict[str, object]:
    default: dict[str, object] = {
        "schema_version": 1,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
    }
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("fingerprint") != fingerprint
    ):
        return default
    return payload
