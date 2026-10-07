import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from histopia.cells._cli import _build_parser
from histopia.cells._reviewed_exclusion import (
    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION,
    REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
    reviewed_artifact_exclusion_manifest_sha256,
    reviewed_artifact_exclusion_qc_evidence,
    reviewed_artifact_exclusion_section_payload,
    validate_reviewed_artifact_exclusion_manifest,
    validate_reviewed_artifact_exclusion_section_payload,
)
from histopia.cells._reviewed_exclusion_refilter import (
    _write_filtered_raw_and_measure,
    refilter_reviewed_artifact_exclusions,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest() -> dict[str, object]:
    label_ids = [3, 7, 11]
    return {
        "schema_version": 1,
        "scope": REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
        "source_algorithm_version": 75,
        "source_result_fingerprint": "a" * 64,
        "source_preflight_fingerprint": "b" * 64,
        "sections": [
            {
                "section": "003",
                "source_identity": "c" * 64,
                "source_labels_sha256": "d" * 64,
                "label_ids": label_ids,
                "label_ids_sha256": _sha(label_ids),
                "selection": {
                    "instance_action": REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION,
                    "regions": [
                        {
                            "region_id": "fold-1",
                            "kind": "unrecoverable-focus-fold",
                            "native_coordinate_space": "source_wsi_pixels",
                            "geometry": {"bbox_xyxy": [10, 20, 30, 40]},
                            "selection_rule": "whole labels touching reviewed region",
                            "parameters": {"maximum_intensity": 145.0},
                        }
                    ],
                },
                "review": {
                    "decision": "approved",
                    "reviewer": "visual-audit",
                    "reviewed_at": "2026-08-31T20:00:00+00:00",
                    "evidence_sha256s": ["e" * 64],
                    "notes": "Detached focus fold only.",
                },
            }
        ],
    }


def test_reviewed_exclusion_manifest_is_exact_path_free_and_digestible() -> None:
    manifest = _manifest()

    sections = validate_reviewed_artifact_exclusion_manifest(
        manifest,
        expected_sections=("003",),
    )

    assert list(sections) == ["003"]
    assert reviewed_artifact_exclusion_manifest_sha256(manifest) == _sha(manifest)
    evidence = reviewed_artifact_exclusion_qc_evidence(manifest, "003")
    assert evidence["manifest_sha256"] == _sha(manifest)
    assert evidence["label_ids_sha256"] == _sha([3, 7, 11])


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("duplicate", "label IDs"),
        ("unsorted", "label IDs"),
        ("stale_digest", "label IDs"),
        ("path", "path-free"),
        ("decision", "decision"),
    ],
)
def test_reviewed_exclusion_manifest_rejects_ambiguous_or_stale_data(
    mutation: str,
    message: str,
) -> None:
    manifest = _manifest()
    section = manifest["sections"][0]  # type: ignore[index]
    if mutation == "duplicate":
        section["label_ids"] = [3, 3]  # type: ignore[index]
    elif mutation == "unsorted":
        section["label_ids"] = [7, 3]  # type: ignore[index]
    elif mutation == "stale_digest":
        section["label_ids_sha256"] = "f" * 64  # type: ignore[index]
    elif mutation == "path":
        section["selection"]["regions"][0]["parameters"]["source_path"] = "/tmp/x"  # type: ignore[index]
    else:
        section["review"]["decision"] = "pending"  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        validate_reviewed_artifact_exclusion_manifest(manifest)


def test_reviewed_exclusion_section_payload_retains_source_seal() -> None:
    manifest = _manifest()

    payload = reviewed_artifact_exclusion_section_payload(manifest, "003")
    validated = validate_reviewed_artifact_exclusion_section_payload(
        payload,
        expected_section="003",
    )

    assert validated["source_result_fingerprint"] == "a" * 64
    assert validated["section"]["label_ids"] == [3, 7, 11]  # type: ignore[index]


def test_reviewed_exclusion_accepts_reviewed_addition_source_only() -> None:
    manifest = _manifest()
    manifest["source_algorithm_version"] = (
        REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
    )

    validate_reviewed_artifact_exclusion_manifest(manifest)

    manifest["source_algorithm_version"] = 96
    with pytest.raises(ValueError, match="source seal"):
        validate_reviewed_artifact_exclusion_manifest(manifest)


class _ArrayImage:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def crop(self, x: int, y: int, width: int, height: int) -> "_ArrayImage":
        return _ArrayImage(self.values[y : y + height, x : x + width])

    def write_to_memory(self) -> bytes:
        return np.asarray(self.values, dtype=np.uint32).tobytes()


def test_reviewed_exclusion_filter_removes_exact_whole_labels(tmp_path: Path) -> None:
    labels = np.array(
        [
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [3, 0, 4, 4],
        ],
        dtype=np.uint32,
    )
    output = tmp_path / "labels.raw"

    areas = _write_filtered_raw_and_measure(
        _ArrayImage(labels),
        output,
        np.array([2, 4], dtype=np.uint32),
        width=4,
        height=3,
        expected_cells=4,
        progress=None,
        section="003",
    )
    filtered = np.memmap(output, mode="r", dtype=np.uint32, shape=labels.shape)

    assert areas.tolist()[:5] == [2, 2, 3, 3, 2]
    assert np.asarray(filtered).tolist() == [
        [0, 1, 1, 0],
        [3, 3, 0, 0],
        [3, 0, 0, 0],
    ]


def test_reviewed_exclusion_filter_rejects_absent_label(tmp_path: Path) -> None:
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint32)

    with pytest.raises(ValueError, match="absent from source"):
        _write_filtered_raw_and_measure(
            _ArrayImage(labels),
            tmp_path / "labels.raw",
            np.array([3], dtype=np.uint32),
            width=2,
            height=2,
            expected_cells=2,
            progress=None,
            section="003",
        )


def test_reviewed_exclusion_cli_accepts_explicit_cpsam_v2_source() -> None:
    args = _build_parser().parse_args(
        [
            "reviewed-exclusion-refilter",
            "--source-run",
            "source",
            "--output",
            "output",
            "--spec",
            "spec.json",
            "--model",
            "cpsam_v2",
        ]
    )

    assert args.model == "cpsam_v2"


def test_reviewed_exclusion_rejects_unknown_source_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source model is unsupported"):
        refilter_reviewed_artifact_exclusions(
            tmp_path / "source",
            tmp_path / "output",
            tmp_path / "spec.json",
            expected_model="not-cellpose",
        )
