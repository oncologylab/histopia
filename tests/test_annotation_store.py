from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from histopia.annotation import (
    AnnotationClass,
    AnnotationOntology,
    AnnotationStore,
    default_pancreas_ontology,
    registration_result_fingerprint,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _collection(*, class_id: str = "epithelial") -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"class_id": class_id},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[1, 1], [8, 1], [8, 9], [1, 1]],
                    ],
                },
            }
        ],
    }


def test_default_ontology_is_stable_and_validated() -> None:
    ontology = default_pancreas_ontology()

    assert ontology.name == "histopia-pancreas-broad"
    assert "uncertain" in ontology.class_ids
    assert AnnotationOntology.from_dict(ontology.as_dict()) == ontology
    with pytest.raises(ValueError, match="unique"):
        AnnotationOntology(
            "invalid",
            (
                AnnotationClass("tissue", "Tissue", "#112233"),
                AnnotationClass("tissue", "Other", "#445566"),
            ),
        )


def test_annotation_store_preserves_revisions_and_path_free_catalog(
    tmp_path: Path,
) -> None:
    store = AnnotationStore.initialize(
        tmp_path,
        registration_fingerprint=_digest("registration"),
        semantic_fingerprint=_digest("semantic"),
    )

    empty = store.read_section("003")
    assert empty["histopia"]["revision"] == 0
    saved = store.save_section(
        "003",
        _collection(),
        reviewer="pathologist",
        expected_revision=0,
        notes="broad lesion outline",
    )

    assert saved["histopia"]["revision"] == 1
    assert saved["features"][0]["properties"]["provenance"] == "manual"
    assert saved["features"][0]["properties"]["accepted"] is True
    assert (tmp_path / "sections" / "003.geojson").is_file()
    assert (tmp_path / "history" / "003" / "000001.geojson").is_file()
    assert store.read_section("003") == saved
    catalog = store.catalog()
    assert catalog["sections"]["003"]["revision"] == 1
    assert str(tmp_path) not in json.dumps(catalog)


def test_annotation_store_rejects_stale_revisions_and_bindings(tmp_path: Path) -> None:
    registration = _digest("registration")
    semantic = _digest("semantic")
    store = AnnotationStore.initialize(
        tmp_path,
        registration_fingerprint=registration,
        semantic_fingerprint=semantic,
    )
    store.save_section(
        "001",
        _collection(),
        reviewer="reviewer",
        expected_revision=0,
    )

    with pytest.raises(ValueError, match="revision conflict"):
        store.save_section(
            "001",
            _collection(),
            reviewer="reviewer",
            expected_revision=0,
        )
    with pytest.raises(ValueError, match="semantic_fingerprint binding is stale"):
        AnnotationStore.initialize(
            tmp_path,
            registration_fingerprint=registration,
            semantic_fingerprint=_digest("changed"),
        )


def test_annotation_store_explicitly_rebinds_only_before_first_revision(
    tmp_path: Path,
) -> None:
    store = AnnotationStore.initialize(
        tmp_path,
        registration_fingerprint=_digest("registration-old"),
        semantic_fingerprint=_digest("semantic-old"),
    )

    returned = store.rebind_empty(
        registration_fingerprint=_digest("registration-new"),
        semantic_fingerprint=_digest("semantic-new"),
    )

    assert returned is store
    assert store.catalog()["registration_fingerprint"] == _digest("registration-new")
    assert store.catalog()["semantic_fingerprint"] == _digest("semantic-new")
    store.save_section(
        "001",
        _collection(),
        reviewer="reviewer",
        expected_revision=0,
    )
    with pytest.raises(ValueError, match="non-empty"):
        store.rebind_empty(
            registration_fingerprint=_digest("registration-later"),
            semantic_fingerprint=_digest("semantic-later"),
        )


def test_annotation_store_never_overwrites_history(tmp_path: Path) -> None:
    store = AnnotationStore.initialize(
        tmp_path,
        registration_fingerprint=_digest("registration"),
        semantic_fingerprint=_digest("semantic"),
    )
    history = tmp_path / "history" / "001" / "000001.geojson"
    history.parent.mkdir(parents=True)
    history.write_text("immutable")

    with pytest.raises(ValueError, match="history revision already exists"):
        store.save_section(
            "001",
            _collection(),
            reviewer="reviewer",
            expected_revision=0,
        )
    assert history.read_text() == "immutable"


def test_annotation_store_validates_geometry_and_classes(tmp_path: Path) -> None:
    store = AnnotationStore.initialize(
        tmp_path,
        registration_fingerprint=_digest("registration"),
        semantic_fingerprint=_digest("semantic"),
    )

    with pytest.raises(ValueError, match="unknown annotation class"):
        store.save_section(
            "001",
            _collection(class_id="tumor"),
            reviewer="reviewer",
            expected_revision=0,
        )
    invalid = _collection()
    invalid["features"][0]["geometry"]["coordinates"][0][-1] = [2, 2]
    with pytest.raises(ValueError, match="must be closed"):
        store.save_section(
            "001",
            invalid,
            reviewer="reviewer",
            expected_revision=0,
        )


def test_registration_result_fingerprint_hashes_exact_bytes(tmp_path: Path) -> None:
    payload = b'{"schema_version":1}\n'
    (tmp_path / "registration_result.json").write_bytes(payload)

    assert (
        registration_result_fingerprint(tmp_path) == hashlib.sha256(payload).hexdigest()
    )


def test_annotation_store_from_runs_validates_semantic_artifacts(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    registration.mkdir()
    semantic.mkdir()
    (registration / "registration_result.json").write_text("{}\n")
    (semantic / "model.npz").write_bytes(b"model")
    (semantic / "labels.npz").write_bytes(b"labels")
    core: dict[str, object] = {
        "schema_version": 3,
        "model": "model.npz",
        "slides": [{"id": "slide", "labels": {"5": "labels.npz"}}],
        "topology_pairs": [],
        "artifacts": {
            "model.npz": hashlib.sha256(b"model").hexdigest(),
            "labels.npz": hashlib.sha256(b"labels").hexdigest(),
        },
    }
    payload = {
        **core,
        "fingerprint": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (semantic / "semantic_result.json").write_text(json.dumps(payload))

    store = AnnotationStore.from_runs(
        tmp_path / "annotations",
        registration_run=registration,
        semantic_run=semantic,
    )

    assert store.catalog()["semantic_fingerprint"] == payload["fingerprint"]
    (semantic / "labels.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        AnnotationStore.from_runs(
            tmp_path / "other-annotations",
            registration_run=registration,
            semantic_run=semantic,
        )
