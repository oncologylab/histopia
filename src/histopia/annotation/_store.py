"""Revisioned GeoJSON storage bound to exact workflow results."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.annotation._model import (
    AnnotationOntology,
    default_pancreas_ontology,
)

ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_COORDINATE_SPACE = "source_native_px"
_SECTION_RE = re.compile(r"[0-9]{3,6}")
_PROVENANCE = {"manual", "propagated"}


class AnnotationStore:
    """Persist current annotations and immutable section revision history."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.Lock()

    @classmethod
    def initialize(
        cls,
        root: Path | str,
        *,
        registration_fingerprint: str,
        semantic_fingerprint: str,
        ontology: AnnotationOntology | None = None,
    ) -> AnnotationStore:
        """Create a store or verify that its workflow bindings still match."""

        store = cls(root)
        ontology = ontology or default_pancreas_ontology()
        expected = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "coordinate_space": ANNOTATION_COORDINATE_SPACE,
            "registration_fingerprint": _required_digest(
                "registration_fingerprint", registration_fingerprint
            ),
            "semantic_fingerprint": _required_digest(
                "semantic_fingerprint", semantic_fingerprint
            ),
            "ontology": ontology.as_dict(),
            "sections": {},
        }
        manifest_path = store.root / "annotation_manifest.json"
        if not manifest_path.is_file():
            write_json_atomic(manifest_path, expected, sort_keys=True)
            return store
        manifest = store._manifest()
        for key in (
            "schema_version",
            "coordinate_space",
            "registration_fingerprint",
            "semantic_fingerprint",
            "ontology",
        ):
            if manifest.get(key) != expected[key]:
                raise ValueError(f"annotation store {key} binding is stale")
        return store

    @classmethod
    def from_runs(
        cls,
        root: Path | str,
        *,
        registration_run: Path | str,
        semantic_run: Path | str,
        ontology: AnnotationOntology | None = None,
    ) -> AnnotationStore:
        """Initialize a store from integrity-checked registration and semantic runs."""

        registration_fingerprint = registration_result_fingerprint(registration_run)
        from histopia.semantic._result_validation import validate_semantic_result

        semantic = validate_semantic_result(semantic_run)
        semantic_fingerprint = semantic.get("fingerprint")
        if not isinstance(semantic_fingerprint, str):
            raise ValueError("semantic result fingerprint is missing")
        return cls.initialize(
            root,
            registration_fingerprint=registration_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
            ontology=ontology,
        )

    @classmethod
    def rebind_empty_from_runs(
        cls,
        root: Path | str,
        *,
        registration_run: Path | str,
        semantic_run: Path | str,
    ) -> AnnotationStore:
        """Explicitly rebind an unused store to validated current results.

        Rebinding is intentionally refused after any annotation revision has
        been written, because existing native coordinates and semantic context
        must remain bound to their original workflow fingerprints.
        """

        registration_fingerprint = registration_result_fingerprint(registration_run)
        from histopia.semantic._result_validation import validate_semantic_result

        semantic = validate_semantic_result(semantic_run)
        semantic_fingerprint = semantic.get("fingerprint")
        if not isinstance(semantic_fingerprint, str):
            raise ValueError("semantic result fingerprint is missing")
        return cls(root).rebind_empty(
            registration_fingerprint=registration_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )

    def rebind_empty(
        self,
        *,
        registration_fingerprint: str,
        semantic_fingerprint: str,
    ) -> AnnotationStore:
        """Rebind a revision-zero store without discarding annotation data."""

        registration_fingerprint = _required_digest(
            "registration_fingerprint", registration_fingerprint
        )
        semantic_fingerprint = _required_digest(
            "semantic_fingerprint", semantic_fingerprint
        )
        with self._lock:
            manifest = self._manifest()
            sections = manifest.get("sections")
            if sections or self._has_annotation_files():
                raise ValueError("non-empty annotation stores cannot be rebound")
            manifest["registration_fingerprint"] = registration_fingerprint
            manifest["semantic_fingerprint"] = semantic_fingerprint
            write_json_atomic(
                self.root / "annotation_manifest.json",
                manifest,
                sort_keys=True,
            )
        return self

    def catalog(self) -> dict[str, object]:
        """Return path-free ontology, binding, and section revision metadata."""

        manifest = self._manifest()
        return {
            "schema_version": manifest["schema_version"],
            "coordinate_space": manifest["coordinate_space"],
            "registration_fingerprint": manifest["registration_fingerprint"],
            "semantic_fingerprint": manifest["semantic_fingerprint"],
            "ontology": manifest["ontology"],
            "sections": manifest["sections"],
        }

    def read_section(self, section: str) -> dict[str, object]:
        """Read the current section or return a revision-zero collection."""

        section = _required_section(section)
        path = self.root / "sections" / f"{section}.geojson"
        if path.is_file():
            payload = json.loads(path.read_text())
            _validate_saved_collection(payload, section, self._manifest())
            return payload
        return self._empty_collection(section)

    def save_section(
        self,
        section: str,
        feature_collection: object,
        *,
        reviewer: str,
        expected_revision: int,
        notes: str = "",
    ) -> dict[str, object]:
        """Save one exact revision, rejecting stale browser state."""

        section = _required_section(section)
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("annotation reviewer must not be blank")
        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision, int
        ):
            raise TypeError("annotation expected revision must be an integer")
        if expected_revision < 0:
            raise ValueError("annotation expected revision must be non-negative")
        if not isinstance(notes, str):
            raise TypeError("annotation notes must be text")
        with self._lock:
            manifest = self._manifest()
            sections = manifest.get("sections")
            assert isinstance(sections, dict)
            previous = sections.get(section)
            current_revision = (
                previous.get("revision", 0) if isinstance(previous, dict) else 0
            )
            if current_revision != expected_revision:
                raise ValueError(
                    f"annotation revision conflict: expected {expected_revision}, "
                    f"current {current_revision}"
                )
            features = _normalize_features(
                feature_collection,
                AnnotationOntology.from_dict(manifest["ontology"]),
            )
            revision = current_revision + 1
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            core: dict[str, object] = {
                "type": "FeatureCollection",
                "histopia": {
                    "schema_version": ANNOTATION_SCHEMA_VERSION,
                    "section": section,
                    "revision": revision,
                    "coordinate_space": ANNOTATION_COORDINATE_SPACE,
                    "registration_fingerprint": manifest["registration_fingerprint"],
                    "semantic_fingerprint": manifest["semantic_fingerprint"],
                    "reviewer": reviewer,
                    "updated_at": timestamp,
                    "notes": notes.strip(),
                },
                "features": features,
            }
            fingerprint = _json_fingerprint(core)
            metadata = core["histopia"]
            assert isinstance(metadata, dict)
            metadata["fingerprint"] = fingerprint
            history_path = self.root / "history" / section / f"{revision:06d}.geojson"
            current_path = self.root / "sections" / f"{section}.geojson"
            if history_path.exists():
                raise ValueError("annotation history revision already exists")
            write_json_atomic(history_path, core, sort_keys=True)
            write_json_atomic(current_path, core, sort_keys=True)
            sections[section] = {
                "revision": revision,
                "fingerprint": fingerprint,
                "feature_count": len(features),
                "reviewer": reviewer,
                "updated_at": timestamp,
            }
            write_json_atomic(
                self.root / "annotation_manifest.json",
                manifest,
                sort_keys=True,
            )
            return core

    def _empty_collection(self, section: str) -> dict[str, object]:
        manifest = self._manifest()
        return {
            "type": "FeatureCollection",
            "histopia": {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "section": section,
                "revision": 0,
                "coordinate_space": ANNOTATION_COORDINATE_SPACE,
                "registration_fingerprint": manifest["registration_fingerprint"],
                "semantic_fingerprint": manifest["semantic_fingerprint"],
            },
            "features": [],
        }

    def _manifest(self) -> dict[str, object]:
        path = self.root / "annotation_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("annotation manifest must be an object")
        if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError("annotation manifest schema version is unsupported")
        if payload.get("coordinate_space") != ANNOTATION_COORDINATE_SPACE:
            raise ValueError("annotation manifest coordinate space is unsupported")
        _required_digest(
            "registration_fingerprint", payload.get("registration_fingerprint")
        )
        _required_digest("semantic_fingerprint", payload.get("semantic_fingerprint"))
        AnnotationOntology.from_dict(payload.get("ontology"))
        if not isinstance(payload.get("sections"), dict):
            raise ValueError("annotation manifest sections must be an object")
        return payload

    def _has_annotation_files(self) -> bool:
        return any(
            path.is_file()
            for directory in (self.root / "sections", self.root / "history")
            if directory.is_dir()
            for path in directory.rglob("*")
        )


def registration_result_fingerprint(registration_run: Path | str) -> str:
    """Hash the exact final registration result bytes used by annotations."""

    path = Path(registration_run) / "registration_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_features(
    payload: object,
    ontology: AnnotationOntology,
) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("annotations must be a GeoJSON FeatureCollection")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("annotation features must be a list")
    normalized: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for raw in raw_features:
        if not isinstance(raw, dict) or raw.get("type") != "Feature":
            raise ValueError("each annotation must be a GeoJSON Feature")
        properties = raw.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("annotation properties must be an object")
        class_id = properties.get("class_id")
        if not isinstance(class_id, str) or class_id not in ontology.class_ids:
            raise ValueError(f"unknown annotation class: {class_id!r}")
        annotation_id = properties.get("annotation_id")
        if annotation_id is None:
            annotation_id = str(uuid.uuid4())
        if not isinstance(annotation_id, str) or not annotation_id.strip():
            raise ValueError("annotation id must not be blank")
        if annotation_id in identifiers:
            raise ValueError("annotation ids must be unique within a section")
        identifiers.add(annotation_id)
        confidence = properties.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("annotation confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("annotation confidence must be between zero and one")
        provenance = properties.get("provenance", "manual")
        if provenance not in _PROVENANCE:
            raise ValueError("annotation provenance must be manual or propagated")
        accepted = properties.get("accepted", provenance == "manual")
        if not isinstance(accepted, bool):
            raise TypeError("annotation accepted must be a boolean")
        source_section = properties.get("source_section")
        if source_section is not None:
            source_section = _required_section(source_section)
        geometry = _normalize_geometry(raw.get("geometry"))
        normalized.append(
            {
                "type": "Feature",
                "properties": {
                    "annotation_id": annotation_id,
                    "class_id": class_id,
                    "confidence": confidence,
                    "provenance": provenance,
                    "accepted": accepted,
                    **(
                        {"source_section": source_section}
                        if source_section is not None
                        else {}
                    ),
                },
                "geometry": geometry,
            }
        )
    return normalized


def _normalize_geometry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("annotation geometry must be an object")
    geometry_type = payload.get("type")
    coordinates = payload.get("coordinates")
    if geometry_type == "Polygon":
        normalized = _normalize_polygon(coordinates)
    elif geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("annotation multipolygon must not be empty")
        normalized = [_normalize_polygon(polygon) for polygon in coordinates]
    else:
        raise ValueError("annotation geometry must be Polygon or MultiPolygon")
    return {"type": geometry_type, "coordinates": normalized}


def _normalize_polygon(payload: object) -> list[list[list[float]]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("annotation polygon must contain at least one ring")
    rings: list[list[list[float]]] = []
    for raw_ring in payload:
        if not isinstance(raw_ring, list) or len(raw_ring) < 4:
            raise ValueError("annotation polygon rings require at least four points")
        ring: list[list[float]] = []
        for point in raw_ring:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("annotation coordinates must be x,y pairs")
            x, y = point
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
            ):
                raise TypeError("annotation coordinates must be numeric")
            x, y = float(x), float(y)
            if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
                raise ValueError(
                    "annotation coordinates must be finite and non-negative"
                )
            ring.append([x, y])
        if ring[0] != ring[-1]:
            raise ValueError("annotation polygon rings must be closed")
        rings.append(ring)
    return rings


def _validate_saved_collection(
    payload: object,
    section: str,
    manifest: dict[str, object],
) -> None:
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("saved annotations must be a GeoJSON FeatureCollection")
    metadata = payload.get("histopia")
    if not isinstance(metadata, dict):
        raise ValueError("saved annotation metadata is missing")
    for key in (
        "schema_version",
        "coordinate_space",
        "registration_fingerprint",
        "semantic_fingerprint",
    ):
        if metadata.get(key) != manifest.get(key):
            raise ValueError(f"saved annotation {key} binding is stale")
    if metadata.get("section") != section:
        raise ValueError("saved annotation section is stale")
    fingerprint = metadata.get("fingerprint")
    core = json.loads(json.dumps(payload))
    core_metadata = core["histopia"]
    del core_metadata["fingerprint"]
    if fingerprint != _json_fingerprint(core):
        raise ValueError("saved annotation fingerprint is stale")
    _normalize_features(payload, AnnotationOntology.from_dict(manifest["ontology"]))


def _required_section(value: object) -> str:
    if not isinstance(value, str) or not _SECTION_RE.fullmatch(value):
        raise ValueError("annotation section must contain 3 to 6 digits")
    return value


def _required_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _json_fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
