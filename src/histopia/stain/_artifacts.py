"""Portable continuous stain-map artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class StainMap:
    """Analysis-resolution target, counterstain, residual, and validity maps."""

    slide_id: str
    raw_target_od: np.ndarray
    corrected_target_od: np.ndarray
    counterstain_od: np.ndarray
    reconstruction_residual: np.ndarray
    tissue_mask: np.ndarray
    confidence: np.ndarray
    positive_mask: np.ndarray
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float]
    provenance: dict[str, object]
    fingerprint: str | None = None
    content_fingerprint: str | None = None

    def __post_init__(self) -> None:
        shape = np.asarray(self.raw_target_od).shape
        arrays = (
            self.corrected_target_od,
            self.counterstain_od,
            self.reconstruction_residual,
            self.tissue_mask,
            self.confidence,
            self.positive_mask,
        )
        if len(shape) != 2 or any(np.asarray(value).shape != shape for value in arrays):
            raise ValueError("stain map arrays must share one two-dimensional shape")
        if not self.slide_id or self.analysis_mpp <= 0:
            raise ValueError("slide identity and analysis MPP must be valid")
        for value in (
            self.raw_target_od,
            self.corrected_target_od,
            self.counterstain_od,
            self.reconstruction_residual,
            self.confidence,
        ):
            array = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError("continuous stain maps must be finite and nonnegative")
        tissue = np.asarray(self.tissue_mask, dtype=bool)
        for name, value in (
            ("raw target OD", self.raw_target_od),
            ("corrected target OD", self.corrected_target_od),
            ("counterstain OD", self.counterstain_od),
            ("reconstruction residual", self.reconstruction_residual),
            ("confidence", self.confidence),
        ):
            if np.any(np.asarray(value)[~tissue] != 0):
                raise ValueError(f"{name} must be zero outside the tissue mask")
        if np.any(np.asarray(self.positive_mask, dtype=bool) & ~tissue):
            raise ValueError("positive mask must stay inside the tissue mask")
        expected = _provenance_fingerprint(self.provenance)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("stain map provenance fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)
        content = _content_fingerprint(self)
        if self.content_fingerprint is not None and self.content_fingerprint != content:
            raise ValueError("stain map content fingerprint does not match")
        object.__setattr__(self, "content_fingerprint", content)

    def save(self, path: Path | str) -> Path:
        """Atomically save a compressed schema-1 artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = _stored_arrays(self)
        content = str(self.content_fingerprint)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.int16(1),
                slide_id=np.asarray(self.slide_id),
                **arrays,
                analysis_mpp=np.float64(self.analysis_mpp),
                content_origin_native_xy=np.asarray(
                    self.content_origin_native_xy, dtype=np.int64
                ),
                source_mpp_xy=np.asarray(self.source_mpp_xy, dtype=np.float64),
                provenance_json=np.asarray(_canonical_json(self.provenance)),
                fingerprint=np.asarray(self.fingerprint),
                content_fingerprint=np.asarray(content),
            )
            _validate_archive_header(
                temporary,
                fingerprint=str(self.fingerprint),
                content_fingerprint=content,
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: Path | str) -> StainMap:
        """Load and fully validate a continuous map artifact."""

        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported stain map schema")
            return cls(
                slide_id=str(data["slide_id"]),
                raw_target_od=np.asarray(data["raw_target_od"], dtype=np.float32),
                corrected_target_od=np.asarray(
                    data["corrected_target_od"], dtype=np.float32
                ),
                counterstain_od=np.asarray(data["counterstain_od"], dtype=np.float32),
                reconstruction_residual=np.asarray(
                    data["reconstruction_residual"], dtype=np.float32
                ),
                tissue_mask=np.asarray(data["tissue_mask"], dtype=bool),
                confidence=np.asarray(data["confidence"], dtype=np.float32),
                positive_mask=np.asarray(data["positive_mask"], dtype=bool),
                analysis_mpp=float(data["analysis_mpp"]),
                content_origin_native_xy=tuple(
                    int(value) for value in data["content_origin_native_xy"]
                ),
                source_mpp_xy=tuple(float(value) for value in data["source_mpp_xy"]),
                provenance=json.loads(str(data["provenance_json"])),
                fingerprint=str(data["fingerprint"]),
                content_fingerprint=str(data["content_fingerprint"]),
            )


@dataclass(frozen=True, slots=True)
class AdaptiveStainMap:
    """Fingerprint-bound derived target OD without duplicating physical maps."""

    slide_id: str
    target_od: np.ndarray
    tissue_mask: np.ndarray
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float]
    source_content_fingerprint: str
    method: str
    diagnostics: dict[str, object]
    content_fingerprint: str | None = None

    def __post_init__(self) -> None:
        target = np.asarray(self.target_od, dtype=np.float32)
        tissue = np.asarray(self.tissue_mask, dtype=bool)
        if target.ndim != 2 or target.shape != tissue.shape:
            raise ValueError("adaptive stain values and tissue support must align")
        if not np.all(np.isfinite(target)) or np.any(target < 0):
            raise ValueError("adaptive stain values must be finite and nonnegative")
        if np.any(target[~tissue] != 0):
            raise ValueError("adaptive stain values must be zero outside tissue")
        if not self.slide_id or not self.method or self.analysis_mpp <= 0:
            raise ValueError("adaptive stain identity and resolution are required")
        if len(self.source_content_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_content_fingerprint
        ):
            raise ValueError("adaptive stain source fingerprint is invalid")
        if self.diagnostics.get("method") != self.method:
            raise ValueError("adaptive stain diagnostics method differs")
        expected = _adaptive_content_fingerprint(self)
        if (
            self.content_fingerprint is not None
            and self.content_fingerprint != expected
        ):
            raise ValueError("adaptive stain content fingerprint does not match")
        object.__setattr__(self, "content_fingerprint", expected)

    def save(self, path: Path | str) -> Path:
        """Atomically save a compact schema-1 derived OD artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.int16(1),
                slide_id=np.asarray(self.slide_id),
                target_od=np.asarray(self.target_od, dtype=np.float32),
                tissue_mask=np.asarray(self.tissue_mask, dtype=np.uint8),
                analysis_mpp=np.float64(self.analysis_mpp),
                content_origin_native_xy=np.asarray(
                    self.content_origin_native_xy, dtype=np.int64
                ),
                source_mpp_xy=np.asarray(self.source_mpp_xy, dtype=np.float64),
                source_content_fingerprint=np.asarray(self.source_content_fingerprint),
                method=np.asarray(self.method),
                diagnostics_json=np.asarray(_canonical_json(self.diagnostics)),
                content_fingerprint=np.asarray(self.content_fingerprint),
            )
            _validate_adaptive_archive_header(
                temporary,
                content_fingerprint=str(self.content_fingerprint),
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: Path | str) -> AdaptiveStainMap:
        """Load and fully validate a derived adaptive target map."""

        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported adaptive stain map schema")
            return cls(
                slide_id=str(data["slide_id"]),
                target_od=np.asarray(data["target_od"], dtype=np.float32),
                tissue_mask=np.asarray(data["tissue_mask"], dtype=bool),
                analysis_mpp=float(data["analysis_mpp"]),
                content_origin_native_xy=tuple(
                    int(value) for value in data["content_origin_native_xy"]
                ),
                source_mpp_xy=tuple(float(value) for value in data["source_mpp_xy"]),
                source_content_fingerprint=str(data["source_content_fingerprint"]),
                method=str(data["method"]),
                diagnostics=json.loads(str(data["diagnostics_json"])),
                content_fingerprint=str(data["content_fingerprint"]),
            )


def _stored_arrays(stain_map: StainMap) -> dict[str, np.ndarray]:
    return {
        "raw_target_od": np.asarray(stain_map.raw_target_od, dtype=np.float32),
        "corrected_target_od": np.asarray(
            stain_map.corrected_target_od, dtype=np.float32
        ),
        "counterstain_od": np.asarray(stain_map.counterstain_od, dtype=np.float32),
        "reconstruction_residual": np.asarray(
            stain_map.reconstruction_residual, dtype=np.float32
        ),
        "tissue_mask": np.asarray(stain_map.tissue_mask, dtype=np.uint8),
        "confidence": np.asarray(stain_map.confidence, dtype=np.float32),
        "positive_mask": np.asarray(stain_map.positive_mask, dtype=np.uint8),
    }


def _validate_archive_header(
    path: Path,
    *,
    fingerprint: str,
    content_fingerprint: str,
) -> None:
    required = {
        "schema_version",
        "slide_id",
        *_stored_array_names(),
        "analysis_mpp",
        "content_origin_native_xy",
        "source_mpp_xy",
        "provenance_json",
        "fingerprint",
        "content_fingerprint",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError("stain map archive fields are incomplete")
        if (
            int(data["schema_version"]) != 1
            or str(data["fingerprint"]) != fingerprint
            or str(data["content_fingerprint"]) != content_fingerprint
        ):
            raise ValueError("stain map archive metadata does not match")


def _validate_adaptive_archive_header(
    path: Path,
    *,
    content_fingerprint: str,
) -> None:
    required = {
        "schema_version",
        "slide_id",
        "target_od",
        "tissue_mask",
        "analysis_mpp",
        "content_origin_native_xy",
        "source_mpp_xy",
        "source_content_fingerprint",
        "method",
        "diagnostics_json",
        "content_fingerprint",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError("adaptive stain archive fields are incomplete")
        if (
            int(data["schema_version"]) != 1
            or str(data["content_fingerprint"]) != content_fingerprint
        ):
            raise ValueError("adaptive stain archive metadata does not match")


def _stored_array_names() -> tuple[str, ...]:
    return (
        "raw_target_od",
        "corrected_target_od",
        "counterstain_od",
        "reconstruction_residual",
        "tissue_mask",
        "confidence",
        "positive_mask",
    )


def _content_fingerprint(stain_map: StainMap) -> str:
    digest = hashlib.sha256(b"histopia-stain-map-v1\0")
    metadata = {
        "slide_id": stain_map.slide_id,
        "analysis_mpp": stain_map.analysis_mpp,
        "content_origin_native_xy": list(stain_map.content_origin_native_xy),
        "source_mpp_xy": list(stain_map.source_mpp_xy),
        "provenance": stain_map.provenance,
    }
    digest.update(_canonical_json(metadata).encode())
    for name, value in _stored_arrays(stain_map).items():
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _adaptive_content_fingerprint(stain_map: AdaptiveStainMap) -> str:
    digest = hashlib.sha256(b"histopia-adaptive-stain-map-v1\0")
    metadata = {
        "slide_id": stain_map.slide_id,
        "analysis_mpp": stain_map.analysis_mpp,
        "content_origin_native_xy": list(stain_map.content_origin_native_xy),
        "source_mpp_xy": list(stain_map.source_mpp_xy),
        "source_content_fingerprint": stain_map.source_content_fingerprint,
        "method": stain_map.method,
        "diagnostics": stain_map.diagnostics,
    }
    digest.update(_canonical_json(metadata).encode())
    for name, value in (
        ("target_od", np.asarray(stain_map.target_od, dtype=np.float32)),
        ("tissue_mask", np.asarray(stain_map.tissue_mask, dtype=np.uint8)),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _provenance_fingerprint(provenance: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(provenance).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
