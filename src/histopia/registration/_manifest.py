"""Local dataset manifest helpers for registration validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RAW_EXTENSIONS = {".ndpi", ".scn"}
REGISTERED_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


@dataclass(frozen=True, slots=True)
class SlidePair:
    """A raw slide and its existing registered reference."""

    raw_path: Path
    reference_path: Path
    key: str


@dataclass(frozen=True, slots=True)
class KpfManifest:
    """Manifest for one KPF mouse validation folder."""

    mouse_dir: Path
    raw_dir: Path
    registered_dir: Path
    pairs: tuple[SlidePair, ...]
    missing_raw_keys: tuple[str, ...]
    missing_reference_keys: tuple[str, ...]
    ambiguous_keys: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not (
            self.missing_raw_keys or self.missing_reference_keys or self.ambiguous_keys
        )


def build_kpf_manifest(mouse_dir: Path | str) -> KpfManifest:
    """Build a non-mutating raw/reference manifest for a KPF mouse folder."""

    mouse_dir = Path(mouse_dir)
    raw_dir = mouse_dir / "raw_wsi"
    registered_dir = mouse_dir / "registered"

    raw_by_key = _paths_by_key(_iter_raw_paths(raw_dir))
    reference_by_key = _paths_by_key(_iter_registered_paths(registered_dir))
    raw_keys = set(raw_by_key)
    reference_keys = set(reference_by_key)

    ambiguous = sorted(
        key
        for key in raw_keys | reference_keys
        if len(raw_by_key.get(key, ())) > 1 or len(reference_by_key.get(key, ())) > 1
    )
    paired_keys = sorted((raw_keys & reference_keys) - set(ambiguous), key=_natural_key)
    pairs = tuple(
        SlidePair(raw_by_key[key][0], reference_by_key[key][0], key)
        for key in paired_keys
    )

    return KpfManifest(
        mouse_dir=mouse_dir,
        raw_dir=raw_dir,
        registered_dir=registered_dir,
        pairs=pairs,
        missing_raw_keys=tuple(sorted(reference_keys - raw_keys, key=_natural_key)),
        missing_reference_keys=tuple(
            sorted(raw_keys - reference_keys, key=_natural_key)
        ),
        ambiguous_keys=tuple(ambiguous),
    )


def normalize_slide_stem(path: Path | str) -> str:
    """Normalize a slide filename for raw/reference matching."""

    name = Path(path).name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ndpi", ".scn", ".tiff", ".tif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    marker_key = _marker_key(name)
    if marker_key == "marker-he":
        return marker_key

    bracket = re.search(r"\[#\s*([0-9]+)\]", name)
    if bracket:
        return f"slide-{int(bracket.group(1)):04d}"
    if marker_key:
        return marker_key

    compact = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return compact


def _iter_raw_paths(raw_dir: Path) -> tuple[Path, ...]:
    if not raw_dir.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in raw_dir.iterdir()
                if path.suffix.lower() in RAW_EXTENSIONS
            ),
            key=lambda path: _natural_key(path.name),
        )
    )


def _iter_registered_paths(registered_dir: Path) -> tuple[Path, ...]:
    if not registered_dir.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in registered_dir.iterdir()
                if path.name.lower().endswith(REGISTERED_SUFFIXES)
            ),
            key=lambda path: _natural_key(path.name),
        )
    )


def _paths_by_key(paths: tuple[Path, ...]) -> dict[str, tuple[Path, ...]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(normalize_slide_stem(path), []).append(path)
    return {key: tuple(value) for key, value in grouped.items()}


def _marker_key(name: str) -> str | None:
    normalized = re.sub(r"panc", " panc ", name)
    normalized = re.sub(r"[^a-z0-9#]+", " ", normalized)
    tokens = normalized.split()
    try:
        panc_index = tokens.index("panc")
    except ValueError:
        return None
    marker_tokens = [
        token
        for token in tokens[panc_index + 1 :]
        if not token.startswith("#")
        and not token.isdigit()
        and token != "collection"
        and len(token) != 8
    ]
    if not marker_tokens:
        return None
    return "marker-" + "-".join(marker_tokens[:3])


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", value)
    )
