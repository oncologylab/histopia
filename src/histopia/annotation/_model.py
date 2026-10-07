"""Small, explicit pathology annotation ontologies."""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_-]*")
_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True, slots=True)
class AnnotationClass:
    """One stable ontology class and its display color."""

    id: str
    label: str
    color: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.id):
            raise ValueError("annotation class id must be a lowercase identifier")
        if not self.label.strip():
            raise ValueError("annotation class label must not be blank")
        if not _COLOR_RE.fullmatch(self.color):
            raise ValueError("annotation class color must be a six-digit hex color")

    def as_dict(self) -> dict[str, str]:
        """Return the portable ontology representation."""

        return {"id": self.id, "label": self.label, "color": self.color.lower()}


@dataclass(frozen=True, slots=True)
class AnnotationOntology:
    """Ordered set of mutually identifiable pathology classes."""

    name: str
    classes: tuple[AnnotationClass, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("annotation ontology name must not be blank")
        if self.version < 1:
            raise ValueError("annotation ontology version must be positive")
        if not self.classes:
            raise ValueError("annotation ontology must define at least one class")
        identifiers = [item.id for item in self.classes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("annotation ontology class ids must be unique")

    @property
    def class_ids(self) -> frozenset[str]:
        """Return accepted class identifiers."""

        return frozenset(item.id for item in self.classes)

    def as_dict(self) -> dict[str, object]:
        """Return the portable ontology representation."""

        return {
            "name": self.name,
            "version": self.version,
            "classes": [item.as_dict() for item in self.classes],
        }

    @classmethod
    def from_dict(cls, payload: object) -> AnnotationOntology:
        """Validate and load an ontology from portable JSON data."""

        if not isinstance(payload, dict):
            raise ValueError("annotation ontology must be an object")
        raw_classes = payload.get("classes")
        if not isinstance(raw_classes, list):
            raise ValueError("annotation ontology classes must be a list")
        classes: list[AnnotationClass] = []
        for row in raw_classes:
            if not isinstance(row, dict):
                raise ValueError("annotation ontology class must be an object")
            values = (row.get("id"), row.get("label"), row.get("color"))
            if any(not isinstance(value, str) for value in values):
                raise ValueError("annotation ontology class fields must be text")
            classes.append(AnnotationClass(*values))  # type: ignore[arg-type]
        name = payload.get("name")
        version = payload.get("version")
        if not isinstance(name, str):
            raise ValueError("annotation ontology name must be text")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("annotation ontology version must be an integer")
        return cls(name=name, version=version, classes=tuple(classes))


def default_pancreas_ontology() -> AnnotationOntology:
    """Return a conservative broad-class ontology for pancreas review."""

    return AnnotationOntology(
        name="histopia-pancreas-broad",
        classes=(
            AnnotationClass("epithelial", "Epithelial / ductal lesion", "#d73027"),
            AnnotationClass("acinar", "Acinar", "#fdae61"),
            AnnotationClass("stroma", "Stroma / ECM", "#1a9850"),
            AnnotationClass("immune", "Immune / lymphoid", "#4575b4"),
            AnnotationClass("vascular", "Vascular", "#984ea3"),
            AnnotationClass("necrosis", "Necrosis / artifact", "#7f8c8d"),
            AnnotationClass("uncertain", "Uncertain", "#e6ab02"),
        ),
    )
