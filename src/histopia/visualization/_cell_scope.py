"""Fingerprint-bound display scopes; scopes do not confer scientific approval."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_DIGEST = re.compile(r"[0-9a-f]{64}")
_SECTION = re.compile(r"[0-9]{3,6}")


@dataclass(frozen=True)
class CellSectionScope:
    """Select exact label artifacts without approving the rest of their run.

    The caller must establish scientific approval separately. Both the static
    reviewer and tile service must receive the same scope when publishing it.
    """

    source_fingerprint: str
    labels_by_section: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.source_fingerprint, str) or not _DIGEST.fullmatch(
            self.source_fingerprint
        ):
            raise ValueError("cell scope requires a source fingerprint")
        if (
            not isinstance(self.labels_by_section, Mapping)
            or not self.labels_by_section
        ):
            raise ValueError("cell scope requires a nonempty section map")
        for section, digest in self.labels_by_section.items():
            if not isinstance(section, str) or not _SECTION.fullmatch(section):
                raise ValueError("cell scope contains an invalid section")
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise ValueError("cell scope requires exact label digests")
        object.__setattr__(
            self, "labels_by_section", MappingProxyType(dict(self.labels_by_section))
        )

    @classmethod
    def from_dict(cls, value: object) -> CellSectionScope:
        """Read the versioned inline scope used by a private review registry."""
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or set(value)
            != {"schema_version", "source_fingerprint", "labels_by_section"}
        ):
            raise ValueError("cell scope must use the version 1 schema")
        return cls(value["source_fingerprint"], value["labels_by_section"])

    def select(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Reject stale/unknown scopes and preserve the source section order."""
        if result.get("fingerprint") != self.source_fingerprint:
            raise ValueError("cell scope source fingerprint differs")
        slides, artifacts = result.get("slides"), result.get("artifacts")
        if not isinstance(slides, list) or not isinstance(artifacts, dict):
            raise ValueError("cell scope requires indexed slides and artifacts")
        found: set[str] = set()
        selected = []
        for row in slides:
            if not isinstance(row, dict) or not isinstance(row.get("section"), str):
                raise ValueError("cell scope source has an invalid section row")
            section = row["section"]
            if section in found:
                raise ValueError("cell scope source has duplicate sections")
            found.add(section)
            if section in self.labels_by_section:
                labels = row.get("labels")
                if (
                    not isinstance(labels, str)
                    or artifacts.get(labels) != self.labels_by_section[section]
                ):
                    raise ValueError("cell scope label artifact differs")
                selected.append(row)
        if set(self.labels_by_section) - found:
            raise ValueError("cell scope contains an unknown section")
        return selected
