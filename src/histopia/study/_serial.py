"""Physical section identity and split checks for serial public specimens."""

from __future__ import annotations

import math
from collections.abc import Mapping

from histopia.study._manifest import fingerprint


def validate_serial_study(manifest: Mapping) -> dict:
    """Validate physical planes independently of acquisitions and channels.

    Z is relative to a declared reference, in micrometres. Unknown depth stays
    null. Acquisitions of the same section (restains, repeated cycles) refer to
    one physical ID. Subjects are scoped by source and species, never inferred
    from organ or filename. This schema does not alter existing mouse studies.
    """
    if manifest.get("schema_version") != "serial-1" or not manifest.get("study_id"):
        raise ValueError("a versioned serial study identity is required")
    subjects = {}
    for subject in manifest.get("subjects", []):
        key = tuple(subject.get(k) for k in ("source_id", "species", "subject_id"))
        if not all(key) or key in subjects:
            raise ValueError(
                "subjects require unique source/species/subject identities"
            )
        if subject.get("role") not in {
            "development",
            "test",
            "protected",
            "unassigned",
        }:
            raise ValueError("subject role must be explicit")
        if subject["role"] == "test" and subject.get("exposed") is not False:
            raise ValueError("untouched tests require explicitly unexposed subjects")
        subjects[key] = subject
    sections, orders, depths = {}, {}, {}
    for section in manifest.get("sections", []):
        key = section.get("physical_section_id")
        subject = tuple(section.get(k) for k in ("source_id", "species", "subject_id"))
        if not key or key in sections or subject not in subjects:
            raise ValueError("sections require unique IDs and a known subject")
        if not all(section.get(k) for k in ("organ", "specimen_id", "block_id")):
            raise ValueError(
                "serial sections require organ, specimen and block identity"
            )
        order = section.get("section_order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("physical section order must be a nonnegative integer")
        stack = (*subject, section["specimen_id"], section["block_id"])
        if order in orders.setdefault(stack, set()):
            raise ValueError(
                "duplicate physical depth/order; restains are acquisitions"
            )
        orders[stack].add(order)
        z = section.get("z_um")
        if z is not None:
            if (
                isinstance(z, bool)
                or not isinstance(z, (int, float))
                or not math.isfinite(z)
            ):
                raise ValueError("Z must be finite or explicitly unknown")
            if not section.get("z_evidence") or not section.get("z_reference"):
                raise ValueError("physical Z requires evidence and a reference")
            depths.setdefault(stack, []).append((order, z))
        sections[key] = section
    for rows in depths.values():
        values = [z for _, z in sorted(rows)]
        if any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("distinct physical sections must have increasing Z")
    acquisitions = set()
    for acquisition in manifest.get("acquisitions", []):
        key = acquisition.get("acquisition_id")
        physical = acquisition.get("physical_section_id")
        if not key or key in acquisitions or physical not in sections:
            raise ValueError("acquisitions require unique IDs and a physical section")
        if not acquisition.get("modality") or not acquisition.get("source_binding"):
            raise ValueError("acquisitions require modality and source bindings")
        if "z_um" in acquisition:
            raise ValueError("Z belongs to physical sections, not image acquisitions")
        acquisitions.add(key)
    for fit in manifest.get("fits", []):
        for key in fit.get("subjects", []):
            if subjects.get(tuple(key), {}).get("role") != "development":
                raise ValueError(
                    "preprocessing, calibration and model fits "
                    "require development subjects"
                )
    result = dict(manifest)
    result.pop("fingerprint", None)
    result["fingerprint"] = fingerprint(result)
    return result
