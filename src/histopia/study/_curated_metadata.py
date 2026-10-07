"""Join curated slide metadata without rewriting registered scientific results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePath

from histopia.study._manifest import fingerprint


def reconcile_curated_metadata(
    records: Sequence[Mapping],
    table: Sequence[Mapping],
    *,
    source_id: str,
    organ: str,
    filename: str,
    source_sha256: str,
) -> tuple[list[dict], dict]:
    """Use curated names/labels, retaining source rows, blanks and disagreements.

    Matching requires cohort, organ, specimen and exact native basename. A
    repeated table row is not an additional physical section. Registration
    order, Z, scientific fingerprints and image links remain bound to their
    existing results; a table-label disagreement cannot relabel an old model.
    """
    output = [dict(r) for r in records]
    index = defaultdict(list)
    for row in output:
        if row["source_id"] == source_id and row["organ"] == organ:
            key = (str(row["subject_id"]), PurePath(row["filename"]).stem)
            index[key].append(row)
            row["metadata_match"] = "outside curated table"
    entries = []
    grouped = defaultdict(list)
    for number, original in enumerate(table, 2):
        record = dict(original)
        tissue = str(record.get("Tissue Type", "")).lower()
        if tissue != organ and not (organ == "pancreas" and tissue == "panc"):
            raise ValueError("curated row conflicts with the declared organ")
        key = (str(record["mouse_id"]), str(record["raw_name"]))
        matches = index.get(key, [])
        if len(matches) > 1:
            raise ValueError("curated row matches multiple scan identities")
        matched = matches[0] if matches else None
        entry = (
            dict(matched)
            if matched
            else dict(
                source_id=source_id,
                organ=organ,
                subject_id=key[0],
                filename=key[1] + "." + str(record["file_type"]).lower(),
                review_status="metadata only",
                scan_id=None,
                review_href=None,
            )
        )
        entry.update(
            metadata_source=filename,
            metadata_source_sha256=source_sha256,
            metadata_row=number,
            metadata_row_fingerprint=fingerprint(record),
            metadata_match="curated" if matched else "unmatched curated row",
            metadata_rows=str(number),
            marker=record.get("antibody") or None,
            antibody_type=record.get("antibody_type") or None,
            table_order=record.get("order") or None,
            table_order_text=record.get("order_text") or None,
            table_label=record.get("label") or None,
            table_note=record.get("note") or None,
            table_order_status="provided" if record.get("order") else "missing",
            analysis_marker=matched.get("marker") if matched else None,
            stain_evidence="Curated metadata; acquisition QC remains separate",
        )
        entry["metadata_marker_difference"] = bool(
            matched and entry["marker"] != entry["analysis_marker"]
        )
        entries.append(entry)
        if matched:
            grouped[matched["scan_id"]].append(entry)
    fields = (
        "marker",
        "antibody_type",
        "table_order",
        "table_order_text",
        "table_label",
        "table_note",
    )
    for row in output:
        members = grouped.get(row["scan_id"], [])
        if not members:
            continue
        row.update(
            metadata_source=filename,
            metadata_source_sha256=source_sha256,
            metadata_match="curated",
            metadata_rows=" | ".join(str(r["metadata_row"]) for r in members),
            analysis_marker=row.get("marker"),
            stain_evidence="Curated metadata; acquisition QC remains separate",
        )
        conflicts = []
        for field in fields:
            values = list(dict.fromkeys(r[field] for r in members))
            row[field] = values[0] if len(values) == 1 else None
            if len(values) > 1:
                conflicts.append(field)
        row["table_order_status"] = (
            "conflicting source rows"
            if "table_order" in conflicts
            else "provided"
            if row["table_order"] is not None
            else "missing"
        )
        row["metadata_conflicts"] = " | ".join(conflicts) or None
        row["metadata_marker_difference"] = row["marker"] != row["analysis_marker"]
        for entry in members:
            entry["metadata_duplicate_scan"] = len(members) > 1
            entry["metadata_conflicts"] = row["metadata_conflicts"]
    report = dict(
        organ=organ,
        source_id=source_id,
        label="Curated " + source_id.upper(),
        filename=filename,
        source_sha256=source_sha256,
        source_rows=len(entries),
        matched_rows=sum(bool(r["scan_id"]) for r in entries),
        unique_scans=len(grouped),
        subjects=len({r["subject_id"] for r in entries}),
        duplicate_scans=sum(len(v) > 1 for v in grouped.values()),
        marker_difference_rows=sum(r["metadata_marker_difference"] for r in entries),
        rows=entries,
    )
    return output, report
