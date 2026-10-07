"""Small, auditable inventory exports, separate from scientific result catalogs.

Inventory inclusion does not confer reconstruction eligibility or model approval.
Counts describe distinct scan-file contents, never distinct physical sections.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import file_sha256, fingerprint

FIELDS = (
    "scan_id",
    "organ",
    "source_id",
    "subject_id",
    "subject_evidence",
    "organ_evidence",
    "filename",
    "file_format",
    "size_bytes",
    "sha1",
    "source_sha256",
    "copy_count",
    "inclusion",
    "stain",
    "marker",
    "stain_evidence",
    "block_id",
    "physical_section_id",
    "section_order",
    "order_evidence",
    "z_um",
    "z_spacing_kind",
    "z_evidence",
    "reconstruction_id",
    "eligibility_fingerprint",
    "mouse_role",
    "mouse_role_evidence",
    "local_status",
    "registration_status",
    "cell_status",
    "stain_status",
    "feature_status",
    "semantic_status",
    "review_status",
    "review_href",
    "legacy_derivative_count",
    "next_step",
    "upstream_fingerprint",
    "metadata_source",
    "metadata_source_sha256",
    "metadata_match",
    "metadata_row",
    "metadata_rows",
    "metadata_row_fingerprint",
    "metadata_duplicate_scan",
    "metadata_conflicts",
    "metadata_marker_difference",
    "analysis_marker",
    "antibody_type",
    "table_order",
    "table_order_text",
    "table_order_status",
    "table_label",
    "table_note",
)
_ASSETS = Path(__file__).with_name("_organ_metadata_assets")
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


def _spreadsheet_value(value: object) -> object:
    # Metadata and filenames are untrusted spreadsheet text, not formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return "" if value is None else value


def _csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {key: _spreadsheet_value(row[key]) for key in FIELDS} for row in rows
    )
    return output.getvalue()


def validate_inventory(manifest: Mapping) -> dict:
    """Validate stable scan identities and a fingerprinted, internal-only scope."""
    if manifest.get("schema_version") != "organ-inventory-1" or manifest.get(
        "fingerprint"
    ) != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("inventory requires a versioned fingerprint binding")
    sources = manifest["sources"]
    source_ids = {s["id"] for s in sources}
    if len(source_ids) != len(sources) or any(
        s.get("external") is not False or not _ID.fullmatch(s["id"]) for s in sources
    ):
        raise ValueError("inventory accepts unique internal sources only")
    seen = set()
    rows = []
    for raw in manifest["rows"]:
        row = {key: raw.get(key) for key in FIELDS}
        if any(isinstance(value, (dict, list, tuple)) for value in row.values()):
            raise ValueError("inventory fields must be scalar")
        key = row["scan_id"]
        if (
            key in seen
            or not isinstance(key, str)
            or not re.fullmatch(r"sha1-[a-f0-9]{40}", key)
            or key != "sha1-" + str(row["sha1"])
        ):
            raise ValueError("scan identities must be unique content hashes")
        seen.add(key)
        if (
            row["source_id"] not in source_ids
            or not _ID.fullmatch(row["organ"] or "")
            or row["inclusion"] != "included"
            or not isinstance(row["copy_count"], int)
            or row["copy_count"] < 1
        ):
            raise ValueError("invalid inventory inclusion or source")
        href = row["review_href"]
        if href:
            url = urlsplit(href)
            query = parse_qs(url.query)
            if (
                url.scheme
                or url.netloc
                or url.path != "../index.html"
                or query.get("view", [""])[0]
                not in {"data-catalog", "internal-inputs", "registration"}
            ):
                raise ValueError(
                    "inventory result links must open existing review views"
                )
        rows.append(row)
    primary = []
    known = {r["scan_id"]: r for r in rows}
    organs_seen = set()
    for table in manifest.get("primary_tables", []):
        organ = table["organ"]
        if (
            not _ID.fullmatch(organ)
            or organ in organs_seen
            or table["source_id"] not in source_ids
            or not re.fullmatch(r"[a-f0-9]{64}", table["source_sha256"])
        ):
            raise ValueError("invalid primary metadata table identity")
        organs_seen.add(organ)
        entries = []
        row_numbers = set()
        for raw in table["rows"]:
            row = {key: raw.get(key) for key in FIELDS}
            if (
                row["organ"] != organ
                or row["source_id"] != table["source_id"]
                or row["metadata_row"] in row_numbers
                or row["metadata_source_sha256"] != table["source_sha256"]
                or row["metadata_source"] != table["filename"]
                or any(isinstance(value, (dict, list, tuple)) for value in row.values())
            ):
                raise ValueError("primary table row differs from its source binding")
            row_numbers.add(row["metadata_row"])
            if row["scan_id"]:
                source = known.get(row["scan_id"])
                if not source or any(
                    row[k] != source[k]
                    for k in FIELDS
                    if not k.startswith(("metadata_", "table_"))
                    and k not in {"marker", "antibody_type", "stain_evidence"}
                ):
                    raise ValueError("primary metadata cannot rewrite result bindings")
            elif row["review_href"]:
                raise ValueError("unmatched metadata cannot link an invented result")
            entries.append(row)
        if (
            len(entries) != table["source_rows"]
            or len({r["scan_id"] for r in entries if r["scan_id"]})
            != table["unique_scans"]
        ):
            raise ValueError("primary table counts differ from their source rows")
        primary.append(dict(table, rows=entries))
    return dict(manifest, rows=rows, primary_tables=primary)


def build_organ_metadata(
    manifest: Mapping,
    output_dir: Path | str,
    *,
    attachments: Mapping[str, Mapping] | None = None,
) -> Path:
    """Write per-organ CSVs and a searchable, offline-capable metadata page.

    Attachments are hash-verified exports (for example a workbook and full file
    index); private source paths are never copied into the browser manifest.
    """
    inventory = validate_inventory(manifest)
    rows = inventory["rows"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloads = out / "downloads"
    downloads.mkdir(exist_ok=True)
    expected = {"all-organs.csv"}
    write_text_atomic(downloads / "all-organs.csv", _csv(rows))
    organs = sorted({r["organ"] for r in rows}, key=lambda s: (s == "unassigned", s))
    counts = []
    primary = {t["organ"]: t for t in inventory["primary_tables"]}
    for organ in organs:
        selected = [r for r in rows if r["organ"] == organ]
        expected.add(organ + ".csv")
        write_text_atomic(
            downloads / (organ + ".csv"),
            _csv(primary[organ]["rows"] if organ in primary else selected),
        )
        if organ in primary:
            expected.add(organ + "-inventory.csv")
            write_text_atomic(downloads / (organ + "-inventory.csv"), _csv(selected))
        counts.append(
            dict(
                organ=organ,
                scans=len(selected),
                result_scans=sum(bool(r["review_href"]) for r in selected),
                sources=dict(Counter(r["source_id"] for r in selected)),
                metadata_only=sum(not r["review_href"] for r in selected),
            )
        )
    links = []
    for name, item in (attachments or {}).items():
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.(?:csv|xlsx|parquet|json)", name)
            or name in expected
        ):
            raise ValueError("invalid or colliding inventory attachment")
        source = Path(item["path"])
        if file_sha256(source) != item["sha256"]:
            raise ValueError("inventory attachment differs from its binding")
        shutil.copyfile(source, downloads / name)
        expected.add(name)
        links.append(
            dict(href="downloads/" + name, label=item["label"], sha256=item["sha256"])
        )
    for path in downloads.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()
    payload = dict(
        schema_version=inventory["schema_version"],
        fingerprint=inventory["fingerprint"],
        updated_at=inventory["updated_at"],
        scope=inventory["scope"],
        count_unit="Distinct native scan-file contents; not physical sections",
        sources=inventory["sources"],
        organs=counts,
        rows=rows,
        primary_tables=inventory["primary_tables"],
        file_summary=inventory.get("file_summary", {}),
        attachments=links,
        evidence=inventory.get("evidence", {}),
    )
    write_json_atomic(out / "inventory.json", payload)
    write_text_atomic(
        out / "inventory-data.js",
        "globalThis.HISTOPIA_ORGAN_INVENTORY="
        + json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
        + ";\n",
    )
    fallback = (
        "".join(
            '<li><a href="downloads/'
            + c["organ"]
            + '.csv">'
            + escape(c["organ"].title())
            + " · "
            + (
                str(primary[c["organ"]]["source_rows"]) + " curated rows"
                if c["organ"] in primary
                else str(c["scans"]) + " scans"
            )
            + "</a></li>"
            for c in counts
        )
        or "<li>No inventoried scans.</li>"
    )
    write_text_atomic(
        out / "index.html",
        (_ASSETS / "index.html")
        .read_text()
        .replace("<!-- ORGAN_DOWNLOADS -->", fallback),
    )
    for name in ("inventory.css", "inventory.js"):
        shutil.copyfile(_ASSETS / name, out / name)
    return out / "index.html"


def organ_metadata_tab(inventory: Mapping) -> dict:
    """Inventory scope is deliberately distinct from image-result eligibility."""
    return dict(
        id="organ-metadata",
        label="Organ metadata",
        href="organ-metadata/index.html",
        export_fingerprint=inventory["fingerprint"],
        scope=dict(
            sources=[s["id"] for s in inventory["sources"]],
            organs=[o["organ"] for o in inventory["organs"]],
        ),
    )
