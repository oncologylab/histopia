"""Source- and organ-specific review of completed research artifacts."""

from __future__ import annotations

import html
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import file_sha256, fingerprint

_ASSETS = Path(__file__).with_name("_results_catalog_assets")
_ORGANS = ("pancreas", "liver", "lung", "kidney")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("catalog identifiers must be nonempty URL-safe strings")
    return value


def _link(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("review links must be relative URLs or HTTPS references")
    parsed = urlsplit(value)
    if parsed.scheme and (parsed.scheme != "https" or not parsed.netloc):
        raise ValueError("external references require HTTPS")
    if not parsed.scheme and (value.startswith("/") or parsed.netloc):
        raise ValueError("local review links must remain relative for proxies")
    return value


def _copy_asset(record: Mapping, output: Path, verified: dict) -> dict:
    source = Path(record["path"])
    digest = record.get("sha256")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("review assets require an expected SHA-256")
    suffix = source.suffix.lower()
    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".pdf",
        ".csv",
        ".parquet",
        ".npz",
    }:
        raise ValueError("unsupported review artifact format")
    destination = output / "assets" / (digest + suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if file_sha256(source) != digest:
            raise ValueError("review artifact differs from its result binding")
        # Copy, rather than link, to keep a later source mutation from changing
        # already exported evidence. The name is immutable and content addressed.
        temporary = destination.with_suffix(suffix + ".partial")
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != digest:
            temporary.unlink()
            raise ValueError("review artifact copy failed verification")
        temporary.replace(destination)
    else:
        stat = destination.stat()
        signature = [stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if (
            verified.get(destination.name) != signature
            and file_sha256(destination) != digest
        ):
            raise ValueError("existing content-addressed review artifact changed")
    stat = destination.stat()
    verified[destination.name] = [
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ]
    public = {
        k: record[k]
        for k in ("label", "caption", "native_xywh", "mpp_xy")
        if k in record
    }
    public.update(href=destination.relative_to(output).as_posix(), sha256=digest)
    return public


def build_results_catalog(
    sources: Sequence[Mapping],
    datasets: Sequence[Mapping],
    output_dir: Path | str,
    *,
    updated_at: str,
    compute: Sequence[Mapping] = (),
    reconstruction_registry: Mapping | None = None,
    analysis_bundles: Mapping | None = None,
) -> Path:
    """Export hash-verified evidence with distinct source and organ identities.

    Caller-supplied private asset paths are removed from the exported catalog.
    ``ready`` means ready to review, never scientific approval. Overlapping
    tiles and HPA specimen annotations must retain their stated measurement
    scope. Review notes are browser-local drafts with a downloadable binding;
    this surface does not mutate upstream approvals or reference labels.
    """
    output = Path(output_dir)
    if reconstruction_registry is None:
        raise ValueError("Tissue review requires a reconstruction eligibility registry")
    from histopia.study._eligibility import filter_reconstruction_catalog

    sources, datasets, _, scope = filter_reconstruction_catalog(
        sources, datasets, reconstruction_registry
    )
    return _write_catalog(
        sources,
        datasets,
        output,
        updated_at=updated_at,
        compute=compute,
        scope=scope,
        analysis_bundles=analysis_bundles,
    )


def _write_catalog(
    sources: Sequence[Mapping],
    datasets: Sequence[Mapping],
    output: Path,
    *,
    updated_at: str,
    compute: Sequence[Mapping] = (),
    scope: Mapping | None = None,
    analysis_bundles: Mapping | None = None,
    benchmark_policy: Mapping | None = None,
    input_review_policy: Mapping | None = None,
) -> Path:
    """Render already authorized records; public builders own separate gates."""
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".verified-assets.json"
    verified = json.loads(cache.read_text()) if cache.exists() else {}
    public_sources, seen = [], set()
    for source in sources:
        key = _identifier(source["id"])
        if key in seen or not isinstance(source.get("external"), bool):
            raise ValueError("sources require unique IDs and an explicit external flag")
        seen.add(key)
        public_sources.append(
            {
                k: source[k]
                for k in ("id", "label", "external", "species", "description")
            }
        )
    public_datasets, identities = [], set()
    for dataset in datasets:
        key = _identifier(dataset["id"])
        source_id = _identifier(dataset["source_id"])
        if key in identities or source_id not in seen:
            raise ValueError(
                "dataset identities must be unique and bind a known source"
            )
        identities.add(key)
        _identifier(dataset.get("organ"))
        if dataset.get("status") not in {"ready", "running", "pending", "failed"}:
            raise ValueError("completion status must not imply scientific approval")
        if not _DIGEST.fullmatch(str(dataset.get("fingerprint", ""))):
            raise ValueError("datasets require an upstream fingerprint")
        row = {
            k: dataset[k]
            for k in (
                "id",
                "source_id",
                "organ",
                "subject_id",
                "title",
                "stage",
                "status",
                "summary",
                "fingerprint",
            )
        }
        # Presentation metadata never changes the upstream evidence identity.
        # Physical section IDs come from a study manifest, not title parsing.
        for name in (
            "specimen_id",
            "physical_section_id",
            "short_title",
            "summary_short",
            "block_id",
            "reconstruction_id",
            "eligibility_fingerprint",
            "image_id",
            "benchmark_fingerprint",
            "role",
            "input_scan_id",
            "input_scan_label",
        ):
            if name in dataset:
                row[name] = str(dataset[name])
        for name in ("acquisition_ids", "acquisition_modalities"):
            if name in dataset:
                row[name] = list(dataset[name])
        if "featured" in dataset:
            if not isinstance(dataset["featured"], bool):
                raise ValueError("featured is a presentation flag, not an approval")
            row["featured"] = dataset["featured"]
        if "evidence_kind" in dataset:
            if dataset["evidence_kind"] not in {
                "image",
                "summary",
                "inventory",
                "stack",
            }:
                raise ValueError("unknown evidence presentation kind")
            row["evidence_kind"] = dataset["evidence_kind"]
        if "section_order" in dataset:
            order = dataset["section_order"]
            if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                raise ValueError("section order must be a nonnegative integer")
            row["section_order"] = order
        row["media"] = [
            _copy_asset(asset, output, verified) for asset in dataset.get("media", [])
        ]
        row["downloads"] = [
            _copy_asset(asset, output, verified)
            for asset in dataset.get("downloads", [])
        ]
        row["links"] = [
            {
                "label": str(link["label"]),
                "href": _link(link["href"]),
                "top": bool(link.get("top", False)),
            }
            for link in dataset.get("links", [])
        ]
        if analysis_bundles and key in analysis_bundles:
            bundle = analysis_bundles[key]
            if (
                bundle.get("kind") != "tissue-regions-1"
                or bundle.get("dataset_fingerprint") != dataset["fingerprint"]
                or bundle.get("eligibility_fingerprint")
                != dataset.get("eligibility_fingerprint")
            ):
                raise ValueError("region bundle must bind the eligible dataset")
            files = bundle.get("files", {})
            allowed = {
                "index.html",
                "regions.css",
                "regions.js",
                "regions-data.js",
                "regions.json",
                "all-regions.svg",
            }
            if set(files) != allowed:
                raise ValueError(
                    "region bundle requires its complete portable file set"
                )
            source = Path(bundle["path"])
            destination = output / "analyses" / key
            destination.mkdir(parents=True, exist_ok=True)
            for name, digest in files.items():
                if (
                    not _DIGEST.fullmatch(str(digest))
                    or file_sha256(source / name) != digest
                ):
                    raise ValueError("region bundle file differs from its binding")
                shutil.copyfile(source / name, destination / name)
                if file_sha256(destination / name) != digest:
                    raise ValueError("region bundle copy failed verification")
            row["analysis_view"] = (
                (destination / "index.html").relative_to(output).as_posix()
            )
            row["links"].insert(
                0,
                dict(label="Open tissue regions", href=row["analysis_view"], top=False),
            )
        if dataset.get("reconstruction_view"):
            if dataset.get("evidence_kind") != "stack":
                raise ValueError("only stack records can embed a reconstruction")
            row["reconstruction_view"] = _link(dataset["reconstruction_view"])
            row["links"].insert(
                0,
                {
                    "label": "Open reconstruction",
                    "href": row["reconstruction_view"],
                    "top": False,
                },
            )
        row["table"] = dataset.get("table", [])
        row["facts"] = dataset.get("facts", {})
        row["scientific_approval"] = False
        public_datasets.append(row)
    # Reject NaN/Infinity rather than silently shipping broken JSON or scores.
    json.dumps(public_datasets, allow_nan=False)
    organs = list(
        dict.fromkeys(
            [*(() if scope else _ORGANS), *(d["organ"] for d in public_datasets)]
        )
    )
    payload = {
        "schema_version": 1,
        "updated_at": updated_at,
        "sources": public_sources,
        "organs": organs,
        "datasets": public_datasets,
        "compute": list(compute),
    }
    if scope is not None:
        payload["eligibility_policy"] = {
            "id": scope["policy_id"],
            "registry_fingerprint": scope["fingerprint"],
        }
    if benchmark_policy is not None:
        payload["benchmark_policy"] = dict(benchmark_policy)
    if input_review_policy is not None:
        payload["input_review_policy"] = dict(input_review_policy)
        payload["organs"] = [
            o for o in organs if any(d["organ"] == o for d in public_datasets)
        ]
    payload["fingerprint"] = fingerprint(payload)
    write_json_atomic(cache, verified)
    write_json_atomic(output / "catalog.json", payload)
    write_text_atomic(
        output / "catalog-data.js",
        "globalThis.HISTOPIA_RESULTS_CATALOG="
        + json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
            "<", "\\u003c"
        )
        + ";\n",
    )
    fallback = []
    for source in public_sources:
        if not any(row["source_id"] == source["id"] for row in public_datasets):
            continue
        fallback.append("<h2>" + html.escape(source["label"]) + "</h2>")
        for organ in organs:
            rows = [
                d
                for d in public_datasets
                if d["source_id"] == source["id"] and d["organ"] == organ
            ]
            if not rows:
                continue
            fallback.append("<h3>" + organ.title() + "</h3><ul>")
            for row in rows:
                links = " ".join(
                    '<a href="'
                    + html.escape(a["href"], quote=True)
                    + '">'
                    + html.escape(str(a.get("label", "Evidence")))
                    + "</a>"
                    for a in row["media"] + row["downloads"] + row["links"]
                )
                fallback.append(
                    "<li>"
                    + html.escape(
                        row["title"] + " — " + row["status"] + ". " + row["summary"]
                    )
                    + " "
                    + links
                    + "</li>"
                )
            fallback.append("</ul>")
    template = (_ASSETS / "index.html").read_text()
    if benchmark_policy is not None:
        template = template.replace("Tissue review", "External validation")
    if input_review_policy is not None:
        template = template.replace("Tissue review", "Internal tissues")
        template = template.replace('for="subject">Specimen', 'for="subject">Mouse')
        template = template.replace('for="section">Section', 'for="section">Scan')
    write_text_atomic(
        output / "index.html",
        template.replace(
            "<!-- FALLBACK -->",
            "\n".join(fallback)
            or (
                "<p>No external benchmark images are available yet.</p>"
                if benchmark_policy is not None
                else "<p>No internal image results are available.</p>"
                if input_review_policy is not None
                else "<p>No eligible serial IHC datasets are available.</p>"
            ),
        ),
    )
    for name in ("catalog.js", "catalog.css"):
        write_text_atomic(output / name, (_ASSETS / name).read_text())
    if (
        scope is not None
        or benchmark_policy is not None
        or input_review_policy is not None
    ):
        # Withdraw obsolete public derivatives as well as their navigation.
        # Source research artifacts are never touched. Production publishers
        # build a fresh release and archive the previous publication separately.
        referenced = {
            a["href"]
            for row in public_datasets
            for a in row["media"] + row["downloads"]
        }
        for asset in (output / "assets").glob("*"):
            if (
                asset.is_file()
                and asset.relative_to(output).as_posix() not in referenced
            ):
                asset.unlink()
        write_json_atomic(
            cache, {k: v for k, v in verified.items() if "assets/" + k in referenced}
        )
    analysis_ids = {row["id"] for row in public_datasets if row.get("analysis_view")}
    for directory in (output / "analyses").glob("*"):
        if directory.is_dir() and directory.name not in analysis_ids:
            shutil.rmtree(directory)
    return output / "index.html"


def catalog_navigation_context(catalog: Mapping) -> dict:
    """Describe available selections from an already filtered public catalog."""
    sources = [source["id"] for source in catalog["sources"]]
    return dict(
        catalog_sources=sources,
        default_source=sources[0] if sources else "",
        catalog_contexts={
            row["id"]: {
                key: str(row[key])
                for key in ("source_id", "organ", "subject_id", "reconstruction_id")
                if key in row
            }
            for row in catalog["datasets"]
        },
    )


def attach_results_catalog(catalog_dir: Path | str, review_dir: Path | str) -> Path:
    """Attach the data catalog first, preserving every existing workflow tab."""
    catalog, review = Path(catalog_dir), Path(review_dir)
    relative = catalog.resolve().relative_to(review.resolve())
    data = json.loads((catalog / "catalog.json").read_text())
    portal = json.loads((review / "manifest.json").read_text())
    if not isinstance(portal.get("tabs"), list):
        raise ValueError("destination is not a workflow review portal")
    tab = {
        "id": "data-catalog",
        "label": "Data & results",
        "href": (relative / "index.html").as_posix(),
        "export_fingerprint": data["fingerprint"],
        **catalog_navigation_context(data),
    }
    portal["tabs"] = [tab] + [r for r in portal["tabs"] if r["id"] != tab["id"]]
    write_json_atomic(review / "manifest.json", portal)
    write_text_atomic(
        review / "manifest-data.js",
        "globalThis.HISTOPIA_WORKFLOW_REVIEW="
        + json.dumps(portal, separators=(",", ":"))
        + ";\n",
    )
    return review / "index.html"
