from __future__ import annotations

import copy
import json

import pytest
from PIL import Image

from histopia.study._eligibility import (
    POLICY_ID,
    assess_reconstruction_eligibility,
    filter_reconstruction_catalog,
)
from histopia.study._manifest import file_sha256, fingerprint
from histopia.study._serial import validate_serial_study
from histopia.visualization._results_catalog import build_results_catalog


def example():
    identity = dict(source_id="serial", species="human", subject_id="patient1")
    study = validate_serial_study(
        dict(
            schema_version="serial-1",
            study_id="example",
            subjects=[dict(**identity, role="development", exposed=True)],
            sections=[
                dict(
                    **identity,
                    physical_section_id=f"p{i}",
                    organ="tongue",
                    specimen_id="sample1",
                    block_id="block1",
                    section_order=order,
                    z_um=z,
                    z_reference="p0",
                    z_evidence="sectioning record",
                )
                for i, (order, z) in enumerate([(1, 0), (3, 8), (6, 20)])
            ],
            acquisitions=[
                dict(
                    acquisition_id=f"a{i}",
                    physical_section_id=f"p{i}",
                    modality="H&E" if i == 0 else "IHC",
                    source_binding={"sha256": fingerprint(i)},
                )
                for i in range(3)
            ],
        )
    )
    assessment = dict(
        reconstruction_id="stack1",
        study_fingerprint=study["fingerprint"],
        external=True,
        acquisition_ids=["a0", "a1", "a2"],
        acquisition_qc={
            a["acquisition_id"]: dict(
                source_binding_fingerprint=fingerprint(a["source_binding"]),
                imaging_mode="brightfield",
                verified_modality=a["modality"],
                documentation_sha256=fingerprint("docs"),
                image_available=True,
                image_sha256=fingerprint(a["acquisition_id"]),
                native_visual_qc="pass",
                native_visual_qc_sha256=fingerprint("native visual check"),
            )
            for a in study["acquisitions"]
        },
        physical_provenance=dict(
            block_verified=True,
            order_verified=True,
            evidence_sha256=fingerprint("serial log"),
            spacing_status="documented",
        ),
        reconstruction_qc=dict(
            study_fingerprint=study["fingerprint"],
            reviewed_section_ids=["p0", "p1", "p2"],
            continuity="pass",
            alignment="pass",
            coverage="pass",
            stack_visual="pass",
            registration_fingerprint=fingerprint("registration"),
            stack_sha256=fingerprint("stack"),
            visual_qc_sha256=fingerprint("visual"),
        ),
    )
    return study, assessment


def rebind(study, assessment):
    study = validate_serial_study(study)
    assessment["study_fingerprint"] = study["fingerprint"]
    assessment["reconstruction_qc"]["study_fingerprint"] = study["fingerprint"]
    return study, assessment


def test_nonuniform_serial_ihc_and_he_are_eligible():
    study, assessment = example()
    result = assess_reconstruction_eligibility(study, assessment)
    assert result["status"] == "eligible"
    assert result["organ"] == "tongue" and result["scientific_approval"] is False
    study["acquisitions"][0]["modality"] = "IHC"
    assessment["acquisition_qc"]["a0"]["verified_modality"] = "IHC"
    assert (
        assess_reconstruction_eligibility(*rebind(study, assessment))["status"]
        == "eligible"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "if",
        "mislabeled_if",
        "he_only",
        "two_planes",
        "different_block",
        "donor_only",
        "no_images",
        "no_order",
        "unknown_z",
        "unverified_stain",
        "no_stack",
        "assumed_public_z",
    ],
)
def test_ineligible_or_unverified_data_fails_closed(failure):
    study, a = example()
    if failure == "if":
        study["acquisitions"][1]["modality"] = "CyCIF"
    elif failure == "mislabeled_if":
        a["acquisition_qc"]["a1"]["imaging_mode"] = "fluorescence"
    elif failure == "he_only":
        for acq in study["acquisitions"]:
            acq["modality"] = "H&E"
            a["acquisition_qc"][acq["acquisition_id"]]["verified_modality"] = "H&E"
    elif failure == "two_planes":
        study["acquisitions"][2]["physical_section_id"] = "p1"
    elif failure == "different_block":
        study["sections"][2]["block_id"] = "another-block"
    elif failure == "donor_only":
        a["physical_provenance"]["block_verified"] = False
    elif failure == "no_images":
        a["acquisition_qc"]["a1"]["image_available"] = False
    elif failure == "no_order":
        a["physical_provenance"]["order_verified"] = False
    elif failure == "unknown_z":
        study["sections"][1]["z_um"] = None
    elif failure == "unverified_stain":
        a["acquisition_qc"]["a1"]["native_visual_qc"] = "pending"
    elif failure == "no_stack":
        a["reconstruction_qc"]["stack_sha256"] = None
    elif failure == "assumed_public_z":
        a["physical_provenance"]["spacing_status"] = "assumed"
        a["physical_provenance"]["prior_approval_sha256"] = fingerprint("old approval")
    assert assess_reconstruction_eligibility(*rebind(study, a))["status"] != "eligible"


def test_internal_assumed_spacing_preserves_its_scope():
    study, a = example()
    a["external"] = False
    a["physical_provenance"].update(
        spacing_status="assumed", prior_approval_sha256=fingerprint("old approval")
    )
    result = assess_reconstruction_eligibility(study, a)
    assert result["status"] == "eligible" and result["spacing_status"] == "assumed"
    del a["physical_provenance"]["prior_approval_sha256"]
    assert assess_reconstruction_eligibility(study, a)["status"] == "pending"


def test_stale_manifest_and_assessment_are_rejected():
    study, a = example()
    a["study_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="different study"):
        assess_reconstruction_eligibility(study, a)
    study, a = example()
    study["sections"][2]["z_um"] += 1
    with pytest.raises(ValueError, match="stale"):
        assess_reconstruction_eligibility(study, a)


def catalog_example(tmp_path):
    study, a = example()
    decision = assess_reconstruction_eligibility(study, a)
    registry = dict(
        schema_version="reconstruction-registry-1",
        policy_id=POLICY_ID,
        entries=[dict(study=study, assessment=a)],
    )
    source = dict(
        id="serial",
        label="Serial IHC",
        external=True,
        species="human",
        description="Serial tissue",
    )
    im = tmp_path / "image.png"
    Image.new("RGB", (12, 12), "brown").save(im)
    row = dict(
        id="result1",
        source_id="serial",
        organ="tongue",
        subject_id="patient1",
        specimen_id="sample1",
        block_id="block1",
        reconstruction_id="stack1",
        eligibility_fingerprint=decision["fingerprint"],
        acquisition_ids=a["acquisition_ids"],
        acquisition_modalities=decision["modalities"],
        title="Eligible stack",
        stage="Reconstruction",
        status="ready",
        summary="Observed serial tissue",
        fingerprint=fingerprint("result"),
        evidence_kind="stack",
        media=[dict(path=im, sha256=file_sha256(im), label="Stack")],
    )
    return [source], [row], registry


def test_region_bundles_are_verified_and_removed_when_eligibility_changes(tmp_path):
    sources, rows, registry = catalog_example(tmp_path)
    folder = tmp_path / "regions"
    folder.mkdir()
    names = [
        "index.html",
        "regions.css",
        "regions.js",
        "regions-data.js",
        "regions.json",
        "all-regions.svg",
    ]
    for name in names:
        (folder / name).write_text("BOUND_REGION_CONTENT")
    bundle = dict(
        kind="tissue-regions-1",
        path=folder,
        dataset_fingerprint=rows[0]["fingerprint"],
        eligibility_fingerprint=rows[0]["eligibility_fingerprint"],
        files={name: file_sha256(folder / name) for name in names},
    )
    out = tmp_path / "out"

    def export(bundles):
        build_results_catalog(
            sources,
            rows,
            out,
            updated_at="now",
            reconstruction_registry=registry,
            analysis_bundles=bundles,
        )

    export({"result1": bundle})
    data = json.loads((out / "catalog.json").read_text())
    assert data["datasets"][0]["analysis_view"] == "analyses/result1/index.html"
    assert "Open tissue regions" in (out / "index.html").read_text()
    for name in names:
        assert file_sha256(out / "analyses/result1" / name) == bundle["files"][name]
    altered = dict(bundle, dataset_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="bind the eligible dataset"):
        export({"result1": altered})
    (folder / "regions.js").write_text("ALTERED")
    with pytest.raises(ValueError, match="differs from its binding"):
        export({"result1": bundle})
    # Exclusion precedes opening invalid bundles and withdraws all prior assets.
    registry["entries"][0]["assessment"]["acquisition_qc"]["a1"]["imaging_mode"] = (
        "fluorescence"
    )
    export({"result1": bundle})
    assert not (out / "analyses/result1").exists()
    assert json.loads((out / "catalog.json").read_text())["datasets"] == []


def test_filter_runs_before_copy_and_all_serialization(tmp_path):
    sources, rows, registry = catalog_example(tmp_path)
    rejected = copy.deepcopy(rows[0])
    rejected.update(
        id="unrelated", title="EXCLUDED_CONTENT", reconstruction_id="unknown"
    )
    rejected["media"] = [dict(path=tmp_path / "missing.png", sha256="0" * 64)]
    rows.append(rejected)
    out = tmp_path / "out"
    build_results_catalog(
        sources, rows, out, updated_at="now", reconstruction_registry=registry
    )
    data = json.loads((out / "catalog.json").read_text())
    assert len(data["datasets"]) == 1 and data["organs"] == ["tongue"]
    for name in ("catalog.json", "catalog-data.js", "index.html"):
        assert "EXCLUDED_CONTENT" not in (out / name).read_text()
    assert len(list((out / "assets").iterdir())) == 1
    with pytest.raises(ValueError, match="requires"):
        build_results_catalog(sources, rows, out, updated_at="later")
    # Replaying an old baseline cannot republish an ineligible result.
    build_results_catalog(
        sources, rows, out, updated_at="later", reconstruction_registry=registry
    )
    assert len(json.loads((out / "catalog.json").read_text())["datasets"]) == 1


def test_new_catalog_cannot_bypass_registry_requirement(tmp_path):
    sources, rows, _ = catalog_example(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        build_results_catalog(sources, rows, tmp_path / "new", updated_at="now")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"eligibility_fingerprint": "0" * 64},
        {"organ": "kidney"},
        {"block_id": "wrong"},
        {"acquisition_ids": ["unknown"]},
        {"acquisition_ids": ["a0"], "acquisition_modalities": ["IHC"]},
        {"physical_section_id": "other"},
        {"evidence_kind": "inventory"},
    ],
)
def test_derived_records_require_exact_bindings(tmp_path, change):
    sources, rows, registry = catalog_example(tmp_path)
    rows[0].update(change)
    kept_sources, kept, removed, _ = filter_reconstruction_catalog(
        sources, rows, registry
    )
    assert kept_sources == kept == [] and len(removed) == 1


def test_empty_registry_and_policy_cache_invalidation(tmp_path):
    sources, rows, registry = catalog_example(tmp_path)
    out = tmp_path / "out"
    build_results_catalog(
        sources, rows, out, updated_at="same", reconstruction_registry=registry
    )
    old = json.loads((out / "catalog.json").read_text())
    registry["entries"][0]["assessment"]["reconstruction_qc"]["coverage"] = "pending"
    build_results_catalog(
        sources, rows, out, updated_at="same", reconstruction_registry=registry
    )
    new = json.loads((out / "catalog.json").read_text())
    assert new["sources"] == new["datasets"] == new["organs"] == []
    assert old["eligibility_policy"] != new["eligibility_policy"]
    assert "No eligible serial IHC datasets" in (out / "index.html").read_text()
    assert not list((out / "assets").glob("*"))


def test_repeated_image_cannot_be_relabelled_as_an_extra_plane():
    study, assessment = example()
    assessment["acquisition_qc"]["a2"]["image_sha256"] = assessment["acquisition_qc"][
        "a1"
    ]["image_sha256"]
    result = assess_reconstruction_eligibility(study, assessment)
    assert result["status"] == "excluded"
    assert "repeated_image_counted_as_physical_sections" in result["reasons"]


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_strict_review_old_urls_empty_catalog_and_block_navigation(tmp_path, prefix):
    import mimetypes
    from urllib.parse import unquote, urlsplit

    playwright = pytest.importorskip("playwright.sync_api")
    sources, rows, registry = catalog_example(tmp_path)
    # One specimen can contain independently selectable, eligible blocks.
    other = copy.deepcopy(registry["entries"][0])
    for section in other["study"]["sections"]:
        section["block_id"] = "block2"
    other["assessment"]["reconstruction_id"] = "stack2"
    other["study"], other["assessment"] = rebind(other["study"], other["assessment"])
    registry["entries"].append(other)
    row = copy.deepcopy(rows[0])
    row.update(
        id="result2",
        reconstruction_id="stack2",
        block_id="block2",
        title="Second block",
        eligibility_fingerprint=assess_reconstruction_eligibility(**other)[
            "fingerprint"
        ],
    )
    rows.append(row)
    out = tmp_path / "review"
    build_results_catalog(
        sources, rows, out, updated_at="now", reconstruction_registry=registry
    )
    origin = "http://histopia.test"
    base = prefix + "/review/"
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1450, "height": 1000})
        errors, escaped = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def respond(route):
            path = unquote(urlsplit(route.request.url).path)
            if not path.startswith(base):
                escaped.append(path)
                route.fulfill(status=404)
                return
            file = out / (path[len(base) :] or "index.html")
            route.fulfill(
                status=200 if file.is_file() else 404,
                content_type=mimetypes.guess_type(file)[0]
                or "application/octet-stream",
                body=file.read_bytes() if file.is_file() else b"",
            )

        page.route(origin + "/**", respond)
        for query in [
            "source=hpa",
            "source=serial&dataset=removed-crc",
            "source=serial&organ=kidney",
            "source=serial&reconstruction=removed-block",
        ]:
            page.goto(origin + base + "index.html?" + query)
            assert page.locator("#empty").is_visible()
            assert page.locator("#detail").is_hidden()
            assert page.locator("#datasets button").count() == 0
            page.reload()
            assert page.locator("#detail").is_hidden()
        page.goto(origin + base + "index.html?source=serial")
        assert page.locator("#title").inner_text() == "Eligible stack"
        assert page.locator("#methods").get_attribute("open") is None
        assert page.locator("#collection").is_hidden()
        page.locator("#reconstruction").select_option("stack2")
        assert page.locator("#title").inner_text() == "Second block"
        assert "reconstruction=stack2" in page.url
        page.reload()
        assert page.locator("#title").inner_text() == "Second block"
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.locator("html").evaluate("el => el.scrollWidth > innerWidth")
        # Revocation also removes the source picker, downloads and thumbnails.
        registry["entries"] = []
        build_results_catalog(
            sources, rows, out, updated_at="revoked", reconstruction_registry=registry
        )
        page.goto(origin + base + "index.html")
        assert page.locator("#empty").is_visible()
        assert page.locator("#sources button").count() == 0
        assert page.locator("#datasets img").count() == 0
        assert page.locator("#detail").is_hidden()
        assert not errors and not escaped
        browser.close()
