"""Curated rows stay authoritative without changing scientific identities."""

import copy

import pytest

from histopia.study._curated_metadata import reconcile_curated_metadata


def inputs():
    rows = [
        dict(
            scan_id="sha1-" + "a" * 40,
            source_id="ours",
            organ="pancreas",
            subject_id="12",
            filename="slide_JunB.ndpi",
            marker="JunB",
            section_order=8,
            z_um=28.0,
            z_spacing_kind="assumed",
            reconstruction_id="block-12",
            eligibility_fingerprint="b" * 64,
            upstream_fingerprint="c" * 64,
            review_href="../index.html?view=data-catalog",
        )
    ]
    table = [
        dict(
            mouse_id="12",
            **{"Tissue Type": "panc"},
            raw_name="slide_JunB",
            file_type="NDPI",
            antibody="cJun",
            antibody_type="nuclear",
            order="2",
            order_text="02",
            label="label supplied by owner",
            note="0.5",
            **{"Original File Path": "private/source/path"},
        )
    ]
    return rows, table


def project(rows, table):
    return reconcile_curated_metadata(
        rows,
        table,
        source_id="ours",
        organ="pancreas",
        filename="curated.csv",
        source_sha256="d" * 64,
    )


def test_curated_fields_win_without_relabeling_or_reordering_existing_results():
    rows, table = inputs()
    original = copy.deepcopy(rows)
    updated, report = project(rows, table)
    assert rows == original
    row = updated[0]
    assert row["marker"] == "cJun"
    assert row["analysis_marker"] == "JunB"
    assert row["table_order"] == "2"
    assert row["table_label"] == table[0]["label"]
    assert row["table_note"] == "0.5"
    assert row["metadata_marker_difference"]
    for key in [
        "section_order",
        "z_um",
        "z_spacing_kind",
        "reconstruction_id",
        "eligibility_fingerprint",
        "upstream_fingerprint",
        "review_href",
    ]:
        assert row[key] == original[0][key]
    assert report["source_rows"] == report["unique_scans"] == 1
    assert "private/source/path" not in str(updated) + str(report)


def test_repeated_source_rows_do_not_invent_planes_or_hide_conflicting_orders():
    rows, table = inputs()
    table.append(dict(table[0], order="3", order_text="03"))
    updated, report = project(rows, table)
    assert len(updated) == 1
    assert report["source_rows"] == 2 and report["unique_scans"] == 1
    assert report["duplicate_scans"] == 1
    assert [r["table_order"] for r in report["rows"]] == ["2", "3"]
    assert [r["metadata_row"] for r in report["rows"]] == [2, 3]
    assert all(r["metadata_duplicate_scan"] for r in report["rows"])
    assert updated[0]["table_order"] is None
    assert updated[0]["table_order_status"] == "conflicting source rows"
    assert updated[0]["section_order"] == 8


def test_missing_curated_values_remain_missing():
    rows, table = inputs()
    table[0].update(order="", antibody="", label="")
    updated, report = project(rows, table)
    assert updated[0]["table_order"] is None
    assert updated[0]["marker"] is None
    assert updated[0]["table_label"] is None
    assert report["rows"][0]["table_order_status"] == "missing"
    assert updated[0]["analysis_marker"] == "JunB"


@pytest.mark.parametrize("key,value", [("subject_id", "99"), ("source_id", "other")])
def test_filename_alone_does_not_join_another_mouse_or_cohort(key, value):
    rows, table = inputs()
    rows[0][key] = value
    updated, report = project(rows, table)
    assert updated[0]["marker"] == "JunB"
    assert report["matched_rows"] == 0 and report["unique_scans"] == 0
    assert report["rows"][0]["scan_id"] is None
    assert report["rows"][0]["review_href"] is None


def test_ambiguous_scan_identity_is_rejected():
    rows, table = inputs()
    rows.append(dict(rows[0], scan_id="sha1-" + "e" * 40))
    with pytest.raises(ValueError, match="multiple scan identities"):
        project(rows, table)
