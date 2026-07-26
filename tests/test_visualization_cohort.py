from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from histopia.semantic._result import _seal_semantic_result
from histopia.visualization import build_section_viewer
from histopia.visualization._viewer import _rasterize_semantic_rectangles


def test_viewer_embeds_seven_mouse_qc_and_exact_review_state(tmp_path: Path) -> None:
    runs: dict[str, Path] = {}
    semantic_runs: dict[str, Path] = {}
    qc_rows: list[dict[str, object]] = []
    for index in range(7):
        mouse_id = str(4_000 + index)
        runs[mouse_id], semantic_runs[mouse_id], fingerprint = _write_mouse(
            tmp_path,
            mouse_id,
            with_topology=index == 0,
        )
        qc_rows.append(
            {
                "mouse_id": mouse_id,
                "fingerprint": fingerprint,
                "selected_k": 7,
                "review_approved": False,
                "flags": ([str(tmp_path / "must-not-be-public")] if index == 0 else []),
                "topology_coverage": 0.8,
                "median_topology_confidence": 0.75,
                "local_path": str(tmp_path / "must-not-be-public"),
            }
        )
    cohort_path = tmp_path / "cohort_qc.json"
    cohort_path.write_text(json.dumps({"schema_version": 1, "mice": qc_rows}))

    with pytest.raises(ValueError, match="invalid cohort QC flag"):
        build_section_viewer(
            runs,
            tmp_path / "viewer-invalid",
            semantic_runs=semantic_runs,
            cohort_qc=cohort_path,
        )

    qc_rows[0]["flags"] = []
    cohort_path.write_text(json.dumps({"schema_version": 1, "mice": qc_rows}))
    index = build_section_viewer(
        runs,
        tmp_path / "viewer",
        semantic_runs=semantic_runs,
        cohort_qc=cohort_path,
    )

    manifest_text = (index.parent / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert len(manifest["mice"]) == 7
    assert str(tmp_path) not in manifest_text
    semantic = manifest["mice"][0]["semantic"]
    assert semantic["cluster_counts"] == list(range(5, 16))
    assert semantic["qc"]["topology_coverage"] == 0.8
    assert list(semantic["qc"]) == [
        "fingerprint",
        "selected_k",
        "median_topology_confidence",
        "topology_coverage",
        "unsupported_sections",
        "review_approved",
        "flags",
    ]
    assert semantic["review"] == {
        "approved": False,
        "fingerprint_matches": True,
    }
    binding = semantic["registration_binding"]
    assert binding["preflight_schema_version"] == 2
    assert len(binding["preflight_fingerprint"]) == 64
    assert len(binding["registration_result_sha256"]) == 64
    assert binding["registration_approval_sha256"] is None
    assert binding["approval_bound"] is False
    assert semantic["link_pair_count"] == 1
    topology = json.loads((index.parent / semantic["links_url"]).read_text())
    assert topology["links"][0]["accepted_links"] == 600
    assert topology["links"][0]["displayed_links"] == 500
    viewer = (index.parent / "viewer.js").read_text()
    assert "Approval required" in viewer
    assert "topology coverage" in viewer


def test_viewer_requires_qc_for_every_semantic_mouse(tmp_path: Path) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=False)
    cohort_path = tmp_path / "cohort_qc.json"
    cohort_path.write_text(json.dumps({"schema_version": 1, "mice": []}))

    with pytest.raises(ValueError, match="missing mouse 4000"):
        build_section_viewer(
            {"4000": run},
            tmp_path / "viewer",
            semantic_runs={"4000": semantic_run},
            cohort_qc=cohort_path,
        )


def test_viewer_rejects_semantics_from_changed_registration(tmp_path: Path) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=False)
    result_path = run / "registration_result.json"
    result = json.loads(result_path.read_text())
    result["changed_after_feature_extraction"] = True
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="different registration result"):
        build_section_viewer(
            {"4000": run},
            tmp_path / "viewer",
            semantic_runs={"4000": semantic_run},
        )


def test_viewer_requires_complete_semantic_approval_metadata(tmp_path: Path) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=False)
    review_path = semantic_run / "semantic_review.json"
    review = json.loads(review_path.read_text())
    review.update({"approved": True, "reviewer": "Reviewer", "notes": ""})
    review_path.write_text(json.dumps(review))

    index = build_section_viewer(
        {"4000": run},
        tmp_path / "viewer",
        semantic_runs={"4000": semantic_run},
    )

    semantic = json.loads((index.parent / "manifest.json").read_text())["mice"][0][
        "semantic"
    ]
    assert semantic["review"] == {
        "approved": False,
        "fingerprint_matches": True,
    }


def test_viewer_reuses_checksum_verified_mouse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=False)
    output = tmp_path / "viewer"

    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
    )
    asset = next((output / "assets" / "4000").glob("*.webp"))
    original_mtime = asset.stat().st_mtime_ns
    monkeypatch.setattr(
        "histopia.visualization._viewer._read_rgb",
        lambda *args, **kwargs: pytest.fail("unchanged mouse was decoded"),
    )
    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
    )

    report = json.loads((output / "build-report.json").read_text())
    assert report["assets_encoded"] == 0
    assert report["assets_reused"] == 13
    assert report["mice_reused"] == 1
    assert report["mice_rendered"] == 0
    assert asset.stat().st_mtime_ns == original_mtime
    viewer_js = (output / "viewer.js").read_text()
    assert "showLinks.disabled = !linksAvailable" in viewer_js


def test_viewer_updates_qc_without_rerendering_mouse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, semantic_run, fingerprint = _write_mouse(
        tmp_path,
        "4000",
        with_topology=False,
    )
    output = tmp_path / "viewer"
    cohort_path = tmp_path / "cohort_qc.json"
    row = {
        "mouse_id": "4000",
        "fingerprint": fingerprint,
        "review_approved": False,
        "flags": [],
    }
    cohort_path.write_text(json.dumps({"schema_version": 1, "mice": [row]}))
    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
        cohort_qc=cohort_path,
    )

    row["flags"] = ["cohort_high_patch_count"]
    cohort_path.write_text(json.dumps({"schema_version": 1, "mice": [row]}))
    monkeypatch.setattr(
        "histopia.visualization._viewer._read_rgb",
        lambda *args, **kwargs: pytest.fail("QC-only update decoded the mouse"),
    )
    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
        cohort_qc=cohort_path,
    )

    report = json.loads((output / "build-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["mice_reused"] == 1
    assert report["mice_rendered"] == 0
    assert report["assets_encoded"] == 0
    assert manifest["mice"][0]["semantic"]["qc"]["flags"] == ["cohort_high_patch_count"]


def test_viewer_rerenders_mouse_when_topology_output_is_changed(
    tmp_path: Path,
) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=True)
    output = tmp_path / "viewer"
    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
    )
    topology = output / "assets" / "4000" / "topology.json"
    topology.write_text("{}")

    build_section_viewer(
        {"4000": run},
        output,
        semantic_runs={"4000": semantic_run},
    )

    report = json.loads((output / "build-report.json").read_text())
    assert report["mice_reused"] == 0
    assert report["mice_rendered"] == 1
    assert json.loads(topology.read_text())["links"]


def test_parallel_viewer_encoding_matches_serial_output(tmp_path: Path) -> None:
    run, semantic_run, _ = _write_mouse(tmp_path, "4000", with_topology=True)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    build_section_viewer(
        {"4000": run},
        serial,
        semantic_runs={"4000": semantic_run},
        workers=1,
    )
    build_section_viewer(
        {"4000": run},
        parallel,
        semantic_runs={"4000": semantic_run},
        workers=4,
    )

    serial_files = {
        path.relative_to(serial): path.read_bytes()
        for path in serial.rglob("*")
        if path.is_file() and path.name != "build-report.json"
    }
    parallel_files = {
        path.relative_to(parallel): path.read_bytes()
        for path in parallel.rglob("*")
        if path.is_file() and path.name != "build-report.json"
    }
    assert parallel_files == serial_files
    report = json.loads((parallel / "build-report.json").read_text())
    assert report["workers"] == 4
    assert report["compute_backend"] == "cpu"
    assert report["peak_pending_assets"] == 8
    assert report["assets_encoded"] == 26


def test_viewer_rejects_nonpositive_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="viewer workers must be positive"):
        build_section_viewer({}, tmp_path / "viewer", workers=0)


@pytest.mark.parametrize(
    ("bounds", "shape"),
    [
        (
            np.array(
                [
                    [-3, -2, 4, 5],
                    [2, 3, 10, 11],
                    [7, 1, 16, 8],
                    [20, 20, 24, 24],
                ]
            ),
            (14, 18),
        ),
        (np.array([[0, 0, 2_000, 2_000]]), (18, 20)),
    ],
)
def test_semantic_rectangle_rasterizer_matches_ordered_pillow_paint(
    bounds: np.ndarray,
    shape: tuple[int, int],
) -> None:
    labels = np.arange(len(bounds), dtype=np.int16) % 3
    colors = np.array(
        [
            [210, 30, 45, 220],
            [25, 145, 80, 220],
            [45, 95, 210, 220],
        ],
        dtype=np.uint8,
    )
    expected = Image.new("RGBA", (shape[1], shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(expected)
    for label, box in zip(labels, bounds, strict=True):
        draw.rectangle(tuple(int(value) for value in box), fill=tuple(colors[label]))

    actual = _rasterize_semantic_rectangles(labels, bounds, colors, shape)

    assert np.array_equal(actual, np.asarray(expected))


def _write_mouse(
    root: Path,
    mouse_id: str,
    *,
    with_topology: bool,
) -> tuple[Path, Path, str]:
    run_dir = root / "registration" / mouse_id
    processed = run_dir / "processed"
    processed.mkdir(parents=True)
    semantic_dir = root / "semantic" / mouse_id
    slide_count = 2 if with_topology else 1
    geometry = {
        "native_shape": [200, 200],
        "content_bbox_xywh": [0, 0, 200, 200],
        "thumbnail_shape": [20, 20],
        "bounds_source": "test",
        "mpp_xy": [0.5, 0.5],
        "mpp_source": "test",
    }
    registration_slides: list[dict[str, object]] = []
    semantic_slides: list[dict[str, object]] = []
    for slide_index in range(slide_count):
        name = f"section-{slide_index + 1}.ndpi"
        image = np.full((20, 20, 3), 220 - slide_index * 10, dtype=np.uint8)
        Image.fromarray(image).save(
            processed / f"section-{slide_index + 1}.thumbnail.png"
        )
        Image.fromarray(np.full((20, 20), 255, dtype=np.uint8)).save(
            processed / f"section-{slide_index + 1}.mask.png"
        )
        registration_slides.append(
            {
                "path": str(root / name),
                "is_reference": slide_index == 0,
                "geometry": geometry,
                "transform": {"matrix": np.eye(3).tolist()},
            }
        )
        labels: dict[str, str] = {}
        for k in range(5, 16):
            label_path = Path("labels") / f"k-{k}" / f"{slide_index + 1:03d}.npz"
            (semantic_dir / label_path.parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                semantic_dir / label_path,
                labels=np.array([0, 1], dtype=np.int16),
                reference_um_xy=np.array([[25, 25], [75, 75]], dtype=float),
                patch_size_px=np.int32(100),
                analysis_mpp=np.float64(0.5),
            )
            labels[str(k)] = str(label_path)
        semantic_slides.append({"id": name, "labels": labels})
    registration_path = run_dir / "registration_result.json"
    registration_path.write_text(
        json.dumps(
            {
                "reference_slide": str(root / "section-1.ndpi"),
                "slides": registration_slides,
            }
        )
    )
    preflight_slides = [
        {
            "slide_name": Path(str(row["path"])).name,
            "source_sha256": f"source-{index}",
            "thumbnail_sha256": f"thumbnail-{index}",
            "mask_sha256": f"mask-{index}",
            "mask_method": "test",
            "mask_review_status": "auto_pass",
            "transform_sha256": f"transform-{index}",
            "thumbnail_shape": [20, 20],
            "mpp_xy": [0.5, 0.5],
            "is_reference": bool(row["is_reference"]),
        }
        for index, row in enumerate(registration_slides)
    ]
    preflight_core = {
        "schema_version": 2,
        "registration_result_sha256": hashlib.sha256(
            registration_path.read_bytes()
        ).hexdigest(),
        "order_review_fingerprint": None,
        "reference_slide": "section-1.ndpi",
        "slides": preflight_slides,
    }
    preflight_fingerprint = hashlib.sha256(
        json.dumps(
            preflight_core,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (semantic_dir / "preflight.json").write_text(
        json.dumps(
            {
                **preflight_core,
                "registration_run": str(run_dir),
                "slides": [
                    {
                        **row,
                        "source_path": str(registration_slides[index]["path"]),
                    }
                    for index, row in enumerate(preflight_slides)
                ],
                "fingerprint": preflight_fingerprint,
                "slide_count": slide_count,
            }
        )
    )
    topology_pairs: list[dict[str, object]] = []
    if with_topology:
        topology_dir = semantic_dir / "topology"
        topology_dir.mkdir()
        artifact = topology_dir / "001-002.npz"
        points = np.column_stack(
            [
                np.linspace(5, 95, 600),
                np.linspace(95, 5, 600),
            ]
        )
        np.savez_compressed(
            artifact,
            source_um_xy=points,
            target_um_xy=points + 1,
            confidence=np.linspace(0.5, 1.0, 600),
        )
        topology_pairs.append(
            {
                "source_section": 0,
                "target_section": 1,
                "accepted_links": 600,
                "artifact": "topology/001-002.npz",
            }
        )
    np.savez_compressed(semantic_dir / "atlas_model.npz", pca_mean=np.zeros(2))
    core = {
        "schema_version": 3,
        "selected_k": 7,
        "primary_clusters": 7,
        "cluster_counts": list(range(5, 16)),
        "palette": ["#d73027"] * 15,
        "model": "atlas_model.npz",
        "slides": semantic_slides,
        "topology_pairs": topology_pairs,
        "batch_correction": {
            "accepted": True,
            "raw": {"slide_variance_fraction": 0.3},
            "corrected": {"slide_variance_fraction": 0.02},
        },
        "k_selection": [{"k": 7, "composite_score": 0.8}],
        "feature_provenance": {
            "preflight_fingerprint": preflight_fingerprint,
            "expected_slide_ids": [
                Path(str(row["path"])).name for row in registration_slides
            ],
        },
    }
    payload = _seal_semantic_result(semantic_dir, core)
    (semantic_dir / "semantic_result.json").write_text(json.dumps(payload))
    fingerprint = str(payload["fingerprint"])
    (semantic_dir / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": False,
                "fingerprint": fingerprint,
                "reviewer": str(root / "private-reviewer-path"),
                "notes": str(root / "private-notes-path"),
            }
        )
    )
    return run_dir, semantic_dir, fingerprint
