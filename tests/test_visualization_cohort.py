from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.semantic._result import _seal_semantic_result
from histopia.visualization import build_section_viewer


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
    assert semantic["review"] == {
        "approved": False,
        "fingerprint_matches": True,
    }
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
    (run_dir / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(root / "section-1.ndpi"),
                "slides": registration_slides,
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
