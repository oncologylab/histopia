import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.registration import (
    RegistrationConfig,
    _pipeline,
    approve_mask_review,
    approve_registration_run,
    approve_section_order,
    register_sections,
    validate_registration_approval,
)
from histopia.registration._errors import RegistrationApprovalRequired
from histopia.registration._masking import TissueMaskResult
from histopia.registration._performance import (
    PERFORMANCE_FILENAME,
    load_performance_report,
)
from histopia.registration._pipeline import (
    _create_tissue_masks,
    _crop_to_mask,
    _load_automatic_mask_snapshot,
    _load_registration_thumbnails,
    _mask_artifact_fingerprint,
    _mask_artifact_paths,
    _mask_artifacts_are_current,
    _record_mask_artifacts,
)
from histopia.registration._slides import SlideGeometry


def test_register_sections_writes_thumbnail_result(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    fixed = np.full((80, 80, 3), 255, dtype=np.uint8)
    fixed[20:55, 22:58] = np.array([238, 223, 204], dtype=np.uint8)
    moving = np.roll(fixed, shift=(3, -4), axis=(0, 1))
    Image.fromarray(fixed).save(input_dir / "[#001] fixed.png")
    Image.fromarray(moving).save(input_dir / "[#002] moving.png")

    result = register_sections(
        RegistrationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            rigid_method="phase_correlation",
            max_processed_image_dim_px=80,
        )
    )

    result_path = output_dir / "registration_result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert len(payload["slides"]) == 2
    assert result.reference_slide.name == "[#001] fixed.png"
    assert payload["slides"][1]["aligned_to"].endswith("[#001] fixed.png")
    assert payload["slides"][1]["alignment_metrics"]["dice"] > 0.9
    assert (output_dir / "qc" / "[#001] fixed.mask_overlay.png").exists()
    assert (output_dir / "validation_report.md").exists()
    performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)
    assert performance["status"] == "completed"
    assert performance["slide_count"] == 2
    assert performance["registered_slide_count"] == 2
    assert performance["controls"]["thumbnail_workers"] == 1
    assert performance["stages"]["result_write"]["status"] == "completed"


def test_strict_registration_advances_through_exact_review_stages(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    image = np.full((90, 100, 3), 255, dtype=np.uint8)
    image[18:72, 22:78] = np.array([185, 100, 120], dtype=np.uint8)
    for index, shift in enumerate((0, 2, 4), start=1):
        Image.fromarray(np.roll(image, shift=(shift, 0), axis=(0, 1))).save(
            input_dir / f"[#{index:03d}] section.png"
        )
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        reference_slide="[#001] section.png",
        reference_policy="explicit",
        rigid_method="phase_correlation",
        section_order_strategy="anchored_similarity",
        require_approved_masks=True,
        require_approved_order=True,
        max_processed_image_dim_px=100,
    )

    with pytest.raises(RegistrationApprovalRequired) as mask_gate:
        register_sections(config)
    assert mask_gate.value.stage == "masks"
    assert len(mask_gate.value.pending_slides) == 3
    mask_performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)
    assert mask_performance["status"] == "review_required"
    assert mask_performance["review_stage"] == "masks"
    assert mask_performance["pending_slide_count"] == 3
    approve_mask_review(
        output_dir,
        reviewer="Test Reviewer",
        notes="Masks visually reviewed.",
    )

    with pytest.raises(RegistrationApprovalRequired) as order_gate:
        register_sections(config)
    assert order_gate.value.stage == "order"
    order_performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)
    assert order_performance["status"] == "review_required"
    assert order_performance["review_stage"] == "order"
    assert order_performance["pending_slide_count"] == 0
    order_payload = json.loads((output_dir / "section_order_review.json").read_text())
    assert order_payload["slides"][0]["slide"] == "[#001] section.png"
    assert order_payload["slides"][0]["fixed"] is True
    approve_section_order(
        output_dir,
        reviewer="Test Reviewer",
        notes="Order visually reviewed.",
    )

    result = register_sections(config)

    assert len(result.slides) == 3
    assert (output_dir / "registration_result.json").is_file()
    mask_payload = json.loads((output_dir / "mask_review.json").read_text())
    assert len(mask_payload["fingerprint"]) == 64
    assert mask_payload["reviewer"] == "Test Reviewer"
    assert json.loads((output_dir / "section_order_review.json").read_text())[
        "approved"
    ]
    assert load_performance_report(output_dir / PERFORMANCE_FILENAME)["status"] == (
        "completed"
    )
    approval = approve_registration_run(
        output_dir,
        reviewer="Test Reviewer",
        notes="Masks, order, and registration reviewed.",
    )
    approved_result = (output_dir / "registration_result.json").read_bytes()

    register_sections(config)

    assert (output_dir / "registration_result.json").read_bytes() == approved_result
    assert validate_registration_approval(output_dir) == approval


def test_registration_failure_is_observationally_recorded(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="no registration input slides"):
        register_sections(
            RegistrationConfig(
                input_dir=input_dir,
                output_dir=output_dir,
            )
        )

    performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)
    assert performance["status"] == "failed"
    assert performance["failure_type"] == "FileNotFoundError"
    assert performance["stages"]["slide_discovery"]["status"] == "failed"


def test_mask_artifact_manifest_requires_exact_complete_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    image = np.full((16, 20, 3), 180, dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=bool)
    mask[3:13, 4:16] = True
    result = TissueMaskResult(
        mask=mask,
        method="test",
        metrics={},
        accepted=True,
        warnings=[],
        candidate_masks={"candidate": mask.copy()},
    )
    paths = _mask_artifact_paths(
        output / "processed",
        output / "qc",
        output / "qc" / "mask_candidates",
        source,
        result,
    )
    fingerprint = _mask_artifact_fingerprint(source, image, result)
    manifest: dict[str, object] = {
        "schema": "histopia-registration-mask-artifacts-v1",
        "slides": {},
    }

    assert not _mask_artifacts_are_current(manifest, source, fingerprint, paths, output)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    _record_mask_artifacts(manifest, source, fingerprint, paths, output)

    assert _mask_artifacts_are_current(manifest, source, fingerprint, paths, output)
    paths[-1].unlink()
    assert not _mask_artifacts_are_current(manifest, source, fingerprint, paths, output)


def test_mask_artifact_fingerprint_changes_with_rendered_mask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    image = np.full((8, 9, 3), 180, dtype=np.uint8)
    first = np.zeros((8, 9), dtype=bool)
    second = first.copy()
    second[2, 3] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "test", {}, True, [])

    assert _mask_artifact_fingerprint(
        source, image, result(first)
    ) != _mask_artifact_fingerprint(source, image, result(second))


def test_tissue_crop_ignores_tiny_remote_artifact() -> None:
    image = np.full((200, 240, 3), 255, dtype=np.uint8)
    mask = np.zeros((200, 240), dtype=bool)
    mask[70:150, 120:210] = True
    mask[5:10, 5:10] = True

    crop = _crop_to_mask(image, mask, target_dim_px=200, padding_fraction=0)

    assert np.array_equal(crop.offset_xy, np.array([120.0, 70.0]))
    assert crop.image.shape[:2] == (178, 200)


def test_parallel_mask_creation_matches_sequential_results(tmp_path: Path) -> None:
    images = {}
    for index, shift in enumerate((0, 3, 6)):
        image = np.full((90, 110, 3), 255, dtype=np.uint8)
        image[20 + shift : 70 + shift, 25:85] = [175, 95, 120]
        images[tmp_path / f"section-{index}.png"] = image
    sequential = _create_tissue_masks(
        images,
        RegistrationConfig(tmp_path, tmp_path / "sequential", mask_workers=1),
    )
    parallel = _create_tissue_masks(
        images,
        RegistrationConfig(tmp_path, tmp_path / "parallel", mask_workers=2),
    )

    assert sequential.keys() == parallel.keys()
    for path in sequential:
        assert sequential[path].method == parallel[path].method
        assert np.array_equal(sequential[path].mask, parallel[path].mask)
        assert sequential[path].candidate_masks.keys() == (
            parallel[path].candidate_masks.keys()
        )
        for method in sequential[path].candidate_masks:
            assert np.array_equal(
                sequential[path].candidate_masks[method],
                parallel[path].candidate_masks[method],
            )


def test_parallel_group_masks_and_artifacts_match_serial_pipeline(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = np.full((90, 110, 3), 255, dtype=np.uint8)
    image[20:72, 25:85] = [175, 95, 120]
    for index, shift in enumerate((0, 3, 6)):
        Image.fromarray(np.roll(image, shift=(shift, -shift), axis=(0, 1))).save(
            input_dir / f"section-{index}.png"
        )

    outputs: list[tuple[dict[str, bytes], bytes, dict[str, object]]] = []
    for workers, name in ((1, "serial"), (3, "parallel")):
        output_dir = tmp_path / name
        register_sections(
            RegistrationConfig(
                input_dir,
                output_dir,
                reference_slide="section-0.png",
                reference_policy="explicit",
                rigid_method="phase_correlation",
                align_strategy="reference",
                max_processed_image_dim_px=110,
                mask_workers=workers,
                write_processed_images=True,
            )
        )
        artifact_paths = (
            *sorted((output_dir / "processed").glob("*.png")),
            *sorted(output_dir.glob("qc/*.mask_overlay.png")),
            *sorted((output_dir / "qc" / "mask_candidates").glob("*.png")),
        )
        outputs.append(
            (
                {
                    str(path.relative_to(output_dir)): path.read_bytes()
                    for path in artifact_paths
                },
                (output_dir / "mask_review.json").read_bytes(),
                load_performance_report(output_dir / PERFORMANCE_FILENAME),
            )
        )

    serial, parallel = outputs
    assert serial[0] == parallel[0]
    assert serial[1] == parallel[1]
    assert serial[2]["mask_artifact_slides_rendered"] == 3
    assert parallel[2]["mask_artifact_slides_rendered"] == 3
    assert parallel[2]["mask_artifact_slides_reused"] == 0
    assert parallel[2]["controls"]["mask_workers"] == 3
    assert parallel[2]["independent_mask_seconds"] >= 0
    assert parallel[2]["group_mask_seconds"] >= 0
    assert parallel[2]["mask_artifact_seconds"] >= 0


def test_mask_workers_must_be_positive(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "mask_workers must be positive"):
        RegistrationConfig(tmp_path, tmp_path / "output", mask_workers=0)


def test_parallel_thumbnail_loading_matches_sequential_and_preserves_order(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple(tmp_path / f"section-{index}.ndpi" for index in range(4))
    thread_names: set[str] = set()

    def fake_load(path: Path, max_dim_px: int):
        thread_names.add(threading.current_thread().name)
        index = paths.index(path)
        image = np.full((2, 3, 3), index + max_dim_px, dtype=np.uint16)
        geometry = SlideGeometry(
            native_shape=(20 + index, 30 + index),
            content_bbox_xywh=(0, 0, 30 + index, 20 + index),
            thumbnail_shape=(2, 3),
            bounds_source="test",
        )
        return image, geometry

    monkeypatch.setattr(_pipeline, "load_slide_thumbnail", fake_load)
    sequential = _load_registration_thumbnails(
        paths,
        RegistrationConfig(
            tmp_path,
            tmp_path / "sequential",
            thumbnail_workers=1,
            max_processed_image_dim_px=12,
        ),
    )
    parallel = _load_registration_thumbnails(
        paths,
        RegistrationConfig(
            tmp_path,
            tmp_path / "parallel",
            thumbnail_workers=3,
            max_processed_image_dim_px=12,
        ),
    )

    assert tuple(sequential[0]) == paths
    assert tuple(parallel[0]) == paths
    assert sequential[1] == parallel[1]
    for path in paths:
        assert np.array_equal(sequential[0][path], parallel[0][path])
    assert any(name.startswith("ThreadPoolExecutor") for name in thread_names)


def test_thumbnail_workers_must_be_positive(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(
        ValueError, "thumbnail_workers must be positive"
    ):
        RegistrationConfig(tmp_path, tmp_path / "output", thumbnail_workers=0)


def test_qc_workers_run_in_bounded_named_pool() -> None:
    barrier = threading.Barrier(3)
    thread_names: set[str] = set()
    lock = threading.Lock()

    def job() -> None:
        with lock:
            thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=5)

    _pipeline._run_qc_jobs((job for _ in range(3)), workers=3)

    assert len(thread_names) == 3
    assert all(name.startswith("histopia-qc") for name in thread_names)


def test_automatic_mask_snapshot_requires_exact_hash_and_slide_set(
    tmp_path: Path,
) -> None:
    slide = tmp_path / "section.ndpi"
    image = np.full((20, 24, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[4:16, 6:18] = 255
    mask_path = tmp_path / "section.mask.png"
    Image.fromarray(mask).save(mask_path)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slides": [
                    {
                        "slide": slide.name,
                        "mask": mask_path.name,
                        "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )

    loaded = _load_automatic_mask_snapshot(
        manifest,
        (slide,),
        {slide: image},
    )

    assert loaded[slide].method == "approved_automatic_snapshot"
    assert np.array_equal(loaded[slide].mask, mask > 127)

    mask_path.write_bytes(b"changed")
    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        _load_automatic_mask_snapshot(manifest, (slide,), {slide: image})


def test_anchored_order_reuses_exact_distance_cache(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base = np.full((80, 80, 3), 255, dtype=np.uint8)
    base[18:62, 22:58] = np.array([185, 100, 120], dtype=np.uint8)
    for index, shift in enumerate((0, 2, 4), start=1):
        Image.fromarray(np.roll(base, shift=(shift, 0), axis=(0, 1))).save(
            input_dir / f"[#{index:03d}] section.png"
        )
    original = _pipeline._section_distance_matrix
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_pipeline, "_section_distance_matrix", counted)
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rigid_method="phase_correlation",
        section_order_strategy="anchored_similarity",
        ordering_workers=2,
        max_processed_image_dim_px=80,
    )

    register_sections(config)
    first = json.loads((output_dir / "section_order_review.json").read_text())
    register_sections(config)
    second = json.loads((output_dir / "section_order_review.json").read_text())

    assert calls == 1
    assert first["fingerprint"] == second["fingerprint"]
    assert (output_dir / ".cache" / "section-order-distances.npz").is_file()

    sequential_output = tmp_path / "sequential"
    register_sections(
        RegistrationConfig(
            input_dir=input_dir,
            output_dir=sequential_output,
            rigid_method="phase_correlation",
            section_order_strategy="anchored_similarity",
            ordering_workers=1,
            max_processed_image_dim_px=80,
        )
    )
    with np.load(
        output_dir / ".cache" / "section-order-distances.npz",
        allow_pickle=False,
    ) as parallel_cache:
        parallel = parallel_cache["distances"]
    with np.load(
        sequential_output / ".cache" / "section-order-distances.npz",
        allow_pickle=False,
    ) as sequential_cache:
        sequential = sequential_cache["distances"]
    assert np.array_equal(parallel, sequential)


def test_registration_reuses_exact_rigid_pair_cache(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fixed = np.full((90, 100, 3), 255, dtype=np.uint8)
    fixed[18:72, 22:78] = np.array([185, 100, 120], dtype=np.uint8)
    moving = np.roll(fixed, shift=(3, -4), axis=(0, 1))
    Image.fromarray(fixed).save(input_dir / "fixed.png")
    Image.fromarray(moving).save(input_dir / "moving.png")
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        reference_slide="fixed.png",
        reference_policy="explicit",
        rigid_method="phase_correlation",
        align_strategy="reference",
        max_processed_image_dim_px=100,
        write_processed_images=False,
    )

    first = register_sections(config)
    first_json = (output_dir / "registration_result.json").read_bytes()
    first_performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)

    def unexpected_estimate(*_args, **_kwargs):
        raise AssertionError("exact rigid pair should have been loaded from cache")

    monkeypatch.setattr(_pipeline, "_estimate_pair_transform", unexpected_estimate)
    second = register_sections(config)
    second_performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)

    assert first.to_json_dict() == second.to_json_dict()
    assert (output_dir / "registration_result.json").read_bytes() == first_json
    assert first_performance["rigid_pair_cache_hits"] == 0
    assert first_performance["rigid_pair_cache_misses"] == 1
    assert first_performance["rigid_pairs_computed"] == 1
    assert second_performance["rigid_pair_cache_hits"] == 1
    assert second_performance["rigid_pair_cache_misses"] == 0
    assert second_performance["rigid_pairs_computed"] == 0
    assert len(tuple((output_dir / ".cache" / "rigid_pairs").glob("*.json"))) == 1


def test_registration_reuses_checksum_validated_qc_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fixed = np.full((80, 90, 3), 255, dtype=np.uint8)
    fixed[16:64, 20:70] = np.array([185, 100, 120], dtype=np.uint8)
    Image.fromarray(fixed).save(input_dir / "fixed.png")
    Image.fromarray(np.roll(fixed, shift=(2, -3), axis=(0, 1))).save(
        input_dir / "moving.png"
    )
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        reference_slide="fixed.png",
        reference_policy="explicit",
        rigid_method="phase_correlation",
        align_strategy="reference",
        max_processed_image_dim_px=90,
        qc_workers=3,
        write_processed_images=True,
    )

    register_sections(config)
    first = load_performance_report(output_dir / PERFORMANCE_FILENAME)
    corrupted_path = output_dir / "qc" / "alignment" / "moving.warped.png"
    unchanged_path = output_dir / "qc" / "review" / "fixed.review.png"
    expected_corrupted_bytes = corrupted_path.read_bytes()
    unchanged_mtime = unchanged_path.stat().st_mtime_ns

    def unexpected_render(*_args, **_kwargs):
        raise AssertionError("exact QC artifact should have been reused")

    monkeypatch.setattr(_pipeline, "save_rgb", unexpected_render)
    monkeypatch.setattr(_pipeline, "write_labeled_review_panel", unexpected_render)
    register_sections(config)
    second = load_performance_report(output_dir / PERFORMANCE_FILENAME)

    assert first["qc_artifact_cache_hits"] == 0
    assert first["qc_artifact_cache_misses"] == 5
    assert first["qc_artifact_bundles_rendered"] == 5
    assert first["controls"]["qc_workers"] == 3
    assert second["qc_artifact_cache_hits"] == 5
    assert second["qc_artifact_cache_misses"] == 0
    assert second["qc_artifact_bundles_rendered"] == 0
    assert (output_dir / ".cache" / "registration-qc-artifacts.json").is_file()

    monkeypatch.undo()
    corrupted_path.write_bytes(b"corrupt")
    register_sections(config)
    repaired = load_performance_report(output_dir / PERFORMANCE_FILENAME)

    assert corrupted_path.read_bytes() == expected_corrupted_bytes
    assert unchanged_path.stat().st_mtime_ns == unchanged_mtime
    assert repaired["qc_artifact_cache_hits"] == 4
    assert repaired["qc_artifact_cache_misses"] == 1
    assert repaired["qc_artifact_bundles_rendered"] == 1


def test_parallel_qc_rendering_is_byte_identical_to_serial(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    image = np.full((75, 85, 3), 255, dtype=np.uint8)
    image[15:60, 20:65] = np.array([185, 100, 120], dtype=np.uint8)
    for index, shift in enumerate((0, 2, 4), start=1):
        Image.fromarray(np.roll(image, shift=(shift, -shift), axis=(0, 1))).save(
            input_dir / f"section-{index}.png"
        )
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        reference_slide="section-2.png",
        reference_policy="explicit",
        rigid_method="phase_correlation",
        align_strategy="hybrid",
        max_processed_image_dim_px=85,
        qc_workers=1,
        alignment_cache=False,
        write_processed_images=True,
    )

    register_sections(config)
    serial = {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in sorted((output_dir / "qc").rglob("*.png"))
    }
    serial_result = (output_dir / "registration_result.json").read_bytes()

    config.qc_workers = 3
    register_sections(config)
    parallel = {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in sorted((output_dir / "qc").rglob("*.png"))
    }

    assert parallel == serial
    assert (output_dir / "registration_result.json").read_bytes() == serial_result


def test_hybrid_renders_only_final_serial_pair_diagnostics(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    image = np.full((70, 80, 3), 255, dtype=np.uint8)
    image[14:56, 18:62] = np.array([185, 100, 120], dtype=np.uint8)
    for index, shift in enumerate((0, 2, 4), start=1):
        Image.fromarray(np.roll(image, shift=(shift, 0), axis=(0, 1))).save(
            input_dir / f"section-{index}.png"
        )

    register_sections(
        RegistrationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            reference_slide="section-2.png",
            reference_policy="explicit",
            rigid_method="phase_correlation",
            align_strategy="hybrid",
            max_processed_image_dim_px=80,
            write_processed_images=True,
        )
    )
    performance = load_performance_report(output_dir / PERFORMANCE_FILENAME)

    assert len(tuple((output_dir / "qc" / "alignment" / "pair_crops").glob("*"))) == 8
    assert performance["qc_artifact_cache_misses"] == 9
    assert performance["qc_artifact_bundles_rendered"] == 9


def test_hybrid_registration_reuses_features_without_changing_results(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(41)
    base = rng.integers(0, 256, size=(140, 160, 3), dtype=np.uint8)
    paths = tuple(tmp_path / f"section-{index}.png" for index in range(4))
    shifts = ((0, 0), (2, -3), (4, -5), (6, -8))
    crops = {
        path: _pipeline._Crop(
            image=np.roll(base, shift=shift, axis=(0, 1)),
            mask=np.ones(base.shape[:2], dtype=bool),
            offset_xy=np.zeros(2, dtype=float),
            scale=1.0,
        )
        for path, shift in zip(paths, shifts, strict=True)
    }
    config = RegistrationConfig(
        tmp_path,
        tmp_path / "output",
        rigid_method="feature",
        align_strategy="hybrid",
        ordering_workers=1,
        write_processed_images=False,
    )
    config.refinement.enabled = False
    reference = paths[1]

    prepare_crop_features = _pipeline._prepare_crop_features
    monkeypatch.setattr(_pipeline, "_prepare_crop_features", lambda *_args, **_kw: None)
    baseline, baseline_parents = _pipeline._estimate_hybrid_transforms(
        paths,
        reference,
        crops,
        config,
        tmp_path / "baseline",
    )

    detections = 0
    prepare_rigid_features = _pipeline.prepare_rigid_features

    def counted_prepare(image, mask):
        nonlocal detections
        detections += 1
        return prepare_rigid_features(image, mask)

    monkeypatch.setattr(_pipeline, "_prepare_crop_features", prepare_crop_features)
    monkeypatch.setattr(_pipeline, "prepare_rigid_features", counted_prepare)
    optimized, optimized_parents = _pipeline._estimate_hybrid_transforms(
        paths,
        reference,
        crops,
        config,
        tmp_path / "optimized",
    )

    assert detections == len(paths)
    assert optimized_parents == baseline_parents
    assert optimized.keys() == baseline.keys()
    for path in optimized:
        assert np.array_equal(optimized[path].matrix, baseline[path].matrix)
        assert optimized[path].method == baseline[path].method
        assert optimized[path].match_count == baseline[path].match_count
        assert optimized[path].inlier_count == baseline[path].inlier_count
        assert optimized[path].warnings == baseline[path].warnings
