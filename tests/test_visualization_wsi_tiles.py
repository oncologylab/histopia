from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from histopia.stain._artifacts import AdaptiveStainMap, StainMap
from histopia.visualization import _wsi_tiles
from histopia.visualization._wsi_tiles import (
    WsiLayer,
    WsiLevel,
    WsiSection,
    WsiTileCapacityError,
    WsiTileService,
    _load_cohort_sections,
    _mask_focus_bbox_fraction,
    _sample_calibrated_od,
    _stain_layer,
    _StainMapSpec,
    _validated_stain_maps,
    _virtual_dzi_levels,
)


def _service(tmp_path: Path) -> tuple[WsiTileService, str]:
    image = tmp_path / "registered.tiff"
    image.write_bytes(b"registered")
    digest = hashlib.sha256(b"registered").hexdigest()
    layer = WsiLayer(
        name="registered",
        path=image,
        digest=digest,
        levels=(
            WsiLevel(256, 128, 1),
            WsiLevel(1024, 512, 0),
        ),
        tile_size=256,
        microns_per_pixel=0.5,
    )
    section = WsiSection(
        cohort="mouse",
        section="001",
        slide="slide.ndpi",
        label="H&E",
        reference=True,
        layers={"registered": layer},
    )
    return (
        WsiTileService(
            {("mouse", "001"): section},
            max_concurrent_tiles=1,
        ),
        digest,
    )


def test_wsi_metadata_is_path_free(tmp_path: Path) -> None:
    service, digest = _service(tmp_path)

    payload = service.metadata("mouse", "001")
    catalog = service.catalog("mouse")

    assert payload["slide"] == "slide.ndpi"
    assert payload["layers"]["registered"] == {
        "digest": digest,
        "tile_size": 256,
        "width": 1024,
        "height": 512,
        "levels": [
            {"width": 256, "height": 128},
            {"width": 1024, "height": 512},
        ],
        "microns_per_pixel": 0.5,
        "format": "jpg",
    }
    assert str(tmp_path) not in str(payload)
    assert catalog == {
        "schema_version": 1,
        "cohort": "mouse",
        "sections": [
            {
                "section": "001",
                "slide": "slide.ndpi",
                "label": "H&E",
                "reference": True,
                "layers": ["registered"],
            }
        ],
    }
    assert str(tmp_path) not in str(catalog)
    with pytest.raises(FileNotFoundError, match="cohort"):
        service.catalog("unknown")


def test_protein_focus_bbox_is_normalized_and_path_free(tmp_path: Path) -> None:
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[10:30, 20:60] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    focus = _mask_focus_bbox_fraction(mask_path, padding_fraction=0.10)
    assert focus == pytest.approx((0.16, 0.16, 0.48, 0.48))

    service, _digest = _service(tmp_path)
    base = service.section("mouse", "001")
    focused = WsiSection(
        cohort=base.cohort,
        section=base.section,
        slide=base.slide,
        label=base.label,
        reference=base.reference,
        layers=base.layers,
        focus_bbox_fraction=focus,
    )
    payload = WsiTileService({("mouse", "001"): focused}).metadata("mouse", "001")

    assert payload["focus_bbox"] == {
        "x": pytest.approx(0.16),
        "y": pytest.approx(0.16),
        "width": pytest.approx(0.48),
        "height": pytest.approx(0.48),
        "coordinate_space": "normalized_native_content_bbox",
    }
    assert str(tmp_path) not in str(payload)


def test_large_cell_and_stain_artifacts_are_verified_on_first_use(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "cells.tiff"
    labels.write_bytes(b"sealed-labels")
    label_digest = hashlib.sha256(labels.read_bytes()).hexdigest()
    cell_layer = WsiLayer(
        name="cells",
        path=labels,
        digest=label_digest,
        levels=(WsiLevel(1, 1, 0),),
        tile_size=256,
        microns_per_pixel=0.5,
        label_path=labels,
        label_artifact_digest=label_digest,
    )
    WsiTileService({})._validate_cell_labels(cell_layer)
    labels.write_bytes(b"changed-labels")
    with pytest.raises(ValueError, match="digest mismatch"):
        WsiTileService({})._validate_cell_labels(cell_layer)

    map_path, stain_map = _stain_map(tmp_path / "map.npz", raw=0.5, corrected=0.2)
    map_digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    stain_layer = WsiLayer(
        name="stain_output",
        path=map_path,
        digest="a" * 64,
        levels=(WsiLevel(5, 3, 0),),
        tile_size=512,
        microns_per_pixel=4.0,
        stain_artifact_digest=map_digest,
    )
    assert (
        WsiTileService({})._load_stain_map(stain_layer).slide_id == stain_map.slide_id
    )
    with map_path.open("ab") as stream:
        stream.write(b"stale")
    with pytest.raises(ValueError, match="digest mismatch"):
        WsiTileService({})._load_stain_map(stain_layer)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("KPF_panc_Ecad-[1]", "E-Cad"),
        ("KPF-panc_E-Cad-[1]", "E-Cad"),
        ("KPF_panc_CK19-[1]", "CK19"),
        ("KPF_panc_Yap-[1]", "YAP"),
        ("KPF_panc_Sirius Red-[1]", "Sirius Red"),
    ],
)
def test_marker_labels_normalize_supported_protein_names(
    stem: str,
    expected: str,
) -> None:
    assert _wsi_tiles._marker_label(stem) == expected


def test_default_protein_model_prefers_best_promoted_held_out_score() -> None:
    summaries = [
        {
            "id": "candidate",
            "promoted": False,
            "approved": False,
            "metrics": {"spearman_64um": 0.99},
        },
        {
            "id": "weak-promoted",
            "promoted": True,
            "approved": True,
            "metrics": {"spearman_64um": 0.58, "spearman": 0.45},
        },
        {
            "id": "strong-promoted",
            "promoted": True,
            "approved": True,
            "metrics": {"spearman_64um": 0.76, "spearman": 0.61},
        },
    ]

    assert _wsi_tiles._select_default_protein_model(summaries) == "strong-promoted"


def test_protein_defaults_are_independent_for_measured_and_transfer_sections() -> None:
    summaries = [
        {
            "id": "yap-fit",
            "target_id": "yap",
            "prediction_protocol": "training-visible",
            "approved": True,
            "metrics": {"spearman_64um": 0.98},
        },
        {
            "id": "yap-transfer",
            "target_id": "yap",
            "prediction_protocol": "leave-one-mouse-out",
            "promoted": True,
            "approved": True,
            "metrics": {"spearman_64um": 0.66},
        },
        {
            "id": "ki67-transfer",
            "target_id": "ki67",
            "prediction_protocol": "leave-one-mouse-out",
            "metrics": {"spearman_64um": 0.55},
        },
        {
            "id": "ecad-fit",
            "target_id": "ecad",
            "prediction_protocol": "training-visible",
            "approved": True,
            "metrics": {"spearman_64um": 0.92},
        },
    ]

    selected = _wsi_tiles._select_recommended_protein_models(summaries)

    assert selected["yap"] == {
        "measured": "yap-fit",
        "unmeasured": "yap-transfer",
    }
    assert "ki67" not in selected
    assert selected["ecad"] == {"measured": "ecad-fit"}


@pytest.mark.parametrize(
    ("layer_names", "counterstain_v3", "expected"),
    [
        (
            {"raw", "stain_adaptive_map", "stain_adaptive_v3_map"},
            True,
            (
                "stain_adaptive_v3_map",
                "current_marker_counterstain_conditioned_od_4um",
            ),
        ),
        (
            {"raw", "stain_adaptive_map", "protein_tissue_support"},
            False,
            ("stain_adaptive_map", "current_marker_adaptive_od_4um"),
        ),
        (
            {"raw", "protein_tissue_support", "mask"},
            False,
            ("raw", "native_histology_context_only"),
        ),
        (
            {"protein_tissue_support", "mask"},
            False,
            ("protein_tissue_support", "no_quantified_stain"),
        ),
    ],
)
def test_unmeasured_protein_context_prefers_actual_current_slide(
    layer_names: set[str],
    counterstain_v3: bool,
    expected: tuple[str, str],
) -> None:
    layers = dict.fromkeys(layer_names)

    assert (
        _wsi_tiles._select_protein_current_layer(
            layers,
            counterstain_v3=counterstain_v3,
        )
        == expected
    )


def test_shared_study_model_is_validated_once_for_multiple_cohorts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protein = tmp_path / "protein"
    protein.mkdir()
    result = {
        "schema_version": 4,
        "model_id": "model",
        "model_label": "YAP model",
        "target_id": "yap",
        "target_label": "YAP curated",
        "architecture": "extra_trees",
        "training_cohorts": ["a", "b", "c"],
        "model_version": "adaptive-v1",
        "prediction_protocol": "training-visible",
        "model_fingerprint": "1" * 64,
        "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        "metrics": {"spearman_64um": 0.7},
        "fingerprint": "2" * 64,
        "status": "candidate",
    }
    calls = 0

    def validate(_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return result

    def load(cohort: str, *_args: object, **_kwargs: object):
        return (
            WsiSection(
                cohort=cohort,
                section="001",
                slide="slide.ndpi",
                label="H&E",
                reference=True,
                layers={},
            ),
        )

    monkeypatch.setattr(
        "histopia.protein._manifest.validate_protein_result_index",
        validate,
    )
    monkeypatch.setattr(_wsi_tiles, "_load_cohort_sections", load)

    service = WsiTileService.from_runs(
        {cohort: (tmp_path / cohort, None) for cohort in ("a", "b", "c")},
        protein_model_runs={cohort: {"model": protein} for cohort in ("a", "b", "c")},
    )

    assert calls == 1
    assert service.catalog("c")["default_protein_model_id"] == "model"
    assert (
        service.catalog("c")["protein_models"][0]["prediction_protocol"]
        == "training-visible"
    )
    assert service.catalog("c")["protein_models"][0]["target_label"] == ("YAP curated")


def test_each_protein_model_uses_its_bound_stain_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_runs = {
        model_id: tmp_path / model_id for model_id in ("model-a", "model-b", "model-c")
    }
    for path in model_runs.values():
        path.mkdir()

    def validate(path: Path) -> dict[str, object]:
        model_id = Path(path).name
        digest_character = {"model-a": "a", "model-b": "b", "model-c": "c"}[model_id]
        return {
            "schema_version": 4,
            "model_id": model_id,
            "model_label": model_id,
            "target_id": model_id,
            "architecture": "test",
            "training_cohorts": ["mouse"],
            "model_version": "v1",
            "model_fingerprint": digest_character * 64,
            "measurement_view": "validated-4um-view",
            "metrics": {},
            "fingerprint": digest_character.upper() * 64,
            "status": "candidate",
        }

    loads: list[tuple[Path | None, Path | None]] = []

    def load(
        cohort: str,
        _registration: Path,
        _registered_wsi: Path | None,
        _cells: Path | None,
        stain: Path | None,
        protein: Path | None,
        **_kwargs: object,
    ) -> tuple[WsiSection, ...]:
        loads.append((stain, protein))
        return (
            WsiSection(
                cohort=cohort,
                section="001",
                slide="slide.ndpi",
                label="H&E",
                reference=True,
                layers={},
            ),
        )

    monkeypatch.setattr(
        "histopia.protein._manifest.validate_protein_result_index",
        validate,
    )
    monkeypatch.setattr(_wsi_tiles, "_load_cohort_sections", load)
    global_stain = tmp_path / "global-stain"
    stain_b = tmp_path / "stain-b"
    stain_c = tmp_path / "stain-c"

    service = WsiTileService.from_runs(
        {"mouse": (tmp_path / "registration", None)},
        stain_runs={"mouse": global_stain},
        protein_model_runs={"mouse": model_runs},
        protein_model_stain_runs={"mouse": {"model-b": stain_b, "model-c": stain_c}},
    )

    assert loads == [
        (global_stain, model_runs["model-a"]),
        (stain_b, model_runs["model-b"]),
        (stain_c, model_runs["model-c"]),
    ]
    catalog = service.catalog("mouse")
    assert [row["id"] for row in catalog["protein_models"]] == [
        "model-a",
        "model-b",
        "model-c",
    ]
    assert str(tmp_path) not in json.dumps(catalog)


def test_protein_model_stain_binding_requires_matching_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no matching model: orphan"):
        WsiTileService.from_runs(
            {"mouse": (tmp_path / "registration", None)},
            protein_model_stain_runs={"mouse": {"orphan": tmp_path / "stain"}},
        )


def test_wsi_metadata_selects_one_path_free_protein_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    prediction = tmp_path / "prediction.npz"
    source.write_bytes(b"source")
    prediction.write_bytes(b"prediction")
    raw = WsiLayer(
        name="raw",
        path=source,
        digest="a" * 64,
        levels=(WsiLevel(32, 24, 0),),
        tile_size=16,
        microns_per_pixel=0.5,
    )
    predicted = WsiLayer(
        name="protein_predicted",
        path=prediction,
        digest="b" * 64,
        levels=(WsiLevel(32, 24, 0),),
        tile_size=16,
        microns_per_pixel=0.5,
        map_kind="predicted_od_reference",
        display_max_od=0.8,
        protein_target="yap",
        protein_model_fingerprint="c" * 64,
    )
    base = WsiSection("mouse", "001", "slide.ndpi", "YAP", True, {"raw": raw})
    model = WsiSection(
        "mouse",
        "001",
        "slide.ndpi",
        "YAP",
        True,
        {"raw": raw, "protein_predicted": predicted},
    )
    service = WsiTileService(
        {("mouse", "001"): base},
        protein_sections={("mouse", "yap-extra-trees", "001"): model},
        protein_models={
            ("mouse", "yap-extra-trees"): {
                "id": "yap-extra-trees",
                "label": "YAP · extra trees",
                "target_id": "yap",
                "result_fingerprint": "d" * 64,
            }
        },
        default_protein_models={"mouse": "yap-extra-trees"},
    )

    catalog = service.catalog("mouse")
    metadata = service.metadata(
        "mouse",
        "001",
        protein_model="yap-extra-trees",
    )

    assert catalog["default_protein_model_id"] == "yap-extra-trees"
    assert catalog["protein_models"][0]["target_id"] == "yap"
    assert metadata["protein_model_id"] == "yap-extra-trees"
    assert "protein_predicted" in metadata["layers"]
    assert str(tmp_path) not in json.dumps(catalog)
    assert str(tmp_path) not in json.dumps(metadata)
    with pytest.raises(FileNotFoundError, match="section"):
        service.metadata("mouse", "001", protein_model="missing")


def test_wsi_tile_validates_digest_and_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, digest = _service(tmp_path)
    monkeypatch.setattr(
        _wsi_tiles,
        "_render_layer_tile",
        lambda layer, level, x, y: f"{layer.name}:{level}:{x}:{y}".encode(),
    )

    payload, media_type, etag = service.render_tile(
        "mouse", "001", "registered", digest, 1, 3, 1
    )

    assert payload == b"registered:1:3:1"
    assert media_type == "image/jpeg"
    assert etag == f'"{digest}-1-3-1"'
    with pytest.raises(FileNotFoundError, match="stale"):
        service.render_tile("mouse", "001", "registered", "0" * 64, 1, 0, 0)
    with pytest.raises(FileNotFoundError, match="coordinates"):
        service.render_tile("mouse", "001", "registered", digest, 1, 4, 0)


def test_wsi_tile_capacity_is_bounded(tmp_path: Path) -> None:
    service, digest = _service(tmp_path)
    assert service._capacity.acquire(blocking=False)
    try:
        with pytest.raises(WsiTileCapacityError, match="capacity"):
            service.render_tile("mouse", "001", "registered", digest, 0, 0, 0)
    finally:
        service._capacity.release()


def test_approved_raw_only_registration_exports_native_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"slide")
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {
                        "path": str(source),
                        "is_reference": True,
                        "geometry": {
                            "native_shape": [24, 40],
                            "content_bbox_xywh": [0, 0, 40, 24],
                        },
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        _wsi_tiles,
        "validate_registration_approval",
        lambda run: SimpleNamespace(registration_result_sha256="a" * 64),
    )
    monkeypatch.setattr(
        _wsi_tiles,
        "_image_layer",
        lambda *args, **kwargs: WsiLayer(
            "raw",
            source,
            "b" * 64,
            (WsiLevel(40, 24, 0),),
            512,
            0.5,
            source_shape=(40, 24),
        ),
    )

    sections = _load_cohort_sections("mouse", registration, None)

    assert len(sections) == 1
    assert sections[0].section == "001"
    assert sections[0].slide == source.name
    assert list(sections[0].layers) == ["raw"]


def test_virtual_stain_pyramid_renders_transparent_od_without_requantifying(
    tmp_path: Path,
) -> None:
    map_path, stain_map = _stain_map(tmp_path / "one.npz", raw=0.9, corrected=0.2)
    spec = _StainMapSpec(5, 3, 4.0, (10, 20))
    artifact_digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    raw = _stain_layer(
        "stain_raw",
        map_path,
        artifact_digest,
        spec,
        display_max_od=1.0,
        map_kind="raw_target_od",
    )
    corrected = _stain_layer(
        "stain_corrected",
        map_path,
        artifact_digest,
        spec,
        display_max_od=1.0,
        map_kind="corrected_target_od",
    )
    output_map = _stain_layer(
        "stain_output_map",
        map_path,
        artifact_digest,
        spec,
        display_max_od=1.0,
        map_kind="corrected_target_od",
        render_mode="opaque_tissue",
        selected_source="corrected",
        stain_family="h-dab",
    )
    adaptive = _stain_layer(
        "stain_adaptive",
        map_path,
        artifact_digest,
        spec,
        display_max_od=1.0,
        map_kind="corrected_target_od",
        selected_source="adaptive_corrected",
        stain_family="pas",
        adaptive_floor_od=0.15,
    )
    section = WsiSection(
        cohort="mouse",
        section="001",
        slide=stain_map.slide_id,
        label="DAB",
        reference=False,
        layers={
            "stain_raw": raw,
            "stain_corrected": corrected,
            "stain_output_map": output_map,
            "stain_adaptive": adaptive,
        },
    )
    service = WsiTileService({("mouse", "001"): section})

    raw_bytes, raw_type, _ = service.render_tile(
        "mouse", "001", "stain_raw", raw.digest, len(raw.levels) - 1, 0, 0
    )
    corrected_bytes, corrected_type, _ = service.render_tile(
        "mouse",
        "001",
        "stain_corrected",
        corrected.digest,
        len(corrected.levels) - 1,
        0,
        0,
    )
    raw_rgba = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGBA"))
    corrected_rgba = np.asarray(Image.open(io.BytesIO(corrected_bytes)).convert("RGBA"))
    map_bytes, _, _ = service.render_tile(
        "mouse",
        "001",
        "stain_output_map",
        output_map.digest,
        len(output_map.levels) - 1,
        0,
        0,
    )
    map_rgba = np.asarray(Image.open(io.BytesIO(map_bytes)).convert("RGBA"))
    adaptive_bytes, _, _ = service.render_tile(
        "mouse",
        "001",
        "stain_adaptive",
        adaptive.digest,
        len(adaptive.levels) - 1,
        0,
        0,
    )
    adaptive_rgba = np.asarray(Image.open(io.BytesIO(adaptive_bytes)).convert("RGBA"))

    assert raw_type == corrected_type == "image/png"
    assert raw_rgba.shape == (3, 5, 4)
    assert raw_rgba[0, 0, 3] == corrected_rgba[0, 0, 3] == 0
    assert raw_rgba[1, 2, 3] > corrected_rgba[1, 2, 3] > 0
    assert not np.array_equal(raw_rgba[1, 2, :3], corrected_rgba[1, 2, :3])
    assert map_rgba[0, 0, 3] == 0
    assert map_rgba[1, 2, 3] == 255
    assert 0 < adaptive_rgba[1, 2, 3] < corrected_rgba[1, 2, 3]
    assert adaptive_rgba[1, 2, 0] > adaptive_rgba[1, 2, 1]
    assert adaptive_rgba[1, 2, 2] > adaptive_rgba[1, 2, 1]
    assert len(service._stain_maps) == 1
    assert list(service._stain_maps) == [map_path]

    metadata = service.metadata("mouse", "001")
    layer_metadata = metadata["layers"]["stain_raw"]
    assert layer_metadata["analysis_mpp"] == 4.0
    assert layer_metadata["display_max_od"] == 1.0
    assert layer_metadata["content_origin_native_xy"] == [10, 20]
    assert layer_metadata["coordinate_space"] == "native_content_bbox"
    assert layer_metadata["format"] == "png"
    assert metadata["layers"]["stain_adaptive"]["adaptive_floor_od"] == 0.15
    assert metadata["layers"]["stain_adaptive"]["stain_family"] == "pas"
    assert str(tmp_path) not in str(metadata)


def test_counterstain_conditioned_sidecar_renders_as_immutable_png(
    tmp_path: Path,
) -> None:
    _physical_path, physical = _stain_map(
        tmp_path / "physical.npz", raw=0.8, corrected=0.5
    )
    target = np.where(physical.tissue_mask, 0.12, 0).astype(np.float32)
    adaptive = AdaptiveStainMap(
        slide_id=physical.slide_id,
        target_od=target,
        tissue_mask=physical.tissue_mask,
        analysis_mpp=4.0,
        content_origin_native_xy=physical.content_origin_native_xy,
        source_mpp_xy=physical.source_mpp_xy,
        source_content_fingerprint=str(physical.content_fingerprint),
        method="counterstain-conditioned-v3",
        diagnostics={"method": "counterstain-conditioned-v3", "accepted": True},
    )
    path = adaptive.save(tmp_path / "adaptive.npz")
    artifact_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    layer = _stain_layer(
        "stain_adaptive_v3",
        path,
        artifact_digest,
        _StainMapSpec(5, 3, 4.0, (10, 20), (0.5, 0.5)),
        display_max_od=0.5,
        map_kind="target_od",
        adaptive_method="counterstain-conditioned-v3",
        stain_artifact_kind="adaptive",
    )
    service = WsiTileService(
        {
            ("mouse", "001"): WsiSection(
                cohort="mouse",
                section="001",
                slide=physical.slide_id,
                label="YAP",
                reference=False,
                layers={"stain_adaptive_v3": layer},
            )
        }
    )

    payload, media_type, etag = service.render_tile(
        "mouse",
        "001",
        "stain_adaptive_v3",
        layer.digest,
        len(layer.levels) - 1,
        0,
        0,
    )
    rgba = np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"))
    metadata = service.metadata("mouse", "001")["layers"]["stain_adaptive_v3"]

    assert media_type == "image/png"
    assert etag.startswith(f'"{layer.digest}-')
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 2, 3] > 0
    assert metadata["correction_version"] == "v3"
    assert metadata["adaptive_method"] == "counterstain-conditioned-v3"
    assert metadata["analysis_mpp"] == 4.0
    assert str(tmp_path) not in str(metadata)


def test_registered_observed_target_uses_corrected_calibrated_od_and_support(
    tmp_path: Path,
) -> None:
    map_path, stain_map = _stain_map(tmp_path / "target.npz", raw=0.9, corrected=0.2)
    target_mask = tmp_path / "target-mask.png"
    Image.fromarray(stain_map.tissue_mask.astype(np.uint8) * 255).save(target_mask)
    layer = WsiLayer(
        name="protein_nearest_observed_target",
        path=map_path,
        digest="a" * 64,
        levels=(WsiLevel(5, 3, 0),),
        tile_size=512,
        microns_per_pixel=4.0,
        mask_path=target_mask,
        analysis_mpp=4.0,
        display_max_od=1.0,
        map_kind="corrected_target_od",
        render_mode="registered_nearest_observed",
        protein_target="yap",
        comparison_source_section="003",
        comparison_source_label="Yap",
        comparison_target_section="004",
        target_map_to_source_map=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        calibration_source_knots=(0.0, 1.0),
        calibration_reference_knots=(0.0, 2.0),
        calibration_fingerprint="b" * 64,
        stain_artifact_digest=hashlib.sha256(map_path.read_bytes()).hexdigest(),
    )
    comparison = {
        "schema_version": 1,
        "mode": "unmeasured-transfer",
        "target_id": "yap",
        "predicted_layer": "protein_predicted",
        "current_layer": "stain_output_map",
        "current_label": "SMA",
        "current_scope": "current_marker_od_4um",
        "comparison_layer": "protein_nearest_observed_target",
        "comparison_label": "Nearest observed YAP",
        "ground_truth_available": False,
        "nearest_target_section": "003",
        "nearest_distance_sections": 1,
    }
    service = WsiTileService(
        {
            ("mouse", "004"): WsiSection(
                "mouse",
                "004",
                "four.ndpi",
                "SMA",
                False,
                {layer.name: layer},
                comparison,
            )
        }
    )

    payload, media_type, _etag = service.render_tile(
        "mouse", "004", layer.name, layer.digest, 0, 0, 0
    )
    rgba = np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"))

    assert media_type == "image/png"
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 2, 3] == 255
    expected_values, expected_support = _sample_calibrated_od(
        stain_map.corrected_target_od,
        stain_map.tissue_mask,
        *np.meshgrid(np.arange(5), np.arange(3)),
        source_knots=(0.0, 1.0),
        reference_knots=(0.0, 2.0),
    )
    assert expected_values[1, 2] == pytest.approx(0.4)
    assert expected_support[1, 2]
    assert str(tmp_path) not in json.dumps(service.metadata("mouse", "004"))
    metadata = service.metadata("mouse", "004")
    assert metadata["protein_comparison"] == comparison
    observed = metadata["layers"][layer.name]
    assert observed["source_section"] == "003"
    assert observed["registered_to_section"] == "004"
    assert observed["ground_truth"] is False


def test_protein_tissue_support_is_neutral_transparent_and_path_free(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.asarray([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)).save(
        mask_path
    )
    raw = WsiLayer(
        name="raw",
        path=tmp_path / "slide.ndpi",
        digest="a" * 64,
        levels=(WsiLevel(3, 2, 0),),
        tile_size=512,
        microns_per_pixel=0.5,
    )
    mask = _wsi_tiles._mask_layer(mask_path, raw)
    support = _wsi_tiles._protein_tissue_support_layer(mask)
    service = WsiTileService(
        {
            ("mouse", "001"): WsiSection(
                "mouse", "001", "slide.ndpi", "H&E", True, {support.name: support}
            )
        }
    )

    payload, media_type, _etag = service.render_tile(
        "mouse", "001", support.name, support.digest, 0, 0, 0
    )
    rgba = np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"))

    assert media_type == "image/png"
    assert rgba[0, 0, 3] == 0
    assert tuple(rgba[1, 1]) == (226, 218, 204, 255)
    metadata = service.metadata("mouse", "001")["layers"][support.name]
    assert metadata["measurement_scope"] == "tissue_support_only"
    assert metadata["render_mode"] == "neutral_stain_free_support"
    assert str(tmp_path) not in json.dumps(metadata)


@pytest.mark.parametrize(
    ("name", "expected_scope"),
    (
        ("protein_predicted", "predicted_target_od_per_native_cell"),
        (
            "protein_probability",
            "predicted_expression_probability_per_native_cell",
        ),
        (
            "protein_uncertainty",
            "predicted_target_od_uncertainty_per_native_cell",
        ),
        (
            "protein_measured",
            "observed_target_od_aggregated_per_native_cell_from_4um",
        ),
        ("protein_residual", "absolute_target_od_error_per_native_cell"),
    ),
)
def test_protein_metadata_discloses_each_cell_value_scope(
    tmp_path: Path,
    name: str,
    expected_scope: str,
) -> None:
    layer = WsiLayer(
        name=name,
        path=tmp_path / "prediction.npz",
        digest="a" * 64,
        levels=(WsiLevel(2, 2, 0),),
        tile_size=512,
        microns_per_pixel=0.5,
        map_kind=name,
        display_max_od=1.0,
    )
    service = WsiTileService(
        {
            ("mouse", "001"): WsiSection(
                "mouse", "001", "slide.ndpi", "YAP", False, {name: layer}
            )
        }
    )

    metadata = service.metadata("mouse", "001")["layers"][name]

    assert metadata["measurement_scope"] == expected_scope
    assert str(tmp_path) not in json.dumps(metadata)


def test_cohort_aware_harmonization_is_filtered_for_observed_tiles() -> None:
    from histopia.protein import fit_equal_section_od_harmonization
    from histopia.visualization._wsi_tiles import _validated_od_calibration

    base = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    calibration = fit_equal_section_od_harmonization(
        np.asarray(["4312"] * 128 + ["4630"] * 128),
        np.asarray(["003"] * 256),
        np.r_[base * 2.0, base * 0.5],
        np.ones(256, dtype=bool),
        minimum_cells=128,
    ).as_dict()

    rows, fingerprint = _validated_od_calibration(calibration, cohort="4312")

    assert set(rows) == {"003"}
    assert rows["003"]["scale"] < 1.0
    assert fingerprint == calibration["fingerprint"]
    with pytest.raises(ValueError, match="requires a cohort"):
        _validated_od_calibration(calibration)


def test_harmonized_observed_metadata_discloses_assay_scale(tmp_path: Path) -> None:
    map_path, stain_map = _stain_map(
        tmp_path / "harmonized.npz", raw=0.9, corrected=0.2
    )
    mask_path = tmp_path / "mask.png"
    Image.fromarray(stain_map.tissue_mask.astype(np.uint8) * 255).save(mask_path)
    layer = WsiLayer(
        name="protein_observed_target",
        path=map_path,
        digest="a" * 64,
        levels=(WsiLevel(5, 3, 0),),
        tile_size=512,
        microns_per_pixel=4.0,
        mask_path=mask_path,
        analysis_mpp=4.0,
        display_max_od=1.0,
        map_kind="harmonized_adaptive_corrected_target_od",
        adaptive_floor_od=0.1,
        protein_target="yap",
        comparison_source_section="003",
        comparison_source_label="YAP",
        comparison_target_section="003",
        target_map_to_source_map=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        calibration_source_knots=(0.0, 1.0),
        calibration_reference_knots=(0.0, 2.0),
        calibration_fingerprint="b" * 64,
        stain_artifact_digest=hashlib.sha256(map_path.read_bytes()).hexdigest(),
    )
    service = WsiTileService(
        {
            ("4312", "003"): WsiSection(
                "4312", "003", "three.ndpi", "YAP", False, {layer.name: layer}
            )
        }
    )

    metadata = service.metadata("4312", "003")["layers"][layer.name]
    assert metadata["measurement_scope"] == (
        "tissue_masked_harmonized_adaptive_corrected_target_od_4um"
    )
    assert metadata["adaptive_floor_od"] == pytest.approx(0.1)
    assert metadata["calibration_fingerprint"] == "b" * 64


def test_protein_overview_reuses_one_label_raster_across_panes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_path = tmp_path / "labels.tiff"
    prediction_path = tmp_path / "prediction.npz"
    calls = 0

    def fake_labels(_layer: WsiLayer) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray([[0, 1], [2, 2]], dtype=np.uint32)

    monkeypatch.setattr(_wsi_tiles, "_build_protein_label_overview", fake_labels)
    levels = (WsiLevel(2, 2, 0),)

    def layer(name: str, digest: str, kind: str) -> WsiLayer:
        return WsiLayer(
            name=name,
            path=prediction_path,
            digest=digest,
            levels=levels,
            tile_size=512,
            microns_per_pixel=0.5,
            source_shape=(2, 2),
            label_path=label_path,
            map_kind=kind,
            display_max_od=1.0,
        )

    prediction = SimpleNamespace(
        label_ids=np.asarray([1, 2], dtype=np.uint32),
        supported=np.asarray([True, True]),
        predicted_od_reference=np.asarray([0.2, 0.8], dtype=np.float32),
        uncertainty=np.asarray([0.1, 0.3], dtype=np.float32),
    )
    service = WsiTileService({})
    predicted = service._load_protein_overview(
        layer("protein_predicted", "a" * 64, "predicted_od_reference"),
        prediction,
    )
    uncertain = service._load_protein_overview(
        layer("protein_uncertainty", "b" * 64, "uncertainty"),
        prediction,
    )

    assert calls == 1
    assert np.asarray(predicted.values)[0, 1] == pytest.approx(0.2)
    assert np.asarray(uncertain.values)[0, 1] == pytest.approx(0.1)


def test_stain_map_cache_retains_current_and_nearest_then_evicts_lru(
    tmp_path: Path,
) -> None:
    first_path, first_map = _stain_map(tmp_path / "one.npz", raw=0.8, corrected=0.4)
    second_path, second_map = _stain_map(
        tmp_path / "two.npz",
        raw=0.7,
        corrected=0.3,
        slide="two.ndpi",
    )
    third_path, third_map = _stain_map(
        tmp_path / "three.npz",
        raw=0.6,
        corrected=0.2,
        slide="three.ndpi",
    )
    spec = _StainMapSpec(5, 3, 4.0, (10, 20))

    def layer(path: Path) -> WsiLayer:
        return _stain_layer(
            "stain_raw",
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            spec,
            display_max_od=1.0,
            map_kind="raw_target_od",
        )

    first_layer = layer(first_path)
    second_layer = layer(second_path)
    third_layer = layer(third_path)
    service = WsiTileService(
        {
            ("mouse", "001"): WsiSection(
                "mouse",
                "001",
                first_map.slide_id,
                "one",
                False,
                {"stain_raw": first_layer},
            ),
            ("mouse", "002"): WsiSection(
                "mouse",
                "002",
                second_map.slide_id,
                "two",
                False,
                {"stain_raw": second_layer},
            ),
            ("mouse", "003"): WsiSection(
                "mouse",
                "003",
                third_map.slide_id,
                "three",
                False,
                {"stain_raw": third_layer},
            ),
        }
    )
    service.render_tile(
        "mouse",
        "001",
        "stain_raw",
        first_layer.digest,
        len(first_layer.levels) - 1,
        0,
        0,
    )
    service.render_tile(
        "mouse",
        "002",
        "stain_raw",
        second_layer.digest,
        len(second_layer.levels) - 1,
        0,
        0,
    )

    assert list(service._stain_maps) == [first_path, second_path]
    service.render_tile(
        "mouse",
        "003",
        "stain_raw",
        third_layer.digest,
        len(third_layer.levels) - 1,
        0,
        0,
    )
    assert list(service._stain_maps) == [second_path, third_path]
    with pytest.raises(FileNotFoundError, match="stale"):
        service.render_tile("mouse", "002", "stain_raw", first_layer.digest, 0, 0, 0)


def test_virtual_stain_levels_follow_deep_zoom_geometry() -> None:
    assert [(row.width, row.height) for row in _virtual_dzi_levels(5, 3)] == [
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 3),
    ]


@pytest.mark.parametrize(
    ("correction_accepted", "expected_kind"),
    ((True, "corrected_target_od"), (False, "raw_target_od")),
)
def test_stain_output_layer_uses_only_accepted_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    correction_accepted: bool,
    expected_kind: str,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"slide")
    result = {
        "slides": [
            {
                "path": str(source),
                "is_reference": True,
                "geometry": {
                    "native_shape": [24, 40],
                    "content_bbox_xywh": [0, 0, 40, 24],
                },
            }
        ]
    }
    (registration / "registration_result.json").write_text(json.dumps(result))
    monkeypatch.setattr(
        _wsi_tiles,
        "validate_registration_approval",
        lambda run: SimpleNamespace(registration_result_sha256="a" * 64),
    )
    spec = _StainMapSpec(5, 3, 4.0, (0, 0))
    monkeypatch.setattr(
        _wsi_tiles,
        "_validated_stain_maps",
        lambda *args: (
            {
                source.name: (
                    {
                        "map": "maps/slide.npz",
                        "map_artifact_digest": "b" * 64,
                        "qc": {"correction_accepted": correction_accepted},
                    },
                    spec,
                )
            },
            1.0,
        ),
    )
    monkeypatch.setattr(
        _wsi_tiles,
        "_image_layer",
        lambda *args, **kwargs: WsiLayer(
            "raw",
            source,
            "c" * 64,
            (WsiLevel(40, 24, 0),),
            512,
            0.5,
            source_shape=(40, 24),
        ),
    )

    sections = _load_cohort_sections(
        "mouse",
        registration,
        None,
        stain_run=tmp_path / "stain",
    )

    assert sections[0].layers["stain_raw"].map_kind == "raw_target_od"
    assert sections[0].layers["stain_corrected"].map_kind == expected_kind
    assert sections[0].layers["stain_output"].map_kind == expected_kind
    assert sections[0].layers["stain_output_map"].map_kind == expected_kind
    assert sections[0].layers["stain_output"].selected_source == (
        "corrected" if correction_accepted else "raw"
    )
    assert sections[0].layers["stain_output_map"].render_mode == "opaque_tissue"


def test_adaptive_output_metadata_distinguishes_physical_and_derived_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"slide")
    registration = tmp_path / "registration"
    registration.mkdir()
    result = {
        "slides": [
            {
                "path": str(source),
                "is_reference": True,
                "geometry": {
                    "native_shape": [24, 40],
                    "content_bbox_xywh": [0, 0, 40, 24],
                },
            }
        ]
    }
    (registration / "registration_result.json").write_text(json.dumps(result))
    monkeypatch.setattr(
        _wsi_tiles,
        "validate_registration_approval",
        lambda run: SimpleNamespace(registration_result_sha256="a" * 64),
    )
    map_path, _ = _stain_map(tmp_path / "stain" / "map.npz", raw=0.9, corrected=0.2)
    spec = _StainMapSpec(5, 3, 4.0, (10, 20))
    monkeypatch.setattr(
        _wsi_tiles,
        "_validated_stain_maps",
        lambda *args, **kwargs: (
            {
                source.name: (
                    {
                        "map": map_path.name,
                        "map_artifact_digest": "b" * 64,
                        "family": "h-dab",
                        "qc": {
                            "correction_accepted": True,
                            "adaptive_background": {
                                "accepted": True,
                                "floor_od": 0.3,
                            },
                        },
                    },
                    spec,
                )
            },
            1.0,
        ),
    )
    monkeypatch.setattr(
        _wsi_tiles,
        "_image_layer",
        lambda *args, **kwargs: WsiLayer(
            "raw",
            source,
            "c" * 64,
            (WsiLevel(40, 24, 0),),
            512,
            0.5,
            source_shape=(40, 24),
        ),
    )

    section = _load_cohort_sections(
        "mouse", registration, None, stain_run=tmp_path / "stain"
    )[0]

    assert section.layers["stain_corrected"].selected_source == "corrected"
    assert section.layers["stain_corrected"].adaptive_floor_od is None
    assert section.layers["stain_output"].selected_source == "adaptive_corrected"
    assert section.layers["stain_output_map"].selected_source == "adaptive_corrected"
    assert section.layers["stain_output_map"].adaptive_floor_od == pytest.approx(0.3)


def test_stain_map_validation_binds_approval_origin_resolution_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histopia.visualization._stain_viewer import StainViewerRun

    registration = tmp_path / "registration"
    registration.mkdir()
    approval_path = registration / "registration_approval.json"
    approval_path.write_bytes(b"approved registration")
    stain_root = tmp_path / "stain"
    stain_root.mkdir()
    map_path, stain_map = _stain_map(
        stain_root / "map.npz",
        raw=0.8,
        corrected=0.4,
    )
    artifact_digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    registration_result = {
        "slides": [
            {
                "path": str(tmp_path / stain_map.slide_id),
                "geometry": {
                    "native_shape": [100, 120],
                    "content_bbox_xywh": [10, 20, 40, 24],
                    "mpp_xy": [0.5, 0.5],
                },
            }
        ]
    }
    registration_sha = "d" * 64
    payload = {
        "registration_result_sha256": registration_sha,
        "registration_approval_sha256": hashlib.sha256(
            approval_path.read_bytes()
        ).hexdigest(),
        "measurement": {"analysis_mpp": 4.0},
        "artifacts": {"map.npz": artifact_digest},
        "slides": [
            {
                "id": stain_map.slide_id,
                "order": 1,
                "quantified": True,
                "map": "map.npz",
                "map_fingerprint": stain_map.content_fingerprint,
                "qc": {"correction_accepted": True},
            }
        ],
    }
    viewer_run = StainViewerRun(
        root=stain_root,
        payload=payload,
        slides={stain_map.slide_id: payload["slides"][0]},
        display_max_od=0.9,
        review={},
    )
    monkeypatch.setattr(
        "histopia.visualization._stain_viewer.load_stain_viewer_run",
        lambda *args, **kwargs: viewer_run,
    )

    rows, display_max = _validated_stain_maps(
        registration,
        registration_result,
        registration_sha,
        stain_root,
    )

    row, spec = rows[stain_map.slide_id]
    assert row["map_artifact_digest"] == artifact_digest
    assert spec == _StainMapSpec(5, 3, 4.0, (10, 20), (0.5, 0.5))
    assert display_max == 0.9

    adaptive = AdaptiveStainMap(
        slide_id=stain_map.slide_id,
        target_od=stain_map.corrected_target_od,
        tissue_mask=stain_map.tissue_mask,
        analysis_mpp=stain_map.analysis_mpp,
        content_origin_native_xy=stain_map.content_origin_native_xy,
        source_mpp_xy=stain_map.source_mpp_xy,
        source_content_fingerprint=str(stain_map.content_fingerprint),
        method="counterstain-conditioned-v3",
        diagnostics={"method": "counterstain-conditioned-v3", "accepted": True},
    )
    adaptive_path = adaptive.save(stain_root / "adaptive.npz")
    payload["artifacts"]["adaptive.npz"] = hashlib.sha256(
        adaptive_path.read_bytes()
    ).hexdigest()
    payload["slides"][0].update(
        {
            "adaptive_map": "adaptive.npz",
            "adaptive_map_fingerprint": adaptive.content_fingerprint,
            "qc": {
                "correction_accepted": True,
                "adaptive_background": adaptive.diagnostics,
            },
        }
    )
    rows, _display_max = _validated_stain_maps(
        registration,
        registration_result,
        registration_sha,
        stain_root,
    )
    assert (
        rows[stain_map.slide_id][0]["adaptive_map_artifact_digest"]
        == (payload["artifacts"]["adaptive.npz"])
    )

    registration_result["slides"][0]["geometry"]["content_bbox_xywh"] = [
        11,
        20,
        40,
        24,
    ]
    with pytest.raises(ValueError, match="origin"):
        _validated_stain_maps(
            registration,
            registration_result,
            registration_sha,
            stain_root,
        )


def _stain_map(
    path: Path,
    *,
    raw: float,
    corrected: float,
    slide: str = "one.ndpi",
) -> tuple[Path, StainMap]:
    tissue = np.zeros((3, 5), dtype=bool)
    tissue[1, 1:4] = True
    stain_map = StainMap(
        slide_id=slide,
        raw_target_od=np.where(tissue, raw, 0).astype(np.float32),
        corrected_target_od=np.where(tissue, corrected, 0).astype(np.float32),
        counterstain_od=np.zeros((3, 5), dtype=np.float32),
        reconstruction_residual=np.zeros((3, 5), dtype=np.float32),
        tissue_mask=tissue,
        confidence=tissue.astype(np.float32),
        positive_mask=np.zeros((3, 5), dtype=bool),
        analysis_mpp=4.0,
        content_origin_native_xy=(10, 20),
        source_mpp_xy=(0.5, 0.5),
        provenance={"test": slide},
    )
    stain_map.save(path)
    return path, stain_map
