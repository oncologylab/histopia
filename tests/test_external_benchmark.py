"""External 2D scope must not bypass donor or reconstruction boundaries."""

import copy

import numpy as np
import pytest

from histopia.study._counterstain_benchmark import (
    H_DAB_BASIS,
    counterstain_patch_inputs,
    separate_fixed_hdab,
)
from histopia.study._external_benchmark import (
    validate_external_benchmark,
    validate_external_job,
)
from histopia.study._manifest import fingerprint


def manifest():
    records = [
        dict(
            image_id=str(i),
            acquisition_id=str(i),
            source_id="hpa",
            organ="lung",
            donor_id=str(i),
            modality="IHC",
            detection="brightfield",
            chromogen="DAB",
            role=role,
            image_url=f"https://example.org/{i}.jpg",
            source_methods_url="https://example.org/methods",
        )
        for i, role in enumerate(["train", "validation", "test"])
    ]
    return seal(
        dict(
            schema_version="external-brightfield-benchmark-1",
            purpose="2d-protein-benchmark",
            images=records,
            development_exposed_donors=["hpa:0"],
        )
    )


def seal(value):
    value["fingerprint"] = fingerprint(
        {k: v for k, v in value.items() if k != "fingerprint"}
    )
    return value


def test_scope_and_split_guards():
    data = manifest()
    assert validate_external_benchmark(data)["donors"] == 3
    for key, value in [
        ("detection", "fluorescence"),
        ("modality", "IF"),
        ("chromogen", "unknown"),
        ("donor_id", "0"),
    ]:
        bad = copy.deepcopy(data)
        bad["images"][2][key] = value
        with pytest.raises(ValueError):
            validate_external_benchmark(seal(bad))
    bad = copy.deepcopy(data)
    bad["images"][2]["image_url"] = bad["images"][0]["image_url"]
    with pytest.raises(ValueError):
        validate_external_benchmark(seal(bad))
    bad = copy.deepcopy(data)
    bad["purpose"] = "3d-reconstruction"
    with pytest.raises(ValueError):
        validate_external_benchmark(seal(bad))
    bad = copy.deepcopy(data)
    bad["development_exposed_donors"].append("hpa:2")
    with pytest.raises(ValueError):
        validate_external_benchmark(seal(bad))
    request = dict(
        manifest_fingerprint=data["fingerprint"],
        stage="fit",
        image_ids=["0", "1"],
        fit_ids=["0"],
        validation_ids=["1"],
        method_fingerprint="a" * 64,
        max_wall_seconds=60,
    )
    assert not validate_external_job(request, data)["reconstruction_allowed"]
    request["fit_ids"] = ["1"]
    with pytest.raises(ValueError):
        validate_external_job(request, data)


def test_fixed_separation_dab_does_not_define_morphology():
    yy, xx = np.mgrid[:224, :448]
    h = 0.3 + 0.2 * (np.sin(xx / 7) ** 2 + np.cos(yy / 9) ** 2)
    dab = np.full_like(h, 0.2)

    def rgb(d):
        od = np.stack([h, d], axis=-1) @ H_DAB_BASIS
        return np.clip(np.rint(256 * np.exp(-od) - 1), 0, 255).astype(np.uint8)

    first = counterstain_patch_inputs(rgb(dab))
    second = counterstain_patch_inputs(rgb(dab * 2))
    assert first["xy_px"].tolist() == [[0, 0], [224, 0]]
    np.testing.assert_array_equal(first["xy_px"], second["xy_px"])
    np.testing.assert_allclose(
        # Quantiles retain the bounded quantization error of 8-bit RGB.
        first["h_statistics"],
        second["h_statistics"],
        atol=0.006,
    )
    assert (
        np.max(np.abs(first["images"].astype(int) - second["images"].astype(int))) <= 2
    )
    assert np.all(second["target_dab_od"] > first["target_dab_od"] * 1.9)
    empty = counterstain_patch_inputs(np.full((240, 240, 3), 255, dtype=np.uint8))
    assert empty["images"].shape == (0, 224, 224, 3)
    with pytest.raises(ValueError):
        separate_fixed_hdab(np.zeros((4, 4)))


def test_evaluation_cache_binds_selected_model():
    data = manifest()
    request = dict(
        manifest_fingerprint=data["fingerprint"],
        stage="evaluate",
        image_ids=["2"],
        test_ids=["2"],
        method_fingerprint="b" * 64,
        model_selection_sha256="a" * 64,
        max_wall_seconds=60,
    )
    first = validate_external_job(request, data)
    request["model_selection_sha256"] = "c" * 64
    assert first["artifact_key"] != validate_external_job(request, data)["artifact_key"]


def test_external_export_is_separate_and_test_images_stay_sealed(tmp_path):
    import json

    from histopia.study._manifest import file_sha256
    from histopia.visualization._external_validation import (
        build_external_validation,
        external_validation_tab,
    )
    from histopia.visualization._results_catalog import build_results_catalog

    data = manifest()
    data["sources"] = [
        dict(
            id="hpa", label="HPA", external=True, species="human", description="2D IHC"
        )
    ]
    seal(data)
    asset = tmp_path / "image.svg"
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"/>')
    row = dict(
        id="r",
        image_id="0",
        benchmark_fingerprint=data["fingerprint"],
        source_id="hpa",
        organ="lung",
        subject_id="0",
        role="train",
        title="IHC",
        stage="Features",
        status="ready",
        summary="Counterstain input",
        evidence_kind="image",
        fingerprint="a" * 64,
        media=[dict(path=str(asset), sha256=file_sha256(asset), label="IHC")],
    )
    out = tmp_path / "review"
    build_external_validation(data, [row], out, updated_at="2026-09-17")
    exported = json.loads((out / "catalog.json").read_text())
    assert "eligibility_policy" not in exported
    assert not exported["benchmark_policy"]["reconstruction_allowed"]
    assert exported["datasets"][0]["image_id"] == "0"
    assert external_validation_tab(exported)["image_contexts"] == {
        "r": dict(source_id="hpa", organ="lung", subject_id="0")
    }
    assert str(tmp_path) not in (out / "catalog.json").read_text()
    with pytest.raises(ValueError, match="eligibility registry"):
        build_results_catalog(
            data["sources"], [row], tmp_path / "serial", updated_at="now"
        )
    row.update(image_id="2", subject_id="2", role="test")
    with pytest.raises(ValueError, match="sealed"):
        build_external_validation(data, [row], out, updated_at="now")
    row["evaluation_complete"] = True
    with pytest.raises(ValueError, match="sealed"):
        build_external_validation(data, [row], out, updated_at="now")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            dict(
                stage="frozen_before_test",
                authorization=dict(
                    stage="fit",
                    manifest_fingerprint=data["fingerprint"],
                    image_ids=["0", "1"],
                    roles=dict(fit_ids=["0"]),
                ),
            )
        )
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            dict(
                status="evaluated",
                model_selection_sha256=file_sha256(selection),
                authorization=dict(
                    stage="evaluate",
                    manifest_fingerprint=data["fingerprint"],
                    image_ids=["2"],
                    model_selection_sha256=file_sha256(selection),
                ),
            )
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(
                status="pass",
                evaluation_sha256=file_sha256(evaluation),
                selection_sha256=file_sha256(selection),
            )
        )
    )
    row["evaluation_evidence"] = dict(
        path=str(evaluation),
        sha256=file_sha256(evaluation),
        selection_path=str(selection),
        selection_sha256=file_sha256(selection),
        audit_path=str(audit),
        audit_sha256=file_sha256(audit),
    )
    build_external_validation(data, [row], out, updated_at="now")
    audit.write_text("{}")
    with pytest.raises(ValueError, match="audit changed"):
        build_external_validation(data, [row], out, updated_at="now")
    row.update(image_id="0", subject_id="0", role="train", reconstruction_id="fake")
    with pytest.raises(ValueError, match="reconstruction"):
        build_external_validation(data, [row], out, updated_at="now")
    build_external_validation(data, [], out, updated_at="now")
    empty = json.loads((out / "catalog.json").read_text())
    assert not empty["sources"]
    assert external_validation_tab(empty)["image_contexts"] == {}
    assert not list((out / "assets").iterdir())
    assert "No external benchmark images" in (out / "index.html").read_text()


def test_donor_weighted_ridge_freezes_training_normalization():
    from histopia.study._external_prediction import (
        donor_metrics,
        donor_weights,
        fit_ridge_path,
        predict_ridge,
    )

    rng = np.random.default_rng(74)
    x = rng.normal(size=(80, 4))
    y = np.exp(x[:, 0] * 0.2 + 1) - 1
    donors = np.array(["a"] * 60 + ["b"] * 20)
    weights = donor_weights(donors)
    assert weights[:60].sum() == pytest.approx(weights[60:].sum())
    model = fit_ridge_path(x, y, donors, [0.01])[0]
    expected = np.average(x, axis=0, weights=weights)
    np.testing.assert_allclose(model["center"], expected)
    # A test outlier cannot affect predictions for other held-out rows.
    prediction = predict_ridge(model, x[:3])
    other = predict_ridge(model, np.concatenate([x[:3], np.full((1, 4), 15)]))
    np.testing.assert_allclose(prediction, other[:3])
    assert donor_metrics(y, predict_ridge(model, x), donors)["donor_mean_mae"] < 0.001
    metrics = donor_metrics(
        np.array([0.0, 0.0, 10.0]), np.zeros(3), np.array(["a", "a", "b"])
    )
    assert metrics["donor_mean_mae"] == 5.0


def test_external_ledger_rechecks_revocation_and_unique_claims(tmp_path):
    import json

    from histopia.study import ExternalBenchmarkLedger

    data = manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    queue = ExternalBenchmarkLedger(tmp_path / "ledger", manifest_path=path)
    request = dict(
        manifest_fingerprint=data["fingerprint"],
        stage="stain_features",
        image_ids=["0"],
        method_fingerprint="a" * 64,
        max_wall_seconds=60,
    )
    queue.add("job", request, {"patch_size": 224})
    with pytest.raises(FileExistsError):
        queue.add("duplicate", request, {"patch_size": 224})
    owner = queue.claim("job")
    with pytest.raises(ValueError, match="already claimed"):
        queue.claim("job")
    data["protocol"] = {"changed": True}
    seal(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stale"):
        queue.finish("job", owner, status="complete", evidence={"done": True})
    queue.finish("job", owner, status="blocked", evidence={"reason": "revoked"})
    assert queue.rows()[0]["status"] == "blocked"


def test_external_ledger_rejects_changed_execution(tmp_path):
    import json

    from histopia.study import ExternalBenchmarkLedger

    data = manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    queue = ExternalBenchmarkLedger(tmp_path / "ledger", manifest_path=path)
    request = dict(
        manifest_fingerprint=data["fingerprint"],
        stage="stain_features",
        image_ids=["0"],
        method_fingerprint="a" * 64,
        max_wall_seconds=60,
    )
    queue.add("job", request, {"patch_size": 224})
    jobpath = next((tmp_path / "ledger/jobs").glob("*.json"))
    row = json.loads(jobpath.read_text())
    row["payload"]["execution"]["patch_size"] = 448
    jobpath.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="changed"):
        queue.claim("job")
    assert queue.rows()[0]["status"] == "blocked"
