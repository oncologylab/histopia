"""Command-line workflow for prepared protein-expression cell tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia._signals import graceful_sigterm


def main(argv: list[str] | None = None) -> int:
    with graceful_sigterm():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit and validate stain-invariant per-cell protein models."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Validate config and cell table.")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("table", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Validate and seal an assembled neutral cell table."
    )
    prepare.add_argument("config", type=Path)
    prepare.add_argument("table", type=Path)
    benchmark = commands.add_parser(
        "benchmark", help="Fit one spatial holdout and report validation metrics."
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("table", type=Path)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument(
        "--group",
        choices=("spatial", "section", "mouse"),
        default="spatial",
        help="Held-out unit; mouse is required for production promotion.",
    )
    fit = commands.add_parser("fit", help="Fit and seal a target model.")
    fit.add_argument("config", type=Path)
    fit.add_argument("table", type=Path)
    predict = commands.add_parser("predict", help="Predict every supported table row.")
    predict.add_argument("model", type=Path)
    predict.add_argument("table", type=Path)
    predict.add_argument("output", type=Path)
    validate = commands.add_parser(
        "validate", help="Validate portable protein artifacts."
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument(
        "--kind", choices=("table", "model", "predictions"), required=True
    )
    approve = commands.add_parser(
        "approve", help="Apply minimum quality gates to benchmark metrics."
    )
    approve.add_argument("run", type=Path)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--binary", action="store_true")
    reassess = commands.add_parser(
        "reassess",
        help="Reapply the current promotion policy to a sealed study result.",
    )
    reassess.add_argument("run", type=Path)
    cohort = commands.add_parser(
        "cohort-qc", help="Aggregate target model status without exposing paths."
    )
    cohort.add_argument("output", type=Path)
    cohort.add_argument("result", type=Path, nargs="+")
    real = commands.add_parser(
        "run-real", help="Assemble and fit a repeated-stain real-data demo."
    )
    real.add_argument("config", type=Path)
    real.add_argument("--registration-run", type=Path, required=True)
    real.add_argument("--cell-run", type=Path, required=True)
    real.add_argument("--stain-run", type=Path, required=True)
    real.add_argument("--semantic-run", type=Path, required=True)
    real.add_argument("--model-cache-dir", type=Path, required=True)
    real.add_argument("--mouse-id", required=True)
    real.add_argument("--device", default="auto")
    real.add_argument("--batch-size", type=int, default=64)
    real.add_argument("--max-training-cells-per-section", type=int, default=12_000)
    real.add_argument(
        "--feature-section",
        action="append",
        default=None,
        help="Extract only this section's shared cell features; repeat as needed.",
    )
    real.add_argument(
        "--features-only",
        action="store_true",
        help="Stop after reusable target-free cell feature extraction.",
    )
    real.add_argument(
        "--source-cache-dir",
        type=Path,
        help="Optional local mirror of source slides, validated by identity.",
    )
    study_table = commands.add_parser(
        "study-table",
        help=(
            "Pool and harmonize strict adaptive target outcomes across a local "
            "study registry."
        ),
    )
    study_table.add_argument("config", type=Path)
    study_table.add_argument("study", type=Path)
    study_table.add_argument("output", type=Path)
    study_table.add_argument("--geometry-cache", type=Path, required=True)
    study_table.add_argument(
        "--max-training-cells-per-section",
        type=int,
        default=8_000,
    )
    study_table.add_argument(
        "--harmonization-reference-cohort",
        action="append",
        default=[],
        help=(
            "Training mouse used to define the repeated-assay reference scale; "
            "repeat to keep external holdouts out of that reference."
        ),
    )
    study_table.add_argument(
        "--feature-schema",
        choices=(
            "native-hdab-neutral-spatial-uni2h-v2",
            "native-hdab-neutral-cell-multiscale-v3",
        ),
        help="Override the config's target-free cell feature schema.",
    )
    extend_table = commands.add_parser(
        "extend-table",
        help=(
            "Append new outcome cohorts to an immutable training table without "
            "replacing or resampling its existing rows."
        ),
    )
    extend_table.add_argument("base", type=Path)
    extend_table.add_argument("addition", type=Path)
    extend_table.add_argument("output", type=Path)
    extend_table.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="New cohort to append; repeat as needed (default: all new cohorts).",
    )
    study_features = commands.add_parser(
        "study-features",
        help="Precompute reusable multiscale features for disjoint sections.",
    )
    study_features.add_argument("config", type=Path)
    study_features.add_argument("study", type=Path)
    study_features.add_argument("--geometry-cache", type=Path, required=True)
    study_features.add_argument("--cohort", action="append", default=[])
    study_features.add_argument("--section", action="append", default=[])
    study_features.add_argument(
        "--feature-schema",
        choices=("native-hdab-neutral-cell-multiscale-v3",),
        default="native-hdab-neutral-cell-multiscale-v3",
    )
    study_fit = commands.add_parser(
        "study-fit",
        help="Fit and seal one multi-mouse model architecture or cohort ablation.",
    )
    study_fit.add_argument("config", type=Path)
    study_fit.add_argument("study", type=Path)
    study_fit.add_argument("table", type=Path)
    study_fit.add_argument("output", type=Path)
    study_fit.add_argument("--geometry-cache", type=Path, required=True)
    study_fit.add_argument(
        "--architecture",
        choices=(
            "cross_attention",
            "hurdle_mlp",
            "extra_trees",
            "multi_tower",
            "dual_bank_attention",
            "graph_transformer",
            "shared_multitask",
        ),
        required=True,
    )
    study_fit.add_argument("--training-cohort", action="append", default=[])
    study_fit.add_argument(
        "--prediction-cohort",
        action="append",
        default=[],
        help=(
            "Write whole-slide predictions only for this cohort while retaining "
            "the complete sealed training/evaluation scope; repeat as needed."
        ),
    )
    study_fit.add_argument(
        "--prediction-protocol",
        choices=("leave-one-mouse-out", "training-visible"),
        default="leave-one-mouse-out",
        help=(
            "Use held-out mouse predictions, or explicitly diagnostic "
            "training-visible predictions for training cohorts."
        ),
    )
    study_fit.add_argument("--device", default="auto")
    study_fit.add_argument(
        "--epochs",
        type=int,
        help="Override the architecture's sealed training-epoch default.",
    )
    study_fit.add_argument(
        "--reuse-models-from",
        type=Path,
        help=(
            "Reuse an exact sealed fit for a different display protocol; "
            "no model optimization is repeated."
        ),
    )
    study_fit.add_argument(
        "--reuse-refinement-evidence",
        type=Path,
        help=(
            "Apply a frozen, confirmation-accepted morphology-transfer "
            "refinement to an exact sealed relational fit without retraining."
        ),
    )
    study_fit.add_argument(
        "--auxiliary-table",
        action="append",
        default=[],
        type=Path,
        help=(
            "Sealed different-antibody table used only to regularize a "
            "shared_multitask trunk; repeat for additional targets."
        ),
    )
    study_fit.add_argument(
        "--feature-schema",
        choices=(
            "native-hdab-neutral-spatial-uni2h-v2",
            "native-hdab-neutral-cell-multiscale-v3",
        ),
        help="Override the config's target-free cell feature schema.",
    )
    study_ensemble = commands.add_parser(
        "study-ensemble",
        help="Seal an equal-weight ensemble of schema-v4 study models.",
    )
    study_ensemble.add_argument("output", type=Path)
    study_ensemble.add_argument("parent", type=Path, nargs="+")
    sweep_create = commands.add_parser(
        "sweep-create",
        help="Create an exact, resumable multi-target candidate sweep.",
    )
    sweep_create.add_argument("specification", type=Path)
    sweep_create.add_argument("output", type=Path)
    sweep_create.add_argument("--hours", type=float, default=8.0)
    sweep_create.add_argument("--epochs", type=int, default=40)
    sweep_create.add_argument("--max-training-cells", type=int, default=24_000)
    sweep_create.add_argument("--max-evaluation-cells", type=int, default=12_000)
    sweep_create.add_argument(
        "--architecture",
        action="append",
        default=[],
        help="Candidate to include; repeat to override the complete default set.",
    )
    sweep_worker = commands.add_parser(
        "sweep-worker",
        help="Claim unique sweep tasks from a shared manifest.",
    )
    sweep_worker.add_argument("manifest", type=Path)
    sweep_worker.add_argument("--worker-id")
    sweep_worker.add_argument("--maximum-tasks", type=int)
    sweep_worker.add_argument("--architecture", action="append", default=[])
    sweep_status = commands.add_parser(
        "sweep-status", help="Report path-free sweep progress."
    )
    sweep_status.add_argument("manifest", type=Path)
    sweep_select = commands.add_parser(
        "sweep-select",
        help="Select separate transfer and measured-section winners.",
    )
    sweep_select.add_argument("manifest", type=Path, nargs="+")
    sweep_select.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import numpy as np

    if args.command == "extend-table":
        from histopia.protein._result import (
            CellExpressionTable,
            extend_cell_expression_table,
        )

        base = CellExpressionTable.load(args.base)
        addition = CellExpressionTable.load(args.addition)
        output = extend_cell_expression_table(
            base,
            addition,
            addition_cohorts=tuple(args.cohort) if args.cohort else None,
        ).save(args.output)
        print(output)
        return 0

    if args.command == "sweep-create":
        from histopia.protein._sweep import (
            SWEEP_ARCHITECTURES,
            create_protein_sweep,
        )

        output = create_protein_sweep(
            args.specification,
            args.output,
            architectures=(
                tuple(args.architecture) if args.architecture else SWEEP_ARCHITECTURES
            ),
            hours=args.hours,
            epochs=args.epochs,
            maximum_training_cells=args.max_training_cells,
            maximum_evaluation_cells=args.max_evaluation_cells,
        )
        print(output)
        return 0
    if args.command == "sweep-worker":
        from histopia.protein._sweep import run_protein_sweep_worker

        status = run_protein_sweep_worker(
            args.manifest,
            worker_id=args.worker_id,
            architectures=(tuple(args.architecture) if args.architecture else None),
            maximum_tasks=args.maximum_tasks,
        )
        print(json.dumps(status, sort_keys=True))
        return 0 if not status["failed"] else 2
    if args.command == "sweep-status":
        from histopia.protein._sweep import protein_sweep_status

        print(json.dumps(protein_sweep_status(args.manifest), sort_keys=True))
        return 0
    if args.command == "sweep-select":
        from histopia.protein._sweep import select_protein_sweep_winners

        output = select_protein_sweep_winners(tuple(args.manifest), args.output)
        print(output)
        return 0

    if args.command == "study-ensemble":
        from histopia.protein._study_workflow import (
            ensemble_protein_study_results,
        )

        output = ensemble_protein_study_results(
            tuple(args.parent),
            args.output,
        )
        print(output)
        return 0

    if args.command in {"study-features", "study-table", "study-fit"}:
        from histopia.protein._config import load_protein_config
        from histopia.protein._study_workflow import (
            fit_real_protein_study_variant,
            load_real_protein_study_manifest,
            prepare_real_protein_study_features,
            prepare_real_protein_study_table,
        )

        config = load_protein_config(args.config)
        if args.feature_schema is not None:
            config.feature_schema_id = args.feature_schema
        study = load_real_protein_study_manifest(args.study)
        if args.command == "study-features":
            output = prepare_real_protein_study_features(
                config,
                study,
                geometry_cache=args.geometry_cache,
                cohort_ids=tuple(args.cohort) if args.cohort else None,
                sections=tuple(args.section) if args.section else None,
            )
        elif args.command == "study-table":
            output = prepare_real_protein_study_table(
                config,
                study,
                args.output,
                geometry_cache=args.geometry_cache,
                max_training_cells_per_section=(args.max_training_cells_per_section),
                harmonization_reference_cohorts=(
                    tuple(args.harmonization_reference_cohort)
                    if args.harmonization_reference_cohort
                    else None
                ),
            )
        else:
            output = fit_real_protein_study_variant(
                config,
                args.table,
                study,
                args.output,
                geometry_cache=args.geometry_cache,
                architecture=args.architecture,
                training_cohorts=(
                    tuple(args.training_cohort) if args.training_cohort else None
                ),
                prediction_cohorts=(
                    tuple(args.prediction_cohort) if args.prediction_cohort else None
                ),
                prediction_protocol=args.prediction_protocol,
                device=args.device,
                epochs=args.epochs,
                reuse_models_from=args.reuse_models_from,
                reuse_refinement_evidence=args.reuse_refinement_evidence,
                auxiliary_table_paths=tuple(args.auxiliary_table),
            )
        rendered = (
            json.dumps(output, sort_keys=True) if isinstance(output, dict) else output
        )
        print(rendered)
        return 0

    if args.command == "run-real":
        from histopia.protein._config import load_protein_config
        from histopia.protein._real_workflow import run_real_protein_workflow

        config = load_protein_config(args.config)
        output = run_real_protein_workflow(
            config,
            registration_run=args.registration_run,
            cell_run=args.cell_run,
            stain_run=args.stain_run,
            semantic_run=args.semantic_run,
            model_cache_dir=args.model_cache_dir,
            mouse_id=args.mouse_id,
            device=args.device,
            batch_size=args.batch_size,
            max_training_cells_per_section=args.max_training_cells_per_section,
            feature_sections=(
                tuple(args.feature_section) if args.feature_section else None
            ),
            features_only=args.features_only,
            source_cache_dir=args.source_cache_dir,
        )
        print(output)
        return 0

    if args.command == "validate":
        _load_artifact(args.artifact, args.kind)
        print(args.artifact)
        return 0
    if args.command == "predict":
        output = _predict(args.model, args.table, args.output)
        print(output)
        return 0
    if args.command == "reassess":
        from histopia.protein._manifest import reassess_protein_result

        output = reassess_protein_result(args.run)
        print(output)
        return 0
    if args.command == "approve":
        from histopia.protein._manifest import (
            approve_protein_result,
            validate_protein_result,
        )

        result = validate_protein_result(args.run)
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("protein result has no validation metrics")
        if result.get("schema_version") == 4:
            from histopia.protein._selection import evaluate_protein_promotion

            decision = evaluate_protein_promotion(metrics)
            accepted, reasons = decision.accepted, decision.reasons
        else:
            from histopia.protein._model import promotion_decision

            accepted, reasons = promotion_decision(
                metrics,
                binary=args.binary or result.get("binary_enabled") is True,
            )
        output = approve_protein_result(
            args.run,
            reviewer=args.reviewer,
            accepted=accepted,
            reasons=reasons,
        )
        print(output)
        return 0 if accepted else 2
    if args.command == "cohort-qc":
        rows = []
        for result in args.result:
            payload = json.loads(result.read_text())
            rows.append(
                {
                    "target_id": payload.get("target_id"),
                    "model_fingerprint": payload.get("model_fingerprint"),
                    "accepted": bool(payload.get("accepted")),
                    "reasons": payload.get("reasons", []),
                }
            )
        write_json_atomic(args.output, {"schema_version": 1, "targets": rows})
        print(args.output)
        return 0

    from histopia.protein._config import load_protein_config
    from histopia.protein._result import CellExpressionTable

    config = load_protein_config(args.config)
    table = CellExpressionTable.load(args.table)
    if table.target_id != config.target.target_id:
        raise ValueError("cell table target differs from protein config")
    if args.command in {"preflight", "prepare"}:
        payload = {
            "schema_version": 1,
            "target_id": table.target_id,
            "cells": len(table.label_ids),
            "measured_cells": int(np.isfinite(table.measured_od).sum()),
            "binary_cells": int((table.binary_label >= 0).sum()),
            "table_fingerprint": table.fingerprint,
        }
        output = config.output_dir / "protein_preflight.json"
        write_json_atomic(output, payload)
        if args.command == "prepare":
            output = table.save(config.output_dir / "cell_expression_table.npz")
        print(output)
        return 0
    if args.command == "benchmark":
        metrics = _benchmark(config, table, group=args.group)
        output = args.output or config.output_dir / f"benchmark-{args.group}.json"
        write_json_atomic(output, metrics)
    else:
        from histopia.protein._manifest import write_protein_result

        evaluation_group = (
            "mouse"
            if len(np.unique(table.mouse_ids)) >= 2
            else "section"
            if len(np.unique(table.section_ids)) >= 2
            else "spatial"
        )
        metrics = _benchmark(config, table, group=evaluation_group)
        measured = np.flatnonzero(
            np.isfinite(table.measured_od) & np.asarray(table.semantic_support)
        )
        model = _fit_model(config, table, measured)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = model.save(config.output_dir / "protein_model.npz")
        prediction_rows = _predict_sections(
            model,
            table,
            config.output_dir / "predictions",
        )
        output = write_protein_result(
            config.output_dir,
            {
                "schema_version": 1,
                "target_id": model.target_id,
                "assay_domain": model.assay_domain,
                "model": model_path.name,
                "slides": prediction_rows,
                "model_fingerprint": model.fingerprint,
                "table_fingerprint": table.fingerprint,
                "registration_result_sha256": table.provenance.get(
                    "registration_result_sha256"
                ),
                "cell_result_fingerprint": table.provenance.get(
                    "cell_result_fingerprint"
                ),
                "metrics": metrics,
                "binary_enabled": config.target.binary_enabled,
            },
        )
    print(output)
    return 0


def _fit_model(config, table, train_indices):
    import numpy as np

    from histopia.protein._model import fit_protein_model

    candidates = np.asarray(train_indices, dtype=np.int64)
    train_indices = candidates[
        np.isfinite(table.measured_od[candidates])
        & np.asarray(table.semantic_support[candidates], dtype=bool)
    ]
    return fit_protein_model(
        table.features,
        table.measured_od,
        table.binary_label,
        train_indices,
        target_id=config.target.target_id,
        assay_domain=config.target.assay_domain,
        pca_components=config.pca_components,
        hidden_units=config.hidden_units,
        hidden_layers=config.hidden_layers,
        ensemble_seeds=config.ensemble_seeds,
        seed=config.seed,
        provenance={
            "table_fingerprint": table.fingerprint,
            "feature_view": "stain_neutral_v1",
            "spatial_block_um": config.spatial_block_um,
            "exclusion_buffer_um": config.exclusion_buffer_um,
        },
    )


def _benchmark(config, table, *, group: str) -> dict[str, object]:
    import numpy as np

    from histopia.protein._splits import build_spatial_split, leave_one_group_out

    if group == "spatial":
        split = build_spatial_split(
            table.reference_um_xy,
            block_um=config.spatial_block_um,
            buffer_um=config.exclusion_buffer_um,
            seed=config.seed,
        )
        folds = ((split.train, split.test, "spatial-test"),)
        evaluation = "spatial_block_holdout"
    else:
        values = table.mouse_ids if group == "mouse" else table.section_ids
        folds = tuple(
            (train, test, str(np.asarray(values)[test[0]]))
            for train, test in leave_one_group_out(values)
        )
        evaluation = f"leave_one_{group}_out"
    candidate_results: dict[str, dict[str, object]] = {}
    for candidate in config.model_candidates:
        fold_metrics = []
        unavailable = None
        for train, test, held_out in folds:
            if not len(train) or not len(test):
                continue
            try:
                fold = _evaluate_candidate(candidate, config, table, train, test)
            except RuntimeError as error:
                unavailable = str(error)
                break
            fold_metrics.append({"held_out": held_out, **fold})
        if unavailable is not None:
            candidate_results[candidate] = {"available": False, "reason": unavailable}
            continue
        candidate_results[candidate] = _aggregate_candidate(fold_metrics)
    available = {
        name: value
        for name, value in candidate_results.items()
        if value.get("available") is True
    }
    if not available:
        raise ValueError(f"{group} benchmark produced no usable folds")
    selected = max(
        available,
        key=lambda name: (
            float(available[name].get("spearman_64um") or -2),
            float(available[name].get("spearman") or -2),
        ),
    )
    aggregate: dict[str, object] = {
        **available[selected],
        "schema_version": 2,
        "evaluation": evaluation,
        "selected_candidate": selected,
        "selection_metric": "spearman_64um_then_cell_spearman",
        "candidates": candidate_results,
    }
    invariance = table.provenance.get("invariance_metrics", {})
    if isinstance(invariance, dict):
        aggregate["invariance"] = invariance
    return aggregate


def _aggregate_candidate(fold_metrics) -> dict[str, object]:
    import numpy as np

    metric_names = (
        "mae",
        "spearman",
        "spearman_64um",
        "mae_improvement",
        "mean_bias_fraction",
        "median_bias_fraction",
        "auroc",
        "average_precision",
        "normalized_ap_gain",
        "ece",
    )
    aggregate: dict[str, object] = {
        "available": True,
        "fold_count": len(fold_metrics),
        "folds": fold_metrics,
        "folds_beating_semantic_fraction": float(
            np.mean([bool(row["beats_semantic_baseline"]) for row in fold_metrics])
        ),
    }
    for name in metric_names:
        values = [row[name] for row in fold_metrics if row.get(name) is not None]
        aggregate[name] = float(np.median(values)) if values else None
    return aggregate


def _evaluate_candidate(candidate, config, table, train, test):
    import numpy as np

    if candidate == "hurdle_mlp":
        return _evaluate_fold(_fit_model(config, table, train), table, train, test)
    if candidate in {
        "graphsage",
        "gatv2",
        "cross_attention",
        "token_decoder",
        "multi_tower",
        "dual_bank_attention",
        "graph_transformer",
    }:
        from histopia.protein._deep import fit_deep_candidate

        predicted = fit_deep_candidate(
            candidate,
            table.features,
            table.measured_od,
            train,
            test,
            reference_um_xyz=np.column_stack(
                (
                    table.reference_um_xy,
                    np.asarray(table.section_ids, dtype=np.float64)
                    * config.section_spacing_um,
                )
            ),
            section_groups=np.asarray(table.section_ids),
            seed=config.seed,
        )
        if (
            candidate == "cross_attention"
            and table.provenance.get("feature_view")
            == "native-hdab-neutral-spatial-uni2h-v2"
        ):
            semantic, _probability = _semantic_baseline(table, train, test)
            predicted = _region_centered_blend(
                predicted,
                semantic,
                np.asarray(table.semantic_region)[test],
                weight=config.semantic_blend_weight,
            )
        return _evaluate_candidate_arrays(table, train, test, predicted, None)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    measured = np.asarray(table.measured_od, dtype=float)
    valid = train[np.isfinite(measured[train])]
    if candidate == "morphospatial_knn":
        estimator = KNeighborsRegressor(
            n_neighbors=min(16, len(valid)), weights="distance", p=2
        )
    elif candidate == "linear":
        estimator = Ridge(alpha=1.0)
    elif candidate == "tree":
        estimator = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=max(2, len(valid) // 1000),
            max_features="sqrt",
            random_state=config.seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unsupported protein candidate: {candidate}")
    estimator.fit(table.features[valid], measured[valid])
    predicted = np.maximum(estimator.predict(table.features[test]), 0)
    return _evaluate_candidate_arrays(table, train, test, predicted, None)


def _evaluate_candidate_arrays(table, train, test, predicted_od, probability):
    import numpy as np

    from histopia.protein._model import evaluate_predictions

    predicted_values = np.asarray(predicted_od, dtype=np.float64)
    if predicted_values.shape != (len(test),) or not np.all(
        np.isfinite(predicted_values)
    ):
        raise ValueError("protein candidate produced invalid OD predictions")

    metrics = evaluate_predictions(
        table.measured_od[test],
        predicted_values,
        binary_label=table.binary_label[test],
        probability=probability,
    )
    return _compare_with_baseline(metrics, table, train, test, predicted_values)


def _evaluate_fold(model, table, train, test) -> dict[str, object]:
    from histopia.protein._model import evaluate_predictions

    probability, _relative, predicted_od, _uncertainty = model.predict(
        table.features[test]
    )
    metrics = evaluate_predictions(
        table.measured_od[test],
        predicted_od,
        binary_label=table.binary_label[test],
        probability=probability,
    )
    return _compare_with_baseline(metrics, table, train, test, predicted_od)


def _compare_with_baseline(metrics, table, train, test, predicted_od):
    import numpy as np

    from histopia.protein._model import evaluate_predictions

    baseline_od, baseline_probability = _semantic_baseline(table, train, test)
    baseline = evaluate_predictions(
        table.measured_od[test],
        baseline_od,
        binary_label=table.binary_label[test],
        probability=baseline_probability,
    )
    mae = metrics.get("mae")
    baseline_mae = baseline.get("mae")
    improvement = (
        (baseline_mae - mae) / max(baseline_mae, 1e-8)
        if isinstance(mae, float) and isinstance(baseline_mae, float)
        else None
    )
    aggregate = _aggregate_at_scale(
        table.reference_um_xy[test], table.measured_od[test], predicted_od, 64.0
    )
    combined_auc = metrics.get("auroc")
    baseline_auc = baseline.get("auroc")
    beats_binary = (
        not isinstance(combined_auc, float)
        or not isinstance(baseline_auc, float)
        or combined_auc >= baseline_auc
    )
    truth = np.asarray(table.measured_od[test], dtype=np.float64)
    predicted_values = np.asarray(predicted_od, dtype=np.float64)
    valid_bias = np.isfinite(truth) & np.isfinite(predicted_values)
    if np.any(valid_bias):
        truth_valid = truth[valid_bias]
        prediction_valid = predicted_values[valid_bias]
        mean_bias_fraction = float(
            (prediction_valid.mean() - truth_valid.mean())
            / max(abs(truth_valid.mean()), 1e-8)
        )
        median_bias_fraction = float(
            (np.median(prediction_valid) - np.median(truth_valid))
            / max(abs(np.median(truth_valid)), 1e-8)
        )
    else:
        mean_bias_fraction = None
        median_bias_fraction = None
    return {
        **metrics,
        "test_cells": int(len(test)),
        "semantic_baseline_mae": baseline_mae,
        "semantic_baseline_auroc": baseline_auc,
        "mae_improvement": improvement,
        "spearman_64um": aggregate.get("spearman"),
        "mean_bias_fraction": mean_bias_fraction,
        "median_bias_fraction": median_bias_fraction,
        "beats_semantic_baseline": bool(
            isinstance(improvement, float) and improvement > 0 and beats_binary
        ),
    }


def _semantic_baseline(table, train, test):
    import numpy as np

    regions = np.asarray(table.semantic_region)
    measured = np.asarray(table.measured_od, dtype=np.float64)
    labels = np.asarray(table.binary_label, dtype=np.int8)
    valid_od = np.isfinite(measured[train])
    global_od = float(np.median(measured[train][valid_od]))
    valid_binary = labels[train] >= 0
    global_probability = (
        float(labels[train][valid_binary].mean()) if np.any(valid_binary) else np.nan
    )
    baseline_od = np.full(len(test), global_od, dtype=np.float64)
    baseline_probability = np.full(len(test), global_probability, dtype=np.float64)
    for region in np.unique(regions[test]):
        train_region = train[(regions[train] == region) & np.isfinite(measured[train])]
        test_region = regions[test] == region
        if len(train_region):
            baseline_od[test_region] = np.median(measured[train_region])
        train_binary = train[(regions[train] == region) & (labels[train] >= 0)]
        if len(train_binary):
            baseline_probability[test_region] = labels[train_binary].mean()
    return baseline_od, baseline_probability


def _region_centered_blend(predicted, semantic, regions, *, weight: float):
    """Anchor region medians while retaining morphology-driven cell variation."""

    import numpy as np

    prediction = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(semantic, dtype=np.float64)
    labels = np.asarray(regions)
    if prediction.shape != reference.shape or labels.shape != prediction.shape:
        raise ValueError("region-centered predictions must align")
    centered = prediction.copy()
    for region in np.unique(labels):
        selected = labels == region
        centered[selected] += np.median(reference[selected]) - np.median(
            prediction[selected]
        )
    return np.maximum(reference + weight * (centered - reference), 0)


def _aggregate_at_scale(xy, truth, predicted, scale_um: float):
    import numpy as np

    from histopia.protein._model import evaluate_predictions

    coordinates = np.asarray(xy, dtype=np.float64)
    measured = np.asarray(truth, dtype=np.float64)
    estimates = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(measured) & np.isfinite(estimates)
    if valid.sum() < 2:
        return {"spearman": None}
    bins = np.floor(coordinates[valid] / scale_um).astype(np.int64)
    _unique, inverse = np.unique(bins, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    measured_mean = np.bincount(inverse, weights=measured[valid]) / counts
    predicted_mean = np.bincount(inverse, weights=estimates[valid]) / counts
    return evaluate_predictions(measured_mean, predicted_mean)


def _predict(model_path: Path, table_path: Path, output: Path) -> Path:
    import numpy as np

    from histopia.protein._model import ProteinModel
    from histopia.protein._result import CellExpressionTable, ProteinPredictions

    model = ProteinModel.load(model_path)
    table = CellExpressionTable.load(table_path)
    if model.target_id != table.target_id:
        raise ValueError("protein model and cell table targets differ")
    probability, relative, od, uncertainty = model.predict(table.features)
    probabilities = (
        np.full(len(table.label_ids), np.nan, dtype=np.float32)
        if probability is None
        else probability
    )
    result = ProteinPredictions(
        target_id=model.target_id,
        model_fingerprint=str(model.fingerprint),
        label_ids=table.label_ids,
        section_ids=table.section_ids,
        expression_probability=probabilities,
        relative_expression=relative,
        predicted_od_reference=od,
        measured_od=table.measured_od,
        uncertainty=uncertainty,
        supported=np.asarray(table.semantic_support, dtype=bool),
        provenance={
            "table_fingerprint": table.fingerprint,
            "feature_view": "stain_neutral_v1",
        },
    )
    return result.save(output)


def _predict_sections(model, table, output_dir: Path) -> list[dict[str, object]]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    probability, relative, od, uncertainty = model.predict(table.features)
    probabilities = (
        np.full(len(table.label_ids), np.nan, dtype=np.float32)
        if probability is None
        else probability
    )
    from histopia.protein._result import ProteinPredictions

    rows: list[dict[str, object]] = []
    for section in np.unique(table.section_ids):
        selected = np.asarray(table.section_ids) == section
        result = ProteinPredictions(
            target_id=model.target_id,
            model_fingerprint=str(model.fingerprint),
            label_ids=table.label_ids[selected],
            section_ids=table.section_ids[selected],
            expression_probability=probabilities[selected],
            relative_expression=relative[selected],
            predicted_od_reference=od[selected],
            measured_od=table.measured_od[selected],
            uncertainty=uncertainty[selected],
            supported=np.asarray(table.semantic_support[selected], dtype=bool),
            provenance={
                "table_fingerprint": table.fingerprint,
                "feature_view": "stain_neutral_v1",
                "section": str(section),
            },
        )
        path = result.save(output_dir / f"{section}.npz")
        rows.append(
            {
                "section": str(section),
                "predictions": path.relative_to(output_dir.parent).as_posix(),
                "prediction_fingerprint": result.fingerprint,
                "cells": int(selected.sum()),
                "measured_cells": int(np.isfinite(table.measured_od[selected]).sum()),
            }
        )
    return rows


def _load_artifact(path: Path, kind: str):
    if kind == "table":
        from histopia.protein._result import CellExpressionTable

        return CellExpressionTable.load(path)
    if kind == "model":
        return _load_portable_model(path)
    from histopia.protein._result import ProteinPredictions

    return ProteinPredictions.load(path)


def _load_portable_model(path: Path):
    """Validate any portable neural protein-model architecture.

    The first portable hurdle model predates the explicit ``architecture``
    metadata field.  Newer study models declare that field and require their
    matching lightweight array loader; dispatching from sealed metadata keeps
    ``histopia-protein validate --kind model`` backward compatible.
    """

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    architecture = metadata.get("architecture")
    if architecture is None:
        from histopia.protein._model import ProteinModel

        return ProteinModel.load(path)
    if architecture == "cross_attention":
        from histopia.protein._attention import PortableCrossAttentionRegressor

        return PortableCrossAttentionRegressor.load(path)
    if architecture == "multi_tower":
        from histopia.protein._advanced import PortableMultiTowerRegressor

        return PortableMultiTowerRegressor.load(path)
    if architecture == "shared_multitask":
        from histopia.protein._deep import PortableSharedMultitaskRegressor

        return PortableSharedMultitaskRegressor.load(path)
    if architecture in {"dual_bank_attention", "graph_transformer"}:
        from histopia.protein._relational import PortableRelationalRegressor

        return PortableRelationalRegressor.load(path)
    raise ValueError(f"unsupported portable protein model architecture: {architecture}")
