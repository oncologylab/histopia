"""Resumable, bounded candidate sweeps for protein-transfer studies.

The sweep manifest is an operational file, not a review artifact. Workers use
an advisory file lock and immutable task fingerprints so several GPU hosts can
share one volume without repeating a target/architecture experiment.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic

SWEEP_ARCHITECTURES = (
    "multi_tower",
    "extra_trees",
    "cross_attention",
    "dual_bank_attention",
    "graph_transformer",
    "shared_multitask",
    "hurdle_mlp",
)
_GPU_ARCHITECTURES = frozenset(
    {
        "cross_attention",
        "multi_tower",
        "dual_bank_attention",
        "graph_transformer",
        "shared_multitask",
    }
)


def create_protein_sweep(
    specification: Path | str,
    output_dir: Path | str,
    *,
    architectures: tuple[str, ...] = SWEEP_ARCHITECTURES,
    hours: float = 8.0,
    epochs: int = 40,
    maximum_training_cells: int = 24_000,
    maximum_evaluation_cells: int = 12_000,
) -> Path:
    """Create or exactly validate an immutable sweep task set.

    The JSON specification uses schema version 1 and an ``entries`` list. Each
    entry supplies ``target_id``, ``config``, ``study``, ``table``, and
    ``geometry_cache``. Relative paths are resolved beside the specification.
    """

    from histopia.protein._config import load_protein_config
    from histopia.protein._result import CellExpressionTable

    source = Path(specification).expanduser().resolve()
    payload = json.loads(source.read_text())
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if payload.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("protein sweep specification must use schema version 1")
    selected_architectures = tuple(dict.fromkeys(str(value) for value in architectures))
    invalid = set(selected_architectures) - set(SWEEP_ARCHITECTURES)
    if not selected_architectures or invalid:
        raise ValueError(
            "unsupported protein sweep architectures: " + ", ".join(sorted(invalid))
        )
    if (
        not np.isfinite(hours)
        or hours <= 0
        or epochs < 1
        or maximum_training_cells < 64
        or maximum_evaluation_cells < 64
    ):
        raise ValueError("protein sweep resource controls are invalid")
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "sweep.json"
    normalized_entries: list[dict[str, object]] = []
    seen_targets: set[str] = set()
    required = {"target_id", "config", "study", "table", "geometry_cache"}
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(
                "each protein sweep entry must declare exactly five fields"
            )
        target_id = str(raw["target_id"])
        if not target_id or target_id in seen_targets:
            raise ValueError("protein sweep target IDs must be unique")
        seen_targets.add(target_id)
        paths = {
            key: _resolve_input_path(source.parent, str(raw[key]))
            for key in ("config", "study", "table", "geometry_cache")
        }
        config = load_protein_config(paths["config"])
        table = CellExpressionTable.load(paths["table"])
        if target_id != config.target.target_id or target_id != table.target_id:
            raise ValueError(f"protein sweep target bindings differ for {target_id}")
        feature_schema = str(table.provenance.get("feature_view", ""))
        if feature_schema not in {
            "native-hdab-neutral-spatial-uni2h-v2",
            "native-hdab-neutral-cell-multiscale-v3",
        }:
            raise ValueError(f"protein sweep feature schema is invalid for {target_id}")
        normalized_entries.append(
            {
                "target_id": target_id,
                "config": str(paths["config"]),
                "study": str(paths["study"]),
                "table": str(paths["table"]),
                "geometry_cache": str(paths["geometry_cache"]),
                "table_fingerprint": table.fingerprint,
                "feature_schema_id": feature_schema,
                "config_sha256": _file_digest(paths["config"]),
                "study_sha256": _file_digest(paths["study"]),
            }
        )
    if len(normalized_entries) < 2:
        raise ValueError("a protein sweep requires at least two target entries")
    created = _now()
    deadline = datetime.now(timezone.utc) + timedelta(hours=float(hours))
    implementation_fingerprint = _implementation_fingerprint()
    resource_controls = {
        "epochs": int(epochs),
        "maximum_training_cells": int(maximum_training_cells),
        "maximum_evaluation_cells": int(maximum_evaluation_cells),
    }
    tasks: list[dict[str, object]] = []
    for architecture in selected_architectures:
        for entry in normalized_entries:
            core = {
                "target_id": entry["target_id"],
                "architecture": architecture,
                "table_fingerprint": entry["table_fingerprint"],
                "feature_schema_id": entry["feature_schema_id"],
                "config_sha256": entry["config_sha256"],
                "study_sha256": entry["study_sha256"],
                "implementation_fingerprint": implementation_fingerprint,
                **resource_controls,
            }
            fingerprint = _json_digest(core)
            tasks.append(
                {
                    **core,
                    "id": f"{entry['target_id']}-{architecture}-{fingerprint[:12]}",
                    "fingerprint": fingerprint,
                    "state": "pending",
                    "attempts": 0,
                    "worker": None,
                    "started_at": None,
                    "lease_expires_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                }
            )
    manifest = {
        "schema_version": 1,
        "created_at": created,
        "deadline_at": deadline.isoformat(),
        "architectures": list(selected_architectures),
        "implementation_fingerprint": implementation_fingerprint,
        "resource_controls": resource_controls,
        "entries": normalized_entries,
        "tasks": tasks,
    }
    if manifest_path.is_file():
        existing = _load_manifest(manifest_path)
        comparable_keys = (
            "architectures",
            "implementation_fingerprint",
            "resource_controls",
            "entries",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise ValueError("existing protein sweep has different immutable inputs")
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    return write_json_atomic(manifest_path, manifest, sort_keys=True)


def run_protein_sweep_worker(
    manifest: Path | str,
    *,
    worker_id: str | None = None,
    architectures: tuple[str, ...] | None = None,
    maximum_tasks: int | None = None,
) -> dict[str, object]:
    """Claim and execute unique tasks until the sweep deadline or exhaustion."""

    path = Path(manifest).expanduser().resolve()
    identifier = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    allowed = set(architectures or SWEEP_ARCHITECTURES)
    if not allowed or allowed - set(SWEEP_ARCHITECTURES):
        raise ValueError("protein sweep worker architecture filter is invalid")
    completed = failed = 0
    while maximum_tasks is None or completed + failed < maximum_tasks:
        task = _claim_task(path, identifier, allowed)
        if task is None:
            break
        try:
            result = _run_task(path, task)
        except BaseException as error:
            _finish_task(path, task, identifier, error=error)
            failed += 1
        else:
            _finish_task(path, task, identifier, result=result)
            completed += 1
    status = protein_sweep_status(path)
    return {"worker": identifier, "completed": completed, "failed": failed, **status}


def protein_sweep_status(manifest: Path | str) -> dict[str, object]:
    """Return counts and deadline without exposing operational input paths."""

    payload = _load_manifest(Path(manifest))
    tasks = payload["tasks"]
    counts = {
        state: sum(task["state"] == state for task in tasks)
        for state in ("pending", "running", "completed", "failed")
    }
    return {
        "schema_version": 1,
        "deadline_at": payload["deadline_at"],
        "total": len(tasks),
        **counts,
    }


def select_protein_sweep_winners(
    manifest: Path | str | tuple[Path | str, ...],
    output: Path | str | None = None,
) -> Path:
    """Select role-specific winners across one or more disjoint sweeps."""

    raw_paths = manifest if isinstance(manifest, tuple) else (manifest,)
    paths = tuple(Path(value).expanduser().resolve() for value in raw_paths)
    if not paths:
        raise ValueError("protein sweep selection requires a manifest")
    payloads = tuple(_load_manifest(path) for path in paths)
    implementation = {row.get("implementation_fingerprint") for row in payloads}
    resources = {
        json.dumps(row.get("resource_controls"), sort_keys=True) for row in payloads
    }
    entries = {json.dumps(row.get("entries"), sort_keys=True) for row in payloads}
    if len(implementation) != 1 or len(resources) != 1 or len(entries) != 1:
        raise ValueError("protein sweep partitions have incompatible inputs")
    by_target: dict[str, list[dict[str, object]]] = {}
    seen_tasks: set[str] = set()
    for path, payload in zip(paths, payloads, strict=True):
        for task in payload["tasks"]:
            fingerprint = str(task.get("fingerprint", ""))
            if fingerprint in seen_tasks:
                raise ValueError("protein sweep partitions contain duplicate tasks")
            seen_tasks.add(fingerprint)
            if task.get("state") != "completed" or not isinstance(
                task.get("result"), str
            ):
                continue
            result_path = path.parent / str(task["result"])
            result = json.loads(result_path.read_text())
            if (
                result.get("task_fingerprint") != fingerprint
                or result.get("target_id") != task.get("target_id")
                or result.get("architecture") != task.get("architecture")
                or result.get("table_fingerprint") != task.get("table_fingerprint")
            ):
                raise ValueError("protein sweep result identity is stale")
            by_target.setdefault(str(task["target_id"]), []).append(result)
    selections: list[dict[str, object]] = []
    for target_id in sorted(by_target):
        results = by_target[target_id]
        fold_sets: list[tuple[str, ...]] = []
        transfer_candidates: dict[str, dict[str, object]] = {}
        for result in results:
            metrics = result.get("transfer")
            folds = metrics.get("folds") if isinstance(metrics, dict) else None
            if (
                not isinstance(folds, list)
                or not folds
                or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("held_out"), str)
                    or not row["held_out"]
                    for row in folds
                )
            ):
                raise ValueError("protein sweep transfer folds are malformed")
            holdouts = tuple(str(row["held_out"]) for row in folds)
            if len(set(holdouts)) != len(holdouts):
                raise ValueError("protein sweep transfer folds are duplicated")
            fold_sets.append(holdouts)
            transfer_candidates[str(result["architecture"])] = {
                **metrics,
                "evaluation": "leave_one_mouse_out",
            }
        if len(set(fold_sets)) != 1:
            raise ValueError("protein sweep architectures use different holdout folds")

        from histopia.protein._selection import (
            select_protein_architecture_candidate,
        )

        decision = select_protein_architecture_candidate(
            transfer_candidates,
            expected_holdouts=fold_sets[0],
            maximum_absolute_mean_bias_fraction=0.25,
        )
        transfer = next(
            (row for row in results if row["architecture"] == decision.selected),
            None,
        )
        diagnostic = max(
            results,
            key=lambda row: _metric_rank_key(row["transfer"]),
        )
        measured = max(results, key=lambda row: _selection_key(row["measured_fit"]))
        selections.append(
            {
                "target_id": target_id,
                "transfer_architecture": (
                    transfer["architecture"] if transfer is not None else None
                ),
                "transfer_task_fingerprint": (
                    transfer["task_fingerprint"] if transfer is not None else None
                ),
                "measured_fit_architecture": measured["architecture"],
                "measured_fit_task_fingerprint": measured["task_fingerprint"],
                "transfer_metrics": (
                    transfer["transfer"] if transfer is not None else None
                ),
                "transfer_eligible_for_protected_evaluation": transfer is not None,
                "transfer_candidate_scores": [
                    asdict(score) for score in decision.candidates
                ],
                "best_ineligible_transfer_architecture": (
                    diagnostic["architecture"] if transfer is None else None
                ),
                "best_ineligible_transfer_task_fingerprint": (
                    diagnostic["task_fingerprint"] if transfer is None else None
                ),
                "measured_fit_metrics": measured["measured_fit"],
            }
        )
    destination = (
        Path(output) if output is not None else paths[0].parent / "winners.json"
    )
    return write_json_atomic(
        destination,
        {
            "schema_version": 1,
            "selection": "guardrailed-transfer-and-measured-fit-v2",
            "targets": selections,
        },
        sort_keys=True,
    )


def _run_task(manifest_path: Path, task: dict[str, object]) -> dict[str, object]:
    from histopia.protein._config import load_protein_config
    from histopia.protein._result import CellExpressionTable

    manifest = _load_manifest(manifest_path)
    entries = {str(row["target_id"]): row for row in manifest["entries"]}
    entry = entries[str(task["target_id"])]
    config = load_protein_config(Path(str(entry["config"])))
    table = CellExpressionTable.load(Path(str(entry["table"])))
    if (
        table.fingerprint != task["table_fingerprint"]
        or _file_digest(Path(str(entry["config"]))) != task["config_sha256"]
        or _file_digest(Path(str(entry["study"]))) != task["study_sha256"]
        or _implementation_fingerprint() != task["implementation_fingerprint"]
    ):
        raise ValueError("protein sweep input changed after task creation")
    config.feature_schema_id = str(task["feature_schema_id"])
    architecture = str(task["architecture"])
    seed = int(config.seed) + int(str(task["fingerprint"])[:8], 16)
    auxiliary_tables = []
    if architecture == "shared_multitask":
        for target_id, auxiliary_entry in sorted(entries.items()):
            if target_id == table.target_id:
                continue
            auxiliary = CellExpressionTable.load(Path(str(auxiliary_entry["table"])))
            if auxiliary.features.shape[1] == table.features.shape[1]:
                auxiliary_tables.append(auxiliary)
    transfer_folds: list[dict[str, object]] = []
    mouse_ids = np.asarray(table.mouse_ids).astype(str)
    for fold_index, held_out in enumerate(sorted(np.unique(mouse_ids))):
        train = np.flatnonzero(mouse_ids != held_out)
        test = np.flatnonzero(mouse_ids == held_out)
        train = _balanced_limit(
            table,
            train,
            int(task["maximum_training_cells"]),
            seed + fold_index,
        )
        test = _balanced_limit(
            table,
            test,
            int(task["maximum_evaluation_cells"]),
            seed + 100 + fold_index,
        )
        auxiliary = _auxiliary_training(auxiliary_tables, held_out=held_out)
        fold = _evaluate_sweep_fold(
            architecture,
            config,
            table,
            train,
            test,
            auxiliary=auxiliary,
            seed=seed + fold_index,
            epochs=int(task["epochs"]),
        )
        transfer_folds.append({"held_out": held_out, **fold})
    measured_train, measured_test = _within_section_split(table, seed=seed)
    measured_train = _balanced_limit(
        table,
        measured_train,
        int(task["maximum_training_cells"]),
        seed + 1000,
    )
    measured_test = _balanced_limit(
        table,
        measured_test,
        int(task["maximum_evaluation_cells"]),
        seed + 1001,
    )
    measured_auxiliary = _auxiliary_training(auxiliary_tables, held_out=None)
    measured_fold = _evaluate_sweep_fold(
        architecture,
        config,
        table,
        measured_train,
        measured_test,
        auxiliary=measured_auxiliary,
        seed=seed + 1000,
        epochs=int(task["epochs"]),
    )
    from histopia.protein._cli import _aggregate_candidate

    return {
        "schema_version": 1,
        "target_id": table.target_id,
        "architecture": architecture,
        "feature_schema_id": table.provenance.get("feature_view"),
        "table_fingerprint": table.fingerprint,
        "task_fingerprint": task["fingerprint"],
        "transfer": _aggregate_candidate(transfer_folds),
        "measured_fit": _aggregate_candidate(
            [{"held_out": "within-measured-sections", **measured_fold}]
        ),
        "evaluation_notes": {
            "transfer": "leave-one-mouse-out; training-only relational banks",
            "measured_fit": "stratified within-section holdout; not transfer evidence",
            "semantic_regions": "auxiliary baseline/reporting only for v3 features",
        },
    }


def _evaluate_sweep_fold(
    architecture: str,
    config: object,
    table: object,
    train: np.ndarray,
    test: np.ndarray,
    *,
    auxiliary: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    seed: int,
    epochs: int,
) -> dict[str, object]:
    if not len(train) or not len(test):
        raise ValueError("protein sweep fold is empty")
    if architecture == "shared_multitask":
        from histopia.protein._cli import _evaluate_candidate_arrays
        from histopia.protein._deep import fit_shared_multitask_candidate

        predicted = fit_shared_multitask_candidate(
            table.features,
            table.measured_od,
            train,
            test,
            auxiliary=auxiliary,
            seed=seed,
            epochs=epochs,
        )
        return _evaluate_candidate_arrays(table, train, test, predicted, None)
    candidate = "tree" if architecture == "extra_trees" else architecture
    if candidate in _GPU_ARCHITECTURES:
        from histopia.protein._cli import _evaluate_candidate_arrays
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
                    * float(config.section_spacing_um),
                )
            ),
            section_groups=np.asarray(table.section_ids),
            seed=seed,
            epochs=epochs,
        )
        return _evaluate_candidate_arrays(table, train, test, predicted, None)
    from histopia.protein._cli import _evaluate_candidate

    return _evaluate_candidate(candidate, config, table, train, test)


def _auxiliary_training(
    tables: list[object], *, held_out: str | None
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    output = []
    for table in tables:
        mouse = np.asarray(table.mouse_ids).astype(str)
        indices = (
            np.flatnonzero(mouse != held_out)
            if held_out is not None
            else np.arange(len(mouse))
        )
        output.append((table.features, table.measured_od, indices))
    return tuple(output)


def _balanced_limit(
    table: object,
    indices: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    measured = np.asarray(table.measured_od, dtype=np.float64)
    selected = selected[np.isfinite(measured[selected])]
    if len(selected) <= maximum:
        return selected
    values = measured[selected]
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 9)[1:-1]))
    od_bin = np.searchsorted(quantiles, values, side="right")
    groups = np.char.add(
        np.char.add(np.asarray(table.mouse_ids)[selected].astype(str), ":"),
        np.asarray(table.section_ids)[selected].astype(str),
    )
    _group, group_index = np.unique(groups, return_inverse=True)
    strata = np.column_stack((group_index, od_bin))
    _unique, inverse, counts = np.unique(
        strata, axis=0, return_inverse=True, return_counts=True
    )
    weight = 1.0 / counts[inverse]
    weight /= weight.sum()
    return np.sort(
        np.random.default_rng(seed).choice(
            selected,
            size=maximum,
            replace=False,
            p=weight,
        )
    )


def _within_section_split(
    table: object, *, seed: int, test_fraction: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    measured = np.asarray(table.measured_od, dtype=np.float64)
    eligible = np.flatnonzero(np.isfinite(measured))
    keys = np.char.add(
        np.char.add(np.asarray(table.mouse_ids)[eligible].astype(str), ":"),
        np.asarray(table.section_ids)[eligible].astype(str),
    )
    generator = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for key in np.unique(keys):
        rows = eligible[keys == key].copy()
        generator.shuffle(rows)
        count = min(max(int(round(len(rows) * test_fraction)), 1), len(rows) - 1)
        test_parts.append(rows[:count])
        train_parts.append(rows[count:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


def _claim_task(path: Path, worker: str, allowed: set[str]) -> dict[str, object] | None:
    with _manifest_lock(path):
        payload = _load_manifest(path)
        now = datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(str(payload["deadline_at"]))
        if now >= deadline:
            return None
        for task in payload["tasks"]:
            if (
                task["state"] == "running"
                and datetime.fromisoformat(str(task["lease_expires_at"])) <= now
            ):
                task.update(
                    state="pending",
                    worker=None,
                    started_at=None,
                    lease_expires_at=None,
                    error="expired worker lease",
                )
        selected = next(
            (
                task
                for task in payload["tasks"]
                if task["state"] == "pending"
                and task["architecture"] in allowed
                and int(task["attempts"]) < 2
            ),
            None,
        )
        if selected is None:
            write_json_atomic(path, payload, sort_keys=True)
            return None
        selected.update(
            state="running",
            attempts=int(selected["attempts"]) + 1,
            worker=worker,
            started_at=_now(),
            lease_expires_at=(now + timedelta(hours=4)).isoformat(),
            error=None,
        )
        write_json_atomic(path, payload, sort_keys=True)
        return dict(selected)


def _finish_task(
    path: Path,
    claimed: dict[str, object],
    worker: str,
    *,
    result: dict[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    with _manifest_lock(path):
        payload = _load_manifest(path)
        task = next(row for row in payload["tasks"] if row["id"] == claimed["id"])
        if task["state"] != "running" or task["worker"] != worker:
            raise RuntimeError("protein sweep task ownership changed during execution")
        if error is None:
            if result is None:
                raise ValueError("completed protein sweep task has no result")
            relative = f"results/{task['id']}.json"
            write_json_atomic(path.parent / relative, result, sort_keys=True)
            task.update(
                state="completed",
                completed_at=_now(),
                result=relative,
                error=None,
                lease_expires_at=None,
            )
        else:
            message = f"{type(error).__name__}: {error}"
            for entry in payload["entries"]:
                for key in ("config", "study", "table", "geometry_cache"):
                    message = message.replace(str(entry[key]), f"<{key}>")
            task.update(
                state="failed" if int(task["attempts"]) >= 2 else "pending",
                completed_at=_now(),
                error=message[:1000],
                worker=None,
                lease_expires_at=None,
            )
        write_json_atomic(path, payload, sort_keys=True)


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    lock = path.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("entries"), list)
        or not isinstance(payload.get("tasks"), list)
    ):
        raise ValueError("protein sweep manifest is malformed")
    return payload


def _metric_rank_key(metrics: object) -> tuple[float, ...]:
    if not isinstance(metrics, dict) or metrics.get("available") is not True:
        return (-2.0,) * 7
    spearman_64 = metrics.get("spearman_64um")
    spearman = metrics.get("spearman")
    bias = metrics.get("mean_bias_fraction")
    mae = metrics.get("mae")
    folds = metrics.get("folds")
    valid_folds = (
        [row for row in folds if isinstance(row, dict)]
        if isinstance(folds, list)
        else []
    )
    fold_spearman = [
        float(row["spearman_64um"])
        for row in valid_folds
        if isinstance(row.get("spearman_64um"), (int, float))
    ]
    fold_bias = [
        abs(float(row["mean_bias_fraction"]))
        for row in valid_folds
        if isinstance(row.get("mean_bias_fraction"), (int, float))
    ]
    return (
        float(spearman_64) if isinstance(spearman_64, (int, float)) else -2.0,
        min(fold_spearman) if fold_spearman else -2.0,
        float(spearman) if isinstance(spearman, (int, float)) else -2.0,
        -max(fold_bias) if fold_bias else -2.0,
        -abs(float(bias)) if isinstance(bias, (int, float)) else -2.0,
        -float(mae) if isinstance(mae, (int, float)) else -1e9,
        float(metrics.get("test_cells", 0)),
    )


def _selection_key(metrics: object) -> tuple[float, ...]:
    """Backward-compatible alias for measured-fit ranking."""

    return _metric_rank_key(metrics)


def _resolve_input_path(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _implementation_fingerprint() -> str:
    """Bind sweep tasks to the exact lightweight modeling implementation."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"histopia-protein-sweep-implementation-v1\0")
    for name in (
        "_advanced.py",
        "_attention.py",
        "_cli.py",
        "_deep.py",
        "_model.py",
        "_sweep.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
