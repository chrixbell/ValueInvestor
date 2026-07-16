"""Time-boxed search for improved ML ranker artifacts.

This is intentionally experiment-oriented: it writes JSONL progress to /tmp
by default and only promotes a candidate when it clears walk-forward selection
and the untouched holdout gate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from valueinvestor.scorer_improver.ml_trainer import (
    ML_SNAPSHOTS_FILE,
    _baseline_metrics,
    _evaluate_predictions,
    _fit_ridge,
    _improvement_ratio,
    _payload_with_ticker_priors,
    _ticker_priors,
    build_feature_matrix,
    evaluate_ml_ranker_payload,
    predict_ml_ranker_payload,
)
from valueinvestor.scorer_improver.promotion_gate import (
    HoldoutGateConfig,
    append_gate_ledger,
    evaluate_promotion_gate,
    make_holdout_split,
)
from valueinvestor.screener.ml_ranker import MODEL_SCHEMA_VERSION, expected_feature_names


LAMBDA_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0)
HORIZONS = ("1m", "3m", "6m")
SAMPLE_CONFIGS = (
    (500_000, 0),
    (500_000, 17),
    (500_000, 41),
    (750_000, 3),
    (750_000, 29),
    (1_000_000, 7),
    (1_000_000, 53),
    (1_500_000, 11),
)
DEFAULT_WALK_FORWARD_FOLDS = 3
DEFAULT_WALK_FORWARD_VALIDATION_MONTHS = 6
DEFAULT_WALK_FORWARD_MAX_ROWS = 500_000
DEFAULT_WALK_FORWARD_MIN_6M_DELTA = 0.001
DEFAULT_WALK_FORWARD_MAX_HORIZON_DEGRADATION = 0.10


def _log(report_path: Path, record: Mapping[str, object]) -> None:
    payload = {"logged_at": datetime.now().isoformat(), **record}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def _print_step(message: str) -> None:
    print(f"[ml-search] {message}", flush=True)


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_payloads(payload: Mapping[str, object]) -> list[dict]:
    models = payload.get("models")
    if isinstance(models, list) and models:
        return [dict(model) for model in models]
    return [{
        "clip_low": payload["clip_low"],
        "clip_high": payload["clip_high"],
        "mean": payload["mean"],
        "scale": payload["scale"],
        "coef": payload["coef"],
        "intercept": payload["intercept"],
        "weight": payload.get("weight", 1.0),
    }]


def _model_weight(model: Mapping[str, object]) -> float:
    try:
        return float(model.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _sample_by_snapshot(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    groups = list(df.groupby("snapshot_date", sort=True))
    rows_per_snapshot = max(10, max_rows // len(groups))
    sampled = []
    for idx, (_snapshot_date, group) in enumerate(groups):
        n_rows = min(len(group), rows_per_snapshot)
        if n_rows > 0:
            sampled.append(group.sample(n=n_rows, random_state=seed + idx))
    sample = pd.concat(sampled)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed)
    return sample.sort_values(["snapshot_date", "ticker"], kind="mergesort").reset_index(drop=True)


def _standardize(X: np.ndarray) -> tuple[np.ndarray, dict]:
    clip_low = np.percentile(X, 1, axis=0)
    clip_high = np.percentile(X, 99, axis=0)
    clipped = np.clip(X, clip_low, clip_high)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (clipped - mean) / scale, {
        "clip_low": clip_low,
        "clip_high": clip_high,
        "mean": mean,
        "scale": scale,
    }


def _target_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rank_1m = pd.to_numeric(df["target_rank_1m"], errors="coerce").to_numpy(dtype="float64")
    rank_3m = pd.to_numeric(df["target_rank_3m"], errors="coerce").to_numpy(dtype="float64")
    rank_6m = pd.to_numeric(df["target_rank_6m"], errors="coerce").to_numpy(dtype="float64")

    def signed_power(values: np.ndarray, power: float) -> np.ndarray:
        return np.sign(values) * (np.abs(values) ** power)

    return {
        "rank_6m": rank_6m,
        "rank_6m_extreme": signed_power(rank_6m, 1.35),
        "rank_6m_soft": signed_power(rank_6m, 0.75),
        "weighted_901": 0.10 * rank_3m + 0.90 * rank_6m,
        "weighted_8515": 0.15 * rank_3m + 0.85 * rank_6m,
        "weighted_703": 0.10 * rank_1m + 0.20 * rank_3m + 0.70 * rank_6m,
        "weighted_802": 0.00 * rank_1m + 0.20 * rank_3m + 0.80 * rank_6m,
        "weighted_7525": 0.25 * rank_3m + 0.75 * rank_6m,
        "weighted_631": 0.15 * rank_1m + 0.25 * rank_3m + 0.60 * rank_6m,
        "mean": (rank_1m + rank_3m + rank_6m) / 3.0,
    }


def _snapshot_dates(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("snapshot_date contains invalid values")
    return dates


def _walk_forward_folds(
    df: pd.DataFrame,
    *,
    n_folds: int,
    validation_months: int,
    embargo_days: int,
    min_train_snapshots: int = 60,
    min_validation_snapshots: int = 20,
) -> list[dict[str, Any]]:
    if n_folds <= 0 or validation_months <= 0:
        return []

    dates = _snapshot_dates(df)
    unique_dates = pd.Index(sorted(dates.unique()))
    if len(unique_dates) < min_train_snapshots + min_validation_snapshots:
        return []

    max_date = pd.Timestamp(unique_dates[-1])
    folds: list[dict[str, Any]] = []
    for offset in range(n_folds - 1, -1, -1):
        validation_end = max_date - pd.DateOffset(months=validation_months * offset)
        validation_start = validation_end - pd.DateOffset(months=validation_months) + timedelta(days=1)
        train_end_exclusive = validation_start - timedelta(days=embargo_days)

        train_mask = dates < train_end_exclusive
        validation_mask = (dates >= validation_start) & (dates <= validation_end)
        train_snapshots = int(dates.loc[train_mask].nunique())
        validation_snapshots = int(dates.loc[validation_mask].nunique())
        if train_snapshots < min_train_snapshots:
            continue
        if validation_snapshots < min_validation_snapshots:
            continue

        folds.append({
            "index": len(folds) + 1,
            "train_end_exclusive": pd.Timestamp(train_end_exclusive),
            "validation_start": pd.Timestamp(validation_start),
            "validation_end": pd.Timestamp(validation_end),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "train_snapshots": train_snapshots,
            "validation_snapshots": validation_snapshots,
        })
    return folds


def _fold_summary(fold: Mapping[str, Any]) -> dict[str, object]:
    return {
        "index": fold["index"],
        "train_end_exclusive": pd.Timestamp(fold["train_end_exclusive"]).date().isoformat(),
        "validation_start": pd.Timestamp(fold["validation_start"]).date().isoformat(),
        "validation_end": pd.Timestamp(fold["validation_end"]).date().isoformat(),
        "train_rows": fold["train_rows"],
        "validation_rows": fold["validation_rows"],
        "train_snapshots": fold["train_snapshots"],
        "validation_snapshots": fold["validation_snapshots"],
    }


def _walk_forward_decision(
    fold_deltas: list[Mapping[str, float]],
    *,
    min_6m_delta: float,
    max_horizon_degradation: float,
) -> dict[str, object]:
    if not fold_deltas:
        return {
            "accepted": False,
            "reason": "no walk-forward folds",
            "mean_deltas": {},
            "median_deltas": {},
            "min_deltas": {},
            "fold_deltas": [],
        }

    arrays = {
        horizon: np.asarray([float(delta[horizon]) for delta in fold_deltas], dtype="float64")
        for horizon in HORIZONS
    }
    mean_deltas = {horizon: float(values.mean()) for horizon, values in arrays.items()}
    median_deltas = {horizon: float(np.median(values)) for horizon, values in arrays.items()}
    min_deltas = {horizon: float(values.min()) for horizon, values in arrays.items()}

    reason = "accepted"
    if any(not np.isfinite(values).all() for values in arrays.values()):
        reason = "non-finite walk-forward delta"
    elif mean_deltas["6m"] < min_6m_delta:
        reason = "mean 6m delta below walk-forward gate"
    elif median_deltas["6m"] < 0.0:
        reason = "median 6m delta below zero"
    elif any(mean_deltas[horizon] < -max_horizon_degradation for horizon in HORIZONS):
        reason = "mean horizon degradation"
    elif min_deltas["6m"] < -max_horizon_degradation:
        reason = "fold 6m degradation"

    return {
        "accepted": reason == "accepted",
        "reason": reason,
        "mean_deltas": {horizon: round(mean_deltas[horizon], 6) for horizon in HORIZONS},
        "median_deltas": {horizon: round(median_deltas[horizon], 6) for horizon in HORIZONS},
        "min_deltas": {horizon: round(min_deltas[horizon], 6) for horizon in HORIZONS},
        "fold_deltas": [
            {horizon: round(float(delta[horizon]), 6) for horizon in HORIZONS}
            for delta in fold_deltas
        ],
    }


def _fit_candidate_model(
    train_snapshots: pd.DataFrame,
    *,
    target_name: str,
    ridge_lambda: float,
    backend: str,
    max_rows: int,
    seed: int,
) -> tuple[dict[str, object], Mapping[str, Mapping[str, float]], str]:
    priors = _ticker_priors(train_snapshots)
    sample_rows = len(train_snapshots) if max_rows <= 0 else max_rows
    sample = _sample_by_snapshot(train_snapshots, sample_rows, seed)
    X = build_feature_matrix(sample, priors)
    X_scaled, standardization = _standardize(X)
    target = _target_arrays(sample)[target_name]
    valid = np.isfinite(target)
    if int(valid.sum()) < 10:
        raise ValueError(f"not enough valid rows for walk-forward target {target_name}")
    coef_with_intercept, used_backend = _fit_ridge(
        X_scaled[valid],
        target[valid],
        ridge_lambda=ridge_lambda,
        backend=backend,
    )
    return {
        "clip_low": standardization["clip_low"].tolist(),
        "clip_high": standardization["clip_high"].tolist(),
        "mean": standardization["mean"].tolist(),
        "scale": standardization["scale"].tolist(),
        "coef": coef_with_intercept[:-1].tolist(),
        "intercept": float(coef_with_intercept[-1]),
    }, priors, used_backend


def _walk_forward_validate_candidate(
    snapshots: pd.DataFrame,
    snapshot_dates: pd.Series,
    folds: list[Mapping[str, Any]],
    incumbent_fold_metrics: list[Mapping[str, Mapping[str, float]]],
    *,
    target_name: str,
    ridge_lambda: float,
    backend: str,
    max_rows: int,
    seed: int,
    chunk_size: int,
    min_6m_delta: float,
    max_horizon_degradation: float,
) -> dict[str, object]:
    fold_deltas: list[Mapping[str, float]] = []
    fold_metrics: list[dict[str, object]] = []

    for fold, incumbent_metrics in zip(folds, incumbent_fold_metrics):
        train_mask = snapshot_dates < pd.Timestamp(fold["train_end_exclusive"])
        validation_mask = (
            (snapshot_dates >= pd.Timestamp(fold["validation_start"]))
            & (snapshot_dates <= pd.Timestamp(fold["validation_end"]))
        )
        train = snapshots.loc[train_mask]
        validation = snapshots.loc[validation_mask]
        model, priors, used_backend = _fit_candidate_model(
            train,
            target_name=target_name,
            ridge_lambda=ridge_lambda,
            backend=backend,
            max_rows=max_rows,
            seed=seed + int(fold["index"]) * 10_000,
        )
        predictions = _predict_linear(validation, priors, model, chunk_size=chunk_size)
        candidate_metrics = _evaluate_predictions(validation, predictions)
        deltas = {
            horizon: (
                float(candidate_metrics[horizon]["spearman_rho"])
                - float(incumbent_metrics[horizon]["spearman_rho"])
            )
            for horizon in HORIZONS
        }
        fold_deltas.append(deltas)
        fold_metrics.append({
            "fold": _fold_summary(fold),
            "backend": used_backend,
            "candidate_rhos": {
                horizon: round(float(candidate_metrics[horizon]["spearman_rho"]), 6)
                for horizon in HORIZONS
            },
            "incumbent_rhos": {
                horizon: round(float(incumbent_metrics[horizon]["spearman_rho"]), 6)
                for horizon in HORIZONS
            },
            "deltas": {horizon: round(float(deltas[horizon]), 6) for horizon in HORIZONS},
        })

    decision = _walk_forward_decision(
        fold_deltas,
        min_6m_delta=min_6m_delta,
        max_horizon_degradation=max_horizon_degradation,
    )
    decision["folds"] = fold_metrics
    return decision


def _predict_linear(
    df: pd.DataFrame,
    priors: Mapping[str, Mapping[str, float]],
    model: Mapping[str, object],
    *,
    chunk_size: int,
) -> np.ndarray:
    clip_low = np.asarray(model["clip_low"], dtype="float64")
    clip_high = np.asarray(model["clip_high"], dtype="float64")
    mean = np.asarray(model["mean"], dtype="float64")
    scale = np.asarray(model["scale"], dtype="float64")
    coef = np.asarray(model["coef"], dtype="float64")
    intercept = float(model["intercept"])
    predictions = np.empty(len(df), dtype="float64")
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        X = build_feature_matrix(df.iloc[start:end], priors)
        X_scaled = (np.clip(X, clip_low, clip_high) - mean) / scale
        predictions[start:end] = X_scaled @ coef + intercept
    return predictions


def _predict_models_average(
    df: pd.DataFrame,
    priors: Mapping[str, Mapping[str, float]],
    models: list[Mapping[str, object]],
    *,
    chunk_size: int,
) -> np.ndarray:
    predictions = np.zeros(len(df), dtype="float64")
    unweighted_predictions = np.zeros(len(df), dtype="float64")
    total_weight = 0.0
    for model in models:
        weight = _model_weight(model)
        model_predictions = _predict_linear(df, priors, model, chunk_size=chunk_size)
        predictions += weight * model_predictions
        unweighted_predictions += model_predictions
        total_weight += weight
    if total_weight <= 0:
        return unweighted_predictions / len(models)
    return predictions / total_weight


def _evaluate_linear_full(
    snapshots: pd.DataFrame,
    priors: Mapping[str, Mapping[str, float]],
    model: Mapping[str, object],
    *,
    chunk_size: int,
) -> tuple[dict, np.ndarray]:
    predictions = _predict_linear(snapshots, priors, model, chunk_size=chunk_size)
    return _evaluate_predictions(snapshots, predictions), predictions


def _candidate_payload(
    *,
    model: Mapping[str, object],
    priors: Mapping[str, Mapping[str, float]],
    baseline_metrics: Mapping[str, object],
    metrics: Mapping[str, object],
    snapshots_path: Path,
    extra_metadata: Mapping[str, object],
) -> dict:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": expected_feature_names(),
        "clip_low": list(model["clip_low"]),
        "clip_high": list(model["clip_high"]),
        "mean": list(model["mean"]),
        "scale": list(model["scale"]),
        "coef": list(model["coef"]),
        "intercept": float(model["intercept"]),
        "ticker_priors": priors,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "backend": extra_metadata.get("backend", "mlx"),
            "target": extra_metadata["target"],
            "ridge_lambda": extra_metadata["ridge_lambda"],
            "baseline_metrics": baseline_metrics,
            "metrics": metrics,
            "final_rho_improvement": _improvement_ratio(
                float(metrics["6m"]["spearman_rho"]),
                float(baseline_metrics["6m"]["spearman_rho"]),
            ),
            "ground_truth_id": (
                f"{snapshots_path.name}:{snapshots_path.stat().st_size}:"
                f"{snapshots_path.stat().st_mtime_ns}|eval=ranked-snapshot-v1|model=ml-ranker-v1"
            ),
            "snapshots_path": str(snapshots_path),
            **extra_metadata,
        },
    }


def _ensemble_payload(
    *,
    models: list[Mapping[str, object]],
    priors: Mapping[str, Mapping[str, float]],
    baseline_metrics: Mapping[str, object],
    metrics: Mapping[str, object],
    snapshots_path: Path,
    extra_metadata: Mapping[str, object],
) -> dict:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": expected_feature_names(),
        "models": [dict(model) for model in models],
        "ticker_priors": priors,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "backend": "ensemble",
            "baseline_metrics": baseline_metrics,
            "metrics": metrics,
            "final_rho_improvement": _improvement_ratio(
                float(metrics["6m"]["spearman_rho"]),
                float(baseline_metrics["6m"]["spearman_rho"]),
            ),
            "ensemble_size": len(models),
            "ground_truth_id": (
                f"{snapshots_path.name}:{snapshots_path.stat().st_size}:"
                f"{snapshots_path.stat().st_mtime_ns}|eval=ranked-snapshot-v1|model=ml-ranker-v1"
            ),
            "snapshots_path": str(snapshots_path),
            **extra_metadata,
        },
    }


def _promote(
    payload: Mapping[str, object],
    *,
    model_path: Path,
    best_meta_path: Path,
) -> None:
    backup_path = model_path.with_suffix(f".backup-{int(time.time())}.json")
    if model_path.exists():
        shutil.copy2(model_path, backup_path)
    model_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    metrics = payload["metadata"]["metrics"]
    best_meta_path.write_text(
        json.dumps({
            "ground_truth_id": payload["metadata"]["ground_truth_id"],
            "iteration": 14256,
            "spearman_rho_1m": round(float(metrics["1m"]["spearman_rho"]), 6),
            "spearman_rho_3m": round(float(metrics["3m"]["spearman_rho"]), 6),
            "spearman_rho_6m": round(float(metrics["6m"]["spearman_rho"]), 6),
            "updated_at": payload["metadata"]["trained_at"],
        }, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--snapshots-path", type=Path, default=ML_SNAPSHOTS_FILE)
    parser.add_argument("--model-path", type=Path, default=Path("data/trainer/ml_ranker_model.json"))
    parser.add_argument("--best-meta-path", type=Path, default=Path("data/trainer/current_best_scorer.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/tmp/valueinvestor_ml_search_report.jsonl"))
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--backend", default="mlx", choices=("mlx", "auto", "numpy"))
    parser.add_argument("--full-evals-per-sample", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-iterations", type=int, default=0, help="Max search samples (0 = time-boxed)")
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=DEFAULT_WALK_FORWARD_FOLDS,
        help="Rolling validation folds before staging candidates (0 disables).",
    )
    parser.add_argument(
        "--walk-forward-validation-months",
        type=int,
        default=DEFAULT_WALK_FORWARD_VALIDATION_MONTHS,
        help="Validation window length per walk-forward fold.",
    )
    parser.add_argument(
        "--walk-forward-max-rows",
        type=int,
        default=DEFAULT_WALK_FORWARD_MAX_ROWS,
        help="Max rows sampled per walk-forward training fold.",
    )
    parser.add_argument(
        "--walk-forward-min-6m-delta",
        type=float,
        default=DEFAULT_WALK_FORWARD_MIN_6M_DELTA,
        help="Required mean 6m rho improvement across walk-forward folds.",
    )
    parser.add_argument(
        "--walk-forward-max-horizon-degradation",
        type=float,
        default=DEFAULT_WALK_FORWARD_MAX_HORIZON_DEGRADATION,
        help=(
            "Allowed mean non-primary rho degradation and worst primary-fold "
            "degradation during walk-forward validation."
        ),
    )
    args = parser.parse_args()

    deadline = time.monotonic() + args.hours * 3600
    _print_step(
        f"loading snapshots from {args.snapshots_path} and starting a bounded search "
        f"({args.max_iterations or 'time-boxed'} iterations)"
    )
    current_payload = _load_payload(args.model_path)

    snapshots = pd.read_parquet(str(args.snapshots_path))
    gate_config = HoldoutGateConfig()
    split = make_holdout_split(snapshots, config=gate_config, source_path=args.snapshots_path)
    search_snapshots = split.train
    gate_snapshots = split.gate
    search_snapshot_dates = _snapshot_dates(search_snapshots)
    priors = _ticker_priors(search_snapshots)
    current_eval_payload = _payload_with_ticker_priors(current_payload, priors)
    if current_eval_payload is None:
        current_eval_payload = current_payload
    baseline_metrics = _baseline_metrics(search_snapshots)
    current_search_metrics = evaluate_ml_ranker_payload(search_snapshots, current_eval_payload)
    current_gate_metrics = evaluate_ml_ranker_payload(gate_snapshots, current_eval_payload)
    walk_forward_folds = _walk_forward_folds(
        search_snapshots,
        n_folds=args.walk_forward_folds,
        validation_months=args.walk_forward_validation_months,
        embargo_days=gate_config.embargo_days,
    )
    incumbent_fold_metrics = [
        evaluate_ml_ranker_payload(
            search_snapshots.loc[
                (search_snapshot_dates >= pd.Timestamp(fold["validation_start"]))
                & (search_snapshot_dates <= pd.Timestamp(fold["validation_end"]))
            ],
            current_eval_payload,
            chunk_size=args.chunk_size,
        )
        for fold in walk_forward_folds
    ]
    current_rho = float(current_gate_metrics["6m"]["spearman_rho"])
    current_search_rho = float(current_search_metrics["6m"]["spearman_rho"])
    best_rho = current_search_rho
    best_payload = current_payload
    current_linear_models = _linear_payloads(current_eval_payload)
    full_candidates: list[dict] = []
    current_predictions: np.ndarray | None = None

    _print_step(
        "split ready: "
        f"train={len(search_snapshots):,} rows / {search_snapshots['snapshot_date'].nunique():,} snapshots, "
        f"gate={len(gate_snapshots):,} rows / {gate_snapshots['snapshot_date'].nunique():,} snapshots, "
        f"embargo={split.manifest['embargo_days']} days"
    )
    _print_step(
        "current model on split: "
        f"search 6m rho={current_search_rho:.6f}, gate 6m rho={current_rho:.6f}"
    )
    if walk_forward_folds:
        _print_step(
            "walk-forward filter ready: "
            f"{len(walk_forward_folds)} folds, "
            f"{args.walk_forward_validation_months} validation months/fold, "
            f"mean Δ6m gate={args.walk_forward_min_6m_delta:+.6f}"
        )
    elif args.walk_forward_folds > 0:
        _print_step("walk-forward filter disabled: not enough dated snapshots for folds")

    _log(args.report_path, {
        "event": "start",
        "current_gate_rho_6m": current_rho,
        "current_search_rho_6m": current_search_rho,
        "rows": len(search_snapshots),
        "snapshots": int(search_snapshots["snapshot_date"].nunique()),
        "gate_rows": len(gate_snapshots),
        "gate_snapshots": int(gate_snapshots["snapshot_date"].nunique()),
        "promotion_gate_manifest": split.manifest,
        "walk_forward_folds": [_fold_summary(fold) for fold in walk_forward_folds],
    })

    sample_index = 0
    while time.monotonic() < deadline and (
        args.max_iterations <= 0 or sample_index < args.max_iterations
    ):
        if sample_index < len(SAMPLE_CONFIGS):
            max_rows, seed = SAMPLE_CONFIGS[sample_index]
        else:
            max_rows = (500_000, 750_000, 1_000_000, 1_500_000)[sample_index % 4]
            seed = 101 + sample_index * 37
        sample_index += 1
        iteration_no = sample_index
        _print_step(f"iteration {iteration_no}: sampling {max_rows:,} rows (seed={seed})")

        sample = _sample_by_snapshot(search_snapshots, max_rows, seed)
        X = build_feature_matrix(sample, priors)
        X_scaled, standardization = _standardize(X)
        targets = _target_arrays(sample)
        sample_candidates = []

        for target_name, target in targets.items():
            valid = np.isfinite(target)
            if int(valid.sum()) < 10:
                continue
            for ridge_lambda in LAMBDA_GRID:
                if time.monotonic() >= deadline:
                    break
                coef_with_intercept, backend = _fit_ridge(
                    X_scaled[valid],
                    target[valid],
                    ridge_lambda=ridge_lambda,
                    backend=args.backend,
                )
                coef = coef_with_intercept[:-1]
                intercept = float(coef_with_intercept[-1])
                predictions = X_scaled @ coef + intercept
                sample_metrics = _evaluate_predictions(sample, predictions)
                sample_rho = float(sample_metrics["6m"]["spearman_rho"])
                model = {
                    "clip_low": standardization["clip_low"].tolist(),
                    "clip_high": standardization["clip_high"].tolist(),
                    "mean": standardization["mean"].tolist(),
                    "scale": standardization["scale"].tolist(),
                    "coef": coef.tolist(),
                    "intercept": intercept,
                }
                candidate = {
                    "model": model,
                    "sample_metrics": sample_metrics,
                    "sample_rho": sample_rho,
                    "target": target_name,
                    "ridge_lambda": float(ridge_lambda),
                    "backend": backend,
                    "sample_rows": len(sample),
                    "sample_seed": seed,
                }
                sample_candidates.append(candidate)
                _log(args.report_path, {
                    "event": "sample",
                    "target": target_name,
                    "ridge_lambda": ridge_lambda,
                    "sample_rows": len(sample),
                    "sample_seed": seed,
                    "rho_6m": sample_rho,
                })

        sample_candidates.sort(key=lambda candidate: candidate["sample_rho"], reverse=True)
        top_sample = sample_candidates[0] if sample_candidates else None
        if top_sample is not None:
            _print_step(
                f"iteration {iteration_no}: best sample candidate "
                f"{top_sample['target']} lambda={top_sample['ridge_lambda']:.1f} "
                f"6m rho={top_sample['sample_rho']:.6f}"
            )
        for candidate in sample_candidates[:args.full_evals_per_sample]:
            if time.monotonic() >= deadline:
                break
            metrics, predictions = _evaluate_linear_full(
                search_snapshots,
                priors,
                candidate["model"],
                chunk_size=args.chunk_size,
            )
            full_rho = float(metrics["6m"]["spearman_rho"])
            walk_forward_result: dict[str, object] | None = None
            if walk_forward_folds:
                walk_forward_result = _walk_forward_validate_candidate(
                    search_snapshots,
                    search_snapshot_dates,
                    walk_forward_folds,
                    incumbent_fold_metrics,
                    target_name=candidate["target"],
                    ridge_lambda=candidate["ridge_lambda"],
                    backend=args.backend,
                    max_rows=min(candidate["sample_rows"], args.walk_forward_max_rows),
                    seed=candidate["sample_seed"],
                    chunk_size=args.chunk_size,
                    min_6m_delta=args.walk_forward_min_6m_delta,
                    max_horizon_degradation=args.walk_forward_max_horizon_degradation,
                )
                mean_deltas = walk_forward_result.get("mean_deltas", {})
                mean_6m_delta = (
                    float(mean_deltas.get("6m", 0.0)) if isinstance(mean_deltas, dict) else 0.0
                )
                _log(args.report_path, {
                    "event": "walk_forward",
                    "target": candidate["target"],
                    "ridge_lambda": candidate["ridge_lambda"],
                    "sample_rows": candidate["sample_rows"],
                    "sample_seed": candidate["sample_seed"],
                    "accepted": walk_forward_result["accepted"],
                    "reason": walk_forward_result["reason"],
                    "mean_deltas": walk_forward_result["mean_deltas"],
                    "median_deltas": walk_forward_result["median_deltas"],
                    "min_deltas": walk_forward_result["min_deltas"],
                })
                if not walk_forward_result["accepted"]:
                    _print_step(
                        f"iteration {iteration_no}: rejected by walk-forward "
                        f"{candidate['target']} lambda={candidate['ridge_lambda']:.1f} "
                        f"mean Δ6m={mean_6m_delta:+.6f} "
                        f"({walk_forward_result['reason']})"
                    )
                    continue
                _print_step(
                    f"iteration {iteration_no}: walk-forward accepted "
                    f"{candidate['target']} lambda={candidate['ridge_lambda']:.1f} "
                    f"mean Δ6m={mean_6m_delta:+.6f}"
                )

            extra_metadata = {
                "backend": candidate["backend"],
                "target": candidate["target"],
                "ridge_lambda": candidate["ridge_lambda"],
                "sample_rows": candidate["sample_rows"],
                "sample_seed": candidate["sample_seed"],
                "sample_metrics": candidate["sample_metrics"],
            }
            if walk_forward_result is not None:
                extra_metadata["walk_forward"] = walk_forward_result
            payload = _candidate_payload(
                model=candidate["model"],
                priors=priors,
                baseline_metrics=baseline_metrics,
                metrics=metrics,
                snapshots_path=args.snapshots_path,
                extra_metadata=extra_metadata,
            )
            full_candidates.append({
                "payload": payload,
                "models": _linear_payloads(payload),
                "predictions": predictions,
                "name": (
                    f"{candidate['target']}/lambda={candidate['ridge_lambda']}/"
                    f"rows={candidate['sample_rows']}/seed={candidate['sample_seed']}"
                ),
                "rho_6m": full_rho,
            })
            full_candidates.sort(key=lambda item: item["rho_6m"], reverse=True)
            full_candidates = full_candidates[:8]
            _log(args.report_path, {
                "event": "full",
                "target": candidate["target"],
                "ridge_lambda": candidate["ridge_lambda"],
                "sample_rows": candidate["sample_rows"],
                "sample_seed": candidate["sample_seed"],
                "rho_1m": metrics["1m"]["spearman_rho"],
                "rho_3m": metrics["3m"]["spearman_rho"],
                "rho_6m": full_rho,
                "best_rho_6m": best_rho,
            })
            if full_rho > best_rho:
                best_rho = full_rho
                best_payload = payload
                _print_step(
                    f"iteration {iteration_no}: staged single candidate "
                    f"{candidate['target']} lambda={candidate['ridge_lambda']:.1f} "
                    f"6m rho={full_rho:.6f}"
                )
                _log(args.report_path, {"event": "stage_single", "search_rho_6m": best_rho})

            ensemble_pool = full_candidates[:5]
            if ensemble_pool:
                if current_predictions is None:
                    current_predictions = predict_ml_ranker_payload(
                        search_snapshots,
                        current_eval_payload,
                        chunk_size=args.chunk_size,
                    )
                current_item = {
                    "models": current_linear_models,
                    "predictions": current_predictions,
                    "name": "current",
                }
                new_item = full_candidates[0]
                for item in full_candidates:
                    if item["payload"] is payload:
                        new_item = item
                        break
                combos = [
                    (current_item, new_item),
                    (current_item, ensemble_pool[0]),
                ]
                if len(ensemble_pool) > 1:
                    combos.append((current_item, ensemble_pool[0], ensemble_pool[1]))
                    combos.append((ensemble_pool[0], ensemble_pool[1]))
                if new_item is not ensemble_pool[0]:
                    combos.append((current_item, ensemble_pool[0], new_item))

                seen_members = set()
                for combo in combos:
                    names = ",".join(item["name"] for item in combo)
                    if names in seen_members:
                        continue
                    seen_members.add(names)
                    predictions = np.mean([item["predictions"] for item in combo], axis=0)
                    metrics = _evaluate_predictions(search_snapshots, predictions)
                    ensemble_rho = float(metrics["6m"]["spearman_rho"])
                    _log(args.report_path, {
                        "event": "ensemble",
                        "members": names,
                        "rho_1m": metrics["1m"]["spearman_rho"],
                        "rho_3m": metrics["3m"]["spearman_rho"],
                        "rho_6m": ensemble_rho,
                        "best_rho_6m": best_rho,
                    })
                    if ensemble_rho > best_rho and math.isfinite(ensemble_rho):
                        models = []
                        for item in combo:
                            models.extend(item["models"])
                        best_rho = ensemble_rho
                        best_payload = _ensemble_payload(
                            models=models,
                            priors=priors,
                            baseline_metrics=baseline_metrics,
                            metrics=metrics,
                            snapshots_path=args.snapshots_path,
                            extra_metadata={"members": names},
                        )
                        _print_step(
                            f"iteration {iteration_no}: staged ensemble "
                            f"{names} 6m rho={ensemble_rho:.6f}"
                        )
                        _log(
                            args.report_path,
                            {"event": "stage_ensemble", "search_rho_6m": best_rho, "members": names},
                        )

    final_path = Path("/tmp/valueinvestor_ml_search_best.json")
    final_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_step(f"search complete: best staged payload written to {final_path}")
    if args.promote and best_payload is not current_payload and best_rho > current_search_rho:
        candidate_gate_metrics = evaluate_ml_ranker_payload(
            gate_snapshots,
            best_payload,
            chunk_size=args.chunk_size,
        )
        _print_step("promotion requested: evaluating finalist on untouched holdout")
        gate_result = evaluate_promotion_gate(
            candidate_metrics=candidate_gate_metrics,
            incumbent_metrics=current_gate_metrics,
            manifest=split.manifest,
            config=gate_config,
        )
        append_gate_ledger(
            gate_result,
            extra={
                "candidate": "search_ml_ranker",
                "model_path": str(args.model_path),
                "report_path": str(args.report_path),
                "best_payload": str(final_path),
            },
        )
        _log(args.report_path, {
            "event": "promotion_gate",
            "accepted": gate_result.accepted,
            "reason": gate_result.reason,
            "delta_1m": gate_result.deltas["1m"],
            "delta_3m": gate_result.deltas["3m"],
            "delta_6m": gate_result.deltas["6m"],
            "weighted_utility": gate_result.weighted_utility,
        })
        if gate_result.accepted:
            _print_step("promotion gate passed: writing production artifact")
            best_payload["metadata"].update({
                "baseline_metrics": current_gate_metrics,
                "metrics": candidate_gate_metrics,
                "promotion_gate": gate_result.to_record(),
                "promotion_gate_manifest": split.manifest,
                "final_rho_improvement": _improvement_ratio(
                    float(candidate_gate_metrics["6m"]["spearman_rho"]),
                    float(current_gate_metrics["6m"]["spearman_rho"]),
                ),
            })
            _promote(best_payload, model_path=args.model_path, best_meta_path=args.best_meta_path)
            _log(args.report_path, {"event": "promote", "gate_rho_6m": candidate_gate_metrics["6m"]["spearman_rho"]})
        else:
            _print_step(f"promotion gate failed: {gate_result.reason}")
    _log(args.report_path, {
        "event": "finish",
        "initial_gate_rho_6m": current_rho,
        "initial_search_rho_6m": current_search_rho,
        "best_search_rho_6m": best_rho,
        "best_payload": str(final_path),
    })
    _print_step(
        "finished: "
        f"initial search 6m rho={current_search_rho:.6f}, "
        f"best staged 6m rho={best_rho:.6f}, "
        f"gate 6m rho={current_rho:.6f}"
    )


if __name__ == "__main__":
    main()
