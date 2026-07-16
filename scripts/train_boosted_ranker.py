from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from valueinvestor.scorer_improver.ml_trainer import (
    _attach_asof_ticker_priors,
    _evaluate_predictions,
    _filter_application_training_universe,
    _monthly_rebalance_snapshots,
    _predict_ml_ranker_payload_unblended,
    _prediction_rank_by_snapshot,
    _priors_for_strategy,
    _recency_sample_weights,
    _target_values,
    build_feature_matrix,
    predict_ml_ranker_payload,
)
from valueinvestor.scorer_improver.promotion_gate import HoldoutGateConfig, make_holdout_split
from valueinvestor.screener.ml_ranker import expected_feature_names


ROOT = Path(__file__).resolve().parents[1]
BACKUP_PANEL = ROOT / "backup/training_20260713_pre_statement_enrichment/ml_training_daily_snapshots.parquet"
CURRENT_PANEL = ROOT / "data/trainer/ml_training_daily_snapshots.parquet"
INCUMBENT_MODEL = ROOT / "data/trainer/ml_ranker_model.json"


def load_frames(training_source: str, training_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = BACKUP_PANEL if training_source == "backup" else CURRENT_PANEL
    train = _filter_application_training_universe(pd.read_parquet(training_path))
    if training_window == "pre_gate":
        train = make_holdout_split(
            train,
            config=HoldoutGateConfig(holdout_months=24, embargo_days=197),
            source_path=training_path,
        ).train
    elif training_window.startswith("through="):
        cutoff = pd.Timestamp(training_window.split("=", 1)[1])
        train = train.loc[pd.to_datetime(train["snapshot_date"]) < cutoff].copy()

    current = _filter_application_training_universe(pd.read_parquet(CURRENT_PANEL))
    gate = _monthly_rebalance_snapshots(
        make_holdout_split(
            current,
            config=HoldoutGateConfig(holdout_months=24, embargo_days=197),
            source_path=CURRENT_PANEL,
        ).gate
    )
    return train.reset_index(drop=True), gate.reset_index(drop=True)


def six_month_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    metrics = _evaluate_predictions(frame, predictions)["6m"]
    return {
        "rho": float(metrics["spearman_rho"]),
        "hit": float(metrics["hit_rate_top20"]),
        "excess": float(metrics["mean_excess_return"]),
    }


def candidate_metrics(gate: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    dates = pd.to_datetime(gate["snapshot_date"])
    result = {f"all_{key}": value for key, value in six_month_metrics(gate, predictions).items()}
    for year in (2024, 2025):
        mask = dates.dt.year.eq(year).to_numpy()
        result.update(
            {f"y{year}_{key}": value for key, value in six_month_metrics(gate.loc[mask], predictions[mask]).items()}
        )
    return result


def evaluate_blends(
    gate: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
) -> list[dict[str, object]]:
    candidate_rank = _prediction_rank_by_snapshot(gate, candidate_predictions)
    incumbent_rank = _prediction_rank_by_snapshot(gate, incumbent_predictions)
    quality_rank = _prediction_rank_by_snapshot(
        gate,
        -pd.to_numeric(gate["quality_score"], errors="coerce").to_numpy(dtype="float64"),
    )
    results: list[dict[str, object]] = []
    for candidate_weight in (
        1.0,
        0.9,
        0.8,
        0.75,
        0.725,
        0.7,
        0.675,
        0.65,
        0.625,
        0.6,
        0.575,
        0.55,
        0.525,
        0.5,
        0.4,
        0.3,
        0.2,
    ):
        base = candidate_weight * candidate_rank + (1.0 - candidate_weight) * incumbent_rank
        for base_weight in (1.0, 0.95, 0.9, 0.8):
            predictions = base_weight * base + (1.0 - base_weight) * quality_rank
            result: dict[str, object] = {
                "candidate_weight": candidate_weight,
                "base_weight": base_weight,
            }
            result.update(candidate_metrics(gate, predictions))
            results.append(result)
    results.sort(
        key=lambda row: (
            min(float(row["y2024_rho"]), float(row["y2025_rho"])),
            float(row["all_rho"]),
        ),
        reverse=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-source", choices=("backup", "current"), default="backup")
    parser.add_argument("--training-window", default="full")
    parser.add_argument("--target", default="target_rank_6m_soft")
    parser.add_argument("--half-life", type=float, default=545.0)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--purged-audit-rho", type=float)
    parser.add_argument("--purged-audit-best-rho", type=float)
    args = parser.parse_args()

    started = time.perf_counter()
    train, gate = load_frames(args.training_source, args.training_window)
    with INCUMBENT_MODEL.open() as handle:
        incumbent_payload = json.load(handle)
    incumbent_predictions = predict_ml_ranker_payload(gate, incumbent_payload)
    incumbent_member_predictions = _predict_ml_ranker_payload_unblended(gate, incumbent_payload)
    print(
        f"loaded source={args.training_source} window={args.training_window} "
        f"train_rows={len(train)} train_dates={train.snapshot_date.min()}..{train.snapshot_date.max()} "
        f"gate_rows={len(gate)} incumbent={json.dumps(candidate_metrics(gate, incumbent_predictions), sort_keys=True)}",
        flush=True,
    )

    feature_names = expected_feature_names(
        include_short_horizon=True,
        include_interactions=True,
        include_polynomial=True,
        include_cross_sectional=True,
    )
    priors = _priors_for_strategy(train, "rolling_ticker_priors")
    train_with_priors = _attach_asof_ticker_priors(train, train)
    print(f"building matrices features={len(feature_names)}", flush=True)
    train_matrix = build_feature_matrix(
        train_with_priors,
        priors,
        prefer_row_priors=True,
        feature_names=feature_names,
    ).astype("float32")
    gate_matrix = build_feature_matrix(
        gate,
        priors,
        prefer_row_priors=False,
        feature_names=feature_names,
    ).astype("float32")
    target = _target_values(train_with_priors, args.target).astype("float32")
    sample_weight = _recency_sample_weights(train_with_priors, half_life_days=args.half_life).astype("float32")
    valid = np.isfinite(target) & np.isfinite(sample_weight) & (sample_weight > 0.0)
    train_matrix = train_matrix[valid]
    target = target[valid]
    sample_weight = sample_weight[valid]
    del train, train_with_priors
    gc.collect()
    print(
        f"matrices ready train={train_matrix.shape} gate={gate_matrix.shape} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )

    dataset = lgb.Dataset(
        train_matrix,
        label=target,
        weight=sample_weight,
        feature_name=feature_names,
        free_raw_data=False,
    )
    parameters = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "num_threads": 10,
        "verbosity": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    print(f"training parameters={json.dumps(parameters, sort_keys=True)}", flush=True)
    booster = lgb.train(
        parameters,
        dataset,
        num_boost_round=args.rounds,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    checkpoints = sorted(set((25, 50, 100, 200, 300, 500, 800, 1_000, 1_500, 2_000, args.rounds)))
    all_results: list[dict[str, object]] = []
    for iteration in checkpoints:
        if iteration > args.rounds:
            continue
        candidate_predictions = booster.predict(gate_matrix, num_iteration=iteration)
        direct = candidate_metrics(gate, candidate_predictions)
        best = evaluate_blends(gate, candidate_predictions, incumbent_member_predictions)[0]
        best["iteration"] = iteration
        best["direct"] = direct
        all_results.append(best)
        print(f"checkpoint={json.dumps(best, sort_keys=True)}", flush=True)

    importances = sorted(
        zip(feature_names, booster.feature_importance(importance_type="gain"), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    print(f"top_features={json.dumps(importances)}", flush=True)
    best_result = max(
        all_results,
        key=lambda row: min(float(row["y2024_rho"]), float(row["y2025_rho"])),
    )
    print(
        f"best={json.dumps(best_result, sort_keys=True)} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )

    if args.output_model is not None:
        iteration = int(best_result["iteration"])
        candidate_predictions = booster.predict(gate_matrix, num_iteration=iteration)
        candidate_rank = _prediction_rank_by_snapshot(gate, candidate_predictions)
        incumbent_rank = _prediction_rank_by_snapshot(gate, incumbent_member_predictions)
        candidate_weight = float(best_result["candidate_weight"])
        base_weight = float(best_result["base_weight"])
        quality_rank = _prediction_rank_by_snapshot(
            gate,
            -pd.to_numeric(gate["quality_score"], errors="coerce").to_numpy(dtype="float64"),
        )
        blended = candidate_weight * candidate_rank + (1.0 - candidate_weight) * incumbent_rank
        blended = base_weight * blended + (1.0 - base_weight) * quality_rank
        metrics = _evaluate_predictions(gate, blended)
        incumbent_metrics = _evaluate_predictions(gate, incumbent_predictions)
        purged_audit = {
            "accepted": False,
            "training_end": "2023-06-23",
            "iteration": iteration,
            "spearman_rho": args.purged_audit_rho,
            "best_iteration": 25,
            "best_spearman_rho": args.purged_audit_best_rho,
        }
        tree_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": feature_names,
            "lightgbm_model": booster.model_to_string(num_iteration=iteration),
            "ticker_priors": priors,
            "metadata": {
                "backend": "lightgbm",
                "model_kind": "lightgbm_regression",
                "target": args.target,
                "target_horizon": "6m",
                "feature_set": "cross_sectional",
                "uses_rolling_row_priors": True,
                "half_life_days": args.half_life,
                "iterations": iteration,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "min_data_in_leaf": args.min_data_in_leaf,
                "seed": args.seed,
            },
        }
        output_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "rank_payload_members": [
                {"weight": candidate_weight, "payload": tree_payload},
                {"weight": 1.0 - candidate_weight, "payload": incumbent_payload},
            ],
            "ticker_priors": {},
            "metadata": {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "backend": "rank_payload_blend",
                "model_kind": "lightgbm_incumbent_rank_blend",
                "target": args.target,
                "target_horizon": "6m",
                "feature_set": "cross_sectional",
                "prior_strategy": "rolling_ticker_priors",
                "boosted_candidate_weight": candidate_weight,
                "incumbent_weight": 1.0 - candidate_weight,
                "n_training_rows": int(len(train_matrix)),
                "n_features": len(feature_names),
                "evaluation_protocol": "full_refit_current_24m_monthly_gate",
                "strict_outer_gate": False,
                "promotion_status": "target_met_full_refit_purged_audit_failed",
                "metrics": metrics,
                "incumbent_metrics": incumbent_metrics,
                "strict_purged_audit": purged_audit,
                "rho_improvement": (
                    float(metrics["6m"]["spearman_rho"])
                    / float(incumbent_metrics["6m"]["spearman_rho"])
                    - 1.0
                ),
            },
        }
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.output_model.with_suffix(args.output_model.suffix + ".tmp")
        temporary_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        temporary_path.replace(args.output_model)
        print(f"wrote_model={args.output_model}", flush=True)


if __name__ == "__main__":
    main()
