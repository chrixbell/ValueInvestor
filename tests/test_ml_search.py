from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_search_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "search_ml_ranker.py"
    spec = importlib.util.spec_from_file_location("search_ml_ranker", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walk_forward_folds_use_past_train_and_embargo() -> None:
    search = _load_search_module()
    df = pd.DataFrame({
        "snapshot_date": pd.date_range("2020-01-01", periods=60, freq="MS"),
        "ticker": ["T"] * 60,
    })

    folds = search._walk_forward_folds(
        df,
        n_folds=2,
        validation_months=6,
        embargo_days=45,
        min_train_snapshots=6,
        min_validation_snapshots=4,
    )

    assert len(folds) == 2
    assert folds[0]["validation_end"] < folds[1]["validation_end"]
    for fold in folds:
        assert fold["train_end_exclusive"] < fold["validation_start"]
        assert (fold["validation_start"] - fold["train_end_exclusive"]).days == 45
        assert fold["train_snapshots"] >= 6
        assert fold["validation_snapshots"] >= 4


def test_walk_forward_decision_rejects_negative_6m_generalization() -> None:
    search = _load_search_module()

    result = search._walk_forward_decision(
        [
            {"1m": 0.01, "3m": 0.01, "6m": -0.02},
            {"1m": 0.01, "3m": 0.01, "6m": 0.00},
        ],
        min_6m_delta=0.001,
        max_horizon_degradation=0.01,
    )

    assert result["accepted"] is False
    assert result["reason"] == "mean 6m delta below walk-forward gate"


def test_walk_forward_decision_accepts_positive_mean_and_median_6m() -> None:
    search = _load_search_module()

    result = search._walk_forward_decision(
        [
            {"1m": 0.00, "3m": 0.00, "6m": 0.004},
            {"1m": 0.01, "3m": 0.01, "6m": 0.002},
        ],
        min_6m_delta=0.001,
        max_horizon_degradation=0.01,
    )

    assert result["accepted"] is True
    assert result["reason"] == "accepted"
    assert result["mean_deltas"]["6m"] == 0.003
