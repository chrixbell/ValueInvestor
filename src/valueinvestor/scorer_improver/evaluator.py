"""Backtest engine — evaluate a scorer against the ground-truth dataset.

Loads the ground truth (scorer outputs + 6-month forward returns) and computes
Spearman rank correlation between composite_score and forward_return_6m as the
primary evaluation metric.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Dict, Optional

import pandas as pd
from scipy import stats

from valueinvestor.scorer_improver.ground_truth import GROUND_TRUTH_FILE

logger = logging.getLogger(__name__)


def _load_ground_truth() -> pd.DataFrame:
    """Load the ground-truth dataset from Parquet."""
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {GROUND_TRUTH_FILE}. "
            "Run `valueinvestor improve-scorer --fetch-only` first."
        )
    return pd.read_parquet(str(GROUND_TRUTH_FILE))


_FINANCIAL_FIELDS = (
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cash_flow",
    "free_cash_flow",
    "gross_margin",
    "net_margin",
    "roe",
    "roa",
    "debt_to_equity",
    "current_ratio",
)

_VALUATION_FIELDS = (
    "price",
    "pe_ratio",
    "pe_forward",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "dividend_yield",
    "ev_to_ebitda",
    "market_cap_rmb",
)


def _rescore_with_current_scorer(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run the current MultiFactorScorer on ground-truth feature data.

    This dynamically reloads the scorer module to pick up any changes
    the agent has made.
    """
    # Force-reload the scorer module to pick up live changes
    mod_name = "valueinvestor.screener.scorer"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

    from valueinvestor.data.models import (
        Company,
        Financials,
        Market,
        ScreeningResult,
        ValuationMetrics,
    )
    from valueinvestor.screener.scorer import MultiFactorScorer

    scorer = MultiFactorScorer()
    new_scores = []

    for row in gt_df.to_dict("records"):
        ticker = str(row["ticker"])
        market = Market.HK_SHARE if ticker.endswith(".HK") else Market.A_SHARE

        company = Company(ticker=ticker, name=ticker, market=market)
        financials = Financials(
            ticker=ticker,
            period="snapshot",
            **{field: _safe(row, field) for field in _FINANCIAL_FIELDS},
        )
        valuation = ValuationMetrics(
            ticker=ticker,
            date=str(row.get("snapshot_date", "")),
            **{
                field: _safe(row, "close" if field == "price" else field)
                for field in _VALUATION_FIELDS
            },
        )
        sr = ScreeningResult(
            company=company,
            financials=financials,
            valuation=valuation,
        )
        try:
            scorer.score(sr)
            new_scores.append(sr.composite_score)
        except Exception:
            new_scores.append(None)

    gt_df = gt_df.copy()
    gt_df["new_composite_score"] = new_scores
    return gt_df


def _safe(row, col) -> Optional[float]:
    val = row.get(col)
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def evaluate_scorer(
    use_original_scores: bool = False,
) -> Dict[str, float]:
    """Evaluate the current scorer against ground truth.

    Parameters
    ----------
    use_original_scores : bool
        If True, use the composite_score column already in ground truth
        (i.e., the scorer as it was when ground truth was built).
        If False, re-run the current scorer.py code to get fresh scores.

    Returns
    -------
    dict with keys:
        - spearman_rho: mean Spearman ρ across all snapshots
        - spearman_p: mean p-value
        - hit_rate_top20: % of top-20 scored stocks that beat the median return
        - mean_excess_return: mean return of top-20 minus median return
        - n_snapshots: number of snapshot dates evaluated
        - n_stocks_per_snapshot: average stocks per snapshot
    """
    gt_df = _load_ground_truth()

    if use_original_scores:
        score_col = "composite_score"
    else:
        gt_df = _rescore_with_current_scorer(gt_df)
        score_col = "new_composite_score"
        gt_df = gt_df.dropna(subset=[score_col])

    if gt_df.empty:
        return {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
        }

    # Compute per-snapshot Spearman correlation
    rhos = []
    pvals = []
    hit_rates = []
    excess_returns = []

    for snap_date, group in gt_df.groupby("snapshot_date"):
        # Need at least 10 stocks for meaningful correlation
        valid = group.dropna(subset=[score_col, "forward_return_6m"])
        if len(valid) < 10:
            continue
        if valid[score_col].nunique() < 2 or valid["forward_return_6m"].nunique() < 2:
            continue

        rho, p = stats.spearmanr(valid[score_col], valid["forward_return_6m"])
        if pd.isna(rho):
            continue
        rhos.append(rho)
        pvals.append(1.0 if pd.isna(p) else p)

        # Hit rate: % of top-20 that beat median
        median_ret = valid["forward_return_6m"].median()
        top20 = valid.nlargest(min(20, len(valid)), score_col)
        hit = (top20["forward_return_6m"] > median_ret).mean()
        hit_rates.append(hit)

        # Mean excess return of top-20
        excess = top20["forward_return_6m"].mean() - median_ret
        excess_returns.append(excess)

    if not rhos:
        return {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
        }

    result = {
        "spearman_rho": sum(rhos) / len(rhos),
        "spearman_p": sum(pvals) / len(pvals),
        "hit_rate_top20": sum(hit_rates) / len(hit_rates),
        "mean_excess_return": sum(excess_returns) / len(excess_returns),
        "n_snapshots": len(rhos),
        "n_stocks_per_snapshot": len(gt_df) / max(gt_df["snapshot_date"].nunique(), 1),
    }

    logger.info(
        "Evaluation: ρ=%.4f (p=%.4f), hit_rate=%.1f%%, excess_ret=%.2f%%, "
        "%d snapshots, ~%.0f stocks/snapshot",
        result["spearman_rho"],
        result["spearman_p"],
        result["hit_rate_top20"] * 100,
        result["mean_excess_return"] * 100,
        result["n_snapshots"],
        result["n_stocks_per_snapshot"],
    )
    return result
