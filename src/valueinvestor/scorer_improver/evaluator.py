"""Backtest engine — evaluate a scorer against the ground-truth dataset.

Loads the ground truth (scorer outputs + forward returns) and computes
Spearman rank correlation between composite_score and the target forward
return column as the primary evaluation metric.

Supported horizons: "1m" (30 days), "3m" (90 days), "6m" (126 days, default).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scipy import stats
from valueinvestor.data.models import (
    Company,
    Financials,
    Market,
    ScreeningResult,
    ValuationMetrics,
)

from valueinvestor.scorer_improver.ground_truth import (
    GROUND_TRUTH_FILE,
    ensure_ground_truth_ready,
    load_ground_truth_cached,
)

logger = logging.getLogger(__name__)

SnapshotTemplates = list[tuple[list[int], list[ScreeningResult]]]

HORIZON_TO_COLUMN: Dict[str, str] = {
    "1m": "forward_return_1m",
    "3m": "forward_return_3m",
    "6m": "forward_return_6m",
}

# Module names for each horizon's scorer
HORIZON_TO_MODULE: Dict[str, str] = {
    "1m": "valueinvestor.screener.scorer",
    "3m": "valueinvestor.screener.scorer",
    "6m": "valueinvestor.screener.scorer",
}


def _load_ground_truth(
    path: Optional[Path] = None,
    *,
    allow_legacy_schema: bool = True,
) -> pd.DataFrame:
    """Load the ground-truth dataset from Parquet."""
    path = path or GROUND_TRUTH_FILE
    ensure_ground_truth_ready(path, allow_legacy_schema=allow_legacy_schema)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. "
            "Run `valueinvestor improve-scorer --fetch-only` first."
        )
    return load_ground_truth_cached(path)


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


def _sanitize_score(score: float) -> float:
    """Replace None, NaN, Inf with 50.0 neutral score."""
    import math

    if score is None:
        return 50.0
    try:
        if math.isnan(score) or math.isinf(score):
            return 50.0
    except TypeError:
        return 50.0
    return max(0.0, min(100.0, float(score)))


def _row_value(row, col):
    if isinstance(row, pd.Series):
        return row.get(col)
    return getattr(row, col, None)


def _screening_result_from_row(row) -> ScreeningResult:
    ticker = str(_row_value(row, "ticker"))
    market = Market.HK_SHARE if ticker.endswith(".HK") else Market.A_SHARE

    company = Company(ticker=ticker, name=ticker, market=market)
    financials = Financials(
        ticker=ticker,
        period="snapshot",
        **{field: _safe(row, field) for field in _FINANCIAL_FIELDS},
    )
    valuation = ValuationMetrics(
        ticker=ticker,
        date=str(_row_value(row, "snapshot_date") or ""),
        **{
            field: _safe(row, "close" if field == "price" else field)
            for field in _VALUATION_FIELDS
        },
    )
    return ScreeningResult(company=company, financials=financials, valuation=valuation)


def _build_snapshot_templates(gt_df: pd.DataFrame) -> SnapshotTemplates:
    if "snapshot_date" in gt_df.columns:
        groups = gt_df.groupby("snapshot_date", sort=False)
    else:
        groups = [(None, gt_df)]

    templates: SnapshotTemplates = []
    for _snap_date, group in groups:
        snapshot_indexes = list(group.index)
        snapshot_results = [
            _screening_result_from_row(row)
            for row in group.itertuples(index=False)
        ]
        templates.append((snapshot_indexes, snapshot_results))
    return templates


def _clone_snapshot_results(
    snapshot_indexes: list[int],
    snapshot_templates: list[ScreeningResult],
) -> tuple[list[int], list[ScreeningResult]]:
    return snapshot_indexes, [template.model_copy(deep=False) for template in snapshot_templates]


def _load_scorer_module(
    scorer_module: str,
    *,
    scorer_path: Optional[Path] = None,
):
    if scorer_path is not None:
        existing_module = sys.modules.pop(scorer_module, None)
        cached_path = getattr(existing_module, "__cached__", None)
        if cached_path:
            try:
                os.remove(cached_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Could not remove scorer bytecode cache: %s", cached_path)
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(scorer_module, scorer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load scorer module from {scorer_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[scorer_module] = module
        spec.loader.exec_module(module)
        return module

    existing_module = sys.modules.get(scorer_module)
    cached_path = getattr(existing_module, "__cached__", None)
    if cached_path:
        try:
            os.remove(cached_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove scorer bytecode cache: %s", cached_path)
    importlib.invalidate_caches()
    if existing_module is not None:
        return importlib.reload(existing_module)
    return importlib.import_module(scorer_module)


def _rescore_with_current_scorer(
    gt_df: pd.DataFrame,
    scorer_module: str = "valueinvestor.screener.scorer",
    *,
    scorer_path: Optional[Path] = None,
    snapshot_templates: Optional[SnapshotTemplates] = None,
) -> tuple:
    """Re-run the specified scorer module on ground-truth feature data.

    This dynamically reloads the scorer module to pick up any changes
    the agent has made.

    Returns (DataFrame, error_count, first_error).
    """
    # Force a fresh import. The improvement loop rewrites scorer.py many times
    # per second, and timestamp-based .pyc files can otherwise preserve a stale
    # scorer even after the source has been restored.
    loaded_module = _load_scorer_module(scorer_module, scorer_path=scorer_path)
    MultiFactorScorer = loaded_module.MultiFactorScorer

    scorer = MultiFactorScorer()
    new_scores = pd.Series(index=gt_df.index, dtype="float64")
    error_count = 0
    first_error: Optional[str] = None
    first_error_ticker: Optional[str] = None

    groups = snapshot_templates or _build_snapshot_templates(gt_df)
    for snapshot_indexes, template_results in groups:
        snapshot_indexes, snapshot_results = _clone_snapshot_results(snapshot_indexes, template_results)
        index_by_id = {id(sr): idx for idx, sr in zip(snapshot_indexes, snapshot_results)}

        try:
            scorer.rank(snapshot_results)
            for sr in snapshot_results:
                idx = index_by_id[id(sr)]
                new_scores.at[idx] = _sanitize_score(sr.composite_score)
        except Exception as e:
            error_count += len(snapshot_results)
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
                first_error_ticker = snapshot_results[0].company.ticker if snapshot_results else None

    if error_count > 0:
        logger.warning(
            "Scorer crashed on %d/%d stocks (%.1f%%). First error: %s on %s",
            error_count, len(gt_df), 100 * error_count / len(gt_df),
            first_error, first_error_ticker,
        )

    gt_df["new_composite_score"] = new_scores
    return gt_df, error_count, first_error


def _safe(row, col) -> Optional[float]:
    val = _row_value(row, col)
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def evaluate_scorer(
    use_original_scores: bool = False,
    horizon: str = "6m",
    scorer_module: Optional[str] = None,
    ground_truth_path: Optional[Path] = None,
    allow_legacy_schema: bool = True,
) -> Dict[str, float]:
    """Evaluate the scorer against ground truth for a specific time horizon.

    Parameters
    ----------
    use_original_scores : bool
        If True, use the composite_score column already in ground truth.
        If False (default), re-run the scorer code to get fresh scores.
    horizon : str
        One of "1m", "3m", "6m" (default). Selects the forward-return column.
    scorer_module : str, optional
        Python module name to load. Defaults to the standard module for
        the given horizon (e.g. ``valueinvestor.screener.scorer_1m`` for "1m").

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
    return_col = HORIZON_TO_COLUMN.get(horizon, "forward_return_6m")
    if scorer_module is None:
        scorer_module = HORIZON_TO_MODULE.get(horizon, "valueinvestor.screener.scorer")

    gt_df = _load_ground_truth(ground_truth_path, allow_legacy_schema=allow_legacy_schema)

    # Ensure the required return column exists
    if return_col not in gt_df.columns:
        logger.warning(
            "Column %s not in ground truth — run augment_ground_truth_with_horizons() first.",
            return_col,
        )
        return {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
            "error_rate": 0.0,
            "error_count": 0,
        }

    error_count = 0
    first_error = None
    if use_original_scores:
        score_col = "composite_score"
    else:
        gt_df, error_count, first_error = _rescore_with_current_scorer(gt_df, scorer_module=scorer_module)
        score_col = "new_composite_score"
        gt_df = gt_df.dropna(subset=[score_col])

    total_stocks = len(gt_df) if use_original_scores else len(gt_df)
    error_rate = error_count / total_stocks if total_stocks > 0 else 0.0

    if gt_df.empty:
        return {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
            "error_rate": error_rate,
            "error_count": error_count,
        }

    # Compute per-snapshot Spearman correlation
    rhos = []
    pvals = []
    hit_rates = []
    excess_returns = []

    for snap_date, group in gt_df.groupby("snapshot_date"):
        valid = group.dropna(subset=[score_col, return_col])
        if len(valid) < 10:
            continue
        if valid[score_col].nunique() < 2 or valid[return_col].nunique() < 2:
            continue

        rho, p = stats.spearmanr(valid[score_col], valid[return_col])
        if pd.isna(rho):
            continue
        rhos.append(rho)
        pvals.append(1.0 if pd.isna(p) else p)

        # Hit rate: % of top-20 that beat median
        median_ret = valid[return_col].median()
        top20 = valid.nlargest(min(20, len(valid)), score_col)
        hit = (top20[return_col] > median_ret).mean()
        hit_rates.append(hit)

        # Mean excess return of top-20
        excess = top20[return_col].mean() - median_ret
        excess_returns.append(excess)

    if not rhos:
        return {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
            "error_rate": error_rate,
            "error_count": error_count,
        }

    result = {
        "spearman_rho": sum(rhos) / len(rhos),
        "spearman_p": sum(pvals) / len(pvals),
        "hit_rate_top20": sum(hit_rates) / len(hit_rates),
        "mean_excess_return": sum(excess_returns) / len(excess_returns),
        "n_snapshots": len(rhos),
        "n_stocks_per_snapshot": len(gt_df) / max(gt_df["snapshot_date"].nunique(), 1),
        "error_rate": error_rate,
        "error_count": error_count,
    }

    logger.info(
        "Evaluation (%s): ρ=%.4f (p=%.4f), hit_rate=%.1f%%, excess_ret=%.2f%%, "
        "%d snapshots, ~%.0f stocks/snapshot (errors: %d, %.1f%%)",
        horizon,
        result["spearman_rho"],
        result["spearman_p"],
        result["hit_rate_top20"] * 100,
        result["mean_excess_return"] * 100,
        result["n_snapshots"],
        result["n_stocks_per_snapshot"],
        error_count,
        error_rate * 100,
    )
    return result


def evaluate_scorer_all_horizons(
    use_original_scores: bool = False,
    ground_truth_path: Optional[Path] = None,
    allow_legacy_schema: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate each horizon's scorer against its respective ground-truth column.

    Returns a dict keyed by horizon ("1m", "3m", "6m") with the same metric
    dict as :func:`evaluate_scorer`.
    """
    results: Dict[str, Dict[str, float]] = {}
    for horizon in ("1m", "3m", "6m"):
        results[horizon] = evaluate_scorer(
            use_original_scores=use_original_scores,
            horizon=horizon,
            ground_truth_path=ground_truth_path,
            allow_legacy_schema=allow_legacy_schema,
        )
    return results


def _compute_snapshot_metrics_for_column(
    gt_df: pd.DataFrame,
    score_col: str,
    return_col: str,
) -> Dict[str, float]:
    """Compute per-snapshot Spearman ρ, hit-rate, and excess return.

    *gt_df* must already have the score and return columns populated.

    Returns the same metric dict as :func:`evaluate_scorer`.
    """
    rhos = []
    pvals = []
    hit_rates = []
    excess_returns = []

    for snap_date, group in gt_df.groupby("snapshot_date"):
        valid = group.dropna(subset=[score_col, return_col])
        if len(valid) < 10:
            continue
        if valid[score_col].nunique() < 2 or valid[return_col].nunique() < 2:
            continue

        rho, p = stats.spearmanr(valid[score_col], valid[return_col])
        if pd.isna(rho):
            continue
        rhos.append(rho)
        pvals.append(1.0 if pd.isna(p) else p)

        median_ret = valid[return_col].median()
        top20 = valid.nlargest(min(20, len(valid)), score_col)
        hit = (top20[return_col] > median_ret).mean()
        hit_rates.append(hit)

        excess = top20[return_col].mean() - median_ret
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
        "Eval %s: ρ=%.4f (p=%.4f), hit_rate=%.1f%%, excess_ret=%.2f%%, "
        "%d snapshots, ~%.0f stocks/snapshot",
        return_col,
        result["spearman_rho"],
        result["spearman_p"],
        result["hit_rate_top20"] * 100,
        result["mean_excess_return"] * 100,
        result["n_snapshots"],
        result["n_stocks_per_snapshot"],
    )
    return result


def _empty_metrics() -> Dict[str, float]:
    return {
        "spearman_rho": 0.0,
        "spearman_p": 1.0,
        "hit_rate_top20": 0.0,
        "mean_excess_return": 0.0,
        "n_snapshots": 0,
        "n_stocks_per_snapshot": 0.0,
    }


def _compute_snapshot_metrics_for_all_horizons(
    gt_df: pd.DataFrame,
    *,
    score_col: str,
) -> Dict[str, Dict[str, float]]:
    metrics_by_horizon = {
        h: {
            "rhos": [],
            "pvals": [],
            "hit_rates": [],
            "excess_returns": [],
        }
        for h in HORIZON_TO_COLUMN
    }
    grouped = gt_df.groupby("snapshot_date") if "snapshot_date" in gt_df.columns else [(None, gt_df)]

    for _snap_date, group in grouped:
        for horizon, return_col in HORIZON_TO_COLUMN.items():
            if return_col not in group.columns:
                continue
            valid = group.dropna(subset=[score_col, return_col])
            if len(valid) < 10:
                continue
            if valid[score_col].nunique() < 2 or valid[return_col].nunique() < 2:
                continue

            rho, p = stats.spearmanr(valid[score_col], valid[return_col])
            if pd.isna(rho):
                continue

            bucket = metrics_by_horizon[horizon]
            bucket["rhos"].append(rho)
            bucket["pvals"].append(1.0 if pd.isna(p) else p)

            median_ret = valid[return_col].median()
            top20 = valid.nlargest(min(20, len(valid)), score_col)
            bucket["hit_rates"].append((top20[return_col] > median_ret).mean())
            bucket["excess_returns"].append(top20[return_col].mean() - median_ret)

    snapshot_count = max(gt_df["snapshot_date"].nunique(), 1) if "snapshot_date" in gt_df.columns else 1
    results: Dict[str, Dict[str, float]] = {}
    for horizon, return_col in HORIZON_TO_COLUMN.items():
        bucket = metrics_by_horizon[horizon]
        if not bucket["rhos"]:
            results[horizon] = _empty_metrics()
            continue

        result = {
            "spearman_rho": sum(bucket["rhos"]) / len(bucket["rhos"]),
            "spearman_p": sum(bucket["pvals"]) / len(bucket["pvals"]),
            "hit_rate_top20": sum(bucket["hit_rates"]) / len(bucket["hit_rates"]),
            "mean_excess_return": sum(bucket["excess_returns"]) / len(bucket["excess_returns"]),
            "n_snapshots": len(bucket["rhos"]),
            "n_stocks_per_snapshot": len(gt_df) / snapshot_count,
        }
        logger.info(
            "Eval %s: ρ=%.4f (p=%.4f), hit_rate=%.1f%%, excess_ret=%.2f%%, "
            "%d snapshots, ~%.0f stocks/snapshot",
            return_col,
            result["spearman_rho"],
            result["spearman_p"],
            result["hit_rate_top20"] * 100,
            result["mean_excess_return"] * 100,
            result["n_snapshots"],
            result["n_stocks_per_snapshot"],
        )
        results[horizon] = result
    return results


def _sample_by_snapshot(gt_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Return a deterministic snapshot-stratified sample.

    The full metric is cross-sectional within each snapshot. A pooled random
    sample changes the objective, so quick-eval keeps the same per-snapshot
    shape while reducing rows.
    """
    if gt_df.empty or len(gt_df) <= sample_size or "snapshot_date" not in gt_df.columns:
        return gt_df

    groups = list(gt_df.groupby("snapshot_date", sort=True))
    if not groups:
        return gt_df.head(sample_size)

    rows_per_snapshot = max(10, sample_size // len(groups))
    sampled = []
    for idx, (_, group) in enumerate(groups):
        n = min(len(group), rows_per_snapshot)
        if n <= 0:
            continue
        sampled.append(group.sample(n=n, random_state=idx))

    if not sampled:
        return gt_df.head(sample_size)

    sample = pd.concat(sampled, ignore_index=True)
    if len(sample) <= sample_size:
        return sample
    return sample.sample(n=sample_size, random_state=0)


def _evaluate_scorer_all_targets_frame(
    gt_df: pd.DataFrame,
    *,
    scorer_module: str,
    scorer_path: Optional[Path] = None,
    snapshot_templates: Optional[SnapshotTemplates] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all targets against an already-loaded ground-truth frame."""
    gt_df, error_count, first_error = _rescore_with_current_scorer(
        gt_df,
        scorer_module=scorer_module,
        scorer_path=scorer_path,
        snapshot_templates=snapshot_templates,
    )
    total_stocks = len(gt_df)
    error_rate = error_count / total_stocks if total_stocks > 0 else 0.0
    score_col = "new_composite_score"
    gt_df = gt_df.dropna(subset=[score_col])

    if gt_df.empty:
        empty = {
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "hit_rate_top20": 0.0,
            "mean_excess_return": 0.0,
            "n_snapshots": 0,
            "n_stocks_per_snapshot": 0.0,
            "error_rate": error_rate,
            "error_count": error_count,
        }
        return {"1m": dict(empty), "3m": dict(empty), "6m": dict(empty)}

    results: Dict[str, Dict[str, float]] = {}
    for horizon, metrics in _compute_snapshot_metrics_for_all_horizons(
        gt_df,
        score_col=score_col,
    ).items():
        metrics["error_rate"] = error_rate
        metrics["error_count"] = error_count
        results[horizon] = metrics

    results["error_rate"] = error_rate
    results["error_count"] = error_count

    logger.info(
        "All-targets evaluation — 1m: ρ=%.4f, 3m: ρ=%.4f, 6m: ρ=%.4f",
        results["1m"]["spearman_rho"],
        results["3m"]["spearman_rho"],
        results["6m"]["spearman_rho"],
    )
    return results


def _quick_evaluate_frame(
    gt_sample: pd.DataFrame,
    *,
    scorer_module: str,
    scorer_path: Optional[Path] = None,
    snapshot_templates: Optional[SnapshotTemplates] = None,
) -> Dict[str, float]:
    """Quick-evaluate an already-built deterministic sample."""
    gt_sample, error_count, first_error = _rescore_with_current_scorer(
        gt_sample,
        scorer_module=scorer_module,
        scorer_path=scorer_path,
        snapshot_templates=snapshot_templates,
    )
    total_sample = len(gt_sample)
    error_rate = error_count / total_sample if total_sample > 0 else 0.0
    score_col = "new_composite_score"
    gt_sample = gt_sample.dropna(subset=[score_col])

    result: Dict[str, float] = {}
    for horizon, metrics in _compute_snapshot_metrics_for_all_horizons(
        gt_sample,
        score_col=score_col,
    ).items():
        result[horizon] = float(metrics["spearman_rho"])

    result["error_rate"] = error_rate
    result["error_count"] = float(error_count)

    logger.debug(
        "Quick evaluation — 1m: ρ=%.4f, 3m: ρ=%.4f, 6m: ρ=%.4f (errors: %d, %.1f%%)",
        result["1m"], result["3m"], result["6m"], error_count, error_rate * 100,
    )
    return result


class ScorerEvaluationContext:
    """Per-run cache for scorer evaluation data and deterministic quick samples."""

    def __init__(
        self,
        *,
        ground_truth_path: Optional[Path] = None,
        allow_legacy_schema: bool = True,
        quick_sample_size: int = 1000,
    ) -> None:
        self.ground_truth_path = ground_truth_path
        self.allow_legacy_schema = allow_legacy_schema
        self.quick_sample_size = quick_sample_size
        self._ground_truth = _load_ground_truth(
            ground_truth_path,
            allow_legacy_schema=allow_legacy_schema,
        )
        self._snapshot_templates = _build_snapshot_templates(self._ground_truth)
        self._quick_samples: dict[int, tuple[pd.DataFrame, SnapshotTemplates]] = {}

    def evaluate_all_targets(
        self,
        scorer_module: str = "valueinvestor.screener.scorer",
        scorer_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the full cached ground truth against all horizons."""
        return _evaluate_scorer_all_targets_frame(
            self._ground_truth.copy(deep=False),
            scorer_module=scorer_module,
            scorer_path=scorer_path,
            snapshot_templates=self._snapshot_templates,
        )

    def quick_evaluate(
        self,
        scorer_module: str = "valueinvestor.screener.scorer",
        sample_size: Optional[int] = None,
        scorer_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate a cached deterministic sample against all horizons."""
        sample_size = sample_size or self.quick_sample_size
        if sample_size not in self._quick_samples:
            sample = _sample_by_snapshot(self._ground_truth, sample_size)
            self._quick_samples[sample_size] = (
                sample,
                _build_snapshot_templates(sample),
            )
        sample, snapshot_templates = self._quick_samples[sample_size]
        return _quick_evaluate_frame(
            sample.copy(deep=False),
            scorer_module=scorer_module,
            scorer_path=scorer_path,
            snapshot_templates=snapshot_templates,
        )


def evaluate_scorer_all_targets(
    scorer_module: str = "valueinvestor.screener.scorer",
    scorer_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    allow_legacy_schema: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a **single** scorer against all three forward-return horizons.

    Rescores the ground truth once, then computes Spearman ρ for 1m, 3m,
    and 6m return columns.  More efficient than calling :func:`evaluate_scorer`
    three times.

    Returns
    -------
    dict keyed by horizon ("1m", "3m", "6m") with metric dicts.
    """
    context = ScorerEvaluationContext(
        ground_truth_path=ground_truth_path,
        allow_legacy_schema=allow_legacy_schema,
    )
    return context.evaluate_all_targets(scorer_module=scorer_module, scorer_path=scorer_path)


def quick_evaluate(
    scorer_module: str = "valueinvestor.screener.scorer",
    scorer_path: Optional[Path] = None,
    sample_size: int = 1000,
    ground_truth_path: Optional[Path] = None,
    allow_legacy_schema: bool = True,
) -> Dict[str, float]:
    """Evaluate a scorer on a random subset of the ground truth.

    Fast rejection check — runs in ~0.02s vs ~0.5s for the full evaluation.
    Returns spearman_rho per horizon plus ``error_rate`` and ``error_count``.
    """
    context = ScorerEvaluationContext(
        ground_truth_path=ground_truth_path,
        allow_legacy_schema=allow_legacy_schema,
        quick_sample_size=sample_size,
    )
    return context.quick_evaluate(
        scorer_module=scorer_module,
        scorer_path=scorer_path,
        sample_size=sample_size,
    )
