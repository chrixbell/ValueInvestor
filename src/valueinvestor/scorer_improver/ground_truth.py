"""Phase 2 — Build ground-truth dataset with forward returns.

For each rolling quarter across the 3-year history window, creates a snapshot
of each stock's fundamentals and valuation, then computes the 6-month forward
return as the ground-truth label.  The current scorer is also run to produce
baseline scores.

Output: ``data/trainer/ground_truth.parquet``
"""

from __future__ import annotations

import importlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from valueinvestor.scorer_improver.data_prep import TRAINER_DIR

logger = logging.getLogger(__name__)

GROUND_TRUTH_FILE = TRAINER_DIR / "ground_truth.parquet"

# Rolling snapshot interval in days (~quarterly)
SNAPSHOT_INTERVAL_DAYS = 90
# Forward return horizon in days (~6 months)
FORWARD_HORIZON_DAYS = 126  # ~6 trading months


def _compute_forward_returns(
    prices_df: pd.DataFrame,
    horizon_days: int = FORWARD_HORIZON_DAYS,
) -> pd.DataFrame:
    """For each (ticker, date), compute the forward return over *horizon_days*.

    Returns a DataFrame with columns: ticker, date, close, forward_return.
    Forward return is (price[t+horizon] - price[t]) / price[t].
    """
    if prices_df.empty:
        return pd.DataFrame()

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    prices_df = prices_df.sort_values(["ticker", "date"])

    results = []
    for ticker, group in prices_df.groupby("ticker"):
        group = group.set_index("date").sort_index()
        closes = group["close"]
        for i, (dt, price) in enumerate(closes.items()):
            if price is None or price <= 0:
                continue
            # Find the price ~horizon_days later
            target_date = dt + timedelta(days=horizon_days)
            # Find nearest available date within ±10 trading days
            mask = (closes.index >= target_date - timedelta(days=15)) & (
                closes.index <= target_date + timedelta(days=15)
            )
            future_prices = closes[mask]
            if future_prices.empty:
                continue
            # Use the closest date to the target
            closest_idx = (future_prices.index - target_date).map(lambda x: abs(x.days)).argmin()
            future_price = future_prices.iloc[closest_idx]
            if future_price is None or future_price <= 0:
                continue
            forward_return = (future_price - price) / price
            results.append({
                "ticker": ticker,
                "date": dt.date(),
                "close": price,
                "forward_return_6m": forward_return,
            })

    return pd.DataFrame(results)


def _get_snapshot_dates(
    prices_df: pd.DataFrame,
    interval_days: int = SNAPSHOT_INTERVAL_DAYS,
) -> List[date]:
    """Generate quarterly snapshot dates from the available price history."""
    if prices_df.empty:
        return []

    dates = pd.to_datetime(prices_df["date"])
    min_date = dates.min().date()
    max_date = dates.max().date()

    # Last snapshot must be at least FORWARD_HORIZON_DAYS before max_date
    last_snap = max_date - timedelta(days=FORWARD_HORIZON_DAYS + 30)
    if last_snap <= min_date:
        return []

    snapshots = []
    current = min_date + timedelta(days=30)  # skip first month (thin data)
    while current <= last_snap:
        snapshots.append(current)
        current += timedelta(days=interval_days)

    return snapshots


def _find_nearest_price(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: date,
    window_days: int = 10,
) -> Optional[float]:
    """Find the closing price nearest to *target_date* within ±window_days."""
    mask = (
        (prices_df["ticker"] == ticker)
        & (prices_df["date_dt"] >= pd.Timestamp(target_date - timedelta(days=window_days)))
        & (prices_df["date_dt"] <= pd.Timestamp(target_date + timedelta(days=window_days)))
    )
    subset = prices_df.loc[mask]
    if subset.empty:
        return None
    # Nearest by absolute date distance
    dists = (subset["date_dt"] - pd.Timestamp(target_date)).abs()
    return float(subset.loc[dists.idxmin(), "close"])


def _build_snapshot_features(
    snapshot_date: date,
    prices_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature rows for all stocks at a given snapshot date.

    For each stock with available data, produces a row with:
    - ticker, snapshot_date, close_price
    - pe_ratio, pb_ratio, market_cap_rmb (from valuations)
    - roe, net_margin, debt_to_equity (from financials)
    """
    records = []
    tickers = prices_df["ticker"].unique()

    for ticker in tickers:
        price = _find_nearest_price(prices_df, ticker, snapshot_date)
        if price is None or price <= 0:
            continue

        # Get valuation data (use latest available, may not be date-specific)
        val_row = val_df[val_df["ticker"] == ticker]
        pe = pb = mktcap = None
        if not val_row.empty:
            row = val_row.iloc[-1]
            pe = _safe_float(row.get("pe_ratio"))
            pb = _safe_float(row.get("pb_ratio"))
            mktcap = _safe_float(row.get("market_cap_rmb"))

        # Get financial data
        fin_row = fin_df[fin_df["ticker"] == ticker]
        roe = net_margin = debt_to_equity = None
        if not fin_row.empty:
            row = fin_row.iloc[-1]
            roe = _safe_float(row.get("roe"))
            net_margin = _safe_float(row.get("net_margin"))
            debt_to_equity = _safe_float(row.get("debt_to_equity"))

        records.append({
            "ticker": ticker,
            "snapshot_date": snapshot_date,
            "close": price,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "market_cap_rmb": mktcap,
            "roe": roe,
            "net_margin": net_margin,
            "debt_to_equity": debt_to_equity,
        })

    return pd.DataFrame(records)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _score_snapshot(features_df: pd.DataFrame) -> pd.DataFrame:
    """Run the current MultiFactorScorer on a snapshot and add score columns."""
    from valueinvestor.data.models import (
        Company,
        Financials,
        Market,
        ScreeningResult,
        ValuationMetrics,
    )
    from valueinvestor.screener.scorer import MultiFactorScorer

    scorer = MultiFactorScorer()
    scored_records = []

    for _, row in features_df.iterrows():
        ticker = row["ticker"]
        # Determine market from ticker format
        if str(ticker).endswith(".HK"):
            market = Market.HK_SHARE
        else:
            market = Market.A_SHARE

        company = Company(ticker=ticker, name=ticker, market=market)
        financials = Financials(
            ticker=ticker,
            period="snapshot",
            roe=row.get("roe"),
            net_margin=row.get("net_margin"),
            debt_to_equity=row.get("debt_to_equity"),
        )
        valuation = ValuationMetrics(
            ticker=ticker,
            date=str(row["snapshot_date"]),
            price=row.get("close"),
            pe_ratio=row.get("pe_ratio"),
            pb_ratio=row.get("pb_ratio"),
            market_cap_rmb=row.get("market_cap_rmb"),
        )
        sr = ScreeningResult(
            company=company,
            financials=financials,
            valuation=valuation,
        )
        scorer.score(sr)

        scored_records.append({
            "ticker": ticker,
            "snapshot_date": row["snapshot_date"],
            "close": row["close"],
            "pe_ratio": row.get("pe_ratio"),
            "pb_ratio": row.get("pb_ratio"),
            "market_cap_rmb": row.get("market_cap_rmb"),
            "roe": row.get("roe"),
            "net_margin": row.get("net_margin"),
            "debt_to_equity": row.get("debt_to_equity"),
            "composite_score": sr.composite_score,
            "value_score": sr.value_score,
            "quality_score": sr.quality_score,
            "growth_score": sr.growth_score,
        })

    return pd.DataFrame(scored_records)


def build_ground_truth(force: bool = False) -> Path:
    """Build the ground-truth dataset from stored training data.

    Returns the path to the output Parquet file.
    """
    if GROUND_TRUTH_FILE.exists() and not force:
        logger.info("Ground truth already exists at %s. Use force=True to rebuild.", GROUND_TRUTH_FILE)
        return GROUND_TRUTH_FILE

    # Load price data
    ashare_path = TRAINER_DIR / "ashare_prices.parquet"
    hkshare_path = TRAINER_DIR / "hkshare_prices.parquet"

    frames = []
    if ashare_path.exists():
        frames.append(pd.read_parquet(str(ashare_path)))
    if hkshare_path.exists():
        frames.append(pd.read_parquet(str(hkshare_path)))

    if not frames:
        raise FileNotFoundError(
            "No price data found. Run `valueinvestor improve-scorer --fetch-only` first."
        )

    all_prices = pd.concat(frames, ignore_index=True)
    all_prices["date_dt"] = pd.to_datetime(all_prices["date"])
    logger.info("Loaded %d price rows for %d tickers", len(all_prices), all_prices["ticker"].nunique())

    # Load valuations & financials
    val_path = TRAINER_DIR / "valuations.parquet"
    fin_path = TRAINER_DIR / "financials.parquet"
    val_df = pd.read_parquet(str(val_path)) if val_path.exists() else pd.DataFrame()
    fin_df = pd.read_parquet(str(fin_path)) if fin_path.exists() else pd.DataFrame()

    # Compute forward returns for all price data
    logger.info("Computing forward returns (horizon=%d days) …", FORWARD_HORIZON_DAYS)
    returns_df = _compute_forward_returns(all_prices)
    if returns_df.empty:
        logger.warning("No forward returns could be computed")
        return GROUND_TRUTH_FILE

    logger.info("Forward returns: %d rows", len(returns_df))

    # Get snapshot dates
    snapshot_dates = _get_snapshot_dates(all_prices)
    logger.info("Snapshot dates: %d (every %d days)", len(snapshot_dates), SNAPSHOT_INTERVAL_DAYS)

    # Build features and scores for each snapshot
    gt_frames = []
    for snap_date in snapshot_dates:
        logger.info("Building snapshot for %s …", snap_date)
        features = _build_snapshot_features(snap_date, all_prices, val_df, fin_df)
        if features.empty:
            continue
        scored = _score_snapshot(features)
        gt_frames.append(scored)

    if not gt_frames:
        logger.warning("No snapshots produced features")
        return GROUND_TRUTH_FILE

    gt_df = pd.concat(gt_frames, ignore_index=True)

    # Merge with forward returns
    returns_df["snapshot_date"] = returns_df["date"]
    merged = gt_df.merge(
        returns_df[["ticker", "snapshot_date", "forward_return_6m"]],
        on=["ticker", "snapshot_date"],
        how="inner",
    )

    # Drop rows without forward returns
    merged = merged.dropna(subset=["forward_return_6m"])
    logger.info(
        "Ground truth: %d rows across %d snapshots for %d tickers",
        len(merged),
        merged["snapshot_date"].nunique(),
        merged["ticker"].nunique(),
    )

    merged.to_parquet(str(GROUND_TRUTH_FILE), index=False)
    logger.info("Saved ground truth → %s", GROUND_TRUTH_FILE)

    return GROUND_TRUTH_FILE
