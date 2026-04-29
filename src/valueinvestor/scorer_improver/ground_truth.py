"""Phase 2 — Build ground-truth dataset with forward returns.

For each rolling quarter across the 10-year history window, creates a snapshot
of each stock's fundamentals and valuation, then computes the 6-month forward
return as the ground-truth label.  The current scorer is also run to produce
baseline scores.

Output: ``data/trainer/ground_truth.parquet``
"""

from __future__ import annotations

import logging
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from valueinvestor.scorer_improver.data_prep import TRAINER_DIR

logger = logging.getLogger(__name__)

GROUND_TRUTH_FILE = TRAINER_DIR / "ground_truth.parquet"
LEGACY_GROUND_TRUTH_FILE = TRAINER_DIR / "ground_truth_legacy.parquet"
CURRENT_GROUND_TRUTH_FILE = TRAINER_DIR / "ground_truth_current.parquet"

# Rolling snapshot interval in days (~quarterly)
SNAPSHOT_INTERVAL_DAYS = 90
# Forward return horizons in calendar days (nearest price within ±15 days is used)
FORWARD_HORIZON_1M_DAYS = 30    # ~1 month
FORWARD_HORIZON_3M_DAYS = 90    # ~3 months
FORWARD_HORIZON_DAYS = 126      # ~6 months (unchanged, backward-compatible alias)
FORWARD_HORIZON_6M_DAYS = 126   # explicit 6m constant

VALUATION_FEATURE_COLUMNS = (
    "pe_ratio",
    "pe_forward",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "dividend_yield",
    "ev_to_ebitda",
    "market_cap_rmb",
)

FINANCIAL_FEATURE_COLUMNS = (
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

RETURN_COLUMNS = (
    "forward_return_1m",
    "forward_return_3m",
    "forward_return_6m",
)

REQUIRED_GROUND_TRUTH_COLUMNS = (
    "ticker",
    "snapshot_date",
    "close",
    *VALUATION_FEATURE_COLUMNS,
    *FINANCIAL_FEATURE_COLUMNS,
    "composite_score",
    "value_score",
    "quality_score",
    "growth_score",
    *RETURN_COLUMNS,
)

MINIMUM_GROUND_TRUTH_COLUMNS = (
    "ticker",
    "snapshot_date",
    "close",
    "forward_return_6m",
)


def ground_truth_fingerprint(path: Path = GROUND_TRUTH_FILE) -> str:
    """Return a cheap identity for the current ground-truth parquet."""
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _compute_forward_returns(
    prices_df: pd.DataFrame,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    column_name: str = "forward_return_6m",
) -> pd.DataFrame:
    """For each (ticker, date), compute the forward return over *horizon_days*.

    Returns a DataFrame with columns: ticker, date, close, <column_name>.
    Forward return is (price[t+horizon] - price[t]) / price[t].
    """
    if prices_df.empty:
        return pd.DataFrame()

    prices = prices_df.loc[:, ["ticker", "date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["ticker", "date", "close"])
    prices = prices[prices["close"] > 0]
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "date", "close", column_name])

    prices = prices.sort_values(["ticker", "date"], kind="mergesort")

    horizon_ns = pd.Timedelta(days=horizon_days).value
    tolerance_ns = pd.Timedelta(days=15).value
    max_distance = np.iinfo(np.int64).max
    frames = []

    for ticker, group in prices.groupby("ticker", sort=False):
        dates_ns = group["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
        closes = group["close"].to_numpy(dtype="float64")
        if len(dates_ns) < 2:
            continue

        target_ns = dates_ns + horizon_ns
        positions = np.searchsorted(dates_ns, target_ns)

        before_idx = np.clip(positions - 1, 0, len(dates_ns) - 1)
        after_idx = np.clip(positions, 0, len(dates_ns) - 1)
        has_before = positions > 0
        has_after = positions < len(dates_ns)

        before_dist = np.full(len(dates_ns), max_distance, dtype="int64")
        after_dist = np.full(len(dates_ns), max_distance, dtype="int64")
        before_dist[has_before] = np.abs(dates_ns[before_idx[has_before]] - target_ns[has_before])
        after_dist[has_after] = np.abs(dates_ns[after_idx[has_after]] - target_ns[has_after])

        use_after = after_dist <= before_dist
        nearest_idx = np.where(use_after, after_idx, before_idx)
        nearest_dist = np.where(use_after, after_dist, before_dist)
        future_closes = closes[nearest_idx]

        valid = (
            (nearest_dist <= tolerance_ns)
            & np.isfinite(closes)
            & np.isfinite(future_closes)
            & (closes > 0)
            & (future_closes > 0)
        )
        if not valid.any():
            continue

        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(dates_ns[valid]).date,
            "close": closes[valid],
            column_name: (future_closes[valid] - closes[valid]) / closes[valid],
        }))

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", column_name])
    return pd.concat(frames, ignore_index=True)


def _get_snapshot_dates(
    prices_df: pd.DataFrame,
    interval_days: int = SNAPSHOT_INTERVAL_DAYS,
    min_forward_days: int = FORWARD_HORIZON_DAYS,
) -> List[date]:
    """Generate quarterly snapshot dates from the available price history."""
    if prices_df.empty:
        return []

    dates = pd.to_datetime(prices_df["date"])
    min_date = dates.min().date()
    max_date = dates.max().date()

    # Last snapshot must leave room for the longest forward horizon
    last_snap = max_date - timedelta(days=min_forward_days + 30)
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


def _latest_feature_frame(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    tickers: pd.Series,
) -> pd.DataFrame:
    """Return latest available feature columns indexed to *tickers*."""
    result = pd.DataFrame({"ticker": tickers})
    if df.empty or "ticker" not in df.columns:
        for col in columns:
            result[col] = None
        return result

    available_cols = ["ticker", *[col for col in columns if col in df.columns]]
    latest = df[available_cols].drop_duplicates("ticker", keep="last")
    result = result.merge(latest, on="ticker", how="left")
    for col in columns:
        if col not in result.columns:
            result[col] = None
    return result[["ticker", *columns]]


def _build_snapshot_features(
    snapshot_date: date,
    prices_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature rows for all stocks at a given snapshot date.

    For each stock with available data, produces a feature row containing
    all valuation and financial fields that ``MultiFactorScorer`` can use.
    """
    if prices_df.empty:
        return pd.DataFrame()

    target = pd.Timestamp(snapshot_date)
    prices = prices_df
    if "date_dt" not in prices.columns:
        prices = prices.copy()
        prices["date_dt"] = pd.to_datetime(prices["date"])

    window = prices.loc[
        (prices["date_dt"] >= target - pd.Timedelta(days=10))
        & (prices["date_dt"] <= target + pd.Timedelta(days=10))
        & (prices["close"] > 0),
        ["ticker", "date_dt", "close"],
    ].copy()
    if window.empty:
        return pd.DataFrame()

    window["_distance"] = (window["date_dt"] - target).abs()
    nearest_idx = window.groupby("ticker", sort=False)["_distance"].idxmin()
    features = window.loc[nearest_idx, ["ticker", "close"]].reset_index(drop=True)
    features["snapshot_date"] = snapshot_date

    valuation_features = _latest_feature_frame(
        val_df,
        VALUATION_FEATURE_COLUMNS,
        features["ticker"],
    )
    financial_features = _latest_feature_frame(
        fin_df,
        FINANCIAL_FEATURE_COLUMNS,
        features["ticker"],
    )

    features = features.merge(valuation_features, on="ticker", how="left")
    features = features.merge(financial_features, on="ticker", how="left")
    ordered_columns = [
        "ticker",
        "snapshot_date",
        "close",
        *VALUATION_FEATURE_COLUMNS,
        *FINANCIAL_FEATURE_COLUMNS,
    ]
    return features[ordered_columns].where(pd.notna(features[ordered_columns]), None)


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

    for row in features_df.to_dict("records"):
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
            revenue=row.get("revenue"),
            net_income=row.get("net_income"),
            total_assets=row.get("total_assets"),
            total_liabilities=row.get("total_liabilities"),
            total_equity=row.get("total_equity"),
            operating_cash_flow=row.get("operating_cash_flow"),
            free_cash_flow=row.get("free_cash_flow"),
            gross_margin=row.get("gross_margin"),
            net_margin=row.get("net_margin"),
            roe=row.get("roe"),
            roa=row.get("roa"),
            debt_to_equity=row.get("debt_to_equity"),
            current_ratio=row.get("current_ratio"),
        )
        valuation = ValuationMetrics(
            ticker=ticker,
            date=str(row["snapshot_date"]),
            price=row.get("close"),
            pe_ratio=row.get("pe_ratio"),
            pe_forward=row.get("pe_forward"),
            pb_ratio=row.get("pb_ratio"),
            ps_ratio=row.get("ps_ratio"),
            peg_ratio=row.get("peg_ratio"),
            dividend_yield=row.get("dividend_yield"),
            ev_to_ebitda=row.get("ev_to_ebitda"),
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
            "pe_forward": row.get("pe_forward"),
            "pb_ratio": row.get("pb_ratio"),
            "ps_ratio": row.get("ps_ratio"),
            "peg_ratio": row.get("peg_ratio"),
            "dividend_yield": row.get("dividend_yield"),
            "ev_to_ebitda": row.get("ev_to_ebitda"),
            "market_cap_rmb": row.get("market_cap_rmb"),
            "revenue": row.get("revenue"),
            "net_income": row.get("net_income"),
            "total_assets": row.get("total_assets"),
            "total_liabilities": row.get("total_liabilities"),
            "total_equity": row.get("total_equity"),
            "operating_cash_flow": row.get("operating_cash_flow"),
            "free_cash_flow": row.get("free_cash_flow"),
            "gross_margin": row.get("gross_margin"),
            "roe": row.get("roe"),
            "roa": row.get("roa"),
            "net_margin": row.get("net_margin"),
            "debt_to_equity": row.get("debt_to_equity"),
            "current_ratio": row.get("current_ratio"),
            "composite_score": sr.composite_score,
            "value_score": sr.value_score,
            "quality_score": sr.quality_score,
            "growth_score": sr.growth_score,
        })

    return pd.DataFrame(scored_records)


def build_ground_truth(force: bool = False, output_path: Path = GROUND_TRUTH_FILE) -> Path:
    """Build the ground-truth dataset from stored training data.

    Returns the path to the output Parquet file.
    """
    if output_path.exists() and not force:
        missing = _missing_ground_truth_columns(output_path)
        if not missing:
            logger.info("Ground truth already exists at %s. Use force=True to rebuild.", output_path)
            return output_path
        logger.info(
            "Ground truth at %s is missing %d required column(s); rebuilding.",
            output_path,
            len(missing),
        )

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

    # Compute forward returns for all 3 horizons
    horizons = [
        (FORWARD_HORIZON_1M_DAYS, "forward_return_1m"),
        (FORWARD_HORIZON_3M_DAYS, "forward_return_3m"),
        (FORWARD_HORIZON_6M_DAYS, "forward_return_6m"),
    ]
    all_returns: dict = {}
    for horizon_days, col_name in horizons:
        logger.info("Computing forward returns (horizon=%d days, col=%s) …", horizon_days, col_name)
        returns_df = _compute_forward_returns(all_prices, horizon_days=horizon_days, column_name=col_name)
        if returns_df.empty:
            logger.warning("No forward returns for horizon %d", horizon_days)
            continue
        all_returns[col_name] = returns_df
        logger.info("Forward returns (%s): %d rows", col_name, len(returns_df))

    if not all_returns:
        logger.warning("No forward returns could be computed for any horizon")
        return output_path

    # Get snapshot dates (use longest horizon for cutoff so all horizons can be evaluated)
    snapshot_dates = _get_snapshot_dates(all_prices, min_forward_days=FORWARD_HORIZON_6M_DAYS)
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
        return output_path

    gt_df = pd.concat(gt_frames, ignore_index=True)

    # Merge with forward returns for each horizon
    merged = gt_df.copy()
    for col_name, returns_df in all_returns.items():
        returns_df = returns_df.copy()
        returns_df["snapshot_date"] = returns_df["date"]
        merged = merged.merge(
            returns_df[["ticker", "snapshot_date", col_name]],
            on=["ticker", "snapshot_date"],
            how="left",
        )

    # Drop rows without 6m returns (primary target)
    merged = merged.dropna(subset=["forward_return_6m"])
    logger.info(
        "Ground truth: %d rows across %d snapshots for %d tickers",
        len(merged),
        merged["snapshot_date"].nunique(),
        merged["ticker"].nunique(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(output_path), index=False)
    logger.info("Saved ground truth → %s", output_path)

    return output_path


def _missing_ground_truth_columns(path: Path = GROUND_TRUTH_FILE) -> List[str]:
    """Return required columns absent from the current ground-truth parquet."""
    if not path.exists():
        return list(REQUIRED_GROUND_TRUTH_COLUMNS)
    try:
        columns = set(pd.read_parquet(str(path)).columns)
    except Exception:
        logger.warning("Could not inspect ground-truth schema; rebuild required.", exc_info=True)
        return list(REQUIRED_GROUND_TRUTH_COLUMNS)
    return [col for col in REQUIRED_GROUND_TRUTH_COLUMNS if col not in columns]


def _ground_truth_columns(path: Path = GROUND_TRUTH_FILE) -> set[str]:
    """Return current ground-truth columns, or an empty set if unreadable."""
    if not path.exists():
        return set()
    try:
        return set(pd.read_parquet(str(path)).columns)
    except Exception:
        logger.warning("Could not inspect ground-truth schema; rebuild required.", exc_info=True)
        return set()


def ensure_ground_truth_ready(
    path: Path = GROUND_TRUTH_FILE,
    *,
    allow_legacy_schema: bool = True,
) -> Path:
    """Ensure the ground-truth parquet has current trainer columns.

    Existing 6m-only files can be augmented cheaply for 1m/3m returns. Files
    missing optional scorer feature columns are allowed so older ground-truth
    datasets remain comparable with historical metrics.
    """
    if not path.exists():
        return build_ground_truth(force=True, output_path=path)

    columns = _ground_truth_columns(path)
    if not columns:
        return build_ground_truth(force=True, output_path=path)

    missing_minimum = [col for col in MINIMUM_GROUND_TRUTH_COLUMNS if col not in columns]
    if missing_minimum:
        logger.info("Ground truth is missing required columns (%s); rebuilding.", ", ".join(missing_minimum))
        return build_ground_truth(force=True, output_path=path)

    missing_returns = [col for col in ("forward_return_1m", "forward_return_3m") if col not in columns]
    if missing_returns:
        return augment_ground_truth_with_horizons(force=True, path=path)

    missing = [col for col in REQUIRED_GROUND_TRUTH_COLUMNS if col not in columns]
    if missing:
        if not allow_legacy_schema:
            logger.info(
                "Ground truth at %s is missing current feature columns (%s); rebuilding.",
                path,
                ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
            )
            return build_ground_truth(force=True, output_path=path)
        logger.info(
            "Using legacy ground-truth feature schema; missing optional columns: %s",
            ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
        )

    if not missing:
        return GROUND_TRUTH_FILE
    return path


def ensure_legacy_ground_truth_ready() -> Path:
    """Ensure the frozen legacy benchmark exists and remains legacy-schema tolerant."""
    if not LEGACY_GROUND_TRUTH_FILE.exists() and GROUND_TRUTH_FILE.exists():
        shutil.copy2(GROUND_TRUTH_FILE, LEGACY_GROUND_TRUTH_FILE)
    return ensure_ground_truth_ready(LEGACY_GROUND_TRUTH_FILE, allow_legacy_schema=True)


def ensure_current_ground_truth_ready(force: bool = False) -> Path:
    """Ensure the updated benchmark exists with the full current feature schema."""
    if force or not CURRENT_GROUND_TRUTH_FILE.exists():
        return build_ground_truth(force=True, output_path=CURRENT_GROUND_TRUTH_FILE)
    return ensure_ground_truth_ready(CURRENT_GROUND_TRUTH_FILE, allow_legacy_schema=False)


def augment_ground_truth_with_horizons(force: bool = False, path: Path = GROUND_TRUTH_FILE) -> Path:
    """Add ``forward_return_1m`` and ``forward_return_3m`` columns to the
    existing ground-truth parquet without rebuilding from scratch.

    This is a one-time migration for existing installations that only have the
    ``forward_return_6m`` column.

    Returns the path to the (updated) ground-truth file.
    """
    if not path.exists():
        logger.warning("Ground truth file not found; run build_ground_truth() first.")
        return path

    gt_df = pd.read_parquet(str(path))

    horizons_needed = []
    for col, horizon_days in [("forward_return_1m", FORWARD_HORIZON_1M_DAYS), ("forward_return_3m", FORWARD_HORIZON_3M_DAYS)]:
        if col not in gt_df.columns or force:
            horizons_needed.append((col, horizon_days))

    if not horizons_needed:
        logger.info("Ground truth already has 1m/3m columns — nothing to do.")
        return path

    # Load price data
    ashare_path = TRAINER_DIR / "ashare_prices.parquet"
    hkshare_path = TRAINER_DIR / "hkshare_prices.parquet"
    frames = []
    if ashare_path.exists():
        frames.append(pd.read_parquet(str(ashare_path)))
    if hkshare_path.exists():
        frames.append(pd.read_parquet(str(hkshare_path)))
    if not frames:
        logger.error("No price data found; cannot augment ground truth.")
        return path

    all_prices = pd.concat(frames, ignore_index=True)

    for col_name, horizon_days in horizons_needed:
        logger.info("Computing %s (horizon=%d days) for augmentation …", col_name, horizon_days)
        returns_df = _compute_forward_returns(all_prices, horizon_days=horizon_days, column_name=col_name)
        if returns_df.empty:
            logger.warning("No forward returns for %s", col_name)
            continue

        returns_df = returns_df.copy()
        returns_df["snapshot_date"] = pd.to_datetime(returns_df["date"]).dt.date

        # Ensure snapshot_date types match
        if "snapshot_date" in gt_df.columns:
            gt_df["snapshot_date"] = pd.to_datetime(gt_df["snapshot_date"]).dt.date

        if col_name in gt_df.columns:
            gt_df = gt_df.drop(columns=[col_name])

        gt_df = gt_df.merge(
            returns_df[["ticker", "snapshot_date", col_name]],
            on=["ticker", "snapshot_date"],
            how="left",
        )
        filled = gt_df[col_name].notna().sum()
        logger.info("Augmented %s: %d/%d rows have values", col_name, filled, len(gt_df))

    gt_df.to_parquet(str(path), index=False)
    logger.info("Saved augmented ground truth → %s", path)
    return path
