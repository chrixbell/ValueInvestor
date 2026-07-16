"""Phase 1 — Fetch and store 10-year historical training data (differential).

Fetches daily price history for A-share (Tencent via akshare) and HK-share
(yfinance) markets, along with valuations and financials snapshots.  Data is
stored as Parquet files in ``data/trainer/`` and metadata in
``data/trainer.db`` (SQLite).

Differential download: only fetches date ranges not already present in the
existing Parquet files.  Re-running is fast and idempotent — it fills gaps
(missing earlier history, missing recent days) without re-downloading data
that is already cached.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from valueinvestor.data.models import Company, Market

logger = logging.getLogger(__name__)

TRAINER_DIR = Path("data/trainer")
TRAINER_DB = Path("data/trainer.db")

_ASHARE_PRICES_FILE = TRAINER_DIR / "ashare_prices.parquet"
_HKSHARE_PRICES_FILE = TRAINER_DIR / "hkshare_prices.parquet"
_VALUATIONS_FILE = TRAINER_DIR / "valuations.parquet"
_FINANCIALS_FILE = TRAINER_DIR / "financials.parquet"
_ASHARE_FINANCIAL_CHECKPOINT_FILE = TRAINER_DIR / "ashare_financials_checkpoint.parquet"
_ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE = (
    TRAINER_DIR / "ashare_bulk_financials_checkpoint.parquet"
)
_ASHARE_STATEMENT_FINANCIAL_CHECKPOINT_FILE = (
    TRAINER_DIR / "ashare_statement_financials_checkpoint.parquet"
)
_ASHARE_STATEMENT_FINANCIAL_MANIFEST_FILE = (
    TRAINER_DIR / "ashare_statement_financials_checkpoint.json"
)
_HK_FINANCIAL_CHECKPOINT_FILE = TRAINER_DIR / "hk_financials_checkpoint.parquet"

# Number of years of historical price data to collect for training
_HISTORY_YEARS = 10

# Tencent history parallel workers (balance speed vs rate limiting)
_TENCENT_WORKERS = 8
# yfinance HK parallel workers
_YF_WORKERS = 10
# Financial statement history workers.  These endpoints are less predictable
# than price history, so keep concurrency lower.
_ASHARE_FINANCIAL_WORKERS = 6
_HKSHARE_FINANCIAL_WORKERS = 4
# ``stock_zh_a_daily`` may initialize mini_racer internally; concurrent first
# use can crash the process, so historical market-cap enrichment is sequential.
_ASHARE_MARKET_CAP_WORKERS = 1
# Max stocks to fetch (0 = all)
_MAX_STOCKS = 0
# Incremental save interval: save partial results every N tasks
_SAVE_INTERVAL = 200
# Brief sleep between requests to avoid rate limiting
_TENCENT_DELAY = 0.05
_FINANCIAL_FETCH_DELAY = 0.05
_FINANCIAL_FETCH_RETRIES = 3

_VALUATION_FEATURE_COLUMNS = (
    "pe_ratio",
    "pe_forward",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "dividend_yield",
    "ev_to_ebitda",
    "market_cap_rmb",
)

_FINANCIAL_FEATURE_COLUMNS = (
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

_FINANCIAL_DERIVATION_COLUMNS = (
    "eps",
    "book_value_per_share",
    "operating_cash_flow_per_share",
    "statement_months",
    "currency_to_rmb",
)

_FINANCIAL_CURRENCY_TO_RMB = {
    "CNY": 1.0,
    "HKD": 0.92,
    "USD": 7.20,
    "EUR": 7.80,
    "GBP": 9.20,
    "SGD": 5.30,
    "AUD": 4.80,
    "CAD": 5.20,
    "JPY": 0.05,
}

_CREATE_TRAINER_TABLES = """
CREATE TABLE IF NOT EXISTS stock_meta (
    ticker    TEXT PRIMARY KEY,
    name      TEXT,
    market    TEXT,
    sector    TEXT,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS fetch_status (
    key       TEXT PRIMARY KEY,
    value     TEXT
);
"""


def _init_db() -> sqlite3.Connection:
    """Create / open the trainer database and ensure tables exist."""
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TRAINER_DB))
    conn.executescript(_CREATE_TRAINER_TABLES)
    return conn


def _get_fetch_status(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM fetch_status WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_fetch_status(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO fetch_status (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def _training_data_complete(
    *,
    force: bool,
    last_fetch: Optional[str],
    stored_start: Optional[str],
    required_start: str,
    required_fetch_date: str,
    required_files: Iterable[Path],
) -> bool:
    return (
        not force
        and last_fetch is not None
        and last_fetch >= required_fetch_date
        and stored_start is not None
        and stored_start <= required_start
        and all(path.exists() for path in required_files)
    )


# -----------------------------------------------------------------------
# Date-range helpers for differential download
# -----------------------------------------------------------------------

def _parse_yyyymmdd(date_str: str) -> date:
    """Parse a ``YYYYMMDD`` string into a :class:`~datetime.date`."""
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def _yyyymmdd_to_iso(date_str: str) -> str:
    """Convert ``YYYYMMDD`` to ``YYYY-MM-DD`` (yfinance / ISO format)."""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _get_ticker_date_ranges(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Return ``{ticker: (min_YYYYMMDD, max_YYYYMMDD)}`` from a prices DataFrame.

    Works regardless of whether the ``date`` column holds strings,
    :class:`~datetime.date` objects, or :class:`~pandas.Timestamp` values.
    """
    if df.empty or "date" not in df.columns or "ticker" not in df.columns:
        return {}
    normed = pd.to_datetime(df["date"])
    tmp = df[["ticker"]].copy()
    tmp["_d"] = normed
    result: Dict[str, Tuple[str, str]] = {}
    for ticker, grp in tmp.groupby("ticker"):
        mn = grp["_d"].min().strftime("%Y%m%d")
        mx = grp["_d"].max().strftime("%Y%m%d")
        result[str(ticker)] = (mn, mx)
    return result


# -----------------------------------------------------------------------
# Stock universe
# -----------------------------------------------------------------------

def _fetch_universe() -> Tuple[List[Company], List[Company]]:
    """Fetch A-share and HK-share stock lists using existing fetchers."""
    from valueinvestor.data.fetcher_ashare import AShareFetcher
    from valueinvestor.data.fetcher_hkshare import HKShareFetcher

    a_fetcher = AShareFetcher()
    hk_fetcher = HKShareFetcher()

    logger.info("Fetching A-share universe …")
    a_companies = a_fetcher.fetch_stock_list()
    logger.info("A-share universe: %d companies", len(a_companies))

    logger.info("Fetching HK-share universe …")
    hk_companies = hk_fetcher.fetch_stock_list()
    logger.info("HK-share universe: %d companies", len(hk_companies))

    return a_companies, hk_companies


def _store_universe(conn: sqlite3.Connection, companies: List[Company]) -> None:
    """Persist stock metadata to trainer.db."""
    now = date.today().isoformat()
    for c in companies:
        conn.execute(
            "INSERT OR REPLACE INTO stock_meta (ticker, name, market, sector, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (c.ticker, c.name, c.market.value, c.sector, now),
        )
    conn.commit()


def _load_stored_universe(conn: sqlite3.Connection) -> Tuple[List[Company], List[Company]]:
    """Load the last fetched stock universe from trainer.db."""
    rows = conn.execute(
        "SELECT ticker, name, market, sector FROM stock_meta ORDER BY ticker"
    ).fetchall()
    a_companies: list[Company] = []
    hk_companies: list[Company] = []
    for ticker, name, market_value, sector in rows:
        try:
            market = Market(market_value)
        except ValueError:
            continue
        company = Company(
            ticker=str(ticker),
            name=str(name or ticker),
            market=market,
            sector=sector,
        )
        if market == Market.A_SHARE:
            a_companies.append(company)
        elif market == Market.HK_SHARE:
            hk_companies.append(company)
    return a_companies, hk_companies


# -----------------------------------------------------------------------
# A-share price history (Tencent via akshare — globally accessible)
# -----------------------------------------------------------------------

def _fetch_ashare_prices(
    companies: List[Company],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Differentially fetch A-share prices via Tencent (akshare).

    For each ticker the function determines which date ranges are missing from
    the existing Parquet file and only downloads those gaps:

    * **New ticker** → full ``start_date … end_date`` range.
    * **Earlier gap** (existing min_date > start_date + 30 d) → fetch from
      ``start_date`` up to one day before the existing min_date.
    * **Recent gap** (existing max_date < end_date − 7 d) → fetch from one day
      after the existing max_date up to ``end_date``.

    Uses a thread pool and saves incrementally every ``_SAVE_INTERVAL`` tasks.

    Returns a deduplicated, sorted DataFrame with columns:
    ``ticker, date, open, close, high, low, volume``.
    """
    import akshare as ak

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]

    # ── Load existing data ────────────────────────────────────────────
    existing_df: pd.DataFrame = pd.DataFrame()
    ticker_ranges: Dict[str, Tuple[str, str]] = {}
    if _ASHARE_PRICES_FILE.exists():
        try:
            existing_df = pd.read_parquet(str(_ASHARE_PRICES_FILE))
            ticker_ranges = _get_ticker_date_ranges(existing_df)
            logger.info(
                "A-share existing data: %d tickers, %d rows",
                len(ticker_ranges), len(existing_df),
            )
        except Exception as exc:
            logger.warning("Could not read existing A-share prices: %s — starting fresh", exc)

    # ── Build differential fetch plan ─────────────────────────────────
    # Each task is (company, fetch_start_YYYYMMDD, fetch_end_YYYYMMDD)
    fetch_tasks: List[Tuple[Company, str, str]] = []
    req_start = _parse_yyyymmdd(start_date)
    req_end   = _parse_yyyymmdd(end_date)

    for company in companies:
        ticker = company.ticker
        if ticker not in ticker_ranges:
            fetch_tasks.append((company, start_date, end_date))
            continue

        ex_min = _parse_yyyymmdd(ticker_ranges[ticker][0])
        ex_max = _parse_yyyymmdd(ticker_ranges[ticker][1])

        # Earlier portion missing?
        if (ex_min - req_start).days > 30:
            gap_end = (ex_min - timedelta(days=1)).strftime("%Y%m%d")
            fetch_tasks.append((company, start_date, gap_end))

        # Recent portion missing?
        if (req_end - ex_max).days > 7:
            gap_start = (ex_max + timedelta(days=1)).strftime("%Y%m%d")
            fetch_tasks.append((company, gap_start, end_date))

    if not fetch_tasks:
        logger.info(
            "A-share prices fully up to date (%d tickers). No fetch needed.",
            len(ticker_ranges),
        )
        return existing_df

    logger.info(
        "A-share differential fetch: %d tasks across %d stocks (workers=%d) …",
        len(fetch_tasks), total, _TENCENT_WORKERS,
    )

    counter: Dict[str, int] = {"done": 0, "ok": 0, "fail": 0}
    new_frames: list[pd.DataFrame] = []
    frames_lock = threading.Lock()

    def _fetch_one(task: Tuple[Company, str, str]) -> Optional[pd.DataFrame]:
        company, t_start, t_end = task
        ticker = company.ticker
        prefix = "sh" if ticker.startswith("6") else "sz"
        symbol = f"{prefix}{ticker}"
        try:
            time.sleep(_TENCENT_DELAY)
            df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=t_start, end_date=t_end)
            if df is not None and not df.empty:
                df = df.rename(columns={"amount": "volume"})
                df["ticker"] = ticker
                cols = [c for c in ["ticker", "date", "open", "close", "high", "low", "volume"] if c in df.columns]
                return df[cols]
        except Exception:
            logger.debug("Failed A-share hist for %s (%s→%s)", ticker, t_start, t_end, exc_info=True)
        return None

    def _worker(task: Tuple[Company, str, str]) -> None:
        result = _fetch_one(task)
        with frames_lock:
            if result is not None:
                new_frames.append(result)
                counter["ok"] += 1
            else:
                counter["fail"] += 1
            counter["done"] += 1
            done = counter["done"]

        if done % 100 == 0:
            logger.info(
                "  A-share: %d/%d tasks (ok=%d, fail=%d)",
                done, len(fetch_tasks), counter["ok"], counter["fail"],
            )

        # Incremental save every _SAVE_INTERVAL tasks
        if done % _SAVE_INTERVAL == 0 and new_frames:
            with frames_lock:
                all_so_far = (
                    [existing_df] if not existing_df.empty else []
                ) + new_frames
                if all_so_far:
                    try:
                        combined = pd.concat(all_so_far, ignore_index=True)
                        combined = combined.drop_duplicates(subset=["ticker", "date"])
                        combined.to_parquet(str(_ASHARE_PRICES_FILE), index=False)
                        logger.info("  ↳ Incremental save: %d rows", len(combined))
                    except Exception as exc:
                        logger.warning("Incremental save failed: %s", exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_TENCENT_WORKERS) as executor:
        list(executor.map(_worker, fetch_tasks))

    logger.info(
        "A-share differential fetch complete: %d ok, %d fail",
        counter["ok"], counter["fail"],
    )

    all_frames = ([existing_df] if not existing_df.empty else []) + new_frames
    if not all_frames:
        logger.warning("No A-share price data fetched")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker", "date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("A-share prices: %d rows for %d stocks", len(result), result["ticker"].nunique())
    return result


# -----------------------------------------------------------------------
# HK-share price history (yfinance — globally accessible)
# -----------------------------------------------------------------------

def _fetch_hkshare_prices(
    companies: List[Company],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Differentially fetch HK-share prices via yfinance.

    Uses explicit ``start`` / ``end`` date parameters (ISO format) instead of
    a fixed ``period`` string, giving precise date-range control.  The same
    gap-detection logic as :func:`_fetch_ashare_prices` is applied:

    * **New ticker** → full ``start_date … end_date`` range.
    * **Earlier gap** → fetch from ``start_date`` up to day before existing min.
    * **Recent gap** → fetch from day after existing max up to ``end_date``.

    Saves incrementally every ``_SAVE_INTERVAL`` tasks so progress survives
    interruption.

    Returns a deduplicated, sorted DataFrame with columns:
    ``ticker, date, open, close, high, low, volume``.
    """
    import yfinance as yf

    # ── Load existing data ────────────────────────────────────────────
    existing_df: pd.DataFrame = pd.DataFrame()
    ticker_ranges: Dict[str, Tuple[str, str]] = {}
    if _HKSHARE_PRICES_FILE.exists():
        try:
            existing_df = pd.read_parquet(str(_HKSHARE_PRICES_FILE))
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
            ticker_ranges = _get_ticker_date_ranges(existing_df)
            logger.info(
                "HK-share existing data: %d tickers, %d rows",
                len(ticker_ranges), len(existing_df),
            )
        except Exception as exc:
            logger.warning("Could not read existing HK-share prices: %s — starting fresh", exc)

    requested_tickers = [company.ticker for company in companies]
    tickers = list(dict.fromkeys([*requested_tickers, *ticker_ranges]))
    if _MAX_STOCKS > 0:
        tickers = tickers[:_MAX_STOCKS]
    total = len(tickers)
    recovered_tickers = len(set(tickers) - set(requested_tickers))
    if recovered_tickers:
        logger.info(
            "HK-share universe augmented with %d tickers from existing price history",
            recovered_tickers,
        )

    # ── Build differential fetch plan ─────────────────────────────────
    # Each task is (ticker, yf_start_ISO, yf_end_ISO)
    fetch_tasks: List[Tuple[str, str, str]] = []
    req_start = _parse_yyyymmdd(start_date)
    req_end   = _parse_yyyymmdd(end_date)
    yf_start  = _yyyymmdd_to_iso(start_date)
    # yfinance end is exclusive — add one day so today's data is included
    yf_end    = (req_end + timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in tickers:
        if ticker not in ticker_ranges:
            fetch_tasks.append((ticker, yf_start, yf_end))
            continue

        ex_min = _parse_yyyymmdd(ticker_ranges[ticker][0])
        ex_max = _parse_yyyymmdd(ticker_ranges[ticker][1])

        # Earlier portion missing?
        if (ex_min - req_start).days > 30:
            gap_yf_end = (ex_min).strftime("%Y-%m-%d")  # exclusive: fetch up to ex_min
            fetch_tasks.append((ticker, yf_start, gap_yf_end))

        # Recent portion missing?
        if (req_end - ex_max).days > 7:
            gap_yf_start = (ex_max + timedelta(days=1)).strftime("%Y-%m-%d")
            fetch_tasks.append((ticker, gap_yf_start, yf_end))

    if not fetch_tasks:
        logger.info(
            "HK-share prices fully up to date (%d tickers). No fetch needed.",
            len(ticker_ranges),
        )
        return existing_df

    logger.info(
        "HK-share differential fetch: %d tasks across %d stocks (workers=%d) …",
        len(fetch_tasks), total, _YF_WORKERS,
    )

    counter: Dict[str, int] = {"done": 0, "ok": 0, "fail": 0}
    new_frames: list[pd.DataFrame] = []
    frames_lock = threading.Lock()

    def _fetch_one(task: Tuple[str, str, str]) -> None:
        ticker, t_start, t_end = task
        try:
            hist = yf.Ticker(ticker).history(start=t_start, end=t_end)
            if hist is not None and not hist.empty:
                df = hist.reset_index()
                df = df.rename(columns={
                    "Date": "date", "Open": "open", "Close": "close",
                    "High": "high", "Low": "low", "Volume": "volume",
                })
                df["ticker"] = ticker
                df["date"] = pd.to_datetime(df["date"]).dt.date
                cols = [c for c in ["ticker", "date", "open", "close", "high", "low", "volume"] if c in df.columns]
                with frames_lock:
                    new_frames.append(df[cols])
                    counter["ok"] += 1
                return
        except Exception:
            logger.debug("Failed HK history for %s (%s→%s)", ticker, t_start, t_end, exc_info=True)
        with frames_lock:
            counter["fail"] += 1

    def _worker(task: Tuple[str, str, str]) -> None:
        _fetch_one(task)
        with frames_lock:
            counter["done"] += 1
            done = counter["done"]

        if done % 50 == 0:
            logger.info(
                "  HK-share: %d/%d tasks (ok=%d, fail=%d)",
                done, len(fetch_tasks), counter["ok"], counter["fail"],
            )

        # Incremental save every _SAVE_INTERVAL tasks
        if done % _SAVE_INTERVAL == 0 and new_frames:
            with frames_lock:
                all_so_far = (
                    [existing_df] if not existing_df.empty else []
                ) + new_frames
                if all_so_far:
                    try:
                        combined = pd.concat(all_so_far, ignore_index=True)
                        combined = combined.drop_duplicates(subset=["ticker", "date"])
                        combined.to_parquet(str(_HKSHARE_PRICES_FILE), index=False)
                        logger.info("  ↳ HK incremental save: %d rows", len(combined))
                    except Exception as exc:
                        logger.warning("HK incremental save failed: %s", exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_YF_WORKERS) as executor:
        list(executor.map(_worker, fetch_tasks))

    logger.info(
        "HK-share differential fetch complete: %d ok, %d fail",
        counter["ok"], counter["fail"],
    )

    all_frames = ([existing_df] if not existing_df.empty else []) + new_frames
    if not all_frames:
        logger.warning("No HK-share price data fetched")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker", "date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("HK-share prices: %d rows for %d stocks", len(result), result["ticker"].nunique())
    return result


# -----------------------------------------------------------------------
# Point-in-time valuation and financial feature sources
# -----------------------------------------------------------------------

def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(str(path))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(str(tmp_path), index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _normalise_date(value: object) -> Optional[str]:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def _feature_float(value: object) -> Optional[float]:
    """Parse provider numeric values, including common Chinese unit suffixes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "")
    if text in ("", "--", "-", "nan", "None", "NaN"):
        return None
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        if text.endswith("亿"):
            return float(text[:-1]) * 1e8
        if text.endswith("万"):
            return float(text[:-1]) * 1e4
        return float(text)
    except (TypeError, ValueError):
        return None


def _find_feature(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    for column in candidates:
        if column not in row:
            continue
        value = _feature_float(row.get(column))
        if value is not None:
            return value
    return None


def _percentage_fraction(value: object) -> Optional[float]:
    parsed = _feature_float(value)
    if parsed is None:
        return None
    return parsed / 100.0 if abs(parsed) > 1.0 else parsed


def _financial_currency_to_rmb(currency: object) -> Optional[float]:
    code = str(currency or "").strip().upper()
    return _FINANCIAL_CURRENCY_TO_RMB.get(code)


def _current_market_cap_share_estimates() -> dict[str, float]:
    """Estimate current shares outstanding from local valuation cache."""
    val_df, _fin_df = _export_cache_data()
    if val_df.empty or not {"ticker", "price", "market_cap_rmb"}.issubset(val_df.columns):
        return {}
    frame = val_df.loc[:, [c for c in ("ticker", "date", "price", "market_cap_rmb") if c in val_df.columns]].copy()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["market_cap_rmb"] = pd.to_numeric(frame["market_cap_rmb"], errors="coerce")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values(["ticker", "date"], kind="mergesort")
    frame = frame.dropna(subset=["ticker", "price", "market_cap_rmb"])
    frame = frame[(frame["price"] > 0) & (frame["market_cap_rmb"] > 0)]
    if frame.empty:
        return {}
    latest = frame.groupby("ticker", sort=False).tail(1)
    shares = latest["market_cap_rmb"] / latest["price"]
    return {
        str(ticker): float(share_count)
        for ticker, share_count in zip(latest["ticker"], shares)
        if pd.notna(share_count) and float(share_count) > 0
    }


def _ordered_valuation_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["ticker", "date", "price", *_VALUATION_FEATURE_COLUMNS]
    df = pd.DataFrame(records)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df[columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["ticker", "date"])
    for column in ["price", *_VALUATION_FEATURE_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["ticker", "date"], kind="mergesort")
    return (
        df.groupby(["ticker", "date"], sort=False, as_index=False)
        .last()
        .reset_index(drop=True)
    )


def _ordered_financial_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    value_columns = [*_FINANCIAL_FEATURE_COLUMNS, *_FINANCIAL_DERIVATION_COLUMNS]
    columns = ["ticker", "period", "report_date", *value_columns]
    df = pd.DataFrame(records)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df[columns].copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.date
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df = df.dropna(subset=["ticker", "period"])
    missing_report_date = df["report_date"].isna()
    if missing_report_date.any():
        df.loc[missing_report_date, "report_date"] = df.loc[
            missing_report_date,
            "period",
        ].map(_conservative_report_date)
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_nonnull"] = df[value_columns].notna().sum(axis=1)
    df = (
        df.sort_values(["ticker", "period", "_nonnull"], kind="mergesort")
        .drop_duplicates(["ticker", "period"], keep="last")
        .drop(columns=["_nonnull"])
        .reset_index(drop=True)
    )
    return df


def _conservative_report_date(period: object) -> Optional[date]:
    """Estimate when a statement could safely have been known to investors."""
    timestamp = pd.to_datetime(period, errors="coerce")
    if pd.isna(timestamp):
        return None
    lag_by_month = {3: 60, 6: 90, 9: 60, 12: 120}
    lag_days = lag_by_month.get(int(timestamp.month), 120)
    return (timestamp + pd.Timedelta(days=lag_days)).date()


def _price_valuation_skeleton(
    prices: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Build non-leaky monthly valuation as-of rows from price history.

    The current live valuation cache is a future-dated snapshot.  These rows
    preserve historical as-of dates and price only; unavailable ratio fields
    remain null rather than being backfilled from today's ratios.
    """
    if prices.empty or not {"ticker", "date", "close"}.issubset(prices.columns):
        return _ordered_valuation_frame([])

    frame = prices.loc[:, ["ticker", "date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "date", "close"])
    frame = frame[(frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)]
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return _ordered_valuation_frame([])

    frame = frame.sort_values(["ticker", "date"], kind="mergesort")
    frame["_month"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby(["ticker", "_month"], sort=False).tail(1)
    share_estimates = _current_market_cap_share_estimates()
    records = [
        {
            "ticker": row["ticker"],
            "date": row["date"].date().isoformat(),
            "price": row["close"],
            "market_cap_rmb": (
                float(row["close"]) * share_estimates[str(row["ticker"])]
                if str(row["ticker"]) in share_estimates
                else None
            ),
        }
        for _, row in monthly.iterrows()
    ]
    return _ordered_valuation_frame(records)


def _exchange_prefix(ticker: str) -> str:
    if ticker.startswith("6"):
        return "sh"
    if ticker.startswith(("4", "8", "9")):
        return "bj"
    return "sz"


def _fetch_ashare_market_cap_history(
    companies: list[Company],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch monthly A-share historical market-cap rows when shares are exposed."""
    import akshare as ak

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]
    if not companies:
        return _ordered_valuation_frame([])

    records: list[dict[str, object]] = []
    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "fail": 0}

    def _fetch_one(company: Company) -> None:
        ticker = company.ticker
        symbol = f"{_exchange_prefix(ticker)}{ticker}"
        try:
            time.sleep(_FINANCIAL_FETCH_DELAY)
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="")
            if df is None or df.empty or "outstanding_share" not in df.columns:
                raise ValueError("no outstanding_share column")
            frame = df.loc[:, [c for c in ("date", "close", "outstanding_share") if c in df.columns]].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["outstanding_share"] = pd.to_numeric(frame["outstanding_share"], errors="coerce")
            frame = frame.dropna(subset=["date", "close", "outstanding_share"])
            frame = frame[(frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)]
            frame = frame[(frame["close"] > 0) & (frame["outstanding_share"] > 0)]
            if frame.empty:
                raise ValueError("no rows in requested date range")
            frame = frame.sort_values("date", kind="mergesort")
            frame["_month"] = frame["date"].dt.to_period("M")
            monthly = frame.groupby("_month", sort=False).tail(1)
            rows = [
                {
                    "ticker": ticker,
                    "date": row["date"].date().isoformat(),
                    "price": row["close"],
                    "market_cap_rmb": row["close"] * row["outstanding_share"],
                }
                for _, row in monthly.iterrows()
            ]
            with lock:
                records.extend(rows)
                counter["ok"] += 1
        except Exception:
            logger.debug("Failed A-share market-cap history for %s", ticker, exc_info=True)
            with lock:
                counter["fail"] += 1
        finally:
            with lock:
                counter["done"] += 1
                done = counter["done"]
            if done % 200 == 0:
                logger.info(
                    "  A-share market-cap history: %d/%d (ok=%d, fail=%d)",
                    done, total, counter["ok"], counter["fail"],
                )

    logger.info("Fetching A-share historical market-cap features (%d tickers) …", total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_ASHARE_MARKET_CAP_WORKERS) as executor:
        list(executor.map(_fetch_one, companies))
    logger.info(
        "A-share market-cap history complete: %d ok, %d fail",
        counter["ok"],
        counter["fail"],
    )
    return _ordered_valuation_frame(records)


def _ashare_financial_records_from_frame(ticker: str, df: pd.DataFrame) -> list[dict[str, object]]:
    if df is None or df.empty:
        return []
    period_col = "报告期" if "报告期" in df.columns else df.columns[0]
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        period = _normalise_date(row.get(period_col))
        if period is None:
            continue
        records.append({
            "ticker": ticker,
            "period": period,
            "report_date": _conservative_report_date(period),
            "revenue": _find_feature(row, ["营业总收入", "营业收入", "主营业务收入", "revenue"]),
            "net_income": _find_feature(row, ["净利润", "归属净利润", "归母净利润", "net_income"]),
            "total_assets": _find_feature(row, ["总资产", "资产总计", "total_assets"]),
            "total_liabilities": _find_feature(row, ["总负债", "负债合计", "total_liabilities"]),
            "total_equity": _find_feature(
                row,
                ["股东权益合计", "归属母公司股东权益", "所有者权益合计", "净资产", "total_equity"],
            ),
            "operating_cash_flow": _find_feature(
                row,
                ["经营现金流", "经营活动现金流量净额", "operating_cash_flow"],
            ),
            "free_cash_flow": _find_feature(row, ["自由现金流", "free_cash_flow"]),
            "gross_margin": _find_feature(row, ["毛利率", "销售毛利率", "gross_margin"]),
            "net_margin": _find_feature(row, ["净利率", "销售净利率", "net_margin"]),
            "roe": _find_feature(row, ["净资产收益率", "摊薄净资产收益率", "ROE", "roe"]),
            "roa": _find_feature(row, ["总资产收益率", "ROA", "roa"]),
            "debt_to_equity": _find_feature(row, ["资产负债率", "debt_to_equity"]),
            "current_ratio": _find_feature(row, ["流动比率", "current_ratio"]),
            "eps": _find_feature(row, ["基本每股收益", "每股收益", "eps"]),
            "book_value_per_share": _find_feature(
                row,
                ["每股净资产", "book_value_per_share"],
            ),
            "operating_cash_flow_per_share": _find_feature(
                row,
                ["每股经营现金流", "operating_cash_flow_per_share"],
            ),
        })
    return records


def _ashare_bulk_financial_records_from_frame(
    period: date,
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    if df is None or df.empty or "股票代码" not in df.columns:
        return []
    report_date = _conservative_report_date(period)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        ticker = str(row.get("股票代码") or "").strip().split(".")[0].zfill(6)
        if not ticker.isdigit() or len(ticker) != 6:
            continue
        records.append({
            "ticker": ticker,
            "period": period,
            "report_date": report_date,
            "revenue": _find_feature(row, ["营业总收入-营业总收入", "营业总收入"]),
            "net_income": _find_feature(row, ["净利润-净利润", "净利润"]),
            "gross_margin": _percentage_fraction(row.get("销售毛利率")),
            "roe": _percentage_fraction(row.get("净资产收益率")),
            "eps": _find_feature(row, ["每股收益"]),
            "book_value_per_share": _find_feature(row, ["每股净资产"]),
            "operating_cash_flow_per_share": _find_feature(
                row,
                ["每股经营现金流量"],
            ),
        })
    return records


def _ashare_statement_financial_records_from_frames(
    period: date,
    *,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    income: pd.DataFrame,
) -> list[dict[str, object]]:
    """Merge all-market statement tables into point-in-time feature rows."""
    rows_by_ticker: dict[str, dict[str, object]] = {}
    report_date = _conservative_report_date(period)
    statement_months = float(period.month)

    def target_for(row: pd.Series) -> Optional[dict[str, object]]:
        ticker = str(row.get("股票代码") or "").strip().split(".")[0].zfill(6)
        if not ticker.isdigit() or len(ticker) != 6:
            return None
        return rows_by_ticker.setdefault(
            ticker,
            {
                "ticker": ticker,
                "period": period,
                "report_date": report_date,
                "statement_months": statement_months,
                "currency_to_rmb": 1.0,
            },
        )

    if balance is not None and not balance.empty:
        for _, row in balance.iterrows():
            target = target_for(row)
            if target is None:
                continue
            target["total_assets"] = _find_feature(row, ["资产-总资产", "总资产"])
            target["total_liabilities"] = _find_feature(
                row,
                ["负债-总负债", "总负债"],
            )
            target["total_equity"] = _find_feature(
                row,
                ["股东权益合计", "所有者权益合计"],
            )
            target["debt_to_equity"] = _percentage_fraction(row.get("资产负债率"))

    if cashflow is not None and not cashflow.empty:
        for _, row in cashflow.iterrows():
            target = target_for(row)
            if target is None:
                continue
            target["operating_cash_flow"] = _find_feature(
                row,
                ["经营性现金流-现金流量净额", "经营活动现金流量净额"],
            )

    if income is not None and not income.empty:
        for _, row in income.iterrows():
            target = target_for(row)
            if target is None:
                continue
            revenue = _find_feature(row, ["营业总收入", "营业收入"])
            net_income = _find_feature(row, ["净利润", "归属净利润"])
            operating_cost = _find_feature(
                row,
                ["营业总支出-营业支出", "营业成本"],
            )
            target["revenue"] = revenue
            target["net_income"] = net_income
            if revenue is not None and revenue > 0:
                if operating_cost is not None:
                    target["gross_margin"] = (revenue - operating_cost) / revenue
                if net_income is not None:
                    target["net_margin"] = net_income / revenue

    for target in rows_by_ticker.values():
        net_income = _feature_float(target.get("net_income"))
        total_assets = _feature_float(target.get("total_assets"))
        total_equity = _feature_float(target.get("total_equity"))
        annualization = 12.0 / statement_months
        if net_income is not None and total_assets is not None and total_assets > 0:
            target["roa"] = annualization * net_income / total_assets
        if net_income is not None and total_equity is not None and total_equity > 0:
            target["roe"] = annualization * net_income / total_equity
    return list(rows_by_ticker.values())


def _quarter_end_dates(start: date, end: date) -> list[date]:
    history_start = start - timedelta(days=400)
    dates: list[date] = []
    for year in range(history_start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if history_start <= period <= end:
                dates.append(period)
    return dates


def _fetch_ashare_financial_history_bulk(
    *,
    start: date,
    end: date,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch one all-market earnings table per report date from Eastmoney."""
    import akshare as ak

    if force and _ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE.exists():
        _ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE.unlink()
    checkpoint = _read_parquet(_ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE)
    records = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed_periods = {
        pd.Timestamp(period).date()
        for period in checkpoint.get("period", pd.Series(dtype=object)).dropna()
    }
    periods = [
        period
        for period in _quarter_end_dates(start, end)
        if period not in completed_periods
    ]
    logger.info(
        "Fetching bulk A-share financial history (%d report dates, %d resumed) …",
        len(periods),
        len(completed_periods),
    )
    for index, period in enumerate(periods, start=1):
        frame = pd.DataFrame()
        for attempt in range(_FINANCIAL_FETCH_RETRIES):
            try:
                frame = ak.stock_yjbb_em(date=period.strftime("%Y%m%d"))
                if frame is not None and not frame.empty:
                    break
            except Exception:
                if attempt + 1 >= _FINANCIAL_FETCH_RETRIES:
                    logger.warning(
                        "Bulk A-share financial fetch failed for %s",
                        period,
                        exc_info=True,
                    )
            time.sleep(1.0 * (2 ** attempt))
        period_records = _ashare_bulk_financial_records_from_frame(period, frame)
        if not period_records:
            logger.warning("No bulk A-share financial rows for %s", period)
            continue
        records.extend(period_records)
        result = _ordered_financial_frame(_combine_financial_records(records))
        _write_parquet_atomic(result, _ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE)
        logger.info(
            "  Bulk A-share financial history: %d/%d period=%s rows=%d tickers=%d",
            index,
            len(periods),
            period,
            len(result),
            result["ticker"].nunique(),
        )
        time.sleep(0.5)
    return _ordered_financial_frame(_combine_financial_records(records))


def _fetch_ashare_statement_financial_history(
    *,
    start: date,
    end: date,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch bulk balance, cash-flow, and income statements by quarter."""
    import akshare as ak

    if force:
        for path in (
            _ASHARE_STATEMENT_FINANCIAL_CHECKPOINT_FILE,
            _ASHARE_STATEMENT_FINANCIAL_MANIFEST_FILE,
        ):
            if path.exists():
                path.unlink()
    checkpoint = _read_parquet(_ASHARE_STATEMENT_FINANCIAL_CHECKPOINT_FILE)
    records = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed_periods: set[date] = set()
    if _ASHARE_STATEMENT_FINANCIAL_MANIFEST_FILE.exists():
        try:
            manifest = json.loads(
                _ASHARE_STATEMENT_FINANCIAL_MANIFEST_FILE.read_text(encoding="utf-8")
            )
            completed_periods = {
                pd.Timestamp(value).date()
                for value in manifest.get("completed_periods", [])
            }
        except (OSError, ValueError, TypeError):
            logger.warning(
                "Could not read A-share statement checkpoint manifest",
                exc_info=True,
            )

    periods = [
        period
        for period in _quarter_end_dates(start, end)
        if period not in completed_periods
    ]
    logger.info(
        "Fetching bulk A-share statement history (%d report dates, %d resumed) ...",
        len(periods),
        len(completed_periods),
    )

    def fetch_frame(function, period: date) -> pd.DataFrame:
        for attempt in range(_FINANCIAL_FETCH_RETRIES):
            try:
                frame = function(date=period.strftime("%Y%m%d"))
                if frame is not None and not frame.empty:
                    return frame
            except Exception:
                if attempt + 1 >= _FINANCIAL_FETCH_RETRIES:
                    logger.warning(
                        "Bulk A-share statement fetch failed function=%s period=%s",
                        function.__name__,
                        period,
                        exc_info=True,
                    )
            time.sleep(1.0 * (2 ** attempt))
        return pd.DataFrame()

    for index, period in enumerate(periods, start=1):
        balance = fetch_frame(ak.stock_zcfz_em, period)
        cashflow = fetch_frame(ak.stock_xjll_em, period)
        income = fetch_frame(ak.stock_lrb_em, period)
        if balance.empty or cashflow.empty or income.empty:
            logger.warning("Incomplete bulk A-share statements for %s", period)
            continue
        period_records = _ashare_statement_financial_records_from_frames(
            period,
            balance=balance,
            cashflow=cashflow,
            income=income,
        )
        if not period_records:
            logger.warning("No bulk A-share statement rows for %s", period)
            continue
        records.extend(period_records)
        completed_periods.add(period)
        result = _ordered_financial_frame(_combine_financial_records(records))
        _write_parquet_atomic(result, _ASHARE_STATEMENT_FINANCIAL_CHECKPOINT_FILE)
        _write_json_atomic(
            {
                "completed_periods": [
                    value.isoformat() for value in sorted(completed_periods)
                ],
                "rows": int(len(result)),
                "tickers": int(result["ticker"].nunique()),
            },
            _ASHARE_STATEMENT_FINANCIAL_MANIFEST_FILE,
        )
        logger.info(
            "  Bulk A-share statements: %d/%d period=%s rows=%d tickers=%d",
            index,
            len(periods),
            period,
            len(result),
            result["ticker"].nunique(),
        )
        time.sleep(0.5)
    return _ordered_financial_frame(_combine_financial_records(records))


def _combine_financial_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        ticker = str(record.get("ticker") or "")
        period = str(record.get("period") or "")
        if not ticker or not period:
            continue
        key = (ticker, period)
        target = merged.setdefault(key, {"ticker": ticker, "period": period})
        for column in (
            "report_date",
            *_FINANCIAL_FEATURE_COLUMNS,
            *_FINANCIAL_DERIVATION_COLUMNS,
        ):
            value = record.get(column)
            if target.get(column) is None and value is not None and pd.notna(value):
                target[column] = value
    return list(merged.values())


def _fetch_ashare_financial_history(
    companies: list[Company],
    *,
    force: bool = False,
) -> pd.DataFrame:
    import akshare as ak

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]
    if not companies:
        return _ordered_financial_frame([])

    if force and _ASHARE_FINANCIAL_CHECKPOINT_FILE.exists():
        _ASHARE_FINANCIAL_CHECKPOINT_FILE.unlink()
    checkpoint = _read_parquet(_ASHARE_FINANCIAL_CHECKPOINT_FILE)
    records: list[dict[str, object]] = (
        checkpoint.to_dict("records") if not checkpoint.empty else []
    )
    completed_tickers = {
        str(ticker) for ticker in checkpoint.get("ticker", pd.Series(dtype=object)).dropna()
    }
    companies = [company for company in companies if company.ticker not in completed_tickers]
    lock = threading.Lock()
    counter = {
        "done": len(completed_tickers),
        "ok": len(completed_tickers),
        "fail": 0,
    }

    def _fetch_one(company: Company) -> None:
        ticker = company.ticker
        ticker_records: list[dict[str, object]] = []
        try:
            for attempt in range(_FINANCIAL_FETCH_RETRIES):
                time.sleep(_FINANCIAL_FETCH_DELAY * (2 ** attempt))
                try:
                    abstract = ak.stock_financial_abstract_ths(symbol=ticker)
                    ticker_records.extend(
                        _ashare_financial_records_from_frame(ticker, abstract)
                    )
                    if ticker_records:
                        break
                except Exception:
                    if attempt + 1 >= _FINANCIAL_FETCH_RETRIES:
                        logger.debug(
                            "A-share financial abstract unavailable for %s",
                            ticker,
                            exc_info=True,
                        )
            if not ticker_records:
                try:
                    indicator = ak.stock_financial_analysis_indicator(symbol=ticker)
                    ticker_records.extend(_ashare_financial_records_from_frame(ticker, indicator))
                except Exception:
                    logger.debug("A-share financial indicator unavailable for %s", ticker, exc_info=True)
            ticker_records = _combine_financial_records(ticker_records)
            if not ticker_records:
                raise ValueError("no financial history rows")
            with lock:
                records.extend(ticker_records)
                counter["ok"] += 1
        except Exception:
            logger.debug("Failed A-share financial history for %s", ticker, exc_info=True)
            with lock:
                counter["fail"] += 1
        finally:
            checkpoint_records: Optional[list[dict[str, object]]] = None
            with lock:
                counter["done"] += 1
                done = counter["done"]
                if done % _SAVE_INTERVAL == 0:
                    checkpoint_records = list(records)
            if checkpoint_records is not None:
                _write_parquet_atomic(
                    _ordered_financial_frame(checkpoint_records),
                    _ASHARE_FINANCIAL_CHECKPOINT_FILE,
                )
            if done % 200 == 0:
                logger.info(
                    "  A-share financial history: %d/%d (ok=%d, fail=%d)",
                    done, total, counter["ok"], counter["fail"],
                )

    logger.info(
        "Fetching A-share historical financial features (%d tickers, %d resumed) …",
        total,
        len(completed_tickers),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=_ASHARE_FINANCIAL_WORKERS) as executor:
        list(executor.map(_fetch_one, companies))
    logger.info(
        "A-share financial history complete: %d ok, %d fail",
        counter["ok"],
        counter["fail"],
    )
    result = _ordered_financial_frame(records)
    if not result.empty:
        _write_parquet_atomic(result, _ASHARE_FINANCIAL_CHECKPOINT_FILE)
    return result


def _statement_value(df: pd.DataFrame, period, aliases: Iterable[str]) -> Optional[float]:
    if df is None or df.empty or period not in df.columns:
        return None
    for alias in aliases:
        if alias not in df.index:
            continue
        value = _feature_float(df.loc[alias, period])
        if value is not None:
            return value
    return None


def _hk_financial_records_from_statements(
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    *,
    statement_months: int,
    currency_to_rmb: Optional[float],
) -> list[dict[str, object]]:
    if income is None or income.empty:
        return []
    records: list[dict[str, object]] = []
    for period in income.columns:
        period_label = _normalise_date(period)
        if period_label is None:
            continue
        revenue = _statement_value(income, period, ["Total Revenue"])
        net_income = _statement_value(income, period, ["Net Income"])
        gross_profit = _statement_value(income, period, ["Gross Profit"])
        total_assets = _statement_value(balance, period, ["Total Assets"])
        total_liabilities = _statement_value(
            balance,
            period,
            ["Total Liabilities Net Minority Interest", "Total Liab"],
        )
        total_equity = _statement_value(
            balance,
            period,
            ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
        )
        current_assets = _statement_value(balance, period, ["Current Assets"])
        current_liabilities = _statement_value(balance, period, ["Current Liabilities"])
        operating_cf = _statement_value(cashflow, period, ["Operating Cash Flow"])
        free_cf = _statement_value(cashflow, period, ["Free Cash Flow"])
        eps = _statement_value(income, period, ["Basic EPS", "Diluted EPS"])
        shares = _statement_value(
            balance,
            period,
            ["Ordinary Shares Number", "Share Issued"],
        )
        report_lag_days = 120 if statement_months >= 12 else 60
        records.append({
            "ticker": ticker,
            "period": period_label,
            "report_date": (
                pd.Timestamp(period_label) + pd.Timedelta(days=report_lag_days)
            ).date(),
            "revenue": revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "total_equity": total_equity,
            "operating_cash_flow": operating_cf,
            "free_cash_flow": free_cf,
            "gross_margin": (gross_profit / revenue) if gross_profit and revenue else None,
            "net_margin": (net_income / revenue) if net_income and revenue else None,
            "roe": (net_income / total_equity) if net_income and total_equity else None,
            "roa": (net_income / total_assets) if net_income and total_assets else None,
            "debt_to_equity": (
                total_liabilities / total_equity
                if total_liabilities and total_equity
                else None
            ),
            "current_ratio": (
                current_assets / current_liabilities
                if current_assets and current_liabilities
                else None
            ),
            "eps": eps,
            "book_value_per_share": (
                total_equity / shares
                if total_equity and shares and shares > 0
                else None
            ),
            "operating_cash_flow_per_share": (
                operating_cf / shares
                if operating_cf and shares and shares > 0
                else None
            ),
            "statement_months": statement_months,
            "currency_to_rmb": currency_to_rmb,
        })
    return records


def _fetch_hkshare_financial_history(
    companies: list[Company],
    *,
    force: bool = False,
) -> pd.DataFrame:
    import yfinance as yf

    from valueinvestor.data.fetcher_hkshare import _quiet_yfinance_errors, _to_yf_ticker

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]
    if not companies:
        return _ordered_financial_frame([])

    if force and _HK_FINANCIAL_CHECKPOINT_FILE.exists():
        _HK_FINANCIAL_CHECKPOINT_FILE.unlink()
    checkpoint = _read_parquet(_HK_FINANCIAL_CHECKPOINT_FILE)
    records: list[dict[str, object]] = (
        checkpoint.to_dict("records") if not checkpoint.empty else []
    )
    completed_tickers = {
        str(ticker) for ticker in checkpoint.get("ticker", pd.Series(dtype=object)).dropna()
    }
    companies = [
        company
        for company in companies
        if _to_yf_ticker(company.ticker) not in completed_tickers
    ]
    lock = threading.Lock()
    counter = {
        "done": len(completed_tickers),
        "ok": len(completed_tickers),
        "fail": 0,
    }

    def _fetch_one(company: Company) -> None:
        ticker = _to_yf_ticker(company.ticker)
        try:
            time.sleep(_FINANCIAL_FETCH_DELAY)
            with _quiet_yfinance_errors():
                yf_ticker = yf.Ticker(ticker)
                try:
                    info = yf_ticker.info
                except Exception:
                    info = {}
                currency_to_rmb = _financial_currency_to_rmb(
                    info.get("financialCurrency") if isinstance(info, dict) else None
                )
                statement_groups = (
                    (
                        yf_ticker.financials,
                        yf_ticker.balance_sheet,
                        yf_ticker.cashflow,
                        12,
                    ),
                    (
                        yf_ticker.quarterly_financials,
                        yf_ticker.quarterly_balance_sheet,
                        yf_ticker.quarterly_cashflow,
                        3,
                    ),
                )
            ticker_records: list[dict[str, object]] = []
            for income, balance, cashflow, statement_months in statement_groups:
                ticker_records.extend(
                    _hk_financial_records_from_statements(
                        ticker,
                        income,
                        balance,
                        cashflow,
                        statement_months=statement_months,
                        currency_to_rmb=currency_to_rmb,
                    )
                )
            ticker_records = _combine_financial_records(ticker_records)
            if not ticker_records:
                raise ValueError("no financial history rows")
            with lock:
                records.extend(ticker_records)
                counter["ok"] += 1
        except Exception:
            logger.debug("Failed HK financial history for %s", ticker, exc_info=True)
            with lock:
                counter["fail"] += 1
        finally:
            checkpoint_records: Optional[list[dict[str, object]]] = None
            with lock:
                counter["done"] += 1
                done = counter["done"]
                if done % 100 == 0:
                    checkpoint_records = list(records)
            if checkpoint_records is not None:
                _write_parquet_atomic(
                    _ordered_financial_frame(checkpoint_records),
                    _HK_FINANCIAL_CHECKPOINT_FILE,
                )
            if done % 100 == 0:
                logger.info(
                    "  HK financial history: %d/%d (ok=%d, fail=%d)",
                    done, total, counter["ok"], counter["fail"],
                )

    logger.info(
        "Fetching HK historical financial features (%d tickers, %d resumed) …",
        total,
        len(completed_tickers),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=_HKSHARE_FINANCIAL_WORKERS) as executor:
        list(executor.map(_fetch_one, companies))
    logger.info(
        "HK financial history complete: %d ok, %d fail",
        counter["ok"],
        counter["fail"],
    )
    result = _ordered_financial_frame(records)
    if not result.empty:
        _write_parquet_atomic(result, _HK_FINANCIAL_CHECKPOINT_FILE)
    return result


def _merge_financial_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame in frames:
        if frame.empty:
            continue
        records.extend(frame.to_dict("records"))
    return _ordered_financial_frame(_combine_financial_records(records))


def _ttm_from_cumulative(group: pd.DataFrame, column: str) -> pd.Series:
    """Convert year-to-date statement values to trailing-twelve-month values."""
    periods = pd.to_datetime(group["period"], errors="coerce")
    values = pd.to_numeric(group[column], errors="coerce")
    lookup = {
        (int(period.year), int(period.month)): float(value)
        for period, value in zip(periods, values)
        if pd.notna(period) and pd.notna(value)
    }
    annualization = {3: 4.0, 6: 2.0, 9: 4.0 / 3.0}
    output = pd.Series(float("nan"), index=group.index, dtype="float64")
    for index, period, value in zip(group.index, periods, values):
        if pd.isna(period) or pd.isna(value):
            continue
        month = int(period.month)
        current = float(value)
        if month == 12:
            output.loc[index] = current
            continue
        previous_annual = lookup.get((int(period.year) - 1, 12))
        previous_same_period = lookup.get((int(period.year) - 1, month))
        if previous_annual is not None and previous_same_period is not None:
            output.loc[index] = current + previous_annual - previous_same_period
        elif month in annualization:
            output.loc[index] = current * annualization[month]
    return output


def _ttm_financial_metric(group: pd.DataFrame, column: str) -> pd.Series:
    """Build TTM values for cumulative A-share and duration-tagged HK statements."""
    output = _ttm_from_cumulative(group, column)
    if "statement_months" not in group.columns:
        return output

    statement_months = pd.to_numeric(group["statement_months"], errors="coerce")
    values = pd.to_numeric(group[column], errors="coerce")
    annual = statement_months >= 10
    output.loc[annual & values.notna()] = values.loc[annual & values.notna()]

    quarterly = group.loc[statement_months.between(1, 4) & values.notna()].copy()
    if quarterly.empty:
        return output
    quarterly["_value"] = pd.to_numeric(quarterly[column], errors="coerce")
    quarterly["_period"] = pd.to_datetime(quarterly["period"], errors="coerce")
    quarterly = quarterly.dropna(subset=["_value", "_period"]).sort_values(
        "_period",
        kind="mergesort",
    )
    rolling = quarterly["_value"].rolling(window=4, min_periods=4).sum()
    output.loc[quarterly.index] = rolling.to_numpy(dtype="float64")
    return output


def _derive_historical_valuation_ratios(
    valuations: pd.DataFrame,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """Fill historical PE/PB/PS from prices and safely available statements."""
    valuations = _ordered_valuation_frame(valuations.to_dict("records"))
    if valuations.empty or financials.empty:
        return _ordered_valuation_frame(valuations.to_dict("records"))
    required = {"ticker", "period", "report_date"}
    if not required.issubset(financials.columns):
        return _ordered_valuation_frame(valuations.to_dict("records"))

    metrics = financials.copy()
    metrics["period"] = pd.to_datetime(
        metrics["period"],
        errors="coerce",
    ).astype("datetime64[ns]")
    metrics["report_date"] = pd.to_datetime(
        metrics["report_date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    metrics = metrics.dropna(subset=["ticker", "period", "report_date"])
    if metrics.empty:
        return _ordered_valuation_frame(valuations.to_dict("records"))
    metrics["_eps_ttm"] = float("nan")
    metrics["_net_income_ttm"] = float("nan")
    metrics["_revenue_ttm"] = float("nan")
    for _ticker, index in metrics.groupby("ticker", sort=False).groups.items():
        group = metrics.loc[index]
        if "eps" in metrics.columns:
            metrics.loc[index, "_eps_ttm"] = _ttm_financial_metric(group, "eps")
        if "net_income" in metrics.columns:
            metrics.loc[index, "_net_income_ttm"] = _ttm_financial_metric(
                group,
                "net_income",
            )
        if "revenue" in metrics.columns:
            metrics.loc[index, "_revenue_ttm"] = _ttm_financial_metric(
                group,
                "revenue",
            )

    result = valuations.copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    available = pd.DataFrame(
        index=result.index,
        columns=[
            "_eps_ttm",
            "book_value_per_share",
            "_net_income_ttm",
            "total_equity",
            "_revenue_ttm",
            "currency_to_rmb",
        ],
        dtype="float64",
    )
    metric_groups = {
        str(ticker): group.sort_values("report_date", kind="mergesort")
        for ticker, group in metrics.groupby("ticker", sort=False)
    }
    for ticker, index in result.groupby("ticker", sort=False).groups.items():
        history = metric_groups.get(str(ticker))
        if history is None or history.empty:
            continue
        left = result.loc[index, ["date"]].copy()
        left["_row_id"] = left.index
        merged = pd.merge_asof(
            left.sort_values("date", kind="mergesort"),
            history[
                ["report_date", *available.columns]
            ].sort_values("report_date", kind="mergesort"),
            left_on="date",
            right_on="report_date",
            direction="backward",
        ).set_index("_row_id")
        available.loc[merged.index, available.columns] = merged[available.columns]

    price = pd.to_numeric(result["price"], errors="coerce")
    market_cap = pd.to_numeric(result["market_cap_rmb"], errors="coerce")
    eps_ttm = pd.to_numeric(available["_eps_ttm"], errors="coerce")
    book_value = pd.to_numeric(available["book_value_per_share"], errors="coerce")
    net_income_ttm = pd.to_numeric(available["_net_income_ttm"], errors="coerce")
    total_equity = pd.to_numeric(available["total_equity"], errors="coerce")
    revenue_ttm = pd.to_numeric(available["_revenue_ttm"], errors="coerce")
    currency_to_rmb = pd.to_numeric(available["currency_to_rmb"], errors="coerce")
    ashare_rows = ~result["ticker"].astype(str).str.endswith(".HK")
    currency_to_rmb = currency_to_rmb.where(~ashare_rows, 1.0)
    quote_currency_to_rmb = pd.Series(
        np.where(
            ashare_rows,
            1.0,
            _FINANCIAL_CURRENCY_TO_RMB["HKD"],
        ),
        index=result.index,
        dtype="float64",
    )
    eps_in_quote_currency = eps_ttm * currency_to_rmb / quote_currency_to_rmb
    book_value_in_quote_currency = (
        book_value * currency_to_rmb / quote_currency_to_rmb
    )
    derived_pe = (price / eps_in_quote_currency).where(
        (price > 0) & (eps_in_quote_currency > 0)
    )
    derived_pb = (price / book_value_in_quote_currency).where(
        (price > 0) & (book_value_in_quote_currency > 0)
    )
    statement_pe = (market_cap / (net_income_ttm * currency_to_rmb)).where(
        (market_cap > 0) & (net_income_ttm > 0) & (currency_to_rmb > 0)
    )
    statement_pb = (market_cap / (total_equity * currency_to_rmb)).where(
        (market_cap > 0) & (total_equity > 0) & (currency_to_rmb > 0)
    )
    derived_ps = (market_cap / (revenue_ttm * currency_to_rmb)).where(
        (market_cap > 0) & (revenue_ttm > 0) & (currency_to_rmb > 0)
    )
    derived_pe = derived_pe.fillna(statement_pe)
    derived_pb = derived_pb.fillna(statement_pb)
    result["pe_ratio"] = pd.to_numeric(result["pe_ratio"], errors="coerce").fillna(
        derived_pe
    )
    result["pb_ratio"] = pd.to_numeric(result["pb_ratio"], errors="coerce").fillna(
        derived_pb
    )
    result["ps_ratio"] = pd.to_numeric(result["ps_ratio"], errors="coerce").fillna(
        derived_ps
    )
    result["date"] = result["date"].dt.date
    return _ordered_valuation_frame(result.to_dict("records"))


def _feature_source_starts_before(path: Path, columns: tuple[str, ...], cutoff: date) -> bool:
    df = _read_parquet(path)
    if df.empty:
        return False
    starts = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce")
        if values.notna().any():
            starts.append(values.min().date())
    return bool(starts) and min(starts) <= cutoff


def _feature_source_has_values_before(
    path: Path,
    *,
    date_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    cutoff: date,
    minimum_rows: int = 100,
) -> bool:
    df = _read_parquet(path)
    if df.empty or not set(feature_columns).issubset(df.columns):
        return False
    date_values: Optional[pd.Series] = None
    for column in date_columns:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            date_values = parsed
            break
    if date_values is None:
        return False
    mask = date_values.dt.date <= cutoff
    for column in feature_columns:
        mask &= pd.to_numeric(df[column], errors="coerce").notna()
    return int(mask.sum()) >= minimum_rows


def _point_in_time_features_ready(cutoff: date) -> bool:
    return (
        _feature_source_has_values_before(
            _VALUATIONS_FILE,
            date_columns=("date",),
            feature_columns=("pe_ratio", "pb_ratio", "market_cap_rmb"),
            cutoff=cutoff,
        )
        and _feature_source_has_values_before(
            _FINANCIALS_FILE,
            date_columns=("report_date", "period"),
            feature_columns=("roe",),
            cutoff=cutoff,
        )
    )


def build_point_in_time_feature_data(
    *,
    a_companies: Optional[list[Company]] = None,
    hk_companies: Optional[list[Company]] = None,
    a_prices: Optional[pd.DataFrame] = None,
    hk_prices: Optional[pd.DataFrame] = None,
    force: bool = False,
    fetch_remote_history: bool = True,
    fetch_market_cap_history: bool = False,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Dict[str, Path]:
    """Build historical feature files used by daily ML snapshot training."""
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today()
    start = start or (end - timedelta(days=_HISTORY_YEARS * 365))
    files: Dict[str, Path] = {}

    a_prices = a_prices if a_prices is not None else _read_parquet(_ASHARE_PRICES_FILE)
    hk_prices = hk_prices if hk_prices is not None else _read_parquet(_HKSHARE_PRICES_FILE)
    price_frames = [frame for frame in (a_prices, hk_prices) if frame is not None and not frame.empty]
    all_prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()

    valuation_frames: list[pd.DataFrame] = []
    if not all_prices.empty:
        valuation_frames.append(_price_valuation_skeleton(all_prices, start=start, end=end))
    if fetch_remote_history and fetch_market_cap_history and a_companies:
        market_cap_history = _fetch_ashare_market_cap_history(a_companies, start=start, end=end)
        if not market_cap_history.empty:
            valuation_frames.append(market_cap_history)

    valuation_records: list[dict[str, object]] = []
    for frame in valuation_frames:
        if not frame.empty:
            valuation_records.extend(frame.to_dict("records"))
    valuations = _ordered_valuation_frame(valuation_records)

    existing_financials = _read_parquet(_FINANCIALS_FILE)
    financial_frames: list[pd.DataFrame] = []
    if fetch_remote_history:
        if a_companies:
            financial_frames.append(
                _fetch_ashare_financial_history_bulk(
                    start=start,
                    end=end,
                    force=force,
                )
            )
            financial_frames.append(
                _fetch_ashare_statement_financial_history(
                    start=start,
                    end=end,
                    force=force,
                )
            )
            if os.environ.get("VALUEINVESTOR_FETCH_PER_TICKER_FINANCIALS", "0") == "1":
                financial_frames.append(
                    _fetch_ashare_financial_history(a_companies, force=force)
                )
        if hk_companies:
            financial_frames.append(
                _fetch_hkshare_financial_history(hk_companies, force=force)
            )
    for checkpoint_path in (
        _ASHARE_BULK_FINANCIAL_CHECKPOINT_FILE,
        _ASHARE_STATEMENT_FINANCIAL_CHECKPOINT_FILE,
        _ASHARE_FINANCIAL_CHECKPOINT_FILE,
        _HK_FINANCIAL_CHECKPOINT_FILE,
    ):
        checkpoint = _read_parquet(checkpoint_path)
        if not checkpoint.empty:
            financial_frames.append(checkpoint)
    if not force and not existing_financials.empty:
        financial_frames.append(existing_financials)
    financials = _merge_financial_frames(financial_frames)
    if financials.empty and not existing_financials.empty:
        financials = _ordered_financial_frame(existing_financials.to_dict("records"))

    if not financials.empty:
        _write_parquet_atomic(financials, _FINANCIALS_FILE)
        files["financials"] = _FINANCIALS_FILE
        logger.info(
            "Saved point-in-time financials → %s (%d rows, %d tickers, %s→%s)",
            _FINANCIALS_FILE,
            len(financials),
            financials["ticker"].nunique(),
            financials["period"].min(),
            financials["period"].max(),
        )

    valuations = _derive_historical_valuation_ratios(valuations, financials)
    if not valuations.empty:
        _write_parquet_atomic(valuations, _VALUATIONS_FILE)
        files["valuations"] = _VALUATIONS_FILE
        logger.info(
            "Saved point-in-time valuations → %s (%d rows, %d tickers, %s→%s; "
            "PE=%d, PB=%d)",
            _VALUATIONS_FILE,
            len(valuations),
            valuations["ticker"].nunique(),
            valuations["date"].min(),
            valuations["date"].max(),
            int(valuations["pe_ratio"].notna().sum()),
            int(valuations["pb_ratio"].notna().sum()),
        )

    return files


# -----------------------------------------------------------------------
# Valuations & financials from existing cache
# -----------------------------------------------------------------------

def _export_cache_data(cache_db_path: str = "data/cache.db") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export valuation and financial data from the main cache.db as DataFrames."""
    import json

    cache_path = Path(cache_db_path)
    if not cache_path.exists():
        logger.warning("Cache database not found at %s", cache_db_path)
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(str(cache_path))

    # Valuations
    val_rows = conn.execute("SELECT ticker, data FROM valuations").fetchall()
    val_records = []
    for ticker, data_json in val_rows:
        try:
            d = json.loads(data_json)
            d["ticker"] = ticker
            val_records.append(d)
        except Exception:
            pass

    # Financials
    fin_rows = conn.execute("SELECT ticker, data FROM financials").fetchall()
    fin_records = []
    for ticker, data_json in fin_rows:
        try:
            d = json.loads(data_json)
            d["ticker"] = ticker
            fin_records.append(d)
        except Exception:
            pass

    conn.close()

    val_df = pd.DataFrame(val_records) if val_records else pd.DataFrame()
    fin_df = pd.DataFrame(fin_records) if fin_records else pd.DataFrame()

    logger.info("Exported from cache: %d valuations, %d financials", len(val_df), len(fin_df))
    return val_df, fin_df


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def fetch_training_data(force: bool = False) -> Dict[str, Path]:
    """Fetch all training data and store to disk.

    Uses differential download: only fetches date ranges not already present
    in the existing Parquet files.  Re-running is fast and idempotent.

    Parameters
    ----------
    force : bool
        If True, discard existing data and re-fetch from scratch.
    """
    conn = _init_db()
    files: Dict[str, Path] = {}
    today = date.today()
    ten_years_ago = today - timedelta(days=_HISTORY_YEARS * 365)
    start_date = ten_years_ago.strftime("%Y%m%d")
    end_date   = today.strftime("%Y%m%d")

    # Skip only when stored data_start_date covers the required window
    stored_start = _get_fetch_status(conn, "data_start_date")
    last_fetch   = _get_fetch_status(conn, "last_full_fetch")
    data_complete = _training_data_complete(
        force=force,
        last_fetch=last_fetch,
        stored_start=stored_start,
        required_start=start_date,          # covers the full 10-year window
        required_fetch_date=today.isoformat(),
        required_files=(_ASHARE_PRICES_FILE, _HKSHARE_PRICES_FILE),
    )
    if data_complete:
        logger.info(
            "Training data already complete (start=%s, last_fetch=%s, required_fetch=%s). "
            "Use --force to re-fetch.",
            stored_start, last_fetch, today.isoformat(),
        )
        files["ashare_prices"] = _ASHARE_PRICES_FILE
        files["hkshare_prices"] = _HKSHARE_PRICES_FILE
        feature_cutoff = ten_years_ago + timedelta(days=2 * 365)
        if _point_in_time_features_ready(feature_cutoff):
            if _VALUATIONS_FILE.exists():
                files["valuations"] = _VALUATIONS_FILE
            if _FINANCIALS_FILE.exists():
                files["financials"] = _FINANCIALS_FILE
        else:
            logger.info("Point-in-time feature files are stale; rebuilding historical features …")
            a_companies, hk_companies = _load_stored_universe(conn)
            feature_files = build_point_in_time_feature_data(
                a_companies=a_companies,
                hk_companies=hk_companies,
                force=False,
                fetch_remote_history=True,
                start=ten_years_ago,
                end=today,
            )
            files.update(feature_files)
        conn.close()
        return files

    # Fetch universe
    a_companies, hk_companies = _fetch_universe()
    _store_universe(conn, a_companies + hk_companies)

    # Fetch A-share prices (differential)
    logger.info("Fetching A-share price history (%s → %s, differential) …", start_date, end_date)
    a_prices = _fetch_ashare_prices(a_companies, start_date, end_date)
    if not a_prices.empty:
        a_prices.to_parquet(str(_ASHARE_PRICES_FILE), index=False)
        logger.info("Saved A-share prices → %s (%d rows)", _ASHARE_PRICES_FILE, len(a_prices))
    files["ashare_prices"] = _ASHARE_PRICES_FILE

    # Fetch HK-share prices (differential)
    logger.info("Fetching HK-share price history (%s → %s, differential) …", start_date, end_date)
    hk_prices = _fetch_hkshare_prices(hk_companies, start_date, end_date)
    if not hk_prices.empty:
        hk_prices.to_parquet(str(_HKSHARE_PRICES_FILE), index=False)
        logger.info("Saved HK-share prices → %s (%d rows)", _HKSHARE_PRICES_FILE, len(hk_prices))
    files["hkshare_prices"] = _HKSHARE_PRICES_FILE

    # Build point-in-time feature sources for historical training snapshots.
    feature_files = build_point_in_time_feature_data(
        a_companies=a_companies,
        hk_companies=hk_companies,
        a_prices=a_prices,
        hk_prices=hk_prices,
        force=force,
        fetch_remote_history=True,
        start=ten_years_ago,
        end=today,
    )
    files.update(feature_files)

    _set_fetch_status(conn, "last_full_fetch", today.isoformat())
    _set_fetch_status(conn, "data_start_date", start_date)   # persist coverage info
    conn.close()

    logger.info("Training data fetch complete.")
    return files
