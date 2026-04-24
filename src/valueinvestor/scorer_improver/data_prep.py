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
import logging
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from valueinvestor.data.models import Company, Market

logger = logging.getLogger(__name__)

TRAINER_DIR = Path("data/trainer")
TRAINER_DB = Path("data/trainer.db")

_ASHARE_PRICES_FILE = TRAINER_DIR / "ashare_prices.parquet"
_HKSHARE_PRICES_FILE = TRAINER_DIR / "hkshare_prices.parquet"
_VALUATIONS_FILE = TRAINER_DIR / "valuations.parquet"
_FINANCIALS_FILE = TRAINER_DIR / "financials.parquet"

# Number of years of historical price data to collect for training
_HISTORY_YEARS = 10

# Tencent history parallel workers (balance speed vs rate limiting)
_TENCENT_WORKERS = 8
# yfinance HK parallel workers
_YF_WORKERS = 10
# Max stocks to fetch (0 = all)
_MAX_STOCKS = 0
# Incremental save interval: save partial results every N tasks
_SAVE_INTERVAL = 200
# Brief sleep between requests to avoid rate limiting
_TENCENT_DELAY = 0.05

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

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]

    # ── Load existing data ────────────────────────────────────────────
    existing_df: pd.DataFrame = pd.DataFrame()
    ticker_ranges: Dict[str, Tuple[str, str]] = {}
    if _HKSHARE_PRICES_FILE.exists():
        try:
            existing_df = pd.read_parquet(str(_HKSHARE_PRICES_FILE))
            ticker_ranges = _get_ticker_date_ranges(existing_df)
            logger.info(
                "HK-share existing data: %d tickers, %d rows",
                len(ticker_ranges), len(existing_df),
            )
        except Exception as exc:
            logger.warning("Could not read existing HK-share prices: %s — starting fresh", exc)

    # ── Build differential fetch plan ─────────────────────────────────
    # Each task is (company, yf_start_ISO, yf_end_ISO)
    fetch_tasks: List[Tuple[Company, str, str]] = []
    req_start = _parse_yyyymmdd(start_date)
    req_end   = _parse_yyyymmdd(end_date)
    yf_start  = _yyyymmdd_to_iso(start_date)
    # yfinance end is exclusive — add one day so today's data is included
    yf_end    = (req_end + timedelta(days=1)).strftime("%Y-%m-%d")

    for company in companies:
        ticker = company.ticker
        if ticker not in ticker_ranges:
            fetch_tasks.append((company, yf_start, yf_end))
            continue

        ex_min = _parse_yyyymmdd(ticker_ranges[ticker][0])
        ex_max = _parse_yyyymmdd(ticker_ranges[ticker][1])

        # Earlier portion missing?
        if (ex_min - req_start).days > 30:
            gap_yf_end = (ex_min).strftime("%Y-%m-%d")  # exclusive: fetch up to ex_min
            fetch_tasks.append((company, yf_start, gap_yf_end))

        # Recent portion missing?
        if (req_end - ex_max).days > 7:
            gap_yf_start = (ex_max + timedelta(days=1)).strftime("%Y-%m-%d")
            fetch_tasks.append((company, gap_yf_start, yf_end))

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

    def _fetch_one(task: Tuple[Company, str, str]) -> None:
        company, t_start, t_end = task
        ticker = company.ticker
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

    def _worker(task: Tuple[Company, str, str]) -> None:
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
    data_complete = (
        not force
        and last_fetch is not None
        and stored_start is not None
        and stored_start <= start_date          # covers the full 10-year window
        and _ASHARE_PRICES_FILE.exists()
        and _HKSHARE_PRICES_FILE.exists()
    )
    if data_complete:
        logger.info(
            "Training data already complete (start=%s, last_fetch=%s). "
            "Use --force to re-fetch.",
            stored_start, last_fetch,
        )
        files["ashare_prices"] = _ASHARE_PRICES_FILE
        files["hkshare_prices"] = _HKSHARE_PRICES_FILE
        if _VALUATIONS_FILE.exists():
            files["valuations"] = _VALUATIONS_FILE
        if _FINANCIALS_FILE.exists():
            files["financials"] = _FINANCIALS_FILE
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

    # Export valuations & financials from cache
    val_df, fin_df = _export_cache_data()
    if not val_df.empty:
        val_df.to_parquet(str(_VALUATIONS_FILE), index=False)
        files["valuations"] = _VALUATIONS_FILE
    if not fin_df.empty:
        fin_df.to_parquet(str(_FINANCIALS_FILE), index=False)
        files["financials"] = _FINANCIALS_FILE

    _set_fetch_status(conn, "last_full_fetch", today.isoformat())
    _set_fetch_status(conn, "data_start_date", start_date)   # persist coverage info
    conn.close()

    logger.info("Training data fetch complete.")
    return files
