"""Phase 1 — Fetch and store 3-year historical training data.

Fetches daily price history for A-share (Tencent via akshare) and HK-share
(yfinance) markets, along with valuations and financials snapshots.  Data is
stored as Parquet files in ``data/trainer/`` and metadata in
``data/trainer.db`` (SQLite).

Designed to be run once; re-running refreshes only stale data.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import date, timedelta
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

# Tencent history fetch rate limit (seconds between requests)
_TENCENT_DELAY = 0.15
# yfinance HK rate limit
_YF_DELAY = 0.3
# Max stocks to fetch (0 = all)
_MAX_STOCKS = 0

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
    """Fetch daily close prices for A-share stocks via Tencent (akshare).

    Returns a DataFrame with columns: ticker, date, open, close, high, low, volume.
    """
    import akshare as ak

    all_frames: list[pd.DataFrame] = []
    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]

    logger.info("Fetching A-share prices for %d stocks …", total)

    for idx, company in enumerate(companies):
        ticker = company.ticker
        prefix = "sh" if ticker.startswith("6") else "sz"
        symbol = f"{prefix}{ticker}"

        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "date": "date", "open": "open", "close": "close",
                    "high": "high", "low": "low", "amount": "volume",
                })
                df["ticker"] = ticker
                all_frames.append(df[["ticker", "date", "open", "close", "high", "low", "volume"]])
        except Exception:
            logger.debug("Failed to fetch A-share history for %s", ticker, exc_info=True)

        if (idx + 1) % 100 == 0:
            logger.info("  A-share prices: %d/%d fetched", idx + 1, total)
        time.sleep(_TENCENT_DELAY)

    if not all_frames:
        logger.warning("No A-share price data fetched")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    logger.info("A-share prices: %d rows for %d stocks", len(result), len(all_frames))
    return result


# -----------------------------------------------------------------------
# HK-share price history (yfinance — globally accessible)
# -----------------------------------------------------------------------

def _fetch_hkshare_prices(
    companies: List[Company],
    period: str = "3y",
) -> pd.DataFrame:
    """Fetch daily close prices for HK stocks via yfinance.

    Returns a DataFrame with columns: ticker, date, open, close, high, low, volume.
    """
    import yfinance as yf

    all_frames: list[pd.DataFrame] = []
    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]

    logger.info("Fetching HK-share prices for %d stocks …", total)

    for idx, company in enumerate(companies):
        ticker = company.ticker
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist is not None and not hist.empty:
                df = hist.reset_index()
                df = df.rename(columns={
                    "Date": "date", "Open": "open", "Close": "close",
                    "High": "high", "Low": "low", "Volume": "volume",
                })
                df["ticker"] = ticker
                # Remove timezone from date
                df["date"] = pd.to_datetime(df["date"]).dt.date
                all_frames.append(df[["ticker", "date", "open", "close", "high", "low", "volume"]])
        except Exception:
            logger.debug("Failed to fetch HK history for %s", ticker, exc_info=True)

        if (idx + 1) % 50 == 0:
            logger.info("  HK-share prices: %d/%d fetched", idx + 1, total)
        time.sleep(_YF_DELAY)

    if not all_frames:
        logger.warning("No HK-share price data fetched")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    logger.info("HK-share prices: %d rows for %d stocks", len(result), len(all_frames))
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

    Returns a dict mapping data type to file path.

    Parameters
    ----------
    force : bool
        If True, re-fetch even if data files already exist.
    """
    conn = _init_db()
    files: Dict[str, Path] = {}
    today = date.today()
    three_years_ago = today - timedelta(days=3 * 365)
    start_date = three_years_ago.strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    # Check if we already have data
    last_fetch = _get_fetch_status(conn, "last_full_fetch")
    if last_fetch and not force:
        if _ASHARE_PRICES_FILE.exists() and _HKSHARE_PRICES_FILE.exists():
            logger.info(
                "Training data already exists (last fetch: %s). "
                "Use --force to re-fetch.",
                last_fetch,
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

    # Fetch A-share prices
    if force or not _ASHARE_PRICES_FILE.exists():
        logger.info("Fetching A-share price history (%s → %s) …", start_date, end_date)
        a_prices = _fetch_ashare_prices(a_companies, start_date, end_date)
        if not a_prices.empty:
            a_prices.to_parquet(str(_ASHARE_PRICES_FILE), index=False)
            logger.info("Saved A-share prices → %s", _ASHARE_PRICES_FILE)
        files["ashare_prices"] = _ASHARE_PRICES_FILE
    else:
        files["ashare_prices"] = _ASHARE_PRICES_FILE

    # Fetch HK-share prices
    if force or not _HKSHARE_PRICES_FILE.exists():
        logger.info("Fetching HK-share price history (3y) …")
        hk_prices = _fetch_hkshare_prices(hk_companies, period="3y")
        if not hk_prices.empty:
            hk_prices.to_parquet(str(_HKSHARE_PRICES_FILE), index=False)
            logger.info("Saved HK-share prices → %s", _HKSHARE_PRICES_FILE)
        files["hkshare_prices"] = _HKSHARE_PRICES_FILE
    else:
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
    conn.close()

    logger.info("Training data fetch complete.")
    return files
