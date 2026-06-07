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
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    return (
        df.sort_values(["ticker", "date"], kind="mergesort")
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def _ordered_financial_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["ticker", "period", *_FINANCIAL_FEATURE_COLUMNS]
    df = pd.DataFrame(records)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df[columns].copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.date
    df = df.dropna(subset=["ticker", "period"])
    for column in _FINANCIAL_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_nonnull"] = df[list(_FINANCIAL_FEATURE_COLUMNS)].notna().sum(axis=1)
    df = (
        df.sort_values(["ticker", "period", "_nonnull"], kind="mergesort")
        .drop_duplicates(["ticker", "period"], keep="last")
        .drop(columns=["_nonnull"])
        .reset_index(drop=True)
    )
    return df


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
                ["经营现金流", "经营活动现金流量净额", "每股经营现金流", "operating_cash_flow"],
            ),
            "free_cash_flow": _find_feature(row, ["自由现金流", "free_cash_flow"]),
            "gross_margin": _find_feature(row, ["毛利率", "销售毛利率", "gross_margin"]),
            "net_margin": _find_feature(row, ["净利率", "销售净利率", "net_margin"]),
            "roe": _find_feature(row, ["净资产收益率", "摊薄净资产收益率", "ROE", "roe"]),
            "roa": _find_feature(row, ["总资产收益率", "ROA", "roa"]),
            "debt_to_equity": _find_feature(row, ["资产负债率", "debt_to_equity"]),
            "current_ratio": _find_feature(row, ["流动比率", "current_ratio"]),
        })
    return records


def _combine_financial_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        ticker = str(record.get("ticker") or "")
        period = str(record.get("period") or "")
        if not ticker or not period:
            continue
        key = (ticker, period)
        target = merged.setdefault(key, {"ticker": ticker, "period": period})
        for column in _FINANCIAL_FEATURE_COLUMNS:
            value = record.get(column)
            if target.get(column) is None and value is not None and pd.notna(value):
                target[column] = value
    return list(merged.values())


def _fetch_ashare_financial_history(companies: list[Company]) -> pd.DataFrame:
    import akshare as ak

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]
    if not companies:
        return _ordered_financial_frame([])

    records: list[dict[str, object]] = []
    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "fail": 0}

    def _fetch_one(company: Company) -> None:
        ticker = company.ticker
        ticker_records: list[dict[str, object]] = []
        try:
            time.sleep(_FINANCIAL_FETCH_DELAY)
            try:
                abstract = ak.stock_financial_abstract_ths(symbol=ticker)
                ticker_records.extend(_ashare_financial_records_from_frame(ticker, abstract))
            except Exception:
                logger.debug("A-share financial abstract unavailable for %s", ticker, exc_info=True)
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
            with lock:
                counter["done"] += 1
                done = counter["done"]
            if done % 200 == 0:
                logger.info(
                    "  A-share financial history: %d/%d (ok=%d, fail=%d)",
                    done, total, counter["ok"], counter["fail"],
                )

    logger.info("Fetching A-share historical financial features (%d tickers) …", total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_ASHARE_FINANCIAL_WORKERS) as executor:
        list(executor.map(_fetch_one, companies))
    logger.info(
        "A-share financial history complete: %d ok, %d fail",
        counter["ok"],
        counter["fail"],
    )
    return _ordered_financial_frame(records)


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
        records.append({
            "ticker": ticker,
            "period": period_label,
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
        })
    return records


def _fetch_hkshare_financial_history(companies: list[Company]) -> pd.DataFrame:
    import yfinance as yf

    from valueinvestor.data.fetcher_hkshare import _quiet_yfinance_errors, _to_yf_ticker

    total = len(companies) if _MAX_STOCKS == 0 else min(_MAX_STOCKS, len(companies))
    companies = companies[:total]
    if not companies:
        return _ordered_financial_frame([])

    records: list[dict[str, object]] = []
    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "fail": 0}

    def _fetch_one(company: Company) -> None:
        ticker = _to_yf_ticker(company.ticker)
        try:
            with _quiet_yfinance_errors():
                yf_ticker = yf.Ticker(ticker)
                statement_groups = (
                    (
                        yf_ticker.financials,
                        yf_ticker.balance_sheet,
                        yf_ticker.cashflow,
                    ),
                    (
                        yf_ticker.quarterly_financials,
                        yf_ticker.quarterly_balance_sheet,
                        yf_ticker.quarterly_cashflow,
                    ),
                )
            ticker_records: list[dict[str, object]] = []
            for income, balance, cashflow in statement_groups:
                ticker_records.extend(
                    _hk_financial_records_from_statements(
                        ticker,
                        income,
                        balance,
                        cashflow,
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
            with lock:
                counter["done"] += 1
                done = counter["done"]
            if done % 100 == 0:
                logger.info(
                    "  HK financial history: %d/%d (ok=%d, fail=%d)",
                    done, total, counter["ok"], counter["fail"],
                )

    logger.info("Fetching HK historical financial features (%d tickers) …", total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_HKSHARE_FINANCIAL_WORKERS) as executor:
        list(executor.map(_fetch_one, companies))
    logger.info(
        "HK financial history complete: %d ok, %d fail",
        counter["ok"],
        counter["fail"],
    )
    return _ordered_financial_frame(records)


def _merge_financial_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame in frames:
        if frame.empty:
            continue
        records.extend(frame.to_dict("records"))
    return _ordered_financial_frame(_combine_financial_records(records))


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


def _point_in_time_features_ready(cutoff: date) -> bool:
    return (
        _feature_source_starts_before(_VALUATIONS_FILE, ("date",), cutoff)
        and _feature_source_starts_before(_FINANCIALS_FILE, ("period", "report_date"), cutoff)
    )


def build_point_in_time_feature_data(
    *,
    a_companies: Optional[list[Company]] = None,
    hk_companies: Optional[list[Company]] = None,
    a_prices: Optional[pd.DataFrame] = None,
    hk_prices: Optional[pd.DataFrame] = None,
    force: bool = False,
    fetch_remote_history: bool = True,
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
    if fetch_remote_history and a_companies:
        market_cap_history = _fetch_ashare_market_cap_history(a_companies, start=start, end=end)
        if not market_cap_history.empty:
            valuation_frames.append(market_cap_history)

    valuation_records: list[dict[str, object]] = []
    for frame in valuation_frames:
        if not frame.empty:
            valuation_records.extend(frame.to_dict("records"))
    valuations = _ordered_valuation_frame(valuation_records)
    if not valuations.empty:
        valuations.to_parquet(str(_VALUATIONS_FILE), index=False)
        files["valuations"] = _VALUATIONS_FILE
        logger.info(
            "Saved point-in-time valuations → %s (%d rows, %d tickers, %s→%s)",
            _VALUATIONS_FILE,
            len(valuations),
            valuations["ticker"].nunique(),
            valuations["date"].min(),
            valuations["date"].max(),
        )

    existing_financials = _read_parquet(_FINANCIALS_FILE)
    financial_frames: list[pd.DataFrame] = []
    if fetch_remote_history:
        if a_companies:
            financial_frames.append(_fetch_ashare_financial_history(a_companies))
        if hk_companies:
            financial_frames.append(_fetch_hkshare_financial_history(hk_companies))
    if not force and not existing_financials.empty:
        financial_frames.append(existing_financials)
    financials = _merge_financial_frames(financial_frames)
    if financials.empty and not existing_financials.empty:
        financials = _ordered_financial_frame(existing_financials.to_dict("records"))

    if not financials.empty:
        financials.to_parquet(str(_FINANCIALS_FILE), index=False)
        files["financials"] = _FINANCIALS_FILE
        logger.info(
            "Saved point-in-time financials → %s (%d rows, %d tickers, %s→%s)",
            _FINANCIALS_FILE,
            len(financials),
            financials["ticker"].nunique(),
            financials["period"].min(),
            financials["period"].max(),
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
        feature_cutoff = today - timedelta(days=365)
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
