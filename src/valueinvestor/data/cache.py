"""SQLite caching layer for market data, financials, and analyses."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from valueinvestor.data.models import (
    Company,
    CompanyAnalysis,
    Financials,
    ValuationMetrics,
)
from valueinvestor.errors import CacheError

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS companies (
    ticker    TEXT PRIMARY KEY,
    data      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS financials (
    ticker    TEXT PRIMARY KEY,
    data      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS valuations (
    ticker    TEXT PRIMARY KEY,
    data      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    ticker    TEXT PRIMARY KEY,
    data      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stock_lists (
    market    TEXT PRIMARY KEY,
    data      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class DataCache:
    """Thread-safe SQLite cache with TTL-based expiration."""

    def __init__(self, db_path: str = "data/cache.db", ttl_hours: int = 24) -> None:
        self.db_path = db_path
        self.ttl = timedelta(hours=ttl_hours)

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as conn:
                conn.executescript(_CREATE_TABLES)
        except sqlite3.Error as exc:
            raise CacheError(f"Failed to initialise cache database at {db_path}") from exc

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise CacheError(f"Cannot connect to cache database: {self.db_path}") from exc
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError("Cache database operation failed") from exc
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def is_expired(self, updated_at: str) -> bool:
        """Return ``True`` if *updated_at* is older than the configured TTL."""
        ts = datetime.fromisoformat(updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - ts > self.ttl

    # ------------------------------------------------------------------
    # Company
    # ------------------------------------------------------------------

    def get_company(self, ticker: str) -> Optional[Company]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, updated_at FROM companies WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None or self.is_expired(row[1]):
            return None
        return Company.model_validate_json(row[0])

    def set_company(self, company: Company) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO companies (ticker, data, updated_at) VALUES (?, ?, ?)",
                (company.ticker, company.model_dump_json(), self._now()),
            )

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------

    def get_financials(self, ticker: str) -> Optional[Financials]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, updated_at FROM financials WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None or self.is_expired(row[1]):
            return None
        return Financials.model_validate_json(row[0])

    def set_financials(self, financials: Financials) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO financials (ticker, data, updated_at) VALUES (?, ?, ?)",
                (financials.ticker, financials.model_dump_json(), self._now()),
            )

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def get_valuation(self, ticker: str) -> Optional[ValuationMetrics]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, updated_at FROM valuations WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None or self.is_expired(row[1]):
            return None
        return ValuationMetrics.model_validate_json(row[0])

    def set_valuation(self, valuation: ValuationMetrics) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO valuations (ticker, data, updated_at) VALUES (?, ?, ?)",
                (valuation.ticker, valuation.model_dump_json(), self._now()),
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_analysis(self, ticker: str) -> Optional[CompanyAnalysis]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, updated_at FROM analyses WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None or self.is_expired(row[1]):
            return None
        return CompanyAnalysis.model_validate_json(row[0])

    def set_analysis(self, analysis: CompanyAnalysis) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analyses (ticker, data, updated_at) VALUES (?, ?, ?)",
                (analysis.ticker, analysis.model_dump_json(), self._now()),
            )

    # ------------------------------------------------------------------
    # Stock lists
    # ------------------------------------------------------------------

    def get_stock_list(self, market: str) -> Optional[List[Company]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, updated_at FROM stock_lists WHERE market = ?",
                (market,),
            ).fetchone()
        if row is None or self.is_expired(row[1]):
            return None
        raw_list: list = json.loads(row[0])
        return [Company.model_validate(item) for item in raw_list]

    def set_stock_list(self, market: str, companies: List[Company]) -> None:
        blob = json.dumps([c.model_dump(mode="json") for c in companies])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stock_lists (market, data, updated_at) VALUES (?, ?, ?)",
                (market, blob, self._now()),
            )

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def list_analyses(self) -> List[CompanyAnalysis]:
        """Return all non-expired analyses."""
        with self._connect() as conn:
            rows = conn.execute("SELECT data, updated_at FROM analyses").fetchall()
        return [
            CompanyAnalysis.model_validate_json(row[0])
            for row in rows
            if not self.is_expired(row[1])
        ]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all cached data."""
        with self._connect() as conn:
            for table in ("companies", "financials", "valuations", "analyses", "stock_lists"):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608

    def clear_expired(self) -> None:
        """Remove only entries whose TTL has elapsed."""
        cutoff = (datetime.now(timezone.utc) - self.ttl).isoformat()
        with self._connect() as conn:
            for table in ("companies", "financials", "valuations", "analyses", "stock_lists"):
                conn.execute(
                    f"DELETE FROM {table} WHERE updated_at < ?",  # noqa: S608
                    (cutoff,),
                )
