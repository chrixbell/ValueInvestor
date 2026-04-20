"""Fetcher for Hong Kong Stock Exchange (HK-share) data via yfinance and akshare."""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf

from valueinvestor.data.models import Company, Financials, Market, ValuationMetrics
from valueinvestor.errors import DataFetchError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HKD_TO_RMB = 0.92

# Fallback list of major HK-listed tickers (top ~50 by market-cap) used when
# akshare is unavailable or fails.  Ticker format follows yfinance convention.
_FALLBACK_TICKERS = [
    "0700.HK", "9988.HK", "0941.HK", "1299.HK", "0005.HK",
    "2318.HK", "0388.HK", "0939.HK", "1398.HK", "3988.HK",
    "2628.HK", "0883.HK", "0001.HK", "0016.HK", "0003.HK",
    "0011.HK", "1928.HK", "0027.HK", "0688.HK", "0002.HK",
    "0006.HK", "0012.HK", "0017.HK", "0066.HK", "0101.HK",
    "0175.HK", "0241.HK", "0267.HK", "0288.HK", "0386.HK",
    "0669.HK", "0762.HK", "0823.HK", "0857.HK", "0868.HK",
    "0960.HK", "0968.HK", "0981.HK", "1038.HK", "1044.HK",
    "1088.HK", "1109.HK", "1113.HK", "1177.HK", "1211.HK",
    "1810.HK", "1876.HK", "1997.HK", "2007.HK", "2020.HK",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(mapping: dict, key: str, default=None):
    """Return *mapping[key]* if the value is not ``None`` / ``'N/A'``."""
    val = mapping.get(key, default)
    if val is None or val == "N/A":
        return default
    return val


def _to_yf_ticker(code: str) -> str:
    """Ensure a ticker string ends with ``.HK``.

    Accepts ``"0700"``, ``"0700.HK"``, or ``"00700"`` and normalises to
    yfinance HK format.
    """
    code = code.strip()
    if code.upper().endswith(".HK"):
        return code.upper()
    # Strip any leading zeros beyond 4 digits (akshare sometimes returns 5-digit codes)
    digits = code.lstrip("0") or "0"
    digits = digits.zfill(4)
    return f"{digits}.HK"


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class HKShareFetcher:
    """Fetch Hong Kong-listed stock data using *yfinance* (and optionally *akshare*)."""

    # -----------------------------------------------------------------
    # Stock list
    # -----------------------------------------------------------------

    def fetch_stock_list(self) -> List[Company]:
        """Return a list of major HK-listed companies.

        Attempts to use ``akshare.stock_hk_spot_em()`` for a comprehensive,
        up-to-date list.  Falls back to a hardcoded list of ~50 blue-chip
        tickers when akshare is unavailable or returns an error.

        Raises
        ------
        DataFetchError
            If both akshare and fallback fail to produce a stock list.
        """
        companies = self._fetch_stock_list_akshare()
        if companies:
            return companies

        logger.info("Falling back to hardcoded HK ticker list (%d tickers)", len(_FALLBACK_TICKERS))
        try:
            # Build companies from fallback tickers, enriching names from yfinance
            companies = []
            for t in _FALLBACK_TICKERS:
                name = self._get_hk_stock_name(t)
                companies.append(
                    Company(
                        ticker=t,
                        name=name,
                        market=Market.HK_SHARE,
                        currency="HKD",
                    )
                )
            return companies
        except Exception as exc:
            raise DataFetchError("Failed to build HK-share fallback stock list") from exc

    def _get_hk_stock_name(self, ticker: str) -> str:
        """Fetch company name for HK ticker from yfinance. Falls back to ticker if unavailable."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info if hasattr(stock, 'info') else {}
            name = info.get('shortName') or info.get('longName') or ticker
            return str(name).strip() if name else ticker
        except Exception:
            return ticker

    def _fetch_stock_list_akshare(self) -> List[Company]:
        """Try fetching HK stock spot data via akshare."""
        try:
            import akshare as ak  # noqa: WPS433 – optional at runtime

            df: pd.DataFrame = ak.stock_hk_spot_em()
            if df is None or df.empty:
                logger.warning("akshare stock_hk_spot_em() returned empty data")
                return []

            companies: List[Company] = []
            for _, row in df.iterrows():
                ticker = _to_yf_ticker(str(row.get("代码", "")))
                name = str(row.get("名称", ticker))
                market_cap_hkd = row.get("总市值")
                market_cap_rmb = (
                    float(market_cap_hkd) * HKD_TO_RMB
                    if market_cap_hkd is not None and pd.notna(market_cap_hkd)
                    else None
                )
                companies.append(
                    Company(
                        ticker=ticker,
                        name=name,
                        market=Market.HK_SHARE,
                        currency="HKD",
                        market_cap_rmb=market_cap_rmb,
                    )
                )

            logger.info("Fetched %d HK companies via akshare", len(companies))
            return companies

        except Exception as exc:
            logger.warning("akshare HK stock list unavailable: %s", exc)
            return []

    # -----------------------------------------------------------------
    # Financials
    # -----------------------------------------------------------------

    def fetch_financials(self, ticker: str) -> Optional[Financials]:
        """Fetch income-statement and balance-sheet data for *ticker*.

        Args:
            ticker: yfinance-style ticker, e.g. ``"0700.HK"``.

        Returns:
            A :class:`Financials` instance for the most recent annual period,
            or ``None`` if data is unavailable.
        """
        ticker = _to_yf_ticker(ticker)
        try:
            yf_ticker = yf.Ticker(ticker)
            income_stmt = yf_ticker.financials
            balance = yf_ticker.balance_sheet
            cashflow = yf_ticker.cashflow

            if income_stmt is None or income_stmt.empty:
                logger.warning("No income-statement data for %s", ticker)
                return None

            # Most recent annual column
            period_col = income_stmt.columns[0]
            period_label = str(period_col.date()) if hasattr(period_col, "date") else str(period_col)

            revenue = self._extract(income_stmt, period_col, "Total Revenue")
            net_income = self._extract(income_stmt, period_col, "Net Income")
            gross_profit = self._extract(income_stmt, period_col, "Gross Profit")

            total_assets = self._extract(balance, period_col, "Total Assets") if balance is not None else None
            total_liabilities = (
                self._extract(balance, period_col, "Total Liabilities Net Minority Interest")
                or self._extract(balance, period_col, "Total Liab")
            ) if balance is not None else None
            total_equity = (
                self._extract(balance, period_col, "Stockholders Equity")
                or self._extract(balance, period_col, "Total Stockholder Equity")
                or self._extract(balance, period_col, "Common Stock Equity")
            ) if balance is not None else None
            current_assets = self._extract(balance, period_col, "Current Assets") if balance is not None else None
            current_liabilities = (
                self._extract(balance, period_col, "Current Liabilities")
            ) if balance is not None else None

            operating_cf = self._extract(cashflow, period_col, "Operating Cash Flow") if cashflow is not None else None
            free_cf = self._extract(cashflow, period_col, "Free Cash Flow") if cashflow is not None else None

            gross_margin = (gross_profit / revenue) if gross_profit and revenue else None
            net_margin = (net_income / revenue) if net_income and revenue else None
            roe = (net_income / total_equity) if net_income and total_equity else None
            roa = (net_income / total_assets) if net_income and total_assets else None
            debt_to_equity = (
                (total_liabilities / total_equity)
                if total_liabilities and total_equity
                else None
            )
            current_ratio = (
                (current_assets / current_liabilities)
                if current_assets and current_liabilities
                else None
            )

            financials = Financials(
                ticker=ticker,
                period=period_label,
                revenue=revenue,
                net_income=net_income,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                total_equity=total_equity,
                operating_cash_flow=operating_cf,
                free_cash_flow=free_cf,
                gross_margin=gross_margin,
                net_margin=net_margin,
                roe=roe,
                roa=roa,
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
            )
            logger.debug("Fetched financials for %s (period %s)", ticker, period_label)
            return financials

        except Exception as exc:
            logger.error("Error fetching financials for %s: %s", ticker, exc)
            return None

    # -----------------------------------------------------------------
    # Valuation
    # -----------------------------------------------------------------

    def fetch_valuation(self, ticker: str) -> Optional[ValuationMetrics]:
        """Fetch valuation metrics for *ticker* from yfinance ``Ticker.info``.

        Market capitalisation is converted from HKD to RMB using
        :data:`HKD_TO_RMB`.

        Args:
            ticker: yfinance-style ticker, e.g. ``"0700.HK"``.

        Returns:
            A :class:`ValuationMetrics` instance, or ``None`` on failure.
        """
        ticker = _to_yf_ticker(ticker)
        try:
            info: dict = yf.Ticker(ticker).info
            if not info:
                logger.warning("Empty info dict for %s", ticker)
                return None

            market_cap_hkd = _safe_get(info, "marketCap")
            market_cap_rmb = (
                float(market_cap_hkd) * HKD_TO_RMB if market_cap_hkd is not None else None
            )

            valuation = ValuationMetrics(
                ticker=ticker,
                date=str(date.today()),
                price=_safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice"),
                pe_ratio=_safe_get(info, "trailingPE"),
                pe_forward=_safe_get(info, "forwardPE"),
                pb_ratio=_safe_get(info, "priceToBook"),
                ps_ratio=_safe_get(info, "priceToSalesTrailing12Months"),
                peg_ratio=_safe_get(info, "pegRatio"),
                dividend_yield=_safe_get(info, "dividendYield"),
                ev_to_ebitda=_safe_get(info, "enterpriseToEbitda"),
                market_cap_rmb=market_cap_rmb,
            )
            logger.debug("Fetched valuation for %s", ticker)
            return valuation

        except Exception as exc:
            logger.error("Error fetching valuation for %s: %s", ticker, exc)
            return None

    # -----------------------------------------------------------------
    # Company detail
    # -----------------------------------------------------------------

    def fetch_company_detail(self, ticker: str) -> Optional[Company]:
        """Fetch detailed company information from yfinance.

        Args:
            ticker: yfinance-style ticker, e.g. ``"0700.HK"``.

        Returns:
            A :class:`Company` with sector, industry, description, etc.,
            or ``None`` on failure.
        """
        ticker = _to_yf_ticker(ticker)
        try:
            info: dict = yf.Ticker(ticker).info
            if not info:
                logger.warning("Empty info dict for %s", ticker)
                return None

            market_cap_hkd = _safe_get(info, "marketCap")
            market_cap_rmb = (
                float(market_cap_hkd) * HKD_TO_RMB if market_cap_hkd is not None else None
            )

            company = Company(
                ticker=ticker,
                name=_safe_get(info, "shortName") or _safe_get(info, "longName") or ticker,
                name_en=_safe_get(info, "longName"),
                market=Market.HK_SHARE,
                sector=_safe_get(info, "sector"),
                industry=_safe_get(info, "industry"),
                description=_safe_get(info, "longBusinessSummary"),
                market_cap_rmb=market_cap_rmb,
                currency="HKD",
            )
            logger.debug("Fetched company detail for %s", ticker)
            return company

        except Exception as exc:
            logger.error("Error fetching company detail for %s: %s", ticker, exc)
            return None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract(df: pd.DataFrame, col, row_label: str) -> Optional[float]:
        """Safely extract a numeric value from a yfinance DataFrame.

        Args:
            df: The DataFrame (income statement, balance sheet, etc.).
            col: The column (period) to read.
            row_label: The index label to look up.

        Returns:
            The value as ``float``, or ``None`` if missing / NaN.
        """
        if df is None or df.empty:
            return None
        if row_label not in df.index:
            return None
        try:
            val = df.loc[row_label, col]
            if pd.isna(val):
                return None
            return float(val)
        except (KeyError, TypeError, ValueError):
            return None
