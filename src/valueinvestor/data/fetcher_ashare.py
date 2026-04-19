"""A-share market data fetcher using the *akshare* library.

akshare provides free access to Chinese financial-market data.  Because its
API surface changes frequently between releases, every call is wrapped in a
``try / except`` block so that individual failures are logged rather than
crashing an entire batch run.

Currency note
-------------
A-share data from akshare is already denominated in **CNY (RMB)**.  The helper
constants ``HKD_TO_RMB`` and ``USD_TO_RMB`` are provided for potential
cross-market use but are *not* applied to the A-share paths.

Known data-source issues
------------------------
* ``ak.stock_a_indicator_lg`` was removed in akshare ≥ 1.14.  Per-stock
  valuation now falls back to a computed approach: price from Tencent's
  ``stock_zh_a_hist_tx`` combined with EPS / book-value from THS
  ``stock_financial_abstract_ths``.
* East Money (EM) push endpoints (``push2.eastmoney.com``) are geo-restricted
  and return ``RemoteDisconnected`` from outside mainland China.  The bulk
  valuation snapshot falls back to Sina Finance's ``Market_Center`` API.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
import requests

from valueinvestor.data.models import Company, Financials, Market, ValuationMetrics
from valueinvestor.errors import DataFetchError

logger = logging.getLogger(__name__)

# Approximate FX rates — used only when data arrives in a foreign currency.
HKD_TO_RMB: float = 0.92
USD_TO_RMB: float = 7.25

# Sina Finance Market_Center bulk API — works outside mainland China.
_SINA_MC_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
    "/Market_Center.getHQNodeData"
)
_SINA_PAGE_SIZE = 100

# Retry settings for East Money endpoints.
_EM_RETRIES = 3
_EM_RETRY_DELAY = 1.0  # seconds between retries

# Tencent Finance real-time quote API.
# Accepts up to ~100 comma-separated symbols per request (e.g. "sh600519,sz000001").
# Field positions in the "~"-separated value string:
#   [3]=current price, [39]=PE(TTM), [44]=流通市值(亿RMB), [45]=总市值(亿RMB), [46]=PB
_TENCENT_QUOTE_URL = "http://qt.gtimg.cn/q="
_TENCENT_BATCH_SIZE = 100  # symbols per request


def _safe_float(value: object) -> Optional[float]:
    """Convert *value* to ``float`` if possible, otherwise return ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ths_numeric(value: object) -> Optional[float]:
    """Parse a THS (Tonghuashun) numeric string into a plain float.

    THS returns financial data with unit/percentage suffixes that ``float()``
    cannot handle.  This function normalises them:

    * ``"32.53%"``    → ``0.3253``  (percentage → ratio)
    * ``"823.20亿"``  → ``82_320_000_000.0``  (亿 = 100 million)
    * ``"1750.00万"`` → ``17_500_000.0``        (万 = 10 thousand)
    * Plain numeric strings and numbers pass through ``_safe_float``.
    * ``"--"`` / ``"-"`` / empty → ``None``
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in ("", "--", "-", "nan", "None"):
        return None
    if s.endswith("%"):
        base = _safe_float(s[:-1])
        return (base / 100.0) if base is not None else None
    if s.endswith("亿"):
        base = _safe_float(s[:-1])
        return (base * 1e8) if base is not None else None
    if s.endswith("万"):
        base = _safe_float(s[:-1])
        return (base * 1e4) if base is not None else None
    return _safe_float(s)


class AShareFetcher:
    """Fetch A-share stock data via the *akshare* library.

    All public methods return model instances defined in
    :pymod:`valueinvestor.data.models`.  Network or parsing errors for
    individual stocks are logged as warnings so that batch operations can
    continue.
    """

    # ------------------------------------------------------------------
    # Stock list
    # ------------------------------------------------------------------

    def fetch_stock_list(self) -> List[Company]:
        """Return every A-share listed company with basic info.

        Uses ``ak.stock_info_a_code_name()`` which returns a DataFrame with
        columns ``code`` (ticker) and ``name`` (Chinese company name).

        Fallback chain (each tried in order):
        1. ``ak.stock_info_a_code_name()`` — EM-based, geo-blocked outside China
        2. SSE + SZSE official exchange lists (publicly accessible globally)
        3. Sina Market_Center paginated API
        4. Tencent Finance batch quotes (derives list + valuations together)
        """
        df: Optional[pd.DataFrame] = None
        for attempt in range(1, _EM_RETRIES + 1):
            try:
                df = ak.stock_info_a_code_name()
                break
            except Exception:
                logger.debug(
                    "stock_info_a_code_name attempt %d/%d failed",
                    attempt,
                    _EM_RETRIES,
                    exc_info=True,
                )
                if attempt < _EM_RETRIES:
                    time.sleep(_EM_RETRY_DELAY)

        # Fallback 1: Official SSE + SZSE exchange lists (globally accessible)
        if df is None or df.empty:
            logger.info("Falling back to SSE/SZSE official exchange lists")
            df = self._fetch_exchange_lists()
            if df is not None and not df.empty:
                logger.info("Exchange lists: %d stocks (SSE+SZSE)", len(df))

        # Fallback 2: Sina Market_Center if exchange lists also failed
        if df is None or df.empty:
            logger.info("Falling back to Sina Market_Center for A-share list")
            df = self._fetch_spot_sina()
            if df is not None and not df.empty:
                df = df.rename(columns={"代码": "code", "名称": "name"})

        # Fallback 3: Tencent Finance batch quotes
        if df is None or df.empty:
            logger.info("Sina failed — falling back to Tencent Finance for A-share list")
            df = self._fetch_spot_tencent()
            if df is not None and not df.empty:
                df = df.rename(columns={"代码": "code", "名称": "name"})

        if df is None or df.empty:
            logger.exception("Failed to fetch A-share stock list from all sources")
            raise DataFetchError("Failed to fetch A-share stock list") from None

        companies: List[Company] = []
        for _, row in df.iterrows():
            try:
                ticker = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip() if row.get("name") is not None else ""
                if not ticker:
                    continue
                companies.append(
                    Company(
                        ticker=ticker,
                        name=name,
                        market=Market.A_SHARE,
                        currency="CNY",
                    )
                )
            except Exception:
                logger.warning("Skipping malformed row: %s", row.to_dict(), exc_info=True)

        logger.info("Fetched %d A-share companies", len(companies))
        return companies

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------

    def fetch_financials(self, ticker: str) -> Optional[Financials]:
        """Get the most-recent financial summary for *ticker*.

        Primary source: ``ak.stock_financial_abstract_ths(symbol=ticker)``
        (Tonghuashun financial abstract).  Falls back to
        ``ak.stock_financial_analysis_indicator(symbol=ticker)`` if the
        primary source is unavailable.

        The returned :class:`Financials` maps the first (latest) row of the
        abstract table to our canonical fields.
        """
        df = self._fetch_financial_abstract(ticker)
        if df is None or df.empty:
            df = self._fetch_financial_indicator(ticker)
        if df is None or df.empty:
            logger.warning("No financial data found for %s", ticker)
            return None

        try:
            # Prefer the most-recent complete annual report (period ending in
            # "-12-31") so we compare full-year figures.  If no annual period
            # is found (e.g., newly listed company with only interim data) fall
            # back to the most-recent row regardless of period type.
            period_col = "报告期" if "报告期" in df.columns else df.columns[0]
            annual = df[df[period_col].astype(str).str.endswith("-12-31", na=False)]
            row = annual.iloc[-1] if not annual.empty else df.iloc[-1]
            period = str(row.get(period_col, date.today().isoformat()))

            return Financials(
                ticker=ticker,
                period=period,
                revenue=self._find_field(row, ["营业总收入", "营业收入", "revenue"]),
                net_income=self._find_field(row, ["净利润", "归属净利润", "net_income"]),
                total_assets=self._find_field(row, ["总资产", "资产总计", "total_assets"]),
                total_liabilities=self._find_field(row, ["总负债", "负债合计", "total_liabilities"]),
                total_equity=self._find_field(
                    row, ["股东权益合计", "归属母公司股东权益", "净资产", "total_equity"]
                ),
                operating_cash_flow=self._find_field(
                    row, ["经营现金流", "经营活动现金流量净额", "operating_cash_flow"]
                ),
                gross_margin=self._find_field(row, ["毛利率", "销售毛利率", "gross_margin"]),
                net_margin=self._find_field(row, ["净利率", "销售净利率", "net_margin"]),
                roe=self._find_field(row, ["净资产收益率", "ROE", "roe"]),
                roa=self._find_field(row, ["总资产收益率", "ROA", "roa"]),
                debt_to_equity=self._find_field(row, ["资产负债率", "debt_to_equity"]),
                current_ratio=self._find_field(row, ["流动比率", "current_ratio"]),
            )
        except Exception:
            logger.warning("Failed to parse financials for %s", ticker, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Valuation — single stock
    # ------------------------------------------------------------------

    def fetch_valuation(self, ticker: str) -> Optional[ValuationMetrics]:
        """Get current valuation metrics for a single *ticker*.

        Uses ``ak.stock_a_indicator_lg(symbol=ticker)`` (Legulegu value
        indicators) which typically returns daily PE / PB / total-market-cap
        history.  The most recent row is used.

        If the primary source fails, falls back to
        ``ak.stock_individual_info_em(symbol=ticker)`` (East-Money individual
        info).
        """
        metrics = self._fetch_indicator_lg(ticker)
        if metrics is not None:
            return metrics

        return self._fetch_valuation_from_individual_info(ticker)

    # ------------------------------------------------------------------
    # Valuation — bulk
    # ------------------------------------------------------------------

    def fetch_all_valuations(self) -> List[ValuationMetrics]:
        """Bulk-fetch valuation data for all A-share stocks.

        Uses ``ak.stock_a_pe_and_market_cap()`` or
        ``ak.stock_zh_a_spot_em()`` (East-Money real-time snapshot) which
        returns PE, PB, market-cap and price for every listed stock in a
        single API call — much more efficient than per-stock requests.
        """
        df = self._fetch_spot_em()
        if df is None or df.empty:
            logger.warning("Bulk valuation snapshot returned no data")
            return []

        today = date.today().isoformat()
        results: List[ValuationMetrics] = []

        for _, row in df.iterrows():
            try:
                ticker = str(row.get("代码", "")).strip()
                if not ticker:
                    continue
                results.append(
                    ValuationMetrics(
                        ticker=ticker,
                        date=today,
                        price=self._find_field(row, ["最新价", "收盘价", "price"]),
                        pe_ratio=self._find_field(row, ["市盈率-动态", "市盈率", "pe"]),
                        pb_ratio=self._find_field(row, ["市净率", "pb"]),
                        ps_ratio=self._find_field(row, ["市销率", "ps"]),
                        market_cap_rmb=self._market_cap_from_row(row),
                        dividend_yield=self._find_field(row, ["股息率", "dividend_yield"]),
                    )
                )
            except Exception:
                logger.warning(
                    "Skipping valuation row for %s",
                    row.get("代码", "?"),
                    exc_info=True,
                )

        logger.info("Bulk-fetched valuation data for %d stocks", len(results))
        return results

    # ------------------------------------------------------------------
    # Company detail
    # ------------------------------------------------------------------

    def fetch_company_detail(self, ticker: str) -> Optional[Company]:
        """Return detailed :class:`Company` info including description.

        Uses ``ak.stock_individual_info_em(symbol=ticker)`` (East-Money)
        which returns a two-column DataFrame (``item`` / ``value``) with
        fields such as 总市值, 行业, 上市时间, etc.
        """
        try:
            df: pd.DataFrame = ak.stock_individual_info_em(symbol=ticker)
        except Exception:
            logger.warning("Failed to fetch company detail for %s", ticker, exc_info=True)
            return None

        if df is None or df.empty:
            return None

        try:
            info = self._em_info_to_dict(df)
            return Company(
                ticker=ticker,
                name=info.get("股票简称", ticker),
                market=Market.A_SHARE,
                sector=info.get("行业", None),
                industry=info.get("行业", None),
                description=info.get("经营范围", None),
                market_cap_rmb=_safe_float(info.get("总市值")),
                currency="CNY",
            )
        except Exception:
            logger.warning("Failed to parse company detail for %s", ticker, exc_info=True)
            return None

    # ==================================================================
    # Private helpers — akshare data-source wrappers
    # ==================================================================

    def _fetch_financial_abstract(self, ticker: str) -> Optional[pd.DataFrame]:
        """Wrap ``ak.stock_financial_abstract_ths(symbol=ticker)``."""
        try:
            df: pd.DataFrame = ak.stock_financial_abstract_ths(symbol=ticker)
            logger.debug("stock_financial_abstract_ths returned %d rows for %s", len(df), ticker)
            return df
        except Exception:
            logger.debug(
                "stock_financial_abstract_ths unavailable for %s, trying fallback",
                ticker,
                exc_info=True,
            )
            return None

    def _fetch_financial_indicator(self, ticker: str) -> Optional[pd.DataFrame]:
        """Wrap ``ak.stock_financial_analysis_indicator(symbol=ticker)``."""
        try:
            df: pd.DataFrame = ak.stock_financial_analysis_indicator(symbol=ticker)
            logger.debug(
                "stock_financial_analysis_indicator returned %d rows for %s", len(df), ticker
            )
            return df
        except Exception:
            logger.debug(
                "stock_financial_analysis_indicator unavailable for %s", ticker, exc_info=True
            )
            return None

    def _fetch_indicator_lg(self, ticker: str) -> Optional[ValuationMetrics]:
        """Compute valuation from Tencent price data + THS financial abstract.

        ``ak.stock_a_indicator_lg`` was removed in akshare ≥ 1.14.  This
        replacement derives PE and PB from first-principles:

        * **Price** — latest close from ``ak.stock_zh_a_hist_tx`` (Tencent).
        * **EPS** — most-recent full-year basic EPS from THS financial abstract.
        * **BV/share** — most-recent book value per share from the same source.
        * **Market cap** — price × outstanding shares from Sina daily data.
        """
        prefix = self._exchange_prefix(ticker)

        # --- 1. Latest close price (Tencent) ---
        try:
            start = (date.today() - timedelta(days=14)).strftime("%Y%m%d")
            end = date.today().strftime("%Y%m%d")
            df_price: pd.DataFrame = ak.stock_zh_a_hist_tx(
                symbol=f"{prefix}{ticker}", start_date=start, end_date=end
            )
        except Exception:
            logger.debug("stock_zh_a_hist_tx unavailable for %s", ticker, exc_info=True)
            return None

        if df_price is None or df_price.empty:
            return None

        price = _safe_float(df_price.iloc[-1].get("close"))
        if price is None:
            return None

        trade_date = str(df_price.iloc[-1].get("date", date.today().isoformat()))

        # --- 2. EPS and BV/share (THS financial abstract) ---
        pe_ratio: Optional[float] = None
        pb_ratio: Optional[float] = None
        try:
            df_fin: pd.DataFrame = ak.stock_financial_abstract_ths(symbol=ticker)
            if df_fin is not None and not df_fin.empty:
                # Most-recent annual EPS (period ending -12-31) for PE
                annual = df_fin[df_fin["报告期"].str.endswith("-12-31", na=False)]
                if not annual.empty:
                    eps = _safe_float(annual.iloc[-1].get("基本每股收益"))
                    if eps and eps > 0:
                        pe_ratio = round(price / eps, 2)

                # Most-recent BV/share (any period) for PB
                bvps = _safe_float(df_fin.iloc[-1].get("每股净资产"))
                if bvps and bvps > 0:
                    pb_ratio = round(price / bvps, 2)
        except Exception:
            logger.debug(
                "stock_financial_abstract_ths unavailable for %s", ticker, exc_info=True
            )

        # --- 3. Market cap (Sina daily — provides outstanding shares) ---
        market_cap_rmb: Optional[float] = None
        try:
            df_daily: pd.DataFrame = ak.stock_zh_a_daily(
                symbol=f"{prefix}{ticker}", adjust=""
            )
            if df_daily is not None and not df_daily.empty:
                outstanding = _safe_float(df_daily.iloc[-1].get("outstanding_share"))
                if outstanding and price:
                    market_cap_rmb = round(outstanding * price)
        except Exception:
            logger.debug("stock_zh_a_daily unavailable for %s", ticker, exc_info=True)

        logger.debug(
            "Computed valuation for %s: price=%s pe=%s pb=%s mktcap=%s",
            ticker, price, pe_ratio, pb_ratio, market_cap_rmb,
        )
        return ValuationMetrics(
            ticker=ticker,
            date=trade_date,
            price=price,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            market_cap_rmb=market_cap_rmb,
        )

    def _fetch_valuation_from_individual_info(self, ticker: str) -> Optional[ValuationMetrics]:
        """Build :class:`ValuationMetrics` from ``ak.stock_individual_info_em``.

        Attempts up to ``_EM_RETRIES`` times before giving up; the endpoint
        is geo-restricted and may return ``RemoteDisconnected`` intermittently.
        """
        df: Optional[pd.DataFrame] = None
        for attempt in range(1, _EM_RETRIES + 1):
            try:
                df = ak.stock_individual_info_em(symbol=ticker)
                break
            except Exception:
                logger.debug(
                    "stock_individual_info_em attempt %d/%d failed for %s",
                    attempt, _EM_RETRIES, ticker,
                    exc_info=True,
                )
                if attempt < _EM_RETRIES:
                    time.sleep(_EM_RETRY_DELAY)

        if df is None or df.empty:
            logger.warning("stock_individual_info_em unavailable for %s", ticker)
            return None

        try:
            info = self._em_info_to_dict(df)
            return ValuationMetrics(
                ticker=ticker,
                date=date.today().isoformat(),
                pe_ratio=_safe_float(info.get("市盈率(动态)")),
                pb_ratio=_safe_float(info.get("市净率")),
                market_cap_rmb=_safe_float(info.get("总市值")),
            )
        except Exception:
            logger.warning(
                "Failed to parse individual_info_em valuation for %s", ticker, exc_info=True
            )
            return None

    def _fetch_spot_em(self) -> Optional[pd.DataFrame]:
        """Bulk A-share snapshot — tries East Money first, then Sina fallback.

        East Money ``push2.eastmoney.com`` endpoints are geo-restricted and
        return ``RemoteDisconnected`` from outside mainland China.  Up to
        ``_EM_RETRIES`` attempts are made before falling back to the Sina
        Finance ``Market_Center`` API which is publicly accessible.

        Returns a DataFrame with at minimum the columns:
        ``代码``, ``最新价``, ``市盈率-动态``, ``市净率``, ``总市值`` (RMB).
        """
        # Primary: East Money (fast, single request)
        for attempt in range(1, _EM_RETRIES + 1):
            try:
                df: pd.DataFrame = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    logger.debug("stock_zh_a_spot_em succeeded on attempt %d", attempt)
                    return df
            except Exception:
                logger.debug(
                    "stock_zh_a_spot_em attempt %d/%d failed",
                    attempt, _EM_RETRIES,
                    exc_info=True,
                )
                if attempt < _EM_RETRIES:
                    time.sleep(_EM_RETRY_DELAY)

        # Fallback 1: Sina Finance Market_Center API
        logger.info(
            "stock_zh_a_spot_em unavailable after %d retries — falling back to Sina",
            _EM_RETRIES,
        )
        df = self._fetch_spot_sina()
        if df is not None and not df.empty:
            return df

        # Fallback 2: Tencent Finance batch quote API (needs stock list first)
        logger.info("Sina fallback failed — trying Tencent Finance batch quotes")
        return self._fetch_spot_tencent()

    def _fetch_spot_sina(self) -> Optional[pd.DataFrame]:
        """Bulk valuation snapshot via Sina Finance Market_Center API.

        Paginates through Sina's ``Market_Center.getHQNodeData`` for the
        ``hs_a`` node (all mainland A-shares including SSE, SZSE, and BSE).

        Returns a DataFrame with columns:
        ``代码``, ``最新价``, ``市盈率-动态``, ``市净率``, ``总市值`` (RMB).
        Sina reports market cap in *万元* (10 000 RMB); this method converts
        it to plain RMB before returning.
        """
        records: list = []
        page = 1
        _page_retries = 3
        _page_retry_delay = 2.0  # base delay for per-page retry (seconds)
        while True:
            params = {
                "page": str(page),
                "num": str(_SINA_PAGE_SIZE),
                "sort": "symbol",
                "asc": "1",
                "node": "hs_a",
                "symbol": "",
                "_s_r_a": "page",
            }
            data = None
            for attempt in range(1, _page_retries + 1):
                try:
                    resp = requests.get(_SINA_MC_URL, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as exc:
                    wait = _page_retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Sina Market_Center page %d attempt %d/%d failed: %s. "
                        "Retrying in %.0fs …",
                        page, attempt, _page_retries, exc, wait,
                    )
                    time.sleep(wait)

            if data is None:
                # All retries exhausted for this page — stop pagination
                logger.warning("Sina Market_Center page %d permanently failed; stopping", page)
                break

            if not data:
                break

            for item in data:
                code = str(item.get("code", "")).strip()
                if not code:
                    continue
                mktcap_wan = _safe_float(item.get("mktcap"))
                records.append(
                    {
                        "代码": code,
                        "名称": item.get("name"),
                        "最新价": _safe_float(item.get("trade")),
                        "市盈率-动态": _safe_float(item.get("per")),
                        "市净率": _safe_float(item.get("pb")),
                        # Sina returns market cap in 万元; convert to RMB
                        "总市值": (mktcap_wan * 10_000) if mktcap_wan else None,
                    }
                )

            if len(data) < _SINA_PAGE_SIZE:
                break  # last page

            page += 1
            time.sleep(0.3)  # polite inter-page delay

        if not records:
            logger.warning("Sina Market_Center returned no data")
            return None

        df = pd.DataFrame(records)
        logger.info(
            "Sina fallback: fetched valuation data for %d stocks (%d pages)",
            len(df), page,
        )
        return df

    @staticmethod
    def _fetch_exchange_lists() -> Optional[pd.DataFrame]:
        """Fetch A-share code+name from SSE and SZSE official listing APIs.

        Both exchanges publish their full listing at publicly accessible URLs
        (no geo-restriction).  BSE (Beijing) is omitted because its endpoint
        uses the geo-blocked EM infrastructure.

        Returns a DataFrame with columns ``code`` (6-digit str) and ``name``.
        """
        frames: list = []

        # Shanghai Stock Exchange — main board + STAR (科创板)
        for segment in ("主板A股", "科创板"):
            try:
                df = ak.stock_info_sh_name_code(symbol=segment)
                if df is not None and not df.empty:
                    sub = df[["证券代码", "证券简称"]].copy()
                    sub.columns = ["code", "name"]
                    sub["code"] = sub["code"].astype(str).str.strip().str.zfill(6)
                    frames.append(sub)
            except Exception:
                logger.debug("SSE list for %s failed", segment, exc_info=True)

        # Shenzhen Stock Exchange — main board + ChiNext (创业板)
        for segment in ("A股列表", "创业板"):
            try:
                df = ak.stock_info_sz_name_code(symbol=segment)
                if df is not None and not df.empty:
                    # SZSE uses "A股代码" / "A股简称" or "证券代码" / "证券简称"
                    code_col = next(
                        (c for c in df.columns if "代码" in c), None
                    )
                    name_col = next(
                        (c for c in df.columns if "简称" in c), None
                    )
                    if code_col and name_col:
                        sub = df[[code_col, name_col]].copy()
                        sub.columns = ["code", "name"]
                        sub["code"] = sub["code"].astype(str).str.strip().str.zfill(6)
                        frames.append(sub)
            except Exception:
                logger.debug("SZSE list for %s failed", segment, exc_info=True)

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="code")
        return combined

    def _fetch_spot_tencent(self) -> Optional[pd.DataFrame]:
        """Bulk valuation snapshot via Tencent Finance real-time quote API.

        This is the third-level fallback (after EM and Sina).  Tencent's
        ``qt.gtimg.cn`` API returns real-time quotes including PE(TTM), PB,
        market cap (亿 RMB), and current price in a single request per batch
        of up to ``_TENCENT_BATCH_SIZE`` symbols.

        Since the Tencent API requires explicit tickers, we first obtain the
        code list via ``ak.stock_info_a_code_name()``, then from the official
        SSE/SZSE exchange lists.  As a last resort, a code-range generator
        covers the full A-share code space.

        Tencent quote string field positions (0-indexed, ``~``-separated):
        - [3]  = current price
        - [39] = PE ratio (TTM)
        - [44] = 流通市值 in 亿 RMB
        - [45] = 总市值 in 亿 RMB
        - [46] = PB ratio
        """
        # Step 1: get the code list (try multiple sources)
        codes: list[str] = []
        try:
            df_codes = ak.stock_info_a_code_name()
            if df_codes is not None and not df_codes.empty:
                codes = df_codes["code"].astype(str).tolist()
        except Exception:
            logger.debug("stock_info_a_code_name failed in Tencent fallback", exc_info=True)

        if not codes:
            df_ex = self._fetch_exchange_lists()
            if df_ex is not None and not df_ex.empty:
                codes = df_ex["code"].astype(str).tolist()
                logger.info("Tencent fallback: using exchange list (%d codes)", len(codes))

        if not codes:
            # Last-resort: generate probable A-share codes
            codes = list(dict.fromkeys(
                [f"{i:06d}" for i in range(600000, 610000)]  # SSE main
                + [f"{i:06d}" for i in range(688000, 690000)]  # SSE STAR
                + [f"{i:06d}" for i in range(0, 3000)]         # SZSE main
                + [f"{i:06d}" for i in range(300000, 302000)]  # ChiNext
            ))
            logger.info(
                "Tencent fallback: using generated code list (%d candidates)", len(codes)
            )

        records: list = []
        total_batches = (len(codes) + _TENCENT_BATCH_SIZE - 1) // _TENCENT_BATCH_SIZE

        for batch_idx in range(0, len(codes), _TENCENT_BATCH_SIZE):
            batch = codes[batch_idx: batch_idx + _TENCENT_BATCH_SIZE]
            # Prefix each code with exchange identifier
            prefixed = [f"{self._exchange_prefix(c)}{c}" for c in batch]
            query = ",".join(prefixed)
            try:
                resp = requests.get(
                    _TENCENT_QUOTE_URL + query,
                    timeout=10,
                    headers={"Referer": "https://finance.qq.com/"},
                )
                resp.raise_for_status()
                for line in resp.text.split(";"):
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    val_str = line.split("=", 1)[1].strip('"')
                    fields = val_str.split("~")
                    if len(fields) < 47:
                        continue
                    code = fields[2].strip()
                    if not code:
                        continue
                    price = _safe_float(fields[3])
                    pe = _safe_float(fields[39]) if fields[39] else None
                    mktcap_yi = _safe_float(fields[45]) if fields[45] else None
                    pb = _safe_float(fields[46]) if fields[46] else None
                    records.append(
                        {
                            "代码": code,
                            "名称": fields[1],
                            "最新价": price,
                            "市盈率-动态": pe,
                            "市净率": pb,
                            # Tencent reports market cap in 亿 (×1e8) RMB
                            "总市值": (mktcap_yi * 1e8) if mktcap_yi else None,
                        }
                    )
            except Exception:
                logger.debug(
                    "Tencent batch %d/%d failed",
                    batch_idx // _TENCENT_BATCH_SIZE + 1,
                    total_batches,
                    exc_info=True,
                )
            time.sleep(0.1)

        if not records:
            logger.warning("Tencent Finance batch fetch returned no data")
            return None

        df = pd.DataFrame(records)
        # Drop rows where price or market cap are missing (unlisted / suspended)
        df = df.dropna(subset=["最新价", "总市值"])
        logger.info("Tencent fallback: fetched quotes for %d stocks", len(df))
        return df

    # ==================================================================
    # Private helpers — field extraction
    # ==================================================================

    @staticmethod
    def _em_info_to_dict(df: pd.DataFrame) -> dict:
        """Convert the two-column East-Money info DataFrame to a plain dict.

        The DataFrame typically has columns ``item`` and ``value``.
        """
        result: dict = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            val = row.iloc[1]
            result[key] = val
        return result

    @staticmethod
    def _find_field(row: pd.Series, candidates: List[str]) -> Optional[float]:
        """Try each column name in *candidates*; return the first valid float.

        Uses :func:`_parse_ths_numeric` so that THS percentage/unit strings
        (e.g. ``"32.53%"``, ``"823.20亿"``) are converted correctly.
        """
        for col in candidates:
            val = row.get(col)
            result = _parse_ths_numeric(val)
            if result is not None:
                return result
        return None

    @staticmethod
    def _market_cap_from_row(row: pd.Series) -> Optional[float]:
        """Extract market cap, converting 亿 (100 million) units if needed."""
        for col in ["总市值", "market_cap"]:
            val = _safe_float(row.get(col))
            if val is not None:
                return val
        return None

    @staticmethod
    def _exchange_prefix(ticker: str) -> str:
        """Return the exchange prefix (``sh``, ``sz``, or ``bj``) for *ticker*.

        Rules based on the leading digit of mainland A-share ticker codes:

        * ``6XXXXX`` → Shanghai Stock Exchange → ``sh``
        * ``4XXXXX``, ``8XXXXX``, ``9XXXXX`` → Beijing Stock Exchange → ``bj``
        * ``0XXXXX``, ``3XXXXX`` → Shenzhen Stock Exchange → ``sz``
        """
        if ticker.startswith("6"):
            return "sh"
        if ticker.startswith(("4", "8", "9")):
            return "bj"
        return "sz"
