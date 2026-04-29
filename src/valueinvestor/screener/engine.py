"""Quantitative screening engine — fetches, filters, scores, and ranks stocks."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from valueinvestor.config import AppConfig
from valueinvestor.data.cache import DataCache
from valueinvestor.data.fetcher_ashare import AShareFetcher
from valueinvestor.data.fetcher_hkshare import HKShareFetcher
from valueinvestor.data.models import (
    Company,
    Financials,
    Market,
    ScreeningResult,
    ValuationMetrics,
)
from valueinvestor.errors import DataFetchError, ScreeningError
from valueinvestor.screener.scorer import MultiFactorScorer

logger = logging.getLogger(__name__)

class ScreeningEngine:
    """End-to-end quantitative stock screener.

    Parameters
    ----------
    config:
        Application configuration (screening thresholds, market list, etc.).
    cache:
        SQLite-backed data cache for avoiding redundant API calls.
    scorer:
        Optional scorer instance. If ``None``, a default ``MultiFactorScorer``
        (optimised for 6-month returns) is created automatically.
    """

    def __init__(
        self,
        config: AppConfig,
        cache: DataCache,
        scorer: Optional[MultiFactorScorer] = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self._a_fetcher = AShareFetcher()
        self._hk_fetcher = HKShareFetcher()
        self._scorer = scorer if scorer is not None else MultiFactorScorer()

    # ------------------------------------------------------------------
    # 1. Universe
    # ------------------------------------------------------------------

    def fetch_universe(self) -> List[Company]:
        """Return the combined list of A-share and HK-share companies.

        Results are read from / written to the cache so that repeated calls
        within the TTL window do not hit external APIs.

        Raises
        ------
        DataFetchError
            If no companies could be fetched from any configured market.
        """
        companies: List[Company] = []

        if "a_share" in self.config.markets:
            try:
                companies.extend(self._fetch_market_list(Market.A_SHARE))
            except DataFetchError:
                logger.warning("Failed to fetch A-share stock list", exc_info=True)

        if "hk_share" in self.config.markets:
            try:
                companies.extend(self._fetch_market_list(Market.HK_SHARE))
            except DataFetchError:
                logger.warning("Failed to fetch HK-share stock list", exc_info=True)

        logger.info("Universe: %d companies (%s)", len(companies), ", ".join(self.config.markets))
        return companies

    # ------------------------------------------------------------------
    # 2. Filtering
    # ------------------------------------------------------------------

    def filter_candidates(
        self,
        companies: List[Company],
        valuations: Dict[str, ValuationMetrics],
        financials: Dict[str, Financials],
    ) -> List[ScreeningResult]:
        """Apply quantitative filters and return passing :class:`ScreeningResult` objects."""
        sc = self.config.screening
        results: List[ScreeningResult] = []

        for company in companies:
            ticker = company.ticker
            val = valuations.get(ticker)
            fin = financials.get(ticker)

            if val is None:
                logger.debug("Skipping %s — no valuation data", ticker)
                continue
            if fin is None:
                logger.debug("Skipping %s — no financial data", ticker)
                continue

            # --- Market-cap filter ---
            cap = val.market_cap_rmb or company.market_cap_rmb
            if cap is None or cap < sc.market_cap_min_rmb:
                continue

            # --- PE filter (positive earnings only) ---
            if val.pe_ratio is None or val.pe_ratio <= 0 or val.pe_ratio > sc.pe_max:
                continue

            # --- PB filter (positive book value only) ---
            if val.pb_ratio is None or val.pb_ratio <= 0 or val.pb_ratio > sc.pb_max:
                continue

            # --- ROE filter ---
            if fin.roe is None or fin.roe < sc.roe_min:
                continue

            # --- Debt-to-equity filter (skip filter if data unavailable) ---
            if fin.debt_to_equity is not None and fin.debt_to_equity > sc.debt_ratio_max:
                continue

            results.append(
                ScreeningResult(
                    company=company,
                    financials=fin,
                    valuation=val,
                )
            )

        logger.info(
            "Filtering complete: %d / %d companies passed all criteria",
            len(results),
            len(companies),
        )
        return results

    # ------------------------------------------------------------------
    # 3. Full pipeline
    # ------------------------------------------------------------------

    def run(self, top_n: Optional[int] = None) -> List[ScreeningResult]:
        """Execute the full screening pipeline and return the top-ranked results.

        Steps
        -----
        1. Fetch stock universe (A-share + HK-share).
        2. Bulk-fetch valuations for A-shares; per-stock for HK-shares.
        3. **Pre-filter by valuation** (PE, PB, market cap) to reduce the
           candidate pool before the expensive per-stock financials fetch.
        4. Fetch financials per stock (cache-first) — only for pre-filtered
           candidates.
        5. Apply full quantitative filters (adds ROE, debt-ratio checks).
        6. Score and rank via :class:`MultiFactorScorer`.
        7. Return the top *top_n* results.
        """
        if top_n is None:
            top_n = self.config.screening.top_n

        # Step 1 — universe
        logger.info("Step 1/6: Fetching stock universe …")
        try:
            companies = self.fetch_universe()
        except DataFetchError as exc:
            raise ScreeningError("Failed to fetch stock universe") from exc
        if not companies:
            logger.warning("No companies in universe — aborting screening run")
            return []

        # Step 2 — valuations
        logger.info("Step 2/6: Fetching valuation data …")
        valuations = self._fetch_all_valuations(companies)
        logger.info("Valuations available for %d companies", len(valuations))

        # Step 3 — valuation pre-filter (PE, PB, market-cap only)
        # This reduces 5 000+ stocks to a manageable subset so that the
        # per-stock financials fetch in Step 4 is fast (≤ ~300 calls instead
        # of 5 000+).
        logger.info("Step 3/6: Pre-filtering by valuation metrics …")
        pre_candidates = self._pre_filter_by_valuation(companies, valuations)
        logger.info(
            "Valuation pre-filter: %d / %d companies remain",
            len(pre_candidates),
            len(companies),
        )
        if not pre_candidates:
            logger.warning("No candidates survived valuation pre-filter")
            return []

        # Step 4 — financials (only for pre-filtered candidates)
        logger.info("Step 4/6: Fetching financials for %d candidates …", len(pre_candidates))
        financials = self._fetch_all_financials(pre_candidates)
        logger.info("Financials available for %d companies", len(financials))

        # Step 5 — full filter (adds ROE and debt-ratio)
        logger.info("Step 5/6: Applying full quantitative filters …")
        candidates = self.filter_candidates(pre_candidates, valuations, financials)
        if not candidates:
            logger.warning("No candidates survived filtering")
            return []

        # Step 6 — score & rank
        logger.info("Step 6/6: Scoring and ranking %d candidates …", len(candidates))
        ranked = self._scorer.rank(candidates)

        top = ranked[:top_n]
        logger.info(
            "Screening complete — returning top %d of %d candidates",
            len(top),
            len(ranked),
        )
        return top

    def _pre_filter_by_valuation(
        self,
        companies: List[Company],
        valuations: Dict[str, ValuationMetrics],
    ) -> List[Company]:
        """Return the subset of *companies* that pass valuation-only criteria.

        Applies PE, PB, and market-cap gates — the three filters that can be
        evaluated purely from the bulk snapshot (no per-stock financials call
        needed).  ROE and debt-ratio are checked later after financials are
        fetched.
        """
        sc = self.config.screening
        pre: List[Company] = []

        for company in companies:
            val = valuations.get(company.ticker)
            if val is None:
                continue
            cap = val.market_cap_rmb or company.market_cap_rmb
            if cap is None or cap < sc.market_cap_min_rmb:
                continue
            if val.pe_ratio is None or val.pe_ratio <= 0 or val.pe_ratio > sc.pe_max:
                continue
            if val.pb_ratio is None or val.pb_ratio <= 0 or val.pb_ratio > sc.pb_max:
                continue
            pre.append(company)

        return pre

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _fetch_market_list(self, market: Market) -> List[Company]:
        """Fetch (or read from cache) the stock list for *market*."""
        cached = self.cache.get_stock_list(market.value)
        if cached is not None:
            logger.info("Using cached stock list for %s (%d companies)", market.value, len(cached))
            return cached

        if market == Market.A_SHARE:
            companies = self._a_fetcher.fetch_stock_list()
        else:
            companies = self._hk_fetcher.fetch_stock_list()

        if companies:
            self.cache.set_stock_list(market.value, companies)
        return companies

    # ------------------------------------------------------------------
    # Valuations
    # ------------------------------------------------------------------

    def _fetch_all_valuations(self, companies: List[Company]) -> Dict[str, ValuationMetrics]:
        """Bulk-fetch A-share valuations; per-stock for HK-shares. Cache everything."""
        result: Dict[str, ValuationMetrics] = {}

        # A-shares — efficient bulk endpoint
        a_tickers = {c.ticker for c in companies if c.market == Market.A_SHARE}
        if a_tickers:
            result.update(self._bulk_a_share_valuations(a_tickers))

        # HK-shares — per-stock via yfinance
        hk_companies = [c for c in companies if c.market == Market.HK_SHARE]
        for company in hk_companies:
            val = self._cached_valuation(company.ticker, market=Market.HK_SHARE)
            if val is not None:
                result[company.ticker] = val

        return result

    def _bulk_a_share_valuations(self, tickers: set[str]) -> Dict[str, ValuationMetrics]:
        """Use the A-share bulk snapshot, caching each result.

        Only triggers a full bulk refetch when the cache miss rate exceeds a
        threshold (``_BULK_REFETCH_THRESHOLD``).  This prevents expensive
        re-fetches when only a handful of tickers (e.g., newly listed or
        suspended stocks) are absent from cache.
        """
        _BULK_REFETCH_THRESHOLD = 0.05  # refetch if >5% of tickers are uncached

        out: Dict[str, ValuationMetrics] = {}
        missing: List[str] = []
        for t in tickers:
            cached = self.cache.get_valuation(t)
            if cached is not None:
                out[t] = cached
            else:
                missing.append(t)

        if not missing:
            logger.info("All %d A-share valuations served from cache", len(out))
            return out

        miss_rate = len(missing) / max(len(tickers), 1)
        if miss_rate <= _BULK_REFETCH_THRESHOLD:
            logger.info(
                "A-share valuations: %d/%d cache hits; %d misses (%.0f%%) — below "
                "refetch threshold, using cached data only",
                len(out), len(tickers), len(missing), miss_rate * 100,
            )
            return out  # accept the small gap; no bulk API call

        logger.info("Fetching bulk A-share valuations (%d cache misses, %.0f%%) …",
                    len(missing), miss_rate * 100)
        bulk = self._a_fetcher.fetch_all_valuations()
        missing_set = set(missing)
        for val in bulk:
            self.cache.set_valuation(val)  # update cache for all returned
            if val.ticker in missing_set:
                out[val.ticker] = val

        return out

    def _cached_valuation(
        self, ticker: str, *, market: Market
    ) -> Optional[ValuationMetrics]:
        """Return a valuation from cache or fetch + cache it."""
        cached = self.cache.get_valuation(ticker)
        if cached is not None:
            return cached

        try:
            if market == Market.A_SHARE:
                val = self._a_fetcher.fetch_valuation(ticker)
            else:
                val = self._hk_fetcher.fetch_valuation(ticker)
        except Exception:
            logger.warning("Failed to fetch valuation for %s", ticker, exc_info=True)
            return None

        if val is not None:
            self.cache.set_valuation(val)
        return val

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------

    def _fetch_all_financials(self, companies: List[Company]) -> Dict[str, Financials]:
        """Fetch financials per stock, preferring the cache."""
        result: Dict[str, Financials] = {}
        for company in companies:
            fin = self._cached_financials(company.ticker, market=company.market)
            if fin is not None:
                result[company.ticker] = fin
        return result

    def _cached_financials(
        self, ticker: str, *, market: Market
    ) -> Optional[Financials]:
        """Return financials from cache or fetch + cache them."""
        cached = self.cache.get_financials(ticker)
        if cached is not None:
            return cached

        try:
            if market == Market.A_SHARE:
                fin = self._a_fetcher.fetch_financials(ticker)
            else:
                fin = self._hk_fetcher.fetch_financials(ticker)
        except Exception:
            logger.warning("Failed to fetch financials for %s", ticker, exc_info=True)
            return None

        if fin is not None:
            self.cache.set_financials(fin)
        return fin
