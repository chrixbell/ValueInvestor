from __future__ import annotations

from unittest.mock import MagicMock, patch

from valueinvestor.config import AppConfig
from valueinvestor.data.models import (
    Company,
    Financials,
    Market,
    ValuationMetrics,
)
from valueinvestor.screener.engine import ScreeningEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _company(ticker: str, market_cap: float = 1e10) -> Company:
    return Company(
        ticker=ticker, name=f"Co {ticker}", market=Market.A_SHARE,
        market_cap_rmb=market_cap,
    )


def _valuation(ticker: str, pe: float = 15.0, pb: float = 2.0, cap: float = 1e10) -> ValuationMetrics:
    return ValuationMetrics(
        ticker=ticker, date="2024-06-01",
        pe_ratio=pe, pb_ratio=pb, market_cap_rmb=cap,
    )


def _financials(ticker: str, roe: float = 0.15, debt_to_equity: float = 0.5) -> Financials:
    return Financials(
        ticker=ticker, period="2024-12-31",
        roe=roe, debt_to_equity=debt_to_equity,
    )


def _make_engine(config: AppConfig | None = None) -> ScreeningEngine:
    """Build a ScreeningEngine with mocked fetchers to avoid network calls."""
    cfg = config or AppConfig()
    cache = MagicMock()
    with patch("valueinvestor.screener.engine.AShareFetcher"), \
         patch("valueinvestor.screener.engine.HKShareFetcher"):
        engine = ScreeningEngine(cfg, cache)
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFilterCandidates:
    def test_passing_company_included(self):
        engine = _make_engine()
        companies = [_company("A")]
        vals = {"A": _valuation("A", pe=15, pb=2, cap=1e10)}
        fins = {"A": _financials("A", roe=0.15)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 1
        assert results[0].company.ticker == "A"

    def test_low_market_cap_filtered(self):
        engine = _make_engine()
        companies = [_company("SMALL", market_cap=1e8)]
        vals = {"SMALL": _valuation("SMALL", pe=10, pb=1, cap=1e8)}
        fins = {"SMALL": _financials("SMALL", roe=0.20)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_high_pe_filtered(self):
        engine = _make_engine()
        companies = [_company("EXPEN")]
        vals = {"EXPEN": _valuation("EXPEN", pe=50, pb=2)}
        fins = {"EXPEN": _financials("EXPEN", roe=0.20)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_negative_pe_filtered(self):
        engine = _make_engine()
        companies = [_company("NEG")]
        vals = {"NEG": _valuation("NEG", pe=-5, pb=2)}
        fins = {"NEG": _financials("NEG", roe=0.20)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_low_roe_filtered(self):
        engine = _make_engine()
        companies = [_company("LOWQ")]
        vals = {"LOWQ": _valuation("LOWQ", pe=12, pb=1.5)}
        fins = {"LOWQ": _financials("LOWQ", roe=0.02)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_high_debt_filtered(self):
        engine = _make_engine()
        companies = [_company("DEBT")]
        vals = {"DEBT": _valuation("DEBT", pe=12, pb=1.5)}
        fins = {"DEBT": _financials("DEBT", roe=0.15, debt_to_equity=0.90)}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_missing_valuation_skipped(self):
        engine = _make_engine()
        companies = [_company("NOVAL")]
        vals: dict = {}
        fins = {"NOVAL": _financials("NOVAL")}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_missing_financials_skipped(self):
        engine = _make_engine()
        companies = [_company("NOFIN")]
        vals = {"NOFIN": _valuation("NOFIN")}
        fins: dict = {}

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 0

    def test_mixed_batch(self):
        """Only the passing company survives."""
        engine = _make_engine()
        companies = [_company("GOOD"), _company("BAD_PE"), _company("SMALL", market_cap=1e8)]
        vals = {
            "GOOD": _valuation("GOOD", pe=12, pb=1.5),
            "BAD_PE": _valuation("BAD_PE", pe=40, pb=1.5),
            "SMALL": _valuation("SMALL", pe=10, pb=1, cap=1e8),
        }
        fins = {
            "GOOD": _financials("GOOD", roe=0.15),
            "BAD_PE": _financials("BAD_PE", roe=0.20),
            "SMALL": _financials("SMALL", roe=0.20),
        }

        results = engine.filter_candidates(companies, vals, fins)
        assert len(results) == 1
        assert results[0].company.ticker == "GOOD"
