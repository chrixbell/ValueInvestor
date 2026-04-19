from __future__ import annotations

import pytest
from pydantic import ValidationError

from valueinvestor.data.models import (
    AnalysisDimension,
    Company,
    CompanyAnalysis,
    Financials,
    InvestmentReport,
    Market,
    ScreeningResult,
    ValuationMetrics,
)


# ---------------------------------------------------------------------------
# Helpers — reusable fixtures
# ---------------------------------------------------------------------------

def _company(**overrides) -> Company:
    defaults = dict(ticker="600519", name="贵州茅台", market=Market.A_SHARE)
    return Company(**{**defaults, **overrides})


def _financials(**overrides) -> Financials:
    defaults = dict(ticker="600519", period="2024-12-31", roe=0.15)
    return Financials(**{**defaults, **overrides})


def _valuation(**overrides) -> ValuationMetrics:
    defaults = dict(ticker="600519", date="2024-06-01", pe_ratio=25.0, pb_ratio=8.0)
    return ValuationMetrics(**{**defaults, **overrides})


def _screening_result(**overrides) -> ScreeningResult:
    return ScreeningResult(
        company=_company(),
        financials=_financials(),
        valuation=_valuation(),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Market enum
# ---------------------------------------------------------------------------

class TestMarketEnum:
    def test_values(self):
        assert Market.A_SHARE.value == "a_share"
        assert Market.HK_SHARE.value == "hk_share"

    def test_from_value(self):
        assert Market("a_share") is Market.A_SHARE
        assert Market("hk_share") is Market.HK_SHARE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Market("us_stock")


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

class TestCompany:
    def test_minimal(self):
        c = _company()
        assert c.ticker == "600519"
        assert c.market == Market.A_SHARE
        assert c.currency == "CNY"

    def test_all_fields(self):
        c = _company(
            name_en="Kweichow Moutai",
            sector="Consumer Staples",
            industry="Liquor",
            description="Premium liquor producer",
            market_cap_rmb=2.5e12,
            currency="CNY",
        )
        assert c.name_en == "Kweichow Moutai"
        assert c.market_cap_rmb == 2.5e12

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Company(ticker="600519", name="贵州茅台")  # missing market

    def test_serialization_roundtrip(self):
        c = _company(sector="Consumer Staples")
        json_str = c.model_dump_json()
        restored = Company.model_validate_json(json_str)
        assert restored == c

    def test_dict_roundtrip(self):
        c = _company()
        d = c.model_dump()
        assert isinstance(d, dict)
        assert Company.model_validate(d) == c


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

class TestFinancials:
    def test_minimal(self):
        f = _financials()
        assert f.ticker == "600519"
        assert f.period == "2024-12-31"

    def test_optional_defaults_none(self):
        f = _financials()
        assert f.revenue is None
        assert f.net_income is None

    def test_full_fields(self):
        f = _financials(
            revenue=1e10, net_income=5e9, total_assets=2e10,
            total_liabilities=8e9, total_equity=1.2e10,
            operating_cash_flow=6e9, free_cash_flow=4e9,
            gross_margin=0.90, net_margin=0.50,
            roe=0.30, roa=0.20, debt_to_equity=0.67,
            current_ratio=2.5,
        )
        assert f.gross_margin == 0.90

    def test_serialization_roundtrip(self):
        f = _financials(revenue=1e10)
        assert Financials.model_validate_json(f.model_dump_json()) == f


# ---------------------------------------------------------------------------
# ValuationMetrics
# ---------------------------------------------------------------------------

class TestValuationMetrics:
    def test_minimal(self):
        v = _valuation()
        assert v.ticker == "600519"
        assert v.pe_ratio == 25.0

    def test_serialization_roundtrip(self):
        v = _valuation(price=1800.0, dividend_yield=0.015)
        assert ValuationMetrics.model_validate_json(v.model_dump_json()) == v


# ---------------------------------------------------------------------------
# ScreeningResult
# ---------------------------------------------------------------------------

class TestScreeningResult:
    def test_defaults(self):
        sr = _screening_result()
        assert sr.composite_score == 0.0
        assert sr.rank == 0

    def test_with_scores(self):
        sr = _screening_result(composite_score=85.5, value_score=90.0, rank=1)
        assert sr.composite_score == 85.5
        assert sr.rank == 1

    def test_serialization_roundtrip(self):
        sr = _screening_result(composite_score=72.3)
        restored = ScreeningResult.model_validate_json(sr.model_dump_json())
        assert restored.composite_score == sr.composite_score
        assert restored.company.ticker == sr.company.ticker


# ---------------------------------------------------------------------------
# AnalysisDimension / CompanyAnalysis / InvestmentReport
# ---------------------------------------------------------------------------

class TestCompanyAnalysis:
    def test_full_analysis(self):
        dim = AnalysisDimension(
            dimension="moat", title="Competitive Moat",
            content="Strong brand moat.", confidence=0.9,
        )
        ca = CompanyAnalysis(
            ticker="600519", company_name="贵州茅台",
            screening_result=_screening_result(),
            analyses=[dim],
            generated_at="2024-06-01T00:00:00Z",
        )
        assert ca.analyses[0].confidence == 0.9

    def test_investment_report(self):
        report = InvestmentReport(
            title="Q2 Report", generated_at="2024-06-01T00:00:00Z",
            config_summary={"markets": ["a_share"]},
            total_screened=500, total_candidates=20,
            candidates=[],
        )
        assert report.total_screened == 500
