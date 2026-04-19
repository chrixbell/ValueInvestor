from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class Market(str, Enum):
    A_SHARE = "a_share"
    HK_SHARE = "hk_share"


class Company(BaseModel):
    ticker: str
    name: str
    name_en: Optional[str] = None
    market: Market
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    market_cap_rmb: Optional[float] = None  # in RMB
    currency: str = "CNY"


class Financials(BaseModel):
    ticker: str
    period: str  # e.g., "2024-12-31"
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None


class ValuationMetrics(BaseModel):
    ticker: str
    date: str
    price: Optional[float] = None
    pe_ratio: Optional[float] = None  # trailing
    pe_forward: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    market_cap_rmb: Optional[float] = None


class ScreeningResult(BaseModel):
    company: Company
    financials: Financials
    valuation: ValuationMetrics
    composite_score: float = 0.0
    value_score: float = 0.0
    quality_score: float = 0.0
    growth_score: float = 0.0
    rank: int = 0


class AnalysisDimension(BaseModel):
    dimension: str  # e.g., "business_nature", "management", "moat", "market_narrative", "recommendation"
    title: str
    content: str
    confidence: Optional[float] = None  # 0.0 to 1.0


class CompanyAnalysis(BaseModel):
    ticker: str
    company_name: str
    screening_result: ScreeningResult
    analyses: List[AnalysisDimension]
    portfolio_weight: Optional[float] = None  # recommended % allocation
    key_milestones: Optional[List[str]] = None
    generated_at: str  # ISO timestamp


class InvestmentReport(BaseModel):
    title: str
    generated_at: str
    config_summary: Dict[str, object]  # snapshot of config used
    total_screened: int
    total_candidates: int
    candidates: List[CompanyAnalysis]
