from __future__ import annotations

import json

from valueinvestor.analysis.pipeline import AnalysisPipeline, _analysis_timeout_seconds
from valueinvestor.config import AppConfig
from valueinvestor.data.models import (
    AnalysisDimension,
    Company,
    CompanyAnalysis,
    Financials,
    Market,
    ScreeningResult,
    ValuationMetrics,
)


def _screening_result() -> ScreeningResult:
    return ScreeningResult(
        company=Company(ticker="000001.SZ", name="Test Co", market=Market.A_SHARE),
        financials=Financials(ticker="000001.SZ", period="2024-12-31"),
        valuation=ValuationMetrics(ticker="000001.SZ", date="2026-01-01"),
        composite_score=88.0,
        value_score=80.0,
        quality_score=75.0,
        growth_score=70.0,
    )


def test_analysis_timeout_seconds_env(monkeypatch) -> None:
    monkeypatch.setenv("VALUEINVESTOR_ANALYSIS_TIMEOUT_SECONDS", "12.5")

    assert _analysis_timeout_seconds() == 12.5


def test_analyze_company_passes_timeout_to_llm(monkeypatch) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.timeouts = []

        def complete(self, **kwargs):
            self.timeouts.append(kwargs.get("timeout"))
            return json.dumps({"title": "ok", "content": "ok", "confidence": 0.5})

    class FakeCache:
        def get_analysis(self, ticker):
            return None

        def set_analysis(self, analysis):
            self.analysis = analysis

    config = AppConfig()
    config.cache.enabled = False
    fake_llm = FakeLLM()
    monkeypatch.setenv("VALUEINVESTOR_ANALYSIS_TIMEOUT_SECONDS", "7")

    pipeline = AnalysisPipeline(llm=fake_llm, cache=FakeCache(), config=config)
    analysis = pipeline.analyze_company(_screening_result())

    assert analysis.ticker == "000001.SZ"
    assert fake_llm.timeouts
    assert set(fake_llm.timeouts) == {7.0}


def test_analyze_company_emits_progress_callback(monkeypatch) -> None:
    class FakeLLM:
        def complete(self, **kwargs):
            return json.dumps({"title": "ok", "content": "ok", "confidence": 0.5})

    class FakeCache:
        def get_analysis(self, ticker):
            return None

        def set_analysis(self, analysis):
            self.analysis = analysis

    config = AppConfig()
    config.cache.enabled = False
    events = []
    monkeypatch.setenv("VALUEINVESTOR_ANALYSIS_TIMEOUT_SECONDS", "7")

    pipeline = AnalysisPipeline(
        llm=FakeLLM(),
        cache=FakeCache(),
        config=config,
        progress_callback=lambda *event: events.append(event),
    )
    pipeline.analyze_company(_screening_result())

    statuses = [event[2] for event in events]
    assert statuses[0] == "analysis started"
    assert statuses[-1] == "analysis finished"
    assert any(status.endswith(" started") for status in statuses)
    assert any(status.endswith(" finished") for status in statuses)


def test_cached_analysis_refreshes_screening_result() -> None:
    old_result = _screening_result()
    cached = CompanyAnalysis(
        ticker="000001.SZ",
        company_name="Old Co",
        screening_result=old_result,
        analyses=[AnalysisDimension(dimension="business", title="t", content="cached")],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    fresh_result = _screening_result()
    fresh_result.company.name = "Fresh Co"
    fresh_result.composite_score = 99.0
    fresh_result.momentum_score = 61.0
    fresh_result.synergy_score = 62.0
    fresh_result.value_growth_score = 63.0

    class FakeCache:
        def __init__(self) -> None:
            self.analysis = cached

        def get_analysis(self, ticker, *, allow_expired=False):
            self.allow_expired = allow_expired
            return self.analysis

        def set_analysis(self, analysis):
            self.analysis = analysis

    config = AppConfig()
    cache = FakeCache()
    pipeline = AnalysisPipeline(llm=None, cache=cache, config=config)  # type: ignore[arg-type]

    refreshed = pipeline.cached_analysis_for_result(fresh_result, allow_expired=True)

    assert refreshed is not None
    assert refreshed.company_name == "Fresh Co"
    assert refreshed.screening_result.composite_score == 99.0
    assert refreshed.screening_result.momentum_score == 61.0
    assert refreshed.screening_result.synergy_score == 62.0
    assert refreshed.screening_result.value_growth_score == 63.0
    assert cache.analysis.screening_result.composite_score == 99.0
    assert cache.allow_expired is True
