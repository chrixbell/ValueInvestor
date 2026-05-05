from __future__ import annotations

from valueinvestor.data.fetcher_hkshare import (
    HKShareFetcher,
    _is_primary_hk_yf_ticker,
    _to_yf_ticker,
)
from valueinvestor.data.models import Company, Market


def test_to_yf_ticker_normalizes_primary_codes() -> None:
    assert _to_yf_ticker("0700") == "0700.HK"
    assert _to_yf_ticker("00700") == "0700.HK"
    assert _to_yf_ticker("700.hk") == "0700.HK"
    assert _to_yf_ticker("40849.HK") == "40849.HK"


def test_primary_hk_ticker_filter_excludes_non_primary_instruments() -> None:
    assert _is_primary_hk_yf_ticker("0700.HK")
    assert _is_primary_hk_yf_ticker("00700.HK")
    assert _is_primary_hk_yf_ticker("700.HK")
    assert _is_primary_hk_yf_ticker("8281.HK")
    assert not _is_primary_hk_yf_ticker("40849.HK")
    assert not _is_primary_hk_yf_ticker("85120.HK")


def test_filter_supported_companies_drops_five_digit_hk_codes() -> None:
    companies = [
        Company(ticker="0700.HK", name="Tencent", market=Market.HK_SHARE),
        Company(ticker="40849.HK", name="GZ METRO N2609", market=Market.HK_SHARE),
        Company(ticker="83151.HK", name="PP科创50-R", market=Market.HK_SHARE),
    ]

    filtered = HKShareFetcher().filter_supported_companies(companies)

    assert [company.ticker for company in filtered] == ["0700.HK"]


def test_fetch_valuation_skips_non_primary_hk_ticker(monkeypatch) -> None:
    def fail_if_called(ticker: str):
        raise AssertionError(f"yfinance should not be called for {ticker}")

    monkeypatch.setattr("valueinvestor.data.fetcher_hkshare.yf.Ticker", fail_if_called)

    assert HKShareFetcher().fetch_valuation("40849.HK") is None


def test_fetch_valuation_returns_empty_negative_cache_for_unusable_yahoo_info(monkeypatch) -> None:
    class FakeTicker:
        @property
        def info(self):
            return {"trailingPegRatio": None}

    monkeypatch.setattr(
        "valueinvestor.data.fetcher_hkshare.yf.Ticker",
        lambda ticker: FakeTicker(),
    )

    valuation = HKShareFetcher().fetch_valuation("9997.HK")

    assert valuation is not None
    assert valuation.ticker == "9997.HK"
    assert valuation.price is None
    assert valuation.pe_ratio is None
    assert valuation.pb_ratio is None
    assert valuation.market_cap_rmb is None
