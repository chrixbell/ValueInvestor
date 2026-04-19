from __future__ import annotations

from valueinvestor.data.cache import DataCache
from valueinvestor.data.models import Company, Financials, Market, ValuationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _company(ticker: str = "600519") -> Company:
    return Company(ticker=ticker, name="贵州茅台", market=Market.A_SHARE)


def _financials(ticker: str = "600519") -> Financials:
    return Financials(ticker=ticker, period="2024-12-31", roe=0.30, net_margin=0.50)


def _valuation(ticker: str = "600519") -> ValuationMetrics:
    return ValuationMetrics(
        ticker=ticker, date="2024-06-01", pe_ratio=25.0, pb_ratio=8.0, price=1800.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataCacheInit:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "cache.db"
        DataCache(db_path=str(db), ttl_hours=24)
        assert db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "sub" / "dir" / "cache.db"
        DataCache(db_path=str(db), ttl_hours=24)
        assert db.exists()


class TestCompanyCache:
    def test_set_and_get(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "c.db"), ttl_hours=24)
        company = _company()
        cache.set_company(company)
        got = cache.get_company("600519")
        assert got is not None
        assert got.ticker == "600519"
        assert got.name == "贵州茅台"

    def test_get_missing_returns_none(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "c.db"), ttl_hours=24)
        assert cache.get_company("MISSING") is None


class TestFinancialsCache:
    def test_set_and_get(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "f.db"), ttl_hours=24)
        fin = _financials()
        cache.set_financials(fin)
        got = cache.get_financials("600519")
        assert got is not None
        assert got.roe == 0.30

    def test_get_missing_returns_none(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "f.db"), ttl_hours=24)
        assert cache.get_financials("MISSING") is None


class TestValuationCache:
    def test_set_and_get(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "v.db"), ttl_hours=24)
        val = _valuation()
        cache.set_valuation(val)
        got = cache.get_valuation("600519")
        assert got is not None
        assert got.pe_ratio == 25.0

    def test_get_missing_returns_none(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "v.db"), ttl_hours=24)
        assert cache.get_valuation("MISSING") is None


class TestCacheExpiration:
    def test_ttl_zero_treats_data_as_expired(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "e.db"), ttl_hours=0)
        cache.set_company(_company())
        cache.set_financials(_financials())
        cache.set_valuation(_valuation())

        assert cache.get_company("600519") is None
        assert cache.get_financials("600519") is None
        assert cache.get_valuation("600519") is None


class TestCacheClear:
    def test_clear_removes_all_data(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "cl.db"), ttl_hours=24)
        cache.set_company(_company())
        cache.set_financials(_financials())
        cache.set_valuation(_valuation())

        cache.clear()

        assert cache.get_company("600519") is None
        assert cache.get_financials("600519") is None
        assert cache.get_valuation("600519") is None

    def test_clear_is_idempotent(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "cl2.db"), ttl_hours=24)
        cache.clear()  # no data yet — should not raise
        cache.clear()
