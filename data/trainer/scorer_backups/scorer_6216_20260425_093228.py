"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.65,     # Retained at 0.65 as decreasing it previously hurt performance.
    "quality": 0.15,   # Core fundamental quality.
    "growth": 0.20,    # Growth factor weight.
    "momentum": 0.00,  # Placeholder.
}


def _linear_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Values beyond the endpoints are clamped.
    """
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Assumes 'value', 'best', and 'worst' are positive for math.log.
    Values beyond the endpoints are clamped.
    """
    if value <= 0 or best <= 0 or worst <= 0:
        return 0.0

    log_value = math.log(value)
    log_best = math.log(best)
    log_worst = math.log(worst)

    if log_best == log_worst:
        return 50.0

    score = (log_value - log_worst) / (log_best - log_worst) * 100.0
    return max(0.0, min(100.0, score))


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        # Momentum is a placeholder — set to 50 (neutral).
        momentum_score = 50.0

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Weighted geometric mean calculation to penalize stocks that are very poor in one category
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0

        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Score every result, sort by composite descending, and assign ranks."""
        for r in results:
            self.score(r)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _value_score(result: ScreeningResult) -> float:
        """Lower valuation multiples → higher score. (Note: PB scoring is still empirically inverted)."""
        weighted_scores: List[Tuple[float, float]] = []

        DIVIDEND_YIELD_WEIGHT = 0.2
        PB_WEIGHT = 0.3
        PS_WEIGHT = 0.2
        MARKET_CAP_WEIGHT = 0.1
        EV_TO_EBITDA_WEIGHT = 0.2

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # Empirical finding: higher PB is currently correlating with better forward returns in the dataset.
            weighted_scores.append((_log_score(pb, best=15.0, worst=1.0), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            # Broadened range and log-scaling to better differentiate between stocks in a skewed distribution.
            weighted_scores.append((_log_score(ps, best=0.4, worst=10.0), PS_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            # Adjusted range for size effect based on A-share market distribution.
            weighted_scores.append(
                (
                    _log_score(
                        market_cap,
                        best=1_000_000_000.0,
                        worst=60_000_000_000.0,
                    ),
                    MARKET_CAP_WEIGHT,
                )
            )

        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield >= 0:
            # Linear scoring for dividend yield remains standard; broadened range to 0.0-0.06.
            weighted_scores.append(
                (_linear_score(dividend_yield, best=0.06, worst=0.0), DIVIDEND_YIELD_WEIGHT)
            )

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            # Broadened range and log-scaling to capture growth valuation variance.
            weighted_scores.append(
                (_log_score(ev_to_ebitda, best=4.0, worst=40.0), EV_TO_EBITDA_WEIGHT)
            )

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Quality score focusing on GP/A (Gross Profitability), ROE, and Cash Return on Assets (CROA)."""
        weighted_scores: List[Tuple[float, float]] = []

        ROE_WEIGHT = 0.20
        GP_A_WEIGHT = 0.50  # Gross Profit / Total Assets
        CROA_WEIGHT = 0.30  # Operating Cash Flow / Total Assets

        # 1. ROE: Profitability relative to equity
        roe = result.financials.roe
        if roe is not None:
            weighted_scores.append((_linear_score(roe, best=0.20, worst=0.0), ROE_WEIGHT))

        # 2. GP/A: Gross Profitability (Revenue * Gross Margin / Total Assets)
        rev = result.financials.revenue
        gm = result.financials.gross_margin
        assets = result.financials.total_assets
        if rev is not None and gm is not None and assets is not None and assets > 0:
            gp_a = (rev * gm) / assets
            weighted_scores.append((_linear_score(gp_a, best=0.40, worst=0.0), GP_A_WEIGHT))

        # 3. CROA: Cash Return on Assets (Operating Cash Flow / Total Assets)
        ocf = result.financials.operating_cash_flow
        if ocf is not None and assets is not None and assets > 0:
            croa = ocf / assets
            weighted_scores.append((_linear_score(croa, best=0.15, worst=0.0), CROA_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Growth score based on PEG ratio, PE ratio, and forward earnings improvement."""
        weighted_scores: List[Tuple[float, float]] = []

        PEG_GROWTH_WEIGHT = 0.4
        PE_GROWTH_WEIGHT = 0.3
        FORWARD_PE_IMPROVEMENT_WEIGHT = 0.3

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            weighted_scores.append((_log_score(peg, best=0.7, worst=1.8), PEG_GROWTH_WEIGHT))

        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # Empirical finding: higher PE (up to a point) correlates with growth/momentum in this market.
            weighted_scores.append((_log_score(pe, best=150.0, worst=10.0), PE_GROWTH_WEIGHT))

        pe_forward = result.valuation.pe_forward
        if pe is not None and pe_forward is not None and pe_forward > 0 and pe > 0:
            forward_pe_ratio = pe / pe_forward
            weighted_scores.append(
                (_linear_score(forward_pe_ratio, best=2.0, worst=0.5), FORWARD_PE_IMPROVEMENT_WEIGHT)
            )

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0
