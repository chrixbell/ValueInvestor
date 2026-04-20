"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.41,
    "quality": 0.35,
    "growth": 0.24,
    "momentum": 0.0, # Changed momentum weight from 0.05 to 0.0
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

        # Momentum is a placeholder — set to 50 (neutral) for now.
        # Its weight is now 0.0, so it won't affect the composite score.
        momentum_score = 50.0

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        # Collect scores with positive weights, ensuring they are at least 1.0 to avoid
        # issues with log(0) or 0^weight, and to ensure a positive composite score.
        # Scores are initially in [0, 100].
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = max(1.0, score_val) # Ensure score is >= 1.0

        if not weighted_scores_to_process:
            # If no factors have positive weights, return a neutral score
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k] # We have already filtered for keys with positive weights
            total_weight += weight
            # Normalize score to [0.01, 1.0] range for geometric mean calculation by dividing by 100
            # score_val is guaranteed to be >= 1.0 here, so score_val / 100.0 is >= 0.01
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            # Calculate the weighted geometric mean: (S1^w1 * S2^w2 * ...) ^ (1 / sum(wi))
            # The product_of_powers already contains (S1/100)^w1 * (S2/100)^w2 * ...
            # The final result is scaled back to [0, 100].
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            # Fallback for an unlikely edge case where total_weight becomes 0 despite checks
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
        """Lower valuation multiples → higher score."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for value sub-factors
        # Adjusted weights to accommodate the new dividend_yield factor,
        # while maintaining a sum of 1.0 for internal weights.
        PE_WEIGHT = 0.15
        PB_WEIGHT = 0.15
        PS_WEIGHT = 0.25
        EV_EBITDA_WEIGHT = 0.25
        DIVIDEND_YIELD_WEIGHT = 0.20 # New factor weight

        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # The current scoring for PE (best=50.0, worst=10.0) means higher PE gets a higher score.
            # This is counter-intuitive for "value" but attempts to "correct" it have historically
            # decreased the correlation. We maintain the current behavior but reduce its weight.
            weighted_scores.append((_linear_score(pe, best=50.0, worst=10.0), PE_WEIGHT))

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # Similar to PE, the current scoring (best=4.0, worst=1.0) means higher PB gets a higher score.
            # We maintain this behavior due to past experiment results, but reduce its weight.
            weighted_scores.append((_linear_score(pb, best=4.0, worst=1.0), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            # Lower PS is better: best=0.5, worst=3.0 correctly assigns 100 to 0.5 and 0 to 3.0
            weighted_scores.append((_linear_score(ps, best=0.5, worst=3.0), PS_WEIGHT))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            # Lower EV/EBITDA is better: best=5.0, worst=15.0 correctly assigns 100 to 5.0 and 0 to 15.0
            weighted_scores.append((_linear_score(ev_to_ebitda, best=5.0, worst=15.0), EV_EBITDA_WEIGHT))

        # New: Add Dividend Yield as a value factor
        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield >= 0:
            # Higher dividend yield is generally better for value.
            # Scores 100 for 4% yield or higher, and 0 for 0% yield or lower.
            weighted_scores.append((_linear_score(dividend_yield, best=0.04, worst=0.00), DIVIDEND_YIELD_WEIGHT))


        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Higher profitability and lower leverage → higher score."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for quality sub-factors
        ROE_WEIGHT = 0.25
        NET_MARGIN_WEIGHT = 0.20
        GROSS_MARGIN_WEIGHT = 0.15
        DEBT_TO_EQUITY_WEIGHT = 0.25
        ROE_TO_DEBT_WEIGHT = 0.15 # Interaction term

        roe = result.financials.roe
        if roe is not None:
            # 0 if ROE <= 0, 100 if ROE >= 0.20
            weighted_scores.append((_linear_score(roe, best=0.20, worst=0.0), ROE_WEIGHT))

        net_margin = result.financials.net_margin
        if net_margin is not None:
            # 0 if margin <= 0, 100 if margin >= 0.15
            weighted_scores.append((_linear_score(net_margin, best=0.15, worst=0.0), NET_MARGIN_WEIGHT))

        gross_margin = result.financials.gross_margin
        if gross_margin is not None:
            weighted_scores.append((_linear_score(gross_margin, best=0.30, worst=0.0), GROSS_MARGIN_WEIGHT))

        debt_ratio = result.financials.debt_to_equity
        if debt_ratio is not None:
            # 100 if debt_ratio <= 0.4, 0 if >= 0.8. Correctly assigns 100 to 0.4 and 0 to 0.8
            weighted_scores.append((_linear_score(debt_ratio, best=0.4, worst=0.8), DEBT_TO_EQUITY_WEIGHT))

        # Add interaction term: ROE / Debt-to-Equity
        if roe is not None and debt_ratio is not None and debt_ratio > 0:
            roe_to_debt = roe / debt_ratio
            weighted_scores.append((_linear_score(roe_to_debt, best=0.5, worst=0.0), ROE_TO_DEBT_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Simplified growth score: lower PEG → higher growth potential."""
        scores: List[float] = []

        # MODIFICATION: Re-introducing ROE into growth score, but with higher thresholds
        # to focus on more robust growth, differentiating it from the quality score's ROE.
        roe = result.financials.roe
        if roe is not None:
            # Score ROE: 0 if ROE <= 0.10, 100 if ROE >= 0.30.
            # This emphasizes strong, growth-oriented ROE performance.
            scores.append(_linear_score(roe, best=0.30, worst=0.10))

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            # Lower PEG is better: 100 if PEG <= 0.7, 0 if PEG >= 1.8. Correctly assigns 100 to 0.7 and 0 to 1.8
            scores.append(_linear_score(peg, best=0.7, worst=1.8))

        return sum(scores) / len(scores) if scores else 50.0