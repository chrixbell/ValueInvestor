"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.41,   # Increased from 0.35
    "quality": 0.35, # Increased from 0.30
    "growth": 0.24,  # Increased from 0.20
    "momentum": 0.00, # Decreased from 0.15
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
        momentum_score = 50.0

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        # --- MODIFICATION START ---
        # Corrected weighted geometric mean calculation
        
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
        # --- MODIFICATION END ---

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
        scores: List[float] = []
        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # 100 if PE <= 10, 0 if PE >= 25.  Inverted for better correlation
            scores.append(_linear_score(pe, best=30.0, worst=10.0))

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # 100 if PB <= 1.5, 0 if PB >= 4.  Inverted for better correlation
            scores.append(_linear_score(pb, best=4.0, worst=1.0))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            scores.append(_linear_score(ps, best=0.5, worst=3.0))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
             scores.append(_linear_score(ev_to_ebitda, best=5.0, worst=15.0))

        return sum(scores) / len(scores) if scores else 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Higher profitability and lower leverage → higher score."""
        scores: List[float] = []

        roe = result.financials.roe
        if roe is not None:
            # 0 if ROE <= 0, 100 if ROE >= 0.20
            scores.append(_linear_score(roe, best=0.20, worst=0.0))

        net_margin = result.financials.net_margin
        if net_margin is not None:
            # 0 if margin <= 0, 100 if margin >= 0.15
            scores.append(_linear_score(net_margin, best=0.15, worst=0.0))

        gross_margin = result.financials.gross_margin
        if gross_margin is not None:
            scores.append(_linear_score(gross_margin, best=0.30, worst=0.0))

        debt_ratio = result.financials.debt_to_equity
        if debt_ratio is not None:
            # 100 if debt_ratio <= 0.4, 0 if >= 0.8
            scores.append(_linear_score(debt_ratio, best=0.4, worst=0.8))

        # Add interaction term: ROE / Debt-to-Equity
        if roe is not None and debt_ratio is not None and debt_ratio > 0:
            roe_to_debt = roe / debt_ratio
            scores.append(_linear_score(roe_to_debt, best=0.5, worst=0.0))

        return sum(scores) / len(scores) if scores else 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Simplified growth score: higher ROE + lower PEG → higher growth potential."""
        scores: List[float] = []

        roe = result.financials.roe
        if roe is not None:
            # Re-use ROE as a growth proxy (0→0, 0.25→100)
            scores.append(_linear_score(roe, best=0.25, worst=0.0))

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            # Lower PEG is better: 100 if PEG <= 0.7, 0 if PEG >= 1.8
            scores.append(_linear_score(peg, best=0.7, worst=1.8))

        return sum(scores) / len(scores) if scores else 50.0