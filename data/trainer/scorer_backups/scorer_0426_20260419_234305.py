"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.35,
    "quality": 0.30,
    "growth": 0.20,
    "momentum": 0.15,
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

        # Use geometric mean for composite score
        value_weight = self.weights.get("value", 0.35)
        quality_weight = self.weights.get("quality", 0.30)
        growth_weight = self.weights.get("growth", 0.20)
        momentum_weight = self.weights.get("momentum", 0.15)

        # Applying weights directly to the scores before geometric mean
        weighted_value_score = result.value_score * value_weight
        weighted_quality_score = result.quality_score * quality_weight
        weighted_growth_score = result.growth_score * growth_weight
        weighted_momentum_score = momentum_score * momentum_weight


        # Use geometric mean with weights as exponents
        # Use geometric mean for composite score
        scores = [
            weighted_value_score,
            weighted_quality_score,
            weighted_growth_score,
            weighted_momentum_score,
        ]
        
        # Filter out zero or negative scores before calculating the geometric mean
        positive_scores = [score for score in scores if score > 0]
        
        if positive_scores:
            product = 1.0
            for score in positive_scores:
                product *= score
            composite_score = product ** (1.0 / len(scores))  # Use the original number of scores for the exponent
        else:
            composite_score = 0.0

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
        scores: List[float] = []
        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # 100 if PE <= 8, 0 if PE >= 30
            scores.append(_linear_score(pe, best=8.0, worst=30.0))

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # 100 if PB <= 1, 0 if PB >= 5
            scores.append(_linear_score(pb, best=1.0, worst=5.0))

        return sum(scores) / len(scores) if scores else 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Higher profitability and lower leverage → higher score."""
        scores: List[float] = []

        roe = result.financials.roe
        if roe is not None:
            # 0 if ROE <= 0, 100 if ROE >= 0.25
            scores.append(_linear_score(roe, best=0.25, worst=0.0))

        net_margin = result.financials.net_margin
        if net_margin is not None:
            # 0 if margin <= 0, 100 if margin >= 0.20
            scores.append(_linear_score(net_margin, best=0.20, worst=0.0))

        debt_ratio = result.financials.debt_to_equity
        if debt_ratio is not None:
            # 100 if debt_ratio <= 0.3, 0 if >= 1.0
            scores.append(_linear_score(debt_ratio, best=0.3, worst=1.0))

        return sum(scores) / len(scores) if scores else 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Simplified growth score: higher ROE + lower PEG → higher growth potential."""
        scores: List[float] = []

        roe = result.financials.roe
        if roe is not None:
            # Re-use ROE as a growth proxy (0→0, 0.30→100)
            scores.append(_linear_score(roe, best=0.30, worst=0.0))

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            # Lower PEG is better: 100 if PEG <= 0.5, 0 if PEG >= 2.0
            scores.append(_linear_score(peg, best=0.5, worst=2.0))

        return sum(scores) / len(scores) if scores else 50.0