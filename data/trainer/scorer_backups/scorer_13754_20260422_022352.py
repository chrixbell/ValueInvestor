"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture stable earnings and balance risk
    "growth": 0.25,    # Increased to capture potential upside
    "momentum": 0.00,  
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
            # To prevent a single zero score from wiping out the entire composite (geometric mean property),
            # we use a small epsilon to ensure the score is at least 0.01/100 if it's not zeroed by logic.
            # However, we keep the raw score_val here to allow legitimate zeroing if intended.
            safe_score = max(0.0, score_val)
            product_of_powers *= (safe_score / 100.0) ** weight

        if total_weight > 0:
            # The product_of_powers is (S1/100)^w1 * (S2/100)^w2...
            # To get the weighted geometric mean in [0, 100]:
            # GM = (product_of_powers)^(1/sum_weights) * 100
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluate valuation metrics."""
        # Use PE and PB as primary value drivers. 
        # We use a mix of forward PE and current PE to balance current vs expected value.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Fallback if PE is None or negative (though we handle via clamping)
        pe_val = pe if (pe is not None and pe > 0) else 20.0
        pb_val = pb if (pb is not None and pb > 0) else 2.0

        # We want low PE/PB to be high score.
        # Using a simple linear scale: 
        # PE target: 15 (best) to 40 (worst). PB target: 1 (best) to 5 (worst).
        pe_s = _linear_score(pe_val, 15.0, 40.0)
        pb_s = _linear_score(pb_val, 1.0, 5.0)
        
        # Dividend yield as a bonus for value
        div_s = 0.0
        if result.valuation.dividend_yield is not None:
            div_s = _linear_score(result.valuation.dividend_yield, 5.0, 0.0)

        return (pe_s * 0.4 + pb_s * 0.4 + div_s * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality metrics (ROE, Margins, Debt)."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE: high is better (target 15% to 30%)
        roe_s = _linear_score(roe * 100, 30.0, 5.0)
        # Margin: high is better (target 20% to 5%)
        margin_s = _linear_score(margin * 100, 20.0, 5.0)
        # Debt: low is better (target 0.2 to 1.5)
        debt_s = _linear_score(debt, 0.2, 1.5)
        
        return (roe_s * 0.5 + margin_s * 0.3 + debt_s * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth metrics (PEG, Revenue Growth)."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        # Use ROE as a proxy for internal growth/efficiency if specific growth data is missing
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG: low is better (target 0.5 to 2.0)
        # Handle PEG = 0 or negative for growth-heavy stocks
        if peg is not None and peg > 0:
            peg_s = _linear_score(peg, 0.5, 2.5)
        else:
            peg_s = 100.0 # High score for negative/zero PEG

        # ROE as growth component
        roe_s = _linear_score(roe * 100, 25.0, 0.0)

        return (peg_s * 0.6 + roe_s * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results