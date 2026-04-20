"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.53,
    "quality": 0.32,
    "growth": 0.15,
    "momentum": 0.0,
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
    # Handle non-positive values gracefully, as math.log is undefined for them
    if value <= 0 or best <= 0 or worst <= 0:
        # For factors where 0 is 'best' (e.g., debt_to_equity), this needs special handling
        # or the factor should be scored differently. For now, returning 50.0 for non-positive
        # values is a neutral fallback.
        return 50.0

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

        # Momentum is a placeholder — set to 50 (neutral) for now.
        # Its weight has been set to 0, so it will not affect the composite score.
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
            # The final result is scaled back to [0, 100).
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
        """Lower valuation multiples → higher score. (Note: current PE/PB scoring is inverted for empirical reasons)."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for value sub-factors
        PE_WEIGHT = 0.10
        PB_WEIGHT = 0.30 
        PS_WEIGHT = 0.20 
        EV_EBITDA_WEIGHT = 0.20
        DIVIDEND_YIELD_WEIGHT = 0.20

        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # The current scoring for PE (best=70.0, worst=10.0) means higher PE gets a higher score.
            # Experiments attempting to reverse this (making lower PE better) resulted in significantly
            # negative correlations. We lean into this empirical finding by increasing its weight.
            # MODIFICATION: Changed from _linear_score to _log_score for PE ratio.
            # This allows for differentiation among high-PE stocks with diminishing returns,
            # potentially better capturing the nuanced positive correlation observed empirically.
            # ONE SPECIFIC MODIFICATION: Increased the 'best' threshold for PE from 70.0 to 100.0.
            weighted_scores.append((_log_score(pe, best=100.0, worst=15.0), PE_WEIGHT))

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # Similar to PE, the current scoring (best=6.0, worst=1.0) means higher PB gets a higher score.
            # We maintain this behavior and increase its weight based on empirical results.
            # MODIFICATION: Changed from _linear_score to _log_score for PB ratio.
            # This allows for differentiation among high-PB stocks with diminishing returns.
            # ONE SPECIFIC MODIFICATION: Increased the 'worst' threshold for PB from 1.0 to 1.5.
            # NEW MODIFICATION: Increased the 'best' threshold for PB from 6.0 to 10.0.
            # ONE SPECIFIC MODIFICATION: Increased the 'best' threshold for PB from 10.0 to 12.0.
            weighted_scores.append((_log_score(pb, best=12.0, worst=1.5), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            # Lower PS is better: best=0.5, worst=3.0 correctly assigns 100 to 0.5 and 0 to 3.0
            weighted_scores.append((_linear_score(ps, best=0.5, worst=3.0), PS_WEIGHT))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            # Lower EV/EBITDA is better: best=5.0, worst=15.0 correctly assigns 100 to 5.0 and 0 to 15.0
            weighted_scores.append((_linear_score(ev_to_ebitda, best=5.0, worst=15.0), EV_EBITDA_WEIGHT))

        # NEW MODIFICATION: Add Dividend Yield as a sub-factor for value score.
        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield > 0:
            # Higher dividend yield is generally better for a value score.
            # Define best at 4% and worst at 1%.
            weighted_scores.append((_linear_score(dividend_yield, best=0.04, worst=0.01), DIVIDEND_YIELD_WEIGHT))


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
        # MODIFICATION: Rebalanced weights to incorporate Operating Cash Flow Margin AND Return on Assets (ROA).
        # NEW MODIFICATION: Added Free Cash Flow Margin as a sub-factor and adjusted weights.
        # NEW MODIFICATION: Added Current Ratio as a sub-factor and rebalanced weights.
        # NEW MODIFICATION: Adjusted weights to make space for ROE/PE interaction term.
        # ONE SPECIFIC MODIFICATION: Increased ROE_PER_PE_WEIGHT from 0.05 to 0.10 and decreased ROE_WEIGHT from 0.18 to 0.13.
        # ONE SPECIFIC MODIFICATION: Rebalanced weights between ROE and ROE_PER_PE to emphasize 'quality at a reasonable price'.
        ROE_WEIGHT = 0.08 
        NET_MARGIN_WEIGHT = 0.10
        GROSS_MARGIN_WEIGHT = 0.10
        # ONE SPECIFIC MODIFICATION: Decreased DEBT_TO_EQUITY_WEIGHT from 0.18 to 0.13.
        DEBT_TO_EQUITY_WEIGHT = 0.13
        ROE_TO_DEBT_WEIGHT = 0.08
        OCF_MARGIN_WEIGHT = 0.12
        ROA_WEIGHT = 0.10
        FCF_MARGIN_WEIGHT = 0.04
        CURRENT_RATIO_WEIGHT = 0.05
        # NEW MODIFICATION: Added ROE/PE as a sub-factor for quality score.
        # ONE SPECIFIC MODIFICATION: Increased ROE_PER_PE_WEIGHT from 0.15 to 0.20.
        ROE_PER_PE_WEIGHT = 0.20

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
            # ORIGINAL: 100 if debt_ratio <= 0.4, 0 if >= 0.8.
            # MODIFICATION: Adjusted thresholds to differentiate more among lower debt levels
            # and be slightly more forgiving for moderate debt.
            weighted_scores.append((_linear_score(debt_ratio, best=0.3, worst=1.0), DEBT_TO_EQUITY_WEIGHT))

        # Add interaction term: ROE / Debt-to-Equity
        if roe is not None and debt_ratio is not None and debt_ratio > 0:
            roe_to_debt = roe / debt_ratio
            weighted_scores.append((_linear_score(roe_to_debt, best=0.5, worst=0.0), ROE_TO_DEBT_WEIGHT))
        
        # NEW MODIFICATION: Add Operating Cash Flow Margin as a sub-factor for quality score.
        operating_cash_flow = result.financials.operating_cash_flow
        revenue = result.financials.revenue
        if operating_cash_flow is not None and revenue is not None and revenue > 0:
            ocf_margin = operating_cash_flow / revenue
            # Higher OCF margin is better: 100 if >= 0.20, 0 if <= 0.0
            weighted_scores.append((_linear_score(ocf_margin, best=0.20, worst=0.0), OCF_MARGIN_WEIGHT))

        # NEW MODIFICATION: Add Return on Assets (ROA) as a sub-factor for quality score.
        roa = result.financials.roa
        if roa is not None:
            # Higher ROA is better: 100 if ROA >= 0.10, 0 if ROA <= 0.0.
            # ROA measures how efficiently a company uses its assets to generate earnings.
            weighted_scores.append((_linear_score(roa, best=0.10, worst=0.0), ROA_WEIGHT))

        # NEW MODIFICATION: Add Free Cash Flow Margin as a sub-factor for quality score.
        free_cash_flow = result.financials.free_cash_flow
        if free_cash_flow is not None and revenue is not None and revenue > 0:
            fcf_margin = free_cash_flow / revenue
            # Higher FCF margin is better: 100 if >= 0.10, 0 if <= 0.0.
            # FCF margin indicates how much cash a company generates after accounting for capital expenditures.
            weighted_scores.append((_linear_score(fcf_margin, best=0.10, worst=0.0), FCF_MARGIN_WEIGHT))

        # NEW MODIFICATION: Add Current Ratio as a sub-factor for quality score.
        current_ratio = result.financials.current_ratio
        if current_ratio is not None:
            # Higher current ratio is better for liquidity: 100 if >= 2.0, 0 if <= 1.0.
            # A current ratio below 1.0 indicates poor liquidity, while above 2.0 is generally considered very healthy.
            weighted_scores.append((_linear_score(current_ratio, best=2.0, worst=1.0), CURRENT_RATIO_WEIGHT))

        # NEW MODIFICATION: Add ROE/PE as an interaction term to the quality score.
        # This aims to capture "quality at a reasonable price".
        pe = result.valuation.pe_ratio
        if roe is not None and pe is not None and pe > 0:
            roe_per_pe = roe / pe
            # Higher ROE/PE is better: 100 if >= 0.03, 0 if <= 0.008.
            # ONE SPECIFIC MODIFICATION: Increased the 'worst' threshold for ROE/PE from 0.005 to 0.008.
            # This makes the factor more selective, penalizing companies with very low ROE relative to their PE ratio more strongly.
            # NEW MODIFICATION: Increased the 'best' threshold for ROE/PE from 0.03 to 0.04.
            # ONE SPECIFIC MODIFICATION: Increased the 'best' threshold for ROE/PE from 0.04 to 0.05.
            # NEW MODIFICATION: Increased the 'best' threshold for ROE/PE from 0.05 to 0.06.
            # ONE SPECIFIC MODIFICATION: Increased the 'worst' threshold for ROE/PE from 0.008 to 0.01.
            # ONE SPECIFIC MODIFICATION: Increased the 'best' threshold for ROE/PE from 0.06 to 0.07.
            # ONE SPECIFIC MODIFICATION: Increased the 'worst' threshold for ROE/PE from 0.01 to 0.015.
            weighted_scores.append((_linear_score(roe_per_pe, best=0.07, worst=0.015), ROE_PER_PE_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Simplified growth score: lower PEG → higher growth potential."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for growth sub-factors
        PEG_GROWTH_WEIGHT = 0.7
        ROE_GROWTH_WEIGHT = 0.3

        roe = result.financials.roe
        if roe is not None:
            # Score ROE: 0 if ROE <= 0.10, 100 if ROE >= 0.30.
            # This emphasizes strong, growth-oriented ROE performance.
            weighted_scores.append((_linear_score(roe, best=0.30, worst=0.10), ROE_GROWTH_WEIGHT))

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            # Lower PEG is better: 100 if PEG <= 0.7, 0 if PEG >= 1.8. Correctly assigns 100 to 0.7 and 0 to 1.8
            weighted_scores.append((_linear_score(peg, best=0.7, worst=1.8), PEG_GROWTH_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0