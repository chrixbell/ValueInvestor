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
    "quality": 0.15,   # Decreased from 0.20 to reallocate weight to growth, continuing a past successful trend.
    "growth": 0.20,    # Increased from 0.15, taking weight from quality. This aligns with a potential market
                       # where growth is more rewarded, which might explain the "higher PE is better" signal
                       # from the value factor.
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score, effectively removing its influence
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
    # Handle non-positive values gracefully, as math.log is undefined for them.
    # If a value is non-positive, it's considered the absolute worst for factors
    # where _log_score is used (e.g., ROE for growth, PE/PB where higher is better).
    # This ensures that truly poor performance is scored at 0, not a neutral 50.0.
    # Note: If 0 is explicitly the BEST score (e.g., for debt), it needs custom handling before calling this.
    if value <= 0 or best <= 0 or worst <= 0:
        # If any input is non-positive, return 0.0. This is a conservative approach.
        # For cases where 'value' can be 0 and is considered 'best' (like 0 debt),
        # that specific case should be handled externally before calling _log_score.
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
        # Its weight has been set to 0 in _DEFAULT_WEIGHTS, so it will not affect the composite score.
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
        """Lower valuation multiples → higher score. (Note: PB scoring is still empirically inverted)."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for value sub-factors. PE_WEIGHT has been removed.
        PB_WEIGHT = 0.25
        PS_WEIGHT = 0.20 
        EV_EBITDA_WEIGHT = 0.20
        DIVIDEND_YIELD_WEIGHT = 0.25
        MARKET_CAP_WEIGHT = 0.10 

        # PE ratio has been moved from _value_score to _growth_score based on empirical findings.

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # Similar to PE, the current scoring (best=6.0, worst=1.0) means higher PB gets a higher score.
            # We maintain this behavior and increase its weight based on empirical results.
            # MODIFICATION: Changed from _linear_score to _log_score for PB ratio.
            # This allows for differentiation among high-PB stocks with diminishing returns.
            weighted_scores.append((_log_score(pb, best=18.0, worst=1.5), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            # Lower PS is better: best=0.5, worst=3.0 correctly assigns 100 to 0.5 and 0 to 3.0
            weighted_scores.append((_linear_score(ps, best=0.5, worst=3.0), PS_WEIGHT))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            # Lower EV/EBITDA is better: best=5.0, worst=15.0 correctly assigns 100 to 5.0 and 0 to 15.0
            weighted_scores.append((_linear_score(ev_to_ebitda, best=5.0, worst=15.0), EV_EBITDA_WEIGHT))

        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield > 0:
            # Higher dividend yield is generally better for a value score.
            # Define best at 4% and worst at 1%.
            weighted_scores.append((_linear_score(dividend_yield, best=0.04, worst=0.01), DIVIDEND_YIELD_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            # Smaller market cap is generally better for a "small cap premium" value factor.
            # Using logarithmic scoring to account for the wide range of market cap values.
            # Define best at 1 billion RMB and worst at 100 billion RMB.
            weighted_scores.append((_log_score(market_cap, best=1_000_000_000.0, worst=100_000_000_000.0), MARKET_CAP_WEIGHT))

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

        # Define internal weights for quality sub-factors.
        # Weights for other factors have been proportionally increased after removing ROE_PER_PE_WEIGHT (0.20).
        # Original sum of other weights = 0.05 + 0.10 + 0.10 + 0.06 + 0.08 + 0.08 + 0.10 + 0.08 + 0.05 + 0.10 = 0.88
        # New sum of weights = 0.88 + 0.20 = 1.08 (for proportional distribution)
        # Multiplier = 1.08 / 0.88 = ~1.22727
        ROE_WEIGHT = 0.05 * 1.22727 # ~0.06136
        NET_MARGIN_WEIGHT = 0.10 * 1.22727 # ~0.12273
        GROSS_MARGIN_WEIGHT = 0.10 * 1.22727 # ~0.12273
        DEBT_TO_EQUITY_WEIGHT = 0.06 * 1.22727 # ~0.07364
        ROE_TO_DEBT_WEIGHT = 0.08 * 1.22727 # ~0.09818
        OCF_MARGIN_WEIGHT = 0.08 * 1.22727 # ~0.09818
        ROA_WEIGHT = 0.10 * 1.22727 # ~0.12273
        FCF_MARGIN_WEIGHT = 0.08 * 1.22727 # ~0.09818
        CURRENT_RATIO_WEIGHT = 0.05 * 1.22727 # ~0.06136
        ROCE_WEIGHT = 0.10 * 1.22727 # ~0.12273

        roe = result.financials.roe
        if roe is not None:
            # MODIFICATION: Changed from _linear_score to _log_score for ROE in quality score.
            # This aims to capture diminishing returns for very high ROE values,
            # while still giving credit to improving ROE at lower levels.
            # Set worst to a small positive value to allow for log transformation.
            weighted_scores.append((_log_score(roe, best=0.20, worst=0.01), ROE_WEIGHT))

        net_margin = result.financials.net_margin
        if net_margin is not None:
            # MODIFICATION: Changed net_margin from _linear_score to _log_score.
            # This aims to better differentiate among lower positive margins and account for
            # diminishing returns at very high margin levels, which is often characteristic of quality metrics.
            # Set worst to a small positive value (0.1%) to allow for log transformation.
            weighted_scores.append((_log_score(net_margin, best=0.15, worst=0.001), NET_MARGIN_WEIGHT))

        gross_margin = result.financials.gross_margin
        if gross_margin is not None:
            weighted_scores.append((_linear_score(gross_margin, best=0.30, worst=0.0), GROSS_MARGIN_WEIGHT))

        debt_ratio = result.financials.debt_to_equity
        if debt_ratio is not None:
            # MODIFICATION: Changed debt_to_equity from _linear_score to _log_score,
            # with explicit handling for zero debt.
            if debt_ratio == 0:
                # Zero debt is generally considered an excellent indicator of quality/financial health.
                score = 100.0
            elif debt_ratio > 0:
                # For positive debt, use log_score to differentiate more effectively among lower debt levels
                # and capture diminishing returns in terms of risk reduction as debt approaches zero.
                # Lower debt is better: best=0.1 (very low but positive), worst=1.2 (high).
                # Values between 0 (handled above) and 0.1 will be clamped to 100 by _log_score.
                score = _log_score(debt_ratio, best=0.1, worst=1.2)
            else: # debt_ratio is negative, indicating negative equity, which is generally very poor.
                score = 0.0
            weighted_scores.append((score, DEBT_TO_EQUITY_WEIGHT))

        # Add interaction term: ROE / Debt-to-Equity
        if roe is not None and debt_ratio is not None and debt_ratio > 0:
            roe_to_debt = roe / debt_ratio
            weighted_scores.append((_linear_score(roe_to_debt, best=0.5, worst=0.0), ROE_TO_DEBT_WEIGHT))
        
        # Add Operating Cash Flow Margin as a sub-factor for quality score.
        operating_cash_flow = result.financials.operating_cash_flow
        revenue = result.financials.revenue
        if operating_cash_flow is not None and revenue is not None and revenue > 0:
            ocf_margin = operating_cash_flow / revenue
            # Higher OCF margin is better: 100 if >= 0.20, 0 if <= 0.0
            weighted_scores.append((_linear_score(ocf_margin, best=0.20, worst=0.0), OCF_MARGIN_WEIGHT))

        # Add Return on Assets (ROA) as a sub-factor for quality score.
        roa = result.financials.roa
        if roa is not None:
            # Higher ROA is better: 100 if ROA >= 0.10, 0 if ROA <= 0.0.
            # ROA measures how efficiently a company uses its assets to generate earnings.
            weighted_scores.append((_linear_score(roa, best=0.10, worst=0.0), ROA_WEIGHT))

        # Add Free Cash Flow Margin as a sub-factor for quality score with a positive weight.
        free_cash_flow = result.financials.free_cash_flow
        revenue = result.financials.revenue
        if free_cash_flow is not None and revenue is not None and revenue > 0:
            fcf_margin = free_cash_flow / revenue
            # Higher FCF margin is better: 100 if >= 0.10, 0 if <= 0.0.
            # FCF margin indicates how much cash a company generates after accounting for capital expenditures.
            weighted_scores.append((_linear_score(fcf_margin, best=0.10, worst=0.0), FCF_MARGIN_WEIGHT))

        # Add Current Ratio as a sub-factor for quality score.
        current_ratio = result.financials.current_ratio
        if current_ratio is not None:
            # Higher current ratio is better for liquidity: 100 if >= 2.0, 0 if <= 1.0.
            # A current ratio below 1.0 indicates poor liquidity, while above 2.0 is generally considered very healthy.
            weighted_scores.append((_linear_score(current_ratio, best=2.0, worst=1.0), CURRENT_RATIO_WEIGHT))

        # REMOVED: ROE/PE as an interaction term to the quality score.
        # This factor's inverse relationship with PE might conflict with the observed positive correlation
        # of high PE with future returns (which is captured by the _growth_score).

        # Add Return on Capital Employed (ROCE) as a sub-factor for quality score.
        net_income = result.financials.net_income
        total_equity = result.financials.total_equity
        debt_to_equity = result.financials.debt_to_equity

        if (net_income is not None and total_equity is not None and total_equity > 0 and
            debt_to_equity is not None and debt_to_equity >= 0): 
            
            # Calculate capital employed as Total Equity + Total Debt (Total Debt = Debt-to-Equity * Total Equity)
            # This is a common approximation for ROCE when EBIT is not available, using Net Income as proxy for earnings.
            capital_employed = total_equity * (1 + debt_to_equity)
            
            if capital_employed > 0:
                roce_proxy = net_income / capital_employed
                # Higher ROCE is better. Best at 15%, worst at 5%.
                weighted_scores.append((_linear_score(roce_proxy, best=0.15, worst=0.05), ROCE_WEIGHT))


        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Growth score based on PEG ratio and PE ratio (as a proxy for growth expectations)."""
        weighted_scores: List[Tuple[float, float]] = [] # (score, weight)

        # Define internal weights for growth sub-factors
        # MODIFICATION: Adjusted PEG_GROWTH_WEIGHT and introduced PE_GROWTH_WEIGHT.
        PEG_GROWTH_WEIGHT = 0.6
        PE_GROWTH_WEIGHT = 0.4 # New weight for PE ratio in growth score.

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            # Lower PEG is better: 100 if PEG <= 0.7, 0 if PEG >= 1.8.
            # Changed from _linear_score to _log_score for PEG ratio.
            weighted_scores.append((_log_score(peg, best=0.7, worst=1.8), PEG_GROWTH_WEIGHT))

        # NEW MODIFICATION: Add PE ratio to the growth score.
        # Empirically, higher PE has shown a positive correlation with forward returns in this context.
        # This aligns with the idea that higher multiples can reflect market expectations of future growth.
        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            # Score PE such that higher PE gets a higher score, using log_score for diminishing returns.
            # Using the same thresholds as previously used for PE in _value_score.
            weighted_scores.append((_log_score(pe, best=100.0, worst=10.0), PE_GROWTH_WEIGHT))


        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0