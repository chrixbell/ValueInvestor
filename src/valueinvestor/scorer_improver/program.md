# Scorer Improvement Agent

## Goal

Maximize Spearman ρ between composite score and forward returns across 1m, 3m, and 6m horizons. A change is kept when it is a **material current-incumbent Pareto improvement**: at least one current horizon improves by more than the acceptance threshold (~0.00010), and no current horizon regresses by more than the behavior-neutral band (~0.00005). Historical-best metrics are tracked separately and should guide retries, but they no longer block a clean current-incumbent improvement.

## File

You modify `src/valueinvestor/screener/scorer.py` only. It contains:
- `_DEFAULT_WEIGHTS`: dict of six factor weights (`value`, `quality`, `growth`, `momentum`, `synergy`, `value_growth`)
- `_linear_score(value, best, worst)`: maps value → [0, 100]
- `MultiFactorScorer` with `score(result)`, `rank(results)`, and per-factor sub-score methods

The trainer requires `_DEFAULT_WEIGHTS` to include all six factors with positive numeric weights summing to `1.0`. Do not remove, zero, or omit `momentum`, `synergy`, or `value_growth`; proposals that start from partial weights are rejected before evaluation.

## Available Fields

`result.valuation`: pe_ratio, pe_forward, pb_ratio, ps_ratio, peg_ratio, dividend_yield, ev_to_ebitda, market_cap_rmb, price — all float|None

`result.financials`: revenue, net_income, gross_margin, net_margin, roe, roa, debt_to_equity, current_ratio, operating_cash_flow, free_cash_flow, total_assets, total_equity — all float|None

`ScreeningResult` only has: `.valuation`, `.financials`, `.company`, `.composite_score`, `.value_score`, `.quality_score`, `.growth_score`, `.momentum_score`, `.synergy_score`, `.value_growth_score`, `.rank`. Do NOT invent fields like `.efficiency_score` — they don't exist and will crash validation.

## Already Implemented — Do NOT Re-Propose

The scorer already has these features. Re-proposing them wastes iterations because they do not produce useful ranking changes. **Critical**: `score()` runs per-stock (no access to other stocks); `rank()` runs across all stocks. Cross-sectional features belong in `rank()`, not `score()`.
- **Cross-sectional percentile ranking**: `rank()` already sorts stocks by `_quality_raw` and `_growth_raw` within each snapshot and assigns percentile-based scores. Value is NOT percentile-ranked; recent ranked-evaluator tests showed adding value percentile ranking regressed all horizons, so do not re-propose it.
- **Geometric mean composite**: `_weighted_geometric_mean()` with scores floored at 10 before log aggregation
- **Synergy term**: `min(value_score, quality_score)` in `rank()`
- **Value-growth interaction**: `sqrt(value_score * growth_score)` in `rank()`
- **Multiplicative loss penalty**: `_loss_penalty()` reduces composite for negative net income relative to market cap
- **Quality raw formula**: `(min(ROE, 0.30) × gross_margin) / (1 + sqrt(max(DTE, 0)))`, with `gross_margin > 0.15`, `DTE <= 2.0`, and an operating-cash-flow/assets multiplier capped at `1.5x`
- **Growth raw formula**: `_compute_growth_raw()` currently ranks inverse PEG (`1 / PEG`)
- **log_score and triangular_score helpers**: for ratio and range-based scoring
- **Momentum proxy**: asset turnover (`revenue / total_assets`) — NOT price momentum
- **Protected value sub-factors**: keep `price × market_cap`, `price × PB`, dividend-yield scoring, and the duplicate/zero earnings-yield append unless you have a genuinely new structural reason. Recent removal attempts regressed the benchmark.

## Constraints

1. `MultiFactorScorer` must keep `score(result)` and `rank(results)` interface
2. Valid Python 3.9+ (no walrus in comprehensions, no 3.10+ features)
3. Only imports: `math`, `logging`, `typing`, and existing modules
4. Handle None fields gracefully — use conditionals, don't crash
5. Scoring ~5000 stocks must complete in <5s
6. Higher composite = better stock
7. No undefined variables — always access via `result.valuation.<field>` or `result.financials.<field>`
8. ScreeningResult objects are not hashable — never use them in sets or as dict keys
9. Verify diff hunks match current code line numbers — stale offsets waste iterations

## Ranked-Evaluator Findings

The trainer now evaluates the actual production workflow: it groups stocks by `snapshot_date`, calls `rank()` for each snapshot, then measures Spearman ρ from the final ranked `composite_score`.

Recent ranked-evaluator run (`eval=ranked-snapshot-v1`) showed:
- **Do NOT try value cross-sectional percentile ranking again.** It was tested as a broad `rank()` change and regressed all horizons.
- **Do NOT flip PB polarity just because low PB sounds cheaper.** Reversing `pb_ratio` scoring was quick-rejected badly; the current unusual PB direction is empirically useful in this dataset.
- **Avoid rare-trigger loss/leverage penalties.** Extra DTE penalties inside `_loss_penalty()` often produce no measurable ranking movement because they affect too few stocks.
- **Avoid OCF/net-income multipliers inside `_compute_quality_raw()`.** Recent earnings-quality multipliers produced no useful movement or were rejected; quality raw is already fragile and heavily used by rank percentiles.
- **Do NOT remove `price × market_cap`, `price × PB`, or dividend yield.** The latest ranked-evaluator runs tested these simplifications and all regressed, with `price × PB` removal damaging 3m and 6m heavily.
- **Do NOT remove the duplicate/zero earnings-yield append.** It looks like a bug, but fresh ranked-evaluator runs showed it is protective; removing it caused large 1m and 6m damage.
- **Do NOT widen the `price × PB` best threshold from `5.0` toward `10.0`.** This fresh test regressed every horizon.
- **Do NOT raise the quality ROE cap from `0.30` to `0.40`.** This was behavior-neutral, not a useful direction.
- **Do NOT loosen the `DTE > 2.0` exclusion in quality raw.** Replacing the exclusion with a leverage cap slightly regressed all horizons.
- **Current-ratio value scoring is a near-miss, not a win.** Adding a triangular current-ratio factor improved 3m/6m slightly but hurt 1m enough to fail target utility; only retry it with an explicit 1m-protection gate.
- **Gross-profit-to-assets is a current-Pareto positive lead.** Retry it only as a 1m-protected or gated variant, not as a broad unconditional weight bump.
- **Do NOT spend another proposal on FCF/assets, FCF/equity, or FCF/debt variants when recent failure guidance marks them as blocked.** These have repeatedly consumed LLM calls and then failed or been skipped before evaluation.
- **Do NOT replace `min(value_score, quality_score)` synergy with a smoother/geometric form.** Fresh ranked quick-eval rejected the geometric-mean replacement across all sampled horizons.
- **Do NOT nudge blend weights for `_growth_abs` vs growth percentile or `log_q` vs quality percentile.** Recent attempts were either below target or 1m-only tradeoffs; they are not strong enough to clear the material threshold.
- **Do NOT retune `boost_alpha`, ROA value thresholds, or add large-cap post-percentile quality bonuses unless recent failure guidance explicitly says the theme is no longer blocked.** Fresh runs showed these are behavior-neutral or below target.
- **Do NOT add broad gross-profit-yield / market-cap factors.** They create the same 3m-only upside with severe 1m/6m damage that quick-eval is designed to reject.
- **Do NOT keep retrying gross-profit-to-assets gates, shape changes, or added asset-turnover gates.** The most recent 10-round batch tested lower gates, net-income gates, turnover gates, log-to-linear shape changes, and weight bumps; all were below target or behavior-neutral.
- **Do NOT tune EV/EBITDA, OCF-yield, PEG, PS, asset-turnover, or loss-penalty thresholds without a specific new structural reason.** Fresh attempts were quick-rejected or failed full eval.
- **Tiny full-eval deltas are not signal.** Treat changes under roughly `0.00005` as behavior-neutral and require roughly `0.00010` upside before keeping a new incumbent. A tiny dip is tolerated if another horizon improves materially, but a proposal with only tiny movement is not useful.
- **Monotonic transforms of ranked raw fields are no-ops.** If `rank()` sorts by `_quality_raw` or `_growth_raw`, replacing raw with `log(raw)`, multiplying all positive values by a constant, or capping without changing order will not change final ranks.

## Strategy

After 11K+ experiments, favor changes that affect many stocks and can change the final per-snapshot ordering:
- **Tune high-coverage value thresholds** by a small amount where most rows have data: market cap preference, OCF yield, gross-profit yield, gross-profit/assets, ROA, earnings yield, or EV/EBITDA. Change one threshold or one sub-factor weight at a time.
- **Preserve empirically useful value interactions.** Do not remove `price × market_cap`, `price × PB`, dividend yield, the duplicate earnings-yield zero append, or the current DTE quality exclusion just because they look unusual.
- **Modify rank-stage composite breadth carefully.** Small structural changes to `synergy` or `value_growth` can matter, but avoid pure weight-only edits. Pair any weight tweak with one concrete structural simplification or threshold change.
- **Try non-monotonic rank-stage gating**, not monotonic raw transforms. For example, use a thresholded bonus/penalty after percentile scores are assigned if it affects a broad subset and is not already captured by existing factors.
- **For near-misses, constrain the retry.** If a candidate only helps 3m/6m while hurting 1m, preserve the 3m/6m mechanism but add a simple gate that avoids degrading high-1m-score stocks.
- **Prefer threshold and gating experiments over deletions.** Recent deletion-style proposals mostly damaged the incumbent scorer; use deletion only when recent history shows the exact component is harmful.
- **Weight-only changes almost never work** (<2% success rate) — only adjust weights alongside a structural change to a sub-score or rank-stage composite term.

## Current Status

{current_status}

## Recent Experiment History

{experiment_history}
