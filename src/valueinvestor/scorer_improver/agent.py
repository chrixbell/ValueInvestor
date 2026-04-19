"""Phase 3 — LLM agent that autonomously improves scorer.py.

Inspired by karpathy/autoresearch: reads the current scorer code, sends it
to an LLM with experiment history and program context, applies the proposed
change, evaluates via backtest, and keeps or reverts.  Loops indefinitely.
"""

from __future__ import annotations

import difflib
import logging
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from valueinvestor.scorer_improver.evaluator import evaluate_scorer
from valueinvestor.scorer_improver.experiment_log import ExperimentLog

logger = logging.getLogger(__name__)

SCORER_PATH = Path("src/valueinvestor/screener/scorer.py")
PROGRAM_MD_PATH = Path("src/valueinvestor/scorer_improver/program.md")
BACKUP_DIR = Path("data/trainer/scorer_backups")

# Prompt sent to the LLM agent
_AGENT_SYSTEM_PROMPT = """\
You are an autonomous research agent improving a stock scoring algorithm.
You will receive:
1. The current scorer.py source code
2. A program.md file describing the system, available data fields, and constraints
3. Recent experiment results (what worked, what didn't)

Your task: propose ONE specific modification to scorer.py that you believe will
improve the Spearman rank correlation (ρ) between composite_score and 6-month
forward stock return.

Rules:
- Output the COMPLETE new scorer.py file (not a diff, not a partial snippet)
- Keep the MultiFactorScorer class interface intact (score/rank methods)
- Valid Python 3.9+ only
- Handle None values gracefully
- Be creative but grounded — use financial domain knowledge
- Learn from past experiments: don't repeat failed approaches
- Make ONE focused change per iteration (easier to attribute improvements)

Wrap your complete scorer.py output in ```python ... ``` code fences.
After the code, briefly explain what you changed and why (1-2 sentences).
"""


def _get_llm_client():
    """Create an LLM client using load_config() — same resolution as the main pipeline."""
    import os

    from valueinvestor.analysis.llm_client import LLMClient
    from valueinvestor.config import load_config

    cfg = load_config()
    cfg.llm.temperature = 0.7  # Higher creativity for exploration

    provider = cfg.llm.provider
    api_key = cfg.llm.api_key

    if not api_key:
        raise RuntimeError(
            "No LLM API key available. Set GITHUB_TOKEN or GEMINI_API_KEY in .env"
        )

    logger.info("Agent using provider=%s model=%s", provider, cfg.llm.model)
    return LLMClient(config=cfg.llm)


def _make_gemini_fallback_client():
    """Create a Gemini client as runtime fallback when primary provider fails."""
    import os

    from valueinvestor.analysis.llm_client import LLMClient
    from valueinvestor.config import LLMConfig

    gemini_key = os.environ.get("GEMINI_API_KEY")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if not gemini_key:
        return None
    config = LLMConfig(
        provider="gemini",
        model=gemini_model,
        api_key=gemini_key,
        base_url=None,
        max_retries=3,
        temperature=0.7,
    )
    logger.info("Falling back to Gemini (%s)", gemini_model)
    return LLMClient(config=config)


def _read_scorer() -> str:
    """Read the current scorer.py source code."""
    return SCORER_PATH.read_text(encoding="utf-8")


def _write_scorer(code: str) -> None:
    """Write new scorer.py source code."""
    SCORER_PATH.write_text(code, encoding="utf-8")


def _backup_scorer(iteration: int) -> Path:
    """Backup current scorer.py before modification."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"scorer_{iteration:04d}_{ts}.py"
    shutil.copy2(SCORER_PATH, backup_path)
    return backup_path


def _build_prompt(
    scorer_code: str,
    program_md: str,
    experiment_history: str,
    baseline_rho: float,
    best_rho: Optional[float],
) -> str:
    """Build the user prompt for the LLM."""
    status = (
        f"Baseline Spearman ρ: {baseline_rho:.4f}\n"
        f"Best ρ achieved: {best_rho:.4f}" if best_rho is not None else f"Baseline Spearman ρ: {baseline_rho:.4f}\nNo experiments yet."
    )

    # Inject status into program.md
    context = program_md.replace("{current_status}", status)
    context = context.replace("{experiment_history}", experiment_history or "No experiments yet.")

    return (
        f"## Program Context\n\n{context}\n\n"
        f"## Current scorer.py\n\n```python\n{scorer_code}\n```\n\n"
        "Now propose your improvement. Output the COMPLETE new scorer.py "
        "wrapped in ```python ... ``` code fences, followed by a brief explanation."
    )


def _extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response (between ```python ... ``` fences)."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        # Try without language specifier
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest match (likely the full file)
        return max(matches, key=len).strip()
    return None


def _extract_explanation(response: str) -> str:
    """Extract the explanation text after the code block."""
    # Find text after the last ``` fence
    parts = response.split("```")
    if len(parts) >= 3:
        explanation = parts[-1].strip()
        return explanation[:300] if explanation else "No explanation provided"
    return "No explanation provided"


def _compute_diff_summary(old_code: str, new_code: str) -> str:
    """Compute a concise diff summary between old and new scorer code."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new", n=1)
    diff_text = "".join(diff)
    # Truncate to 500 chars
    return diff_text[:500] if diff_text else "(no changes)"


def _validate_scorer() -> bool:
    """Check if the current scorer.py is valid Python and doesn't crash."""
    try:
        code = _read_scorer()
        compile(code, str(SCORER_PATH), "exec")
    except SyntaxError as e:
        logger.warning("Scorer has syntax error: %s", e)
        return False

    # Try importing and running a basic score
    try:
        import importlib

        mod_name = "valueinvestor.screener.scorer"
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.scorer import MultiFactorScorer

        scorer = MultiFactorScorer()
        sr = ScreeningResult(
            company=Company(ticker="TEST", name="Test", market=Market.A_SHARE),
            financials=Financials(ticker="TEST", period="test", roe=0.15, net_margin=0.10, debt_to_equity=0.5),
            valuation=ValuationMetrics(ticker="TEST", date="2024-01-01", pe_ratio=15.0, pb_ratio=2.0, market_cap_rmb=1e10),
        )
        scorer.score(sr)
        assert sr.composite_score >= 0, "Composite score must be non-negative"
        return True
    except Exception as e:
        logger.warning("Scorer validation failed: %s", e)
        return False


def _print_banner(
    iteration: int,
    rho: float,
    baseline_rho: float,
    best_rho: float,
    kept: bool,
    description: str,
) -> None:
    """Print a progress banner to stdout."""
    delta = rho - baseline_rho
    status = "✅ KEPT" if kept else "❌ REVERTED"
    print(f"\n{'='*60}")
    print(f"  Iteration #{iteration}  |  {status}")
    print(f"  ρ = {rho:.4f}  (Δ = {delta:+.4f})  |  Best: {best_rho:.4f}")
    print(f"  {description[:80]}")
    print(f"{'='*60}\n")


def run_improvement_loop(max_iterations: int = 0) -> None:
    """Run the autonomous scorer improvement loop.

    Parameters
    ----------
    max_iterations : int
        If > 0, stop after this many iterations. If 0, run indefinitely.
    """
    exp_log = ExperimentLog()
    program_md = PROGRAM_MD_PATH.read_text(encoding="utf-8")

    # Graceful shutdown on Ctrl-C
    _stop = False

    def _signal_handler(sig, frame):
        nonlocal _stop
        print("\n⏹  Stopping after current iteration …")
        _stop = True

    signal.signal(signal.SIGINT, _signal_handler)

    # Get baseline evaluation
    logger.info("Computing baseline evaluation …")
    baseline_metrics = evaluate_scorer(use_original_scores=False)
    baseline_rho = float(baseline_metrics["spearman_rho"])
    best_rho = float(exp_log.best_rho() or baseline_rho)

    print(f"\n🎯 Baseline Spearman ρ = {baseline_rho:.4f}")
    print(f"📊 Best achieved ρ = {best_rho:.4f}")
    print(f"🔄 Starting improvement loop …\n")

    # Create LLM client
    llm = _get_llm_client()
    _fallback_llm = None  # Gemini fallback, created on first 401

    iteration = exp_log.total_experiments()
    new_iterations = 0  # Count only iterations in THIS run (not historical)
    while not _stop:
        if max_iterations > 0 and new_iterations >= max_iterations:
            print(f"\n🏁 Reached max iterations ({max_iterations}). Stopping.")
            break
        iteration += 1
        new_iterations += 1

        logger.info("=== Iteration #%d ===", iteration)

        # 1. Read current scorer
        original_code = _read_scorer()

        # 2. Backup
        backup_path = _backup_scorer(iteration)
        logger.info("Backed up scorer → %s", backup_path)

        # 3. Get experiment history
        recent = exp_log.read_last_n(10)
        if recent:
            history_lines = []
            for r in recent:
                status = "✅" if r["kept"] else "❌"
                history_lines.append(
                    f"{status} #{r['iteration']}: ρ={r['spearman_rho']:.4f} "
                    f"(Δ={r['delta']:+.4f}) — {r['description'][:100]}"
                )
            experiment_history = "\n".join(history_lines)
        else:
            experiment_history = ""

        # 4. Build prompt and call LLM
        user_prompt = _build_prompt(
            scorer_code=original_code,
            program_md=program_md,
            experiment_history=experiment_history,
            baseline_rho=baseline_rho,
            best_rho=best_rho,
        )

        try:
            response = llm.complete(
                system_prompt=_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as e:
            err_str = str(e).lower()
            # On auth failure, try Gemini fallback once
            if any(k in err_str for k in ("401", "unauthorized", "authentication", "invalid api key")):
                if _fallback_llm is None:
                    _fallback_llm = _make_gemini_fallback_client()
                if _fallback_llm is not None and _fallback_llm is not llm:
                    logger.warning("Primary LLM auth failed, switching to Gemini fallback")
                    llm = _fallback_llm
                    try:
                        response = llm.complete(
                            system_prompt=_AGENT_SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                        )
                    except Exception as e2:
                        logger.error("Gemini fallback also failed: %s", e2)
                        exp_log.log(
                            iteration=iteration,
                            spearman_rho=baseline_rho,
                            baseline_rho=baseline_rho,
                            kept=False,
                            description=f"LLM call failed (both providers): {e2}",
                        )
                        time.sleep(5)
                        continue
                else:
                    logger.error("LLM call failed (auth): %s", e)
                    exp_log.log(
                        iteration=iteration,
                        spearman_rho=baseline_rho,
                        baseline_rho=baseline_rho,
                        kept=False,
                        description=f"LLM call failed: {e}",
                    )
                    time.sleep(5)
                    continue
            else:
                logger.error("LLM call failed: %s", e)
                exp_log.log(
                    iteration=iteration,
                    spearman_rho=baseline_rho,
                    baseline_rho=baseline_rho,
                    kept=False,
                    description=f"LLM call failed: {e}",
                )
                time.sleep(5)
                continue

        # 5. Extract code from response
        new_code = _extract_code(response)
        explanation = _extract_explanation(response)

        if not new_code:
            logger.warning("Could not extract code from LLM response")
            exp_log.log(
                iteration=iteration,
                spearman_rho=baseline_rho,
                baseline_rho=baseline_rho,
                kept=False,
                description="Failed to extract code from LLM response",
            )
            continue

        # 6. Apply change
        _write_scorer(new_code)
        diff_summary = _compute_diff_summary(original_code, new_code)

        # 7. Validate (syntax + basic test)
        if not _validate_scorer():
            logger.warning("Modified scorer failed validation — reverting")
            _write_scorer(original_code)
            exp_log.log(
                iteration=iteration,
                spearman_rho=baseline_rho,
                baseline_rho=baseline_rho,
                kept=False,
                description=f"Validation failed: {explanation}",
                diff_summary=diff_summary,
            )
            _print_banner(iteration, baseline_rho, baseline_rho, best_rho, False, "Validation failed")
            continue

        # 8. Evaluate
        try:
            new_metrics = evaluate_scorer(use_original_scores=False)
            new_rho = new_metrics["spearman_rho"]
        except Exception as e:
            logger.warning("Evaluation failed: %s — reverting", e)
            _write_scorer(original_code)
            exp_log.log(
                iteration=iteration,
                spearman_rho=baseline_rho,
                baseline_rho=baseline_rho,
                kept=False,
                description=f"Evaluation crashed: {e}",
                diff_summary=diff_summary,
            )
            _print_banner(iteration, baseline_rho, baseline_rho, best_rho, False, f"Eval crashed: {e}")
            continue

        # 9. Keep or revert
        improved = bool(new_rho > baseline_rho)
        if improved:
            baseline_rho = float(new_rho)
            if new_rho > best_rho:
                best_rho = float(new_rho)
            logger.info("✅ Improvement! ρ: %.4f → %.4f", baseline_rho, new_rho)
        else:
            _write_scorer(original_code)
            logger.info("❌ No improvement (ρ=%.4f ≤ %.4f) — reverted", new_rho, baseline_rho)

        exp_log.log(
            iteration=iteration,
            spearman_rho=float(new_rho),
            baseline_rho=float(baseline_rho),
            kept=improved,
            description=explanation,
            diff_summary=diff_summary,
            extra={
                "hit_rate_top20": float(new_metrics.get("hit_rate_top20", 0)),
                "mean_excess_return": float(new_metrics.get("mean_excess_return", 0)),
            },
        )

        _print_banner(iteration, new_rho, baseline_rho, best_rho, improved, explanation)

        # Brief pause between iterations
        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  🏁 Improvement loop finished after {new_iterations} new iterations")
    print(f"  📊 Final Spearman ρ = {baseline_rho:.4f}")
    print(f"  🏆 Best ρ = {best_rho:.4f}")
    print(f"  📝 Experiment log: {exp_log.path}")
    print(f"{'='*60}\n")


def show_status() -> None:
    """Print a summary of experiment history."""
    exp_log = ExperimentLog()
    records = exp_log.read_all()

    if not records:
        print("No experiments recorded yet.")
        return

    total = len(records)
    kept = sum(1 for r in records if r.get("kept"))
    best = exp_log.best_rho()
    latest_rho = records[-1]["spearman_rho"]

    print(f"\n📊 Scorer Improvement Status")
    print(f"{'─'*40}")
    print(f"  Total experiments: {total}")
    print(f"  Kept (improved):   {kept}")
    print(f"  Reverted:          {total - kept}")
    print(f"  Success rate:      {kept/total*100:.1f}%")
    print(f"  Latest ρ:          {latest_rho:.4f}")
    print(f"  Best ρ:            {best:.4f}" if best else "  Best ρ:            N/A")
    print(f"{'─'*40}")
    print(f"\nLast 10 experiments:")
    for r in records[-10:]:
        status = "✅" if r["kept"] else "❌"
        print(
            f"  {status} #{r['iteration']}: ρ={r['spearman_rho']:.4f} "
            f"(Δ={r['delta']:+.4f}) — {r.get('description', '')[:60]}"
        )
    print()
