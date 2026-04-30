"""One-time backfill: evaluate all historical scorer variants against 1m/3m targets.

Usage (via CLI)::

    valueinvestor improve-scorer --backfill

For each unique scorer variant found in the backup directory, this module:
1. Writes the scorer code to the target horizon file (e.g. scorer_1m.py)
2. Evaluates it against forward_return_1m / forward_return_3m
3. Keeps the best-performing variant for each horizon

After the backfill, ``scorer_1m.py`` and ``scorer_3m.py`` contain the best
historically tested scorer for their respective horizons (or fall back to the
current ``scorer.py`` if no variant beats it).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BACKUP_DIR = Path("data/trainer/scorer_backups")
SCORER_6M_PATH = Path("src/valueinvestor/screener/scorer.py")
SCORER_1M_PATH = Path("src/valueinvestor/screener/scorer_1m.py")
SCORER_3M_PATH = Path("src/valueinvestor/screener/scorer_3m.py")


def _collect_unique_scorer_variants() -> Dict[str, str]:
    """Return a dict of {md5_hash: code_str} for all unique backup scorers plus current scorer.

    Deduplicates by content hash so we don't evaluate the same code twice.
    """
    variants: Dict[str, str] = {}

    # Include the current scorer.py
    if SCORER_6M_PATH.exists():
        code = SCORER_6M_PATH.read_text(encoding="utf-8")
        h = hashlib.md5(code.encode()).hexdigest()
        variants[h] = code

    # Include existing scorer_1m.py / scorer_3m.py if they differ
    for p in (SCORER_1M_PATH, SCORER_3M_PATH):
        if p.exists():
            code = p.read_text(encoding="utf-8")
            h = hashlib.md5(code.encode()).hexdigest()
            variants[h] = code

    # Scan all backup files
    if BACKUP_DIR.exists():
        for f in BACKUP_DIR.glob("scorer_*.py"):
            try:
                code = f.read_text(encoding="utf-8")
            except Exception:
                continue
            h = hashlib.md5(code.encode()).hexdigest()
            variants[h] = code

    return variants


def _evaluate_scorer_code(
    code: str,
    target_path: Path,
    horizon: str,
) -> Optional[float]:
    """Write *code* to *target_path*, evaluate against *horizon*, return rho.

    Returns ``None`` if evaluation fails.
    """
    from valueinvestor.scorer_improver.evaluator import evaluate_scorer

    original_code: Optional[str] = None
    if target_path.exists():
        original_code = target_path.read_text(encoding="utf-8")

    try:
        target_path.write_text(code, encoding="utf-8")

        # Force-reload the module so the evaluator picks up the new code
        mod_name = f"valueinvestor.screener.{target_path.stem}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        metrics = evaluate_scorer(
            use_original_scores=False,
            horizon=horizon,
            scorer_module=mod_name,
        )
        return float(metrics["spearman_rho"])
    except Exception as exc:
        logger.debug("Evaluation failed for horizon %s: %s", horizon, exc)
        return None
    finally:
        # Restore original file
        if original_code is not None:
            target_path.write_text(original_code, encoding="utf-8")
        # Clean up module cache so we don't leave stale state
        mod_name = f"valueinvestor.screener.{target_path.stem}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]


def _validate_scorer_code(code: str) -> bool:
    """Return True if *code* compiles and passes basic instantiation."""
    try:
        compile(code, "<backfill>", "exec")
    except SyntaxError:
        return False

    # Quick smoke-test via exec
    try:
        namespace: dict = {}
        exec(code, namespace)
        cls = namespace.get("MultiFactorScorer")
        if cls is None:
            return False
        from valueinvestor.data.models import (
            Company, Financials, Market, ScreeningResult, ValuationMetrics,
        )
        sr = ScreeningResult(
            company=Company(ticker="TEST", name="TEST", market=Market.A_SHARE),
            financials=Financials(ticker="TEST", period="test", roe=0.15),
            valuation=ValuationMetrics(ticker="TEST", date="2024-01-01", pe_ratio=15.0, pb_ratio=2.0, market_cap_rmb=1e10),
        )
        cls().score(sr)
        return sr.composite_score >= 0
    except Exception:
        return False


def run_backfill(horizons: Tuple[str, ...] = ("1m", "3m")) -> Dict[str, float]:
    """Evaluate all unique historical scorer variants for the requested horizons.

    For each horizon, the best-performing scorer code is saved to its
    corresponding file (``scorer_1m.py`` / ``scorer_3m.py``).

    Returns a dict of ``{horizon: best_rho}`` for each processed horizon.
    """

    horizon_targets: Dict[str, Path] = {
        "1m": SCORER_1M_PATH,
        "3m": SCORER_3M_PATH,
    }

    print("\n🔍 Collecting unique scorer variants …")
    variants = _collect_unique_scorer_variants()
    print(f"  Found {len(variants)} unique scorer variants to evaluate\n")

    best_results: Dict[str, Tuple[float, str]] = {}  # horizon → (rho, code)

    for horizon in horizons:
        if horizon not in horizon_targets:
            logger.warning("Unknown horizon %s — skipping", horizon)
            continue

        target_path = horizon_targets[horizon]
        print(f"📊 Evaluating for {horizon} target …")

        best_rho = float("-inf")
        best_code: Optional[str] = None
        n_valid = 0
        n_total = len(variants)

        for i, (hash_key, code) in enumerate(variants.items(), 1):
            if i % 20 == 0 or i == n_total:
                print(f"  [{i}/{n_total}] best so far: ρ={best_rho:.4f}")

            if not _validate_scorer_code(code):
                continue

            rho = _evaluate_scorer_code(code, target_path, horizon)
            if rho is None:
                continue
            n_valid += 1

            if rho > best_rho:
                best_rho = rho
                best_code = code

        if best_code is None:
            print(f"  ⚠  No valid scorer found for {horizon} — keeping current {target_path.name}")
        else:
            target_path.write_text(best_code, encoding="utf-8")
            print(f"  ✅ Best {horizon} scorer: ρ={best_rho:.4f} → saved to {target_path}")
            best_results[horizon] = (best_rho, best_code)

        # Clean module cache after horizon
        mod_name = f"valueinvestor.screener.{target_path.stem}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    return {h: rho for h, (rho, _) in best_results.items()}
