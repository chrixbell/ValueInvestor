"""Phase 3 — LLM agent that autonomously improves scorer.py.

Inspired by karpathy/autoresearch: reads the current scorer code, sends it
to an LLM with experiment history and program context, evaluates the proposed
change against **all three** forward-return horizons (1m, 3m, 6m), and keeps
the change if **any** target improves.  Loops indefinitely.
"""

from __future__ import annotations

import difflib
import hashlib
import ast
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from valueinvestor.scorer_improver.evaluator import (
    ScorerEvaluationContext,
)
from valueinvestor.scorer_improver.experiment_log import ExperimentLog
from valueinvestor.scorer_improver.ground_truth import (
    CURRENT_GROUND_TRUTH_FILE,
    LEGACY_GROUND_TRUTH_FILE,
    ensure_current_ground_truth_ready,
    ensure_legacy_ground_truth_ready,
    ground_truth_fingerprint,
)

logger = logging.getLogger(__name__)

SCORER_PATH = Path("src/valueinvestor/screener/scorer.py")
PROGRAM_MD_PATH = Path("src/valueinvestor/scorer_improver/program.md")
BACKUP_DIR = Path("data/trainer/scorer_backups")
BEST_SCORER_PATH = Path("data/trainer/current_best_scorer.py")
BEST_SCORER_META_PATH = Path("data/trainer/current_best_scorer.json")
PROPOSAL_DEBUG_DIR = Path(os.environ.get("IMPROVE_SCORER_PROPOSAL_DEBUG_DIR", "data/trainer/proposal_debug"))
SCORER_MODULE = "valueinvestor.screener.scorer"

_HORIZONS = ("1m", "3m", "6m")
_HORIZON_LABELS = {"1m": "1-month", "3m": "3-month", "6m": "6-month"}
_EVALUATION_SCHEMA_ID = "ranked-snapshot-v1"
_REQUIRED_DEFAULT_WEIGHT_KEYS = (
    "value",
    "quality",
    "growth",
    "momentum",
    "synergy",
    "value_growth",
)

# Number of parallel LLM proposals per iteration (env-overridable)
_N_PARALLEL = int(os.environ.get("IMPROVE_SCORER_PARALLEL", "2"))
# Epsilon for quick-reject threshold
_QUICK_REJECT_MARGIN = 0.02
_QUICK_EVAL_SAMPLE_SIZE = int(os.environ.get("IMPROVE_SCORER_QUICK_SAMPLE_SIZE", "3000"))
_QUICK_1M_DEGRADATION_MARGIN = float(os.environ.get("IMPROVE_SCORER_QUICK_1M_DEGRADATION", "0.01"))
_QUICK_NEUTRAL_DELTA = float(os.environ.get("IMPROVE_SCORER_QUICK_NEUTRAL_DELTA", "0.0005"))
_QUICK_NOOP_DELTA = float(os.environ.get("IMPROVE_SCORER_QUICK_NOOP_DELTA", "0.00005"))
_QUICK_TRADEOFF_DEGRADATION = float(os.environ.get("IMPROVE_SCORER_QUICK_TRADEOFF_DEGRADATION", "0.001"))
_QUICK_UTILITY_REJECT_MARGIN = float(os.environ.get("IMPROVE_SCORER_QUICK_UTILITY_REJECT_MARGIN", "0.0005"))
_BEHAVIOR_NEUTRAL_DELTA = float(os.environ.get("IMPROVE_SCORER_NEUTRAL_DELTA", "0.00005"))
# Maximum allowed degradation in any single horizon for a change to be kept.
# Prevents severe trade-offs where a small improvement in one horizon
# destroys others (e.g. +0.002 in 1m but -0.018 in 3m, -0.013 in 6m).
_MAX_DEGRADATION = 0.01
_LEGACY_GUARD_MAX_DEGRADATION = float(os.environ.get("IMPROVE_SCORER_LEGACY_MAX_DEGRADATION", "0.005"))
_MIN_RHO_IMPROVEMENT = 1e-6
_MIN_ACCEPTED_RHO_IMPROVEMENT = float(os.environ.get("IMPROVE_SCORER_MIN_ACCEPT_DELTA", "0.0001"))
_HORIZON_WEIGHTS = {"1m": 0.25, "3m": 0.35, "6m": 0.40}
# How many recent experiments to show the LLM
_EXPERIMENT_HISTORY_COUNT = int(os.environ.get("IMPROVE_SCORER_HISTORY", "10"))
_proj_root: Optional[Path] = None


def _project_root() -> Path:
    """Return the project root (cached)."""
    global _proj_root
    if _proj_root is None:
        _proj_root = Path(__file__).resolve().parent.parent.parent.parent
    return _proj_root


_REPAIR_INVALID_PROPOSALS = os.environ.get("IMPROVE_SCORER_REPAIR_PROPOSALS", "true").lower() not in {
    "0", "false", "no", "off",
}
_SAVE_INVALID_PROPOSALS = os.environ.get("IMPROVE_SCORER_SAVE_INVALID_PROPOSALS", "true").lower() not in {
    "0", "false", "no", "off",
}
_PARALLEL_PROPOSAL_TIMEOUT_SECONDS = float(os.environ.get("IMPROVE_SCORER_PARALLEL_TIMEOUT_SECONDS", "120"))
_PARALLEL_EARLY_STOP_AFTER_SECONDS = float(
    os.environ.get("IMPROVE_SCORER_PARALLEL_EARLY_STOP_AFTER_SECONDS", "60")
)
_PARALLEL_COMPLETION_GRACE_SECONDS = float(
    os.environ.get("IMPROVE_SCORER_PARALLEL_COMPLETION_GRACE_SECONDS", "10")
)
_PARALLEL_MIN_COMPLETED_PROPOSALS = int(os.environ.get("IMPROVE_SCORER_PARALLEL_MIN_COMPLETED", "1"))
_LLM_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get(
        "IMPROVE_SCORER_LLM_REQUEST_TIMEOUT_SECONDS",
        str(max(1.0, _PARALLEL_PROPOSAL_TIMEOUT_SECONDS - 5.0))
        if _PARALLEL_PROPOSAL_TIMEOUT_SECONDS > 0
        else "0",
    )
)
_MAX_PROPOSAL_TOKENS = int(os.environ.get("IMPROVE_SCORER_MAX_PROPOSAL_TOKENS", "4000"))
_MAX_REPAIR_TOKENS = int(os.environ.get("IMPROVE_SCORER_MAX_REPAIR_TOKENS", "2500"))

_EXPLORATION_LANES = (
    "Lane A, exclusive: make one minimal shape/sign/scaling correction in an existing value sub-factor. Do not change numeric gates, thresholds, market-cap gates, or composite aggregation.",
    "Lane B, exclusive: tune one existing value sub-factor threshold or gate. Do not alter ROE/EV-to-EBITDA, liabilities-to-market-cap, FCF/assets, or FCF/equity variants when failure guidance blocks them.",
    "Lane C, exclusive: make one focused quality-score change only if it is non-monotonic and broad enough to change ranks. Avoid quality_raw multipliers, percentile-blend weight nudges, and value-factor edits.",
    "Lane D, exclusive: simplify one noisy interaction term without changing factor weights, thresholds, gates, or rank/composite mechanics.",
)

_FAILURE_THEME_LABELS = {
    "remove_price_market_cap_subfactor": "removing the price x market-cap value sub-factor",
    "remove_price_pb_subfactor": "removing the price x PB value sub-factor",
    "remove_dividend_yield_subfactor": "removing dividend-yield scoring",
    "remove_earnings_yield_zero_append": "removing the earnings-yield zero append",
    "widen_price_pb_threshold": "widening the price x PB threshold",
    "price_pb_shape_retry": "price x PB scoring shape or threshold changes",
    "raise_quality_roe_cap": "raising the quality ROE cap",
    "current_ratio_retry": "current-ratio factors, gates, or bonuses",
    "fcf_assets_or_equity_retry": "FCF/assets, FCF/equity, or FCF/debt retry variants",
    "quality_raw_multiplier_retry": "new quality_raw multipliers or gates",
    "pe_safe_quality_raw_repair": "pe_safe or quality_raw bug-repair proposals",
    "replace_synergy_min_retry": "replacing min(value_score, quality_score) synergy",
    "growth_blend_weight_retry": "changing the absolute-vs-percentile growth blend weights",
    "quality_percentile_blend_retry": "changing quality percentile/log blend weights",
    "value_boost_alpha_retry": "changing the value-score boost exponent",
    "gross_profit_market_cap_yield_retry": "gross-profit-yield / market-cap value factors",
    "gross_profit_assets_retry": "gross-profit-to-assets gate or shape retries",
    "asset_turnover_retry": "asset-turnover / momentum proxy changes",
    "ocf_yield_threshold_retry": "operating-cash-flow-yield shape or threshold changes",
    "ev_ebitda_threshold_retry": "EV/EBITDA threshold or shape changes",
    "peg_threshold_retry": "PEG growth threshold changes",
    "loss_penalty_scaling_retry": "loss-penalty scaling changes",
    "earnings_yield_zero_weight_retry": "earnings-yield zero-weight changes",
    "large_cap_composite_bonus_retry": "large-cap composite bonus gates",
    "roa_value_threshold_retry": "ROA value-threshold nudges",
    "large_cap_quality_bonus_retry": "large-cap post-percentile quality bonuses",
    "large_cap_direct_roe_value_retry": "large-cap direct ROE value-factor gates",
    "liabilities_market_cap_gate_retry": "liabilities-to-market-cap gate or threshold changes",
    "roe_ev_ebitda_value_shape_retry": "ROE/EV-to-EBITDA value sub-factor threshold or shape changes",
    "growth_forward_pe_fallback_retry": "PEG fallback to forward-PE improvement",
    "forward_pe_value_threshold_retry": "forward-PE value threshold or forward-PE improvement threshold changes",
    "value_quality_composite_bonus_retry": "value-quality composite bonus gates",
}

# Prompt sent to the LLM agent
_AGENT_SYSTEM_PROMPT = """\
You are an autonomous research agent improving a stock scoring algorithm.
You will receive:
1. The current scorer.py source code
2. A program.md file describing the system, available data fields, and constraints
3. Recent experiment results (what worked, what didn't)

Your task: propose ONE specific modification to scorer.py that you believe will
improve the Spearman rank correlation (ρ) between composite_score and forward
stock returns across **three time horizons simultaneously**: 1-month, 3-month,
and 6-month.

Rules:
- Propose a SINGLE focused modification per iteration.
- Use SEARCH/REPLACE blocks for the code change. This is the most reliable format.
- A SEARCH/REPLACE block must look exactly like this:
<<<<<<< SEARCH
[exact code from the file to be replaced]
=======
[new code to replace it with]
>>>>>>> REPLACE
- Keep the SEARCH block as small as possible while remaining unique.
- Do not output a complete replacement file.
- If you use a unified diff patch (---/+++/@@), it must apply cleanly.
- Valid Python 3.9+ only. Handle None values gracefully.
- Learn from past experiments: don't repeat failed approaches.
- Keep the patch as small as possible: change only the lines required.

After the code block, briefly explain what you changed and why (1-2 sentences).
"""


def _get_llm_client():
    """Create an LLM client for improve_scorer.

    Priority order for trainer:
    1. Local LLM (if LOCAL_LLM_ENABLED=true and server is reachable)
    2. Primary provider from config (GitHub, Gemini, etc.)
    3. Gemini fallback
    """
    import os

    from valueinvestor.analysis.llm_client import LLMClient
    from valueinvestor.config import load_config, LLMConfig

    # First, try local LLM if enabled
    local_llm_enabled = os.environ.get("LOCAL_LLM_ENABLED", "").lower() == "true"
    if local_llm_enabled:
        local_base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234")
        local_model = os.environ.get("LOCAL_LLM_MODEL", "default")

        try:
            logger.info("Attempting to connect to local LLM at %s (model: %s)", local_base_url, local_model)
            local_config = LLMConfig(
                provider="local_llm",
                model=local_model,
                api_key="not-needed",
                base_url=local_base_url,
                max_retries=2,
                temperature=0.7,
            )
            client = LLMClient(config=local_config)
            logger.info("✓ Using Local LLM at %s", local_base_url)
            return client
        except Exception as e:
            logger.warning("Local LLM connection failed (%s). Falling back to primary provider.", e)

    # Fall back to primary provider
    cfg = load_config()
    cfg.llm.temperature = 0.7

    provider = cfg.llm.provider

    if provider == "local_llm":
        if not cfg.llm.api_key:
            cfg.llm.api_key = "not-needed"
        logger.info("Agent using local LLM at %s (model: %s)", cfg.llm.base_url, cfg.llm.model)
        return LLMClient(config=cfg.llm)

    api_key = cfg.llm.api_key
    if not api_key:
        raise RuntimeError(
            "No LLM API key available. Set LOCAL_LLM_ENABLED=true or the matching provider key "
            "(for example DEEPSEEK_API_KEY, GITHUB_TOKEN, or GEMINI_API_KEY) in .env"
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
    """Read the current scorer source."""
    return SCORER_PATH.read_text(encoding="utf-8")


def _write_scorer(code: str) -> None:
    """Write new scorer source."""
    SCORER_PATH.write_text(code, encoding="utf-8")


@dataclass(frozen=True)
class _ScorerModuleRef:
    module_name: str
    path: Path
    uses_worktree: bool = False


def _current_scorer_ref() -> _ScorerModuleRef:
    return _ScorerModuleRef(module_name=SCORER_MODULE, path=SCORER_PATH, uses_worktree=True)


def _materialize_temp_scorer(
    workspace_dir: Path,
    code: str,
    *,
    label: str,
    iteration: Optional[int] = None,
) -> _ScorerModuleRef:
    """Persist scorer code to a temp file so rejected proposals never touch SCORER_PATH."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_") or "candidate"
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
    iter_label = f"{iteration}" if iteration is not None else "base"
    path = workspace_dir / f"{safe_label}_{iter_label}_{digest}.py"
    path.write_text(code, encoding="utf-8")
    module_name = f"valueinvestor.scorer_improver._{safe_label}_{iter_label}_{digest}"
    return _ScorerModuleRef(module_name=module_name, path=path, uses_worktree=False)


def _release_scorer_module(ref: _ScorerModuleRef) -> None:
    if ref.uses_worktree:
        return
    sys.modules.pop(ref.module_name, None)


def _backup_scorer(iteration: int) -> Path:
    """Backup the current scorer before modification."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"scorer_{iteration:04d}_{ts}.py"
    shutil.copy2(SCORER_PATH, backup_path)
    return backup_path


_MAX_BACKUPS = int(os.environ.get("IMPROVE_SCORER_MAX_BACKUPS", "500"))


def _cleanup_old_backups() -> int:
    """Remove oldest untracked backups keeping at most _MAX_BACKUPS files.

    The trainer historically created many backup files that may be tracked in
    existing worktrees. Avoid deleting tracked files by default, since that
    creates large, unrelated git diffs during evaluation runs.
    """
    if not BACKUP_DIR.exists():
        return 0
    files = sorted(BACKUP_DIR.glob("scorer_*.py"))

    clean_tracked = os.environ.get("IMPROVE_SCORER_CLEAN_TRACKED_BACKUPS", "").lower() in {
        "1", "true", "yes", "on",
    }
    tracked: set[Path] = set()
    if not clean_tracked:
        try:
            result = subprocess.run(
                ["git", "ls-files", "--", str(BACKUP_DIR)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(_project_root()),
            )
            if result.returncode == 0:
                tracked = {
                    (_project_root() / line.strip()).resolve()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
        except (subprocess.TimeoutExpired, OSError):
            logger.debug("Could not inspect tracked scorer backups", exc_info=True)

    cleanup_candidates = [f for f in files if clean_tracked or f.resolve() not in tracked]
    if len(cleanup_candidates) <= _MAX_BACKUPS:
        return 0
    to_remove = cleanup_candidates[:len(cleanup_candidates) - _MAX_BACKUPS]
    for f in to_remove:
        f.unlink()
    return len(to_remove)


def _save_best_scorer_snapshot(
    *,
    iteration: int,
    rhos: dict[str, float],
    ground_truth_id: str,
) -> None:
    """Persist the post-keep scorer so resume can restore the true best state."""
    BEST_SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCORER_PATH, BEST_SCORER_PATH)
    meta = {
        "iteration": int(iteration),
        "ground_truth_id": ground_truth_id,
        "spearman_rho_1m": round(float(rhos["1m"]), 6),
        "spearman_rho_3m": round(float(rhos["3m"]), 6),
        "spearman_rho_6m": round(float(rhos["6m"]), 6),
        "updated_at": datetime.now().isoformat(),
    }
    BEST_SCORER_META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot_rhos_from_meta(meta: dict) -> dict[str, float]:
    """Extract per-horizon rhos from best-scorer metadata."""
    return {
        h: float(meta[f"spearman_rho_{h}"])
        for h in _HORIZONS
        if meta.get(f"spearman_rho_{h}") is not None
    }


def _is_better_historical_snapshot(
    candidate_rhos: dict[str, float],
    current_best_rhos: dict[str, Optional[float]],
) -> bool:
    """Return true when a candidate exceeds the historical benchmark."""
    return any(
        float(candidate_rhos[h])
        > float(current_best_rhos.get(h) or float("-inf")) + _MIN_ACCEPTED_RHO_IMPROVEMENT
        for h in _HORIZONS
    )


def _load_best_scorer_meta() -> dict:
    """Return best scorer snapshot metadata, or an empty dict if unavailable."""
    if not BEST_SCORER_META_PATH.exists():
        return {}
    try:
        return json.loads(BEST_SCORER_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read best scorer metadata: %s", BEST_SCORER_META_PATH)
        return {}


def _restore_best_scorer_snapshot(ground_truth_id: str) -> Optional[dict]:
    """Restore persisted best scorer for the active ground truth if available."""
    if not BEST_SCORER_PATH.exists():
        return None
    meta = _load_best_scorer_meta()
    meta_gt = meta.get("ground_truth_id")
    if meta_gt and meta_gt != ground_truth_id:
        logger.info(
            "Best scorer snapshot ground truth differs (%s != %s); using current scorer.",
            meta_gt,
            ground_truth_id,
        )
        return None
    current = SCORER_PATH.read_text(encoding="utf-8") if SCORER_PATH.exists() else ""
    best = BEST_SCORER_PATH.read_text(encoding="utf-8")
    if current != best:
        shutil.copy2(BEST_SCORER_PATH, SCORER_PATH)
        try:
            os.utime(SCORER_PATH, None)
        except OSError:
            logger.debug("Could not update scorer mtime after restore", exc_info=True)
        logger.info("Restored best scorer snapshot → %s", SCORER_PATH)
    return meta


def _load_resume_baseline(ground_truth_id: str) -> tuple[Optional[dict], str, bool]:
    """Return the scorer code that should act as the trainer baseline.

    When a best-scorer snapshot exists for the active ground truth, use its
    code in-memory without mutating the tracked scorer file. This keeps
    rejected trainer runs from dirtying the worktree.
    """
    if BEST_SCORER_PATH.exists():
        meta = _load_best_scorer_meta()
        meta_gt = meta.get("ground_truth_id")
        if not meta_gt or meta_gt == ground_truth_id:
            return meta, BEST_SCORER_PATH.read_text(encoding="utf-8"), True
    return None, _read_scorer(), False


def _write_candidate_and_test(
    candidate_code: str,
    *,
    restore_code: str,
) -> tuple[bool, str]:
    """Run the scorer test suite against a candidate worktree scorer."""
    _write_scorer(candidate_code)
    tests_passed, test_summary = _run_scorer_test_suite()
    if not tests_passed:
        _write_scorer(restore_code)
    return tests_passed, test_summary


def _compute_summary_stats(all_records: list[dict]) -> str:
    """Build summary statistics from experiment records for the LLM prompt."""
    if not all_records:
        return ""

    total = len(all_records)
    kept = [r for r in all_records if r.get("kept")]
    kept_count = len(kept)
    rate = kept_count / total * 100 if total else 0

    # Per-horizon improvement counts
    improved_1m = sum(1 for r in kept
                      if r.get("improved_horizons") is not None and "1m" in r["improved_horizons"])
    improved_3m = sum(1 for r in kept
                      if r.get("improved_horizons") is not None and "3m" in r["improved_horizons"])
    improved_6m = sum(1 for r in kept
                      if r.get("improved_horizons") is not None and "6m" in r["improved_horizons"])
    # Fallback for older records that only logged the 6m delta.
    improved_6m += sum(1 for r in kept if r.get("improved_horizons") is None and r.get("delta", 0) > 0)

    # Common failure patterns (simple heuristic: weight-only changes)
    reverted = [r for r in all_records if not r.get("kept")]
    weight_only_patterns = sum(
        1 for r in reverted
        if r.get("description", "") and any(
            kw in r["description"].lower()
            for kw in ("weight", "allocation", "rebalance")
        )
    )

    parts = [
        f"Summary of last {total} experiments:",
        f"  {kept_count} kept ({rate:.1f}%), {total - kept_count} reverted.",
        f"  Per-horizon improvements: 1m={improved_1m}, 3m={improved_3m}, 6m={improved_6m}",
    ]

    if weight_only_patterns / max(total, 1) > 0.3:
        parts.append(
            f"  ⚠️  Weight-only changes appear in {weight_only_patterns} of {total} "
            f"experiments but almost never improve ρ. Avoid proposing weight-only changes."
        )

    invalid = [r for r in all_records if not r.get("diff_summary") and not r.get("kept")]
    if invalid:
        parts.append(
            f"  ⚠️  {len(invalid)} recent attempts did not produce an applicable patch. "
            "Return a real unified diff with @@ hunks."
        )

    return "\n".join(parts)


def _records_for_prompt(records: list[dict], max_entries: int) -> list[dict]:
    """Return a balanced mix of successful and diverse failed experiments.

    Instead of showing only the most recent entries (which are nearly all
    failures at a 2.3% acceptance rate), this surfaces both what worked
    and what failed, giving the LLM signal about productive directions.
    """
    kept = [r for r in records if r.get("kept")]
    failed = [r for r in records if not r.get("kept") and r.get("diff_summary")]

    # Deduplicate failures by theme so the LLM sees variety, not repeats
    seen_themes: set[str] = set()
    diverse_failed: list[dict] = []
    for r in reversed(failed):
        theme = (r.get("diff_summary") or r.get("description", ""))[:80]
        if theme and theme not in seen_themes:
            seen_themes.add(theme)
            diverse_failed.append(r)

    n_successes = min(len(kept), max(5, max_entries // 3))
    n_failures = max_entries - n_successes

    best_successes = sorted(
        kept, key=lambda r: r.get("spearman_rho_6m", 0) or 0, reverse=True,
    )[:n_successes]
    recent_failures = diverse_failed[:n_failures]

    combined = best_successes + recent_failures
    combined.sort(key=lambda r: r.get("timestamp", ""))
    return combined


def _compute_failure_guidance(
    records: list[dict],
    *,
    blocked_theme_records: Optional[list[dict]] = None,
) -> str:
    """Summarize repeated failed themes so the next prompts diversify."""
    failed_text = "\n".join(
        f"{r.get('description', '')}\n{r.get('diff_summary', '')}".lower()
        for r in records
        if not r.get("kept")
    )
    blocked_themes = _blocked_failure_themes(blocked_theme_records or records)
    if not failed_text and not blocked_themes:
        return ""

    patterns = [
        ("aggregation-only changes", ("harmonic mean", "geometric mean", "arithmetic mean", "minimum of")),
        ("value-score percentile/rank transforms", ("value sub-score is converted", "value score is converted", "cross-sectional percentile")),
        ("removing protected value sub-factors", ("price × market_cap", "price * market_cap", "price × pb", "price * pb", "dividend-yield", "dividend yield")),
        ("removing earnings-yield zero append", ("earnings_yield_weight", "earnings yield", "weighted_scores.append((0.0", "duplicate zero")),
        ("price×PB threshold widening", ("price * pb threshold", "price × pb threshold", "price * pb best", "ppb")),
        ("price×PB scoring shape/threshold changes", ("price × pb", "price * pb", "price x pb", "price_pb_weight", "ppb")),
        ("quality ROE cap nudges", ("roe cap", "roe = min", "min(roe", "0.40")),
        ("quality-raw multipliers", ("quality raw", "quality‑raw", "quality_raw")),
        ("rank-stage blend weight nudges", ("absolute growth score", "quality_score from 0.8 to 0.9", "cross-sectional percentile component", "blend weights")),
        ("synergy replacement retries", ("min(value_score, quality_score)", "geometric mean", "synergy term")),
        ("value boost exponent tweaks", ("boost_alpha", "raw value score", "value score exponent")),
        ("gross-profit-yield market-cap retries", ("gross-profit-yield", "gross profit / market cap", "gross profit yield")),
        ("gross-profit-to-assets retries", ("gross-profit-to-assets", "gross profit / total_assets", "gpa_yield")),
        ("asset-turnover retries", ("asset-turnover", "asset turnover", "momentum score", "revenue/total_assets")),
        ("OCF-yield shape or threshold changes", (
            "operating-cash-flow yield",
            "operating-cash-flow-yield",
            "operating cash flow yield",
            "ocf yield",
            "ocf/market cap",
            "ocf_yield",
        )),
        ("EV/EBITDA threshold or shape changes", ("ev/ebitda", "ev_to_ebitda", "enterprise multiples")),
        ("PEG threshold changes", ("peg scoring", "inverse peg", "inv_peg")),
        ("loss-penalty scaling", ("loss penalty", "_loss_penalty")),
        ("ROA threshold nudges", ("roa value sub-factor", "roa value", "best threshold for the roa")),
        ("large-cap direct ROE value gates", ("direct roe value", "roe value sub-factor", "roe_dir", "large-cap-gated direct roe")),
        ("liabilities-to-market-cap gate retries", ("liabilities-to-market-cap", "liabilities to market cap", "liabilities/market cap")),
        ("ROE/EV-to-EBITDA value-shape retries", ("roe_ey", "roe/ev-to-ebitda", "roe/ev‑to‑ebitda", "roe to ev/ebitda")),
        ("PEG fallback-to-forward-PE retries", ("pe_forward", "forward pe improvement", "trailing-to-forward pe")),
        ("forward-PE threshold retries", ("pe_forward", "forward p/e", "forward-pe", "forward pe", "forward_pe_improve")),
        ("value-quality composite bonus gates", ("value_score > 60", "quality_score > 60", "value and quality")),
        ("current-ratio retries", ("current_ratio", "current ratio", "current-ratio")),
        ("FCF/assets or FCF/equity retries", (
            "fcf/ta", "fcf to total assets", "free cash flow to total assets",
            "fcf/equity", "fcf_equity", "free cash flow to equity",
            "free‑cash‑flow‑to‑equity",
        )),
        ("ROE/ROA substitutions", ("replace roe", "replacing roe", "roa")),
        ("PB-sign fixes", ("pb scoring", "price-to-book", "price‑to‑book")),
        ("undefined pe_safe / _compute_quality_raw repairs", ("pe_safe", "undefined", "_compute_quality_raw", "broken quality raw")),
    ]
    lines = []
    if blocked_themes:
        lines.append("Hard proposal exclusions for the next round:")
        for theme in blocked_themes:
            label = _FAILURE_THEME_LABELS.get(theme, theme.replace("_", " "))
            lines.append(f"- Do not propose {label}; this theme is already skipped before evaluation.")
    for label, needles in patterns:
        count = sum(failed_text.count(needle) for needle in needles)
        if count >= 3:
            lines.append(f"- Avoid repeating {label}; it has dominated recent failed/no-op proposals.")
    quick_reason_counts: dict[str, int] = {}
    quick_improved_counts: dict[str, int] = {}
    quick_degraded_counts: dict[str, int] = {}
    for r in records:
        if r.get("kept"):
            continue
        reason = str(r.get("quick_reject_reason") or "").strip()
        if not reason:
            continue
        quick_reason_counts[reason] = quick_reason_counts.get(reason, 0) + 1
        for h in r.get("quick_material_improved_horizons") or []:
            quick_improved_counts[h] = quick_improved_counts.get(h, 0) + 1
        for h in r.get("quick_material_degraded_horizons") or []:
            quick_degraded_counts[h] = quick_degraded_counts.get(h, 0) + 1
    for reason, count in sorted(quick_reason_counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= 2:
            lines.append(f"- Quick-eval has rejected {count} recent proposal(s) for {reason}; avoid that tradeoff pattern.")
    if quick_improved_counts.get("1m", 0) >= 2 and (
        quick_degraded_counts.get("3m", 0) + quick_degraded_counts.get("6m", 0)
    ) >= 2:
        lines.append("- Recent quick-eval rejects show 1m upside paired with 3m/6m damage; propose changes designed for 3m/6m durability, not 1m-only gains.")
    if not lines:
        return ""
    return "Recent failure guidance:\n" + "\n".join(lines)


def _near_miss_guidance(records: list[dict]) -> str:
    """Return prompt guidance from rejected candidates with useful long-horizon upside."""
    near_misses: list[dict] = []
    for record in reversed(records):
        if record.get("kept"):
            continue
        if not record.get("near_miss"):
            continue
        improved = set(record.get("current_material_improved_horizons") or [])
        degraded = set(record.get("current_material_degraded_horizons") or [])
        if not improved.intersection({"3m", "6m"}) or "1m" not in degraded:
            continue
        near_misses.append(record)
        if len(near_misses) >= 2:
            break

    if not near_misses:
        return ""

    lines = [
        "Near-miss repair objective:",
        "- Recent proposals found 3m/6m upside but were rejected because 1m degraded materially.",
        "- Do not repeat those exact diffs. Propose a smaller, offsetting change designed to preserve 3m/6m upside while repairing 1m damage.",
    ]
    for record in near_misses:
        iteration = record.get("iteration", "?")
        delta_1m = float(record.get("current_delta_1m", record.get("delta_1m", 0.0)) or 0.0)
        delta_3m = float(record.get("current_delta_3m", record.get("delta_3m", 0.0)) or 0.0)
        delta_6m = float(record.get("current_delta_6m", record.get("delta", 0.0)) or 0.0)
        summary = str(record.get("diff_summary") or "").replace("\n", " ")
        if len(summary) > 220:
            summary = summary[:217] + "..."
        lines.append(
            f"- #{iteration}: Δ1m={delta_1m:+.4f}, Δ3m={delta_3m:+.4f}, "
            f"Δ6m={delta_6m:+.4f}; avoid repeating: {summary}"
        )
    return "\n".join(lines)


def _strip_scorer_for_prompt(code: str) -> str:
    """Remove docstrings and collapse blank lines from scorer code.

    Saves ~250-400 tokens per LLM call without losing any logic or inline comments.
    """
    import re
    # Remove triple-quoted docstrings (module, class, and function level)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    # Collapse 2+ consecutive blank lines to 1
    code = re.sub(r'\n{2,}', '\n', code)
    return code.strip()


def _add_line_numbers(code: str) -> str:
    """Prefix each line with its 1-based line number for accurate diff hunks."""
    lines = code.split('\n')
    width = len(str(len(lines)))
    return '\n'.join(f'{i+1:>{width}}|{line}' for i, line in enumerate(lines))


def _build_prompt(
    scorer_code: str,
    program_md: str,
    experiment_history: str,
    baseline_rhos: dict[str, float],
    best_rhos: dict[str, Optional[float]],
    is_local_llm: bool = False,
    summary_stats: str = "",
    failure_guidance: str = "",
    lane_instruction: str = "",
) -> str:
    """Build the user prompt for the LLM.

    *baseline_rhos* and *best_rhos* are dicts keyed by horizon ("1m", "3m", "6m").
    """
    # Build status string showing all three horizons + ground truth structure
    status_parts = [
        "Targets: 1-month, 3-month, and 6-month forward returns",
        "Acceptance gate: improve at least one horizon by more than "
        f"{_MIN_ACCEPTED_RHO_IMPROVEMENT:.4f} versus the current baseline, with no "
        "material degradation in any other horizon; historical and legacy guards may "
        "still reject tradeoffs.",
        "You are evaluated on ~108K rows across 28 quarterly snapshots (2016-2025). "
        "The holdout guard uses 8 more-recent snapshots (2023-2024) which naturally show "
        "higher ρ due to temporal correlation — focus on the primary benchmark.",
    ]
    for h in _HORIZONS:
        label = _HORIZON_LABELS[h]
        baseline = baseline_rhos.get(h, 0.0)
        best = best_rhos.get(h)
        if best is not None:
            status_parts.append(f"  {label} — baseline ρ: {baseline:.4f}, best ρ: {best:.4f}")
        else:
            status_parts.append(f"  {label} — baseline ρ: {baseline:.4f}, no experiments yet")
    status = "\n".join(status_parts)

    # Inject status into program.md
    context = program_md.replace("{current_status}", status)

    # Prepend summary stats to experiment history
    full_history = experiment_history or "No experiments yet."
    if summary_stats:
        full_history = summary_stats + "\n\n" + full_history
    if failure_guidance:
        full_history = failure_guidance + "\n\n" + full_history
    if lane_instruction:
        full_history = lane_instruction + "\n\n" + full_history

    if is_local_llm:
        history_lines = full_history.split("\n")[:3]
        context = context.replace("{experiment_history}", "\n".join(history_lines) or "No experiments yet.")

        score_start = scorer_code.find("    def score(")
        if score_start > 0:
            score_end = scorer_code.find("\n    def ", score_start + 1)
            if score_end < 0:
                score_end = len(scorer_code)
            truncated_code = scorer_code[:score_start] + scorer_code[score_start:score_end]
        else:
            truncated_code = scorer_code[:1500]
    else:
        context = context.replace("{experiment_history}", full_history)
        # Use stripped scorer to save tokens, NO line numbers (they confuse LLMs with SEARCH/REPLACE)
        truncated_code = _strip_scorer_for_prompt(scorer_code)

    return (
        f"## Program Context\n\n{context}\n\n"
        f"## Current scorer.py Source\n\n"
        f"```python\n{truncated_code}\n```\n\n"
        "## Instructions\n\n"
        "- Propose ONE modification to improve the metrics above.\n"
        "- Use a SEARCH/REPLACE block for the change. This is the most reliable format.\n"
        "- The SEARCH section MUST match the existing code exactly (whitespace/indentation).\n"
        "- Keep it brief and focused. Do not include prose before the code block.\n"
        "- Alternatively, you may use a unified diff in ```diff fences.\n"
    )


def _extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response (between ```python ... ``` fences)."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return None


def _extract_patch(response: str) -> Optional[str]:
    """Extract a unified diff from an LLM response."""
    for pattern in (r"```diff\s*\n(.*?)```", r"```patch\s*\n(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return _normalize_patch_text(max(matches, key=len))

    generic_blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    for block in generic_blocks:
        stripped = textwrap.dedent(block).lstrip()
        if stripped.startswith(("diff --git", "--- ", "@@ ")):
            return _normalize_patch_text(block)

    lines = response.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("diff --git", "--- ", "@@ ")):
            return _normalize_patch_text("\n".join(lines[idx:]))
    return None


def _normalize_patch_text(patch_text: str) -> str:
    """Normalize common LLM formatting around unified diffs."""
    patch_text = textwrap.dedent(patch_text).lstrip("\r\n")
    lines = patch_text.splitlines(keepends=True)
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(("diff --git", "--- ", "@@ ")):
            start_idx = idx
            break
    normalized = []
    for line in lines[start_idx:]:
        stripped = line.lstrip(" \t")
        if stripped.startswith(("diff --git", "index ", "--- ", "+++ ", "@@ ")):
            normalized.append(stripped)
        elif not line.startswith((" ", "+", "-", "@")) and stripped.startswith(("+", "-", "@")):
            normalized.append(stripped)
        else:
            normalized.append(line)
    return "".join(normalized)


_EXPLANATION_STRIP_PREFIXES = (
    "**What I changed and why:**",
    "**What changed and why:**",
    "**What changed and why**:",
    "**What I changed:**",
    "**Change made:**",
    "**Change explanation:**",
    "**Explanation:**",
    "**Explanation of change:**",
    "**Changes:**",
    "**Summary:**",
    "**What changed:**",
    "What I changed and why:",
    "What changed and why:",
    "Explanation:",
    "Explanation of change:",
    "Change made:",
    "Changes:",
)


def _clean_explanation(text: str) -> str:
    """Strip boilerplate prefixes and normalize whitespace in LLM explanations."""
    cleaned = text.strip()
    for prefix in _EXPLANATION_STRIP_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            break
    # Collapse multiple newlines/spaces
    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned[:300] if cleaned else "No explanation provided"


def _extract_explanation(response: str) -> str:
    """Extract the explanation text after the code block."""
    parts = response.split("```")
    if len(parts) >= 3:
        return _clean_explanation(parts[-1])
    return "No explanation provided"


def _compute_diff_summary(old_code: str, new_code: str) -> str:
    """Compute a concise diff summary between old and new scorer code."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new", n=1)
    diff_text = "".join(diff)
    return diff_text[:500] if diff_text else "(no changes)"


def _compute_full_diff(old_code: str, new_code: str) -> str:
    """Compute the full unified diff between old and new code (not truncated)."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="a", tofile="b", n=1)
    return "".join(diff)


def _lines_equal(left: str, right: str) -> bool:
    """Compare source lines while ignoring newline style only."""
    return left.rstrip("\r\n") == right.rstrip("\r\n")


def _find_hunk_match(
    lines: list[str],
    old_hunk: list[str],
    preferred: int,
) -> tuple[int, int] | None:
    """Find where a hunk's old lines match, preferring the diff line number."""
    if not old_hunk:
        return max(0, min(preferred, len(lines))), 0

    def matches(pos: int) -> bool:
        if pos < 0 or pos + len(old_hunk) > len(lines):
            return False
        return all(_lines_equal(lines[pos + i], old_hunk[i]) for i in range(len(old_hunk)))

    if matches(preferred):
        return preferred, len(old_hunk)

    # LLMs often emit stale line numbers. Search nearby first, then the whole file.
    for radius in (8, 25, 80):
        start = max(0, preferred - radius)
        end = min(len(lines) - len(old_hunk), preferred + radius)
        for pos in range(start, end + 1):
            if matches(pos):
                return pos, len(old_hunk)

    max_start = len(lines) - len(old_hunk)
    for pos in range(max_start + 1):
        if matches(pos):
            return pos, len(old_hunk)

    # Last resort: tolerate tiny blank-line or comment-context drift in LLM hunks.
    old_norm = [line.strip() for line in old_hunk if line.strip()]
    if not old_norm:
        return None

    best: tuple[float, int, int] | None = None
    min_len = max(1, len(old_hunk) - 3)
    max_len = min(len(lines), len(old_hunk) + 3)
    candidate_positions = list(range(max(0, preferred - 80), min(len(lines), preferred + 80) + 1))
    candidate_positions.extend(range(len(lines)))

    seen: set[int] = set()
    for pos in candidate_positions:
        if pos in seen:
            continue
        seen.add(pos)
        for window_len in range(min_len, max_len + 1):
            if pos + window_len > len(lines):
                continue
            window_norm = [line.strip() for line in lines[pos:pos + window_len] if line.strip()]
            if not window_norm:
                continue
            ratio = difflib.SequenceMatcher(None, old_norm, window_norm).ratio()
            if ratio < 0.86:
                continue
            distance_penalty = min(abs(pos - preferred) / 1000, 0.05)
            adjusted = ratio - distance_penalty
            if best is None or adjusted > best[0]:
                best = (adjusted, pos, window_len)

    if best is not None:
        _, pos, window_len = best
        logger.debug("Applied fuzzy patch hunk match at line %d (window=%d)", pos + 1, window_len)
        return pos, window_len
    return None


def _try_apply_patch_with_system_patch(base_code: str, patch_text: str) -> str | None:
    """Fallback to the system patch command, which can apply fuzzy hunks."""
    if not shutil.which("patch"):
        return None

    normalized = _normalize_patch_text(patch_text)
    hunk_start = normalized.find("@@")
    if hunk_start < 0:
        return None
    scorer_patch = "--- scorer.py\n+++ scorer.py\n" + normalized[hunk_start:]

    with tempfile.TemporaryDirectory(prefix="valueinvestor_patch_") as tmp:
        tmp_path = Path(tmp)
        scorer_path = tmp_path / "scorer.py"
        scorer_path.write_text(base_code, encoding="utf-8")
        try:
            result = subprocess.run(
                ["patch", "--batch", "--forward", "--silent", "-p0"],
                input=scorer_patch,
                text=True,
                cwd=tmp_path,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:
            logger.debug("system patch fallback failed", exc_info=True)
            return None
        if result.returncode != 0:
            logger.debug("system patch rejected proposal: %s", result.stderr.strip())
            return None
        patched = scorer_path.read_text(encoding="utf-8")
        return patched if patched != base_code else None


def _try_apply_patch(base_code: str, patch_text: str) -> str | None:
    """Try to apply a unified-diff patch. Returns patched code or None on failure."""
    base_lines = base_code.splitlines(keepends=True)
    patch_lines = _normalize_patch_text(patch_text).splitlines(keepends=True)

    result = list(base_lines)
    offset = 0
    applied_any = False
    i = 0

    while i < len(patch_lines):
        m = re.match(r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@', patch_lines[i])
        if not m:
            i += 1
            continue

        preferred = int(m.group(1)) - 1 + offset
        i += 1
        old_hunk: list[str] = []
        new_hunk: list[str] = []

        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith("@@"):
                break
            if line.startswith(("diff --git", "--- ", "+++ ")):
                break
            if line.startswith("\\ No newline"):
                i += 1
                continue
            if line.startswith(" "):
                old_hunk.append(line[1:])
                new_hunk.append(line[1:])
            elif line.startswith("-"):
                old_hunk.append(line[1:])
            elif line.startswith("+"):
                new_hunk.append(line[1:])
            elif line.strip():
                # Tolerate context lines where the LLM omitted the diff-prefix space.
                old_hunk.append(line)
                new_hunk.append(line)
            i += 1

        match = _find_hunk_match(result, old_hunk, preferred)
        if match is None:
            logger.debug("Hunk at preferred line %d failed to match (%d old lines); "
                         "preview: %s", preferred + 1, len(old_hunk),
                         "".join(old_hunk[:3]).strip()[:120])
            return _try_apply_patch_with_system_patch(base_code, patch_text)
        pos, matched_len = match
        result[pos:pos + matched_len] = new_hunk
        offset += len(new_hunk) - matched_len
        applied_any = True

    patched = "".join(result) if applied_any else None
    if patched is not None and patched != base_code:
        return patched
    return _try_apply_patch_with_system_patch(base_code, patch_text)


def _try_apply_search_replace(base_code: str, response: str) -> str | None:
    """Try to apply aider-style search/replace blocks."""
    import re
    pattern = re.compile(
        r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE", re.DOTALL
    )
    matches = pattern.findall(response)
    if not matches:
        return None

    result = base_code
    applied_any = False
    for search, replace in matches:
        # Search/replace blocks often have slightly different leading/trailing whitespace
        # if the LLM isn't careful. We try to be a bit flexible but prioritize exact match.
        if search in result:
            result = result.replace(search, replace, 1)
            applied_any = True
        else:
            # Try matching while ignoring leading/trailing blank lines in the search block
            search_stripped = search.strip("\r\n")
            if search_stripped and search_stripped in result:
                result = result.replace(search_stripped, replace.strip("\r\n"), 1)
                applied_any = True
            else:
                logger.debug("Search block not found in base code:\n%s", search[:100])

    return result if applied_any else None


def _extract_proposal_code(response: str, original_code: str) -> tuple[Optional[str], str, str]:
    """Extract proposal code from a unified diff or search/replace response."""
    explanation = _extract_explanation(response)

    # 1. Try search/replace blocks first (very robust for LLMs)
    patched_code = _try_apply_search_replace(original_code, response)
    if patched_code is not None:
        if patched_code != original_code:
            return patched_code, explanation, _compute_diff_summary(original_code, patched_code)

    # 2. Try unified diff patch
    patch_text = _extract_patch(response)
    if patch_text:
        patched_code = _try_apply_patch(original_code, patch_text)
        if patched_code is not None:
            return patched_code, explanation, _compute_diff_summary(original_code, patched_code)
        logger.warning("LLM returned a patch that could not be applied")

    return None, explanation, ""


def _proposal_is_valid(new_code: Optional[str], diff_summary: str) -> bool:
    """Return True when a response produced an actual scorer change."""
    return bool(new_code) and diff_summary not in {"", "(no changes)"}


def _proposal_syntax_error(new_code: Optional[str]) -> str:
    """Return a syntax-error message for proposed code, or empty string."""
    if not new_code:
        return ""
    try:
        compile(new_code, str(SCORER_PATH), "exec")
    except SyntaxError as exc:
        return str(exc)
    return ""


def _build_repair_prompt(user_prompt: str, bad_response: str, reason: str = "") -> str:
    """Ask the LLM to repair an invalid/no-op proposal without changing intent."""
    excerpt = bad_response[:3500]
    reason_text = f"Reason: {reason}\n\n" if reason else ""
    return (
        f"{user_prompt}\n\n"
        "## Repair Required\n\n"
        f"{reason_text}"
        "Your previous response did not produce an applicable code change. "
        "Return ONLY a valid unified diff for src/valueinvestor/screener/scorer.py. "
        "The diff must include ---/+++ file headers and @@ hunk headers, and it must "
        "apply cleanly to the exact scorer.py shown above. The resulting scorer.py must "
        "compile as valid Python. Do not include prose before the diff.\n\n"
        f"Previous response excerpt:\n```\n{excerpt}\n```"
    )


def _llm_complete(
    llm_client,
    user_prompt: str,
    max_tokens: int,
    *,
    retry_empty: bool = True,
    request_timeout: Optional[float] = None,
) -> str:
    """Call an LLM client with a token cap, tolerating test doubles without kwargs."""
    kwargs = {
        "system_prompt": _AGENT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
    }
    if request_timeout is not None and request_timeout > 0:
        kwargs["timeout"] = request_timeout

    while True:
        try:
            response = llm_client.complete(**kwargs)
            break
        except TypeError as exc:
            message = str(exc)
            if "timeout" in message and "timeout" in kwargs:
                kwargs.pop("timeout", None)
                continue
            if "max_tokens" in message and "max_tokens" in kwargs:
                kwargs.pop("max_tokens", None)
                continue
            raise

    if response.strip() or max_tokens <= 0 or not retry_empty:
        return response

    retry_max_tokens = min(max_tokens, _MAX_REPAIR_TOKENS)
    logger.warning(
        "LLM returned empty content at max_tokens=%d; retrying once with max_tokens=%d",
        max_tokens,
        retry_max_tokens,
    )
    retry_kwargs = {
        "system_prompt": _AGENT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "max_tokens": retry_max_tokens,
    }
    if request_timeout is not None and request_timeout > 0:
        retry_kwargs["timeout"] = request_timeout
    while True:
        try:
            return llm_client.complete(**retry_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "timeout" in message and "timeout" in retry_kwargs:
                retry_kwargs.pop("timeout", None)
                continue
            if "max_tokens" in message and "max_tokens" in retry_kwargs:
                retry_kwargs.pop("max_tokens", None)
                continue
            raise


def _save_proposal_debug(
    *,
    iteration: int | None,
    stage: str,
    response: str,
    diff_summary: str,
) -> str:
    """Persist invalid raw LLM output so failed rounds can be inspected."""
    if not _SAVE_INVALID_PROPOSALS:
        return ""

    try:
        PROPOSAL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        iter_label = f"{iteration:06d}" if iteration is not None else "unknown"
        path = PROPOSAL_DEBUG_DIR / f"{int(time.time_ns())}_iter{iter_label}_{stage}.md"
        patch_text = _extract_patch(response) or ""
        body = (
            f"# Invalid scorer proposal\n\n"
            f"- iteration: {iter_label}\n"
            f"- stage: {stage}\n"
            f"- diff_summary: {diff_summary or '(none)'}\n\n"
            f"## Extracted Patch\n\n```diff\n{patch_text}\n```\n\n"
            f"## Raw Response\n\n{response}\n"
        )
        path.write_text(body, encoding="utf-8")
        return str(path)
    except Exception:
        logger.debug("Could not save proposal debug artifact", exc_info=True)
        return ""


def _complete_proposal(
    llm_client,
    user_prompt: str,
    original_code: str,
    iteration: int | None = None,
    request_timeout: Optional[float] = None,
) -> dict:
    """Call the LLM and optionally repair invalid/no-op proposal responses."""
    response = _llm_complete(
        llm_client,
        user_prompt,
        _MAX_PROPOSAL_TOKENS,
        request_timeout=request_timeout,
    )
    new_code, explanation, diff_summary = _extract_proposal_code(response, original_code)
    repaired = False
    debug_paths: list[str] = []
    syntax_error = _proposal_syntax_error(new_code)

    if not _proposal_is_valid(new_code, diff_summary) or syntax_error:
        debug_path = _save_proposal_debug(
            iteration=iteration,
            stage="initial",
            response=response,
            diff_summary=diff_summary,
        )
        if debug_path:
            debug_paths.append(debug_path)

    if _REPAIR_INVALID_PROPOSALS and (
        not _proposal_is_valid(new_code, diff_summary) or syntax_error
    ):
        reason = syntax_error or "No applicable code change was extracted."
        repair_prompt = _build_repair_prompt(user_prompt, response, reason=reason)
        try:
            repaired_response = _llm_complete(
                llm_client,
                repair_prompt,
                _MAX_REPAIR_TOKENS,
                retry_empty=False,
                request_timeout=request_timeout,
            )
            repaired_code, repaired_explanation, repaired_diff = _extract_proposal_code(
                repaired_response,
                original_code,
            )
            repaired_syntax_error = _proposal_syntax_error(repaired_code)
            if _proposal_is_valid(repaired_code, repaired_diff) and not repaired_syntax_error:
                new_code = repaired_code
                explanation = repaired_explanation
                diff_summary = repaired_diff
                syntax_error = ""
                repaired = True
            else:
                debug_path = _save_proposal_debug(
                    iteration=iteration,
                    stage="repair",
                    response=repaired_response,
                    diff_summary=repaired_diff,
                )
                if debug_path:
                    debug_paths.append(debug_path)
        except Exception as e:
            logger.warning("Proposal repair call failed: %s", e)

    syntax_error = _proposal_syntax_error(new_code)
    if syntax_error:
        explanation = f"Syntax-invalid proposal: {syntax_error}. {explanation}".strip()
        new_code = None
        diff_summary = ""

    return {
        "code": new_code,
        "explanation": explanation,
        "diff_summary": diff_summary,
        "repaired": repaired,
        "debug_paths": debug_paths,
    }


def _proposal_worker_entry(
    iteration: int,
    prompt: str,
    original_code: str,
    request_timeout: float,
    result_queue,
) -> None:
    """Generate one proposal in a child process.

    Keeping network calls in child processes lets the parent terminate a stuck
    provider request. Thread cancellation cannot interrupt an in-flight HTTP
    request, which caused completed trainer runs to wait for late LLM responses.
    """
    try:
        llm = _create_single_client()
        result = _complete_proposal(
            llm,
            prompt,
            original_code,
            iteration=iteration,
            request_timeout=request_timeout if request_timeout > 0 else None,
        )
        result_queue.put({"ok": True, "iteration": iteration, "result": result})
    except BaseException as exc:
        result_queue.put({
            "ok": False,
            "iteration": iteration,
            "error": f"{type(exc).__name__}: {exc}",
        })


def _multiprocessing_context():
    """Return the configured multiprocessing context for proposal workers."""
    method = os.environ.get("IMPROVE_SCORER_MULTIPROCESS_CONTEXT", "spawn").strip() or "spawn"
    try:
        return multiprocessing.get_context(method)
    except ValueError:
        logger.warning("Invalid multiprocessing context %r; using platform default", method)
        return multiprocessing.get_context()


def _terminate_proposal_process(process: multiprocessing.Process) -> None:
    """Terminate a proposal process and escalate to kill if it ignores terminate."""
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        try:
            process.kill()
        except AttributeError:
            logger.warning("Process %s did not exit after terminate()", process.name)
        else:
            process.join(timeout=2)


def _run_parallel_proposal_workers(
    prompts: list[tuple[int, str]],
    original_code: str,
    *,
    timeout_seconds: float,
    request_timeout_seconds: float,
    early_stop_after_seconds: float = 0.0,
    completion_grace_seconds: float = 0.0,
    min_completed_proposals: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run proposal generation in cancellable child processes.

    Returns ``(proposals, events)``. Events contain failed/timed-out iterations
    for the caller to log in the experiment history.
    """
    if not prompts:
        return [], []

    ctx = _multiprocessing_context()
    result_queue = ctx.Queue()
    processes: dict[int, multiprocessing.Process] = {}
    proposals: list[dict] = []
    events: list[dict] = []
    received: set[int] = set()
    latest_valid_proposal_at: Optional[float] = None

    for iteration, prompt in prompts:
        process = ctx.Process(
            target=_proposal_worker_entry,
            args=(iteration, prompt, original_code, request_timeout_seconds, result_queue),
            name=f"scorer-proposal-{iteration}",
        )
        process.start()
        processes[iteration] = process

    started_at = time.monotonic()
    deadline = started_at + timeout_seconds if timeout_seconds > 0 else None
    early_deadline = (
        started_at + early_stop_after_seconds
        if early_stop_after_seconds > 0 and min_completed_proposals > 0
        else None
    )
    stopped_early = False

    def _handle_message(message: dict) -> None:
        nonlocal latest_valid_proposal_at
        iteration = int(message.get("iteration", 0))
        if not iteration or iteration in received:
            return
        received.add(iteration)
        process = processes.pop(iteration, None)
        if process is not None:
            process.join(timeout=1)
            if process.is_alive():
                _terminate_proposal_process(process)
        if message.get("ok"):
            result = dict(message.get("result") or {})
            result["iteration"] = iteration
            proposals.append(result)
            if _proposal_is_valid(result.get("code"), result.get("diff_summary", "")):
                latest_valid_proposal_at = time.monotonic()
        else:
            events.append({
                "iteration": iteration,
                "kind": "error",
                "message": message.get("error") or "proposal worker failed",
            })

    while processes:
        while True:
            try:
                _handle_message(result_queue.get_nowait())
            except queue.Empty:
                break

        for iteration, process in list(processes.items()):
            if process.exitcode is None:
                continue
            process.join(timeout=1)
            processes.pop(iteration, None)
            if iteration not in received:
                received.add(iteration)
                events.append({
                    "iteration": iteration,
                    "kind": "error",
                    "message": f"proposal worker exited with code {process.exitcode}",
                })

        if not processes:
            break

        valid_proposal_count = sum(
            1
            for proposal in proposals
            if _proposal_is_valid(proposal.get("code"), proposal.get("diff_summary", ""))
        )

        if (
            early_deadline is not None
            and valid_proposal_count >= min_completed_proposals
            and time.monotonic() >= early_deadline
        ):
            stopped_early = True
            break

        if (
            completion_grace_seconds > 0
            and min_completed_proposals > 0
            and valid_proposal_count >= min_completed_proposals
            and latest_valid_proposal_at is not None
            and time.monotonic() - latest_valid_proposal_at >= completion_grace_seconds
        ):
            stopped_early = True
            break

        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_time = min(0.25, remaining)
        else:
            wait_time = 0.25

        try:
            _handle_message(result_queue.get(timeout=wait_time))
        except queue.Empty:
            pass

    for iteration, process in list(processes.items()):
        if iteration not in received:
            _terminate_proposal_process(process)
            received.add(iteration)
            elapsed = time.monotonic() - started_at
            valid_proposal_count = sum(
                1
                for proposal in proposals
                if _proposal_is_valid(proposal.get("code"), proposal.get("diff_summary", ""))
            )
            message = (
                f"LLM call stopped after {elapsed:.1f}s once {valid_proposal_count} valid proposal(s) returned"
                if stopped_early
                else f"LLM call timed out after {timeout_seconds:.1f}s"
            )
            events.append({
                "iteration": iteration,
                "kind": "timeout",
                "message": message,
                "proposal_timeout_seconds": round(elapsed if stopped_early else timeout_seconds, 3),
            })
        process.join(timeout=1)
        processes.pop(iteration, None)

    result_queue.close()
    result_queue.join_thread()

    proposals.sort(key=lambda p: p["iteration"])
    events.sort(key=lambda e: e["iteration"])
    return proposals, events


def _is_duplicate_diff(diff_summary: str, recent_records: list[dict]) -> bool:
    """Return True when this proposal repeats a recent evaluated change."""
    if not diff_summary or diff_summary == "(no changes)":
        return False
    return any(r.get("diff_summary") == diff_summary for r in recent_records)


def _semantic_signal_text(text: str) -> str:
    """Return proposal text excluding unchanged unified-diff context lines."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("---", "+++", "@@")):
            continue
        if line.startswith(" ") and not stripped.startswith(("+", "-")):
            continue
        lines.append(line)
    normalized = "\n".join(lines).lower()
    return normalized.translate(str.maketrans({
        "‑": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }))


def _semantic_failure_theme(text: str) -> str:
    """Return a coarse theme for proposals that repeatedly fail despite varied diffs."""
    normalized = _semantic_signal_text(text)
    remove_words = (
        "remove", "removed", "removing", "delete", "deleted", "deleting",
        "drop", "dropped", "dropping", "fix", "fixed", "fixing",
    )
    if any(word in normalized for word in remove_words):
        if any(
            needle in normalized
            for needle in (
                "price × market_cap",
                "price x market_cap",
                "price * market_cap",
                "price×market_cap",
                "price×market-cap",
                "price × market-cap",
                "price * market-cap",
                "price_market_cap_weight",
            )
        ):
            return "remove_price_market_cap_subfactor"
        if any(
            needle in normalized
            for needle in (
                "price × pb",
                "price x pb",
                "price * pb",
                "price×pb",
                "price_pb_weight",
                "ppb",
            )
        ):
            return "remove_price_pb_subfactor"
        if any(
            needle in normalized
            for needle in (
                "dividend-yield",
                "dividend yield",
                "dividend_yield",
                "dividend_yield_weight",
            )
        ):
            return "remove_dividend_yield_subfactor"
        if any(
            needle in normalized
            for needle in (
                "earnings_yield_weight",
                "earnings yield",
                "earn-yield",
                "weighted_scores.append((0.0",
            )
        ) and any(
            needle in normalized
            for needle in ("zero", "double", "duplicate", "redundant", "append")
        ):
            return "remove_earnings_yield_zero_append"

    threshold_words = ("raise", "raised", "raising", "increase", "increased", "widen", "widened", "widening")
    if any(word in normalized for word in threshold_words):
        if any(
            needle in normalized
            for needle in ("price * pb", "price × pb", "price x pb", "ppb", "price_pb_weight")
        ) and any(needle in normalized for needle in ("threshold", "best", "ideal", "log-score", "log score")):
            return "widen_price_pb_threshold"
        if any(
            needle in normalized
            for needle in ("roe cap", "roe_cap", "min(roe", "roe = min", "quality roe")
        ):
            return "raise_quality_roe_cap"

    if any(
        needle in normalized
        for needle in ("price * pb", "price × pb", "price x pb", "price×pb", "price_pb_weight", "ppb")
    ) and any(
        needle in normalized
        for needle in (
            "_log_score",
            "_linear_score",
            "best=",
            "worst=",
            "threshold",
            "logarithmic",
            "linear",
            "scoring",
        )
    ):
        return "price_pb_shape_retry"

    if any(
        needle in normalized
        for needle in ("current_ratio", "current ratio", "current-ratio")
    ) and any(
        word in normalized
        for word in ("add", "added", "bonus", "factor", "sub-factor", "gate", "gated", "threshold", "triangular")
    ):
        return "current_ratio_retry"

    if any(needle in normalized for needle in ("fcf", "free cash flow", "free‑cash‑flow")) and any(
        needle in normalized
        for needle in (
            "total assets",
            "to assets",
            "fcf/ta",
            "fcf_ta",
            "equity",
            "fcf/equity",
            "fcf_equity",
        )
    ) and any(
        word in normalized
        for word in ("add", "added", "bonus", "factor", "sub-factor", "multiplier", "yield", "reward")
    ):
        return "fcf_assets_or_equity_retry"

    if any(needle in normalized for needle in ("quality_raw", "quality raw", "quality‑raw")) and any(
        word in normalized
        for word in ("bonus", "multiplier", "reward", "gate", "gated", "net margin", "current ratio", "fcf")
    ):
        return "quality_raw_multiplier_retry"

    if any(
        needle in normalized
        for needle in ("min(value_score, quality_score)", "min(r.value_score, r.quality_score)", "synergy term")
    ) and any(
        needle in normalized
        for needle in ("geometric mean", "sqrt", "replace", "replaced", "smooth")
    ):
        return "replace_synergy_min_retry"

    if any(
        needle in normalized
        for needle in ("absolute growth score", "_growth_abs", "final growth blend", "blended =")
    ) and any(
        needle in normalized
        for needle in ("0.4", "0.6", "0.3", "0.7", "weight", "influence")
    ):
        return "growth_blend_weight_retry"

    if any(
        needle in normalized
        for needle in ("quality_score", "quality sub-score", "cross-sectional percentile component")
    ) and any(
        needle in normalized
        for needle in ("0.1 * log_q", "0.9 * cross_score", "0.8 to 0.9", "percentile component")
    ):
        return "quality_percentile_blend_retry"

    if "boost_alpha" in normalized or (
        "value score" in normalized
        and any(needle in normalized for needle in ("exponent", "flatten", "flattens"))
    ):
        return "value_boost_alpha_retry"

    if any(
        needle in normalized
        for needle in ("gross-profit-yield", "gross profit yield", "gross profit / market cap", "gp_yield")
    ):
        return "gross_profit_market_cap_yield_retry"

    if any(
        needle in normalized
        for needle in ("gross-profit-to-assets", "gross profit to assets", "gross profit / total_assets", "gpa_yield")
    ):
        return "gross_profit_assets_retry"

    if any(
        needle in normalized
        for needle in ("asset-turnover", "asset turnover", "_momentum_score", "momentum score", "revenue/total_assets")
    ):
        return "asset_turnover_retry"

    if any(
        needle in normalized
        for needle in (
            "operating-cash-flow yield",
            "operating-cash-flow-yield",
            "operating cash flow yield",
            "ocf yield",
            "ocf/market cap",
            "ocf_yield",
        )
    ):
        return "ocf_yield_threshold_retry"

    if any(
        needle in normalized
        for needle in ("ev/ebitda", "ev_to_ebitda", "ev‑ebitda", "enterprise multiples")
    ) and any(
        needle in normalized
        for needle in ("optimal", "threshold", "triangular", "log-linear", "log score", "weight")
    ):
        return "ev_ebitda_threshold_retry"

    if any(
        needle in normalized
        for needle in ("peg scoring", "inverse peg", "inv_peg", "peg_growth_weight")
    ) and any(
        needle in normalized
        for needle in ("best", "worst", "threshold", "window", "reference")
    ):
        return "peg_threshold_retry"

    if any(needle in normalized for needle in ("loss penalty", "_loss_penalty")) and any(
        needle in normalized
        for needle in ("scaling", "constant", "5.0", "6.0", "more severe")
    ):
        return "loss_penalty_scaling_retry"

    if any(
        needle in normalized
        for needle in (
            "liabilities-to-market-cap",
            "liabilities‑to‑market‑cap",
            "liabilities to market cap",
            "liabilities/market cap",
            "liabilities-to-market cap",
        )
    ) and any(
        needle in normalized
        for needle in ("gate", "threshold", "50_000_000_000", "30_000_000_000", "market_cap >")
    ):
        return "liabilities_market_cap_gate_retry"

    if (
        any(
            needle in normalized
            for needle in (
                "roe_ey",
                "roe/ev-to-ebitda",
                "roe/ev‑to‑ebitda",
                "roe to ev/ebitda",
                "roe-to-ev/ebitda",
            )
        )
        or ("roe" in normalized and "ev_to_ebitda" in normalized)
    ) and any(
        needle in normalized
        for needle in ("best", "worst", "threshold", "shape", "linear_score", "log_score", "logarithmic")
    ):
        return "roe_ev_ebitda_value_shape_retry"

    if any(
        needle in normalized
        for needle in ("_compute_growth_raw", "pe_forward", "forward pe", "forward-p/e", "forward p/e")
    ) and any(
        needle in normalized
        for needle in ("fallback", "peg unavailable", "trailing-to-forward", "pe/pe_forward")
    ):
        return "growth_forward_pe_fallback_retry"

    if any(
        needle in normalized
        for needle in (
            "pe_forward",
            "forward pe",
            "forward-pe",
            "forward p/e",
            "forward‑pe",
            "forward_pe_improve",
            "forward-pe improvement",
        )
    ) and any(
        needle in normalized
        for needle in ("best", "worst", "threshold", "log-score", "log score", "linear_score")
    ):
        return "forward_pe_value_threshold_retry"

    if any(
        needle in normalized
        for needle in ("value_score > 60", "quality_score > 60", "value and quality", "value-quality")
    ) and any(
        needle in normalized
        for needle in ("composite_score", "bonus", "multiplicative", "post-rank", "rank()")
    ):
        return "value_quality_composite_bonus_retry"

    if any(
        needle in normalized
        for needle in ("earnings_yield_zero_weight", "duplicate zero", "zero-score entry")
    ) and any(needle in normalized for needle in ("weight", "smaller", "weaken", "dilution")):
        return "earnings_yield_zero_weight_retry"

    if any(
        needle in normalized
        for needle in ("large-cap", "large cap", "market cap > ¥50b", "market cap > 50b")
    ) and any(
        needle in normalized
        for needle in ("composite score", "composite ranking", "composite_score")
    ):
        return "large_cap_composite_bonus_retry"

    if any(
        needle in normalized
        for needle in ("roa value sub-factor", "roa value", "_linear_score(roa")
    ) and any(
        needle in normalized
        for needle in ("best", "threshold", "0.12", "0.15", "lowered", "raised")
    ):
        return "roa_value_threshold_retry"

    if any(
        needle in normalized
        for needle in (
            "direct roe value",
            "roe value sub-factor",
            "roe_dir",
            "_linear_score(roe_dir",
            "_linear_score(roe",
            "large-cap-gated direct roe",
            "large-cap gated direct roe",
        )
    ) and any(
        needle in normalized
        for needle in (
            "large-cap",
            "large cap",
            "market_cap >",
            "market cap >",
            "30_000_000_000",
            "50_000_000_000",
        )
    ):
        return "large_cap_direct_roe_value_retry"

    if any(
        needle in normalized
        for needle in ("large-cap", "large cap", "market cap > 50b", "market_cap > 50")
    ) and any(
        needle in normalized
        for needle in ("quality_score", "post-percentile", "roa", "bonus")
    ):
        return "large_cap_quality_bonus_retry"

    pe_safe_bug = "pe_safe" in normalized and any(
        phrase in normalized
        for phrase in ("undefined", "nameerror", "broken", "fatal bug", "bug", "fix", "repair")
    )
    if (
        pe_safe_bug
        or ("_compute_quality_raw" in normalized and any(
            phrase in normalized
            for phrase in ("undefined", "broken", "fatal bug", "bug", "fix")
        ))
        or ("quality raw" in normalized and any(
            phrase in normalized
            for phrase in ("undefined", "broken", "fatal bug", "bug", "fix", "repair")
        ))
    ):
        return "pe_safe_quality_raw_repair"
    return ""


def _blocked_failure_themes(records: list[dict], *, min_failures: int = 3) -> list[str]:
    """Return semantic themes with repeated failures and no material recent wins."""
    counts: dict[str, int] = {}
    successful_or_material: set[str] = set()
    for record in records:
        theme = _semantic_failure_theme(
            f"{record.get('diff_summary', '')}\n{record.get('description', '')}"
        )
        if not theme:
            continue
        materially_improved = any(
            float(record.get(f"delta_{h}", 0.0)) > _MIN_ACCEPTED_RHO_IMPROVEMENT
            for h in _HORIZONS
            if h != "6m"
        )
        materially_improved = materially_improved or float(record.get("delta", 0.0)) > _MIN_ACCEPTED_RHO_IMPROVEMENT
        if record.get("kept") or materially_improved:
            successful_or_material.add(theme)
            continue
        counts[theme] = counts.get(theme, 0) + 1

    return sorted(
        theme
        for theme, count in counts.items()
        if count >= min_failures and theme not in successful_or_material
    )


def _is_repeated_failed_theme(
    diff_summary: str,
    explanation: str,
    recent_records: list[dict],
    *,
    min_failures: int = 3,
) -> str:
    """Return the repeated failed theme name when a candidate should be skipped."""
    theme = _semantic_failure_theme(f"{diff_summary}\n{explanation}")
    if not theme:
        return ""

    failures = 0
    for record in recent_records:
        record_text = f"{record.get('diff_summary', '')}\n{record.get('description', '')}"
        if _semantic_failure_theme(record_text) != theme:
            continue
        improved = any(
            float(record.get(f"delta_{h}", 0.0)) > _MIN_ACCEPTED_RHO_IMPROVEMENT
            for h in ("1m", "3m")
        )
        improved = improved or float(record.get("delta", 0.0)) > _MIN_ACCEPTED_RHO_IMPROVEMENT
        if record.get("kept") or improved:
            return ""
        failures += 1

    return theme if failures >= min_failures else ""


def _rho_deltas(rhos: dict[str, float], baseline_rhos: dict[str, float]) -> dict[str, float]:
    """Return per-horizon rho deltas against the current scorer baseline."""
    return {h: float(rhos.get(h, 0.0)) - float(baseline_rhos.get(h, 0.0)) for h in _HORIZONS}


def _proposal_utility(rhos: dict[str, float], baseline_rhos: dict[str, float]) -> float:
    """Weighted multi-horizon utility for ranking proposal candidates."""
    deltas = _rho_deltas(rhos, baseline_rhos)
    return sum(_HORIZON_WEIGHTS[h] * deltas[h] for h in _HORIZONS)


def _is_behavior_neutral(rhos: dict[str, float], baseline_rhos: dict[str, float]) -> bool:
    """Return true when full evaluation is effectively unchanged."""
    deltas = _rho_deltas(rhos, baseline_rhos)
    return all(abs(delta) <= _BEHAVIOR_NEUTRAL_DELTA for delta in deltas.values())


def _quick_reject_diagnostics(
    quick_rhos: dict[str, float],
    quick_baseline_rhos: dict[str, float],
    target_rhos: Optional[dict[str, float]] = None,
) -> dict:
    """Return quick-eval rejection details using same-sample baseline deltas.

    Quick-eval rho levels are biased by the deterministic sample, so rejection
    should be based primarily on deltas against a baseline quick-eval run on
    the same sample. The historical target check is kept as a coarse fail-fast
    guard for obviously bad candidates.
    """
    sample_deltas = _rho_deltas(quick_rhos, quick_baseline_rhos)
    target_deltas = _rho_deltas(quick_rhos, target_rhos) if target_rhos else {}
    material_improved = [h for h in _HORIZONS if sample_deltas[h] > _QUICK_NEUTRAL_DELTA]
    material_degraded = [h for h in _HORIZONS if sample_deltas[h] < -_QUICK_NEUTRAL_DELTA]
    severe_degraded = [h for h in _HORIZONS if sample_deltas[h] < -_QUICK_TRADEOFF_DEGRADATION]
    utility = sum(_HORIZON_WEIGHTS[h] * sample_deltas[h] for h in _HORIZONS)
    noop_sample = all(abs(sample_deltas[h]) <= _QUICK_NOOP_DELTA for h in _HORIZONS)
    no_sample_upside = (
        not material_improved
        and all(sample_deltas[h] <= _QUICK_NOOP_DELTA for h in _HORIZONS)
        and any(sample_deltas[h] < -_QUICK_NOOP_DELTA for h in _HORIZONS)
    )

    reason = ""
    if target_rhos and all(
        quick_rhos[h] < target_rhos[h] - _QUICK_REJECT_MARGIN
        for h in _HORIZONS
    ):
        reason = "below target"
    elif noop_sample:
        reason = "behavior-neutral sample"
    elif sample_deltas["1m"] < -_QUICK_1M_DEGRADATION_MARGIN:
        reason = "1m degradation"
    elif (
        sample_deltas["1m"] < -_QUICK_TRADEOFF_DEGRADATION
        and ("6m" not in material_improved or utility < -_QUICK_UTILITY_REJECT_MARGIN)
    ):
        reason = "1m tradeoff"
    elif (
        material_improved == ["1m"]
        and any(h in severe_degraded for h in ("3m", "6m"))
        and utility < -_QUICK_UTILITY_REJECT_MARGIN
    ):
        reason = "1m-only tradeoff"
    elif not material_improved and severe_degraded:
        reason = "below sample baseline"
    elif no_sample_upside:
        reason = "no sample upside"
    elif material_degraded and utility < -_QUICK_UTILITY_REJECT_MARGIN:
        reason = "negative utility"

    return {
        "reject": bool(reason),
        "reason": reason,
        "quick_rhos": {h: float(quick_rhos[h]) for h in _HORIZONS},
        "quick_baseline_rhos": {h: float(quick_baseline_rhos[h]) for h in _HORIZONS},
        "sample_deltas": sample_deltas,
        "target_deltas": target_deltas,
        "sample_utility": utility,
        "material_improved_horizons": material_improved,
        "material_degraded_horizons": material_degraded,
        "severe_degraded_horizons": severe_degraded,
        "noop_sample": noop_sample,
        "no_sample_upside": no_sample_upside,
    }


def _quick_reject_log_fields(diagnostics: dict) -> dict:
    """Flatten quick-reject diagnostics into JSONL-friendly fields."""
    sample_deltas = diagnostics.get("sample_deltas", {})
    fields = {
        "quick_reject_reason": diagnostics.get("reason", ""),
        "quick_sample_utility": round(float(diagnostics.get("sample_utility", 0.0)), 6),
        "quick_material_improved_horizons": diagnostics.get("material_improved_horizons", []),
        "quick_material_degraded_horizons": diagnostics.get("material_degraded_horizons", []),
        "quick_severe_degraded_horizons": diagnostics.get("severe_degraded_horizons", []),
    }
    for h in _HORIZONS:
        if h in sample_deltas:
            quick_rhos = diagnostics.get("quick_rhos", {})
            quick_baseline_rhos = diagnostics.get("quick_baseline_rhos", {})
            if h in quick_rhos:
                fields[f"quick_rho_{h}"] = round(float(quick_rhos[h]), 6)
            if h in quick_baseline_rhos:
                fields[f"quick_baseline_rho_{h}"] = round(float(quick_baseline_rhos[h]), 6)
            fields[f"quick_delta_{h}"] = round(float(sample_deltas[h]), 6)
    return fields


def _acceptance_diagnostics(
    rhos: dict[str, float],
    current_rhos: dict[str, float],
    target_rhos: dict[str, float],
) -> dict:
    """Explain whether a proposal clears the current and historical targets."""
    current_deltas = _rho_deltas(rhos, current_rhos)
    target_deltas = _rho_deltas(rhos, target_rhos)
    current_improved = [h for h in _HORIZONS if current_deltas[h] > _MIN_RHO_IMPROVEMENT]
    current_material_improved = [
        h for h in _HORIZONS if current_deltas[h] > _MIN_ACCEPTED_RHO_IMPROVEMENT
    ]
    target_improved = [h for h in _HORIZONS if target_deltas[h] > _MIN_RHO_IMPROVEMENT]
    current_degraded = [h for h in _HORIZONS if current_deltas[h] < -_MIN_RHO_IMPROVEMENT]
    current_material_degraded = [h for h in _HORIZONS if current_deltas[h] < -_BEHAVIOR_NEUTRAL_DELTA]
    current_utility = sum(_HORIZON_WEIGHTS[h] * current_deltas[h] for h in _HORIZONS)
    worst_target_degradation = max(-delta for delta in target_deltas.values())
    target_utility = sum(_HORIZON_WEIGHTS[h] * target_deltas[h] for h in _HORIZONS)

    current_pareto_accept = bool(current_material_improved) and not current_material_degraded

    if not target_improved:
        reject_reason = "below target"
    elif worst_target_degradation > _MAX_DEGRADATION:
        reject_reason = "target degradation"
    elif target_utility <= _MIN_RHO_IMPROVEMENT:
        reject_reason = "target utility tradeoff"
    else:
        reject_reason = ""

    accepted_horizons = current_material_improved if current_pareto_accept else []

    return {
        "accepted_horizons": accepted_horizons,
        "current_improved_horizons": current_improved,
        "current_material_improved_horizons": current_material_improved,
        "current_degraded_horizons": current_degraded,
        "current_material_degraded_horizons": current_material_degraded,
        "target_improved_horizons": target_improved,
        "current_deltas": current_deltas,
        "target_deltas": target_deltas,
        "current_utility": current_utility,
        "target_utility": target_utility,
        "worst_target_degradation": worst_target_degradation,
        "historical_reject_reason": reject_reason,
        "reject_reason": "" if current_pareto_accept else reject_reason,
        "accepted_by": "current_material_pareto" if current_pareto_accept else "",
        "near_miss": bool(current_improved or target_improved) and not current_pareto_accept,
    }


def _acceptance_log_fields(diagnostics: dict) -> dict:
    """Flatten acceptance diagnostics into JSONL-friendly record fields."""
    fields = {
        "acceptance_reject_reason": diagnostics.get("reject_reason", ""),
        "historical_reject_reason": diagnostics.get("historical_reject_reason", ""),
        "accepted_by": diagnostics.get("accepted_by", ""),
        "near_miss": bool(diagnostics.get("near_miss")),
        "current_utility": round(float(diagnostics.get("current_utility", 0.0)), 6),
        "target_utility": round(float(diagnostics.get("target_utility", 0.0)), 6),
        "worst_target_degradation": round(float(diagnostics.get("worst_target_degradation", 0.0)), 6),
        "current_improved_horizons": diagnostics.get("current_improved_horizons", []),
        "current_material_improved_horizons": diagnostics.get("current_material_improved_horizons", []),
        "current_degraded_horizons": diagnostics.get("current_degraded_horizons", []),
        "current_material_degraded_horizons": diagnostics.get("current_material_degraded_horizons", []),
        "target_improved_horizons": diagnostics.get("target_improved_horizons", []),
    }
    current_deltas = diagnostics.get("current_deltas", {})
    target_deltas = diagnostics.get("target_deltas", {})
    for h in _HORIZONS:
        fields[f"current_delta_{h}"] = round(float(current_deltas.get(h, 0.0)), 6)
        fields[f"target_delta_{h}"] = round(float(target_deltas.get(h, 0.0)), 6)
    return fields


def _accepted_horizons(rhos: dict[str, float], baseline_rhos: dict[str, float]) -> list[str]:
    """Return improved horizons if a candidate is a material incumbent improvement."""
    deltas = _rho_deltas(rhos, baseline_rhos)
    material_improved = [
        h for h in _HORIZONS if deltas[h] > _MIN_ACCEPTED_RHO_IMPROVEMENT
    ]
    if not material_improved:
        return []

    if any(delta < -_BEHAVIOR_NEUTRAL_DELTA for delta in deltas.values()):
        return []

    return material_improved


def _dual_benchmark_enabled() -> bool:
    """Return whether to use current-vs-legacy dual benchmark evaluation."""
    setting = os.environ.get("IMPROVE_SCORER_DUAL_BENCHMARK", "auto").strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        ensure_legacy_ground_truth_ready()
        ensure_current_ground_truth_ready()
        return True
    if LEGACY_GROUND_TRUTH_FILE.exists() and CURRENT_GROUND_TRUTH_FILE.exists():
        ensure_legacy_ground_truth_ready()
        ensure_current_ground_truth_ready()
        return True
    return False


def _legacy_guard_rejections(
    legacy_rhos: dict[str, float],
    legacy_baseline_rhos: dict[str, float],
) -> list[str]:
    """Return horizons where a candidate damages the legacy benchmark too much."""
    return [
        h for h in _HORIZONS
        if legacy_baseline_rhos[h] - legacy_rhos.get(h, 0.0) > _LEGACY_GUARD_MAX_DEGRADATION
    ]


def _validate_static(code: str) -> list[str]:
    """Run ruff on proposed code. Returns non-empty list of errors on failure."""
    if not shutil.which("ruff"):
        logger.warning("ruff not found; skipping static analysis")
        return []
    try:
        result = subprocess.run(
            ["ruff", "check", "--select", "F821", "--no-cache", "-"],
            input=code, capture_output=True, text=True, timeout=10,
            cwd=str(_project_root()),
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("ruff static analysis failed: %s", e)
        return []
    if result.returncode == 0:
        return []
    # ruff writes errors to stdout on failure
    errors = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return errors


def _default_weight_issues(code: str) -> list[str]:
    """Return structural issues with the scorer's default factor weights."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error before weight validation: {exc}"]

    weights_node: ast.Dict | None = None
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            target = node.targets[0] if node.targets else None
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if isinstance(target, ast.Name) and target.id == "_DEFAULT_WEIGHTS":
            if isinstance(value, ast.Dict):
                weights_node = value
            break

    if weights_node is None:
        return ["_DEFAULT_WEIGHTS must be a module-level dict literal"]

    weights: dict[str, float] = {}
    for key_node, value_node in zip(weights_node.keys, weights_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            return ["_DEFAULT_WEIGHTS keys must be string literals"]
        if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
            return ["_DEFAULT_WEIGHTS values must be numeric literals"]
        weights[key_node.value] = float(value_node.value)

    required = set(_REQUIRED_DEFAULT_WEIGHT_KEYS)
    missing = sorted(required - set(weights))
    extra = sorted(set(weights) - required)
    non_positive = sorted(k for k in required if k in weights and weights[k] <= 0)
    total = sum(weights.get(k, 0.0) for k in _REQUIRED_DEFAULT_WEIGHT_KEYS)

    issues: list[str] = []
    if missing:
        issues.append(f"_DEFAULT_WEIGHTS missing required factors: {', '.join(missing)}")
    if extra:
        issues.append(f"_DEFAULT_WEIGHTS has unsupported factors: {', '.join(extra)}")
    if non_positive:
        issues.append(f"_DEFAULT_WEIGHTS factors must be positive: {', '.join(non_positive)}")
    if abs(total - 1.0) > 1e-9:
        issues.append(f"_DEFAULT_WEIGHTS must sum to 1.0 across all factors; got {total:.6f}")
    return issues


def _validate_scorer(
    *,
    code: Optional[str] = None,
    scorer_module: str = SCORER_MODULE,
    scorer_path: Optional[Path] = None,
) -> bool:
    """Check if the current scorer is valid Python and doesn't crash.

    Runs compile() + 6 smoke-test fixtures covering edge cases, plus a rank()
    call to verify cross-sectional ranking produces valid output.
    """
    try:
        source_path = scorer_path or SCORER_PATH
        source_code = code if code is not None else source_path.read_text(encoding="utf-8")
        compile(source_code, str(source_path), "exec")
        weight_issues = _default_weight_issues(source_code)
        if weight_issues:
            logger.warning("Scorer default weight validation failed: %s", "; ".join(weight_issues))
            return False
    except SyntaxError as e:
        logger.warning("Scorer has syntax error: %s", e)
        return False

    try:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.evaluator import _load_scorer_module

        loaded_mod = _load_scorer_module(scorer_module, scorer_path=scorer_path)
        MultiFactorScorer = loaded_mod.MultiFactorScorer

        def _mk(pe=15.0, pb=2.0, roe=0.15, gm=0.30, dte=0.5, fcf=1e9,
                mcap=1e10, current_ratio=1.5, market=Market.A_SHARE, **kw):
            return ScreeningResult(
                company=Company(ticker="TEST", name="Test", market=market),
                financials=Financials(
                    ticker="TEST", period="test",
                    roe=roe, gross_margin=gm, debt_to_equity=dte,
                    free_cash_flow=fcf, current_ratio=current_ratio,
                    operating_cash_flow=1e9, revenue=5e9, net_income=1e9,
                    roa=0.08, total_assets=1e10, total_equity=5e9, total_liabilities=5e9,
                    net_margin=0.10,
                ),
                valuation=ValuationMetrics(
                    ticker="TEST", date="2024-01-01",
                    pe_ratio=pe, pb_ratio=pb, market_cap_rmb=mcap,
                    ps_ratio=1.5, peg_ratio=1.0, dividend_yield=0.02,
                    ev_to_ebitda=10.0, price=50.0, pe_forward=12.0,
                    **kw,
                ),
            )

        fixtures: dict[str, ScreeningResult] = {
            "fully_populated": _mk(),
            "zero_pe": _mk(pe=0.0),
            "negative_roe": _mk(roe=-0.10),
            "no_fcf": _mk(fcf=None),
            "hk_share": _mk(pe=8.0, pb=1.5, roe=0.12, market=Market.HK_SHARE),
            "sparse": ScreeningResult(
                company=Company(ticker="SPARSE", name="Sparse", market=Market.A_SHARE),
                financials=Financials(ticker="SPARSE", period="test"),
                valuation=ValuationMetrics(ticker="SPARSE", date="2024-01-01"),
            ),
        }

        scorer = MultiFactorScorer()
        for name, sr in fixtures.items():
            try:
                scorer.score(sr)
                if not (0.0 <= sr.composite_score <= 100.0):
                    logger.warning("Validation failed on fixture '%s': composite_score=%.2f out of range",
                                   name, sr.composite_score)
                    return False
            except Exception as e:
                logger.warning("Validation failed on fixture '%s': %s: %s", name, type(e).__name__, e)
                return False

        # Cross-sectional rank validation
        rank_fixtures = [
            _mk(pe=25, pb=4, roe=0.05, gm=0.15, dte=1.0),
            _mk(pe=5, pb=0.8, roe=0.25, gm=0.40, dte=0.2),
            _mk(pe=12, pb=1.8, roe=0.15, gm=0.30, dte=0.5),
        ]
        ranked = scorer.rank(rank_fixtures)
        ranks = [r.rank for r in ranked]
        if set(ranks) != {1, 2, 3}:
            logger.warning("Rank validation failed: got ranks %s", ranks)
            return False
        for r in ranked:
            if not (0.0 <= r.composite_score <= 100.0):
                logger.warning("Rank validation: composite_score=%.2f out of range", r.composite_score)
                return False

        return True
    except Exception as e:
        logger.warning("Scorer validation failed: %s", e)
        return False


def _run_scorer_test_suite() -> tuple:
    """Run the scorer test suite. Returns (passed, output_summary)."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        logger.warning("pytest not installed; skipping test suite")
        return True, "pytest_not_available"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_scorer.py",
             "-x", "-q", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_project_root()),
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("pytest run failed: %s", e)
        return True, f"pytest_error: {e}"

    passed = result.returncode == 0
    # Extract failing test name from pytest short output, plus the count line
    stdout = result.stdout or ""
    failed_match = re.search(r"FAILED\s+(tests/\S+)", stdout)
    count_line = stdout.strip().split("\n")[-1] if stdout.strip() else ""
    if failed_match:
        summary = f"{failed_match.group(1)} — {count_line}"
    else:
        summary = count_line or result.stderr[:200]
    if not passed:
        logger.warning("Test suite failed: %s", result.stderr[:500] if result.stderr else summary)
    return passed, summary


def _print_iteration_summary(
    iteration: int,
    rhos: dict[str, float],
    best_rhos: dict[str, Optional[float]],
    status: str,
    reason: str = "",
) -> None:
    """Print the compact per-iteration summary shown in the trainer console."""
    suffix = f" ({reason})" if reason else ""
    print(f"\n{'='*60}")
    print(f"  Iteration #{iteration}  |  {status}{suffix}")
    for h in _HORIZONS:
        rho = float(rhos.get(h, 0.0))
        best = best_rhos.get(h)
        best = rho if best is None else float(best)
        print(f"  {_HORIZON_LABELS[h]:>10}  ρ={rho:.4f}  |  Best: {best:.4f}")
    print(f"{'='*60}\n")


def _print_banner(
    iteration: int,
    rhos: dict[str, float],
    baseline_rhos: dict[str, float],
    best_rhos: dict[str, float],
    kept: bool,
    description: str,
    improved_horizons: list[str],
) -> None:
    """Print a progress banner to stdout showing all three horizon rhos."""
    reason = ""
    if kept and improved_horizons:
        reason = f"improved {','.join(improved_horizons)}"
    elif not kept:
        reason = description or "no improvement"
    _print_iteration_summary(
        iteration=iteration,
        rhos=rhos,
        best_rhos=best_rhos,
        status="✅ KEPT" if kept else "❌ REVERTED",
        reason=reason,
    )


def _build_experiment_history(
    exp_log: ExperimentLog,
    unified_records: list[dict],
    include_legacy: bool = True,
    max_entries: int = 15,
) -> str:
    """Build a formatted experiment history string for the LLM prompt.

    Merges unified log records with legacy horizon-specific records,
    sorted chronologically, and formats each with status and horizon label.
    """
    all_records = list(unified_records)
    if include_legacy:
        all_records.extend(ExperimentLog.read_legacy_logs())
    all_records.sort(key=lambda r: r.get("timestamp", ""))

    # Take the most recent entries
    recent = all_records[-max_entries:] if len(all_records) > max_entries else all_records

    lines = []
    for r in recent:
        status = "✅" if r.get("kept") else "❌"
        # Show which horizon(s) improved, if available
        improved = r.get("improved_horizons", [])
        improved_tag = f" [↑{','.join(improved)}]" if improved else ""
        horizon_tag = r.get("source_horizon") or r.get("horizon", "")
        tag = f"[{horizon_tag}]" if horizon_tag and horizon_tag != "all" else ""
        rho = r.get("spearman_rho", 0)
        delta = r.get("delta", 0)
        desc = r.get("description", "")[:100]
        lines.append(
            f"{status} {tag} #{r.get('iteration', '?')}: ρ={rho:.4f}"
            f" (Δ={delta:+.4f}){improved_tag} — {desc}"
        )

    return "\n".join(lines) if lines else ""


def _create_parallel_clients():
    """Create N independent LLM client instances for parallel proposal generation."""
    clients = []
    for _ in range(_N_PARALLEL):
        try:
            client = _create_single_client()
            if client is not None:
                clients.append(client)
        except Exception as e:
            logger.warning("Failed to create parallel client: %s", e)
    if not clients:
        raise RuntimeError("Failed to create any LLM clients")
    logger.info("Created %d parallel LLM client(s)", len(clients))
    return clients


def _create_single_client():
    """Create one LLM client (same logic as _get_llm_client but doesn't cache fallback)."""
    from valueinvestor.analysis.llm_client import LLMClient
    from valueinvestor.config import load_config, LLMConfig

    local_llm_enabled = os.environ.get("LOCAL_LLM_ENABLED", "").lower() == "true"
    if local_llm_enabled:
        local_base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234")
        local_model = os.environ.get("LOCAL_LLM_MODEL", "default")
        try:
            local_config = LLMConfig(
                provider="local_llm", model=local_model, api_key="not-needed",
                base_url=local_base_url, max_retries=2, temperature=0.7,
            )
            return LLMClient(config=local_config)
        except Exception:
            pass

    cfg = load_config()
    cfg.llm.temperature = 0.7
    if cfg.llm.provider == "local_llm":
        if not cfg.llm.api_key:
            cfg.llm.api_key = "not-needed"
    if not cfg.llm.api_key and cfg.llm.provider != "local_llm":
        raise RuntimeError("No LLM API key available")
    return LLMClient(config=cfg.llm)


def run_improvement_loop(max_iterations: int = 0) -> None:
    """Run the autonomous scorer improvement loop.

    Evaluates each proposed change against all three forward-return horizons
    (1m, 3m, 6m) and keeps the change if ANY target improves.

    Generates up to N parallel proposals per iteration (controlled by
    ``IMPROVE_SCORER_PARALLEL`` env var, default 1).  Uses quick subset
    evaluation to reject clearly-bad proposals before running the full
    40K-row evaluation.

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

    # Load legacy experiment history for LLM context
    legacy_history = _build_experiment_history(
        exp_log, exp_log.read_all(), include_legacy=True, max_entries=_EXPERIMENT_HISTORY_COUNT,
    )
    if legacy_history:
        logger.info("Loaded historical experiments from legacy logs for LLM context")

    dual_benchmark = _dual_benchmark_enabled()
    if dual_benchmark:
        primary_ground_truth_path = CURRENT_GROUND_TRUTH_FILE
        legacy_ground_truth_path: Optional[Path] = LEGACY_GROUND_TRUTH_FILE
        allow_primary_legacy_schema = False
        logger.info(
            "Dual benchmark mode enabled — primary=%s, legacy guard=%s",
            primary_ground_truth_path,
            legacy_ground_truth_path,
        )
    else:
        primary_ground_truth_path = None
        legacy_ground_truth_path = None
        allow_primary_legacy_schema = True

    ground_truth_fingerprint_id = (
        ground_truth_fingerprint(primary_ground_truth_path)
        if primary_ground_truth_path is not None
        else ground_truth_fingerprint()
    )
    ground_truth_id = f"{ground_truth_fingerprint_id}|eval={_EVALUATION_SCHEMA_ID}"
    os.environ["IMPROVE_SCORER_ACTIVE_GROUND_TRUTH_ID"] = ground_truth_id
    comparable_history = exp_log.read_current_lineage(ground_truth_id=ground_truth_id)
    if not comparable_history and exp_log.total_experiments() > 0:
        logger.info(
            "No prior experiments match current ground truth (%s); historical best metrics "
            "will be ignored for this run.",
            ground_truth_id,
        )

    primary_eval_context = ScorerEvaluationContext(
        ground_truth_path=primary_ground_truth_path,
        allow_legacy_schema=allow_primary_legacy_schema,
        quick_sample_size=_QUICK_EVAL_SAMPLE_SIZE,
    )
    legacy_eval_context = (
        ScorerEvaluationContext(
            ground_truth_path=legacy_ground_truth_path,
            allow_legacy_schema=True,
            quick_sample_size=_QUICK_EVAL_SAMPLE_SIZE,
        )
        if legacy_ground_truth_path is not None
        else None
    )

    worktree_scorer_code = _read_scorer()

    scorer_workspace = tempfile.TemporaryDirectory(prefix="valueinvestor-scorer-")
    scorer_workspace_dir = Path(scorer_workspace.name)
    restored_best_meta, active_scorer_code, using_snapshot_baseline = _load_resume_baseline(
        ground_truth_id
    )
    if restored_best_meta:
        logger.info(
            "Resume baseline loaded from best scorer snapshot: iteration #%s",
            restored_best_meta.get("iteration", "?"),
        )

    if using_snapshot_baseline and active_scorer_code != worktree_scorer_code:
        active_scorer_ref = _materialize_temp_scorer(
            scorer_workspace_dir,
            active_scorer_code,
            label="baseline",
            iteration=int(restored_best_meta.get("iteration", 0)) if restored_best_meta else None,
        )
    else:
        active_scorer_code = worktree_scorer_code
        active_scorer_ref = _current_scorer_ref()

    if not _validate_scorer(
        code=active_scorer_code,
        scorer_module=active_scorer_ref.module_name,
        scorer_path=active_scorer_ref.path,
    ):
        raise RuntimeError(
            "Trainer cannot start: scorer.py failed validation. Ensure _DEFAULT_WEIGHTS "
            "contains positive value, quality, growth, momentum, synergy, and "
            "value_growth weights summing to 1.0."
        )

    # Get baseline evaluation across all three horizons
    logger.info("Computing baseline evaluation (1m, 3m, 6m) …")
    baseline_metrics = primary_eval_context.evaluate_all_targets(
        scorer_module=active_scorer_ref.module_name,
        scorer_path=active_scorer_ref.path,
    )
    baseline_rhos: dict[str, float] = {
        h: float(baseline_metrics[h]["spearman_rho"]) for h in _HORIZONS
    }
    legacy_baseline_rhos: Optional[dict[str, float]] = None
    if legacy_eval_context is not None:
        legacy_metrics = legacy_eval_context.evaluate_all_targets(
            scorer_module=active_scorer_ref.module_name,
            scorer_path=active_scorer_ref.path,
        )
        legacy_baseline_rhos = {
            h: float(legacy_metrics[h]["spearman_rho"]) for h in _HORIZONS
        }

    best_rhos: dict[str, Optional[float]] = exp_log.best_rho_per_horizon(
        current_lineage=True,
        ground_truth_id=ground_truth_id,
    )
    if restored_best_meta:
        for h, rho in _snapshot_rhos_from_meta(restored_best_meta).items():
            if best_rhos.get(h) is None or rho > float(best_rhos[h]):
                best_rhos[h] = rho
    for h in _HORIZONS:
        lineage_val = best_rhos.get(h)
        if lineage_val is None or baseline_rhos[h] > lineage_val:
            best_rhos[h] = baseline_rhos[h]

    # Proposals must beat the higher of current baseline or all-time best
    target_rhos: dict[str, float] = {
        h: max(baseline_rhos[h], float(best_rhos.get(h) or 0.0)) for h in _HORIZONS
    }

    print("\n🎯 Current Benchmark Spearman ρ:" if dual_benchmark else "\n🎯 Baseline Spearman ρ:")
    for h in _HORIZONS:
        print(f"     {_HORIZON_LABELS[h]:>10}  {baseline_rhos[h]:.4f}")
    if any(abs(float(best_rhos[h] or 0.0) - baseline_rhos[h]) > 0.00005 for h in _HORIZONS):
        print("🏆 Historical Best Spearman ρ:")
        for h in _HORIZONS:
            print(f"     {_HORIZON_LABELS[h]:>10}  {float(best_rhos[h] or 0.0):.4f}")
    if legacy_baseline_rhos is not None:
        print("🛡️  Legacy Guard Spearman ρ:")
        for h in _HORIZONS:
            print(f"     {_HORIZON_LABELS[h]:>10}  {legacy_baseline_rhos[h]:.4f}")
    parallel_info = f" ({_N_PARALLEL} parallel proposals)" if _N_PARALLEL > 1 else ""
    print(f"🔄 Starting improvement loop{parallel_info} …\n")

    # Create parallel LLM clients
    parallel_clients = _create_parallel_clients()
    primary_llm = parallel_clients[0]

    iteration = exp_log.total_experiments()
    rounds_completed = 0
    quick_baseline_cache_key: Optional[tuple] = None
    quick_baseline_cache_rhos: Optional[dict[str, float]] = None
    while not _stop:
        if max_iterations > 0 and rounds_completed >= max_iterations:
            print(f"\n🏁 Reached max iterations ({max_iterations}). Stopping.")
            break

        # 1. Read the active baseline scorer (may come from a persisted snapshot)
        original_code = active_scorer_code

        # 2. Get experiment history. Keep invalid/no-patch attempts out of the
        # prompt so they do not drown out real evaluated changes.
        lineage_records = exp_log.read_current_lineage(ground_truth_id=ground_truth_id)
        recent = _records_for_prompt(lineage_records, _EXPERIMENT_HISTORY_COUNT)
        experiment_history = _build_experiment_history(
            exp_log, recent, include_legacy=False, max_entries=_EXPERIMENT_HISTORY_COUNT,
        )

        # Compute summary stats from the records
        summary_stats = _compute_summary_stats(lineage_records[-50:]) if lineage_records else ""
        failed_theme_history = lineage_records[-150:]
        failure_guidance = _compute_failure_guidance(
            lineage_records[-75:],
            blocked_theme_records=failed_theme_history,
        )
        near_miss_guidance = _near_miss_guidance(lineage_records[-75:])
        history_for_prompt = experiment_history or legacy_history

        # 3. Build prompt(s), one exploration lane per parallel proposal.
        is_local = primary_llm.provider == "local_llm"

        def _prompt_for_lane(lane_index: int = 0) -> str:
            lane = _EXPLORATION_LANES[lane_index % len(_EXPLORATION_LANES)]
            return _build_prompt(
                scorer_code=original_code,
                program_md=program_md,
                experiment_history=history_for_prompt,
                baseline_rhos=baseline_rhos,
                best_rhos=best_rhos,
                is_local_llm=is_local,
                summary_stats=summary_stats,
                failure_guidance="\n\n".join(
                    guidance for guidance in (near_miss_guidance, failure_guidance) if guidance
                ),
                lane_instruction=lane,
            )

        user_prompt = _prompt_for_lane(rounds_completed)

        # 4. Generate proposals — each is its own iteration
        proposals: list[dict] = []
        if len(parallel_clients) == 1:
            iteration += 1
            llm = parallel_clients[0]
            try:
                proposal_result = _complete_proposal(
                    llm,
                    user_prompt,
                    original_code,
                    iteration=iteration,
                    request_timeout=_LLM_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception as e:
                _log_skip(exp_log, iteration, baseline_rhos, f"LLM call failed: {e}")
                _print_skip_banner(iteration, baseline_rhos, best_rhos, f"LLM error: {e}")
                rounds_completed += 1
                time.sleep(5)
                continue
            new_code = proposal_result["code"]
            explanation = proposal_result["explanation"]
            diff_summary = proposal_result["diff_summary"]
            if not _proposal_is_valid(new_code, diff_summary):
                logger.info("  #%d: ∅ no code changes in response", iteration)
                _log_skip(
                    exp_log,
                    iteration,
                    baseline_rhos,
                    explanation or "No code changes proposed",
                    extra={"proposal_debug_paths": proposal_result.get("debug_paths", [])},
                )
                _print_skip_banner(iteration, baseline_rhos, best_rhos, "no code changes")
                rounds_completed += 1
                time.sleep(1)
                continue
            proposals.append({
                "code": new_code, "explanation": explanation, "diff_summary": diff_summary,
                "iteration": iteration, "repaired": proposal_result.get("repaired", False),
            })
        else:
            # Parallel path — each LLM call gets its own iteration number and
            # child process. Stuck provider requests are terminated at timeout.
            n_parallel = len(parallel_clients)
            base_iter = iteration
            prompt_jobs = [
                (base_iter + i + 1, _prompt_for_lane(i))
                for i in range(n_parallel)
            ]
            generated, proposal_events = _run_parallel_proposal_workers(
                prompt_jobs,
                original_code,
                timeout_seconds=_PARALLEL_PROPOSAL_TIMEOUT_SECONDS,
                request_timeout_seconds=_LLM_REQUEST_TIMEOUT_SECONDS,
                early_stop_after_seconds=_PARALLEL_EARLY_STOP_AFTER_SECONDS,
                completion_grace_seconds=_PARALLEL_COMPLETION_GRACE_SECONDS,
                min_completed_proposals=_PARALLEL_MIN_COMPLETED_PROPOSALS,
            )
            iteration = max(iteration, base_iter + n_parallel)
            proposals.extend(generated)

            for event in proposal_events:
                iter_num = int(event["iteration"])
                kind = event.get("kind")
                message = str(event.get("message") or "LLM call failed")
                if kind == "timeout":
                    timeout_seconds = float(
                        event.get("proposal_timeout_seconds")
                        or _PARALLEL_PROPOSAL_TIMEOUT_SECONDS
                    )
                    logger.warning(
                        "Parallel LLM call stopped after %.1fs for iteration #%d",
                        timeout_seconds,
                        iter_num,
                    )
                    _log_skip(
                        exp_log,
                        iter_num,
                        baseline_rhos,
                        message,
                        extra={"proposal_timeout_seconds": timeout_seconds},
                    )
                    _print_skip_banner(
                        iter_num,
                        baseline_rhos,
                        best_rhos,
                        f"LLM timeout after {timeout_seconds:.1f}s",
                    )
                else:
                    logger.warning("Parallel LLM call failed for iteration #%d: %s", iter_num, message)
                    _log_skip(exp_log, iter_num, baseline_rhos, f"LLM call failed: {message}")
                    _print_skip_banner(iter_num, baseline_rhos, best_rhos, f"LLM error: {message}")

            # Log no-code proposals that came back from parallel calls
            for p in list(proposals):
                if not _proposal_is_valid(p.get("code"), p.get("diff_summary", "")):
                    logger.info("  #%d: ∅ no code changes in response", p["iteration"])
                    _log_skip(exp_log, p["iteration"], baseline_rhos,
                              p.get("explanation", "") or "No code changes proposed",
                              extra={"proposal_debug_paths": p.get("debug_paths", [])})
                    _print_skip_banner(p["iteration"], baseline_rhos, best_rhos, "no code changes")
                    proposals.remove(p)

        if not proposals:
            logger.info("  ❌ No valid proposals generated this round")
            rounds_completed += 1
            time.sleep(1)
            continue

        rounds_completed += 1
        first_iter = min(p["iteration"] for p in proposals)
        last_iter = max(p["iteration"] for p in proposals)
        iter_label = f"#{first_iter}" if len(proposals) == 1 else f"#{first_iter}-{last_iter}"
        logger.info("=== Round %d — %d proposal(s) [%s] ===", rounds_completed, len(proposals), iter_label)

        proposal_lines: list[str] = []
        prefiltered_proposals: list[dict] = []
        seen_round_diffs: set[str] = set()
        for proposal in proposals:
            diff_summary = proposal.get("diff_summary", "")
            explanation = proposal.get("explanation", "")
            iter_num = proposal["iteration"]

            if _is_duplicate_diff(diff_summary, recent):
                logger.debug("  #%d: duplicate diff — skipping", iter_num)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Duplicate diff: {explanation}", diff_summary=diff_summary)
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "duplicate diff")
                continue

            if diff_summary in seen_round_diffs:
                logger.info("  #%d: ⏩ skipped duplicate diff in current round", iter_num)
                _log_skip(
                    exp_log,
                    iter_num,
                    baseline_rhos,
                    f"Duplicate diff in current round: {explanation}",
                    diff_summary=diff_summary,
                )
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "duplicate diff in round")
                continue
            seen_round_diffs.add(diff_summary)

            failed_theme = _is_repeated_failed_theme(diff_summary, explanation, failed_theme_history)
            if failed_theme:
                logger.info("  #%d: ⏩ skipped repeated failed theme (%s)", iter_num, failed_theme)
                _log_skip(
                    exp_log,
                    iter_num,
                    baseline_rhos,
                    f"Repeated failed theme ({failed_theme}): {explanation}",
                    diff_summary=diff_summary,
                )
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "repeated failed theme")
                continue

            prefiltered_proposals.append(proposal)

        if len(prefiltered_proposals) != len(proposals):
            logger.info(
                "Pre-evaluation filters kept %d/%d proposal(s)",
                len(prefiltered_proposals),
                len(proposals),
            )
        proposals = prefiltered_proposals

        if not proposals:
            print(f"\n--- Round {rounds_completed} proposals [{iter_label}] ---")
            removed = _cleanup_old_backups()
            if removed:
                logger.info("Cleaned up %d old backup(s); %d retained", removed, _MAX_BACKUPS)
            logger.info("❌ No proposal reached evaluation — skipped before quick baseline")
            continue

        quick_baseline_rhos: Optional[dict[str, float]] = None
        quick_cache_key = (
            tuple(round(float(baseline_rhos[h]), 12) for h in _HORIZONS),
            hashlib.sha256(original_code.encode("utf-8")).hexdigest(),
            _QUICK_EVAL_SAMPLE_SIZE,
            str(primary_ground_truth_path),
        )
        if quick_baseline_cache_key == quick_cache_key and quick_baseline_cache_rhos is not None:
            quick_baseline_rhos = dict(quick_baseline_cache_rhos)
            logger.info(
                "Quick baseline cache hit — 1m=%.4f, 3m=%.4f, 6m=%.4f",
                quick_baseline_rhos["1m"],
                quick_baseline_rhos["3m"],
                quick_baseline_rhos["6m"],
            )
        else:
            try:
                quick_baseline = primary_eval_context.quick_evaluate(
                    scorer_module=active_scorer_ref.module_name,
                    scorer_path=active_scorer_ref.path,
                    sample_size=_QUICK_EVAL_SAMPLE_SIZE,
                )
                quick_baseline_rhos = {h: float(quick_baseline[h]) for h in _HORIZONS}
                quick_baseline_cache_key = quick_cache_key
                quick_baseline_cache_rhos = dict(quick_baseline_rhos)
                logger.info(
                    "Quick baseline — 1m=%.4f, 3m=%.4f, 6m=%.4f",
                    quick_baseline_rhos["1m"],
                    quick_baseline_rhos["3m"],
                    quick_baseline_rhos["6m"],
                )
            except Exception as e:
                logger.warning("Quick baseline evaluation failed; disabling same-sample quick gates: %s", e)
                quick_baseline_cache_key = None
                quick_baseline_cache_rhos = None

        # 5. Evaluate each proposal, log individually, collect improving ones
        improving_proposals: list[dict] = []
        baseline_rhos_original = dict(baseline_rhos)

        for pi, proposal in enumerate(proposals):
            new_code = proposal.get("code")
            explanation = proposal.get("explanation", "")
            diff_summary = proposal.get("diff_summary", "")
            iter_num = proposal["iteration"]

            # Static analysis (ruff) — catch NameError before runtime
            static_errors = _validate_static(new_code)
            if static_errors:
                logger.info("  #%d: ❌ static analysis failed — %s", iter_num, static_errors[0])
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Static analysis failed: {static_errors[0]}", diff_summary=diff_summary)
                proposal_lines.append(f"  #{iter_num}: ❌ static analysis — {static_errors[0]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "static analysis failed")
                continue

            candidate_ref = _materialize_temp_scorer(
                scorer_workspace_dir,
                new_code,
                label="proposal",
                iteration=iter_num,
            )

            if not _validate_scorer(
                code=new_code,
                scorer_module=candidate_ref.module_name,
                scorer_path=candidate_ref.path,
            ):
                logger.info("  #%d: ❌ validation failed — %s", iter_num, explanation[:80])
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Validation failed: {explanation}", diff_summary=diff_summary)
                proposal_lines.append(f"  #{iter_num}: ❌ validation failed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "validation failed")
                _release_scorer_module(candidate_ref)
                continue

            # Quick evaluation on a sample before spending a full benchmark run.
            try:
                quick_rhos = primary_eval_context.quick_evaluate(
                    scorer_module=candidate_ref.module_name,
                    scorer_path=candidate_ref.path,
                    sample_size=_QUICK_EVAL_SAMPLE_SIZE,
                )
            except Exception as e:
                logger.info("  #%d: ❌ quick-eval crashed — %s", iter_num, e)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Quick-eval crashed: {e}", diff_summary=diff_summary)
                proposal_lines.append(f"  #{iter_num}: ❌ eval crashed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "quick eval crashed")
                _release_scorer_module(candidate_ref)
                continue

            quick_reference = quick_baseline_rhos or {h: float(quick_rhos[h]) for h in _HORIZONS}
            quick_rejection = _quick_reject_diagnostics(quick_rhos, quick_reference, target_rhos)
            if quick_rejection["reject"]:
                quick_deltas = quick_rejection["sample_deltas"]
                reason = quick_rejection["reason"]
                logger.info(
                    "  #%d: ⏩ quick-rejected %s (Δ 1m=%+.4f, 3m=%+.4f, 6m=%+.4f; utility=%+.4f)",
                    iter_num,
                    reason,
                    quick_deltas["1m"],
                    quick_deltas["3m"],
                    quick_deltas["6m"],
                    quick_rejection["sample_utility"],
                )
                _log_skip(
                    exp_log,
                    iter_num,
                    baseline_rhos,
                    f"Quick-rejected ({reason}): {explanation}",
                    diff_summary=diff_summary,
                    extra=_quick_reject_log_fields(quick_rejection),
                )
                proposal_lines.append(
                    f"  #{iter_num}: ⏩ quick {reason} — "
                    f"Δ1m={quick_deltas['1m']:+.4f} "
                    f"Δ3m={quick_deltas['3m']:+.4f} "
                    f"Δ6m={quick_deltas['6m']:+.4f} — {explanation[:60]}"
                )
                _print_iteration_summary(
                    iteration=iter_num,
                    rhos=quick_rhos,
                    best_rhos=best_rhos,
                    status="❌ REVERTED",
                    reason=f"quick {reason}",
                )
                _release_scorer_module(candidate_ref)
                continue

            # Full evaluation
            try:
                new_metrics = primary_eval_context.evaluate_all_targets(
                    scorer_module=candidate_ref.module_name,
                    scorer_path=candidate_ref.path,
                )
                new_rhos: dict[str, float] = {
                    h: float(new_metrics[h]["spearman_rho"]) for h in _HORIZONS
                }

                # Reject if scorer crashes on too many stocks
                error_rate = float(new_metrics.get("error_rate", 0.0))
                if error_rate > 0.0:
                    logger.info("  #%d: ❌ high error rate — %.2f%% (%s)",
                                iter_num, error_rate * 100,
                                new_metrics.get("error_count", 0))
                    _log_skip(exp_log, iter_num, baseline_rhos,
                              f"Validation failed: {error_rate:.1%} error rate "
                              f"({new_metrics.get('error_count', 0)} stocks)",
                              diff_summary=diff_summary)
                    proposal_lines.append(
                        f"  #{iter_num}: ❌ high error rate ({error_rate:.1%}) — {explanation[:60]}"
                    )
                    _print_skip_banner(iter_num, baseline_rhos, best_rhos, "high error rate")
                    _release_scorer_module(candidate_ref)
                    continue

            except Exception as e:
                logger.info("  #%d: ❌ full eval crashed — %s", iter_num, e)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Full eval crashed: {e}", diff_summary=diff_summary)
                proposal_lines.append(f"  #{iter_num}: ❌ eval crashed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "full eval crashed")
                _release_scorer_module(candidate_ref)
                continue

            acceptance = _acceptance_diagnostics(new_rhos, baseline_rhos, target_rhos)
            improved_horizons = acceptance["accepted_horizons"]
            improved_count = len(improved_horizons)
            behavior_neutral = _is_behavior_neutral(new_rhos, baseline_rhos)
            legacy_candidate_metrics: Optional[dict[str, dict[str, float]]] = None
            legacy_candidate_rhos: Optional[dict[str, float]] = None
            legacy_rejections: list[str] = []

            if improved_count > 0 and legacy_eval_context is not None and legacy_baseline_rhos is not None:
                legacy_candidate_metrics = legacy_eval_context.evaluate_all_targets(
                    scorer_module=candidate_ref.module_name,
                    scorer_path=candidate_ref.path,
                )
                legacy_candidate_rhos = {
                    h: float(legacy_candidate_metrics[h]["spearman_rho"]) for h in _HORIZONS
                }
                legacy_rejections = _legacy_guard_rejections(
                    legacy_candidate_rhos,
                    legacy_baseline_rhos,
                )

            # Log this proposal
            if improved_count == 0 or legacy_rejections:
                reject_reason = (
                    f"legacy guard failed ({','.join(legacy_rejections)})"
                    if legacy_rejections
                    else "behavior-neutral"
                    if behavior_neutral
                    else acceptance["reject_reason"] or "no improvement"
                )
                _log_evaluated(
                    exp_log=exp_log,
                    iteration=iter_num,
                    baseline_rhos=baseline_rhos,
                    rhos=new_rhos,
                    kept=False,
                    description=f"{reject_reason}: {explanation}",
                    diff_summary=diff_summary,
                    metrics=new_metrics,
                    legacy_rhos=legacy_candidate_rhos,
                    legacy_baseline_rhos=legacy_baseline_rhos,
                    extra=_acceptance_log_fields(acceptance),
                )
                _print_banner(
                    iteration=iter_num,
                    rhos=new_rhos,
                    baseline_rhos=baseline_rhos,
                    best_rhos=best_rhos,
                    kept=False,
                    description=reject_reason,
                    improved_horizons=[],
                )

            # Build per-proposal summary line
            deltas_str = "  ".join(
                f"{h}={new_rhos[h]:.4f} (Δ={new_rhos[h] - baseline_rhos[h]:+.4f})"
                for h in _HORIZONS
            )
            if improved_count > 0 and not legacy_rejections:
                historical_note = (
                    f"; historical {acceptance['historical_reject_reason']}"
                    if acceptance.get("historical_reject_reason")
                    else ""
                )
                status = f"✅ kept material-pareto [↑{','.join(improved_horizons)}{historical_note}]"
            elif legacy_rejections:
                status = f"❌ legacy guard [{','.join(legacy_rejections)}]"
            elif behavior_neutral:
                status = "∅ behavior-neutral"
            elif acceptance["near_miss"]:
                current_up = ",".join(acceptance["current_improved_horizons"]) or "none"
                target_up = ",".join(acceptance["target_improved_horizons"]) or "none"
                status = f"↯ {acceptance['reject_reason']} [current ↑{current_up}; target ↑{target_up}]"
            else:
                status = f"❌ {acceptance['reject_reason'] or 'no improvement'}"
            proposal_lines.append(
                f"  #{iter_num}: {status} — {deltas_str} — {explanation[:80]}"
            )

            # Track all improving proposals (not just best) for potential stacking
            if improved_count > 0 and not legacy_rejections:
                improving_proposals.append({
                    **proposal,
                    "new_rhos": new_rhos,
                    "new_metrics": new_metrics,
                    "quick_rhos": {h: float(quick_rhos[h]) for h in _HORIZONS},
                    "legacy_rhos": legacy_candidate_rhos,
                    "legacy_metrics": legacy_candidate_metrics,
                    "improved_count": improved_count,
                    "improved_horizons": improved_horizons,
                    "acceptance": acceptance,
                    "utility": float(acceptance["current_utility"]),
                })

            _release_scorer_module(candidate_ref)

        # 6. Print per-proposal summary
        print(f"\n--- Round {rounds_completed} proposals [{iter_label}] ---")
        for line in proposal_lines:
            print(line)

        # 7. Cleanup old backups (keep most recent _MAX_BACKUPS)
        removed = _cleanup_old_backups()
        if removed:
            logger.info("Cleaned up %d old backup(s); %d retained", removed, _MAX_BACKUPS)

        # 8. Apply best or try stacking if multiple improved
        if not improving_proposals:
            logger.info("❌ No proposal improved any horizon — reverted")
        else:
            # Sort by weighted utility first, then breadth, then 6m delta.
            improving_proposals.sort(
                key=lambda p: (
                    p["utility"],
                    p["improved_count"],
                    p["new_rhos"]["6m"] - baseline_rhos["6m"],
                ),
                reverse=True,
            )

            applied: list[dict] = []
            applied_iterations: set[int] = set()
            current_code = original_code
            current_rhos = dict(baseline_rhos)
            current_quick_rhos = quick_baseline_rhos or dict(baseline_rhos)
            backup_path: Optional[Path] = None

            for idx, imp in enumerate(improving_proposals):
                if idx == 0:
                    # First (best) improving proposal — apply directly
                    if backup_path is None:
                        backup_path = _backup_scorer(imp["iteration"])
                        logger.info("Backed up scorer → %s", backup_path)
                    tests_passed, test_summary = _write_candidate_and_test(
                        imp["code"],
                        restore_code=worktree_scorer_code,
                    )
                    if not tests_passed:
                        logger.info("  #%d: ❌ candidate test suite failed — %s", imp["iteration"], test_summary[:120])
                        imp["application_rejection"] = f"Candidate improved but test suite failed: {test_summary[:120]}"
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ candidate test suite failed — {test_summary[:80]}"
                        )
                        continue

                    previous_active_ref = active_scorer_ref
                    worktree_scorer_code = imp["code"]
                    current_code = imp["code"]
                    current_rhos = dict(imp["new_rhos"])
                    current_quick_rhos = dict(imp.get("quick_rhos") or current_quick_rhos)
                    active_scorer_code = current_code
                    active_scorer_ref = _current_scorer_ref()
                    _release_scorer_module(previous_active_ref)
                    should_save_snapshot = _is_better_historical_snapshot(current_rhos, best_rhos)
                    if should_save_snapshot:
                        _save_best_scorer_snapshot(
                            iteration=imp["iteration"],
                            rhos=current_rhos,
                            ground_truth_id=ground_truth_id,
                        )
                    for h in _HORIZONS:
                        if imp["new_rhos"][h] > (best_rhos.get(h) or 0):
                            best_rhos[h] = float(imp["new_rhos"][h])

                    # Update baselines so subsequent rounds must beat the new bar
                    baseline_rhos = dict(current_rhos)
                    target_rhos = {
                        h: max(baseline_rhos[h], float(best_rhos.get(h) or 0.0))
                        for h in _HORIZONS
                    }

                    applied.append(imp)
                    applied_iterations.add(imp["iteration"])
                    _log_evaluated(
                        exp_log=exp_log,
                        iteration=imp["iteration"],
                        baseline_rhos=baseline_rhos_original,
                        rhos=imp["new_rhos"],
                        kept=True,
                        description=imp.get("explanation", ""),
                        diff_summary=imp.get("diff_summary", ""),
                        improved_horizons=imp.get("improved_horizons", []),
                        metrics=imp.get("new_metrics"),
                        legacy_rhos=imp.get("legacy_rhos"),
                        legacy_baseline_rhos=legacy_baseline_rhos,
                        extra=_acceptance_log_fields(imp.get("acceptance", {})),
                    )
                    _print_banner(
                        iteration=imp["iteration"],
                        rhos=imp["new_rhos"],
                        baseline_rhos=baseline_rhos_original,
                        best_rhos=best_rhos,
                        kept=True,
                        description=imp.get("explanation", ""),
                        improved_horizons=imp.get("improved_horizons", []),
                    )
                    logger.info("✅ Applied #%d — horizons improved: %s",
                                imp["iteration"], ", ".join(imp["improved_horizons"]))
                else:
                    # Try stacking: patch this proposal's diff onto current scorer
                    imp_diff = _compute_full_diff(original_code, imp["code"])
                    stacked_code = _try_apply_patch(current_code, imp_diff)
                    if stacked_code is None:
                        logger.info("  #%d: ⚠️ patch conflict — cannot stack with previous changes",
                                    imp["iteration"])
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ⚠️ patch conflict, skipped stacking — {imp.get('explanation', '')[:60]}"
                        )
                        continue

                    # Static analysis on stacked code
                    static_errors = _validate_static(stacked_code)
                    if static_errors:
                        logger.info("  #%d: ❌ stacked static analysis failed — %s",
                                    imp["iteration"], static_errors[0])
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ stacked static analysis — {static_errors[0]}"
                        )
                        continue

                    stacked_ref = _materialize_temp_scorer(
                        scorer_workspace_dir,
                        stacked_code,
                        label="stacked",
                        iteration=imp["iteration"],
                    )
                    if not _validate_scorer(
                        code=stacked_code,
                        scorer_module=stacked_ref.module_name,
                        scorer_path=stacked_ref.path,
                    ):
                        logger.info("  #%d: ❌ stacked validation failed", imp["iteration"])
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ stacked validation failed — {imp.get('explanation', '')[:60]}"
                        )
                        _release_scorer_module(stacked_ref)
                        continue

                    # Quick-eval stacked
                    try:
                        stacked_quick = primary_eval_context.quick_evaluate(
                            scorer_module=stacked_ref.module_name,
                            scorer_path=stacked_ref.path,
                            sample_size=_QUICK_EVAL_SAMPLE_SIZE,
                        )
                    except Exception as e:
                        logger.info("  #%d: ❌ stacked quick-eval crashed: %s", imp["iteration"], e)
                        _release_scorer_module(stacked_ref)
                        continue

                    stacked_quick_rejection = _quick_reject_diagnostics(
                        stacked_quick,
                        current_quick_rhos,
                        current_rhos,
                    )
                    if stacked_quick_rejection["reject"]:
                        stacked_deltas = stacked_quick_rejection["sample_deltas"]
                        reason = stacked_quick_rejection["reason"]
                        logger.info(
                            "  #%d: ⏩ stacked quick-rejected %s "
                            "(Δ 1m=%+.4f, 3m=%+.4f, 6m=%+.4f; utility=%+.4f)",
                            imp["iteration"],
                            reason,
                            stacked_deltas["1m"],
                            stacked_deltas["3m"],
                            stacked_deltas["6m"],
                            stacked_quick_rejection["sample_utility"],
                        )
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ⏩ stacked quick {reason} — "
                            f"{imp.get('explanation', '')[:60]}"
                        )
                        _release_scorer_module(stacked_ref)
                        continue

                    # Full eval stacked
                    try:
                        stacked_metrics = primary_eval_context.evaluate_all_targets(
                            scorer_module=stacked_ref.module_name,
                            scorer_path=stacked_ref.path,
                        )
                        stacked_rhos = {
                            h: float(stacked_metrics[h]["spearman_rho"]) for h in _HORIZONS
                        }

                        error_rate = float(stacked_metrics.get("error_rate", 0.0))
                        if error_rate > 0.0:
                            logger.info("  #%d: ❌ stacked high error rate — %.2f%%",
                                        imp["iteration"], error_rate * 100)
                            proposal_lines.append(
                                f"  #{imp['iteration']}: ❌ stacked high error rate ({error_rate:.1%})"
                            )
                            _release_scorer_module(stacked_ref)
                            continue

                    except Exception as e:
                        logger.info("  #%d: ❌ stacked full-eval crashed: %s", imp["iteration"], e)
                        _release_scorer_module(stacked_ref)
                        continue

                    previous_rhos = dict(current_rhos)
                    stacked_improved = _accepted_horizons(stacked_rhos, previous_rhos)
                    if not stacked_improved:
                        logger.info("  #%d: stacking did not further improve", imp["iteration"])
                        proposal_lines.append(
                            f"  #{imp['iteration']}: stacked no further improvement — {imp.get('explanation', '')[:60]}"
                        )
                        _release_scorer_module(stacked_ref)
                        continue

                    stacked_legacy_rhos = None
                    if legacy_eval_context is not None and legacy_baseline_rhos is not None:
                        stacked_legacy_metrics = legacy_eval_context.evaluate_all_targets(
                            scorer_module=stacked_ref.module_name,
                            scorer_path=stacked_ref.path,
                        )
                        stacked_legacy_rhos = {
                            h: float(stacked_legacy_metrics[h]["spearman_rho"]) for h in _HORIZONS
                        }
                        legacy_rejections = _legacy_guard_rejections(
                            stacked_legacy_rhos,
                            legacy_baseline_rhos,
                        )
                        if legacy_rejections:
                            logger.info(
                                "  #%d: stacked legacy guard failed (%s)",
                                imp["iteration"],
                                ", ".join(legacy_rejections),
                            )
                            proposal_lines.append(
                                f"  #{imp['iteration']}: ❌ stacked legacy guard [{','.join(legacy_rejections)}]"
                            )
                            _release_scorer_module(stacked_ref)
                            continue

                    # Stacking improved further — keep it
                    logger.info("✅ Stacked #%d — further improved: %s",
                                imp["iteration"], ", ".join(stacked_improved))
                    previous_code = current_code
                    tests_passed, test_summary = _write_candidate_and_test(
                        stacked_code,
                        restore_code=worktree_scorer_code,
                    )
                    if not tests_passed:
                        logger.info("  #%d: ❌ stacked test suite failed — %s", imp["iteration"], test_summary[:120])
                        imp["application_rejection"] = f"Stacked candidate test suite failed: {test_summary[:120]}"
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ stacked test suite failed — {test_summary[:80]}"
                        )
                        _release_scorer_module(stacked_ref)
                        continue

                    worktree_scorer_code = stacked_code
                    current_code = stacked_code
                    current_rhos = dict(stacked_rhos)
                    current_quick_rhos = {h: float(stacked_quick[h]) for h in _HORIZONS}
                    active_scorer_code = current_code
                    active_scorer_ref = _current_scorer_ref()
                    should_save_snapshot = _is_better_historical_snapshot(current_rhos, best_rhos)
                    if should_save_snapshot:
                        _save_best_scorer_snapshot(
                            iteration=imp["iteration"],
                            rhos=current_rhos,
                            ground_truth_id=ground_truth_id,
                        )
                    for h in _HORIZONS:
                        if stacked_rhos[h] > (best_rhos.get(h) or 0):
                            best_rhos[h] = float(stacked_rhos[h])

                    # Update baselines so subsequent rounds must beat the new bar
                    baseline_rhos = dict(current_rhos)
                    target_rhos = {
                        h: max(baseline_rhos[h], float(best_rhos.get(h) or 0.0))
                        for h in _HORIZONS
                    }

                    applied.append(imp)
                    applied_iterations.add(imp["iteration"])
                    _log_evaluated(
                        exp_log=exp_log,
                        iteration=imp["iteration"],
                        baseline_rhos=previous_rhos,
                        rhos=stacked_rhos,
                        kept=True,
                        description=f"Stacked: {imp.get('explanation', '')}",
                        diff_summary=_compute_diff_summary(previous_code, stacked_code),
                        improved_horizons=stacked_improved,
                        metrics=stacked_metrics,
                        legacy_rhos=stacked_legacy_rhos,
                        legacy_baseline_rhos=legacy_baseline_rhos,
                    )
                    _print_banner(
                        iteration=imp["iteration"],
                        rhos=stacked_rhos,
                        baseline_rhos=previous_rhos,
                        best_rhos=best_rhos,
                        kept=True,
                        description=f"Stacked: {imp.get('explanation', '')}",
                        improved_horizons=stacked_improved,
                    )
                    proposal_lines.append(
                        f"  #{imp['iteration']}: ✅ stacked [↑{','.join(stacked_improved)}] — "
                        f"{'  '.join(f'{h}={stacked_rhos[h]:.4f} (Δ={stacked_rhos[h] - previous_rhos[h]:+.4f})' for h in _HORIZONS)}"
                    )
                    _release_scorer_module(stacked_ref)

            for imp in improving_proposals:
                if imp["iteration"] in applied_iterations:
                    continue
                _log_evaluated(
                    exp_log=exp_log,
                    iteration=imp["iteration"],
                    baseline_rhos=baseline_rhos_original,
                    rhos=imp["new_rhos"],
                    kept=False,
                    description=imp.get("application_rejection")
                    or f"Candidate improved but was not applied: {imp.get('explanation', '')}",
                    diff_summary=imp.get("diff_summary", ""),
                    improved_horizons=imp.get("improved_horizons", []),
                    metrics=imp.get("new_metrics"),
                    legacy_rhos=imp.get("legacy_rhos"),
                    legacy_baseline_rhos=legacy_baseline_rhos,
                    extra=_acceptance_log_fields(imp.get("acceptance", {})),
                )
                _print_banner(
                    iteration=imp["iteration"],
                    rhos=imp["new_rhos"],
                    baseline_rhos=baseline_rhos_original,
                    best_rhos=best_rhos,
                    kept=False,
                    description="improved but not applied",
                    improved_horizons=[],
                )

            # Print stacking results
            if len(applied) > 1:
                print(f"\n  ⟳ Stacked {len(applied)} improvements — cumulative ρ:")
                for h in _HORIZONS:
                    print(f"     {_HORIZON_LABELS[h]:>10}  ρ={current_rhos[h]:.4f}  (Δ={current_rhos[h] - baseline_rhos[h]:+.4f})")

            # Update baseline for next round
            baseline_rhos = current_rhos

        time.sleep(1)

    # Summary
    total_experiments = exp_log.total_experiments()
    print(f"\n{'='*60}")
    print(f"  🏁 Improvement loop finished after {rounds_completed} rounds ({total_experiments} total experiments)")
    for h in _HORIZONS:
        print(f"     {_HORIZON_LABELS[h]:>10}  Final ρ = {baseline_rhos[h]:.4f}")
    print(f"  📝 Experiment log: {exp_log.path}")
    print(f"{'='*60}\n")
    _release_scorer_module(active_scorer_ref)
    scorer_workspace.cleanup()


def _log_skip(
    exp_log: ExperimentLog,
    iteration: int,
    baseline_rhos: dict[str, float],
    description: str,
    diff_summary: str = "",
    extra: Optional[dict] = None,
) -> None:
    """Log a skipped/errored iteration with all three horizon baselines."""
    exp_log.log(
        iteration=iteration,
        spearman_rho=float(baseline_rhos["6m"]),
        baseline_rho=float(baseline_rhos["6m"]),
        kept=False,
        description=description,
        diff_summary=diff_summary,
        spearman_rho_1m=float(baseline_rhos["1m"]),
        baseline_rho_1m=float(baseline_rhos["1m"]),
        spearman_rho_3m=float(baseline_rhos["3m"]),
        baseline_rho_3m=float(baseline_rhos["3m"]),
        extra=extra,
    )


def _log_evaluated(
    exp_log: ExperimentLog,
    iteration: int,
    baseline_rhos: dict[str, float],
    rhos: dict[str, float],
    kept: bool,
    description: str,
    diff_summary: str = "",
    improved_horizons: Optional[list[str]] = None,
    metrics: Optional[dict[str, dict[str, float]]] = None,
    legacy_rhos: Optional[dict[str, float]] = None,
    legacy_baseline_rhos: Optional[dict[str, float]] = None,
    extra: Optional[dict] = None,
) -> None:
    """Log an evaluated proposal with actual per-horizon metrics."""
    extra_fields = dict(extra or {})
    if metrics:
        for h in _HORIZONS:
            horizon_metrics = metrics.get(h, {})
            extra_fields[f"hit_rate_top20_{h}"] = float(horizon_metrics.get("hit_rate_top20", 0))
            extra_fields[f"mean_excess_return_{h}"] = float(horizon_metrics.get("mean_excess_return", 0))
        extra_fields["proposal_utility"] = round(_proposal_utility(rhos, baseline_rhos), 6)

    if legacy_rhos and legacy_baseline_rhos:
        for h in _HORIZONS:
            extra_fields[f"legacy_spearman_rho_{h}"] = round(float(legacy_rhos[h]), 6)
            extra_fields[f"legacy_baseline_rho_{h}"] = round(float(legacy_baseline_rhos[h]), 6)
            extra_fields[f"legacy_delta_{h}"] = round(float(legacy_rhos[h]) - float(legacy_baseline_rhos[h]), 6)

    exp_log.log(
        iteration=iteration,
        spearman_rho=float(rhos["6m"]),
        baseline_rho=float(baseline_rhos["6m"]),
        kept=kept,
        description=description,
        diff_summary=diff_summary,
        spearman_rho_1m=float(rhos["1m"]),
        baseline_rho_1m=float(baseline_rhos["1m"]),
        spearman_rho_3m=float(rhos["3m"]),
        baseline_rho_3m=float(baseline_rhos["3m"]),
        improved_horizons=improved_horizons if kept else None,
        extra=extra_fields,
    )


def _print_skip_banner(
    iteration: int,
    baseline_rhos: dict[str, float],
    best_rhos: dict[str, Optional[float]],
    reason: str,
) -> None:
    """Print a banner for a skipped/errored iteration."""
    _print_iteration_summary(
        iteration=iteration,
        rhos=baseline_rhos,
        best_rhos=best_rhos,
        status="❌ REVERTED",
        reason=reason,
    )


def show_status() -> None:
    """Print a summary of experiment history across all horizons."""
    exp_log = ExperimentLog()
    records = exp_log.read_all()
    legacy = ExperimentLog.read_legacy_logs()
    total_legacy = len(legacy)

    if not records:
        print(f"No unified experiments recorded yet. ({total_legacy} legacy experiments found.)")
        return

    total = len(records)
    kept = sum(1 for r in records if r.get("kept"))
    best_by_horizon = exp_log.best_rho_per_horizon(current_lineage=True)
    latest = records[-1]

    print("\n📊 Scorer Improvement Status")
    print(f"{'─'*50}")
    print(f"  Unified experiments:  {total}")
    print(f"  Legacy (historical):  {total_legacy}")
    print(f"  Kept (improved):      {kept}")
    print(f"  Reverted:             {total - kept}")
    print(f"  Success rate:         {kept/total*100:.1f}%")
    print(f"{'─'*50}")
    for h in _HORIZONS:
        latest_h = latest.get(f"spearman_rho_{h}") or (latest["spearman_rho"] if h == "6m" else None)
        best_h = best_by_horizon.get(h)
        latest_str = f"{latest_h:.4f}" if latest_h else "N/A"
        best_str = f"{best_h:.4f}" if best_h else "N/A"
        print(f"  {_HORIZON_LABELS[h]:>12}:  Latest ρ = {latest_str}  |  Best ρ = {best_str}")
    print(f"{'─'*50}")
    print("\nLast 10 unified experiments:")
    for r in records[-10:]:
        status = "✅" if r.get("kept") else "❌"
        improved = r.get("improved_horizons", [])
        tag = f" [{', '.join(improved)}]" if improved else ""
        print(
            f"  {status} #{r['iteration']}: ρ6m={r['spearman_rho']:.4f}"
            f"{tag} — {r.get('description', '')[:60]}"
        )
    if legacy:
        print(f"\n… and {total_legacy} legacy experiments from separate horizon optimizers")
    print()
