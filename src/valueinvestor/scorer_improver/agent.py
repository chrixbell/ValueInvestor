"""Phase 3 — LLM agent that autonomously improves scorer.py.

Inspired by karpathy/autoresearch: reads the current scorer code, sends it
to an LLM with experiment history and program context, evaluates the proposed
change against **all three** forward-return horizons (1m, 3m, 6m), and keeps
the change if **any** target improves.  Loops indefinitely.
"""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from valueinvestor.scorer_improver.evaluator import (
    evaluate_scorer_all_targets,
    quick_evaluate,
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

# Number of parallel LLM proposals per iteration (env-overridable)
_N_PARALLEL = int(os.environ.get("IMPROVE_SCORER_PARALLEL", "3"))
# Epsilon for quick-reject threshold
_QUICK_REJECT_MARGIN = 0.02
# Maximum allowed degradation in any single horizon for a change to be kept.
# Prevents severe trade-offs where a small improvement in one horizon
# destroys others (e.g. +0.002 in 1m but -0.018 in 3m, -0.013 in 6m).
_MAX_DEGRADATION = 0.01
_LEGACY_GUARD_MAX_DEGRADATION = float(os.environ.get("IMPROVE_SCORER_LEGACY_MAX_DEGRADATION", "0.005"))
_MIN_RHO_IMPROVEMENT = 1e-6
_HORIZON_WEIGHTS = {"1m": 0.25, "3m": 0.35, "6m": 0.40}
# How many recent experiments to show the LLM
_EXPERIMENT_HISTORY_COUNT = int(os.environ.get("IMPROVE_SCORER_HISTORY", "15"))
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
_MAX_PROPOSAL_TOKENS = int(os.environ.get("IMPROVE_SCORER_MAX_PROPOSAL_TOKENS", "7000"))
_MAX_REPAIR_TOKENS = int(os.environ.get("IMPROVE_SCORER_MAX_REPAIR_TOKENS", "7000"))

_EXPLORATION_LANES = (
    "Lane A: Make one minimal threshold/sign/scaling correction in an existing value sub-factor. Avoid composite aggregation changes.",
    "Lane B: Add or remove exactly one value sub-factor using a raw field already present in scorer.py. Keep the composite formula unchanged.",
    "Lane C: Make one focused quality-score change. Avoid changing value_score percentile/ranking logic or the composite aggregation formula.",
    "Lane D: Simplify one noisy interaction term without changing factor weights or the rank/composite mechanics.",
)

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
and 6-month. A change is kept if it improves ρ for ANY of the three horizons.

Rules:
- Output a compact unified diff patch for src/valueinvestor/screener/scorer.py
- Include ---/+++ file headers and at least one @@ hunk
- Do not use ellipses, placeholders, or omitted context in the patch
- Keep the MultiFactorScorer class interface intact (score/rank methods)
- Valid Python 3.9+ only
- Handle None values gracefully
- Be creative but grounded — use financial domain knowledge
- Learn from past experiments: don't repeat failed approaches
- Make ONE focused change per iteration (easier to attribute improvements)
- Keep the patch as small as possible: change only the lines required

Wrap your unified diff in ```diff ... ``` code fences.
After the code, briefly explain what you changed and why (1-2 sentences).
If a patch is impossible, fall back to the complete scorer.py in ```python ... ```
code fences.
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


def _backup_scorer(iteration: int) -> Path:
    """Backup the current scorer before modification."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"scorer_{iteration:04d}_{ts}.py"
    shutil.copy2(SCORER_PATH, backup_path)
    return backup_path


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
        float(candidate_rhos[h]) > float(current_best_rhos.get(h) or float("-inf")) + _MIN_RHO_IMPROVEMENT
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
    """Return history records that teach the LLM something concrete."""
    informative = [
        r for r in records
        if r.get("kept") or r.get("diff_summary") or "legacy_" in " ".join(r.keys())
    ]
    if not informative:
        return records[-max_entries:] if len(records) > max_entries else records
    return informative[-max_entries:] if len(informative) > max_entries else informative


def _compute_failure_guidance(records: list[dict]) -> str:
    """Summarize repeated failed themes so the next prompts diversify."""
    failed_text = "\n".join(
        str(r.get("description", "")).lower()
        for r in records
        if not r.get("kept")
    )
    if not failed_text:
        return ""

    patterns = [
        ("aggregation-only changes", ("harmonic mean", "geometric mean", "arithmetic mean", "minimum of")),
        ("value-score percentile/rank transforms", ("value sub-score is converted", "value score is converted", "cross-sectional percentile")),
        ("quality-raw multipliers", ("quality raw", "quality‑raw", "quality_raw")),
        ("ROE/ROA substitutions", ("replace roe", "replacing roe", "roa")),
        ("PB-sign fixes", ("pb scoring", "price-to-book", "price‑to‑book")),
        ("undefined pe_safe / _compute_quality_raw repairs", ("pe_safe", "undefined", "_compute_quality_raw", "broken quality raw")),
    ]
    lines = []
    for label, needles in patterns:
        count = sum(failed_text.count(needle) for needle in needles)
        if count >= 3:
            lines.append(f"- Avoid repeating {label}; it has dominated recent failed/no-op proposals.")
    if not lines:
        return ""
    return "Recent failure guidance:\n" + "\n".join(lines)


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
    # Build status string showing all three horizons
    status_parts = ["Targets: 1-month, 3-month, and 6-month forward returns"]
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
        truncated_code = scorer_code

    return (
        f"## Program Context\n\n{context}\n\n"
        f"## Current scorer.py\n\n```python\n{truncated_code}\n```\n\n"
        "Now propose your improvement. Prefer a compact unified diff patch "
        "wrapped in ```diff ... ``` code fences, followed by a brief explanation. "
        "Only output the complete scorer.py in ```python ... ``` fences if you cannot "
        "express the change as a patch."
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
            return _try_apply_patch_with_system_patch(base_code, patch_text)
        pos, matched_len = match
        result[pos:pos + matched_len] = new_hunk
        offset += len(new_hunk) - matched_len
        applied_any = True

    patched = "".join(result) if applied_any else None
    if patched is not None and patched != base_code:
        return patched
    return _try_apply_patch_with_system_patch(base_code, patch_text)


def _extract_proposal_code(response: str, original_code: str) -> tuple[Optional[str], str, str]:
    """Extract proposal code from a patch response, falling back to full-file code."""
    explanation = _extract_explanation(response)
    patch_text = _extract_patch(response)
    if patch_text:
        patched_code = _try_apply_patch(original_code, patch_text)
        if patched_code is not None:
            return patched_code, explanation, _compute_diff_summary(original_code, patched_code)
        logger.warning("LLM returned a patch that could not be applied; trying full-file fallback")

    new_code = _extract_code(response)
    diff_summary = _compute_diff_summary(original_code, new_code) if new_code else ""
    return new_code, explanation, diff_summary


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
) -> str:
    """Call an LLM client with a token cap, tolerating test doubles without kwargs."""
    try:
        response = llm_client.complete(
            system_prompt=_AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    except TypeError as exc:
        if "max_tokens" not in str(exc):
            raise
        return llm_client.complete(system_prompt=_AGENT_SYSTEM_PROMPT, user_prompt=user_prompt)
    if response.strip() or max_tokens <= 0 or not retry_empty:
        return response

    logger.warning(
        "LLM returned empty content at max_tokens=%d; retrying once without a token cap",
        max_tokens,
    )
    return llm_client.complete(system_prompt=_AGENT_SYSTEM_PROMPT, user_prompt=user_prompt)


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
) -> dict:
    """Call the LLM and optionally repair invalid/no-op proposal responses."""
    response = _llm_complete(llm_client, user_prompt, _MAX_PROPOSAL_TOKENS)
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


def _is_duplicate_diff(diff_summary: str, recent_records: list[dict]) -> bool:
    """Return True when this proposal repeats a recent evaluated change."""
    if not diff_summary or diff_summary == "(no changes)":
        return False
    return any(r.get("diff_summary") == diff_summary for r in recent_records)


def _semantic_failure_theme(text: str) -> str:
    """Return a coarse theme for proposals that repeatedly fail despite varied diffs."""
    normalized = text.lower()
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
        improved = any(float(record.get(f"delta_{h}", 0.0)) > _MIN_RHO_IMPROVEMENT for h in ("1m", "3m"))
        improved = improved or float(record.get("delta", 0.0)) > _MIN_RHO_IMPROVEMENT
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


def _accepted_horizons(rhos: dict[str, float], baseline_rhos: dict[str, float]) -> list[str]:
    """Return improved horizons if a candidate passes the keep policy."""
    deltas = _rho_deltas(rhos, baseline_rhos)
    improved = [h for h in _HORIZONS if deltas[h] > _MIN_RHO_IMPROVEMENT]
    if not improved:
        return []

    worst_degradation = max(-delta for delta in deltas.values())
    if worst_degradation > _MAX_DEGRADATION:
        return []

    if _proposal_utility(rhos, baseline_rhos) <= _MIN_RHO_IMPROVEMENT:
        return []

    return improved


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
    errors = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return errors


def _validate_scorer() -> bool:
    """Check if the current scorer is valid Python and doesn't crash.

    Runs compile() + 6 smoke-test fixtures covering edge cases, plus a rank()
    call to verify cross-sectional ranking produces valid output.
    """
    try:
        code = SCORER_PATH.read_text(encoding="utf-8")
        compile(code, str(SCORER_PATH), "exec")
    except SyntaxError as e:
        logger.warning("Scorer has syntax error: %s", e)
        return False

    try:
        import importlib

        if SCORER_MODULE in sys.modules:
            importlib.reload(sys.modules[SCORER_MODULE])
        else:
            importlib.import_module(SCORER_MODULE)

        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        loaded_mod = sys.modules[SCORER_MODULE]
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
    summary = result.stdout.strip().split("\n")[-1] if result.stdout else result.stderr[:200]
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
    ``IMPROVE_SCORER_PARALLEL`` env var, default 3).  Uses quick subset
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

    ground_truth_id = (
        ground_truth_fingerprint(primary_ground_truth_path)
        if primary_ground_truth_path is not None
        else ground_truth_fingerprint()
    )
    os.environ["IMPROVE_SCORER_ACTIVE_GROUND_TRUTH_ID"] = ground_truth_id
    comparable_history = exp_log.read_current_lineage(ground_truth_id=ground_truth_id)
    if not comparable_history and exp_log.total_experiments() > 0:
        logger.info(
            "No prior experiments match current ground truth (%s); historical best metrics "
            "will be ignored for this run.",
            ground_truth_id,
        )

    restored_best_meta = _restore_best_scorer_snapshot(ground_truth_id)
    if restored_best_meta:
        logger.info(
            "Resume baseline restored from best scorer snapshot: iteration #%s",
            restored_best_meta.get("iteration", "?"),
        )

    # Get baseline evaluation across all three horizons
    logger.info("Computing baseline evaluation (1m, 3m, 6m) …")
    baseline_metrics = evaluate_scorer_all_targets(
        scorer_module=SCORER_MODULE,
        ground_truth_path=primary_ground_truth_path,
        allow_legacy_schema=allow_primary_legacy_schema,
    )
    baseline_rhos: dict[str, float] = {
        h: float(baseline_metrics[h]["spearman_rho"]) for h in _HORIZONS
    }
    legacy_baseline_rhos: Optional[dict[str, float]] = None
    if legacy_ground_truth_path is not None:
        legacy_metrics = evaluate_scorer_all_targets(
            scorer_module=SCORER_MODULE,
            ground_truth_path=legacy_ground_truth_path,
            allow_legacy_schema=True,
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
    while not _stop:
        if max_iterations > 0 and rounds_completed >= max_iterations:
            print(f"\n🏁 Reached max iterations ({max_iterations}). Stopping.")
            break

        # 1. Read current scorer
        original_code = _read_scorer()

        # 2. Get experiment history. Keep invalid/no-patch attempts out of the
        # prompt so they do not drown out real evaluated changes.
        lineage_records = exp_log.read_current_lineage(ground_truth_id=ground_truth_id)
        recent = _records_for_prompt(lineage_records, _EXPERIMENT_HISTORY_COUNT)
        experiment_history = _build_experiment_history(
            exp_log, recent, include_legacy=False, max_entries=_EXPERIMENT_HISTORY_COUNT,
        )

        # Compute summary stats from the records
        summary_stats = _compute_summary_stats(lineage_records[-50:]) if lineage_records else ""
        failure_guidance = _compute_failure_guidance(lineage_records[-75:])
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
                failure_guidance=failure_guidance,
                lane_instruction=lane,
            )

        user_prompt = _prompt_for_lane(rounds_completed)

        # 4. Generate proposals — each is its own iteration
        proposals: list[dict] = []
        if len(parallel_clients) == 1:
            iteration += 1
            llm = parallel_clients[0]
            try:
                proposal_result = _complete_proposal(llm, user_prompt, original_code, iteration=iteration)
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
            # Parallel path — each LLM call gets its own iteration number
            def _call_llm(llm_client, iter_num, prompt):
                proposal_result = _complete_proposal(
                    llm_client,
                    prompt,
                    original_code,
                    iteration=iter_num,
                )
                return {
                    **proposal_result,
                    "iteration": iter_num,
                }

            n_parallel = len(parallel_clients)
            base_iter = iteration
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
                futures = {
                    executor.submit(_call_llm, c, base_iter + i + 1, _prompt_for_lane(i)): base_iter + i + 1
                    for i, c in enumerate(parallel_clients)
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        iteration = max(iteration, result["iteration"])
                        proposals.append(result)
                    except Exception as e:
                        iter_num = futures[future]
                        iteration = max(iteration, iter_num)
                        logger.warning("Parallel LLM call failed: %s", e)
                        _log_skip(exp_log, iter_num, baseline_rhos, f"LLM call failed: {e}")
                        _print_skip_banner(iter_num, baseline_rhos, best_rhos, f"LLM error: {e}")

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

        # 5. Evaluate each proposal, log individually, collect improving ones
        improving_proposals: list[dict] = []
        baseline_rhos_original = dict(baseline_rhos)
        proposal_lines: list[str] = []

        for pi, proposal in enumerate(proposals):
            new_code = proposal.get("code")
            explanation = proposal.get("explanation", "")
            diff_summary = proposal.get("diff_summary", "")
            iter_num = proposal["iteration"]

            if _is_duplicate_diff(diff_summary, recent):
                logger.debug("  #%d: duplicate diff — skipping", iter_num)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Duplicate diff: {explanation}", diff_summary=diff_summary)
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "duplicate diff")
                continue

            failed_theme = _is_repeated_failed_theme(diff_summary, explanation, recent)
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

            # Write proposal and validate
            _write_scorer(new_code)

            # Static analysis (ruff) — catch NameError before runtime
            static_errors = _validate_static(new_code)
            if static_errors:
                logger.info("  #%d: ❌ static analysis failed — %s", iter_num, static_errors[0])
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Static analysis failed: {static_errors[0]}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(f"  #{iter_num}: ❌ static analysis — {static_errors[0]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "static analysis failed")
                continue

            if not _validate_scorer():
                logger.info("  #%d: ❌ validation failed — %s", iter_num, explanation[:80])
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Validation failed: {explanation}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(f"  #{iter_num}: ❌ validation failed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "validation failed")
                continue

            # Run test suite before accepting
            tests_passed, test_summary = _run_scorer_test_suite()
            if not tests_passed:
                logger.info("  #%d: ❌ test suite failed — %s", iter_num, test_summary[:120])
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Test suite failed: {test_summary[:120]}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(f"  #{iter_num}: ❌ test suite failed — {test_summary[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "test suite failed")
                continue

            # Quick evaluation on 1K-row sample
            try:
                quick_rhos = quick_evaluate(
                    scorer_module=SCORER_MODULE,
                    sample_size=1000,
                    ground_truth_path=primary_ground_truth_path,
                    allow_legacy_schema=allow_primary_legacy_schema,
                )
            except Exception as e:
                logger.info("  #%d: ❌ quick-eval crashed — %s", iter_num, e)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Quick-eval crashed: {e}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(f"  #{iter_num}: ❌ eval crashed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "quick eval crashed")
                continue

            # Quick reject: skip if all horizons are clearly below target
            all_significantly_worse = all(
                quick_rhos[h] < target_rhos[h] - _QUICK_REJECT_MARGIN
                for h in _HORIZONS
            )
            if all_significantly_worse:
                logger.info(
                    "  #%d: ⏩ quick-rejected (ρ 1m=%.4f, 3m=%.4f, 6m=%.4f)",
                    iter_num, quick_rhos["1m"], quick_rhos["3m"], quick_rhos["6m"],
                )
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Quick-rejected: {explanation}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(
                    f"  #{iter_num}: ⏩ quick-rejected — ρ1m={quick_rhos['1m']:.4f} "
                    f"ρ3m={quick_rhos['3m']:.4f} ρ6m={quick_rhos['6m']:.4f} — {explanation[:60]}"
                )
                _print_iteration_summary(
                    iteration=iter_num,
                    rhos=quick_rhos,
                    best_rhos=best_rhos,
                    status="❌ REVERTED",
                    reason="quick-rejected",
                )
                continue

            # Full evaluation
            try:
                new_metrics = evaluate_scorer_all_targets(
                    scorer_module=SCORER_MODULE,
                    ground_truth_path=primary_ground_truth_path,
                    allow_legacy_schema=allow_primary_legacy_schema,
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
                    _write_scorer(original_code)
                    proposal_lines.append(
                        f"  #{iter_num}: ❌ high error rate ({error_rate:.1%}) — {explanation[:60]}"
                    )
                    _print_skip_banner(iter_num, baseline_rhos, best_rhos, "high error rate")
                    continue

            except Exception as e:
                logger.info("  #%d: ❌ full eval crashed — %s", iter_num, e)
                _log_skip(exp_log, iter_num, baseline_rhos,
                          f"Full eval crashed: {e}", diff_summary=diff_summary)
                _write_scorer(original_code)
                proposal_lines.append(f"  #{iter_num}: ❌ eval crashed — {explanation[:80]}")
                _print_skip_banner(iter_num, baseline_rhos, best_rhos, "full eval crashed")
                continue

            improved_horizons = _accepted_horizons(new_rhos, target_rhos)
            improved_count = len(improved_horizons)
            legacy_candidate_metrics: Optional[dict[str, dict[str, float]]] = None
            legacy_candidate_rhos: Optional[dict[str, float]] = None
            legacy_rejections: list[str] = []

            if improved_count > 0 and legacy_ground_truth_path is not None and legacy_baseline_rhos is not None:
                legacy_candidate_metrics = evaluate_scorer_all_targets(
                    scorer_module=SCORER_MODULE,
                    ground_truth_path=legacy_ground_truth_path,
                    allow_legacy_schema=True,
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
                    if legacy_rejections else "no improvement"
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
                status = f"✅ kept [↑{','.join(improved_horizons)}]"
            elif legacy_rejections:
                status = f"❌ legacy guard [{','.join(legacy_rejections)}]"
            else:
                status = "❌ no improvement"
            proposal_lines.append(
                f"  #{iter_num}: {status} — {deltas_str} — {explanation[:80]}"
            )

            # Track all improving proposals (not just best) for potential stacking
            if improved_count > 0 and not legacy_rejections:
                improving_proposals.append({
                    **proposal,
                    "new_rhos": new_rhos,
                    "new_metrics": new_metrics,
                    "legacy_rhos": legacy_candidate_rhos,
                    "legacy_metrics": legacy_candidate_metrics,
                    "improved_count": improved_count,
                    "improved_horizons": improved_horizons,
                    "utility": _proposal_utility(new_rhos, target_rhos),
                })

            # Restore original for next proposal eval
            _write_scorer(original_code)

        # 6. Print per-proposal summary
        print(f"\n--- Round {rounds_completed} proposals [{iter_label}] ---")
        for line in proposal_lines:
            print(line)

        # 7. Apply best or try stacking if multiple improved
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

            for idx, imp in enumerate(improving_proposals):
                if idx == 0:
                    # First (best) improving proposal — apply directly
                    backup_path = _backup_scorer(imp["iteration"])
                    logger.info("Backed up scorer → %s", backup_path)
                    _write_scorer(imp["code"])
                    current_code = imp["code"]
                    current_rhos = dict(imp["new_rhos"])
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

                    # Validate stacked scorer
                    _write_scorer(stacked_code)

                    # Static analysis on stacked code
                    static_errors = _validate_static(stacked_code)
                    if static_errors:
                        logger.info("  #%d: ❌ stacked static analysis failed — %s",
                                    imp["iteration"], static_errors[0])
                        _write_scorer(current_code)
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ stacked static analysis — {static_errors[0]}"
                        )
                        continue

                    if not _validate_scorer():
                        logger.info("  #%d: ❌ stacked validation failed", imp["iteration"])
                        _write_scorer(current_code)
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ❌ stacked validation failed — {imp.get('explanation', '')[:60]}"
                        )
                        continue

                    # Quick-eval stacked
                    try:
                        stacked_quick = quick_evaluate(
                            scorer_module=SCORER_MODULE,
                            sample_size=1000,
                            ground_truth_path=primary_ground_truth_path,
                            allow_legacy_schema=allow_primary_legacy_schema,
                        )
                    except Exception as e:
                        logger.info("  #%d: ❌ stacked quick-eval crashed: %s", imp["iteration"], e)
                        _write_scorer(current_code)
                        continue

                    all_worse = all(
                        stacked_quick[h] < current_rhos[h] - _QUICK_REJECT_MARGIN
                        for h in _HORIZONS
                    )
                    if all_worse:
                        logger.info("  #%d: ⏩ stacked quick-rejected (ρ 1m=%.4f, 3m=%.4f, 6m=%.4f)",
                                    imp["iteration"],
                                    stacked_quick["1m"], stacked_quick["3m"], stacked_quick["6m"])
                        _write_scorer(current_code)
                        proposal_lines.append(
                            f"  #{imp['iteration']}: ⏩ stacked regressed — {imp.get('explanation', '')[:60]}"
                        )
                        continue

                    # Full eval stacked
                    try:
                        stacked_metrics = evaluate_scorer_all_targets(
                            scorer_module=SCORER_MODULE,
                            ground_truth_path=primary_ground_truth_path,
                            allow_legacy_schema=allow_primary_legacy_schema,
                        )
                        stacked_rhos = {
                            h: float(stacked_metrics[h]["spearman_rho"]) for h in _HORIZONS
                        }

                        error_rate = float(stacked_metrics.get("error_rate", 0.0))
                        if error_rate > 0.0:
                            logger.info("  #%d: ❌ stacked high error rate — %.2f%%",
                                        imp["iteration"], error_rate * 100)
                            _write_scorer(current_code)
                            proposal_lines.append(
                                f"  #{imp['iteration']}: ❌ stacked high error rate ({error_rate:.1%})"
                            )
                            continue

                    except Exception as e:
                        logger.info("  #%d: ❌ stacked full-eval crashed: %s", imp["iteration"], e)
                        _write_scorer(current_code)
                        continue

                    previous_rhos = dict(current_rhos)
                    stacked_improved = _accepted_horizons(stacked_rhos, previous_rhos)
                    if not stacked_improved:
                        logger.info("  #%d: stacking did not further improve", imp["iteration"])
                        _write_scorer(current_code)
                        proposal_lines.append(
                            f"  #{imp['iteration']}: stacked no further improvement — {imp.get('explanation', '')[:60]}"
                        )
                        continue

                    stacked_legacy_rhos = None
                    if legacy_ground_truth_path is not None and legacy_baseline_rhos is not None:
                        stacked_legacy_metrics = evaluate_scorer_all_targets(
                            scorer_module=SCORER_MODULE,
                            ground_truth_path=legacy_ground_truth_path,
                            allow_legacy_schema=True,
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
                            _write_scorer(current_code)
                            proposal_lines.append(
                                f"  #{imp['iteration']}: ❌ stacked legacy guard [{','.join(legacy_rejections)}]"
                            )
                            continue

                    # Stacking improved further — keep it
                    logger.info("✅ Stacked #%d — further improved: %s",
                                imp["iteration"], ", ".join(stacked_improved))
                    previous_code = current_code
                    current_code = stacked_code
                    current_rhos = dict(stacked_rhos)
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
                        f"{'  '.join(f'{h}={stacked_rhos[h]:.4f} (Δ={stacked_rhos[h] - baseline_rhos[h]:+.4f})' for h in _HORIZONS)}"
                    )

            for imp in improving_proposals:
                if imp["iteration"] in applied_iterations:
                    continue
                _log_evaluated(
                    exp_log=exp_log,
                    iteration=imp["iteration"],
                    baseline_rhos=baseline_rhos_original,
                    rhos=imp["new_rhos"],
                    kept=False,
                    description=f"Candidate improved but was not applied: {imp.get('explanation', '')}",
                    diff_summary=imp.get("diff_summary", ""),
                    improved_horizons=imp.get("improved_horizons", []),
                    metrics=imp.get("new_metrics"),
                    legacy_rhos=imp.get("legacy_rhos"),
                    legacy_baseline_rhos=legacy_baseline_rhos,
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
) -> None:
    """Log an evaluated proposal with actual per-horizon metrics."""
    extra = {}
    if metrics:
        for h in _HORIZONS:
            horizon_metrics = metrics.get(h, {})
            extra[f"hit_rate_top20_{h}"] = float(horizon_metrics.get("hit_rate_top20", 0))
            extra[f"mean_excess_return_{h}"] = float(horizon_metrics.get("mean_excess_return", 0))
        extra["proposal_utility"] = round(_proposal_utility(rhos, baseline_rhos), 6)

    if legacy_rhos and legacy_baseline_rhos:
        for h in _HORIZONS:
            extra[f"legacy_spearman_rho_{h}"] = round(float(legacy_rhos[h]), 6)
            extra[f"legacy_baseline_rho_{h}"] = round(float(legacy_baseline_rhos[h]), 6)
            extra[f"legacy_delta_{h}"] = round(float(legacy_rhos[h]) - float(legacy_baseline_rhos[h]), 6)

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
        extra=extra,
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
