"""JSONL experiment logger for the scorer improvement loop.

Each experiment is logged as a single JSON line in
``data/trainer/experiments.jsonl``.  Records include all three horizon
rhos (1m, 3m, 6m).  The agent reads the last N experiments to provide
context for the LLM, and can also read legacy horizon-specific logs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from valueinvestor.scorer_improver.data_prep import TRAINER_DIR

logger = logging.getLogger(__name__)

EXPERIMENT_LOG_FILE = TRAINER_DIR / "experiments.jsonl"

# Legacy log files from separate horizon-specific optimizers
_LEGACY_LOG_FILES: dict = {
    "1m": TRAINER_DIR / "experiments_1m.jsonl",
    "3m": TRAINER_DIR / "experiments_3m.jsonl",
    "6m": TRAINER_DIR / "experiments.jsonl",  # the original, now legacy
}

_HORIZONS = ("1m", "3m", "6m")

# Minimum absolute drop in ρ to infer a manual rollback / lineage reset.
# Smaller drops are treated as noise or minor trade-off regressions rather
# than intentional resets.
_MIN_RESET_THRESHOLD = 0.005


class ExperimentLog:
    """Append-only JSONL logger for scorer improvement experiments."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or EXPERIMENT_LOG_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        iteration: int,
        spearman_rho: float,
        baseline_rho: float,
        kept: bool,
        description: str,
        diff_summary: str = "",
        extra: Optional[Dict[str, Any]] = None,
        # Multi-horizon fields
        spearman_rho_1m: Optional[float] = None,
        baseline_rho_1m: Optional[float] = None,
        spearman_rho_3m: Optional[float] = None,
        baseline_rho_3m: Optional[float] = None,
        improved_horizons: Optional[List[str]] = None,
    ) -> None:
        """Append one experiment record.

        *spearman_rho* / *baseline_rho* are the 6m values (primary).
        The 1m and 3m values are stored in separate fields.
        """
        record: dict[str, Any] = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spearman_rho": round(spearman_rho, 6),
            "baseline_rho": round(baseline_rho, 6),
            "delta": round(spearman_rho - baseline_rho, 6),
            "kept": kept,
            "description": description,
            "diff_summary": diff_summary[:500],
            "horizon": "all",
        }
        # Multi-horizon fields
        if spearman_rho_1m is not None:
            record["spearman_rho_1m"] = round(spearman_rho_1m, 6)
        if baseline_rho_1m is not None:
            record["baseline_rho_1m"] = round(baseline_rho_1m, 6)
        if spearman_rho_3m is not None:
            record["spearman_rho_3m"] = round(spearman_rho_3m, 6)
        if baseline_rho_3m is not None:
            record["baseline_rho_3m"] = round(baseline_rho_3m, 6)
        if "spearman_rho" in record:
            record["spearman_rho_6m"] = record["spearman_rho"]
        if "baseline_rho" in record:
            record["baseline_rho_6m"] = record["baseline_rho"]
        if improved_horizons:
            record["improved_horizons"] = improved_horizons

        active_ground_truth_id = os.environ.get("IMPROVE_SCORER_ACTIVE_GROUND_TRUTH_ID")
        if active_ground_truth_id:
            record["ground_truth_id"] = active_ground_truth_id
        else:
            try:
                from valueinvestor.scorer_improver.ground_truth import ground_truth_fingerprint

                record["ground_truth_id"] = ground_truth_fingerprint()
            except Exception:
                logger.debug("Could not attach ground-truth fingerprint", exc_info=True)

        # Compute deltas for 1m/3m where both new and baseline are present
        if "spearman_rho_1m" in record and "baseline_rho_1m" in record:
            record["delta_1m"] = round(record["spearman_rho_1m"] - record["baseline_rho_1m"], 6)
        if "spearman_rho_3m" in record and "baseline_rho_3m" in record:
            record["delta_3m"] = round(record["spearman_rho_3m"] - record["baseline_rho_3m"], 6)

        if extra:
            record.update(extra)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug("Logged experiment #%d (ρ=%.4f, kept=%s)", iteration, spearman_rho, kept)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_ground_truth(
        records: List[Dict[str, Any]],
        ground_truth_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter records to the same ground-truth build when requested."""
        if ground_truth_id is None:
            return records
        return [r for r in records if r.get("ground_truth_id") == ground_truth_id]

    def read_last_n(
        self,
        n: int = 10,
        current_lineage: bool = False,
        include_legacy: bool = False,
        ground_truth_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the last *n* experiment records.

        When *current_lineage* is true, only consider the active experiment
        lineage after the most recent kept-score reset from the unified log.

        When *include_legacy* is true, also include legacy horizon-specific
        records (merged chronologically) so the LLM can learn from previous
        optimization runs.
        """
        if include_legacy:
            unified = self.read_current_lineage(ground_truth_id=ground_truth_id) if current_lineage else self.read_all()
            unified = self._filter_ground_truth(unified, ground_truth_id)
            legacy = self.read_legacy_logs()
            if ground_truth_id is not None:
                legacy = []
            merged = sorted(unified + legacy, key=lambda r: r.get("timestamp", ""))
            if not merged:
                return []
            return merged[-n:] if len(merged) > n else merged

        records = self.read_current_lineage(ground_truth_id=ground_truth_id) if current_lineage else self.read_all()
        records = self._filter_ground_truth(records, ground_truth_id)
        if not records:
            return []
        return records[-n:] if len(records) > n else records

    def total_experiments(self) -> int:
        """Count total experiments logged."""
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().strip().split("\n") if line.strip())

    def best_rho(
        self,
        current_lineage: bool = False,
        ground_truth_id: Optional[str] = None,
    ) -> Optional[float]:
        """Return the best Spearman ρ (6m) achieved so far.

        When *current_lineage* is true, only consider the active experiment
        lineage after the most recent kept-score reset.
        """
        records = self.read_current_lineage(ground_truth_id=ground_truth_id) if current_lineage else self.read_all()
        records = self._filter_ground_truth(records, ground_truth_id)
        if not records:
            return None
        kept = [r for r in records if r.get("kept")]
        if not kept:
            return None
        return max(r["spearman_rho"] for r in kept)

    def best_rho_per_horizon(
        self,
        current_lineage: bool = False,
        ground_truth_id: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """Return the best Spearman ρ for each horizon."""
        records = self.read_current_lineage(ground_truth_id=ground_truth_id) if current_lineage else self.read_all()
        records = self._filter_ground_truth(records, ground_truth_id)
        best: dict[str, Optional[float]] = {"1m": None, "3m": None, "6m": None}
        kept = [r for r in records if r.get("kept")]
        if not kept:
            return best
        for h in _HORIZONS:
            col = f"spearman_rho_{h}" if h != "6m" else "spearman_rho"
            vals = [r[col] for r in kept if col in r and r[col] is not None]
            if vals:
                best[h] = max(vals)
        return best

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all experiment records from the unified log."""
        if not self.path.exists():
            return []

        records = []
        for line in self.path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def read_current_lineage(self, ground_truth_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read the active lineage after the most recent kept-score reset.

        A reset is inferred when a kept experiment has a lower ρ than the
        previous kept experiment, which indicates the trainer was intentionally
        rolled back to an earlier scorer and resumed from there.
        """
        records = self.read_all()
        records = self._filter_ground_truth(records, ground_truth_id)
        if not records:
            return []

        last_reset_index = 0
        previous_kept_rho: Optional[float] = None

        for idx, record in enumerate(records):
            if not record.get("kept"):
                continue

            rho = record.get("spearman_rho")
            if rho is None:
                continue

            rho = float(rho)
            if previous_kept_rho is not None and (previous_kept_rho - rho) > _MIN_RESET_THRESHOLD:
                last_reset_index = idx
            previous_kept_rho = rho

        return records[last_reset_index:]

    # ------------------------------------------------------------------
    # Legacy log support
    # ------------------------------------------------------------------

    @classmethod
    def read_legacy_logs(cls) -> List[Dict[str, Any]]:
        """Read all legacy horizon-specific experiment logs.

        Returns a merged, chronologically-sorted list of records with a
        ``source_horizon`` key added so callers can label experiments.
        """
        all_records: list[dict[str, Any]] = []
        for horizon, path in _LEGACY_LOG_FILES.items():
            if not path.exists():
                logger.debug("Legacy log not found (skipping): %s", path)
                continue
            try:
                for line in path.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        record = json.loads(line)
                        record["source_horizon"] = horizon
                        all_records.append(record)
            except Exception:
                logger.warning("Failed to read legacy log: %s", path)
        all_records.sort(key=lambda r: r.get("timestamp", ""))
        return all_records
