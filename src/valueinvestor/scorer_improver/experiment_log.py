"""JSONL experiment logger for the scorer improvement loop.

Each experiment is logged as a single JSON line in
``data/trainer/experiments.jsonl``.  The agent reads the last N experiments
to provide context for the LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from valueinvestor.scorer_improver.data_prep import TRAINER_DIR

logger = logging.getLogger(__name__)

EXPERIMENT_LOG_FILE = TRAINER_DIR / "experiments.jsonl"


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
    ) -> None:
        """Append one experiment record."""
        record = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spearman_rho": round(spearman_rho, 6),
            "baseline_rho": round(baseline_rho, 6),
            "delta": round(spearman_rho - baseline_rho, 6),
            "kept": kept,
            "description": description,
            "diff_summary": diff_summary[:500],
        }
        if extra:
            record.update(extra)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug("Logged experiment #%d (ρ=%.4f, kept=%s)", iteration, spearman_rho, kept)

    def read_last_n(self, n: int = 10) -> List[Dict[str, Any]]:
        """Read the last *n* experiment records."""
        if not self.path.exists():
            return []

        lines = self.path.read_text(encoding="utf-8").strip().split("\n")
        lines = [l for l in lines if l.strip()]
        recent = lines[-n:] if len(lines) > n else lines

        records = []
        for line in recent:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def total_experiments(self) -> int:
        """Count total experiments logged."""
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().strip().split("\n") if line.strip())

    def best_rho(self) -> Optional[float]:
        """Return the best Spearman ρ achieved so far."""
        records = self.read_all()
        if not records:
            return None
        kept = [r for r in records if r.get("kept")]
        if not kept:
            return None
        return max(r["spearman_rho"] for r in kept)

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all experiment records."""
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
