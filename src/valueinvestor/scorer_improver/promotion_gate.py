"""Strict holdout gate for promoting trained scorer artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from valueinvestor.scorer_improver.data_prep import TRAINER_DIR
from valueinvestor.scorer_improver.ground_truth import FORWARD_HORIZON_6M_DAYS

HORIZONS = ("1m", "3m", "6m")
PROMOTION_GATE_SCHEMA_VERSION = "promotion-gate-v1"
DEFAULT_HOLDOUT_MONTHS = 24
DEFAULT_EMBARGO_DAYS = FORWARD_HORIZON_6M_DAYS + 15
DEFAULT_MIN_6M_DELTA = 0.001
DEFAULT_MAX_HORIZON_DEGRADATION = 0.001
DEFAULT_MIN_WEIGHTED_UTILITY = 0.0
DEFAULT_MIN_GATE_SNAPSHOTS = 4
DEFAULT_MIN_TRAIN_SNAPSHOTS = 4
DEFAULT_MIN_REGIME_6M_WIN_RATE = 0.50
DEFAULT_MAX_REGIME_6M_DEGRADATION = 0.05
DEFAULT_MIN_RECENT_PRIMARY_RHO = 0.0
PROMOTION_GATE_LEDGER_PATH = TRAINER_DIR / "promotion_gate_ledger.jsonl"

_HORIZON_WEIGHTS = {"1m": 0.25, "3m": 0.35, "6m": 0.40}


def _gate_horizons(primary_horizon: str) -> tuple[str, ...]:
    if primary_horizon in HORIZONS:
        return HORIZONS
    return (primary_horizon, *HORIZONS)


def _gate_weights(primary_horizon: str, horizons: tuple[str, ...]) -> dict[str, float]:
    if primary_horizon == "6m":
        return dict(_HORIZON_WEIGHTS)
    secondary_total = sum(_HORIZON_WEIGHTS.get(horizon, 0.0) for horizon in HORIZONS)
    weights = {primary_horizon: 0.50}
    for horizon in horizons:
        if horizon == primary_horizon:
            continue
        base_weight = _HORIZON_WEIGHTS.get(horizon, 0.0)
        weights[horizon] = 0.50 * base_weight / secondary_total if secondary_total > 0 else 0.0
    return weights


@dataclass(frozen=True)
class HoldoutGateConfig:
    """Configuration for the untouched holdout promotion gate."""

    holdout_months: int = DEFAULT_HOLDOUT_MONTHS
    embargo_days: int = DEFAULT_EMBARGO_DAYS
    min_6m_delta: float = DEFAULT_MIN_6M_DELTA
    max_horizon_degradation: float = DEFAULT_MAX_HORIZON_DEGRADATION
    min_weighted_utility: float = DEFAULT_MIN_WEIGHTED_UTILITY
    min_gate_snapshots: int = DEFAULT_MIN_GATE_SNAPSHOTS
    min_train_snapshots: int = DEFAULT_MIN_TRAIN_SNAPSHOTS
    min_primary_rho: Optional[float] = None
    min_regime_6m_win_rate: float = DEFAULT_MIN_REGIME_6M_WIN_RATE
    max_regime_6m_degradation: float = DEFAULT_MAX_REGIME_6M_DEGRADATION
    min_recent_primary_rho: float = DEFAULT_MIN_RECENT_PRIMARY_RHO
    require_6m_top20_excess_non_degradation: bool = True


@dataclass(frozen=True)
class HoldoutSplit:
    """Train/gate split with an embargo between the two windows."""

    train: pd.DataFrame
    embargo: pd.DataFrame
    gate: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PromotionGateResult:
    """Serializable result from evaluating a candidate against the holdout gate."""

    accepted: bool
    reason: str
    deltas: dict[str, float]
    weighted_utility: float
    candidate_metrics: Mapping[str, Mapping[str, float]]
    incumbent_metrics: Mapping[str, Mapping[str, float]]
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    regime_diagnostics: Optional[Mapping[str, Any]] = None

    def to_record(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _snapshot_series(df: pd.DataFrame) -> pd.Series:
    if "snapshot_date" not in df.columns:
        raise ValueError("promotion gate requires a snapshot_date column")
    dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("promotion gate found invalid snapshot_date values")
    return dates.dt.normalize()


def _date_str(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _source_fingerprint(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def make_holdout_split(
    df: pd.DataFrame,
    *,
    config: Optional[HoldoutGateConfig] = None,
    source_path: Optional[Path] = None,
) -> HoldoutSplit:
    """Return train, embargo, and untouched-gate frames split by snapshot date."""
    config = config or HoldoutGateConfig()
    if df.empty:
        raise ValueError("promotion gate cannot split an empty snapshot frame")

    dates = _snapshot_series(df)
    unique_dates = pd.Index(sorted(dates.unique()))
    if len(unique_dates) < config.min_gate_snapshots + config.min_train_snapshots:
        raise ValueError(
            "promotion gate needs at least "
            f"{config.min_gate_snapshots + config.min_train_snapshots} snapshot dates; "
            f"found {len(unique_dates)}"
        )

    max_date = pd.Timestamp(unique_dates[-1])
    target_gate_start = max_date - pd.DateOffset(months=config.holdout_months)
    gate_dates = unique_dates[unique_dates >= target_gate_start]
    if len(gate_dates) < config.min_gate_snapshots:
        gate_dates = unique_dates[-config.min_gate_snapshots:]
    gate_start = pd.Timestamp(gate_dates[0])
    train_end_exclusive = gate_start - timedelta(days=config.embargo_days)

    train_mask = dates < train_end_exclusive
    gate_mask = dates >= gate_start
    embargo_mask = ~(train_mask | gate_mask)

    train = df.loc[train_mask].copy()
    embargo = df.loc[embargo_mask].copy()
    gate = df.loc[gate_mask].copy()

    train_snapshot_count = int(_snapshot_series(train).nunique()) if not train.empty else 0
    gate_snapshot_count = int(_snapshot_series(gate).nunique()) if not gate.empty else 0
    if train_snapshot_count < config.min_train_snapshots:
        raise ValueError(
            f"promotion gate train window has {train_snapshot_count} snapshot dates; "
            f"need at least {config.min_train_snapshots}"
        )
    if gate_snapshot_count < config.min_gate_snapshots:
        raise ValueError(
            f"promotion gate holdout has {gate_snapshot_count} snapshot dates; "
            f"need at least {config.min_gate_snapshots}"
        )

    manifest = {
        "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
        "source_path": str(source_path) if source_path is not None else "",
        "source_fingerprint": _source_fingerprint(source_path),
        "snapshot_min": _date_str(pd.Timestamp(unique_dates[0])),
        "snapshot_max": _date_str(max_date),
        "train_start": _date_str(pd.Timestamp(_snapshot_series(train).min())),
        "train_end_exclusive": _date_str(train_end_exclusive),
        "gate_start": _date_str(gate_start),
        "gate_end": _date_str(pd.Timestamp(_snapshot_series(gate).max())),
        "holdout_months": int(config.holdout_months),
        "embargo_days": int(config.embargo_days),
        "train_rows": int(len(train)),
        "embargo_rows": int(len(embargo)),
        "gate_rows": int(len(gate)),
        "train_snapshots": train_snapshot_count,
        "embargo_snapshots": int(_snapshot_series(embargo).nunique()) if not embargo.empty else 0,
        "gate_snapshots": gate_snapshot_count,
    }
    return HoldoutSplit(train=train, embargo=embargo, gate=gate, manifest=manifest)


def _rho(metrics: Mapping[str, Mapping[str, float]], horizon: str) -> float:
    horizon_metrics = metrics.get(horizon, {})
    try:
        return float(horizon_metrics.get("spearman_rho", 0.0))
    except (TypeError, ValueError):
        return float("nan")


def _metric_delta(
    candidate_metrics: Mapping[str, Mapping[str, float]],
    incumbent_metrics: Mapping[str, Mapping[str, float]],
    horizon: str,
    metric: str,
) -> float:
    candidate = candidate_metrics.get(horizon, {}).get(metric, 0.0)
    incumbent = incumbent_metrics.get(horizon, {}).get(metric, 0.0)
    try:
        return float(candidate) - float(incumbent)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_promotion_gate(
    *,
    candidate_metrics: Mapping[str, Mapping[str, float]],
    incumbent_metrics: Mapping[str, Mapping[str, float]],
    manifest: Mapping[str, Any],
    config: Optional[HoldoutGateConfig] = None,
    regime_diagnostics: Optional[Mapping[str, Any]] = None,
    primary_horizon: str = "6m",
) -> PromotionGateResult:
    """Return whether candidate metrics clear the untouched holdout gate."""
    config = config or HoldoutGateConfig()
    horizons = _gate_horizons(primary_horizon)
    weights = _gate_weights(primary_horizon, horizons)
    deltas = {
        horizon: _rho(candidate_metrics, horizon) - _rho(incumbent_metrics, horizon)
        for horizon in horizons
    }
    weighted_utility = sum(weights[horizon] * deltas[horizon] for horizon in horizons)

    reason = "accepted"
    if any(not math.isfinite(_rho(candidate_metrics, horizon)) for horizon in horizons):
        reason = "non-finite candidate rho"
    elif any(not math.isfinite(delta) for delta in deltas.values()):
        reason = "non-finite gate delta"
    elif int(manifest.get("gate_snapshots", 0)) < config.min_gate_snapshots:
        reason = "insufficient gate snapshots"
    elif (
        config.min_primary_rho is not None
        and _rho(candidate_metrics, primary_horizon) < config.min_primary_rho
    ):
        reason = f"{primary_horizon} rho below absolute gate"
    elif deltas[primary_horizon] < config.min_6m_delta:
        reason = f"{primary_horizon} delta below gate"
    elif any(delta < -config.max_horizon_degradation for delta in deltas.values()):
        reason = "horizon degradation"
    elif weighted_utility <= config.min_weighted_utility:
        reason = "weighted utility below gate"
    elif (
        config.require_6m_top20_excess_non_degradation
        and
        _metric_delta(candidate_metrics, incumbent_metrics, primary_horizon, "hit_rate_top20") < 0
        and _metric_delta(candidate_metrics, incumbent_metrics, primary_horizon, "mean_excess_return") < 0
    ):
        reason = f"{primary_horizon} top20 and excess return degraded"
    elif regime_diagnostics:
        year_records = regime_diagnostics.get("year", [])
        recent_records = []
        if isinstance(year_records, list):
            for record in year_records:
                if not isinstance(record, Mapping):
                    continue
                try:
                    recent_records.append(
                        (int(str(record.get("label"))), float(record["candidate_primary"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        if (
            recent_records
            and max(recent_records)[1] < config.min_recent_primary_rho
        ):
            reason = f"recent-year {primary_horizon} rho below gate"

        for dimension, records in regime_diagnostics.items():
            if reason != "accepted":
                break
            if not isinstance(records, list) or len(records) < 2:
                continue
            primary_deltas = [
                float(record.get("deltas", {}).get(primary_horizon, 0.0))
                for record in records
                if isinstance(record, Mapping)
            ]
            if not primary_deltas:
                continue
            win_rate = sum(delta >= 0.0 for delta in primary_deltas) / len(primary_deltas)
            if min(primary_deltas) < -config.max_regime_6m_degradation:
                reason = f"regime {dimension} {primary_horizon} degradation"
                break
            if (
                win_rate < config.min_regime_6m_win_rate
                and sum(primary_deltas) / len(primary_deltas) <= 0.0
            ):
                reason = f"regime {dimension} win rate below gate"
                break

    return PromotionGateResult(
        accepted=reason == "accepted",
        reason=reason,
        deltas={horizon: round(float(delta), 6) for horizon, delta in deltas.items()},
        weighted_utility=round(float(weighted_utility), 6),
        candidate_metrics=candidate_metrics,
        incumbent_metrics=incumbent_metrics,
        manifest=manifest,
        config={**asdict(config), "primary_horizon": primary_horizon},
        regime_diagnostics=regime_diagnostics,
    )


def append_gate_ledger(
    result: PromotionGateResult,
    *,
    ledger_path: Path = PROMOTION_GATE_LEDGER_PATH,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append a promotion gate decision to the JSONL ledger."""
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **result.to_record(),
    }
    if extra:
        record.update(_json_safe(dict(extra)))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
