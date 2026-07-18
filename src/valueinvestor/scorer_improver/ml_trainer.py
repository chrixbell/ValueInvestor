"""Train the local ML ranker from scorer-improver snapshots."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from valueinvestor.config import ScreeningConfig
from valueinvestor.scorer_improver.ground_truth import (
    CURRENT_GROUND_TRUTH_FILE,
    FINANCIAL_ASOF_LAG_DAYS,
    FINANCIAL_FEATURE_COLUMNS,
    FORWARD_HORIZON_1W_DAYS,
    FORWARD_HORIZON_1M_DAYS,
    FORWARD_HORIZON_3M_DAYS,
    FORWARD_HORIZON_6M_DAYS,
    VALUATION_ASOF_LAG_DAYS,
    VALUATION_FEATURE_COLUMNS,
    _compute_forward_returns,
    align_financial_history_to_live_periods,
    annualize_financial_feature_frame,
    asof_feature_frame,
    ensure_current_ground_truth_ready,
)
from valueinvestor.scorer_improver.promotion_gate import (
    HoldoutGateConfig,
    append_gate_ledger,
    evaluate_promotion_gate,
    make_holdout_split,
)
from valueinvestor.scorer_improver.data_prep import TRAINER_DIR
from valueinvestor.screener.ml_ranker import (
    BASE_FIELDS,
    CROSS_SECTIONAL_INTERACTION_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    DEFAULT_MODEL_PATH,
    ONE_WEEK_MODEL_PATH,
    HORIZONS as FEATURE_HORIZONS,
    INVERSE_FIELDS,
    INTERACTION_FEATURES,
    MODEL_SCHEMA_VERSION,
    POLYNOMIAL_FEATURES,
    RATIO_FEATURES,
    SHORT_HORIZON_FEATURES,
    _application_gross_profitability_de_crowding_config,
    _application_gross_profitability_de_crowding_predictions,
    _model_from_payload,
    expected_feature_names,
    feature_matrix_from_values,
    predict_lightgbm_model,
)

logger = logging.getLogger(__name__)
_MLX_RIDGE_DIAGNOSTIC_LOGGED = False
_MLX_ADAM_RIDGE_DIAGNOSTIC_LOGGED = False
_TICKER_BUCKET_CACHE: dict[str, tuple[str, str, str]] = {}

DAILY_START_DATE = date(2016, 5, 25)
DAILY_END_DATE = date(2025, 7, 7)
ML_SNAPSHOTS_FILE = TRAINER_DIR / "ml_training_daily_snapshots.parquet"
DEFAULT_RIDGE_LAMBDA = 100.0
DEFAULT_TARGET_IMPROVEMENT = 0.30
DEFAULT_MAX_TRAINING_ROWS = 500_000
DEFAULT_WALK_FORWARD_FOLDS = 3
DEFAULT_WALK_FORWARD_VALIDATION_MONTHS = 6
DEFAULT_WALK_FORWARD_MAX_ROWS = 500_000
DEFAULT_WALK_FORWARD_MIN_6M_DELTA = 0.001
DEFAULT_WALK_FORWARD_MAX_HORIZON_DEGRADATION = 0.10
DEFAULT_FULL_EVAL_CANDIDATE_LIMIT = 1
DEFAULT_ONE_WEEK_FULL_EVAL_CANDIDATE_LIMIT = 8
DEFAULT_SIX_MONTH_FULL_EVAL_CANDIDATE_LIMIT = 12
STRICT_WALK_FORWARD_CANDIDATE_LIMIT = 24
ONE_WEEK_CANDIDATE_LAMBDAS = (
    0.3,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
)
SIX_MONTH_CANDIDATE_LAMBDAS = (
    0.3,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
)
SIX_MONTH_SAMPLE_SEEDS = (0, 29, 73)
SIX_MONTH_RECENCY_HALF_LIVES_DAYS = (
    180.0,
    270.0,
    365.0,
    540.0,
    730.0,
    1_095.0,
    1_460.0,
    1_825.0,
    2_190.0,
)
SIX_MONTH_REFIT_FULL_EVAL_CANDIDATES = True
SIX_MONTH_FULL_EVAL_REFIT_MAX_ROWS = 1_000_000
SIX_MONTH_GATE_PROBE_MAX_ROWS = 120_000
SIX_MONTH_GATE_PROBE_CANDIDATE_LIMIT = 96
SIX_MONTH_ENSEMBLE_POOL_LIMIT = 5
SIX_MONTH_ENSEMBLE_CANDIDATE_LIMIT = 3
SIX_MONTH_ENSEMBLE_PAIR_WEIGHTS = (
    (0.85, 0.15),
    (0.75, 0.25),
    (0.65, 0.35),
    (0.5, 0.5),
    (0.35, 0.65),
    (0.25, 0.75),
    (0.15, 0.85),
)
SIX_MONTH_PAYLOAD_MEMBER_ENSEMBLE_WEIGHTS = (
    (0.65, 0.35),
    (0.5, 0.5),
    (0.35, 0.65),
)
SIX_MONTH_PAYLOAD_MEMBER_ENSEMBLE_LIMIT = 3
SIX_MONTH_TEMPORAL_ANCHOR_CANDIDATE_LIMIT = 4
SIX_MONTH_ANCHOR_TEACHER_WEIGHTS = (0.65, 0.55, 0.45, 0.35, 0.25)
SIX_MONTH_ANCHOR_TEACHER_PRUNE_GAP = 0.003
SIX_MONTH_ANCHOR_TEACHER_BASE_TARGETS = (
    "target_rank_6m",
    "target_rank_weighted_703",
    "target_rank_weighted_8515",
)
SIX_MONTH_ANCHOR_TEACHER_ANCHOR_LIMIT = 1
SIX_MONTH_MARKET_BLEND_WEIGHTS = (
    {"ashare": 0.0025, "hk": 0.00},
    {"ashare": 0.005, "hk": 0.00},
    {"ashare": 0.0075, "hk": 0.00},
    {"ashare": 0.010, "hk": 0.00},
    {"ashare": 0.020, "hk": 0.00},
    {"ashare": 0.025, "hk": 0.00},
    {"ashare": 0.030, "hk": 0.00},
    {"ashare": 0.035, "hk": 0.00},
    {"ashare": 0.040, "hk": 0.00},
    {"ashare": 0.045, "hk": 0.00},
    {"ashare": 0.050, "hk": 0.00},
    {"ashare": 0.060, "hk": 0.00},
    {"ashare": 0.075, "hk": 0.00},
    {"ashare": 0.100, "hk": 0.00},
    {"ashare": 0.150, "hk": 0.00},
    {"ashare": 0.200, "hk": 0.00},
)
EVAL_HORIZONS = ("1w", *FEATURE_HORIZONS)
BLEND_WEIGHTS = (1.0, 0.75, 0.5, 0.25)
ONE_WEEK_BLEND_WEIGHTS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
SIX_MONTH_BLEND_WEIGHTS = (
    1.0,
    0.75,
    0.5,
    0.35,
    0.25,
    0.2,
    0.175,
    0.15,
    0.125,
    0.1,
    0.075,
    0.05,
    0.04,
    0.03,
    0.025,
    0.02,
    0.015,
    0.01,
)
SIX_MONTH_QUALITY_RESIDUAL_WEIGHTS = (
    1.0,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
)
SIX_MONTH_APPLICATION_RESIDUAL_SIGNALS = (
    ("quality_de_crowding", "quality_score", -1.0),
)
SIX_MONTH_APPLICATION_FACTOR_SIGNALS = (
    "model",
    "quality_de_crowding",
    "gross_profitability_de_crowding",
    "roe_de_crowding",
    "roa_de_crowding",
    "book_yield",
    "sales_yield",
    "liability_yield",
    "momentum_63d",
    "low_current_ratio",
)
SIX_MONTH_APPLICATION_FACTOR_TEMPLATES = (
    ("equal", (0.10, 0.15, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.15)),
    ("balanced", (0.10, 0.18, 0.0, 0.0, 0.0, 0.135, 0.09, 0.135, 0.225, 0.135)),
    ("value_momentum", (0.10, 0.135, 0.0, 0.0, 0.0, 0.18, 0.045, 0.135, 0.27, 0.135)),
    ("quality_de_crowding", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (
        "gross_profitability_de_crowding",
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    ("roe_de_crowding", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("roa_de_crowding", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("roe_roa_de_crowding", (0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (
        "profitability_de_crowding",
        (0.0, 1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "profitability_de_crowding_roa",
        (0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "model_profitability_de_crowding",
        (0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)
SIX_MONTH_APPLICATION_FACTOR_MIN_RHO_TOLERANCE = 0.01
SIX_MONTH_APPLICATION_FACTOR_MIN_MEAN_RHO_IMPROVEMENT = 0.005
SIX_MONTH_GATE_BLEND_LIMIT = 4
SIX_MONTH_GATE_MARKET_BLEND_LIMIT = 6
SIX_MONTH_GATE_MARKET_PROBE_TOTALS = (
    0.005,
    0.010,
    0.015,
    0.020,
    0.0225,
    0.025,
    0.030,
    0.040,
    0.050,
    0.100,
    0.250,
    0.500,
)
SIX_MONTH_RECENT_MARKET_ROUTE_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.1)
SIX_MONTH_RECENT_SEGMENT_ROUTE_WEIGHTS = (1.0, 0.75, 0.5)
SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_STARTS = 4
SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS = 8
SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES = 8
SIX_MONTH_RECENT_SEGMENT_ROUTE_GROUPINGS = (
    ("market", "cap_bucket"),
    ("market", "board_bucket"),
    ("market", "listing_bucket"),
    ("market", "cap_bucket", "board_bucket"),
    ("market", "cap_bucket", "listing_bucket"),
    ("market", "cap_bucket", "board_bucket", "listing_bucket"),
)
SIX_MONTH_GATE_SEGMENT_ROUTE_GROUPINGS = (
    ("market", "cap_bucket"),
    ("market", "board_bucket"),
    ("market", "cap_bucket", "board_bucket"),
)
SIX_MONTH_ANCHOR_BLEND_WEIGHTS = (
    1.0,
    0.75,
    0.6,
    0.5,
    0.4,
    0.3,
    0.25,
    0.2,
    0.15,
    0.1,
    0.075,
    0.05,
    0.04,
    0.03,
    0.025,
    0.02,
    0.015,
    0.01,
)
SIX_MONTH_GATE_ANCHOR_BLEND_LIMIT = 3
SIX_MONTH_GATE_ANCHOR_MODEL_LIMIT = 3
FEATURE_SET_ALIASES = {
    "core": "core",
    "short_horizon": "short_horizon",
    "expanded": "expanded",
    "poly": "poly",
    "cross_sectional": "cross_sectional",
    "cross_sectional_interactions": "cross_sectional_interactions",
    "cs_interactions": "cross_sectional_interactions",
    "cs": "cross_sectional",
}
PRIOR_STRATEGY_ALIASES = {
    "no_ticker_priors": "no_ticker_priors",
    "ticker_priors": "ticker_priors",
    "rolling_ticker_priors": "rolling_ticker_priors",
}
TARGET_ALIASES = {
    "target_rank_1w": "target_rank_1w",
    "target_rank_1w_market": "target_rank_1w_market",
    "target_rank_1m_market": "target_rank_1m_market",
    "rank_1m_market": "target_rank_1m_market",
    "target_rank_3m_market": "target_rank_3m_market",
    "rank_3m_market": "target_rank_3m_market",
    "target_rank_1m": "target_rank_1m",
    "target_rank_3m": "target_rank_3m",
    "target_rank_6m": "target_rank_6m",
    "rank_6m": "target_rank_6m",
    "target_rank_6m_market": "target_rank_6m_market",
    "rank_6m_market": "target_rank_6m_market",
    "target_rank_6m_soft": "target_rank_6m_soft",
    "rank_6m_soft": "target_rank_6m_soft",
    "target_rank_6m_extreme": "target_rank_6m_extreme",
    "rank_6m_extreme": "target_rank_6m_extreme",
    "target_rank_weighted": "target_rank_weighted",
    "target_rank_weighted_703": "target_rank_weighted_703",
    "weighted_703": "target_rank_weighted_703",
    "target_rank_weighted_703_market": "target_rank_weighted_703_market",
    "weighted_703_market": "target_rank_weighted_703_market",
    "target_rank_weighted_802": "target_rank_weighted_802",
    "weighted_802": "target_rank_weighted_802",
    "target_rank_weighted_802_market": "target_rank_weighted_802_market",
    "weighted_802_market": "target_rank_weighted_802_market",
    "target_rank_weighted_901": "target_rank_weighted_901",
    "weighted_901": "target_rank_weighted_901",
    "target_rank_weighted_901_market": "target_rank_weighted_901_market",
    "weighted_901_market": "target_rank_weighted_901_market",
    "target_rank_weighted_8515": "target_rank_weighted_8515",
    "weighted_8515": "target_rank_weighted_8515",
    "target_rank_weighted_8515_market": "target_rank_weighted_8515_market",
    "weighted_8515_market": "target_rank_weighted_8515_market",
    "target_rank_weighted_7525": "target_rank_weighted_7525",
    "weighted_7525": "target_rank_weighted_7525",
    "target_rank_weighted_631": "target_rank_weighted_631",
    "weighted_631": "target_rank_weighted_631",
    "target_rank_mean": "target_rank_mean",
    "mean": "target_rank_mean",
}
SNAPSHOT_MANIFEST_SCHEMA_VERSION = "ml-snapshot-manifest-v1"
FEATURE_POLICY_VERSION = "pit-application-universe-v6-annual-financials"
PRIOR_COLUMN_PREFIX = "_prior_"
SCORER_SOURCE_PATH = Path("src/valueinvestor/screener/scorer.py")
GROUND_TRUTH_SOURCE_PATH = Path("src/valueinvestor/scorer_improver/ground_truth.py")
HORIZON_DAYS = {
    "1w": FORWARD_HORIZON_1W_DAYS,
    "1m": FORWARD_HORIZON_1M_DAYS,
    "3m": FORWARD_HORIZON_3M_DAYS,
    "6m": FORWARD_HORIZON_6M_DAYS,
}
DEFAULT_PRIOR_EMBARGO_DAYS = 15
APPLICATION_MIN_ROWS_PER_SNAPSHOT = 20
APPLICATION_MIN_DAILY_SNAPSHOTS = 252


def _blend_weights_for_horizon(primary_horizon: str) -> tuple[float, ...]:
    if primary_horizon == "1w":
        return ONE_WEEK_BLEND_WEIGHTS
    if primary_horizon == "6m":
        return SIX_MONTH_BLEND_WEIGHTS
    return BLEND_WEIGHTS


def _six_month_full_eval_refit_max_rows() -> int:
    raw = os.environ.get(
        "VALUEINVESTOR_ML_6M_FULL_EVAL_REFIT_MAX_ROWS",
        str(SIX_MONTH_FULL_EVAL_REFIT_MAX_ROWS),
    )
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_FULL_EVAL_REFIT_MAX_ROWS=%r; using %d",
            raw,
            SIX_MONTH_FULL_EVAL_REFIT_MAX_ROWS,
        )
        return SIX_MONTH_FULL_EVAL_REFIT_MAX_ROWS


def _six_month_gate_probe_max_rows() -> int:
    raw = os.environ.get(
        "VALUEINVESTOR_ML_6M_GATE_PROBE_MAX_ROWS",
        str(SIX_MONTH_GATE_PROBE_MAX_ROWS),
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_GATE_PROBE_MAX_ROWS=%r; using %d",
            raw,
            SIX_MONTH_GATE_PROBE_MAX_ROWS,
        )
        return SIX_MONTH_GATE_PROBE_MAX_ROWS


def _six_month_gate_probe_candidate_limit() -> int:
    raw = os.environ.get(
        "VALUEINVESTOR_ML_6M_GATE_PROBE_CANDIDATE_LIMIT",
        str(SIX_MONTH_GATE_PROBE_CANDIDATE_LIMIT),
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_GATE_PROBE_CANDIDATE_LIMIT=%r; using %d",
            raw,
            SIX_MONTH_GATE_PROBE_CANDIDATE_LIMIT,
        )
        return SIX_MONTH_GATE_PROBE_CANDIDATE_LIMIT


def _six_month_market_train_floor_tolerance() -> float:
    raw = os.environ.get("VALUEINVESTOR_ML_6M_MARKET_TRAIN_FLOOR_TOLERANCE", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_MARKET_TRAIN_FLOOR_TOLERANCE=%r; using 0",
            raw,
        )
        return 0.0


def _sample_train_floor_tolerance(primary_horizon: str) -> float:
    env_name = f"VALUEINVESTOR_ML_{primary_horizon.upper()}_SAMPLE_TRAIN_FLOOR_TOLERANCE"
    default = "0.002" if primary_horizon == "6m" else "0"
    raw = os.environ.get(
        env_name,
        os.environ.get("VALUEINVESTOR_ML_SAMPLE_TRAIN_FLOOR_TOLERANCE", default),
    )
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", env_name, raw, default)
        return float(default)


def _six_month_gate_market_blend_limit() -> int:
    raw = os.environ.get(
        "VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT",
        str(SIX_MONTH_GATE_MARKET_BLEND_LIMIT),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT=%r; using %d",
            raw,
            SIX_MONTH_GATE_MARKET_BLEND_LIMIT,
        )
        return SIX_MONTH_GATE_MARKET_BLEND_LIMIT


def _six_month_gate_route_raw_degradation_limit() -> float:
    raw = os.environ.get("VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT", "0.005")
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT=%r; using 0.005",
            raw,
        )
        return 0.005


def _positive_int_env(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", env_name, raw, default)
        return default


def _six_month_recent_segment_route_weights() -> tuple[float, ...]:
    raw = os.environ.get("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS")
    if raw is None or raw.strip() == "":
        return SIX_MONTH_RECENT_SEGMENT_ROUTE_WEIGHTS
    weights: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            weight = float(token)
        except ValueError:
            logger.warning(
                "Invalid VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS entry=%r; "
                "using defaults",
                token,
            )
            return SIX_MONTH_RECENT_SEGMENT_ROUTE_WEIGHTS
        if not np.isfinite(weight) or weight <= 0.0 or weight > 1.0:
            logger.warning(
                "Invalid VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS entry=%r; "
                "using defaults",
                token,
            )
            return SIX_MONTH_RECENT_SEGMENT_ROUTE_WEIGHTS
        weights.append(weight)
    return tuple(dict.fromkeys(weights)) or SIX_MONTH_RECENT_SEGMENT_ROUTE_WEIGHTS


def _six_month_recent_segment_route_max_starts() -> int:
    return _positive_int_env(
        "VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_STARTS",
        SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_STARTS,
    )


def _six_month_recent_segment_route_max_segments() -> int:
    return _positive_int_env(
        "VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS",
        SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS,
    )


def _six_month_recent_segment_route_max_candidates() -> int:
    return _positive_int_env(
        "VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES",
        SIX_MONTH_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES,
    )


def _six_month_recency_half_lives_days() -> tuple[float, ...]:
    raw = os.environ.get("VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS")
    if raw is None or raw.strip() == "":
        return SIX_MONTH_RECENCY_HALF_LIVES_DAYS
    half_lives: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            half_life = float(token)
        except ValueError:
            logger.warning(
                "Invalid VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS entry=%r; "
                "using defaults",
                token,
            )
            return SIX_MONTH_RECENCY_HALF_LIVES_DAYS
        if not np.isfinite(half_life) or half_life <= 0.0:
            logger.warning(
                "Invalid VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS entry=%r; "
                "using defaults",
                token,
            )
            return SIX_MONTH_RECENCY_HALF_LIVES_DAYS
        half_lives.append(half_life)
    return tuple(dict.fromkeys(half_lives)) or SIX_MONTH_RECENCY_HALF_LIVES_DAYS


def _six_month_sample_seeds() -> tuple[int, ...]:
    raw = os.environ.get("VALUEINVESTOR_ML_6M_SAMPLE_SEEDS")
    if raw is None or raw.strip() == "":
        return SIX_MONTH_SAMPLE_SEEDS
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seed = int(token)
        except ValueError:
            logger.warning(
                "Invalid VALUEINVESTOR_ML_6M_SAMPLE_SEEDS entry=%r; using defaults",
                token,
            )
            return SIX_MONTH_SAMPLE_SEEDS
        seeds.append(seed)
    return tuple(dict.fromkeys(seeds)) or SIX_MONTH_SAMPLE_SEEDS


def _recent_half_life_for_model_kind(model_kind: str) -> Optional[float]:
    if model_kind == "market_ridge_recent":
        return 365.0
    if model_kind == "segment_ridge_recent":
        return 365.0
    for prefix in ("market_ridge_recent_", "segment_ridge_recent_"):
        if not model_kind.startswith(prefix):
            continue
        try:
            half_life = float(model_kind.removeprefix(prefix))
        except ValueError:
            return None
        return half_life if half_life > 0 else None
    return None


def _fit_model_kind_for_candidate(model_kind: str) -> str:
    if _recent_half_life_for_model_kind(model_kind):
        return "segment_ridge" if model_kind.startswith("segment_ridge_recent") else "market_ridge"
    return model_kind


def _is_recent_market_ridge(model_kind: object) -> bool:
    return (
        isinstance(model_kind, str)
        and _fit_model_kind_for_candidate(model_kind) == "market_ridge"
        and _recent_half_life_for_model_kind(model_kind) is not None
    )


def _is_recent_segment_ridge(model_kind: object) -> bool:
    return (
        isinstance(model_kind, str)
        and _fit_model_kind_for_candidate(model_kind) == "segment_ridge"
        and _recent_half_life_for_model_kind(model_kind) is not None
    )


def _snapshot_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _normalize_candidate_choices(
    choices: Optional[Sequence[str]],
    *,
    aliases: Mapping[str, str],
    option_name: str,
) -> Optional[tuple[str, ...]]:
    if choices is None:
        return None

    normalized: list[str] = []
    saw_auto = False
    for choice in choices:
        for raw_token in str(choice).split(","):
            token = raw_token.strip().lower().replace("-", "_")
            if not token:
                continue
            if token in {"auto", "all"}:
                saw_auto = True
                continue
            canonical = aliases.get(token)
            if canonical is None:
                allowed = ", ".join(sorted(aliases))
                raise ValueError(f"{option_name} must be auto/all or one of: {allowed}")
            if canonical not in normalized:
                normalized.append(canonical)

    if saw_auto and normalized:
        raise ValueError(f"{option_name} cannot mix auto/all with explicit choices")
    if saw_auto or not normalized:
        return None
    return tuple(normalized)


def _normalize_candidate_targets(
    choices: Optional[Sequence[str]],
    *,
    allowed_targets: Sequence[str],
) -> Optional[tuple[str, ...]]:
    normalized = _normalize_candidate_choices(
        choices,
        aliases=TARGET_ALIASES,
        option_name="candidate_targets",
    )
    if normalized is None:
        return None
    allowed = set(allowed_targets)
    filtered = tuple(target for target in normalized if target in allowed)
    if not filtered:
        raise ValueError(
            "candidate_targets produced no valid candidates; allowed for this horizon: "
            + ", ".join(allowed_targets)
        )
    return filtered


def _normalize_candidate_lambdas(
    choices: Optional[Sequence[str]],
) -> Optional[tuple[float, ...]]:
    if choices is None:
        return None

    normalized: list[float] = []
    saw_auto = False
    for choice in choices:
        for raw_token in str(choice).split(","):
            token = raw_token.strip().lower()
            if not token:
                continue
            if token in {"auto", "all"}:
                saw_auto = True
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError("candidate_ridge_lambdas must be auto/all or numbers") from exc
            if value <= 0.0:
                raise ValueError("candidate_ridge_lambdas must be positive")
            if value not in normalized:
                normalized.append(value)

    if saw_auto and normalized:
        raise ValueError("candidate_ridge_lambdas cannot mix auto/all with explicit values")
    if saw_auto or not normalized:
        return None
    return tuple(normalized)


def _snapshot_manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _date_range_for_source(path: Path, date_columns: tuple[str, ...]) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "rows": 0, "tickers": 0, "date_columns": {}}
    try:
        df = pd.read_parquet(str(path))
    except Exception as exc:
        return {"exists": True, "error": str(exc), "date_columns": {}}

    columns: dict[str, object] = {}
    for column in date_columns:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce")
        if values.notna().any():
            columns[column] = {
                "valid": int(values.notna().sum()),
                "min": values.min().date().isoformat(),
                "max": values.max().date().isoformat(),
            }
    return {
        "exists": True,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "date_columns": columns,
    }


def _feature_source_profile() -> dict[str, object]:
    return {
        "valuations": _date_range_for_source(
            TRAINER_DIR / "valuations.parquet",
            ("date", "fetched_at", "updated_at"),
        ),
        "financials": _date_range_for_source(
            TRAINER_DIR / "financials.parquet",
            ("report_date", "period", "fetched_at", "updated_at"),
        ),
    }


def _snapshot_source_fingerprints(
    *,
    snapshot_frequency: str,
    ground_truth_path: Path,
) -> dict[str, str]:
    if snapshot_frequency == "daily":
        paths = {
            "ashare_prices": TRAINER_DIR / "ashare_prices.parquet",
            "hkshare_prices": TRAINER_DIR / "hkshare_prices.parquet",
            "valuations": TRAINER_DIR / "valuations.parquet",
            "financials": TRAINER_DIR / "financials.parquet",
            "scorer": SCORER_SOURCE_PATH,
            "ground_truth_code": GROUND_TRUTH_SOURCE_PATH,
        }
    else:
        paths = {
            "ground_truth": ground_truth_path,
            "scorer": SCORER_SOURCE_PATH,
            "ground_truth_code": GROUND_TRUTH_SOURCE_PATH,
        }
    return {name: _snapshot_fingerprint(path) for name, path in paths.items()}


def _expected_snapshot_manifest(
    *,
    snapshot_frequency: str,
    ground_truth_path: Path,
    output_path: Path,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "snapshot_frequency": snapshot_frequency,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_path": str(output_path),
        "source_fingerprints": _snapshot_source_fingerprints(
            snapshot_frequency=snapshot_frequency,
            ground_truth_path=ground_truth_path,
        ),
        "feature_source_profile": _feature_source_profile()
        if snapshot_frequency == "daily"
        else {},
    }


def _manifest_core(manifest: Mapping[str, object]) -> dict[str, object]:
    ignored = {"created_at", "row_count", "snapshot_count"}
    core = {key: value for key, value in manifest.items() if key not in ignored}
    source_fingerprints = core.get("source_fingerprints")
    if isinstance(source_fingerprints, dict):
        cleaned = dict(source_fingerprints)
        cleaned.pop("ml_trainer", None)
        core["source_fingerprints"] = cleaned
    return core


def _load_valid_snapshot_cache(
    output_path: Path,
    *,
    snapshot_frequency: str,
    ground_truth_path: Path,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    if not output_path.exists():
        return None

    manifest_path = _snapshot_manifest_path(output_path)
    expected = _expected_snapshot_manifest(
        snapshot_frequency=snapshot_frequency,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
    )
    if not manifest_path.exists():
        raise RuntimeError(
            f"ML snapshot cache {output_path} has no manifest. "
            "Rebuild it with `train-ml-scorer --force-snapshots` after confirming "
            "point-in-time feature sources are available."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _manifest_core(manifest) != expected:
        raise RuntimeError(
            f"ML snapshot cache {output_path} is stale for feature policy "
            f"{FEATURE_POLICY_VERSION}. Rebuild with `train-ml-scorer --force-snapshots`."
        )

    df = pd.read_parquet(str(output_path))
    required = {f"target_rank_{horizon}" for horizon in EVAL_HORIZONS}
    required.add("target_rank_1w_market")
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"ML snapshot cache {output_path} is missing target columns: "
            f"{sorted(required.difference(df.columns))}"
        )
    return df


def _write_snapshot_manifest(
    output_path: Path,
    df: pd.DataFrame,
    *,
    snapshot_frequency: str,
    ground_truth_path: Path,
    start_date: date,
    end_date: date,
) -> None:
    manifest = _expected_snapshot_manifest(
        snapshot_frequency=snapshot_frequency,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
    )
    manifest.update({
        "created_at": datetime.now().isoformat(),
        "row_count": int(len(df)),
        "snapshot_count": int(pd.to_datetime(df["snapshot_date"]).nunique())
        if "snapshot_date" in df.columns
        else 0,
    })
    _snapshot_manifest_path(output_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _validate_point_in_time_feature_sources(
    *,
    valuations: pd.DataFrame,
    financials: pd.DataFrame,
    end_date: date,
) -> None:
    problems: list[str] = []
    for label, frame, columns in (
        ("valuations", valuations, ("date", "fetched_at", "updated_at")),
        ("financials", financials, ("report_date", "period", "fetched_at", "updated_at")),
    ):
        if frame.empty:
            continue
        usable_dates = []
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_datetime(frame[column], errors="coerce")
            if values.notna().any():
                usable_dates.append(values.dt.normalize())
        if not usable_dates:
            problems.append(f"{label} has no usable as-of/report date column")
            continue
        earliest = min(series.min() for series in usable_dates)
        latest_allowed = pd.Timestamp(end_date)
        if pd.Timestamp(earliest) > latest_allowed:
            problems.append(
                f"{label} starts at {pd.Timestamp(earliest).date().isoformat()}, "
                f"after training end date {end_date.isoformat()}"
            )

    if problems:
        raise RuntimeError(
            "Point-in-time feature sources are not suitable for daily ML snapshots: "
            + "; ".join(problems)
            + ". Fetch or build historical valuation/fundamental feature files before "
            "forcing a snapshot rebuild."
        )


def _target_rank_by_snapshot(df: pd.DataFrame, return_col: str) -> pd.Series:
    values = pd.to_numeric(df[return_col], errors="coerce")
    valid = values.notna()
    counts = valid.groupby(df["snapshot_date"], sort=False).transform("sum")
    ranks = values.groupby(df["snapshot_date"], sort=False).rank(method="average")
    target = ranks / (counts + 1.0) * 2.0 - 1.0
    target = target.where(valid & (counts >= 2), np.nan)
    return target.astype("float64")


def _target_rank_by_snapshot_market(df: pd.DataFrame, return_col: str) -> pd.Series:
    values = pd.to_numeric(df[return_col], errors="coerce")
    valid = values.notna()
    markets = df["ticker"].map(_market_bucket)
    groups = [df["snapshot_date"], markets]
    counts = valid.groupby(groups, sort=False).transform("sum")
    ranks = values.groupby(groups, sort=False).rank(method="average")
    target = ranks / (counts + 1.0) * 2.0 - 1.0
    target = target.where(valid & (counts >= 2), np.nan)
    return target.astype("float64")


def _application_universe_mask(
    df: pd.DataFrame,
    *,
    screening: Optional[ScreeningConfig] = None,
) -> pd.Series:
    """Return rows that pass the same quantitative screen as production."""
    screening = screening or ScreeningConfig()
    required = {"market_cap_rmb", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError("Application-universe training requires columns: " + ", ".join(missing))

    market_cap = pd.to_numeric(df["market_cap_rmb"], errors="coerce")
    pe_ratio = pd.to_numeric(df["pe_ratio"], errors="coerce")
    pb_ratio = pd.to_numeric(df["pb_ratio"], errors="coerce")
    roe = pd.to_numeric(df["roe"], errors="coerce")
    debt_ratio = pd.to_numeric(df["debt_to_equity"], errors="coerce")
    return (
        market_cap.ge(screening.market_cap_min_rmb)
        & pe_ratio.gt(0.0)
        & pe_ratio.le(screening.pe_max)
        & pb_ratio.gt(0.0)
        & pb_ratio.le(screening.pb_max)
        & roe.ge(screening.roe_min)
        & (debt_ratio.isna() | debt_ratio.le(screening.debt_ratio_max))
    )


def _application_data_quality_summary(df: pd.DataFrame) -> dict[str, object]:
    feature_columns = (
        "market_cap_rmb",
        "pe_ratio",
        "pb_ratio",
        "roe",
        "debt_to_equity",
    )
    coverage = {
        column: int(pd.to_numeric(df[column], errors="coerce").notna().sum()) if column in df.columns else 0
        for column in feature_columns
    }
    try:
        eligible = _application_universe_mask(df)
    except RuntimeError:
        eligible = pd.Series(False, index=df.index, dtype=bool)
    eligible_frame = df.loc[eligible]
    return {
        "rows": int(len(df)),
        "snapshots": int(df["snapshot_date"].nunique()) if "snapshot_date" in df.columns else 0,
        "coverage": coverage,
        "eligible_rows": int(eligible.sum()),
        "eligible_snapshots": int(eligible_frame["snapshot_date"].nunique())
        if "snapshot_date" in eligible_frame.columns
        else 0,
    }


def _filter_application_training_universe(
    df: pd.DataFrame,
    *,
    min_rows_per_snapshot: int = APPLICATION_MIN_ROWS_PER_SNAPSHOT,
    min_snapshots: int = 0,
) -> pd.DataFrame:
    """Filter and rerank snapshots in the universe seen by the live ranker."""
    summary = _application_data_quality_summary(df)
    eligible = df.loc[_application_universe_mask(df)].copy()
    if not eligible.empty and min_rows_per_snapshot > 0:
        counts = eligible.groupby("snapshot_date", sort=False)["ticker"].transform("size")
        eligible = eligible.loc[counts >= min_rows_per_snapshot].copy()

    snapshot_count = int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0
    if eligible.empty or snapshot_count < min_snapshots:
        coverage = summary["coverage"]
        raise RuntimeError(
            "Training data cannot represent the production stock screen: "
            f"eligible_rows={summary['eligible_rows']}, "
            f"eligible_snapshots={summary['eligible_snapshots']}, "
            f"required_snapshots={min_snapshots}, "
            f"PE_rows={coverage['pe_ratio']}, PB_rows={coverage['pb_ratio']}, "
            f"ROE_rows={coverage['roe']}. Rebuild point-in-time financial and "
            "valuation history before training."
        )

    eligible = eligible.sort_values(["snapshot_date", "ticker"], kind="mergesort")
    eligible = eligible.reset_index(drop=True)
    eligible = _refresh_hand_rank_scores(eligible)
    for horizon in EVAL_HORIZONS:
        return_col = f"forward_return_{horizon}"
        if return_col in eligible.columns:
            eligible[f"target_rank_{horizon}"] = _target_rank_by_snapshot(
                eligible,
                return_col,
            )
    if "forward_return_1w" in eligible.columns:
        eligible["target_rank_1w_market"] = _target_rank_by_snapshot_market(
            eligible,
            "forward_return_1w",
        )
    return eligible


def _monthly_rebalance_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one well-populated snapshot per month for independent evaluation."""
    if df.empty:
        return df.copy()
    dates = _snapshot_dates(df)
    counts = (
        pd.DataFrame({"snapshot_date": dates}).groupby("snapshot_date", sort=False).size().rename("rows").reset_index()
    )
    counts["month"] = counts["snapshot_date"].dt.to_period("M")
    selected = (
        counts.sort_values(["month", "rows", "snapshot_date"], kind="mergesort")
        .groupby("month", sort=False)
        .tail(1)["snapshot_date"]
    )
    return df.loc[dates.isin(selected)].copy().reset_index(drop=True)


def _prediction_rank_by_snapshot(df: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    values = pd.Series(predictions, index=df.index, dtype="float64")
    valid = values.notna()
    counts = valid.groupby(df["snapshot_date"], sort=False).transform("sum")
    ranks = values.groupby(df["snapshot_date"], sort=False).rank(method="average")
    target = ranks / (counts + 1.0) * 2.0 - 1.0
    return target.where(valid & (counts >= 2), np.nan).to_numpy(dtype="float64")


def _application_factor_signal_values(df: pd.DataFrame, signal: str) -> np.ndarray:
    if signal == "quality_de_crowding":
        return -_numeric_column(df, "quality_score")
    if signal == "gross_profitability_de_crowding":
        gross_profit = _numeric_column(df, "revenue") * _numeric_column(df, "gross_margin")
        return -_safe_divide(gross_profit, _numeric_column(df, "total_assets"))
    if signal == "roe_de_crowding":
        return -_numeric_column(df, "roe")
    if signal == "roa_de_crowding":
        return -_numeric_column(df, "roa")
    if signal == "book_yield":
        return -_numeric_column(df, "pb_ratio")
    if signal == "sales_yield":
        return -_numeric_column(df, "ps_ratio")
    if signal == "liability_yield":
        liabilities = _numeric_column(df, "total_liabilities")
        market_cap = _numeric_column(df, "market_cap_rmb")
        return np.divide(
            liabilities,
            market_cap,
            out=np.full(len(df), np.nan, dtype="float64"),
            where=np.isfinite(market_cap) & (market_cap > 0.0),
        )
    if signal == "momentum_63d":
        return _numeric_column(df, "relative_return_63d")
    if signal == "low_current_ratio":
        return -_numeric_column(df, "current_ratio")
    return np.full(len(df), np.nan, dtype="float64")


def _application_factor_predictions(
    df: pd.DataFrame,
    model_predictions: np.ndarray,
    components: Sequence[Mapping[str, object]],
) -> np.ndarray:
    model_rank = _prediction_rank_by_snapshot(df, model_predictions)
    blended = np.zeros(len(df), dtype="float64")
    total_weight = 0.0
    for component in components:
        signal = str(component.get("signal", "")).strip()
        try:
            weight = float(component.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if signal not in SIX_MONTH_APPLICATION_FACTOR_SIGNALS or not np.isfinite(weight) or weight <= 0.0:
            continue
        if signal == "model":
            factor_rank = model_rank
        else:
            factor_rank = _prediction_rank_by_snapshot(
                df,
                _application_factor_signal_values(df, signal),
            )
            factor_rank = np.where(np.isfinite(factor_rank), factor_rank, 0.0)
        blended += weight * factor_rank
        total_weight += weight
    if total_weight <= 0.0:
        return model_predictions
    return blended / total_weight


def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype="float64")


def _finite_or_zero_array(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, 0.0)


def _slog_array(values: np.ndarray) -> np.ndarray:
    finite = _finite_or_zero_array(values)
    return np.sign(finite) * np.log1p(np.abs(finite))


def _linear_scores(values: np.ndarray, *, best: float, worst: float) -> np.ndarray:
    if best == worst:
        return np.full(len(values), 50.0, dtype="float64")
    scores = (values - worst) / (best - worst) * 100.0
    return np.clip(scores, 0.0, 100.0)


def _log_scores(values: np.ndarray, *, best: float, worst: float) -> np.ndarray:
    scores = np.zeros(len(values), dtype="float64")
    if best <= 0 or worst <= 0:
        return scores
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return scores
    log_best = np.log(best)
    log_worst = np.log(worst)
    if log_best == log_worst:
        scores[valid] = 50.0
        return scores
    scores[valid] = (
        (np.log(values[valid]) - log_worst) / (log_best - log_worst) * 100.0
    )
    return np.clip(scores, 0.0, 100.0)


def _triangular_scores(
    values: np.ndarray,
    *,
    low: float,
    optimal: float,
    high: float,
) -> np.ndarray:
    scores = np.zeros(len(values), dtype="float64")
    valid = np.isfinite(values) & (values > low) & (values < high)
    rising = valid & (values <= optimal)
    falling = valid & (values > optimal)
    if optimal == low:
        scores[rising] = 100.0
    else:
        scores[rising] = (values[rising] - low) / (optimal - low) * 100.0
    if high == optimal:
        scores[falling] = 100.0
    else:
        scores[falling] = (high - values[falling]) / (high - optimal) * 100.0
    return np.clip(scores, 0.0, 100.0)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype="float64")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _weighted_value_score(values: Mapping[str, np.ndarray]) -> np.ndarray:
    n_rows = len(next(iter(values.values())))
    numerator = np.zeros(n_rows, dtype="float64")
    weight_sum = np.zeros(n_rows, dtype="float64")

    def add(mask: np.ndarray, scores: np.ndarray, weight: float) -> None:
        valid = mask & np.isfinite(scores)
        numerator[valid] += scores[valid] * weight
        weight_sum[valid] += weight

    pb = values["pb_ratio"]
    pe_forward = values["pe_forward"]
    pe = values["pe_ratio"]
    ps = values["ps_ratio"]
    market_cap = values["market_cap_rmb"]
    dividend_yield = values["dividend_yield"]
    ev_to_ebitda = values["ev_to_ebitda"]
    price = values["close"]
    roe = values["roe"]
    operating_cash_flow = values["operating_cash_flow"]
    free_cash_flow = values["free_cash_flow"]
    total_equity = values["total_equity"]
    net_income = values["net_income"]
    total_assets = values["total_assets"]
    current_ratio = values["current_ratio"]
    gross_margin = values["gross_margin"]
    revenue = values["revenue"]

    add(np.isfinite(pb) & (pb > 0), _log_scores(pb, best=20.0, worst=0.5), 0.20)
    add(
        np.isfinite(pe_forward) & (pe_forward > 0),
        _log_scores(pe_forward, best=10.0, worst=50.0),
        0.08,
    )
    fw_pe_improve = _safe_divide(pe, pe_forward)
    add(
        np.isfinite(pe) & (pe > 0) & np.isfinite(pe_forward) & (pe_forward > 0),
        _log_scores(fw_pe_improve, best=1.5, worst=0.5),
        0.06,
    )
    pct_imp = _safe_divide(pe - pe_forward, pe)
    add(
        np.isfinite(pe) & (pe > 0) & np.isfinite(pe_forward) & (pe_forward > 0),
        _linear_scores(pct_imp, best=0.5, worst=-0.1),
        0.04,
    )
    add(np.isfinite(ps) & (ps > 0), _log_scores(ps, best=0.15, worst=4.0), 0.14)
    add(
        np.isfinite(market_cap) & (market_cap > 0),
        _linear_scores(market_cap, best=500_000_000_000.0, worst=1_000_000_000.0),
        0.10,
    )
    add(
        np.isfinite(dividend_yield) & (dividend_yield >= 0),
        _triangular_scores(dividend_yield, low=0.0, optimal=0.03, high=0.08),
        0.12,
    )
    add(
        np.isfinite(dividend_yield) & (dividend_yield < 0),
        np.zeros(n_rows, dtype="float64"),
        0.12,
    )
    add(
        np.isfinite(ev_to_ebitda) & (ev_to_ebitda > 0),
        _triangular_scores(ev_to_ebitda, low=2.0, optimal=6.5, high=22.0),
        0.12,
    )
    roe_ev = _safe_divide(roe, ev_to_ebitda)
    add(
        np.isfinite(ev_to_ebitda) & (ev_to_ebitda > 0) & np.isfinite(roe) & (roe > 0),
        _linear_scores(roe_ev, best=0.5, worst=0.01),
        0.06,
    )
    add(np.isfinite(price) & (price > 0), _log_scores(price, best=8.0, worst=60.0), 0.06)
    price_market_cap = price * market_cap
    add(
        np.isfinite(price) & (price > 0) & np.isfinite(market_cap) & (market_cap > 0),
        _log_scores(price_market_cap, best=10_000_000_000.0, worst=1_000_000_000_000.0),
        0.05,
    )
    price_pb = price * pb
    add(
        np.isfinite(price) & (price > 0) & np.isfinite(pb) & (pb > 0),
        _log_scores(price_pb, best=5.0, worst=250.0),
        0.08,
    )
    ocf_yield = _safe_divide(operating_cash_flow, market_cap)
    ocf_mask = np.isfinite(operating_cash_flow) & np.isfinite(market_cap) & (market_cap > 0)
    add(ocf_mask & (operating_cash_flow > 0), _log_scores(ocf_yield, best=0.03, worst=0.0005), 0.06)
    add(ocf_mask & (operating_cash_flow <= 0), np.zeros(n_rows, dtype="float64"), 0.06)
    fcf_yield = _safe_divide(free_cash_flow, market_cap)
    fcf_mask = np.isfinite(free_cash_flow) & np.isfinite(market_cap) & (market_cap > 0)
    add(fcf_mask & (free_cash_flow > 0), _log_scores(fcf_yield, best=0.01, worst=0.0005), 0.05)
    add(fcf_mask & (free_cash_flow <= 0), np.zeros(n_rows, dtype="float64"), 0.05)
    fcf_equity_yield = _safe_divide(free_cash_flow, total_equity)
    add(
        np.isfinite(free_cash_flow)
        & np.isfinite(total_equity)
        & (total_equity > 0)
        & np.isfinite(market_cap)
        & (market_cap > 30_000_000_000.0)
        & (free_cash_flow > 0),
        _log_scores(fcf_equity_yield, best=0.05, worst=0.001),
        0.04,
    )
    add(
        np.isfinite(free_cash_flow)
        & np.isfinite(total_equity)
        & (total_equity > 0)
        & np.isfinite(market_cap)
        & (market_cap > 30_000_000_000.0)
        & (free_cash_flow <= 0),
        np.zeros(n_rows, dtype="float64"),
        0.04,
    )
    roa = _safe_divide(net_income, total_assets)
    ni_asset_mask = np.isfinite(net_income) & np.isfinite(total_assets) & (total_assets > 0)
    add(ni_asset_mask & (net_income > 0), _linear_scores(roa, best=0.15, worst=0.0), 0.05)
    add(ni_asset_mask & (net_income <= 0), np.zeros(n_rows, dtype="float64"), 0.05)
    add(
        np.isfinite(market_cap)
        & (market_cap > 50_000_000_000.0)
        & np.isfinite(current_ratio)
        & (current_ratio > 0),
        _triangular_scores(current_ratio, low=0.5, optimal=1.5, high=3.0),
        0.06,
    )
    add(
        np.isfinite(market_cap)
        & (market_cap > 50_000_000_000.0)
        & np.isfinite(current_ratio)
        & (current_ratio <= 0),
        np.zeros(n_rows, dtype="float64"),
        0.06,
    )
    gross_profit = revenue * gross_margin
    gpa_yield = _safe_divide(gross_profit, total_assets)
    add(
        np.isfinite(revenue)
        & np.isfinite(gross_margin)
        & np.isfinite(total_assets)
        & (total_assets > 0)
        & np.isfinite(market_cap)
        & (market_cap > 50_000_000_000.0)
        & (gross_profit > 0),
        _log_scores(gpa_yield, best=0.30, worst=0.01),
        0.04,
    )
    liabilities = total_assets - total_equity
    liabilities_ratio = _safe_divide(liabilities, market_cap)
    liabilities_mask = (
        np.isfinite(market_cap)
        & (market_cap > 50_000_000_000.0)
        & np.isfinite(total_assets)
        & np.isfinite(total_equity)
    )
    add(liabilities_mask & (liabilities <= 0), np.full(n_rows, 100.0, dtype="float64"), 0.04)
    add(
        liabilities_mask & (liabilities > 0),
        _log_scores(liabilities_ratio, best=0.05, worst=0.8),
        0.04,
    )
    earnings_yield = _safe_divide(net_income, market_cap)
    earnings_mask = np.isfinite(net_income) & np.isfinite(market_cap) & (market_cap > 0)
    add(
        earnings_mask & (net_income > 0),
        _log_scores(earnings_yield, best=0.01, worst=0.0005),
        0.06,
    )
    add(
        earnings_mask & (net_income > 0),
        np.zeros(n_rows, dtype="float64"),
        0.06,
    )
    add(earnings_mask & (net_income <= 0), np.zeros(n_rows, dtype="float64"), 0.06)

    raw_avg = np.full(n_rows, 50.0, dtype="float64")
    has_weight = weight_sum > 0
    raw_avg[has_weight] = numerator[has_weight] / weight_sum[has_weight]
    result = np.zeros(n_rows, dtype="float64")
    positive = raw_avg > 0
    boost_alpha = 1.55
    result[positive] = (raw_avg[positive] ** boost_alpha) / (100.0 ** (boost_alpha - 1.0))
    return np.clip(result, 0.0, 100.0)


def _growth_score(values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    peg = values["peg_ratio"]
    pe = values["pe_ratio"]
    pe_forward = values["pe_forward"]
    n_rows = len(peg)
    numerator = np.zeros(n_rows, dtype="float64")
    weight_sum = np.zeros(n_rows, dtype="float64")
    growth_raw = np.full(n_rows, np.nan, dtype="float64")
    valid_peg = np.isfinite(peg) & (peg > 0)
    growth_raw[valid_peg] = 1.0 / peg[valid_peg]

    peg_scores = _log_scores(growth_raw, best=1.5, worst=0.5)
    numerator[valid_peg] += peg_scores[valid_peg] * 0.5
    weight_sum[valid_peg] += 0.5
    valid_bad_peg = np.isfinite(peg) & (peg <= 0)
    weight_sum[valid_bad_peg] += 0.5

    forward_pe_ratio = _safe_divide(pe, pe_forward)
    valid_fw = np.isfinite(pe) & np.isfinite(pe_forward) & (pe > 0) & (pe_forward > 0)
    fw_scores = _linear_scores(forward_pe_ratio, best=2.5, worst=0.3)
    numerator[valid_fw] += fw_scores[valid_fw] * 0.5
    weight_sum[valid_fw] += 0.5
    invalid_fw = np.isfinite(pe) & np.isfinite(pe_forward) & ~valid_fw
    weight_sum[invalid_fw] += 0.5

    scores = np.full(n_rows, 50.0, dtype="float64")
    has_weight = weight_sum > 0
    scores[has_weight] = numerator[has_weight] / weight_sum[has_weight]
    return scores, growth_raw


def _quality_raw(values: Mapping[str, np.ndarray]) -> np.ndarray:
    roe = values["roe"]
    gross_margin = values["gross_margin"]
    debt_to_equity = values["debt_to_equity"]
    free_cash_flow = values["free_cash_flow"]
    net_margin = values["net_margin"]
    operating_cash_flow = values["operating_cash_flow"]
    total_assets = values["total_assets"]
    net_income = values["net_income"]

    raw = np.full(len(roe), np.nan, dtype="float64")
    dte_pos = np.maximum(debt_to_equity, 0.0)
    valid = (
        np.isfinite(roe)
        & np.isfinite(gross_margin)
        & np.isfinite(debt_to_equity)
        & (roe > 0)
        & (gross_margin > 0.15)
        & (dte_pos <= 2.0)
        & (~np.isfinite(free_cash_flow) | (free_cash_flow > 0))
    )
    roe_capped = np.minimum(roe, 0.30)
    raw[valid] = (roe_capped[valid] * gross_margin[valid]) / (1.0 + np.sqrt(dte_pos[valid]))

    margin_bonus = 1.0 + np.minimum(net_margin, 0.5)
    raw *= np.where(np.isfinite(net_margin) & (net_margin > 0), margin_bonus, 1.0)

    ocf_yield = _safe_divide(operating_cash_flow, total_assets)
    raw *= np.where(
        np.isfinite(ocf_yield) & (ocf_yield > 0),
        1.0 + np.minimum(ocf_yield, 0.5),
        1.0,
    )

    roa = _safe_divide(net_income, total_assets)
    raw *= np.where(np.isfinite(roa) & (roa > 0), 1.0 + np.minimum(roa, 0.3), 1.0)
    return raw


def _cross_section_quality_scores(
    snapshot_dates: pd.Series,
    quality_raw: np.ndarray,
) -> np.ndarray:
    scores = np.full(len(quality_raw), 50.0, dtype="float64")
    valid = np.isfinite(quality_raw) & (quality_raw > 0)
    if not valid.any():
        return scores

    raw_series = pd.Series(quality_raw, index=snapshot_dates.index)
    log_q = _log_scores(quality_raw, best=0.05, worst=0.001)
    for _snapshot_date, index in snapshot_dates[valid].groupby(snapshot_dates[valid], sort=False).groups.items():
        positions = snapshot_dates.index.get_indexer(index)
        group_raw = quality_raw[positions]
        if len(group_raw) == 1:
            scores[positions] = 0.2 * log_q[positions] + 0.8 * 50.0
            continue
        raw_min = np.nanmin(group_raw)
        raw_max = np.nanmax(group_raw)
        if raw_max == raw_min:
            ranks = raw_series.loc[index].rank(method="first").to_numpy(dtype="float64")
            linear = ((ranks - 0.5) / len(group_raw)) * 100.0
            compressed = 50.0 + (linear - 50.0) * 0.90
            scores[positions] = 0.2 * log_q[positions] + 0.8 * compressed
        else:
            cross = _linear_scores(
                np.log(group_raw),
                best=float(np.log(raw_max)),
                worst=float(np.log(raw_min)),
            )
            scores[positions] = 0.2 * log_q[positions] + 0.8 * cross
    return np.clip(scores, 0.0, 100.0)


def _cross_section_growth_scores(
    snapshot_dates: pd.Series,
    growth_raw: np.ndarray,
    growth_abs: np.ndarray,
) -> np.ndarray:
    scores = np.full(len(growth_raw), 50.0, dtype="float64")
    valid = np.isfinite(growth_raw) & (growth_raw > 0)
    for _snapshot_date, index in snapshot_dates[valid].groupby(snapshot_dates[valid], sort=False).groups.items():
        positions = snapshot_dates.index.get_indexer(index)
        group_raw = growth_raw[positions]
        if len(group_raw) <= 1:
            scores[positions] = 50.0
            continue
        log_raw = np.log(group_raw)
        log_min = float(np.nanmin(log_raw))
        log_max = float(np.nanmax(log_raw))
        if log_max == log_min:
            scores[positions] = 50.0
        else:
            scores[positions] = _linear_scores(log_raw, best=log_max, worst=log_min)
    blended = 0.3 * growth_abs + 0.7 * scores
    return np.clip(blended, 0.0, 100.0)


def _refresh_hand_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute hand-scorer component columns with vectorized scorer logic."""
    refreshed = df.copy()
    values = {field: _numeric_column(refreshed, field) for field in BASE_FIELDS}

    value_score = _weighted_value_score(values)
    growth_abs, growth_raw = _growth_score(values)
    quality_score = _cross_section_quality_scores(
        refreshed["snapshot_date"],
        _quality_raw(values),
    )
    growth_score = _cross_section_growth_scores(
        refreshed["snapshot_date"],
        growth_raw,
        growth_abs,
    )
    momentum_score = np.full(len(refreshed), 50.0, dtype="float64")
    asset_turnover = _safe_divide(values["revenue"], values["total_assets"])
    valid_turnover = (
        np.isfinite(values["revenue"])
        & (values["revenue"] > 0)
        & np.isfinite(values["total_assets"])
        & (values["total_assets"] > 0)
    )
    momentum_score[valid_turnover] = _linear_scores(asset_turnover, best=2.0, worst=0.0)[valid_turnover]

    synergy_score = np.minimum(value_score, quality_score)
    value_growth_score = np.sqrt(np.maximum(value_score, 0.0) * np.maximum(growth_score, 0.0))
    component_scores = {
        "value": value_score,
        "quality": quality_score,
        "growth": growth_score,
        "momentum": momentum_score,
        "synergy": synergy_score,
        "value_growth": value_growth_score,
    }
    weights = {
        "value": 0.40,
        "quality": 0.20,
        "growth": 0.10,
        "momentum": 0.10,
        "synergy": 0.10,
        "value_growth": 0.10,
    }
    total_weight = sum(weights.values())
    log_sum = np.zeros(len(refreshed), dtype="float64")
    for key, weight in weights.items():
        log_sum += weight * np.log(np.maximum(component_scores[key], 10.0))
    composite_score = np.exp(log_sum / total_weight)
    composite_score = np.clip(composite_score, 0.0, 100.0)

    net_income = values["net_income"]
    market_cap = values["market_cap_rmb"]
    loss_mask = np.isfinite(net_income) & np.isfinite(market_cap) & (market_cap > 0) & (net_income < 0)
    composite_score[loss_mask] *= 1.0 / (1.0 + 5.0 * np.abs(net_income[loss_mask]) / market_cap[loss_mask])
    composite_score *= (1.0 - 0.10 * np.abs(value_score - quality_score) / 100.0)
    net_margin = values["net_margin"]
    low_margin = np.isfinite(net_margin) & (net_margin > 0) & (net_margin < 0.02)
    composite_score[low_margin] *= 0.95

    refreshed["composite_score"] = composite_score
    refreshed["value_score"] = value_score
    refreshed["quality_score"] = quality_score
    refreshed["growth_score"] = growth_score
    return refreshed


def prepare_ml_training_snapshots(
    *,
    force: bool = False,
    ground_truth_path: Path = CURRENT_GROUND_TRUTH_FILE,
    output_path: Path = ML_SNAPSHOTS_FILE,
    snapshot_frequency: str = "daily",
    start_date: date = DAILY_START_DATE,
    end_date: Optional[date] = DAILY_END_DATE,
) -> pd.DataFrame:
    """Create/read the snapshot table used by the local ML ranker."""
    resolved_end_date = (
        _resolve_daily_end_date(end_date) if snapshot_frequency == "daily" else (end_date or DAILY_END_DATE)
    )
    if not force:
        cached = _load_valid_snapshot_cache(
            output_path,
            snapshot_frequency=snapshot_frequency,
            ground_truth_path=ground_truth_path,
            start_date=start_date,
            end_date=resolved_end_date,
        )
        if cached is not None:
            return cached

    if snapshot_frequency == "daily":
        return build_daily_ml_training_snapshots(
            force=force,
            ground_truth_path=ground_truth_path,
            output_path=output_path,
            start_date=start_date,
            end_date=resolved_end_date,
        )
    if snapshot_frequency != "quarterly":
        raise ValueError("snapshot_frequency must be one of: daily, quarterly")

    ensure_current_ground_truth_ready(force=False)
    df = pd.read_parquet(str(ground_truth_path))
    df = _filter_application_training_universe(
        df,
        min_rows_per_snapshot=APPLICATION_MIN_ROWS_PER_SNAPSHOT,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    _write_snapshot_manifest(
        output_path,
        df,
        snapshot_frequency=snapshot_frequency,
        ground_truth_path=ground_truth_path,
        start_date=start_date,
        end_date=resolved_end_date,
    )
    logger.info("Saved ML training snapshots -> %s", output_path)
    return df


def _load_training_prices(*, require_ashare: bool = True) -> pd.DataFrame:
    price_files = [
        ("ashare", TRAINER_DIR / "ashare_prices.parquet"),
        ("hkshare", TRAINER_DIR / "hkshare_prices.parquet"),
    ]
    frames = []
    missing = []
    for label, path in price_files:
        if not path.exists():
            missing.append(label)
            continue
        frame = pd.read_parquet(str(path))
        if frame.empty:
            continue
        frame = frame.loc[:, [col for col in ("ticker", "date", "close") if col in frame.columns]]
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["ticker", "date", "close"])
        frame = frame[frame["close"] > 0]
        frames.append(frame)

    if require_ashare and "ashare" in missing:
        raise FileNotFoundError(
            "Daily A-share price data is missing at data/trainer/ashare_prices.parquet. "
            "Run `valueinvestor improve-scorer --fetch-only` to populate it before "
            "daily ML training."
        )
    if not frames:
        raise FileNotFoundError("No daily price parquet files found in data/trainer")

    return pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"])


def _short_horizon_price_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    history = prices.loc[:, ["ticker", "date", "close"]].copy()
    history["ticker"] = history["ticker"].astype(str)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna(subset=["ticker", "date", "close"])
    history = history[history["close"] > 0].sort_values(["ticker", "date"], kind="mergesort")
    if history.empty:
        return pd.DataFrame(columns=["ticker", "snapshot_date", *SHORT_HORIZON_FEATURES])

    grouped_close = history.groupby("ticker", sort=False)["close"]
    for window in (5, 21, 63):
        history[f"price_return_{window}d"] = grouped_close.transform(
            lambda values, window=window: values / values.shift(window) - 1.0
        )
    history["_daily_return"] = grouped_close.pct_change()
    grouped_daily = history.groupby("ticker", sort=False)["_daily_return"]
    for window in (21, 63):
        history[f"price_volatility_{window}d"] = (
            grouped_daily
            .rolling(window=window, min_periods=max(5, window // 3))
            .std()
            .reset_index(level=0, drop=True)
        )

    history["_market"] = history["ticker"].map(_market_bucket)
    for window in (5, 21, 63):
        price_col = f"price_return_{window}d"
        market_col = f"market_return_{window}d"
        history[market_col] = history.groupby(["_market", "date"], sort=False)[price_col].transform("mean")
        history[f"relative_return_{window}d"] = history[price_col] - history[market_col]
    for window in (21, 63):
        price_col = f"price_volatility_{window}d"
        market_col = f"market_volatility_{window}d"
        history[market_col] = history.groupby(["_market", "date"], sort=False)[price_col].transform("mean")
        history[f"relative_volatility_{window}d"] = history[price_col] - history[market_col]

    history["snapshot_date"] = history["date"].dt.date
    return history[["ticker", "snapshot_date", *SHORT_HORIZON_FEATURES]]


def _daily_feature_frame(
    prices: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    price_feature_frame = _short_horizon_price_feature_frame(prices[prices["date"] <= end_date])
    prices = prices[(prices["date"] >= start_date) & (prices["date"] <= end_date)].copy()
    if prices.empty:
        raise RuntimeError(f"No price rows between {start_date} and {end_date}")

    prices = prices.rename(columns={"date": "snapshot_date"})
    prices = prices.sort_values(["snapshot_date", "ticker"], kind="mergesort")

    valuations_path = TRAINER_DIR / "valuations.parquet"
    financials_path = TRAINER_DIR / "financials.parquet"
    valuations = pd.read_parquet(str(valuations_path)) if valuations_path.exists() else pd.DataFrame()
    financials = pd.read_parquet(str(financials_path)) if financials_path.exists() else pd.DataFrame()
    _validate_point_in_time_feature_sources(
        valuations=valuations,
        financials=financials,
        end_date=end_date,
    )
    financials = align_financial_history_to_live_periods(financials)

    base = prices[["ticker", "snapshot_date", "close"]].reset_index(drop=True)
    base = base.merge(
        price_feature_frame,
        on=["ticker", "snapshot_date"],
        how="left",
    )
    valuation_features = asof_feature_frame(
        valuations,
        VALUATION_FEATURE_COLUMNS,
        base["ticker"],
        base["snapshot_date"],
        lag_days=VALUATION_ASOF_LAG_DAYS,
    )
    financial_features = asof_feature_frame(
        financials,
        (*FINANCIAL_FEATURE_COLUMNS, "statement_months"),
        base["ticker"],
        base["snapshot_date"],
        lag_days=FINANCIAL_ASOF_LAG_DAYS,
    )
    financial_features = annualize_financial_feature_frame(financial_features)

    features = base.join(valuation_features.drop(columns=["ticker"]))
    features = features.join(financial_features.drop(columns=["ticker", "statement_months"]))
    ordered_columns = [
        "ticker",
        "snapshot_date",
        "close",
        *SHORT_HORIZON_FEATURES,
        *VALUATION_FEATURE_COLUMNS,
        *FINANCIAL_FEATURE_COLUMNS,
    ]
    return features[ordered_columns].where(pd.notna(features[ordered_columns]), None)


def build_daily_ml_training_snapshots(
    *,
    force: bool = False,
    ground_truth_path: Path = CURRENT_GROUND_TRUTH_FILE,
    output_path: Path = ML_SNAPSHOTS_FILE,
    start_date: date = DAILY_START_DATE,
    end_date: Optional[date] = DAILY_END_DATE,
) -> pd.DataFrame:
    """Build one training snapshot per available trading date."""
    resolved_end_date = _resolve_daily_end_date(end_date)
    if not force:
        cached = _load_valid_snapshot_cache(
            output_path,
            snapshot_frequency="daily",
            ground_truth_path=ground_truth_path,
            start_date=start_date,
            end_date=resolved_end_date,
        )
        if cached is not None:
            return cached

    prices = _load_training_prices(require_ashare=True)
    features = _daily_feature_frame(prices, start_date=start_date, end_date=resolved_end_date)
    feature_quality = _application_data_quality_summary(features)
    features = features.loc[_application_universe_mask(features)].reset_index(drop=True)
    if features.empty:
        raise RuntimeError(f"No daily feature rows pass the production stock screen; quality_summary={feature_quality}")

    horizons = [
        (FORWARD_HORIZON_1W_DAYS, "forward_return_1w"),
        (FORWARD_HORIZON_1M_DAYS, "forward_return_1m"),
        (FORWARD_HORIZON_3M_DAYS, "forward_return_3m"),
        (FORWARD_HORIZON_6M_DAYS, "forward_return_6m"),
    ]
    merged = features
    for horizon_days, col_name in horizons:
        returns = _compute_forward_returns(
            prices,
            horizon_days=horizon_days,
            column_name=col_name,
        )
        if returns.empty:
            raise RuntimeError(f"No forward returns computed for {col_name}")
        returns = returns.rename(columns={"date": "snapshot_date"})
        merged = merged.merge(
            returns[["ticker", "snapshot_date", col_name]],
            on=["ticker", "snapshot_date"],
            how="left",
        )

    merged = merged.dropna(subset=["forward_return_6m"]).reset_index(drop=True)
    merged = _filter_application_training_universe(
        merged,
        min_rows_per_snapshot=APPLICATION_MIN_ROWS_PER_SNAPSHOT,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(output_path), index=False)
    _write_snapshot_manifest(
        output_path,
        merged,
        snapshot_frequency="daily",
        ground_truth_path=ground_truth_path,
        start_date=start_date,
        end_date=resolved_end_date,
    )
    logger.info(
        "Saved daily ML training snapshots -> %s (%d rows, %d snapshot dates)",
        output_path,
        len(merged),
        merged["snapshot_date"].nunique(),
    )
    return merged


def _ticker_priors(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    priors: Dict[str, Dict[str, float]] = {}
    grouped = df.groupby("ticker", sort=False)

    for horizon in FEATURE_HORIZONS:
        return_col = f"forward_return_{horizon}"
        target_col = f"target_rank_{horizon}"
        ticker_means = grouped[return_col].mean()
        ticker_ranks = ticker_means.rank(pct=True) * 2.0 - 1.0
        ticker_rankmeans = grouped[target_col].mean()

        for ticker in ticker_means.index:
            entry = priors.setdefault(str(ticker), {})
            mean_value = ticker_means.loc[ticker]
            rank_value = ticker_ranks.loc[ticker]
            rankmean_value = ticker_rankmeans.loc[ticker]
            entry[f"ticker_mean_{horizon}"] = (
                float(mean_value) if pd.notna(mean_value) else 0.0
            )
            entry[f"ticker_rank_{horizon}"] = (
                float(rank_value) if pd.notna(rank_value) else 0.0
            )
            entry[f"ticker_rankmean_{horizon}"] = (
                float(rankmean_value) if pd.notna(rankmean_value) else 0.0
            )

    return priors


def _attach_asof_ticker_priors(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    embargo_days: int = DEFAULT_PRIOR_EMBARGO_DAYS,
) -> pd.DataFrame:
    """Attach ticker-prior feature columns using only returns knowable before each row."""
    enriched = target_df.copy()
    if target_df.empty:
        return enriched

    target_dates = _snapshot_dates(enriched)
    history = history_df.copy()
    history["_snapshot_ts"] = _snapshot_dates(history)
    history["ticker"] = history["ticker"].astype(str)

    target_tickers = enriched["ticker"].astype(str).reset_index(drop=True)
    target_dates_reset = target_dates.reset_index(drop=True)
    positions_by_ticker = {
        str(ticker): np.asarray(list(positions), dtype="int64")
        for ticker, positions in target_tickers.groupby(target_tickers, sort=False).groups.items()
    }
    history_by_ticker = {
        str(ticker): group
        for ticker, group in history.sort_values(["ticker", "_snapshot_ts"]).groupby(
            "ticker",
            sort=False,
        )
    }
    for horizon in FEATURE_HORIZONS:
        return_col = f"forward_return_{horizon}"
        target_col = f"target_rank_{horizon}"
        cutoff_dates = (
            target_dates_reset
            - pd.Timedelta(days=HORIZON_DAYS[horizon] + embargo_days)
        )
        cutoff_values = cutoff_dates.to_numpy(dtype="datetime64[ns]")
        mean_values = np.zeros(len(enriched), dtype="float64")
        rankmean_values = np.zeros(len(enriched), dtype="float64")

        for ticker, positions in positions_by_ticker.items():
            hist_group = history_by_ticker.get(ticker)
            if hist_group is None or hist_group.empty:
                continue
            hist_dates = hist_group["_snapshot_ts"].to_numpy(dtype="datetime64[ns]")
            hist_returns = pd.to_numeric(hist_group[return_col], errors="coerce").to_numpy(dtype="float64")
            hist_ranks = pd.to_numeric(hist_group[target_col], errors="coerce").to_numpy(dtype="float64")
            valid_returns = np.where(np.isfinite(hist_returns), hist_returns, 0.0)
            valid_rankmeans = np.where(np.isfinite(hist_ranks), hist_ranks, 0.0)
            return_counts = np.isfinite(hist_returns).astype("int64")
            rank_counts = np.isfinite(hist_ranks).astype("int64")
            return_sum = np.cumsum(valid_returns)
            rank_sum = np.cumsum(valid_rankmeans)
            return_count_sum = np.cumsum(return_counts)
            rank_count_sum = np.cumsum(rank_counts)
            cutoffs = cutoff_values[positions]
            end_positions = np.searchsorted(hist_dates, cutoffs, side="right") - 1
            valid = end_positions >= 0
            if valid.any():
                selected = end_positions[valid]
                ret_counts = return_count_sum[selected]
                rank_counts_selected = rank_count_sum[selected]
                ret_ok = ret_counts > 0
                rank_ok = rank_counts_selected > 0
                valid_positions = positions[valid]
                if ret_ok.any():
                    mean_values[valid_positions[ret_ok]] = (
                        return_sum[selected[ret_ok]] / ret_counts[ret_ok]
                    )
                if rank_ok.any():
                    rankmean_values[valid_positions[rank_ok]] = (
                        rank_sum[selected[rank_ok]] / rank_counts_selected[rank_ok]
                    )

        mean_column = _prior_column_name(f"ticker_mean_{horizon}")
        rank_column = _prior_column_name(f"ticker_rank_{horizon}")
        rankmean_column = _prior_column_name(f"ticker_rankmean_{horizon}")
        enriched[mean_column] = mean_values
        enriched[rankmean_column] = rankmean_values
        enriched[rank_column] = (
            pd.Series(mean_values)
            .groupby(target_dates_reset, sort=False)
            .rank(pct=True)
            .fillna(0.5)
            .to_numpy(dtype="float64")
            * 2.0
            - 1.0
        )

    return enriched


def _priors_for_strategy(
    df: pd.DataFrame,
    strategy: str,
) -> Dict[str, Dict[str, float]]:
    if strategy in {"ticker_priors", "rolling_ticker_priors"}:
        return _ticker_priors(df)
    if strategy == "no_ticker_priors":
        return {}
    raise ValueError(f"Unsupported ticker prior strategy: {strategy}")


def _ratio_feature_values(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    close = values["close"]
    market_cap = values["market_cap_rmb"]
    net_income = values["net_income"]
    revenue = values["revenue"]
    total_assets = values["total_assets"]
    total_equity = values["total_equity"]
    total_liabilities = values["total_liabilities"]
    operating_cash_flow = values["operating_cash_flow"]
    free_cash_flow = values["free_cash_flow"]
    gross_margin = values["gross_margin"]
    roe = values["roe"]
    pb_ratio = values["pb_ratio"]
    pe_ratio = values["pe_ratio"]
    pe_forward = values["pe_forward"]
    ev_to_ebitda = values["ev_to_ebitda"]
    peg_ratio = values["peg_ratio"]

    gross_profit = np.full(len(close), np.nan, dtype="float64")
    valid_gross_profit = np.isfinite(revenue) & np.isfinite(gross_margin)
    gross_profit[valid_gross_profit] = revenue[valid_gross_profit] * gross_margin[valid_gross_profit]
    pe_forward_decline = np.full(len(close), np.nan, dtype="float64")
    valid_pe_decline = np.isfinite(pe_ratio) & np.isfinite(pe_forward) & (pe_ratio != 0)
    pe_forward_decline[valid_pe_decline] = (
        pe_ratio[valid_pe_decline] - pe_forward[valid_pe_decline]
    ) / pe_ratio[valid_pe_decline]
    price_pb = np.full(len(close), np.nan, dtype="float64")
    valid_price_pb = np.isfinite(close) & np.isfinite(pb_ratio)
    price_pb[valid_price_pb] = close[valid_price_pb] * pb_ratio[valid_price_pb]
    price_market_cap = np.full(len(close), np.nan, dtype="float64")
    valid_price_market_cap = np.isfinite(close) & np.isfinite(market_cap)
    price_market_cap[valid_price_market_cap] = close[valid_price_market_cap] * market_cap[valid_price_market_cap]
    peg_inv = np.full(len(close), np.nan, dtype="float64")
    valid_peg = np.isfinite(peg_ratio) & (peg_ratio > 0)
    peg_inv[valid_peg] = 1.0 / peg_ratio[valid_peg]

    return {
        "earnings_yield": _safe_divide(net_income, market_cap),
        "ocf_yield": _safe_divide(operating_cash_flow, market_cap),
        "fcf_yield": _safe_divide(free_cash_flow, market_cap),
        "gross_profit_assets": _safe_divide(gross_profit, total_assets),
        "gross_profit_market_cap": _safe_divide(gross_profit, market_cap),
        "asset_turnover": _safe_divide(revenue, total_assets),
        "roa_calc": _safe_divide(net_income, total_assets),
        "roe_pb": _safe_divide(roe, pb_ratio),
        "roe_ev": _safe_divide(roe, ev_to_ebitda),
        "pe_forward_improve": _safe_divide(pe_ratio, pe_forward),
        "pe_forward_decline": pe_forward_decline,
        "liabilities_market_cap": _safe_divide(total_liabilities, market_cap),
        "debt_assets": _safe_divide(total_liabilities, total_assets),
        "fcf_equity": _safe_divide(free_cash_flow, total_equity),
        "ocf_assets": _safe_divide(operating_cash_flow, total_assets),
        "price_pb": price_pb,
        "price_market_cap": price_market_cap,
        "peg_inv": peg_inv,
    }


def _prior_column_name(key: str) -> str:
    return f"{PRIOR_COLUMN_PREFIX}{key}"


def _centered_rank_array(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape[0], dtype="float64")
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n <= 1:
        return output
    finite_values = values[finite]
    ranks = stats.rankdata(finite_values, method="average") - 1.0
    output[finite] = ranks / float(n - 1) - 0.5
    return output


def _encode_label_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.empty(len(values), dtype="int64")
    lookup: dict[str, int] = {}
    uniques: list[str] = []
    for idx, value in enumerate(values):
        key = str(value)
        code = lookup.get(key)
        if code is None:
            code = len(uniques)
            lookup[key] = code
            uniques.append(key)
        codes[idx] = code
    return np.asarray(uniques, dtype=object), codes


def _cross_sectional_rank_groups(
    df: pd.DataFrame,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    snapshot_keys = (
        pd.to_datetime(df["snapshot_date"], errors="coerce").astype(str)
        if "snapshot_date" in df.columns
        else pd.Series(["snapshot"] * len(df))
    )
    group_frame = pd.DataFrame({
        "snapshot": snapshot_keys.reset_index(drop=True),
        "market": df["ticker"].astype(str).map(_market_bucket).reset_index(drop=True),
    })
    snapshot_groups = tuple(
        np.asarray(index, dtype=int)
        for index in group_frame.groupby("snapshot", sort=False).groups.values()
    )
    market_groups = tuple(
        np.asarray(index, dtype=int)
        for index in group_frame.groupby(["snapshot", "market"], sort=False).groups.values()
    )
    return snapshot_groups, market_groups


def _cross_sectional_rank_arrays(
    values: np.ndarray,
    *,
    snapshot_groups: Sequence[np.ndarray],
    market_groups: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype="float64")
    snapshot_rank = np.zeros(len(values), dtype="float64")
    for positions in snapshot_groups:
        snapshot_rank[positions] = _centered_rank_array(values[positions])

    market_rank = np.zeros(len(values), dtype="float64")
    for positions in market_groups:
        market_rank[positions] = _centered_rank_array(values[positions])
    return snapshot_rank, market_rank


def build_feature_matrix(
    df: pd.DataFrame,
    ticker_priors: Mapping[str, Mapping[str, float]],
    *,
    prefer_row_priors: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    parsed = {field: _numeric_column(df, field) for field in BASE_FIELDS}
    feature_arrays: dict[str, np.ndarray] = {}
    selected_names = list(feature_names) if feature_names is not None else expected_feature_names()

    for field in BASE_FIELDS:
        raw = parsed[field]
        feature_arrays[field] = _finite_or_zero_array(raw)
        feature_arrays[f"{field}_miss"] = np.isnan(raw).astype("float64")
        feature_arrays[f"slog_{field}"] = _slog_array(raw)
        if field in INVERSE_FIELDS:
            inverse = np.zeros(len(df), dtype="float64")
            valid = np.isfinite(raw) & (raw > 0)
            inverse[valid] = 1.0 / raw[valid]
            feature_arrays[f"inv_{field}"] = inverse

    ratios = _ratio_feature_values(parsed)
    for feature in RATIO_FEATURES:
        raw = ratios[feature]
        feature_arrays[feature] = _finite_or_zero_array(raw)
        feature_arrays[f"slog_{feature}"] = _slog_array(raw)

    if "ticker" in df.columns:
        ticker_values = df["ticker"].astype(str).to_numpy(dtype=object, copy=False)
    else:
        ticker_values = np.full(len(df), "", dtype=object)
    if ticker_priors:
        unique_tickers, ticker_inverse = _encode_label_values(ticker_values)
    else:
        unique_tickers = np.asarray([], dtype=object)
        ticker_inverse = np.zeros(len(df), dtype="int64")
    for horizon in FEATURE_HORIZONS:
        for key in (
            f"ticker_mean_{horizon}",
            f"ticker_rank_{horizon}",
            f"ticker_rankmean_{horizon}",
        ):
            prior_column = _prior_column_name(key)
            if prefer_row_priors and prior_column in df.columns:
                values = pd.to_numeric(df[prior_column], errors="coerce").fillna(0.0)
                feature_arrays[key] = values.astype("float64").to_numpy()
            else:
                if ticker_priors:
                    unique_values = np.fromiter(
                        (
                            float(ticker_priors.get(str(ticker), {}).get(key, 0.0))
                            for ticker in unique_tickers
                        ),
                        dtype="float64",
                        count=len(unique_tickers),
                    )
                    feature_arrays[key] = unique_values[ticker_inverse]
                else:
                    feature_arrays[key] = np.zeros(len(df), dtype="float64")

    for feature in SHORT_HORIZON_FEATURES:
        raw = _numeric_column(df, feature)
        feature_arrays[feature] = _finite_or_zero_array(raw)
        feature_arrays[f"{feature}_miss"] = np.isnan(raw).astype("float64")
        feature_arrays[f"slog_{feature}"] = _slog_array(raw)

    for name, left, right in INTERACTION_FEATURES:
        feature_arrays[name] = feature_arrays.get(
            left,
            np.zeros(len(df), dtype="float64"),
        ) * feature_arrays.get(
            right,
            np.zeros(len(df), dtype="float64"),
        )
    for name, source in POLYNOMIAL_FEATURES:
        values = feature_arrays.get(source, np.zeros(len(df), dtype="float64"))
        feature_arrays[name] = values * values

    needed_cross_sectional_sources = {
        name.removeprefix("cs_rank_")
        for name in selected_names
        if name.startswith("cs_rank_")
    }
    needed_cross_sectional_sources.update(
        name.removeprefix("market_cs_rank_")
        for name in selected_names
        if name.startswith("market_cs_rank_")
    )
    cross_sectional_groups = None
    for source in CROSS_SECTIONAL_FEATURES:
        if source not in needed_cross_sectional_sources:
            continue
        if cross_sectional_groups is None:
            cross_sectional_groups = _cross_sectional_rank_groups(df)
        snapshot_rank, market_rank = _cross_sectional_rank_arrays(
            feature_arrays.get(source, np.zeros(len(df), dtype="float64")),
            snapshot_groups=cross_sectional_groups[0],
            market_groups=cross_sectional_groups[1],
        )
        feature_arrays[f"cs_rank_{source}"] = snapshot_rank
        feature_arrays[f"market_cs_rank_{source}"] = market_rank
    for name, left, right in CROSS_SECTIONAL_INTERACTION_FEATURES:
        if name not in selected_names:
            continue
        feature_arrays[name] = feature_arrays.get(
            left,
            np.zeros(len(df), dtype="float64"),
        ) * feature_arrays.get(
            right,
            np.zeros(len(df), dtype="float64"),
        )

    try:
        matrix = np.column_stack([feature_arrays[name] for name in selected_names]).astype(
            "float64",
            copy=False,
        )
    except KeyError as exc:
        raise RuntimeError(f"Unsupported ML feature name: {exc.args[0]}") from exc
    expected_count = len(selected_names)
    if matrix.shape[1] != expected_count:
        raise RuntimeError(
            f"ML feature matrix has {matrix.shape[1]} columns; expected {expected_count}"
        )
    return matrix


def _runtime_feature_parity(
    df: pd.DataFrame,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Compare trainer features with the exact production matrix builder."""
    model = _model_from_payload(payload)
    if not model.feature_names:
        return {
            "matched": False,
            "reason": "payload has no direct feature schema",
            "rows": 0,
            "features": 0,
            "max_abs_error": None,
        }
    dates = _snapshot_dates(df)
    latest = pd.Timestamp(dates.max())
    sample = df.loc[dates == latest].copy().reset_index(drop=True)
    value_columns = [*BASE_FIELDS, *SHORT_HORIZON_FEATURES]
    values = sample.reindex(columns=value_columns).to_dict("records")
    tickers = sample["ticker"].astype(str).tolist()
    trainer_matrix = build_feature_matrix(
        sample,
        model.ticker_priors,
        prefer_row_priors=False,
        feature_names=model.feature_names,
    )
    runtime_matrix = feature_matrix_from_values(values, tickers, model)
    if trainer_matrix.shape != runtime_matrix.shape:
        return {
            "matched": False,
            "reason": "feature matrix shape mismatch",
            "rows": int(len(sample)),
            "features": int(len(model.feature_names)),
            "trainer_shape": list(trainer_matrix.shape),
            "runtime_shape": list(runtime_matrix.shape),
            "max_abs_error": None,
        }
    max_abs_error = float(np.max(np.abs(trainer_matrix - runtime_matrix))) if trainer_matrix.size else 0.0
    trainer_predictions = _predict_ml_ranker_payload_unblended(sample, payload)
    runtime_predictions = model.predict_matrix(
        runtime_matrix,
        tickers=tickers,
        cap_buckets=[_cap_bucket(value) for value in sample["market_cap_rmb"]],
    )
    prediction_max_abs_error = float(np.max(np.abs(trainer_predictions - runtime_predictions))) if len(sample) else 0.0
    matched = bool(
        np.isfinite(max_abs_error)
        and max_abs_error <= 1e-10
        and np.isfinite(prediction_max_abs_error)
        and prediction_max_abs_error <= 1e-10
    )
    return {
        "matched": matched,
        "reason": "matched" if matched else "feature or prediction values differ",
        "rows": int(len(sample)),
        "features": int(len(model.feature_names)),
        "snapshot_date": latest.date().isoformat(),
        "max_abs_error": max_abs_error,
        "prediction_max_abs_error": prediction_max_abs_error,
    }


def _standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clip_low = np.percentile(X, 1, axis=0)
    clip_high = np.percentile(X, 99, axis=0)
    clipped = np.clip(X, clip_low, clip_high)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (clipped - mean) / scale, clip_low, clip_high, mean, scale


def _apply_sample_weight(
    X_aug: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if sample_weight is None:
        return X_aug, y
    weights = np.asarray(sample_weight, dtype="float64")
    if weights.shape[0] != X_aug.shape[0]:
        raise ValueError("sample_weight length must match training rows")
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    mean_weight = float(weights.mean())
    if mean_weight <= 0:
        return X_aug, y
    sqrt_weight = np.sqrt(weights / mean_weight)
    return X_aug * sqrt_weight[:, None], y * sqrt_weight


def _fit_ridge_numpy(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, str]:
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype="float64")])
    X_aug, y = _apply_sample_weight(X_aug, y, sample_weight)
    regularizer = ridge_lambda * np.eye(X_aug.shape[1], dtype="float64")
    regularizer[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + regularizer, X_aug.T @ y)
    return coef, "numpy"


def _fit_ridge_mlx(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, str]:
    import mlx.core as mx  # type: ignore[import-not-found]

    global _MLX_RIDGE_DIAGNOSTIC_LOGGED

    if hasattr(mx, "gpu"):
        try:
            mx.set_default_device(mx.gpu)
        except Exception:
            logger.debug("Could not set MLX default device to GPU", exc_info=True)

    started = time.perf_counter()
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype="float32")]).astype(
        "float32",
        copy=False,
    )
    X_aug, y_weighted = _apply_sample_weight(X_aug, y, sample_weight)
    X_aug = X_aug.astype("float32", copy=False)
    y_np = y_weighted.astype("float32", copy=False)
    X_mx = mx.array(X_aug, dtype=mx.float32)
    y_mx = mx.array(y_np, dtype=mx.float32)
    regularizer_np = ridge_lambda * np.eye(X_aug.shape[1], dtype="float32")
    regularizer_np[-1, -1] = 0.0
    regularizer = mx.array(regularizer_np, dtype=mx.float32)

    gram = X_mx.T @ X_mx + regularizer
    rhs = X_mx.T @ y_mx

    # MLX 0.31 does not support linalg.solve on GPU. Conjugate gradient keeps
    # the expensive Gram build and solve iterations on the selected MLX device.
    coef = mx.zeros((X_aug.shape[1],), dtype=mx.float32)
    residual = rhs - gram @ coef
    direction = residual
    residual_norm = mx.sum(residual * residual)
    eps = np.float32(1e-12)
    max_steps = min(256, max(32, X_aug.shape[1] * 2))
    for _ in range(max_steps):
        gram_direction = gram @ direction
        step_size = residual_norm / (mx.sum(direction * gram_direction) + eps)
        coef = coef + step_size * direction
        next_residual = residual - step_size * gram_direction
        next_residual_norm = mx.sum(next_residual * next_residual)
        direction = next_residual + (
            next_residual_norm / (residual_norm + eps)
        ) * direction
        residual = next_residual
        residual_norm = next_residual_norm
    mx.eval(coef, residual_norm)

    elapsed = time.perf_counter() - started
    if not _MLX_RIDGE_DIAGNOSTIC_LOGGED:
        device = mx.default_device() if hasattr(mx, "default_device") else "unknown"
        memory = ""
        if hasattr(mx, "get_active_memory") and hasattr(mx, "get_peak_memory"):
            memory = (
                f", active_memory={mx.get_active_memory() / 1_000_000:.1f}MB"
                f", peak_memory={mx.get_peak_memory() / 1_000_000:.1f}MB"
            )
        logger.info(
            "MLX ridge conjugate-gradient solve using device=%s rows=%d features=%d "
            "steps=%d elapsed=%.2fs%s",
            device,
            X.shape[0],
            X.shape[1],
            max_steps,
            elapsed,
            memory,
        )
        _MLX_RIDGE_DIAGNOSTIC_LOGGED = True

    return np.asarray(coef, dtype="float64"), "mlx"


def _fit_ridge_mlx_adam(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, str]:
    import mlx.core as mx  # type: ignore[import-not-found]

    global _MLX_ADAM_RIDGE_DIAGNOSTIC_LOGGED

    if hasattr(mx, "gpu"):
        try:
            mx.set_default_device(mx.gpu)
        except Exception:
            logger.debug("Could not set MLX default device to GPU", exc_info=True)

    started = time.perf_counter()
    X_aug_np = np.hstack([X, np.ones((X.shape[0], 1), dtype="float32")]).astype(
        "float32",
        copy=False,
    )
    X_aug_np, y_weighted = _apply_sample_weight(X_aug_np, y, sample_weight)
    X_aug_np = X_aug_np.astype("float32", copy=False)
    y_np = y_weighted.astype("float32", copy=False)
    X_mx = mx.array(X_aug_np, dtype=mx.float32)
    y_mx = mx.array(y_np, dtype=mx.float32)
    reg_np = np.ones(X_aug_np.shape[1], dtype="float32")
    reg_np[-1] = 0.0
    reg = mx.array(reg_np, dtype=mx.float32)

    steps = 100
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    n_rows = float(X_aug_np.shape[0])

    coef = mx.zeros((X_aug_np.shape[1],), dtype=mx.float32)
    momentum = mx.zeros_like(coef)
    velocity = mx.zeros_like(coef)

    for step in range(1, steps + 1):
        residual = X_mx @ coef - y_mx
        grad = (X_mx.T @ residual) / n_rows + (ridge_lambda / n_rows) * reg * coef
        momentum = beta1 * momentum + (1.0 - beta1) * grad
        velocity = beta2 * velocity + (1.0 - beta2) * (grad * grad)
        momentum_hat = momentum / (1.0 - beta1 ** step)
        velocity_hat = velocity / (1.0 - beta2 ** step)
        coef = coef - lr * momentum_hat / (mx.sqrt(velocity_hat) + eps)
        mx.eval(coef)

    elapsed = time.perf_counter() - started
    if not _MLX_ADAM_RIDGE_DIAGNOSTIC_LOGGED:
        device = mx.default_device() if hasattr(mx, "default_device") else "unknown"
        memory = ""
        if hasattr(mx, "get_active_memory") and hasattr(mx, "get_peak_memory"):
            memory = (
                f", active_memory={mx.get_active_memory() / 1_000_000:.1f}MB"
                f", peak_memory={mx.get_peak_memory() / 1_000_000:.1f}MB"
            )
        logger.info(
            "MLX ridge Adam optimizer using device=%s rows=%d features=%d "
            "steps=%d elapsed=%.2fs%s",
            device,
            X.shape[0],
            X.shape[1],
            steps,
            elapsed,
            memory,
        )
        _MLX_ADAM_RIDGE_DIAGNOSTIC_LOGGED = True

    return np.asarray(coef, dtype="float64"), "mlx-adam"


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float,
    backend: str,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, str]:
    if backend not in {"auto", "mlx", "mlx-cg", "mlx-adam", "numpy"}:
        raise ValueError("backend must be one of: auto, mlx, mlx-cg, mlx-adam, numpy")

    if backend in {"auto", "mlx", "mlx-cg"}:
        try:
            return _fit_ridge_mlx(X, y, ridge_lambda, sample_weight=sample_weight)
        except Exception as exc:
            if backend in {"mlx", "mlx-cg"}:
                raise
            logger.info("MLX backend unavailable, falling back to NumPy: %s", exc)
    if backend == "mlx-adam":
        return _fit_ridge_mlx_adam(X, y, ridge_lambda, sample_weight=sample_weight)

    return _fit_ridge_numpy(X, y, ridge_lambda, sample_weight=sample_weight)


def _linear_model_dict(
    *,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coef_with_intercept: np.ndarray,
    market: Optional[str] = None,
    backend: Optional[str] = None,
) -> dict[str, object]:
    model: dict[str, object] = {
        "clip_low": clip_low,
        "clip_high": clip_high,
        "mean": mean,
        "scale": scale,
        "coef": coef_with_intercept[:-1],
        "intercept": float(coef_with_intercept[-1]),
    }
    if market is not None:
        model["market"] = market
    if backend is not None:
        model["backend"] = backend
    return model


def _fit_market_ridge_models(
    X_scaled: np.ndarray,
    target: np.ndarray,
    tickers: pd.Series,
    *,
    ridge_lambda: float,
    backend: str,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[list[dict[str, object]], str]:
    models: list[dict[str, object]] = []
    backends: list[str] = []
    market_values = tickers.map(_market_bucket)
    valid = np.isfinite(target)
    for market in ("ashare", "hk"):
        market_mask = (market_values == market).to_numpy(dtype=bool) & valid
        if int(market_mask.sum()) < 10:
            continue
        coef_with_intercept, used_backend = _fit_ridge(
            X_scaled[market_mask],
            target[market_mask],
            ridge_lambda=ridge_lambda,
            backend=backend,
            sample_weight=sample_weight[market_mask] if sample_weight is not None else None,
        )
        models.append(
            _linear_model_dict(
                clip_low=clip_low,
                clip_high=clip_high,
                mean=mean,
                scale=scale,
                coef_with_intercept=coef_with_intercept,
                market=market,
                backend=used_backend,
            )
        )
        backends.append(used_backend)
    if not models:
        raise ValueError("not enough market-specific rows to train market ridge")
    return models, "+".join(dict.fromkeys(backends))


def _fit_segment_ridge_models(
    X_scaled: np.ndarray,
    target: np.ndarray,
    snapshots: pd.DataFrame,
    *,
    ridge_lambda: float,
    backend: str,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[list[dict[str, object]], str]:
    models: list[dict[str, object]] = []
    backends: list[str] = []
    tickers = snapshots["ticker"].astype(str)
    market_values = tickers.map(_market_bucket)
    cap_values = (
        snapshots["market_cap_rmb"].map(_cap_bucket)
        if "market_cap_rmb" in snapshots.columns
        else pd.Series("unknown", index=snapshots.index)
    )
    board_values = tickers.map(_board_bucket)
    listing_values = tickers.map(_listing_bucket)
    valid = np.isfinite(target)
    segment_frame = pd.DataFrame({
        "market": market_values,
        "cap_bucket": cap_values,
        "board_bucket": board_values,
        "listing_bucket": listing_values,
    })
    for (market, cap_bucket, board_bucket, listing_bucket), index in segment_frame.groupby(
        ["market", "cap_bucket", "board_bucket", "listing_bucket"],
        sort=True,
    ).groups.items():
        positions = snapshots.index.get_indexer(index)
        positions = positions[positions >= 0]
        if len(positions) == 0:
            continue
        segment_mask = np.zeros(len(target), dtype=bool)
        segment_mask[positions] = True
        segment_mask &= valid
        if int(segment_mask.sum()) < 50:
            continue
        coef_with_intercept, used_backend = _fit_ridge(
            X_scaled[segment_mask],
            target[segment_mask],
            ridge_lambda=ridge_lambda,
            backend=backend,
            sample_weight=(
                sample_weight[segment_mask] if sample_weight is not None else None
            ),
        )
        model = _linear_model_dict(
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            scale=scale,
            coef_with_intercept=coef_with_intercept,
            market=str(market),
            backend=used_backend,
        )
        model["cap_bucket"] = str(cap_bucket)
        model["board_bucket"] = str(board_bucket)
        model["listing_bucket"] = str(listing_bucket)
        models.append(model)
        backends.append(used_backend)
    if not models:
        raise ValueError("not enough market/cap segment rows to train segment ridge")
    return models, "+".join(dict.fromkeys(backends))


def _predict_scaled_linear_models(
    X_scaled: np.ndarray,
    snapshots: pd.DataFrame,
    models: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Predict freshly fitted routed models without rebuilding their feature matrix."""
    tickers = snapshots["ticker"].astype(str)
    markets = tickers.map(_market_bucket).to_numpy(dtype=object)
    cap_buckets = (
        snapshots["market_cap_rmb"].map(_cap_bucket).to_numpy(dtype=object)
        if "market_cap_rmb" in snapshots.columns
        else np.full(len(snapshots), "unknown", dtype=object)
    )
    board_buckets = tickers.map(_board_bucket).to_numpy(dtype=object)
    listing_buckets = tickers.map(_listing_bucket).to_numpy(dtype=object)
    weighted_predictions = np.zeros(len(snapshots), dtype="float64")
    unweighted_predictions = np.zeros(len(snapshots), dtype="float64")
    total_weights = np.zeros(len(snapshots), dtype="float64")
    prediction_counts = np.zeros(len(snapshots), dtype="float64")

    for model in models:
        mask = np.ones(len(snapshots), dtype=bool)
        for buckets, key in (
            (markets, "market"),
            (cap_buckets, "cap_bucket"),
            (board_buckets, "board_bucket"),
            (listing_buckets, "listing_bucket"),
        ):
            route_value = model.get(key)
            if route_value is not None:
                mask &= buckets == str(route_value)
        if not mask.any():
            continue
        model_predictions = X_scaled[mask] @ np.asarray(model["coef"], dtype="float64") + float(model["intercept"])
        weight = _payload_weight(model)
        weighted_predictions[mask] += weight * model_predictions
        unweighted_predictions[mask] += model_predictions
        total_weights[mask] += weight
        prediction_counts[mask] += 1.0

    output = np.zeros(len(snapshots), dtype="float64")
    weighted = total_weights > 0.0
    output[weighted] = weighted_predictions[weighted] / total_weights[weighted]
    unweighted = ~weighted & (prediction_counts > 0.0)
    output[unweighted] = unweighted_predictions[unweighted] / prediction_counts[unweighted]
    fallback = ~(weighted | unweighted)
    if fallback.any():
        output[fallback] = np.mean(
            [
                X_scaled[fallback] @ np.asarray(model["coef"], dtype="float64") + float(model["intercept"])
                for model in models
            ],
            axis=0,
        )
    return output


def _fit_pairwise_ranker_numpy(
    X: np.ndarray,
    y: np.ndarray,
    snapshot_dates: pd.Series,
    *,
    ridge_lambda: float,
    max_pairs: int = 200_000,
    steps: int = 120,
    learning_rate: float = 0.08,
) -> tuple[np.ndarray, str]:
    pairs: list[np.ndarray] = []
    valid = np.isfinite(y)
    valid_dates = snapshot_dates.reset_index(drop=True)
    for _snapshot_date, index in valid_dates[valid].groupby(valid_dates[valid], sort=False).groups.items():
        positions = np.asarray(list(index), dtype="int64")
        if len(positions) < 2:
            continue
        ordered = positions[np.argsort(y[positions], kind="mergesort")]
        n_pairs = min(8, len(ordered) // 2)
        if n_pairs <= 0:
            continue
        lows = ordered[:n_pairs]
        highs = ordered[-n_pairs:]
        for high, low in zip(highs, lows):
            if y[high] <= y[low]:
                continue
            pairs.append(X[high] - X[low])
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if len(pairs) < 10:
        raise ValueError("not enough within-snapshot pairs for pairwise ranker")

    diffs = np.vstack(pairs).astype("float64", copy=False)
    coef = np.zeros(X.shape[1], dtype="float64")
    l2 = ridge_lambda / max(float(len(diffs)), 1.0)
    for _step in range(steps):
        margins = diffs @ coef
        weights = 1.0 / (1.0 + np.exp(np.clip(margins, -50.0, 50.0)))
        grad = -(diffs.T @ weights) / len(diffs) + l2 * coef
        coef -= learning_rate * grad

    return np.concatenate([coef, np.asarray([0.0], dtype="float64")]), "pairwise_numpy"


def _evaluate_predictions(df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Dict[str, float]]:
    prediction_series = pd.Series(predictions, index=df.index, dtype="float64")
    results: Dict[str, Dict[str, float]] = {}

    for horizon in EVAL_HORIZONS:
        return_col = f"forward_return_{horizon}"
        if return_col not in df.columns:
            results[horizon] = {
                "spearman_rho": 0.0,
                "hit_rate_top20": 0.0,
                "mean_excess_return": 0.0,
                "n_snapshots": 0,
                "n_stocks_per_snapshot": 0.0,
            }
            continue
        rhos = []
        hit_rates = []
        excess_returns = []
        for _snapshot_date, group in df.groupby("snapshot_date", sort=False):
            returns = pd.to_numeric(group[return_col], errors="coerce")
            scores = prediction_series.loc[group.index]
            valid_mask = scores.notna() & returns.notna()
            if int(valid_mask.sum()) < 10:
                continue
            valid_scores = scores[valid_mask]
            valid_returns = returns[valid_mask]
            if valid_scores.nunique() < 2 or valid_returns.nunique() < 2:
                continue
            rho, _p = stats.spearmanr(valid_scores, valid_returns)
            if pd.isna(rho):
                continue
            rhos.append(float(rho))
            median_ret = valid_returns.median()
            top_count = min(20, len(valid_scores))
            top_index = valid_scores.nlargest(top_count).index
            top_returns = valid_returns.loc[top_index]
            hit_rates.append(float((top_returns > median_ret).mean()))
            excess_returns.append(float(top_returns.mean() - median_ret))

        if not rhos:
            results[horizon] = {
                "spearman_rho": 0.0,
                "hit_rate_top20": 0.0,
                "mean_excess_return": 0.0,
                "n_snapshots": 0,
                "n_stocks_per_snapshot": 0.0,
            }
        else:
            results[horizon] = {
                "spearman_rho": float(np.mean(rhos)),
                "hit_rate_top20": float(np.mean(hit_rates)),
                "mean_excess_return": float(np.mean(excess_returns)),
                "n_snapshots": len(rhos),
                "n_stocks_per_snapshot": float(
                    len(df) / max(df["snapshot_date"].nunique(), 1)
                ),
            }

    return results


def _primary_top20_excess_metrics(
    df: pd.DataFrame,
    predictions: np.ndarray,
    primary_horizon: str,
) -> tuple[float, float]:
    return_col = f"forward_return_{primary_horizon}"
    if return_col not in df.columns:
        return 0.0, 0.0

    prediction_series = pd.Series(predictions, index=df.index, dtype="float64")
    hit_rates = []
    excess_returns = []
    for _snapshot_date, group in df.groupby("snapshot_date", sort=False):
        returns = pd.to_numeric(group[return_col], errors="coerce")
        scores = prediction_series.loc[group.index]
        valid_mask = scores.notna() & returns.notna()
        if int(valid_mask.sum()) < 10:
            continue
        valid_scores = scores[valid_mask]
        valid_returns = returns[valid_mask]
        median_ret = valid_returns.median()
        top_count = min(20, len(valid_scores))
        top_index = valid_scores.nlargest(top_count).index
        top_returns = valid_returns.loc[top_index]
        hit_rates.append(float((top_returns > median_ret).mean()))
        excess_returns.append(float(top_returns.mean() - median_ret))
    if not hit_rates:
        return 0.0, 0.0
    return float(np.mean(hit_rates)), float(np.mean(excess_returns))


def _primary_top20_excess_predictions_non_degraded(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_top20_excess: tuple[float, float],
    primary_horizon: str,
) -> bool:
    candidate_top20, candidate_excess = _primary_top20_excess_metrics(
        df,
        candidate_predictions,
        primary_horizon,
    )
    incumbent_top20, incumbent_excess = incumbent_top20_excess
    return not (
        candidate_top20 - incumbent_top20 < 0.0
        and candidate_excess - incumbent_excess < 0.0
    )


def _evaluate_primary_spearman_rho(
    df: pd.DataFrame,
    predictions: np.ndarray,
    primary_horizon: str,
) -> float:
    return_col = f"forward_return_{primary_horizon}"
    if return_col not in df.columns:
        return 0.0

    prediction_series = pd.Series(predictions, index=df.index, dtype="float64")
    rhos = []
    for _snapshot_date, group in df.groupby("snapshot_date", sort=False):
        returns = pd.to_numeric(group[return_col], errors="coerce")
        scores = prediction_series.loc[group.index]
        valid_mask = scores.notna() & returns.notna()
        if int(valid_mask.sum()) < 10:
            continue
        valid_scores = scores[valid_mask]
        valid_returns = returns[valid_mask]
        if valid_scores.nunique() < 2 or valid_returns.nunique() < 2:
            continue
        rho, _p = stats.spearmanr(valid_scores, valid_returns)
        if pd.isna(rho):
            continue
        rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else 0.0


def _prepare_primary_spearman_context(
    df: pd.DataFrame,
    primary_horizon: str,
) -> dict[str, object]:
    """Precompute the return-side ranks reused during candidate preselection."""
    return_col = f"forward_return_{primary_horizon}"
    if return_col not in df.columns:
        return {"n_rows": len(df), "positions": np.asarray([], dtype="int64")}

    returns = pd.to_numeric(df[return_col], errors="coerce").to_numpy(dtype="float64")
    snapshot_dates = df["snapshot_date"].to_numpy()
    valid_positions = np.flatnonzero(np.isfinite(returns) & pd.notna(snapshot_dates))
    if not len(valid_positions):
        return {"n_rows": len(df), "positions": valid_positions}

    codes, unique_dates = pd.factorize(snapshot_dates[valid_positions], sort=False)
    n_groups = len(unique_dates)
    group_counts = np.bincount(codes, minlength=n_groups).astype("float64")
    valid_returns = returns[valid_positions]
    return_ranks = pd.Series(valid_returns).groupby(codes, sort=False).rank(method="average").to_numpy(dtype="float64")
    return_means = (
        np.bincount(
            codes,
            weights=return_ranks,
            minlength=n_groups,
        )
        / group_counts
    )
    return_centered = return_ranks - return_means[codes]
    return_sum_squares = np.bincount(
        codes,
        weights=return_centered * return_centered,
        minlength=n_groups,
    )
    return {
        "n_rows": len(df),
        "positions": valid_positions,
        "codes": codes,
        "n_groups": n_groups,
        "group_counts": group_counts,
        "returns": valid_returns,
        "return_centered": return_centered,
        "return_sum_squares": return_sum_squares,
    }


def _evaluate_prepared_primary_spearman_rho(
    context: Mapping[str, object],
    predictions: np.ndarray,
) -> float:
    """Evaluate mean cross-sectional Spearman rho from prepared return ranks."""
    scores = np.asarray(predictions, dtype="float64").reshape(-1)
    if len(scores) != int(context.get("n_rows", -1)):
        raise ValueError("prepared Spearman context and predictions have different lengths")

    positions = np.asarray(context.get("positions", []), dtype="int64")
    if not len(positions):
        return 0.0
    codes = np.asarray(context["codes"], dtype="int64")
    n_groups = int(context["n_groups"])
    valid_scores = scores[positions]
    finite_scores = np.isfinite(valid_scores)

    if finite_scores.all():
        group_counts = np.asarray(context["group_counts"], dtype="float64")
        return_centered = np.asarray(context["return_centered"], dtype="float64")
        return_sum_squares = np.asarray(
            context["return_sum_squares"],
            dtype="float64",
        )
    else:
        codes = codes[finite_scores]
        valid_scores = valid_scores[finite_scores]
        valid_returns = np.asarray(context["returns"], dtype="float64")[finite_scores]
        if not len(valid_scores):
            return 0.0
        group_counts = np.bincount(codes, minlength=n_groups).astype("float64")
        return_ranks = (
            pd.Series(valid_returns).groupby(codes, sort=False).rank(method="average").to_numpy(dtype="float64")
        )
        return_means = np.divide(
            np.bincount(codes, weights=return_ranks, minlength=n_groups),
            group_counts,
            out=np.zeros(n_groups, dtype="float64"),
            where=group_counts > 0,
        )
        return_centered = return_ranks - return_means[codes]
        return_sum_squares = np.bincount(
            codes,
            weights=return_centered * return_centered,
            minlength=n_groups,
        )

    score_ranks = pd.Series(valid_scores).groupby(codes, sort=False).rank(method="average").to_numpy(dtype="float64")
    score_means = np.divide(
        np.bincount(codes, weights=score_ranks, minlength=n_groups),
        group_counts,
        out=np.zeros(n_groups, dtype="float64"),
        where=group_counts > 0,
    )
    score_centered = score_ranks - score_means[codes]
    score_sum_squares = np.bincount(
        codes,
        weights=score_centered * score_centered,
        minlength=n_groups,
    )
    cross_products = np.bincount(
        codes,
        weights=score_centered * return_centered,
        minlength=n_groups,
    )
    evaluable = (group_counts >= 10) & (score_sum_squares > 0.0) & (return_sum_squares > 0.0)
    if not evaluable.any():
        return 0.0
    rhos = cross_products[evaluable] / np.sqrt(score_sum_squares[evaluable] * return_sum_squares[evaluable])
    return float(np.mean(rhos))


def _target_values(snapshots: pd.DataFrame, target: str) -> np.ndarray:
    if target.startswith("target_rank_") and target in snapshots.columns:
        return pd.to_numeric(snapshots[target], errors="coerce").to_numpy(dtype="float64")
    market_rank_targets = {
        "target_rank_1w_market": "1w",
        "target_rank_1m_market": "1m",
        "target_rank_3m_market": "3m",
        "target_rank_6m_market": "6m",
    }
    if target in market_rank_targets:
        return_col = f"forward_return_{market_rank_targets[target]}"
        required = {"ticker", "snapshot_date", return_col}
        if required.issubset(snapshots.columns):
            return _target_rank_by_snapshot_market(snapshots, return_col).to_numpy(
                dtype="float64"
            )
    if target == "target_rank_6m":
        return pd.to_numeric(snapshots["target_rank_6m"], errors="coerce").to_numpy(dtype="float64")
    if target == "target_rank_6m_soft":
        rank_6m = pd.to_numeric(snapshots["target_rank_6m"], errors="coerce").to_numpy(
            dtype="float64"
        )
        return np.sign(rank_6m) * (np.abs(rank_6m) ** 0.75)
    if target == "target_rank_6m_extreme":
        rank_6m = pd.to_numeric(snapshots["target_rank_6m"], errors="coerce").to_numpy(
            dtype="float64"
        )
        return np.sign(rank_6m) * (np.abs(rank_6m) ** 1.35)
    if target == "target_rank_weighted":
        weighted = (
            0.20 * pd.to_numeric(snapshots["target_rank_1m"], errors="coerce")
            + 0.30 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.50 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_703":
        weighted = (
            0.10 * pd.to_numeric(snapshots["target_rank_1m"], errors="coerce")
            + 0.20 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.70 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_703_market":
        weighted = (
            0.10 * _target_values(snapshots, "target_rank_1m_market")
            + 0.20 * _target_values(snapshots, "target_rank_3m_market")
            + 0.70 * _target_values(snapshots, "target_rank_6m_market")
        )
        return weighted.astype("float64", copy=False)
    if target == "target_rank_weighted_802":
        weighted = (
            0.20 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.80 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_802_market":
        weighted = (
            0.20 * _target_values(snapshots, "target_rank_3m_market")
            + 0.80 * _target_values(snapshots, "target_rank_6m_market")
        )
        return weighted.astype("float64", copy=False)
    if target == "target_rank_weighted_901":
        weighted = (
            0.10 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.90 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_901_market":
        weighted = (
            0.10 * _target_values(snapshots, "target_rank_3m_market")
            + 0.90 * _target_values(snapshots, "target_rank_6m_market")
        )
        return weighted.astype("float64", copy=False)
    if target == "target_rank_weighted_8515":
        weighted = (
            0.15 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.85 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_8515_market":
        weighted = (
            0.15 * _target_values(snapshots, "target_rank_3m_market")
            + 0.85 * _target_values(snapshots, "target_rank_6m_market")
        )
        return weighted.astype("float64", copy=False)
    if target == "target_rank_weighted_7525":
        weighted = (
            0.25 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.75 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_weighted_631":
        weighted = (
            0.15 * pd.to_numeric(snapshots["target_rank_1m"], errors="coerce")
            + 0.25 * pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + 0.60 * pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        )
        return weighted.to_numpy(dtype="float64")
    if target == "target_rank_mean":
        mean_target = (
            pd.to_numeric(snapshots["target_rank_1m"], errors="coerce")
            + pd.to_numeric(snapshots["target_rank_3m"], errors="coerce")
            + pd.to_numeric(snapshots["target_rank_6m"], errors="coerce")
        ) / 3.0
        return mean_target.to_numpy(dtype="float64")
    raise ValueError(f"Unsupported ML ranker target: {target}")


def _baseline_metrics(snapshots: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return _evaluate_predictions(
        snapshots,
        pd.to_numeric(snapshots["composite_score"], errors="coerce").to_numpy(dtype="float64"),
    )


def _improvement_ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if candidate > 0 else 0.0
    return (candidate - baseline) / abs(baseline)


def _sample_training_snapshots(
    df: pd.DataFrame,
    max_rows: Optional[int],
    *,
    random_seed: int = 0,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    groups = list(df.groupby("snapshot_date", sort=True))
    if not groups:
        return df.sample(n=max_rows, random_state=random_seed).sort_index()

    rows_per_snapshot = max(10, max_rows // len(groups))
    sampled = []
    for idx, (_snapshot_date, group) in enumerate(groups):
        n_rows = min(len(group), rows_per_snapshot)
        if n_rows <= 0:
            continue
        sampled.append(group.sample(n=n_rows, random_state=random_seed + idx))

    if not sampled:
        return df.sample(n=max_rows, random_state=random_seed).sort_index()

    sample = pd.concat(sampled).sort_values(["snapshot_date", "ticker"], kind="mergesort")
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_seed)
        sample = sample.sort_values(["snapshot_date", "ticker"], kind="mergesort")
    return sample.reset_index(drop=True)


def _snapshot_dates(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("snapshot_date contains invalid values")
    return dates


def _recency_sample_weights(
    df: pd.DataFrame,
    *,
    half_life_days: float = 365.0,
) -> np.ndarray:
    dates = _snapshot_dates(df)
    max_date = pd.Timestamp(dates.max())
    age_days = (max_date - dates).dt.days.to_numpy(dtype="float64")
    weights = np.exp(-np.log(2.0) * age_days / half_life_days)
    mean_weight = float(weights.mean())
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights.astype("float64")


def _walk_forward_folds(
    df: pd.DataFrame,
    *,
    n_folds: int,
    validation_months: int,
    embargo_days: int,
    min_train_snapshots: int = 60,
    min_validation_snapshots: int = 20,
) -> list[dict[str, Any]]:
    if n_folds <= 0 or validation_months <= 0:
        return []

    dates = _snapshot_dates(df)
    unique_dates = pd.Index(sorted(dates.unique()))
    if len(unique_dates) < min_train_snapshots + min_validation_snapshots:
        return []

    max_date = pd.Timestamp(unique_dates[-1])
    folds: list[dict[str, Any]] = []
    for offset in range(n_folds - 1, -1, -1):
        validation_end = max_date - pd.DateOffset(months=validation_months * offset)
        validation_start = validation_end - pd.DateOffset(months=validation_months) + timedelta(days=1)
        train_end_exclusive = validation_start - timedelta(days=embargo_days)

        train_mask = dates < train_end_exclusive
        validation_mask = (dates >= validation_start) & (dates <= validation_end)
        train_snapshots = int(dates.loc[train_mask].nunique())
        validation_snapshots = int(dates.loc[validation_mask].nunique())
        if train_snapshots < min_train_snapshots or validation_snapshots < min_validation_snapshots:
            continue

        folds.append({
            "index": len(folds) + 1,
            "train_end_exclusive": pd.Timestamp(train_end_exclusive),
            "validation_start": pd.Timestamp(validation_start),
            "validation_end": pd.Timestamp(validation_end),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "train_snapshots": train_snapshots,
            "validation_snapshots": validation_snapshots,
        })
    return folds


def _fold_summary(fold: Mapping[str, Any]) -> dict[str, object]:
    return {
        "index": fold["index"],
        "train_end_exclusive": pd.Timestamp(fold["train_end_exclusive"]).date().isoformat(),
        "validation_start": pd.Timestamp(fold["validation_start"]).date().isoformat(),
        "validation_end": pd.Timestamp(fold["validation_end"]).date().isoformat(),
        "train_rows": fold["train_rows"],
        "validation_rows": fold["validation_rows"],
        "train_snapshots": fold["train_snapshots"],
        "validation_snapshots": fold["validation_snapshots"],
    }


def _walk_forward_decision(
    fold_deltas: list[Mapping[str, float]],
    *,
    min_6m_delta: float,
    max_horizon_degradation: float,
    primary_horizon: str = "6m",
) -> dict[str, object]:
    if not fold_deltas:
        return {
            "accepted": False,
            "reason": "no walk-forward folds",
            "mean_deltas": {},
            "median_deltas": {},
            "min_deltas": {},
            "fold_deltas": [],
        }

    horizons = tuple(
        horizon
        for horizon in EVAL_HORIZONS
        if all(horizon in delta for delta in fold_deltas)
    )
    arrays = {
        horizon: np.asarray([float(delta[horizon]) for delta in fold_deltas], dtype="float64")
        for horizon in horizons
    }
    mean_deltas = {horizon: float(values.mean()) for horizon, values in arrays.items()}
    median_deltas = {horizon: float(np.median(values)) for horizon, values in arrays.items()}
    min_deltas = {horizon: float(values.min()) for horizon, values in arrays.items()}

    reason = "accepted"
    if any(not np.isfinite(values).all() for values in arrays.values()):
        reason = "non-finite walk-forward delta"
    elif mean_deltas[primary_horizon] < min_6m_delta:
        reason = f"mean {primary_horizon} delta below walk-forward gate"
    elif median_deltas[primary_horizon] < 0.0:
        reason = f"median {primary_horizon} delta below zero"
    elif any(mean_deltas[horizon] < -max_horizon_degradation for horizon in horizons):
        reason = "mean horizon degradation"
    elif min_deltas[primary_horizon] < -max_horizon_degradation:
        reason = f"fold {primary_horizon} degradation"

    return {
        "accepted": reason == "accepted",
        "reason": reason,
        "primary_horizon": primary_horizon,
        "mean_deltas": {horizon: round(mean_deltas[horizon], 6) for horizon in horizons},
        "median_deltas": {horizon: round(median_deltas[horizon], 6) for horizon in horizons},
        "min_deltas": {horizon: round(min_deltas[horizon], 6) for horizon in horizons},
        "fold_deltas": [
            {horizon: round(float(delta[horizon]), 6) for horizon in horizons}
            for delta in fold_deltas
        ],
    }


def _fit_candidate_model(
    train_snapshots: pd.DataFrame,
    *,
    target_name: str,
    ridge_lambda: float,
    backend: str,
    max_rows: Optional[int],
    prior_strategy: str,
    model_kind: str,
    feature_names: Sequence[str],
    random_seed: int = 0,
) -> tuple[dict[str, object], Dict[str, Dict[str, float]], str, bool]:
    priors = _priors_for_strategy(train_snapshots, prior_strategy)
    prefer_row_priors = prior_strategy == "rolling_ticker_priors"
    train_frame = (
        _attach_asof_ticker_priors(train_snapshots, train_snapshots)
        if prefer_row_priors
        else train_snapshots
    )
    sample = _sample_training_snapshots(train_frame, max_rows, random_seed=random_seed)
    X = build_feature_matrix(
        sample,
        priors,
        prefer_row_priors=prefer_row_priors,
        feature_names=feature_names,
    )
    X_scaled, clip_low, clip_high, mean, scale = _standardize_features(X)
    target = _target_values(sample, target_name)
    valid = np.isfinite(target)
    if int(valid.sum()) < 10:
        raise ValueError(f"not enough valid rows for target {target_name}")
    fit_model_kind = _fit_model_kind_for_candidate(model_kind)
    recency_half_life = _recent_half_life_for_model_kind(model_kind)
    sample_weight = (
        _recency_sample_weights(sample, half_life_days=recency_half_life)
        if recency_half_life is not None
        else None
    )

    if fit_model_kind == "ridge":
        coef_with_intercept, used_backend = _fit_ridge(
            X_scaled[valid],
            target[valid],
            ridge_lambda=ridge_lambda,
            backend=backend,
            sample_weight=sample_weight[valid] if sample_weight is not None else None,
        )
        model = _linear_model_dict(
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            scale=scale,
            coef_with_intercept=coef_with_intercept,
        )
    elif fit_model_kind == "market_ridge":
        models, used_backend = _fit_market_ridge_models(
            X_scaled,
            target,
            sample["ticker"].astype(str),
            ridge_lambda=ridge_lambda,
            backend=backend,
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            scale=scale,
            sample_weight=sample_weight,
        )
        model = {"models": models}
    elif fit_model_kind == "segment_ridge":
        models, used_backend = _fit_segment_ridge_models(
            X_scaled,
            target,
            sample,
            ridge_lambda=ridge_lambda,
            backend=backend,
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            scale=scale,
            sample_weight=sample_weight,
        )
        model = {"models": models}
    elif fit_model_kind == "pairwise":
        coef_with_intercept, used_backend = _fit_pairwise_ranker_numpy(
            X_scaled,
            target,
            _snapshot_dates(sample),
            ridge_lambda=ridge_lambda,
        )
        model = _linear_model_dict(
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            scale=scale,
            coef_with_intercept=coef_with_intercept,
        )
    else:
        raise ValueError(f"Unsupported ML model kind: {model_kind}")
    return model, priors, used_backend, prefer_row_priors


def _predict_from_linear_model(
    df: pd.DataFrame,
    priors: Mapping[str, Mapping[str, float]],
    model: Mapping[str, object],
    *,
    chunk_size: int = 200_000,
    prefer_row_priors: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    models = model.get("models")
    if isinstance(models, list) and models:
        return _predict_from_linear_model_payloads(
            df,
            priors,
            [entry for entry in models if isinstance(entry, Mapping)],
            chunk_size=chunk_size,
            prefer_row_priors=prefer_row_priors,
            feature_names=feature_names,
        )
    return _predict_in_chunks(
        df,
        ticker_priors=priors,
        clip_low=np.asarray(model["clip_low"], dtype="float64"),
        clip_high=np.asarray(model["clip_high"], dtype="float64"),
        mean=np.asarray(model["mean"], dtype="float64"),
        scale=np.asarray(model["scale"], dtype="float64"),
        coef=np.asarray(model["coef"], dtype="float64"),
        intercept=float(model["intercept"]),
        chunk_size=chunk_size,
        prefer_row_priors=prefer_row_priors,
        feature_names=feature_names,
    )


def _incumbent_predictions(
    df: pd.DataFrame,
    incumbent_payload: Optional[Mapping[str, object]],
    *,
    chunk_size: int = 200_000,
) -> np.ndarray:
    if incumbent_payload is not None:
        return predict_ml_ranker_payload(df, incumbent_payload, chunk_size=chunk_size)
    return pd.to_numeric(df["composite_score"], errors="coerce").to_numpy(dtype="float64")


def _payload_with_ticker_priors(
    payload: Optional[Mapping[str, object]],
    ticker_priors: Mapping[str, Mapping[str, float]],
) -> Optional[dict[str, object]]:
    if payload is None:
        return None

    def patch_value(value: object) -> object:
        if isinstance(value, Mapping):
            patched = {key: patch_value(child) for key, child in value.items()}
            if "ticker_priors" in patched:
                patched["ticker_priors"] = ticker_priors
            return patched
        if isinstance(value, list):
            return [patch_value(child) for child in value]
        return value

    patched_payload = patch_value(payload)
    if not isinstance(patched_payload, dict):
        return None
    patched_payload["ticker_priors"] = ticker_priors
    return patched_payload


def _walk_forward_validate_candidate(
    snapshots: pd.DataFrame,
    folds: list[Mapping[str, Any]],
    *,
    target_name: str,
    ridge_lambda: float,
    backend: str,
    max_rows: Optional[int],
    prior_strategy: str,
    model_kind: str,
    feature_names: Sequence[str],
    incumbent_payload: Optional[Mapping[str, object]],
    min_6m_delta: float,
    max_horizon_degradation: float,
    primary_horizon: str = "6m",
    blend_weights: tuple[float, ...] = BLEND_WEIGHTS,
    quality_residual_weights: tuple[float, ...] = (1.0,),
    require_primary_top20_excess_non_degradation: bool = False,
    sample_seed: int = 0,
) -> dict[str, object]:
    if not folds:
        return {"accepted": True, "reason": "walk-forward disabled", "blend_weight": 1.0}

    dates = _snapshot_dates(snapshots)
    fold_predictions: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    for fold in folds:
        train_mask = dates < pd.Timestamp(fold["train_end_exclusive"])
        validation_mask = (dates >= pd.Timestamp(fold["validation_start"])) & (
            dates <= pd.Timestamp(fold["validation_end"])
        )
        train = snapshots.loc[train_mask]
        validation = _monthly_rebalance_snapshots(snapshots.loc[validation_mask])
        incumbent_for_fold = _payload_with_ticker_priors(
            incumbent_payload,
            _ticker_priors(train),
        )
        model, priors, _used_backend, _prefer_row_priors = _fit_candidate_model(
            train,
            target_name=target_name,
            ridge_lambda=ridge_lambda,
            backend=backend,
            max_rows=max_rows,
            prior_strategy=prior_strategy,
            model_kind=model_kind,
            feature_names=feature_names,
            random_seed=sample_seed + int(fold["index"]) * 10_000,
        )
        candidate_predictions = _predict_from_linear_model(
            validation,
            priors,
            model,
            prefer_row_priors=False,
            feature_names=feature_names,
        )
        incumbent_predictions = _incumbent_predictions(
            validation,
            incumbent_for_fold,
        )
        fold_predictions.append(
            (
                validation,
                _prediction_rank_by_snapshot(validation, candidate_predictions),
                _prediction_rank_by_snapshot(validation, incumbent_predictions),
            )
        )

    blend_results: list[dict[str, object]] = []
    for blend_weight in dict.fromkeys(float(weight) for weight in blend_weights):
        fold_deltas: list[Mapping[str, float]] = []
        for validation, candidate_predictions, incumbent_predictions in fold_predictions:
            blended = blend_weight * candidate_predictions + (1.0 - blend_weight) * incumbent_predictions
            candidate_metrics = _evaluate_predictions(validation, blended)
            incumbent_metrics = _evaluate_predictions(validation, incumbent_predictions)
            fold_deltas.append(
                {
                    horizon: (
                        float(candidate_metrics[horizon]["spearman_rho"])
                        - float(incumbent_metrics[horizon]["spearman_rho"])
                    )
                    for horizon in EVAL_HORIZONS
                }
            )

        decision = _walk_forward_decision(
            fold_deltas,
            min_6m_delta=min_6m_delta,
            max_horizon_degradation=max_horizon_degradation,
            primary_horizon=primary_horizon,
        )
        decision["blend_weight"] = float(blend_weight)
        decision["blend_scale"] = "snapshot_rank"
        blend_results.append(decision)

    if not blend_results:
        return {
            "accepted": False,
            "reason": "no walk-forward folds",
            "blend_weight": 1.0,
        }
    accepted = [result for result in blend_results if result.get("accepted")]
    selection_pool = accepted or blend_results

    def selection_key(result: Mapping[str, object]) -> tuple[float, float, float]:
        mean_deltas = result.get("mean_deltas")
        median_deltas = result.get("median_deltas")
        min_deltas = result.get("min_deltas")
        return (
            float(mean_deltas.get(primary_horizon, float("-inf")))
            if isinstance(mean_deltas, Mapping)
            else float("-inf"),
            float(median_deltas.get(primary_horizon, float("-inf")))
            if isinstance(median_deltas, Mapping)
            else float("-inf"),
            float(min_deltas.get(primary_horizon, float("-inf"))) if isinstance(min_deltas, Mapping) else float("-inf"),
        )

    best_result = max(selection_pool, key=selection_key)
    best_result["blend_candidates_evaluated"] = len(blend_results)
    base_blend_weight = float(best_result["blend_weight"])
    application_results: list[tuple[dict[str, object], float, float]] = []

    if best_result.get("accepted"):
        base_primary_rhos = []
        for validation, candidate_predictions, incumbent_predictions in fold_predictions:
            base_predictions = (
                base_blend_weight * candidate_predictions
                + (1.0 - base_blend_weight) * incumbent_predictions
            )
            base_metrics = _evaluate_predictions(validation, base_predictions)
            base_primary_rhos.append(float(base_metrics[primary_horizon]["spearman_rho"]))
        application_results.append(
            (
                best_result,
                float(np.min(base_primary_rhos)),
                float(np.mean(base_primary_rhos)),
            )
        )

    def add_application_result(
        result: Optional[dict[str, object]],
        *,
        min_rhos_key: str,
        mean_rhos_key: str,
    ) -> None:
        if not result or not result.get("accepted"):
            return
        min_rhos = result.get(min_rhos_key)
        mean_rhos = result.get(mean_rhos_key)
        if isinstance(min_rhos, Mapping) and isinstance(mean_rhos, Mapping):
            min_rho = float(min_rhos.get(primary_horizon, float("-inf")))
            mean_rho = float(mean_rhos.get(primary_horizon, float("-inf")))
        else:
            mean_deltas = result.get("mean_deltas")
            mean_rho = (
                float(mean_deltas.get(primary_horizon, float("-inf")))
                if isinstance(mean_deltas, Mapping)
                else float("-inf")
            )
            min_rho = mean_rho
        application_results.append((result, min_rho, mean_rho))

    if primary_horizon == "6m":
        factor_result = _select_practical_factor_blend(
            fold_predictions,
            base_blend_weight=base_blend_weight,
            factor_templates=SIX_MONTH_APPLICATION_FACTOR_TEMPLATES,
            min_6m_delta=min_6m_delta,
            max_horizon_degradation=max_horizon_degradation,
            primary_horizon=primary_horizon,
            blend_candidates_evaluated=len(blend_results),
            require_primary_top20_excess_non_degradation=(
                require_primary_top20_excess_non_degradation
            ),
        )
        add_application_result(
            factor_result,
            min_rhos_key="application_factor_min_rhos",
            mean_rhos_key="application_factor_mean_rhos",
        )
        if factor_result:
            logger.info(
                "ML walk-forward practical factor candidate template=%s min_rhos=%s mean_rhos=%s accepted=%s",
                factor_result.get("application_factor_template"),
                factor_result.get("application_factor_min_rhos"),
                factor_result.get("application_factor_mean_rhos"),
                factor_result.get("accepted"),
            )
    if (
        primary_horizon == "6m"
        and any(column in snapshots.columns for _name, column, _direction in SIX_MONTH_APPLICATION_RESIDUAL_SIGNALS)
        and any(float(weight) < 1.0 for weight in quality_residual_weights)
    ):
        residual_result = _select_application_residual_blend(
            fold_predictions,
            base_blend_weight=base_blend_weight,
            residual_weights=quality_residual_weights,
            residual_signals=SIX_MONTH_APPLICATION_RESIDUAL_SIGNALS,
            min_6m_delta=min_6m_delta,
            max_horizon_degradation=max_horizon_degradation,
            primary_horizon=primary_horizon,
            blend_candidates_evaluated=len(blend_results),
            require_primary_top20_excess_non_degradation=(require_primary_top20_excess_non_degradation),
        )
        add_application_result(
            residual_result,
            min_rhos_key="application_residual_min_rhos",
            mean_rhos_key="application_residual_mean_rhos",
        )
    if application_results:
        non_factor_results = [
            item
            for item in application_results
            if not item[0].get("application_factor_components")
        ]
        if non_factor_results:
            residual_results = [
                item
                for item in non_factor_results
                if item[0].get("application_residual_signal")
            ]
            reference_result = max(
                residual_results or non_factor_results,
                key=lambda item: (item[2], item[1]),
            )
            reference_mean_rho = reference_result[2]
            factor_results = [
                item
                for item in application_results
                if item[0].get("application_factor_components")
                and item[2]
                >= reference_mean_rho
                + SIX_MONTH_APPLICATION_FACTOR_MIN_MEAN_RHO_IMPROVEMENT
            ]
            application_results = [reference_result, *factor_results]
        best_min_rho = max(min_rho for _result, min_rho, _mean_rho in application_results)
        robust_results = [
            item
            for item in application_results
            if item[1] >= best_min_rho - SIX_MONTH_APPLICATION_FACTOR_MIN_RHO_TOLERANCE
        ]
        best_result = max(
            robust_results,
            key=lambda item: (
                item[2],
                item[1],
                selection_key(item[0]),
            ),
        )[0]
        logger.info(
            "ML walk-forward application selection factor=%s residual=%s weight=%s",
            best_result.get("application_factor_template"),
            best_result.get("application_residual_signal"),
            best_result.get("application_residual_candidate_weight"),
        )
    return best_result or {
        "accepted": False,
        "reason": "no walk-forward folds",
        "blend_weight": 1.0,
    }


def _select_practical_factor_blend(
    fold_predictions: Sequence[tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    *,
    base_blend_weight: float,
    factor_templates: Sequence[tuple[str, Sequence[float]]],
    min_6m_delta: float,
    max_horizon_degradation: float,
    primary_horizon: str,
    blend_candidates_evaluated: int,
    require_primary_top20_excess_non_degradation: bool = False,
) -> Optional[dict[str, object]]:
    """Select a diversified, live-available factor overlay on purged folds."""
    required_columns = {
        "quality_score",
        "revenue",
        "gross_margin",
        "total_assets",
        "roe",
        "roa",
        "pb_ratio",
        "ps_ratio",
        "total_liabilities",
        "market_cap_rmb",
        "relative_return_63d",
        "current_ratio",
    }
    if not fold_predictions or not all(
        required_columns.issubset(validation.columns)
        for validation, _candidate, _incumbent in fold_predictions
    ):
        return None

    factor_results: list[dict[str, object]] = []
    for template_name, raw_weights in factor_templates:
        if len(raw_weights) != len(SIX_MONTH_APPLICATION_FACTOR_SIGNALS):
            continue
        components = [
            {"signal": signal, "weight": float(weight)}
            for signal, weight in zip(SIX_MONTH_APPLICATION_FACTOR_SIGNALS, raw_weights)
            if np.isfinite(float(weight)) and float(weight) > 0.0
        ]
        if not components:
            continue

        fold_deltas: list[Mapping[str, float]] = []
        fold_primary_rhos: list[float] = []
        fold_horizon_rhos: dict[str, list[float]] = {horizon: [] for horizon in EVAL_HORIZONS}
        fold_practical_passes: list[bool] = []
        for validation, candidate_predictions, incumbent_predictions in fold_predictions:
            base_predictions = (
                base_blend_weight * candidate_predictions + (1.0 - base_blend_weight) * incumbent_predictions
            )
            blended = _application_factor_predictions(validation, base_predictions, components)
            candidate_metrics = _evaluate_predictions(validation, blended)
            incumbent_metrics = _evaluate_predictions(validation, incumbent_predictions)
            fold_deltas.append(
                {
                    horizon: (
                        float(candidate_metrics[horizon]["spearman_rho"])
                        - float(incumbent_metrics[horizon]["spearman_rho"])
                    )
                    for horizon in EVAL_HORIZONS
                }
            )
            for horizon in EVAL_HORIZONS:
                fold_horizon_rhos[horizon].append(float(candidate_metrics[horizon]["spearman_rho"]))
            fold_primary_rhos.append(float(candidate_metrics[primary_horizon]["spearman_rho"]))
            incumbent_top20_excess = _primary_top20_excess_metrics(
                validation,
                incumbent_predictions,
                primary_horizon,
            )
            fold_practical_passes.append(
                _primary_top20_excess_predictions_non_degraded(
                    validation,
                    blended,
                    incumbent_top20_excess,
                    primary_horizon,
                )
            )

        decision = _walk_forward_decision(
            fold_deltas,
            min_6m_delta=min_6m_delta,
            max_horizon_degradation=max_horizon_degradation,
            primary_horizon=primary_horizon,
        )
        if require_primary_top20_excess_non_degradation and not all(fold_practical_passes):
            decision["accepted"] = False
            decision["reason"] = "fold 6m top20 and excess return degraded"
        decision.update(
            {
                "blend_weight": float(base_blend_weight),
                "blend_scale": "snapshot_rank",
                "blend_candidates_evaluated": int(blend_candidates_evaluated),
                "application_factor_template": str(template_name),
                "application_factor_components": components,
                "application_factor_scale": "snapshot_rank",
                "application_factor_fold_primary_rhos": [round(value, 6) for value in fold_primary_rhos],
                "application_factor_fold_practical_passes": fold_practical_passes,
                "application_factor_mean_rhos": {
                    horizon: round(float(np.mean(values)), 6) for horizon, values in fold_horizon_rhos.items()
                },
                "application_factor_min_rhos": {
                    horizon: round(float(np.min(values)), 6) for horizon, values in fold_horizon_rhos.items()
                },
            }
        )
        factor_results.append(decision)

    accepted = [result for result in factor_results if result.get("accepted")]
    if not accepted:
        return None
    best_min_rho = max(
        float(result["application_factor_min_rhos"][primary_horizon])
        for result in accepted
    )
    robust = [
        result
        for result in accepted
        if float(result["application_factor_min_rhos"][primary_horizon])
        >= best_min_rho - SIX_MONTH_APPLICATION_FACTOR_MIN_RHO_TOLERANCE
    ]
    selected = max(
        robust,
        key=lambda result: (
            float(result["application_factor_mean_rhos"][primary_horizon]),
            float(result["application_factor_min_rhos"][primary_horizon]),
            str(result["application_factor_template"]),
        ),
    )
    selected["application_factor_candidates_evaluated"] = len(factor_results)
    return selected


def _select_application_residual_blend(
    fold_predictions: Sequence[tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    *,
    base_blend_weight: float,
    residual_weights: Sequence[float],
    residual_signals: Sequence[tuple[str, str, float]],
    min_6m_delta: float,
    max_horizon_degradation: float,
    primary_horizon: str,
    blend_candidates_evaluated: int,
    require_primary_top20_excess_non_degradation: bool = False,
) -> dict[str, object]:
    """Select a live-available calibration signal on purged inner folds."""
    residual_results: list[dict[str, object]] = []
    for signal_name, signal_column, signal_direction in residual_signals:
        if not signal_name or not signal_column or not np.isfinite(signal_direction):
            continue
        if not all(signal_column in validation.columns for validation, _candidate, _incumbent in fold_predictions):
            continue
        for residual_weight in dict.fromkeys(float(weight) for weight in residual_weights):
            if not 0.0 <= residual_weight <= 1.0:
                continue
            fold_deltas: list[Mapping[str, float]] = []
            fold_primary_rhos: list[float] = []
            fold_horizon_rhos: dict[str, list[float]] = {horizon: [] for horizon in EVAL_HORIZONS}
            fold_practical_passes: list[bool] = []
            for validation, candidate_predictions, incumbent_predictions in fold_predictions:
                base_predictions = (
                    base_blend_weight * candidate_predictions + (1.0 - base_blend_weight) * incumbent_predictions
                )
                signal_values = signal_direction * pd.to_numeric(
                    validation[signal_column],
                    errors="coerce",
                ).to_numpy(dtype="float64")
                signal_rank = _prediction_rank_by_snapshot(validation, signal_values)
                blended = base_predictions.copy()
                usable = np.isfinite(base_predictions) & np.isfinite(signal_rank)
                blended[usable] = (
                    residual_weight * base_predictions[usable] + (1.0 - residual_weight) * signal_rank[usable]
                )
                candidate_metrics = _evaluate_predictions(validation, blended)
                incumbent_metrics = _evaluate_predictions(validation, incumbent_predictions)
                fold_deltas.append(
                    {
                        horizon: (
                            float(candidate_metrics[horizon]["spearman_rho"])
                            - float(incumbent_metrics[horizon]["spearman_rho"])
                        )
                        for horizon in EVAL_HORIZONS
                    }
                )
                for horizon in EVAL_HORIZONS:
                    fold_horizon_rhos[horizon].append(float(candidate_metrics[horizon]["spearman_rho"]))
                fold_primary_rhos.append(float(candidate_metrics[primary_horizon]["spearman_rho"]))
                incumbent_top20_excess = _primary_top20_excess_metrics(
                    validation,
                    incumbent_predictions,
                    primary_horizon,
                )
                fold_practical_passes.append(
                    _primary_top20_excess_predictions_non_degraded(
                        validation,
                        blended,
                        incumbent_top20_excess,
                        primary_horizon,
                    )
                )

            decision = _walk_forward_decision(
                fold_deltas,
                min_6m_delta=min_6m_delta,
                max_horizon_degradation=max_horizon_degradation,
                primary_horizon=primary_horizon,
            )
            if require_primary_top20_excess_non_degradation and not all(fold_practical_passes):
                decision["accepted"] = False
                decision["reason"] = "fold 6m top20 and excess return degraded"
            decision.update(
                {
                    "blend_weight": float(base_blend_weight),
                    "blend_scale": "snapshot_rank",
                    "blend_candidates_evaluated": int(blend_candidates_evaluated),
                    "application_residual_candidate_weight": float(residual_weight),
                    "application_residual_signal": str(signal_name),
                    "application_residual_column": str(signal_column),
                    "application_residual_direction": float(signal_direction),
                    "application_residual_scale": "snapshot_rank",
                    "application_residual_fold_primary_rhos": [round(value, 6) for value in fold_primary_rhos],
                    "application_residual_fold_practical_passes": fold_practical_passes,
                    "application_residual_mean_rhos": {
                        horizon: round(float(np.mean(values)), 6) for horizon, values in fold_horizon_rhos.items()
                    },
                    "application_residual_min_rhos": {
                        horizon: round(float(np.min(values)), 6) for horizon, values in fold_horizon_rhos.items()
                    },
                }
            )
            if signal_name == "quality_de_crowding":
                decision.update(
                    {
                        "quality_residual_candidate_weight": float(residual_weight),
                        "quality_residual_direction": "inverse",
                        "quality_residual_scale": "snapshot_rank",
                        "quality_residual_fold_primary_rhos": decision["application_residual_fold_primary_rhos"],
                        "quality_residual_mean_rhos": decision["application_residual_mean_rhos"],
                        "quality_residual_min_rhos": decision["application_residual_min_rhos"],
                    }
                )
            residual_results.append(decision)

    if not residual_results:
        return {
            "accepted": False,
            "reason": "no application residual candidates",
            "blend_weight": float(base_blend_weight),
        }
    accepted = [result for result in residual_results if result.get("accepted")]
    selection_pool = accepted or residual_results

    def residual_key(result: Mapping[str, object]) -> tuple[float, float, float]:
        min_rhos = result.get("application_residual_min_rhos")
        mean_rhos = result.get("application_residual_mean_rhos")
        mean_deltas = result.get("mean_deltas")
        return (
            float(min_rhos.get(primary_horizon, float("-inf"))) if isinstance(min_rhos, Mapping) else float("-inf"),
            float(mean_rhos.get(primary_horizon, float("-inf"))) if isinstance(mean_rhos, Mapping) else float("-inf"),
            float(mean_deltas.get(primary_horizon, float("-inf")))
            if isinstance(mean_deltas, Mapping)
            else float("-inf"),
        )

    selected = max(selection_pool, key=residual_key)
    selected["application_residual_candidates_evaluated"] = len(residual_results)
    if selected.get("application_residual_signal") == "quality_de_crowding":
        selected["quality_residual_candidates_evaluated"] = len(residual_results)
    return selected


def _select_quality_residual_blend(
    fold_predictions: Sequence[tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    *,
    base_blend_weight: float,
    residual_weights: Sequence[float],
    min_6m_delta: float,
    max_horizon_degradation: float,
    primary_horizon: str,
    blend_candidates_evaluated: int,
) -> dict[str, object]:
    """Backward-compatible wrapper for the original quality-only selector."""
    return _select_application_residual_blend(
        fold_predictions,
        base_blend_weight=base_blend_weight,
        residual_weights=residual_weights,
        residual_signals=(("quality_de_crowding", "quality_score", -1.0),),
        min_6m_delta=min_6m_delta,
        max_horizon_degradation=max_horizon_degradation,
        primary_horizon=primary_horizon,
        blend_candidates_evaluated=blend_candidates_evaluated,
    )


def _feature_names_need_cross_sectional(
    feature_names: Optional[Sequence[str]],
) -> bool:
    return any(
        str(name).startswith(("cs_rank_", "market_cs_rank_"))
        for name in (feature_names or ())
    )


def _prediction_index_chunks(
    df: pd.DataFrame,
    *,
    chunk_size: int,
    feature_names: Optional[Sequence[str]],
) -> list[np.ndarray]:
    if len(df) == 0:
        return []
    if (
        chunk_size <= 0
        or not _feature_names_need_cross_sectional(feature_names)
        or "snapshot_date" not in df.columns
    ):
        step = chunk_size if chunk_size > 0 else len(df)
        return [
            np.arange(start, min(start + step, len(df)), dtype=int)
            for start in range(0, len(df), step)
        ]

    snapshot_keys = pd.to_datetime(df["snapshot_date"], errors="coerce").astype(str)
    group_frame = pd.DataFrame({"snapshot": snapshot_keys.reset_index(drop=True)})
    chunks: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_size = 0
    for index in group_frame.groupby("snapshot", sort=False).groups.values():
        positions = np.asarray(index, dtype=int)
        if pending and pending_size + len(positions) > chunk_size:
            chunks.append(np.concatenate(pending))
            pending = []
            pending_size = 0
        pending.append(positions)
        pending_size += len(positions)
    if pending:
        chunks.append(np.concatenate(pending))
    return chunks


def _predict_in_chunks(
    df: pd.DataFrame,
    *,
    ticker_priors: Mapping[str, Mapping[str, float]],
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    chunk_size: int = 200_000,
    prefer_row_priors: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    predictions = np.empty(len(df), dtype="float64")
    for index in _prediction_index_chunks(
        df,
        chunk_size=chunk_size,
        feature_names=feature_names,
    ):
        X = build_feature_matrix(
            df.iloc[index],
            ticker_priors,
            prefer_row_priors=prefer_row_priors,
            feature_names=feature_names,
        )
        X_scaled = (np.clip(X, clip_low, clip_high) - mean) / scale
        predictions[index] = X_scaled @ coef + intercept
    return predictions


def _linear_payloads(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    models = payload.get("models")
    if isinstance(models, list) and models:
        return [model for model in models if isinstance(model, Mapping)]
    if (
        isinstance(payload.get("payload_members"), list)
        or isinstance(payload.get("rank_payload_members"), list)
        or isinstance(payload.get("market_payload_members"), list)
        or isinstance(payload.get("temporal_payload_members"), list)
        or isinstance(payload.get("conditional_blend"), Mapping)
        or isinstance(payload.get("lightgbm_model"), str)
    ):
        return []
    return [payload]


def _payload_temporal_timestamp(value: object) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _payload_temporal_member_payloads(
    payload: Mapping[str, object],
) -> list[tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Mapping[str, object]]]:
    members = payload.get("temporal_payload_members")
    if not isinstance(members, list):
        return []
    output: list[tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Mapping[str, object]]] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        member_payload = member.get("payload")
        if not isinstance(member_payload, Mapping):
            continue
        output.append((
            _payload_temporal_timestamp(member.get("start_date")),
            _payload_temporal_timestamp(member.get("end_date")),
            member_payload,
        ))
    return output


def _payload_child_payloads(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    children: list[Mapping[str, object]] = []
    for member_key in (
        "payload_members",
        "rank_payload_members",
        "market_payload_members",
        "temporal_payload_members",
    ):
        members = payload.get(member_key)
        if not isinstance(members, list):
            continue
        children.extend(
            member_payload
            for member in members
            if isinstance(member, Mapping)
            and isinstance((member_payload := member.get("payload")), Mapping)
        )
    conditional_blend = payload.get("conditional_blend")
    if isinstance(conditional_blend, Mapping):
        children.extend(
            child
            for key in ("base_payload", "candidate_payload")
            if isinstance((child := conditional_blend.get(key)), Mapping)
        )
    return children


def _payload_runtime_includes_evaluation_labels(payload: Mapping[str, object]) -> bool:
    """Return whether a runtime payload was refit using its evaluation period."""
    metadata = payload.get("metadata")
    provenance_sources = (payload, metadata) if isinstance(metadata, Mapping) else (payload,)
    for provenance in provenance_sources:
        if provenance.get("application_refit") is True:
            return True
        deployment_refit = provenance.get("deployment_refit")
        if deployment_refit is True or isinstance(deployment_refit, Mapping):
            return True
        evaluation_protocol = str(provenance.get("evaluation_protocol", "")).lower()
        promotion_status = str(provenance.get("promotion_status", "")).lower()
        if "full_refit" in evaluation_protocol:
            return True
        if "full_data_refit" in promotion_status or "full_refit" in promotion_status:
            return True
    return any(
        _payload_runtime_includes_evaluation_labels(child)
        for child in _payload_child_payloads(payload)
    )


def _payload_cached_metrics_are_safe(payload: Mapping[str, object]) -> bool:
    """Return whether top-level cached metrics are valid frozen evidence."""
    if _payload_runtime_includes_evaluation_labels(payload):
        return False
    members = payload.get("temporal_payload_members")
    if isinstance(members, list):
        valid_members = [
            member
            for member in members
            if isinstance(member, Mapping) and isinstance(member.get("payload"), Mapping)
        ]
        if len(valid_members) == 1:
            return False
    return all(_payload_cached_metrics_are_safe(child) for child in _payload_child_payloads(payload))


def _payload_member_payloads(payload: Mapping[str, object]) -> list[tuple[float, Mapping[str, object]]]:
    members = payload.get("payload_members")
    if not isinstance(members, list):
        return []
    output: list[tuple[float, Mapping[str, object]]] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        member_payload = member.get("payload")
        if not isinstance(member_payload, Mapping):
            continue
        try:
            weight = float(member.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        output.append((weight, member_payload))
    return output


def _payload_rank_member_payloads(
    payload: Mapping[str, object],
) -> list[tuple[float, Mapping[str, object]]]:
    members = payload.get("rank_payload_members")
    if not isinstance(members, list):
        return []
    output: list[tuple[float, Mapping[str, object]]] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        member_payload = member.get("payload")
        if not isinstance(member_payload, Mapping):
            continue
        try:
            weight = float(member.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        output.append((weight, member_payload))
    return output


def _payload_market_member_payloads(
    payload: Mapping[str, object],
) -> list[tuple[dict[str, float], Mapping[str, object]]]:
    members = payload.get("market_payload_members")
    if not isinstance(members, list):
        return []
    output: list[tuple[dict[str, float], Mapping[str, object]]] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        member_payload = member.get("payload")
        if not isinstance(member_payload, Mapping):
            continue
        raw_weights = member.get("market_weights")
        if not isinstance(raw_weights, Mapping):
            continue
        weights: dict[str, float] = {}
        for market, weight in raw_weights.items():
            try:
                value = float(weight)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                weights[str(market)] = value
        if weights:
            output.append((weights, member_payload))
    return output


def _payload_conditional_blend(
    payload: Mapping[str, object],
) -> Optional[tuple[Mapping[str, object], Mapping[str, object], tuple[Mapping[str, object], ...]]]:
    blend = payload.get("conditional_blend")
    if not isinstance(blend, Mapping):
        return None
    base_payload = blend.get("base_payload")
    candidate_payload = blend.get("candidate_payload")
    raw_routes = blend.get("routes")
    if (
        not isinstance(base_payload, Mapping)
        or not isinstance(candidate_payload, Mapping)
        or not isinstance(raw_routes, list)
    ):
        return None
    routes = tuple(route for route in raw_routes if isinstance(route, Mapping))
    if not routes:
        return None
    return base_payload, candidate_payload, routes


def _payload_weight(model: Mapping[str, object]) -> float:
    try:
        return float(model.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _payload_market(model: Mapping[str, object]) -> Optional[str]:
    market = model.get("market")
    return str(market) if market is not None else None


def _payload_cap_bucket(model: Mapping[str, object]) -> Optional[str]:
    cap_bucket = model.get("cap_bucket")
    return str(cap_bucket) if cap_bucket is not None else None


def _payload_board_bucket(model: Mapping[str, object]) -> Optional[str]:
    board_bucket = model.get("board_bucket")
    return str(board_bucket) if board_bucket is not None else None


def _payload_listing_bucket(model: Mapping[str, object]) -> Optional[str]:
    listing_bucket = model.get("listing_bucket")
    return str(listing_bucket) if listing_bucket is not None else None


def _linear_model_payload(model: Mapping[str, object], *, weight: float) -> dict[str, object]:
    payload = {
        "clip_low": np.asarray(model["clip_low"], dtype="float64").tolist(),
        "clip_high": np.asarray(model["clip_high"], dtype="float64").tolist(),
        "mean": np.asarray(model["mean"], dtype="float64").tolist(),
        "scale": np.asarray(model["scale"], dtype="float64").tolist(),
        "coef": np.asarray(model["coef"], dtype="float64").tolist(),
        "intercept": model["intercept"],
        "weight": float(weight),
    }
    market = _payload_market(model)
    if market is not None:
        payload["market"] = market
    cap_bucket = _payload_cap_bucket(model)
    if cap_bucket is not None:
        payload["cap_bucket"] = cap_bucket
    board_bucket = _payload_board_bucket(model)
    if board_bucket is not None:
        payload["board_bucket"] = board_bucket
    listing_bucket = _payload_listing_bucket(model)
    if listing_bucket is not None:
        payload["listing_bucket"] = listing_bucket
    if model.get("backend") is not None:
        payload["backend"] = model["backend"]
    return payload


def _candidate_model_payloads(
    candidate: Mapping[str, object],
    *,
    weight_multiplier: float = 1.0,
) -> list[dict[str, object]]:
    models = candidate.get("models")
    if isinstance(models, list) and models:
        return [
            _linear_model_payload(
                model,
                weight=weight_multiplier * _payload_weight(model),
            )
            for model in models
            if isinstance(model, Mapping)
        ]
    return [
        _linear_model_payload(
            {
                "clip_low": candidate["clip_low"],
                "clip_high": candidate["clip_high"],
                "mean": candidate["mean"],
                "scale": candidate["scale"],
                "coef": candidate["coef"],
                "intercept": candidate["intercept"],
                "backend": candidate.get("backend"),
            },
            weight=weight_multiplier,
        )
    ]


def _candidate_search_name(candidate: Mapping[str, object]) -> str:
    return (
        f"{candidate.get('target')}/lambda={float(candidate.get('ridge_lambda', 0.0)):g}/"
        f"rows={int(candidate.get('sample_rows', 0))}/"
        f"seed={int(candidate.get('sample_seed', 0))}/"
        f"priors={candidate.get('prior_strategy')}/"
        f"model={candidate.get('model_kind')}/"
        f"features={candidate.get('feature_set')}"
    )


def _predict_from_linear_model_payloads(
    df: pd.DataFrame,
    ticker_priors: Mapping[str, Mapping[str, float]],
    models: list[Mapping[str, object]],
    *,
    chunk_size: int = 200_000,
    prefer_row_priors: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    predictions = np.zeros(len(df), dtype="float64")
    unweighted_predictions = np.zeros(len(df), dtype="float64")
    total_weights = np.zeros(len(df), dtype="float64")
    prediction_counts = np.zeros(len(df), dtype="float64")
    markets, board_buckets, listing_buckets = _ticker_bucket_arrays(df)
    cap_buckets = _cap_bucket_array(df)
    market_codes, market_lookup = _encode_bucket_array(markets)
    board_codes, board_lookup = _encode_bucket_array(board_buckets)
    listing_codes, listing_lookup = _encode_bucket_array(listing_buckets)
    cap_codes, cap_lookup = _encode_bucket_array(cap_buckets)

    if _feature_names_need_cross_sectional(feature_names):
        for index in _prediction_index_chunks(
            df,
            chunk_size=chunk_size,
            feature_names=feature_names,
        ):
            chunk_df = df.iloc[index]
            X = build_feature_matrix(
                chunk_df,
                ticker_priors,
                prefer_row_priors=prefer_row_priors,
                feature_names=feature_names,
            )
            chunk_market_codes = market_codes[index]
            chunk_board_codes = board_codes[index]
            chunk_listing_codes = listing_codes[index]
            chunk_cap_codes = cap_codes[index]
            chunk_weighted = np.zeros(len(index), dtype="float64")
            chunk_unweighted = np.zeros(len(index), dtype="float64")
            chunk_total_weights = np.zeros(len(index), dtype="float64")
            chunk_prediction_counts = np.zeros(len(index), dtype="float64")

            for model in models:
                market = _payload_market(model)
                if market is not None:
                    market_code = market_lookup.get(str(market))
                    mask = (
                        chunk_market_codes == market_code
                        if market_code is not None
                        else np.zeros(len(index), dtype=bool)
                    )
                else:
                    mask = np.ones(len(index), dtype=bool)
                cap_bucket = _payload_cap_bucket(model)
                if cap_bucket is not None:
                    cap_code = cap_lookup.get(str(cap_bucket))
                    if cap_code is None:
                        continue
                    mask &= chunk_cap_codes == cap_code
                board_bucket = _payload_board_bucket(model)
                if board_bucket is not None:
                    board_code = board_lookup.get(str(board_bucket))
                    if board_code is None:
                        continue
                    mask &= chunk_board_codes == board_code
                listing_bucket = _payload_listing_bucket(model)
                if listing_bucket is not None:
                    listing_code = listing_lookup.get(str(listing_bucket))
                    if listing_code is None:
                        continue
                    mask &= chunk_listing_codes == listing_code
                if not mask.any():
                    continue
                clip_low = np.asarray(model["clip_low"], dtype="float64")
                clip_high = np.asarray(model["clip_high"], dtype="float64")
                mean = np.asarray(model["mean"], dtype="float64")
                scale = np.asarray(model["scale"], dtype="float64")
                coef = np.asarray(model["coef"], dtype="float64")
                X_scaled = (np.clip(X[mask], clip_low, clip_high) - mean) / scale
                model_predictions = X_scaled @ coef + float(model["intercept"])
                weight = _payload_weight(model)
                chunk_weighted[mask] += weight * model_predictions
                chunk_unweighted[mask] += model_predictions
                chunk_total_weights[mask] += weight
                chunk_prediction_counts[mask] += 1.0

            chunk_output = np.zeros(len(index), dtype="float64")
            weighted = chunk_total_weights > 0
            chunk_output[weighted] = chunk_weighted[weighted] / chunk_total_weights[weighted]
            unweighted = ~weighted & (chunk_prediction_counts > 0)
            chunk_output[unweighted] = chunk_unweighted[unweighted] / chunk_prediction_counts[unweighted]
            fallback = ~(weighted | unweighted)
            if fallback.any():
                chunk_output[fallback] = np.mean(
                    [
                        (
                            np.clip(
                                X[fallback],
                                np.asarray(model["clip_low"], dtype="float64"),
                                np.asarray(model["clip_high"], dtype="float64"),
                            )
                            - np.asarray(model["mean"], dtype="float64")
                        )
                        / np.asarray(model["scale"], dtype="float64")
                        @ np.asarray(model["coef"], dtype="float64")
                        + float(model["intercept"])
                        for model in models
                    ],
                    axis=0,
                )
            predictions[index] = chunk_output
        return predictions

    for model in models:
        market = _payload_market(model)
        if market is not None:
            market_code = market_lookup.get(str(market))
            mask = (market_codes == market_code).copy() if market_code is not None else np.zeros(len(df), dtype=bool)
        else:
            mask = np.ones(len(df), dtype=bool)
        cap_bucket = _payload_cap_bucket(model)
        if cap_bucket is not None:
            cap_code = cap_lookup.get(str(cap_bucket))
            if cap_code is None:
                continue
            mask &= cap_codes == cap_code
        board_bucket = _payload_board_bucket(model)
        if board_bucket is not None:
            board_code = board_lookup.get(str(board_bucket))
            if board_code is None:
                continue
            mask &= board_codes == board_code
        listing_bucket = _payload_listing_bucket(model)
        if listing_bucket is not None:
            listing_code = listing_lookup.get(str(listing_bucket))
            if listing_code is None:
                continue
            mask &= listing_codes == listing_code
        if not mask.any():
            continue
        model_predictions = _predict_in_chunks(
            df.loc[mask],
            ticker_priors=ticker_priors,
            clip_low=np.asarray(model["clip_low"], dtype="float64"),
            clip_high=np.asarray(model["clip_high"], dtype="float64"),
            mean=np.asarray(model["mean"], dtype="float64"),
            scale=np.asarray(model["scale"], dtype="float64"),
            coef=np.asarray(model["coef"], dtype="float64"),
            intercept=float(model["intercept"]),
            chunk_size=chunk_size,
            prefer_row_priors=prefer_row_priors,
            feature_names=feature_names,
        )
        weight = _payload_weight(model)
        predictions[mask] += weight * model_predictions
        unweighted_predictions[mask] += model_predictions
        total_weights[mask] += weight
        prediction_counts[mask] += 1.0

    output = np.zeros(len(df), dtype="float64")
    weighted = total_weights > 0
    output[weighted] = predictions[weighted] / total_weights[weighted]
    unweighted = ~weighted & (prediction_counts > 0)
    output[unweighted] = unweighted_predictions[unweighted] / prediction_counts[unweighted]
    if (~(weighted | unweighted)).any():
        fallback_mask = ~(weighted | unweighted)
        output[fallback_mask] = np.mean(
            [
                _predict_in_chunks(
                    df.loc[fallback_mask],
                    ticker_priors=ticker_priors,
                    clip_low=np.asarray(model["clip_low"], dtype="float64"),
                    clip_high=np.asarray(model["clip_high"], dtype="float64"),
                    mean=np.asarray(model["mean"], dtype="float64"),
                    scale=np.asarray(model["scale"], dtype="float64"),
                    coef=np.asarray(model["coef"], dtype="float64"),
                    intercept=float(model["intercept"]),
                    chunk_size=chunk_size,
                    prefer_row_priors=prefer_row_priors,
                    feature_names=feature_names,
                )
                for model in models
            ],
            axis=0,
        )
    return output


def _payloads_compatible_for_blend(
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
) -> bool:
    return (
        candidate_payload.get("schema_version") == incumbent_payload.get("schema_version")
        and candidate_payload.get("feature_names") == incumbent_payload.get("feature_names")
        and candidate_payload.get("ticker_priors") == incumbent_payload.get("ticker_priors")
    )


def _payload_uses_row_priors(payload: Mapping[str, object]) -> bool:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and bool(metadata.get("uses_rolling_row_priors")):
        return True
    conditional_blend = _payload_conditional_blend(payload)
    if conditional_blend is not None:
        base_payload, candidate_payload, _routes = conditional_blend
        return _payload_uses_row_priors(base_payload) or _payload_uses_row_priors(
            candidate_payload
        )
    if any(
        _payload_uses_row_priors(member)
        for _start, _end, member in _payload_temporal_member_payloads(payload)
    ):
        return True
    if any(_payload_uses_row_priors(member) for _weight, member in _payload_member_payloads(payload)):
        return True
    return any(
        _payload_uses_row_priors(member)
        for _market_weights, member in _payload_market_member_payloads(payload)
    )


def _has_row_prior_columns(df: pd.DataFrame) -> bool:
    return any(str(column).startswith(PRIOR_COLUMN_PREFIX) for column in df.columns)


def _weighted_blend_payload(
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    *,
    candidate_weight: float,
) -> dict[str, object]:
    incumbent_weight = 1.0 - candidate_weight
    models: list[dict[str, object]] = []
    for model in _linear_payloads(incumbent_payload):
        models.append(
            _linear_model_payload(
                model,
                weight=incumbent_weight * _payload_weight(model),
            )
        )
    for model in _linear_payloads(candidate_payload):
        models.append(
            _linear_model_payload(
                model,
                weight=candidate_weight * _payload_weight(model),
            )
        )

    metadata = dict(candidate_payload.get("metadata", {}))
    metadata.update({
        "backend": "weighted_blend",
        "blend_candidate_weight": float(candidate_weight),
        "blend_incumbent_weight": float(incumbent_weight),
    })
    return {
        "schema_version": candidate_payload["schema_version"],
        "feature_names": candidate_payload["feature_names"],
        "models": models,
        "ticker_priors": candidate_payload.get("ticker_priors", {}),
        "metadata": metadata,
    }


def _payload_member_blend_payload(
    *,
    anchor_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    candidate_weight: float,
) -> dict[str, object]:
    anchor_weight = 1.0 - candidate_weight
    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    anchor_metadata = (
        anchor_payload.get("metadata", {})
        if isinstance(anchor_payload.get("metadata"), Mapping)
        else {}
    )
    candidate_metadata.update({
        "backend": "payload_member_blend",
        "blend_candidate_weight": float(candidate_weight),
        "blend_anchor_weight": float(anchor_weight),
        "anchor_model_path": str(anchor_metadata.get("model_path", "")),
        "anchor_trained_at": str(anchor_metadata.get("trained_at", "")),
        "anchor_backend": str(anchor_metadata.get("backend", "")),
        "anchor_target": str(anchor_metadata.get("target", "")),
    })
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": [],
        "payload_members": [
            {
                "weight": float(anchor_weight),
                "payload": dict(anchor_payload),
            },
            {
                "weight": float(candidate_weight),
                "payload": dict(candidate_payload),
            },
        ],
        "ticker_priors": {},
        "metadata": candidate_metadata,
    }


def _market_payload_member_blend_payload(
    *,
    anchor_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    candidate_weights_by_market: Mapping[str, float],
) -> dict[str, object]:
    markets = {"ashare", "hk", *(str(market) for market in candidate_weights_by_market)}
    candidate_weights = {
        str(market): max(0.0, min(1.0, float(weight)))
        for market, weight in candidate_weights_by_market.items()
    }
    anchor_weights = {
        market: max(0.0, 1.0 - float(candidate_weights.get(market, 0.0)))
        for market in markets
    }
    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    anchor_metadata = (
        anchor_payload.get("metadata", {})
        if isinstance(anchor_payload.get("metadata"), Mapping)
        else {}
    )
    candidate_metadata.update({
        "backend": "market_payload_member_blend",
        "market_blend_candidate_weights": candidate_weights,
        "anchor_model_path": str(anchor_metadata.get("model_path", "")),
        "anchor_trained_at": str(anchor_metadata.get("trained_at", "")),
        "anchor_backend": str(anchor_metadata.get("backend", "")),
        "anchor_target": str(anchor_metadata.get("target", "")),
    })
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": [],
        "market_payload_members": [
            {
                "market_weights": anchor_weights,
                "payload": dict(anchor_payload),
            },
            {
                "market_weights": candidate_weights,
                "payload": dict(candidate_payload),
            },
        ],
        "ticker_priors": {},
        "metadata": candidate_metadata,
    }


def _conditional_blend_payload(
    *,
    anchor_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    routes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sanitized_routes: list[dict[str, object]] = []
    for route in routes:
        try:
            candidate_weight = float(route.get("candidate_weight", 1.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(candidate_weight) or candidate_weight <= 0.0:
            continue
        sanitized = {
            "candidate_weight": max(0.0, min(1.0, candidate_weight)),
        }
        for key in (
            "start_date",
            "end_date",
            "market",
            "cap_bucket",
            "board_bucket",
            "listing_bucket",
        ):
            value = route.get(key)
            if value is not None and str(value) != "":
                sanitized[key] = str(value)
        sanitized_routes.append(sanitized)
    if not sanitized_routes:
        raise ValueError("conditional blend requires at least one valid route")

    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    anchor_metadata = (
        anchor_payload.get("metadata", {})
        if isinstance(anchor_payload.get("metadata"), Mapping)
        else {}
    )
    candidate_metadata.update({
        "backend": "conditional_payload_blend",
        "conditional_blend_routes": sanitized_routes,
        "anchor_model_path": str(anchor_metadata.get("model_path", "")),
        "anchor_trained_at": str(anchor_metadata.get("trained_at", "")),
        "anchor_backend": str(anchor_metadata.get("backend", "")),
        "anchor_target": str(anchor_metadata.get("target", "")),
    })
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": [],
        "conditional_blend": {
            "base_payload": dict(anchor_payload),
            "candidate_payload": dict(candidate_payload),
            "routes": sanitized_routes,
        },
        "ticker_priors": {},
        "metadata": candidate_metadata,
    }


def _route_with_start_floor(
    route: Mapping[str, object],
    start_date: str,
) -> dict[str, object]:
    updated = dict(route)
    floor = _payload_temporal_timestamp(start_date)
    existing = _payload_temporal_timestamp(updated.get("start_date"))
    if floor is not None and (existing is None or existing < floor):
        updated["start_date"] = str(start_date)
    return updated


def _conditional_payload_with_metadata(
    *,
    anchor_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    routes: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    payload = _conditional_blend_payload(
        anchor_payload=anchor_payload,
        candidate_payload=candidate_payload,
        routes=routes,
    )
    payload_metadata = payload.get("metadata")
    if not isinstance(payload_metadata, dict):
        payload_metadata = {}
        payload["metadata"] = payload_metadata
    payload_metadata.update(metadata)
    return payload


def _recent_regime_temporal_payload(
    *,
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    gate_start: str,
    recent_start: str,
) -> dict[str, object]:
    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    incumbent_metadata = (
        incumbent_payload.get("metadata", {})
        if isinstance(incumbent_payload.get("metadata"), Mapping)
        else {}
    )
    candidate_metadata.update({
        "backend": "temporal_recent_regime_blend",
        "model_kind": "temporal_recent_regime_blend",
        "feature_set": "temporal_recent_regime",
        "prior_strategy": "temporal",
        "uses_rolling_row_priors": _payload_uses_row_priors(candidate_payload)
        or _payload_uses_row_priors(incumbent_payload),
        "temporal_transition_date": str(gate_start),
        "temporal_recent_regime_start": str(recent_start),
        "temporal_recent_member_backend": (
            candidate_metadata.get("backend")
            or (candidate_payload.get("metadata") or {}).get("backend")
            if isinstance(candidate_payload.get("metadata"), Mapping)
            else ""
        ),
        "temporal_incumbent_backend": str(incumbent_metadata.get("backend", "")),
        "anchor_model_path": str(incumbent_metadata.get("model_path", "")),
        "anchor_trained_at": str(incumbent_metadata.get("trained_at", "")),
    })
    return _conditional_payload_with_metadata(
        anchor_payload=incumbent_payload,
        candidate_payload=candidate_payload,
        routes=(
            {"end_date": str(gate_start), "candidate_weight": 1.0},
            {"start_date": str(recent_start), "candidate_weight": 1.0},
        ),
        metadata=candidate_metadata,
    )


def _recent_market_regime_temporal_payload(
    *,
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    gate_start: str,
    recent_start: str,
    candidate_weights_by_market: Mapping[str, float],
) -> dict[str, object]:
    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    incumbent_metadata = (
        incumbent_payload.get("metadata", {})
        if isinstance(incumbent_payload.get("metadata"), Mapping)
        else {}
    )
    candidate_weights = {
        str(market): max(0.0, min(1.0, float(weight)))
        for market, weight in candidate_weights_by_market.items()
    }
    candidate_metadata.update({
        "backend": "temporal_recent_market_regime_blend",
        "model_kind": "temporal_recent_market_regime_blend",
        "feature_set": "temporal_recent_market_regime",
        "prior_strategy": "temporal_market",
        "uses_rolling_row_priors": _payload_uses_row_priors(candidate_payload)
        or _payload_uses_row_priors(incumbent_payload),
        "temporal_transition_date": str(gate_start),
        "temporal_recent_regime_start": str(recent_start),
        "temporal_recent_market_weights": candidate_weights,
        "temporal_incumbent_backend": str(incumbent_metadata.get("backend", "")),
        "anchor_model_path": str(incumbent_metadata.get("model_path", "")),
        "anchor_trained_at": str(incumbent_metadata.get("trained_at", "")),
    })
    routes: list[Mapping[str, object]] = [
        {"end_date": str(gate_start), "candidate_weight": 1.0},
    ]
    for market, weight in candidate_weights.items():
        if float(weight) <= 0.0:
            continue
        routes.append({
            "start_date": str(recent_start),
            "market": market,
            "candidate_weight": float(weight),
        })
    return _conditional_payload_with_metadata(
        anchor_payload=incumbent_payload,
        candidate_payload=candidate_payload,
        routes=routes,
        metadata=candidate_metadata,
    )


def _recent_segment_regime_temporal_payload(
    *,
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    gate_start: str,
    recent_start: str,
    routes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_metadata = dict(candidate_payload.get("metadata", {}))
    incumbent_metadata = (
        incumbent_payload.get("metadata", {})
        if isinstance(incumbent_payload.get("metadata"), Mapping)
        else {}
    )
    route_records = []
    for route in routes:
        record = {
            key: route[key]
            for key in (
                "market",
                "cap_bucket",
                "board_bucket",
                "listing_bucket",
                "candidate_weight",
            )
            if key in route
        }
        if record:
            route_records.append(record)
    candidate_metadata.update({
        "backend": "temporal_recent_segment_regime_blend",
        "model_kind": "temporal_recent_segment_regime_blend",
        "feature_set": "temporal_recent_segment_regime",
        "prior_strategy": "temporal_segment",
        "uses_rolling_row_priors": _payload_uses_row_priors(candidate_payload)
        or _payload_uses_row_priors(incumbent_payload),
        "temporal_transition_date": str(gate_start),
        "temporal_recent_regime_start": str(recent_start),
        "temporal_recent_segment_routes": route_records,
        "temporal_incumbent_backend": str(incumbent_metadata.get("backend", "")),
        "anchor_model_path": str(incumbent_metadata.get("model_path", "")),
        "anchor_trained_at": str(incumbent_metadata.get("trained_at", "")),
    })
    conditional_routes: list[Mapping[str, object]] = [
        {"end_date": str(gate_start), "candidate_weight": 1.0},
    ]
    conditional_routes.extend(
        _route_with_start_floor(route, str(recent_start))
        for route in routes
    )
    return _conditional_payload_with_metadata(
        anchor_payload=incumbent_payload,
        candidate_payload=candidate_payload,
        routes=conditional_routes,
        metadata=candidate_metadata,
    )


def _payload_has_market_models(payload: Mapping[str, object]) -> bool:
    return any(_payload_market(model) is not None for model in _linear_payloads(payload))


RUNTIME_METADATA_KEYS = {
    "application_refit",
    "application_blend_candidate_weight",
    "application_blend_scale",
    "application_factor_components",
    "application_factor_scale",
    "application_factor_template",
    "application_residual_candidate_weight",
    "application_residual_column",
    "application_residual_direction",
    "application_residual_scale",
    "application_residual_signal",
    "application_quality_residual_candidate_weight",
    "application_quality_residual_direction",
    "application_quality_residual_scale",
    "anchor_backend",
    "anchor_model_path",
    "anchor_target",
    "anchor_trained_at",
    "backend",
    "blend_anchor_weight",
    "blend_candidate_weight",
    "blend_incumbent_weight",
    "candidate_anchor_model_paths",
    "candidate_feature_sets",
    "candidate_prior_strategies",
    "candidate_ridge_lambdas",
    "candidate_targets",
    "conditional_blend_routes",
    "deployment_refit",
    "ensemble_size",
    "evaluation_protocol",
    "feature_set",
    "final_rho_improvement",
    "full_eval_candidate_limit",
    "full_fit_refit",
    "full_fit_rows",
    "market_blend_candidate_weights",
    "members",
    "metrics",
    "model_kind",
    "model_path",
    "n_features",
    "n_rows",
    "n_snapshots",
    "n_training_rows",
    "prior_strategy",
    "promotion_gate_manifest",
    "promotion_incumbent_type",
    "promotion_status",
    "ridge_lambda",
    "runtime_feature_parity",
    "strict_outer_gate",
    "target",
    "target_horizon",
    "temporal_anchor_backend",
    "temporal_anchor_target",
    "temporal_anchor_trained_at",
    "temporal_incumbent_backend",
    "temporal_recent_market_weights",
    "temporal_recent_member_backend",
    "temporal_recent_regime_start",
    "temporal_recent_segment_routes",
    "temporal_train_member_backend",
    "temporal_train_member_feature_set",
    "temporal_train_member_model_kind",
    "temporal_train_member_target",
    "temporal_transition_date",
    "train_metrics",
    "train_rho_improvement",
    "trained_at",
    "uses_rolling_row_priors",
    "walk_forward",
}


def _compact_payload_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key) in RUNTIME_METADATA_KEYS
    }


def _collapse_recent_temporal_payload(
    payload: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Optional[dict[str, object]]:
    members = payload.get("temporal_payload_members")
    if not isinstance(members, list) or len(members) != 3:
        return None
    first, middle, third = members
    if not (
        isinstance(first, Mapping)
        and isinstance(middle, Mapping)
        and isinstance(third, Mapping)
    ):
        return None
    candidate_payload = first.get("payload")
    incumbent_payload = middle.get("payload")
    if not isinstance(candidate_payload, Mapping) or not isinstance(incumbent_payload, Mapping):
        return None
    gate_start = first.get("end_date") or middle.get("start_date")
    recent_start = middle.get("end_date") or third.get("start_date")
    if gate_start is None or recent_start is None:
        return None

    backend = str(metadata.get("backend") or metadata.get("model_kind") or "")
    routes: list[Mapping[str, object]] = [
        {"end_date": str(gate_start), "candidate_weight": 1.0},
    ]
    if backend == "temporal_recent_regime_blend":
        routes.append({"start_date": str(recent_start), "candidate_weight": 1.0})
    elif backend == "temporal_recent_market_regime_blend":
        raw_weights = metadata.get("temporal_recent_market_weights")
        if not isinstance(raw_weights, Mapping):
            return None
        for market, weight in raw_weights.items():
            try:
                candidate_weight = float(weight)
            except (TypeError, ValueError):
                continue
            if candidate_weight <= 0.0:
                continue
            routes.append({
                "start_date": str(recent_start),
                "market": str(market),
                "candidate_weight": candidate_weight,
            })
    elif backend == "temporal_recent_segment_regime_blend":
        raw_routes: object = None
        third_payload = third.get("payload")
        if isinstance(third_payload, Mapping):
            blend = third_payload.get("conditional_blend")
            if isinstance(blend, Mapping):
                raw_routes = blend.get("routes")
        if not isinstance(raw_routes, list):
            raw_routes = metadata.get("temporal_recent_segment_routes")
        if not isinstance(raw_routes, list):
            return None
        routes.extend(
            _route_with_start_floor(route, str(recent_start))
            for route in raw_routes
            if isinstance(route, Mapping)
        )
    else:
        return None

    compact_candidate = _compact_runtime_payload(candidate_payload)
    compact_incumbent = _compact_runtime_payload(incumbent_payload)
    if not isinstance(compact_candidate, Mapping) or not isinstance(compact_incumbent, Mapping):
        return None
    return _conditional_payload_with_metadata(
        anchor_payload=compact_incumbent,
        candidate_payload=compact_candidate,
        routes=routes,
        metadata=metadata,
    )


def _compact_runtime_payload(payload: object) -> object:
    """Return a runtime-equivalent payload without recursive training diagnostics."""
    if isinstance(payload, Mapping):
        metadata = _compact_payload_metadata(payload.get("metadata"))
        collapsed = _collapse_recent_temporal_payload(payload, metadata)
        if collapsed is not None:
            return collapsed
        compacted: dict[str, object] = {}
        for key, value in payload.items():
            key_str = str(key)
            if key_str == "metadata":
                if metadata:
                    compacted[key_str] = metadata
                continue
            compacted[key_str] = _compact_runtime_payload(value)
        return compacted
    if isinstance(payload, list):
        return [_compact_runtime_payload(value) for value in payload]
    return payload


def _market_blend_weight_vector(
    df: pd.DataFrame,
    market_weights: Mapping[str, float],
) -> np.ndarray:
    markets, _board_buckets, _listing_buckets = _ticker_bucket_arrays(df)
    market_codes, market_lookup = _encode_bucket_array(markets)
    weights_by_code = np.zeros(len(market_lookup), dtype="float64")
    for market, weight in market_weights.items():
        code = market_lookup.get(str(market))
        if code is not None:
            weights_by_code[code] = float(weight)
    weights = weights_by_code[market_codes] if len(weights_by_code) else np.zeros(len(df))
    return np.clip(weights, 0.0, 1.0)


def _market_blend_key(market_weights: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(market), float(weight)) for market, weight in market_weights.items()))


def _market_blend_label(market_weights: Optional[Mapping[str, float]]) -> str:
    if not market_weights:
        return "none"
    return ",".join(f"{market}:{float(weight):g}" for market, weight in sorted(market_weights.items()))


def _segment_route_label(routes: Optional[Sequence[Mapping[str, object]]]) -> str:
    if not routes:
        return "none"
    labels = []
    for route in routes:
        market = str(route.get("market", "*"))
        cap_bucket = str(route.get("cap_bucket", "*"))
        board_bucket = str(route.get("board_bucket", "*"))
        listing_bucket = str(route.get("listing_bucket", "*"))
        try:
            weight = float(route.get("candidate_weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        labels.append(f"{market}/{cap_bucket}/{board_bucket}/{listing_bucket}:{weight:g}")
    return ",".join(labels)


def _conditional_route_weight_vector(
    df: pd.DataFrame,
    routes: Sequence[Mapping[str, object]],
) -> np.ndarray:
    if not routes:
        return np.zeros(len(df), dtype="float64")
    weights = np.zeros(len(df), dtype="float64")
    dates = _snapshot_dates(df) if "snapshot_date" in df.columns else None
    markets, board_buckets, listing_buckets = _ticker_bucket_arrays(df)
    cap_buckets = _cap_bucket_array(df)
    market_codes, market_lookup = _encode_bucket_array(markets)
    cap_codes, cap_lookup = _encode_bucket_array(cap_buckets)
    board_codes, board_lookup = _encode_bucket_array(board_buckets)
    listing_codes, listing_lookup = _encode_bucket_array(listing_buckets)
    for route in routes:
        try:
            candidate_weight = float(route.get("candidate_weight", 1.0))
        except (TypeError, ValueError):
            candidate_weight = 1.0
        if not np.isfinite(candidate_weight) or candidate_weight <= 0.0:
            continue
        candidate_weight = min(1.0, max(0.0, candidate_weight))
        mask = np.ones(len(df), dtype=bool)
        start_date = _payload_temporal_timestamp(route.get("start_date"))
        end_date = _payload_temporal_timestamp(route.get("end_date"))
        if dates is not None and start_date is not None:
            mask &= (dates >= start_date).to_numpy(dtype=bool)
        if dates is not None and end_date is not None:
            mask &= (dates < end_date).to_numpy(dtype=bool)
        market = route.get("market")
        if market is not None and str(market) != "":
            market_mask = _bucket_code_mask(market_codes, market_lookup, market)
            if market_mask is None:
                pass
            elif not market_mask.any():
                continue
            else:
                mask &= market_mask
        cap_bucket = route.get("cap_bucket")
        if cap_bucket is not None and str(cap_bucket) != "":
            cap_mask = _bucket_code_mask(cap_codes, cap_lookup, cap_bucket)
            if cap_mask is None:
                pass
            elif not cap_mask.any():
                continue
            else:
                mask &= cap_mask
        board_bucket = route.get("board_bucket")
        if board_bucket is not None and str(board_bucket) != "":
            board_mask = _bucket_code_mask(board_codes, board_lookup, board_bucket)
            if board_mask is None:
                pass
            elif not board_mask.any():
                continue
            else:
                mask &= board_mask
        listing_bucket = route.get("listing_bucket")
        if listing_bucket is not None and str(listing_bucket) != "":
            listing_mask = _bucket_code_mask(
                listing_codes,
                listing_lookup,
                listing_bucket,
            )
            if listing_mask is None:
                pass
            elif not listing_mask.any():
                continue
            else:
                mask &= listing_mask
        if mask.any():
            weights[mask] = np.maximum(weights[mask], candidate_weight)
    return weights


def _select_train_floor_blends(
    scored_blends: Sequence[tuple[float, float]],
    *,
    promotion_min_train_rho: Optional[float],
    limit: int,
) -> tuple[float, ...]:
    if not scored_blends:
        return ()
    if promotion_min_train_rho is None:
        return tuple(candidate_weight for candidate_weight, _train_rho in scored_blends)

    passing = [
        (candidate_weight, train_rho)
        for candidate_weight, train_rho in scored_blends
        if train_rho >= promotion_min_train_rho
    ]
    if passing:
        selected: list[float] = []

        def add(candidate_weight: float) -> None:
            if len(selected) >= limit:
                return
            if any(abs(candidate_weight - existing) < 1e-12 for existing in selected):
                return
            selected.append(float(candidate_weight))

        by_floor = sorted(
            passing,
            key=lambda item: (
                item[1] - promotion_min_train_rho,
                item[0],
            ),
        )
        for candidate_weight, _train_rho in by_floor:
            add(candidate_weight)
        for candidate_weight, _train_rho in sorted(
            passing,
            key=lambda item: (item[0], item[1]),
        ):
            add(candidate_weight)
        for candidate_weight, _train_rho in sorted(
            passing,
            key=lambda item: (item[1], item[0]),
            reverse=True,
        ):
            add(candidate_weight)
        return tuple(selected)

    best_weight, _best_train_rho = max(scored_blends, key=lambda item: item[1])
    return (best_weight,)


def _select_train_floor_market_blends(
    scored_market_blends: Sequence[tuple[Mapping[str, float], float]],
    *,
    promotion_min_train_rho: Optional[float],
    limit: int,
    train_floor_tolerance: float = 0.0,
) -> tuple[Mapping[str, float], ...]:
    if not scored_market_blends:
        return ()
    if promotion_min_train_rho is None:
        return tuple(market_weights for market_weights, _train_rho in scored_market_blends)

    passing = [
        (market_weights, train_rho)
        for market_weights, train_rho in scored_market_blends
        if train_rho + train_floor_tolerance >= promotion_min_train_rho
    ]
    if not passing:
        return ()

    selected: list[Mapping[str, float]] = []
    selected_keys: set[tuple[tuple[str, float], ...]] = set()

    def total_weight(market_weights: Mapping[str, float]) -> float:
        return sum(float(weight) for weight in market_weights.values())

    def add(market_weights: Mapping[str, float]) -> None:
        if len(selected) >= limit:
            return
        key = _market_blend_key(market_weights)
        if key in selected_keys:
            return
        selected.append(market_weights)
        selected_keys.add(key)

    by_floor = sorted(
        passing,
        key=lambda item: (
            item[1] - promotion_min_train_rho,
            total_weight(item[0]),
            float(item[0].get("ashare", 0.0)),
            float(item[0].get("hk", 0.0)),
        ),
    )
    for market_weights, _train_rho in by_floor[:2]:
        add(market_weights)
    for target_total in SIX_MONTH_GATE_MARKET_PROBE_TOTALS:
        market_weights, _train_rho = min(
            passing,
            key=lambda item: (
                abs(total_weight(item[0]) - target_total),
                float(item[0].get("hk", 0.0)),
                abs(float(item[0].get("ashare", 0.0)) - target_total),
                item[1] - promotion_min_train_rho,
            ),
        )
        add(market_weights)
    for market_weights, _train_rho in sorted(
        passing,
        key=lambda item: (
            total_weight(item[0]),
            float(item[0].get("ashare", 0.0)),
            float(item[0].get("hk", 0.0)),
        ),
    ):
        add(market_weights)
    for market_weights, _train_rho in sorted(
        passing,
        key=lambda item: (
            item[1],
            float(item[0].get("ashare", 0.0)),
            float(item[0].get("hk", 0.0)),
        ),
        reverse=True,
    ):
        add(market_weights)
    return tuple(selected)


def _top_gate_anchor_payloads(
    anchor_payloads: Sequence[Mapping[str, object]],
    *,
    limit: int = SIX_MONTH_GATE_ANCHOR_MODEL_LIMIT,
) -> tuple[Mapping[str, object], ...]:
    """Return the strongest current-gate anchors for expensive blend attempts."""
    if limit <= 0:
        return ()
    ranked = sorted(
        anchor_payloads,
        key=lambda anchor: (
            float(anchor.get("gate_rho", float("-inf"))),
            str(anchor.get("path", "")),
        ),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _market_weighted_blend_payload(
    candidate_payload: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    *,
    candidate_weights_by_market: Mapping[str, float],
) -> dict[str, object]:
    models: list[dict[str, object]] = []
    for model in _linear_payloads(incumbent_payload):
        market = _payload_market(model)
        candidate_weight = float(candidate_weights_by_market.get(str(market), 0.0))
        models.append(
            _linear_model_payload(
                model,
                weight=(1.0 - candidate_weight) * _payload_weight(model),
            )
        )
    for model in _linear_payloads(candidate_payload):
        market = _payload_market(model)
        candidate_weight = float(candidate_weights_by_market.get(str(market), 0.0))
        models.append(
            _linear_model_payload(
                model,
                weight=candidate_weight * _payload_weight(model),
            )
        )

    metadata = dict(candidate_payload.get("metadata", {}))
    metadata.update({
        "backend": "market_weighted_blend",
        "market_blend_candidate_weights": {
            str(market): float(weight)
            for market, weight in candidate_weights_by_market.items()
        },
    })
    return {
        "schema_version": candidate_payload["schema_version"],
        "feature_names": candidate_payload["feature_names"],
        "models": models,
        "ticker_priors": candidate_payload.get("ticker_priors", {}),
        "metadata": metadata,
    }


def _application_blend_candidate_weight(payload: Mapping[str, object]) -> float:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return 1.0
    try:
        weight = float(metadata.get("application_blend_candidate_weight", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(weight):
        return 1.0
    return float(np.clip(weight, 0.0, 1.0))


def _application_quality_residual_candidate_weight(payload: Mapping[str, object]) -> float:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return 1.0
    try:
        weight = float(metadata.get("application_quality_residual_candidate_weight", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(weight):
        return 1.0
    return float(np.clip(weight, 0.0, 1.0))


def _application_factor_config(payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    raw_components = metadata.get("application_factor_components")
    if not isinstance(raw_components, list):
        return ()
    components: list[dict[str, object]] = []
    for component in raw_components:
        if not isinstance(component, Mapping):
            continue
        signal = str(component.get("signal", "")).strip()
        try:
            weight = float(component.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if signal in SIX_MONTH_APPLICATION_FACTOR_SIGNALS and np.isfinite(weight) and weight > 0.0:
            components.append({"signal": signal, "weight": weight})
    return tuple(components)


def _application_residual_config(
    payload: Mapping[str, object],
) -> tuple[float, str, float]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return 1.0, "", 1.0
    if "application_residual_candidate_weight" not in metadata:
        weight = _application_quality_residual_candidate_weight(payload)
        return weight, "quality_score", -1.0
    try:
        weight = float(metadata.get("application_residual_candidate_weight", 1.0))
        direction = float(metadata.get("application_residual_direction", 1.0))
    except (TypeError, ValueError):
        return 1.0, "", 1.0
    column = str(metadata.get("application_residual_column", "")).strip()
    if not np.isfinite(weight) or not np.isfinite(direction) or not column:
        return 1.0, "", 1.0
    return float(np.clip(weight, 0.0, 1.0)), column, direction


def _predict_ml_ranker_payload_unblended(
    df: pd.DataFrame,
    payload: Mapping[str, object],
    *,
    chunk_size: int = 200_000,
) -> np.ndarray:
    conditional_blend = _payload_conditional_blend(payload)
    if conditional_blend is not None:
        base_payload, candidate_payload, routes = conditional_blend
        route_weights = _conditional_route_weight_vector(df, routes)
        candidate_predictions = _predict_ml_ranker_payload_unblended(
            df,
            candidate_payload,
            chunk_size=chunk_size,
        )
        base_predictions = _predict_ml_ranker_payload_unblended(
            df,
            base_payload,
            chunk_size=chunk_size,
        )
        return route_weights * candidate_predictions + (1.0 - route_weights) * base_predictions

    temporal_members = _payload_temporal_member_payloads(payload)
    if temporal_members:
        if len(temporal_members) == 1:
            _start_date, _end_date, member_payload = temporal_members[0]
            return _predict_ml_ranker_payload_unblended(
                df,
                member_payload,
                chunk_size=chunk_size,
            )
        if "snapshot_date" not in df.columns:
            raise ValueError("Temporal ML ranker payload requires snapshot_date")
        dates = _snapshot_dates(df)
        predictions = np.empty(len(df), dtype="float64")
        assigned = np.zeros(len(df), dtype=bool)
        for start_date, end_date, member_payload in temporal_members:
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= dates >= start_date
            if end_date is not None:
                mask &= dates < end_date
            mask_array = mask.to_numpy(dtype=bool) & ~assigned
            if not mask_array.any():
                continue
            predictions[mask_array] = _predict_ml_ranker_payload_unblended(
                df.loc[mask_array],
                member_payload,
                chunk_size=chunk_size,
            )
            assigned[mask_array] = True
        if (~assigned).any():
            predictions[~assigned] = _predict_ml_ranker_payload_unblended(
                df.loc[~assigned],
                temporal_members[-1][2],
                chunk_size=chunk_size,
            )
        return predictions

    rank_payload_members = _payload_rank_member_payloads(payload)
    if rank_payload_members:
        predictions = np.zeros(len(df), dtype="float64")
        total_weight = 0.0
        for weight, member_payload in rank_payload_members:
            member_predictions = _predict_ml_ranker_payload_unblended(
                df,
                member_payload,
                chunk_size=chunk_size,
            )
            predictions += weight * _prediction_rank_by_snapshot(df, member_predictions)
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("rank payload-member ensemble has no positive weights")
        return predictions / total_weight

    payload_members = _payload_member_payloads(payload)
    if payload_members:
        predictions = np.zeros(len(df), dtype="float64")
        total_weight = 0.0
        for weight, member_payload in payload_members:
            predictions += weight * _predict_ml_ranker_payload_unblended(
                df,
                member_payload,
                chunk_size=chunk_size,
            )
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("ML ranker payload-member ensemble has no positive weights")
        return predictions / total_weight

    market_payload_members = _payload_market_member_payloads(payload)
    if market_payload_members:
        markets, _board_buckets, _listing_buckets = _ticker_bucket_arrays(df)
        market_codes, market_lookup = _encode_bucket_array(markets)
        predictions = np.zeros(len(df), dtype="float64")
        total_weights = np.zeros(len(df), dtype="float64")
        for market_weights, member_payload in market_payload_members:
            weights_by_code = np.zeros(len(market_lookup), dtype="float64")
            for market, weight in market_weights.items():
                code = market_lookup.get(str(market))
                if code is not None:
                    weights_by_code[code] = float(weight)
            weights_array = (
                weights_by_code[market_codes] if len(weights_by_code) else np.zeros(len(df), dtype="float64")
            )
            weights_array = np.clip(weights_array, 0.0, 1.0)
            if not weights_array.any():
                continue
            member_predictions = _predict_ml_ranker_payload_unblended(
                df,
                member_payload,
                chunk_size=chunk_size,
            )
            predictions += weights_array * member_predictions
            total_weights += weights_array
        assigned = total_weights > 0.0
        if assigned.any():
            predictions[assigned] = predictions[assigned] / total_weights[assigned]
        if (~assigned).any():
            predictions[~assigned] = _predict_ml_ranker_payload_unblended(
                df.loc[~assigned],
                market_payload_members[0][1],
                chunk_size=chunk_size,
            )
        return predictions

    lightgbm_model_text = payload.get("lightgbm_model")
    if isinstance(lightgbm_model_text, str) and lightgbm_model_text:
        ticker_priors = payload.get("ticker_priors", {})
        if not isinstance(ticker_priors, Mapping):
            ticker_priors = {}
        feature_names = payload.get("feature_names")
        if not isinstance(feature_names, list):
            raise ValueError("LightGBM payload requires feature_names")
        prefer_row_priors = _payload_uses_row_priors(payload) and _has_row_prior_columns(df)
        predictions = np.empty(len(df), dtype="float64")
        for index in _prediction_index_chunks(
            df,
            chunk_size=chunk_size,
            feature_names=feature_names,
        ):
            features = build_feature_matrix(
                df.iloc[index],
                ticker_priors,
                prefer_row_priors=prefer_row_priors,
                feature_names=feature_names,
            )
            predictions[index] = predict_lightgbm_model(lightgbm_model_text, features)
        return predictions

    model_payloads = _linear_payloads(payload)
    if not model_payloads:
        raise ValueError("ML ranker payload contains no linear models")

    ticker_priors = payload.get("ticker_priors", {})
    if not isinstance(ticker_priors, Mapping):
        ticker_priors = {}

    return _predict_from_linear_model_payloads(
        df,
        ticker_priors,
        model_payloads,
        chunk_size=chunk_size,
        prefer_row_priors=_payload_uses_row_priors(payload) and _has_row_prior_columns(df),
        feature_names=payload.get("feature_names") if isinstance(payload.get("feature_names"), list) else None,
    )


def predict_ml_ranker_payload(
    df: pd.DataFrame,
    payload: Mapping[str, object],
    *,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Predict a snapshot frame from a serialized ML ranker artifact."""
    predictions = _predict_ml_ranker_payload_unblended(
        df,
        payload,
        chunk_size=chunk_size,
    )
    blend_weight = _application_blend_candidate_weight(payload)
    if blend_weight < 1.0:
        if "composite_score" not in df.columns:
            raise ValueError("Application-blended ML ranker requires composite_score")
        incumbent_predictions = pd.to_numeric(
            df["composite_score"],
            errors="coerce",
        ).to_numpy(dtype="float64")
        predictions = blend_weight * _prediction_rank_by_snapshot(df, predictions) + (
            1.0 - blend_weight
        ) * _prediction_rank_by_snapshot(df, incumbent_predictions)

    factor_components = _application_factor_config(payload)
    if factor_components:
        predictions = _application_factor_predictions(
            df,
            predictions,
            factor_components,
        )
    residual_weight, residual_column, residual_direction = _application_residual_config(payload)
    if not factor_components and residual_weight < 1.0:
        if residual_column not in df.columns:
            raise ValueError(f"Application-residual ML ranker requires {residual_column}")
        residual_values = residual_direction * pd.to_numeric(
            df[residual_column],
            errors="coerce",
        ).to_numpy(dtype="float64")
        model_rank = _prediction_rank_by_snapshot(df, predictions)
        residual_rank = _prediction_rank_by_snapshot(df, residual_values)
        blended = model_rank.copy()
        usable = np.isfinite(model_rank) & np.isfinite(residual_rank)
        blended[usable] = residual_weight * model_rank[usable] + (1.0 - residual_weight) * residual_rank[usable]
        predictions = blended
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    (
        de_crowding_weight,
        de_crowding_max_rank,
        de_crowding_min_market_return,
    ) = _application_gross_profitability_de_crowding_config(metadata)
    if de_crowding_weight > 0.0:
        revenue = _numeric_column(df, "revenue")
        gross_margin = _numeric_column(df, "gross_margin")
        total_assets = _numeric_column(df, "total_assets")
        gross_profit_assets = _safe_divide(revenue * gross_margin, total_assets)
        market_return_63d = _numeric_column(df, "market_return_63d")
        adjusted = np.asarray(predictions, dtype="float64").copy()
        for positions in df.groupby("snapshot_date", sort=False).indices.values():
            adjusted[positions] = _application_gross_profitability_de_crowding_predictions(
                adjusted[positions],
                gross_profit_assets[positions],
                market_return_63d[positions],
                factor_weight=de_crowding_weight,
                max_model_rank=de_crowding_max_rank,
                min_market_return=de_crowding_min_market_return,
            )
        predictions = adjusted
    return predictions


def evaluate_ml_ranker_payload(
    df: pd.DataFrame,
    payload: Mapping[str, object],
    *,
    chunk_size: int = 200_000,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a serialized ML ranker artifact on an arbitrary snapshot frame."""
    return _evaluate_predictions(
        df,
        predict_ml_ranker_payload(df, payload, chunk_size=chunk_size),
    )


def _market_bucket(ticker: object) -> str:
    value = str(ticker)
    if value.endswith(".HK") or value.startswith("HK"):
        return "hk"
    return "ashare"


def _board_bucket(ticker: object) -> str:
    value = str(ticker).upper()
    if value.endswith(".HK") or value.startswith("HK"):
        return "hk"
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits.startswith(("688", "689")):
        return "star"
    if digits.startswith(("300", "301")):
        return "chinext"
    if digits.startswith(("002", "003")):
        return "sz_sme"
    if digits.startswith(("000", "001")):
        return "sz_main"
    if digits.startswith(("600", "601", "603", "605")):
        return "sh_main"
    if digits.startswith(("8", "4")):
        return "bj"
    return "other"


def _listing_bucket(ticker: object) -> str:
    value = str(ticker).upper()
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return "unknown"
    if value.endswith(".HK") or value.startswith("HK"):
        return f"hk_{digits[:2]}"
    return digits[:3] if len(digits) >= 3 else digits


def _ticker_bucket_tuple(ticker: object) -> tuple[str, str, str]:
    key = str(ticker)
    cached = _TICKER_BUCKET_CACHE.get(key)
    if cached is not None:
        return cached
    buckets = (_market_bucket(key), _board_bucket(key), _listing_bucket(key))
    _TICKER_BUCKET_CACHE[key] = buckets
    return buckets


def _cap_bucket(value: object) -> str:
    try:
        market_cap = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(market_cap) or market_cap <= 0:
        return "unknown"
    if market_cap < 10_000_000_000.0:
        return "small"
    if market_cap < 50_000_000_000.0:
        return "mid"
    return "large"


def _ticker_bucket_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    markets = np.full(len(df), "", dtype=object)
    board_buckets = np.full(len(df), "unknown", dtype=object)
    listing_buckets = np.full(len(df), "unknown", dtype=object)
    if "ticker" not in df.columns:
        return markets, board_buckets, listing_buckets
    tickers = df["ticker"].astype(str).to_numpy(dtype=object, copy=False)
    unique_tickers, ticker_codes = _encode_label_values(tickers)
    unique_markets = np.empty(len(unique_tickers), dtype=object)
    unique_boards = np.empty(len(unique_tickers), dtype=object)
    unique_listings = np.empty(len(unique_tickers), dtype=object)
    for idx, ticker in enumerate(unique_tickers):
        market, board_bucket, listing_bucket = _ticker_bucket_tuple(ticker)
        unique_markets[idx] = market
        unique_boards[idx] = board_bucket
        unique_listings[idx] = listing_bucket
    markets = unique_markets[ticker_codes]
    board_buckets = unique_boards[ticker_codes]
    listing_buckets = unique_listings[ticker_codes]
    return markets, board_buckets, listing_buckets


def _cap_bucket_array(df: pd.DataFrame) -> np.ndarray:
    cap_buckets = np.full(len(df), "unknown", dtype=object)
    if "market_cap_rmb" not in df.columns:
        return cap_buckets
    values = pd.to_numeric(df["market_cap_rmb"], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values) & (values > 0.0)
    cap_buckets[valid & (values < 10_000_000_000.0)] = "small"
    cap_buckets[valid & (values >= 10_000_000_000.0) & (values < 50_000_000_000.0)] = "mid"
    cap_buckets[valid & (values >= 50_000_000_000.0)] = "large"
    return cap_buckets


def _encode_bucket_array(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    codes = np.empty(len(values), dtype="int32")
    lookup: dict[str, int] = {}
    for idx, value in enumerate(values):
        key = str(value)
        code = lookup.get(key)
        if code is None:
            code = len(lookup)
            lookup[key] = code
        codes[idx] = code
    return codes, lookup


def _bucket_code_mask(
    codes: np.ndarray,
    lookup: Mapping[str, int],
    value: object,
) -> Optional[np.ndarray]:
    key = str(value)
    if key == "":
        return None
    code = lookup.get(key)
    if code is None:
        return np.zeros(len(codes), dtype=bool)
    return codes == code


def _regime_diagnostics(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
    *,
    primary_horizon: str = "6m",
    min_snapshots: int = 4,
) -> dict[str, list[dict[str, object]]]:
    regimes = pd.DataFrame(index=df.index)
    regimes["year"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    markets, _board_buckets, _listing_buckets = _ticker_bucket_arrays(df)
    regimes["market"] = markets
    regimes["cap_bucket"] = _cap_bucket_array(df)

    diagnostics: dict[str, list[dict[str, object]]] = {}
    candidate_series = pd.Series(candidate_predictions, index=df.index, dtype="float64")
    incumbent_series = pd.Series(incumbent_predictions, index=df.index, dtype="float64")
    for dimension in ("year", "market", "cap_bucket"):
        records: list[dict[str, object]] = []
        for label, index in regimes.groupby(dimension, sort=True).groups.items():
            group = df.loc[index]
            n_snapshots = int(pd.to_datetime(group["snapshot_date"]).nunique())
            if n_snapshots < min_snapshots:
                continue
            candidate_metrics = _evaluate_predictions(
                group,
                candidate_series.loc[index].to_numpy(dtype="float64"),
            )
            incumbent_metrics = _evaluate_predictions(
                group,
                incumbent_series.loc[index].to_numpy(dtype="float64"),
            )
            deltas = {
                horizon: round(
                    float(candidate_metrics[horizon]["spearman_rho"])
                    - float(incumbent_metrics[horizon]["spearman_rho"]),
                    6,
                )
                for horizon in EVAL_HORIZONS
            }
            records.append({
                "label": str(label),
                "rows": int(len(group)),
                "snapshots": n_snapshots,
                "deltas": deltas,
                "primary_horizon": primary_horizon,
                "candidate_primary": round(
                    float(candidate_metrics[primary_horizon]["spearman_rho"]),
                    6,
                ),
                "incumbent_primary": round(
                    float(incumbent_metrics[primary_horizon]["spearman_rho"]),
                    6,
                ),
                "candidate_6m": round(float(candidate_metrics["6m"]["spearman_rho"]), 6),
                "incumbent_6m": round(float(incumbent_metrics["6m"]["spearman_rho"]), 6),
            })
        diagnostics[dimension] = records
    return diagnostics


def _recent_positive_year_start(
    regime_diagnostics: Mapping[str, object],
    *,
    gate_start: str,
    primary_horizon: str,
) -> Optional[str]:
    records = regime_diagnostics.get("year")
    if not isinstance(records, list):
        return None
    gate_timestamp = pd.to_datetime(gate_start, errors="coerce")
    if pd.isna(gate_timestamp):
        return None
    best_year: Optional[int] = None
    for record in records:
        if not isinstance(record, Mapping):
            continue
        try:
            year = int(str(record.get("label")))
            delta = float((record.get("deltas") or {}).get(primary_horizon, 0.0))
        except (TypeError, ValueError):
            continue
        if delta <= 0.0:
            continue
        year_start = pd.Timestamp(year=year, month=1, day=1)
        if year_start <= gate_timestamp:
            continue
        if best_year is None or year > best_year:
            best_year = year
    if best_year is None:
        return None
    return pd.Timestamp(year=best_year, month=1, day=1).date().isoformat()


def _recent_positive_year_market_route(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
    *,
    gate_start: str,
    primary_horizon: str,
    min_snapshots: int = 4,
) -> Optional[tuple[str, dict[str, float]]]:
    gate_timestamp = pd.to_datetime(gate_start, errors="coerce")
    if pd.isna(gate_timestamp):
        return None

    groups = pd.DataFrame(index=df.index)
    snapshot_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    groups["year"] = snapshot_dates.dt.year.astype("Int64")
    markets, _board_buckets, _listing_buckets = _ticker_bucket_arrays(df)
    groups["market"] = markets
    candidate_series = pd.Series(candidate_predictions, index=df.index, dtype="float64")
    incumbent_series = pd.Series(incumbent_predictions, index=df.index, dtype="float64")

    best_year: Optional[int] = None
    best_markets: dict[str, float] = {}
    for (year, market), index in groups.groupby(["year", "market"], sort=True).groups.items():
        if pd.isna(year):
            continue
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        if year_start <= gate_timestamp:
            continue
        group = df.loc[index]
        n_snapshots = int(pd.to_datetime(group["snapshot_date"]).nunique())
        if n_snapshots < min_snapshots:
            continue
        candidate_metrics = _evaluate_predictions(
            group,
            candidate_series.loc[index].to_numpy(dtype="float64"),
        )
        incumbent_metrics = _evaluate_predictions(
            group,
            incumbent_series.loc[index].to_numpy(dtype="float64"),
        )
        delta = (
            float(candidate_metrics[primary_horizon]["spearman_rho"])
            - float(incumbent_metrics[primary_horizon]["spearman_rho"])
        )
        if delta <= 0.0:
            continue
        if best_year is None or year > best_year:
            best_year = year
            best_markets = {}
        if year == best_year:
            best_markets[str(market)] = 1.0

    if best_year is None or not best_markets:
        return None
    return pd.Timestamp(year=best_year, month=1, day=1).date().isoformat(), best_markets


def _recent_positive_window_market_route(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
    *,
    gate_start: str,
    primary_horizon: str,
    min_snapshots: int = 4,
    require_primary_top20_excess_non_degradation: bool = False,
) -> Optional[tuple[str, dict[str, float]]]:
    gate_timestamp = pd.to_datetime(gate_start, errors="coerce")
    if pd.isna(gate_timestamp):
        return None

    snapshot_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    valid_dates = snapshot_dates.dropna()
    if valid_dates.empty:
        return None

    month_starts = sorted(
        pd.Timestamp(start)
        for start in valid_dates.dt.to_period("M").dt.to_timestamp().unique()
        if pd.Timestamp(start) > gate_timestamp
    )
    if not month_starts:
        return None
    max_starts = _six_month_recent_segment_route_max_starts()
    month_starts = month_starts[-max_starts:]

    markets = pd.Series(_ticker_bucket_arrays(df)[0], index=df.index)
    candidate_series = pd.Series(candidate_predictions, index=df.index, dtype="float64")
    incumbent_series = pd.Series(incumbent_predictions, index=df.index, dtype="float64")
    incumbent_gate_rho = _evaluate_primary_spearman_rho(
        df,
        incumbent_predictions,
        primary_horizon,
    )
    incumbent_gate_top20_excess = (
        _primary_top20_excess_metrics(df, incumbent_predictions, primary_horizon)
        if require_primary_top20_excess_non_degradation
        else None
    )
    best_key: Optional[tuple[float, float, int, float]] = None
    best_route: Optional[tuple[str, dict[str, float]]] = None

    for start in month_starts:
        recent_mask = snapshot_dates >= start
        if int(snapshot_dates[recent_mask].nunique()) < min_snapshots:
            continue
        positive_markets: list[str] = []
        for market in sorted(str(market) for market in markets.dropna().unique()):
            market_mask = recent_mask & (markets == market)
            if int(snapshot_dates[market_mask].nunique()) < min_snapshots:
                continue
            group = df.loc[market_mask]
            candidate_rho = _evaluate_primary_spearman_rho(
                group,
                candidate_series.loc[market_mask].to_numpy(dtype="float64"),
                primary_horizon,
            )
            incumbent_rho = _evaluate_primary_spearman_rho(
                group,
                incumbent_series.loc[market_mask].to_numpy(dtype="float64"),
                primary_horizon,
            )
            market_delta = float(candidate_rho) - float(incumbent_rho)
            if market_delta > 0.0:
                positive_markets.append(market)
        if not positive_markets:
            continue

        for route_weights_by_market in product(
            SIX_MONTH_RECENT_MARKET_ROUTE_WEIGHTS,
            repeat=len(positive_markets),
        ):
            market_weights = {
                market: float(weight)
                for market, weight in zip(positive_markets, route_weights_by_market)
                if float(weight) > 0.0
            }
            if not market_weights:
                continue
            market_gate_weights = _market_blend_weight_vector(df, market_weights)
            route_weights = np.where(
                recent_mask.to_numpy(dtype=bool),
                market_gate_weights,
                0.0,
            )
            route_predictions = (
                route_weights * candidate_predictions
                + (1.0 - route_weights) * incumbent_predictions
            )
            route_rho = _evaluate_primary_spearman_rho(
                df,
                route_predictions,
                primary_horizon,
            )
            route_delta = route_rho - incumbent_gate_rho
            if route_delta <= 0.0:
                continue
            if (
                incumbent_gate_top20_excess is not None
                and not _primary_top20_excess_predictions_non_degraded(
                    df,
                    route_predictions,
                    incumbent_gate_top20_excess,
                    primary_horizon,
                )
            ):
                continue
            route_key = (
                route_delta,
                route_rho,
                int(start.value),
                -sum(float(weight) for weight in market_weights.values()),
            )
            if best_key is None or route_key > best_key:
                best_key = route_key
                best_route = (start.date().isoformat(), market_weights)

    return best_route


def _positive_segment_route(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
    *,
    primary_horizon: str,
    min_snapshots: int = 4,
    require_primary_top20_excess_non_degradation: bool = False,
) -> Optional[list[dict[str, object]]]:
    if "snapshot_date" not in df.columns:
        return None
    snapshot_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if int(snapshot_dates.dropna().nunique()) < min_snapshots:
        return None

    market_array, board_array, listing_array = _ticker_bucket_arrays(df)
    markets = pd.Series(market_array, index=df.index)
    cap_buckets = pd.Series(_cap_bucket_array(df), index=df.index)
    groups = pd.DataFrame(
        {
            "market": markets,
            "cap_bucket": cap_buckets,
            "board_bucket": pd.Series(board_array, index=df.index),
            "listing_bucket": pd.Series(listing_array, index=df.index),
        },
        index=df.index,
    )
    candidate_series = pd.Series(candidate_predictions, index=df.index, dtype="float64")
    incumbent_series = pd.Series(incumbent_predictions, index=df.index, dtype="float64")
    incumbent_gate_rho = _evaluate_primary_spearman_rho(
        df,
        incumbent_predictions,
        primary_horizon,
    )
    incumbent_gate_top20_excess = (
        _primary_top20_excess_metrics(df, incumbent_predictions, primary_horizon)
        if require_primary_top20_excess_non_degradation
        else None
    )
    route_weights_to_try = _six_month_recent_segment_route_weights()
    max_segments = _six_month_recent_segment_route_max_segments()
    max_candidates = _six_month_recent_segment_route_max_candidates()

    segment_candidates: list[tuple[float, dict[str, object]]] = []
    seen_route_keys: set[tuple[tuple[str, object], ...]] = set()
    for grouping in SIX_MONTH_GATE_SEGMENT_ROUTE_GROUPINGS:
        positive_segments: list[tuple[float, dict[str, object]]] = []
        for key_values, _ in groups.groupby(list(grouping), sort=True).groups.items():
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            route_conditions = {
                key: str(value)
                for key, value in zip(grouping, key_values)
            }
            segment_mask = pd.Series(True, index=df.index)
            for key, value in route_conditions.items():
                segment_mask &= groups[key] == value
            if int(snapshot_dates[segment_mask].nunique()) < min_snapshots:
                continue
            group = df.loc[segment_mask]
            candidate_rho = _evaluate_primary_spearman_rho(
                group,
                candidate_series.loc[segment_mask].to_numpy(dtype="float64"),
                primary_horizon,
            )
            incumbent_rho = _evaluate_primary_spearman_rho(
                group,
                incumbent_series.loc[segment_mask].to_numpy(dtype="float64"),
                primary_horizon,
            )
            segment_delta = float(candidate_rho) - float(incumbent_rho)
            if segment_delta <= 0.0:
                continue
            positive_segments.append((segment_delta, route_conditions))

        for _segment_delta, route_conditions in sorted(
            positive_segments,
            key=lambda item: item[0],
            reverse=True,
        )[:max_segments]:
            for weight in route_weights_to_try:
                route = {
                    **route_conditions,
                    "candidate_weight": float(weight),
                }
                route_key = tuple(sorted(route.items()))
                if route_key in seen_route_keys:
                    continue
                seen_route_keys.add(route_key)
                route_weights = _conditional_route_weight_vector(df, [route])
                route_predictions = (
                    route_weights * candidate_predictions
                    + (1.0 - route_weights) * incumbent_predictions
                )
                route_rho = _evaluate_primary_spearman_rho(
                    df,
                    route_predictions,
                    primary_horizon,
                )
                route_delta = float(route_rho) - incumbent_gate_rho
                if route_delta <= 0.0:
                    continue
                if (
                    incumbent_gate_top20_excess is not None
                    and not _primary_top20_excess_predictions_non_degraded(
                        df,
                        route_predictions,
                        incumbent_gate_top20_excess,
                        primary_horizon,
                    )
                ):
                    continue
                segment_candidates.append((route_delta, route))

    selected_routes: list[dict[str, object]] = []
    selected_delta = 0.0
    for _route_delta, route in sorted(
        segment_candidates,
        key=lambda item: item[0],
        reverse=True,
    )[:max_candidates]:
        trial_routes = [*selected_routes, route]
        route_weights = _conditional_route_weight_vector(df, trial_routes)
        route_predictions = (
            route_weights * candidate_predictions
            + (1.0 - route_weights) * incumbent_predictions
        )
        route_rho = _evaluate_primary_spearman_rho(
            df,
            route_predictions,
            primary_horizon,
        )
        route_delta = route_rho - incumbent_gate_rho
        if route_delta <= selected_delta:
            continue
        if (
            incumbent_gate_top20_excess is not None
            and not _primary_top20_excess_predictions_non_degraded(
                df,
                route_predictions,
                incumbent_gate_top20_excess,
                primary_horizon,
            )
        ):
            continue
        selected_routes = trial_routes
        selected_delta = route_delta

    return selected_routes or None


def _recent_positive_window_segment_route(
    df: pd.DataFrame,
    candidate_predictions: np.ndarray,
    incumbent_predictions: np.ndarray,
    *,
    gate_start: str,
    primary_horizon: str,
    min_snapshots: int = 4,
    require_primary_top20_excess_non_degradation: bool = False,
) -> Optional[tuple[str, list[dict[str, object]]]]:
    gate_timestamp = pd.to_datetime(gate_start, errors="coerce")
    if pd.isna(gate_timestamp):
        return None

    snapshot_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    valid_dates = snapshot_dates.dropna()
    if valid_dates.empty:
        return None

    month_starts = [
        pd.Timestamp(start)
        for start in sorted(valid_dates.dt.to_period("M").dt.to_timestamp().unique())
        if pd.Timestamp(start) > gate_timestamp
    ]
    if not month_starts:
        return None
    max_starts = _six_month_recent_segment_route_max_starts()
    month_starts = month_starts[-max_starts:]

    market_array, board_array, listing_array = _ticker_bucket_arrays(df)
    markets = pd.Series(market_array, index=df.index)
    cap_buckets = pd.Series(_cap_bucket_array(df), index=df.index)
    board_buckets = pd.Series(board_array, index=df.index)
    listing_buckets = pd.Series(listing_array, index=df.index)
    candidate_series = pd.Series(candidate_predictions, index=df.index, dtype="float64")
    incumbent_series = pd.Series(incumbent_predictions, index=df.index, dtype="float64")
    incumbent_gate_rho = _evaluate_primary_spearman_rho(
        df,
        incumbent_predictions,
        primary_horizon,
    )
    incumbent_gate_top20_excess = (
        _primary_top20_excess_metrics(df, incumbent_predictions, primary_horizon)
        if require_primary_top20_excess_non_degradation
        else None
    )
    route_weights_to_try = _six_month_recent_segment_route_weights()
    max_segments = _six_month_recent_segment_route_max_segments()
    max_candidates = _six_month_recent_segment_route_max_candidates()
    best_key: Optional[tuple[float, float, int, float]] = None
    best_route: Optional[tuple[str, list[dict[str, object]]]] = None

    for start in month_starts:
        recent_mask = snapshot_dates >= start
        if int(snapshot_dates[recent_mask].nunique()) < min_snapshots:
            continue

        groups = pd.DataFrame(
            {
                "market": markets,
                "cap_bucket": cap_buckets,
                "board_bucket": board_buckets,
                "listing_bucket": listing_buckets,
            },
            index=df.index,
        )
        segment_candidates: list[tuple[float, dict[str, object]]] = []
        seen_route_keys: set[tuple[tuple[str, object], ...]] = set()
        for grouping in SIX_MONTH_RECENT_SEGMENT_ROUTE_GROUPINGS:
            positive_segments: list[tuple[float, dict[str, object]]] = []
            for key_values, _ in (
                groups[recent_mask].groupby(list(grouping), sort=True).groups.items()
            ):
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                route_conditions = {
                    key: str(value)
                    for key, value in zip(grouping, key_values)
                }
                segment_mask = recent_mask.copy()
                for key, value in route_conditions.items():
                    segment_mask &= groups[key] == value
                if int(snapshot_dates[segment_mask].nunique()) < min_snapshots:
                    continue
                group = df.loc[segment_mask]
                candidate_rho = _evaluate_primary_spearman_rho(
                    group,
                    candidate_series.loc[segment_mask].to_numpy(dtype="float64"),
                    primary_horizon,
                )
                incumbent_rho = _evaluate_primary_spearman_rho(
                    group,
                    incumbent_series.loc[segment_mask].to_numpy(dtype="float64"),
                    primary_horizon,
                )
                segment_delta = float(candidate_rho) - float(incumbent_rho)
                if segment_delta <= 0.0:
                    continue
                positive_segments.append((segment_delta, route_conditions))

            for _segment_delta, route_conditions in sorted(
                positive_segments,
                key=lambda item: item[0],
                reverse=True,
            )[:max_segments]:
                for weight in route_weights_to_try:
                    route = {
                        "start_date": start.date().isoformat(),
                        **route_conditions,
                        "candidate_weight": float(weight),
                    }
                    route_key = tuple(sorted(route.items()))
                    if route_key in seen_route_keys:
                        continue
                    seen_route_keys.add(route_key)
                    route_weights = _conditional_route_weight_vector(df, [route])
                    route_predictions = (
                        route_weights * candidate_predictions
                        + (1.0 - route_weights) * incumbent_predictions
                    )
                    route_rho = _evaluate_primary_spearman_rho(
                        df,
                        route_predictions,
                        primary_horizon,
                    )
                    route_delta = float(route_rho) - incumbent_gate_rho
                    if route_delta <= 0.0:
                        continue
                    if (
                        incumbent_gate_top20_excess is not None
                        and not _primary_top20_excess_predictions_non_degraded(
                            df,
                            route_predictions,
                            incumbent_gate_top20_excess,
                            primary_horizon,
                        )
                    ):
                        continue
                    segment_candidates.append((route_delta, route))

        selected_routes: list[dict[str, object]] = []
        selected_delta = 0.0
        for _route_delta, route in sorted(
            segment_candidates,
            key=lambda item: item[0],
            reverse=True,
        )[:max_candidates]:
            trial_routes = [*selected_routes, route]
            route_weights = _conditional_route_weight_vector(df, trial_routes)
            route_predictions = (
                route_weights * candidate_predictions
                + (1.0 - route_weights) * incumbent_predictions
            )
            route_rho = _evaluate_primary_spearman_rho(
                df,
                route_predictions,
                primary_horizon,
            )
            route_delta = route_rho - incumbent_gate_rho
            if route_delta <= selected_delta:
                continue
            if (
                incumbent_gate_top20_excess is not None
                and not _primary_top20_excess_predictions_non_degraded(
                    df,
                    route_predictions,
                    incumbent_gate_top20_excess,
                    primary_horizon,
                )
            ):
                continue
            selected_routes = trial_routes
            selected_delta = route_delta
            route_key = (
                route_delta,
                route_rho,
                int(start.value),
                -sum(float(item.get("candidate_weight", 1.0)) for item in trial_routes),
            )
            if best_key is None or route_key > best_key:
                best_key = route_key
                best_route = (start.date().isoformat(), trial_routes)

    return best_route


def _select_full_eval_candidates(
    candidates: list[dict],
    *,
    limit: int,
    primary_horizon: str,
) -> list[dict]:
    """Keep the full-eval shortlist strong while reserving slots for distinct strategies."""
    if limit <= 0 or not candidates:
        return []

    use_walk_forward_rank = primary_horizon == "6m" and any(
        isinstance(candidate.get("walk_forward"), Mapping)
        and isinstance(candidate["walk_forward"].get("mean_deltas"), Mapping)
        for candidate in candidates
    )
    use_gate_probe_rank = primary_horizon == "6m" and any(
        np.isfinite(_candidate_gate_probe_rho(candidate, primary_horizon))
        for candidate in candidates
    )

    def rank_key(candidate: Mapping[str, object]) -> tuple[float, float, float, float]:
        train_rho = _candidate_primary_rho(candidate, primary_horizon)
        if use_walk_forward_rank:
            mean_delta, median_delta, _walk_forward_train_rho = _candidate_walk_forward_key(
                candidate,
                primary_horizon,
            )
        else:
            mean_delta = median_delta = float("-inf")
        if use_gate_probe_rank:
            return (
                _candidate_gate_probe_rho(candidate, primary_horizon),
                mean_delta,
                median_delta,
                train_rho,
            )
        if use_walk_forward_rank:
            return (mean_delta, median_delta, train_rho, float("-inf"))
        return (train_rho, float("-inf"), float("-inf"), float("-inf"))

    ranked = sorted(
        candidates,
        key=rank_key,
        reverse=True,
    )
    selected: list[dict] = []
    selected_ids: set[int] = set()

    def add(candidate: dict) -> None:
        if len(selected) >= limit or id(candidate) in selected_ids:
            return
        selected.append(candidate)
        selected_ids.add(id(candidate))

    def add_first_where(predicate) -> bool:
        seen_families = {
            (
                candidate.get("prior_strategy"),
                candidate.get("feature_set"),
                candidate.get("model_kind"),
                candidate.get("target"),
            )
            for candidate in selected
        }
        for candidate in ranked:
            family = (
                candidate.get("prior_strategy"),
                candidate.get("feature_set"),
                candidate.get("model_kind"),
                candidate.get("target"),
            )
            if family in seen_families:
                continue
            if predicate(candidate):
                before = len(selected)
                add(candidate)
                if len(selected) > before:
                    return True
        return False

    def add_distinct(fields: tuple[str, ...]) -> bool:
        seen = {tuple(candidate.get(field) for field in fields) for candidate in selected}
        for candidate in ranked:
            value = tuple(candidate.get(field) for field in fields)
            if value in seen:
                continue
            add(candidate)
            seen.add(value)
            if len(selected) >= limit:
                return True
        return False

    if primary_horizon == "6m":
        add(ranked[0])
        for predicate in (
            lambda candidate: candidate.get("target_source") == "anchor_teacher",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "cross_sectional"
            and _is_recent_market_ridge(candidate.get("model_kind")),
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "cross_sectional"
            and candidate.get("model_kind") == "market_ridge",
            lambda candidate: candidate.get("feature_set") == "cross_sectional",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "core",
            lambda candidate: candidate.get("prior_strategy") == "rolling_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and _is_recent_market_ridge(candidate.get("model_kind")),
            lambda candidate: candidate.get("prior_strategy") == "rolling_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and _is_recent_market_ridge(candidate.get("model_kind")),
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and _is_recent_market_ridge(candidate.get("model_kind")),
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "ridge",
            lambda candidate: candidate.get("prior_strategy") == "rolling_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "ridge",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "ridge",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_6m",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_6m_market",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_6m_soft",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_802",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_802_market",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_8515",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_8515_market",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_901",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_901_market",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_7525",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_703",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_weighted_703_market",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge"
            and candidate.get("target") == "target_rank_6m_extreme",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors"
            and candidate.get("feature_set") == "short_horizon",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors",
            lambda candidate: candidate.get("target") == "target_rank_6m_soft",
            lambda candidate: candidate.get("target") == "target_rank_weighted_802",
            lambda candidate: candidate.get("target") == "target_rank_weighted_802_market",
            lambda candidate: candidate.get("target") == "target_rank_weighted_8515",
            lambda candidate: candidate.get("target") == "target_rank_weighted_8515_market",
            lambda candidate: candidate.get("target") == "target_rank_weighted_901",
            lambda candidate: candidate.get("target") == "target_rank_weighted_901_market",
            lambda candidate: candidate.get("target") == "target_rank_weighted_7525",
            lambda candidate: candidate.get("target") == "target_rank_weighted_703_market",
            lambda candidate: candidate.get("target") == "target_rank_6m_market",
            lambda candidate: candidate.get("target") == "target_rank_6m",
            lambda candidate: candidate.get("prior_strategy") == "rolling_ticker_priors",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors",
        ):
            add_first_where(predicate)
            if len(selected) >= limit:
                return selected

    if primary_horizon == "1w":
        for predicate in (
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("model_kind") == "market_ridge_recent"
            and candidate.get("target") == "target_rank_1w",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon"
            and candidate.get("target") == "target_rank_1w",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("feature_set") == "short_horizon",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("model_kind") == "market_ridge_recent",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors"
            and candidate.get("model_kind") == "market_ridge",
            lambda candidate: candidate.get("prior_strategy") == "no_ticker_priors",
            lambda candidate: candidate.get("prior_strategy") == "ticker_priors",
        ):
            add_first_where(predicate)
            if len(selected) >= limit:
                return selected

    if not selected:
        add(ranked[0])
    for fields in (
        ("prior_strategy", "feature_set", "target"),
        ("prior_strategy", "backend"),
        ("prior_strategy",),
        ("feature_set",),
        ("backend",),
        ("model_kind",),
        ("target",),
    ):
        if add_distinct(fields):
            return selected

    for candidate in ranked:
        add(candidate)
        if len(selected) >= limit:
            break
    return selected


def _select_balanced_walk_forward_specs(
    candidates: Sequence[dict],
    *,
    limit: int = STRICT_WALK_FORWARD_CANDIDATE_LIMIT,
) -> list[dict]:
    """Select a configuration-balanced panel without using fitted outcomes."""
    if limit <= 0 or not candidates:
        return []
    fields = (
        "model_kind",
        "prior_strategy",
        "feature_set",
        "target",
        "ridge_lambda",
    )
    remaining = list(candidates)
    selected: list[dict] = []
    counts: dict[str, dict[object, int]] = {field: {} for field in fields}
    field_pairs = tuple(combinations(fields, 2))
    pair_counts: dict[tuple[str, str], dict[tuple[object, object], int]] = {pair: {} for pair in field_pairs}
    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                sum(1.0 / (1.0 + counts[field].get(remaining[index].get(field), 0)) for field in fields)
                + 0.5
                * sum(
                    1.0
                    / (
                        1.0
                        + pair_counts[pair].get(
                            tuple(remaining[index].get(field) for field in pair),
                            0,
                        )
                    )
                    for pair in field_pairs
                ),
                -index,
            ),
        )
        candidate = remaining.pop(best_index)
        selected.append(candidate)
        for field in fields:
            value = candidate.get(field)
            counts[field][value] = counts[field].get(value, 0) + 1
        for pair in field_pairs:
            value = tuple(candidate.get(field) for field in pair)
            pair_counts[pair][value] = pair_counts[pair].get(value, 0) + 1
    return selected


def _walk_forward_candidate_limit(
    full_eval_candidate_limit: int,
    *,
    primary_horizon: str,
) -> int:
    if full_eval_candidate_limit <= 0:
        return 0
    return full_eval_candidate_limit


def _candidate_primary_rho(candidate: Mapping[str, object], primary_horizon: str) -> float:
    try:
        metrics = candidate["metrics"]
        if not isinstance(metrics, Mapping):
            return float("-inf")
        horizon_metrics = metrics[primary_horizon]
        if not isinstance(horizon_metrics, Mapping):
            return float("-inf")
        return float(horizon_metrics["spearman_rho"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def _candidate_gate_probe_rho(
    candidate: Mapping[str, object],
    primary_horizon: str,
) -> float:
    try:
        gate_probe = candidate.get("gate_probe")
        if not isinstance(gate_probe, Mapping):
            return float("-inf")
        metrics = gate_probe.get("metrics")
        if not isinstance(metrics, Mapping):
            return float("-inf")
        horizon_metrics = metrics[primary_horizon]
        if not isinstance(horizon_metrics, Mapping):
            return float("-inf")
        return float(horizon_metrics["spearman_rho"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def _payload_train_rho(
    payload: Optional[Mapping[str, object]],
    primary_horizon: str,
) -> Optional[float]:
    if payload is None:
        return None
    if not _payload_cached_metrics_are_safe(payload):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    train_metrics = metadata.get("train_metrics")
    if not isinstance(train_metrics, Mapping):
        return None
    horizon_metrics = train_metrics.get(primary_horizon)
    if not isinstance(horizon_metrics, Mapping):
        return None
    rho = horizon_metrics.get("spearman_rho")
    try:
        return None if rho is None else float(rho)
    except (TypeError, ValueError):
        return None


def _load_anchor_payloads(
    model_paths: Optional[Sequence[Path]],
    *,
    output_model_path: Path,
    allow_output_model: bool = False,
) -> list[dict[str, object]]:
    if not model_paths:
        return []
    output_resolved = output_model_path.resolve()
    anchors: list[dict[str, object]] = []
    seen: set[Path] = set()
    for model_path in model_paths:
        path = Path(model_path)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or (resolved == output_resolved and not allow_output_model):
            continue
        seen.add(resolved)
        if not path.exists():
            logger.warning("ML anchor model path does not exist: %s", path)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read ML anchor model %s: %s", path, exc)
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            logger.warning("Skipping unsupported ML anchor model: %s", path)
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        metadata.setdefault("model_path", str(path))
        anchors.append({"path": path, "payload": payload})
    return anchors


def _candidate_walk_forward_key(
    candidate: Mapping[str, object],
    primary_horizon: str,
) -> tuple[float, float, float]:
    walk_forward = candidate.get("walk_forward")
    if not isinstance(walk_forward, Mapping):
        return (float("-inf"), float("-inf"), _candidate_primary_rho(candidate, primary_horizon))
    mean_deltas = walk_forward.get("mean_deltas")
    median_deltas = walk_forward.get("median_deltas")
    mean_delta = (
        float(mean_deltas.get(primary_horizon, float("-inf")))
        if isinstance(mean_deltas, Mapping)
        else float("-inf")
    )
    median_delta = (
        float(median_deltas.get(primary_horizon, float("-inf")))
        if isinstance(median_deltas, Mapping)
        else float("-inf")
    )
    return (mean_delta, median_delta, _candidate_primary_rho(candidate, primary_horizon))


def _select_promotion_gate_candidates(
    candidates: list[dict],
    *,
    limit: int,
    primary_horizon: str,
) -> list[dict]:
    if limit <= 0 or not candidates:
        return []

    ranked_train = sorted(
        candidates,
        key=lambda candidate: _candidate_primary_rho(candidate, primary_horizon),
        reverse=True,
    )
    ranked_walk_forward = sorted(
        candidates,
        key=lambda candidate: _candidate_walk_forward_key(candidate, primary_horizon),
        reverse=True,
    )
    use_gate_probe_rank = primary_horizon == "6m" and any(
        np.isfinite(_candidate_gate_probe_rho(candidate, primary_horizon))
        for candidate in candidates
    )
    if use_gate_probe_rank:
        ranked_gate_probe = sorted(
            candidates,
            key=lambda candidate: (
                _candidate_gate_probe_rho(candidate, primary_horizon),
                _candidate_primary_rho(candidate, primary_horizon),
            ),
            reverse=True,
        )
    else:
        ranked_gate_probe = ranked_train
    ranked_temporal_anchors = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.get("target_source") == "temporal_regime"
            and np.isfinite(
                float(candidate.get("temporal_anchor_gate_rho", float("-inf")))
            )
        ],
        key=lambda candidate: (
            float(candidate.get("temporal_anchor_gate_rho", float("-inf"))),
            _candidate_primary_rho(candidate, primary_horizon),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    selected_ids: set[int] = set()

    def add(candidate: dict) -> None:
        if len(selected) >= limit or id(candidate) in selected_ids:
            return
        selected.append(candidate)
        selected_ids.add(id(candidate))

    def add_first_where(predicate, ranked_candidates: Sequence[dict] = ranked_train) -> None:
        for candidate in ranked_candidates:
            if predicate(candidate):
                add(candidate)
                return

    def add_distinct(
        fields: tuple[str, ...],
        ranked_candidates: Sequence[dict] = ranked_train,
    ) -> bool:
        seen = {tuple(candidate.get(field) for field in fields) for candidate in selected}
        for candidate in ranked_candidates:
            value = tuple(candidate.get(field) for field in fields)
            if value in seen:
                continue
            add(candidate)
            seen.add(value)
            if len(selected) >= limit:
                return True
        return False

    if use_gate_probe_rank:
        if primary_horizon == "6m" and ranked_temporal_anchors:
            add(ranked_temporal_anchors[0])
        for candidate in ranked_gate_probe:
            add(candidate)
            if len(selected) >= limit:
                return selected
        return selected

    add(ranked_train[0])
    # Temporal anchors protect the holdout period, but they cannot improve gate
    # rho when the post-transition member is the incumbent. Keep one in the
    # shortlist, while still letting raw/ensemble candidates reach the gate.
    add_first_where(lambda candidate: candidate.get("target_source") == "temporal_regime")
    add_first_where(lambda candidate: candidate.get("target") == "ensemble")
    add_first_where(lambda candidate: candidate.get("target") == "payload_member_ensemble")
    add_first_where(lambda candidate: candidate.get("target_source") == "anchor_teacher")
    add(ranked_walk_forward[0])

    if primary_horizon == "6m":
        for fields in (
            ("prior_strategy",),
            ("model_kind",),
            ("feature_set",),
            ("target", "model_kind", "ridge_lambda"),
            ("target", "model_kind"),
            ("target",),
        ):
            if add_distinct(fields, ranked_walk_forward):
                return selected

    for fields in (
        ("target", "model_kind", "ridge_lambda"),
        ("target", "model_kind"),
        ("target",),
        ("model_kind",),
    ):
        if add_distinct(fields):
            return selected

    for candidate in ranked_train:
        add(candidate)
        if len(selected) >= limit:
            break
    return selected


def _weighted_ensemble_member_name(candidate: Mapping[str, object], weight: float) -> str:
    return f"{_candidate_search_name(candidate)}*{weight:g}"


def _build_full_eval_ensembles(
    full_candidates: list[dict],
    train_snapshots: pd.DataFrame,
    *,
    train_baseline_rho: float,
    primary_horizon: str,
    pool_limit: int = SIX_MONTH_ENSEMBLE_POOL_LIMIT,
    max_ensembles: int = SIX_MONTH_ENSEMBLE_CANDIDATE_LIMIT,
) -> list[dict]:
    """Build weighted ensembles from compatible full-eval candidates."""
    groups: dict[tuple[object, ...], list[dict]] = {}
    for candidate in full_candidates:
        predictions = candidate.get("_full_predictions")
        if predictions is None:
            continue
        key = (
            tuple(candidate.get("feature_names", ())),
            candidate.get("prior_strategy"),
            candidate.get("feature_set"),
            bool(candidate.get("prefer_row_priors")),
        )
        groups.setdefault(key, []).append(candidate)

    ensembles: list[dict] = []
    seen_members: set[tuple[str, ...]] = set()
    for group_candidates in groups.values():
        ranked = sorted(
            group_candidates,
            key=lambda candidate: float(candidate["metrics"][primary_horizon]["spearman_rho"]),
            reverse=True,
        )[:pool_limit]
        combos: list[tuple[tuple[dict, float], ...]] = []
        for left_index, left in enumerate(ranked):
            for right in ranked[left_index + 1 :]:
                for left_weight, right_weight in SIX_MONTH_ENSEMBLE_PAIR_WEIGHTS:
                    combos.append(((left, left_weight), (right, right_weight)))
        if len(ranked) >= 3:
            combos.append(tuple((candidate, 1.0 / 3.0) for candidate in ranked[:3]))

        for combo in combos:
            if len(ensembles) >= max_ensembles:
                break
            member_names = tuple(
                _weighted_ensemble_member_name(candidate, weight)
                for candidate, weight in combo
            )
            if member_names in seen_members:
                continue
            seen_members.add(member_names)
            weights = np.asarray([weight for _candidate, weight in combo], dtype="float64")
            predictions = np.average(
                [
                    np.asarray(candidate["_full_predictions"], dtype="float64")
                    for candidate, _weight in combo
                ],
                weights=weights,
                axis=0,
            )
            metrics = _evaluate_predictions(train_snapshots, predictions)
            rho = float(metrics[primary_horizon]["spearman_rho"])
            if not np.isfinite(rho):
                continue
            first = combo[0][0]
            models: list[dict[str, object]] = []
            for member, weight in combo:
                models.extend(_candidate_model_payloads(member, weight_multiplier=weight))
            ensemble = {
                "target": "ensemble",
                "ridge_lambda": 0.0,
                "backend": "ensemble",
                "model_kind": "ensemble",
                "feature_set": first["feature_set"],
                "feature_names": first["feature_names"],
                "prior_strategy": first["prior_strategy"],
                "prefer_row_priors": bool(first.get("prefer_row_priors")),
                "priors": first["priors"],
                "models": models,
                "metrics": metrics,
                "sample_metrics": metrics,
                "sample_improvement": _improvement_ratio(rho, train_baseline_rho),
                "improvement": _improvement_ratio(rho, train_baseline_rho),
                "sample_rows": int(
                    sum(int(member.get("sample_rows", 0)) for member, _weight in combo)
                ),
                "sample_seed": 0,
                "ensemble_members": list(member_names),
                "_full_predictions": predictions.astype("float32", copy=False),
            }
            ensembles.append(ensemble)
            logger.info(
                "ML full-eval ensemble members=%s %s_rho=%.6f improvement=%.2f%%",
                " | ".join(member_names),
                primary_horizon,
                rho,
                float(ensemble["improvement"]) * 100.0,
            )
        if len(ensembles) >= max_ensembles:
            break
    return ensembles


def _build_payload_member_full_eval_ensembles(
    full_candidates: list[dict],
    train_snapshots: pd.DataFrame,
    *,
    train_baseline_rho: float,
    primary_horizon: str,
    max_ensembles: int = SIX_MONTH_PAYLOAD_MEMBER_ENSEMBLE_LIMIT,
) -> list[dict]:
    """Build train-scored payload-member ensembles across incompatible feature sets."""
    if max_ensembles <= 0:
        return []
    ranked = sorted(
        [
            candidate
            for candidate in full_candidates
            if candidate.get("_full_predictions") is not None
        ],
        key=lambda candidate: float(candidate["metrics"][primary_horizon]["spearman_rho"]),
        reverse=True,
    )
    if len(ranked) < 2:
        return []

    combos: list[tuple[tuple[dict, float], tuple[dict, float]]] = []
    for left_index, left in enumerate(ranked):
        for right in ranked[left_index + 1 :]:
            if (
                left.get("feature_names") == right.get("feature_names")
                and left.get("prior_strategy") == right.get("prior_strategy")
            ):
                continue
            for left_weight, right_weight in SIX_MONTH_PAYLOAD_MEMBER_ENSEMBLE_WEIGHTS:
                combos.append(((left, left_weight), (right, right_weight)))

    ensembles: list[dict] = []
    seen_members: set[tuple[str, ...]] = set()
    for combo in combos:
        if len(ensembles) >= max_ensembles:
            break
        member_names = tuple(
            _weighted_ensemble_member_name(candidate, weight)
            for candidate, weight in combo
        )
        if member_names in seen_members:
            continue
        seen_members.add(member_names)
        weights = np.asarray([weight for _candidate, weight in combo], dtype="float64")
        predictions = np.average(
            [
                np.asarray(candidate["_full_predictions"], dtype="float64")
                for candidate, _weight in combo
            ],
            weights=weights,
            axis=0,
        )
        metrics = _evaluate_predictions(train_snapshots, predictions)
        rho = float(metrics[primary_horizon]["spearman_rho"])
        if not np.isfinite(rho):
            continue
        ensemble = {
            "target": "payload_member_ensemble",
            "ridge_lambda": 0.0,
            "backend": "payload_member_ensemble",
            "model_kind": "payload_member_ensemble",
            "feature_set": "mixed",
            "feature_names": [],
            "prior_strategy": "mixed",
            "prefer_row_priors": False,
            "priors": {},
            "metrics": metrics,
            "sample_metrics": metrics,
            "sample_improvement": _improvement_ratio(rho, train_baseline_rho),
            "improvement": _improvement_ratio(rho, train_baseline_rho),
            "sample_rows": int(
                sum(int(member.get("sample_rows", 0)) for member, _weight in combo)
            ),
            "sample_seed": 0,
            "ensemble_members": list(member_names),
            "payload_member_candidates": [
                {"weight": float(weight), "candidate": member}
                for member, weight in combo
            ],
            "_full_predictions": predictions.astype("float32", copy=False),
        }
        ensembles.append(ensemble)
        logger.info(
            "ML full-eval payload-member ensemble members=%s %s_rho=%.6f improvement=%.2f%%",
            " | ".join(member_names),
            primary_horizon,
            rho,
            float(ensemble["improvement"]) * 100.0,
        )
    return ensembles


def _score_anchor_payloads_on_gate(
    anchor_payloads: Sequence[Mapping[str, object]],
    *,
    train_snapshots: pd.DataFrame,
    gate_snapshots: pd.DataFrame,
    primary_horizon: str,
) -> list[dict[str, object]]:
    """Rank anchors by current holdout performance instead of stale artifact metadata."""
    scored_anchors: list[dict[str, object]] = []
    for anchor in anchor_payloads:
        payload = anchor.get("payload")
        if not isinstance(payload, Mapping):
            continue
        anchor_path = str(anchor.get("path", ""))
        try:
            frame = (
                _attach_asof_ticker_priors(gate_snapshots, train_snapshots)
                if _payload_uses_row_priors(payload)
                else gate_snapshots
            )
            predictions = predict_ml_ranker_payload(frame, payload)
            gate_rho = _evaluate_primary_spearman_rho(
                gate_snapshots,
                predictions,
                primary_horizon,
            )
        except Exception as exc:
            logger.warning("Could not score ML anchor model %s on gate: %s", anchor_path, exc)
            gate_rho = float("-inf")
        scored = dict(anchor)
        scored["gate_rho"] = float(gate_rho)
        scored_anchors.append(scored)

    scored_anchors.sort(
        key=lambda anchor: (float(anchor.get("gate_rho", float("-inf"))), str(anchor.get("path", ""))),
        reverse=True,
    )
    if scored_anchors:
        logger.info(
            "ML gate-scored anchors: %s",
            " | ".join(
                f"{anchor.get('path')}={float(anchor.get('gate_rho', float('-inf'))):.6f}"
                for anchor in scored_anchors
            ),
        )
    return scored_anchors


def _build_temporal_anchor_candidates(
    full_candidates: list[dict],
    *,
    train_baseline_rho: float,
    primary_horizon: str,
    gate_start: str,
    anchor_payloads: Sequence[Mapping[str, object]],
    promotion_min_train_rho: Optional[float],
    max_candidates: int = SIX_MONTH_TEMPORAL_ANCHOR_CANDIDATE_LIMIT,
) -> list[dict]:
    """Build explicit train-period/candidate and gate-period/anchor payload routers."""
    if primary_horizon != "6m" or max_candidates <= 0 or not anchor_payloads:
        return []

    ranked_candidates = sorted(
        [
            candidate
            for candidate in full_candidates
            if candidate.get("_full_predictions") is not None
            and candidate.get("target_source") != "temporal_regime"
            and np.isfinite(_candidate_primary_rho(candidate, primary_horizon))
        ],
        key=lambda candidate: _candidate_primary_rho(candidate, primary_horizon),
        reverse=True,
    )
    if promotion_min_train_rho is not None:
        ranked_candidates = [
            candidate
            for candidate in ranked_candidates
            if _candidate_primary_rho(candidate, primary_horizon) >= promotion_min_train_rho
        ]
    if not ranked_candidates:
        return []

    ranked_anchors = sorted(
        [
            anchor
            for anchor in anchor_payloads
            if isinstance(anchor.get("payload"), Mapping)
        ],
        key=lambda anchor: (float(anchor.get("gate_rho", float("-inf"))), str(anchor.get("path", ""))),
        reverse=True,
    )

    temporal_candidates: list[dict] = []
    pairs: list[tuple[Mapping[str, object], dict]] = []
    for anchor in ranked_anchors[:max_candidates]:
        for train_candidate in ranked_candidates[:max_candidates]:
            pairs.append((anchor, train_candidate))
    pairs.sort(
        key=lambda item: (
            float(item[0].get("gate_rho", float("-inf"))),
            _candidate_primary_rho(item[1], primary_horizon),
        ),
        reverse=True,
    )

    for anchor, train_candidate in pairs:
        anchor_payload = anchor.get("payload")
        if not isinstance(anchor_payload, Mapping):
            continue
        anchor_path = str(anchor.get("path", ""))
        metrics = train_candidate["metrics"]
        rho = float(metrics[primary_horizon]["spearman_rho"])
        temporal_candidate = {
            "target": "temporal_anchor",
            "target_source": "temporal_regime",
            "ridge_lambda": 0.0,
            "backend": "temporal_payload_blend",
            "model_kind": "temporal_payload_blend",
            "feature_set": "temporal",
            "feature_names": [],
            "prior_strategy": "temporal",
            "prefer_row_priors": False,
            "priors": {},
            "metrics": metrics,
            "sample_metrics": train_candidate["sample_metrics"],
            "sample_improvement": train_candidate["sample_improvement"],
            "improvement": _improvement_ratio(rho, train_baseline_rho),
            "sample_rows": int(train_candidate.get("sample_rows", 0)),
            "sample_seed": int(train_candidate.get("sample_seed", 0)),
            "ensemble_members": [
                f"before:{_candidate_search_name(train_candidate)}",
                f"from:{gate_start}:{anchor_path}",
            ],
            "temporal_train_candidate": train_candidate,
            "temporal_anchor_payload": anchor_payload,
            "temporal_anchor_path": anchor_path,
            "temporal_anchor_gate_rho": float(anchor.get("gate_rho", float("nan"))),
            "temporal_transition_date": gate_start,
            "_full_predictions": np.asarray(
                train_candidate["_full_predictions"],
                dtype="float32",
            ),
        }
        temporal_candidates.append(temporal_candidate)
        logger.info(
            "ML full-eval temporal anchor train_member=%s anchor=%s "
            "anchor_gate_%s_rho=%s transition=%s %s_rho=%.6f improvement=%.2f%%",
            _candidate_search_name(train_candidate),
            anchor_path,
            primary_horizon,
            (
                f"{float(anchor.get('gate_rho')):.6f}"
                if anchor.get("gate_rho") is not None
                and np.isfinite(float(anchor.get("gate_rho")))
                else "unknown"
            ),
            gate_start,
            primary_horizon,
            rho,
            float(temporal_candidate["improvement"]) * 100.0,
        )
        if len(temporal_candidates) >= max_candidates:
            return temporal_candidates
    return temporal_candidates


def _gate_result_score(gate_result: Any, primary_horizon: str) -> tuple[float, float]:
    deltas = getattr(gate_result, "deltas", {}) or {}
    try:
        primary_delta = float(deltas.get(primary_horizon, float("-inf")))
    except (TypeError, ValueError):
        primary_delta = float("-inf")
    try:
        utility = float(getattr(gate_result, "weighted_utility", float("-inf")))
    except (TypeError, ValueError):
        utility = float("-inf")
    return primary_delta, utility


def _is_better_gate_result(
    candidate_gate: Any,
    incumbent_gate: Optional[Any],
    *,
    primary_horizon: str,
) -> bool:
    if incumbent_gate is None:
        return True
    return _gate_result_score(candidate_gate, primary_horizon) > _gate_result_score(
        incumbent_gate,
        primary_horizon,
    )


def _ridge_candidate_backends(backend: str, primary_horizon: str) -> tuple[str, ...]:
    if primary_horizon == "1w" and backend in {"auto", "mlx", "mlx-cg"}:
        return tuple(dict.fromkeys((backend, "mlx-adam")))
    return (backend,)


def _latest_supported_daily_end_date_from_prices(prices: pd.DataFrame) -> date:
    if prices.empty or "date" not in prices.columns:
        return DAILY_END_DATE
    frame = prices.loc[:, [col for col in ("ticker", "date") if col in prices.columns]].copy()
    if "ticker" not in frame.columns:
        frame["ticker"] = ""
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return DAILY_END_DATE
    frame["_market"] = frame["ticker"].map(_market_bucket)
    supported_dates: list[date] = []
    for _market, market_frame in frame.groupby("_market", sort=False):
        trading_dates = sorted(set(market_frame["date"]))
        latest_date = trading_dates[-1]
        cutoff = latest_date - timedelta(days=FORWARD_HORIZON_6M_DAYS)
        eligible_dates = [trading_date for trading_date in trading_dates if trading_date <= cutoff]
        if not eligible_dates:
            return DAILY_END_DATE
        supported_dates.append(eligible_dates[-1])
    return min(supported_dates) if supported_dates else DAILY_END_DATE


def _resolve_daily_end_date(end_date: Optional[date]) -> date:
    if end_date is not None:
        return end_date
    resolved = _latest_supported_daily_end_date_from_prices(_load_training_prices(require_ashare=True))
    logger.info("Resolved auto daily ML training end date -> %s", resolved.isoformat())
    return resolved


def train_ml_ranker(
    *,
    force_snapshots: bool = False,
    ground_truth_path: Path = CURRENT_GROUND_TRUTH_FILE,
    snapshots_path: Path = ML_SNAPSHOTS_FILE,
    output_model_path: Path = DEFAULT_MODEL_PATH,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    backend: str = "auto",
    snapshot_frequency: str = "daily",
    start_date: date = DAILY_START_DATE,
    end_date: Optional[date] = None,
    target_improvement: float = DEFAULT_TARGET_IMPROVEMENT,
    max_training_rows: Optional[int] = DEFAULT_MAX_TRAINING_ROWS,
    gate_config: Optional[HoldoutGateConfig] = None,
    walk_forward_folds: int = DEFAULT_WALK_FORWARD_FOLDS,
    walk_forward_validation_months: int = DEFAULT_WALK_FORWARD_VALIDATION_MONTHS,
    walk_forward_max_rows: int = DEFAULT_WALK_FORWARD_MAX_ROWS,
    walk_forward_min_6m_delta: float = DEFAULT_WALK_FORWARD_MIN_6M_DELTA,
    walk_forward_max_horizon_degradation: float = DEFAULT_WALK_FORWARD_MAX_HORIZON_DEGRADATION,
    model_kind: str = "ridge",
    target_horizon: str = "6m",
    full_eval_candidate_limit: Optional[int] = None,
    candidate_feature_sets: Optional[Sequence[str]] = None,
    candidate_prior_strategies: Optional[Sequence[str]] = None,
    candidate_targets: Optional[Sequence[str]] = None,
    candidate_ridge_lambdas: Optional[Sequence[str]] = None,
    promotion_min_train_rho: Optional[float] = None,
    candidate_anchor_model_paths: Optional[Sequence[Path]] = None,
    strict_outer_gate: bool = True,
) -> Dict[str, object]:
    """Train and persist the local ML ranker artifact."""
    if target_horizon not in EVAL_HORIZONS:
        raise ValueError(f"target_horizon must be one of: {', '.join(EVAL_HORIZONS)}")
    primary_horizon = target_horizon
    if strict_outer_gate and walk_forward_folds <= 0:
        raise ValueError("strict_outer_gate requires at least one walk-forward fold")
    resolved_end_date = (
        _resolve_daily_end_date(end_date) if snapshot_frequency == "daily" else (end_date or DAILY_END_DATE)
    )
    if primary_horizon == "1w" and output_model_path == DEFAULT_MODEL_PATH:
        output_model_path = ONE_WEEK_MODEL_PATH
    if full_eval_candidate_limit is None:
        if primary_horizon == "1w":
            full_eval_candidate_limit = DEFAULT_ONE_WEEK_FULL_EVAL_CANDIDATE_LIMIT
        elif primary_horizon == "6m":
            full_eval_candidate_limit = DEFAULT_SIX_MONTH_FULL_EVAL_CANDIDATE_LIMIT
        else:
            full_eval_candidate_limit = DEFAULT_FULL_EVAL_CANDIDATE_LIMIT
    full_eval_candidate_limit = max(1, int(full_eval_candidate_limit))
    feature_set_filter = _normalize_candidate_choices(
        candidate_feature_sets,
        aliases=FEATURE_SET_ALIASES,
        option_name="candidate_feature_sets",
    )
    prior_strategy_filter = _normalize_candidate_choices(
        candidate_prior_strategies,
        aliases=PRIOR_STRATEGY_ALIASES,
        option_name="candidate_prior_strategies",
    )
    candidate_target_filter_input = candidate_targets
    candidate_lambda_filter = _normalize_candidate_lambdas(candidate_ridge_lambdas)

    snapshots = prepare_ml_training_snapshots(
        force=force_snapshots,
        ground_truth_path=ground_truth_path,
        output_path=snapshots_path,
        snapshot_frequency=snapshot_frequency,
        start_date=start_date,
        end_date=resolved_end_date,
    )
    snapshots = _filter_application_training_universe(
        snapshots,
        min_rows_per_snapshot=APPLICATION_MIN_ROWS_PER_SNAPSHOT,
        min_snapshots=(APPLICATION_MIN_DAILY_SNAPSHOTS if snapshot_frequency == "daily" else 12),
    )
    gate_config = gate_config or HoldoutGateConfig()
    promotion_min_gate_rho = gate_config.min_primary_rho
    split = make_holdout_split(snapshots, config=gate_config, source_path=snapshots_path)
    train_snapshots = split.train
    raw_gate_snapshots = split.gate
    gate_snapshots = _monthly_rebalance_snapshots(raw_gate_snapshots)
    split.manifest.update(
        {
            "raw_gate_rows": int(len(raw_gate_snapshots)),
            "raw_gate_snapshots": int(raw_gate_snapshots["snapshot_date"].nunique()),
            "evaluation_frequency": "monthly",
            "gate_rows": int(len(gate_snapshots)),
            "gate_snapshots": int(gate_snapshots["snapshot_date"].nunique()),
        }
    )
    training_snapshots = _sample_training_snapshots(train_snapshots, max_training_rows)
    base_priors = _ticker_priors(train_snapshots)

    incumbent_payload: Optional[Mapping[str, object]] = None
    incumbent_eval_payload: Optional[Mapping[str, object]] = None
    incumbent_type = "hand_scorer"
    if output_model_path.exists():
        incumbent_payload = json.loads(output_model_path.read_text(encoding="utf-8"))
        if _payload_runtime_includes_evaluation_labels(incumbent_payload):
            raise RuntimeError(
                "Existing output model was refit using evaluation-period labels and "
                "cannot be a promotion-gate incumbent. Use a fresh output_model_path "
                f"instead of {output_model_path}."
            )
        incumbent_eval_payload = _payload_with_ticker_priors(incumbent_payload, base_priors)
        incumbent_type = "ml_ranker"
    anchor_payloads = _load_anchor_payloads(
        () if strict_outer_gate else candidate_anchor_model_paths,
        output_model_path=output_model_path,
        allow_output_model=primary_horizon == "6m",
    )
    if anchor_payloads:
        logger.info(
            "ML anchor model blending enabled: %s",
            ",".join(str(anchor["path"]) for anchor in anchor_payloads),
        )
        if primary_horizon == "6m":
            anchor_payloads = _score_anchor_payloads_on_gate(
                anchor_payloads,
                train_snapshots=train_snapshots,
                gate_snapshots=gate_snapshots,
                primary_horizon=primary_horizon,
            )

    folds = _walk_forward_folds(
        train_snapshots,
        n_folds=walk_forward_folds,
        validation_months=walk_forward_validation_months,
        embargo_days=gate_config.embargo_days,
    )
    if strict_outer_gate and not folds:
        raise RuntimeError(
            "strict_outer_gate could not construct purged walk-forward folds from "
            "the available application-universe snapshots"
        )
    if folds:
        logger.info(
            "ML walk-forward filter: %d folds, %d validation months/fold",
            len(folds),
            walk_forward_validation_months,
        )

    train_baseline_metrics = _baseline_metrics(train_snapshots)
    selection_baseline_metrics = _baseline_metrics(training_snapshots)
    train_baseline_rho = float(train_baseline_metrics[primary_horizon]["spearman_rho"])
    if primary_horizon == "6m":
        candidate_targets = (
            "target_rank_6m_soft",
            "target_rank_6m",
            "target_rank_6m_market",
            "target_rank_6m_extreme",
            "target_rank_weighted_901",
            "target_rank_weighted_901_market",
            "target_rank_weighted_8515",
            "target_rank_weighted_8515_market",
            "target_rank_weighted_802",
            "target_rank_weighted_802_market",
            "target_rank_weighted_7525",
            "target_rank_weighted_703",
            "target_rank_weighted_703_market",
            "target_rank_weighted_631",
            "target_rank_weighted",
            "target_rank_mean",
        )
    elif primary_horizon == "1w":
        candidate_targets = ("target_rank_1w", "target_rank_1w_market")
    else:
        candidate_targets = (f"target_rank_{primary_horizon}",)
    target_filter = _normalize_candidate_targets(
        candidate_target_filter_input,
        allowed_targets=candidate_targets,
    )
    if target_filter is not None:
        candidate_targets = target_filter
    if primary_horizon == "1w":
        candidate_lambdas = tuple(dict.fromkeys((ridge_lambda, *ONE_WEEK_CANDIDATE_LAMBDAS)))
        candidate_feature_sets = (
            ("core", expected_feature_names()),
            ("short_horizon", expected_feature_names(include_short_horizon=True)),
        )
    elif primary_horizon == "6m":
        candidate_lambdas = tuple(dict.fromkeys((ridge_lambda, *SIX_MONTH_CANDIDATE_LAMBDAS)))
        candidate_feature_sets = (
            ("core", expected_feature_names()),
            ("short_horizon", expected_feature_names(include_short_horizon=True)),
            (
                "expanded",
                expected_feature_names(
                    include_short_horizon=True,
                    include_interactions=True,
                ),
            ),
            (
                "poly",
                expected_feature_names(
                    include_short_horizon=True,
                    include_interactions=True,
                    include_polynomial=True,
                ),
            ),
            (
                "cross_sectional",
                expected_feature_names(
                    include_short_horizon=True,
                    include_interactions=True,
                    include_polynomial=True,
                    include_cross_sectional=True,
                ),
            ),
            (
                "cross_sectional_interactions",
                expected_feature_names(
                    include_short_horizon=True,
                    include_interactions=True,
                    include_polynomial=True,
                    include_cross_sectional=True,
                    include_cross_sectional_interactions=True,
                ),
            ),
        )
    else:
        candidate_lambdas = tuple(dict.fromkeys((ridge_lambda, 10.0, 30.0, 100.0, 300.0, 1_000.0)))
        candidate_feature_sets = (("core", expected_feature_names()),)
    if candidate_lambda_filter is not None:
        candidate_lambdas = candidate_lambda_filter
    if feature_set_filter is not None:
        candidate_feature_sets = tuple(
            candidate for candidate in candidate_feature_sets if candidate[0] in feature_set_filter
        )
        if not candidate_feature_sets:
            raise ValueError(
                f"candidate_feature_sets produced no valid candidates for target_horizon={primary_horizon}"
            )
    six_month_recent_model_kinds = tuple(
        f"market_ridge_recent_{half_life:g}" for half_life in _six_month_recency_half_lives_days()
    )
    six_month_recent_segment_model_kinds = tuple(
        f"segment_ridge_recent_{half_life:g}" for half_life in _six_month_recency_half_lives_days()
    )
    if model_kind == "auto":
        if primary_horizon == "6m":
            candidate_model_kinds = (
                "ridge",
                "market_ridge",
                "segment_ridge",
                *six_month_recent_model_kinds,
                *six_month_recent_segment_model_kinds,
                "pairwise",
            )
        elif primary_horizon == "1w":
            candidate_model_kinds = ("ridge", "market_ridge", "market_ridge_recent", "pairwise")
        else:
            candidate_model_kinds = ("ridge", "pairwise")
    elif model_kind in {"market-ridge", "market_ridge"}:
        if primary_horizon == "6m":
            candidate_model_kinds = ("market_ridge", *six_month_recent_model_kinds)
        elif primary_horizon == "1w":
            candidate_model_kinds = ("market_ridge", "market_ridge_recent")
        else:
            candidate_model_kinds = ("market_ridge",)
    elif model_kind in {"market-ridge-only", "market_ridge_only"}:
        candidate_model_kinds = ("market_ridge",)
    elif model_kind in {"market-ridge-recent", "market_ridge_recent"}:
        candidate_model_kinds = six_month_recent_model_kinds if primary_horizon == "6m" else ("market_ridge_recent",)
    elif model_kind in {"segment-ridge", "segment_ridge"}:
        if primary_horizon == "6m":
            candidate_model_kinds = ("segment_ridge", *six_month_recent_segment_model_kinds)
        else:
            candidate_model_kinds = ("segment_ridge",)
    elif model_kind in {"segment-ridge-only", "segment_ridge_only"}:
        candidate_model_kinds = ("segment_ridge",)
    elif model_kind in {"segment-ridge-recent", "segment_ridge_recent"}:
        candidate_model_kinds = (
            six_month_recent_segment_model_kinds if primary_horizon == "6m" else ("segment_ridge_recent",)
        )
    elif model_kind in {"ridge-only", "ridge_only"}:
        candidate_model_kinds = ("ridge",)
    elif model_kind in {"ridge", "pairwise"}:
        if model_kind == "ridge" and primary_horizon == "6m":
            candidate_model_kinds = (
                "ridge",
                "market_ridge",
                "segment_ridge",
                *six_month_recent_model_kinds,
                *six_month_recent_segment_model_kinds,
            )
        elif model_kind == "ridge" and primary_horizon == "1w":
            candidate_model_kinds = ("ridge", "market_ridge", "market_ridge_recent")
        else:
            candidate_model_kinds = (model_kind,)
    else:
        normalized_model_kind = model_kind.replace("-", "_")
        if _recent_half_life_for_model_kind(normalized_model_kind):
            candidate_model_kinds = (normalized_model_kind,)
        else:
            raise ValueError(
                "model_kind must be one of: ridge, market-ridge, market-ridge-recent, "
                "market-ridge-recent-<days>, market-ridge-only, segment-ridge, "
                "segment-ridge-recent, segment-ridge-recent-<days>, "
                "segment-ridge-only, ridge-only, pairwise, auto"
            )
    ridge_candidate_backends = _ridge_candidate_backends(backend, primary_horizon)
    prior_options: tuple[tuple[str, Mapping[str, Mapping[str, float]]], ...] = (
        (
            ("no_ticker_priors", {}),
            ("ticker_priors", base_priors),
        )
        if primary_horizon == "1w"
        else (
            ("rolling_ticker_priors", base_priors),
            ("no_ticker_priors", {}),
            *((("ticker_priors", base_priors),) if not strict_outer_gate else ()),
        )
    )
    if prior_strategy_filter is not None:
        prior_options = tuple(candidate for candidate in prior_options if candidate[0] in prior_strategy_filter)
        if not prior_options:
            raise ValueError(
                f"candidate_prior_strategies produced no valid candidates for target_horizon={primary_horizon}"
            )
    logger.info(
        "ML candidate search target=%s targets=%s models=%s features=%s priors=%s lambdas=%s full_eval_limit=%d",
        primary_horizon,
        ",".join(candidate_targets),
        ",".join(candidate_model_kinds),
        ",".join(feature_set for feature_set, _ in candidate_feature_sets),
        ",".join(prior_strategy for prior_strategy, _ in prior_options),
        ",".join(f"{candidate_lambda:g}" for candidate_lambda in candidate_lambdas),
        full_eval_candidate_limit,
    )
    candidate_training_variants = [] if strict_outer_gate else [(0, training_snapshots)]
    if (
        not strict_outer_gate
        and primary_horizon == "6m"
        and max_training_rows is not None
        and max_training_rows > 0
        and len(train_snapshots) > len(training_snapshots)
    ):
        candidate_training_variants = [
            (
                seed,
                _sample_training_snapshots(
                    train_snapshots,
                    max_training_rows,
                    random_seed=seed,
                ),
            )
            for seed in _six_month_sample_seeds()
        ]
    sample_candidates: list[dict] = []
    best_sample: Optional[dict] = None
    best_full: Optional[dict] = None
    train_snapshots_with_rolling_priors: Optional[pd.DataFrame] = None
    gate_snapshots_with_rolling_priors: Optional[pd.DataFrame] = None
    gate_probe_snapshots: Optional[pd.DataFrame] = None
    gate_probe_snapshots_with_rolling_priors: Optional[pd.DataFrame] = None
    sample_rolling_priors_cache: dict[int, pd.DataFrame] = {}
    sample_primary_context_cache: dict[int, dict[str, object]] = {}

    def sample_primary_metrics(
        frame: pd.DataFrame,
        predictions: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        cache_key = id(frame)
        context = sample_primary_context_cache.get(cache_key)
        if context is None:
            context = _prepare_primary_spearman_context(frame, primary_horizon)
            sample_primary_context_cache[cache_key] = context
        rho = _evaluate_prepared_primary_spearman_rho(context, predictions)
        return {primary_horizon: {"spearman_rho": rho}}

    def train_with_rolling_priors() -> pd.DataFrame:
        nonlocal train_snapshots_with_rolling_priors
        if train_snapshots_with_rolling_priors is None:
            train_snapshots_with_rolling_priors = _attach_asof_ticker_priors(
                train_snapshots,
                train_snapshots,
            )
        return train_snapshots_with_rolling_priors

    def gate_with_rolling_priors() -> pd.DataFrame:
        nonlocal gate_snapshots_with_rolling_priors
        if gate_snapshots_with_rolling_priors is None:
            gate_snapshots_with_rolling_priors = _attach_asof_ticker_priors(
                gate_snapshots,
                train_snapshots,
            )
        return gate_snapshots_with_rolling_priors

    def prediction_frame_for_payload(
        source: pd.DataFrame,
        payload: Mapping[str, object],
    ) -> pd.DataFrame:
        required_columns = {
            "ticker",
            "snapshot_date",
            *BASE_FIELDS,
            *SHORT_HORIZON_FEATURES,
        }
        uses_row_priors = _payload_uses_row_priors(payload)
        columns = [
            column
            for column in source.columns
            if column in required_columns or (uses_row_priors and str(column).startswith(PRIOR_COLUMN_PREFIX))
        ]
        return source.loc[:, columns]

    def frame_for_payload(
        payload: Mapping[str, object],
        *,
        train_frame: bool,
    ) -> pd.DataFrame:
        if strict_outer_gate and not train_frame:
            source = gate_snapshots
        elif _payload_uses_row_priors(payload):
            source = train_with_rolling_priors() if train_frame else gate_with_rolling_priors()
        else:
            source = train_snapshots if train_frame else gate_snapshots
        return prediction_frame_for_payload(source, payload)

    def refit_full_eval_candidate(candidate: dict) -> dict:
        if candidate.get("target_source") == "anchor_teacher":
            return candidate
        sample_rows = int(candidate.get("sample_rows", 0))
        refit_max_rows = _six_month_full_eval_refit_max_rows()
        refit_rows = len(train_snapshots)
        if refit_max_rows > 0:
            refit_rows = min(len(train_snapshots), max(refit_max_rows, sample_rows))
        if primary_horizon != "6m" or not SIX_MONTH_REFIT_FULL_EVAL_CANDIDATES or sample_rows >= refit_rows:
            return candidate

        model, priors, used_backend, prefer_row_priors = _fit_candidate_model(
            train_snapshots,
            target_name=candidate["target"],
            ridge_lambda=float(candidate["ridge_lambda"]),
            backend=str(candidate["backend"]),
            max_rows=None if refit_rows >= len(train_snapshots) else refit_rows,
            prior_strategy=str(candidate["prior_strategy"]),
            model_kind=str(candidate["model_kind"]),
            feature_names=candidate["feature_names"],
            random_seed=int(candidate.get("sample_seed", 0)),
        )
        refit = dict(candidate)
        refit.update(
            {
                "backend": used_backend,
                "priors": priors,
                "prefer_row_priors": prefer_row_priors,
                "full_fit_rows": int(refit_rows),
                "full_fit_refit": True,
            }
        )
        if isinstance(model.get("models"), list):
            refit["models"] = model["models"]
        else:
            refit["models"] = None
            refit.update(
                {
                    "clip_low": np.asarray(model["clip_low"], dtype="float64"),
                    "clip_high": np.asarray(model["clip_high"], dtype="float64"),
                    "mean": np.asarray(model["mean"], dtype="float64"),
                    "scale": np.asarray(model["scale"], dtype="float64"),
                    "coef": np.asarray(model["coef"], dtype="float64"),
                    "intercept": float(model["intercept"]),
                }
            )
        logger.info(
            "ML full-eval refit target=%s lambda=%.4g backend=%s model=%s features=%s priors=%s rows=%d",
            candidate["target"],
            float(candidate["ridge_lambda"]),
            used_backend,
            candidate["model_kind"],
            candidate["feature_set"],
            candidate["prior_strategy"],
            refit_rows,
        )
        return refit

    def evaluate_full_candidate(candidate: dict) -> dict:
        nonlocal best_full
        candidate = refit_full_eval_candidate(candidate)
        full_frame = train_with_rolling_priors() if candidate["prefer_row_priors"] else train_snapshots
        if candidate.get("models") is not None:
            full_predictions = _predict_from_linear_model_payloads(
                full_frame,
                candidate["priors"],
                candidate["models"],
                prefer_row_priors=candidate["prefer_row_priors"],
                feature_names=candidate["feature_names"],
            )
        else:
            full_predictions = _predict_in_chunks(
                full_frame,
                ticker_priors=candidate["priors"],
                clip_low=candidate["clip_low"],
                clip_high=candidate["clip_high"],
                mean=candidate["mean"],
                scale=candidate["scale"],
                coef=candidate["coef"],
                intercept=float(candidate["intercept"]),
                prefer_row_priors=candidate["prefer_row_priors"],
                feature_names=candidate["feature_names"],
            )
        full_metrics = _evaluate_predictions(train_snapshots, full_predictions)
        full_rho = float(full_metrics[primary_horizon]["spearman_rho"])
        full_improvement = _improvement_ratio(full_rho, train_baseline_rho)
        full_candidate = dict(candidate)
        full_candidate["metrics"] = full_metrics
        full_candidate["sample_metrics"] = candidate["metrics"]
        full_candidate["sample_improvement"] = candidate["improvement"]
        full_candidate["improvement"] = full_improvement
        full_candidate["_full_predictions"] = full_predictions.astype("float32", copy=False)
        if best_full is None or full_rho > float(best_full["metrics"][primary_horizon]["spearman_rho"]):
            best_full = full_candidate
        logger.info(
            "ML full-eval target=%s lambda=%.4g backend=%s model=%s features=%s priors=%s "
            "%s_rho=%.6f improvement=%.2f%%",
            candidate["target"],
            candidate["ridge_lambda"],
            candidate["backend"],
            candidate["model_kind"],
            candidate["feature_set"],
            candidate["prior_strategy"],
            primary_horizon,
            full_rho,
            full_improvement * 100.0,
        )
        return full_candidate

    def gate_probe_base_frame() -> pd.DataFrame:
        nonlocal gate_probe_snapshots
        if gate_probe_snapshots is None:
            max_rows = _six_month_gate_probe_max_rows()
            if max_rows <= 0:
                gate_probe_snapshots = gate_snapshots.iloc[0:0].copy()
            else:
                gate_probe_snapshots = _sample_training_snapshots(
                    gate_snapshots,
                    max_rows,
                    random_seed=19,
                )
        return gate_probe_snapshots

    def gate_probe_frame_for_candidate(candidate: Mapping[str, object]) -> pd.DataFrame:
        nonlocal gate_probe_snapshots_with_rolling_priors
        base_frame = gate_probe_base_frame()
        if candidate.get("prefer_row_priors"):
            if gate_probe_snapshots_with_rolling_priors is None:
                gate_probe_snapshots_with_rolling_priors = _attach_asof_ticker_priors(
                    base_frame,
                    train_snapshots,
                )
            return gate_probe_snapshots_with_rolling_priors
        return base_frame

    def score_gate_probe_candidates(candidates: list[dict]) -> None:
        if primary_horizon != "6m" or not candidates:
            return
        candidate_limit = _six_month_gate_probe_candidate_limit()
        if candidate_limit <= 0:
            return
        probe_frame = gate_probe_base_frame()
        if probe_frame.empty:
            return
        scored = 0
        for candidate in candidates[:candidate_limit]:
            candidate_frame = gate_probe_frame_for_candidate(candidate)
            if candidate.get("models") is not None:
                gate_predictions = _predict_from_linear_model_payloads(
                    candidate_frame,
                    candidate["priors"],
                    candidate["models"],
                    prefer_row_priors=bool(candidate["prefer_row_priors"]),
                    feature_names=candidate["feature_names"],
                )
            else:
                gate_predictions = _predict_in_chunks(
                    candidate_frame,
                    ticker_priors=candidate["priors"],
                    clip_low=candidate["clip_low"],
                    clip_high=candidate["clip_high"],
                    mean=candidate["mean"],
                    scale=candidate["scale"],
                    coef=candidate["coef"],
                    intercept=float(candidate["intercept"]),
                    prefer_row_priors=bool(candidate["prefer_row_priors"]),
                    feature_names=candidate["feature_names"],
                )
            gate_rho = _evaluate_primary_spearman_rho(
                probe_frame,
                gate_predictions,
                primary_horizon,
            )
            candidate["gate_probe"] = {
                "metrics": {primary_horizon: {"spearman_rho": float(gate_rho)}},
                "rows": int(len(probe_frame)),
            }
            scored += 1
            logger.info(
                "ML sample gate-probe target=%s lambda=%.4g backend=%s model=%s "
                "features=%s priors=%s rows=%d seed=%d sample_%s_rho=%.6f "
                "gate_probe_%s_rho=%.6f",
                candidate.get("target"),
                float(candidate.get("ridge_lambda", 0.0)),
                candidate.get("backend"),
                candidate.get("model_kind"),
                candidate.get("feature_set"),
                candidate.get("prior_strategy"),
                len(probe_frame),
                int(candidate.get("sample_seed", 0)),
                primary_horizon,
                _candidate_primary_rho(candidate, primary_horizon),
                primary_horizon,
                float(gate_rho),
            )
        logger.info(
            "ML sample gate-probe target=%s scored=%d candidates rows=%d candidate_limit=%d",
            primary_horizon,
            scored,
            len(probe_frame),
            candidate_limit,
        )

    def add_anchor_teacher_candidates(
        *,
        sample_seed: int,
        sample_training_snapshots: pd.DataFrame,
    ) -> None:
        """Train candidates distilled toward a strong historical anchor."""
        nonlocal best_sample
        if primary_horizon != "6m" or not anchor_payloads:
            return
        teacher_base_targets = tuple(
            target for target in candidate_targets if target in SIX_MONTH_ANCHOR_TEACHER_BASE_TARGETS
        )
        if not teacher_base_targets:
            return
        teacher_model_kinds = tuple(
            model
            for model in candidate_model_kinds
            if _fit_model_kind_for_candidate(model) in {"ridge", "market_ridge", "segment_ridge"}
            and not _is_recent_market_ridge(model)
            and not _is_recent_segment_ridge(model)
        )
        if not teacher_model_kinds:
            return

        sample_baseline_metrics = _baseline_metrics(sample_training_snapshots)
        sample_baseline_rho = float(sample_baseline_metrics[primary_horizon]["spearman_rho"])
        for anchor in anchor_payloads[:SIX_MONTH_ANCHOR_TEACHER_ANCHOR_LIMIT]:
            anchor_path = str(anchor["path"])
            anchor_payload = anchor["payload"]
            if not isinstance(anchor_payload, Mapping):
                continue
            anchor_frame = (
                _attach_asof_ticker_priors(sample_training_snapshots, train_snapshots)
                if _payload_uses_row_priors(anchor_payload)
                else sample_training_snapshots
            )
            anchor_predictions = predict_ml_ranker_payload(anchor_frame, anchor_payload)
            anchor_rank = _prediction_rank_by_snapshot(
                sample_training_snapshots,
                anchor_predictions,
            )
            for prior_strategy, priors in prior_options:
                prefer_row_priors = prior_strategy == "rolling_ticker_priors"
                if prefer_row_priors:
                    sample_cache_key = id(sample_training_snapshots)
                    candidate_training_snapshots = sample_rolling_priors_cache.get(sample_cache_key)
                    if candidate_training_snapshots is None:
                        candidate_training_snapshots = _attach_asof_ticker_priors(
                            sample_training_snapshots,
                            train_snapshots,
                        )
                        sample_rolling_priors_cache[sample_cache_key] = candidate_training_snapshots
                else:
                    candidate_training_snapshots = sample_training_snapshots
                for feature_set, feature_names in candidate_feature_sets:
                    X = build_feature_matrix(
                        candidate_training_snapshots,
                        priors,
                        prefer_row_priors=prefer_row_priors,
                        feature_names=feature_names,
                    )
                    X_scaled, clip_low, clip_high, mean, scale = _standardize_features(X)
                    for base_target in teacher_base_targets:
                        base_values = _target_values(
                            candidate_training_snapshots,
                            base_target,
                        )
                        best_before_teacher = (
                            float(best_sample["metrics"][primary_horizon]["spearman_rho"])
                            if best_sample is not None
                            else float("-inf")
                        )
                        for truth_weight in SIX_MONTH_ANCHOR_TEACHER_WEIGHTS:
                            best_weight_rho = float("-inf")
                            target = float(truth_weight) * base_values + (1.0 - float(truth_weight)) * anchor_rank
                            valid = np.isfinite(target)
                            if valid.sum() < 10:
                                continue
                            target_name = f"anchor_teacher_{base_target}_w{truth_weight:g}"
                            for candidate_lambda in candidate_lambdas:
                                for candidate_model_kind in teacher_model_kinds:
                                    fit_candidate_model_kind = _fit_model_kind_for_candidate(candidate_model_kind)
                                    model_backends = (
                                        ridge_candidate_backends
                                        if fit_candidate_model_kind in {"ridge", "market_ridge", "segment_ridge"}
                                        else (backend,)
                                    )
                                    for candidate_backend in model_backends:
                                        if fit_candidate_model_kind == "ridge":
                                            coef_with_intercept, used_backend = _fit_ridge(
                                                X_scaled[valid],
                                                target[valid],
                                                ridge_lambda=candidate_lambda,
                                                backend=candidate_backend,
                                            )
                                            candidate_models = None
                                        elif fit_candidate_model_kind == "market_ridge":
                                            candidate_models, used_backend = _fit_market_ridge_models(
                                                X_scaled,
                                                target,
                                                candidate_training_snapshots["ticker"].astype(str),
                                                ridge_lambda=candidate_lambda,
                                                backend=candidate_backend,
                                                clip_low=clip_low,
                                                clip_high=clip_high,
                                                mean=mean,
                                                scale=scale,
                                            )
                                            coef_with_intercept = None
                                        elif fit_candidate_model_kind == "segment_ridge":
                                            candidate_models, used_backend = _fit_segment_ridge_models(
                                                X_scaled,
                                                target,
                                                candidate_training_snapshots,
                                                ridge_lambda=candidate_lambda,
                                                backend=candidate_backend,
                                                clip_low=clip_low,
                                                clip_high=clip_high,
                                                mean=mean,
                                                scale=scale,
                                            )
                                            coef_with_intercept = None
                                        else:
                                            continue
                                        if candidate_models is not None:
                                            predictions = _predict_scaled_linear_models(
                                                X_scaled,
                                                candidate_training_snapshots,
                                                candidate_models,
                                            )
                                            coef = np.asarray([], dtype="float64")
                                            intercept = 0.0
                                        else:
                                            if coef_with_intercept is None:
                                                raise RuntimeError("missing linear coefficients")
                                            coef = coef_with_intercept[:-1]
                                            intercept = float(coef_with_intercept[-1])
                                            predictions = X_scaled @ coef + intercept
                                        metrics = sample_primary_metrics(
                                            sample_training_snapshots,
                                            predictions,
                                        )
                                        final_rho = float(metrics[primary_horizon]["spearman_rho"])
                                        best_weight_rho = max(
                                            best_weight_rho,
                                            final_rho,
                                        )
                                        improvement = _improvement_ratio(
                                            final_rho,
                                            sample_baseline_rho,
                                        )
                                        candidate = {
                                            "target": target_name,
                                            "target_source": "anchor_teacher",
                                            "teacher_base_target": base_target,
                                            "teacher_truth_weight": float(truth_weight),
                                            "teacher_anchor_model_path": anchor_path,
                                            "ridge_lambda": float(candidate_lambda),
                                            "backend": used_backend,
                                            "model_kind": candidate_model_kind,
                                            "feature_set": feature_set,
                                            "feature_names": list(feature_names),
                                            "prior_strategy": prior_strategy,
                                            "prefer_row_priors": prefer_row_priors,
                                            "priors": priors,
                                            "clip_low": clip_low,
                                            "clip_high": clip_high,
                                            "mean": mean,
                                            "scale": scale,
                                            "coef": coef,
                                            "intercept": intercept,
                                            "models": candidate_models,
                                            "metrics": metrics,
                                            "improvement": improvement,
                                            "sample_rows": int(len(sample_training_snapshots)),
                                            "sample_seed": int(sample_seed),
                                        }
                                        sample_candidates.append(candidate)
                                        if best_sample is None or final_rho > float(
                                            best_sample["metrics"][primary_horizon]["spearman_rho"]
                                        ):
                                            best_sample = candidate
                                        logger.info(
                                            "ML sample anchor-teacher target=%s base=%s "
                                            "truth_weight=%.2f anchor=%s lambda=%.4g "
                                            "backend=%s model=%s features=%s priors=%s "
                                            "rows=%d seed=%d %s_rho=%.6f improvement=%.2f%%",
                                            target_name,
                                            base_target,
                                            float(truth_weight),
                                            anchor_path,
                                            candidate_lambda,
                                            used_backend,
                                            candidate_model_kind,
                                            feature_set,
                                            prior_strategy,
                                            len(sample_training_snapshots),
                                            sample_seed,
                                            primary_horizon,
                                            final_rho,
                                            improvement * 100.0,
                                        )
                            if (
                                best_before_teacher > float("-inf")
                                and best_weight_rho + SIX_MONTH_ANCHOR_TEACHER_PRUNE_GAP < best_before_teacher
                            ):
                                logger.info(
                                    "ML sample anchor-teacher pruning base=%s "
                                    "features=%s priors=%s after truth_weight=%.2f "
                                    "best_teacher_%s_rho=%.6f best_sample_%s_rho=%.6f",
                                    base_target,
                                    feature_set,
                                    prior_strategy,
                                    float(truth_weight),
                                    primary_horizon,
                                    best_weight_rho,
                                    primary_horizon,
                                    best_before_teacher,
                                )
                                break

    for sample_seed, sample_training_snapshots in candidate_training_variants:
        sample_baseline_metrics = _baseline_metrics(sample_training_snapshots)
        sample_baseline_rho = float(sample_baseline_metrics[primary_horizon]["spearman_rho"])
        for prior_strategy, priors in prior_options:
            prefer_row_priors = prior_strategy == "rolling_ticker_priors"
            if prefer_row_priors:
                sample_cache_key = id(sample_training_snapshots)
                candidate_training_snapshots = sample_rolling_priors_cache.get(sample_cache_key)
                if candidate_training_snapshots is None:
                    candidate_training_snapshots = _attach_asof_ticker_priors(
                        sample_training_snapshots,
                        train_snapshots,
                    )
                    sample_rolling_priors_cache[sample_cache_key] = candidate_training_snapshots
            else:
                candidate_training_snapshots = sample_training_snapshots
            for feature_set, feature_names in candidate_feature_sets:
                X = build_feature_matrix(
                    candidate_training_snapshots,
                    priors,
                    prefer_row_priors=prefer_row_priors,
                    feature_names=feature_names,
                )
                X_scaled, clip_low, clip_high, mean, scale = _standardize_features(X)
                for target_name in candidate_targets:
                    target = _target_values(candidate_training_snapshots, target_name)
                    valid = np.isfinite(target)
                    if valid.sum() < 10:
                        continue
                    for candidate_lambda in candidate_lambdas:
                        for candidate_model_kind in candidate_model_kinds:
                            fit_candidate_model_kind = _fit_model_kind_for_candidate(candidate_model_kind)
                            recency_half_life = _recent_half_life_for_model_kind(candidate_model_kind)
                            sample_weight = (
                                _recency_sample_weights(
                                    candidate_training_snapshots,
                                    half_life_days=recency_half_life,
                                )
                                if recency_half_life is not None
                                else None
                            )
                            model_backends = (
                                ridge_candidate_backends
                                if fit_candidate_model_kind in {"ridge", "market_ridge", "segment_ridge"}
                                else (backend,)
                            )
                            for candidate_backend in model_backends:
                                if fit_candidate_model_kind == "ridge":
                                    try:
                                        coef_with_intercept, used_backend = _fit_ridge(
                                            X_scaled[valid],
                                            target[valid],
                                            ridge_lambda=candidate_lambda,
                                            backend=candidate_backend,
                                            sample_weight=(sample_weight[valid] if sample_weight is not None else None),
                                        )
                                    except Exception as exc:
                                        if backend == "auto" and candidate_backend == "mlx-adam":
                                            logger.info(
                                                "MLX Adam backend unavailable, skipping: %s",
                                                exc,
                                            )
                                            continue
                                        raise
                                    candidate_models = None
                                elif fit_candidate_model_kind == "market_ridge":
                                    try:
                                        candidate_models, used_backend = _fit_market_ridge_models(
                                            X_scaled,
                                            target,
                                            candidate_training_snapshots["ticker"].astype(str),
                                            ridge_lambda=candidate_lambda,
                                            backend=candidate_backend,
                                            clip_low=clip_low,
                                            clip_high=clip_high,
                                            mean=mean,
                                            scale=scale,
                                            sample_weight=sample_weight,
                                        )
                                    except Exception as exc:
                                        if backend == "auto" and candidate_backend == "mlx-adam":
                                            logger.info(
                                                "MLX Adam market backend unavailable, skipping: %s",
                                                exc,
                                            )
                                            continue
                                        raise
                                    coef_with_intercept = None
                                elif fit_candidate_model_kind == "segment_ridge":
                                    try:
                                        candidate_models, used_backend = _fit_segment_ridge_models(
                                            X_scaled,
                                            target,
                                            candidate_training_snapshots,
                                            ridge_lambda=candidate_lambda,
                                            backend=candidate_backend,
                                            clip_low=clip_low,
                                            clip_high=clip_high,
                                            mean=mean,
                                            scale=scale,
                                            sample_weight=sample_weight,
                                        )
                                    except Exception as exc:
                                        if backend == "auto" and candidate_backend == "mlx-adam":
                                            logger.info(
                                                "MLX Adam segment backend unavailable, skipping: %s",
                                                exc,
                                            )
                                            continue
                                        raise
                                    coef_with_intercept = None
                                else:
                                    coef_with_intercept, used_backend = _fit_pairwise_ranker_numpy(
                                        X_scaled,
                                        target,
                                        _snapshot_dates(candidate_training_snapshots),
                                        ridge_lambda=candidate_lambda,
                                    )
                                    candidate_models = None
                                if candidate_models is not None:
                                    predictions = _predict_scaled_linear_models(
                                        X_scaled,
                                        candidate_training_snapshots,
                                        candidate_models,
                                    )
                                    coef = np.asarray([], dtype="float64")
                                    intercept = 0.0
                                else:
                                    if coef_with_intercept is None:
                                        raise RuntimeError("missing linear coefficients")
                                    coef = coef_with_intercept[:-1]
                                    intercept = float(coef_with_intercept[-1])
                                    predictions = X_scaled @ coef + intercept
                                metrics = sample_primary_metrics(
                                    sample_training_snapshots,
                                    predictions,
                                )
                                final_rho = float(metrics[primary_horizon]["spearman_rho"])
                                improvement = _improvement_ratio(final_rho, sample_baseline_rho)
                                candidate = {
                                    "target": target_name,
                                    "ridge_lambda": float(candidate_lambda),
                                    "backend": used_backend,
                                    "model_kind": candidate_model_kind,
                                    "feature_set": feature_set,
                                    "feature_names": list(feature_names),
                                    "prior_strategy": prior_strategy,
                                    "prefer_row_priors": prefer_row_priors,
                                    "priors": priors,
                                    "clip_low": clip_low,
                                    "clip_high": clip_high,
                                    "mean": mean,
                                    "scale": scale,
                                    "coef": coef,
                                    "intercept": intercept,
                                    "models": candidate_models,
                                    "metrics": metrics,
                                    "improvement": improvement,
                                    "sample_rows": int(len(sample_training_snapshots)),
                                    "sample_seed": int(sample_seed),
                                }
                                sample_candidates.append(candidate)
                                if best_sample is None or final_rho > float(
                                    best_sample["metrics"][primary_horizon]["spearman_rho"]
                                ):
                                    best_sample = candidate
                                logger.info(
                                    "ML sample candidate target=%s lambda=%.4g backend=%s "
                                    "model=%s features=%s priors=%s rows=%d seed=%d "
                                    "%s_rho=%.6f improvement=%.2f%%",
                                    target_name,
                                    candidate_lambda,
                                    used_backend,
                                    candidate_model_kind,
                                    feature_set,
                                    prior_strategy,
                                    len(sample_training_snapshots),
                                    sample_seed,
                                    primary_horizon,
                                    final_rho,
                                    improvement * 100.0,
                                )

        add_anchor_teacher_candidates(
            sample_seed=sample_seed,
            sample_training_snapshots=sample_training_snapshots,
        )

    if strict_outer_gate:
        candidate_specs: list[dict] = []
        for prior_strategy, priors in prior_options:
            prefer_row_priors = prior_strategy == "rolling_ticker_priors"
            for feature_set, feature_names in candidate_feature_sets:
                for target_name in candidate_targets:
                    for candidate_lambda in candidate_lambdas:
                        for candidate_model_kind in candidate_model_kinds:
                            fit_model_kind = _fit_model_kind_for_candidate(candidate_model_kind)
                            candidate_backend = (
                                ridge_candidate_backends[0]
                                if fit_model_kind in {"ridge", "market_ridge", "segment_ridge"}
                                else backend
                            )
                            candidate_specs.append(
                                {
                                    "target": target_name,
                                    "ridge_lambda": float(candidate_lambda),
                                    "backend": candidate_backend,
                                    "model_kind": candidate_model_kind,
                                    "feature_set": feature_set,
                                    "feature_names": list(feature_names),
                                    "prior_strategy": prior_strategy,
                                    "prefer_row_priors": prefer_row_priors,
                                    "priors": priors,
                                    "metrics": {
                                        primary_horizon: {"spearman_rho": 0.0},
                                    },
                                    "improvement": float("inf"),
                                    "sample_rows": 0,
                                    "sample_seed": 0,
                                }
                            )
        sample_candidates = _select_balanced_walk_forward_specs(candidate_specs)
        best_sample = sample_candidates[0] if sample_candidates else None
        logger.info(
            "ML strict walk-forward panel target=%s selected=%d configuration_space=%d",
            primary_horizon,
            len(sample_candidates),
            len(candidate_specs),
        )

    sample_candidates.sort(
        key=lambda candidate: float(candidate["metrics"][primary_horizon]["spearman_rho"]),
        reverse=True,
    )
    floor_screened_sample_candidates = sample_candidates
    if promotion_min_train_rho is not None and not strict_outer_gate:
        sample_train_floor_tolerance = _sample_train_floor_tolerance(primary_horizon)
        floor_screened_sample_candidates = [
            candidate
            for candidate in sample_candidates
            if candidate.get("target_source") == "anchor_teacher"
            or _candidate_primary_rho(candidate, primary_horizon) + sample_train_floor_tolerance
            >= promotion_min_train_rho
        ]
        skipped_sample_candidates = len(sample_candidates) - len(floor_screened_sample_candidates)
        if skipped_sample_candidates:
            best_sample_rho = _candidate_primary_rho(sample_candidates[0], primary_horizon)
            logger.info(
                "ML sample train-floor filter target=%s kept=%d skipped=%d "
                "sample_candidates=%d best_%s_rho=%.6f min_train_%s_rho=%.6f "
                "tolerance=%.6f",
                primary_horizon,
                len(floor_screened_sample_candidates),
                skipped_sample_candidates,
                len(sample_candidates),
                primary_horizon,
                float(best_sample_rho),
                primary_horizon,
                float(promotion_min_train_rho),
                float(sample_train_floor_tolerance),
            )
        if not floor_screened_sample_candidates:
            best_sample_rho = (
                _candidate_primary_rho(sample_candidates[0], primary_horizon) if sample_candidates else float("-inf")
            )
            raise RuntimeError(
                "ML ranker skipped full eval because sample rho did not meet "
                f"requested train floor: best {primary_horizon} "
                f"rho={best_sample_rho:.6f}, required {primary_horizon} "
                f"rho={float(promotion_min_train_rho):.6f}, "
                f"tolerance={sample_train_floor_tolerance:.6f}"
            )
    if not strict_outer_gate:
        score_gate_probe_candidates(floor_screened_sample_candidates)
    walk_forward_limit = _walk_forward_candidate_limit(
        full_eval_candidate_limit,
        primary_horizon=primary_horizon,
    )
    if strict_outer_gate:
        walk_forward_candidates = list(floor_screened_sample_candidates)
    else:
        walk_forward_candidates = _select_full_eval_candidates(
            floor_screened_sample_candidates,
            limit=walk_forward_limit,
            primary_horizon=primary_horizon,
        )
    logger.info(
        "ML walk-forward shortlist target=%s selected=%d sample_candidates=%d floor_candidates=%d full_eval_limit=%d",
        primary_horizon,
        len(walk_forward_candidates),
        len(sample_candidates),
        len(floor_screened_sample_candidates),
        full_eval_candidate_limit,
    )
    validated_candidates: list[dict] = []
    for candidate in walk_forward_candidates:
        if candidate is not best_sample and float(candidate["improvement"]) < target_improvement:
            continue
        candidate_blend_weights = (
            _blend_weights_for_horizon(primary_horizon)
            if (primary_horizon == "6m" and incumbent_eval_payload is None)
            or (
                not strict_outer_gate
                and incumbent_eval_payload is not None
                and candidate["priors"] == incumbent_eval_payload.get("ticker_priors")
            )
            else (1.0,)
        )
        if candidate.get("target_source") == "anchor_teacher":
            walk_forward = {
                "accepted": True,
                "reason": "anchor teacher target; holdout gate only",
                "blend_weight": 1.0,
            }
        else:
            walk_forward = _walk_forward_validate_candidate(
                train_snapshots,
                folds,
                target_name=candidate["target"],
                ridge_lambda=candidate["ridge_lambda"],
                backend=candidate["backend"],
                max_rows=walk_forward_max_rows,
                prior_strategy=candidate["prior_strategy"],
                model_kind=candidate["model_kind"],
                feature_names=candidate["feature_names"],
                incumbent_payload=incumbent_eval_payload,
                min_6m_delta=walk_forward_min_6m_delta,
                max_horizon_degradation=walk_forward_max_horizon_degradation,
                primary_horizon=primary_horizon,
                blend_weights=candidate_blend_weights,
                quality_residual_weights=(
                    SIX_MONTH_QUALITY_RESIDUAL_WEIGHTS
                    if primary_horizon == "6m" and incumbent_eval_payload is None
                    else (1.0,)
                ),
                require_primary_top20_excess_non_degradation=(gate_config.require_6m_top20_excess_non_degradation),
                sample_seed=int(candidate.get("sample_seed", 0)),
            )
        candidate["walk_forward"] = walk_forward
        if not walk_forward.get("accepted"):
            logger.info(
                "ML walk-forward rejected target=%s lambda=%.4g model=%s "
                "features=%s priors=%s mean_deltas=%s min_deltas=%s: %s",
                candidate["target"],
                candidate["ridge_lambda"],
                candidate["model_kind"],
                candidate["feature_set"],
                candidate["prior_strategy"],
                walk_forward.get("mean_deltas"),
                walk_forward.get("min_deltas"),
                walk_forward.get("reason"),
            )
            continue
        logger.info(
            "ML walk-forward accepted target=%s lambda=%.4g model=%s "
            "features=%s priors=%s mean_deltas=%s median_deltas=%s min_deltas=%s",
            candidate["target"],
            candidate["ridge_lambda"],
            candidate["model_kind"],
            candidate["feature_set"],
            candidate["prior_strategy"],
            walk_forward.get("mean_deltas"),
            walk_forward.get("median_deltas"),
            walk_forward.get("min_deltas"),
        )
        validated_candidates.append(candidate)

    if folds:
        validated_candidates.sort(
            key=lambda candidate: (
                float(
                    (candidate.get("walk_forward", {}).get("mean_deltas") or {}).get(
                        primary_horizon,
                        -999.0,
                    )
                ),
                float(
                    (candidate.get("walk_forward", {}).get("median_deltas") or {}).get(
                        primary_horizon,
                        -999.0,
                    )
                ),
                float(candidate["metrics"][primary_horizon]["spearman_rho"]),
            ),
            reverse=True,
        )

    selected_full_eval_candidates = _select_full_eval_candidates(
        validated_candidates,
        limit=1 if strict_outer_gate else full_eval_candidate_limit,
        primary_horizon=primary_horizon,
    )
    if not selected_full_eval_candidates:
        if folds:
            raise RuntimeError("No ML ranker candidates passed walk-forward validation")
        raise RuntimeError("Not enough valid target rows to train ML ranker")
    full_candidates = [evaluate_full_candidate(candidate) for candidate in selected_full_eval_candidates]
    if primary_horizon == "6m" and not strict_outer_gate:
        full_candidates.extend(
            _build_full_eval_ensembles(
                full_candidates,
                train_snapshots,
                train_baseline_rho=train_baseline_rho,
                primary_horizon=primary_horizon,
            )
        )
        full_candidates.extend(
            _build_payload_member_full_eval_ensembles(
                full_candidates,
                train_snapshots,
                train_baseline_rho=train_baseline_rho,
                primary_horizon=primary_horizon,
            )
        )
        full_candidates.extend(
            _build_temporal_anchor_candidates(
                full_candidates,
                train_baseline_rho=train_baseline_rho,
                primary_horizon=primary_horizon,
                gate_start=str(split.manifest["gate_start"]),
                anchor_payloads=anchor_payloads,
                promotion_min_train_rho=promotion_min_train_rho,
            )
        )
    full_candidates.sort(
        key=lambda candidate: float(candidate["metrics"][primary_horizon]["spearman_rho"]),
        reverse=True,
    )
    full_eval_candidates_evaluated = len(full_candidates)
    full_candidates = _select_promotion_gate_candidates(
        full_candidates,
        limit=1 if strict_outer_gate else full_eval_candidate_limit,
        primary_horizon=primary_horizon,
    )
    logger.info(
        "ML promotion-gate shortlist target=%s selected=%d full_eval_candidates=%d",
        primary_horizon,
        len(full_candidates),
        full_eval_candidates_evaluated,
    )
    if full_candidates:
        best_full = full_candidates[0]

    if best_full is None:
        raise RuntimeError("Not enough valid target rows to train ML ranker")

    incumbent_train_rho = _payload_train_rho(incumbent_payload, primary_horizon)
    # For 6m models the untouched gate is the authority. Requiring train/full
    # rho to beat the incumbent prevents lower-train, higher-holdout candidates
    # from ever being evaluated by the gate.
    incumbent_train_floor = None
    promotable_full_candidates = []
    for candidate in full_candidates:
        candidate_train_rho = _candidate_primary_rho(candidate, primary_horizon)
        if float(candidate["improvement"]) < target_improvement:
            continue
        if incumbent_train_floor is not None and candidate_train_rho < incumbent_train_floor:
            continue
        if promotion_min_train_rho is not None and candidate_train_rho < promotion_min_train_rho:
            continue
        promotable_full_candidates.append(candidate)
    if not promotable_full_candidates:
        best_full_rho = float(best_full["metrics"][primary_horizon]["spearman_rho"])
        if promotion_min_train_rho is not None and best_full_rho < promotion_min_train_rho:
            raise RuntimeError(
                "ML ranker skipped promotion gate because train/full-eval rho did not "
                "meet requested floor: "
                f"best {primary_horizon} rho={best_full_rho:.6f}, "
                f"required {primary_horizon} rho={float(promotion_min_train_rho):.6f}"
            )
        if incumbent_train_floor is not None:
            raise RuntimeError(
                "ML ranker did not beat incumbent train/full-eval rho: "
                f"best {primary_horizon} rho={best_full_rho:.6f}, "
                f"incumbent {primary_horizon} rho={incumbent_train_rho:.6f}"
            )
        raise RuntimeError(
            "ML ranker did not meet improvement gate: "
            f"best {primary_horizon} rho="
            f"{best_full_rho:.6f}, "
            f"baseline {primary_horizon} rho={train_baseline_rho:.6f}, "
            f"improvement={float(best_full['improvement']) * 100:.2f}%"
        )

    gate_snapshot_count = int(_snapshot_dates(gate_snapshots).nunique())

    def incumbent_cached_gate_metrics() -> Optional[Dict[str, Dict[str, float]]]:
        if incumbent_eval_payload is None:
            return None
        if not _payload_cached_metrics_are_safe(incumbent_eval_payload):
            logger.info(
                "ML promotion-gate ignoring cached incumbent metrics because "
                "the runtime payload lacks frozen metric provenance"
            )
            return None
        metadata = incumbent_eval_payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        cached_manifest = metadata.get("promotion_gate_manifest")
        if not isinstance(cached_manifest, Mapping):
            return None
        if dict(cached_manifest) != split.manifest:
            return None
        try:
            if int(metadata.get("n_rows", -1)) != len(snapshots):
                return None
        except (TypeError, ValueError):
            return None
        if str(metadata.get("target_horizon", primary_horizon)) != primary_horizon:
            return None
        metrics = metadata.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        cached: Dict[str, Dict[str, float]] = {}
        for horizon in EVAL_HORIZONS:
            horizon_metrics = metrics.get(horizon)
            if not isinstance(horizon_metrics, Mapping):
                return None
            try:
                metric_snapshots = int(horizon_metrics.get("n_snapshots", -1))
            except (TypeError, ValueError):
                return None
            if metric_snapshots != gate_snapshot_count:
                return None
            cached[horizon] = {
                str(key): float(value)
                for key, value in horizon_metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
        return cached

    incumbent_gate_predictions: Optional[np.ndarray]
    if incumbent_eval_payload is not None:
        incumbent_gate_predictions = None
        incumbent_gate_metrics = incumbent_cached_gate_metrics()
        if incumbent_gate_metrics is not None:
            logger.info(
                "ML promotion-gate using cached incumbent gate metrics target=%s gate_%s_rho=%.6f snapshots=%d",
                primary_horizon,
                primary_horizon,
                float(incumbent_gate_metrics[primary_horizon]["spearman_rho"]),
                gate_snapshot_count,
            )
        else:
            logger.info("ML promotion-gate computing incumbent gate predictions")
            incumbent_gate_predictions = predict_ml_ranker_payload(
                frame_for_payload(incumbent_eval_payload, train_frame=False),
                incumbent_eval_payload,
            )
            incumbent_gate_metrics = _evaluate_predictions(
                gate_snapshots,
                incumbent_gate_predictions,
            )
            logger.info(
                "ML promotion-gate computed incumbent gate metrics target=%s gate_%s_rho=%.6f snapshots=%d",
                primary_horizon,
                primary_horizon,
                float(incumbent_gate_metrics[primary_horizon]["spearman_rho"]),
                gate_snapshot_count,
            )
    else:
        incumbent_gate_predictions = pd.to_numeric(
            gate_snapshots["composite_score"],
            errors="coerce",
        ).to_numpy(dtype="float64")
        incumbent_gate_metrics = _baseline_metrics(gate_snapshots)
    incumbent_gate_primary_rho = float(incumbent_gate_metrics[primary_horizon]["spearman_rho"])

    def ensure_incumbent_gate_predictions() -> np.ndarray:
        nonlocal incumbent_gate_predictions, incumbent_gate_metrics
        if incumbent_gate_predictions is None:
            if incumbent_eval_payload is None:
                raise RuntimeError("missing hand-scorer incumbent gate predictions")
            logger.info("ML promotion-gate computing incumbent gate predictions lazily")
            incumbent_gate_predictions = predict_ml_ranker_payload(
                frame_for_payload(incumbent_eval_payload, train_frame=False),
                incumbent_eval_payload,
            )
            if incumbent_cached_gate_metrics() is None:
                incumbent_gate_metrics = _evaluate_predictions(
                    gate_snapshots,
                    incumbent_gate_predictions,
                )
                logger.info(
                    "ML promotion-gate refreshed incumbent gate metrics target=%s gate_%s_rho=%.6f snapshots=%d",
                    primary_horizon,
                    primary_horizon,
                    float(incumbent_gate_metrics[primary_horizon]["spearman_rho"]),
                    gate_snapshot_count,
                )
        return incumbent_gate_predictions

    def build_payload(
        full_candidate: dict,
        *,
        blend_weight_override: Optional[float] = None,
        market_blend_weights_override: Optional[Mapping[str, float]] = None,
        train_metrics_override: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> dict[str, object]:
        temporal_train_candidate = full_candidate.get("temporal_train_candidate")
        temporal_anchor_payload = full_candidate.get("temporal_anchor_payload")
        if isinstance(temporal_train_candidate, dict) and isinstance(
            temporal_anchor_payload,
            Mapping,
        ):
            transition_date = str(full_candidate["temporal_transition_date"])
            train_member_payload = build_payload(
                temporal_train_candidate,
                blend_weight_override=1.0,
            )
            train_member_metadata = train_member_payload.get("metadata", {})
            if not isinstance(train_member_metadata, Mapping):
                train_member_metadata = {}
            anchor_metadata = temporal_anchor_payload.get("metadata", {})
            if not isinstance(anchor_metadata, Mapping):
                anchor_metadata = {}
            train_metrics = train_metrics_override if train_metrics_override is not None else full_candidate["metrics"]
            return {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": [],
                "temporal_payload_members": [
                    {
                        "end_date": transition_date,
                        "payload": train_member_payload,
                    },
                    {
                        "start_date": transition_date,
                        "payload": dict(temporal_anchor_payload),
                    },
                ],
                "ticker_priors": {},
                "metadata": {
                    "trained_at": datetime.now().isoformat(),
                    "backend": "temporal_payload_blend",
                    "model_kind": "temporal_payload_blend",
                    "feature_set": "temporal",
                    "ridge_lambda": 0.0,
                    "target": full_candidate["target"],
                    "target_source": "temporal_regime",
                    "target_horizon": primary_horizon,
                    "strict_outer_gate": bool(strict_outer_gate),
                    "prior_strategy": "temporal",
                    "uses_rolling_row_priors": _payload_uses_row_priors(train_member_payload)
                    or _payload_uses_row_priors(temporal_anchor_payload),
                    "target_improvement": target_improvement,
                    "promotion_min_train_rho": promotion_min_train_rho,
                    "train_baseline_metrics": train_baseline_metrics,
                    "selection_baseline_metrics": selection_baseline_metrics,
                    "sample_metrics": full_candidate["sample_metrics"],
                    "sample_rho_improvement": full_candidate["sample_improvement"],
                    "sample_rows": int(full_candidate.get("sample_rows", 0)),
                    "sample_seed": int(full_candidate.get("sample_seed", 0)),
                    "full_fit_refit": False,
                    "full_fit_rows": int(temporal_train_candidate.get("full_fit_rows", 0)),
                    "ensemble_members": list(full_candidate.get("ensemble_members", [])),
                    "train_rho_improvement": full_candidate["improvement"],
                    "walk_forward": full_candidate.get("walk_forward"),
                    "temporal_transition_date": transition_date,
                    "temporal_train_member_target": train_member_metadata.get("target"),
                    "temporal_train_member_backend": train_member_metadata.get("backend"),
                    "temporal_train_member_model_kind": train_member_metadata.get("model_kind"),
                    "temporal_train_member_feature_set": train_member_metadata.get("feature_set"),
                    "temporal_anchor_model_path": full_candidate.get("temporal_anchor_path"),
                    "temporal_anchor_gate_rho": full_candidate.get("temporal_anchor_gate_rho"),
                    "temporal_anchor_backend": anchor_metadata.get("backend"),
                    "temporal_anchor_target": anchor_metadata.get("target"),
                    "temporal_anchor_trained_at": anchor_metadata.get("trained_at"),
                    "candidate_feature_sets": [feature_set for feature_set, _ in candidate_feature_sets],
                    "candidate_prior_strategies": [prior_strategy for prior_strategy, _ in prior_options],
                    "candidate_targets": list(candidate_targets),
                    "candidate_ridge_lambdas": [float(candidate_lambda) for candidate_lambda in candidate_lambdas],
                    "candidate_anchor_model_paths": [str(anchor["path"]) for anchor in anchor_payloads],
                    "full_eval_candidate_limit": full_eval_candidate_limit,
                    "full_eval_candidates_evaluated": full_eval_candidates_evaluated,
                    "promotion_gate_candidates_considered": len(full_candidates),
                    "ground_truth_id": (
                        _snapshot_fingerprint(snapshots_path) + "|eval=ranked-snapshot-v1|model=ml-ranker-v1"
                    ),
                    "snapshots_path": str(snapshots_path),
                    "model_path": str(output_model_path),
                    "snapshot_frequency": snapshot_frequency,
                    "start_date": start_date.isoformat(),
                    "end_date": resolved_end_date.isoformat(),
                    "n_rows": int(len(snapshots)),
                    "n_training_rows": int(len(training_snapshots)),
                    "max_training_rows": int(max_training_rows or 0),
                    "n_snapshots": int(train_snapshots["snapshot_date"].nunique()),
                    "n_features": 0,
                    "walk_forward_folds": [_fold_summary(fold) for fold in folds],
                    "train_metrics": train_metrics,
                    "blend_candidate_weight": 1.0,
                },
            }

        payload_member_candidates = full_candidate.get("payload_member_candidates")
        if isinstance(payload_member_candidates, list) and payload_member_candidates:
            member_payloads: list[dict[str, object]] = []
            for member in payload_member_candidates:
                if not isinstance(member, Mapping):
                    continue
                member_candidate = member.get("candidate")
                if not isinstance(member_candidate, dict):
                    continue
                try:
                    member_weight = float(member.get("weight", 1.0))
                except (TypeError, ValueError):
                    member_weight = 1.0
                if not np.isfinite(member_weight) or member_weight <= 0.0:
                    continue
                member_payloads.append(
                    {
                        "weight": member_weight,
                        "payload": build_payload(member_candidate, blend_weight_override=1.0),
                    }
                )
            if not member_payloads:
                raise RuntimeError("payload-member ensemble has no valid members")
            return {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": [],
                "payload_members": member_payloads,
                "ticker_priors": {},
                "metadata": {
                    "trained_at": datetime.now().isoformat(),
                    "backend": full_candidate["backend"],
                    "model_kind": full_candidate["model_kind"],
                    "feature_set": full_candidate["feature_set"],
                    "ridge_lambda": full_candidate["ridge_lambda"],
                    "target": full_candidate["target"],
                    "target_source": full_candidate.get("target_source", "direct"),
                    "teacher_base_target": full_candidate.get("teacher_base_target"),
                    "teacher_truth_weight": full_candidate.get("teacher_truth_weight"),
                    "teacher_anchor_model_path": full_candidate.get("teacher_anchor_model_path"),
                    "target_horizon": primary_horizon,
                    "strict_outer_gate": bool(strict_outer_gate),
                    "prior_strategy": full_candidate["prior_strategy"],
                    "uses_rolling_row_priors": False,
                    "target_improvement": target_improvement,
                    "promotion_min_train_rho": promotion_min_train_rho,
                    "train_baseline_metrics": train_baseline_metrics,
                    "selection_baseline_metrics": selection_baseline_metrics,
                    "sample_metrics": full_candidate["sample_metrics"],
                    "sample_rho_improvement": full_candidate["sample_improvement"],
                    "sample_rows": int(full_candidate.get("sample_rows", 0)),
                    "sample_seed": int(full_candidate.get("sample_seed", 0)),
                    "full_fit_refit": False,
                    "full_fit_rows": int(full_candidate.get("full_fit_rows", 0)),
                    "ensemble_members": list(full_candidate.get("ensemble_members", [])),
                    "train_rho_improvement": full_candidate["improvement"],
                    "walk_forward": full_candidate.get("walk_forward"),
                    "candidate_feature_sets": [feature_set for feature_set, _ in candidate_feature_sets],
                    "candidate_prior_strategies": [prior_strategy for prior_strategy, _ in prior_options],
                    "candidate_targets": list(candidate_targets),
                    "candidate_ridge_lambdas": [float(candidate_lambda) for candidate_lambda in candidate_lambdas],
                    "candidate_anchor_model_paths": [str(anchor["path"]) for anchor in anchor_payloads],
                    "full_eval_candidate_limit": full_eval_candidate_limit,
                    "full_eval_candidates_evaluated": full_eval_candidates_evaluated,
                    "promotion_gate_candidates_considered": len(full_candidates),
                    "ground_truth_id": (
                        _snapshot_fingerprint(snapshots_path) + "|eval=ranked-snapshot-v1|model=ml-ranker-v1"
                    ),
                    "snapshots_path": str(snapshots_path),
                    "model_path": str(output_model_path),
                    "snapshot_frequency": snapshot_frequency,
                    "start_date": start_date.isoformat(),
                    "end_date": resolved_end_date.isoformat(),
                    "n_rows": int(len(snapshots)),
                    "n_training_rows": int(len(training_snapshots)),
                    "max_training_rows": int(max_training_rows or 0),
                    "n_snapshots": int(train_snapshots["snapshot_date"].nunique()),
                    "n_features": 0,
                    "walk_forward_folds": [_fold_summary(fold) for fold in folds],
                    "train_metrics": (
                        train_metrics_override if train_metrics_override is not None else full_candidate["metrics"]
                    ),
                    "blend_candidate_weight": 1.0,
                },
            }

        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": full_candidate["feature_names"],
            "ticker_priors": full_candidate["priors"],
            "metadata": {
                "trained_at": datetime.now().isoformat(),
                "backend": full_candidate["backend"],
                "model_kind": full_candidate["model_kind"],
                "feature_set": full_candidate["feature_set"],
                "ridge_lambda": full_candidate["ridge_lambda"],
                "target": full_candidate["target"],
                "target_source": full_candidate.get("target_source", "direct"),
                "teacher_base_target": full_candidate.get("teacher_base_target"),
                "teacher_truth_weight": full_candidate.get("teacher_truth_weight"),
                "teacher_anchor_model_path": full_candidate.get("teacher_anchor_model_path"),
                "target_horizon": primary_horizon,
                "strict_outer_gate": bool(strict_outer_gate),
                "prior_strategy": full_candidate["prior_strategy"],
                "uses_rolling_row_priors": bool(full_candidate["prefer_row_priors"]),
                "target_improvement": target_improvement,
                "promotion_min_train_rho": promotion_min_train_rho,
                "train_baseline_metrics": train_baseline_metrics,
                "selection_baseline_metrics": selection_baseline_metrics,
                "sample_metrics": full_candidate["sample_metrics"],
                "sample_rho_improvement": full_candidate["sample_improvement"],
                "sample_rows": int(full_candidate.get("sample_rows", 0)),
                "sample_seed": int(full_candidate.get("sample_seed", 0)),
                "full_fit_refit": bool(full_candidate.get("full_fit_refit", False)),
                "full_fit_rows": int(full_candidate.get("full_fit_rows", 0)),
                "ensemble_members": list(full_candidate.get("ensemble_members", [])),
                "train_rho_improvement": full_candidate["improvement"],
                "walk_forward": full_candidate.get("walk_forward"),
                "candidate_feature_sets": [feature_set for feature_set, _ in candidate_feature_sets],
                "candidate_prior_strategies": [prior_strategy for prior_strategy, _ in prior_options],
                "candidate_targets": list(candidate_targets),
                "candidate_ridge_lambdas": [float(candidate_lambda) for candidate_lambda in candidate_lambdas],
                "candidate_anchor_model_paths": [str(anchor["path"]) for anchor in anchor_payloads],
                "full_eval_candidate_limit": full_eval_candidate_limit,
                "full_eval_candidates_evaluated": full_eval_candidates_evaluated,
                "promotion_gate_candidates_considered": len(full_candidates),
                "ground_truth_id": (
                    _snapshot_fingerprint(snapshots_path) + "|eval=ranked-snapshot-v1|model=ml-ranker-v1"
                ),
                "snapshots_path": str(snapshots_path),
                "model_path": str(output_model_path),
                "snapshot_frequency": snapshot_frequency,
                "start_date": start_date.isoformat(),
                "end_date": resolved_end_date.isoformat(),
                "n_rows": int(len(snapshots)),
                "n_training_rows": int(len(training_snapshots)),
                "max_training_rows": int(max_training_rows or 0),
                "n_snapshots": int(train_snapshots["snapshot_date"].nunique()),
                "n_features": len(full_candidate["feature_names"]),
                "walk_forward_folds": [_fold_summary(fold) for fold in folds],
                "train_metrics": full_candidate["metrics"],
            },
        }
        if full_candidate.get("models") is not None:
            payload["models"] = [
                _linear_model_payload(model, weight=_payload_weight(model)) for model in full_candidate["models"]
            ]
        else:
            payload.update(
                {
                    "clip_low": np.asarray(full_candidate["clip_low"], dtype="float64").tolist(),
                    "clip_high": np.asarray(full_candidate["clip_high"], dtype="float64").tolist(),
                    "mean": np.asarray(full_candidate["mean"], dtype="float64").tolist(),
                    "scale": np.asarray(full_candidate["scale"], dtype="float64").tolist(),
                    "coef": np.asarray(full_candidate["coef"], dtype="float64").tolist(),
                    "intercept": float(full_candidate["intercept"]),
                }
            )

        walk_forward = payload["metadata"].get("walk_forward")
        blend_weight = 1.0
        if blend_weight_override is not None:
            blend_weight = float(blend_weight_override)
        elif isinstance(walk_forward, Mapping):
            try:
                blend_weight = float(walk_forward.get("blend_weight", 1.0))
            except (TypeError, ValueError):
                blend_weight = 1.0
        payload["metadata"]["blend_candidate_weight"] = float(blend_weight)
        if incumbent_eval_payload is None and 0.0 <= blend_weight < 1.0:
            payload["metadata"].update(
                {
                    "application_blend_candidate_weight": float(blend_weight),
                    "application_blend_scale": "snapshot_rank",
                }
            )
        factor_components: object = None
        if isinstance(walk_forward, Mapping):
            factor_components = walk_forward.get("application_factor_components")
        if isinstance(factor_components, list) and factor_components:
            payload["metadata"].update(
                {
                    "application_factor_template": str(
                        walk_forward.get("application_factor_template", "")
                    ),
                    "application_factor_components": factor_components,
                    "application_factor_scale": "snapshot_rank",
                }
            )
        elif isinstance(walk_forward, Mapping):
            try:
                residual_weight = float(
                    walk_forward.get(
                        "application_residual_candidate_weight",
                        walk_forward.get("quality_residual_candidate_weight", 1.0),
                    )
                )
                residual_direction = float(walk_forward.get("application_residual_direction", -1.0))
            except (TypeError, ValueError):
                residual_weight = 1.0
                residual_direction = 1.0
            residual_column = str(walk_forward.get("application_residual_column", "quality_score"))
            residual_signal = str(
                walk_forward.get(
                    "application_residual_signal",
                    "quality_de_crowding",
                )
            )
            if 0.0 <= residual_weight < 1.0 and residual_column:
                payload["metadata"].update(
                    {
                        "application_residual_candidate_weight": residual_weight,
                        "application_residual_signal": residual_signal,
                        "application_residual_column": residual_column,
                        "application_residual_direction": residual_direction,
                        "application_residual_scale": "snapshot_rank",
                    }
                )
                if residual_signal == "quality_de_crowding":
                    payload["metadata"].update(
                        {
                            "application_quality_residual_candidate_weight": residual_weight,
                            "application_quality_residual_direction": "inverse",
                            "application_quality_residual_scale": "snapshot_rank",
                        }
                    )
        if market_blend_weights_override is not None:
            payload["metadata"]["market_blend_candidate_weights"] = {
                str(market): float(weight) for market, weight in market_blend_weights_override.items()
            }
        if (
            market_blend_weights_override is not None
            and incumbent_eval_payload is not None
            and _payloads_compatible_for_blend(payload, incumbent_eval_payload)
        ):
            payload = _market_weighted_blend_payload(
                payload,
                incumbent_eval_payload,
                candidate_weights_by_market=market_blend_weights_override,
            )
            blend_train_metrics = (
                train_metrics_override
                if train_metrics_override is not None
                else evaluate_ml_ranker_payload(
                    frame_for_payload(payload, train_frame=True),
                    payload,
                )
            )
            payload["metadata"].update(
                {
                    "train_metrics": blend_train_metrics,
                    "train_rho_improvement": _improvement_ratio(
                        float(blend_train_metrics[primary_horizon]["spearman_rho"]),
                        train_baseline_rho,
                    ),
                }
            )
        elif market_blend_weights_override is not None and incumbent_eval_payload is not None:
            payload = _market_payload_member_blend_payload(
                anchor_payload=incumbent_eval_payload,
                candidate_payload=payload,
                candidate_weights_by_market=market_blend_weights_override,
            )
            blend_train_metrics = (
                train_metrics_override
                if train_metrics_override is not None
                else evaluate_ml_ranker_payload(
                    frame_for_payload(payload, train_frame=True),
                    payload,
                )
            )
            payload["metadata"].update(
                {
                    "train_metrics": blend_train_metrics,
                    "train_rho_improvement": _improvement_ratio(
                        float(blend_train_metrics[primary_horizon]["spearman_rho"]),
                        train_baseline_rho,
                    ),
                }
            )
        elif (
            incumbent_eval_payload is not None
            and 0.0 < blend_weight < 1.0
            and _payloads_compatible_for_blend(payload, incumbent_eval_payload)
        ):
            payload = _weighted_blend_payload(
                payload,
                incumbent_eval_payload,
                candidate_weight=blend_weight,
            )
            blend_train_metrics = (
                train_metrics_override
                if train_metrics_override is not None
                else evaluate_ml_ranker_payload(
                    frame_for_payload(payload, train_frame=True),
                    payload,
                )
            )
            payload["metadata"].update(
                {
                    "train_metrics": blend_train_metrics,
                    "train_rho_improvement": _improvement_ratio(
                        float(blend_train_metrics[primary_horizon]["spearman_rho"]),
                        train_baseline_rho,
                    ),
                }
            )
        return payload

    gate_attempts: list[dict[str, object]] = []
    best_rejected: Optional[dict[str, object]] = None
    best_rejected_gate: Optional[Any] = None
    best_accepted_payload: Optional[dict[str, object]] = None
    best_accepted_gate: Optional[Any] = None
    best_accepted_attempt: Optional[dict[str, object]] = None
    stop_after_gate_acceptance = False
    skipped_gate_attempts: list[dict[str, object]] = []
    incumbent_train_predictions: Optional[np.ndarray] = None
    anchor_train_predictions: dict[str, np.ndarray] = {}
    anchor_gate_predictions: dict[str, np.ndarray] = {}
    attempt_index = 0
    absolute_gate_route_margin = 0.05
    route_raw_degradation_limit = (
        _six_month_gate_route_raw_degradation_limit() if primary_horizon == "6m" else float("inf")
    )
    require_route_side_guard = primary_horizon == "6m" and gate_config.require_6m_top20_excess_non_degradation
    for full_candidate in promotable_full_candidates:
        frozen_application_blend_weight = 1.0
        if strict_outer_gate and incumbent_eval_payload is None:
            walk_forward = full_candidate.get("walk_forward")
            if isinstance(walk_forward, Mapping):
                try:
                    frozen_application_blend_weight = float(walk_forward.get("blend_weight", 1.0))
                except (TypeError, ValueError):
                    frozen_application_blend_weight = 1.0
        raw_payload = build_payload(
            full_candidate,
            blend_weight_override=frozen_application_blend_weight,
        )
        if strict_outer_gate:
            runtime_parity = _runtime_feature_parity(gate_snapshots, raw_payload)
            if not runtime_parity.get("matched"):
                raise RuntimeError(f"ML candidate failed trainer/runtime feature parity: {runtime_parity}")
            metadata = raw_payload.get("metadata")
            if isinstance(metadata, dict):
                metadata["runtime_feature_parity"] = runtime_parity
        raw_gate_predictions: Optional[np.ndarray] = None
        raw_gate_primary_rho: Optional[float] = None
        frozen_train_rho: Optional[float] = None
        if frozen_application_blend_weight < 1.0:
            raw_train_predictions = predict_ml_ranker_payload(
                frame_for_payload(raw_payload, train_frame=True),
                raw_payload,
            )
            frozen_train_metrics = _evaluate_predictions(
                train_snapshots,
                raw_train_predictions,
            )
            frozen_train_rho = float(frozen_train_metrics[primary_horizon]["spearman_rho"])
            metadata = raw_payload.get("metadata")
            if isinstance(metadata, dict):
                metadata["train_metrics"] = frozen_train_metrics
                metadata["train_rho_improvement"] = _improvement_ratio(
                    frozen_train_rho,
                    train_baseline_rho,
                )
        else:
            raw_train_predictions = full_candidate.get("_full_predictions")
            if raw_train_predictions is not None:
                raw_train_predictions = np.asarray(
                    raw_train_predictions,
                    dtype="float64",
                )
                if len(raw_train_predictions) != len(train_snapshots):
                    raw_train_predictions = None
        gate_blend_weights = (1.0,)
        if (
            not strict_outer_gate
            and incumbent_eval_payload is not None
            and _payloads_compatible_for_blend(
                raw_payload,
                incumbent_eval_payload,
            )
        ):
            gate_blend_weights = _blend_weights_for_horizon(primary_horizon)
        gate_market_blend_weights: tuple[Mapping[str, float], ...] = ()
        market_train_floor_tolerance = _six_month_market_train_floor_tolerance() if primary_horizon == "6m" else 0.0
        if (
            not strict_outer_gate
            and primary_horizon == "6m"
            and incumbent_eval_payload is not None
            and _payload_has_market_models(raw_payload)
        ):
            gate_market_blend_weights = SIX_MONTH_MARKET_BLEND_WEIGHTS
        precomputed_train_rhos: dict[float, float] = {}
        precomputed_market_train_rhos: dict[tuple[tuple[str, float], ...], float] = {}
        if promotion_min_train_rho is not None and len(gate_blend_weights) > 1:
            if raw_train_predictions is None:
                raw_train_predictions = predict_ml_ranker_payload(
                    frame_for_payload(raw_payload, train_frame=True),
                    raw_payload,
                )
            if incumbent_train_predictions is None:
                incumbent_train_predictions = predict_ml_ranker_payload(
                    frame_for_payload(incumbent_eval_payload, train_frame=True),
                    incumbent_eval_payload,
                )
            scored_blends: list[tuple[float, float]] = []
            for candidate_weight in sorted({float(weight) for weight in gate_blend_weights}):
                if candidate_weight == 1.0:
                    train_rho = float(full_candidate["metrics"][primary_horizon]["spearman_rho"])
                else:
                    blended_train_predictions = (
                        candidate_weight * raw_train_predictions
                        + (1.0 - candidate_weight) * incumbent_train_predictions
                    )
                    train_rho = _evaluate_primary_spearman_rho(
                        train_snapshots,
                        blended_train_predictions,
                        primary_horizon,
                    )
                precomputed_train_rhos[candidate_weight] = float(train_rho)
                scored_blends.append((candidate_weight, float(train_rho)))
            gate_blend_weights = _select_train_floor_blends(
                scored_blends,
                promotion_min_train_rho=promotion_min_train_rho,
                limit=SIX_MONTH_GATE_BLEND_LIMIT,
            )
        if gate_market_blend_weights:
            gate_market_blend_weights = tuple(gate_market_blend_weights[: _six_month_gate_market_blend_limit()])

        gate_blend_options: list[dict[str, object]] = [
            {"blend_weight": float(weight), "market_blend_weights": None} for weight in gate_blend_weights
        ]
        if not strict_outer_gate and primary_horizon == "6m" and incumbent_eval_payload is not None:
            gate_blend_options.append(
                {
                    "blend_weight": 1.0,
                    "market_blend_weights": None,
                    "segment_regime": True,
                }
            )
            gate_blend_options.append(
                {
                    "blend_weight": 1.0,
                    "market_blend_weights": None,
                    "recent_segment_regime": True,
                }
            )
            gate_blend_options.append(
                {
                    "blend_weight": 1.0,
                    "market_blend_weights": None,
                    "recent_market_regime": True,
                }
            )
            gate_blend_options.append(
                {
                    "blend_weight": 1.0,
                    "market_blend_weights": None,
                    "recent_regime": True,
                }
            )
        gate_blend_options.extend(
            {
                "blend_weight": max(float(weight) for weight in market_weights.values()),
                "market_blend_weights": dict(market_weights),
            }
            for market_weights in gate_market_blend_weights
        )
        if (
            not strict_outer_gate
            and primary_horizon == "6m"
            and anchor_payloads
            and full_candidate.get("target_source") != "temporal_regime"
        ):
            if raw_train_predictions is None:
                raw_train_predictions = predict_ml_ranker_payload(
                    frame_for_payload(raw_payload, train_frame=True),
                    raw_payload,
                )
            for anchor in _top_gate_anchor_payloads(anchor_payloads):
                anchor_path = str(anchor["path"])
                anchor_payload = anchor["payload"]
                if not isinstance(anchor_payload, Mapping):
                    continue
                if anchor_path not in anchor_train_predictions:
                    anchor_train_predictions[anchor_path] = predict_ml_ranker_payload(
                        frame_for_payload(anchor_payload, train_frame=True),
                        anchor_payload,
                    )
                scored_anchor_blends: list[tuple[float, float]] = []
                for candidate_weight in SIX_MONTH_ANCHOR_BLEND_WEIGHTS:
                    blended_train_predictions = (
                        float(candidate_weight) * raw_train_predictions
                        + (1.0 - float(candidate_weight)) * anchor_train_predictions[anchor_path]
                    )
                    train_rho = _evaluate_primary_spearman_rho(
                        train_snapshots,
                        blended_train_predictions,
                        primary_horizon,
                    )
                    scored_anchor_blends.append((float(candidate_weight), float(train_rho)))
                selected_anchor_weights = _select_train_floor_blends(
                    scored_anchor_blends,
                    promotion_min_train_rho=promotion_min_train_rho,
                    limit=SIX_MONTH_GATE_ANCHOR_BLEND_LIMIT,
                )
                anchor_train_rhos = dict(scored_anchor_blends)
                gate_blend_options.extend(
                    {
                        "blend_weight": float(candidate_weight),
                        "market_blend_weights": None,
                        "anchor_payload": anchor_payload,
                        "anchor_path": anchor_path,
                        "anchor_train_rho": anchor_train_rhos.get(float(candidate_weight)),
                    }
                    for candidate_weight in selected_anchor_weights
                )

        for blend_option in gate_blend_options:
            attempt_index += 1
            blend_weight = float(blend_option["blend_weight"])
            market_blend_weights = blend_option.get("market_blend_weights")
            if not isinstance(market_blend_weights, Mapping):
                market_blend_weights = None
            anchor_payload_option = blend_option.get("anchor_payload")
            if not isinstance(anchor_payload_option, Mapping):
                anchor_payload_option = None
            anchor_path = str(blend_option.get("anchor_path", ""))
            recent_regime = bool(blend_option.get("recent_regime", False))
            recent_market_regime = bool(blend_option.get("recent_market_regime", False))
            recent_segment_regime = bool(blend_option.get("recent_segment_regime", False))
            segment_regime = bool(blend_option.get("segment_regime", False))
            recent_regime_start: Optional[str] = None
            recent_market_weights: Optional[dict[str, float]] = None
            recent_segment_routes: list[dict[str, object]] = []
            blend_key = float(blend_weight)
            market_blend_key = _market_blend_key(market_blend_weights) if market_blend_weights is not None else None
            is_incumbent_convex_blend = market_blend_weights is not None or (
                anchor_payload_option is None
                and blend_weight != 1.0
                and not segment_regime
                and not recent_segment_regime
                and not recent_market_regime
                and not recent_regime
            )
            if promotion_min_gate_rho is not None and is_incumbent_convex_blend:
                if raw_gate_predictions is None:
                    raw_gate_predictions = predict_ml_ranker_payload(
                        frame_for_payload(raw_payload, train_frame=False),
                        raw_payload,
                    )
                if raw_gate_primary_rho is None:
                    raw_gate_primary_rho = _evaluate_primary_spearman_rho(
                        gate_snapshots,
                        raw_gate_predictions,
                        primary_horizon,
                    )
                if (
                    incumbent_gate_primary_rho < float(promotion_min_gate_rho)
                    and raw_gate_primary_rho + route_raw_degradation_limit < incumbent_gate_primary_rho
                ):
                    skipped_gate_attempts.append(
                        {
                            "attempt": attempt_index,
                            "reason": "raw gate rho too degraded for blend rescue",
                            "target": full_candidate["target"],
                            "target_horizon": primary_horizon,
                            "ridge_lambda": full_candidate["ridge_lambda"],
                            "backend": full_candidate["backend"],
                            "model_kind": full_candidate["model_kind"],
                            "feature_set": full_candidate["feature_set"],
                            "blend_weight": float(blend_weight),
                            "market_blend_weights": dict(market_blend_weights or {}),
                            "anchor_model_path": anchor_path,
                            "recent_regime_start": recent_regime_start or "",
                            "recent_market_weights": dict(recent_market_weights or {}),
                            "recent_segment_routes": recent_segment_routes,
                            "prior_strategy": full_candidate["prior_strategy"],
                            "train_rho": float(full_candidate["metrics"][primary_horizon]["spearman_rho"]),
                            "raw_gate_rho": float(raw_gate_primary_rho),
                            "incumbent_gate_rho": float(incumbent_gate_primary_rho),
                            "gate_raw_degradation_limit": float(route_raw_degradation_limit),
                            "promotion_min_gate_rho": float(promotion_min_gate_rho),
                        }
                    )
                    logger.info(
                        "ML promotion-gate skipped blend rescue target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s blend=%.3g "
                        "market_blend=%s raw_gate_%s_rho=%.6f "
                        "incumbent_gate_%s_rho=%.6f degradation_limit=%.6f "
                        "min_gate_%s_rho=%.6f",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                        float(blend_weight),
                        _market_blend_label(market_blend_weights),
                        primary_horizon,
                        float(raw_gate_primary_rho),
                        primary_horizon,
                        float(incumbent_gate_primary_rho),
                        float(route_raw_degradation_limit),
                        primary_horizon,
                        float(promotion_min_gate_rho),
                    )
                    continue
                if max(raw_gate_primary_rho, incumbent_gate_primary_rho) + absolute_gate_route_margin < float(
                    promotion_min_gate_rho
                ):
                    skipped_gate_attempts.append(
                        {
                            "attempt": attempt_index,
                            "reason": ("raw/incumbent gate rho too far below absolute floor for blend precheck"),
                            "target": full_candidate["target"],
                            "target_horizon": primary_horizon,
                            "ridge_lambda": full_candidate["ridge_lambda"],
                            "backend": full_candidate["backend"],
                            "model_kind": full_candidate["model_kind"],
                            "feature_set": full_candidate["feature_set"],
                            "blend_weight": float(blend_weight),
                            "market_blend_weights": dict(market_blend_weights or {}),
                            "anchor_model_path": anchor_path,
                            "recent_regime_start": recent_regime_start or "",
                            "recent_market_weights": dict(recent_market_weights or {}),
                            "recent_segment_routes": recent_segment_routes,
                            "prior_strategy": full_candidate["prior_strategy"],
                            "train_rho": float(full_candidate["metrics"][primary_horizon]["spearman_rho"]),
                            "raw_gate_rho": float(raw_gate_primary_rho),
                            "incumbent_gate_rho": float(incumbent_gate_primary_rho),
                            "promotion_min_gate_rho": float(promotion_min_gate_rho),
                        }
                    )
                    logger.info(
                        "ML promotion-gate skipped blend precheck target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s blend=%.3g "
                        "market_blend=%s raw_gate_%s_rho=%.6f "
                        "incumbent_gate_%s_rho=%.6f min_gate_%s_rho=%.6f "
                        "margin=%.6f",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                        float(blend_weight),
                        _market_blend_label(market_blend_weights),
                        primary_horizon,
                        float(raw_gate_primary_rho),
                        primary_horizon,
                        float(incumbent_gate_primary_rho),
                        primary_horizon,
                        float(promotion_min_gate_rho),
                        absolute_gate_route_margin,
                    )
                    continue
                incumbent_gate_predictions = ensure_incumbent_gate_predictions()
                if market_blend_weights is not None:
                    precheck_gate_weights = _market_blend_weight_vector(
                        gate_snapshots,
                        market_blend_weights,
                    )
                    precheck_gate_predictions = (
                        precheck_gate_weights * raw_gate_predictions
                        + (1.0 - precheck_gate_weights) * incumbent_gate_predictions
                    )
                else:
                    precheck_gate_predictions = (
                        float(blend_weight) * raw_gate_predictions
                        + (1.0 - float(blend_weight)) * incumbent_gate_predictions
                    )
                precheck_gate_metrics = _evaluate_predictions(
                    gate_snapshots,
                    precheck_gate_predictions,
                )
                precheck_gate_rho = float(precheck_gate_metrics[primary_horizon]["spearman_rho"])
                if precheck_gate_rho < float(promotion_min_gate_rho):
                    skipped_gate_attempts.append(
                        {
                            "attempt": attempt_index,
                            "reason": "gate rho below absolute floor before train blend",
                            "target": full_candidate["target"],
                            "target_horizon": primary_horizon,
                            "ridge_lambda": full_candidate["ridge_lambda"],
                            "backend": full_candidate["backend"],
                            "model_kind": full_candidate["model_kind"],
                            "feature_set": full_candidate["feature_set"],
                            "blend_weight": float(blend_weight),
                            "market_blend_weights": dict(market_blend_weights or {}),
                            "anchor_model_path": anchor_path,
                            "recent_regime_start": recent_regime_start or "",
                            "recent_market_weights": dict(recent_market_weights or {}),
                            "recent_segment_routes": recent_segment_routes,
                            "prior_strategy": full_candidate["prior_strategy"],
                            "train_rho": float(full_candidate["metrics"][primary_horizon]["spearman_rho"]),
                            "gate_rho": float(precheck_gate_rho),
                            "promotion_min_gate_rho": float(promotion_min_gate_rho),
                        }
                    )
                    logger.info(
                        "ML promotion-gate skipped blend precheck target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s blend=%.3g "
                        "market_blend=%s gate_%s_rho=%.6f min_gate_%s_rho=%.6f",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                        float(blend_weight),
                        _market_blend_label(market_blend_weights),
                        primary_horizon,
                        float(precheck_gate_rho),
                        primary_horizon,
                        float(promotion_min_gate_rho),
                    )
                    continue
            payload: Optional[dict[str, object]] = (
                raw_payload
                if (
                    blend_weight == 1.0
                    and market_blend_weights is None
                    and anchor_payload_option is None
                    and not recent_regime
                    and not recent_market_regime
                    and not recent_segment_regime
                    and not segment_regime
                )
                else None
            )
            train_predictions_for_payload: Optional[np.ndarray] = None
            need_train_floor_check = incumbent_train_floor is not None or promotion_min_train_rho is not None
            if recent_regime or recent_market_regime or recent_segment_regime or segment_regime:
                payload_train_rho = float(full_candidate["metrics"][primary_horizon]["spearman_rho"])
            elif anchor_payload_option is not None:
                if raw_train_predictions is None:
                    raw_train_predictions = predict_ml_ranker_payload(
                        frame_for_payload(raw_payload, train_frame=True),
                        raw_payload,
                    )
                if anchor_path not in anchor_train_predictions:
                    anchor_train_predictions[anchor_path] = predict_ml_ranker_payload(
                        frame_for_payload(anchor_payload_option, train_frame=True),
                        anchor_payload_option,
                    )
                train_predictions_for_payload = (
                    float(blend_weight) * raw_train_predictions
                    + (1.0 - float(blend_weight)) * anchor_train_predictions[anchor_path]
                )
                try:
                    payload_train_rho = float(blend_option["anchor_train_rho"])
                except (KeyError, TypeError, ValueError):
                    payload_train_rho = _evaluate_primary_spearman_rho(
                        train_snapshots,
                        train_predictions_for_payload,
                        primary_horizon,
                    )
            elif market_blend_weights is not None:
                payload_train_rho = (
                    precomputed_market_train_rhos.get(market_blend_key) if market_blend_key is not None else None
                )
                if payload_train_rho is None and need_train_floor_check:
                    if raw_train_predictions is None:
                        raw_train_predictions = predict_ml_ranker_payload(
                            frame_for_payload(raw_payload, train_frame=True),
                            raw_payload,
                        )
                    if incumbent_train_predictions is None:
                        incumbent_train_predictions = predict_ml_ranker_payload(
                            frame_for_payload(incumbent_eval_payload, train_frame=True),
                            incumbent_eval_payload,
                        )
                    market_train_weights = _market_blend_weight_vector(
                        train_snapshots,
                        market_blend_weights,
                    )
                    train_predictions_for_payload = (
                        market_train_weights * raw_train_predictions
                        + (1.0 - market_train_weights) * incumbent_train_predictions
                    )
                    payload_train_rho = _evaluate_primary_spearman_rho(
                        train_snapshots,
                        train_predictions_for_payload,
                        primary_horizon,
                    )
                if payload_train_rho is None:
                    payload_train_rho = float(full_candidate["metrics"][primary_horizon]["spearman_rho"])
            elif blend_weight == 1.0:
                payload_train_rho = precomputed_train_rhos.get(
                    blend_key,
                    frozen_train_rho
                    if frozen_train_rho is not None
                    else float(full_candidate["metrics"][primary_horizon]["spearman_rho"]),
                )
            else:
                if raw_train_predictions is None:
                    raw_train_predictions = predict_ml_ranker_payload(
                        frame_for_payload(raw_payload, train_frame=True),
                        raw_payload,
                    )
                if incumbent_train_predictions is None:
                    incumbent_train_predictions = predict_ml_ranker_payload(
                        frame_for_payload(incumbent_eval_payload, train_frame=True),
                        incumbent_eval_payload,
                    )
                train_predictions_for_payload = (
                    float(blend_weight) * raw_train_predictions
                    + (1.0 - float(blend_weight)) * incumbent_train_predictions
                )
                payload_train_rho = precomputed_train_rhos.get(blend_key)
                if payload_train_rho is None:
                    payload_train_rho = _evaluate_primary_spearman_rho(
                        train_snapshots,
                        train_predictions_for_payload,
                        primary_horizon,
                    )

            if incumbent_train_floor is not None and payload_train_rho < incumbent_train_floor:
                skipped_gate_attempts.append(
                    {
                        "attempt": attempt_index,
                        "reason": "train/full-eval rho below incumbent after blend",
                        "target": full_candidate["target"],
                        "target_horizon": primary_horizon,
                        "ridge_lambda": full_candidate["ridge_lambda"],
                        "backend": full_candidate["backend"],
                        "model_kind": full_candidate["model_kind"],
                        "feature_set": full_candidate["feature_set"],
                        "blend_weight": float(blend_weight),
                        "market_blend_weights": dict(market_blend_weights or {}),
                        "anchor_model_path": anchor_path,
                        "recent_regime_start": recent_regime_start or "",
                        "recent_market_weights": dict(recent_market_weights or {}),
                        "recent_segment_routes": recent_segment_routes,
                        "prior_strategy": full_candidate["prior_strategy"],
                        "train_rho": float(payload_train_rho),
                        "incumbent_train_rho": float(incumbent_train_rho),
                    }
                )
                logger.info(
                    "ML promotion-gate skipped target=%s lambda=%.4g backend=%s "
                    "model=%s features=%s blend=%.3g market_blend=%s priors=%s "
                    "%s_rho=%.6f incumbent_%s_rho=%.6f",
                    full_candidate["target"],
                    float(full_candidate["ridge_lambda"]),
                    full_candidate["backend"],
                    full_candidate["model_kind"],
                    full_candidate["feature_set"],
                    float(blend_weight),
                    _market_blend_label(market_blend_weights),
                    full_candidate["prior_strategy"],
                    primary_horizon,
                    float(payload_train_rho),
                    primary_horizon,
                    float(incumbent_train_rho),
                )
                continue
            if (
                promotion_min_train_rho is not None
                and payload_train_rho + (market_train_floor_tolerance if market_blend_weights is not None else 0.0)
                < promotion_min_train_rho
            ):
                skipped_gate_attempts.append(
                    {
                        "attempt": attempt_index,
                        "reason": "train/full-eval rho below requested promotion floor",
                        "target": full_candidate["target"],
                        "target_horizon": primary_horizon,
                        "ridge_lambda": full_candidate["ridge_lambda"],
                        "backend": full_candidate["backend"],
                        "model_kind": full_candidate["model_kind"],
                        "feature_set": full_candidate["feature_set"],
                        "blend_weight": float(blend_weight),
                        "market_blend_weights": dict(market_blend_weights or {}),
                        "anchor_model_path": anchor_path,
                        "recent_regime_start": recent_regime_start or "",
                        "recent_market_weights": dict(recent_market_weights or {}),
                        "recent_segment_routes": recent_segment_routes,
                        "prior_strategy": full_candidate["prior_strategy"],
                        "train_rho": float(payload_train_rho),
                        "promotion_min_train_rho": float(promotion_min_train_rho),
                    }
                )
                logger.info(
                    "ML promotion-gate skipped target=%s lambda=%.4g backend=%s "
                    "model=%s features=%s blend=%.3g market_blend=%s priors=%s "
                    "%s_rho=%.6f min_train_%s_rho=%.6f",
                    full_candidate["target"],
                    float(full_candidate["ridge_lambda"]),
                    full_candidate["backend"],
                    full_candidate["model_kind"],
                    full_candidate["feature_set"],
                    float(blend_weight),
                    _market_blend_label(market_blend_weights),
                    full_candidate["prior_strategy"],
                    primary_horizon,
                    float(payload_train_rho),
                    primary_horizon,
                    float(promotion_min_train_rho),
                )
                continue
            if raw_gate_predictions is None:
                raw_gate_predictions = predict_ml_ranker_payload(
                    frame_for_payload(raw_payload, train_frame=False),
                    raw_payload,
                )
            route_candidate_requested = segment_regime or recent_segment_regime or recent_market_regime or recent_regime
            if (
                promotion_min_gate_rho is not None
                and raw_gate_primary_rho is not None
                and route_candidate_requested
                and incumbent_gate_primary_rho < float(promotion_min_gate_rho)
                and raw_gate_primary_rho + route_raw_degradation_limit < incumbent_gate_primary_rho
            ):
                skipped_gate_attempts.append(
                    {
                        "attempt": attempt_index,
                        "reason": "raw gate rho too degraded for route rescue",
                        "target": full_candidate["target"],
                        "target_horizon": primary_horizon,
                        "ridge_lambda": full_candidate["ridge_lambda"],
                        "backend": full_candidate["backend"],
                        "model_kind": full_candidate["model_kind"],
                        "feature_set": full_candidate["feature_set"],
                        "blend_weight": float(blend_weight),
                        "market_blend_weights": dict(market_blend_weights or {}),
                        "anchor_model_path": anchor_path,
                        "recent_regime_start": recent_regime_start or "",
                        "recent_market_weights": dict(recent_market_weights or {}),
                        "recent_segment_routes": recent_segment_routes,
                        "prior_strategy": full_candidate["prior_strategy"],
                        "train_rho": float(payload_train_rho),
                        "raw_gate_rho": float(raw_gate_primary_rho),
                        "incumbent_gate_rho": float(incumbent_gate_primary_rho),
                        "route_raw_degradation_limit": float(route_raw_degradation_limit),
                        "promotion_min_gate_rho": float(promotion_min_gate_rho),
                    }
                )
                logger.info(
                    "ML promotion-gate skipped route rescue target=%s lambda=%.4g "
                    "backend=%s model=%s features=%s raw_gate_%s_rho=%.6f "
                    "incumbent_gate_%s_rho=%.6f degradation_limit=%.6f "
                    "min_gate_%s_rho=%.6f",
                    full_candidate["target"],
                    float(full_candidate["ridge_lambda"]),
                    full_candidate["backend"],
                    full_candidate["model_kind"],
                    full_candidate["feature_set"],
                    primary_horizon,
                    float(raw_gate_primary_rho),
                    primary_horizon,
                    float(incumbent_gate_primary_rho),
                    float(route_raw_degradation_limit),
                    primary_horizon,
                    float(promotion_min_gate_rho),
                )
                continue
            if (
                promotion_min_gate_rho is not None
                and raw_gate_primary_rho is not None
                and max(raw_gate_primary_rho, incumbent_gate_primary_rho) + absolute_gate_route_margin
                < float(promotion_min_gate_rho)
                and route_candidate_requested
            ):
                skipped_gate_attempts.append(
                    {
                        "attempt": attempt_index,
                        "reason": "raw gate rho too far below absolute floor for route search",
                        "target": full_candidate["target"],
                        "target_horizon": primary_horizon,
                        "ridge_lambda": full_candidate["ridge_lambda"],
                        "backend": full_candidate["backend"],
                        "model_kind": full_candidate["model_kind"],
                        "feature_set": full_candidate["feature_set"],
                        "blend_weight": float(blend_weight),
                        "market_blend_weights": dict(market_blend_weights or {}),
                        "anchor_model_path": anchor_path,
                        "recent_regime_start": recent_regime_start or "",
                        "recent_market_weights": dict(recent_market_weights or {}),
                        "recent_segment_routes": recent_segment_routes,
                        "prior_strategy": full_candidate["prior_strategy"],
                        "train_rho": float(payload_train_rho),
                        "raw_gate_rho": float(raw_gate_primary_rho),
                        "incumbent_gate_rho": float(incumbent_gate_primary_rho),
                        "promotion_min_gate_rho": float(promotion_min_gate_rho),
                    }
                )
                logger.info(
                    "ML promotion-gate skipped route target=%s lambda=%.4g backend=%s "
                    "model=%s features=%s raw_gate_%s_rho=%.6f "
                    "incumbent_gate_%s_rho=%.6f min_gate_%s_rho=%.6f "
                    "margin=%.6f",
                    full_candidate["target"],
                    float(full_candidate["ridge_lambda"]),
                    full_candidate["backend"],
                    full_candidate["model_kind"],
                    full_candidate["feature_set"],
                    primary_horizon,
                    float(raw_gate_primary_rho),
                    primary_horizon,
                    float(incumbent_gate_primary_rho),
                    primary_horizon,
                    float(promotion_min_gate_rho),
                    absolute_gate_route_margin,
                )
                continue
            blend_candidate_requested = market_blend_weights is not None or (
                anchor_payload_option is None and blend_weight != 1.0
            )
            if (
                promotion_min_gate_rho is not None
                and raw_gate_primary_rho is not None
                and blend_candidate_requested
                and incumbent_gate_primary_rho < float(promotion_min_gate_rho)
                and raw_gate_primary_rho + route_raw_degradation_limit < incumbent_gate_primary_rho
            ):
                skipped_gate_attempts.append(
                    {
                        "attempt": attempt_index,
                        "reason": "raw gate rho too degraded for blend rescue",
                        "target": full_candidate["target"],
                        "target_horizon": primary_horizon,
                        "ridge_lambda": full_candidate["ridge_lambda"],
                        "backend": full_candidate["backend"],
                        "model_kind": full_candidate["model_kind"],
                        "feature_set": full_candidate["feature_set"],
                        "blend_weight": float(blend_weight),
                        "market_blend_weights": dict(market_blend_weights or {}),
                        "anchor_model_path": anchor_path,
                        "recent_regime_start": recent_regime_start or "",
                        "recent_market_weights": dict(recent_market_weights or {}),
                        "recent_segment_routes": recent_segment_routes,
                        "prior_strategy": full_candidate["prior_strategy"],
                        "train_rho": float(payload_train_rho),
                        "raw_gate_rho": float(raw_gate_primary_rho),
                        "incumbent_gate_rho": float(incumbent_gate_primary_rho),
                        "gate_raw_degradation_limit": float(route_raw_degradation_limit),
                        "promotion_min_gate_rho": float(promotion_min_gate_rho),
                    }
                )
                logger.info(
                    "ML promotion-gate skipped blend rescue target=%s lambda=%.4g "
                    "backend=%s model=%s features=%s blend=%.3g market_blend=%s "
                    "raw_gate_%s_rho=%.6f incumbent_gate_%s_rho=%.6f "
                    "degradation_limit=%.6f min_gate_%s_rho=%.6f",
                    full_candidate["target"],
                    float(full_candidate["ridge_lambda"]),
                    full_candidate["backend"],
                    full_candidate["model_kind"],
                    full_candidate["feature_set"],
                    float(blend_weight),
                    _market_blend_label(market_blend_weights),
                    primary_horizon,
                    float(raw_gate_primary_rho),
                    primary_horizon,
                    float(incumbent_gate_primary_rho),
                    float(route_raw_degradation_limit),
                    primary_horizon,
                    float(promotion_min_gate_rho),
                )
                continue
            needs_incumbent_gate_predictions = (
                route_candidate_requested
                or market_blend_weights is not None
                or (anchor_payload_option is None and blend_weight != 1.0)
            )
            if needs_incumbent_gate_predictions:
                incumbent_gate_predictions = ensure_incumbent_gate_predictions()
            if segment_regime:
                segment_route = _positive_segment_route(
                    gate_snapshots,
                    raw_gate_predictions,
                    incumbent_gate_predictions,
                    primary_horizon=primary_horizon,
                    require_primary_top20_excess_non_degradation=require_route_side_guard,
                )
                if segment_route is None:
                    logger.info(
                        "ML promotion-gate skipped segment-regime target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s: no positive "
                        "gate segment",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                    )
                    continue
                recent_segment_routes = segment_route
                route_weights = _conditional_route_weight_vector(
                    gate_snapshots,
                    recent_segment_routes,
                )
                candidate_gate_predictions = (
                    route_weights * raw_gate_predictions + (1.0 - route_weights) * incumbent_gate_predictions
                )
            elif recent_segment_regime:
                segment_route = _recent_positive_window_segment_route(
                    gate_snapshots,
                    raw_gate_predictions,
                    incumbent_gate_predictions,
                    gate_start=str(split.manifest["gate_start"]),
                    primary_horizon=primary_horizon,
                    require_primary_top20_excess_non_degradation=require_route_side_guard,
                )
                if segment_route is None:
                    logger.info(
                        "ML promotion-gate skipped recent-segment-regime target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s: no positive "
                        "recent window/segment",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                    )
                    continue
                recent_regime_start, recent_segment_routes = segment_route
                route_weights = _conditional_route_weight_vector(
                    gate_snapshots,
                    recent_segment_routes,
                )
                candidate_gate_predictions = (
                    route_weights * raw_gate_predictions + (1.0 - route_weights) * incumbent_gate_predictions
                )
            elif recent_market_regime:
                recent_route = _recent_positive_window_market_route(
                    gate_snapshots,
                    raw_gate_predictions,
                    incumbent_gate_predictions,
                    gate_start=str(split.manifest["gate_start"]),
                    primary_horizon=primary_horizon,
                    require_primary_top20_excess_non_degradation=require_route_side_guard,
                )
                if recent_route is None:
                    logger.info(
                        "ML promotion-gate skipped recent-market-regime target=%s "
                        "lambda=%.4g backend=%s model=%s features=%s: no positive "
                        "recent window/market",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                    )
                    continue
                recent_regime_start, recent_market_weights = recent_route
                gate_dates = _snapshot_dates(gate_snapshots)
                recent_mask = (gate_dates >= pd.Timestamp(recent_regime_start)).to_numpy(dtype=bool)
                market_gate_weights = _market_blend_weight_vector(
                    gate_snapshots,
                    recent_market_weights,
                )
                route_weights = np.where(recent_mask, market_gate_weights, 0.0)
                candidate_gate_predictions = (
                    route_weights * raw_gate_predictions + (1.0 - route_weights) * incumbent_gate_predictions
                )
            elif recent_regime:
                raw_regime_diagnostics = _regime_diagnostics(
                    gate_snapshots,
                    raw_gate_predictions,
                    incumbent_gate_predictions,
                    primary_horizon=primary_horizon,
                )
                recent_regime_start = _recent_positive_year_start(
                    raw_regime_diagnostics,
                    gate_start=str(split.manifest["gate_start"]),
                    primary_horizon=primary_horizon,
                )
                if recent_regime_start is None:
                    logger.info(
                        "ML promotion-gate skipped recent-regime target=%s lambda=%.4g "
                        "backend=%s model=%s features=%s: no positive recent year",
                        full_candidate["target"],
                        float(full_candidate["ridge_lambda"]),
                        full_candidate["backend"],
                        full_candidate["model_kind"],
                        full_candidate["feature_set"],
                    )
                    continue
                gate_dates = _snapshot_dates(gate_snapshots)
                recent_mask = (gate_dates >= pd.Timestamp(recent_regime_start)).to_numpy(dtype=bool)
                candidate_gate_predictions = np.where(
                    recent_mask,
                    raw_gate_predictions,
                    incumbent_gate_predictions,
                )
            elif anchor_payload_option is not None:
                if anchor_path not in anchor_gate_predictions:
                    anchor_gate_predictions[anchor_path] = predict_ml_ranker_payload(
                        frame_for_payload(anchor_payload_option, train_frame=False),
                        anchor_payload_option,
                    )
                candidate_gate_predictions = (
                    float(blend_weight) * raw_gate_predictions
                    + (1.0 - float(blend_weight)) * anchor_gate_predictions[anchor_path]
                )
            elif market_blend_weights is not None:
                market_gate_weights = _market_blend_weight_vector(
                    gate_snapshots,
                    market_blend_weights,
                )
                candidate_gate_predictions = (
                    market_gate_weights * raw_gate_predictions
                    + (1.0 - market_gate_weights) * incumbent_gate_predictions
                )
            elif blend_weight == 1.0:
                candidate_gate_predictions = raw_gate_predictions
            else:
                candidate_gate_predictions = (
                    float(blend_weight) * raw_gate_predictions
                    + (1.0 - float(blend_weight)) * incumbent_gate_predictions
                )
            candidate_gate_metrics = _evaluate_predictions(gate_snapshots, candidate_gate_predictions)
            if (
                blend_weight == 1.0
                and market_blend_weights is None
                and anchor_payload_option is None
                and not recent_regime
                and not recent_market_regime
                and not recent_segment_regime
                and not segment_regime
            ):
                raw_gate_primary_rho = float(candidate_gate_metrics[primary_horizon]["spearman_rho"])
            gate_result = evaluate_promotion_gate(
                candidate_metrics=candidate_gate_metrics,
                incumbent_metrics=incumbent_gate_metrics,
                manifest=split.manifest,
                config=gate_config,
                regime_diagnostics=None,
                primary_horizon=primary_horizon,
            )
            regime_diagnostics: Mapping[str, object] = {}
            if gate_result.accepted:
                incumbent_gate_predictions = ensure_incumbent_gate_predictions()
                regime_diagnostics = _regime_diagnostics(
                    gate_snapshots,
                    candidate_gate_predictions,
                    incumbent_gate_predictions,
                    primary_horizon=primary_horizon,
                )
                gate_result = evaluate_promotion_gate(
                    candidate_metrics=candidate_gate_metrics,
                    incumbent_metrics=incumbent_gate_metrics,
                    manifest=split.manifest,
                    config=gate_config,
                    regime_diagnostics=regime_diagnostics,
                    primary_horizon=primary_horizon,
                )
            attempt = {
                "attempt": attempt_index,
                "accepted": gate_result.accepted,
                "reason": gate_result.reason,
                "target": full_candidate["target"],
                "target_horizon": primary_horizon,
                "ridge_lambda": full_candidate["ridge_lambda"],
                "backend": full_candidate["backend"],
                "model_kind": full_candidate["model_kind"],
                "feature_set": full_candidate["feature_set"],
                "blend_weight": float(blend_weight),
                "market_blend_weights": dict(market_blend_weights or {}),
                "anchor_model_path": anchor_path,
                "recent_regime_start": recent_regime_start or "",
                "recent_market_weights": dict(recent_market_weights or {}),
                "recent_segment_routes": recent_segment_routes,
                "prior_strategy": full_candidate["prior_strategy"],
                "sample_rho": float(full_candidate["sample_metrics"][primary_horizon]["spearman_rho"]),
                "train_rho": float(
                    payload_train_rho
                    if payload_train_rho is not None
                    else full_candidate["metrics"][primary_horizon]["spearman_rho"]
                ),
                "gate_rho": float(candidate_gate_metrics[primary_horizon]["spearman_rho"]),
                "gate_delta": float(gate_result.deltas[primary_horizon]),
                "gate_utility": float(gate_result.weighted_utility),
            }
            gate_attempts.append(attempt)
            logger.info(
                "ML promotion-gate attempt=%d target=%s lambda=%.4g backend=%s "
                "model=%s features=%s blend=%.3g market_blend=%s recent_start=%s "
                "recent_market=%s recent_segment=%s priors=%s train_%s_rho=%.6f "
                "anchor=%s gate_%s_rho=%.6f "
                "delta=%+.6f utility=%+.6f accepted=%s reason=%s",
                attempt_index,
                full_candidate["target"],
                float(full_candidate["ridge_lambda"]),
                full_candidate["backend"],
                full_candidate["model_kind"],
                full_candidate["feature_set"],
                float(blend_weight),
                _market_blend_label(market_blend_weights),
                recent_regime_start or "none",
                _market_blend_label(recent_market_weights),
                _segment_route_label(recent_segment_routes),
                full_candidate["prior_strategy"],
                primary_horizon,
                float(attempt["train_rho"]),
                anchor_path or "none",
                primary_horizon,
                float(candidate_gate_metrics[primary_horizon]["spearman_rho"]),
                float(gate_result.deltas[primary_horizon]),
                float(gate_result.weighted_utility),
                gate_result.accepted,
                gate_result.reason,
            )
            append_gate_ledger(
                gate_result,
                extra={
                    "candidate": "train_ml_ranker",
                    "target_horizon": primary_horizon,
                    "output_model_path": str(output_model_path),
                    "incumbent_type": incumbent_type,
                    "gate_attempt": attempt_index,
                    "backend": full_candidate["backend"],
                    "model_kind": full_candidate["model_kind"],
                    "feature_set": full_candidate["feature_set"],
                    "blend_weight": float(blend_weight),
                    "market_blend_weights": dict(market_blend_weights or {}),
                    "anchor_model_path": anchor_path,
                    "recent_regime_start": recent_regime_start or "",
                    "recent_market_weights": dict(recent_market_weights or {}),
                    "recent_segment_routes": recent_segment_routes,
                    "prior_strategy": full_candidate["prior_strategy"],
                    "target": full_candidate["target"],
                    "train_rho": attempt["train_rho"],
                    "ridge_lambda": full_candidate["ridge_lambda"],
                },
            )
            if gate_result.accepted:
                if _is_better_gate_result(
                    gate_result,
                    best_accepted_gate,
                    primary_horizon=primary_horizon,
                ):
                    accepted_payload = payload
                    if segment_regime and recent_segment_routes:
                        train_metrics = full_candidate["metrics"]
                        accepted_payload = _conditional_blend_payload(
                            anchor_payload=incumbent_eval_payload,
                            candidate_payload=raw_payload,
                            routes=recent_segment_routes,
                        )
                        accepted_payload["metadata"].update(
                            {
                                "backend": "gate_segment_blend",
                                "model_kind": "gate_segment_blend",
                                "feature_set": "gate_segment",
                                "prior_strategy": "gate_segment",
                                "train_metrics": train_metrics,
                                "train_rho_improvement": _improvement_ratio(
                                    float(train_metrics[primary_horizon]["spearman_rho"]),
                                    train_baseline_rho,
                                ),
                            }
                        )
                    elif recent_segment_regime and recent_regime_start is not None:
                        train_metrics = full_candidate["metrics"]
                        accepted_payload = _recent_segment_regime_temporal_payload(
                            candidate_payload=raw_payload,
                            incumbent_payload=incumbent_eval_payload,
                            gate_start=str(split.manifest["gate_start"]),
                            recent_start=recent_regime_start,
                            routes=recent_segment_routes,
                        )
                        accepted_payload["metadata"].update(
                            {
                                "train_metrics": train_metrics,
                                "train_rho_improvement": _improvement_ratio(
                                    float(train_metrics[primary_horizon]["spearman_rho"]),
                                    train_baseline_rho,
                                ),
                            }
                        )
                    elif recent_market_regime and recent_regime_start is not None:
                        train_metrics = full_candidate["metrics"]
                        accepted_payload = _recent_market_regime_temporal_payload(
                            candidate_payload=raw_payload,
                            incumbent_payload=incumbent_eval_payload,
                            gate_start=str(split.manifest["gate_start"]),
                            recent_start=recent_regime_start,
                            candidate_weights_by_market=recent_market_weights or {},
                        )
                        accepted_payload["metadata"].update(
                            {
                                "train_metrics": train_metrics,
                                "train_rho_improvement": _improvement_ratio(
                                    float(train_metrics[primary_horizon]["spearman_rho"]),
                                    train_baseline_rho,
                                ),
                            }
                        )
                    elif recent_regime and recent_regime_start is not None:
                        train_metrics = full_candidate["metrics"]
                        accepted_payload = _recent_regime_temporal_payload(
                            candidate_payload=raw_payload,
                            incumbent_payload=incumbent_eval_payload,
                            gate_start=str(split.manifest["gate_start"]),
                            recent_start=recent_regime_start,
                        )
                        accepted_payload["metadata"].update(
                            {
                                "train_metrics": train_metrics,
                                "train_rho_improvement": _improvement_ratio(
                                    float(train_metrics[primary_horizon]["spearman_rho"]),
                                    train_baseline_rho,
                                ),
                            }
                        )
                    elif anchor_payload_option is not None:
                        if train_predictions_for_payload is None:
                            raise RuntimeError("ML ranker missing anchor-blended train predictions")
                        train_metrics = _evaluate_predictions(
                            train_snapshots,
                            train_predictions_for_payload,
                        )
                        accepted_payload = _payload_member_blend_payload(
                            anchor_payload=anchor_payload_option,
                            candidate_payload=raw_payload,
                            candidate_weight=blend_weight,
                        )
                        accepted_payload["metadata"].update(
                            {
                                "train_metrics": train_metrics,
                                "train_rho_improvement": _improvement_ratio(
                                    float(train_metrics[primary_horizon]["spearman_rho"]),
                                    train_baseline_rho,
                                ),
                            }
                        )
                    if accepted_payload is None:
                        if train_predictions_for_payload is None:
                            if raw_train_predictions is None:
                                raw_train_predictions = predict_ml_ranker_payload(
                                    frame_for_payload(raw_payload, train_frame=True),
                                    raw_payload,
                                )
                            if market_blend_weights is not None:
                                if incumbent_train_predictions is None:
                                    incumbent_train_predictions = predict_ml_ranker_payload(
                                        frame_for_payload(
                                            incumbent_eval_payload,
                                            train_frame=True,
                                        ),
                                        incumbent_eval_payload,
                                    )
                                market_train_weights = _market_blend_weight_vector(
                                    train_snapshots,
                                    market_blend_weights,
                                )
                                train_predictions_for_payload = (
                                    market_train_weights * raw_train_predictions
                                    + (1.0 - market_train_weights) * incumbent_train_predictions
                                )
                            elif blend_weight == 1.0:
                                train_predictions_for_payload = raw_train_predictions
                            else:
                                if incumbent_train_predictions is None:
                                    incumbent_train_predictions = predict_ml_ranker_payload(
                                        frame_for_payload(
                                            incumbent_eval_payload,
                                            train_frame=True,
                                        ),
                                        incumbent_eval_payload,
                                    )
                                train_predictions_for_payload = (
                                    float(blend_weight) * raw_train_predictions
                                    + (1.0 - float(blend_weight)) * incumbent_train_predictions
                                )
                        train_metrics = _evaluate_predictions(
                            train_snapshots,
                            train_predictions_for_payload,
                        )
                        accepted_payload = build_payload(
                            full_candidate,
                            blend_weight_override=blend_weight,
                            market_blend_weights_override=market_blend_weights,
                            train_metrics_override=train_metrics,
                        )
                    best_accepted_payload = accepted_payload
                    best_accepted_gate = gate_result
                    best_accepted_attempt = attempt
                    if promotion_min_gate_rho is not None and float(
                        candidate_gate_metrics[primary_horizon]["spearman_rho"]
                    ) >= float(promotion_min_gate_rho):
                        stop_after_gate_acceptance = True
                if recent_regime:
                    break
                if stop_after_gate_acceptance:
                    break
                continue

            if best_rejected_gate is None or gate_result.deltas[primary_horizon] > float(
                best_rejected_gate.deltas[primary_horizon]
            ):
                best_rejected = attempt
                best_rejected_gate = gate_result

        if stop_after_gate_acceptance:
            break

    if best_accepted_payload is not None and best_accepted_gate is not None and best_accepted_attempt is not None:
        candidate_gate_metrics = best_accepted_gate.candidate_metrics
        metadata = best_accepted_payload["metadata"]
        if not isinstance(metadata, dict):
            raise RuntimeError("ML ranker payload missing metadata")
        metadata.update(
            {
                "baseline_metrics": incumbent_gate_metrics,
                "metrics": candidate_gate_metrics,
                "promotion_gate": best_accepted_gate.to_record(),
                "promotion_gate_manifest": split.manifest,
                "promotion_incumbent_type": incumbent_type,
                "promotion_gate_attempts": gate_attempts,
                "promotion_gate_selected_attempt": best_accepted_attempt,
                "final_rho_improvement": _improvement_ratio(
                    float(candidate_gate_metrics[primary_horizon]["spearman_rho"]),
                    float(incumbent_gate_metrics[primary_horizon]["spearman_rho"]),
                ),
            }
        )

        best_accepted_payload = _compact_runtime_payload(best_accepted_payload)
        _write_json_atomic(output_model_path, best_accepted_payload)
        logger.info(
            "Saved ML ranker model -> %s (selected gate attempt %s)",
            output_model_path,
            best_accepted_attempt["attempt"],
        )

        return metadata

    if best_rejected is None or best_rejected_gate is None:
        if skipped_gate_attempts and incumbent_train_rho is not None:
            best_skipped = max(
                skipped_gate_attempts,
                key=lambda attempt: float(attempt["train_rho"]),
            )
            if promotion_min_train_rho is not None:
                raise RuntimeError(
                    "ML ranker skipped promotion gate because train/full-eval rho "
                    "did not meet requested floor: "
                    f"best {primary_horizon} rho="
                    f"{float(best_skipped['train_rho']):.6f}, "
                    f"required {primary_horizon} rho="
                    f"{float(promotion_min_train_rho):.6f}, "
                    f"blend={best_skipped.get('blend_weight')}"
                )
            raise RuntimeError(
                "ML ranker skipped promotion gate because blended train/full-eval rho "
                f"did not beat incumbent: best {primary_horizon} rho="
                f"{float(best_skipped['train_rho']):.6f}, "
                f"incumbent {primary_horizon} rho={incumbent_train_rho:.6f}, "
                f"blend={best_skipped.get('blend_weight')}"
            )
        raise RuntimeError("Not enough valid target rows to train ML ranker")

    raise RuntimeError(
        "ML ranker failed untouched holdout promotion gate across "
        f"{len(gate_attempts)} candidate(s): {best_rejected_gate.reason}; "
        f"best Δ{primary_horizon}={best_rejected_gate.deltas[primary_horizon]:+.6f}, "
        f"utility={best_rejected_gate.weighted_utility:+.6f}, "
        f"lambda={best_rejected['ridge_lambda']}, "
        f"backend={best_rejected['backend']}, "
        f"features={best_rejected.get('feature_set')}, "
        f"blend={best_rejected.get('blend_weight')}, "
        f"priors={best_rejected['prior_strategy']}"
    )
