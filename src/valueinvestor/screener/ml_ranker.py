"""Optional trained ML ranker used by the production scorer.

The model is a compact ridge ranker trained by
``valueinvestor.scorer_improver.ml_trainer``.  It is deliberately artifact
driven: if the model file is absent or disabled, the hand-written scorer keeps
the existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/trainer/ml_ranker_model.json")
ONE_WEEK_MODEL_PATH = Path("data/trainer/ml_ranker_model_1w.json")
MODEL_SCHEMA_VERSION = "ml-ranker-v1"
# Runtime feature priors intentionally stay on the existing 1m/3m/6m schema so
# current 6m artifacts remain load-compatible while new target horizons can be
# trained against the same feature matrix.
HORIZONS = ("1m", "3m", "6m")

BASE_FIELDS = (
    "close",
    "pe_ratio",
    "pe_forward",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "dividend_yield",
    "ev_to_ebitda",
    "market_cap_rmb",
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cash_flow",
    "free_cash_flow",
    "gross_margin",
    "roe",
    "roa",
    "net_margin",
    "debt_to_equity",
    "current_ratio",
    "composite_score",
    "value_score",
    "quality_score",
    "growth_score",
)

INVERSE_FIELDS = {
    "close",
    "pe_ratio",
    "pe_forward",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "ev_to_ebitda",
    "market_cap_rmb",
    "debt_to_equity",
    "current_ratio",
}

RATIO_FEATURES = (
    "earnings_yield",
    "ocf_yield",
    "fcf_yield",
    "gross_profit_assets",
    "gross_profit_market_cap",
    "asset_turnover",
    "roa_calc",
    "roe_pb",
    "roe_ev",
    "pe_forward_improve",
    "pe_forward_decline",
    "liabilities_market_cap",
    "debt_assets",
    "fcf_equity",
    "ocf_assets",
    "price_pb",
    "price_market_cap",
    "peg_inv",
)

SHORT_HORIZON_FEATURES = (
    "price_return_5d",
    "price_return_21d",
    "price_return_63d",
    "price_volatility_21d",
    "price_volatility_63d",
    "market_return_5d",
    "market_return_21d",
    "market_return_63d",
    "market_volatility_21d",
    "market_volatility_63d",
    "relative_return_5d",
    "relative_return_21d",
    "relative_return_63d",
    "relative_volatility_21d",
    "relative_volatility_63d",
)

APPLICATION_FACTOR_SIGNALS = (
    "model",
    "quality_de_crowding",
    "book_yield",
    "sales_yield",
    "liability_yield",
    "momentum_63d",
    "low_current_ratio",
)

INTERACTION_FEATURES = (
    ("value_quality_score", "value_score", "quality_score"),
    ("value_growth_score", "value_score", "growth_score"),
    ("quality_growth_score", "quality_score", "growth_score"),
    ("roe_earnings_yield", "roe", "earnings_yield"),
    ("roe_fcf_yield", "roe", "fcf_yield"),
    ("relative_return_63d_value_score", "relative_return_63d", "value_score"),
    ("relative_return_21d_quality_score", "relative_return_21d", "quality_score"),
    ("price_return_63d_growth_score", "price_return_63d", "growth_score"),
    ("ticker_rankmean_6m_value_score", "ticker_rankmean_6m", "value_score"),
    ("ticker_rankmean_6m_relative_return_63d", "ticker_rankmean_6m", "relative_return_63d"),
)

POLYNOMIAL_FEATURES = (
    ("value_score_sq", "value_score"),
    ("quality_score_sq", "quality_score"),
    ("growth_score_sq", "growth_score"),
    ("composite_score_sq", "composite_score"),
    ("roe_sq", "roe"),
    ("earnings_yield_sq", "earnings_yield"),
    ("fcf_yield_sq", "fcf_yield"),
    ("gross_profit_assets_sq", "gross_profit_assets"),
    ("relative_return_21d_sq", "relative_return_21d"),
    ("relative_return_63d_sq", "relative_return_63d"),
    ("price_return_21d_sq", "price_return_21d"),
    ("price_return_63d_sq", "price_return_63d"),
    ("ticker_rankmean_3m_sq", "ticker_rankmean_3m"),
    ("ticker_rankmean_6m_sq", "ticker_rankmean_6m"),
)

CROSS_SECTIONAL_FEATURES = (
    "composite_score",
    "value_score",
    "quality_score",
    "growth_score",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "market_cap_rmb",
    "roe",
    "roa",
    "net_margin",
    "debt_to_equity",
    "earnings_yield",
    "fcf_yield",
    "gross_profit_assets",
    "asset_turnover",
    "ticker_rankmean_3m",
    "ticker_rankmean_6m",
    "relative_return_21d",
    "relative_return_63d",
    "price_return_63d",
)

CROSS_SECTIONAL_INTERACTION_FEATURES = (
    ("csx_value_ticker_rankmean_6m", "cs_rank_value_score", "cs_rank_ticker_rankmean_6m"),
    ("csx_quality_ticker_rankmean_6m", "cs_rank_quality_score", "cs_rank_ticker_rankmean_6m"),
    ("csx_roe_ticker_rankmean_6m", "cs_rank_roe", "cs_rank_ticker_rankmean_6m"),
    (
        "csx_relative_return_63d_ticker_rankmean_6m",
        "cs_rank_relative_return_63d",
        "cs_rank_ticker_rankmean_6m",
    ),
    (
        "market_csx_value_ticker_rankmean_6m",
        "market_cs_rank_value_score",
        "market_cs_rank_ticker_rankmean_6m",
    ),
    (
        "market_csx_quality_ticker_rankmean_6m",
        "market_cs_rank_quality_score",
        "market_cs_rank_ticker_rankmean_6m",
    ),
    (
        "market_csx_roe_ticker_rankmean_6m",
        "market_cs_rank_roe",
        "market_cs_rank_ticker_rankmean_6m",
    ),
    (
        "market_csx_relative_return_63d_ticker_rankmean_6m",
        "market_cs_rank_relative_return_63d",
        "market_cs_rank_ticker_rankmean_6m",
    ),
)


def cross_sectional_feature_names() -> List[str]:
    names: List[str] = []
    for source in CROSS_SECTIONAL_FEATURES:
        names.append(f"cs_rank_{source}")
        names.append(f"market_cs_rank_{source}")
    return names


def cross_sectional_interaction_feature_names() -> List[str]:
    return [name for name, _left, _right in CROSS_SECTIONAL_INTERACTION_FEATURES]


def expected_feature_names(
    *,
    include_short_horizon: bool = False,
    include_interactions: bool = False,
    include_polynomial: bool = False,
    include_cross_sectional: bool = False,
    include_cross_sectional_interactions: bool = False,
) -> List[str]:
    names: List[str] = []
    for field in BASE_FIELDS:
        names.append(field)
        names.append(f"{field}_miss")
        names.append(f"slog_{field}")
        if field in INVERSE_FIELDS:
            names.append(f"inv_{field}")
    for feature in RATIO_FEATURES:
        names.append(feature)
        names.append(f"slog_{feature}")
    for horizon in HORIZONS:
        names.append(f"ticker_mean_{horizon}")
        names.append(f"ticker_rank_{horizon}")
        names.append(f"ticker_rankmean_{horizon}")
    if include_short_horizon:
        for feature in SHORT_HORIZON_FEATURES:
            names.append(feature)
            names.append(f"{feature}_miss")
            names.append(f"slog_{feature}")
    if include_interactions:
        names.extend(name for name, _left, _right in INTERACTION_FEATURES)
    if include_polynomial:
        names.extend(name for name, _source in POLYNOMIAL_FEATURES)
    if include_cross_sectional:
        names.extend(cross_sectional_feature_names())
    if include_cross_sectional_interactions:
        names.extend(cross_sectional_interaction_feature_names())
    return names


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if not math.isnan(parsed) else float("nan")


def _slog(value: float) -> float:
    value = _finite_or_zero(value)
    return math.copysign(math.log1p(abs(value)), value) if value != 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _ratio_values(values: Mapping[str, float]) -> Dict[str, float]:
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

    gross_profit = (
        revenue * gross_margin
        if math.isfinite(revenue) and math.isfinite(gross_margin)
        else float("nan")
    )
    pe_forward_decline = (
        (pe_ratio - pe_forward) / pe_ratio
        if math.isfinite(pe_ratio)
        and math.isfinite(pe_forward)
        and pe_ratio != 0
        else float("nan")
    )

    return {
        "earnings_yield": _safe_div(net_income, market_cap),
        "ocf_yield": _safe_div(operating_cash_flow, market_cap),
        "fcf_yield": _safe_div(free_cash_flow, market_cap),
        "gross_profit_assets": _safe_div(gross_profit, total_assets),
        "gross_profit_market_cap": _safe_div(gross_profit, market_cap),
        "asset_turnover": _safe_div(revenue, total_assets),
        "roa_calc": _safe_div(net_income, total_assets),
        "roe_pb": _safe_div(roe, pb_ratio),
        "roe_ev": _safe_div(roe, ev_to_ebitda),
        "pe_forward_improve": _safe_div(pe_ratio, pe_forward),
        "pe_forward_decline": pe_forward_decline,
        "liabilities_market_cap": _safe_div(total_liabilities, market_cap),
        "debt_assets": _safe_div(total_liabilities, total_assets),
        "fcf_equity": _safe_div(free_cash_flow, total_equity),
        "ocf_assets": _safe_div(operating_cash_flow, total_assets),
        "price_pb": (
            close * pb_ratio if math.isfinite(close) and math.isfinite(pb_ratio) else float("nan")
        ),
        "price_market_cap": (
            close * market_cap
            if math.isfinite(close) and math.isfinite(market_cap)
            else float("nan")
        ),
        "peg_inv": 1.0 / peg_ratio if math.isfinite(peg_ratio) and peg_ratio > 0 else float("nan"),
    }


def feature_vector_from_values(
    values: Mapping[str, object],
    ticker: str,
    ticker_priors: Mapping[str, Mapping[str, float]],
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> List[float]:
    parsed = {field: _to_float(values.get(field)) for field in BASE_FIELDS}
    features: Dict[str, float] = {}

    for field in BASE_FIELDS:
        raw = parsed[field]
        features[field] = _finite_or_zero(raw)
        features[f"{field}_miss"] = 1.0 if math.isnan(raw) else 0.0
        features[f"slog_{field}"] = _slog(raw)
        if field in INVERSE_FIELDS:
            features[f"inv_{field}"] = 1.0 / raw if math.isfinite(raw) and raw > 0 else 0.0

    ratios = _ratio_values(parsed)
    for feature in RATIO_FEATURES:
        raw = ratios[feature]
        features[feature] = _finite_or_zero(raw)
        features[f"slog_{feature}"] = _slog(raw)

    prior = ticker_priors.get(ticker, {})
    for horizon in HORIZONS:
        features[f"ticker_mean_{horizon}"] = float(prior.get(f"ticker_mean_{horizon}", 0.0))
        features[f"ticker_rank_{horizon}"] = float(prior.get(f"ticker_rank_{horizon}", 0.0))
        features[f"ticker_rankmean_{horizon}"] = float(prior.get(f"ticker_rankmean_{horizon}", 0.0))

    for feature in SHORT_HORIZON_FEATURES:
        raw = _to_float(values.get(feature))
        features[feature] = _finite_or_zero(raw)
        features[f"{feature}_miss"] = 1.0 if math.isnan(raw) else 0.0
        features[f"slog_{feature}"] = _slog(raw)

    for name, left, right in INTERACTION_FEATURES:
        features[name] = _finite_or_zero(features.get(left, 0.0)) * _finite_or_zero(
            features.get(right, 0.0)
        )
    for name, source in POLYNOMIAL_FEATURES:
        value = _finite_or_zero(features.get(source, 0.0))
        features[name] = value * value

    selected_names = list(feature_names) if feature_names is not None else expected_feature_names()
    return [features.get(name, 0.0) for name in selected_names]


def values_from_result(result: ScreeningResult) -> Dict[str, object]:
    return {
        "close": result.valuation.price,
        "pe_ratio": result.valuation.pe_ratio,
        "pe_forward": result.valuation.pe_forward,
        "pb_ratio": result.valuation.pb_ratio,
        "ps_ratio": result.valuation.ps_ratio,
        "peg_ratio": result.valuation.peg_ratio,
        "dividend_yield": result.valuation.dividend_yield,
        "ev_to_ebitda": result.valuation.ev_to_ebitda,
        "market_cap_rmb": result.valuation.market_cap_rmb,
        "revenue": result.financials.revenue,
        "net_income": result.financials.net_income,
        "total_assets": result.financials.total_assets,
        "total_liabilities": result.financials.total_liabilities,
        "total_equity": result.financials.total_equity,
        "operating_cash_flow": result.financials.operating_cash_flow,
        "free_cash_flow": result.financials.free_cash_flow,
        "gross_margin": result.financials.gross_margin,
        "roe": result.financials.roe,
        "roa": result.financials.roa,
        "net_margin": result.financials.net_margin,
        "debt_to_equity": result.financials.debt_to_equity,
        "current_ratio": result.financials.current_ratio,
        "composite_score": result.composite_score,
        "value_score": result.value_score,
        "quality_score": result.quality_score,
        "growth_score": result.growth_score,
    }


@dataclass(frozen=True)
class MLRankerLinearModel:
    clip_low: np.ndarray
    clip_high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    weight: float = 1.0
    market: Optional[str] = None
    cap_bucket: Optional[str] = None
    board_bucket: Optional[str] = None
    listing_bucket: Optional[str] = None

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        clipped = np.clip(features, self.clip_low, self.clip_high)
        scaled = (clipped - self.mean) / self.scale
        return scaled @ self.coef + self.intercept


@dataclass(frozen=True)
class MLRankerModel:
    feature_names: Sequence[str]
    linear_models: Sequence[MLRankerLinearModel]
    ticker_priors: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, object]
    lightgbm_model_text: Optional[str] = None
    rank_member_models: Sequence[tuple[float, "MLRankerModel"]] = ()
    member_models: Sequence[tuple[float, "MLRankerModel"]] = ()
    market_member_models: Sequence[tuple[Mapping[str, float], "MLRankerModel"]] = ()
    temporal_member_models: Sequence[tuple[Optional[date], Optional[date], "MLRankerModel"]] = ()
    conditional_base_model: Optional["MLRankerModel"] = None
    conditional_candidate_model: Optional["MLRankerModel"] = None
    conditional_routes: Sequence[Mapping[str, object]] = ()

    @property
    def coef(self) -> np.ndarray:
        return self.linear_models[0].coef

    @property
    def intercept(self) -> float:
        return self.linear_models[0].intercept

    def predict_matrix(
        self,
        features: np.ndarray,
        tickers: Optional[Sequence[str]] = None,
        cap_buckets: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.lightgbm_model_text is not None:
            return predict_lightgbm_model(self.lightgbm_model_text, features)
        if len(self.linear_models) == 1:
            return self.linear_models[0].predict_matrix(features)
        predictions = np.zeros(features.shape[0], dtype="float64")
        unweighted_predictions = np.zeros(features.shape[0], dtype="float64")
        total_weights = np.zeros(features.shape[0], dtype="float64")
        prediction_counts = np.zeros(features.shape[0], dtype="float64")
        markets = [_market_bucket(ticker) for ticker in tickers] if tickers is not None else None
        board_buckets = [_board_bucket(ticker) for ticker in tickers] if tickers is not None else None
        listing_buckets = [_listing_bucket(ticker) for ticker in tickers] if tickers is not None else None
        for linear_model in self.linear_models:
            if linear_model.market and markets is not None:
                mask = np.asarray(
                    [market == linear_model.market for market in markets],
                    dtype=bool,
                )
            else:
                mask = np.ones(features.shape[0], dtype=bool)
            if linear_model.cap_bucket and cap_buckets is not None:
                mask &= np.asarray(
                    [cap_bucket == linear_model.cap_bucket for cap_bucket in cap_buckets],
                    dtype=bool,
                )
            if linear_model.board_bucket and board_buckets is not None:
                mask &= np.asarray(
                    [board_bucket == linear_model.board_bucket for board_bucket in board_buckets],
                    dtype=bool,
                )
            if linear_model.listing_bucket and listing_buckets is not None:
                mask &= np.asarray(
                    [
                        listing_bucket == linear_model.listing_bucket
                        for listing_bucket in listing_buckets
                    ],
                    dtype=bool,
                )
            if not mask.any():
                continue
            weight = float(linear_model.weight)
            model_predictions = linear_model.predict_matrix(features[mask])
            predictions[mask] += weight * model_predictions
            unweighted_predictions[mask] += model_predictions
            total_weights[mask] += weight
            prediction_counts[mask] += 1.0
        fallback = prediction_counts > 0
        output = np.zeros(features.shape[0], dtype="float64")
        weighted = total_weights > 0
        output[weighted] = predictions[weighted] / total_weights[weighted]
        unweighted = ~weighted & fallback
        output[unweighted] = unweighted_predictions[unweighted] / prediction_counts[unweighted]
        if (~fallback).any():
            output[~fallback] = np.mean(
                [model.predict_matrix(features[~fallback]) for model in self.linear_models],
                axis=0,
            )
        return output


_MODEL_CACHE: Dict[str, tuple[tuple[int, int] | None, Optional[MLRankerModel]]] = {}
_PRICE_FEATURE_CACHE: Dict[tuple[tuple[str, int, int], ...], object] = {}


@lru_cache(maxsize=4)
def _load_lightgbm_booster(model_text: str) -> object:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency validation covers this
        raise RuntimeError("LightGBM model requires the lightgbm package") from exc
    return lgb.Booster(model_str=model_text)


def predict_lightgbm_model(model_text: str, features: np.ndarray) -> np.ndarray:
    booster = _load_lightgbm_booster(model_text)
    return np.asarray(booster.predict(features), dtype="float64")


def _parse_temporal_date(value: object) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def _valid_feature_schemas() -> tuple[List[str], ...]:
    return (
        expected_feature_names(),
        expected_feature_names(include_short_horizon=True),
        expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
        ),
        expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
        ),
        expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
            include_cross_sectional=True,
        ),
        expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
            include_cross_sectional=True,
            include_cross_sectional_interactions=True,
        ),
    )


def _centered_rank(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape[0], dtype="float64")
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n <= 1:
        return output
    finite_values = values[finite]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], n]
    sorted_ranks = np.empty(n, dtype="float64")
    for start, end in zip(group_starts, group_ends):
        sorted_ranks[start:end] = (float(start) + float(end - 1)) / 2.0
    ranks = np.empty(n, dtype="float64")
    ranks[order] = sorted_ranks
    output[finite] = ranks / float(n - 1) - 0.5
    return output


def _application_snapshot_rank(values: np.ndarray) -> np.ndarray:
    """Match the trainer's one-based snapshot rank used for application blends."""
    output = np.full(values.shape[0], np.nan, dtype="float64")
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n <= 1:
        return output
    finite_values = values[finite]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], n]
    sorted_ranks = np.empty(n, dtype="float64")
    for start, end in zip(group_starts, group_ends):
        sorted_ranks[start:end] = (float(start + 1) + float(end)) / 2.0
    ranks = np.empty(n, dtype="float64")
    ranks[order] = sorted_ranks
    output[finite] = ranks / float(n + 1) * 2.0 - 1.0
    return output


def _application_factor_config(
    metadata: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
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
        if signal in APPLICATION_FACTOR_SIGNALS and np.isfinite(weight) and weight > 0.0:
            components.append({"signal": signal, "weight": weight})
    return tuple(components)


def _application_factor_value(values: Mapping[str, object], signal: str) -> float:
    if signal == "quality_de_crowding":
        return -_to_float(values.get("quality_score"))
    if signal == "book_yield":
        return -_to_float(values.get("pb_ratio"))
    if signal == "sales_yield":
        return -_to_float(values.get("ps_ratio"))
    if signal == "liability_yield":
        return _safe_div(
            _to_float(values.get("total_liabilities")),
            _to_float(values.get("market_cap_rmb")),
        )
    if signal == "momentum_63d":
        return _to_float(values.get("relative_return_63d"))
    if signal == "low_current_ratio":
        return -_to_float(values.get("current_ratio"))
    return float("nan")


def _application_factor_predictions(
    model_predictions: np.ndarray,
    values_by_result: Sequence[Mapping[str, object]],
    components: Sequence[Mapping[str, object]],
) -> np.ndarray:
    model_rank = _application_snapshot_rank(model_predictions)
    blended = np.zeros(len(model_predictions), dtype="float64")
    total_weight = 0.0
    for component in components:
        signal = str(component.get("signal", "")).strip()
        try:
            weight = float(component.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if signal not in APPLICATION_FACTOR_SIGNALS or not np.isfinite(weight) or weight <= 0.0:
            continue
        if signal == "model":
            factor_rank = model_rank
        else:
            factor_values = np.asarray(
                [_application_factor_value(values, signal) for values in values_by_result],
                dtype="float64",
            )
            factor_rank = _application_snapshot_rank(factor_values)
            factor_rank = np.where(np.isfinite(factor_rank), factor_rank, 0.0)
        blended += weight * factor_rank
        total_weight += weight
    if total_weight <= 0.0:
        return model_predictions
    return blended / total_weight


def _application_residual_config(
    metadata: Mapping[str, object],
) -> tuple[float, str, float]:
    if "application_residual_candidate_weight" not in metadata:
        raw_weight = metadata.get(
            "application_quality_residual_candidate_weight",
            1.0,
        )
        column = "quality_score"
        direction = -1.0
    else:
        raw_weight = metadata.get("application_residual_candidate_weight", 1.0)
        column = str(metadata.get("application_residual_column", "")).strip()
        raw_direction = metadata.get("application_residual_direction", 1.0)
        try:
            direction = float(raw_direction)
        except (TypeError, ValueError):
            return 1.0, "", 1.0
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        return 1.0, "", 1.0
    if not np.isfinite(weight) or not np.isfinite(direction) or not column:
        return 1.0, "", 1.0
    return float(np.clip(weight, 0.0, 1.0)), column, direction


def _application_gross_profitability_de_crowding_config(
    metadata: Mapping[str, object],
) -> tuple[float, float, float]:
    try:
        factor_weight = float(
            metadata.get(
                "application_gross_profitability_de_crowding_factor_weight",
                0.0,
            )
        )
        max_model_rank = float(
            metadata.get(
                "application_gross_profitability_de_crowding_max_model_rank",
                0.4,
            )
        )
        min_market_return = float(
            metadata.get(
                "application_gross_profitability_de_crowding_min_market_return_63d",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return 0.0, 0.4, 0.0
    if not all(np.isfinite(value) for value in (factor_weight, max_model_rank, min_market_return)):
        return 0.0, 0.4, 0.0
    return (
        float(np.clip(factor_weight, 0.0, 1.0)),
        float(np.clip(max_model_rank, -1.0, 1.0)),
        min_market_return,
    )


def _application_gross_profitability_de_crowding_predictions(
    model_predictions: np.ndarray,
    gross_profit_assets: np.ndarray,
    market_return_63d: np.ndarray,
    *,
    factor_weight: float,
    max_model_rank: float,
    min_market_return: float,
) -> np.ndarray:
    """Re-rank lower-confidence stocks when the 63-day market regime is positive."""
    model_rank = _application_snapshot_rank(np.asarray(model_predictions, dtype="float64"))
    market_returns = np.asarray(market_return_63d, dtype="float64")
    finite_market = market_returns[np.isfinite(market_returns)]
    if not len(finite_market) or float(finite_market.mean()) <= min_market_return:
        return model_rank

    factor_rank = _application_snapshot_rank(
        np.asarray(gross_profit_assets, dtype="float64")
    )
    blended = model_rank.copy()
    usable = (
        np.isfinite(model_rank)
        & np.isfinite(factor_rank)
        & (model_rank <= max_model_rank)
    )
    if usable.any():
        blended[usable] = (
            (1.0 - factor_weight) * model_rank[usable]
            - factor_weight * factor_rank[usable]
        )
        blended[usable] = np.minimum(
            blended[usable],
            np.nextafter(max_model_rank, float("-inf")),
        )
    return blended


def _cross_sectional_feature_matrix(
    rows_by_feature: Mapping[str, np.ndarray],
    tickers: Sequence[str],
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    needed = set(feature_names)
    if not (
        any(name.startswith(("cs_rank_", "market_cs_rank_")) for name in needed)
        or any(name in needed for name, _left, _right in CROSS_SECTIONAL_INTERACTION_FEATURES)
    ):
        return {}

    markets = np.asarray([_market_bucket(ticker) for ticker in tickers], dtype=object)
    output: dict[str, np.ndarray] = {}
    for source in CROSS_SECTIONAL_FEATURES:
        values = rows_by_feature.get(source)
        if values is None:
            continue
        cs_name = f"cs_rank_{source}"
        if cs_name in needed:
            output[cs_name] = _centered_rank(np.asarray(values, dtype="float64"))

        market_name = f"market_cs_rank_{source}"
        if market_name in needed:
            ranked = np.zeros(len(tickers), dtype="float64")
            for market in dict.fromkeys(markets.tolist()):
                mask = markets == market
                ranked[mask] = _centered_rank(np.asarray(values, dtype="float64")[mask])
            output[market_name] = ranked
    for feature_name, left, right in CROSS_SECTIONAL_INTERACTION_FEATURES:
        if feature_name not in needed:
            continue
        left_values = output.get(left)
        right_values = output.get(right)
        if left_values is not None and right_values is not None:
            output[feature_name] = left_values * right_values
    return output


def _model_from_payload(payload: Mapping[str, object]) -> MLRankerModel:
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported model schema {payload.get('schema_version')!r}")

    conditional_blend = payload.get("conditional_blend")
    if isinstance(conditional_blend, Mapping):
        base_payload = conditional_blend.get("base_payload")
        candidate_payload = conditional_blend.get("candidate_payload")
        raw_routes = conditional_blend.get("routes")
        if (
            not isinstance(base_payload, Mapping)
            or not isinstance(candidate_payload, Mapping)
            or not isinstance(raw_routes, list)
        ):
            raise ValueError("conditional blend payload is malformed")
        routes = tuple(route for route in raw_routes if isinstance(route, Mapping))
        if not routes:
            raise ValueError("conditional blend payload contains no valid routes")
        return MLRankerModel(
            feature_names=(),
            linear_models=(),
            ticker_priors={},
            metadata=payload.get("metadata", {}),
            conditional_base_model=_model_from_payload(base_payload),
            conditional_candidate_model=_model_from_payload(candidate_payload),
            conditional_routes=routes,
        )

    temporal_payloads = payload.get("temporal_payload_members")
    if isinstance(temporal_payloads, list) and temporal_payloads:
        temporal_models: list[tuple[Optional[date], Optional[date], MLRankerModel]] = []
        for member in temporal_payloads:
            if not isinstance(member, Mapping):
                continue
            child_payload = member.get("payload")
            if not isinstance(child_payload, Mapping):
                continue
            temporal_models.append((
                _parse_temporal_date(member.get("start_date")),
                _parse_temporal_date(member.get("end_date")),
                _model_from_payload(child_payload),
            ))
        if not temporal_models:
            raise ValueError("temporal payload contains no valid members")
        return MLRankerModel(
            feature_names=(),
            linear_models=(),
            ticker_priors={},
            metadata=payload.get("metadata", {}),
            temporal_member_models=tuple(temporal_models),
        )

    rank_member_payloads = payload.get("rank_payload_members")
    if isinstance(rank_member_payloads, list) and rank_member_payloads:
        rank_member_models: list[tuple[float, MLRankerModel]] = []
        for member in rank_member_payloads:
            if not isinstance(member, Mapping):
                continue
            child_payload = member.get("payload")
            if not isinstance(child_payload, Mapping):
                continue
            try:
                weight = float(member.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            rank_member_models.append((weight, _model_from_payload(child_payload)))
        if not rank_member_models:
            raise ValueError("rank payload-member ensemble contains no valid members")
        return MLRankerModel(
            feature_names=(),
            linear_models=(),
            ticker_priors={},
            metadata=payload.get("metadata", {}),
            rank_member_models=tuple(rank_member_models),
        )

    member_payloads = payload.get("payload_members")
    if isinstance(member_payloads, list) and member_payloads:
        member_models: list[tuple[float, MLRankerModel]] = []
        for member in member_payloads:
            if not isinstance(member, Mapping):
                continue
            child_payload = member.get("payload")
            if not isinstance(child_payload, Mapping):
                continue
            try:
                weight = float(member.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            member_models.append((weight, _model_from_payload(child_payload)))
        if not member_models:
            raise ValueError("payload-member ensemble contains no valid members")
        return MLRankerModel(
            feature_names=(),
            linear_models=(),
            ticker_priors={},
            metadata=payload.get("metadata", {}),
            member_models=tuple(member_models),
        )

    market_payloads = payload.get("market_payload_members")
    if isinstance(market_payloads, list) and market_payloads:
        market_models: list[tuple[Mapping[str, float], MLRankerModel]] = []
        for member in market_payloads:
            if not isinstance(member, Mapping):
                continue
            child_payload = member.get("payload")
            raw_weights = member.get("market_weights")
            if not isinstance(child_payload, Mapping) or not isinstance(raw_weights, Mapping):
                continue
            weights: dict[str, float] = {}
            for market, weight in raw_weights.items():
                try:
                    value = float(weight)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    weights[str(market)] = value
            if weights:
                market_models.append((weights, _model_from_payload(child_payload)))
        if not market_models:
            raise ValueError("market payload-member ensemble contains no valid members")
        return MLRankerModel(
            feature_names=(),
            linear_models=(),
            ticker_priors={},
            metadata=payload.get("metadata", {}),
            market_member_models=tuple(market_models),
        )

    feature_names = payload["feature_names"]
    if list(feature_names) not in _valid_feature_schemas():
        raise ValueError("model feature schema does not match runtime feature schema")
    expected_len = len(feature_names)
    lightgbm_model_text = payload.get("lightgbm_model")
    if isinstance(lightgbm_model_text, str) and lightgbm_model_text:
        return MLRankerModel(
            feature_names=feature_names,
            linear_models=(),
            ticker_priors=payload.get("ticker_priors", {}),
            metadata=payload.get("metadata", {}),
            lightgbm_model_text=lightgbm_model_text,
        )
    model_payloads = payload.get("models")
    if model_payloads is None:
        model_payloads = [payload]
    if not isinstance(model_payloads, list) or not model_payloads:
        raise ValueError("model artifact contains no linear models")
    linear_models = []
    for model_payload in model_payloads:
        scale = np.asarray(model_payload["scale"], dtype="float64")
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        coef = np.asarray(model_payload["coef"], dtype="float64")
        if coef.shape[0] != expected_len:
            raise ValueError("linear model coefficient count does not match feature schema")
        linear_models.append(
            MLRankerLinearModel(
                clip_low=np.asarray(model_payload["clip_low"], dtype="float64"),
                clip_high=np.asarray(model_payload["clip_high"], dtype="float64"),
                mean=np.asarray(model_payload["mean"], dtype="float64"),
                scale=scale,
                coef=coef,
                intercept=float(model_payload["intercept"]),
                weight=float(model_payload.get("weight", 1.0)),
                market=(
                    str(model_payload["market"])
                    if model_payload.get("market") is not None
                    else None
                ),
                cap_bucket=(
                    str(model_payload["cap_bucket"])
                    if model_payload.get("cap_bucket") is not None
                    else None
                ),
                board_bucket=(
                    str(model_payload["board_bucket"])
                    if model_payload.get("board_bucket") is not None
                    else None
                ),
                listing_bucket=(
                    str(model_payload["listing_bucket"])
                    if model_payload.get("listing_bucket") is not None
                    else None
                ),
            )
        )
    return MLRankerModel(
        feature_names=feature_names,
        linear_models=tuple(linear_models),
        ticker_priors=payload.get("ticker_priors", {}),
        metadata=payload.get("metadata", {}),
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


def _cap_bucket(value: object) -> str:
    try:
        market_cap = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(market_cap) or market_cap <= 0:
        return "unknown"
    if market_cap < 10_000_000_000.0:
        return "small"
    if market_cap < 50_000_000_000.0:
        return "mid"
    return "large"


def _env_disabled() -> bool:
    return os.environ.get("VALUEINVESTOR_DISABLE_ML_SCORER", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Optional[MLRankerModel]:
    if _env_disabled():
        return None

    key = str(path)
    try:
        stat = path.stat()
        fingerprint: tuple[int, int] | None = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        fingerprint = None

    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    if fingerprint is None:
        _MODEL_CACHE[key] = (None, None)
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = _model_from_payload(payload)
    except Exception as exc:
        logger.warning("Could not load ML ranker model from %s: %s", path, exc)
        model = None

    _MODEL_CACHE[key] = (fingerprint, model)
    return model


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _needs_short_horizon_features(feature_names: Sequence[str]) -> bool:
    return any(
        name == feature or name == f"{feature}_miss" or name == f"slog_{feature}"
        for feature in SHORT_HORIZON_FEATURES
        for name in feature_names
    )


def _price_feature_paths() -> tuple[Path, ...]:
    return (
        Path("data/trainer/ashare_prices.parquet"),
        Path("data/trainer/hkshare_prices.parquet"),
    )


def _price_feature_cache_key(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    key: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        key.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(key)


def _recent_price_features_for_results(
    results: Sequence[ScreeningResult],
) -> Dict[int, Mapping[str, float]]:
    if not results:
        return {}

    paths = _price_feature_paths()
    cache_key = _price_feature_cache_key(paths)
    if not cache_key:
        return {}

    try:
        import pandas as pd
    except Exception:
        return {}

    cached = _PRICE_FEATURE_CACHE.get(cache_key)
    if cached is None:
        frames = []
        for path in paths:
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(str(path), columns=["ticker", "date", "close"])
            except Exception:
                continue
            if frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return {}
        prices = pd.concat(frames, ignore_index=True)
        prices["ticker"] = prices["ticker"].astype(str)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna(subset=["ticker", "date", "close"])
        prices = prices[prices["close"] > 0].sort_values(["ticker", "date"], kind="mergesort")
        if prices.empty:
            return {}

        grouped_close = prices.groupby("ticker", sort=False)["close"]
        for window in (5, 21, 63):
            prices[f"price_return_{window}d"] = grouped_close.transform(
                lambda values, window=window: values / values.shift(window) - 1.0
            )
        prices["_daily_return"] = grouped_close.pct_change()
        grouped_daily = prices.groupby("ticker", sort=False)["_daily_return"]
        for window in (21, 63):
            prices[f"price_volatility_{window}d"] = (
                grouped_daily
                .rolling(window=window, min_periods=max(5, window // 3))
                .std()
                .reset_index(level=0, drop=True)
            )
        prices["_market"] = prices["ticker"].map(_market_bucket)
        for window in (5, 21, 63):
            price_col = f"price_return_{window}d"
            market_col = f"market_return_{window}d"
            prices[market_col] = prices.groupby(["_market", "date"], sort=False)[price_col].transform("mean")
            prices[f"relative_return_{window}d"] = prices[price_col] - prices[market_col]
        for window in (21, 63):
            price_col = f"price_volatility_{window}d"
            market_col = f"market_volatility_{window}d"
            prices[market_col] = prices.groupby(["_market", "date"], sort=False)[price_col].transform("mean")
            prices[f"relative_volatility_{window}d"] = prices[price_col] - prices[market_col]
        cached = prices[["ticker", "date", *SHORT_HORIZON_FEATURES]].reset_index(drop=True)
        _PRICE_FEATURE_CACHE.clear()
        _PRICE_FEATURE_CACHE[cache_key] = cached

    price_features = cached
    tickers = {result.company.ticker for result in results}
    price_features = price_features[price_features["ticker"].isin(tickers)]
    if price_features.empty:
        return {}

    output: Dict[int, Mapping[str, float]] = {}
    by_ticker = {
        ticker: group.sort_values("date", kind="mergesort")
        for ticker, group in price_features.groupby("ticker", sort=False)
    }
    for index, result in enumerate(results):
        group = by_ticker.get(result.company.ticker)
        if group is None or group.empty:
            continue
        asof = pd.to_datetime(result.valuation.date, errors="coerce")
        if pd.isna(asof):
            row = group.iloc[-1]
        else:
            eligible = group[group["date"] <= asof]
            if eligible.empty:
                continue
            row = eligible.iloc[-1]
        output[index] = {
            feature: float(row[feature])
            for feature in SHORT_HORIZON_FEATURES
            if feature in row and pd.notna(row[feature])
        }
    return output


def feature_matrix_from_values(
    values_by_result: Sequence[Mapping[str, object]],
    tickers: Sequence[str],
    model: MLRankerModel,
) -> np.ndarray:
    """Build the production feature matrix from already-resolved value rows."""
    if len(values_by_result) != len(tickers):
        raise ValueError("values_by_result and tickers must have equal lengths")
    if any(name.startswith(("cs_rank_", "market_cs_rank_")) for name in model.feature_names):
        source_features = {
            source: np.asarray(
                [
                    feature_vector_from_values(
                        values,
                        ticker,
                        model.ticker_priors,
                        feature_names=[source],
                    )[0]
                    for values, ticker in zip(values_by_result, tickers)
                ],
                dtype="float64",
            )
            for source in CROSS_SECTIONAL_FEATURES
        }
        cross_sectional_features = _cross_sectional_feature_matrix(
            source_features,
            tickers,
            model.feature_names,
        )
    else:
        cross_sectional_features = {}
    feature_indexes = {
        feature_name: model.feature_names.index(feature_name)
        for feature_name in cross_sectional_features
        if feature_name in model.feature_names
    }
    rows = []
    for index, ticker in enumerate(tickers):
        row = feature_vector_from_values(
            values_by_result[index],
            ticker,
            model.ticker_priors,
            feature_names=model.feature_names,
        )
        for feature_name, feature_index in feature_indexes.items():
            row[feature_index] = float(cross_sectional_features[feature_name][index])
        rows.append(row)
    return np.asarray(rows, dtype="float64")


def feature_matrix_for_results(results: Sequence[ScreeningResult], model: MLRankerModel) -> np.ndarray:
    short_features = (
        _recent_price_features_for_results(results) if _needs_short_horizon_features(model.feature_names) else {}
    )
    values_by_result = [
        {**values_from_result(result), **short_features.get(index, {})} for index, result in enumerate(results)
    ]
    return feature_matrix_from_values(
        values_by_result,
        [result.company.ticker for result in results],
        model,
    )


def _percentile_scores(predictions: np.ndarray) -> np.ndarray:
    n = len(predictions)
    if n == 0:
        return predictions
    if n == 1:
        return np.asarray([50.0], dtype="float64")

    order = np.argsort(predictions, kind="mergesort")
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.arange(1, n + 1, dtype="float64")
    return ranks / (n + 1.0) * 100.0


def _result_asof_date(result: ScreeningResult) -> date:
    return _parse_temporal_date(getattr(result.valuation, "date", None)) or date.today()


def _temporal_member_index(
    asof: date,
    members: Sequence[tuple[Optional[date], Optional[date], MLRankerModel]],
) -> int:
    for index, (start_date, end_date, _model) in enumerate(members):
        if start_date is not None and asof < start_date:
            continue
        if end_date is not None and asof >= end_date:
            continue
        return index
    return len(members) - 1


def _conditional_route_weights_for_results(
    results: Sequence[ScreeningResult],
    routes: Sequence[Mapping[str, object]],
) -> np.ndarray:
    weights = np.zeros(len(results), dtype="float64")
    if not routes:
        return weights
    asof_dates = [_result_asof_date(result) for result in results]
    markets = [_market_bucket(result.company.ticker) for result in results]
    cap_buckets = [_cap_bucket(result.valuation.market_cap_rmb) for result in results]
    board_buckets = [_board_bucket(result.company.ticker) for result in results]
    listing_buckets = [_listing_bucket(result.company.ticker) for result in results]
    for route in routes:
        try:
            candidate_weight = float(route.get("candidate_weight", 1.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(candidate_weight) or candidate_weight <= 0.0:
            continue
        candidate_weight = min(1.0, max(0.0, candidate_weight))
        start_date = _parse_temporal_date(route.get("start_date"))
        end_date = _parse_temporal_date(route.get("end_date"))
        market = route.get("market")
        cap_bucket = route.get("cap_bucket")
        board_bucket = route.get("board_bucket")
        listing_bucket = route.get("listing_bucket")
        for index, asof in enumerate(asof_dates):
            if start_date is not None and asof < start_date:
                continue
            if end_date is not None and asof >= end_date:
                continue
            if market is not None and str(market) != "" and markets[index] != str(market):
                continue
            if (
                cap_bucket is not None
                and str(cap_bucket) != ""
                and cap_buckets[index] != str(cap_bucket)
            ):
                continue
            if (
                board_bucket is not None
                and str(board_bucket) != ""
                and board_buckets[index] != str(board_bucket)
            ):
                continue
            if (
                listing_bucket is not None
                and str(listing_bucket) != ""
                and listing_buckets[index] != str(listing_bucket)
            ):
                continue
            weights[index] = max(weights[index], candidate_weight)
    return weights


def _predict_results_with_model(
    results: Sequence[ScreeningResult],
    model: MLRankerModel,
) -> np.ndarray:
    if model.conditional_base_model is not None and model.conditional_candidate_model is not None:
        route_weights = _conditional_route_weights_for_results(
            results,
            model.conditional_routes,
        )
        candidate_predictions = _predict_results_with_model(
            results,
            model.conditional_candidate_model,
        )
        base_predictions = _predict_results_with_model(
            results,
            model.conditional_base_model,
        )
        return route_weights * candidate_predictions + (1.0 - route_weights) * base_predictions

    if model.temporal_member_models:
        if len(model.temporal_member_models) == 1:
            _start_date, _end_date, member_model = model.temporal_member_models[0]
            return _predict_results_with_model(results, member_model)
        predictions = np.empty(len(results), dtype="float64")
        grouped: dict[int, list[int]] = {}
        for index, result in enumerate(results):
            member_index = _temporal_member_index(
                _result_asof_date(result),
                model.temporal_member_models,
            )
            grouped.setdefault(member_index, []).append(index)
        for member_index, result_indexes in grouped.items():
            _start_date, _end_date, member_model = model.temporal_member_models[member_index]
            subset = [results[index] for index in result_indexes]
            predictions[np.asarray(result_indexes, dtype=int)] = _predict_results_with_model(
                subset,
                member_model,
            )
        return predictions

    if model.rank_member_models:
        predictions = np.zeros(len(results), dtype="float64")
        total_weight = 0.0
        for weight, member_model in model.rank_member_models:
            member_predictions = _predict_results_with_model(results, member_model)
            predictions += weight * _application_snapshot_rank(member_predictions)
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("rank payload-member ensemble has no positive weights")
        return predictions / total_weight

    if model.member_models:
        predictions = np.zeros(len(results), dtype="float64")
        total_weight = 0.0
        for weight, member_model in model.member_models:
            predictions += weight * _predict_results_with_model(results, member_model)
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("payload-member ensemble has no positive weights")
        return predictions / total_weight

    if model.market_member_models:
        markets = [_market_bucket(result.company.ticker) for result in results]
        predictions = np.zeros(len(results), dtype="float64")
        total_weights = np.zeros(len(results), dtype="float64")
        for market_weights, member_model in model.market_member_models:
            weights = np.asarray(
                [float(market_weights.get(market, 0.0)) for market in markets],
                dtype="float64",
            )
            if not weights.any():
                continue
            predictions += weights * _predict_results_with_model(results, member_model)
            total_weights += weights
        assigned = total_weights > 0.0
        if assigned.any():
            predictions[assigned] = predictions[assigned] / total_weights[assigned]
        if (~assigned).any():
            predictions[~assigned] = _predict_results_with_model(
                [result for result, is_assigned in zip(results, assigned) if not is_assigned],
                model.market_member_models[0][1],
            )
        return predictions

    features = feature_matrix_for_results(results, model)
    return model.predict_matrix(
        features,
        tickers=[result.company.ticker for result in results],
        cap_buckets=[_cap_bucket(result.valuation.market_cap_rmb) for result in results],
    )


def score_results_with_ml_ranker(
    results: Sequence[ScreeningResult],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> bool:
    """Replace composite scores with trained ML-ranker scores when available."""
    model = load_model(model_path)
    if model is None or not results:
        return False

    hand_scores = np.asarray(
        [float(result.composite_score) for result in results],
        dtype="float64",
    )
    predictions = _predict_results_with_model(results, model)
    try:
        application_blend_weight = float(model.metadata.get("application_blend_candidate_weight", 1.0))
    except (TypeError, ValueError):
        application_blend_weight = 1.0
    if 0.0 <= application_blend_weight < 1.0:
        predictions = application_blend_weight * _application_snapshot_rank(predictions) + (
            1.0 - application_blend_weight
        ) * _application_snapshot_rank(hand_scores)
    factor_components = _application_factor_config(model.metadata)
    if factor_components:
        needs_momentum = any(component.get("signal") == "momentum_63d" for component in factor_components)
        short_features = _recent_price_features_for_results(results) if needs_momentum else {}
        values_by_result = [
            {**values_from_result(result), **short_features.get(index, {})}
            for index, result in enumerate(results)
        ]
        predictions = _application_factor_predictions(
            predictions,
            values_by_result,
            factor_components,
        )
    residual_weight, residual_column, residual_direction = _application_residual_config(model.metadata)
    if not factor_components and residual_weight < 1.0:
        residual_values = residual_direction * np.asarray(
            [_to_float(values_from_result(result).get(residual_column)) for result in results],
            dtype="float64",
        )
        model_rank = _application_snapshot_rank(predictions)
        residual_rank = _application_snapshot_rank(residual_values)
        blended = model_rank.copy()
        usable = np.isfinite(model_rank) & np.isfinite(residual_rank)
        blended[usable] = residual_weight * model_rank[usable] + (1.0 - residual_weight) * residual_rank[usable]
        predictions = blended
    (
        de_crowding_weight,
        de_crowding_max_rank,
        de_crowding_min_market_return,
    ) = _application_gross_profitability_de_crowding_config(model.metadata)
    if de_crowding_weight > 0.0:
        price_features = _recent_price_features_for_results(results)
        values_by_result = [
            {**values_from_result(result), **price_features.get(index, {})}
            for index, result in enumerate(results)
        ]
        gross_profit_assets = np.asarray(
            [
                _safe_div(
                    _to_float(values.get("revenue"))
                    * _to_float(values.get("gross_margin")),
                    _to_float(values.get("total_assets")),
                )
                for values in values_by_result
            ],
            dtype="float64",
        )
        market_return_63d = np.asarray(
            [_to_float(values.get("market_return_63d")) for values in values_by_result],
            dtype="float64",
        )
        predictions = _application_gross_profitability_de_crowding_predictions(
            predictions,
            gross_profit_assets,
            market_return_63d,
            factor_weight=de_crowding_weight,
            max_model_rank=de_crowding_max_rank,
            min_market_return=de_crowding_min_market_return,
        )
    percentile_scores = _percentile_scores(predictions)
    for result, raw_score, percentile_score in zip(results, predictions, percentile_scores):
        result._ml_ranker_raw_score = float(raw_score)
        result.composite_score = float(percentile_score)
        result._ml_ranker_model_path = str(model_path)
        result._ml_ranker_target_horizon = str(model.metadata.get("target_horizon", "6m"))
    return True
