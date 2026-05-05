"""End-to-end analysis pipeline: screening results → LLM analysis → report."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, List, Optional

from valueinvestor.analysis.llm_client import LLMClient
from valueinvestor.analysis.prompts import DIMENSION_PROMPTS, SYSTEM_PROMPT
from valueinvestor.config import AppConfig
from valueinvestor.data.cache import DataCache
from valueinvestor.errors import AnalysisError, LLMError
from valueinvestor.data.models import (
    AnalysisDimension,
    CompanyAnalysis,
    InvestmentReport,
    MultiTimeframeReport,
    ScreeningResult,
)

logger = logging.getLogger(__name__)


def _analysis_timeout_seconds() -> Optional[float]:
    """Return per-dimension LLM analysis timeout, or None when disabled."""
    raw = os.environ.get("VALUEINVESTOR_ANALYSIS_TIMEOUT_SECONDS", "90").strip()
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning("Invalid VALUEINVESTOR_ANALYSIS_TIMEOUT_SECONDS=%r; using 90s", raw)
        timeout = 90.0
    return timeout if timeout > 0 else None


def _coerce_to_str(value) -> str:
    """Ensure a value is a plain string — some models return nested dicts/lists."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Flatten nested dict: join values that are strings, recurse otherwise
        parts = []
        for v in value.values():
            parts.append(_coerce_to_str(v))
        return "\n\n".join(p for p in parts if p)
    if isinstance(value, list):
        return "\n".join(_coerce_to_str(item) for item in value)
    return str(value)


def _screening_to_data(result: ScreeningResult) -> dict:
    """Convert a ScreeningResult into the flat dict expected by prompt builders."""
    return {
        "company": result.company.model_dump(),
        "financials": result.financials.model_dump(),
        "valuation": result.valuation.model_dump(),
        "scores": {
            "composite_score": result.composite_score,
            "value_score": result.value_score,
            "quality_score": result.quality_score,
            "growth_score": result.growth_score,
            "momentum_score": result.momentum_score,
            "synergy_score": result.synergy_score,
            "value_growth_score": result.value_growth_score,
        },
    }


class AnalysisPipeline:
    """Orchestrates multi-dimension LLM analysis for screened companies."""

    def __init__(
        self,
        llm: LLMClient,
        cache: DataCache,
        config: AppConfig,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.config = config
        self.progress_callback = progress_callback

    def _emit_progress(self, ticker: str, company_name: str, status: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(ticker, company_name, status)
        except Exception:
            logger.debug("Analysis progress callback failed", exc_info=True)

    def cached_analysis_for_result(
        self,
        screening_result: ScreeningResult,
        *,
        allow_expired: bool = False,
    ) -> Optional[CompanyAnalysis]:
        """Return cached analysis text with the latest screening result attached."""
        cached = self.cache.get_analysis(
            screening_result.company.ticker,
            allow_expired=allow_expired,
        )
        if cached is None:
            return None
        return self._refresh_cached_analysis(cached, screening_result)

    def _refresh_cached_analysis(
        self,
        cached: CompanyAnalysis,
        screening_result: ScreeningResult,
    ) -> CompanyAnalysis:
        """Keep cached LLM text but replace stale score/rank data."""
        refreshed = cached.model_copy(
            update={
                "company_name": screening_result.company.name,
                "screening_result": screening_result,
            }
        )
        if self.config.cache.enabled:
            self.cache.set_analysis(refreshed)
        return refreshed

    # ------------------------------------------------------------------
    # Single-company analysis
    # ------------------------------------------------------------------

    def analyze_company(self, screening_result: ScreeningResult) -> CompanyAnalysis:
        """Run all analysis dimensions for one company.

        Returns a cached result when available and not expired.
        """
        ticker = screening_result.company.ticker
        company_name = screening_result.company.name

        # Check cache
        if self.config.cache.enabled:
            cached = self.cache.get_analysis(ticker)
            if cached is not None:
                logger.info("Using cached analysis for %s (%s)", ticker, company_name)
                self._emit_progress(ticker, company_name, "cache hit")
                return self._refresh_cached_analysis(cached, screening_result)

        logger.info("Starting LLM analysis for %s (%s)", ticker, company_name)
        self._emit_progress(ticker, company_name, "analysis started")
        company_data = _screening_to_data(screening_result)

        dimensions: List[AnalysisDimension] = []
        portfolio_weight: Optional[float] = None
        key_milestones: Optional[List[str]] = None

        try:
            request_timeout = _analysis_timeout_seconds()
            for dim_name, prompt_fn in DIMENSION_PROMPTS.items():
                logger.info("  → dimension: %s", dim_name)
                self._emit_progress(ticker, company_name, f"{dim_name} started")
                user_prompt = prompt_fn(company_data)

                raw = self.llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_format="json",
                    timeout=request_timeout,
                )

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse JSON for %s/%s – storing raw text", ticker, dim_name
                    )
                    parsed = {"title": dim_name, "content": raw}

                # Normalise: some models return a JSON array instead of an object.
                # Take the first element if it's a dict, otherwise treat as raw text.
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else {"title": dim_name, "content": raw}

                # Ensure we have a dict before calling .get()
                if not isinstance(parsed, dict):
                    parsed = {"title": dim_name, "content": str(parsed)}

                dim = AnalysisDimension(
                    dimension=dim_name,
                    title=parsed.get("title", dim_name),
                    content=_coerce_to_str(parsed.get("content", raw)),
                    confidence=parsed.get("confidence"),
                )
                dimensions.append(dim)

                # Extract recommendation-specific fields
                if dim_name == "recommendation":
                    portfolio_weight = parsed.get("portfolio_weight")
                    key_milestones = parsed.get("milestones")
                self._emit_progress(ticker, company_name, f"{dim_name} finished")
        except LLMError as exc:
            self._emit_progress(ticker, company_name, "analysis failed")
            raise AnalysisError(
                f"LLM analysis failed for {ticker} ({company_name})"
            ) from exc

        analysis = CompanyAnalysis(
            ticker=ticker,
            company_name=company_name,
            screening_result=screening_result,
            analyses=dimensions,
            portfolio_weight=portfolio_weight,
            key_milestones=key_milestones,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Persist to cache
        if self.config.cache.enabled:
            self.cache.set_analysis(analysis)
            logger.info("Cached analysis for %s", ticker)

        self._emit_progress(ticker, company_name, "analysis finished")
        return analysis

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        results: List[ScreeningResult],
    ) -> List[CompanyAnalysis]:
        """Analyse a list of screened companies sequentially.

        Processes companies one at a time to respect API rate limits.
        Individual failures are logged and skipped.
        """
        total = len(results)
        logger.info("Starting batch analysis for %d companies", total)

        analyses: List[CompanyAnalysis] = []

        for idx, sr in enumerate(results, 1):
            ticker = sr.company.ticker
            logger.info("[%d/%d] Analysing %s …", idx, total, ticker)
            try:
                analysis = self.analyze_company(sr)
                analyses.append(analysis)
            except Exception:
                logger.warning(
                    "[%d/%d] Failed to analyse %s – skipping",
                    idx,
                    total,
                    ticker,
                    exc_info=True,
                )

        logger.info(
            "Batch analysis complete: %d/%d succeeded", len(analyses), total
        )
        return analyses

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def build_report(
        self,
        analyses: List[CompanyAnalysis],
        config: AppConfig,
        total_screened: int = 0,
    ) -> InvestmentReport:
        """Assemble a final :class:`InvestmentReport` from completed analyses."""
        config_summary = {
            "markets": config.markets,
            "screening": config.screening.model_dump(),
            "investment": config.investment.model_dump(),
            "llm_model": config.llm.model,
            "output_language": config.output.language,
        }

        return InvestmentReport(
            title="ValueInvestor Investment Report",
            generated_at=datetime.now(timezone.utc).isoformat(),
            config_summary=config_summary,
            total_screened=total_screened,
            total_candidates=len(analyses),
            candidates=analyses,
        )

    def build_multi_timeframe_report(
        self,
        results_by_horizon: dict,
        analyses: List[CompanyAnalysis],
        config: AppConfig,
        total_screened: int = 0,
        spearman_rhos: dict | None = None,
    ) -> MultiTimeframeReport:
        """Assemble a :class:`MultiTimeframeReport` from multi-horizon screening.

        Parameters
        ----------
        results_by_horizon:
            Dict keyed by "1m", "3m", "6m" with top-N ``ScreeningResult`` lists.
        analyses:
            Deduplicated ``CompanyAnalysis`` list.
        config:
            Application config.
        total_screened:
            Total stocks evaluated before filtering (for summary display).
        spearman_rhos:
            Optional dict of Spearman ρ per horizon (keys: "1m", "3m", "6m").
        """
        config_summary = {
            "markets": config.markets,
            "screening": config.screening.model_dump(),
            "investment": config.investment.model_dump(),
            "llm_model": config.llm.model,
            "output_language": config.output.language,
        }
        return MultiTimeframeReport(
            title="价值投资多时间框架筛选报告",
            generated_at=datetime.now(timezone.utc).isoformat(),
            config_summary=config_summary,
            total_screened=total_screened,
            results_by_horizon=results_by_horizon,
            company_analyses=analyses,
            spearman_rhos=spearman_rhos or {},
        )
