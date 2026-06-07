"""Typer CLI for the ValueInvestor application.

Commands
--------
- ``scan``    – full pipeline: fetch → screen → score → LLM analyse → report
- ``screen``  – quantitative screening only (no LLM)
- ``analyze`` – deep-dive a single ticker via LLM
- ``report``  – regenerate reports from cached analyses
- ``config``  – show / initialise configuration
- ``serve``   – launch the web dashboard via uvicorn
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from valueinvestor.config import AppConfig, load_config, save_default_config
from valueinvestor.data.cache import DataCache
from valueinvestor.data.models import ScreeningResult
from valueinvestor.errors import ValueInvestorError

# ---------------------------------------------------------------------------
# App & console
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="valueinvestor",
    help="Automated Chinese stock value-investing research tool.",
    add_completion=False,
    no_args_is_help=True,
)
config_app = typer.Typer(help="Show or initialise configuration.")
app.add_typer(config_app, name="config")

console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger("valueinvestor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str) -> AppConfig:
    """Load config or exit with a friendly message."""
    try:
        return load_config(config_path)
    except Exception as exc:
        err_console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _make_cache(cfg: AppConfig) -> DataCache:
    try:
        return DataCache(db_path=cfg.cache.db_path, ttl_hours=cfg.cache.ttl_hours)
    except ValueInvestorError as exc:
        err_console.print(f"[red]Cache error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _fmt(value: Optional[float], decimals: int = 2, fallback: str = "—") -> str:
    if value is None:
        return fallback
    return f"{value:.{decimals}f}"


def _pct(value: Optional[float], fallback: str = "—") -> str:
    if value is None:
        return fallback
    return f"{value * 100:.1f}%"


def _csv_option(value: str) -> Optional[tuple[str, ...]]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(items) == 1 and items[0].lower() in {"auto", "all"}:
        return None
    return items or None


def _anchor_model_paths_option(value: str, *, target_horizon: str, output_model_path: Path) -> tuple[Path, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or (len(items) == 1 and items[0].lower() in {"none", "off", "false", "0"}):
        return ()
    if len(items) == 1 and items[0].lower() in {"auto", "all"}:
        if target_horizon != "6m":
            return ()
        patterns = (
            "ml_ranker_model.verify-auto.json",
            "ml_ranker_model.verify-ridge.json",
            "ml_ranker_model.backup-*.json",
        )
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(Path("data/trainer").glob(pattern)))
        output_resolved = output_model_path.resolve()
        unique: list[Path] = []
        seen: set[Path] = set()
        if output_model_path.exists():
            seen.add(output_resolved)
        for path in paths:
            resolved = path.resolve()
            if resolved == output_resolved or resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)

        def anchor_score(path: Path) -> tuple[float, str]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return (float("-inf"), "")
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            if not isinstance(metadata, dict):
                return (float("-inf"), "")
            metrics = metadata.get("metrics")
            train_metrics = metadata.get("train_metrics")
            rho = None
            if isinstance(metrics, dict):
                rho = (metrics.get("6m") or {}).get("spearman_rho")
            if rho is None and isinstance(train_metrics, dict):
                rho = (train_metrics.get("6m") or {}).get("spearman_rho")
            try:
                score = float(rho)
            except (TypeError, ValueError):
                score = float("-inf")
            return (score, str(metadata.get("trained_at", "")))

        try:
            anchor_limit = int(os.environ.get("VALUEINVESTOR_ML_ANCHOR_LIMIT", "3"))
        except ValueError:
            anchor_limit = 3
        ranked = sorted(unique, key=anchor_score, reverse=True)
        if anchor_limit > 0:
            return tuple(ranked[:anchor_limit])
        return tuple(ranked)
    return tuple(Path(item) for item in items)


def _scan_progress_interval_seconds() -> float:
    """Return seconds between durable scan heartbeat lines; 0 disables them."""
    raw = os.environ.get("VALUEINVESTOR_SCAN_PROGRESS_INTERVAL_SECONDS", "15").strip()
    try:
        interval = float(raw)
    except ValueError:
        logger.warning(
            "Invalid VALUEINVESTOR_SCAN_PROGRESS_INTERVAL_SECONDS=%r; using 15s",
            raw,
        )
        return 15.0
    return max(0.0, interval)


@contextmanager
def _scan_heartbeat(
    label: str,
    *,
    interval_seconds: Optional[float] = None,
) -> Iterator[Callable[[str], None]]:
    """Emit periodic durable scan progress while a blocking operation runs."""
    interval = (
        _scan_progress_interval_seconds()
        if interval_seconds is None
        else max(0.0, interval_seconds)
    )
    started_at = time.monotonic()
    current_label = label
    label_lock = threading.Lock()
    stop_event = threading.Event()

    def update_label(next_label: str) -> None:
        nonlocal current_label
        with label_lock:
            current_label = next_label

    def read_label() -> str:
        with label_lock:
            return current_label

    def run_heartbeat() -> None:
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - started_at
            console.print(
                f"[dim]still running: {read_label()} ({elapsed:.0f}s elapsed)[/dim]"
            )

    thread: Optional[threading.Thread] = None
    if interval > 0:
        thread = threading.Thread(
            target=run_heartbeat,
            daemon=True,
            name="scan-progress-heartbeat",
        )
        thread.start()

    try:
        yield update_label
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=1)


def _analysis_progress_callback(
    prefix: str,
    update_heartbeat: Callable[[str], None],
) -> Callable[[str, str, str], None]:
    def callback(ticker: str, company_name: str, status: str) -> None:
        label = f"{prefix} {ticker} {company_name}: {status}"
        update_heartbeat(label)
        console.print(f"[dim]progress: {label}[/dim]")

    return callback


def _load_current_scorer_metadata(
    model_path: Path | None = None,
    *,
    target_horizon: str = "6m",
) -> dict[str, object]:
    """Return report-friendly metadata for the active production scorer."""

    def _spearman_rhos(metrics: object) -> dict[str, float]:
        if not isinstance(metrics, dict):
            return {}
        metric_rhos: dict[str, float] = {}
        for horizon in ("1w", "1m", "3m", "6m"):
            horizon_metrics = metrics.get(horizon)
            if not isinstance(horizon_metrics, dict):
                continue
            rho = horizon_metrics.get("spearman_rho")
            if rho is None:
                continue
            try:
                metric_rhos[horizon] = float(rho)
            except (TypeError, ValueError):
                logger.warning("Invalid %s rho in ML metadata: %r", horizon, rho)
        return metric_rhos

    model_path = model_path or Path("data/trainer/ml_ranker_model.json")
    if model_path.exists():
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read ML ranker metadata from %s: %s", model_path, exc)
        else:
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            models = payload.get("models")
            payload_members = payload.get("payload_members")
            temporal_payload_members = payload.get("temporal_payload_members")
            if isinstance(payload_members, list) and payload_members:
                ensemble_size = len(payload_members)
            elif isinstance(temporal_payload_members, list) and temporal_payload_members:
                ensemble_size = len(temporal_payload_members)
            elif isinstance(models, list) and models:
                ensemble_size = len(models)
            else:
                ensemble_size = 1
            summary: dict[str, object] = {
                "type": "ml_ranker",
                "model_path": str(model_path),
                "schema_version": payload.get("schema_version"),
                "backend": metadata.get("backend"),
                "ensemble_size": metadata.get("ensemble_size", ensemble_size),
                "target_horizon": metadata.get("target_horizon", target_horizon),
            }
            if metadata.get("members") is not None:
                summary["members"] = metadata["members"]
            if metadata.get("trained_at") is not None:
                summary["trained_at"] = metadata["trained_at"]
            if metadata.get("final_rho_improvement") is not None:
                summary["final_rho_improvement"] = metadata["final_rho_improvement"]

            train_rhos = _spearman_rhos(metadata.get("train_metrics"))
            holdout_rhos = _spearman_rhos(metadata.get("metrics"))
            if train_rhos:
                summary["train_spearman_rhos"] = train_rhos
                summary["spearman_rhos"] = train_rhos
                summary["spearman_rho_basis"] = "train_metrics"
            elif holdout_rhos:
                summary["spearman_rhos"] = holdout_rhos
                summary["spearman_rho_basis"] = "holdout_metrics"
            if holdout_rhos:
                summary["holdout_spearman_rhos"] = holdout_rhos
            return summary

    meta_path = Path("data/trainer/current_best_scorer.json")
    summary = {"type": "hand_scorer"}
    if not meta_path.exists():
        return summary
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read current scorer metadata from %s: %s", meta_path, exc)
        return summary

    rhos: dict[str, float] = {}
    for horizon in ("1m", "3m", "6m"):
        raw = meta.get(f"spearman_rho_{horizon}")
        if raw is None:
            continue
        try:
            rhos[horizon] = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s rho in %s: %r", horizon, meta_path, raw)
    if rhos:
        summary["spearman_rhos"] = rhos
    updated_at = meta.get("updated_at")
    if updated_at is not None:
        summary["updated_at"] = updated_at
    return summary


def _load_current_spearman_rhos() -> dict[str, float]:
    """Read current scorer rho metadata without running scorer evaluation."""
    metadata = _load_current_scorer_metadata()
    rhos = metadata.get("spearman_rhos")
    return rhos if isinstance(rhos, dict) else {}


def _attach_current_scorer_metadata(report) -> None:
    scorer_metadata = _load_current_scorer_metadata()
    report.config_summary["scorer_model"] = scorer_metadata
    rhos = scorer_metadata.get("spearman_rhos")
    if isinstance(rhos, dict) and rhos:
        report.config_summary["spearman_rhos"] = rhos


def _attach_dual_scorer_metadata(
    report,
    scorer_models: dict[str, dict[str, object]],
) -> None:
    report.config_summary["scorer_models"] = scorer_models
    report.config_summary["score_targets"] = list(scorer_models)
    primary_rhos: dict[str, float] = {}
    for target, metadata in scorer_models.items():
        rhos = metadata.get("spearman_rhos")
        if not isinstance(rhos, dict):
            continue
        rho = rhos.get(target)
        if rho is None:
            continue
        try:
            primary_rhos[target] = float(rho)
        except (TypeError, ValueError):
            continue
    if primary_rhos:
        report.spearman_rhos = primary_rhos
        report.config_summary["spearman_rhos"] = primary_rhos


def _model_path_for_score_target(target: str) -> Path:
    from valueinvestor.screener.ml_ranker import DEFAULT_MODEL_PATH, ONE_WEEK_MODEL_PATH

    if target == "1w":
        return ONE_WEEK_MODEL_PATH
    if target == "6m":
        return DEFAULT_MODEL_PATH
    raise ValueError("score target must be one of: 1w, 6m, both")


def _rank_with_model(
    base_results: list[ScreeningResult],
    *,
    model_path: Path,
    top_n: int,
) -> list[ScreeningResult]:
    from valueinvestor.screener.ml_ranker import score_results_with_ml_ranker

    ranked = [result.model_copy(deep=True) for result in base_results]
    if not score_results_with_ml_ranker(ranked, model_path=model_path):
        console.print(
            f"[yellow]ML ranker unavailable at {model_path}; using hand scorer fallback[/yellow]"
        )
    ranked.sort(key=lambda result: result.composite_score, reverse=True)
    for idx, result in enumerate(ranked, start=1):
        result.rank = idx
    return ranked[:top_n]


def _dedupe_results_for_analysis(
    results_by_target: dict[str, list[ScreeningResult]],
) -> list[ScreeningResult]:
    ordered: list[ScreeningResult] = []
    seen: set[str] = set()
    for target in ("6m", "1w"):
        for result in results_by_target.get(target, []):
            ticker = result.company.ticker
            if ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(result)
    for target, results in results_by_target.items():
        if target in {"6m", "1w"}:
            continue
        for result in results:
            ticker = result.company.ticker
            if ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(result)
    return ordered


def _screening_progress_callback(
    prefix: str,
    update_heartbeat: Callable[[str], None],
) -> Callable[[str, bool], None]:
    def callback(message: str, checkpoint: bool = True) -> None:
        label = f"{prefix}: {message}"
        update_heartbeat(label)
        if checkpoint:
            console.print(f"[dim]progress: {label}[/dim]")

    return callback


def _screening_table(results: list[ScreeningResult]) -> Table:
    """Build a Rich table from screening results."""
    table = Table(title="Screening Results", show_lines=False, expand=False)
    table.add_column("Rank", justify="right", style="bold cyan")
    table.add_column("Ticker", style="green")
    table.add_column("Name")
    table.add_column("Market")
    table.add_column("Score", justify="right", style="bold yellow")
    table.add_column("PE", justify="right")
    table.add_column("PB", justify="right")
    table.add_column("ROE", justify="right")
    for r in results:
        table.add_row(
            str(r.rank),
            r.company.ticker,
            r.company.name,
            r.company.market.value,
            _fmt(r.composite_score, 1),
            _fmt(r.valuation.pe_ratio, 1),
            _fmt(r.valuation.pb_ratio, 1),
            _pct(r.financials.roe),
        )
    return table


def _cached_analyses_for_results(
    results: list[ScreeningResult],
    cfg: AppConfig,
    cache: DataCache,
) -> list:
    """Load cached LLM analyses and attach freshly screened score/rank data."""
    from valueinvestor.analysis.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline(llm=None, cache=cache, config=cfg)  # type: ignore[arg-type]
    analyses = []
    for sr in results:
        analysis = pipeline.cached_analysis_for_result(sr, allow_expired=True)
        if analysis is None:
            err_console.print(
                f"[yellow]Cached analysis missing for {sr.company.ticker}; skipping analysis text[/yellow]"
            )
            continue
        analyses.append(analysis)
    return analyses


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def _run_multi_timeframe_scan(
    engine,
    cfg,
    cache,
    skip_analysis: bool,
    cached_analysis_only: bool = False,
) -> None:
    """Run single-screen multi-timeframe pipeline and generate a Chinese report.

    Screens once using the same scoring algorithm for all horizons, then reports
    the current Spearman ρ for 1m/3m/6m forward returns alongside the results.
    """
    # ── Screen once ──────────────────────────────────────────────────
    screening_started = time.monotonic()
    console.print("[dim]progress: screening candidates started[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Screening candidates…", total=None)
        try:
            with _scan_heartbeat("screening candidates") as update_heartbeat:
                engine.progress_callback = _screening_progress_callback(
                    "screening", update_heartbeat
                )
                try:
                    results = engine.run(top_n=cfg.screening.top_n)
                finally:
                    engine.progress_callback = None
        except Exception as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    console.print(
        f"[dim]progress: screening candidates finished "
        f"({time.monotonic() - screening_started:.1f}s elapsed)[/dim]"
    )

    if not results:
        console.print("[yellow]No candidates survived screening.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"\n[green]✓[/green] {len(results)} candidates\n")
    console.print(_screening_table(results))

    # ── Get current ρ values from the active scorer metadata ─────────
    spearman_rhos: dict = _load_current_spearman_rhos()
    if spearman_rhos:
        console.print(
            "[dim]ρ 1m=%.4f  3m=%.4f  6m=%.4f[/dim]"
            % (spearman_rhos.get("1m", 0), spearman_rhos.get("3m", 0), spearman_rhos.get("6m", 0))
        )
    else:
        try:
            from valueinvestor.scorer_improver.evaluator import evaluate_scorer_all_targets

            eval_started = time.monotonic()
            console.print("[dim]progress: evaluating scorer rho started[/dim]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Evaluating ρ against ground truth…", total=None)
                with _scan_heartbeat("evaluating scorer rho"):
                    eval_results = evaluate_scorer_all_targets()
            console.print(
                f"[dim]progress: evaluating scorer rho finished "
                f"({time.monotonic() - eval_started:.1f}s elapsed)[/dim]"
            )
            for h in ("1m", "3m", "6m"):
                spearman_rhos[h] = eval_results.get(h, {}).get("spearman_rho", None)
            console.print(
                "[dim]ρ 1m=%.4f  3m=%.4f  6m=%.4f[/dim]"
                % (spearman_rhos.get("1m", 0), spearman_rhos.get("3m", 0), spearman_rhos.get("6m", 0))
            )
        except Exception:
            console.print("[dim]Ground truth not available — skipping ρ evaluation[/dim]")

    # ── LLM analysis ─────────────────────────────────────────────────
    analyses = []
    if cached_analysis_only:
        analyses = _cached_analyses_for_results(results, cfg, cache)
        console.print(
            f"[green]✓[/green] Cached analyses refreshed — {len(analyses)}/{len(results)} matched"
        )
    elif not skip_analysis:
        try:
            from valueinvestor.analysis.llm_client import LLMClient
            from valueinvestor.analysis.pipeline import AnalysisPipeline

            llm = LLMClient(config=cfg.llm)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Analyzing {len(results)} candidates…", total=len(results)
                )
                for idx, sr in enumerate(results, 1):
                    progress.update(task, description=f"Analyzing [{idx}/{len(results)}] {sr.company.ticker}…")
                    prefix = f"analysis {idx}/{len(results)}"
                    company_label = f"{prefix} {sr.company.ticker} {sr.company.name}"
                    analysis_started = time.monotonic()
                    console.print(f"[dim]progress: {company_label} started[/dim]")
                    try:
                        with _scan_heartbeat(company_label) as update_heartbeat:
                            pipeline = AnalysisPipeline(
                                llm=llm,
                                cache=cache,
                                config=cfg,
                                progress_callback=_analysis_progress_callback(
                                    prefix, update_heartbeat
                                ),
                            )
                            analysis = pipeline.analyze_company(sr)
                        analyses.append(analysis)
                        console.print(
                            f"[dim]progress: {company_label} finished "
                            f"({time.monotonic() - analysis_started:.1f}s elapsed)[/dim]"
                        )
                    except Exception:
                        err_console.print(f"[yellow]⚠ Analysis failed for {sr.company.ticker} — skipping[/yellow]")
                    progress.advance(task)

            console.print(f"[green]✓[/green] Analysis complete — {len(analyses)}/{len(results)} succeeded")
        except Exception as exc:
            err_console.print(f"[red]LLM analysis error:[/red] {exc}")
            err_console.print("[yellow]Continuing without analysis…[/yellow]")

    # ── Build and save report ────────────────────────────────────────
    from valueinvestor.analysis.pipeline import AnalysisPipeline as AP
    from valueinvestor.reports.md_generator import MultiTimeframeMarkdownReportGenerator

    report_started = time.monotonic()
    console.print("[dim]progress: report generation started[/dim]")
    with _scan_heartbeat("report generation"):
        rpt_pipeline = AP(llm=None, cache=cache, config=cfg)  # type: ignore[arg-type]
        report = rpt_pipeline.build_multi_timeframe_report(
            results_by_horizon={"1m": results, "3m": results, "6m": results},
            analyses=analyses,
            config=cfg,
            total_screened=engine.last_total_screened,
            spearman_rhos=spearman_rhos,
        )
        _attach_current_scorer_metadata(report)

        md_gen = MultiTimeframeMarkdownReportGenerator()
        md_path = md_gen.save(report, cfg.output.reports_dir)
    console.print(
        f"[dim]progress: report generation finished "
        f"({time.monotonic() - report_started:.1f}s elapsed)[/dim]"
    )
    console.print(f"\n[green]✓[/green] Report saved → {md_path}")

    if "pdf" in cfg.output.formats:
        try:
            from valueinvestor.reports.pdf_generator import PDFReportGenerator

            pdf_started = time.monotonic()
            console.print("[dim]progress: PDF report generation started[/dim]")
            with _scan_heartbeat("PDF report generation"):
                pdf_gen = PDFReportGenerator()
                pdf_path = pdf_gen.generate_from_multi_timeframe_report(
                    report, cfg.output.reports_dir
                )
            console.print(
                f"[dim]progress: PDF report generation finished "
                f"({time.monotonic() - pdf_started:.1f}s elapsed)[/dim]"
            )
            console.print(f"[green]✓[/green] PDF saved → {pdf_path}")
        except Exception as exc:
            err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")


def _run_dual_target_scan(
    cfg,
    cache,
    *,
    skip_analysis: bool,
    cached_analysis_only: bool,
    report_mode: str,
    cache_only: bool,
) -> None:
    from valueinvestor.analysis.pipeline import AnalysisPipeline
    from valueinvestor.reports.md_generator import MultiTimeframeMarkdownReportGenerator
    from valueinvestor.screener.engine import ScreeningEngine
    from valueinvestor.screener.ml_ranker import clear_model_cache
    from valueinvestor.screener.scorer import MultiFactorScorer

    clear_model_cache()
    scorer_models = {
        "1w": _load_current_scorer_metadata(
            _model_path_for_score_target("1w"),
            target_horizon="1w",
        ),
        "6m": _load_current_scorer_metadata(
            _model_path_for_score_target("6m"),
            target_horizon="6m",
        ),
    }
    for target, metadata in scorer_models.items():
        console.print(
            f"[dim]active {target} ML ranker: {metadata.get('model_path')} "
            f"(backend={metadata.get('backend', 'unknown')}, "
            f"trained_at={metadata.get('trained_at', 'unknown')})[/dim]"
        )

    engine = ScreeningEngine(
        config=cfg,
        cache=cache,
        scorer=MultiFactorScorer(use_ml_ranker=False),
        cache_only=cache_only,
    )

    screening_started = time.monotonic()
    console.print("[dim]progress: fetching universe and screening candidates started[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching universe & screening candidates…", total=None)
        try:
            with _scan_heartbeat(
                "fetching universe and screening candidates"
            ) as update_heartbeat:
                engine.progress_callback = _screening_progress_callback(
                    "screening",
                    update_heartbeat,
                )
                try:
                    base_results = engine.run(top_n=1_000_000)
                finally:
                    engine.progress_callback = None
        except ValueInvestorError as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    console.print(
        f"[dim]progress: fetching universe and screening candidates finished "
        f"({time.monotonic() - screening_started:.1f}s elapsed)[/dim]"
    )
    if not base_results:
        console.print("[yellow]No candidates survived screening.[/yellow]")
        raise typer.Exit(code=0)

    results_by_target = {
        target: _rank_with_model(
            base_results,
            model_path=_model_path_for_score_target(target),
            top_n=cfg.screening.top_n,
        )
        for target in ("1w", "6m")
    }
    for target, results in results_by_target.items():
        label = "1-week" if target == "1w" else "6-month"
        console.print(f"\n[green]✓[/green] {label} scoring complete — {len(results)} candidates\n")
        console.print(_screening_table(results))

    analysis_results = _dedupe_results_for_analysis(results_by_target)
    analyses = []
    if cached_analysis_only:
        analyses = _cached_analyses_for_results(analysis_results, cfg, cache)
        console.print(
            f"[green]✓[/green] Cached analyses refreshed — "
            f"{len(analyses)}/{len(analysis_results)} matched"
        )
    elif not skip_analysis:
        try:
            from valueinvestor.analysis.llm_client import LLMClient

            llm = LLMClient(config=cfg.llm)
            pipeline = AnalysisPipeline(llm=llm, cache=cache, config=cfg)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Analyzing {len(analysis_results)} unique candidates…",
                    total=len(analysis_results),
                )
                for idx, sr in enumerate(analysis_results, 1):
                    progress.update(
                        task,
                        description=f"Analyzing [{idx}/{len(analysis_results)}] {sr.company.ticker}…",
                    )
                    try:
                        analyses.append(pipeline.analyze_company(sr))
                    except ValueInvestorError:
                        err_console.print(
                            f"[yellow]⚠ Analysis failed for {sr.company.ticker} — skipping[/yellow]"
                        )
                    progress.advance(task)
            console.print(
                f"[green]✓[/green] LLM analysis complete — "
                f"{len(analyses)}/{len(analysis_results)} succeeded"
            )
        except ValueInvestorError as exc:
            err_console.print(f"[red]LLM analysis error:[/red] {exc}")
            err_console.print("[yellow]Continuing without analysis…[/yellow]")

    report_paths: list[str] = []
    if analyses or skip_analysis or cached_analysis_only:
        rpt_pipeline = AnalysisPipeline(llm=None, cache=cache, config=cfg)  # type: ignore[arg-type]
        report_started = time.monotonic()
        console.print("[dim]progress: report generation started[/dim]")
        with _scan_heartbeat("report generation"):
            if report_mode in {"combined", "both"}:
                report = rpt_pipeline.build_multi_timeframe_report(
                    results_by_horizon=results_by_target,
                    analyses=analyses,
                    config=cfg,
                    total_screened=engine.last_total_screened,
                )
                report.title = "价值投资双目标筛选报告"
                _attach_dual_scorer_metadata(report, scorer_models)
                md_path = MultiTimeframeMarkdownReportGenerator().save(
                    report,
                    cfg.output.reports_dir,
                )
                report_paths.append(md_path)
                if "pdf" in cfg.output.formats:
                    try:
                        from valueinvestor.reports.pdf_generator import PDFReportGenerator

                        pdf_path = PDFReportGenerator().generate_from_multi_timeframe_report(
                            report,
                            cfg.output.reports_dir,
                        )
                        report_paths.append(pdf_path)
                    except Exception as exc:
                        err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")

            if report_mode in {"separate", "both"}:
                analyses_by_ticker = {analysis.ticker: analysis for analysis in analyses}
                for target, results in results_by_target.items():
                    target_analyses = [
                        analyses_by_ticker[result.company.ticker]
                        for result in results
                        if result.company.ticker in analyses_by_ticker
                    ]
                    report = rpt_pipeline.build_multi_timeframe_report(
                        results_by_horizon={target: results},
                        analyses=target_analyses,
                        config=cfg,
                        total_screened=engine.last_total_screened,
                    )
                    report.title = f"价值投资{target}目标筛选报告"
                    _attach_dual_scorer_metadata(report, {target: scorer_models[target]})
                    md_path = MultiTimeframeMarkdownReportGenerator().save(
                        report,
                        cfg.output.reports_dir,
                    )
                    report_paths.append(md_path)
                    if "pdf" in cfg.output.formats:
                        try:
                            from valueinvestor.reports.pdf_generator import PDFReportGenerator

                            pdf_path = PDFReportGenerator().generate_from_multi_timeframe_report(
                                report,
                                cfg.output.reports_dir,
                            )
                            report_paths.append(pdf_path)
                        except Exception as exc:
                            err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")

        console.print(
            f"[dim]progress: report generation finished "
            f"({time.monotonic() - report_started:.1f}s elapsed)[/dim]"
        )
        console.print(f"[green]✓[/green] Reports generated — {len(report_paths)} file(s)")

    summary_lines = [
        f"Screened candidates: [cyan]{len(base_results)}[/cyan]",
        f"1-week top list:   [cyan]{len(results_by_target['1w'])}[/cyan]",
        f"6-month top list:  [cyan]{len(results_by_target['6m'])}[/cyan]",
    ]
    if analyses:
        summary_lines.append(f"Analyses completed: [cyan]{len(analyses)}[/cyan]")
    if report_paths:
        summary_lines.append("")
        summary_lines.append("[bold]Report files:[/bold]")
        summary_lines.extend(report_paths)
    console.print(Panel("\n".join(summary_lines), title="Dual-Target Scan Complete"))


@app.command()
def scan(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Override screening.top_n."),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Skip LLM analysis step."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Report output dir."),
    multi_timeframe: bool = typer.Option(False, "--multi-timeframe", help="Run screening for 1m/3m/6m horizons and generate Chinese multi-timeframe report."),
    cache_only: bool = typer.Option(False, "--cache-only", help="Use cached market data only; never call external data fetchers."),
    cached_analysis_only: bool = typer.Option(False, "--cached-analysis-only", help="Use cached LLM analyses only and refresh them with the latest screening scores."),
    score_target: str = typer.Option("6m", "--score-target", help="Scoring target: 6m, 1w, or both."),
    report_mode: str = typer.Option("combined", "--report-mode", help="For --score-target both: combined, separate, or both."),
) -> None:
    """Full pipeline: fetch → screen → score → LLM analysis → generate reports."""
    cfg = _load_cfg(config)
    if top_n is not None:
        cfg.screening.top_n = top_n
    if output_dir is not None:
        cfg.output.reports_dir = output_dir
    score_target = score_target.lower()
    if score_target not in {"6m", "1w", "both"}:
        err_console.print("[red]Invalid --score-target; use 6m, 1w, or both.[/red]")
        raise typer.Exit(code=1)
    report_mode = report_mode.lower()
    if report_mode not in {"combined", "separate", "both"}:
        err_console.print("[red]Invalid --report-mode; use combined, separate, or both.[/red]")
        raise typer.Exit(code=1)

    cache = _make_cache(cfg)

    if score_target == "both":
        _run_dual_target_scan(
            cfg,
            cache,
            skip_analysis=skip_analysis,
            cached_analysis_only=cached_analysis_only,
            report_mode=report_mode,
            cache_only=cache_only,
        )
        return

    from valueinvestor.screener.engine import ScreeningEngine
    from valueinvestor.screener.ml_ranker import clear_model_cache, load_model
    from valueinvestor.screener.scorer import MultiFactorScorer

    clear_model_cache()
    active_model_path = _model_path_for_score_target(score_target)
    active_model = load_model(active_model_path)
    if active_model is None:
        console.print("[dim]active ML ranker: unavailable; using hand scorer fallback[/dim]")
    else:
        model_meta = active_model.metadata if isinstance(active_model.metadata, dict) else {}
        trained_at = model_meta.get("trained_at", "unknown")
        target = model_meta.get("target", "unknown")
        backend = model_meta.get("backend", "unknown")
        console.print(
            f"[dim]active {score_target} ML ranker: {active_model_path} "
            f"(backend={backend}, target={target}, trained_at={trained_at})[/dim]"
        )

    engine = ScreeningEngine(
        config=cfg,
        cache=cache,
        scorer=MultiFactorScorer(ml_model_path=active_model_path),
        cache_only=cache_only,
    )

    if multi_timeframe:
        _run_multi_timeframe_scan(
            engine,
            cfg,
            cache,
            skip_analysis,
            cached_analysis_only=cached_analysis_only,
        )
        return

    screening_started = time.monotonic()
    console.print("[dim]progress: fetching universe and screening candidates started[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1 — fetch & screen
        progress.add_task("Fetching universe & screening candidates…", total=None)
        try:
            with _scan_heartbeat(
                "fetching universe and screening candidates"
            ) as update_heartbeat:
                engine.progress_callback = _screening_progress_callback(
                    "screening", update_heartbeat
                )
                try:
                    results = engine.run(top_n=cfg.screening.top_n)
                finally:
                    engine.progress_callback = None
        except ValueInvestorError as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    console.print(
        f"[dim]progress: fetching universe and screening candidates finished "
        f"({time.monotonic() - screening_started:.1f}s elapsed)[/dim]"
    )

    if not results:
        console.print("[yellow]No candidates survived screening.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"\n[green]✓[/green] Screening complete — {len(results)} candidates\n")
    console.print(_screening_table(results))

    # Step 2 — LLM analysis (optional)
    analyses = []
    llm_usage: Optional[dict] = None
    if cached_analysis_only:
        analyses = _cached_analyses_for_results(results, cfg, cache)
        console.print(
            f"[green]✓[/green] Cached analyses refreshed — {len(analyses)}/{len(results)} matched"
        )
    elif not skip_analysis:
        try:
            from valueinvestor.analysis.llm_client import LLMClient
            from valueinvestor.analysis.pipeline import AnalysisPipeline

            llm = LLMClient(config=cfg.llm)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Analyzing top {len(results)} candidates…", total=len(results)
                )
                for idx, sr in enumerate(results, 1):
                    progress.update(
                        task,
                        description=f"Analyzing [{idx}/{len(results)}] {sr.company.ticker}…",
                    )
                    prefix = f"analysis {idx}/{len(results)}"
                    company_label = f"{prefix} {sr.company.ticker} {sr.company.name}"
                    analysis_started = time.monotonic()
                    console.print(f"[dim]progress: {company_label} started[/dim]")
                    try:
                        with _scan_heartbeat(company_label) as update_heartbeat:
                            pipeline = AnalysisPipeline(
                                llm=llm,
                                cache=cache,
                                config=cfg,
                                progress_callback=_analysis_progress_callback(
                                    prefix, update_heartbeat
                                ),
                            )
                            analysis = pipeline.analyze_company(sr)
                        analyses.append(analysis)
                        console.print(
                            f"[dim]progress: {company_label} finished "
                            f"({time.monotonic() - analysis_started:.1f}s elapsed)[/dim]"
                        )
                    except ValueInvestorError:
                        err_console.print(
                            f"[yellow]⚠ Analysis failed for {sr.company.ticker} — skipping[/yellow]"
                        )
                    progress.advance(task)

            llm_usage = llm.get_usage_summary()
            console.print(
                f"[green]✓[/green] LLM analysis complete — {len(analyses)}/{len(results)} succeeded"
            )
        except ValueInvestorError as exc:
            err_console.print(f"[red]LLM analysis error:[/red] {exc}")
            err_console.print("[yellow]Continuing without analysis…[/yellow]")

    # Step 3 — generate reports
    report_paths: list[str] = []
    if analyses:
        from valueinvestor.analysis.pipeline import AnalysisPipeline as AP
        from valueinvestor.reports.md_generator import (
            MarkdownReportGenerator,
            MultiTimeframeMarkdownReportGenerator,
        )

        temp_pipeline = AP(
            llm=None,  # type: ignore[arg-type]
            cache=cache,
            config=cfg,
        )
        scorer_metadata = _load_current_scorer_metadata(
            active_model_path,
            target_horizon=score_target,
        )
        if score_target == "6m":
            report = temp_pipeline.build_report(
                analyses,
                cfg,
                total_screened=engine.last_total_screened,
            )
            report.config_summary["scorer_model"] = scorer_metadata
            rhos = scorer_metadata.get("spearman_rhos")
            if isinstance(rhos, dict) and rhos:
                report.config_summary["spearman_rhos"] = rhos
        else:
            report = temp_pipeline.build_multi_timeframe_report(
                results_by_horizon={score_target: results},
                analyses=analyses,
                config=cfg,
                total_screened=engine.last_total_screened,
            )
            report.title = f"价值投资{score_target}目标筛选报告"
            _attach_dual_scorer_metadata(report, {score_target: scorer_metadata})

        report_started = time.monotonic()
        console.print("[dim]progress: report generation started[/dim]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating reports…", total=None)

            with _scan_heartbeat("report generation"):
                if "md" in cfg.output.formats:
                    if score_target == "6m":
                        md_path = MarkdownReportGenerator().save(
                            report,
                            cfg.output.reports_dir,
                        )
                    else:
                        md_path = MultiTimeframeMarkdownReportGenerator().save(
                            report,
                            cfg.output.reports_dir,
                        )
                    report_paths.append(md_path)

                if "pdf" in cfg.output.formats:
                    try:
                        from valueinvestor.reports.pdf_generator import PDFReportGenerator

                        pdf_gen = PDFReportGenerator()
                        if score_target == "6m":
                            pdf_path = pdf_gen.generate_from_report(
                                report,
                                cfg.output.reports_dir,
                            )
                        else:
                            pdf_path = pdf_gen.generate_from_multi_timeframe_report(
                                report,
                                cfg.output.reports_dir,
                            )
                        report_paths.append(pdf_path)
                    except Exception as exc:
                        err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")

        console.print(
            f"[dim]progress: report generation finished "
            f"({time.monotonic() - report_started:.1f}s elapsed)[/dim]"
        )
        console.print(f"[green]✓[/green] Reports generated — {len(report_paths)} file(s)")

    # Summary panel
    summary_lines = [
        f"Candidates screened: [cyan]{len(results)}[/cyan]",
    ]
    if analyses:
        summary_lines.append(f"Analyses completed:  [cyan]{len(analyses)}[/cyan]")
    if report_paths:
        summary_lines.append("")
        summary_lines.append("[bold]Report files:[/bold]")
        for p in report_paths:
            summary_lines.append(f"  • {p}")
    if llm_usage:
        cost = llm_usage.get("total_cost_usd", 0)
        tokens = llm_usage.get("total_tokens", 0)
        summary_lines.append("")
        summary_lines.append(
            f"LLM usage: [cyan]{tokens:,}[/cyan] tokens · [cyan]${cost:.4f}[/cyan] USD"
        )

    console.print()
    console.print(Panel("\n".join(summary_lines), title="Scan Summary", border_style="green"))


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

@app.command()
def screen(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Override screening.top_n."),
    cache_only: bool = typer.Option(False, "--cache-only", help="Use cached market data only; never call external data fetchers."),
) -> None:
    """Run quantitative screening only (no LLM analysis)."""
    cfg = _load_cfg(config)
    if top_n is not None:
        cfg.screening.top_n = top_n

    cache = _make_cache(cfg)

    from valueinvestor.screener.engine import ScreeningEngine

    engine = ScreeningEngine(config=cfg, cache=cache, cache_only=cache_only)

    screening_started = time.monotonic()
    console.print("[dim]progress: fetching universe and screening candidates started[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching universe & screening candidates…", total=None)
        try:
            with _scan_heartbeat(
                "fetching universe and screening candidates"
            ) as update_heartbeat:
                engine.progress_callback = _screening_progress_callback(
                    "screening", update_heartbeat
                )
                try:
                    results = engine.run(top_n=cfg.screening.top_n)
                finally:
                    engine.progress_callback = None
        except ValueInvestorError as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    console.print(
        f"[dim]progress: fetching universe and screening candidates finished "
        f"({time.monotonic() - screening_started:.1f}s elapsed)[/dim]"
    )

    if not results:
        console.print("[yellow]No candidates survived screening.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"\n[green]✓[/green] {len(results)} candidates found\n")
    console.print(_screening_table(results))


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker to analyse (e.g. 600519, 0700.HK)."),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
) -> None:
    """Deep-dive LLM analysis of a single company."""
    cfg = _load_cfg(config)
    cache = _make_cache(cfg)

    from valueinvestor.data.fetcher_ashare import AShareFetcher
    from valueinvestor.data.fetcher_hkshare import HKShareFetcher
    from valueinvestor.data.models import Company, Market
    from valueinvestor.screener.scorer import MultiFactorScorer

    # Determine market from ticker format
    is_hk = ticker.upper().endswith(".HK")
    market = Market.HK_SHARE if is_hk else Market.A_SHARE

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Fetching data for {ticker}…", total=None)

        try:
            if market == Market.HK_SHARE:
                fetcher = HKShareFetcher()
            else:
                fetcher = AShareFetcher()  # type: ignore[assignment]

            company = fetcher.fetch_company_detail(ticker)
            if company is None:
                company = Company(ticker=ticker, name=ticker, market=market)

            financials = fetcher.fetch_financials(ticker)
            valuation = fetcher.fetch_valuation(ticker)
        except ValueInvestorError as exc:
            err_console.print(f"[red]Data fetch failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    if financials is None or valuation is None:
        err_console.print(f"[red]Insufficient data for {ticker}.[/red]")
        raise typer.Exit(code=1)

    sr = ScreeningResult(company=company, financials=financials, valuation=valuation)
    MultiFactorScorer().score(sr)
    sr.rank = 1

    # Run LLM analysis
    try:
        from valueinvestor.analysis.llm_client import LLMClient
        from valueinvestor.analysis.pipeline import AnalysisPipeline

        llm = LLMClient(config=cfg.llm)
        pipeline = AnalysisPipeline(llm=llm, cache=cache, config=cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Running LLM analysis on {ticker}…", total=None)
            analysis = pipeline.analyze_company(sr)
    except ValueInvestorError as exc:
        err_console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Display results
    console.print()
    console.print(
        Panel(
            f"[bold]{analysis.company_name}[/bold]  ({analysis.ticker})\n"
            f"Score: [yellow]{sr.composite_score:.1f}[/yellow]  "
            f"PE: {_fmt(valuation.pe_ratio, 1)}  "
            f"PB: {_fmt(valuation.pb_ratio, 1)}  "
            f"ROE: {_pct(financials.roe)}",
            title="Company Overview",
            border_style="cyan",
        )
    )

    for dim in analysis.analyses:
        confidence = f"  (confidence: {dim.confidence:.0%})" if dim.confidence is not None else ""
        console.print(
            Panel(
                dim.content,
                title=f"[bold]{dim.title}[/bold]{confidence}",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analysis.key_milestones:
        milestone_text = "\n".join(f"• {m}" for m in analysis.key_milestones)
        console.print(
            Panel(milestone_text, title="Key Milestones to Watch", border_style="magenta")
        )

    usage = llm.get_usage_summary()
    console.print(
        f"\n[dim]LLM: {usage['total_tokens']:,} tokens · ${usage['total_cost_usd']:.4f} USD[/dim]"
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command()
def report(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Report output dir."),
    fmt: str = typer.Option("both", "--format", "-f", help="Output format: md, pdf, or both."),
) -> None:
    """Regenerate reports from cached analysis data (no re-fetching)."""
    cfg = _load_cfg(config)
    if output_dir is not None:
        cfg.output.reports_dir = output_dir

    cache = _make_cache(cfg)

    cached_analyses = cache.list_analyses()
    if not cached_analyses:
        err_console.print("[yellow]No cached analyses found. Run 'scan' first.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Found [cyan]{len(cached_analyses)}[/cyan] cached analyses.")

    from valueinvestor.analysis.pipeline import AnalysisPipeline
    from valueinvestor.reports.md_generator import MarkdownReportGenerator

    pipeline = AnalysisPipeline(llm=None, cache=cache, config=cfg)  # type: ignore[arg-type]
    inv_report = pipeline.build_report(cached_analyses, cfg)
    current_rhos = _load_current_spearman_rhos()
    if current_rhos:
        inv_report.config_summary["spearman_rhos"] = current_rhos
    _attach_current_scorer_metadata(inv_report)

    formats = (
        ["md", "pdf"] if fmt == "both" else [fmt]
    )
    report_paths: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Generating reports…", total=None)

        if "md" in formats:
            md_gen = MarkdownReportGenerator()
            md_path = md_gen.save(inv_report, cfg.output.reports_dir)
            report_paths.append(md_path)

        if "pdf" in formats:
            try:
                from valueinvestor.reports.pdf_generator import PDFReportGenerator

                pdf_gen = PDFReportGenerator()
                pdf_path = pdf_gen.generate_from_report(inv_report, cfg.output.reports_dir)
                report_paths.append(pdf_path)
            except Exception as exc:
                err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")

    console.print(f"\n[green]✓[/green] Generated {len(report_paths)} report(s):")
    for p in report_paths:
        console.print(f"  • {p}")


# ---------------------------------------------------------------------------
# config show / config init
# ---------------------------------------------------------------------------

@config_app.command("show")
def config_show(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
) -> None:
    """Print the current resolved configuration."""
    cfg = _load_cfg(config)
    import yaml

    console.print(Panel(yaml.dump(cfg.model_dump(), sort_keys=False), title="Configuration"))


@config_app.command("init")
def config_init(
    path: str = typer.Option("config.yaml", "--path", "-p", help="Destination path."),
) -> None:
    """Generate a default config.yaml file."""
    target = Path(path)
    if target.exists():
        overwrite = typer.confirm(f"{target} already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)
    save_default_config(str(target))
    console.print(f"[green]✓[/green] Default config written to [cyan]{target}[/cyan]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Start the ValueInvestor web dashboard."""
    try:
        import uvicorn
    except ImportError as exc:
        err_console.print("[red]uvicorn is not installed.[/red] pip install uvicorn")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Starting web dashboard at[/green] http://{host}:{port}"
    )
    uvicorn.run("valueinvestor.web.app:app", host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# improve-scorer
# ---------------------------------------------------------------------------

@app.command("train-ml-scorer")
def train_ml_scorer(
    ground_truth_path: str = typer.Option(
        "data/trainer/ground_truth_current.parquet",
        "--ground-truth-path",
        help="Ground-truth parquet used for training.",
    ),
    snapshots_path: str = typer.Option(
        "data/trainer/ml_training_daily_snapshots.parquet",
        "--snapshots-path",
        help="Prepared ML snapshot parquet output path.",
    ),
    output_model_path: str = typer.Option(
        "data/trainer/ml_ranker_model.json",
        "--output-model-path",
        help="Trained ML ranker artifact output path. Defaults to the 1w path when --target-horizon=1w.",
    ),
    target_horizon: str = typer.Option(
        "6m",
        "--target-horizon",
        help="Primary training target horizon: 6m or 1w.",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Training backend: auto, mlx, mlx-cg, mlx-adam, or numpy.",
    ),
    model_kind: str = typer.Option(
        "ridge",
        "--model-kind",
        help=(
            "Model kind: ridge, ridge-only, market-ridge, market-ridge-only, "
            "market-ridge-recent, market-ridge-recent-<days>, pairwise, or auto."
        ),
    ),
    force_snapshots: bool = typer.Option(
        False,
        "--force-snapshots",
        help="Rebuild ML snapshots before training.",
    ),
    ridge_lambda: float = typer.Option(
        100.0,
        "--ridge-lambda",
        help="Ridge regularization strength.",
    ),
    snapshot_frequency: str = typer.Option(
        "daily",
        "--snapshot-frequency",
        help="Snapshot frequency: daily or quarterly.",
    ),
    start_date: str = typer.Option(
        "2016-05-25",
        "--start-date",
        help="Daily snapshot start date, YYYY-MM-DD.",
    ),
    end_date: str = typer.Option(
        "2025-07-07",
        "--end-date",
        help="Daily snapshot end date, YYYY-MM-DD or auto for the latest supported 6m label date.",
    ),
    target_improvement: float = typer.Option(
        0.30,
        "--target-improvement",
        help="Required primary-horizon rho improvement over the same-snapshot hand baseline.",
    ),
    max_training_rows: int = typer.Option(
        500_000,
        "--max-training-rows",
        help="Deterministic daily-snapshot training sample size; use 0 for all rows.",
    ),
    gate_holdout_months: int = typer.Option(
        24,
        "--gate-holdout-months",
        help="Latest N months reserved as untouched promotion holdout.",
    ),
    gate_embargo_days: int = typer.Option(
        141,
        "--gate-embargo-days",
        help="Embargo days between training and promotion holdout.",
    ),
    gate_min_6m_delta: float = typer.Option(
        0.001,
        "--gate-min-6m-delta",
        help="Required absolute primary-horizon rho improvement on untouched holdout.",
    ),
    gate_max_degradation: float = typer.Option(
        0.001,
        "--gate-max-degradation",
        help="Maximum allowed absolute rho degradation in any holdout horizon.",
    ),
    gate_min_weighted_utility: float = typer.Option(
        0.0,
        "--gate-min-weighted-utility",
        help="Minimum weighted holdout utility required for promotion.",
    ),
    gate_min_regime_6m_win_rate: float = typer.Option(
        0.50,
        "--gate-min-regime-6m-win-rate",
        help="Minimum 6m regime win rate required for promotion.",
    ),
    gate_max_regime_6m_degradation: float = typer.Option(
        0.05,
        "--gate-max-regime-6m-degradation",
        help="Maximum allowed 6m rho degradation inside any reported regime bucket.",
    ),
    gate_require_top20_excess_non_degradation: bool = typer.Option(
        True,
        "--gate-require-top20-excess-non-degradation/--gate-allow-top20-excess-degradation",
        help="Require primary-horizon top20 hit rate and excess return not to degrade together.",
    ),
    walk_forward_folds: int = typer.Option(
        3,
        "--walk-forward-folds",
        help="Rolling validation folds before holdout promotion (0 disables).",
    ),
    walk_forward_validation_months: int = typer.Option(
        6,
        "--walk-forward-validation-months",
        help="Validation window length per walk-forward fold.",
    ),
    walk_forward_max_rows: int = typer.Option(
        500_000,
        "--walk-forward-max-rows",
        help="Max sampled rows per walk-forward training fold; use 0 for all rows.",
    ),
    walk_forward_min_6m_delta: float = typer.Option(
        0.001,
        "--walk-forward-min-6m-delta",
        help="Required mean primary-horizon rho improvement across walk-forward folds.",
    ),
    walk_forward_max_horizon_degradation: float = typer.Option(
        0.01,
        "--walk-forward-max-horizon-degradation",
        help="Allowed mean walk-forward rho degradation for non-primary horizons.",
    ),
    full_eval_candidate_limit: int = typer.Option(
        0,
        "--full-eval-candidate-limit",
        help="Full-train/gate candidates to try; 0 uses auto defaults (8 for 1w, 12 for 6m).",
    ),
    candidate_feature_sets: str = typer.Option(
        "auto",
        "--candidate-feature-sets",
        help="Candidate feature sets: auto, core, short-horizon, expanded, poly, or comma-separated list.",
    ),
    candidate_prior_strategies: str = typer.Option(
        "auto",
        "--candidate-prior-strategies",
        help=(
            "Candidate prior strategies: auto, no-ticker-priors, ticker-priors, "
            "rolling-ticker-priors, or comma-separated list."
        ),
    ),
    candidate_targets: str = typer.Option(
        "auto",
        "--candidate-targets",
        help=(
            "Candidate target columns: auto, target-rank-1w, target-rank-1w-market, "
            "target-rank-6m, target-rank-weighted-901, target-rank-weighted-8515, "
            "target-rank-weighted-802, target-rank-weighted, target-rank-mean, "
            "market-normalized 6m variants such as target-rank-6m-market or "
            "target-rank-weighted-703-market, or comma-separated list."
        ),
    ),
    candidate_ridge_lambdas: str = typer.Option(
        "auto",
        "--candidate-ridge-lambdas",
        help="Candidate ridge lambdas: auto or comma-separated positive numbers.",
    ),
    promotion_min_train_rho: Optional[float] = typer.Option(
        None,
        "--promotion-min-train-rho",
        help="Minimum full-train primary-horizon Spearman rho required before holdout promotion.",
    ),
    promotion_min_gate_rho: Optional[float] = typer.Option(
        None,
        "--promotion-min-gate-rho",
        help="Minimum untouched-holdout primary-horizon Spearman rho required for promotion.",
    ),
    anchor_model_paths: str = typer.Option(
        "auto",
        "--anchor-model-paths",
        help=(
            "Historical ML ranker artifacts to blend during 6m promotion: auto, none, "
            "or comma-separated paths."
        ),
    ),
) -> None:
    """Train the local ML ranker used by the default screening flow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from datetime import date

        from valueinvestor.scorer_improver.ml_trainer import train_ml_ranker
        from valueinvestor.scorer_improver.promotion_gate import HoldoutGateConfig

        target_horizon = target_horizon.lower()
        effective_output_model_path = Path(output_model_path)
        if target_horizon == "1w" and output_model_path == "data/trainer/ml_ranker_model.json":
            effective_output_model_path = Path("data/trainer/ml_ranker_model_1w.json")
        parsed_end_date = None if end_date.lower() == "auto" else date.fromisoformat(end_date)

        metadata = train_ml_ranker(
            force_snapshots=force_snapshots,
            ground_truth_path=Path(ground_truth_path),
            snapshots_path=Path(snapshots_path),
            output_model_path=effective_output_model_path,
            ridge_lambda=ridge_lambda,
            backend=backend,
            snapshot_frequency=snapshot_frequency,
            start_date=date.fromisoformat(start_date),
            end_date=parsed_end_date,
            target_improvement=target_improvement,
            max_training_rows=max_training_rows,
            gate_config=HoldoutGateConfig(
                holdout_months=gate_holdout_months,
                embargo_days=gate_embargo_days,
                min_6m_delta=gate_min_6m_delta,
                max_horizon_degradation=gate_max_degradation,
                min_weighted_utility=gate_min_weighted_utility,
                min_regime_6m_win_rate=gate_min_regime_6m_win_rate,
                max_regime_6m_degradation=gate_max_regime_6m_degradation,
                min_primary_rho=promotion_min_gate_rho,
                require_6m_top20_excess_non_degradation=gate_require_top20_excess_non_degradation,
            ),
            walk_forward_folds=walk_forward_folds,
            walk_forward_validation_months=walk_forward_validation_months,
            walk_forward_max_rows=walk_forward_max_rows,
            walk_forward_min_6m_delta=walk_forward_min_6m_delta,
            walk_forward_max_horizon_degradation=walk_forward_max_horizon_degradation,
            model_kind=model_kind,
            target_horizon=target_horizon,
            full_eval_candidate_limit=(
                None if full_eval_candidate_limit <= 0 else full_eval_candidate_limit
            ),
            candidate_feature_sets=_csv_option(candidate_feature_sets),
            candidate_prior_strategies=_csv_option(candidate_prior_strategies),
            candidate_targets=_csv_option(candidate_targets),
            candidate_ridge_lambdas=_csv_option(candidate_ridge_lambdas),
            promotion_min_train_rho=promotion_min_train_rho,
            candidate_anchor_model_paths=_anchor_model_paths_option(
                anchor_model_paths,
                target_horizon=target_horizon,
                output_model_path=effective_output_model_path,
            ),
        )
    except Exception as exc:
        err_console.print(f"[red]ML scorer training failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    metrics = metadata.get("metrics", {})
    console.print(f"[green]✓[/green] ML scorer trained with backend: {metadata.get('backend')}")
    console.print(f"  snapshots: {metadata.get('snapshots_path')}")
    console.print(f"  model:     {metadata.get('model_path', output_model_path)}")
    target_horizon = str(metadata.get("target_horizon", "6m"))
    gate = metadata.get("promotion_gate")
    if isinstance(gate, dict):
        console.print(
            "  gate:      %s (Δ%s=%+.6f, utility=%+.6f)"
            % (
                gate.get("reason", "unknown"),
                target_horizon,
                float(gate.get("deltas", {}).get(target_horizon, 0.0)),
                float(gate.get("weighted_utility", 0.0)),
            )
        )
    for horizon in ("1w", "1m", "3m", "6m"):
        horizon_metrics = metrics.get(horizon, {}) if isinstance(metrics, dict) else {}
        rho = horizon_metrics.get("spearman_rho", 0.0)
        console.print(f"  ρ {horizon}:    {float(rho):.6f}")


@app.command("build-training-features")
def build_training_features(
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace existing valuation/financial feature files.",
    ),
    fetch_remote_history: bool = typer.Option(
        True,
        "--fetch-remote-history/--no-fetch-remote-history",
        help="Fetch provider statement and market-cap history; otherwise build local price-dated skeletons.",
    ),
    start_date: str = typer.Option(
        "2016-05-12",
        "--start-date",
        help="Historical feature start date, YYYY-MM-DD.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="Historical feature end date, YYYY-MM-DD. Defaults to today.",
    ),
) -> None:
    """Build point-in-time valuation and financial feature files for ML training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        from datetime import date

        from valueinvestor.scorer_improver.data_prep import (
            _init_db,
            _load_stored_universe,
            build_point_in_time_feature_data,
        )

        conn = _init_db()
        try:
            a_companies, hk_companies = _load_stored_universe(conn)
        finally:
            conn.close()

        files = build_point_in_time_feature_data(
            a_companies=a_companies,
            hk_companies=hk_companies,
            force=force,
            fetch_remote_history=fetch_remote_history,
            start=date.fromisoformat(start_date),
            end=date.fromisoformat(end_date) if end_date else date.today(),
        )
    except Exception as exc:
        err_console.print(f"[red]Training feature build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] Training feature files ready — {len(files)} files")
    for name, path in files.items():
        console.print(f"  • {name}: {path}")


@app.command("improve-scorer")
def improve_scorer(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
    fetch_only: bool = typer.Option(False, "--fetch-only", help="Only fetch/rebuild training data."),
    resume: bool = typer.Option(False, "--resume", help="Skip data fetch, resume improvement loop."),
    status: bool = typer.Option(False, "--status", help="Show experiment history summary."),
    structured: bool = typer.Option(False, "--structured", help="Run bounded parameter search instead of LLM patching."),
    force: bool = typer.Option(False, "--force", help="Force re-fetch of training data."),
    max_iterations: int = typer.Option(0, "--max-iter", "-n", help="Max iterations (0 = unlimited)."),
) -> None:
    """Autonomous scorer improvement loop (autoresearch-inspired).

    Fetches 10-year historical data, builds ground truth with forward returns
    for 1-month, 3-month, and 6-month horizons, then iteratively uses an LLM
    to improve scorer.py.  Each change is evaluated against all three horizons
    and kept if ANY target improves.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if status:
        from valueinvestor.scorer_improver.agent import show_status
        show_status()
        return

    if not resume:
        # Phase 1: Fetch training data
        console.print("\n[bold cyan]Phase 1:[/bold cyan] Fetching training data …")
        try:
            from valueinvestor.scorer_improver.data_prep import fetch_training_data

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Fetching 10-year historical data (differential) …", total=None)
                files = fetch_training_data(force=force)

            console.print(f"[green]✓[/green] Training data ready — {len(files)} files")
            for name, path in files.items():
                console.print(f"  • {name}: {path}")
        except Exception as exc:
            err_console.print(f"[red]Data fetch failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        # Phase 2: Build ground truth
        console.print("\n[bold cyan]Phase 2:[/bold cyan] Building ground truth …")
        try:
            from valueinvestor.scorer_improver.ground_truth import build_ground_truth

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Computing forward returns & scoring snapshots …", total=None)
                gt_path = build_ground_truth(force=force)

            console.print(f"[green]✓[/green] Ground truth ready → {gt_path}")
        except Exception as exc:
            err_console.print(f"[red]Ground truth build failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        if fetch_only:
            console.print("\n[yellow]--fetch-only mode: stopping after data prep.[/yellow]")
            return

    # Phase 3: Improvement loop (all horizons — 1m, 3m, 6m)
    console.print("\n[bold cyan]Phase 3:[/bold cyan] Starting improvement loop …")
    console.print("  Evaluating 1-month, 3-month, and 6-month forward returns")
    console.print("[dim]Press Ctrl-C to stop gracefully[/dim]\n")

    try:
        if structured:
            from valueinvestor.scorer_improver.agent import run_structured_parameter_search

            run_structured_parameter_search(max_iterations=max_iterations)
        else:
            from valueinvestor.scorer_improver.agent import run_improvement_loop

            run_improvement_loop(max_iterations=max_iterations)
    except Exception as exc:
        err_console.print(f"[red]Improvement loop error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
