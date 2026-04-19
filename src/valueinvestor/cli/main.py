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

import logging
import sys
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

@app.command()
def scan(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML."),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Override screening.top_n."),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Skip LLM analysis step."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Report output dir."),
) -> None:
    """Full pipeline: fetch → screen → score → LLM analysis → generate reports."""
    cfg = _load_cfg(config)
    if top_n is not None:
        cfg.screening.top_n = top_n
    if output_dir is not None:
        cfg.output.reports_dir = output_dir

    cache = _make_cache(cfg)

    from valueinvestor.screener.engine import ScreeningEngine

    engine = ScreeningEngine(config=cfg, cache=cache)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1 — fetch & screen
        progress.add_task("Fetching universe & screening candidates…", total=None)
        try:
            results = engine.run(top_n=cfg.screening.top_n)
        except ValueInvestorError as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]No candidates survived screening.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"\n[green]✓[/green] Screening complete — {len(results)} candidates\n")
    console.print(_screening_table(results))

    # Step 2 — LLM analysis (optional)
    analyses = []
    llm_usage: Optional[dict] = None
    if not skip_analysis:
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
                task = progress.add_task(
                    f"Analyzing top {len(results)} candidates…", total=len(results)
                )
                for idx, sr in enumerate(results, 1):
                    progress.update(
                        task,
                        description=f"Analyzing [{idx}/{len(results)}] {sr.company.ticker}…",
                    )
                    try:
                        analysis = pipeline.analyze_company(sr)
                        analyses.append(analysis)
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
        from valueinvestor.reports.md_generator import MarkdownReportGenerator

        temp_pipeline = AP(
            llm=None,  # type: ignore[arg-type]
            cache=cache,
            config=cfg,
        )
        report = temp_pipeline.build_report(analyses, cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating reports…", total=None)

            if "md" in cfg.output.formats:
                md_gen = MarkdownReportGenerator()
                md_path = md_gen.save(report, cfg.output.reports_dir)
                report_paths.append(md_path)

            if "pdf" in cfg.output.formats:
                try:
                    from valueinvestor.reports.pdf_generator import PDFReportGenerator

                    pdf_gen = PDFReportGenerator()
                    pdf_path = pdf_gen.generate_from_report(report, cfg.output.reports_dir)
                    report_paths.append(pdf_path)
                except Exception as exc:
                    err_console.print(f"[yellow]PDF generation skipped:[/yellow] {exc}")

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
) -> None:
    """Run quantitative screening only (no LLM analysis)."""
    cfg = _load_cfg(config)
    if top_n is not None:
        cfg.screening.top_n = top_n

    cache = _make_cache(cfg)

    from valueinvestor.screener.engine import ScreeningEngine

    engine = ScreeningEngine(config=cfg, cache=cache)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching universe & screening candidates…", total=None)
        try:
            results = engine.run(top_n=cfg.screening.top_n)
        except ValueInvestorError as exc:
            err_console.print(f"[red]Screening failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

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
    from valueinvestor.data.models import Company, Financials, Market, ValuationMetrics
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
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
