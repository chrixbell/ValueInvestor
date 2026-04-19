"""FastAPI web application for ValueInvestor dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from valueinvestor.config import AppConfig, load_config
from valueinvestor.data.cache import DataCache
from valueinvestor.data.models import CompanyAnalysis, ScreeningResult

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path.cwd()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache(app: FastAPI) -> DataCache:
    return app.state.cache  # type: ignore[no-any-return]


def _get_config(app: FastAPI) -> AppConfig:
    return app.state.config  # type: ignore[no-any-return]


def _reports_dir(config: AppConfig) -> Path:
    return _PROJECT_ROOT / config.output.reports_dir


def _list_report_files(config: AppConfig) -> List[Dict[str, str]]:
    """Scan the reports directory and return metadata for each file."""
    rdir = _reports_dir(config)
    if not rdir.is_dir():
        return []
    files: List[Dict[str, str]] = []
    for p in sorted(rdir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix in {".md", ".pdf", ".html", ".txt"}:
            stat = p.stat()
            files.append(
                {
                    "filename": p.name,
                    "title": p.stem.replace("_", " ").replace("-", " "),
                    "ext": p.suffix.lstrip("."),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                }
            )
    return files


def _cached_candidates(cache: DataCache) -> List[ScreeningResult]:
    """Build a ranked candidate list from cached analyses."""
    analyses = cache.list_analyses()
    results = [a.screening_result for a in analyses]
    results.sort(key=lambda r: r.composite_score, reverse=True)
    for idx, r in enumerate(results, 1):
        r.rank = idx
    return results


def _fmt_market_cap(value: Optional[float]) -> str:
    """Human-readable market-cap string."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Lifespan – load config and initialise cache on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    cache = DataCache(
        db_path=config.cache.db_path,
        ttl_hours=config.cache.ttl_hours,
    )
    app.state.config = config
    app.state.cache = cache
    logger.info("ValueInvestor web app started (cache=%s)", config.cache.db_path)
    yield


app = FastAPI(title="ValueInvestor", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_BASE_DIR / "templates")
templates.env.filters["fmt_market_cap"] = _fmt_market_cap


# ---------------------------------------------------------------------------
# HTML page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request) -> HTMLResponse:
    """Dashboard home page with scan summary and top picks."""
    cache = _get_cache(request.app)
    config = _get_config(request.app)
    candidates = _cached_candidates(cache)
    has_data = len(candidates) > 0

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "has_data": has_data,
            "last_scan": candidates[0].valuation.date if has_data else None,
            "total_screened": config.screening.top_n if has_data else 0,
            "num_candidates": len(candidates),
            "top_candidates": candidates[:5],
        },
    )


@app.get("/candidates", response_class=HTMLResponse)
async def candidates_page(request: Request) -> HTMLResponse:
    """Full candidate list page."""
    cache = _get_cache(request.app)
    candidates = _cached_candidates(cache)
    return templates.TemplateResponse(
        "candidates.html",
        {"request": request, "candidates": candidates},
    )


@app.get("/candidates/{ticker}", response_class=HTMLResponse)
async def candidate_detail(request: Request, ticker: str) -> HTMLResponse:
    """Single company detail page."""
    cache = _get_cache(request.app)
    analysis: Optional[CompanyAnalysis] = cache.get_analysis(ticker)
    return templates.TemplateResponse(
        "detail.html",
        {"request": request, "analysis": analysis, "ticker": ticker},
    )


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request) -> HTMLResponse:
    """Reports download page."""
    config = _get_config(request.app)
    reports = _list_report_files(config)
    return templates.TemplateResponse(
        "reports.html",
        {"request": request, "reports": reports},
    )


# ---------------------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------------------


@app.get("/api/candidates")
async def api_candidates(request: Request) -> List[Dict[str, Any]]:
    """Return all candidates as JSON."""
    cache = _get_cache(request.app)
    candidates = _cached_candidates(cache)
    return [sr.model_dump(mode="json") for sr in candidates]


@app.get("/api/candidates/{ticker}", response_model=None)
async def api_candidate_detail(request: Request, ticker: str) -> Union[Dict[str, Any], JSONResponse]:
    """Return single candidate analysis as JSON."""
    cache = _get_cache(request.app)
    analysis = cache.get_analysis(ticker)
    if analysis is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "ticker": ticker},
        )
    return analysis.model_dump(mode="json")


@app.post("/api/scan")
async def api_trigger_scan() -> Dict[str, Any]:
    """Trigger a new screening scan (stub — actual background execution added with CLI)."""
    return {
        "status": "accepted",
        "message": "Scan started. Results will appear once processing completes.",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/reports")
async def api_reports(request: Request) -> List[Dict[str, str]]:
    """List available report files."""
    config = _get_config(request.app)
    return _list_report_files(config)


@app.get("/api/reports/download/{filename}", response_model=None)
async def api_download_report(request: Request, filename: str) -> Union[FileResponse, JSONResponse]:
    """Serve a report file for download."""
    config = _get_config(request.app)
    filepath = _reports_dir(config) / filename
    if not filepath.is_file() or not filepath.resolve().is_relative_to(_reports_dir(config).resolve()):
        return JSONResponse(status_code=404, content={"error": "not_found", "filename": filename})
    return FileResponse(path=filepath, filename=filename)
