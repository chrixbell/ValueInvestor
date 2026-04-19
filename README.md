# ValueInvestor 🇨🇳📊

Automated value-investing research tool for **Chinese A-share** and **Hong Kong-share** markets.
ValueInvestor screens thousands of listed companies, scores them with a multi-factor model, runs AI-powered qualitative analysis, and produces ready-to-share investment reports — all from a single CLI command.

---

## Features

- **Quantitative screening** — filters A-share (SSE + SZSE) and HK-share stocks by market cap (>5 B RMB), P/E, P/B, ROE, and debt ratio.
- **Multi-factor scoring** — ranks candidates across value, quality, and growth dimensions.
- **AI qualitative analysis** — uses OpenAI (GPT-4o by default) to evaluate business nature, management quality, competitive moats, market narrative, and investment recommendation for each pick.
- **Report generation** — outputs polished Markdown and PDF reports listing the top 20 candidates with full reasoning.
- **CLI interface** — end-to-end pipeline via the `valueinvestor` command.
- **Web dashboard** — FastAPI-based UI for browsing candidates, viewing analysis detail, and downloading reports.
- **SQLite caching** — avoids redundant API calls; configurable TTL.

---

## Quick Start

```bash
# Clone the repository
git clone <repo-url> ValueInvestor
cd ValueInvestor

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode (with dev extras)
pip install -e ".[dev]"

# Generate a default config file
valueinvestor config init

# Set your OpenAI API key (required for LLM analysis)
export OPENAI_API_KEY="sk-..."

# Run the full pipeline: screen → score → analyse → report
valueinvestor scan

# Or run screening only (no LLM / API key needed)
valueinvestor screen
```

> **Note:** PDF generation requires [WeasyPrint](https://doc.courtbouillon.org/weasyprint/) and its system-level dependencies. On macOS: `brew install pango`.

---

## CLI Commands

The entry point is `valueinvestor` (installed via `pyproject.toml`).

### `valueinvestor scan`

Full pipeline: fetch market data → quantitative screen → multi-factor score → LLM analysis → generate reports.

```bash
valueinvestor scan                        # default (top 20, config.yaml)
valueinvestor scan --top-n 10             # only keep top 10
valueinvestor scan --skip-analysis        # screen + score, skip LLM step
valueinvestor scan -o ./my_reports        # custom output directory
valueinvestor scan -c custom_config.yaml  # use a different config file
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config.yaml` | Path to the YAML configuration file |
| `--top-n` | `-n` | from config (20) | Override number of top picks |
| `--skip-analysis` | | `false` | Skip the LLM analysis step |
| `--output-dir` | `-o` | from config | Override report output directory |

### `valueinvestor screen`

Run quantitative screening only — no LLM calls, no API key required.

```bash
valueinvestor screen
valueinvestor screen --top-n 50
```

### `valueinvestor analyze TICKER`

Deep-dive LLM analysis of a single company. Fetches data, scores, and runs the full AI analysis pipeline for one stock.

```bash
valueinvestor analyze 600519          # A-share (Kweichow Moutai)
valueinvestor analyze 0700.HK         # HK-share (Tencent)
```

### `valueinvestor report`

Regenerate reports from previously cached analysis data (no re-fetching or re-analysis).

```bash
valueinvestor report                      # regenerate MD + PDF
valueinvestor report --format md          # Markdown only
valueinvestor report --format pdf         # PDF only
valueinvestor report -o ./export          # custom output directory
```

### `valueinvestor config show`

Print the currently resolved configuration (file + environment overrides).

```bash
valueinvestor config show
valueinvestor config show -c custom.yaml
```

### `valueinvestor config init`

Generate a default `config.yaml` with explanatory comments.

```bash
valueinvestor config init
valueinvestor config init --path my_config.yaml
```

### `valueinvestor serve`

Launch the web dashboard (FastAPI + Uvicorn).

```bash
valueinvestor serve                       # http://127.0.0.1:8000
valueinvestor serve --host 0.0.0.0 --port 9000
```

---

## Web Dashboard

Start the dashboard with `valueinvestor serve` and open `http://127.0.0.1:8000` in your browser.

| Page | Path | Description |
|---|---|---|
| **Home** | `/` | Scan summary, last run date, and top 5 candidates at a glance |
| **Candidates** | `/candidates` | Full ranked table of all screened candidates with scores and metrics |
| **Detail** | `/candidates/{ticker}` | Single company deep-dive — financials, AI analysis sections, milestones |
| **Reports** | `/reports` | Browse and download generated Markdown / PDF reports |

A JSON API is also available under `/api/`:

- `GET /api/candidates` — all candidates as JSON
- `GET /api/candidates/{ticker}` — single candidate analysis
- `GET /api/reports` — list report files
- `GET /api/reports/download/{filename}` — download a report file
- `POST /api/scan` — trigger a new scan (stub)

---

## Configuration

All settings live in `config.yaml` at the project root. Generate a default one with `valueinvestor config init`.

Environment variables override file values — notably `OPENAI_API_KEY`.

```yaml
# Target markets
markets:
  - a_share          # China A-shares (SSE + SZSE)
  - hk_share         # Hong Kong Stock Exchange

# Screening thresholds
screening:
  market_cap_min_rmb: 5000000000   # 5 billion RMB
  pe_max: 30.0
  pb_max: 5.0
  roe_min: 0.08                    # 8%
  debt_ratio_max: 0.70             # 70%
  top_n: 20

# Investment parameters
investment:
  amount_rmb: 500000
  cycle_months_min: 6
  cycle_months_max: 24
  target_return_min: 0.20          # 20%
  target_return_max: 2.0           # 200%

# LLM provider
llm:
  provider: openai
  model: gpt-4o
  api_key: ""                      # prefer OPENAI_API_KEY env var
  max_retries: 3
  temperature: 0.3

# Report output
output:
  reports_dir: reports
  formats: [md, pdf]

# Data cache (SQLite)
cache:
  enabled: true
  ttl_hours: 24
  db_path: data/cache.db
```

### Key config notes

| Section | Field | Description |
|---|---|---|
| `screening` | `market_cap_min_rmb` | Minimum market capitalisation in RMB |
| `screening` | `top_n` | Number of top candidates to keep after scoring |
| `llm` | `model` | OpenAI model to use (e.g. `gpt-4o`, `gpt-4o-mini`) |
| `llm` | `temperature` | Lower = more deterministic analysis |
| `cache` | `ttl_hours` | How long cached data stays valid |

---

## Architecture

```
src/valueinvestor/
├── cli/                 # Typer CLI commands
│   └── main.py
├── data/                # Market data layer
│   ├── fetcher_ashare.py   # A-share data via akshare
│   ├── fetcher_hkshare.py  # HK-share data via yfinance
│   ├── models.py           # Pydantic domain models
│   └── cache.py            # SQLite cache
├── screener/            # Quantitative screening
│   ├── engine.py           # Screening pipeline
│   └── scorer.py           # Multi-factor scoring
├── analysis/            # AI qualitative analysis
│   ├── llm_client.py       # OpenAI API wrapper
│   ├── pipeline.py         # Analysis orchestration
│   └── prompts.py          # Prompt templates
├── reports/             # Report generation
│   ├── md_generator.py     # Markdown reports
│   └── pdf_generator.py    # PDF reports (WeasyPrint)
├── web/                 # Web dashboard
│   ├── app.py              # FastAPI application
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # CSS / JS assets
├── config.py            # YAML config loading (Pydantic)
└── errors.py            # Custom exceptions
```

### Pipeline flow

```
Market Data (akshare / yfinance)
    ↓
Screening Engine (market cap, PE, PB, ROE, debt filters)
    ↓
Multi-Factor Scorer (value + quality + growth composite)
    ↓
LLM Analysis Pipeline (business, management, moats, narrative, recommendation)
    ↓
Report Generator (Markdown → PDF)
```

---

## Data Sources

| Source | Usage | Notes |
|---|---|---|
| [akshare](https://github.com/akfamily/akshare) | A-share market data (SSE + SZSE) | Free, no API key needed |
| [yfinance](https://github.com/ranaroussi/yfinance) | HK-share market data (HKEX) | Free, no API key needed |
| [OpenAI API](https://platform.openai.com/) | Qualitative analysis (GPT-4o) | Requires `OPENAI_API_KEY`; billed per token |

---

## Development

### Prerequisites

- Python ≥ 3.9
- (Optional) System dependencies for PDF: [WeasyPrint requirements](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)

### Install dev dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
pytest --cov=valueinvestor          # with coverage
```

### Lint

```bash
ruff check src/ tests/
ruff format --check src/ tests/     # formatting check
```

### Project layout

```
ValueInvestor/
├── config.yaml          # User configuration
├── pyproject.toml       # Build & dependency config
├── mission.md           # Investment thesis / goals
├── src/valueinvestor/   # Application source
├── tests/               # Test suite
├── data/                # Runtime data (cache DB)
└── reports/             # Generated reports
```

---

## Disclaimer

> **This tool is for educational and personal research purposes only.**
> Nothing produced by ValueInvestor constitutes financial advice, a solicitation, or a recommendation to buy or sell any securities.
> Investment decisions should be based on your own due diligence and, where appropriate, the advice of a qualified financial adviser.
> The authors accept no liability for any losses arising from the use of this tool.

---

## License

See [LICENSE](LICENSE) for details.
