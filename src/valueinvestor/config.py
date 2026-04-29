"""YAML-based configuration for the ValueInvestor application."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ScreeningConfig(BaseModel):
    market_cap_min_rmb: int = 5_000_000_000
    pe_max: float = 30.0
    pb_max: float = 5.0
    roe_min: float = 0.08
    debt_ratio_max: float = 0.70
    top_n: int = 20


class InvestmentConfig(BaseModel):
    amount_rmb: int = 500_000
    cycle_months_min: int = 6
    cycle_months_max: int = 24
    target_return_min: float = 0.20
    target_return_max: float = 2.0


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "DeepSeek-V4-Flash"
    api_key: str = ""
    base_url: Optional[str] = "https://api.deepseek.com/v1"
    max_retries: int = 3
    temperature: float = 0.3


class OutputConfig(BaseModel):
    reports_dir: str = "reports"
    formats: List[str] = Field(default_factory=lambda: ["md", "pdf"])
    language: str = "en"


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_hours: int = 24
    db_path: str = "data/cache.db"


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    markets: List[str] = Field(default_factory=lambda: ["a_share", "hk_share"])
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    investment: InvestmentConfig = Field(default_factory=InvestmentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cfg_logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS = {
    "deepseek": {"model": "DeepSeek-V4-Flash", "base_url": "https://api.deepseek.com/v1"},
    "openai": {"model": "gpt-4o", "base_url": None},
    "gemini": {"model": "gemini-2.0-flash", "base_url": None},
    "openrouter": {"model": "deepseek/deepseek-v4-flash", "base_url": "https://openrouter.ai/api/v1"},
    "github": {"model": "claude-sonnet-4.6", "base_url": "https://api.githubcopilot.com"},
    "kimi": {"model": "Kimi Code 2.5", "base_url": "https://api.moonshot.cn/v1"},
    "nvidia_nim": {"model": "minimax-m2-7", "base_url": "https://api.studio.nvidia.com/v1"},
    "local_llm": {"model": "default", "base_url": "http://127.0.0.1:1234"},
}

_PROVIDER_KEYS = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    # GitHub Copilot / GitHub Models: prefer GITHUB_TOKEN (standard GitHub
    # env var), fall back to GITHUB_API_KEY (legacy ValueInvestor naming).
    "github": ("GITHUB_TOKEN", "GITHUB_API_KEY"),
    "kimi": ("KIMI_API_KEY",),
    "nvidia_nim": ("NVIDIA_NIM_API_KEY",),
    "local_llm": (),  # Local LLM doesn't require API key
}


def _apply_provider_defaults(
    cfg: AppConfig,
    *,
    llm_config: Optional[dict] = None,
    provider_changed_by_env: bool = False,
) -> AppConfig:
    defaults = _PROVIDER_DEFAULTS.get(cfg.llm.provider)
    if defaults is None:
        return cfg

    if provider_changed_by_env:
        if not os.environ.get("LLM_MODEL"):
            cfg.llm.model = defaults["model"]
        if not os.environ.get("LLM_BASE_URL"):
            cfg.llm.base_url = defaults["base_url"]
        return cfg

    llm_config = llm_config or {}
    if "model" not in llm_config:
        cfg.llm.model = defaults["model"]
    if "base_url" not in llm_config:
        cfg.llm.base_url = defaults["base_url"]

    return cfg


def _apply_env_overrides(cfg: AppConfig, *, llm_config: Optional[dict] = None) -> AppConfig:
    """Merge selected environment variables over file/default values.

    Auto-fallback: if provider is ``github`` but ``GITHUB_TOKEN`` is not set
    (or is a placeholder), the function automatically switches to ``gemini``
    when ``GEMINI_API_KEY`` is available.  This lets the app run out-of-the-box
    without a Copilot subscription while keeping GitHub Copilot as the
    preferred default when a real token is provided.
    """
    # Override LLM selection via environment variables
    original_provider = cfg.llm.provider
    if env_provider := os.environ.get("LLM_PROVIDER"):
        cfg.llm.provider = env_provider
    provider_changed_by_env = bool(env_provider and env_provider != original_provider)

    cfg = _apply_provider_defaults(
        cfg,
        llm_config=llm_config,
        provider_changed_by_env=provider_changed_by_env,
    )

    if env_model := os.environ.get("LLM_MODEL"):
        cfg.llm.model = env_model
    if env_base_url := os.environ.get("LLM_BASE_URL"):
        cfg.llm.base_url = env_base_url

    env_keys = _PROVIDER_KEYS.get(cfg.llm.provider, ())
    for env_key in env_keys:
        api_key = os.environ.get(env_key)
        if api_key:
            cfg.llm.api_key = api_key
            break

    # Auto-fallback: GitHub provider without any usable key → try Gemini.
    # We check whether cfg.llm.api_key looks like a real GitHub token
    # (starts with a known prefix and is ≥ 36 characters long).  If it's
    # empty or a placeholder, switch to Gemini when GEMINI_API_KEY is set.
    if cfg.llm.provider == "github":
        current_key = cfg.llm.api_key or ""
        _is_real_token = (
            len(current_key) >= 36
            and any(current_key.startswith(pfx) for pfx in ("ghp_", "ghu_", "ghs_", "github_pat_"))
        )
        if not _is_real_token:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            if gemini_key:
                _cfg_logger.warning(
                    "GITHUB_TOKEN not set — auto-falling back to Gemini. "
                    "Set GITHUB_TOKEN in .env to use GitHub Copilot."
                )
                cfg.llm.provider = "gemini"
                cfg.llm.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
                cfg.llm.base_url = None
                cfg.llm.api_key = gemini_key

    return cfg


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Environment variables always take precedence over values found in the file.
    """
    load_dotenv()
    config_path = Path(path)
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Optional[dict] = yaml.safe_load(fh)
        cfg = AppConfig.model_validate(raw) if raw else AppConfig()
    else:
        raw = None
        cfg = AppConfig()

    return _apply_env_overrides(cfg, llm_config=(raw or {}).get("llm"))


# ---------------------------------------------------------------------------
# Default config writer
# ---------------------------------------------------------------------------

_DEFAULT_YAML = """\
# ==========================================================================
# ValueInvestor configuration
# ==========================================================================
# Copy this file to `config.yaml` at the project root and adjust as needed.
# Environment variables (e.g. DEEPSEEK_API_KEY) override values set here.

# Target stock markets to screen
markets:
  - a_share      # China A-shares (SSE + SZSE)
  - hk_share     # Hong Kong Stock Exchange

# Stock screening criteria
screening:
  market_cap_min_rmb: 5000000000   # Minimum market-cap in RMB (5 billion)
  pe_max: 30.0                     # Maximum P/E ratio
  pb_max: 5.0                      # Maximum P/B ratio
  roe_min: 0.08                    # Minimum return on equity (8%)
  debt_ratio_max: 0.70             # Maximum debt-to-asset ratio (70%)
  top_n: 20                        # Number of top picks to keep

# Investment parameters
investment:
  amount_rmb: 500000               # Capital to deploy per cycle (RMB)
  cycle_months_min: 6              # Minimum holding period (months)
  cycle_months_max: 24             # Maximum holding period (months)
  target_return_min: 0.20          # Minimum target return (20%)
  target_return_max: 2.0           # Maximum target return (200%)

# LLM / AI provider settings
llm:
  provider: deepseek              # deepseek, openai, gemini, openrouter, github, kimi, nvidia_nim
  model: DeepSeek-V4-Flash        # Provider defaults apply when model/base_url are omitted
  api_key: ""                      # Leave blank; set appropriate env var instead (e.g. DEEPSEEK_API_KEY)
  base_url: https://api.deepseek.com/v1
  max_retries: 3
  temperature: 0.3

# Report output settings
output:
  reports_dir: reports             # Directory for generated reports
  formats:
    - md                           # Markdown
    - pdf                          # PDF (requires weasyprint)
  language: en                     # Output language (e.g. en, zh-CN)

# Data cache settings
cache:
  enabled: true
  ttl_hours: 24                    # Cache time-to-live in hours
  db_path: data/cache.db           # SQLite cache database path
"""


def save_default_config(path: str = "config.yaml") -> None:
    """Write a default configuration file with explanatory comments."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(_DEFAULT_YAML)
