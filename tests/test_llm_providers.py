from __future__ import annotations

import pytest

from valueinvestor.config import AppConfig, load_config
from valueinvestor.analysis.llm_client import LLMClient
from valueinvestor.errors import LLMAuthError

def test_config_supports_base_url():
    cfg = AppConfig()
    cfg.llm.base_url = "https://mock.api/v1"
    assert cfg.llm.base_url == "https://mock.api/v1"

def test_load_config_gemini_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_val")
    cfg_path = tmp_path / "config.yaml"
    # Set provider to gemini in config
    cfg_path.write_text("llm:\n  provider: gemini", encoding="utf-8")
    
    cfg = load_config(str(cfg_path))
    assert cfg.llm.provider == "gemini"
    assert cfg.llm.api_key == "gemini_val"


def test_load_config_deepseek_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_val")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("llm:\n  provider: deepseek", encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.model == "DeepSeek-V4-Flash"
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.api_key == "deepseek_val"


def test_load_config_openrouter_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_val")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("llm:\n  provider: openrouter", encoding="utf-8")
    
    cfg = load_config(str(cfg_path))
    assert cfg.llm.provider == "openrouter"
    assert cfg.llm.api_key == "or_val"


def test_env_override_to_deepseek_uses_provider_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_val")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("llm:\n  provider: openai\n  model: gpt-4o", encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.model == "DeepSeek-V4-Flash"
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.api_key == "deepseek_val"


def test_load_config_env_selection(tmp_path, monkeypatch):
    # Use a realistic-length mock token (real GitHub PATs are >= 40 chars and start with ghp_)
    mock_token = "ghp_" + "A" * 36  # 40 chars, starts with ghp_
    monkeypatch.setenv("LLM_PROVIDER", "github")
    monkeypatch.setenv("LLM_MODEL", "Copilot/Claude Sonnet 4.6")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.github.com/")
    monkeypatch.setenv("GITHUB_API_KEY", mock_token)
    # Ensure GEMINI_API_KEY is unset so the fallback doesn't trigger
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("llm:\n  provider: openai\n  model: gpt-4o", encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert cfg.llm.provider == "github"
    assert cfg.llm.model == "Copilot/Claude Sonnet 4.6"
    assert cfg.llm.base_url == "https://api.github.com/"
    assert cfg.llm.api_key == mock_token

def test_llm_client_uses_base_url():
    cfg = AppConfig()
    cfg.llm.api_key = "test_key"
    cfg.llm.base_url = "https://mock.api/v1"
    
    client = LLMClient(config=cfg.llm)
    assert client._client.base_url == "https://mock.api/v1/"


def test_llm_client_normalizes_deepseek_model_alias():
    cfg = AppConfig()
    cfg.llm.provider = "deepseek"
    cfg.llm.model = "DeepSeek-V4-Flash"
    cfg.llm.api_key = "test_key"
    cfg.llm.base_url = "https://api.deepseek.com/v1"

    client = LLMClient(config=cfg.llm)
    assert client.model == "deepseek-v4-flash"
    assert client._client.base_url == "https://api.deepseek.com/v1/"


def test_llm_client_auth_error_message():
    cfg = AppConfig()
    cfg.llm.provider = "kimi"
    cfg.llm.api_key = ""  # clear it

    with pytest.raises(LLMAuthError) as excinfo:
        LLMClient(config=cfg.llm)
    assert "No API key provided for kimi" in str(excinfo.value)
