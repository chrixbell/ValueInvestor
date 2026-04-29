from __future__ import annotations

import yaml

from valueinvestor.config import AppConfig, load_config, save_default_config


class TestLoadConfigDefaults:
    """load_config with no file on disk returns default AppConfig."""

    def test_returns_defaults_when_no_file(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.screening.pe_max == 30.0
        assert cfg.screening.top_n == 20
        assert cfg.markets == ["a_share", "hk_share"]

    def test_default_cache_config(self, tmp_path):
        cfg = load_config(str(tmp_path / "missing.yaml"))
        assert cfg.cache.enabled is True
        assert cfg.cache.ttl_hours == 24

    def test_default_llm_is_deepseek(self, tmp_path, monkeypatch):
        for env_key in (
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "GITHUB_TOKEN",
            "GITHUB_API_KEY",
            "KIMI_API_KEY",
            "NVIDIA_NIM_API_KEY",
        ):
            monkeypatch.setenv(env_key, "")

        cfg = load_config(str(tmp_path / "missing.yaml"))
        assert cfg.llm.provider == "deepseek"
        assert cfg.llm.model == "DeepSeek-V4-Flash"
        assert cfg.llm.base_url == "https://api.deepseek.com/v1"


class TestLoadConfigFromFile:
    """load_config correctly reads values from a YAML file."""

    def test_custom_values(self, tmp_path):
        content = {
            "markets": ["a_share"],
            "screening": {"pe_max": 15.0, "top_n": 10},
            "investment": {"amount_rmb": 1_000_000},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(content), encoding="utf-8")

        cfg = load_config(str(cfg_path))
        assert cfg.markets == ["a_share"]
        assert cfg.screening.pe_max == 15.0
        assert cfg.screening.top_n == 10
        assert cfg.investment.amount_rmb == 1_000_000
        # Unset values keep defaults
        assert cfg.screening.roe_min == 0.08

    def test_empty_yaml_returns_defaults(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("", encoding="utf-8")
        cfg = load_config(str(cfg_path))
        assert cfg.screening.pe_max == 30.0

    def test_env_override_api_key(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        # Explicitly use openai provider
        cfg_path.write_text(
            yaml.dump({"llm": {"provider": "openai", "api_key": "file_key"}}),
            encoding="utf-8",
        )
        # Must SET (not delete) these to prevent load_dotenv() re-reading .env
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("OPENAI_API_KEY", "env_key")
        # Ensure GITHUB_API_KEY is unset (load_dotenv won't re-add since LLM_PROVIDER is set)
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setenv("GITHUB_API_KEY", "")

        cfg = load_config(str(cfg_path))
        assert cfg.llm.api_key == "env_key"


class TestSaveDefaultConfig:
    """save_default_config writes a valid, parseable YAML file."""

    def test_creates_file(self, tmp_path):
        out = tmp_path / "defaults.yaml"
        save_default_config(str(out))
        assert out.exists()

    def test_content_is_valid_yaml(self, tmp_path):
        out = tmp_path / "defaults.yaml"
        save_default_config(str(out))
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "screening" in data

    def test_roundtrip_through_load(self, tmp_path):
        out = tmp_path / "defaults.yaml"
        save_default_config(str(out))
        cfg = load_config(str(out))
        assert cfg.screening.pe_max == 30.0
        assert cfg.cache.ttl_hours == 24

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "subdir" / "deep" / "config.yaml"
        save_default_config(str(out))
        assert out.exists()
