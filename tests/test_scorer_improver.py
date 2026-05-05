"""Tests for the scorer_improver sub-package.

Tests cover experiment_log, evaluator, agent, and ground_truth helpers.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd


# ── ExperimentLog tests ─────────────────────────────────────────────────────


class TestExperimentLog:
    """Tests for ExperimentLog JSONL logger."""

    def test_log_creates_file(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.25, baseline_rho=0.20, kept=True, description="test")
        assert log_file.exists()
        records = log.read_all()
        assert len(records) == 1
        assert records[0]["iteration"] == 1
        assert records[0]["spearman_rho"] == 0.25
        assert records[0]["kept"] is True
        assert records[0]["delta"] == 0.05

    def test_log_preserves_zero_6m_fields(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.0, baseline_rho=0.0, kept=False, description="zero")
        record = log.read_all()[0]
        assert record["spearman_rho_6m"] == 0.0
        assert record["baseline_rho_6m"] == 0.0

    def test_read_last_n(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        for i in range(20):
            log.log(iteration=i, spearman_rho=0.1 * i, baseline_rho=0.0, kept=True, description=f"iter-{i}")
        assert log.total_experiments() == 20
        last5 = log.read_last_n(5)
        assert len(last5) == 5
        assert last5[0]["iteration"] == 15
        assert last5[-1]["iteration"] == 19

    def test_best_rho(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.1, baseline_rho=0.0, kept=True, description="a")
        log.log(iteration=2, spearman_rho=0.5, baseline_rho=0.1, kept=True, description="b")
        log.log(iteration=3, spearman_rho=0.3, baseline_rho=0.5, kept=False, description="c")
        assert log.best_rho() == 0.5

    def test_best_rho_no_kept(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.1, baseline_rho=0.2, kept=False, description="x")
        assert log.best_rho() is None

    def test_current_lineage_ignores_older_branch(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.10, baseline_rho=0.00, kept=True, description="a")
        log.log(iteration=2, spearman_rho=0.50, baseline_rho=0.10, kept=True, description="b")
        log.log(iteration=3, spearman_rho=0.20, baseline_rho=0.20, kept=True, description="manual reset")
        log.log(iteration=4, spearman_rho=0.18, baseline_rho=0.20, kept=False, description="reverted")
        log.log(iteration=5, spearman_rho=0.30, baseline_rho=0.20, kept=True, description="improved")

        current = log.read_current_lineage()
        assert [record["iteration"] for record in current] == [3, 4, 5]
        assert log.read_last_n(2, current_lineage=True)[0]["iteration"] == 4
        assert log.best_rho(current_lineage=True) == 0.30
        assert log.best_rho() == 0.50

    def test_best_rho_filters_by_ground_truth_id(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import experiment_log
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        gt_id = "ground_truth.parquet:1:100"
        monkeypatch.setattr(
            "valueinvestor.scorer_improver.ground_truth.ground_truth_fingerprint",
            lambda: gt_id,
        )

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.20, baseline_rho=0.10, kept=True, description="current")

        old_record = log.read_all()[0] | {
            "iteration": 2,
            "spearman_rho": 0.50,
            "spearman_rho_1m": 0.40,
            "spearman_rho_3m": 0.45,
            "ground_truth_id": "old-ground-truth",
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(experiment_log.json.dumps(old_record) + "\n")

        assert log.best_rho(ground_truth_id=gt_id) == 0.20
        assert log.best_rho(ground_truth_id="old-ground-truth") == 0.50
        assert log.best_rho_per_horizon(ground_truth_id=gt_id)["6m"] == 0.20

    def test_empty_log(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        assert log.total_experiments() == 0
        assert log.read_last_n(5) == []
        assert log.best_rho() is None
        assert log.read_all() == []


# ── Agent helper tests ──────────────────────────────────────────────────────


class TestAgentHelpers:
    """Tests for agent.py helper functions (code extraction, diff, etc.)."""

    def test_get_llm_client_uses_deepseek_config(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver.agent import _get_llm_client

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url=None, max_retries=0):
                self.api_key = api_key
                self.base_url = f"{base_url.rstrip('/')}/" if base_url else None
                self.max_retries = max_retries
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kwargs: None)
                )

        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_MODEL", "DeepSeek-V4-Flash")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_test_key")

        with patch("valueinvestor.analysis.llm_client.OpenAI", FakeOpenAI):
            client = _get_llm_client()

        assert client.provider == "deepseek"
        assert client.model == "deepseek-v4-flash"
        assert client._client.base_url == "https://api.deepseek.com/v1/"

    def test_run_improvement_loop_smoke_with_deepseek(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        scorer_code = textwrap.dedent("""\
        class MultiFactorScorer:
            def score(self, result):
                return result
        """).strip()
        program_md = tmp_path / "program.md"
        program_md.write_text(
            "Current:\n{current_status}\n\nHistory:\n{experiment_history}\n",
            encoding="utf-8",
        )
        exp_path = tmp_path / "experiments.jsonl"

        class TempExperimentLog(ExperimentLog):
            def __init__(self, path=None) -> None:
                super().__init__(path=exp_path)

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url=None, max_retries=0):
                self.api_key = api_key
                self.base_url = f"{base_url.rstrip('/')}/" if base_url else None
                self.max_retries = max_retries
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                return SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=f"```python\n{scorer_code}\n```\nNo changes needed."
                            )
                        )
                    ],
                )

        class FakeLLMClient:
            """Fake LLMClient with .provider attribute for the parallel path."""
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None):
                return f"```python\n{scorer_code}\n```\nNo changes needed."

        class FakeEvalContext:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def evaluate_all_targets(self, scorer_module=None, scorer_path=None):
                return {
                    "1m": {"spearman_rho": 0.12, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                    "3m": {"spearman_rho": 0.23, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                    "6m": {"spearman_rho": 0.34, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                }

            def quick_evaluate(self, scorer_module=None, scorer_path=None, sample_size=1000):
                return {"1m": 0.11, "3m": 0.22, "6m": 0.33}

        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_MODEL", "DeepSeek-V4-Flash")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_test_key")
        monkeypatch.setenv("IMPROVE_SCORER_PARALLEL", "1")
        monkeypatch.setattr(agent, "PROGRAM_MD_PATH", program_md)
        monkeypatch.setattr(agent, "ExperimentLog", TempExperimentLog)
        monkeypatch.setattr(
            agent,
            "_create_parallel_clients",
            lambda: [FakeLLMClient()],
        )
        monkeypatch.setattr(agent, "ScorerEvaluationContext", FakeEvalContext)
        monkeypatch.setattr(agent, "_read_scorer", lambda: scorer_code)
        writes: list[str] = []
        monkeypatch.setattr(agent, "_write_scorer", lambda code: writes.append(code))
        monkeypatch.setattr(agent, "_validate_scorer", lambda **kwargs: True)

        with patch("valueinvestor.analysis.llm_client.OpenAI", FakeOpenAI):
            agent.run_improvement_loop(max_iterations=1)

        records = ExperimentLog(path=exp_path).read_all()
        assert len(records) == 1
        assert records[0]["kept"] is False
        assert records[0]["description"]  # should have a reason logged
        assert writes == []

    def test_extract_code_python_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = textwrap.dedent("""\
        Here's the improved scorer:
        ```python
        class MultiFactorScorer:
            pass
        ```
        This improves the scoring by ...
        """)
        code = _extract_code(response)
        assert code is not None
        assert "class MultiFactorScorer:" in code

    def test_semantic_failure_theme_does_not_block_normal_pe_safe_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = "Removed the pe_safe scaling from the quality raw metric to avoid valuation overlap."

        assert _semantic_failure_theme(text) == ""

    def test_semantic_failure_theme_blocks_pe_safe_bug_repairs(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = "Fixed the undefined pe_safe NameError in _compute_quality_raw."

        assert _semantic_failure_theme(text) == "pe_safe_quality_raw_repair"

    def test_semantic_failure_theme_blocks_protected_value_removals(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        assert (
            _semantic_failure_theme("Removed the price × market_cap sub-factor from value scoring")
            == "remove_price_market_cap_subfactor"
        )
        assert (
            _semantic_failure_theme("Drop the price * PB interaction because it looks noisy")
            == "remove_price_pb_subfactor"
        )
        assert (
            _semantic_failure_theme("Delete the dividend-yield triangular sub-factor")
            == "remove_dividend_yield_subfactor"
        )
        assert (
            _semantic_failure_theme("Raised the ideal price * pb threshold in the log-score")
            == "widen_price_pb_threshold"
        )
        assert (
            _semantic_failure_theme("Switched the price × PB sub-factor scoring from logarithmic to linear")
            == "price_pb_shape_retry"
        )
        assert (
            _semantic_failure_theme("_log_score(ppb, best=4.0, worst=250.0)")
            == "price_pb_shape_retry"
        )
        assert (
            _semantic_failure_theme("Raised the quality ROE cap from 0.30 to 0.40")
            == "raise_quality_roe_cap"
        )
        assert (
            _semantic_failure_theme("Fix duplicate weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))")
            == "remove_earnings_yield_zero_append"
        )
        assert (
            _semantic_failure_theme("Replaced min(value_score, quality_score) synergy term with geometric mean")
            == "replace_synergy_min_retry"
        )
        assert (
            _semantic_failure_theme("Increased the influence of the absolute growth score from 0.3 to 0.4")
            == "growth_blend_weight_retry"
        )
        assert (
            _semantic_failure_theme("Increased quality_score cross-sectional percentile component from 0.8 to 0.9")
            == "quality_percentile_blend_retry"
        )
        assert _semantic_failure_theme("Reduced boost_alpha from 1.55 to 1.45") == "value_boost_alpha_retry"
        assert (
            _semantic_failure_theme("Added a gross-profit-yield (gross profit / market cap) value factor")
            == "gross_profit_market_cap_yield_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the market-cap gate for the gross-profit-to-assets value sub-factor")
            == "gross_profit_assets_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the linear scoring of asset turnover in _momentum_score")
            == "asset_turnover_retry"
        )
        assert (
            _semantic_failure_theme("Raised the best threshold of the operating-cash-flow yield")
            == "ocf_yield_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the log‑scale scoring of operating‑cash‑flow yield")
            == "ocf_yield_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Shifted the optimal EV/EBITDA from 8.0 to 9.0")
            == "ev_ebitda_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Widened the PEG scoring reference window")
            == "peg_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Increased the loss penalty scaling constant from 5.0 to 6.0")
            == "loss_penalty_scaling_retry"
        )
        assert (
            _semantic_failure_theme("The duplicate zero-score entry now uses a smaller weight")
            == "earnings_yield_zero_weight_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap composite score bonus gate")
            == "large_cap_composite_bonus_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the best threshold for the ROA value sub-factor from 0.15 to 0.12")
            == "roa_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap post-percentile quality_score ROA bonus")
            == "large_cap_quality_bonus_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap-gated direct ROE value sub-factor")
            == "large_cap_direct_roe_value_retry"
        )
        assert (
            _semantic_failure_theme(
                "Lowered the market-cap gate for the liabilities-to-market-cap value sub-factor"
            )
            == "liabilities_market_cap_gate_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the linear scoring of roe_ey with a logarithmic shape")
            == "roe_ev_ebitda_value_shape_retry"
        )
        assert (
            _semantic_failure_theme("Added a fallback to pe_forward when PEG is unavailable")
            == "growth_forward_pe_fallback_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the best threshold for the forward P/E log-score")
            == "forward_pe_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Raised the forward_pe_improve linear_score best threshold")
            == "forward_pe_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Added a value_score > 60 and quality_score > 60 composite_score bonus")
            == "value_quality_composite_bonus_retry"
        )

    def test_semantic_failure_theme_ignores_unchanged_diff_context(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = """\
--- old
+++ new
@@ -442,2 +442,12 @@
                     weighted_scores.append((0.0, CURRENT_RATIO_WEIGHT))
+        # ROE / PB - profitable book value
+        roe_pb = roe_val / pb_val
Added a new value sub-factor ROE / PB that rewards profitability relative to book value.
"""

        assert _semantic_failure_theme(text) == ""

    def test_best_scorer_snapshot_round_trip(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("BEST = True\n", encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        agent._save_best_scorer_snapshot(
            iteration=42,
            rhos={"1m": 0.11, "3m": 0.12, "6m": 0.13},
            ground_truth_id="gt-current",
        )
        scorer_path.write_text("BEST = False\n", encoding="utf-8")

        meta = agent._restore_best_scorer_snapshot("gt-current")

        assert scorer_path.read_text(encoding="utf-8") == "BEST = True\n"
        assert meta is not None
        assert meta["iteration"] == 42
        assert json.loads(meta_path.read_text(encoding="utf-8"))["spearman_rho_6m"] == 0.13

    def test_cleanup_old_backups_preserves_tracked_files(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        tracked = backup_dir / "scorer_0000.py"
        old_untracked = backup_dir / "scorer_0001.py"
        new_untracked = backup_dir / "scorer_0002.py"
        for path in (tracked, old_untracked, new_untracked):
            path.write_text(path.name, encoding="utf-8")

        monkeypatch.setattr(agent, "BACKUP_DIR", backup_dir)
        monkeypatch.setattr(agent, "_MAX_BACKUPS", 1)
        monkeypatch.setattr(agent, "_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            agent.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout="backups/scorer_0000.py\n",
            ),
        )

        assert agent._cleanup_old_backups() == 1
        assert tracked.exists()
        assert not old_untracked.exists()
        assert new_untracked.exists()

    def test_best_scorer_snapshot_ignores_other_ground_truth(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("CURRENT = True\n", encoding="utf-8")
        best_path.write_text("BEST = True\n", encoding="utf-8")
        meta_path.write_text(json.dumps({"ground_truth_id": "old-gt"}), encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        assert agent._restore_best_scorer_snapshot("new-gt") is None
        assert scorer_path.read_text(encoding="utf-8") == "CURRENT = True\n"

    def test_load_resume_baseline_uses_snapshot_without_mutating_worktree(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("WORKTREE = True\n", encoding="utf-8")
        best_path.write_text("BEST = True\n", encoding="utf-8")
        meta_path.write_text(json.dumps({"ground_truth_id": "gt-current"}), encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        meta, code, from_snapshot = agent._load_resume_baseline("gt-current")

        assert from_snapshot is True
        assert meta == {"ground_truth_id": "gt-current"}
        assert code == "BEST = True\n"
        assert scorer_path.read_text(encoding="utf-8") == "WORKTREE = True\n"

    def test_historical_snapshot_guard_only_saves_true_bests(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_better_historical_snapshot

        best = {"1m": 0.11, "3m": 0.12, "6m": 0.13}

        assert _is_better_historical_snapshot(
            {"1m": 0.1102, "3m": 0.119, "6m": 0.129},
            best,
        )
        assert not _is_better_historical_snapshot(
            {"1m": 0.11001, "3m": 0.119, "6m": 0.129},
            best,
        )
        assert not _is_better_historical_snapshot(
            {"1m": 0.10, "3m": 0.119, "6m": 0.129},
            best,
        )

    def test_print_skip_banner_shows_each_horizon(self, capsys) -> None:
        from valueinvestor.scorer_improver.agent import _print_skip_banner

        rhos = {"1m": 0.0455, "3m": 0.0878, "6m": 0.1035}
        best = {"1m": 0.1646, "3m": 0.1771, "6m": 0.2430}

        _print_skip_banner(7992, rhos, best, "no improvement")

        output = capsys.readouterr().out
        assert "Iteration #7992  |  ❌ REVERTED (no improvement)" in output
        assert "1-month  ρ=0.0455  |  Best: 0.1646" in output
        assert "3-month  ρ=0.0878  |  Best: 0.1771" in output
        assert "6-month  ρ=0.1035  |  Best: 0.2430" in output

    def test_extract_code_no_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = "No code block here, just plain text."
        code = _extract_code(response)
        assert code is None

    def test_extract_code_generic_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = textwrap.dedent("""\
        ```
        print("hello")
        ```
        """)
        code = _extract_code(response)
        assert code is not None
        assert "print" in code

    def test_extract_patch_diff_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_patch

        response = textwrap.dedent("""\
        ```diff
        --- a
        +++ b
        @@ -1 +1 @@
        -old
        +new
        ```
        Changed one line.
        """)
        patch = _extract_patch(response)
        assert patch is not None
        assert "@@ -1 +1 @@" in patch
        assert "+new" in patch

    def test_extract_proposal_code_prefers_patch(self) -> None:
        from valueinvestor.scorer_improver.agent import (
            _compute_full_diff,
            _extract_proposal_code,
        )

        old = "def score():\n    return 1\n"
        new = "def score():\n    return 2\n"
        patch = _compute_full_diff(old, new)
        response = f"```diff\n{patch}```\nUse a stronger score."

        code, explanation, diff_summary = _extract_proposal_code(response, old)

        assert code == new
        assert "stronger" in explanation.lower()
        assert "return 2" in diff_summary

    def test_extract_proposal_code_handles_indented_diff(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_proposal_code

        old = "def score():\n    return 1\n"
        response = textwrap.dedent("""\
            ```diff
                --- a
                +++ b
                @@ -1,2 +1,2 @@
                 def score():
                -    return 1
                +    return 4
            ```
            Repaired patch.
        """)

        code, _, diff_summary = _extract_proposal_code(response, old)

        assert code == "def score():\n    return 4\n"
        assert "return 4" in diff_summary

    def test_extract_proposal_code_rejects_full_file_fallback(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_proposal_code

        old = "def score():\n    return 1\n"
        new = "def score():\n    return 3\n"
        response = f"```python\n{new}```\nFallback full file."

        code, explanation, diff_summary = _extract_proposal_code(response, old)

        assert code is None
        assert "fallback" in explanation.lower()
        assert diff_summary == ""

    def test_extract_explanation(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_explanation

        response = textwrap.dedent("""\
        ```python
        class Foo:
            pass
        ```
        I changed the weights to be more balanced.
        """)
        explanation = _extract_explanation(response)
        assert "weights" in explanation.lower() or "balanced" in explanation.lower()

    def test_compute_diff_summary(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_diff_summary

        old = "line1\nline2\nline3"
        new = "line1\nmodified\nline3"
        diff = _compute_diff_summary(old, new)
        assert "line2" in diff or "modified" in diff
        assert len(diff) <= 500

    def test_compute_diff_no_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_diff_summary

        code = "same code"
        diff = _compute_diff_summary(code, code)
        assert "no changes" in diff.lower()

    def test_compute_full_diff(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff

        old = "line1\nline2\nline3\n"
        new = "line1\nmodified\nline3\n"
        diff = _compute_full_diff(old, new)
        assert "line2" in diff or "modified" in diff

    def test_try_apply_patch_applies_clean_change(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        old = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        new = "def foo():\n    x = 1\n    y = 3\n    return x + y\n"
        patch = _compute_full_diff(old, new)
        result = _try_apply_patch(old, patch)
        assert result == new

    def test_try_apply_patch_stacks_independent_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        base = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        change1 = "def a():\n    return 10\n\ndef b():\n    return 2\n"
        change2 = "def a():\n    return 1\n\ndef b():\n    return 20\n"

        # Apply change1
        patch1 = _compute_full_diff(base, change1)
        after1 = _try_apply_patch(base, patch1)
        assert after1 == change1

        # Stack change2 on top
        patch2 = _compute_full_diff(base, change2)  # diff vs original base
        stacked = _try_apply_patch(after1, patch2)
        expected = "def a():\n    return 10\n\ndef b():\n    return 20\n"
        assert stacked == expected

    def test_try_apply_patch_tolerates_stale_line_numbers(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        base = (
            "def first():\n"
            "    return 1\n"
            "\n"
            "def target():\n"
            "    score = 2\n"
            "    return score\n"
        )
        patch = textwrap.dedent("""\
            --- a/src/valueinvestor/screener/scorer.py
            +++ b/src/valueinvestor/screener/scorer.py
            @@ -1,3 +1,3 @@
             def target():
            -    score = 2
            +    score = 5
        """)

        result = _try_apply_patch(base, patch)

        assert result is not None
        assert "score = 5" in result
        assert "def first" in result

    def test_try_apply_patch_tolerates_omitted_blank_context(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        base = (
            "def quality():\n"
            "    pe = result.valuation.pe_ratio\n"
            "\n"
            "    # Require all inputs\n"
            "    if pe is None:\n"
            "        return None\n"
        )
        patch = textwrap.dedent("""\
            --- a/scorer.py
            +++ b/scorer.py
            @@ -1,5 +1,6 @@
             def quality():
                 pe = result.valuation.pe_ratio
            +    current_ratio = result.financials.current_ratio
                 # Require all inputs
                 if pe is None:
                     return None
        """)

        result = _try_apply_patch(base, patch)

        assert result is not None
        assert "current_ratio" in result

    def test_try_apply_patch_returns_none_on_context_mismatch(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        old = "def foo():\n    return 1\n"
        new = "def foo():\n    return 2\n"
        patch = _compute_full_diff(old, new)
        different_base = "def bar():\n    return 3\n"
        assert _try_apply_patch(different_base, patch) is None

    def test_try_apply_patch_returns_none_without_hunks(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        assert _try_apply_patch("x = 1\n", "This is not a patch") is None

    def test_records_for_prompt_skips_no_patch_noise(self) -> None:
        from valueinvestor.scorer_improver.agent import _records_for_prompt

        records = [
            {"iteration": 1, "kept": False, "description": "prose only", "diff_summary": ""},
            {"iteration": 2, "kept": False, "description": "actual patch", "diff_summary": "--- old\n+++ new\n"},
        ]

        assert [r["iteration"] for r in _records_for_prompt(records, 10)] == [2]

    def test_failure_guidance_summarizes_quick_reject_patterns(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_failure_guidance

        records = [
            {
                "kept": False,
                "quick_reject_reason": "1m-only tradeoff",
                "quick_material_improved_horizons": ["1m"],
                "quick_material_degraded_horizons": ["3m", "6m"],
            },
            {
                "kept": False,
                "quick_reject_reason": "1m-only tradeoff",
                "quick_material_improved_horizons": ["1m"],
                "quick_material_degraded_horizons": ["6m"],
            },
        ]

        guidance = _compute_failure_guidance(records)

        assert "1m-only tradeoff" in guidance
        assert "1m upside paired with 3m/6m damage" in guidance

    def test_near_miss_guidance_targets_1m_repair(self) -> None:
        from valueinvestor.scorer_improver.agent import _near_miss_guidance

        records = [
            {
                "iteration": 13743,
                "kept": False,
                "near_miss": True,
                "current_material_improved_horizons": ["3m", "6m"],
                "current_material_degraded_horizons": ["1m"],
                "current_delta_1m": -0.001176,
                "current_delta_3m": 0.000453,
                "current_delta_6m": 0.000106,
                "diff_summary": "PB best threshold 20.0 -> 22.0",
            },
        ]

        guidance = _near_miss_guidance(records)

        assert "Near-miss repair objective" in guidance
        assert "repairing 1m damage" in guidance
        assert "#13743" in guidance
        assert "Δ3m=+0.0005" in guidance

    def test_failure_guidance_promotes_blocked_themes_to_hard_exclusions(self) -> None:
        from valueinvestor.scorer_improver.agent import _blocked_failure_themes, _compute_failure_guidance

        records = [
            {
                "kept": False,
                "description": "Added FCF/TA value sub-factor",
                "diff_summary": "fcf / total_assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Added free cash flow to total assets yield",
                "diff_summary": "free cash flow to total assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Reward FCF/equity in value score",
                "diff_summary": "fcf_equity_yield",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        assert _blocked_failure_themes(records) == ["fcf_assets_or_equity_retry"]
        guidance = _compute_failure_guidance(records)
        assert "Hard proposal exclusions" in guidance
        assert "FCF/assets, FCF/equity, or FCF/debt retry variants" in guidance

    def test_failure_guidance_uses_wider_blocked_theme_window(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_failure_guidance

        blocked_records = [
            {
                "kept": False,
                "description": "Lowered the best threshold for forward P/E",
                "diff_summary": "pe_forward best threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Raised the forward_pe_improve linear_score best threshold",
                "diff_summary": "forward_pe_improve threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Adjusted forward-pe improvement threshold",
                "diff_summary": "forward-pe improvement threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        guidance = _compute_failure_guidance([], blocked_theme_records=blocked_records)

        assert "Hard proposal exclusions" in guidance
        assert "forward-PE value threshold" in guidance

    def test_llm_complete_passes_request_timeout_when_supported(self) -> None:
        from valueinvestor.scorer_improver.agent import _llm_complete

        class FakeLLM:
            def __init__(self) -> None:
                self.timeout = None

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None, timeout=None):
                self.timeout = timeout
                return "ok"

        fake = FakeLLM()

        assert _llm_complete(fake, "prompt", 10, request_timeout=2.5) == "ok"
        assert fake.timeout == 2.5

    def test_llm_complete_falls_back_for_test_doubles_without_timeout(self) -> None:
        from valueinvestor.scorer_improver.agent import _llm_complete

        class FakeLLM:
            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return "ok"

        assert _llm_complete(FakeLLM(), "prompt", 10, request_timeout=2.5) == "ok"

    def test_llm_complete_empty_retry_stays_bounded(self) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None, timeout=None):
                self.calls.append(max_tokens)
                return "" if len(self.calls) == 1 else "ok"

        fake = FakeLLM()

        assert agent._llm_complete(fake, "prompt", 6000) == "ok"
        assert fake.calls == [6000, agent._MAX_REPAIR_TOKENS]

    def test_repeated_failed_theme_detects_pe_safe_repairs(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Fixed undefined pe_safe in _compute_quality_raw",
                "diff_summary": "pe_safe",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Repair broken quality raw bug",
                "diff_summary": "_compute_quality_raw",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Remove fatal bug in quality raw",
                "diff_summary": "undefined",
                "delta_1m": 0.0,
                "delta_3m": 0.0,
                "delta": 0.0,
            },
        ]

        theme = _is_repeated_failed_theme(
            "replace pe_safe with pe ratio",
            "fix undefined _compute_quality_raw",
            recent,
        )

        assert theme == "pe_safe_quality_raw_repair"

    def test_repeated_failed_theme_detects_protected_value_removals(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Removed price × market_cap from value score",
                "diff_summary": "- PRICE_MARKET_CAP_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Drop price * market_cap because it overlaps",
                "diff_summary": "- pmc = price * market_cap",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Remove price_market_cap_weight sub-factor",
                "diff_summary": "PRICE_MARKET_CAP_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "- PRICE_MARKET_CAP_WEIGHT",
            "Removed the price x market_cap sub-factor",
            recent,
        )

        assert theme == "remove_price_market_cap_subfactor"

    def test_repeated_failed_theme_detects_earnings_yield_zero_append_removal(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Removed duplicate earnings yield zero append",
                "diff_summary": "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.03,
            },
            {
                "kept": False,
                "description": "Delete redundant earnings_yield_weight zero branch",
                "diff_summary": "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Fix double earnings yield zero append",
                "diff_summary": "EARNINGS_YIELD_WEIGHT duplicate",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
            "Remove duplicate earnings yield zero append",
            recent,
        )

        assert theme == "remove_earnings_yield_zero_append"

    def test_repeated_failed_theme_detects_fcf_assets_retries(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Added FCF/TA value sub-factor",
                "diff_summary": "FCF_TO_ASSETS_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Added free cash flow to total assets yield",
                "diff_summary": "fcf / total_assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Reward FCF/equity in value score",
                "diff_summary": "fcf_equity_yield",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "fcf_ta = fcf / ta",
            "Add a free cash flow to total assets value factor",
            recent,
        )

        assert theme == "fcf_assets_or_equity_retry"

    def test_repeated_failed_theme_detects_current_ratio_retries(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Added current-ratio triangular bonus",
                "diff_summary": "current_ratio",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Large-cap gated current ratio factor",
                "diff_summary": "CURRENT_RATIO_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Quality raw current ratio gate",
                "diff_summary": "cr = result.financials.current_ratio",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "current_ratio = result.financials.current_ratio",
            "Add a current ratio bonus",
            recent,
        )

        assert theme == "current_ratio_retry"

    def test_behavior_neutral_detects_tiny_full_eval_deltas(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_behavior_neutral

        baseline = {"1m": 0.10, "3m": 0.20, "6m": 0.30}

        assert _is_behavior_neutral({"1m": 0.10002, "3m": 0.19998, "6m": 0.30001}, baseline)
        assert not _is_behavior_neutral({"1m": 0.1002, "3m": 0.20, "6m": 0.30}, baseline)

    def test_agent_prompt_describes_current_pareto_acceptance_gate(self) -> None:
        from valueinvestor.scorer_improver.agent import _AGENT_SYSTEM_PROMPT, _build_prompt

        assert "ANY of the three horizons" not in _AGENT_SYSTEM_PROMPT
        assert "clears the configured material" in _AGENT_SYSTEM_PROMPT
        assert "avoiding material degradation" in _AGENT_SYSTEM_PROMPT
        assert "complete replacement file" in _AGENT_SYSTEM_PROMPT

        prompt = _build_prompt(
            scorer_code="class MultiFactorScorer:\n    pass\n",
            program_md="Status:\n{current_status}\n\nHistory:\n{experiment_history}\n",
            experiment_history="No history",
            baseline_rhos={"1m": 0.10, "3m": 0.20, "6m": 0.30},
            best_rhos={"1m": 0.10, "3m": 0.20, "6m": 0.30},
        )

        assert "Acceptance gate: improve at least one horizon by more than 0.0001" in prompt
        assert "Do not output markdown-only commentary" in prompt
        assert "full replacement file" in prompt
        assert "```python fence" in prompt

    def test_default_weight_guard_accepts_full_six_factor_weights(self) -> None:
        from valueinvestor.scorer_improver.agent import _default_weight_issues

        code = """
_DEFAULT_WEIGHTS = {
    "value": 0.40,
    "quality": 0.20,
    "growth": 0.10,
    "momentum": 0.10,
    "synergy": 0.10,
    "value_growth": 0.10,
}
"""

        assert _default_weight_issues(code) == []

    def test_default_weight_guard_rejects_partial_weights(self) -> None:
        from valueinvestor.scorer_improver.agent import _default_weight_issues

        code = """
_DEFAULT_WEIGHTS = {
    "value": 0.70,
    "quality": 0.20,
    "growth": 0.10,
}
"""

        issues = _default_weight_issues(code)

        assert any("momentum" in issue for issue in issues)
        assert any("synergy" in issue for issue in issues)

    def test_quick_reject_diagnostics_rejects_1m_only_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1370, "3m": 0.1370, "6m": 0.1660}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "1m-only tradeoff"
        assert diagnostics["material_improved_horizons"] == ["1m"]
        assert diagnostics["material_degraded_horizons"] == ["3m", "6m"]

    def test_quick_reject_diagnostics_keeps_multi_horizon_upside(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1352, "3m": 0.1418, "6m": 0.1701}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is False
        assert diagnostics["material_improved_horizons"] == ["3m", "6m"]

    def test_quick_reject_diagnostics_rejects_severe_1m_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1368, "3m": 0.1436, "6m": 0.1724}
        candidate = {"1m": 0.1345, "3m": 0.1458, "6m": 0.1721}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "1m tradeoff"
        assert diagnostics["material_improved_horizons"] == ["3m"]
        assert "1m" in diagnostics["severe_degraded_horizons"]

    def test_quick_reject_diagnostics_tolerates_sample_noise(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1351, "3m": 0.1407, "6m": 0.1688}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is False
        assert diagnostics["material_improved_horizons"] == []
        assert diagnostics["material_degraded_horizons"] == []

    def test_quick_reject_diagnostics_rejects_no_sample_upside(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1368, "3m": 0.1436, "6m": 0.1724}
        candidate = {"1m": 0.1364, "3m": 0.1434, "6m": 0.1723}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "no sample upside"
        assert diagnostics["no_sample_upside"] is True
        assert diagnostics["material_improved_horizons"] == []

    def test_quick_reject_diagnostics_rejects_sample_noop(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "behavior-neutral sample"
        assert diagnostics["noop_sample"] is True

    def test_acceptance_diagnostics_explains_target_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.1315, "3m": 0.1355, "6m": 0.1663}
        target = {"1m": 0.1347, "3m": 0.1355, "6m": 0.1663}
        candidate = {"1m": 0.1307, "3m": 0.1360, "6m": 0.1665}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == []
        assert diagnostics["current_improved_horizons"] == ["3m", "6m"]
        assert diagnostics["target_improved_horizons"] == ["3m", "6m"]
        assert diagnostics["reject_reason"] == "target utility tradeoff"
        assert diagnostics["near_miss"] is True

    def test_acceptance_diagnostics_accepts_current_pareto_despite_historical_target(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.130593, "3m": 0.139496, "6m": 0.170561}
        target = {"1m": 0.134683, "3m": 0.139496, "6m": 0.170561}
        candidate = {"1m": 0.131434, "3m": 0.139771, "6m": 0.170960}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == ["1m", "3m", "6m"]
        assert diagnostics["accepted_by"] == "current_material_pareto"
        assert diagnostics["historical_reject_reason"] == "target utility tradeoff"
        assert diagnostics["reject_reason"] == ""
        assert diagnostics["near_miss"] is False

    def test_acceptance_diagnostics_accepts_material_improvement_with_neutral_dip(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.130593, "3m": 0.139496, "6m": 0.170561}
        target = {"1m": 0.134683, "3m": 0.139496, "6m": 0.170561}
        candidate = {"1m": 0.130790, "3m": 0.139946, "6m": 0.170533}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == ["1m", "3m"]
        assert diagnostics["accepted_by"] == "current_material_pareto"
        assert diagnostics["current_degraded_horizons"] == ["6m"]
        assert diagnostics["current_material_degraded_horizons"] == []
        assert diagnostics["historical_reject_reason"] == "target utility tradeoff"
        assert diagnostics["reject_reason"] == ""
        assert diagnostics["near_miss"] is False

    def test_acceptance_diagnostics_rejects_tiny_noise_gain(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.133937, "3m": 0.143385, "6m": 0.174097}
        target = dict(current)
        candidate = {"1m": 0.133918, "3m": 0.143421, "6m": 0.174177}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == []
        assert diagnostics["current_material_improved_horizons"] == []
        assert diagnostics["near_miss"] is True

    def test_complete_proposal_saves_invalid_debug_artifact(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return "```diff\n--- a\n+++ b\n@@ -1 +1 @@\n-missing\n+new\n```\n"

        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", False)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(FakeLLM(), "prompt", "old\n", iteration=123)

        assert result["code"] is None
        assert result["debug_paths"]
        assert "Raw Response" in Path(result["debug_paths"][0]).read_text()

    def test_complete_proposal_rejects_syntax_invalid_code(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return textwrap.dedent("""\
                    ```diff
                    --- a/scorer.py
                    +++ b/scorer.py
                    @@ -1,2 +1,2 @@
                    -def score():
                    -    return 1
                    +def score():
                    +return 2
                    ```
                """)

        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", False)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(FakeLLM(), "prompt", "def score():\n    return 1\n", iteration=456)

        assert result["code"] is None
        assert "Syntax-invalid proposal" in result["explanation"]
        assert result["debug_paths"]

    def test_complete_proposal_does_not_uncap_empty_repair(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def __init__(self) -> None:
                self.calls = []

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                self.calls.append(max_tokens)
                if len(self.calls) == 1:
                    return "This is not a patch."
                return ""

        fake = FakeLLM()
        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", True)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(fake, "prompt", "old\n", iteration=789)

        assert result["code"] is None
        assert len(fake.calls) == 2
        assert fake.calls[0] == agent._MAX_PROPOSAL_TOKENS
        assert fake.calls[1] == agent._MAX_REPAIR_TOKENS
        assert result["debug_paths"]

    def test_accept_policy_rejects_bad_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.102, "3m": 0.091, "6m": 0.091}
        assert _accepted_horizons(candidate, baseline) == []

    def test_accept_policy_accepts_weighted_improvement(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.100, "3m": 0.101, "6m": 0.103}
        assert _accepted_horizons(candidate, baseline) == ["3m", "6m"]

    def test_accept_policy_tolerates_behavior_neutral_dip(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.1002, "3m": 0.09997, "6m": 0.10}
        assert _accepted_horizons(candidate, baseline) == ["1m"]

    def test_accept_policy_rejects_only_behavior_neutral_movement(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.10002, "3m": 0.10001, "6m": 0.09999}
        assert _accepted_horizons(candidate, baseline) == []


# ── Evaluator tests (with mocked ground truth) ──────────────────────────────


class TestEvaluator:
    """Tests for evaluate_scorer with synthetic ground truth."""

    def _make_gt(self) -> pd.DataFrame:
        """Create a small synthetic ground truth DataFrame."""
        import numpy as np

        np.random.seed(42)
        n = 50
        tickers = [f"SH{i:06d}" for i in range(n)]
        return pd.DataFrame({
            "ticker": tickers,
            "snapshot_date": pd.Timestamp("2023-06-01"),
            "close": np.random.uniform(10, 100, n),
            "pe_ratio": np.random.uniform(5, 50, n),
            "pb_ratio": np.random.uniform(0.5, 8, n),
            "market_cap_rmb": np.random.uniform(1e9, 1e11, n),
            "pe_forward": np.random.uniform(4, 40, n),
            "ps_ratio": np.random.uniform(0.2, 8, n),
            "peg_ratio": np.random.uniform(0.2, 3, n),
            "dividend_yield": np.random.uniform(0, 0.08, n),
            "ev_to_ebitda": np.random.uniform(2, 30, n),
            "revenue": np.random.uniform(1e8, 1e11, n),
            "net_income": np.random.uniform(-1e9, 1e10, n),
            "total_assets": np.random.uniform(1e9, 2e11, n),
            "total_liabilities": np.random.uniform(1e8, 1e11, n),
            "total_equity": np.random.uniform(1e8, 1e11, n),
            "operating_cash_flow": np.random.uniform(-1e9, 1e10, n),
            "free_cash_flow": np.random.uniform(-1e9, 1e10, n),
            "gross_margin": np.random.uniform(0, 0.8, n),
            "roa": np.random.uniform(-0.1, 0.2, n),
            "current_ratio": np.random.uniform(0.2, 4, n),
            "roe": np.random.uniform(-0.1, 0.3, n),
            "net_margin": np.random.uniform(-0.05, 0.3, n),
            "debt_to_equity": np.random.uniform(0, 3, n),
            # Make composite_score correlated with forward_return for a non-zero ρ
            "composite_score": np.random.uniform(20, 90, n),
            "value_score": np.random.uniform(20, 90, n),
            "quality_score": np.random.uniform(20, 90, n),
            "growth_score": np.random.uniform(20, 90, n),
            "forward_return_1m": np.random.uniform(-0.1, 0.2, n),
            "forward_return_3m": np.random.uniform(-0.2, 0.35, n),
            "forward_return_6m": np.random.uniform(-0.3, 0.5, n),
        })

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_evaluate_with_original_scores(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver.evaluator import evaluate_scorer

        gt = self._make_gt()
        mock_load.return_value = gt

        result = evaluate_scorer(use_original_scores=True)
        assert "spearman_rho" in result
        assert "hit_rate_top20" in result
        assert result["n_snapshots"] == 1
        # Rho should be some float (sign depends on random seed)
        assert isinstance(result["spearman_rho"], float)

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_evaluate_empty_gt(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver.evaluator import evaluate_scorer

        mock_load.return_value = pd.DataFrame()
        result = evaluate_scorer(use_original_scores=True)
        assert result["spearman_rho"] == 0.0
        assert result["n_snapshots"] == 0

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_rescore_passes_full_feature_set(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver import evaluator

        gt = self._make_gt().head(12).copy()
        mock_load.return_value = gt
        seen = {}

        class RecordingScorer:
            def score(self, result):
                seen["financials"] = result.financials
                seen["valuation"] = result.valuation
                result.composite_score = 42.0
                return result

            def rank(self, results):
                seen["rank_called"] = True
                for result in results:
                    self.score(result)
                return results

        with patch.object(evaluator.importlib, "reload", side_effect=lambda mod: mod):
            with patch("valueinvestor.screener.scorer.MultiFactorScorer", return_value=RecordingScorer()):
                evaluator.evaluate_scorer(use_original_scores=False)

        assert seen["valuation"].ps_ratio is not None
        assert seen["rank_called"] is True
        assert seen["valuation"].peg_ratio is not None
        assert seen["valuation"].dividend_yield is not None
        assert seen["financials"].revenue is not None
        assert seen["financials"].gross_margin is not None
        assert seen["financials"].operating_cash_flow is not None

    def test_sample_by_snapshot_keeps_cross_section_shape(self) -> None:
        from valueinvestor.scorer_improver.evaluator import _sample_by_snapshot

        frames = []
        for idx, snapshot in enumerate(pd.date_range("2023-01-01", periods=3, freq="90D")):
            frames.append(pd.DataFrame({
                "ticker": [f"TEST{idx:02d}{i:03d}" for i in range(30)],
                "snapshot_date": snapshot,
                "close": range(30),
            }))
        gt = pd.concat(frames, ignore_index=True)
        sampled = _sample_by_snapshot(gt, sample_size=30)
        counts = sampled.groupby("snapshot_date").size()
        assert len(sampled) == 30
        assert set(counts) == {10}
        assert len(counts) == 3

    def test_evaluation_context_reuses_loaded_frame_and_quick_sample(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver import evaluator
        from valueinvestor.scorer_improver.evaluator import ScorerEvaluationContext

        gt = self._make_gt()
        load_calls = []
        sample_calls = []

        def fake_load(path=None, *, allow_legacy_schema=True):
            load_calls.append((path, allow_legacy_schema))
            return gt

        def fake_sample(frame, sample_size):
            sample_calls.append(sample_size)
            return frame.head(12)

        monkeypatch.setattr(evaluator, "_load_ground_truth", fake_load)
        monkeypatch.setattr(evaluator, "_sample_by_snapshot", fake_sample)
        monkeypatch.setattr(
            evaluator,
            "_evaluate_scorer_all_targets_frame",
            lambda frame, *, scorer_module, scorer_path=None, snapshot_templates=None: {
                "1m": {"spearman_rho": 0.1},
                "3m": {"spearman_rho": 0.2},
                "6m": {"spearman_rho": 0.3},
            },
        )
        monkeypatch.setattr(
            evaluator,
            "_quick_evaluate_frame",
            lambda frame, *, scorer_module, scorer_path=None, snapshot_templates=None: {
                "1m": 0.1,
                "3m": 0.2,
                "6m": 0.3,
            },
        )

        context = ScorerEvaluationContext(quick_sample_size=12)
        context.evaluate_all_targets()
        context.evaluate_all_targets()
        context.quick_evaluate(sample_size=12)
        context.quick_evaluate(sample_size=12)

        assert len(load_calls) == 1
        assert sample_calls == [12]

    def test_evaluation_context_builds_snapshot_templates_once_per_cached_frame(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver import evaluator
        from valueinvestor.scorer_improver.evaluator import ScorerEvaluationContext

        gt = pd.DataFrame({
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "snapshot_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "close": [10.0, 11.0, 12.0, 13.0],
        })
        template_calls = []

        monkeypatch.setattr(
            evaluator,
            "_load_ground_truth",
            lambda path=None, *, allow_legacy_schema=True: gt,
        )
        monkeypatch.setattr(
            evaluator,
            "_sample_by_snapshot",
            lambda frame, sample_size: frame.head(2),
        )

        def fake_build(frame):
            template_calls.append(tuple(frame.index))
            return []

        monkeypatch.setattr(evaluator, "_build_snapshot_templates", fake_build)
        monkeypatch.setattr(
            evaluator,
            "_evaluate_scorer_all_targets_frame",
            lambda frame, **kwargs: {
                "1m": {"spearman_rho": 0.1},
                "3m": {"spearman_rho": 0.2},
                "6m": {"spearman_rho": 0.3},
            },
        )
        monkeypatch.setattr(
            evaluator,
            "_quick_evaluate_frame",
            lambda frame, **kwargs: {"1m": 0.1, "3m": 0.2, "6m": 0.3},
        )

        context = ScorerEvaluationContext(quick_sample_size=2)
        context.evaluate_all_targets()
        context.evaluate_all_targets()
        context.quick_evaluate(sample_size=2)
        context.quick_evaluate(sample_size=2)

        assert template_calls == [(0, 1, 2, 3), (0, 1)]


# ── Ground truth helper tests ───────────────────────────────────────────────


class TestGroundTruthHelpers:
    """Tests for forward return and snapshot date helpers."""

    def test_get_snapshot_dates(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _get_snapshot_dates

        # 3 years of daily dates
        dates = pd.date_range("2021-01-01", "2024-01-01", freq="B")  # business days
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": range(len(dates)),
        })
        snaps = _get_snapshot_dates(prices)
        assert len(snaps) > 0
        # All snapshots should be at least FORWARD_HORIZON_DAYS + 30 before end
        from valueinvestor.scorer_improver.ground_truth import FORWARD_HORIZON_DAYS
        from datetime import timedelta

        max_date = dates.max().date()
        for s in snaps:
            assert s <= max_date - timedelta(days=FORWARD_HORIZON_DAYS + 30)

    def test_get_snapshot_dates_short_data(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _get_snapshot_dates

        # Too short for any snapshots
        dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": range(len(dates)),
        })
        snaps = _get_snapshot_dates(prices)
        assert len(snaps) == 0

    def test_compute_forward_returns(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        # Create 2 years of daily prices for one stock
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="B")
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": [100.0 + i * 0.1 for i in range(len(dates))],
        })
        returns = _compute_forward_returns(prices, horizon_days=126)
        assert len(returns) > 0
        # Forward returns should be positive (monotonically increasing price)
        assert (returns["forward_return_6m"] > 0).all()

    def test_compute_forward_returns_uses_nearest_future_price(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        prices = pd.DataFrame({
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-30", "2024-02-02", "2024-03-20"]),
            "close": [100.0, 130.0, 140.0, 200.0],
        })

        returns = _compute_forward_returns(prices, horizon_days=30).set_index("date")

        assert returns.loc[pd.Timestamp("2024-01-01").date(), "forward_return_6m"] == 0.3
        assert pd.Timestamp("2024-02-02").date() not in returns.index

    def test_compute_forward_returns_empty(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        result = _compute_forward_returns(pd.DataFrame())
        assert result.empty

    def test_build_snapshot_features_vectorized_nearest_price(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _build_snapshot_features

        prices = pd.DataFrame({
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "date": pd.to_datetime([
                "2024-01-08",
                "2024-01-11",
                "2024-01-02",
                "2024-01-14",
                "2023-12-01",
            ]),
            "close": [10.0, 11.0, 20.0, 22.0, 30.0],
        })
        prices["date_dt"] = pd.to_datetime(prices["date"])
        valuations = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "pe_ratio": [12.0, 18.0],
            "pb_ratio": [1.2, 2.0],
        })
        financials = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "roe": [0.15, 0.20],
            "net_margin": [0.08, 0.10],
        })

        features = _build_snapshot_features(
            pd.Timestamp("2024-01-10").date(),
            prices,
            valuations,
            financials,
        ).set_index("ticker")

        assert set(features.index) == {"AAA", "BBB"}
        assert features.loc["AAA", "close"] == 11.0
        assert features.loc["BBB", "close"] == 22.0
        assert features.loc["AAA", "pe_ratio"] == 12.0
        assert features.loc["BBB", "roe"] == 0.20
