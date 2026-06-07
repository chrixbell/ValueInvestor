from __future__ import annotations

import json
from pathlib import Path


def test_anchor_model_paths_auto_returns_requested_number_of_non_output_anchors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from valueinvestor.cli.main import _anchor_model_paths_option

    trainer_dir = tmp_path / "data" / "trainer"
    trainer_dir.mkdir(parents=True)
    output_model_path = trainer_dir / "ml_ranker_model.json"

    def write_payload(path: Path, rho: float) -> None:
        path.write_text(
            json.dumps({
                "schema_version": 1,
                "metadata": {
                    "metrics": {"6m": {"spearman_rho": rho}},
                    "trained_at": f"2026-06-06T00:00:0{int(rho * 10)}",
                },
            }),
            encoding="utf-8",
        )

    write_payload(output_model_path, 0.99)
    write_payload(trainer_dir / "ml_ranker_model.backup-1.json", 0.10)
    write_payload(trainer_dir / "ml_ranker_model.backup-2.json", 0.30)
    write_payload(trainer_dir / "ml_ranker_model.backup-3.json", 0.20)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VALUEINVESTOR_ML_ANCHOR_LIMIT", "2")

    selected = _anchor_model_paths_option(
        "auto",
        target_horizon="6m",
        output_model_path=output_model_path,
    )

    assert selected == (
        Path("data/trainer/ml_ranker_model.backup-2.json"),
        Path("data/trainer/ml_ranker_model.backup-3.json"),
    )
