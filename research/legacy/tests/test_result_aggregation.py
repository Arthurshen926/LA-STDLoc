from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.aggregate import aggregate_registered_benchmark, latex_rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _result(te: float, correct: int) -> dict[str, float | int]:
    return {
        "translation_error_cm": te,
        "rotation_error_deg": te / 10,
        "raw_count": 10,
        "raw_correct_2px": correct,
        "inlier_count": 5,
        "inlier_correct_2px": correct,
        "ransac_iterations": 100,
        "total_ms": 20 + te,
    }


def _scene(root: Path, name: str, *, leaked: bool = False) -> None:
    scene = root / name
    marker = {
        "schema": "lafgs_indoor_pgt_scene_v1",
        "family": "Fixture",
        "config_sha256": "config",
        "runner_sha256": "runner",
        "function_graph_shards": 4,
        "provenance_shards": 4,
        "observation_shards": 1,
        "pose_scoring_shards": 4,
        "mapping_only_prior": True,
        "test_images_used_for_training": leaked,
        "a0_seeds": [2026, 2027, 2028],
        "a1_seeds": [2026, 2027, 2028],
    }
    _write_json(scene / "full_benchmark_complete.json", marker)
    for seed in (2026, 2027, 2028):
        _write_json(
            scene / f"evaluation_a0_seed{seed}" / "results.json",
            [_result(2.0, 1), _result(4.0, 2)],
        )
        output = "evaluation" if seed == 2026 else f"evaluation_a1_seed{seed}"
        _write_json(
            scene / output / "results.json",
            [_result(1.0, 2), _result(3.0, 3)],
        )
    _write_json(
        scene / "map_learning" / "training_report.json",
        {"config": {"track_anchor_count": 8_000, "base_anchor_count": 500}},
    )


def test_registered_aggregate_pools_queries_and_formats_rows(tmp_path):
    _scene(tmp_path, "one")
    _scene(tmp_path, "two")
    result = aggregate_registered_benchmark(tmp_path, ["one", "two"])
    assert result["scene_count"] == 2
    assert result["a1_anchor_count_mean"] == 8_500
    assert result["stages"]["a0"]["pooled_seed_mean"]["median_te_cm"] == 3
    assert result["stages"]["a1"]["pooled_seed_mean"]["median_te_cm"] == 2
    rows = latex_rows("Fixture", result)
    assert "Fixture & A0 & 3.000" in rows
    assert "Fixture & A1 & 2.000" in rows


def test_registered_aggregate_rejects_test_leakage(tmp_path):
    _scene(tmp_path, "bad", leaked=True)
    with pytest.raises(ValueError, match="zero test leakage"):
        aggregate_registered_benchmark(tmp_path, ["bad"])
