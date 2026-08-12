import hashlib
import json
import sys
from pathlib import Path

import torch

from common.hashing import sha256_file
from scripts import evaluate_mapping_cache


EVALUATION_CODE = {
    "schema": "lafgs_mapping_pose_evaluation_code",
    "version": 1,
    "repository": "/clean/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {
        "scripts/evaluate_mapping_cache.py": "b" * 64,
        "scripts/compare_mapping_pose_gate.py": "c" * 64,
    },
}


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_mapping_cache_report_self_binds_inputs_seed_and_query_subset(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {
        "map": tmp_path / "map.pt",
        "metric": tmp_path / "metric.pt",
        "teacher": tmp_path / "teacher.pt",
        "query_cache": tmp_path / "query_cache.pt",
        "calibration": tmp_path / "calibration.json",
    }
    torch.save({"anchor_ids": torch.tensor([0])}, paths["map"])
    torch.save({"metric": "stub"}, paths["metric"])
    query_names = [f"query-{index:04d}" for index in range(300)]
    torch.save(
        {
            "anchor_count": 1,
            "query_names": query_names,
            "records": [{} for _ in query_names],
            "query_cache": str(paths["query_cache"].resolve()),
        },
        paths["teacher"],
    )
    torch.save({"queries": {}}, paths["query_cache"])
    paths["calibration"].write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": {
                    "query_cache": str(paths["query_cache"].resolve()),
                    "uses_test_queries": False,
                },
                "parameters": {
                    "ransac_reprojection_px": 12.0,
                    "clean_radius_px": 1.0,
                    "task_translation_m": 0.05,
                    "task_rotation_deg": 5.0,
                },
            }
        )
    )
    captured = {}

    def collect_stub(**kwargs):
        captured.update(kwargs)
        return {
            "summary": {
                "query_count": 256,
                "median_te_cm": 1.0,
                "mean_te_cm": 1.1,
                "p90_te_cm": 2.0,
                "p95_te_cm": 2.5,
                "p99_te_cm": 3.0,
                "cvar95_te_cm": 3.1,
                "median_ae_deg": 0.2,
                "mean_ae_deg": 0.3,
                "p90_ae_deg": 0.5,
                "p95_ae_deg": 0.6,
                "recall_5cm_5deg_percent": 99.0,
                "catastrophic_100cm_count": 0,
                "raw_gt_precision_percent": 6.0,
            }
        }

    monkeypatch.setattr(
        evaluate_mapping_cache, "collect_deployment_statistics", collect_stub
    )
    monkeypatch.setattr(
        evaluate_mapping_cache,
        "mapping_pose_evaluation_code_identity",
        lambda **_: EVALUATION_CODE,
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.evaluate_mapping_cache",
            "--map",
            str(paths["map"]),
            "--metric-state",
            str(paths["metric"]),
            "--complete-positive-teacher",
            str(paths["teacher"]),
            "--query-cache",
            str(paths["query_cache"]),
            "--scene-calibration",
            str(paths["calibration"]),
            "--query-count",
            "256",
            "--seed",
            "2027",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    evaluate_mapping_cache.main()

    report = json.loads((output / "mapping_cache_summary.json").read_text())
    expected_indices = (
        torch.linspace(0, len(query_names) - 1, steps=256)
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    expected_names = [query_names[index] for index in expected_indices]
    assert report["version"] == 2
    assert report["uses_test_queries"] is False
    assert report["seed"] == 2027
    assert report["evaluation_code"] == EVALUATION_CODE
    assert report["query_count"] == 256
    assert report["query_selection"] == "uniform_mapping_gate"
    assert report["evaluation_protocol"] == {
        "split": "mapping_only",
        "query_selection": "uniform_mapping_gate",
        "requested_query_count": 256,
        "evaluated_query_count": 256,
        "teacher_query_count": 300,
        "ordered_teacher_query_names_sha256": _json_sha256(query_names),
        "selected_query_indices": expected_indices,
        "selected_query_indices_sha256": _json_sha256(expected_indices),
        "selected_query_names_sha256": _json_sha256(expected_names),
        "deployment_row_limit": 0,
    }
    for role, path in paths.items():
        assert report["artifacts"][role] == {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    assert captured["seed"] == 2027
    assert torch.equal(captured["query_indices"], torch.tensor(expected_indices))
    assert captured["deployment_row_limit"] == 0
