import json

import torch

import map_learning.pipeline as pipeline
from map_learning.pipeline import (
    _assert_adaptive_threshold_contract,
    _assert_compact_training_threshold_contract,
)


def test_adaptive_threshold_contract_accepts_one_resolved_calibration(tmp_path):
    parameters = {
        "positive_radius_px": 0.75,
        "clean_radius_px": 1.5,
        "negative_radius_px": 3.0,
        "ransac_reprojection_px": 4.5,
        "harm_radius_px": 4.5,
        "evidence_depth_abs_tolerance_m": 0.01,
    }
    graph = tmp_path / "graph.pt"
    provenance = tmp_path / "provenance.pt"
    teacher = tmp_path / "teacher.pt"
    torch.save(
        {
            "resolved_thresholds": {
                "strong_radius_px": 0.75,
                "clean_radius_px": 1.5,
                "ambiguous_radius_px": 3.0,
                "pnp_reprojection_error_px": 4.5,
                "harm_radius_px": 4.5,
                "depth_abs_tolerance_m": 0.01,
            }
        },
        graph,
    )
    torch.save({"config": {"depth_abs_tolerance_m": 0.01}}, provenance)
    torch.save(
        {
            "config": {
                "strong_radius_px": 0.75,
                "ambiguous_radius_px": 3.0,
                "depth_abs_tolerance_m": 0.01,
            }
        },
        teacher,
    )
    _assert_adaptive_threshold_contract(
        graph=graph,
        provenance=provenance,
        teacher=teacher,
        parameters=parameters,
    )


def test_adaptive_threshold_contract_rejects_stale_teacher(tmp_path):
    graph = tmp_path / "graph.pt"
    provenance = tmp_path / "provenance.pt"
    teacher = tmp_path / "teacher.pt"
    parameters = {
        "positive_radius_px": 1.0,
        "clean_radius_px": 2.0,
        "negative_radius_px": 4.0,
        "ransac_reprojection_px": 6.0,
        "harm_radius_px": 6.0,
        "evidence_depth_abs_tolerance_m": 0.02,
    }
    torch.save(
        {
            "resolved_thresholds": {
                "strong_radius_px": 1.0,
                "clean_radius_px": 2.0,
                "ambiguous_radius_px": 4.0,
                "pnp_reprojection_error_px": 6.0,
                "harm_radius_px": 6.0,
                "depth_abs_tolerance_m": 0.02,
            }
        },
        graph,
    )
    torch.save({"config": {"depth_abs_tolerance_m": 0.02}}, provenance)
    torch.save(
        {
            "config": {
                "strong_radius_px": 2.0,
                "ambiguous_radius_px": 4.0,
                "depth_abs_tolerance_m": 0.02,
            }
        },
        teacher,
    )
    try:
        _assert_adaptive_threshold_contract(
            graph=graph,
            provenance=provenance,
            teacher=teacher,
            parameters=parameters,
        )
    except ValueError as error:
        assert "teacher.strong_radius_px" in str(error)
    else:
        raise AssertionError("stale teacher threshold was accepted")


def test_compact_training_contract_rejects_stale_ransac_threshold(tmp_path):
    report = tmp_path / "training_report.json"
    report.write_text(
        json.dumps(
            {
                "config": {
                    "ransac_reprojection_px": 4.0,
                    "clean_reprojection_px": 1.5,
                }
            }
        )
    )
    parameters = {"ransac_reprojection_px": 8.0, "clean_radius_px": 1.5}
    try:
        _assert_compact_training_threshold_contract(report, parameters)
    except ValueError as error:
        assert "ransac_reprojection_px" in str(error)
    else:
        raise AssertionError("stale compact training threshold was accepted")


def test_parallel_shards_are_round_robin_pinned_to_visible_gpus(monkeypatch):
    environments = []

    class CompletedProcess:
        args = ["python"]

        @staticmethod
        def poll():
            return 0

    def fake_popen(command, env=None):
        environments.append(env)
        return CompletedProcess()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    pipeline._run_parallel("example.module", [["a"], ["b"], ["c"]])
    assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == ["0", "2", "0"]
