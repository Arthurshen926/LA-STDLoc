import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.hashing import sha256_file
from scripts import compare_rendered_track_pose_feedback as compare_feedback
from scripts import evaluate_rendered_track_closed_loop as evaluate_closed_loop
from scripts import materialize_rendered_track_pose_feedback as materialize_feedback


def _save(payload, path: Path) -> str:
    torch.save(payload, path)
    return sha256_file(path)


def _identity():
    return {
        "git_commit": "a" * 40,
        "worktree_clean": True,
        "source_sha256": {"x.py": "b" * 64},
        "torch_version": torch.__version__,
    }


def _materialization_inputs(tmp_path: Path):
    anchor_map = tmp_path / "map.pt"
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0]] * 3),
        "anchor_features": torch.eye(3),
        "v7_metric_raw_features": torch.eye(3),
        "anchor_type": torch.ones(3, dtype=torch.long),
        "source_primitive_ids": torch.full((3,), -1, dtype=torch.long),
        "track_cluster_ids": torch.tensor([10, 11, 12]),
        "base_anchor_count": 0,
        "micro_anchor_count": 3,
        "requested_micro_anchor_budget": 3,
        "canonical_anchor_count": 3,
        "track_centric_reconstruction": {
            "track_indices": torch.tensor([10, 11, 12]),
            "base_canonical_rows": torch.empty(0, dtype=torch.long),
        },
    }
    map_sha = _save(state, anchor_map)
    metric_path = tmp_path / "metric.pt"
    metric = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(3),
        "map_path": str(anchor_map.resolve()),
        "map_sha256": map_sha,
    }
    metric_sha = _save(metric, metric_path)
    query_cache = tmp_path / "cache.pt"
    query_cache_sha = _save({"uses_test_queries": False}, query_cache)
    calibration_path = tmp_path / "calibration.json"
    calibration = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "uses_test_queries": False,
        "sources": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "mapping_source": "gaussian_render",
            "query_cache": str(query_cache.resolve()),
        },
        "parameters": {},
    }
    calibration_path.write_text(json.dumps(calibration))
    calibration_sha = sha256_file(calibration_path)
    teacher_path = tmp_path / "teacher.pt"
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": 3,
        "query_names": ["q"],
        "query_cache": str(query_cache.resolve()),
        "query_cache_sha256": query_cache_sha,
        "records": [
            {
                "query_index": 0,
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
        "diagnostics": {
            "positive_rows": 2,
            "strong_pair_count": 2,
            "ambiguous_pair_count": 0,
        },
    }
    teacher_sha = _save(teacher, teacher_path)
    statistics_path = tmp_path / "statistics.pt"
    counter_names = (
        "winner_count",
        "correct_winner_count",
        "false_attractor_count",
        "ambiguous_winner_count",
        "clean_inlier_count",
        "harmful_inlier_count",
        "counterfactual_clean_gain",
        "information_deletion_loss",
    )
    counters = {name: torch.zeros(3, dtype=torch.float64) for name in counter_names}
    counters["winner_count"][2] = 4
    counters["false_attractor_count"][2] = 4
    counters["harmful_inlier_count"][2] = 2
    counters["counterfactual_clean_gain"][2] = 3
    statistics = {
        "schema": "lafgs_rendered_track_full_mapping_loo_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": [
            {
                "query_index": 0,
                "image_name": "q",
                "te_cm": 1.0,
                "ae_deg": 1.0,
                "hypotheses": 10,
            }
        ],
        "counters": counters,
        "summary": {"query_count": 1},
    }
    statistics_sha = _save(statistics, statistics_path)
    report_path = tmp_path / "baseline.json"
    paths = {
        "map": anchor_map,
        "metric": metric_path,
        "teacher": teacher_path,
        "query_cache": query_cache,
        "scene_calibration": calibration_path,
    }
    hashes = {
        "map": map_sha,
        "metric": metric_sha,
        "teacher": teacher_sha,
        "query_cache": query_cache_sha,
        "scene_calibration": calibration_sha,
    }
    report = {
        "schema": "lafgs_rendered_track_full_mapping_loo_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "input_sha256": hashes,
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": statistics_sha,
        "summary": statistics["summary"],
    }
    report_path.write_text(json.dumps(report))
    return SimpleNamespace(
        anchor_map=anchor_map,
        expected_map_sha256=map_sha,
        metric_state=metric_path,
        expected_metric_sha256=metric_sha,
        teacher=teacher_path,
        expected_teacher_sha256=teacher_sha,
        statistics=statistics_path,
        expected_statistics_sha256=statistics_sha,
        baseline_report=report_path,
        expected_baseline_report_sha256=sha256_file(report_path),
        query_cache=query_cache,
        expected_query_cache_sha256=query_cache_sha,
        scene_calibration=calibration_path,
        expected_scene_calibration_sha256=calibration_sha,
        output_dir=tmp_path / "revision",
        matching_rows_target=2,
        maximum_prune_fraction=0.01,
        minimum_counterfactual_gain=1.0,
        maximum_tail_nonimproving_wins=2,
    )


def test_pose_feedback_materializes_atomic_map_metric_teacher(monkeypatch, tmp_path):
    args = _materialization_inputs(tmp_path)
    monkeypatch.setattr(materialize_feedback, "_producer_identity", _identity)
    monkeypatch.setattr(
        materialize_feedback,
        "select_revision",
        lambda *unused_args, **unused_kwargs: (
            torch.tensor([2]),
            {
                "final_prune_count": 1,
                "matching_constraint": {"unmet_query_count": 0},
            },
        ),
    )
    report = materialize_feedback.run(args)

    assert report["pruned_anchor_rows"] == [2]
    assert report["retained_anchor_count"] == 2
    completion_path = args.output_dir / "pose_feedback_revision_completion.json"
    completion = json.loads(completion_path.read_text())
    revised_map = torch.load(
        completion["outputs"]["map"], map_location="cpu", weights_only=False
    )
    revised_metric = torch.load(
        completion["outputs"]["metric"], map_location="cpu", weights_only=False
    )
    revised_teacher = torch.load(
        completion["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    assert revised_map["pose_feedback_source_anchor_rows"].tolist() == [0, 1]
    assert revised_map["track_cluster_ids"].tolist() == [10, 11]
    assert revised_metric["map_sha256"] == sha256_file(
        Path(completion["outputs"]["map"])
    )
    assert revised_teacher["anchor_count"] == 2
    assert revised_teacher["records"][0]["positive_indices"].tolist() == [0, 1]


def test_pose_feedback_rejects_metric_not_bound_to_map(monkeypatch, tmp_path):
    args = _materialization_inputs(tmp_path)
    metric = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    metric["map_sha256"] = "0" * 64
    args.expected_metric_sha256 = _save(metric, args.metric_state)
    monkeypatch.setattr(materialize_feedback, "_producer_identity", _identity)
    with pytest.raises(ValueError, match="metric is not bound"):
        materialize_feedback.run(args)
    assert not args.output_dir.exists()


def _evaluation_report(tmp_path: Path, label: str, te: float):
    artifacts = {}
    artifact_sha = {}
    for role in ("map", "metric", "teacher", "query_cache", "scene_calibration"):
        path = tmp_path / f"{label}_{role}.bin"
        path.write_bytes(f"{label}-{role}".encode())
        artifacts[role] = str(path.resolve())
        artifact_sha[role] = sha256_file(path)
    statistics_path = tmp_path / f"{label}_statistics.pt"
    statistics = {
        "schema": "lafgs_rendered_track_full_mapping_loo_statistics",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": [
            {
                "query_index": 0,
                "image_name": "q",
                "te_cm": te,
                "ae_deg": te,
                "hypotheses": 100,
            }
        ],
    }
    _save(statistics, statistics_path)
    report_path = tmp_path / f"{label}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_rendered_track_full_mapping_loo_report",
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "inputs": artifacts,
                "input_sha256": artifact_sha,
                "statistics": str(statistics_path.resolve()),
                "statistics_sha256": sha256_file(statistics_path),
                "summary": {"mean_te_cm": te},
            }
        )
    )
    return report_path, artifacts, artifact_sha


def test_closed_loop_selects_lower_aggregate_pose_risk(monkeypatch, tmp_path):
    baseline_path, _, _ = _evaluation_report(tmp_path, "baseline", 10.0)
    candidate_path, artifacts, artifact_sha = _evaluation_report(
        tmp_path, "candidate", 1.0
    )
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_rendered_track_pose_feedback_revision_completion",
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "outputs": {
                    role: artifacts[role] for role in ("map", "metric", "teacher")
                },
                "output_sha256": {
                    role: artifact_sha[role] for role in ("map", "metric", "teacher")
                },
            }
        )
    )
    args = SimpleNamespace(
        baseline_report=baseline_path,
        expected_baseline_report_sha256=sha256_file(baseline_path),
        candidate_report=candidate_path,
        expected_candidate_report_sha256=sha256_file(candidate_path),
        revision_completion=completion_path,
        expected_revision_completion_sha256=sha256_file(completion_path),
        output=tmp_path / "closed_loop.json",
    )
    monkeypatch.setattr(compare_feedback, "_producer_identity", _identity)
    result = compare_feedback.run(args)

    assert result["decision"] == "SELECT_POSE_FEEDBACK_REVISION"
    assert result["selected_artifacts"]["map"] == artifacts["map"]
    assert result["selection_objective"]["absolute_improvement"] > 0


def test_closed_loop_rejects_spliced_candidate_artifact(monkeypatch, tmp_path):
    baseline_path, _, _ = _evaluation_report(tmp_path, "baseline", 10.0)
    candidate_path, artifacts, artifact_sha = _evaluation_report(
        tmp_path, "candidate", 1.0
    )
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_rendered_track_pose_feedback_revision_completion",
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "outputs": {
                    role: artifacts[role] for role in ("map", "metric", "teacher")
                },
                "output_sha256": {
                    **{
                        role: artifact_sha[role]
                        for role in ("map", "metric", "teacher")
                    },
                    "map": "f" * 64,
                },
            }
        )
    )
    args = SimpleNamespace(
        baseline_report=baseline_path,
        expected_baseline_report_sha256=sha256_file(baseline_path),
        candidate_report=candidate_path,
        expected_candidate_report_sha256=sha256_file(candidate_path),
        revision_completion=completion_path,
        expected_revision_completion_sha256=sha256_file(completion_path),
        output=tmp_path / "closed_loop.json",
    )
    monkeypatch.setattr(compare_feedback, "_producer_identity", _identity)
    with pytest.raises(ValueError, match="revision map SHA differs"):
        compare_feedback.run(args)
    assert not args.output.exists()


def _closed_loop_selection(tmp_path: Path):
    artifacts = {}
    hashes = {}
    for role in ("map", "metric", "teacher", "query_cache", "scene_calibration"):
        path = tmp_path / f"selected_{role}.bin"
        path.write_bytes(role.encode())
        artifacts[role] = str(path.resolve())
        hashes[role] = sha256_file(path)
    selection_path = tmp_path / "closed_loop.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_rendered_track_closed_loop_selection",
                "valid": True,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "decision": "SELECT_POSE_FEEDBACK_REVISION",
                "selected_label": "pose_feedback_revision",
                "selected_artifacts": artifacts,
                "selected_artifact_sha256": hashes,
                "authorization": {
                    "mapping_selection_complete": True,
                    "test_may_be_used_only_for_frozen_final_evaluation": True,
                    "test_may_change_map_or_selection": False,
                },
            }
        )
    )
    return selection_path, artifacts


def test_closed_loop_test_loader_rehashes_every_selected_artifact(tmp_path):
    selection_path, artifacts = _closed_loop_selection(tmp_path)
    loaded = evaluate_closed_loop.load_closed_loop_selection(
        selection_path, sha256_file(selection_path)
    )
    assert loaded["artifacts"]["map"] == Path(artifacts["map"])

    Path(artifacts["teacher"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="selected teacher SHA differs"):
        evaluate_closed_loop.load_closed_loop_selection(
            selection_path, sha256_file(selection_path)
        )
