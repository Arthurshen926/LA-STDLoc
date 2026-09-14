from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import scripts.run_v6_mainline_convergence as runner


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        map=tmp_path / "map.pt",
        expected_map_sha256="m" * 64,
        metric=tmp_path / "metric.pt",
        expected_metric_sha256="i" * 64,
        observation_cache=tmp_path / "observations.pt",
        expected_observation_cache_sha256="o" * 64,
        association_graph=tmp_path / "association.pt",
        expected_association_graph_sha256="a" * 64,
        output_dir=tmp_path / "run",
        device="cpu",
        cpu_threads=1,
        seed=2026,
        positive_radius_px=2.0,
        alpha_minimum=0.05,
        required_rank=16,
        required_visibility_rank=4,
        required_detectable_rank=16,
        loo_pose_neighbors=3,
        loo_affected_anchor_policy="rebuild",
        ransac_reprojection_px=4.0,
        descriptor_rounds=1,
        descriptor_trust_region=0.05,
        descriptor_margin=0.05,
        descriptor_temperature=0.04,
        descriptor_learning_rate=0.02,
        descriptor_epochs=1,
        descriptor_batch_size=8,
        descriptor_maximum_triplets_per_query=4,
        descriptor_clean_fraction=0.25,
        descriptor_clean_weight=0.25,
        descriptor_trust_weight=0.1,
        descriptor_pose_critical_weight=0.0,
        descriptor_tail_query_weight=0.0,
        descriptor_training_query_indices=None,
        expected_descriptor_training_query_indices_sha256=None,
        run_reconstruction=False,
        run_selection=False,
        selection_maximum_anchors=100,
        pose_logdet_target=0.0,
        pose_min_eigenvalue_target=0.0,
    )


def _summary(*, l1: int, l3: int, learned: bool = False) -> dict:
    return {
        "summary": {"median_translation_cm": 1.0},
        "failure_layer_counts": {"L1": l1, "L2": 0, "L3": l3, "L4": 0},
        "failure_layer_counts_are_overlapping": True,
        "failure_query_count": int(bool(l1 or l3)),
        "multi_layer_failure_query_count": int(bool(l1 and l3)),
        "descriptor_validation_summary": None,
        "descriptor_training_replay_summary": (
            {"median_translation_cm": 1.0} if learned else None
        ),
        "descriptor_gradient_reuse_summary": (
            {"median_translation_cm": 1.0} if learned else None
        ),
        "reconstruction_target_replay_summary": None,
        "selection_training_replay_summary": None,
        "contract": {"affected_anchor_policy": "rebuild"},
    }


def _fake_clean_git(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )


def test_runner_does_not_label_baseline_as_candidate(tmp_path, monkeypatch) -> None:
    _fake_clean_git(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_evaluate",
        lambda *args, **kwargs: (tmp_path / "summary.json", _summary(l1=0, l3=0)),
    )
    result = runner.run(_args(tmp_path))
    assert result["version"] == 2
    assert result["candidate_available"] is False
    assert result["last_candidate_map"] is None
    assert result["last_candidate_deployment_map"] is None
    assert result["stages"][0]["deployment_map_sha256"] == "m" * 64


def test_runner_serializes_a_consistent_candidate_stage(tmp_path, monkeypatch) -> None:
    _fake_clean_git(monkeypatch)
    calls = iter(
        [
            _summary(l1=0, l3=1),
            _summary(l1=0, l3=0, learned=True),
        ]
    )
    monkeypatch.setattr(
        runner,
        "_evaluate",
        lambda *args, **kwargs: (tmp_path / "summary.json", next(calls)),
    )

    def propose(*args, **kwargs):
        output = tmp_path / "proposal"
        return {
            "proposal_available": True,
            "output": {
                "map": str(output / "proposal_map.pt"),
                "map_sha256": "p" * 64,
                "metric": str(output / "metric.pt"),
                "metric_sha256": "q" * 64,
                "deployment_map": str(output / "deployment_map.pt"),
                "deployment_map_sha256": "d" * 64,
                "deployment_metric": str(output / "deployment_metric.pt"),
                "deployment_metric_sha256": "e" * 64,
            },
        }

    monkeypatch.setattr(runner, "_propose", propose)
    result = runner.run(_args(tmp_path))
    stage = result["stages"][1]
    for field in (
        "map",
        "map_sha256",
        "metric",
        "metric_sha256",
        "deployment_map",
        "deployment_map_sha256",
        "deployment_metric",
        "deployment_metric_sha256",
        "evaluation_role",
        "contract",
    ):
        assert field in stage
    assert result["candidate_available"] is True
    assert result["last_candidate_deployment_map_sha256"] == "d" * 64


def test_runner_records_unavailable_proposal(tmp_path, monkeypatch) -> None:
    _fake_clean_git(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_evaluate",
        lambda *args, **kwargs: (tmp_path / "summary.json", _summary(l1=0, l3=1)),
    )
    monkeypatch.setattr(
        runner,
        "_propose",
        lambda *args, **kwargs: {
            "proposal_available": False,
            "unavailable_reason": "no_trainable_l3_descriptor_triplets",
            "_report_path": str(tmp_path / "proposal.json"),
            "_report_sha256": "r" * 64,
        },
    )
    result = runner.run(_args(tmp_path))
    attempt = result["stages"][1]
    assert attempt["stage_kind"] == "proposal_attempt"
    assert attempt["proposal_available"] is False
    assert attempt["unavailable_reason"] == "no_trainable_l3_descriptor_triplets"
    assert attempt["proposal_report_sha256"] == "r" * 64


def test_runner_passes_independent_layer_and_pose_targets(tmp_path, monkeypatch) -> None:
    captured = []
    output = tmp_path / "proposal"
    args = _args(tmp_path)
    args.descriptor_training_query_indices = tmp_path / "sequence-split.json"
    args.expected_descriptor_training_query_indices_sha256 = "s" * 64

    def fake_run(command, *, root):
        captured.extend(command)
        output.mkdir()
        (output / "proposal.json").write_text("{}\n")

    monkeypatch.setattr(runner, "_run", fake_run)
    runner._propose(
        args,
        root=tmp_path,
        arm="selection",
        map_path=tmp_path / "map.pt",
        map_sha="m" * 64,
        feedback_summary={
            "feedback_path": str(tmp_path / "feedback.pt"),
            "feedback_sha256": "f" * 64,
        },
        output=output,
    )

    def value(flag):
        return captured[captured.index(flag) + 1]

    assert value("--visibility-target") == "4"
    assert value("--detectability-target") == "16"
    assert value("--matching-target") == "16"
    assert value("--pose-min-eigenvalue-target") == "0.0"
    assert value("--mapping-training-query-indices") == str(
        tmp_path / "sequence-split.json"
    )
    assert value("--expected-mapping-training-query-indices-sha256") == "s" * 64
