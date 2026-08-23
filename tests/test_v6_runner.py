from argparse import Namespace
from pathlib import Path

from scripts.run_closed_loop_projective_distillation import (
    _eligible_arms,
    _evaluation_command,
    _load_initial_baseline,
    _proposal_command,
)
from common.hashing import sha256_file


def test_formal_runner_includes_joint_descriptor_selection_arm() -> None:
    arms, skipped = _eligible_arms(
        {
            "summary": {"anchor_count": 100},
            "failure_layer_counts": {"L1": 0},
        },
        maximum_anchor_count=100,
        has_association_graph=True,
    )
    assert arms == ["descriptor", "selection", "descriptor_selection"]
    assert skipped == []


def test_formal_runner_skips_descriptor_only_when_it_cannot_be_compact() -> None:
    arms, skipped = _eligible_arms(
        {
            "summary": {"anchor_count": 101},
            "failure_layer_counts": {"L1": 1},
        },
        maximum_anchor_count=100,
        has_association_graph=True,
    )
    assert arms == ["selection", "descriptor_selection", "reconstruction"]
    assert skipped == [
        {
            "arm": "descriptor",
            "reason": "descriptor_only_preserves_anchor_count_above_hard_limit",
        }
    ]


def test_initial_baseline_is_bound_to_exact_inputs(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.pt"
    feedback.write_bytes(b"feedback")
    summary = tmp_path / "summary.json"
    summary.write_text(
        __import__("json").dumps(
            {
                "input_sha256": {
                    "map": "m" * 64,
                    "metric": "i" * 64,
                    "observation_cache": "o" * 64,
                },
                "feedback_path": str(feedback),
                "feedback_sha256": sha256_file(feedback),
            }
        )
    )
    args = _args()
    args.initial_baseline_summary = summary
    args.expected_initial_baseline_summary_sha256 = sha256_file(summary)
    loaded = _load_initial_baseline(args, map_sha="m" * 64, metric_sha="i" * 64)
    assert loaded is not None
    assert loaded[0] == summary.resolve()


def _args() -> Namespace:
    return Namespace(
        observation_cache=Path("observations.pt"),
        expected_observation_cache_sha256="o" * 64,
        device="cpu",
        cpu_threads=1,
        positive_radius_px=2.0,
        alpha_minimum=0.05,
        required_rank=4,
        ransac_reprojection_px=4.0,
        seed=2026,
        descriptor_trust_region=0.05,
        maximum_anchor_count=100,
        matching_target=4,
        pose_logdet_target=0.0,
        association_graph=Path("association.pt"),
        expected_association_graph_sha256="a" * 64,
    )


def test_closed_loop_runner_freezes_single_pass_evaluation_contract() -> None:
    command = _evaluation_command(
        _args(),
        root=Path("/repo"),
        map_path=Path("map.pt"),
        map_sha="m" * 64,
        metric_path=Path("metric.pt"),
        metric_sha="i" * 64,
        output=Path("round/baseline"),
    )
    assert command.count("--device") == 1
    assert "evaluate_v6_self_localization.py" in command[1]
    assert "--observation-cache" in command
    assert "--expected-observation-cache-sha256" in command


def test_reconstruction_arm_is_bound_to_frozen_association() -> None:
    command = _proposal_command(
        _args(),
        root=Path("/repo"),
        arm="reconstruction",
        map_path=Path("map.pt"),
        map_sha="m" * 64,
        feedback_path=Path("feedback.pt"),
        feedback_sha="f" * 64,
        output=Path("round/proposal"),
    )
    assert command[-4:] == [
        "--association-graph",
        "association.pt",
        "--expected-association-graph-sha256",
        "a" * 64,
    ]
