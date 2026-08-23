from argparse import Namespace
from pathlib import Path

from scripts.run_closed_loop_projective_distillation import (
    _evaluation_command,
    _proposal_command,
)


def test_formal_runner_includes_joint_descriptor_selection_arm() -> None:
    source = Path(
        "/tmp/stdloc-wt-v6-closed-loop-projective/"
        "scripts/run_closed_loop_projective_distillation.py"
    ).read_text()
    assert '["descriptor", "selection", "descriptor_selection"]' in source


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
