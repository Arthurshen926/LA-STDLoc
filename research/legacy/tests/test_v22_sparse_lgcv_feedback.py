from __future__ import annotations

import torch

from map_learning.v22_sparse_lgcv_feedback import (
    default_config,
    filter_provisional_assignment_with_sparse_lgcv,
    finalize_evaluations,
    sparse_lgcv_candidate_support,
)


def _ring() -> torch.Tensor:
    return torch.tensor(
        [
            [20.0, 0.0],
            [14.0, 14.0],
            [0.0, 20.0],
            [-14.0, 14.0],
            [-20.0, 0.0],
            [-14.0, -14.0],
            [0.0, -20.0],
            [14.0, -14.0],
        ]
    )


def test_sparse_lgcv_accepts_locally_consistent_candidate() -> None:
    ring = _ring()
    result = sparse_lgcv_candidate_support(
        candidate_query_xy=torch.tensor([[0.0, 0.0]]),
        candidate_projected_xy=torch.tensor([[0.0, 0.0]]),
        reference_query_xy=ring,
        reference_projected_xy=ring,
        config=default_config(),
    )
    assert result["effective_neighbor_count"] == 8
    assert result["support_scores"].item() > 4


def test_sparse_lgcv_rejects_orientation_reversal() -> None:
    ring = _ring()
    mirrored = ring.clone()
    mirrored[:, 0] *= -1
    result = sparse_lgcv_candidate_support(
        candidate_query_xy=torch.tensor([[0.0, 0.0]]),
        candidate_projected_xy=torch.tensor([[0.0, 0.0]]),
        reference_query_xy=ring,
        reference_projected_xy=mirrored,
        config=default_config(),
    )
    assert result["support_scores"].item() == 0


def test_filter_preserves_inliers_and_keeps_supported_replacement() -> None:
    ring = _ring()
    keypoints = torch.cat((ring, torch.tensor([[0.0, 0.0]])), dim=0)
    xyz = torch.cat(
        (
            torch.cat((ring, torch.ones(8, 1)), dim=1),
            torch.tensor([[80.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        ),
        dim=0,
    )
    baseline = torch.tensor(list(range(8)) + [8])
    provisional = baseline.clone()
    provisional[8] = 9
    result = filter_provisional_assignment_with_sparse_lgcv(
        keypoints=keypoints,
        baseline_anchor_rows=baseline,
        provisional_anchor_rows=provisional,
        provisional_changed_query_rows=torch.tensor([8]),
        baseline_inlier_rows=torch.arange(8),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=default_config(),
    )
    assert torch.equal(result["anchor_rows"], provisional)
    assert result["supported_changed_query_rows"].tolist() == [8]
    assert torch.equal(result["anchor_rows"][:8], baseline[:8])


def test_filter_rejects_candidate_without_enough_local_inlier_support() -> None:
    keypoints = torch.tensor([[0.0, 0.0], [200.0, 0.0], [0.0, 200.0]])
    baseline = torch.tensor([0, 1, 2])
    provisional = torch.tensor([0, 1, 3])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [200.0, 0.0, 1.0],
            [100.0, 100.0, 1.0],
            [0.0, 200.0, 1.0],
        ]
    )
    result = filter_provisional_assignment_with_sparse_lgcv(
        keypoints=keypoints,
        baseline_anchor_rows=baseline,
        provisional_anchor_rows=provisional,
        provisional_changed_query_rows=torch.tensor([2]),
        baseline_inlier_rows=torch.tensor([0, 1]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=default_config(),
    )
    assert torch.equal(result["anchor_rows"], baseline)
    assert result["rejected_changed_query_rows"].tolist() == [2]


def test_finalizer_stops_when_confirmation_has_no_continuous_gain(monkeypatch) -> None:
    def summary(delta_te: float, delta_re: float) -> dict:
        return {
            "paired_r5_gain_count": 1,
            "paired_r5_loss_count": 0,
            "continuous_pose_metrics": {
                "translation_error_cm": {
                    "baseline": {"median": 4.0, "p90": 10.0},
                    "candidate": {"median": 4.0 + delta_te, "p90": 10.0},
                },
                "rotation_error_deg": {
                    "baseline": {"median": 0.2, "p90": 0.5},
                    "candidate": {"median": 0.2 + delta_re, "p90": 0.5},
                },
            },
        }

    monkeypatch.setattr(
        "map_learning.v22_sparse_lgcv_feedback.validate_evaluation", lambda _: None
    )
    monkeypatch.setattr(
        "map_learning.v22_sparse_lgcv_feedback.summarize",
        lambda _: {
            "continuous_pose_metrics": {
                "translation_error_cm": {
                    "baseline": {"median": 4.0},
                    "candidate": {"median": 3.9},
                },
                "rotation_error_deg": {
                    "baseline": {"median": 0.2},
                    "candidate": {"median": 0.19},
                },
            }
        },
    )
    payloads = []
    for query, role, deltas in (
        (0, "adaptation", (-0.1, -0.01)),
        (1, "control", (-0.1, 0.0)),
        (2, "confirmation", (0.0, 0.001)),
    ):
        payloads.append(
            {
                "evaluation_role": role,
                "configuration": default_config(),
                "baseline_contract": {"frozen": True},
                "inputs": {
                    "stable_map": {"path": "/map", "sha256": "a" * 64},
                    "split_manifest": {"path": "/split", "sha256": "b" * 64},
                },
                "records": [{"query_index": query}],
                "summary": summary(*deltas),
            }
        )
    sources = [
        {"path": f"/{role}", "sha256": str(index) * 64, "size_bytes": 1}
        for index, role in enumerate(
            ("adaptation", "control", "confirmation"), start=1
        )
    ]
    result = finalize_evaluations(payloads, sources)
    assert result["deployment_authorized"] is False
    assert result["phase_gates"]["confirmation"]["passed"] is False
    assert result["decision"].startswith("STOP_")
