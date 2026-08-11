import torch

from map_learning.pose_set_refinement import (
    PoseSetConstraint,
    _bounded_local_bank,
    _oracle_target_gains,
    _uniform_cap,
    materialize_pose_set_map,
    train_pose_set_residual,
)


def test_pose_set_local_residual_is_bounded():
    base = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    revised, residual = _bounded_local_bank(base, torch.full_like(base, 10.0), 0.02)
    assert float(torch.linalg.norm(residual, dim=1).max()) <= 0.020001
    assert torch.allclose(torch.linalg.norm(revised, dim=1), torch.ones(4))


def test_materialized_pose_set_map_changes_only_selected_anchors():
    features = torch.eye(4)
    state = {"anchor_features": features.clone(), "anchor_ids": torch.arange(4)}
    output = materialize_pose_set_map(
        state=state,
        trainable_anchors=torch.tensor([1]),
        residual=torch.tensor([[0.1, 0.0, 0.0, 0.0]]),
        report={"ok": True},
    )
    assert torch.equal(output["anchor_features"][[0, 2, 3]], features[[0, 2, 3]])
    assert not torch.equal(output["anchor_features"][1], features[1])


def test_oracle_targets_aggregate_joint_pose_gain_by_identity():
    oracle = {
        "queries": [
            {
                "current_risk": 0.5,
                "joint_risk": 0.3,
                "joint_action_count": 1,
                "joint_trace": [
                    {
                        "risk": 0.3,
                        "actions": [{"kind": "swap", "row": 2, "anchor": 7}],
                    }
                ],
            },
            {
                "current_risk": 0.4,
                "joint_risk": 0.3,
                "joint_action_count": 1,
                "joint_trace": [
                    {
                        "risk": 0.3,
                        "actions": [{"kind": "swap", "row": 4, "anchor": 7}],
                    }
                ],
            },
        ]
    }
    assert abs(_oracle_target_gains(oracle)[7] - 0.15) < 1e-6


def test_uniform_cap_keeps_endpoints_deterministically():
    assert _uniform_cap(list(range(10)), 3) == [0, 4, 9]


def test_pose_set_training_allows_fixed_bad_anchor():
    state = {
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
    }
    constraints = [
        PoseSetConstraint(
            query_index=index,
            query=torch.tensor([0.8, 0.6]),
            bad_anchor=torch.tensor(0),
            good_anchor=torch.tensor(1),
            weight=1.0,
            pose_gain=0.1,
        )
        for index in range(5)
    ]
    residual, report = train_pose_set_residual(
        state=state,
        constraints=constraints,
        clean_constraints=[],
        trainable_anchors=torch.tensor([1]),
        maximum_norm=0.02,
        steps=2,
        learning_rate=1e-3,
        margin=0.03,
        temperature=0.04,
        trust_weight=0.0,
        clean_weight=0.0,
        holdout_modulus=5,
        holdout_remainder=0,
        device=torch.device("cpu"),
    )
    assert residual.shape == (1, 2)
    assert report["realizability"]["initial_pair_accuracy"] == 0.0
