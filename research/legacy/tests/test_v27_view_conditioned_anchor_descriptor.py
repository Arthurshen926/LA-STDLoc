from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from map_learning.v27_view_conditioned_anchor_descriptor import (
    authorize_mapping_view_modes,
    build_mapping_view_conditioned_descriptors,
    select_view_conditioned_anchor_features,
    validate_artifact,
)


def _inputs() -> tuple[dict, dict]:
    base = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), dim=1)
    map_state = {
        "schema": "lafgs_materialized_anchor_map",
        "provenance": {"uses_test_queries": False},
        "anchor_ids": torch.tensor([7, 11]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "anchor_features": base,
        "v6_mapping_query_names": ["a.png", "b.png"],
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 2, 4]),
            "query_indices": torch.tensor([0, 1, 0, 1]),
            "keypoint_indices": torch.tensor([0, 0, 1, 1]),
        },
        "anchor_view_support": {
            "schema": "lafgs_v24_anchor_view_support",
            "uses_test_queries": False,
            "direction_modes": torch.tensor(
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[-0.2, 0.0, 0.98], [-0.2, 0.0, 0.98]],
                ]
            ),
            "mode_count": torch.tensor([1, 1]),
        },
    }
    pose = torch.eye(4)
    pose[2, 3] = -5.0
    cache = {
        "schema": "render_observation_cache_v2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": {
            "a.png": {
                "native_descriptors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "pose_w2c": pose,
            },
            "b.png": {
                "native_descriptors": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
                "pose_w2c": pose,
            },
        },
    }
    return map_state, cache


def test_build_uses_mapping_observations_and_keeps_owner_count() -> None:
    map_state, cache = _inputs()
    built = build_mapping_view_conditioned_descriptors(
        map_state=map_state, observation_cache=cache
    )
    assert built["mode_features"].shape == (2, 2, 2)
    assert built["mode_valid"].tolist() == [[True, False], [True, False]]
    assert built["mode_observation_count"].tolist() == [[2, 0], [2, 0]]
    assert torch.allclose(built["mode_features"].norm(dim=2), torch.ones(2, 2))


def test_selector_changes_only_valid_nearest_mode() -> None:
    base = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), dim=1)
    pose = torch.eye(4)
    pose[2, 3] = -5.0
    selected, report = select_view_conditioned_anchor_features(
        base_anchor_features=base,
        anchor_xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        direction_modes=torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ]
        ),
        baseline_pose_w2c=pose,
        mode_features=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ),
        mode_valid=torch.tensor([[True, False], [False, False]]),
    )
    assert torch.equal(selected[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(selected[1], base[1])
    assert report == {
        "selected_mode_anchor_count": 1,
        "base_fallback_anchor_count": 1,
        "selected_mode_mean_alpha": 1.0,
    }


def test_exact_owner_authorization_rejects_false_stealing_mode() -> None:
    base = torch.eye(2)
    result = authorize_mapping_view_modes(
        mode_features=torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        mode_valid=torch.tensor([[True, False], [True, True]]),
        base_anchor_features=base,
        device="cpu",
    )
    assert result["mode_authorized"].tolist() == [
        [True, False],
        [False, True],
    ]


def test_fail_closed_on_test_cache_and_corrupt_artifact() -> None:
    map_state, cache = _inputs()
    cache["uses_test_queries"] = True
    with pytest.raises(ValueError):
        build_mapping_view_conditioned_descriptors(
            map_state=map_state, observation_cache=cache
        )

    payload = {
        "schema": "lafgs_v27_mapping_view_conditioned_anchor_descriptors",
        "version": 1,
        "protocol": "mapping_only_view_conditioned_anchor_descriptor",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_test_poses": False,
        "map_mutated": False,
        "adds_anchor_owners": False,
        "maximum_modes_per_anchor": 2,
        "minimum_mode_observations": 2,
        "selection_authority": "first_pass_estimated_pose_only",
        "inputs": {"stable_map": {}, "mapping_observation_cache": {}},
        "mode_features": torch.ones(2, 2, 2),
        "mode_observation_count": torch.ones(2, 2, dtype=torch.long) * 2,
        "mode_concentration": torch.ones(2, 2),
        "mode_valid": torch.ones(2, 2, dtype=torch.bool),
        "anchor_ids": torch.tensor([7, 11]),
    }
    invalid = copy.deepcopy(payload)
    invalid["mode_features"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_artifact(invalid)


def test_registered_mapping_view_may_have_no_surviving_anchor_observation() -> None:
    map_state, cache = _inputs()
    map_state["v6_mapping_query_names"].append("empty.png")
    cache["queries"]["empty.png"] = {
        "native_descriptors": torch.ones(1, 2),
        "pose_w2c": torch.eye(4),
    }
    built = build_mapping_view_conditioned_descriptors(
        map_state=map_state, observation_cache=cache
    )
    assert built["mode_observation_count"].sum().item() == 4
