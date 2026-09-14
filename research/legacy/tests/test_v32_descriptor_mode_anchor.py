from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from map_learning.v27_view_conditioned_anchor_descriptor import (
    select_view_conditioned_anchor_features,
)
from map_learning.v32_descriptor_mode_anchor import (
    build_mapping_descriptor_modes,
    make_artifact,
    validate_artifact,
)


def _inputs(tmp_path) -> tuple[dict, dict, object, object]:
    names = [
        "s0/positive.png",
        "s1/positive.png",
        "s2/positive.png",
        "s3/positive.png",
        "s0/negative.png",
        "s1/negative.png",
        "s2/negative.png",
        "s3/negative.png",
    ]
    anchor_count = 3
    observations_per_anchor = len(names)
    query_rows = torch.arange(len(names)).repeat(anchor_count)
    keypoint_rows = torch.arange(anchor_count).repeat_interleave(len(names))
    base = F.normalize(
        torch.tensor([[1.0, 1.0], [-1.0, 0.1], [0.1, -1.0]]), dim=1
    )
    map_state = {
        "schema": "lafgs_materialized_anchor_map",
        "provenance": {"uses_test_queries": False},
        "anchor_ids": torch.tensor([4, 7, 9]),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "anchor_features": base,
        "v6_mapping_query_names": names,
        "projective_anchor_observations": {
            "observation_offsets": torch.arange(4) * observations_per_anchor,
            "query_indices": query_rows,
            "keypoint_indices": keypoint_rows,
        },
        "anchor_view_support": {
            "schema": "lafgs_v24_anchor_view_support",
            "uses_test_queries": False,
            "direction_modes": torch.tensor(
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                ]
            ),
            "mode_count": torch.tensor([2, 2, 2]),
        },
    }
    queries = {}
    for index, name in enumerate(names):
        pose = torch.eye(4)
        # C = -R^T t.  The first four views observe +z and the last four -z.
        pose[2, 3] = -5.0 if index < 4 else 5.0
        first = torch.tensor([1.0, 0.02]) if index < 4 else torch.tensor([0.02, 1.0])
        queries[name] = {
            "native_descriptors": torch.stack(
                (first, torch.tensor([-1.0, 0.1]), torch.tensor([0.1, -1.0]))
            ),
            "pose_w2c": pose,
        }
    cache = {
        "schema": "render_observation_cache_v2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": queries,
    }
    map_path = tmp_path / "map.pt"
    cache_path = tmp_path / "cache.pt"
    torch.save(map_state, map_path)
    torch.save(cache, cache_path)
    return map_state, cache, map_path, cache_path


def test_descriptor_clustering_discovers_repeatable_two_mode_anchor(tmp_path) -> None:
    map_state, cache, _, _ = _inputs(tmp_path)
    built = build_mapping_descriptor_modes(
        map_state=map_state,
        observation_cache=cache,
        maximum_modes_per_anchor=3,
        minimum_mode_observations=3,
        minimum_mapping_families=2,
        minimum_distortion_improvement=0.05,
        maximum_mode_cosine=0.9,
    )
    assert built["selected_mode_count"].tolist() == [2, 0, 0]
    assert built["mode_valid"][0].tolist() == [True, True, False]
    assert built["mode_observation_count"][0].tolist() == [4, 4, 0]
    assert built["mode_mapping_family_count"][0].tolist() == [4, 4, 0]
    assert built["single_mode_distortion"][0] > built["selected_mode_distortion"][0]
    assert torch.allclose(
        built["mode_features"].norm(dim=2), torch.ones(3, 3), atol=1e-6
    )


def test_estimated_pose_selects_descriptor_cluster_direction(tmp_path) -> None:
    map_state, cache, _, _ = _inputs(tmp_path)
    built = build_mapping_descriptor_modes(
        map_state=map_state,
        observation_cache=cache,
        minimum_distortion_improvement=0.05,
        maximum_mode_cosine=0.9,
    )
    pose = torch.eye(4)
    pose[2, 3] = -5.0
    selected, report = select_view_conditioned_anchor_features(
        base_anchor_features=map_state["anchor_features"],
        anchor_xyz=map_state["anchor_xyz"],
        direction_modes=built["mode_direction_vectors"],
        direction_radius_deg=built["mode_direction_radius_deg"],
        baseline_pose_w2c=pose,
        mode_features=built["mode_features"],
        mode_valid=built["mode_valid"],
        mode_concentration=built["mode_concentration"],
    )
    assert selected[0, 0] > selected[0, 1]
    assert torch.allclose(selected[1:], map_state["anchor_features"][1:])
    assert report["selected_mode_anchor_count"] == 1


def test_artifact_is_mapping_only_and_globally_owner_authorized(tmp_path) -> None:
    map_state, cache, map_path, cache_path = _inputs(tmp_path)
    payload = make_artifact(
        map_path=map_path,
        observation_cache_path=cache_path,
        map_state=map_state,
        observation_cache=cache,
        minimum_distortion_improvement=0.05,
        maximum_mode_cosine=0.9,
    )
    validate_artifact(payload, map_state=map_state)
    assert payload["uses_test_queries"] is False
    assert payload["map_mutated"] is False
    assert payload["adds_anchor_owners"] is False
    assert payload["mode_authorized"][0, :2].all()

    corrupt = copy.deepcopy(payload)
    corrupt["mode_direction_vectors"][0, 0] = 0
    with pytest.raises(ValueError):
        validate_artifact(corrupt, map_state=map_state)


def test_rejects_test_or_real_rgb_mapping_cache(tmp_path) -> None:
    map_state, cache, _, _ = _inputs(tmp_path)
    cache["uses_source_mapping_rgb"] = True
    with pytest.raises(ValueError):
        build_mapping_descriptor_modes(map_state=map_state, observation_cache=cache)
