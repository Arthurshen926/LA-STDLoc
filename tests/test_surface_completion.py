from __future__ import annotations

import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from topology.anchor_construction import SurfaceCompletionProvider
from topology.surface_completion import (
    materialize_gaussian_surface_completion,
    surface_completion_selector_inputs,
)


def _provider() -> GaussianRenderObservationProvider:
    records = {}
    for index in range(3):
        records[f"seq-{index}/image.png"] = {
            "native_keypoints": torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
            "native_descriptors": torch.nn.functional.normalize(
                torch.tensor([[1.0, 0.1 * index, 0.0], [0.0, 1.0, float(index)]]),
                dim=1,
            ),
            "native_scores": torch.tensor([0.9, 0.1]),
            "native_K": torch.tensor(
                [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]
            ),
            "pose_w2c": torch.eye(4),
            "native_input_hw": [4, 4],
            "native_valid_mask": torch.ones((4, 4), dtype=torch.bool),
            "native_depth": torch.full((4, 4), 2.0),
            "native_alpha": torch.ones((4, 4)),
        }
    return GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": records},
        query_bins=torch.tensor([0, 1, 2]),
    )


def test_rendered_surface_completion_builds_nontrack_multiview_candidate() -> None:
    result = materialize_gaussian_surface_completion(
        _provider(),
        voxel_size_m=0.02,
        maximum_candidates=4,
        maximum_rows_per_view=1,
        alpha_minimum=0.05,
        minimum_observations=3,
        minimum_views=3,
        minimum_pose_bins=3,
        descriptor_trim_fraction=0.0,
    )
    assert result["surface_completion"]["selected_surface_component_count"] == 1
    assert result["surface_completion"]["legal_observation_count"] == 3
    assert result["track_cluster_ids"].tolist() == [-1]
    assert result["source_primitive_ids"].tolist() == [-1]
    assert result["surface_completion_observations"][
        "observation_offsets"
    ].tolist() == [0, 3]
    batch = SurfaceCompletionProvider(
        result, torch.tensor([0]), maximum_candidates=1
    ).materialize()
    assert batch.observation_offsets.tolist() == [0, 3]
    assert (
        batch.parent_identity_ids.tolist()
        == result["gaussian_support_component_ids"].tolist()
    )
    teacher, graph = surface_completion_selector_inputs(result, _provider())
    assert teacher["anchor_count"] == 1
    assert (
        sum(int(record["positive_indices"].numel()) for record in teacher["records"])
        == 3
    )
    assert graph["provenance_legal_hit_strong_count"].tolist() == [3]
    assert graph["provenance_harmful_solver_inlier_count"].tolist() == [0]


def test_rendered_surface_completion_is_disabled_by_zero_capacity() -> None:
    result = materialize_gaussian_surface_completion(
        _provider(),
        voxel_size_m=0.02,
        maximum_candidates=0,
        maximum_rows_per_view=1,
        alpha_minimum=0.05,
        minimum_observations=3,
        minimum_views=3,
        minimum_pose_bins=3,
        descriptor_trim_fraction=0.0,
    )
    assert result["anchor_ids"].numel() == 0
    assert result["surface_completion"]["eligible_surface_component_count"] == 1
