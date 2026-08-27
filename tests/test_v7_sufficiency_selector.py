import pytest
import torch

from topology.v7_sufficiency_selector import (
    COMPLETION_LAYERS,
    CompactEdgeRegistry,
    CompactPoseInformation,
    EligibilityThresholds,
    LazyPoseInformation,
    SufficiencyTargets,
    eligibility_mask,
    reconstruct_mapping_candidate_evidence,
    select_v7_sufficiency,
)


def test_compact_edge_registry_matches_mapping_semantics() -> None:
    registry = CompactEdgeRegistry(
        torch.tensor([0, 3, 4]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 1, 5, 7]),
    )
    assert len(registry) == 2
    assert registry[0] == {0: (1, 2), 1: (5,)}
    assert registry[-1] == {1: (7,)}


def test_lazy_pose_information_is_full_se3_psd() -> None:
    pose = LazyPoseInformation(
        torch.tensor([[0.2, 0.1, 2.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0]),
        torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]),
        torch.eye(4).unsqueeze(0),
    )[0][0]
    assert pose.shape == (6, 6)
    assert torch.allclose(pose, pose.T)
    assert torch.linalg.eigvalsh(pose).min() >= -1e-8
    assert torch.count_nonzero(pose[:3, 3:]) > 0


def test_compact_pose_information_sums_same_query_edges() -> None:
    values = torch.stack((torch.eye(6), torch.eye(6) * 2, torch.eye(6) * 3))
    pose = CompactPoseInformation(
        torch.tensor([0, 2, 3]), torch.tensor([0, 0, 1]), values
    )
    assert torch.equal(pose[0][0], torch.eye(6) * 3)
    assert torch.equal(pose[1][1], torch.eye(6) * 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reconstruction")
def test_reconstructs_quality_from_rendered_mapping_rows() -> None:
    record = {
        "native_keypoints": torch.tensor([[55.0, 50.0], [45.0, 50.0]]),
        "native_descriptors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "native_K": torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        "pose_w2c": torch.eye(4),
        "native_input_hw": torch.tensor([100, 100]),
    }
    result = reconstruct_mapping_candidate_evidence(
        anchor_xyz=torch.tensor([[0.1, 0.0, 2.0], [-0.1, 0.0, 2.0]]),
        anchor_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        observation_offsets=torch.tensor([0, 1, 2]),
        query_indices=torch.tensor([0, 0]),
        keypoint_indices=torch.tensor([0, 1]),
        query_names=["q"],
        query_bins=torch.tensor([0]),
        rendered_feature_records={"q": record},
    )
    assert torch.allclose(result["descriptor_dispersion"], torch.zeros(2))
    assert torch.allclose(result["reprojection_error_px_mean"], torch.zeros(2))
    assert result["image_cell_identities"].tolist() == [10, 9]
    assert result["contract"]["uses_source_mapping_rgb"] is False


def _inputs():
    edges = {
        "matching": [{0: [0]}, {0: [1]}, {1: [0]}, {1: [1]}, {0: [2], 1: [2]}],
        "image_cell": [{0: [0]}, {0: [1]}, {1: [0]}, {1: [1]}, {0: [2], 1: [2]}],
        "view_family": [{0: [0]}, {0: [1]}, {1: [0]}, {1: [1]}, {0: [2], 1: [2]}],
        "depth_range": [{0: [0]}, {0: [1]}, {1: [0]}, {1: [1]}, {0: [2], 1: [2]}],
    }
    pose = [
        {0: torch.eye(6)},
        {0: torch.eye(6)},
        {1: torch.eye(6)},
        {1: torch.eye(6)},
        {0: torch.eye(6), 1: torch.eye(6)},
    ]
    targets = SufficiencyTargets(1, 2, 2, 2, 2, 0.0, 1.0, 5)
    return dict(
        anchor_ids=torch.tensor([10, 11, 12, 13, 14]),
        reliability=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        eligible=torch.ones(5, dtype=torch.bool),
        layer_edges=edges,
        pose_information=pose,
        query_count=2,
        targets=targets,
    )


def test_eligibility_is_fail_closed_and_reports_each_exclusion() -> None:
    threshold = EligibilityThresholds(0.5, 2, 2, 0.3, 2.0, 1.0, 0.1)
    kwargs = dict(
        geometry_reliability=torch.tensor([0.9, float("nan"), 0.9]),
        observation_count=torch.tensor([3, 3, 1]),
        view_family_count=torch.tensor([2, 2, 2]),
        descriptor_dispersion=torch.tensor([0.1, 0.1, 0.1]),
        reprojection_error=torch.tensor([1.0, 1.0, 1.0]),
        covariance_trace=torch.tensor([0.2, 0.2, 0.2]),
        parallax=torch.tensor([0.2, 0.2, 0.2]),
        render_artifact_supported=torch.tensor([False, False, True]),
        lineage_complete=torch.tensor([True, True, True]),
        thresholds=threshold,
    )
    eligible, report = eligibility_mask(**kwargs)
    assert eligible.tolist() == [True, False, False]
    assert report["non_finite"] == 1
    assert report["observation_count"] == 1
    assert report["render_artifact_support"] == 1


def test_initial_selection_has_all_layers_and_exact_reasons() -> None:
    result = select_v7_sufficiency(**_inputs())
    assert result["contract"]["same_api_initialization_and_update"] is True
    assert result["trust_region"]["initialization_unlimited"] is True
    assert result["unmet"] == {**{name: 0 for name in COMPLETION_LAYERS}, "pose": 0}
    assert set(result["primary_selection_reason"].values()) <= {
        "precision_core",
        "matching_completion",
        "image_cell_completion",
        "view_family_completion",
        "depth_range_completion",
        "pose_redundancy_completion",
    }


def test_selection_is_deterministic_and_anchor_id_breaks_ties() -> None:
    inputs = _inputs()
    inputs["reliability"] = torch.ones(5)
    first = select_v7_sufficiency(**inputs)
    second = select_v7_sufficiency(**inputs)
    assert torch.equal(first["selected_anchor_ids"], second["selected_anchor_ids"])
    assert first["selected_anchor_ids"][0].item() == 10


def test_update_obeys_symmetric_difference_budget_and_protects_critical() -> None:
    inputs = _inputs()
    inputs["previous_active"] = torch.tensor([10, 11, 12, 13])
    inputs["previous_clean_critical"] = torch.tensor([13])
    inputs["active_set_change_fraction"] = 0.25
    result = select_v7_sufficiency(**inputs)
    selected = set(result["selected_anchor_ids"].tolist())
    previous = {10, 11, 12, 13}
    assert 13 in selected
    assert len(selected ^ previous) <= 1
    assert result["trust_region"]["change_budget"] == 1


def test_nonaccept_like_ineligible_candidate_cannot_enter_initial_map() -> None:
    inputs = _inputs()
    inputs["eligible"][0] = False
    result = select_v7_sufficiency(**inputs)
    assert 10 not in result["selected_anchor_ids"].tolist()


def test_rejects_wrong_layer_order_and_unknown_previous_anchor() -> None:
    inputs = _inputs()
    inputs["layer_edges"] = dict(reversed(list(inputs["layer_edges"].items())))
    with pytest.raises(ValueError, match="layer order"):
        select_v7_sufficiency(**inputs)
    inputs = _inputs()
    inputs["previous_active"] = torch.tensor([999])
    with pytest.raises(ValueError, match="unknown anchor IDs"):
        select_v7_sufficiency(**inputs)
