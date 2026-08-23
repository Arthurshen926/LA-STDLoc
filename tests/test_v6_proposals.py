import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_proposals import descriptor_loss_proposal
from scripts.propose_v6_round import _jsonable

from topology.v6_anchor_map import subset_projective_anchor_map


def test_selection_report_tensors_are_json_serializable() -> None:
    assert _jsonable({"rows": torch.tensor([1, 2]), "nested": (torch.tensor(3),)}) == {
        "rows": [1, 2],
        "nested": [3],
    }


def test_subset_rebuilds_projective_csr() -> None:
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.arange(9).reshape(3, 3),
        "anchor_features": torch.eye(3),
        "source_primitive_ids": torch.full((3,), -1),
        "track_cluster_ids": torch.arange(3),
        "anchor_type": torch.ones(3, dtype=torch.long),
        "dependency_group_ids": torch.arange(3),
        "coarse_dependency_group_ids": torch.arange(3),
        "fine_identity_ids": torch.arange(3),
        "anchor_parent_identity_ids": torch.arange(3),
        "anchor_correlation_group_ids": torch.arange(3),
        "anchor_position_covariance": torch.eye(3).repeat(3, 1, 1),
        "anchor_matchability": torch.ones(3),
        "anchor_candidate_kind": ["a", "b", "c"],
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 3, 4]),
            "query_indices": torch.tensor([0, 0, 1, 2]),
            "keypoint_indices": torch.tensor([1, 2, 3, 4]),
        },
    }
    selected = subset_projective_anchor_map(state, torch.tensor([0, 2]))
    assert selected["anchor_ids"].tolist() == [0, 1]
    assert selected["projective_anchor_observations"]["observation_offsets"].tolist() == [0, 1, 2]
    assert selected["projective_anchor_observations"]["query_indices"].tolist() == [0, 2]


def test_descriptor_loss_uses_confusion_triplet_and_stores_residual() -> None:
    provider = GaussianRenderObservationProvider(
        {
            "uses_source_mapping_rgb": False,
            "queries": {
                "q": {
                    "native_keypoints": torch.tensor([[0.0, 0.0]]),
                    "native_descriptors": torch.tensor([[1.0, 0.0]]),
                    "native_scores": torch.tensor([1.0]),
                    "native_K": torch.eye(3),
                    "pose_w2c": torch.eye(4),
                    "native_input_hw": torch.tensor([2, 2]),
                }
            },
        }
    )
    state = {
        "anchor_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    feedback = {
        "schema": "self_localization_feedback_v1",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q"],
        "records": [
            {
                "descriptor_triplets": torch.tensor([[0, 0, 1, 0]]),
            }
        ],
    }
    before = float(
        provider.build_view(0).descriptors[0]
        @ (state["anchor_features"][0] - state["anchor_features"][1])
    )
    proposal = descriptor_loss_proposal(
        state,
        provider,
        feedback,
        trust_region=0.2,
        learning_rate=0.1,
        epochs=20,
        batch_size=1,
        maximum_triplets_per_query=1,
        clean_fraction=0.0,
        device="cpu",
    )
    after = float(
        provider.build_view(0).descriptors[0]
        @ (proposal["anchor_features"][0] - proposal["anchor_features"][1])
    )
    assert after > before
    assert proposal["anchor_descriptor_residual"].shape == (2, 2)
    assert proposal["v6_descriptor_distillation"]["final_ranking_loss"] < proposal[
        "v6_descriptor_distillation"
    ]["initial_ranking_loss"]
