import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_proposals import descriptor_loss_proposal
from common.hashing import sha256_file
from common.v6_contracts import ordered_query_registry_sha256
from scripts.propose_v6_round import (
    _attach_reconstruction_distillation,
    _jsonable,
    _load_query_indices,
    _validate_proposal_inputs,
)

from topology.v6_anchor_map import (
    compact_projective_deployment_map,
    subset_projective_anchor_map,
)


def test_selection_report_tensors_are_json_serializable() -> None:
    assert _jsonable({"rows": torch.tensor([1, 2]), "nested": (torch.tensor(3),)}) == {
        "rows": [1, 2],
        "nested": [3],
    }


def test_descriptor_training_split_is_sha_bound(tmp_path) -> None:
    split = tmp_path / "split.json"
    names = ["seq1/a", "seq2/a", "seq3/a"]
    split.write_text(
        __import__("json").dumps(
            {
                "schema": "lafgs_v6_sequence_block_descriptor_split",
                "version": 1,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "source_feedback_sha256": "f" * 64,
                "query_names_sha256": ordered_query_registry_sha256(names),
                "training_query_indices": [0, 2],
                "validation_query_indices": [1],
            }
        )
        + "\n"
    )
    rows, actual = _load_query_indices(
        split,
        sha256_file(split),
        feedback_sha256="f" * 64,
        query_names=names,
        require_source_feedback_match=True,
    )
    assert rows == [0, 2]
    assert actual == sha256_file(split)
    with pytest.raises(ValueError, match="split SHA differs"):
        _load_query_indices(split, "0" * 64, query_names=names)


def test_reconstruction_preserves_training_dependencies() -> None:
    state = {
        "v6_descriptor_distillation": {"training_query_indices": torch.tensor([0])},
        "v6_selection_distillation": {"training_query_indices": torch.tensor([1])},
        "v6_reconstruction_distillation": {
            "target_query_indices": torch.tensor([2]),
            "excluded_support_query_indices": torch.tensor([2, 3]),
            "reconstruction_round": 1,
        },
    }
    proposal = {}
    _attach_reconstruction_distillation(
        proposal,
        state,
        {"contract": {"target_queries_used_as_anchor_support": False}},
        target_query_indices=[4],
        excluded_support_query_indices=[4, 5],
    )
    assert proposal["v6_descriptor_distillation"] is not state[
        "v6_descriptor_distillation"
    ]
    assert proposal["v6_selection_distillation"][
        "training_query_indices"
    ].tolist() == [1]
    report = proposal["v6_reconstruction_distillation"]
    assert report["target_query_indices"].tolist() == [2, 4]
    assert report["excluded_support_query_indices"].tolist() == [2, 3, 4, 5]
    assert report["reconstruction_round"] == 2


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
        "v6_descriptor_distillation": {
            "updated_anchor_rows": torch.tensor([0, 2]),
            "round_updated_anchor_rows": torch.tensor([2]),
        },
    }
    selected = subset_projective_anchor_map(state, torch.tensor([0, 2]))
    assert selected["anchor_ids"].tolist() == [0, 1]
    assert selected["projective_anchor_observations"]["observation_offsets"].tolist() == [0, 1, 2]
    assert selected["projective_anchor_observations"]["query_indices"].tolist() == [0, 2]
    report = selected["v6_descriptor_distillation"]
    assert report["updated_anchor_rows"].tolist() == [0, 1]
    assert report["round_updated_anchor_rows"].tolist() == [1]


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
    report = proposal["v6_descriptor_distillation"]
    assert report["selected_query_indices"].tolist() == [0]
    assert 0.0 <= report["residual_cap_hit_fraction"] <= 1.0
    assert report["final_objective"] >= report["final_ranking_loss"]
    assert report["final_objective"] <= report["initial_objective"] + 1e-8


def test_compact_deployment_export_removes_dense_training_state() -> None:
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_features": torch.eye(2),
        "anchor_observation_features": torch.eye(2),
        "anchor_descriptor_residual": torch.ones((2, 2)) * 0.01,
        "v6_descriptor_distillation": {
            "updated_anchor_rows": torch.tensor([0, 1]),
            "selected_query_indices": torch.tensor([3]),
        },
        "provenance": {"uses_test_queries": False},
    }
    compact = compact_projective_deployment_map(state)
    assert "anchor_observation_features" not in compact
    assert "anchor_descriptor_residual" not in compact
    assert torch.equal(compact["anchor_features"], state["anchor_features"])
    assert compact["v6_descriptor_distillation"]["updated_anchor_count"] == 2
    assert compact["v6_descriptor_distillation"]["training_state_available"] is False
    assert compact["v6_descriptor_distillation"]["selected_query_indices"].tolist() == [3]


def test_descriptor_training_dependencies_accumulate_across_rounds() -> None:
    provider = GaussianRenderObservationProvider(
        {
            "uses_source_mapping_rgb": False,
            "queries": {
                "q0": {
                    "native_keypoints": torch.tensor([[0.0, 0.0]]),
                    "native_descriptors": torch.tensor([[1.0, 0.0, 0.0]]),
                    "native_scores": torch.tensor([1.0]),
                    "native_K": torch.eye(3),
                    "pose_w2c": torch.eye(4),
                    "native_input_hw": torch.tensor([2, 2]),
                },
                "q1": {
                    "native_keypoints": torch.tensor([[0.0, 0.0]]),
                    "native_descriptors": torch.tensor([[0.0, 1.0, 0.0]]),
                    "native_scores": torch.tensor([1.0]),
                    "native_K": torch.eye(3),
                    "pose_w2c": torch.eye(4),
                    "native_input_hw": torch.tensor([2, 2]),
                },
            },
        }
    )
    state = {
        "anchor_features": torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    }
    feedback = {
        "schema": "self_localization_feedback_v1",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q0", "q1"],
        "records": [
            {
                "failure_layers": ["L3"],
                "descriptor_triplets": torch.tensor([[0, 0, 1, 0]]),
            },
            {
                "failure_layers": ["L3"],
                "descriptor_triplets": torch.tensor([[0, 2, 3, 0]]),
            },
        ],
    }
    first = descriptor_loss_proposal(
        state,
        provider,
        feedback,
        training_query_indices=[0],
        trust_region=0.2,
        learning_rate=0.05,
        epochs=1,
        maximum_triplets_per_query=1,
        clean_fraction=0.0,
        device="cpu",
    )
    second = descriptor_loss_proposal(
        first,
        provider,
        feedback,
        training_query_indices=[1],
        trust_region=0.2,
        learning_rate=0.05,
        epochs=1,
        maximum_triplets_per_query=1,
        clean_fraction=0.0,
        device="cpu",
    )
    report = second["v6_descriptor_distillation"]
    assert report["training_query_indices"].tolist() == [0, 1]
    assert report["selected_query_indices"].tolist() == [0, 1]
    assert report["updated_anchor_rows"].tolist() == [0, 1, 2, 3]
    assert report["round_updated_anchor_rows"].tolist() == [2, 3]
    assert report["descriptor_training_round"] == 2


def test_proposal_inputs_fail_closed_on_cache_mismatch() -> None:
    state = {
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        }
    }
    cache = {
        "schema": "render_observation_cache_v2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
    }
    feedback = {
        "schema": "self_localization_feedback_v1",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "input_sha256": {"map": "m", "query_cache": "wrong"},
    }
    with pytest.raises(
        ValueError, match="feedback is not bound to the observation cache"
    ):
        _validate_proposal_inputs(
            state=state,
            cache=cache,
            feedback=feedback,
            map_sha="m",
            cache_sha="c",
        )


def test_proposal_inputs_reject_compact_map_and_registry_mismatch() -> None:
    cache = {
        "schema": "render_observation_cache_v2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": {"q": {}},
    }
    feedback = {
        "schema": "self_localization_feedback_v1",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["other"],
        "input_sha256": {"map": "m", "query_cache": "c"},
    }
    state = {
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        }
    }
    with pytest.raises(ValueError, match="registries differ"):
        _validate_proposal_inputs(
            state=state,
            cache=cache,
            feedback=feedback,
            map_sha="m",
            cache_sha="c",
        )
    state["provenance"]["v6_compact_deployment_export"] = True
    with pytest.raises(ValueError, match="compact deployment maps"):
        _validate_proposal_inputs(
            state=state,
            cache=cache,
            feedback=feedback,
            map_sha="m",
            cache_sha="c",
        )
