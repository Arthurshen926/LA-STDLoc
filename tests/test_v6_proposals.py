import pytest
import torch
import torch.nn.functional as F

from evidence.observation_provider import GaussianRenderObservationProvider
import map_learning.v6_proposals as v6_proposals
from map_learning.v6_proposals import (
    descriptor_loss_proposal,
    selection_only_proposal,
)
from common.hashing import sha256_file
from common.v6_contracts import (
    DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
    DESCRIPTOR_POSE_WEIGHT_SEMANTICS,
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
    ordered_query_registry_sha256,
)
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


def _with_unaffected_projective_loo(
    state: dict,
    provider: GaussianRenderObservationProvider,
) -> dict:
    """Attach a minimal V6 replay contract for unit losses with no affected row."""

    output = dict(state)
    anchor_count = int(torch.as_tensor(output["anchor_features"]).shape[0])
    output["anchor_ids"] = torch.arange(anchor_count)
    output["v6_mapping_query_names"] = list(provider.names)
    output["v6_mapping_query_bins"] = torch.arange(len(provider))
    output["projective_anchor_construction"] = {
        "final_xyz_source": "fixed_camera_robust_ray_triangulation"
    }
    output["projective_anchor_observations"] = {
        "observation_offsets": torch.zeros(anchor_count + 1, dtype=torch.long),
        "query_indices": torch.empty(0, dtype=torch.long),
        "keypoint_indices": torch.empty(0, dtype=torch.long),
    }
    return output


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
    assert (
        proposal["v6_descriptor_distillation"]
        is not state["v6_descriptor_distillation"]
    )
    assert proposal["v6_selection_distillation"]["training_query_indices"].tolist() == [
        1
    ]
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
    assert selected["projective_anchor_observations"][
        "observation_offsets"
    ].tolist() == [0, 1, 2]
    assert selected["projective_anchor_observations"]["query_indices"].tolist() == [
        0,
        2,
    ]
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
    state = _with_unaffected_projective_loo(
        {
            "anchor_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        },
        provider,
    )
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "descriptor_triplet_pose_weight_semantics": (
            DESCRIPTOR_POSE_WEIGHT_SEMANTICS
        ),
        "descriptor_triplet_clean_semantics": DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q"],
        "records": [
            {
                # The cached label is intentionally wrong.  Proposal training
                # must classify this from the current query-local margin.
                "descriptor_triplets": torch.tensor([[0, 0, 1, 1]]),
                "descriptor_triplet_pose_weights": torch.tensor([1.0]),
                "descriptor_triplet_harmful_inlier_mask": torch.tensor([True]),
                "descriptor_identity_supervision_available": True,
                "exact_identity_positive_pairs": torch.tensor([[0, 0]]),
                "affected_anchor_policy": "rebuild",
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
        pose_critical_weight=2.0,
        device="cpu",
    )
    after = float(
        provider.build_view(0).descriptors[0]
        @ (proposal["anchor_features"][0] - proposal["anchor_features"][1])
    )
    assert after > before
    assert proposal["anchor_descriptor_residual"].shape == (2, 2)
    assert (
        proposal["v6_descriptor_distillation"]["final_ranking_loss"]
        < proposal["v6_descriptor_distillation"]["initial_ranking_loss"]
    )
    report = proposal["v6_descriptor_distillation"]
    assert report["selected_query_indices"].tolist() == [0]
    assert 0.0 <= report["residual_cap_hit_fraction"] <= 1.0
    assert report["final_objective"] >= report["final_ranking_loss"]
    assert report["final_objective"] <= report["initial_objective"] + 1e-8
    assert report["effective_coordinate_learning_rate"] == pytest.approx(0.1 / 2**0.5)
    assert report["error_triplet_count"] == 1
    assert report["clean_triplet_count"] == 0
    assert report["clean_labels_recomputed_from_query_local_current_margin"] is True
    assert report["positive_pose_weight_triplet_count"] == 1
    assert report["pose_critical_weight"] == 2.0

    feedback["records"][0]["affected_anchor_policy"] = "purge"
    with pytest.raises(ValueError, match="purge feedback is diagnostic-only"):
        descriptor_loss_proposal(state, provider, feedback, device="cpu")
    feedback["records"][0]["affected_anchor_policy"] = "rebuild"
    feedback["records"][0]["exact_identity_positive_pairs"] = torch.empty(
        (0, 2), dtype=torch.long
    )
    with pytest.raises(ValueError, match="lacks exact active identity"):
        descriptor_loss_proposal(state, provider, feedback, device="cpu")


def test_descriptor_loss_scores_sparse_query_local_loo_bases() -> None:
    xyz = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.5, 6.0]])
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    names = []
    queries = {}
    for query_index, center_x in enumerate((0.0, 0.5, 1.0, 1.5, 2.0)):
        name = f"q{query_index}"
        names.append(name)
        pose = torch.eye(4)
        pose[0, 3] = -center_x
        camera = xyz @ pose[:3, :3].T + pose[:3, 3]
        physical = (camera @ K.T)[:, :2] / camera[:, 2:]
        # q0 makes the full positive look perfect.  Every remaining observation
        # makes its exact LOO descriptor orthogonal to q0 instead.
        descriptors = (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            if query_index == 0
            else torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        queries[name] = {
            "native_keypoints": physical - 0.5,
            "native_descriptors": descriptors,
            "native_scores": torch.ones(2),
            "native_K": K,
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
        }
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries},
        query_names=names,
    )
    state = {
        "anchor_ids": torch.arange(2),
        "anchor_xyz": xyz,
        "anchor_features": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "anchor_observation_features": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "v6_mapping_query_names": names,
        "v6_mapping_query_bins": torch.arange(5),
        "projective_anchor_construction": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation"
        },
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": torch.tensor([0, 5, 10]),
            "query_indices": torch.arange(5).repeat(2),
            "keypoint_indices": torch.cat(
                (
                    torch.zeros(5, dtype=torch.long),
                    torch.ones(5, dtype=torch.long),
                )
            ),
        },
    }
    records = []
    for query_index in range(5):
        records.append(
            {
                "failure_layers": ["L3"] if query_index == 0 else [],
                "descriptor_triplets": (
                    torch.tensor([[0, 0, 1, 0]])
                    if query_index == 0
                    else torch.empty((0, 4), dtype=torch.long)
                ),
                "descriptor_triplet_pose_weights": (
                    torch.tensor([0.0])
                    if query_index == 0
                    else torch.empty(0)
                ),
                "descriptor_triplet_harmful_inlier_mask": (
                    torch.tensor([False])
                    if query_index == 0
                    else torch.empty(0, dtype=torch.bool)
                ),
                "descriptor_identity_supervision_available": True,
                "exact_identity_positive_pairs": (
                    torch.tensor([[0, 0]])
                    if query_index == 0
                    else torch.empty((0, 2), dtype=torch.long)
                ),
                "excluded_query_indices": torch.tensor([query_index]),
                "affected_anchor_policy": "rebuild",
            }
        )
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "descriptor_triplet_pose_weight_semantics": (
            DESCRIPTOR_POSE_WEIGHT_SEMANTICS
        ),
        "descriptor_triplet_clean_semantics": DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": names,
        "records": records,
    }
    proposal = descriptor_loss_proposal(
        state,
        provider,
        feedback,
        training_query_indices=[0],
        trust_region=0.2,
        margin=0.0,
        temperature=0.04,
        learning_rate=0.01,
        epochs=1,
        batch_size=1,
        maximum_triplets_per_query=1,
        clean_fraction=0.0,
        trust_weight=0.0,
        device="cpu",
    )
    report = proposal["v6_descriptor_distillation"]
    expected_loo_loss = float(F.softplus(torch.tensor(25.0)) * 0.04)
    full_bank_loss = float(F.softplus(torch.tensor(0.0)) * 0.04)
    assert report["initial_ranking_loss"] == pytest.approx(expected_loo_loss)
    assert report["initial_ranking_loss"] != pytest.approx(full_bank_loss)
    assert report["query_local_loo_pair_count"] == 2
    assert report["query_local_loo_affected_pair_count"] == 2
    assert report["query_observations_excluded_from_training_anchor_bases"] is True
    assert report["query_local_loo_dense_query_anchor_bank_materialized"] is False


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
    assert compact["v6_descriptor_distillation"]["selected_query_indices"].tolist() == [
        3
    ]


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
    state = _with_unaffected_projective_loo(
        {
            "anchor_features": torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        },
        provider,
    )
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "descriptor_triplet_pose_weight_semantics": (
            DESCRIPTOR_POSE_WEIGHT_SEMANTICS
        ),
        "descriptor_triplet_clean_semantics": DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q0", "q1"],
        "records": [
            {
                "failure_layers": ["L3"],
                "descriptor_triplets": torch.tensor([[0, 0, 1, 0]]),
                "descriptor_triplet_pose_weights": torch.tensor([0.0]),
                "descriptor_triplet_harmful_inlier_mask": torch.tensor([False]),
                "descriptor_identity_supervision_available": True,
                "exact_identity_positive_pairs": torch.tensor([[0, 0]]),
                "affected_anchor_policy": "rebuild",
            },
            {
                "failure_layers": ["L3"],
                "descriptor_triplets": torch.tensor([[0, 2, 3, 0]]),
                "descriptor_triplet_pose_weights": torch.tensor([0.0]),
                "descriptor_triplet_harmful_inlier_mask": torch.tensor([False]),
                "descriptor_identity_supervision_available": True,
                "exact_identity_positive_pairs": torch.tensor([[0, 2]]),
                "affected_anchor_policy": "rebuild",
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
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
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
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
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


def test_selection_uses_image_cells_and_independent_layer_targets(monkeypatch) -> None:
    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return {
            "selected_anchor_rows": torch.tensor([0, 2]),
            "unmet": {},
        }

    monkeypatch.setattr(v6_proposals, "select_layered_sufficiency", fake_select)
    monkeypatch.setattr(
        v6_proposals,
        "subset_projective_anchor_map",
        lambda state, selected: {"anchor_ids": state["anchor_ids"][selected]},
    )
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_matchability": torch.tensor([0.9, 0.8, 0.7]),
        "v6_selection_distillation": {
            "training_query_indices": torch.tensor([0]),
            "training_query_registry_explicit": True,
            "selection_round": 1,
        },
    }
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "records": [
            {
                "visible_anchor_ids": torch.tensor([1]),
                "visible_anchor_image_cells": torch.tensor([1]),
                "detectable_pairs": torch.tensor([[1, 1]]),
                "matching_pairs": torch.tensor([[1, 1]]),
                "clean_inlier_pose_anchor_ids": torch.tensor([1]),
                "clean_inlier_pose_information": torch.eye(6).unsqueeze(0),
            },
            {
                "visible_anchor_ids": torch.tensor([0, 1, 2]),
                "visible_anchor_image_cells": torch.tensor([5, 5, 7]),
                "detectable_pairs": torch.tensor([[10, 0], [11, 1]]),
                "matching_pairs": torch.tensor([[10, 0]]),
                "clean_inlier_pose_anchor_ids": torch.tensor([0, 2]),
                "clean_inlier_pose_information": torch.stack(
                    [torch.eye(6), torch.eye(6) * 2]
                ),
            }
        ],
    }
    proposal, _ = selection_only_proposal(
        state,
        feedback,
        maximum_anchors=2,
        visibility_target=2,
        detectability_target=3,
        matching_target=4,
        pose_logdet_target=5.0,
        pose_min_eigenvalue_target=0.25,
        training_query_indices=[1],
    )

    assert captured["layer_edges"]["visibility"] == [
        {0: (5,)},
        {0: (5,)},
        {0: (7,)},
    ]
    assert captured["visibility_target"] == 2
    assert captured["detectability_target"] == 3
    assert captured["matching_target"] == 4
    assert captured["pose_min_eigenvalue_target"] == 0.25
    assert captured["query_count"] == 1
    assert set(captured["pose_information"][0]) == {0}
    assert set(captured["pose_information"][2]) == {0}
    report = proposal["v6_selection_distillation"]
    assert report["visibility_evidence_unit"] == "query_image_grid_cell"
    assert report["pose_evidence_unit"] == "unique_anchor_per_query"
    assert report["training_query_indices"].tolist() == [0, 1]
    assert report["round_training_query_indices"].tolist() == [1]
    assert report["training_query_registry_explicit"] is True
    assert report["selection_round"] == 2


def test_selection_rejects_duplicate_pose_rows_for_one_anchor() -> None:
    state = {
        "anchor_ids": torch.arange(1),
        "anchor_matchability": torch.ones(1),
    }
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "records": [
            {
                "visible_anchor_ids": torch.tensor([0]),
                "visible_anchor_image_cells": torch.tensor([0]),
                "detectable_pairs": torch.tensor([[0, 0]]),
                "matching_pairs": torch.tensor([[0, 0]]),
                "clean_inlier_pose_anchor_ids": torch.tensor([0, 0]),
                "clean_inlier_pose_information": torch.stack(
                    [torch.eye(6), torch.eye(6)]
                ),
            }
        ],
    }
    with pytest.raises(ValueError, match="one row per unique Anchor"):
        selection_only_proposal(
            state,
            feedback,
            maximum_anchors=1,
            visibility_target=1,
            detectability_target=1,
            matching_target=1,
            pose_logdet_target=0.0,
        )
