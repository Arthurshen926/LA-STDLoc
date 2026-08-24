import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from evidence.observation_provider import GaussianRenderObservationProvider
import map_learning.v6_proposals as v6_proposals
from map_learning.v6_proposals import (
    descriptor_loss_proposal,
    geometry_consensus_descriptor_feedback,
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
    _reconstruction_training_scope,
    _training_split_input_sha,
    _validate_proposal_inputs,
)
import scripts.propose_v6_round as propose_v6_round

from topology.v6_anchor_map import (
    compact_projective_deployment_map,
    subset_projective_anchor_map,
)
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
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


def test_geometry_consensus_adds_only_negative_winner_alternatives() -> None:
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "records": [
            {
                "query_rows": torch.tensor([3, 4]),
                "winner_anchor_ids": torch.tensor([8, 9]),
                "top1_negative_mask": torch.tensor([True, False]),
                "projective_compatible_ambiguous_pairs": torch.tensor(
                    [[3, 7], [3, 6], [4, 5]]
                ),
                "descriptor_triplets": torch.empty((0, 4), dtype=torch.long),
                "descriptor_triplet_harmful_inlier_mask": torch.empty(
                    0, dtype=torch.bool
                ),
                "descriptor_triplet_pose_weights": torch.empty(0),
                "descriptor_triplet_legal_pair_clean_mask": torch.empty(
                    0, dtype=torch.bool
                ),
                "inlier_query_rows": torch.tensor([3]),
            }
        ],
    }
    augmented, count = geometry_consensus_descriptor_feedback(feedback)
    record = augmented["records"][0]
    assert count == 1
    assert record["descriptor_triplets"].tolist() == [[3, 6, 8, 0]]
    assert record["descriptor_triplet_harmful_inlier_mask"].tolist() == [True]
    assert record["descriptor_triplet_pose_weights"].tolist() == [0.0]
    assert feedback["records"][0]["descriptor_triplets"].numel() == 0


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
    old_split_sha = "b" * 64
    new_split_sha = "a" * 64
    state = {
        "v6_descriptor_distillation": {"training_query_indices": torch.tensor([0])},
        "v6_selection_distillation": {"training_query_indices": torch.tensor([1])},
        "v6_reconstruction_distillation": {
            "version": 2,
            "target_query_indices": torch.tensor([0]),
            "excluded_support_query_indices": torch.tensor([0, 1]),
            "training_query_indices": torch.tensor([0, 1]),
            "training_query_registry_explicit": True,
            "training_split_artifact_sha256s": [old_split_sha],
            "reconstruction_round": 1,
        },
    }
    proposal = {}
    _attach_reconstruction_distillation(
        proposal,
        state,
        {
            "contract": {
                "target_queries_seed_regions": True,
                "support_queries_restricted": True,
                "target_queries_used_as_anchor_support": False,
            }
        },
        target_query_indices=[4],
        excluded_support_query_indices=[4, 5],
        training_query_indices=[4, 5],
        query_count=6,
        training_split_sha256=new_split_sha,
    )
    assert (
        proposal["v6_descriptor_distillation"]
        is not state["v6_descriptor_distillation"]
    )
    assert proposal["v6_selection_distillation"]["training_query_indices"].tolist() == [
        1
    ]
    report = proposal["v6_reconstruction_distillation"]
    assert report["version"] == 2
    assert report["target_query_indices"].tolist() == [0, 4]
    assert report["excluded_support_query_indices"].tolist() == [0, 1, 4, 5]
    assert report["training_query_indices"].tolist() == [0, 1, 4, 5]
    assert report["round_training_query_indices"].tolist() == [4, 5]
    assert report["validation_query_indices"].tolist() == [2, 3]
    assert report["round_validation_query_indices"].tolist() == [0, 1, 2, 3]
    assert report["training_split_artifact_sha256s"] == [
        old_split_sha,
        new_split_sha,
    ]
    assert report["validation_queries_used_as_target_seed_or_support"] is False
    assert report["reconstruction_round"] == 2


def test_reconstruction_scope_keeps_validation_sequence_out_of_arm() -> None:
    records = [
        {
            "query_index": 0,
            "failure_layers": ["L1"],
            "excluded_query_indices": torch.tensor([0, 1]),
        },
        {
            "query_index": 1,
            "failure_layers": ["L1"],
            "excluded_query_indices": torch.tensor([0, 1, 2]),
        },
        {"query_index": 2, "failure_layers": ["L3"]},
        {"query_index": 3, "failure_layers": ["L1"]},
        {
            "query_index": 4,
            "failure_layers": ["L1"],
            "excluded_query_indices": torch.tensor([3, 4]),
        },
    ]
    scope = _reconstruction_training_scope(
        {"records": records}, training_query_indices=[0, 2, 4]
    )
    assert scope == {
        "training_query_indices": [0, 2, 4],
        "validation_query_indices": [1, 3],
        "target_query_indices": [0, 4],
        "excluded_support_query_indices": [0, 4],
    }


def test_reconstruction_split_sha_uses_mapping_and_arm_specific_keys() -> None:
    digest = "a" * 64
    assert _training_split_input_sha("reconstruction", digest) == {
        "mapping_training_query_indices": digest,
        "reconstruction_training_query_indices": digest,
    }
    assert _training_split_input_sha("selection", digest) == {
        "mapping_training_query_indices": digest,
        "descriptor_training_query_indices": digest,
    }


def test_reconstruction_run_passes_only_training_queries_to_completion(
    tmp_path, monkeypatch
) -> None:
    split_sha = "a" * 64
    feedback = {
        "query_names": ["train/l1", "seq2/l1", "train/ok"],
        "records": [
            {
                "query_index": 0,
                "failure_layers": ["L1"],
                "excluded_query_indices": torch.tensor([0, 1]),
            },
            {
                "query_index": 1,
                "failure_layers": ["L1"],
                "excluded_query_indices": torch.tensor([0, 1]),
            },
            {"query_index": 2, "failure_layers": []},
        ],
    }
    artifacts = {
        "map": ({"provenance": {}, "anchor_ids": torch.tensor([0])}, "m" * 64),
        "observation cache": ({}, "c" * 64),
        "feedback": ({"feedback": feedback}, "f" * 64),
        "association graph": ({}, "e" * 64),
    }
    captured = {}
    split_call = {}

    monkeypatch.setattr(propose_v6_round, "_clean_commit", lambda: "0" * 40)
    monkeypatch.setattr(
        propose_v6_round,
        "_load",
        lambda path, expected, label: artifacts[label],
    )
    monkeypatch.setattr(
        propose_v6_round, "_validate_proposal_inputs", lambda **kwargs: None
    )
    monkeypatch.setattr(propose_v6_round, "require_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        propose_v6_round, "GaussianRenderObservationProvider", lambda cache: object()
    )
    def load_split(*args, **kwargs):
        split_call.update(kwargs)
        return [0, 2], split_sha

    monkeypatch.setattr(propose_v6_round, "_load_query_indices", load_split)

    def unavailable_completion(*args, **kwargs):
        captured.update(kwargs)
        raise ValueError("no unused render-valid observations for completion")

    monkeypatch.setattr(
        propose_v6_round, "build_projective_completion", unavailable_completion
    )
    args = SimpleNamespace(
        arm="reconstruction",
        map=tmp_path / "map.pt",
        expected_map_sha256="m" * 64,
        observation_cache=tmp_path / "cache.pt",
        expected_observation_cache_sha256="c" * 64,
        feedback=tmp_path / "feedback.pt",
        expected_feedback_sha256="f" * 64,
        descriptor_training_query_indices=tmp_path / "split.json",
        expected_descriptor_training_query_indices_sha256=split_sha,
        association_graph=tmp_path / "association.pt",
        expected_association_graph_sha256="e" * 64,
        output_dir=tmp_path / "proposal",
        device="cpu",
        descriptor_trust_region=0.05,
        descriptor_margin=0.05,
        descriptor_temperature=0.04,
        descriptor_learning_rate=0.02,
        descriptor_epochs=1,
        descriptor_batch_size=1,
        descriptor_maximum_triplets_per_query=1,
        descriptor_clean_fraction=0.25,
        descriptor_clean_weight=0.25,
        descriptor_trust_weight=0.1,
        descriptor_pose_critical_weight=0.0,
        descriptor_tail_query_weight=0.0,
        maximum_anchors=10,
        visibility_target=1,
        detectability_target=1,
        matching_target=1,
        pose_logdet_target=0.0,
        pose_min_eigenvalue_target=0.0,
        selection_pose_information_chunk_size=16,
        completion_voxel_size_m=0.05,
        alpha_minimum=0.05,
        completion_minimum_similarity=0.7,
        minimum_margin=0.01,
        maximum_epipolar_error_px=2.0,
        minimum_views=3,
        minimum_camera_families=2,
        completion_maximum_rows_per_view=32,
        completion_safety_maximum_components=100,
    )

    report = propose_v6_round.run(args)

    assert split_call["require_source_feedback_match"] is True
    assert split_call["feedback_sha256"] == "f" * 64
    assert captured["eligible_query_indices"] == [0, 2]
    assert captured["target_query_indices"] == [0]
    assert captured["excluded_support_query_indices"] == [0]
    assert report["proposal_available"] is False
    assert report["reconstruction_training_scope"] == {
        "training_query_indices": [0, 2],
        "validation_query_indices": [1],
        "target_query_indices": [0],
        "excluded_support_query_indices": [0],
    }
    assert report["input_sha256"] == {
        "map": "m" * 64,
        "observation_cache": "c" * 64,
        "feedback": "f" * 64,
        "mapping_training_query_indices": split_sha,
        "reconstruction_training_query_indices": split_sha,
        "association_graph": "e" * 64,
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
                "query_rows": torch.tensor([0]),
                "winner_anchor_ids": torch.tensor([1]),
                "exact_identity_pairs": torch.tensor([[0, 0]]),
                "active_identity_pairs": torch.tensor([[0, 0]]),
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

    set_proposal = descriptor_loss_proposal(
        state,
        provider,
        feedback,
        trust_region=0.2,
        learning_rate=0.1,
        epochs=10,
        batch_size=1,
        maximum_triplets_per_query=1,
        clean_fraction=0.0,
        loss_mode="set_consensus",
        consensus_count_target=1.0,
        consensus_cell_target=1.0,
        device="cpu",
    )
    set_report = set_proposal["v6_descriptor_distillation"]
    assert set_report["loss_mode"] == "set_consensus"
    assert set_report["set_consensus_joint_query_objective"] is True
    assert set_report["final_objective"] < set_report["initial_objective"]

    feedback["records"][0]["affected_anchor_policy"] = "purge"
    with pytest.raises(ValueError, match="purge feedback is diagnostic-only"):
        descriptor_loss_proposal(state, provider, feedback, device="cpu")
    feedback["records"][0]["affected_anchor_policy"] = "rebuild"
    feedback["records"][0]["exact_identity_positive_pairs"] = torch.empty(
        (0, 2), dtype=torch.long
    )
    with pytest.raises(ValueError, match="active identity partition differs"):
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
                "query_rows": torch.tensor([0, 1]),
                "winner_anchor_ids": torch.tensor([1, 1]),
                "exact_identity_pairs": (
                    torch.tensor([[0, 0]])
                    if query_index == 0
                    else torch.empty((0, 2), dtype=torch.long)
                ),
                "active_identity_pairs": (
                    torch.tensor([[0, 0]])
                    if query_index == 0
                    else torch.empty((0, 2), dtype=torch.long)
                ),
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


def _pose_weight_fixture() -> tuple[dict, GaussianRenderObservationProvider, dict]:
    provider = GaussianRenderObservationProvider(
        {
            "uses_source_mapping_rgb": False,
            "queries": {
                "q": {
                    "native_keypoints": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                    "native_descriptors": torch.eye(2),
                    "native_scores": torch.ones(2),
                    "native_K": torch.eye(3),
                    "pose_w2c": torch.eye(4),
                    "native_input_hw": torch.tensor([2, 2]),
                }
            },
        }
    )
    state = _with_unaffected_projective_loo(
        {
            "anchor_features": torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        },
        provider,
    )
    identity = torch.tensor([[0, 0], [1, 0]])
    record = {
        "failure_layers": ["L3", "L4"],
        "descriptor_triplets": torch.tensor([[0, 0, 1, 0], [1, 0, 1, 0]]),
        "descriptor_triplet_pose_weights": torch.tensor([1.0, 0.0]),
        "descriptor_triplet_harmful_inlier_mask": torch.tensor([True, False]),
        "descriptor_identity_supervision_available": True,
        "query_rows": torch.tensor([0, 1]),
        "winner_anchor_ids": torch.tensor([1, 1]),
        "exact_identity_pairs": identity,
        "active_identity_pairs": identity,
        "exact_identity_positive_pairs": identity,
        "affected_anchor_policy": "rebuild",
        "te_cm": 100.0,
    }
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
        "records": [record],
    }
    return state, provider, feedback


def test_pose_weights_survive_memory_chunking_and_pure_l4_filters_rows() -> None:
    state, provider, feedback = _pose_weight_fixture()

    def train(*, batch_size: int, pose_weight: float) -> dict:
        return descriptor_loss_proposal(
            state,
            provider,
            feedback,
            trust_region=0.2,
            learning_rate=0.1,
            epochs=1,
            batch_size=batch_size,
            maximum_triplets_per_query=2,
            clean_fraction=0.0,
            trust_weight=0.0,
            pose_critical_weight=pose_weight,
            device="cpu",
        )

    weighted_chunked = train(batch_size=1, pose_weight=3.0)
    weighted_full = train(batch_size=2, pose_weight=3.0)
    unweighted = train(batch_size=1, pose_weight=0.0)
    assert torch.allclose(
        weighted_chunked["anchor_descriptor_residual"],
        weighted_full["anchor_descriptor_residual"],
        atol=1e-6,
    )
    assert not torch.allclose(
        weighted_chunked["anchor_descriptor_residual"],
        unweighted["anchor_descriptor_residual"],
    )
    assert weighted_chunked["v6_descriptor_distillation"][
        "weighted_gradient_uses_fixed_global_denominator"
    ] is True

    feedback["records"][0]["failure_layers"] = ["L4"]
    pure_l4 = train(batch_size=1, pose_weight=3.0)
    report = pure_l4["v6_descriptor_distillation"]
    assert report["triplet_count"] == 1
    assert report["positive_pose_weight_triplet_count"] == 1
    del feedback["records"][0]["affected_anchor_policy"]
    with pytest.raises(ValueError, match="explicit LOO policy"):
        train(batch_size=1, pose_weight=3.0)


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
                "query_rows": torch.tensor([0]),
                "winner_anchor_ids": torch.tensor([1]),
                "exact_identity_pairs": torch.tensor([[0, 0]]),
                "active_identity_pairs": torch.tensor([[0, 0]]),
                "exact_identity_positive_pairs": torch.tensor([[0, 0]]),
                "affected_anchor_policy": "rebuild",
            },
            {
                "failure_layers": ["L3"],
                "descriptor_triplets": torch.tensor([[0, 2, 3, 0]]),
                "descriptor_triplet_pose_weights": torch.tensor([0.0]),
                "descriptor_triplet_harmful_inlier_mask": torch.tensor([False]),
                "descriptor_identity_supervision_available": True,
                "query_rows": torch.tensor([0]),
                "winner_anchor_ids": torch.tensor([3]),
                "exact_identity_pairs": torch.tensor([[0, 2]]),
                "active_identity_pairs": torch.tensor([[0, 2]]),
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


def _selection_provider() -> GaussianRenderObservationProvider:
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    queries = {}
    for name, row_count in (("q0", 2), ("q1", 12)):
        queries[name] = {
            "native_keypoints": torch.zeros((row_count, 2)),
            "native_descriptors": torch.ones((row_count, 2)),
            "native_scores": torch.ones(row_count),
            "native_K": intrinsics,
            "pose_w2c": torch.eye(4),
            "native_input_hw": torch.tensor([100, 100]),
        }
    return GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries}
    )


def test_selection_uses_potential_pose_information_and_independent_layers(
    monkeypatch,
) -> None:
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
        "anchor_xyz": torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]),
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
        "query_names": ["q0", "q1"],
        "records": [
            {
                "visible_anchor_ids": torch.tensor([1]),
                "visible_anchor_image_cells": torch.tensor([1]),
                "detectable_pairs": torch.tensor([[1, 1]]),
                "matching_pairs": torch.tensor([[1, 1]]),
                "exact_identity_positive_pairs": torch.tensor([[1, 1]]),
                "projective_compatible_ambiguous_pairs": torch.empty(
                    (0, 2), dtype=torch.long
                ),
                "clean_inlier_pose_anchor_ids": torch.tensor([1]),
                "clean_inlier_pose_information": torch.eye(6).unsqueeze(0),
            },
            {
                "visible_anchor_ids": torch.tensor([0, 1, 2]),
                "visible_anchor_image_cells": torch.tensor([5, 5, 7]),
                "detectable_pairs": torch.tensor([[10, 0], [11, 1]]),
                "matching_pairs": torch.tensor([[10, 0]]),
                "exact_identity_positive_pairs": torch.tensor([[10, 0]]),
                "projective_compatible_ambiguous_pairs": torch.tensor(
                    [[11, 1], [9, 2]]
                ),
                "clean_inlier_pose_anchor_ids": torch.tensor([0]),
                "clean_inlier_pose_information": torch.eye(6).unsqueeze(0),
            },
        ],
    }
    proposal, _ = selection_only_proposal(
        state,
        _selection_provider(),
        feedback,
        maximum_anchors=2,
        visibility_target=2,
        detectability_target=3,
        matching_target=4,
        pose_logdet_target=5.0,
        pose_min_eigenvalue_target=0.25,
        pose_information_chunk_size=1,
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
    assert set(captured["pose_information"][1]) == {0}
    assert set(captured["pose_information"][2]) == {0}
    view = _selection_provider().build_view(1)
    jacobian = pose_jacobian_analytic(
        state["anchor_xyz"][:1].double(),
        view.intrinsics.double(),
        view.pose_w2c.double(),
    )
    expected = fisher_contributions(
        task_scaled_pose_jacobian(
            jacobian,
            translation_scale=0.05,
            rotation_scale=math.radians(5.0),
        )
    )[0]
    assert torch.allclose(captured["pose_information"][0][0], expected)
    report = proposal["v6_selection_distillation"]
    assert report["visibility_evidence_unit"] == "query_image_grid_cell"
    assert report["version"] == 4
    assert report["pose_information_realized_clean_inlier_conditioned"] is False
    assert report["potential_pose_information_edge_count"] == 3
    assert report["pose_information_chunk_size"] == 1
    potential = report["report"]["potential_pose_information"]
    assert potential["candidate_pool"] == (
        "all_unique_feedback_exact_or_ambiguous_geometry_candidate_anchors"
    )
    assert potential["dense_query_anchor_tensor_materialized"] is False
    assert report["training_query_indices"].tolist() == [0, 1]
    assert report["round_training_query_indices"].tolist() == [1]
    assert report["training_query_registry_explicit"] is True
    assert report["selection_round"] == 2


def test_selection_deduplicates_potential_anchor_and_ignores_realized_pose_rows(
    monkeypatch,
) -> None:
    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return {"selected_anchor_rows": torch.tensor([0]), "unmet": {}}

    monkeypatch.setattr(v6_proposals, "select_layered_sufficiency", fake_select)
    monkeypatch.setattr(
        v6_proposals,
        "subset_projective_anchor_map",
        lambda state, selected: {"anchor_ids": state["anchor_ids"][selected]},
    )
    state = {
        "anchor_ids": torch.arange(1),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 5.0]]),
        "anchor_matchability": torch.ones(1),
    }
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "positive_identity_contract": exact_identity_positive_contract(),
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q0", "q1"],
        "records": [
            {
                "visible_anchor_ids": torch.tensor([0]),
                "visible_anchor_image_cells": torch.tensor([0]),
                "detectable_pairs": torch.tensor([[0, 0], [1, 0]]),
                "matching_pairs": torch.tensor([[0, 0]]),
                "exact_identity_positive_pairs": torch.tensor([[0, 0], [1, 0]]),
                "projective_compatible_ambiguous_pairs": torch.empty(
                    (0, 2), dtype=torch.long
                ),
                "clean_inlier_pose_anchor_ids": torch.tensor([0, 0]),
                "clean_inlier_pose_information": torch.stack(
                    [torch.eye(6), torch.eye(6)]
                ),
            },
            {
                "visible_anchor_ids": torch.tensor([0]),
                "visible_anchor_image_cells": torch.tensor([0]),
                "detectable_pairs": torch.tensor([[0, 0]]),
                "matching_pairs": torch.empty((0, 2), dtype=torch.long),
                "exact_identity_positive_pairs": torch.tensor([[0, 0]]),
                "projective_compatible_ambiguous_pairs": torch.empty(
                    (0, 2), dtype=torch.long
                ),
            },
        ],
    }
    proposal, _ = selection_only_proposal(
        state,
        _selection_provider(),
        feedback,
        maximum_anchors=1,
        visibility_target=1,
        detectability_target=1,
        matching_target=1,
        pose_logdet_target=0.0,
        training_query_indices=[0],
    )
    assert set(captured["pose_information"][0]) == {0}
    assert (
        proposal["v6_selection_distillation"]["potential_pose_information_edge_count"]
        == 1
    )
    with pytest.raises(ValueError, match="chunk size must be positive"):
        selection_only_proposal(
            state,
            _selection_provider(),
            feedback,
            maximum_anchors=1,
            visibility_target=1,
            detectability_target=1,
            matching_target=1,
            pose_logdet_target=0.0,
            pose_information_chunk_size=0,
        )
