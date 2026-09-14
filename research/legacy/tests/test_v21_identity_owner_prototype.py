from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from map_learning.v21_identity_owner_prototype import (
    CANDIDATE_SCHEMA,
    CONTROL_GATE_POLICY,
    METADATA_FIELD,
    _block_balanced_prototype,
    build_identity_owner_cached_evaluation,
    control_gate,
    validate_identity_owner_cached_evaluation,
    validate_identity_owner_candidate,
)
from map_learning.v21_pose_feedback_transductive import (
    PROTOTYPE_FEATURE_FIELD,
    PROTOTYPE_OWNER_FIELD,
)
from map_learning.v21_test_cache import tensor_sha256
from tests.test_v21_pose_feedback_transductive import (
    BASELINE_CONTRACT,
    MarkerSolver,
    _one_record_cache,
    _replay,
    _set_cached_baseline,
    _source,
    _stable_map,
)


def _candidate(stable: dict, adaptation: dict) -> dict:
    stable = deepcopy(stable)
    stable["fine_identity_ids"] = torch.tensor([10, 11, 12])
    descriptors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.00, 0.0, 0.0],
                [1.0, 0.10, 0.0, 0.0],
                [1.0, -0.08, 0.0, 0.0],
            ]
        ),
        dim=1,
    )
    evidence = _block_balanced_prototype(
        descriptors=descriptors,
        block_ids=("s0:b0", "s0:b1", "s1:b0"),
        sequence_ids=("s0", "s0", "s1"),
    )
    prototype = evidence["prototype"][None]
    owners = torch.tensor([2])
    action = {
        "fine_identity_id": 12,
        "owner_anchor_row": 2,
        "prototype_index": 0,
        "prototype_sha256": tensor_sha256(prototype[0]),
        "stored_medoid": evidence["medoid"],
        "stored_medoid_sha256": evidence["medoid_sha256"],
        "block_centroids": evidence["block_centroids"],
        "block_centroid_sha256s": evidence["block_centroid_sha256s"],
        "block_ids": evidence["block_ids"],
        "block_sequence_ids": evidence["block_sequence_ids"],
        "source_descriptor_sha256s": evidence["source_descriptor_sha256s"],
        "source_query_indices": (
            int(adaptation["records"][0]["query_index"]),
        ),
        "source_edge_count": 3,
        "adaptation_block_count": 3,
        "adaptation_sequence_count": 2,
        "mapping_track_bank_observation_count": 4,
        "mapping_track_bank_family_count": 2,
        "descriptor_medoid_min_cosine": evidence["medoid_min_cosine"],
        "prototype_to_stored_medoid_cosine": float(
            evidence["prototype"] @ evidence["medoid"]
        ),
        "promotion_wrong_top1_edge_count": 2,
        "preservation_correct_top1_edge_count": 1,
        "prototype_uses_multiple_blocks": True,
        "prototype_is_single_observation_copy": False,
    }
    stable_source = _source("/stable_map.pt", "b" * 64)
    split_source = dict(adaptation["inputs"]["split_manifest"])
    calibration_source = _source("/strict_calibration.pt", "e" * 64)
    producer_source = _source("/producer.py", "1" * 64)
    metadata = {
        "schema": CANDIDATE_SCHEMA,
        "version": 1,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "formation_role": "adaptation",
        "complete_adaptation_registry_consumed": True,
        "adaptation_features_consumed": True,
        "control_features_consumed": False,
        "confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "identity_evidence_authority": "provisional_candidate_only",
        "identity_truth_claimed": False,
        "negative_anchor_labels_created": False,
        "base_anchor_fields_bit_exact": True,
        "base_anchor_descriptors_retained": True,
        "base_anchor_features_moved_or_lowered": False,
        "geometry_changed": False,
        "matching_semantics": "global_owner_prototype_top1",
        "prototype_feature_source": (
            "normalized_equal_block_mean_after_exact_calibration_medoid_recompute"
        ),
        "prototype_activation_threshold": None,
        "deployment_authorized": False,
        "controller_authorized": False,
        "heldout_control_required": True,
        "heldout_confirmation_required": True,
        "confirmation_evaluation_authorized": False,
        "inputs": {
            "stable_map": stable_source,
            "identity_calibration": calibration_source,
            "adaptation_caches": [_source("/adaptation.pt", "c" * 64)],
            "split_manifest": split_source,
            "producer_sources": [producer_source],
        },
        "frontend_shard_registry_sha256": adaptation["shard_registry"][
            "registry_sha256"
        ],
        "preprocessing_config_sha256": adaptation["preprocessing_config_sha256"],
        "baseline_contract": dict(BASELINE_CONTRACT),
        "calibration_thresholds": {
            "minimum_adaptation_blocks": 3,
            "minimum_adaptation_sequences": 2,
            "minimum_mapping_observations": 2,
            "minimum_mapping_families": 2,
            "minimum_descriptor_medoid_cosine": 0.8,
        },
        "maximum_total_prototypes": 128,
        "maximum_prototypes_per_fine_identity": 1,
        "accepted_identity_count": 1,
        "added_prototype_count": 1,
        "source_query_count": 1,
        "source_query_indices": torch.tensor(
            [int(adaptation["records"][0]["query_index"])]
        ),
        "prototype_features_sha256": tensor_sha256(prototype),
        "prototype_owner_rows_sha256": tensor_sha256(owners),
        "selected_actions": [action],
    }
    candidate = deepcopy(stable)
    candidate[PROTOTYPE_FEATURE_FIELD] = prototype
    candidate[PROTOTYPE_OWNER_FIELD] = owners
    candidate[METADATA_FIELD] = metadata
    validate_identity_owner_candidate(candidate, stable_map=stable)
    return candidate


def test_block_balanced_prototype_is_derived_from_all_blocks() -> None:
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.98, 0.1], [0.99, -0.1], [0.97, 0.03]]
    )
    result = _block_balanced_prototype(
        descriptors=descriptors,
        block_ids=("s0:b0", "s0:b0", "s0:b1", "s1:b0"),
        sequence_ids=("s0", "s0", "s0", "s1"),
    )
    expected = F.normalize(result["block_centroids"].mean(0), dim=0)
    assert torch.allclose(result["prototype"], expected)
    assert result["block_centroids"].shape[0] == 3
    assert len(set(result["block_sequence_ids"])) == 2
    assert result["prototype_sha256"] not in result["source_descriptor_sha256s"]


def test_candidate_keeps_base_fields_and_rejects_single_block_tamper(
    tmp_path: Path,
) -> None:
    stable = _stable_map()
    stable["fine_identity_ids"] = torch.tensor([10, 11, 12])
    adaptation = _one_record_cache(tmp_path, role="adaptation", marker=1.0)
    candidate = _candidate(stable, adaptation)
    for key in stable:
        if isinstance(stable[key], torch.Tensor):
            assert torch.equal(candidate[key], stable[key])
    assert candidate[PROTOTYPE_OWNER_FIELD].tolist() == [2]
    assert candidate[METADATA_FIELD]["deployment_authorized"] is False
    assert candidate[METADATA_FIELD]["identity_truth_claimed"] is False

    unsafe = deepcopy(candidate)
    unsafe[METADATA_FIELD]["selected_actions"][0]["block_centroids"] = unsafe[
        METADATA_FIELD
    ]["selected_actions"][0]["block_centroids"][:1]
    with pytest.raises(ValueError, match="action row is invalid"):
        validate_identity_owner_candidate(unsafe, stable_map=stable)


def test_exact_cached_control_failure_blocks_confirmation(tmp_path: Path) -> None:
    stable = _stable_map()
    stable["fine_identity_ids"] = torch.tensor([10, 11, 12])
    adaptation = _one_record_cache(tmp_path, role="adaptation", marker=1.0)
    control = _one_record_cache(tmp_path, role="control", marker=3.0)
    solver = MarkerSolver(
        {
            2: adaptation["records"][0]["pose_w2c"].numpy(),
            4: control["records"][0]["pose_w2c"].numpy(),
        }
    )
    _set_cached_baseline(
        adaptation,
        _replay(
            adaptation,
            stable,
            solver,
            adaptation["records"][0]["winner_anchor_rows"],
        ),
    )
    _set_cached_baseline(
        control,
        _replay(control, stable, solver, control["records"][0]["winner_anchor_rows"]),
    )
    candidate = _candidate(stable, adaptation)
    common = {
        "stable_map": stable,
        "candidate": candidate,
        "stable_map_source": _source("/stable_map.pt", "b" * 64),
        "candidate_source": _source("/candidate.pt", "f" * 64),
        "producer_sources": [_source("/eval.py", "2" * 64)],
        "device": "cpu",
        "matcher_chunk_size": 2,
        "solver": solver,
    }
    adaptation_result = build_identity_owner_cached_evaluation(
        cache_payloads=[adaptation],
        cache_sources=[_source("/adaptation.pt", "c" * 64)],
        **common,
    )
    assert adaptation_result["summary"]["paired_r5_gain_count"] == 1
    assert adaptation_result["control_gate"]["evaluated"] is False
    control_result = build_identity_owner_cached_evaluation(
        cache_payloads=[control],
        cache_sources=[_source("/control.pt", "9" * 64)],
        **common,
    )
    validate_identity_owner_cached_evaluation(control_result)
    assert control_result["summary"]["paired_r5_loss_count"] == 1
    assert control_result["control_gate"]["passed"] is False
    assert control_result["confirmation_evaluation_authorized"] is False


def test_control_gate_requires_gain_no_loss_and_nonpositive_median() -> None:
    summary = {
        "paired_r5_gain_count": 1,
        "paired_r5_loss_count": 0,
        "paired_delta_task_error": {"median": -0.01},
    }
    result = control_gate(summary, role="control")
    assert result["passed"] is True
    assert result["policy"] == CONTROL_GATE_POLICY
    summary["paired_r5_loss_count"] = 1
    assert control_gate(summary, role="control")["passed"] is False
