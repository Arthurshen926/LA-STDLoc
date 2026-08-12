from argparse import Namespace

import pytest
import torch

from map_learning.frontend_upper_bound import (
    DESCRIPTOR_REPRESENTATION_EQUAL_ENERGY,
    PROBE_SCHEMA,
    _build_descriptor_banks,
    audit_descriptor_equal_energy_crossfit,
    audit_descriptor_identity_crossfit,
    audit_detector_repeatability,
    file_sha256,
    probe_contract,
    tensor_sha256,
    validate_probe,
)
import scripts.audit_frontend_upper_bound as frontend_runner
from scripts.audit_frontend_upper_bound import preflight


def _synthetic_inputs(tmp_path):
    weight = tmp_path / "candidate.pth"
    weight.write_bytes(b"locked synthetic candidate")
    names = ["seq/frame000.png", "seq/frame001.png"]
    native_keypoints = torch.tensor([[0.5, 0.5], [2.5, 2.5]])
    raw_descriptors = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    queries = {}
    records = []
    for query_index, (name, descriptors) in enumerate(zip(names, raw_descriptors)):
        queries[name] = {
            "native_keypoints": native_keypoints.clone(),
            "native_descriptors": descriptors.clone(),
            "native_scores": torch.tensor([0.9, 0.8]),
            "native_valid_mask": torch.ones((8, 8), dtype=torch.bool),
            "native_depth": torch.ones((8, 8)),
            "native_alpha": torch.ones((8, 8)),
            "native_K": torch.eye(3),
            "native_input_hw": [8, 8],
            "pose_w2c": torch.eye(4),
            "pixel_center_offset": 0.5,
            "native_sparse_metadata": {
                "detect_num": 2,
                "nms_radius": 4,
            },
        }
        records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        )
    query_cache = {"queries": queries}
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": 2,
        "query_names": names,
        "records": records,
        "config": {
            "depth_abs_tolerance_m": 0.05,
            "depth_rel_tolerance": 0.02,
            "alpha_minimum": 0.01,
        },
    }
    state = {
        "anchor_xyz": torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 1.0]]),
        "anchor_type": torch.tensor([1, 0]),
    }
    candidate_descriptors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    probe = {
        "schema": PROBE_SCHEMA,
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference": {
            "query_cache_signature": None,
            "teacher_schema": teacher["schema"],
        },
        "frontend": {
            "name": "synthetic-independent",
            "family": "independent_local_frontend",
            "implementation_id": "synthetic-test-v1",
            "coordinate_convention": (
                "reference_grid_index_then_cached_pixel_center_offset"
            ),
            "descriptor_dim": 3,
            "requested_keypoint_count": 2,
            "weights": {
                "path": str(weight),
                "sha256": file_sha256(weight),
            },
        },
        "capabilities": {
            "detector_repeatability": True,
            "descriptor_identity": True,
        },
        "queries": {
            name: {
                "reference_keypoints_sha256": tensor_sha256(
                    queries[name]["native_keypoints"]
                ),
                "detector_keypoints": torch.tensor([[0.5, 0.5], [4.5, 4.5]]),
                "detector_scores": torch.tensor([0.95, 0.85]),
                "detected_count_before_mask": 2,
                "descriptor_at_reference_keypoints": (candidate_descriptors.clone()),
            }
            for name in names
        },
    }
    return state, query_cache, teacher, probe


def test_probe_contract_and_validation_fail_closed(tmp_path):
    _, query_cache, teacher, probe = _synthetic_inputs(tmp_path)
    contract = probe_contract(query_cache, teacher)
    assert contract["mapping_only"] is True
    assert contract["requested_keypoint_count"] == 2
    assert contract["required_capabilities"]["detector_repeatability"][
        "same_requested_k"
    ]
    assert contract["required_capabilities"]["descriptor_identity"][
        "dimension_may_differ"
    ]
    result = validate_probe(
        probe,
        query_cache,
        teacher,
        require_detector=True,
        require_descriptor=True,
    )
    assert result["validated_descriptor_rows"] == 4
    assert result["reference_descriptor_dim"] == 2
    assert result["candidate_descriptor_dim"] == 3
    assert result["validated_detector_keypoints"] == 4

    invalid_hash = dict(probe)
    invalid_hash["queries"] = {
        name: dict(payload) for name, payload in probe["queries"].items()
    }
    invalid_hash["queries"][teacher["query_names"][0]]["reference_keypoints_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="registry mismatch"):
        validate_probe(
            invalid_hash,
            query_cache,
            teacher,
            require_descriptor=True,
        )

    pair_matcher = dict(probe)
    pair_matcher["frontend"] = {**probe["frontend"], "family": "pair_matcher"}
    with pytest.raises(ValueError, match="pair matchers"):
        validate_probe(
            pair_matcher,
            query_cache,
            teacher,
            require_descriptor=True,
        )


def test_detector_repeatability_is_same_k_and_descriptor_free(tmp_path):
    state, query_cache, teacher, probe = _synthetic_inputs(tmp_path)
    report = audit_detector_repeatability(
        state=state,
        query_cache=query_cache,
        teacher=teacher,
        probe=probe,
        radii_px=[0.2],
    )
    baseline = report["frozen_superpoint"]["by_anchor_kind"]["all"]
    candidate = report["candidate"]["by_anchor_kind"]["all"]
    assert baseline["target_count"] == 4
    assert baseline["reachable_fraction"]["0.2"] == pytest.approx(0.5)
    assert candidate["reachable_fraction"]["0.2"] == pytest.approx(1.0)
    assert report["delta_candidate_minus_superpoint"]["all"]["0.2"] == pytest.approx(
        0.5
    )


def test_descriptor_identity_is_paired_and_bidirectional(tmp_path):
    state, query_cache, teacher, probe = _synthetic_inputs(tmp_path)
    report = audit_descriptor_identity_crossfit(
        state=state,
        query_cache=query_cache,
        teacher=teacher,
        probe=probe,
        crossfit_blocks=2,
        minimum_support_views=1,
        topks=[1, 2],
    )
    assert len(report["directions"]) == 2
    assert report["pooled"]["frozen_superpoint"]["positive_recall_at_k"][
        "1"
    ] == pytest.approx(0.0)
    assert report["pooled"]["candidate"]["positive_recall_at_k"]["1"] == pytest.approx(
        1.0
    )
    assert report["delta_candidate_minus_superpoint"]["1"] == pytest.approx(1.0)
    assert report["protocol"]["candidate_detector_used"] is False
    for direction in report["directions"]:
        memory = direction["support"]["map_descriptor_memory_float32"]
        assert memory["frozen_superpoint_dim"] == 2
        assert memory["candidate_dim"] == 3
        assert memory["candidate_to_superpoint_ratio"] == pytest.approx(1.5)
        resources = direction["ranking_resources"]
        assert resources["frozen_superpoint"]["descriptor_dim"] == 2
        assert resources["candidate"]["descriptor_dim"] == 3
        assert resources["candidate_to_superpoint_ratio"][
            "dot_product_multiply_accumulates"
        ] == pytest.approx(1.5)


def test_equal_energy_descriptor_is_one_global_cosine_with_fixed_half_scores(
    tmp_path,
):
    state, query_cache, teacher, probe = _synthetic_inputs(tmp_path)
    banks, support = _build_descriptor_banks(
        query_cache=query_cache,
        teacher=teacher,
        probe=probe,
        support_query_indices=[0],
        minimum_support_views=1,
        candidate_representation=DESCRIPTOR_REPRESENTATION_EQUAL_ENERGY,
    )
    gate_name = teacher["query_names"][1]
    raw_query = torch.nn.functional.normalize(
        query_cache["queries"][gate_name]["native_descriptors"], dim=1
    )
    candidate_query = torch.nn.functional.normalize(
        probe["queries"][gate_name]["descriptor_at_reference_keypoints"], dim=1
    )
    composite_query = torch.cat((raw_query, candidate_query), dim=1) / (2**0.5)
    composite_scores = composite_query @ banks["candidate"].T

    support_name = teacher["query_names"][0]
    raw_bank = torch.nn.functional.normalize(
        query_cache["queries"][support_name]["native_descriptors"], dim=1
    )
    candidate_bank = torch.nn.functional.normalize(
        probe["queries"][support_name]["descriptor_at_reference_keypoints"],
        dim=1,
    )
    expected_scores = 0.5 * (raw_query @ raw_bank.T) + 0.5 * (
        candidate_query @ candidate_bank.T
    )
    assert torch.allclose(composite_scores, expected_scores, atol=1e-7, rtol=0)
    assert support["map_descriptor_memory_float32"]["candidate_dim"] == 5

    report = audit_descriptor_equal_energy_crossfit(
        state=state,
        query_cache=query_cache,
        teacher=teacher,
        probe=probe,
        crossfit_blocks=2,
        minimum_support_views=1,
        topks=[1, 2],
    )
    assert report["schema"] == ("lafgs_mapping_descriptor_equal_energy_ceiling_probe")
    assert report["protocol"]["effective_candidate_descriptor_dim"] == 5
    assert report["protocol"]["learned_fusion_parameters"] is False
    assert report["protocol"]["source_specific_descriptor_routing"] is False


def test_equal_energy_bank_preserves_view_balance_with_duplicate_anchor_edges(
    tmp_path,
):
    _, query_cache, teacher, probe = _synthetic_inputs(tmp_path)
    teacher["records"][0]["positive_offsets"] = torch.tensor([0, 1, 2])
    teacher["records"][0]["positive_indices"] = torch.tensor([0, 0])
    teacher["records"][1]["positive_offsets"] = torch.tensor([0, 1, 1])
    teacher["records"][1]["positive_indices"] = torch.tensor([0])
    banks, report = _build_descriptor_banks(
        query_cache=query_cache,
        teacher=teacher,
        probe=probe,
        support_query_indices=[0, 1],
        minimum_support_views=2,
        candidate_representation=DESCRIPTOR_REPRESENTATION_EQUAL_ENERGY,
    )
    first_superpoint_view = torch.nn.functional.normalize(
        torch.tensor([1.0, 1.0]), dim=0
    )
    second_superpoint_view = torch.tensor([0.0, 1.0])
    expected_superpoint = torch.nn.functional.normalize(
        first_superpoint_view + second_superpoint_view, dim=0
    )
    first_xfeat_view = torch.nn.functional.normalize(
        torch.tensor([1.0, 1.0, 0.0]), dim=0
    )
    second_xfeat_view = torch.tensor([1.0, 0.0, 0.0])
    expected_xfeat = torch.nn.functional.normalize(
        first_xfeat_view + second_xfeat_view, dim=0
    )
    expected = torch.cat((expected_superpoint, expected_xfeat)) / (2**0.5)
    assert banks["anchor_indices"].tolist() == [0]
    assert torch.allclose(banks["candidate"][0], expected, atol=1e-7, rtol=0)
    assert report["positive_edge_count"] == 3
    assert report["supported_anchor_count"] == 1


def test_preflight_blocks_when_only_code_or_pair_matcher_exists(tmp_path):
    superpoint = tmp_path / "superpoint.pth"
    superpoint.write_bytes(b"not the official checkpoint")
    loftr = tmp_path / "loftr.ckpt"
    loftr.write_bytes(b"pair matcher")
    args = Namespace(
        superpoint_weights=str(superpoint),
        featurebooster_weights=str(tmp_path / "missing-boost.pth"),
        loftr_weights=str(loftr),
        kornia_python=str(tmp_path / "missing-python"),
        candidate_name="unprovisioned",
        candidate_family="independent_local_frontend",
        candidate_code_id="",
        candidate_weights=None,
        candidate_weights_sha256=None,
        candidate_descriptor_dim=256,
    )
    report = preflight(args)
    assert (
        report["ceiling_probe_arms"]["A_detector_repeatability"]["status"]
        == "BLOCKED_BY_ARTIFACT"
    )
    assert (
        report["ceiling_probe_arms"]["B_descriptor_identity"]["status"]
        == "BLOCKED_BY_ARTIFACT"
    )
    assert (
        report["available_but_not_admissible_as_stronger_frontend"]["loftr"][
            "classification"
        ]
        == "pair_matcher"
    )


def test_preflight_allows_a_locked_non_256d_descriptor_candidate(tmp_path, monkeypatch):
    superpoint = tmp_path / "superpoint.pth"
    superpoint.write_bytes(b"synthetic frozen baseline")
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"synthetic locked 64D frontend")
    monkeypatch.setattr(frontend_runner, "SUPERPOINT_SHA256", file_sha256(superpoint))
    args = Namespace(
        superpoint_weights=str(superpoint),
        featurebooster_weights=str(tmp_path / "missing-boost.pth"),
        loftr_weights=str(tmp_path / "missing-loftr.ckpt"),
        kornia_python=str(tmp_path / "missing-python"),
        candidate_name="synthetic-64d",
        candidate_family="independent_local_frontend",
        candidate_code_id="locked-implementation-commit",
        candidate_weights=str(candidate),
        candidate_weights_sha256=file_sha256(candidate),
        candidate_descriptor_dim=64,
    )
    report = preflight(args)
    assert report["candidate"]["descriptor_arm_eligible"] is True
    assert (
        report["ceiling_probe_arms"]["B_descriptor_identity"]["status"]
        == "READY_FOR_PROBE_MATERIALIZATION"
    )
