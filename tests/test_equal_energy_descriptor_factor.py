from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.localizer import load_shared_metric
from map_learning import equal_energy_descriptor_factor as descriptor_factor
from map_learning.equal_energy_descriptor_factor import (
    audit_descriptor_factor_pair,
    materialize_equal_energy_descriptor_factor,
    validate_descriptor_factor_contract,
)
from map_learning.frontend_upper_bound import tensor_sha256
from map_learning.metric import SharedLowRankMetric
from scripts import compare_mapping_pose_gate as pose_gate
from scripts import evaluate_mapping_cache


EVALUATION_CODE = {
    "schema": "lafgs_mapping_pose_evaluation_code",
    "version": 1,
    "repository": "/clean/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {
        "map_learning/equal_energy_descriptor_factor.py": "a" * 64,
        "scripts/materialize_equal_energy_descriptor_factor.py": "b" * 64,
        "scripts/evaluate_mapping_cache.py": "c" * 64,
        "scripts/compare_mapping_pose_gate.py": "d" * 64,
    },
}
PRODUCER_IDENTITY = {
    "schema": "lafgs_equal_energy_descriptor_factor_producer_code",
    "version": 1,
    "repository": "/clean/repository",
    "git_commit": EVALUATION_CODE["git_commit"],
    "git_worktree_clean": True,
    "entrypoints": EVALUATION_CODE["entrypoints"],
}
REAL_PRODUCER_IDENTITY = descriptor_factor.descriptor_factor_producer_identity


@pytest.fixture(autouse=True)
def _clean_synthetic_producer_identity(monkeypatch):
    monkeypatch.setattr(
        descriptor_factor,
        "descriptor_factor_producer_identity",
        lambda **_: json.loads(json.dumps(PRODUCER_IDENTITY)),
    )


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle(tmp_path: Path, *, query_count: int = 4, anchor_count: int = 2) -> dict:
    tmp_path.mkdir(parents=True)
    paths = {
        "source_map": tmp_path / "source_map.pt",
        "source_metric": tmp_path / "source_metric.pt",
        "teacher_anchor_map": tmp_path / "teacher_anchor_map.pt",
        "source_query_cache": tmp_path / "source_query_cache.pt",
        "refreshed_query_cache": tmp_path / "refreshed_query_cache.pt",
        "teacher": tmp_path / "teacher.pt",
        "calibration": tmp_path / "calibration.json",
        "probe": tmp_path / "probe.pt",
        "mechanism_report": tmp_path / "mechanism_report.json",
        "mechanism_gate": tmp_path / "mechanism_gate.json",
        "deployment_extension": tmp_path / "deployment_extension.json",
        "xfeat_weights": tmp_path / "xfeat.pt",
    }
    paths["xfeat_weights"].write_bytes(b"locked synthetic XFeat weights")
    names = [f"mapping-{index:04d}.png" for index in range(query_count)]
    source_queries = {}
    refreshed_queries = {}
    probe_queries = {}
    teacher_records = []
    for index, name in enumerate(names):
        superpoint = torch.zeros((1, 256), dtype=torch.float16)
        superpoint[0, index % 256] = 1
        keypoints = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        record = {
            "native_keypoints": keypoints,
            "native_descriptors": superpoint,
            "native_scores": torch.tensor([1.0], dtype=torch.float16),
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
            "native_depth": torch.ones((3, 3)),
            "native_alpha": torch.ones((3, 3), dtype=torch.float16),
            "native_valid_mask": torch.ones((3, 3), dtype=torch.bool),
            "native_input_hw": [3, 3],
            "pixel_center_offset": 0.5,
            "native_sparse_metadata": {
                "detect_num": 1,
                "requested_keypoint_count": 1,
                "nms_radius": 4,
            },
            "immutable_payload": {"query_index": index},
        }
        source_queries[name] = record
        refreshed_queries[name] = dict(record)
        xfeat = torch.zeros((1, 64), dtype=torch.float32)
        xfeat[0, (index * 3) % 64] = 1
        probe_queries[name] = {
            "reference_keypoints_sha256": tensor_sha256(keypoints),
            "descriptor_at_reference_keypoints": xfeat,
        }
        anchor = 1 if index == 1 and anchor_count > 1 else 0
        teacher_records.append(
            {
                "query_index": index,
                "query_name": name,
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([anchor]),
                "ambiguous_offsets": torch.tensor([0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        )
    source_cache = {
        "version": 3,
        "signature": "source-signature",
        "signature_payload": {
            "version": 10,
            "descriptor_source": "superpoint",
            "native_sparse_keypoint_count": 1,
            "native_sparse_nms_radius": 4,
        },
        "queries": source_queries,
        "immutable_top_level": {"scene": "synthetic"},
    }
    refreshed_signature_payload = {
        "version": 11,
        "descriptor_source": "superpoint",
        "native_sparse_keypoint_count": 1,
        "native_sparse_nms_radius": 4,
    }
    refreshed_cache = {
        "version": 3,
        "signature": _json_sha256(refreshed_signature_payload),
        "signature_payload": refreshed_signature_payload,
        "queries": refreshed_queries,
    }
    torch.save(source_cache, paths["source_query_cache"])
    torch.save(refreshed_cache, paths["refreshed_query_cache"])

    generator = torch.Generator().manual_seed(7)
    anchor_ids = torch.arange(anchor_count, dtype=torch.long) * 2 + 11
    anchor_features = F.normalize(
        torch.randn((anchor_count, 256), generator=generator), dim=1
    )
    source_map = {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": anchor_ids,
        "source_primitive_ids": torch.arange(anchor_count),
        "track_cluster_ids": torch.arange(anchor_count) + 30,
        "anchor_xyz": torch.randn((anchor_count, 3), generator=generator),
        "anchor_type": torch.tensor(
            [1 if index % 2 == 0 else 0 for index in range(anchor_count)]
        ),
        "dependency_group_ids": torch.arange(anchor_count) + 40,
        "coarse_dependency_group_ids": torch.arange(anchor_count) + 50,
        "fine_identity_ids": torch.arange(anchor_count) + 60,
        "source_dependency_group_ids": torch.arange(anchor_count) + 70,
        "anchor_features": anchor_features,
        "v7_metric_raw_features": anchor_features.clone(),
        "v7_anchor_residual_parameter": torch.ones((anchor_count, 256)),
        "v7_anchor_residual": torch.ones((anchor_count, 256)),
        "v7_online_metric": {"schema": "source"},
        "immutable_payload": {"geometry": "frozen"},
    }
    torch.save(source_map, paths["source_map"])
    teacher_anchor_map = {
        key: value
        for key, value in source_map.items()
        if key
        not in {
            "v7_metric_raw_features",
            "v7_anchor_residual_parameter",
            "v7_anchor_residual",
            "v7_online_metric",
        }
    }
    teacher_anchor_map["anchor_features"] = F.normalize(
        torch.flip(anchor_features, dims=(1,)), dim=1
    )
    teacher_anchor_map["teacher_topology_only"] = True
    torch.save(teacher_anchor_map, paths["teacher_anchor_map"])
    metric = SharedLowRankMetric(descriptor_dim=256, rank=2, max_residual_norm=0.1)
    with torch.no_grad():
        metric.down.weight.fill_(0.01)
        metric.down.bias.copy_(torch.tensor([0.1, -0.2]))
        metric.up.weight.fill_(0.002)
    torch.save(
        {
            "schema": "lafgs_shared_metric_state",
            "version": 1,
            "landmark_indices": anchor_ids.clone(),
            "metric_config": metric.export_config(),
            "metric_state_dict": metric.state_dict(),
            "map_path": str(paths["source_map"].resolve()),
            "step": 1520,
        },
        paths["source_metric"],
    )
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "uses_test_queries": False,
        "anchor_count": anchor_count,
        "query_names": names,
        "records": teacher_records,
        "anchor_map": str(paths["teacher_anchor_map"].resolve()),
        "query_cache": str(paths["source_query_cache"].resolve()),
        "config": {"radius": 2.0},
        "diagnostics": {"positive_rows": query_count},
    }
    torch.save(teacher, paths["teacher"])
    calibration = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "uses_test_queries": False,
        "sources": {
            "query_cache": str(paths["source_query_cache"].resolve()),
            "query_cache_sha256": sha256_file(paths["source_query_cache"]),
            "track_payload": "/frozen/track_payload.pt",
            "uses_test_queries": False,
        },
        "statistics": {"mapping_views": query_count},
        "parameters": {
            "ransac_reprojection_px": 12.0,
            "clean_radius_px": 1.0,
            "task_translation_m": 0.05,
            "task_rotation_deg": 5.0,
        },
        "policy": {"calibration_split": "mapping_only"},
    }
    paths["calibration"].write_text(json.dumps(calibration))

    probe = {
        "schema": "lafgs_frontend_ceiling_probe_cache",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference": {
            "query_cache_path": str(paths["refreshed_query_cache"].resolve()),
            "query_cache_sha256": sha256_file(paths["refreshed_query_cache"]),
            "teacher_path": str(paths["teacher"].resolve()),
            "teacher_sha256": sha256_file(paths["teacher"]),
            "query_cache_signature": refreshed_cache["signature"],
            "teacher_schema": teacher["schema"],
        },
        "frontend": {
            "family": "independent_local_frontend",
            "implementation_id": "synthetic-xfeat64-v1",
            "coordinate_convention": (
                "reference_grid_index_then_cached_pixel_center_offset"
            ),
            "requested_keypoint_count": 1,
            "descriptor_dim": 64,
            "weights": {
                "path": str(paths["xfeat_weights"].resolve()),
                "sha256": sha256_file(paths["xfeat_weights"]),
            },
        },
        "capabilities": {"descriptor_identity": True},
        "queries": probe_queries,
    }
    torch.save(probe, paths["probe"])
    mechanism_evaluation_code = {
        "schema": "lafgs_frontend_descriptor_evaluation_code",
        "version": 1,
        "repository": "/clean/repository",
        "git_commit": "9" * 40,
        "git_worktree_clean": True,
        "entrypoints": {"mechanism.py": "8" * 64},
    }
    mechanism_sources = {
        "state": {
            "path": str(paths["source_map"].resolve()),
            "sha256": sha256_file(paths["source_map"]),
        },
        "query_cache": {
            "path": str(paths["refreshed_query_cache"].resolve()),
            "sha256": sha256_file(paths["refreshed_query_cache"]),
        },
        "teacher": {
            "path": str(paths["teacher"].resolve()),
            "sha256": sha256_file(paths["teacher"]),
        },
        "probe_cache": {
            "path": str(paths["probe"].resolve()),
            "sha256": sha256_file(paths["probe"]),
        },
    }
    mechanism_report = {
        "schema": "lafgs_frontend_ceiling_probe_audit_bundle",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "deployment_modified": False,
        "probe_cache": str(paths["probe"].resolve()),
        "source_artifacts": mechanism_sources,
        "evaluation_code": mechanism_evaluation_code,
        "descriptor_identity": {
            "schema": "lafgs_mapping_descriptor_equal_energy_ceiling_probe",
            "mapping_only": True,
            "uses_test_queries": False,
            "protocol": {
                "candidate_representation": (
                    "l2_concat(l2(superpoint),l2(candidate))/sqrt(2)"
                ),
                "score_identity": "0.5*cosine_superpoint+0.5*cosine_candidate",
                "ranking": "single_global_cosine",
                "map_bank": "same_positive_edges_view_balanced_support_only",
                "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
                "learned_fusion_parameters": False,
                "source_specific_descriptor_routing": False,
                "candidate_detector_used": False,
                "source_candidate_descriptor_dim": 64,
                "effective_candidate_descriptor_dim": 320,
                "minimum_support_views": 2,
            },
            "attestation": {
                "artifact": {
                    "path": str(paths["xfeat_weights"].resolve()),
                    "sha256": sha256_file(paths["xfeat_weights"]),
                },
                "reference_artifacts": {
                    "query_cache": mechanism_sources["query_cache"],
                    "teacher": mechanism_sources["teacher"],
                },
                "reference_descriptor_dim": 256,
                "candidate_descriptor_dim": 64,
                "query_count": query_count,
                "validated_descriptor_rows": query_count,
            },
        },
    }
    paths["mechanism_report"].write_text(json.dumps(mechanism_report))
    mechanism_gate = {
        "schema": "lafgs_frontend_descriptor_equal_energy_mechanism_gate",
        "version": 1,
        "valid": True,
        "mapping_only": True,
        "uses_test_queries": False,
        "mechanism_gate_passed": True,
        "advance_to_mapping_only_descriptor_rebuild": True,
        "decision": "GO",
        "single_factor": "equal_energy_single_descriptor_at_exact_superpoint_rows",
        "gates": {
            "selection_to_gate_candidate_r1_strictly_positive": True,
            "gate_to_selection_candidate_r1_strictly_positive": True,
            "pooled_r8_non_regression": True,
            "pooled_track_core_r1_non_regression": True,
            "pooled_gaussian_reserve_r1_non_regression": True,
        },
        "protocol": {
            "candidate_representation": "equal_energy_superpoint_candidate",
            "source_candidate_descriptor_dim": 64,
            "effective_candidate_descriptor_dim": 320,
            "crossfit": "bidirectional_temporal_block",
            "minimum_support_views": 2,
        },
        "inputs": {
            "descriptor_report": {
                "path": str(paths["mechanism_report"].resolve()),
                "sha256": sha256_file(paths["mechanism_report"]),
            },
            "candidate_weights": {
                "path": str(paths["xfeat_weights"].resolve()),
                "sha256": sha256_file(paths["xfeat_weights"]),
            },
            "source_artifacts": mechanism_sources,
            "evaluation_code": mechanism_evaluation_code,
        },
    }
    paths["mechanism_gate"].write_text(json.dumps(mechanism_gate))
    support_counts = torch.zeros(anchor_count, dtype=torch.long)
    for record in teacher_records:
        support_counts[torch.as_tensor(record["positive_indices"]).unique()] += 1
    single_view = torch.nonzero(support_counts == 1, as_tuple=False).reshape(-1)
    single_types = source_map["anchor_type"][single_view]
    single_type_histogram = {
        str(int(value)): int((single_types == value).sum())
        for value in torch.unique(single_types, sorted=True)
    }
    deployment_extension = {
        "schema": descriptor_factor.DEPLOYMENT_EXTENSION_SCHEMA,
        "version": 1,
        "valid": True,
        "mapping_only": True,
        "uses_test_queries": False,
        "single_factor": "fixed_equal_energy_descriptor_representation",
        "source_artifacts": {
            "source_map": mechanism_sources["state"],
            "source_metric": {
                "path": str(paths["source_metric"].resolve()),
                "sha256": sha256_file(paths["source_metric"]),
            },
            "teacher_anchor_map": {
                "path": str(paths["teacher_anchor_map"].resolve()),
                "sha256": sha256_file(paths["teacher_anchor_map"]),
            },
            "teacher": mechanism_sources["teacher"],
            "mechanism_report": {
                "path": str(paths["mechanism_report"].resolve()),
                "sha256": sha256_file(paths["mechanism_report"]),
            },
            "mechanism_gate": {
                "path": str(paths["mechanism_gate"].resolve()),
                "sha256": sha256_file(paths["mechanism_gate"]),
            },
        },
        "estimator_extension": {
            "mechanism_support_domain_min": 2,
            "deployment_estimator_domain_min": 1,
            "single_view_extension_preregistered": True,
            "estimator": descriptor_factor.DEPLOYMENT_ESTIMATOR,
            "all_frozen_anchors_retained": True,
            "no_unsupported_anchor_fallback": True,
            "no_source_or_anchor_type_routing": True,
            "no_anchor_removal": True,
            "anchor_ids_geometry_topology_frozen": True,
        },
        "proxy_to_deployment_transfer": {
            "preregistered_before_pose": True,
            "mechanism_superpoint_branch": descriptor_factor.MECHANISM_SP_BRANCH,
            "deployment_score": descriptor_factor.DEPLOYMENT_SCORE,
            "mechanism_and_deployment_superpoint_banks_bitwise_identical": False,
            "reason": "preserve_the_frozen_v3_baseline_branch",
            "requires_pose_tail_and_cross_domain_adjudication": True,
        },
        "expected_support": {
            "anchor_count": anchor_count,
            "single_view_anchor_count": int(single_view.numel()),
            "single_view_anchor_indices_sha256": tensor_sha256(single_view),
            "single_view_anchor_ids_sha256": tensor_sha256(anchor_ids[single_view]),
            "single_view_anchor_type_histogram": single_type_histogram,
        },
        "adjudication": descriptor_factor.DEPLOYMENT_ADJUDICATION,
    }
    paths["deployment_extension"].write_text(json.dumps(deployment_extension))
    kwargs = {}
    for name, path in paths.items():
        kwargs[f"{name}_path"] = path
        kwargs[f"{name}_sha256"] = sha256_file(path)
    return {
        "paths": paths,
        "kwargs": kwargs,
        "source_map": source_map,
        "source_cache": source_cache,
        "teacher": teacher,
        "calibration": calibration,
        "probe": probe,
        "names": names,
    }


def _materialize(bundle: dict, output: Path) -> dict:
    return materialize_equal_energy_descriptor_factor(
        **bundle["kwargs"], output_dir=output
    )


def _pair_audit(bundle: dict, result: dict) -> dict:
    outputs = {
        name: Path(value["path"]) for name, value in result["contract"]["outputs"].items()
    }
    return audit_descriptor_factor_pair(
        result["contract_path"],
        source_map_path=bundle["paths"]["source_map"],
        source_metric_path=bundle["paths"]["source_metric"],
        source_query_cache_path=bundle["paths"]["source_query_cache"],
        teacher_path=bundle["paths"]["teacher"],
        calibration_path=bundle["paths"]["calibration"],
        variant_map_path=outputs["map"],
        variant_metric_path=outputs["metric"],
        variant_query_cache_path=outputs["descriptor_cache"],
        variant_teacher_path=outputs["teacher"],
        variant_calibration_path=outputs["calibration"],
    )


def test_materializer_preserves_every_non_descriptor_factor_and_exact_score(tmp_path):
    bundle = _bundle(tmp_path / "inputs")
    result = _materialize(bundle, tmp_path / "factor")
    contract_path = Path(result["contract_path"])
    contract = result["contract"]
    outputs = {name: Path(value["path"]) for name, value in contract["outputs"].items()}
    validated = validate_descriptor_factor_contract(
        contract_path,
        source_map_path=bundle["paths"]["source_map"],
        source_metric_path=bundle["paths"]["source_metric"],
        source_query_cache_path=bundle["paths"]["source_query_cache"],
        teacher_path=bundle["paths"]["teacher"],
        calibration_path=bundle["paths"]["calibration"],
        variant_map_path=outputs["map"],
        variant_metric_path=outputs["metric"],
        variant_query_cache_path=outputs["descriptor_cache"],
        variant_teacher_path=outputs["teacher"],
        variant_calibration_path=outputs["calibration"],
    )
    assert validated["factor_id"] == contract["factor_id"]
    assert contract["support_audit"]["unsupported_anchor_count"] == 0
    assert bundle["paths"]["teacher_anchor_map"] != bundle["paths"]["source_map"]
    assert contract["teacher_anchor_map_audit"][
        "anchor_ids_geometry_type_and_topology_bitwise_equal"
    ]

    candidate_map = torch.load(outputs["map"], weights_only=False)
    candidate_cache = torch.load(outputs["descriptor_cache"], weights_only=False)
    candidate_metric = torch.load(outputs["metric"], weights_only=False)
    candidate_teacher = torch.load(outputs["teacher"], weights_only=False)
    candidate_calibration = json.loads(outputs["calibration"].read_text())
    name = bundle["names"][0]
    descriptor = candidate_cache["queries"][name]["native_descriptors"]
    assert descriptor.dtype == torch.float32
    assert descriptor.shape == (1, 320)
    assert candidate_map["anchor_features"].shape[1] == 320
    assert all(
        bool((torch.as_tensor(value) == 0).all())
        for value in candidate_metric["metric_state_dict"].values()
    )
    assert candidate_teacher["query_names"] == bundle["teacher"]["query_names"]
    for source_record, candidate_record in zip(
        bundle["teacher"]["records"], candidate_teacher["records"]
    ):
        assert set(source_record) == set(candidate_record)
        for field in source_record:
            if isinstance(source_record[field], torch.Tensor):
                assert torch.equal(source_record[field], candidate_record[field])
            else:
                assert source_record[field] == candidate_record[field]
    assert candidate_calibration["statistics"] == bundle["calibration"]["statistics"]
    assert candidate_calibration["parameters"] == bundle["calibration"]["parameters"]
    assert candidate_calibration["policy"] == bundle["calibration"]["policy"]

    old_metric = load_shared_metric(
        bundle["paths"]["source_metric"],
        anchor_ids=bundle["source_map"]["anchor_ids"],
        device=torch.device("cpu"),
    )
    source_query = F.normalize(
        bundle["source_cache"]["queries"][name]["native_descriptors"].float(), dim=1
    )
    adapted_query, _ = old_metric(source_query)
    source_bank = F.normalize(bundle["source_map"]["anchor_features"].float(), dim=1)
    xfeat_query = F.normalize(
        bundle["probe"]["queries"][name]["descriptor_at_reference_keypoints"],
        dim=1,
    )
    xfeat_bank = candidate_map["anchor_features"][:, 256:] * (2.0**0.5)
    expected = 0.5 * (adapted_query @ source_bank.T) + 0.5 * (
        xfeat_query @ xfeat_bank.T
    )
    identity = load_shared_metric(
        outputs["metric"],
        anchor_ids=candidate_map["anchor_ids"],
        device=torch.device("cpu"),
    )
    actual_query, residual = identity(descriptor)
    actual = actual_query @ F.normalize(candidate_map["anchor_features"], dim=1).T
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)


def test_materializer_fails_closed_instead_of_falling_back_for_unseen_anchor(tmp_path):
    bundle = _bundle(tmp_path / "inputs", anchor_count=3, query_count=2)
    with pytest.raises(ValueError, match="forbids unsupported-anchor fallback"):
        _materialize(bundle, tmp_path / "factor")


def test_materializer_rejects_teacher_anchor_map_registry_tamper_before_write(
    tmp_path,
):
    bundle = _bundle(tmp_path / "inputs")
    teacher_anchor_map_path = bundle["paths"]["teacher_anchor_map"]
    teacher_anchor_map = torch.load(teacher_anchor_map_path, weights_only=False)
    teacher_anchor_map["anchor_xyz"][0, 0] += 1
    torch.save(teacher_anchor_map, teacher_anchor_map_path)
    teacher_anchor_map_sha256 = sha256_file(teacher_anchor_map_path)
    bundle["kwargs"]["teacher_anchor_map_sha256"] = teacher_anchor_map_sha256

    extension_path = bundle["paths"]["deployment_extension"]
    extension = json.loads(extension_path.read_text())
    extension["source_artifacts"]["teacher_anchor_map"][
        "sha256"
    ] = teacher_anchor_map_sha256
    extension_path.write_text(json.dumps(extension))
    bundle["kwargs"]["deployment_extension_sha256"] = sha256_file(extension_path)
    output = tmp_path / "factor"
    with pytest.raises(ValueError, match="teacher anchor map registry differs"):
        _materialize(bundle, output)
    assert not output.exists()


def test_live_pair_audit_rejects_candidate_geometry_mutation(tmp_path):
    bundle = _bundle(tmp_path / "inputs")
    result = _materialize(bundle, tmp_path / "factor")
    contract = result["contract"]
    outputs = {name: Path(value["path"]) for name, value in contract["outputs"].items()}
    candidate_map = torch.load(outputs["map"], weights_only=False)
    candidate_map["anchor_xyz"][0, 0] += 1
    torch.save(candidate_map, outputs["map"])
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_descriptor_factor_pair(
            result["contract_path"],
            source_map_path=bundle["paths"]["source_map"],
            source_metric_path=bundle["paths"]["source_metric"],
            source_query_cache_path=bundle["paths"]["source_query_cache"],
            teacher_path=bundle["paths"]["teacher"],
            calibration_path=bundle["paths"]["calibration"],
            variant_map_path=outputs["map"],
            variant_metric_path=outputs["metric"],
            variant_query_cache_path=outputs["descriptor_cache"],
            variant_teacher_path=outputs["teacher"],
            variant_calibration_path=outputs["calibration"],
        )


def test_live_pair_audit_rejects_cache_tamper_even_with_resigned_contract(tmp_path):
    bundle = _bundle(tmp_path / "inputs")
    result = _materialize(bundle, tmp_path / "factor")
    contract_path = Path(result["contract_path"])
    contract = json.loads(contract_path.read_text())
    outputs = {name: Path(value["path"]) for name, value in contract["outputs"].items()}

    candidate_cache = torch.load(outputs["descriptor_cache"], weights_only=False)
    first = bundle["names"][0]
    candidate_cache["queries"][first]["native_keypoints"] += 123
    candidate_cache["queries"][first]["native_descriptors"].zero_()
    torch.save(candidate_cache, outputs["descriptor_cache"])
    cache_sha = sha256_file(outputs["descriptor_cache"])

    calibration = json.loads(outputs["calibration"].read_text())
    calibration["sources"]["query_cache_sha256"] = cache_sha
    outputs["calibration"].write_text(json.dumps(calibration))
    contract["outputs"]["descriptor_cache"]["sha256"] = cache_sha
    contract["outputs"]["calibration"]["sha256"] = sha256_file(
        outputs["calibration"]
    )
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="immutable field differs at native_keypoints"):
        _pair_audit(bundle, {**result, "contract": contract})


def test_materializer_rejects_non_go_mechanism_even_when_resigned(tmp_path):
    bundle = _bundle(tmp_path / "inputs")
    gate_path = bundle["paths"]["mechanism_gate"]
    gate = json.loads(gate_path.read_text())
    gate["gates"]["pooled_r8_non_regression"] = False
    gate["decision"] = "STOP"
    gate["mechanism_gate_passed"] = False
    gate_path.write_text(json.dumps(gate))
    bundle["kwargs"]["mechanism_gate_sha256"] = sha256_file(gate_path)
    with pytest.raises(ValueError, match="mapping-only GO"):
        _materialize(bundle, tmp_path / "factor")


def test_contract_rejects_producer_commit_mismatch(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "inputs")
    result = _materialize(bundle, tmp_path / "factor")
    mismatch = {**PRODUCER_IDENTITY, "git_commit": "b" * 40}
    monkeypatch.setattr(
        descriptor_factor,
        "descriptor_factor_producer_identity",
        lambda **_: mismatch,
    )
    with pytest.raises(ValueError, match="producer Git commit differs"):
        validate_descriptor_factor_contract(result["contract_path"])


def test_real_producer_identity_rejects_dirty_worktree(monkeypatch):
    monkeypatch.setattr(
        descriptor_factor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M tracked_file.py\n"),
    )
    with pytest.raises(RuntimeError, match="requires a clean worktree"):
        REAL_PRODUCER_IDENTITY(require_clean=True)


def _metrics(**updates) -> dict:
    value = {
        "query_count": 256,
        "raw_gt_precision_percent": 12.0,
        "median_te_cm": 1.0,
        "mean_te_cm": 1.2,
        "p90_te_cm": 2.0,
        "cvar95_te_cm": 3.0,
        "median_ae_deg": 0.5,
        "mean_ae_deg": 0.6,
        "p90_ae_deg": 0.8,
        "p95_ae_deg": 1.0,
        "recall_5cm_5deg_percent": 60.0,
        "catastrophic_100cm_count": 0,
    }
    value.update(updates)
    return value


def _summaries(
    root: Path,
    *,
    artifacts: dict[str, Path],
    query_names: list[str],
    metrics: dict,
    factor: dict | None,
) -> dict[int, Path]:
    root.mkdir(parents=True)
    indices = (
        torch.linspace(0, len(query_names) - 1, steps=256)
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    selected_names = [query_names[index] for index in indices]
    artifact_records = {
        role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for role, path in artifacts.items()
    }
    result = {}
    for seed in (2026, 2027, 2028):
        path = root / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "lafgs_mapping_cache_evaluation",
                    "version": 2,
                    "uses_test_queries": False,
                    "seed": seed,
                    "evaluation_code": EVALUATION_CODE,
                    "map": str(artifacts["map"].resolve()),
                    "metric_state": str(artifacts["metric"].resolve()),
                    "complete_positive_teacher": str(artifacts["teacher"].resolve()),
                    "query_cache": str(artifacts["query_cache"].resolve()),
                    "descriptor_cache": str(artifacts["query_cache"].resolve()),
                    "descriptor_factor_contract": (
                        str(artifacts["descriptor_factor"].resolve())
                        if factor is not None
                        else None
                    ),
                    "scene_calibration": str(artifacts["calibration"].resolve()),
                    "artifacts": artifact_records,
                    "deployment_row_limit": 0,
                    "pose_error_units": {"translation": "cm", "rotation": "deg"},
                    "query_count": 256,
                    "query_selection": "uniform_mapping_gate",
                    "evaluation_protocol": {
                        "split": "mapping_only",
                        "query_selection": "uniform_mapping_gate",
                        "requested_query_count": 256,
                        "evaluated_query_count": 256,
                        "teacher_query_count": len(query_names),
                        "ordered_teacher_query_names_sha256": _json_sha256(query_names),
                        "selected_query_indices": indices,
                        "selected_query_indices_sha256": _json_sha256(indices),
                        "selected_query_names_sha256": _json_sha256(selected_names),
                        "deployment_row_limit": 0,
                        "descriptor_protocol": (
                            {
                                "kind": "equal_energy_descriptor_factor",
                                "factor_id": factor["factor_id"],
                                "source_descriptor_dim": 256,
                                "xfeat_descriptor_dim": 64,
                                "effective_descriptor_dim": 320,
                                "strict_identity_metric": True,
                                "one_materialized_bank": True,
                                "one_global_top1": True,
                                "one_poselib_call_per_query": True,
                            }
                            if factor is not None
                            else {
                                "kind": "canonical_query_cache_shared_metric",
                                "descriptor_cache_equals_query_cache": True,
                            }
                        ),
                    },
                    "summary": metrics,
                }
            )
        )
        result[seed] = path
    return result


def test_pose_gate_accepts_two_cache_paths_only_through_strict_factor_contract(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path / "inputs", query_count=300)
    materialized = _materialize(bundle, tmp_path / "factor")
    contract = materialized["contract"]
    output = {name: Path(value["path"]) for name, value in contract["outputs"].items()}
    baseline = {
        "map": bundle["paths"]["source_map"],
        "metric": bundle["paths"]["source_metric"],
        "teacher": bundle["paths"]["teacher"],
        "query_cache": bundle["paths"]["source_query_cache"],
        "calibration": bundle["paths"]["calibration"],
    }
    variant = {
        "map": output["map"],
        "metric": output["metric"],
        "teacher": output["teacher"],
        "query_cache": output["descriptor_cache"],
        "calibration": output["calibration"],
    }
    baseline_summaries = _summaries(
        tmp_path / "baseline_summaries",
        artifacts=baseline,
        query_names=bundle["names"],
        metrics=_metrics(),
        factor=None,
    )
    variant_with_factor = {**variant, "descriptor_factor": Path(materialized["contract_path"])}
    variant_summaries = _summaries(
        tmp_path / "variant_summaries",
        artifacts=variant_with_factor,
        query_names=bundle["names"],
        metrics=_metrics(mean_te_cm=1.16),
        factor=contract,
    )
    monkeypatch.setattr(
        pose_gate, "mapping_pose_evaluation_code_identity", lambda **_: EVALUATION_CODE
    )
    report = pose_gate.compare_mapping_pose_gate(
        baseline_summaries=baseline_summaries,
        variant_summaries=variant_summaries,
        baseline_artifacts=baseline,
        variant_artifacts=variant,
        variant_descriptor_factor=materialized["contract_path"],
        expected_sha256={
            "variant.descriptor_factor": materialized["contract_sha256"]
        },
    )
    assert report["decision"]["verdict"] == "PASS"
    assert report["lineage"]["checks"]["query_caches_descriptor_factor_equivalent"]
    assert report["lineage"]["checks"]["anchor_registry_bitwise_equal"]
    assert report["lineage"]["checks"]["teacher_rebind_only"]
    assert report["lineage"]["checks"]["calibration_rebind_only"]

    with pytest.raises(ValueError, match="requires expected SHA-256"):
        pose_gate.compare_mapping_pose_gate(
            baseline_summaries=baseline_summaries,
            variant_summaries=variant_summaries,
            baseline_artifacts=baseline,
            variant_artifacts=variant,
            variant_descriptor_factor=materialized["contract_path"],
        )
    with pytest.raises(ValueError, match="paired mapping-pose lineage differs"):
        pose_gate.compare_mapping_pose_gate(
            baseline_summaries=baseline_summaries,
            variant_summaries=variant_summaries,
            baseline_artifacts=baseline,
            variant_artifacts=variant,
        )


def test_evaluator_consumes_candidate_cache_and_self_binds_factor(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path / "inputs", query_count=300)
    materialized = _materialize(bundle, tmp_path / "factor")
    outputs = {
        name: Path(value["path"])
        for name, value in materialized["contract"]["outputs"].items()
    }
    captured = {}

    def collect_stub(**kwargs):
        captured.update(kwargs)
        indices = [int(value) for value in kwargs["query_indices"].tolist()]
        return {
            "summary": _metrics(),
            "queries": [
                {"query_index": index, "image_name": bundle["names"][index]}
                for index in indices
            ],
            "counters": {"winner_count": torch.zeros(1, dtype=torch.float64)},
        }

    monkeypatch.setattr(
        evaluate_mapping_cache, "collect_deployment_statistics", collect_stub
    )
    monkeypatch.setattr(
        evaluate_mapping_cache,
        "mapping_pose_evaluation_code_identity",
        lambda **_: EVALUATION_CODE,
    )
    report_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.evaluate_mapping_cache",
            "--map",
            str(outputs["map"]),
            "--metric-state",
            str(outputs["metric"]),
            "--complete-positive-teacher",
            str(outputs["teacher"]),
            "--query-cache",
            str(outputs["descriptor_cache"]),
            "--scene-calibration",
            str(outputs["calibration"]),
            "--descriptor-factor-contract",
            materialized["contract_path"],
            "--query-count",
            "256",
            "--device",
            "cpu",
            "--output",
            str(report_dir),
        ],
    )
    evaluate_mapping_cache.main()
    report = json.loads((report_dir / "mapping_cache_summary.json").read_text())
    first = captured["query_cache"]["queries"][bundle["names"][0]][
        "native_descriptors"
    ]
    assert first.shape[1] == 320
    assert set(report["artifacts"]) == {
        "map",
        "metric",
        "teacher",
        "query_cache",
        "calibration",
        "descriptor_factor",
    }
    assert report["descriptor_factor_contract"] == materialized["contract_path"]
    assert report["evaluation_protocol"]["descriptor_protocol"][
        "one_global_top1"
    ]
