from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from map_learning.frontend_upper_bound import tensor_sha256
import scripts.compare_frontend_detector_arm_a as detector_gate


EVALUATION_CODE = {
    "schema": "lafgs_frontend_detector_evaluation_code",
    "version": 1,
    "repository": "/synthetic/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {
        "map_learning/frontend_upper_bound.py": "b" * 64,
        "scripts/audit_frontend_upper_bound.py": "c" * 64,
        "scripts/compare_frontend_detector_arm_a.py": "d" * 64,
    },
}
RADII = ("2.0", "4.0", "8.0")
KINDS = ("all", "track_core", "gaussian_reserve")


def _source_entry(path: Path) -> dict:
    return {
        "path": str(path),
        "status": "present_unverified",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "expected_sha256": None,
    }


def _counts(track_hits: tuple[int, int, int], reserve_hits: tuple[int, int, int]):
    return {
        "all": {
            "target_count": 20,
            "hit_count": {
                radius: track_hits[index] + reserve_hits[index]
                for index, radius in enumerate(RADII)
            },
        },
        "track_core": {
            "target_count": 12,
            "hit_count": dict(zip(RADII, track_hits)),
        },
        "gaussian_reserve": {
            "target_count": 8,
            "hit_count": dict(zip(RADII, reserve_hits)),
        },
    }


def _pooled(counts: dict) -> dict:
    return {
        "query_count": 2,
        "by_anchor_kind": {
            kind: {
                **counts[kind],
                "reachable_fraction": {
                    radius: counts[kind]["hit_count"][radius]
                    / counts[kind]["target_count"]
                    for radius in RADII
                },
            }
            for kind in KINDS
        },
    }


def _per_query(counts: dict) -> dict:
    kinds = {
        kind: {
            "target_count": counts[kind]["target_count"] // 2,
            "hit_count": {
                radius: counts[kind]["hit_count"][radius] // 2
                for radius in RADII
            },
        }
        for kind in KINDS
    }
    return {
        "target_count": kinds["all"]["target_count"],
        "hit_count": kinds["all"]["hit_count"],
        "by_anchor_kind": kinds,
    }


def _deltas(baseline: dict, candidate: dict) -> dict:
    return {
        kind: {
            radius: (
                candidate[kind]["hit_count"][radius]
                - baseline[kind]["hit_count"][radius]
            )
            / baseline[kind]["target_count"]
            for radius in RADII
        }
        for kind in KINDS
    }


def _bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_track=(4, 8, 10),
    candidate_reserve=(2, 4, 8),
    probe_mutator=None,
    report_mutator=None,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("state", "query_cache", "teacher", "weights"):
        path = (tmp_path / f"{name}.pt").resolve()
        path.write_bytes(f"synthetic-{name}".encode())
        paths[name] = path
    external = tmp_path / "external"
    external.mkdir()
    external_files = {}
    for name in ("model", "interpolator", "wrapper"):
        path = (external / f"{name}.py").resolve()
        path.write_text(f"# synthetic {name}\n", encoding="utf-8")
        external_files[name] = path
    monkeypatch.setattr(
        detector_gate,
        "frontend_detector_evaluation_code_identity",
        lambda **_: copy.deepcopy(EVALUATION_CODE),
    )
    monkeypatch.setattr(
        detector_gate, "EXPECTED_XFEAT_WEIGHTS_SHA256", sha256_file(paths["weights"])
    )
    monkeypatch.setattr(detector_gate, "EXPECTED_XFEAT_TREE", "e" * 40)
    monkeypatch.setattr(
        detector_gate, "EXPECTED_XFEAT_MODEL_SHA256", sha256_file(external_files["model"])
    )
    monkeypatch.setattr(
        detector_gate,
        "EXPECTED_XFEAT_INTERPOLATOR_SHA256",
        sha256_file(external_files["interpolator"]),
    )
    monkeypatch.setattr(
        detector_gate,
        "EXPECTED_XFEAT_WRAPPER_SHA256",
        sha256_file(external_files["wrapper"]),
    )
    monkeypatch.setattr(detector_gate, "EXPECTED_IMPLEMENTATION_ID", "synthetic-arm-a")
    monkeypatch.setattr(detector_gate, "EXPECTED_QUERY_COUNT", 2)

    root = Path(detector_gate.__file__).resolve().parents[1]
    implementation_files = {
        relative: {
            "path": str((root / relative).resolve()),
            "sha256": sha256_file(root / relative),
        }
        for relative in detector_gate.EXPECTED_PRODUCER_IMPLEMENTATION_FILES
    }
    query_names = ("seq/frame000.png", "seq/frame001.png")
    query_names_sha256 = hashlib.sha256(
        "".join(f"{len(name)}:{name}\n" for name in query_names).encode("utf-8")
    ).hexdigest()
    query_records = {}
    for index, name in enumerate(query_names):
        keypoints = torch.tensor([[9.0, 9.0], [17.0, 9.0]])
        scores = torch.tensor([0.9, 0.8])
        query_records[name] = {
            "query_index": index,
            "query_name": name,
            "reference_keypoints_sha256": "f" * 64,
            "detector_keypoints": keypoints,
            "detector_keypoints_sha256": tensor_sha256(keypoints),
            "detector_scores": scores,
            "detector_scores_sha256": tensor_sha256(scores),
            "detected_count_before_mask": 2,
            "detected_count_after_mask": 2,
            "image_lineage": {},
            "detector_lineage": {
                "keypoint_heatmap": "softmax_65_discard_dustbin_then_8x8_unpack",
                "nms_kernel_size": 5,
                "nms_radius": 2,
                "nms_passes": 1,
                "strict_probability_threshold": 0.05,
                "score_semantics": "nearest_probability_times_bilinear_reliability",
                "origin_padding_sentinel_excluded": True,
                "sort": "descending_score_stable_row_major_ties",
                "top_k_before_native_mask": 1024,
                "candidate_count_after_threshold_nms": 2,
                "positive_top_k_count_before_mask": 2,
                "post_mask_count": 2,
                "native_mask_filter": "sample_mask_at_grid_uv_nearest_round",
                "shared_forward_descriptor_output_used": False,
                "candidate_descriptor_rows_materialized": False,
                "pair_matcher_used": False,
                "mask_equivalence_proof": {
                    "required_native_hw_divisible_by_32": True,
                    "native_hw_divisible_by_32": True,
                    "identity_xfeat_resize": True,
                    "integer_xfeat_coordinates": True,
                    "round_floor_indices_equal": True,
                    "round_floor_mask_decisions_equal": True,
                    "checked_pre_mask_rows": 2,
                    "round_indices_sha256": "1" * 64,
                    "floor_indices_sha256": "1" * 64,
                    "round_mask_keep_sha256": "2" * 64,
                    "floor_mask_keep_sha256": "2" * 64,
                },
            },
        }
    producer_protocol = detector_gate._expected_candidate_protocol()
    probe = {
        "schema": "lafgs_frontend_ceiling_probe_cache",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference": {
            "query_cache_path": str(paths["query_cache"]),
            "query_cache_sha256": sha256_file(paths["query_cache"]),
            "teacher_path": str(paths["teacher"]),
            "teacher_sha256": sha256_file(paths["teacher"]),
            "teacher_schema": "lafgs_v9_active_map_complete_positive_teacher",
            "query_names": list(query_names),
            "query_names_sha256": query_names_sha256,
            "reference_detector_protocol": {
                "name": "frozen_superpoint",
                "requested_keypoint_count": 1024,
                "nms_radius": 4,
                "mask_filter": "sample_mask_at_grid_uv_nearest_round",
                "top_k_before_native_mask": True,
            },
        },
        "frontend": {
            "name": "xfeat_sparse_64d_detector_only",
            "family": "independent_local_frontend",
            "implementation_id": "synthetic-arm-a",
            "coordinate_convention": (
                "reference_grid_index_then_cached_pixel_center_offset"
            ),
            "descriptor_dim": 64,
            "requested_keypoint_count": 1024,
            "weights": {
                "path": str(paths["weights"]),
                "sha256": sha256_file(paths["weights"]),
                "size_bytes": paths["weights"].stat().st_size,
            },
            "code": {
                "xfeat_tree": "e" * 40,
                "git_clean": True,
                **{
                    name: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for name, path in external_files.items()
                },
            },
        },
        "capabilities": {
            "detector_repeatability": True,
            "descriptor_identity": False,
        },
        "producer": {
            "schema": "lafgs_xfeat_arm_a_producer",
            "version": 1,
            "arm": "A_detector_repeatability",
            "device": "cpu",
            "dtype": "float32",
            "gpu_used": False,
            "network_access_used": False,
            "candidate_detector_used": True,
            "shared_forward_descriptor_output_used": False,
            "candidate_descriptor_rows_materialized": False,
            "pair_matcher_used": False,
            "implementation_files": implementation_files,
            "state_dict": {},
            "query_count": 2,
            "detected_count_before_mask": 4,
            "detected_count_after_mask": 4,
            "all_queries_identity_xfeat_resize": True,
            "all_queries_round_floor_mask_equivalent": True,
            "detector_protocol": producer_protocol,
            "consumer_validation": {
                "query_count": 2,
                "requested_keypoint_count": 1024,
                "reference_descriptor_dim": 256,
                "candidate_descriptor_dim": None,
                "validated_descriptor_rows": 0,
                "validated_detector_keypoints": 4,
            },
            "cli": {
                "path": str((root / "scripts/materialize_xfeat_arm_a.py").resolve()),
                "sha256": sha256_file(root / "scripts/materialize_xfeat_arm_a.py"),
            },
        },
        "queries": query_records,
    }
    if probe_mutator is not None:
        probe_mutator(probe)
    probe_path = (tmp_path / "probe_cache.pt").resolve()
    torch.save(probe, probe_path)
    paths["probe_cache"] = probe_path

    baseline_counts = _counts((4, 6, 10), (2, 4, 6))
    candidate_counts = _counts(candidate_track, candidate_reserve)
    per_baseline = _per_query(baseline_counts)
    per_candidate = _per_query(candidate_counts)
    detector = {
        "schema": "lafgs_mapping_detector_repeatability_ceiling_probe",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "attestation": {
            "artifact": {
                "path": str(paths["weights"]),
                "sha256": sha256_file(paths["weights"]),
                "verified": True,
            },
            "reference_artifacts": {
                name: {
                    "path": str(paths[name]),
                    "sha256": sha256_file(paths[name]),
                    "verified": True,
                }
                for name in ("query_cache", "teacher")
            },
            "query_count": 2,
            "requested_keypoint_count": 1024,
            "reference_descriptor_dim": 256,
            "candidate_descriptor_dim": None,
            "validated_descriptor_rows": 0,
            "validated_detector_keypoints": 4,
        },
        "config": {
            "radii_px": [2.0, 4.0, 8.0],
            "depth_abs_tolerance_m": 0.05,
            "depth_rel_tolerance": 0.02,
            "alpha_minimum": 0.01,
            "target_universe": "frozen_map_gt_projection_depth_alpha_mask_legal",
            "same_requested_k": True,
        },
        "frozen_superpoint": _pooled(baseline_counts),
        "candidate": _pooled(candidate_counts),
        "delta_candidate_minus_superpoint": _deltas(
            baseline_counts, candidate_counts
        ),
        "per_query": [
            {
                "query_name": name,
                "legal_anchor_count": 10,
                "reference_keypoint_count": 2,
                "candidate_keypoint_count": 2,
                "frozen_superpoint": per_baseline,
                "candidate": per_candidate,
            }
            for name in query_names
        ],
    }
    report = {
        "schema": "lafgs_frontend_ceiling_probe_audit_bundle",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "deployment_modified": False,
        "evaluation_code": copy.deepcopy(EVALUATION_CODE),
        "probe_cache": str(probe_path),
        "source_artifacts": {
            name: _source_entry(paths[name])
            for name in ("state", "query_cache", "teacher", "probe_cache")
        },
        "detector_repeatability": detector,
    }
    if report_mutator is not None:
        report_mutator(report)
    report_path = (tmp_path / "report.json").resolve()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output = (tmp_path / "gate.json").resolve()
    argv = [
        "--report",
        str(report_path),
        "--expected-report-sha256",
        sha256_file(report_path),
        "--state",
        str(paths["state"]),
        "--expected-state-sha256",
        sha256_file(paths["state"]),
        "--query-cache",
        str(paths["query_cache"]),
        "--expected-query-cache-sha256",
        sha256_file(paths["query_cache"]),
        "--teacher",
        str(paths["teacher"]),
        "--expected-teacher-sha256",
        sha256_file(paths["teacher"]),
        "--probe-cache",
        str(paths["probe_cache"]),
        "--expected-probe-cache-sha256",
        sha256_file(paths["probe_cache"]),
        "--candidate-weights",
        str(paths["weights"]),
        "--expected-candidate-weights-sha256",
        sha256_file(paths["weights"]),
        "--expected-query-count",
        "2",
        "--expected-validated-detector-keypoints",
        "4",
        "--expected-all-target-count",
        "20",
        "--expected-track-target-count",
        "12",
        "--expected-reserve-target-count",
        "8",
        "--output",
        str(output),
    ]
    return {"paths": paths, "report": report_path, "output": output, "argv": argv}


def test_detector_arm_a_gate_passes_exact_detector_only_contract(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path, monkeypatch)
    detector_gate.main(bundle["argv"])
    result = json.loads(bundle["output"].read_text())
    assert result["decision"] == "GO"
    assert result["mechanism_gate_passed"] is True
    assert all(result["gates"].values())
    assert result["comparisons"]["all_at_4px"]["delta_hit_count"] == 2
    assert result["protocol"]["radii_px"] == [2.0, 4.0, 8.0]
    assert result["protocol"]["requested_keypoint_count"] == 1024


@pytest.mark.parametrize(
    ("track", "reserve", "failed_gate"),
    (
        ((2, 8, 10), (2, 4, 8), "all_at_2px_non_regression"),
        ((4, 6, 10), (2, 4, 8), "all_at_4px_strict_positive"),
        ((4, 8, 8), (2, 4, 6), "all_at_8px_non_regression"),
        ((4, 4, 10), (2, 8, 8), "track_core_at_4px_non_regression"),
        ((4, 10, 10), (2, 2, 8), "gaussian_reserve_at_4px_non_regression"),
    ),
)
def test_detector_arm_a_gate_writes_stop_for_each_preregistered_failure(
    tmp_path, monkeypatch, track, reserve, failed_gate
):
    bundle = _bundle(
        tmp_path,
        monkeypatch,
        candidate_track=track,
        candidate_reserve=reserve,
    )
    with pytest.raises(SystemExit) as error:
        detector_gate.main(bundle["argv"])
    assert error.value.code == 2
    result = json.loads(bundle["output"].read_text())
    assert result["decision"] == "STOP"
    assert result["gates"][failed_gate] is False


def test_detector_gate_rejects_descriptor_mixed_report(tmp_path, monkeypatch):
    bundle = _bundle(
        tmp_path,
        monkeypatch,
        report_mutator=lambda report: report.update({"descriptor_identity": {}}),
    )
    with pytest.raises(ValueError, match="mixes another arm"):
        detector_gate.main(bundle["argv"])
    assert not bundle["output"].exists()


def test_detector_gate_rejects_candidate_nms_or_descriptor_payload(
    tmp_path, monkeypatch
):
    def mutate(probe):
        probe["producer"]["detector_protocol"]["nms_kernel_size"] = 3
        first = next(iter(probe["queries"].values()))
        first["descriptor_at_reference_keypoints"] = torch.ones((2, 64))

    bundle = _bundle(tmp_path, monkeypatch, probe_mutator=mutate)
    with pytest.raises(ValueError, match="protocol differs|descriptor rows"):
        detector_gate.main(bundle["argv"])
    assert not bundle["output"].exists()


def test_detector_gate_rejects_evaluator_identity_or_source_mutation(
    tmp_path, monkeypatch
):
    bundle = _bundle(
        tmp_path,
        monkeypatch,
        report_mutator=lambda report: report["evaluation_code"].update(
            {"git_commit": "9" * 40}
        ),
    )
    with pytest.raises(ValueError, match="clean-code identity differs"):
        detector_gate.main(bundle["argv"])

    second = _bundle(tmp_path / "source", monkeypatch)
    second["paths"]["teacher"].write_bytes(b"mutated after evaluation")
    with pytest.raises(ValueError, match="current artifact"):
        detector_gate.main(second["argv"])


def test_detector_gate_rejects_nonadditive_target_counts(tmp_path, monkeypatch):
    def mutate(report):
        report["detector_repeatability"]["candidate"]["by_anchor_kind"]["all"][
            "target_count"
        ] += 1

    bundle = _bundle(tmp_path, monkeypatch, report_mutator=mutate)
    with pytest.raises(ValueError, match="target|fraction"):
        detector_gate.main(bundle["argv"])
