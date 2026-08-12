import json

import pytest

from common.hashing import sha256_file
import scripts.compare_frontend_descriptor_arm_b as descriptor_gate
from scripts.compare_frontend_descriptor_arm_b import main


TOPKS = (1, 2, 4, 8, 16, 32)
TEACHER_SCHEMA = "lafgs_v9_active_map_complete_positive_teacher"
EVALUATION_CODE = {
    "schema": "lafgs_frontend_descriptor_evaluation_code",
    "version": 1,
    "repository": "/synthetic/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {
        "map_learning/frontend_upper_bound.py": "b" * 64,
        "scripts/audit_frontend_upper_bound.py": "c" * 64,
        "scripts/compare_frontend_descriptor_arm_b.py": "d" * 64,
    },
}


def _summary(
    *,
    positive_recall,
    track_recall,
    reserve_recall,
    correct,
    false,
    ambiguous,
    replayed=10,
    positive=10,
    track=6,
    reserve=4,
):
    def keyed(values):
        return {str(topk): float(value) for topk, value in zip(TOPKS, values)}

    return {
        "row_counts": {
            "replayed_rows": replayed,
            "positive_eligible_rows": positive,
            "track_positive_eligible_rows": track,
            "reserve_positive_eligible_rows": reserve,
            "top1_correct": correct,
            "top1_false": false,
            "top1_ambiguous": ambiguous,
        },
        "positive_recall_at_k": keyed(positive_recall),
        "positive_recall_at_k_by_anchor_kind": {
            "track_core": keyed(track_recall),
            "gaussian_reserve": keyed(reserve_recall),
        },
        "false_top1_recoverable_at_k": keyed([0.0] * len(TOPKS)),
    }


def _delta(candidate, baseline):
    result = {
        str(topk): (
            candidate["positive_recall_at_k"][str(topk)]
            - baseline["positive_recall_at_k"][str(topk)]
        )
        for topk in TOPKS
    }
    result["by_anchor_kind"] = {
        kind: {
            str(topk): (
                candidate["positive_recall_at_k_by_anchor_kind"][kind][str(topk)]
                - baseline["positive_recall_at_k_by_anchor_kind"][kind][str(topk)]
            )
            for topk in TOPKS
        }
        for kind in ("track_core", "gaussian_reserve")
    }
    return result


def _direction(name, baseline, candidate):
    return {
        "direction": name,
        "support": {
            "support_query_count": 4,
            "positive_edge_count": 20,
            "minimum_support_views": 2,
            "supported_anchor_count": 10,
            "map_descriptor_memory_float32": {
                "formula": "supported_anchor_count * descriptor_dim * 4",
                "bytes_per_scalar": 4,
                "frozen_superpoint_dim": 256,
                "candidate_dim": 64,
                "frozen_superpoint_bytes": 10 * 256 * 4,
                "candidate_bytes": 10 * 64 * 4,
                "candidate_to_superpoint_ratio": 0.25,
            },
        },
        "ranking_resources": {
            "frozen_superpoint": {
                "descriptor_dim": 256,
                "ranking_wall_seconds": 0.01,
                "query_rows": 10,
                "score_elements": 100,
                "dot_product_multiply_accumulates": 25_600,
            },
            "candidate": {
                "descriptor_dim": 64,
                "ranking_wall_seconds": 0.005,
                "query_rows": 10,
                "score_elements": 100,
                "dot_product_multiply_accumulates": 6_400,
            },
            "candidate_to_superpoint_ratio": {
                "ranking_wall_seconds": 0.5,
                "dot_product_multiply_accumulates": 0.25,
            },
            "latency_note": "synthetic",
        },
        "heldout_query_count": 4,
        "frozen_superpoint": baseline,
        "candidate": candidate,
        "delta_candidate_minus_superpoint": _delta(candidate, baseline),
    }


def _artifacts(tmp_path):
    paths = {}
    for name in ("state", "query_cache", "teacher", "probe_cache", "weights"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(f"synthetic-{name}".encode())
        paths[name] = path.resolve()
    return paths


def _report(paths):
    baseline_one = _summary(
        positive_recall=[0.2, 0.4, 0.5, 0.7, 0.8, 0.9],
        track_recall=[0.3, 0.5, 0.6, 0.75, 0.85, 0.9],
        reserve_recall=[0.2, 0.3, 0.5, 0.65, 0.75, 0.85],
        correct=2,
        false=6,
        ambiguous=2,
    )
    candidate_one = _summary(
        positive_recall=[0.3, 0.45, 0.55, 0.75, 0.82, 0.91],
        track_recall=[0.31, 0.52, 0.62, 0.76, 0.86, 0.91],
        reserve_recall=[0.2, 0.35, 0.52, 0.67, 0.77, 0.86],
        correct=3,
        false=5,
        ambiguous=2,
    )
    baseline_two = _summary(
        positive_recall=[0.3, 0.5, 0.6, 0.75, 0.85, 0.9],
        track_recall=[0.3, 0.5, 0.65, 0.8, 0.88, 0.92],
        reserve_recall=[0.2, 0.4, 0.55, 0.7, 0.8, 0.88],
        correct=3,
        false=5,
        ambiguous=2,
    )
    candidate_two = _summary(
        positive_recall=[0.4, 0.55, 0.65, 0.8, 0.88, 0.93],
        track_recall=[0.32, 0.52, 0.67, 0.82, 0.89, 0.93],
        reserve_recall=[0.21, 0.42, 0.57, 0.72, 0.82, 0.89],
        correct=4,
        false=4,
        ambiguous=2,
    )
    pooled_baseline = _summary(
        positive_recall=[0.25, 0.45, 0.55, 0.725, 0.825, 0.9],
        track_recall=[0.3, 0.5, 0.625, 0.775, 0.865, 0.91],
        reserve_recall=[0.25, 0.35, 0.525, 0.675, 0.775, 0.865],
        correct=5,
        false=11,
        ambiguous=4,
        replayed=20,
        positive=20,
        track=12,
        reserve=8,
    )
    pooled_candidate = _summary(
        positive_recall=[0.35, 0.5, 0.6, 0.775, 0.85, 0.92],
        track_recall=[0.3, 0.52, 0.645, 0.79, 0.875, 0.92],
        reserve_recall=[0.25, 0.385, 0.545, 0.695, 0.795, 0.875],
        correct=7,
        false=9,
        ambiguous=4,
        replayed=20,
        positive=20,
        track=12,
        reserve=8,
    )
    sources = {
        name: {
            "path": str(paths[name]),
            "status": "present_unverified",
            "size_bytes": paths[name].stat().st_size,
            "sha256": sha256_file(paths[name]),
            "expected_sha256": None,
        }
        for name in ("state", "query_cache", "teacher", "probe_cache")
    }
    descriptor = {
        "schema": "lafgs_mapping_descriptor_identity_ceiling_probe",
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
            "query_count": 8,
            "requested_keypoint_count": 1024,
            "reference_descriptor_dim": 256,
            "candidate_descriptor_dim": 64,
            "validated_descriptor_rows": 8192,
            "validated_detector_keypoints": 0,
        },
        "protocol": {
            "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
            "positive_labels": TEACHER_SCHEMA,
            "map_bank": "same_positive_edges_view_balanced_support_only",
            "ranking": "global_cosine",
            "crossfit": "bidirectional_temporal_block",
            "crossfit_blocks": 8,
            "minimum_support_views": 2,
            "topks": list(TOPKS),
            "candidate_detector_used": False,
            "descriptor_dimension_policy": (
                "native_dimensions_may_differ; rows_edges_folds_K_are_paired"
            ),
        },
        "split": {
            "policy": "per_sequence_alternating_contiguous_temporal_blocks",
            "block_count": 8,
            "selection_query_count": 4,
            "gate_query_count": 4,
            "assignments": {f"seq/frame{index}.png": index for index in range(8)},
            "uses_test_queries": False,
        },
        "directions": [
            _direction("selection_to_gate", baseline_one, candidate_one),
            _direction("gate_to_selection", baseline_two, candidate_two),
        ],
        "pooled": {
            "frozen_superpoint": pooled_baseline,
            "candidate": pooled_candidate,
        },
        "delta_candidate_minus_superpoint": _delta(pooled_candidate, pooled_baseline),
    }
    return {
        "schema": "lafgs_frontend_ceiling_probe_audit_bundle",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "deployment_modified": False,
        "probe_cache": str(paths["probe_cache"]),
        "source_artifacts": sources,
        "descriptor_identity": descriptor,
    }


def _write_report(tmp_path, payload, name="report"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path.resolve()


def _argv(paths, report, output):
    return [
        "--report",
        str(report),
        "--expected-report-sha256",
        sha256_file(report),
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
        "8",
        "--expected-requested-keypoint-count",
        "1024",
        "--expected-reference-descriptor-dim",
        "256",
        "--expected-candidate-descriptor-dim",
        "64",
        "--expected-validated-descriptor-rows",
        "8192",
        "--expected-teacher-schema",
        TEACHER_SCHEMA,
        "--output",
        str(output),
    ]


def _refresh_delta(report):
    descriptor = report["descriptor_identity"]
    for direction in descriptor["directions"]:
        direction["delta_candidate_minus_superpoint"] = _delta(
            direction["candidate"], direction["frozen_superpoint"]
        )
    descriptor["delta_candidate_minus_superpoint"] = _delta(
        descriptor["pooled"]["candidate"],
        descriptor["pooled"]["frozen_superpoint"],
    )


def _equal_energy_report(paths):
    payload = _report(paths)
    payload["evaluation_code"] = EVALUATION_CODE
    descriptor = payload["descriptor_identity"]
    descriptor["schema"] = "lafgs_mapping_descriptor_equal_energy_ceiling_probe"
    descriptor["protocol"] = {
        "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
        "positive_labels": TEACHER_SCHEMA,
        "map_bank": "same_positive_edges_view_balanced_support_only",
        "ranking": "single_global_cosine",
        "crossfit": "bidirectional_temporal_block",
        "crossfit_blocks": 8,
        "minimum_support_views": 2,
        "topks": list(TOPKS),
        "candidate_detector_used": False,
        "candidate_representation": ("l2_concat(l2(superpoint),l2(candidate))/sqrt(2)"),
        "score_identity": "0.5*cosine_superpoint+0.5*cosine_candidate",
        "source_candidate_descriptor_dim": 64,
        "effective_candidate_descriptor_dim": 320,
        "learned_fusion_parameters": False,
        "source_specific_descriptor_routing": False,
    }
    for direction in descriptor["directions"]:
        support = direction["support"]
        anchors = support["supported_anchor_count"]
        memory = support["map_descriptor_memory_float32"]
        memory["candidate_dim"] = 320
        memory["candidate_bytes"] = anchors * 320 * 4
        memory["candidate_to_superpoint_ratio"] = 1.25
        candidate = direction["ranking_resources"]["candidate"]
        candidate["descriptor_dim"] = 320
        candidate["dot_product_multiply_accumulates"] = (
            candidate["query_rows"] * anchors * 320
        )
        direction["ranking_resources"]["candidate_to_superpoint_ratio"][
            "dot_product_multiply_accumulates"
        ] = 1.25
    return payload


def test_descriptor_arm_b_gate_passes_exact_bidirectional_contract(tmp_path):
    paths = _artifacts(tmp_path)
    report = _write_report(tmp_path, _report(paths))
    output = tmp_path / "gate.json"
    main(_argv(paths, report, output))
    result = json.loads(output.read_text())
    assert result["decision"] == "GO"
    assert result["mechanism_gate_passed"] is True
    assert all(result["gates"].values())
    assert result["protocol"]["strict_positive_r1_delta_threshold"] == 0.0
    assert result["protocol"]["non_regression_absolute_tolerance"] == 1e-12


def test_equal_energy_gate_requires_exact_single_320d_representation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        descriptor_gate,
        "frontend_descriptor_evaluation_code_identity",
        lambda **_: EVALUATION_CODE,
    )
    paths = _artifacts(tmp_path)
    report = _write_report(tmp_path, _equal_energy_report(paths))
    output = tmp_path / "equal_energy_gate.json"
    argv = _argv(paths, report, output)
    argv.extend(
        [
            "--candidate-representation",
            "equal_energy_superpoint_candidate",
            "--expected-effective-candidate-descriptor-dim",
            "320",
        ]
    )
    main(argv)
    result = json.loads(output.read_text())
    assert result["schema"] == ("lafgs_frontend_descriptor_equal_energy_mechanism_gate")
    assert result["decision"] == "GO"
    assert result["protocol"]["source_candidate_descriptor_dim"] == 64
    assert result["protocol"]["effective_candidate_descriptor_dim"] == 320

    invalid = _equal_energy_report(paths)
    invalid["descriptor_identity"]["protocol"]["source_specific_descriptor_routing"] = (
        True
    )
    invalid_report = _write_report(tmp_path, invalid, "invalid_equal_energy")
    invalid_argv = _argv(paths, invalid_report, tmp_path / "invalid_gate.json")
    invalid_argv.extend(
        [
            "--candidate-representation",
            "equal_energy_superpoint_candidate",
            "--expected-effective-candidate-descriptor-dim",
            "320",
        ]
    )
    with pytest.raises(ValueError, match="protocol differs"):
        main(invalid_argv)


def test_equal_energy_gate_rejects_missing_dimension_or_resource_tamper(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        descriptor_gate,
        "frontend_descriptor_evaluation_code_identity",
        lambda **_: EVALUATION_CODE,
    )
    payload = _equal_energy_report(_artifacts(tmp_path))
    paths = {
        name: (tmp_path / f"{name}.pt").resolve()
        for name in ("state", "query_cache", "teacher", "probe_cache", "weights")
    }
    report = _write_report(tmp_path, payload)
    argv = _argv(paths, report, tmp_path / "missing_dim_gate.json")
    argv.extend(
        [
            "--candidate-representation",
            "equal_energy_superpoint_candidate",
        ]
    )
    with pytest.raises(ValueError, match="dimension is not additive"):
        main(argv)

    payload = _equal_energy_report(paths)
    payload["descriptor_identity"]["directions"][0]["ranking_resources"]["candidate"][
        "dot_product_multiply_accumulates"
    ] += 1
    report = _write_report(tmp_path, payload, "resource_tamper")
    argv = _argv(paths, report, tmp_path / "resource_gate.json")
    argv.extend(
        [
            "--candidate-representation",
            "equal_energy_superpoint_candidate",
            "--expected-effective-candidate-descriptor-dim",
            "320",
        ]
    )
    with pytest.raises(ValueError, match="MAC formula differs"):
        main(argv)

    payload = _equal_energy_report(paths)
    payload["evaluation_code"] = {
        **EVALUATION_CODE,
        "git_commit": "e" * 40,
    }
    report = _write_report(tmp_path, payload, "code_mismatch")
    argv = _argv(paths, report, tmp_path / "code_gate.json")
    argv.extend(
        [
            "--candidate-representation",
            "equal_energy_superpoint_candidate",
            "--expected-effective-candidate-descriptor-dim",
            "320",
        ]
    )
    with pytest.raises(ValueError, match="code identity differs"):
        main(argv)


@pytest.mark.parametrize(
    ("regression", "expected_pass"),
    ((5e-13, True), (2e-12, False)),
)
def test_descriptor_arm_b_gate_uses_exact_non_regression_tolerance(
    tmp_path, regression, expected_pass
):
    paths = _artifacts(tmp_path)
    payload = _report(paths)
    descriptor = payload["descriptor_identity"]
    baseline = descriptor["pooled"]["frozen_superpoint"]["positive_recall_at_k"]["8"]
    descriptor["pooled"]["candidate"]["positive_recall_at_k"]["8"] = (
        baseline - regression
    )
    _refresh_delta(payload)
    report = _write_report(tmp_path, payload)
    output = tmp_path / "gate.json"
    if expected_pass:
        main(_argv(paths, report, output))
    else:
        with pytest.raises(SystemExit) as error:
            main(_argv(paths, report, output))
        assert error.value.code == 2
    assert json.loads(output.read_text())["mechanism_gate_passed"] is expected_pass


@pytest.mark.parametrize(
    "failure",
    (
        "first_direction_r1",
        "second_direction_r1",
        "pooled_r8",
        "track_r1",
        "reserve_r1",
    ),
)
def test_descriptor_arm_b_gate_writes_stop_and_exits_two(tmp_path, failure):
    paths = _artifacts(tmp_path)
    payload = _report(paths)
    descriptor = payload["descriptor_identity"]
    if failure == "first_direction_r1":
        candidate = descriptor["directions"][0]["candidate"]
        candidate["positive_recall_at_k"]["1"] = 0.2
        candidate["row_counts"]["top1_correct"] = 2
        candidate["row_counts"]["top1_false"] = 6
        pooled = descriptor["pooled"]["candidate"]
        pooled["positive_recall_at_k"]["1"] = 0.3
        pooled["row_counts"]["top1_correct"] = 6
        pooled["row_counts"]["top1_false"] = 10
    elif failure == "second_direction_r1":
        candidate = descriptor["directions"][1]["candidate"]
        candidate["positive_recall_at_k"]["1"] = 0.3
        candidate["row_counts"]["top1_correct"] = 3
        candidate["row_counts"]["top1_false"] = 5
        pooled = descriptor["pooled"]["candidate"]
        pooled["positive_recall_at_k"]["1"] = 0.3
        pooled["row_counts"]["top1_correct"] = 6
        pooled["row_counts"]["top1_false"] = 10
    elif failure == "pooled_r8":
        descriptor["pooled"]["candidate"]["positive_recall_at_k"]["8"] = 0.7
    elif failure == "track_r1":
        descriptor["pooled"]["candidate"]["positive_recall_at_k_by_anchor_kind"][
            "track_core"
        ]["1"] = 0.2
    else:
        descriptor["pooled"]["candidate"]["positive_recall_at_k_by_anchor_kind"][
            "gaussian_reserve"
        ]["1"] = 0.1
    _refresh_delta(payload)
    report = _write_report(tmp_path, payload)
    output = tmp_path / "gate.json"
    with pytest.raises(SystemExit) as error:
        main(_argv(paths, report, output))
    assert error.value.code == 2
    result = json.loads(output.read_text())
    assert result["decision"] == "STOP"
    assert result["mechanism_gate_passed"] is False
    assert not all(result["gates"].values())


def test_descriptor_arm_b_gate_rejects_mutated_source_artifact(tmp_path):
    paths = _artifacts(tmp_path)
    report = _write_report(tmp_path, _report(paths))
    argv = _argv(paths, report, tmp_path / "gate.json")
    paths["query_cache"].write_bytes(b"mutated after descriptor audit")
    with pytest.raises(ValueError, match="current artifact"):
        main(argv)


def test_descriptor_arm_b_gate_rejects_relabelled_lineage(tmp_path):
    paths = _artifacts(tmp_path)
    payload = _report(paths)
    payload["source_artifacts"]["query_cache"]["path"] = str(paths["teacher"])
    report = _write_report(tmp_path, payload)
    with pytest.raises(ValueError, match="query_cache report path differs"):
        main(_argv(paths, report, tmp_path / "gate.json"))


def test_descriptor_arm_b_gate_rejects_unidirectional_or_test_report(tmp_path):
    paths = _artifacts(tmp_path)
    payload = _report(paths)
    payload["descriptor_identity"]["directions"] = payload["descriptor_identity"][
        "directions"
    ][:1]
    report = _write_report(tmp_path, payload, "unidirectional")
    with pytest.raises(ValueError, match="not bidirectional"):
        main(_argv(paths, report, tmp_path / "gate.json"))

    payload = _report(paths)
    payload["uses_test_queries"] = True
    report = _write_report(tmp_path, payload, "test_tainted")
    with pytest.raises(ValueError, match="not mapping-only/test-free"):
        main(_argv(paths, report, tmp_path / "gate2.json"))
