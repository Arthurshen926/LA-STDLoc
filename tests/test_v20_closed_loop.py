from argparse import Namespace
import hashlib
import json

import pytest
import torch

from common.hashing import sha256_file
from map_learning.metric import SharedLowRankMetric
from scripts.aggregate_v20_sparse_map import aggregate
from scripts.finalize_v20_closed_loop import finalize


def _pose(task: float, translation: float) -> dict:
    return {
        "task_error": task,
        "translation_error_cm": translation,
        "rotation_error_deg": 1.0,
    }


def _source_sha(namespace: str, query: int) -> str:
    return hashlib.sha256(f"{namespace}:{query}".encode()).hexdigest()


def _identity_metric_payload(map_path, *, protocol: str, arm=None) -> dict:
    map_payload = torch.load(map_path, map_location="cpu", weights_only=False)
    descriptor_dim = int(torch.as_tensor(map_payload["anchor_features"]).shape[1])
    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    payload = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "protocol": protocol,
        "step": 0,
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
        "landmark_indices": torch.as_tensor(map_payload["anchor_ids"])
        .long()
        .clone(),
        "map_path": str(map_path.resolve()),
        "map_sha256": sha256_file(map_path),
    }
    if arm is not None:
        payload.update(
            {
                "deployment_arm": arm,
                "loo_used": False,
                "feedback_descriptors_copied_into_map": False,
                "photometric_canonicalization_contract": map_payload.get(
                    "photometric_canonicalization_contract"
                ),
            }
        )
    return payload


def _shard(
    path,
    candidate_map,
    *,
    authorized: bool,
    role: str = "feedback_query",
    query_offset: int = 0,
    family_offset: int = 10,
    baseline_map_sha256: str = "b" * 64,
    source_namespace: str = "control",
) -> list[str]:
    arm = "sparse_positive_only_angle_5"
    source_paths = [
        path.with_name(f"{path.stem}_source_{offset}.pt") for offset in range(2)
    ]
    for offset, source_path in enumerate(source_paths):
        torch.save(
            {
                "schema": "lafgs_v7_certified_clean_render",
                "version": 1,
                "view_role": role,
                "query_index": query_offset + offset,
                "pose_family_id": family_offset + 10 * offset,
                "source_namespace": source_namespace,
                "certificate": {"decision": "ACCEPT"},
            },
            source_path,
        )
    source_sha256s = [sha256_file(source_path) for source_path in source_paths]
    certified_path = path.with_name(f"{path.stem}_certified.json")
    certified_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_v14_observer_split_certified_view",
                "version": 1,
                "view_role": role,
                "uses_test_queries": False,
                "map_mutation_count": 0,
                "decision_counts": {"ACCEPT": 2, "UNCERTAIN": 0, "REJECT": 0},
                "records": [
                    {
                        "query_index": query_offset + offset,
                        "decision": "ACCEPT",
                        "path": str(source_paths[offset].resolve()),
                        "sha256": source_sha256s[offset],
                    }
                    for offset in range(2)
                ],
            }
        )
    )
    payload = {
        "schema": "lafgs_v20_sparse_map_replay_shard",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "plant_row_policy": "all_detected_rows",
        "evaluation_role": role,
        "candidate_arm": arm,
        "strong_feedback_authorized": authorized,
        "analysis_only": not authorized,
        "descriptor_training_safe": True,
        "shard_index": 0,
        "shard_count": 1,
        "source_query_count": 2,
        "accepted_query_count": 2,
        "input": {
            "candidate_map_sha256": sha256_file(candidate_map),
            "baseline_map_sha256": baseline_map_sha256,
            "certified_batch": str(certified_path),
            "certified_batch_sha256": sha256_file(certified_path),
        },
        "records": [
            {
                "query_index": query_offset,
                "pose_family_id": family_offset,
                "source_record_sha256": source_sha256s[0],
                "baseline": _pose(0.5, 10.0),
                arm: _pose(0.1, 1.0),
            },
            {
                "query_index": query_offset + 1,
                "pose_family_id": family_offset + 10,
                "source_record_sha256": source_sha256s[1],
                "baseline": _pose(0.6, 12.0),
                arm: _pose(0.1, 1.0),
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return source_sha256s


def test_v20_control_cannot_advance_unauthorized_teacher(tmp_path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    shard = tmp_path / "shard.json"
    _shard(shard, candidate, authorized=False)
    result = aggregate(
        Namespace(
            shards=[shard],
            phase="control",
            selected_arm=None,
            output=tmp_path / "control.json",
        )
    )
    assert result["decision_report"]["classification"] == "DEFAULT_CANDIDATE"
    assert result["selected_arm"] is None
    assert result["decision"] == "NO_ACTION"


def test_v20_aggregate_rejects_missing_certified_accept_query(tmp_path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    shard = tmp_path / "shard.json"
    _shard(shard, candidate, authorized=False)
    payload = json.loads(shard.read_text())
    payload["records"] = payload["records"][:1]
    payload["accepted_query_count"] = 1
    shard.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exactly cover certified ACCEPT"):
        aggregate(
            Namespace(
                shards=[shard],
                phase="control",
                selected_arm=None,
                output=tmp_path / "control.json",
            )
        )


def test_v20_aggregate_rejects_unbound_pose_family(tmp_path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    shard = tmp_path / "shard.json"
    _shard(shard, candidate, authorized=False)
    payload = json.loads(shard.read_text())
    payload["records"][0]["pose_family_id"] = 999_999
    shard.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exactly cover certified ACCEPT"):
        aggregate(
            Namespace(
                shards=[shard],
                phase="control",
                selected_arm=None,
                output=tmp_path / "control.json",
            )
        )


def test_v20_confirmed_action_deploys_exact_sparse_map(tmp_path) -> None:
    arm = "sparse_positive_only_angle_5"
    # Non-contiguous IDs catch accidental row-index registries in both the
    # candidate and rollback identity metrics.
    anchor_ids = torch.tensor([1761, 2142])
    anchor_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    stable_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    stable_map = tmp_path / "stable.pt"
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_ids": anchor_ids,
            "anchor_xyz": anchor_xyz,
            "anchor_features": stable_features,
        },
        stable_map,
    )
    stable_map_sha = sha256_file(stable_map)
    design_sources = sorted(
        [_source_sha("design", 200), _source_sha("design", 201)]
    )
    evidence = tmp_path / "evidence.pt"
    torch.save(
        {
            "schema": "lafgs_v20_topk_competition_evidence",
            "version": 2,
            "uses_test_queries": False,
            "loo_used": False,
            "strong_feedback_authorized": True,
            "design_source_record_sha256s": design_sources,
            "inputs": {
                "anchor_map_sha256": stable_map_sha,
                "design_batch_sha256": "d" * 64,
            },
            "protection_query_descriptors": torch.tensor([[1.0, 0.0]]),
            "protection_positive_offsets": torch.tensor([0, 1]),
            "protection_positive_anchor_rows": torch.tensor([0]),
            "protection_negative_offsets": torch.tensor([0, 1]),
            "protection_negative_anchor_rows": torch.tensor([1]),
        },
        evidence,
    )
    audit = {
        "schema": "lafgs_v20_materialized_sparse_action_audit",
        "version": 1,
        "passed": True,
    }
    candidate_features = torch.tensor([[0.99, 0.01], [0.0, 1.0]])
    selected_angle = float(
        torch.rad2deg(
            torch.acos(
                torch.nn.functional.cosine_similarity(
                    stable_features[0].reshape(1, -1),
                    candidate_features[0].reshape(1, -1),
                ).clamp(-1.0, 1.0)
            )
        )[0]
    )
    candidate = tmp_path / "candidate.pt"
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_ids": anchor_ids,
            "anchor_xyz": anchor_xyz,
            "anchor_features": candidate_features,
            "v20_sparse_descriptor_action": {
                "schema": "lafgs_v20_sparse_descriptor_action",
                "arm": arm,
                "mode": "positive_only",
                "training_status": "REQUIRES_EXACT_POSE_CONTROL",
                "query_descriptor_action": "native_unchanged",
                "selected_anchor_rows": torch.tensor([0]),
                "maximum_angle_deg": 5.0,
                "per_anchor_observed_angle_deg": torch.tensor([selected_angle]),
                "positive_objective": "per_positive_listwise_mean",
                "strong_feedback_authorized": True,
                "clean_protection_passed": True,
                "materialized_action_audit": audit,
                "positive_win_nonregression_passed": True,
                "post_training_action_scale": 1.0,
                "global_seed_action_scale": 1.0,
                "clean_margin_slack": 0.002,
                "per_anchor_action_scales": torch.tensor([1.0]),
                "maximum_applied_anchor_scale": 1.0,
                "baseline_map_sha256": stable_map_sha,
                "evidence": str(evidence),
                "evidence_sha256": sha256_file(evidence),
                "design_batch_sha256": "d" * 64,
                "design_query_indices": [200, 201],
                "design_source_record_sha256s": design_sources,
                "design_pose_family_ids": [1, 2],
            },
        },
        candidate,
    )
    metric = tmp_path / "candidate_metric.pt"
    torch.save(
        _identity_metric_payload(
            candidate,
            protocol="v20_sparse_anchor_native_query_identity",
            arm=arm,
        ),
        metric,
    )
    control_shard = tmp_path / "control_shard.json"
    control_sources = _shard(
        control_shard,
        candidate,
        authorized=True,
        baseline_map_sha256=stable_map_sha,
    )
    control_path = tmp_path / "control.json"
    control = aggregate(
        Namespace(
            shards=[control_shard],
            phase="control",
            selected_arm=None,
            output=control_path,
        )
    )
    assert control["decision"] == "ADVANCE_TO_CONFIRMATION"
    confirmation_shard = tmp_path / "confirmation_shard.json"
    confirmation_sources = _shard(
        confirmation_shard,
        candidate,
        authorized=True,
        role="confirmation_query",
        query_offset=100,
        family_offset=30,
        baseline_map_sha256=stable_map_sha,
        source_namespace="confirmation",
    )
    confirmation_path = tmp_path / "confirmation.json"
    confirmation = aggregate(
        Namespace(
            shards=[confirmation_shard],
            phase="confirmation",
            selected_arm=arm,
            output=confirmation_path,
        )
    )
    assert confirmation["decision"] == "DEFAULT_CANDIDATE"
    assert set(control_sources).isdisjoint(confirmation_sources)
    report_path = tmp_path / "action.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "lafgs_v20_sparse_anchor_descriptor_action_report",
                "uses_test_queries": False,
                "loo_used": False,
                "arm": arm,
                "status": "REQUIRES_EXACT_POSE_CONTROL",
                "training": {
                    "strong_feedback_authorized": True,
                    "mode": "positive_only",
                    "positive_objective": "per_positive_listwise_mean",
                    "clean_protection_passed": True,
                    "materialized_action_audit": audit,
                    "positive_win_nonregression_passed": True,
                    "protection_row_count": 2,
                    "post_training_action_scale": 1.0,
                    "global_seed_action_scale": 1.0,
                    "clean_margin_slack": 0.002,
                    "selected_anchor_rows": [0],
                    "maximum_angle_deg": 5.0,
                    "per_anchor_observed_angle_deg": [selected_angle],
                    "per_anchor_action_scales": [1.0],
                    "maximum_applied_anchor_scale": 1.0,
                },
                "design_split": {
                    "batch_sha256": "d" * 64,
                    "query_indices": [200, 201],
                    "source_record_sha256s": design_sources,
                    "pose_family_ids": [1, 2],
                },
                "outputs": {
                    "candidate_map": str(candidate),
                    "candidate_map_sha256": sha256_file(candidate),
                    "identity_metric": str(metric),
                    "identity_metric_sha256": sha256_file(metric),
                },
                "inputs": {
                    "baseline_map_sha256": stable_map_sha,
                    "evidence": str(evidence),
                    "evidence_sha256": sha256_file(evidence),
                },
            }
        )
    )
    stable_metric = tmp_path / "stable_metric.pt"
    torch.save(
        _identity_metric_payload(
            stable_map, protocol="v6_identity_shared_metric"
        ),
        stable_metric,
    )
    result = finalize(
        Namespace(
            action_report=report_path,
            control_decision=control_path,
            confirmation_decision=confirmation_path,
            stable_map=stable_map,
            stable_metric=stable_metric,
            output=tmp_path / "deployment.json",
        )
    )
    assert result["formal_deployment_authorized"] is True
    assert result["decision"] == "DEPLOY_CANDIDATE"
    assert result["deployment"]["map"] == str(candidate.resolve())
    assert result["deployment"]["map_mutation"] == "sparse_anchor_descriptors"

    overlapping = json.loads(confirmation_path.read_text())
    overlapping["evaluation_pose_family_ids"] = [10]
    overlapping_path = tmp_path / "overlapping_confirmation.json"
    overlapping_path.write_text(json.dumps(overlapping))
    with pytest.raises(ValueError, match="splits are not disjoint"):
        finalize(
            Namespace(
                action_report=report_path,
                control_decision=control_path,
                confirmation_decision=overlapping_path,
                stable_map=stable_map,
                stable_metric=stable_metric,
                output=tmp_path / "rejected_overlap.json",
            )
        )

    overlapping_sources = json.loads(confirmation_path.read_text())
    overlapping_sources["evaluation_source_record_sha256s"] = [
        control_sources[0],
        confirmation_sources[1],
    ]
    overlapping_source_path = tmp_path / "overlapping_source_confirmation.json"
    overlapping_source_path.write_text(json.dumps(overlapping_sources))
    with pytest.raises(ValueError, match="splits are not disjoint"):
        finalize(
            Namespace(
                action_report=report_path,
                control_decision=control_path,
                confirmation_decision=overlapping_source_path,
                stable_map=stable_map,
                stable_metric=stable_metric,
                output=tmp_path / "rejected_source_overlap.json",
            )
        )

    wrong_stable_map = tmp_path / "wrong_stable.pt"
    wrong_payload = torch.load(stable_map, map_location="cpu", weights_only=False)
    wrong_payload["anchor_features"] = wrong_payload["anchor_features"].clone()
    wrong_payload["anchor_features"][0, 0] = 0.5
    torch.save(wrong_payload, wrong_stable_map)
    with pytest.raises(ValueError, match="frozen candidate artifacts"):
        finalize(
            Namespace(
                action_report=report_path,
                control_decision=control_path,
                confirmation_decision=confirmation_path,
                stable_map=wrong_stable_map,
                stable_metric=stable_metric,
                output=tmp_path / "rejected_wrong_stable.json",
            )
        )

    malicious_metric = tmp_path / "malicious_metric.pt"
    malicious_payload = torch.load(
        metric, map_location="cpu", weights_only=False
    )
    malicious_payload["metric_state_dict"]["up.weight"][0, 0] = 0.25
    torch.save(malicious_payload, malicious_metric)
    malicious_report = json.loads(report_path.read_text())
    malicious_report["outputs"]["identity_metric"] = str(malicious_metric)
    malicious_report["outputs"]["identity_metric_sha256"] = sha256_file(
        malicious_metric
    )
    malicious_report_path = tmp_path / "malicious_action.json"
    malicious_report_path.write_text(json.dumps(malicious_report))
    with pytest.raises(ValueError, match="learned descriptor transform"):
        finalize(
            Namespace(
                action_report=malicious_report_path,
                control_decision=control_path,
                confirmation_decision=confirmation_path,
                stable_map=stable_map,
                stable_metric=stable_metric,
                output=tmp_path / "rejected_metric.json",
            )
        )
