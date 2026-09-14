from pathlib import Path

import pytest
import torch

from map_learning.v21_pose_leverage import (
    GAUSSIAN_GEOMETRY_SOURCE,
    REPROJECTION_UPPER_BOUND,
    TRACK_CONSENSUS_DIAGNOSTIC,
    summarize_pose_recovery,
)
from scripts.aggregate_v21_pose_recovery_oracle import (
    SHARD_SCHEMA,
    aggregate_payloads,
    summarize_aggregate,
)


def _record(query_index: int, *, success: bool, recovered: bool = False) -> dict:
    bundle = None
    one_assignment = None
    route = "protection_only" if success else "geometry_limited"
    if recovered:
        route = "descriptor_controllable"
        one_assignment = {"r5_success": True}
        bundle = {
            "exact_delta_r5": 1,
            "query_rows": torch.tensor([0, 2], dtype=torch.long),
        }
    return {
        "query_index": query_index,
        "image_name": f"seq/frame{query_index:05d}.png",
        "sequence_id": "seq",
        "block_id": f"seq:block:{query_index // 2}",
        "source_cache_index": query_index % 2,
        "source_record_index": query_index // 2,
        "baseline": {"r5_success": success},
        "route": route,
        "controller_authorized": False,
        "legal_positive_csr": {
            "positive_evidence_mode": REPROJECTION_UPPER_BOUND,
        },
        "one_assignment_lower_bound": one_assignment,
        "recovery_bundle": bundle,
        "exact_replay_count": query_index + 1,
    }


def _source_records() -> list[tuple[int, int, dict]]:
    return [
        (
            query_index % 2,
            query_index // 2,
            {
                "query_index": query_index,
                "image_name": f"seq/frame{query_index:05d}.png",
                "sequence_id": "seq",
                "block_id": f"seq:block:{query_index // 2}",
                "source_record_sha256": f"{query_index + 1:064x}",
            },
        )
        for query_index in range(4)
    ]


def _payload(shard_index: int, records: list[dict]) -> dict:
    payload = {
        "schema": SHARD_SCHEMA,
        "version": 1,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "role": "adaptation",
        "loo_used": False,
        "ground_truth_pose_is_feedback_authority": True,
        "topk_is_candidate_mining_only": True,
        "all_action_authority_is_exact_poselib": True,
        "unlabeled_rows_are_negative": False,
        "negative_anchor_label_count": 0,
        "gaussian_support_is_geometry_only": True,
        "correspondence_identity_authority_present": False,
        "controller_authorized_query_count_must_be_zero": True,
        "shard_index": shard_index,
        "shard_count": 2,
        "source_query_count": 4,
        "input": {
            "adaptation_caches": [
                {"path": "/tmp/cache0.pt", "sha256": "a" * 64},
                {"path": "/tmp/cache1.pt", "sha256": "b" * 64},
            ],
            "gaussian_support": [],
            "frozen_map": "/tmp/map.pt",
            "frozen_map_sha256": "c" * 64,
        },
        "parameters": {"seed": 2026, "positive_reprojection_px": 2.0},
        "records": records,
    }
    payload["summary"] = summarize_pose_recovery(records)
    return payload


def _entries() -> list[tuple[Path, str, dict]]:
    records = [
        _record(0, success=True),
        _record(1, success=False, recovered=True),
        _record(2, success=False),
        _record(3, success=True),
    ]
    return [
        (Path("/tmp/oracle0.pt"), "d" * 64, _payload(0, records[0::2])),
        (Path("/tmp/oracle1.pt"), "e" * 64, _payload(1, records[1::2])),
    ]


def _track_entries() -> list[tuple[Path, str, dict]]:
    entries = _entries()
    for _, _, payload in entries:
        payload.pop("all_action_authority_is_exact_poselib")
        payload.update(
            {
                "positive_source": TRACK_CONSENSUS_DIAGNOSTIC,
                "all_pose_recovery_claims_use_exact_poselib": True,
                "exact_poselib_is_controller_action_authority": False,
                "track_consensus_identity_evidence_present": True,
                "track_consensus_identity_evidence_is_deployment_authority": False,
                "track_consensus_diagnostic_is_action_authority": False,
            }
        )
        payload["input"]["correspondence_truth"] = {
            "path": "/tmp/truth.pt",
            "sha256": "f" * 64,
        }
        for record in payload["records"]:
            record["positive_source"] = TRACK_CONSENSUS_DIAGNOSTIC
            record["legal_positive_csr"].update(
                {
                    "positive_evidence_mode": TRACK_CONSENSUS_DIAGNOSTIC,
                    "positive_source": TRACK_CONSENSUS_DIAGNOSTIC,
                    "source_decisive_row_count": (
                        3 if record["query_index"] != 2 else 0
                    ),
                    "legal_positive_row_count": (
                        3 if record["query_index"] != 2 else 0
                    ),
                    "legal_positive_edge_count": (
                        4 if record["query_index"] != 2 else 0
                    ),
                }
            )
            if isinstance(record.get("one_assignment_lower_bound"), dict):
                record["one_assignment_lower_bound"]["positive_source"] = (
                    TRACK_CONSENSUS_DIAGNOSTIC
                )
        payload["summary"] = summarize_pose_recovery(
            payload["records"], positive_source=TRACK_CONSENSUS_DIAGNOSTIC
        )
    return entries


def test_complete_aggregate_preserves_all_queries_and_upper_bound_semantics() -> None:
    records, summary = aggregate_payloads(_entries(), _source_records())
    assert [record["query_index"] for record in records] == [0, 1, 2, 3]
    assert summary == summarize_aggregate(records)
    assert summary["baseline_r5_success_count"] == 2
    assert summary["baseline_failure_count"] == 2
    assert summary["one_assignment_recovered_failure_count"] == 1
    assert summary["one_assignment_recovery_upper_bound_r5_success_count"] == 3
    assert summary["recovery_bundle_sizes"] == [2]
    assert summary["exact_replay_count_total"] == 10
    assert summary["controller_authorized_query_count"] == 0
    assert summary["geometry_recovery_is_upper_bound_only"] is True
    assert summary["deployment_authorized"] is False
    assert summary["sequence_distribution"]["seq"]["query_count"] == 4
    assert len(summary["block_distribution"]) == 2
    assert summary["positive_source"] == GAUSSIAN_GEOMETRY_SOURCE


def test_missing_oracle_shard_is_rejected() -> None:
    with pytest.raises(ValueError, match="full registry"):
        aggregate_payloads(_entries()[:1], _source_records())


def test_omitted_failure_record_is_rejected() -> None:
    entries = _entries()
    entries[1][2]["records"] = entries[1][2]["records"][:1]
    entries[1][2]["summary"] = summarize_pose_recovery(entries[1][2]["records"])
    with pytest.raises(ValueError, match="omitted or added"):
        aggregate_payloads(entries, _source_records())


def test_parameter_or_input_lineage_mismatch_is_rejected() -> None:
    entries = _entries()
    entries[1][2]["parameters"] = {"seed": 7, "positive_reprojection_px": 2.0}
    with pytest.raises(ValueError, match="contracts differ"):
        aggregate_payloads(entries, _source_records())


def test_wrong_modulo_or_frontend_coordinate_is_rejected() -> None:
    entries = _entries()
    entries[0][2]["records"][0]["source_cache_index"] = 1
    with pytest.raises(ValueError, match="lineage differs"):
        aggregate_payloads(entries, _source_records())


def test_any_controller_authorization_is_rejected() -> None:
    entries = _entries()
    entries[0][2]["records"][0]["controller_authorized"] = True
    with pytest.raises(ValueError, match="record is invalid"):
        aggregate_payloads(entries, _source_records())


def test_track_consensus_aggregate_reports_diagnostic_coverage_and_recovery() -> None:
    records, summary = aggregate_payloads(_track_entries(), _source_records())
    assert len(records) == 4
    assert summary["positive_source"] == TRACK_CONSENSUS_DIAGNOSTIC
    assert summary["track_consensus_diagnostic_available"] is True
    assert summary["track_consensus_diagnostic_truth_available_query_count"] == 3
    assert summary["track_consensus_diagnostic_truth_query_coverage"] == 0.75
    assert summary["track_consensus_diagnostic_truth_available_failure_count"] == 1
    assert summary["track_consensus_diagnostic_positive_row_count_total"] == 9
    assert summary["track_consensus_diagnostic_positive_edge_count_total"] == 12
    assert summary["track_consensus_one_assignment_recovered_failure_count"] == 1
    assert summary["track_consensus_exact_bundle_recovered_failure_count"] == 1
    assert summary["controller_authorized_query_count"] == 0
    assert summary["deployment_authorized"] is False
    assert "geometry_recovery_is_upper_bound_only" not in summary


def test_geometry_and_track_sources_cannot_be_mixed() -> None:
    geometry = _entries()[0]
    track = _track_entries()[1]
    with pytest.raises(ValueError, match="mix positive sources/contracts"):
        aggregate_payloads([geometry, track], _source_records())


def test_track_shards_require_identical_correspondence_truth_identity() -> None:
    entries = _track_entries()
    entries[1][2]["input"]["correspondence_truth"] = {
        "path": "/tmp/other_truth.pt",
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="contracts differ"):
        aggregate_payloads(entries, _source_records())
