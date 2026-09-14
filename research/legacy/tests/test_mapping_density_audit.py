from __future__ import annotations

from pathlib import Path

import torch

from evidence.mapping_density_audit import (
    audit_mapping_density,
    explain_area_density_resolution,
)


DEPLOYMENT = {
    "keypoints": 2048,
    "keypoint_reference_area_px": 1920 * 1080,
    "keypoint_minimum": 1024,
    "keypoint_maximum": 2048,
}


def _cache(path: Path, *, request: int, nms: int | None, masked: bool) -> Path:
    before = request
    after = request - 3 if masked else request
    sparse_metadata = {
        "detect_num": request,
        "keypoint_count_before_mask": before,
        "keypoint_count_after_mask": after,
    }
    if nms is not None:
        sparse_metadata["nms_radius"] = nms
    signature_payload = {
        "version": 11,
        "native_sparse_keypoint_count": request,
    }
    if nms is not None:
        signature_payload["native_sparse_nms_radius"] = nms
    record = {
        "native_keypoints": torch.zeros(after, 2),
        "native_input_hw": [480, 640],
        "native_valid_mask": torch.ones(480, 640, dtype=torch.bool),
        "native_sparse_metadata": sparse_metadata,
    }
    torch.save(
        {
            "version": 11,
            "signature": "cache-signature",
            "signature_payload": signature_payload,
            "queries": {"mapping/frame.png": record},
        },
        path,
    )
    return path


def test_area_resolution_reports_minimum_clamp():
    report = explain_area_density_resolution(DEPLOYMENT, [640 * 480] * 5)
    assert report["area_scaled_unclamped_keypoints"] == 303
    assert report["resolved_keypoints"] == 1024
    assert report["clamp_reason"] == "minimum_clamp"


def test_low_density_cache_blocks_paired_factor_and_separates_axes(tmp_path):
    cache = _cache(tmp_path / "cache.pt", request=1024, nms=4, masked=False)
    report = audit_mapping_density(
        scene="heads",
        query_cache_path=cache,
        deployment=DEPLOYMENT,
    )
    assert report["protocol"]["mapping_target_satisfied"] is False
    assert report["protocol"]["nms_contract_status"] == "pass"
    assert report["mapping_cache"]["native_tensor_rows"]["median"] == 1024
    factor = report["factor_manifest"]
    assert factor["ready_for_paired_deployment_factor"] is False
    assert [variant["k_deployment"] for variant in factor["variants"]] == [1024, 2048]
    assert {variant["k_mapping"] for variant in factor["variants"]} == {2048}


def test_high_density_attested_cache_freezes_one_graph_for_both_variants(tmp_path):
    cache = _cache(tmp_path / "cache.pt", request=2048, nms=4, masked=True)
    graph = tmp_path / "function_graph.pt"
    graph.write_bytes(b"one immutable mapping graph")
    report = audit_mapping_density(
        scene="stairs",
        query_cache_path=cache,
        mapping_graph_path=graph,
        deployment=DEPLOYMENT,
    )
    factor = report["factor_manifest"]
    assert factor["ready_for_paired_deployment_factor"] is True
    identities = [variant["mapping_graph_identity"] for variant in factor["variants"]]
    assert identities[0] == identities[1]
    assert identities[0]["mapping_graph_artifact"]["sha256"]
    assert report["mapping_cache"]["queries_with_masked_keypoint_drop_rate"] == 1.0


def test_legacy_cache_without_nms_metadata_is_unattested(tmp_path):
    cache = _cache(tmp_path / "legacy.pt", request=2048, nms=None, masked=False)
    report = audit_mapping_density(
        scene="legacy",
        query_cache_path=cache,
        deployment=DEPLOYMENT,
    )
    assert report["protocol"]["nms_contract_status"] == "unattested"
    assert any(
        gap["kind"] == "nms_contract_not_attested"
        for gap in report["mechanism_gaps"]
    )
