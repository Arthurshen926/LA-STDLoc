import torch

from scripts.refresh_mapping_sparse_cache import (
    refreshed_signature_payload,
    update_sparse_record,
)


def test_refreshed_signature_changes_k_and_nms_explicitly():
    signature, payload = refreshed_signature_payload(
        {"version": 10, "source_path": "/data"},
        mapping_keypoints=2048,
        nms_radius=4,
    )
    assert len(signature) == 64
    assert payload["version"] == 11
    assert payload["native_sparse_keypoint_count"] == 2048
    assert payload["native_sparse_nms_radius"] == 4


def test_sparse_refresh_replaces_rows_and_records_contract():
    record = {"dense": torch.ones(1), "native_depth_at_keypoints": torch.ones(2)}
    sparse = {
        "keypoints": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        "descriptors": torch.ones(2, 4),
        "keypoint_scores": torch.tensor([0.9, 0.8]),
    }
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[2, 2] = False
    output = update_sparse_record(
        record,
        sparse=sparse,
        valid_mask=mask,
        mapping_keypoints=2048,
        nms_radius=4,
    )
    assert output["dense"].item() == 1
    assert output["native_keypoints"].shape[0] == 1
    assert "native_depth_at_keypoints" not in output
    metadata = output["native_sparse_metadata"]
    assert metadata["requested_keypoint_count"] == 2048
    assert metadata["nms_radius"] == 4
