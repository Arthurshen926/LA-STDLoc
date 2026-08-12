import torch

from evidence.mapping_density_factor import (
    audit_density_cache_pair,
    audit_sparse_refresh_equivalence,
    compare_density_arms,
)


def _arm(scale=1.0, covariance=1.0, parallax=1.0):
    return {
        "raw_reciprocal_epipolar_edge_count": round(100 * scale),
        "accepted_edge_count": round(80 * scale),
        "track_count": round(50 * scale),
        "triangulated_track_count": round(40 * scale),
        "strict_track_count": round(20 * scale),
        "broad_track_count": round(30 * scale),
        "broad_covariance_trace_m2": {"median": covariance, "p90": covariance},
        "broad_parallax_deg": {"median": parallax},
    }


def test_density_gate_requires_more_stable_tracks_not_only_raw_edges():
    report = compare_density_arms(_arm(), _arm(scale=2.0, covariance=1.3))
    assert report["mechanism_gate_passed"] is False
    assert report["checks"]["broad_covariance_median_at_most_1p10x"] is False


def test_density_gate_passes_clear_quantity_gain_with_stable_quality():
    report = compare_density_arms(
        _arm(), _arm(scale=1.3, covariance=1.02, parallax=0.95)
    )
    assert report["mechanism_gate_passed"] is True


def test_density_cache_pair_proves_prefix_and_geometry_identity():
    def payload(keypoints):
        return {
            "signature_payload": {
                "version": 11,
                "source_path": "/data",
                "native_sparse_keypoint_count": keypoints,
                "native_sparse_nms_radius": 4,
            },
            "queries": {
                "frame.png": {
                    "native_keypoints": torch.arange(keypoints * 2).reshape(
                        keypoints, 2
                    ),
                    "native_descriptors": torch.ones(keypoints, 2),
                    "native_scores": torch.arange(keypoints),
                    "native_K": torch.eye(3),
                    "pose_w2c": torch.eye(4),
                    "native_depth": torch.ones(2, 2),
                    "native_alpha": torch.ones(2, 2),
                    "native_sparse_metadata": {
                        "detect_num": keypoints,
                        "nms_radius": 4,
                    },
                }
            },
        }

    control = payload(2)
    high = payload(4)
    high["queries"]["frame.png"]["native_keypoints"][:2] = control["queries"][
        "frame.png"
    ]["native_keypoints"]
    high["queries"]["frame.png"]["native_scores"][:2] = control["queries"]["frame.png"][
        "native_scores"
    ]
    report = audit_density_cache_pair(control, high)
    assert report["strict_single_factor_contract_passed"] is True


def test_sparse_refresh_equivalence_authorizes_exact_payload_reuse():
    record = {
        "native_keypoints": torch.ones(2, 2),
        "native_descriptors": torch.ones(2, 4),
        "native_scores": torch.ones(2),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_depth": torch.ones(2, 2),
        "native_alpha": torch.ones(2, 2),
        "native_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "native_input_hw": [2, 2],
    }
    refreshed_record = dict(record)
    refreshed_record["native_sparse_metadata"] = {
        "requested_keypoint_count": 2,
        "nms_radius": 4,
    }
    report = audit_sparse_refresh_equivalence(
        {"queries": {"frame.png": record}},
        {
            "signature_payload": {
                "native_sparse_keypoint_count": 2,
                "native_sparse_nms_radius": 4,
            },
            "queries": {"frame.png": refreshed_record},
        },
    )
    assert report["content_equivalent_track_payload_reuse_authorized"] is True


def test_sparse_refresh_equivalence_rejects_changed_effective_sparse_depth():
    record = {
        "native_keypoints": torch.tensor([[0.0, 0.0]]),
        "native_descriptors": torch.ones(1, 4),
        "native_scores": torch.ones(1),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_depth": torch.ones(2, 2),
        "native_depth_at_keypoints": torch.tensor([2.0]),
        "native_alpha": torch.ones(2, 2),
        "native_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "native_input_hw": [2, 2],
    }
    refreshed = dict(record)
    refreshed.pop("native_depth_at_keypoints")
    refreshed["native_sparse_metadata"] = {
        "requested_keypoint_count": 1,
        "nms_radius": 4,
    }
    report = audit_sparse_refresh_equivalence(
        {"queries": {"frame.png": record}},
        {
            "signature_payload": {
                "native_sparse_keypoint_count": 1,
                "native_sparse_nms_radius": 4,
            },
            "queries": {"frame.png": refreshed},
        },
    )
    assert report["effective_sparse_depth_exact_query_count"] == 0
    assert report["content_equivalent_track_payload_reuse_authorized"] is False
