import torch

from evidence.track_pair_audit import audit_track_pair_graph


def _pose(center):
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, 3] = -torch.as_tensor(center, dtype=torch.float64)
    return pose


def test_pair_graph_audit_separates_adjacent_repeats_from_geometric_pairs():
    names = [f"seq-00/frame-{index:06d}.color.png" for index in range(4)]
    centers = [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0], [0.8, 0.0, 0.0]]
    K = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    cache = {
        "queries": {
            name: {
                "pose_w2c": _pose(center),
                "native_K": K,
                "native_input_hw": [100, 100],
            }
            for name, center in zip(names, centers)
        }
    }
    payload = {
        "query_names": names,
        "tracks": {
            "track_index": torch.tensor([0, 0, 0, 1, 1, 1]),
            "query_index": torch.tensor([0, 1, 3, 0, 2, 3]),
            "keypoint_index": torch.zeros(6, dtype=torch.long),
            "confidence": torch.ones(6),
        },
        "track_geometry": {
            "triangulated": torch.tensor([True, True]),
            "triangulated_xyz": torch.tensor(
                [[0.0, 0.0, 4.0], [0.2, 0.0, 4.0]]
            ),
        },
        "diagnostics": {
            "track_camera_pair_candidate_count": 6,
            "track_camera_pair_matched_count": 4,
        },
    }
    audit = audit_track_pair_graph(
        payload,
        cache,
        pair_neighbors=3,
        minimum_baseline_m=0.0,
        maximum_baseline_m=2.0,
        minimum_effective_parallax_deg=1.0,
        maximum_visibility_points=16,
    )
    report = audit["report"]
    assert report["candidate_graph_exact_count_reconstructed"] is True
    assert report["temporal_adjacent_pair_count"] == 3
    assert report["short_baseline_near_repeat_proxy_count"] >= 1
    assert report["effective_geometry_proxy_count"] >= 1
    assert audit["uses_test_queries"] is False
    assert audit["pair_selection_mutated"] is False
    assert audit["deployment_mutated"] is False
    assert (
        report["provenance_contract"][
            "exact_short_baseline_duplicate_texture_edge_fraction"
        ]
        is None
    )


def test_pair_graph_audit_rejects_missing_mapping_camera():
    payload = {
        "query_names": ["seq/frame-000001.png"],
        "tracks": {
            "track_index": torch.zeros(0, dtype=torch.long),
            "query_index": torch.zeros(0, dtype=torch.long),
        },
        "track_geometry": {
            "triangulated": torch.zeros(0, dtype=torch.bool),
            "triangulated_xyz": torch.zeros((0, 3)),
        },
    }
    try:
        audit_track_pair_graph(payload, {"queries": {}})
    except ValueError as error:
        assert "lacks mapping camera" in str(error)
    else:
        raise AssertionError("missing mapping camera must fail closed")
