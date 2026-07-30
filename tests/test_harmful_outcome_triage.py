import torch

from localization_training.harmful_outcome_triage import (
    HarmfulTriageConfig,
    _verified_teacher_candidates,
    project_depth_legal_candidates,
    triage_harmful_outcomes,
)


def test_project_depth_legal_candidates_checks_reprojection_and_depth():
    errors, legal = project_depth_legal_candidates(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        pose_w2c=torch.eye(4),
        K=torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]),
        keypoints=torch.tensor([[5.0, 5.0]]),
        rendered_depth=torch.tensor([2.0]),
        rendered_alpha=torch.tensor([1.0]),
        config=HarmfulTriageConfig(depth_abs_tolerance_m=0.1, depth_rel_tolerance=0.0),
        device=torch.device("cpu"),
    )
    assert torch.allclose(errors, torch.zeros_like(errors))
    assert legal.tolist() == [[True, False]]


def test_project_depth_legal_candidates_cuda_device_contract():
    if not torch.cuda.is_available():
        return
    errors, legal = project_depth_legal_candidates(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        pose_w2c=torch.eye(4),
        K=torch.eye(3),
        keypoints=torch.tensor([[0.0, 0.0]]),
        rendered_depth=torch.tensor([2.0]),
        rendered_alpha=torch.tensor([1.0]),
        config=HarmfulTriageConfig(),
        device=torch.device("cuda"),
    )
    assert errors.is_cuda
    assert legal.is_cuda
    candidates, invalid = _verified_teacher_candidates(
        query_row=3,
        top1=0,
        lookup={3: torch.tensor([0, 1])},
        errors=torch.tensor([3.0, 1.0], device="cuda"),
        legal=torch.tensor([True, True], device="cuda"),
        radius_px=2.0,
    )
    assert candidates.tolist() == [1]
    assert invalid == 0


def _teacher_record(rows, positives):
    values = []
    offsets = [0]
    for row in rows:
        values.extend(positives.get(int(row), []))
        offsets.append(len(values))
    return {
        "query_name": "seq0/frame0.png",
        "query_rows": torch.tensor(rows),
        "positive_offsets": torch.tensor(offsets),
        "positive_indices": torch.tensor(values, dtype=torch.long),
        "ambiguous_offsets": torch.zeros(len(rows) + 1, dtype=torch.long),
        "ambiguous_indices": torch.empty(0, dtype=torch.long),
    }


def test_triage_distinguishes_rank_teacher_coverage_and_unmatchable():
    rows = torch.tensor([0, 1, 2, 3])
    selected_record = {
        "query_name": "seq0/frame0.png",
        "query_rows": rows,
        "topk_anchor_indices": torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        ),
        "selected_row_mask": torch.ones(4, dtype=torch.bool),
    }
    dynamic_record = {
        "query_name": "seq0/frame0.png",
        "query_rows": rows,
        "harmful_inlier_mask": torch.ones(4, dtype=torch.bool),
    }
    cache = {
        "native_keypoints": torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [40.0, 0.0]]
        ),
        "native_input_hw": (2, 64),
        "native_depth": torch.full((2, 64), 2.0),
        "native_alpha": torch.ones((2, 64)),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "pixel_center_offset": 0.0,
    }
    active = {
        "anchor_xyz": torch.tensor(
            [[100.0, 0.0, 2.0], [0.0, 0.0, 2.0], [20.0, 0.0, 2.0]]
        ),
        "track_cluster_ids": torch.full((3,), -1),
    }
    canonical = {
        "anchor_xyz": torch.tensor(
            [[100.0, 0.0, 2.0], [40.0, 0.0, 2.0]]
        )
    }
    track_payload = {
        "query_names": ["seq0/frame0.png"],
        "query_bins": torch.tensor([0]),
        "tracks": {
            "track_index": torch.empty(0, dtype=torch.long),
            "query_index": torch.empty(0, dtype=torch.long),
            "keypoint_index": torch.empty(0, dtype=torch.long),
        },
        "track_geometry": {
            "triangulated": torch.empty(0, dtype=torch.bool),
            "triangulation_distinct_view_count": torch.empty(0, dtype=torch.long),
            "triangulation_distinct_view_bin_count": torch.empty(0, dtype=torch.long),
            "triangulation_reprojection_p90_px": torch.empty(0),
        },
        "assignment": {
            "track_landmark_index": torch.empty(0, dtype=torch.long)
        },
    }
    triage, completed = triage_harmful_outcomes(
        active_map=active,
        canonical_map=canonical,
        selected_outcomes={
            "query_names": ["seq0/frame0.png"],
            "anchor_count": 3,
            "records": [selected_record],
        },
        dynamic_outcomes={
            "query_names": ["seq0/frame0.png"],
            "records": [dynamic_record],
        },
        active_positive_teacher={
            "query_names": ["seq0/frame0.png"],
            "anchor_count": 3,
            "records": [_teacher_record(rows.tolist(), {0: [1]})],
        },
        query_cache={"seq0/frame0.png": cache},
        track_payload=track_payload,
        config=HarmfulTriageConfig(
            strict_radius_px=0.1,
            depth_abs_tolerance_m=0.1,
            depth_rel_tolerance=0.0,
        ),
        device=torch.device("cpu"),
    )
    assert triage["records"][0]["category"].tolist() == [0, 1, 2, 4]
    assert triage["records"][0]["surface_support_valid"].tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert completed["records"][0]["positive_indices"].tolist() == [1, 2]
    assert completed["diagnostics"]["positive_rows"] == 2
    assert completed["diagnostics"]["strong_pair_count"] == 2
