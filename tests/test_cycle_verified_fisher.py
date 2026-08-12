import copy

import pytest
import torch

from evidence.cycle_verified_fisher import (
    POLICY_NAME,
    PROBE_SCHEMA,
    bounded_union_candidate_pool,
    materialize_pair_match_probe,
    pair_matches_from_probe,
    probe_track_build_inputs,
    select_cycle_verified_fisher_pairs,
    validate_cycle_verified_fisher_selection,
    validate_pair_match_probe,
)


def _look_at_pose(center, target):
    center = torch.as_tensor(center, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    forward = torch.nn.functional.normalize(target - center, dim=0)
    right = torch.nn.functional.normalize(
        torch.cross(
            forward,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            dim=0,
        ),
        dim=0,
    )
    down = torch.cross(forward, right, dim=0)
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = torch.stack((right, down, forward))
    pose[:3, 3] = -(pose[:3, :3] @ center)
    return pose


def _project(points, K, pose):
    camera = torch.einsum("ij,pj->pi", pose[:3, :3], points) + pose[:3, 3]
    pixel = torch.einsum("ij,pj->pi", K, camera)
    return pixel[:, :2] / pixel[:, 2:]


def _matcher_parameters():
    return {
        "minimum_similarity": 0.65,
        "minimum_margin": 0.01,
        "maximum_epipolar_error_px": 2.0,
        "epipolar_candidate_topk": 1,
        "epipolar_recovered_minimum_similarity": -1.0,
        "epipolar_recovered_minimum_margin": -1.0,
    }


def _diagnostics(pair, match, keypoint_count=3):
    count = len(match[0])
    return {
        "source_keypoint_count": keypoint_count,
        "target_keypoint_count": keypoint_count,
        "raw_top1_reciprocal_count": count,
        "descriptor_accepted_before_epipolar_count": count,
        "epipolar_accepted_top1_count": count,
        "epipolar_rejected_after_descriptor_count": 0,
        "ambiguity_rejected_count": 0,
        "final_reciprocal_epipolar_count": count,
        "epipolar_recovered_final_count": 0,
    }


def _synthetic_probe_and_geometry(*, translation_scale=1.0):
    # 0/1/2 form an exact descriptor triangle.  Cameras 3/4 are connected by
    # edges to camera 2, while their direct (large-baseline) edge has no match.
    # A fixed budget of five must spend its final non-backbone edge completing
    # the verified triangle rather than selecting the unverified 3/4 edge.
    centers = torch.tensor(
        [
            [-0.8, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.8, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ) * float(translation_scale)
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64) * float(
        translation_scale
    )
    poses = torch.stack([_look_at_pose(center, target) for center in centers])
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(5, 1, 1)
    points = torch.tensor(
        [[-0.2, 0.0, 4.0], [0.0, 0.15, 4.2], [0.2, -0.1, 3.8]],
        dtype=torch.float64,
    ) * float(translation_scale)
    keypoints = [_project(points, K[index], poses[index]) for index in range(5)]
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    identity = torch.arange(points.shape[0], dtype=torch.long)
    pair_matches = {
        pair: (identity.clone(), identity.clone(), torch.ones(points.shape[0]))
        for pair in pairs
    }
    pair_matches[(3, 4)] = (
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0),
    )
    probe = materialize_pair_match_probe(
        candidate_pairs=pairs,
        pair_matches=pair_matches,
        pair_diagnostics={
            pair: _diagnostics(pair, match)
            for pair, match in pair_matches.items()
        },
        keypoint_counts=[len(value) for value in keypoints],
        query_names_sha256="a" * 64,
        query_cache_sha256="b" * 64,
        mapping_keypoint_count=1024,
        mapping_nms_radius=4,
        candidate_pool_construction="synthetic_bounded_pose_pool_v1",
        candidate_pool_parameters={"per_camera": 4},
        matcher_parameters=_matcher_parameters(),
        detector_scores_applied=True,
    )
    return probe, keypoints, K, poses


def test_probe_contract_is_hash_bound_and_rejects_aggregate_sidecar():
    probe, _, _, _ = _synthetic_probe_and_geometry()
    validate_pair_match_probe(
        probe,
        expected_query_names_sha256="a" * 64,
        expected_query_cache_sha256="b" * 64,
        expected_mapping_keypoint_count=1024,
        expected_mapping_nms_radius=4,
        expected_content_sha256=probe["content_sha256"],
    )
    assert probe["schema"] == PROBE_SCHEMA

    tampered = copy.deepcopy(probe)
    tampered["matches"]["confidence"][0] += 0.01
    with pytest.raises(ValueError, match="content SHA-256"):
        validate_pair_match_probe(tampered)

    # The current post-Track sidecar only carries aggregate counts.  It cannot
    # prove which keypoint closes which exact descriptor triangle.
    with pytest.raises(ValueError, match="schema"):
        validate_pair_match_probe(
            {
                "schema": "lafgs_mapping_track_pair_sidecar",
                "pair": {"cycle_supported_edge_count": torch.tensor([3])},
            }
        )


def test_selected_matches_are_an_exact_reusable_probe_subset():
    probe, keypoints, K, poses = _synthetic_probe_and_geometry()
    selected, selection = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_budget=5,
    )
    matches, diagnostics = pair_matches_from_probe(
        probe, selected_pairs=selected
    )
    assert list(matches) == selected
    assert list(diagnostics) == selected
    for pair in selected:
        expected, _ = pair_matches_from_probe(probe, selected_pairs=[pair])
        for actual_column, expected_column in zip(matches[pair], expected[pair]):
            torch.testing.assert_close(actual_column, expected_column)

    tampered_selection = copy.deepcopy(selection)
    tampered_selection["selected_pair"]["right_query_index"][0] = 4
    with pytest.raises(ValueError, match="content SHA-256"):
        validate_cycle_verified_fisher_selection(
            tampered_selection, pair_match_probe=probe
        )


def test_track_builder_reuses_probe_without_calling_matcher(monkeypatch):
    from evidence import triangulation

    probe, keypoints, K, poses = _synthetic_probe_and_geometry()
    selected, selection = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
    )

    def forbidden_matcher(*args, **kwargs):
        raise AssertionError("selected pairs must reuse the hash-bound probe")

    monkeypatch.setattr(triangulation, "reciprocal_epipolar_matches", forbidden_matcher)
    tracks, diagnostics, sidecar = triangulation.build_cycle_consistent_tracks(
        descriptors=[torch.eye(3) for _ in range(5)],
        keypoints=keypoints,
        detector_scores=[torch.ones(3) for _ in range(5)],
        camera_K=K,
        pose_w2c=poses,
        pair_policy=POLICY_NAME,
        pair_budget=5,
        minimum_track_views=3,
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        device="cpu",
        **probe_track_build_inputs(probe, selection),
    )
    assert diagnostics["track_pair_matches_reused"] == 1
    assert diagnostics["track_camera_pair_candidate_count"] == len(selected)
    assert diagnostics["track_count"] == 3
    assert tracks["query_index"].unique().numel() == 5
    assert sidecar["policy"]["uses_precomputed_pair_matches"] is True


def test_cycle_verified_fisher_has_exact_budget_closure_and_connectivity():
    probe, keypoints, K, poses = _synthetic_probe_and_geometry()
    selected, sidecar = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_budget=5,
        minimum_camera_degree=1,
        maximum_cycle_reprojection_error_px=0.01,
    )
    assert selected == [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    assert (3, 4) not in selected
    assert len(selected) == 5
    assert sidecar["policy"] == POLICY_NAME
    assert sidecar["exact_pair_budget"] == 5
    assert sidecar["graph"] == {
        "component_count": 1,
        "isolated_camera_count": 0,
        "minimum_degree": 1,
        "maximum_degree": 4,
    }
    assert sidecar["candidate_graph"]["component_count"] == 1
    assert sidecar["verified_triangle"]["candidate_count"] == 3
    assert sidecar["verified_triangle"]["selected_completed_count"] == 3
    assert sidecar["verified_triangle"]["selected_camera_count"] == 3
    assert sidecar["verified_triangle"]["selected_camera_fraction"] == 0.6
    assert sidecar["verified_triangle"]["selected_fisher_logdet_gain_sum"] > 0
    assert (
        sidecar["verified_triangle"]["selected_confidence_weighted_utility_sum"]
        > 0
    )


def test_dimensionless_fisher_utility_is_invariant_to_world_scale():
    base_probe, base_keypoints, K, base_poses = _synthetic_probe_and_geometry(
        translation_scale=1.0
    )
    scaled_probe, scaled_keypoints, _, scaled_poses = _synthetic_probe_and_geometry(
        translation_scale=10.0
    )
    base_selected, base_sidecar = select_cycle_verified_fisher_pairs(
        pair_match_probe=base_probe,
        keypoints=base_keypoints,
        camera_K=K,
        pose_w2c=base_poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
    )
    scaled_selected, scaled_sidecar = select_cycle_verified_fisher_pairs(
        pair_match_probe=scaled_probe,
        keypoints=scaled_keypoints,
        camera_K=K,
        pose_w2c=scaled_poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
    )
    assert scaled_selected == base_selected
    assert scaled_sidecar["verified_triangle"]["scene_scale_m"] == pytest.approx(
        10.0 * base_sidecar["verified_triangle"]["scene_scale_m"]
    )
    assert scaled_sidecar["verified_triangle"][
        "selected_confidence_weighted_utility_sum"
    ] == pytest.approx(
        base_sidecar["verified_triangle"][
            "selected_confidence_weighted_utility_sum"
        ],
        rel=1e-10,
        abs=1e-10,
    )


def test_selector_fails_closed_when_budget_cannot_connect_all_cameras():
    probe, keypoints, K, poses = _synthetic_probe_and_geometry()
    with pytest.raises(ValueError, match="preserve"):
        select_cycle_verified_fisher_pairs(
            pair_match_probe=probe,
            keypoints=keypoints,
            camera_K=K,
            pose_w2c=poses,
            pair_budget=3,
        )

    candidate_matches, _ = pair_matches_from_probe(probe)
    disconnected_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    disconnected = materialize_pair_match_probe(
        candidate_pairs=disconnected_pairs,
        pair_matches={pair: candidate_matches[pair] for pair in disconnected_pairs},
        pair_diagnostics={
            pair: _diagnostics(pair, candidate_matches[pair])
            for pair in disconnected_pairs
        },
        keypoint_counts=[len(value) for value in keypoints],
        query_names_sha256="a" * 64,
        query_cache_sha256="b" * 64,
        mapping_keypoint_count=1024,
        mapping_nms_radius=4,
        candidate_pool_construction="synthetic_disconnected_pool_v1",
        candidate_pool_parameters={},
        matcher_parameters=_matcher_parameters(),
        detector_scores_applied=True,
    )
    with pytest.raises(RuntimeError, match="isolated"):
        select_cycle_verified_fisher_pairs(
            pair_match_probe=disconnected,
            keypoints=keypoints,
            camera_K=K,
            pose_w2c=poses,
            pair_budget=4,
        )


def test_bounded_candidate_union_preserves_components_without_soft_scoring():
    pairs, graph = bounded_union_candidate_pool(
        pair_sets=[
            [(0, 1), (1, 2), (3, 4)],
            [(0, 2), (2, 3), (3, 4)],
        ],
        query_count=5,
        maximum_pair_count=6,
    )
    assert pairs == [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    assert graph["arm_count"] == 2
    assert graph["component_count"] == 1
    assert graph["isolated_camera_count"] == 0
    with pytest.raises(ValueError, match="hard probe bound"):
        bounded_union_candidate_pool(
            pair_sets=[pairs, pairs],
            query_count=5,
            maximum_pair_count=4,
        )
    bounded_union_candidate_pool,
