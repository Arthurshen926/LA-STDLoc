import copy
import itertools
import random

import pytest
import torch

from evidence import cycle_verified_fisher as cycle_fisher
from evidence.cycle_verified_fisher import (
    COVERAGE_POLICY_NAME,
    POLICY_NAME,
    PROBE_SCHEMA,
    bounded_union_candidate_pool,
    materialize_pair_match_probe,
    materialize_verified_cycle_table,
    pair_matches_from_probe,
    probe_track_build_inputs,
    select_cycle_verified_fisher_pairs,
    select_cycle_verified_fisher_coverage_pairs,
    validate_cycle_verified_fisher_coverage_selection,
    validate_cycle_verified_fisher_selection,
    validate_pair_match_probe,
    validate_verified_cycle_table,
    verified_cycle_table_content_sha256,
)


def _closure_pair(triangle_edges, triangle_utility, selected, pair_budget, edge_count):
    expected = cycle_fisher._complete_verified_triangles_bruteforce(
        triangle_edges=triangle_edges,
        triangle_utility=triangle_utility,
        selected=selected,
        pair_budget=pair_budget,
    )
    actual = cycle_fisher._complete_verified_triangles_incremental(
        triangle_edges=triangle_edges,
        triangle_utility=triangle_utility,
        selected=selected,
        pair_budget=pair_budget,
        edge_count=edge_count,
    )
    return expected, actual


def test_incremental_closure_matches_bruteforce_for_random_tied_utilities():
    generator = random.Random(20260813)
    for trial in range(2000):
        edge_count = generator.randint(3, 36)
        triangle_count = generator.randint(0, 80)
        rows = [generator.sample(range(edge_count), 3) for _ in range(triangle_count)]
        triangle_edges = (
            torch.tensor(rows, dtype=torch.long)
            if rows
            else torch.zeros((0, 3), dtype=torch.long)
        )
        # A small discrete set deliberately creates exact priority and tuple ties.
        triangle_utility = torch.tensor(
            [generator.choice((0.125, 0.25, 0.5, 1.0, 2.0, 4.0)) for _ in rows],
            dtype=torch.float64,
        )
        initial_count = generator.randint(0, min(edge_count, 10))
        selected = set(generator.sample(range(edge_count), initial_count))
        pair_budget = generator.randint(initial_count, edge_count)
        expected, actual = _closure_pair(
            triangle_edges,
            triangle_utility,
            selected,
            pair_budget,
            edge_count,
        )
        assert actual == expected, f"random closure parity failed at trial {trial}"


@pytest.mark.parametrize("remaining", [1, 2, 3])
def test_incremental_closure_preserves_final_slot_bundle_eligibility(remaining):
    triangle_edges = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]], dtype=torch.long
    )
    triangle_utility = torch.tensor([9.0, 8.0, 7.0, 6.0], dtype=torch.float64)
    selected = {0, 1, 3}
    pair_budget = len(selected) + remaining
    expected, actual = _closure_pair(
        triangle_edges,
        triangle_utility,
        selected,
        pair_budget,
        edge_count=6,
    )
    assert actual == expected
    assert len(actual) <= pair_budget


def test_incremental_closure_preserves_first_triangle_exact_tie_break():
    # Duplicate rows and utilities have identical registered priority.  The
    # original scan keeps the first triangle; the lazy heap must do likewise.
    triangle_edges = torch.tensor([[0, 2, 4], [0, 2, 4], [1, 3, 5]], dtype=torch.long)
    triangle_utility = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
    expected, actual = _closure_pair(
        triangle_edges,
        triangle_utility,
        selected={0},
        pair_budget=3,
        edge_count=6,
    )
    assert actual == expected == {0, 2, 4}


def test_incremental_closure_rejects_out_of_range_edges():
    edges = torch.tensor([[0, 1, 2]], dtype=torch.long)
    utility = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="verified triangle"):
        cycle_fisher._complete_verified_triangles_incremental(
            triangle_edges=edges + 3,
            triangle_utility=utility,
            selected=set(),
            pair_budget=1,
            edge_count=3,
        )
    for selected in ({-1}, {3}):
        with pytest.raises(ValueError, match="preselected edge"):
            cycle_fisher._complete_verified_triangles_incremental(
                triangle_edges=edges,
                triangle_utility=utility,
                selected=selected,
                pair_budget=1,
                edge_count=3,
            )


def test_full_selector_content_matches_bruteforce_closure(monkeypatch):
    probe, keypoints, K, poses = _synthetic_probe_and_geometry()
    incremental_pairs, incremental_sidecar = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
    )

    def brute_adapter(
        *, triangle_edges, triangle_utility, selected, pair_budget, edge_count
    ):
        del edge_count
        return cycle_fisher._complete_verified_triangles_bruteforce(
            triangle_edges=triangle_edges,
            triangle_utility=triangle_utility,
            selected=selected,
            pair_budget=pair_budget,
        )

    monkeypatch.setattr(
        cycle_fisher, "_complete_verified_triangles_incremental", brute_adapter
    )
    brute_pairs, brute_sidecar = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
    )
    assert incremental_pairs == brute_pairs
    assert incremental_sidecar["content_sha256"] == brute_sidecar["content_sha256"]


def _completed_cameras(triangle_camera, triangle_edges, selected):
    return {
        int(camera)
        for cameras, edges in zip(triangle_camera.tolist(), triangle_edges.tolist())
        if all(int(edge) in selected for edge in edges)
        for camera in cameras
    }


def test_coverage_scaffold_matches_random_small_graph_feasibility_oracle():
    generator = random.Random(20260813)
    successful = 0
    for trial in range(30):
        query_count = generator.randint(4, 6)
        pairs = list(itertools.combinations(range(query_count), 2))
        pair_index = {pair: index for index, pair in enumerate(pairs)}
        triangle_camera = torch.tensor(
            list(itertools.combinations(range(query_count), 3)), dtype=torch.long
        )
        triangle_edges = torch.tensor(
            [
                [
                    pair_index[(left, middle)],
                    pair_index[(left, right)],
                    pair_index[(middle, right)],
                ]
                for left, middle, right in triangle_camera.tolist()
            ],
            dtype=torch.long,
        )
        triangle_utility = torch.tensor(
            [generator.choice((0.5, 1.0, 2.0, 4.0)) for _ in triangle_camera],
            dtype=torch.float64,
        )
        reference_pairs = {(0, 1), (0, 2), (1, 2)}
        reference_pairs.update((2, camera) for camera in range(3, query_count))
        budget = min(len(pairs), query_count + 1)
        for pair in pairs:
            if len(reference_pairs) >= budget:
                break
            reference_pairs.add(pair)
        reference = {pair_index[pair] for pair in reference_pairs}
        edge_utility, edge_count = cycle_fisher._edge_evidence(
            triangle_edges, triangle_utility, edge_count=len(pairs)
        )
        edge_order = sorted(
            range(len(pairs)),
            key=lambda edge: (
                -float(edge_utility[edge]),
                -int(edge_count[edge]),
                pairs[edge],
            ),
        )
        try:
            scaffold, _, _, target, _ = cycle_fisher._coverage_scaffold_indices(
                pairs=pairs,
                triangle_camera=triangle_camera,
                triangle_edges=triangle_edges,
                triangle_utility=triangle_utility,
                coverage_reference_indices=reference,
                query_count=query_count,
                pair_budget=budget,
                minimum_camera_degree=1,
                edge_order=edge_order,
            )
        except RuntimeError as error:
            # The deterministic greedy may fail even when the NP-hard joint
            # problem has a feasible solution; it must fail closed, not claim
            # feasibility or silently exceed the exact budget.
            assert "exceeds the exact pair budget" in str(error)
            continue
        successful += 1

        # Exhaustive small-graph oracle proves the joint hard constraints are
        # feasible; V2 promises a deterministic feasible solution, not optimum.
        feasible = []
        for combination in itertools.combinations(range(len(pairs)), budget):
            chosen = set(combination)
            graph = cycle_fisher._graph_diagnostics(pairs, chosen, query_count)
            if (
                graph["component_count"] == 1
                and graph["isolated_camera_count"] == 0
                and set(target).issubset(
                    _completed_cameras(triangle_camera, triangle_edges, chosen)
                )
            ):
                feasible.append(chosen)
                break
        assert feasible, f"oracle found no feasible graph at trial {trial}"
        assert len(scaffold) <= budget
        assert set(target).issubset(
            _completed_cameras(triangle_camera, triangle_edges, scaffold)
        )
        graph = cycle_fisher._graph_diagnostics(pairs, scaffold, query_count)
        assert graph["component_count"] == 1
        assert graph["isolated_camera_count"] == 0
    assert 0 < successful < 30


def test_coverage_scaffold_fails_closed_without_reference_verified_triangle():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    with pytest.raises(RuntimeError, match="completes no verified triangle"):
        cycle_fisher._coverage_scaffold_indices(
            pairs=pairs,
            triangle_camera=torch.tensor([[0, 1, 2]]),
            triangle_edges=torch.tensor([[0, 1, 3]]),
            triangle_utility=torch.tensor([1.0]),
            coverage_reference_indices={0, 1, 2},
            query_count=4,
            pair_budget=3,
            minimum_camera_degree=1,
            edge_order=list(range(4)),
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


def test_coverage_selector_hard_covers_control_and_reuses_verified_table():
    probe, keypoints, camera_K, poses = _synthetic_probe_and_geometry()
    control = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    verified = materialize_verified_cycle_table(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=camera_K,
        pose_w2c=poses,
        maximum_reprojection_error_px=0.01,
    )
    validate_verified_cycle_table(
        verified,
        pair_match_probe=probe,
        expected_content_sha256=verified["content_sha256"],
    )
    selected, sidecar = select_cycle_verified_fisher_coverage_pairs(
        pair_match_probe=probe,
        coverage_reference_pairs=control,
        keypoints=keypoints,
        camera_K=camera_K,
        pose_w2c=poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
        verified_cycle_table=verified,
    )
    assert len(selected) == 5
    assert sidecar["policy"] == COVERAGE_POLICY_NAME
    assert sidecar["coverage_certificate"]["target_camera_index"] == [0, 1, 2]
    assert sidecar["coverage_certificate"]["all_target_cameras_covered"] is True
    assert sidecar["coverage_certificate"]["lost_control_camera_count"] == 0
    assert sidecar["coverage_certificate"]["remaining_budget_after_stage1"] >= 0
    validate_cycle_verified_fisher_coverage_selection(
        sidecar,
        pair_match_probe=probe,
        coverage_reference_pairs=control,
        verified_cycle_table=verified,
    )

    tampered = copy.deepcopy(verified)
    tampered["verified_triangle"]["utility"][0] += 0.25
    with pytest.raises(ValueError, match="metric columns|content SHA-256"):
        validate_verified_cycle_table(tampered, pair_match_probe=probe)


def test_coverage_validator_rejects_target_derived_from_camera_degree():
    probe, keypoints, camera_K, poses = _synthetic_probe_and_geometry()
    control = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    verified = materialize_verified_cycle_table(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=camera_K,
        pose_w2c=poses,
        maximum_reprojection_error_px=0.01,
    )
    _, sidecar = select_cycle_verified_fisher_coverage_pairs(
        pair_match_probe=probe,
        coverage_reference_pairs=control,
        keypoints=keypoints,
        camera_K=camera_K,
        pose_w2c=poses,
        pair_budget=5,
        maximum_cycle_reprojection_error_px=0.01,
        verified_cycle_table=verified,
    )
    tampered = copy.deepcopy(sidecar)
    # Cameras 3/4 have positive graph degree but are not in any completed
    # control verified triangle.  A degree-based implementation would accept.
    tampered["coverage_certificate"]["target_camera_index"] = [0, 1, 2, 3, 4]
    tampered["coverage_certificate"]["target_camera_count"] = 5
    tampered["coverage_certificate"]["target_camera_index_sha256"] = (
        cycle_fisher._integer_set_sha256(range(5))
    )
    tampered["content_sha256"] = cycle_fisher._selection_content_sha256(tampered)
    with pytest.raises(ValueError, match="same-probe control"):
        validate_cycle_verified_fisher_coverage_selection(
            tampered,
            pair_match_probe=probe,
            coverage_reference_pairs=control,
            verified_cycle_table=verified,
        )


def test_coverage_selector_rejects_precomputed_table_from_different_threshold():
    probe, keypoints, camera_K, poses = _synthetic_probe_and_geometry()
    verified = materialize_verified_cycle_table(
        pair_match_probe=probe,
        keypoints=keypoints,
        camera_K=camera_K,
        pose_w2c=poses,
        maximum_reprojection_error_px=0.01,
    )
    modified = copy.deepcopy(verified)
    modified["parameters"]["maximum_cycle_reprojection_error_px"] = 0.02
    modified["content_sha256"] = verified_cycle_table_content_sha256(modified)
    validate_verified_cycle_table(modified, pair_match_probe=probe)
    with pytest.raises(ValueError, match="different threshold"):
        select_cycle_verified_fisher_coverage_pairs(
            pair_match_probe=probe,
            coverage_reference_pairs=[
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (2, 4),
            ],
            keypoints=keypoints,
            camera_K=camera_K,
            pose_w2c=poses,
            pair_budget=5,
            maximum_cycle_reprojection_error_px=0.01,
            verified_cycle_table=modified,
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
