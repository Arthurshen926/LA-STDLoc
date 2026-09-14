"""Bounded synthetic regressions: no reconstruction or dataset benchmark."""

import ast
from pathlib import Path
import torch
import pytest


def test_full_selector_matches_original_control_flow():
    from anygsloc.mapping.compact import select_compact_rows
    from topology.sufficiency_selector import HierarchicalSufficiencySelector
    from topology.adaptive_distillation import _build_pose_evidence, _initial_pose_state
    from topology.dynamic_reserve import spatial_voxel_ids

    torch.set_num_threads(4)
    torch.manual_seed(12)
    n, q = 1200, 3
    xyz = torch.randn(n, 3)
    xyz[:, 2] += 8
    queries = torch.arange(q).repeat(n)
    points = torch.arange(n).remainder(400).repeat_interleave(q)
    state = dict(
        anchor_ids=torch.arange(n),
        anchor_xyz=xyz,
        anchor_matchability=torch.rand(n),
        anchor_position_covariance=torch.eye(3)[None].repeat(n, 1, 1) * 0.0001,
        v6_mapping_query_names=[f"q{i}" for i in range(q)],
        v6_mapping_query_bins=torch.arange(q),
        projective_anchor_observations=dict(
            observation_offsets=torch.arange(n + 1) * q,
            query_indices=queries,
            keypoint_indices=points,
        ),
        provenance=dict(uses_test_queries=False),
    )
    cache = {
        name: dict(
            native_scores=torch.rand(400),
            native_K=torch.tensor([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]]),
            pose_w2c=torch.eye(4),
            native_keypoints=torch.rand(400, 2) * torch.tensor([640, 480]),
            native_input_hw=[480, 640],
        )
        for name in state["v6_mapping_query_names"]
    }
    source = (
        Path(__file__).resolve().parents[1]
        / "research/legacy/scripts/run_full_selector_pilot.py"
    )
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    body = []
    collect = False
    for node in function.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "names"
        ):
            collect = True
        if collect:
            body.append(node)
        if (
            collect
            and isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "full_selector"
        ):
            break
    reference = dict(
        torch=torch,
        state=state,
        cache=cache,
        HierarchicalSufficiencySelector=HierarchicalSufficiencySelector,
        _build_pose_evidence=_build_pose_evidence,
        _initial_pose_state=_initial_pose_state,
        spatial_voxel_ids=spatial_voxel_ids,
    )
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), reference
    )
    expected = reference["variants"]["full_selector"]
    actual, trace = select_compact_rows(state, cache)
    assert torch.equal(actual, expected)


def test_fresh_process_triangulation_uses_new_namespace(tmp_path):
    from anygsloc.mapping.parallel_triangulation import _default_command

    command = _default_command(tmp_path / "job.json", 0)
    assert "anygsloc.mapping.parallel_triangulation" in command
    assert "evidence.parallel_triangulation" not in command


def test_synthetic_observations_to_compact_deployment_map(tmp_path):
    from anygsloc.mapping.observation_provider import GaussianRenderObservationProvider
    from anygsloc.mapping.projective_association import (
        build_projective_association_graph,
    )
    from anygsloc.mapping.projective_reconstruction import (
        reconstruct_projective_anchors,
    )
    from anygsloc.mapping.projective_completion import build_projective_completion
    from anygsloc.mapping.anchor_map import (
        merge_projective_candidates,
        materialize_projective_anchor_map,
        identity_metric_state,
    )
    from anygsloc.mapping.view_support import build_anchor_view_support
    from anygsloc.mapping.compact import compact_map
    from anygsloc.artifacts import sha256, validate

    torch.manual_seed(8)
    count = 32
    truth = torch.rand(count, 3)
    truth[:, 2] += 5
    features = torch.eye(256)[:count]
    K = torch.tensor([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
    records = {}
    for index in range(3):
        pose = torch.eye(4)
        pose[0, 3] = -0.5 * index
        camera = truth + pose[:3, 3]
        projected = camera @ K.T
        records[f"q{index}"] = dict(
            native_keypoints=projected[:, :2] / projected[:, 2:] - 0.5,
            native_descriptors=features,
            native_scores=torch.ones(count),
            native_K=K,
            pose_w2c=pose,
            native_input_hw=[480, 640],
            native_valid_keypoint_mask=torch.ones(count, dtype=torch.bool),
            native_alpha_at_keypoints=torch.ones(count),
            native_depth_at_keypoints=camera[:, 2],
        )
    cache = dict(
        uses_source_mapping_rgb=False, uses_test_queries=False, queries=records
    )
    provider = GaussianRenderObservationProvider(cache)
    association = build_projective_association_graph(provider, device="cpu")
    candidates = reconstruct_projective_anchors(
        provider, association, parallel_workers=1
    )
    sharded = reconstruct_projective_anchors(
        provider, association, parallel_workers=2, parallel_minimum_tracks=1
    )
    torch.testing.assert_close(
        candidates["anchor_xyz"], sharded["anchor_xyz"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        candidates["anchor_features"], sharded["anchor_features"], atol=0, rtol=0
    )
    # The frozen materializer explicitly accepts this no-unused-observation
    # outcome and proceeds with the base candidates alone.
    with pytest.raises(ValueError, match="no unused render-valid observations"):
        build_projective_completion(
            provider,
            association,
            voxel_size_m=0.05,
            alpha_minimum=0.05,
            minimum_similarity=0.7,
            device="cpu",
        )
    state = materialize_projective_anchor_map(
        merge_projective_candidates([candidates]), lineage={}
    )
    csr = state["projective_anchor_observations"]
    state["anchor_view_support"] = build_anchor_view_support(
        anchor_xyz=state["anchor_xyz"],
        observation_offsets=csr["observation_offsets"],
        observation_query_indices=csr["query_indices"],
        mapping_pose_w2c=torch.stack([r["pose_w2c"] for r in records.values()]),
    )
    expected = truth[state["anchor_features"].argmax(dim=1)]
    torch.testing.assert_close(state["anchor_xyz"], expected, atol=1e-4, rtol=1e-4)
    assert len(state["anchor_ids"]) == count
    full = tmp_path / "full"
    full.mkdir()
    path = full / "projective_anchor_map.pt"
    torch.save(state, path)
    torch.save(
        identity_metric_state(state, map_path=str(path), map_sha256=sha256(path)),
        full / "identity_metric.pt",
    )
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    stats = compact_map(full, cache_path, tmp_path / "compact")
    exported = validate(
        torch.load(tmp_path / "compact/deployment_map.pt", weights_only=True)
    )
    assert stats["anchor_count"] == len(exported["anchor_ids"]) > 0
