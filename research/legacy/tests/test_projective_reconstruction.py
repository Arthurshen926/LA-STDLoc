import inspect
from pathlib import Path

import pytest
import torch

from common.v6_contracts import ASSOCIATION_GRAPH_SCHEMA
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_reconstruction import (
    joint_track_observation_weights,
    reconstruct_projective_anchors,
)
from scripts.materialize_v8_v2_projective_map import _implementation_sha256


def test_reconstruction_rejects_legacy_association() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError):
        reconstruct_projective_anchors(provider, {"schema": "legacy"})


def test_association_scope_must_be_mapping_only() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError):
        reconstruct_projective_anchors(
            provider,
            {
                "schema": ASSOCIATION_GRAPH_SCHEMA,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": True,
            },
        )


def test_base_reconstruction_does_not_require_cross_global_pose_bins() -> None:
    parameter = inspect.signature(reconstruct_projective_anchors).parameters[
        "minimum_view_bins"
    ]
    assert parameter.default == 1


def test_materializer_hashes_project_sources_not_torch_decorator_wrapper() -> None:
    hashes = _implementation_sha256()
    from common.hashing import sha256_file

    assert hashes["projective_reconstruction"] == sha256_file(
        Path(__file__).parents[1] / "evidence/projective_reconstruction.py"
    )
    assert hashes["v2_filtered_observations"] == sha256_file(
        Path(__file__).parents[1] / "evidence/v2_filtered_observations.py"
    )
    assert hashes["cycle_core_reserve"] == sha256_file(
        Path(__file__).parents[1] / "evidence/cycle_core_reserve.py"
    )


def test_stable_group_materialization_preserves_legacy_row_order() -> None:
    group = torch.tensor([2, 0, 1, 2, 0, 2, 1])
    order = torch.argsort(group, stable=True)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), torch.bincount(group).cumsum(0))
    )
    for index in range(3):
        legacy = torch.nonzero(group == index, as_tuple=False).reshape(-1)
        packed = order[offsets[index] : offsets[index + 1]]
        assert torch.equal(packed, legacy)


def _purification_inputs() -> dict:
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    return {
        "landmark_count": 1,
        "landmark_index": torch.zeros(5, dtype=torch.long),
        "query_index": torch.arange(5),
        "uv": torch.tensor(
            [
                [50.0, 50.0],
                [50.0, 50.0],
                [50.0, 50.0],
                [50.0, 50.0],
                [70.0, 50.0],
            ]
        ),
        "descriptors": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "confidence": torch.ones(5),
        "camera_K": intrinsic.repeat(5, 1, 1),
        "pose_w2c": torch.eye(4).repeat(5, 1, 1),
        "initial_xyz": torch.tensor([[0.0, 0.0, 5.0]]),
    }


def test_joint_track_purification_downweights_joint_outlier() -> None:
    weight, report = joint_track_observation_weights(
        **_purification_inputs(),
        rendered_depth=torch.tensor([5.0, 5.0, 5.0, 5.0, 2.0]),
    )
    assert float(weight[-1]) < 0.25 * float(weight[:3].min())
    assert report["gaussian_depth_role"] == "observation_consistency_only"
    assert report["final_xyz_source"] == (
        "second_fixed_camera_robust_ray_triangulation"
    )
    assert report["version"] == 3
    assert report["descriptor_reference"] == (
        "leave_one_observation_out_weighted_track_consensus"
    )
    assert report["relative_depth_reference"] == (
        "leave_one_observation_out_track_mean_bias"
    )


def test_consistent_depth_bias_does_not_move_or_penalize_track() -> None:
    inputs = _purification_inputs()
    inputs["uv"][-1] = torch.tensor([50.0, 50.0])
    inputs["descriptors"][-1] = torch.tensor([1.0, 0.0])
    with_depth, _ = joint_track_observation_weights(
        **inputs, rendered_depth=torch.full((5,), 4.0)
    )
    without_depth, _ = joint_track_observation_weights(
        **inputs, rendered_depth=None
    )
    assert torch.allclose(with_depth, without_depth)


def test_fixed_ray_reconstruction_is_bitwise_invariant_to_auxiliary_depth():
    # Conditional representation test only: fixed pixels, cameras, Track IDs,
    # detector/descriptor evidence. V2 and completion are deliberately frozen.
    K=torch.tensor([[100.,0.,64.],[0.,100.,64.],[0.,0.,1.]])
    records={}
    for i,center in enumerate([-.8,-.4,0.,.4,.8]):
        T=torch.eye(4);T[0,3]=-center
        records[str(i)]={'native_keypoints':torch.tensor([[64.-20.*center-.5,63.5]]),
            'native_descriptors':torch.tensor([[1.,0.,0.]]),'native_scores':torch.ones(1),
            'native_K':K,'pose_w2c':T,'native_input_hw':torch.tensor([128,128]),
            'native_depth':torch.full((128,128),5.)}
    association={'schema':ASSOCIATION_GRAPH_SCHEMA,'uses_source_mapping_rgb':False,'uses_test_queries':False,
        'query_names':list(records),'query_bins':torch.arange(5),'diagnostics':{'track_count':1},
        'component_statistics':{'identity_reliability':torch.ones(1)},
        'tracks':{'track_index':torch.zeros(5,dtype=torch.long),'query_index':torch.arange(5),
                  'keypoint_index':torch.zeros(5,dtype=torch.long),'confidence':torch.ones(5)}}
    results=[]
    for scale in [.01,1.,20.]:
        payload={'schema':'render_observation_cache_v2','version':2,
                 'uses_source_mapping_rgb':False,'uses_test_queries':False,
                 'queries':{k:{**r,'native_depth':r['native_depth']*scale*(int(k)+1)} for k,r in records.items()}}
        results.append(reconstruct_projective_anchors(GaussianRenderObservationProvider(payload),association,parallel_workers=1))
    for result in results:
        assert result['anchor_xyz'].shape==(1,3)
        assert torch.allclose(result['anchor_xyz'],torch.tensor([[0.,0.,5.]]),atol=1e-5)
        for field in ['anchor_xyz','anchor_features','anchor_position_covariance']:
            assert torch.equal(result[field],results[0][field])
