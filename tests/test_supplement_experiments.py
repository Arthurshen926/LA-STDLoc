"""Scientific controls: exact budgets, fixed cameras, conflict-free tracks."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from supplement_experiments import spatial_rows, quality_rows
from supplement_classic import assemble_tracks, fit_point
from supplement_geometry import refit_ray
from figure_cache_reader import read_cache


def cameras_and_measurements():
    centers = np.array([[-1., 0, 0], [1., 0, 0], [0, 1., 0], [0, -1., 0]])
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = -centers
    K = np.repeat(np.array([[700., 0, 320.], [0, 710., 240.], [0, 0, 1.]])[None], 4, axis=0)
    point = np.array([.2, .1, 5.])
    h = np.einsum('nij,nj->ni', K, point-centers)
    return point, h[:, :2]/h[:, 2:], K, poses


def test_spatial_exact_budget_duplicate_coordinates_and_rounding():
    xyz = np.array([[0, 0, 0]]*7 + [[2., 0, 0]]*3 + [[5., 0, 0]]*2)
    for k in range(1, len(xyz)+1):
        rows = spatial_rows(xyz, k)
        assert len(rows) == k and len(set(rows)) == k
        assert np.array_equal(rows, spatial_rows(xyz, k))
    assert len(np.unique(xyz[spatial_rows(xyz, 3)], axis=0)) == 3


def test_quality_topk_uses_global_scores_with_stable_source_tie():
    assert quality_rows([1., 3., 3., 0., 5.], 3).tolist() == [1, 2, 4]


def test_conflicting_track_merge_preserves_good_subtrack():
    pairs = [(0, 1, np.array([0]), np.array([0]), np.array([.99])),
             (1, 2, np.array([0]), np.array([0]), np.array([.98])),
             (0, 1, np.array([0]), np.array([1]), np.array([.8]))]
    tracks, stats = assemble_tracks(pairs, [1, 2, 1])
    assert len(tracks) == 1 and stats['conflicting_unions_rejected'] == 1
    assert tracks[0][0].tolist() == [0, 1, 2]
    assert tracks[0][1].tolist() == [0, 0, 0]


def test_classic_fixed_camera_triangulation_recovers_point_without_pose_changes():
    point, uv, K, poses = cameras_and_measurements()
    copy = poses.copy()
    result = fit_point(uv, K, poses)
    assert result is not None
    xyz, keep, error = result
    assert np.allclose(xyz, point, atol=1e-6) and len(keep) == 4 and error < 1e-6
    assert np.array_equal(poses, copy)


def test_common_ray_refit_uses_supplied_observations_and_pixel_convention():
    point, uv, K, poses = cameras_and_measurements()
    assert np.allclose(refit_ray(uv[:3], K[:3], poses[:3]), point, atol=1e-6)
    moved = uv[:3].copy(); moved[:, 0] += 4
    assert not np.allclose(refit_ray(moved, K[:3], poses[:3]), point, atol=1e-3)


def test_selective_trusted_cache_loader_preserves_dtype_stride_and_offset(tmp_path):
    values = dict(float=torch.arange(40, dtype=torch.float32).reshape(5, 8)[1:, ::2],
                  integer=torch.tensor([2, 5, 7]), boolean=torch.tensor([True, False]))
    path = tmp_path/'cache.pt'
    torch.save(values, path)
    loaded = read_cache(path)
    for key, value in values.items():
        assert np.array_equal(loaded[key].numpy(), value.numpy())
        assert loaded[key].numpy().dtype == value.numpy().dtype
