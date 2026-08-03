import torch

from evidence.triangulation import robust_triangulate_associations


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
    rotation = torch.stack((right, down, forward))
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -(rotation @ center)
    return pose


def _project(point, K, pose):
    camera = (pose @ torch.cat((point, point.new_ones(1))))[:3]
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def test_robust_triangulation_recovers_track_and_rejects_duplicate_view():
    point = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    centers = [[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0], [0.3, -0.3, 0.0]]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(len(centers), 1, 1)
    observed = [_project(point, K[index], poses[index]) for index in range(4)]
    uv = torch.stack(
        [
            observed[0] + torch.tensor([20.0, 15.0]),
            observed[0],
            observed[1],
            observed[2],
            observed[3],
        ]
    )
    result = robust_triangulate_associations(
        landmark_count=2,
        landmark_index=torch.zeros(5, dtype=torch.long),
        query_index=torch.tensor([0, 0, 1, 2, 3]),
        uv=uv,
        confidence=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]),
        camera_K=K,
        pose_w2c=poses,
        query_bin=torch.tensor([0, 1, 2, 3]),
        minimum_views=3,
        minimum_view_bins=2,
        maximum_reprojection_px=1.0,
    )
    assert result["triangulation_high_confidence"].tolist() == [True, False]
    torch.testing.assert_close(result["triangulated_xyz"][0], point.float(), atol=1e-4, rtol=0)
