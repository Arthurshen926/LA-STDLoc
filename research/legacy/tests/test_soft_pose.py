import torch

from map_learning.soft_pose import soft_pose_bias_loss


def test_soft_pose_bias_prefers_geometry_consistent_candidate():
    dtype = torch.float64
    xyz = torch.tensor(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [3.0, 0.0, 4.0],
        ],
        dtype=dtype,
    )
    K = torch.tensor([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1]], dtype=dtype)
    pose = torch.eye(4, dtype=dtype)
    keypoints = torch.stack(
        (100.0 * xyz[:4, 0] / xyz[:4, 2] + 50.0, 100.0 * xyz[:4, 1] / xyz[:4, 2] + 50.0),
        dim=1,
    )
    anchors = torch.eye(5, dtype=dtype)
    clean_query = anchors[:4].clone().requires_grad_(True)
    wrong_query = torch.stack([anchors[4], anchors[4], anchors[4], anchors[4]]).requires_grad_(True)
    clean, _ = soft_pose_bias_loss(
        query_features=clean_query,
        anchor_features=anchors,
        anchor_xyz=xyz,
        keypoint_xy=keypoints,
        intrinsic=K,
        pose_gt_w2c=pose,
        topk=2,
    )
    wrong, _ = soft_pose_bias_loss(
        query_features=wrong_query,
        anchor_features=anchors,
        anchor_xyz=xyz,
        keypoint_xy=keypoints,
        intrinsic=K,
        pose_gt_w2c=pose,
        topk=2,
    )
    assert torch.isfinite(clean)
    assert torch.isfinite(wrong)
    assert wrong > clean
    wrong.backward()
    assert torch.isfinite(wrong_query.grad).all()


def test_soft_pose_bias_stays_finite_with_invalid_depth_candidates():
    dtype = torch.float32
    xyz = torch.tensor(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [10.0, 10.0, -1.0],
            [1.0, 1.0, 1e-9],
        ],
        dtype=dtype,
    )
    K = torch.tensor([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1]], dtype=dtype)
    pose = torch.eye(4, dtype=dtype)
    keypoints = torch.stack(
        (100.0 * xyz[:4, 0] / xyz[:4, 2] + 50.0, 100.0 * xyz[:4, 1] / xyz[:4, 2] + 50.0),
        dim=1,
    )
    anchors = torch.eye(6, dtype=dtype)
    query = torch.nn.Parameter(anchors[:4].clone() + 0.01)
    optimizer = torch.optim.Adam([query], lr=1e-2)
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = soft_pose_bias_loss(
            query_features=query,
            anchor_features=anchors,
            anchor_xyz=xyz,
            keypoint_xy=keypoints,
            intrinsic=K,
            pose_gt_w2c=pose,
            topk=6,
        )
        assert torch.isfinite(loss)
        assert diagnostics["soft_pose_valid_candidate_fraction"] < 1.0
        loss.backward()
        assert torch.isfinite(query.grad).all()
        optimizer.step()
        assert torch.isfinite(query).all()
