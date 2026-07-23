import numpy as np
import torch
import cv2
import poselib
import math

def solve_pose(
    p2d,
    p3d,
    K,
    solver="poselib",
    reprojection_error=8.0,
    confidence=0.9999,
    max_iterations=100000,
    min_iterations=1000,
    scores=None,
    progressive_sampling=False,
    max_prosac_iterations=100000,
    ransac_seed=0,
    return_diagnostics=False,
):
    p2d = np.asarray(p2d)
    p3d = np.asarray(p3d)
    ransac_seed = int(ransac_seed)
    if ransac_seed < 0:
        raise ValueError("ransac_seed must be non-negative")
    match_num = p2d.shape[0]
    diagnostics = {
        "solver": str(solver),
        "ransac_candidate_count": int(match_num),
        "ransac_configured_max_iterations": int(max_iterations),
        "ransac_configured_min_iterations": int(min_iterations),
        "ransac_confidence": float(confidence),
        "ransac_progressive_sampling": bool(progressive_sampling),
        "ransac_max_prosac_iterations": int(max_prosac_iterations),
        "ransac_seed": ransac_seed,
        "ransac_actual_hypotheses": None,
        "ransac_actual_hypotheses_available": False,
        "ransac_inlier_ratio": 0.0,
        "ransac_required_hypotheses_at_confidence": None,
    }

    def finish(pose, inliers, *, info=None):
        inliers = np.asarray(inliers).reshape(-1)
        inlier_ratio = float(inliers.shape[0] / max(match_num, 1))
        diagnostics["ransac_inlier_ratio"] = inlier_ratio
        if info is not None:
            # Poselib exposes the number of sampled RANSAC hypotheses as
            # ``iterations``. Preserve it verbatim instead of presenting a
            # confidence-derived estimate as an actual count.
            iterations = info.get("iterations") if isinstance(info, dict) else None
            if iterations is not None:
                diagnostics["ransac_actual_hypotheses"] = int(iterations)
                diagnostics["ransac_actual_hypotheses_available"] = True
            if isinstance(info, dict):
                diagnostics["ransac_refinements"] = int(info.get("refinements", 0))
                diagnostics["ransac_model_score"] = float(info.get("model_score", 0.0))
        if inlier_ratio > 0.0:
            success_probability = min(max(float(confidence), 1e-12), 1.0 - 1e-12)
            all_inlier_probability = min(max(inlier_ratio**4, 1e-12), 1.0 - 1e-12)
            diagnostics["ransac_required_hypotheses_at_confidence"] = int(
                math.ceil(
                    math.log(1.0 - success_probability)
                    / math.log(1.0 - all_inlier_probability)
                )
            )
        if return_diagnostics:
            return pose, inliers, diagnostics
        return pose, inliers

    if match_num < 4:
        print("[SKIP] No enough matches")
        return finish(np.eye(4, dtype=np.float32), np.array([]))

    solver_to_input = np.arange(match_num, dtype=np.int64)
    if progressive_sampling:
        if scores is None:
            raise ValueError("progressive pose sampling requires correspondence scores")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != match_num:
            raise ValueError("pose scores must match the correspondence count")
        ranking_scores = np.where(np.isfinite(scores), scores, -np.inf)
        solver_to_input = np.argsort(-ranking_scores, kind="stable")
        p2d = p2d[solver_to_input]
        p3d = p3d[solver_to_input]

    if solver == "opencv":
        # OpenCV keeps RANSAC state globally, so set it before every solve.
        cv2.setRNGSeed(ransac_seed)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d,
            p2d,
            K,
            distCoeffs=np.zeros((4, 1)),
            reprojectionError=reprojection_error,
            confidence=confidence,
            iterationsCount=max_iterations
        )
        if success:
            w2c = np.eye(4)
            cv2.Rodrigues(rvec, w2c[:3, :3])
            w2c[:3, -1] = tvec.flatten()
            w2c = w2c.astype(np.float32)
            inliers = solver_to_input[np.asarray(inliers).reshape(-1)]
            return finish(w2c, inliers)

    elif solver == "poselib":
        camera = {
            "model": "PINHOLE",
            "width": int(K[0, 2] * 2),
            "height": int(K[1, 2] * 2),
            "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        }

        max_reproj_error = reprojection_error
        confidence = confidence

        pose, info = poselib.estimate_absolute_pose(
            p2d,
            p3d,
            camera,
            {
                "max_iterations": max_iterations,
                "min_iterations": min_iterations,
                "max_reproj_error": max_reproj_error,
                "success_prob": confidence,
                "progressive_sampling": bool(progressive_sampling),
                "max_prosac_iterations": int(max_prosac_iterations),
                "seed": ransac_seed,
            },
            {
                "verbose": False,
            },
        )

        if info["num_inliers"] > 0:
            w2c = pose.Rt
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0).astype(
                np.float32
            )
            inliers = info["inliers"]
            indices = np.where(inliers)[0]
            inliers = solver_to_input[indices].reshape(-1, 1).astype(np.int32)
            return finish(w2c, inliers.flatten(), info=info)

    return finish(np.eye(4, dtype=np.float32), np.array([]))


def covariance_weighted_pose_refinement(
    p2d,
    p3d,
    K,
    initial_pose_w2c,
    covariance,
    inliers,
    iterations=10,
    mahalanobis_threshold=3.0,
    robust_delta=2.5,
    model_mismatch_floor_px=1.0,
    damping=1e-6,
):
    """Refine a robust PnP hypothesis using pair-specific 2D covariance."""
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(-1, 2, 2)
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    valid_inliers = inliers[(inliers >= 0) & (inliers < p2d.shape[0])]
    if p2d.shape[0] != p3d.shape[0] or covariance.shape[0] != p2d.shape[0]:
        raise ValueError("p2d, p3d, and covariance must have the same count")
    if valid_inliers.shape[0] < 4:
        return np.asarray(initial_pose_w2c, dtype=np.float64), valid_inliers

    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    model_mismatch_variance = max(float(model_mismatch_floor_px), 0.0) ** 2
    covariance = covariance + np.eye(2, dtype=np.float64)[None] * (
        model_mismatch_variance + 1e-8
    )
    pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec = cv2.Rodrigues(pose[:3, :3])[0].reshape(3)
    tvec = pose[:3, 3].copy()

    def project_and_whiten(indices):
        projected, jacobian = cv2.projectPoints(
            p3d[indices],
            rvec,
            tvec,
            K,
            np.zeros((4, 1), dtype=np.float64),
        )
        residual = p2d[indices] - projected.reshape(-1, 2)
        jacobian = jacobian[:, :6].reshape(-1, 2, 6)
        cholesky = np.linalg.cholesky(covariance[indices])
        whitened_residual = np.linalg.solve(
            cholesky, residual[..., None]
        )[..., 0]
        whitened_jacobian = np.linalg.solve(cholesky, jacobian)
        return whitened_residual, whitened_jacobian

    threshold = float(mahalanobis_threshold)
    if np.isfinite(threshold) and threshold > 0.0:
        residual, _ = project_and_whiten(valid_inliers)
        mahalanobis = np.linalg.norm(residual, axis=1)
        selected = valid_inliers[
            np.isfinite(mahalanobis) & (mahalanobis <= threshold)
        ]
        if selected.shape[0] < 4:
            selected = valid_inliers
    else:
        selected = valid_inliers

    for _ in range(max(int(iterations), 0)):
        residual, jacobian = project_and_whiten(selected)
        radial = np.linalg.norm(residual, axis=1)
        delta = float(robust_delta)
        robust = (
            np.minimum(1.0, delta / np.maximum(radial, 1e-8))
            if delta > 0.0
            else np.ones_like(radial)
        )
        scale = np.sqrt(robust)[:, None]
        design = (jacobian * scale[:, :, None]).reshape(-1, 6)
        target = (residual * scale).reshape(-1)
        information = design.T @ design
        information += np.eye(6, dtype=np.float64) * float(damping)
        try:
            step = np.linalg.solve(information, design.T @ target)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ (design.T @ target)
        rvec += step[:3]
        tvec += step[3:]
        if np.linalg.norm(step) < 1e-9:
            break

    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(rvec)[0]
    refined[:3, 3] = tvec
    if not np.isfinite(refined).all():
        return pose, valid_inliers
    return refined, selected


def cal_pose_error(pred_w2c, gt_w2c):
    """
    Calculate the pose error between the predicted pose and the ground truth pose.
    """
    pred_R = pred_w2c[:3, :3]
    pred_t = np.linalg.inv(pred_w2c)[:3, -1]
    gt_R = gt_w2c[:3, :3]
    gt_t = np.linalg.inv(gt_w2c)[:3, -1]

    # calculate angle error
    r_err = np.matmul(gt_R, np.transpose(pred_R))
    r_err = cv2.Rodrigues(r_err)[0]
    # Extract the angle.
    ae = np.linalg.norm(r_err) * 180 / math.pi

    # calculate translation error
    te = np.linalg.norm(pred_t - gt_t) * 100

    return ae, te


def compute_reprojection_error(points_3D, points_2D, camera_matrix, w2c):
    """
    Compute the reprojection error between the 3D points and the 2D points.
    """
    projection_matrix = camera_matrix @ w2c[:3, :]
    projected_points = (
        projection_matrix
        @ torch.cat(
            [points_3D, torch.ones((points_3D.shape[0], 1), device=points_3D.device)],
            dim=1,
        ).t()
    )
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_points = projected_points.t()
    reprojection_error = torch.linalg.norm(points_2D - projected_points, dim=1)
    return reprojection_error.mean()


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(views, focal=30, zrate=0.5, rots=2, N=120):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    # poses = np.stack([np.concatenate([view.R.T, view.T[:, None]], 1) for view in views], 0)
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses


def spherify_poses(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_pose = np.eye(4)
        render_pose[:3] = p
        # render_pose[:3, 1:3] *= -1
        new_poses.append(render_pose)

    new_poses = np.stack(new_poses, 0)
    print(new_poses.shape)
    return new_poses
