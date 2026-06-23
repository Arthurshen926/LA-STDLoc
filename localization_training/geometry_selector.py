from dataclasses import dataclass

import torch


@dataclass
class GeometryBalancedSelector:
    image_width: int
    image_height: int
    grid_rows: int = 4
    grid_cols: int = 4
    max_per_cell: int = 64
    voxel_size: float = 0.25
    max_per_voxel: int = 64
    max_matches: int = 0
    post_max_matches: int = 0
    post_candidate_pool: int = 1024
    post_regularization: float = 1e-4
    post_score_weight: float = 1e-3

    def select(self, p2d, p3d, scores=None):
        p2d = torch.as_tensor(p2d, dtype=torch.float32)
        p3d = torch.as_tensor(p3d, dtype=torch.float32)
        if p2d.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=p2d.device)
        if p2d.ndim != 2 or p2d.shape[1] != 2:
            raise ValueError(f"p2d must have shape [N, 2], got {tuple(p2d.shape)}.")
        if p3d.ndim != 2 or p3d.shape[1] != 3 or p3d.shape[0] != p2d.shape[0]:
            raise ValueError(f"p3d must have shape [N, 3] and match p2d rows, got {tuple(p3d.shape)}.")
        if scores is None:
            scores = torch.arange(p2d.shape[0], 0, -1, dtype=torch.float32, device=p2d.device)
        else:
            scores = torch.as_tensor(scores, dtype=torch.float32, device=p2d.device).reshape(-1)
            if scores.shape[0] != p2d.shape[0]:
                raise ValueError(f"scores must have one value per correspondence, got {scores.shape[0]} for {p2d.shape[0]}.")

        order = torch.argsort(scores, descending=True)
        if int(self.max_matches) <= 0:
            return order

        grid_counts = {}
        voxel_counts = {}
        selected = []
        width = max(float(self.image_width), 1.0)
        height = max(float(self.image_height), 1.0)
        rows = max(int(self.grid_rows), 1)
        cols = max(int(self.grid_cols), 1)
        max_per_cell = max(int(self.max_per_cell), 1)
        max_per_voxel = max(int(self.max_per_voxel), 1)
        voxel_size = max(float(self.voxel_size), 1e-8)
        max_matches = int(self.max_matches)

        for idx in order.tolist():
            x = float(p2d[idx, 0].item())
            y = float(p2d[idx, 1].item())
            col = min(cols - 1, max(0, int(x / width * cols)))
            row = min(rows - 1, max(0, int(y / height * rows)))
            grid_key = (row, col)
            if grid_counts.get(grid_key, 0) >= max_per_cell:
                continue

            voxel = torch.floor(p3d[idx] / voxel_size).to(dtype=torch.long)
            voxel_key = tuple(int(v) for v in voxel.tolist())
            if voxel_counts.get(voxel_key, 0) >= max_per_voxel:
                continue

            selected.append(idx)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
            voxel_counts[voxel_key] = voxel_counts.get(voxel_key, 0) + 1
            if len(selected) >= max_matches:
                break

        return torch.as_tensor(selected, dtype=torch.long, device=p2d.device)

    def select_pose_informative_inliers(self, p3d, pose_w2c, intrinsic, inliers, scores=None):
        p3d = torch.as_tensor(p3d, dtype=torch.float32)
        inliers = torch.as_tensor(inliers, dtype=torch.long, device=p3d.device).reshape(-1)
        if inliers.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=p3d.device)
        if int(self.post_max_matches) <= 0 or inliers.numel() <= int(self.post_max_matches):
            return inliers
        if p3d.ndim != 2 or p3d.shape[1] != 3:
            raise ValueError(f"p3d must have shape [N, 3], got {tuple(p3d.shape)}.")

        pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float32, device=p3d.device)
        intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32, device=p3d.device)
        if pose_w2c.shape == (3, 4):
            bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=p3d.device, dtype=torch.float32)
            pose_w2c = torch.cat([pose_w2c, bottom], dim=0)
        if pose_w2c.shape != (4, 4):
            raise ValueError(f"pose_w2c must have shape [4, 4] or [3, 4], got {tuple(pose_w2c.shape)}.")
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must have shape [3, 3], got {tuple(intrinsic.shape)}.")

        if scores is None:
            score_values = torch.zeros(inliers.numel(), dtype=torch.float32, device=p3d.device)
        else:
            scores = torch.as_tensor(scores, dtype=torch.float32, device=p3d.device).reshape(-1)
            if scores.shape[0] == p3d.shape[0]:
                score_values = scores[inliers]
            elif scores.shape[0] == inliers.shape[0]:
                score_values = scores
            else:
                raise ValueError(
                    "scores must have one value per correspondence or per inlier, "
                    f"got {scores.shape[0]} for {p3d.shape[0]} points and {inliers.shape[0]} inliers."
                )
            score_values = (score_values - score_values.mean()) / score_values.std().clamp_min(1e-6)

        ones = torch.ones((inliers.numel(), 1), dtype=torch.float32, device=p3d.device)
        homo = torch.cat([p3d[inliers], ones], dim=1)
        cam = (pose_w2c @ homo.T).T[:, :3]
        valid = cam[:, 2] > 1e-6
        if not bool(valid.all()):
            inliers = inliers[valid]
            cam = cam[valid]
            score_values = score_values[valid]
            if inliers.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=p3d.device)
            if inliers.numel() <= int(self.post_max_matches):
                return inliers

        pool = int(self.post_candidate_pool)
        if pool > 0 and inliers.numel() > pool:
            keep = torch.topk(score_values, k=pool, largest=True).indices
            inliers = inliers[keep]
            cam = cam[keep]
            score_values = score_values[keep]

        jacobians = _pose_projection_jacobians(cam, intrinsic)
        selected_local = []
        remaining = torch.arange(inliers.numel(), dtype=torch.long, device=p3d.device)
        information = torch.eye(6, dtype=torch.float32, device=p3d.device) * float(self.post_regularization)
        max_matches = min(int(self.post_max_matches), int(inliers.numel()))
        score_weight = float(self.post_score_weight)

        for step in range(max_matches):
            if remaining.numel() == 0:
                break
            if step == 0:
                best_pos = int(torch.argmax(score_values[remaining]).item())
            else:
                candidate_j = jacobians[remaining]
                candidate_info = information.unsqueeze(0) + torch.einsum("nri,nrj->nij", candidate_j, candidate_j)
                sign, logabsdet = torch.linalg.slogdet(candidate_info)
                values = logabsdet + score_weight * score_values[remaining]
                values = torch.where(sign > 0, values, torch.full_like(values, float("-inf")))
                best_pos = int(torch.argmax(values).item())
            best_idx = int(remaining[best_pos].item())
            selected_local.append(best_idx)
            j = jacobians[best_idx]
            information = information + j.T @ j
            remaining = torch.cat([remaining[:best_pos], remaining[best_pos + 1:]], dim=0)

        return inliers[torch.as_tensor(selected_local, dtype=torch.long, device=p3d.device)]


def _pose_projection_jacobians(points_camera, intrinsic):
    points_camera = torch.as_tensor(points_camera, dtype=torch.float32)
    intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32, device=points_camera.device)
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2].clamp_min(1e-6)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z

    jac = torch.zeros((points_camera.shape[0], 2, 6), dtype=torch.float32, device=points_camera.device)
    jac[:, 0, 0] = fx * inv_z
    jac[:, 0, 2] = -fx * x * inv_z2
    jac[:, 0, 3] = -fx * x * y * inv_z2
    jac[:, 0, 4] = fx * (1.0 + x * x * inv_z2)
    jac[:, 0, 5] = -fx * y * inv_z
    jac[:, 1, 1] = fy * inv_z
    jac[:, 1, 2] = -fy * y * inv_z2
    jac[:, 1, 3] = -fy * (1.0 + y * y * inv_z2)
    jac[:, 1, 4] = fy * x * y * inv_z2
    jac[:, 1, 5] = fy * x * inv_z
    return jac
