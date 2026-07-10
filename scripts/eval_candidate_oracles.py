import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from utils.pose_utils import cal_pose_error


def project_points(points, K, pose_w2c):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (np.asarray(pose_w2c, dtype=np.float64) @ points_h.T)[:3].T
    depth = camera[:, 2]
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (depth > 1e-8)
    uv[valid, 0] = K[0, 0] * camera[valid, 0] / depth[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * camera[valid, 1] / depth[valid] + K[1, 2]
    return uv, depth, valid


def deterministic_pnp(p2d, p3d, K):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if p2d.shape[0] < 4:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return np.eye(4, dtype=np.float64), False
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.reshape(3)
    return pose, True


def pose_information(points, K, pose_w2c, damping=1e-6):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 6:
        return {
            "full_logdet": float("nan"),
            "full_condition": float("nan"),
            "translation_logdet": float("nan"),
            "translation_condition": float("nan"),
            "translation_min_eig": float("nan"),
        }
    _, _, valid = project_points(points, K, pose_w2c)
    points = points[valid]
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (np.asarray(pose_w2c, dtype=np.float64) @ points_h.T)[:3].T
    fx, fy = float(K[0, 0]), float(K[1, 1])
    H = np.eye(6, dtype=np.float64) * float(damping)
    for x, y, z in camera:
        dproj = np.array(
            [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )
        jacobian = dproj @ np.concatenate([np.eye(3), -skew], axis=1)
        H += jacobian.T @ jacobian
    H_tt = H[:3, :3]
    H_tr = H[:3, 3:]
    H_rr = H[3:, 3:]
    translation = H_tt - H_tr @ np.linalg.pinv(
        H_rr + np.eye(3, dtype=np.float64) * float(damping)
    ) @ H_tr.T
    translation = 0.5 * (translation + translation.T)
    full_eig = np.linalg.eigvalsh(H).clip(1e-12, None)
    trans_eig = np.linalg.eigvalsh(translation).clip(1e-12, None)
    return {
        "full_logdet": float(np.log(full_eig).sum()),
        "full_condition": float(full_eig[-1] / full_eig[0]),
        "translation_logdet": float(np.log(trans_eig).sum()),
        "translation_condition": float(trans_eig[-1] / trans_eig[0]),
        "translation_min_eig": float(trans_eig[0]),
    }


def visibility_filter(projected, depth, valid, width, height, abs_tol=0.25, rel_tol=0.02):
    in_image = (
        valid
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    candidate = np.flatnonzero(in_image)
    if candidate.size == 0:
        return in_image
    x = np.clip(np.floor(projected[candidate, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(projected[candidate, 1]).astype(np.int64), 0, height - 1)
    cell = y * width + x
    min_depth = np.full(width * height, np.inf, dtype=np.float64)
    np.minimum.at(min_depth, cell, depth[candidate])
    tolerance = np.maximum(float(abs_tol), float(rel_tol) * min_depth[cell])
    visible_candidate = depth[candidate] <= min_depth[cell] + tolerance
    visible = np.zeros_like(in_image)
    visible[candidate[visible_candidate]] = True
    return visible


def oracle_assign_detector_points(p2d, bank_xyz, K, pose_w2c, width, height, radius_px):
    projected, depth, valid = project_points(bank_xyz, K, pose_w2c)
    visible = visibility_filter(projected, depth, valid, width, height)
    visible_idx = np.flatnonzero(visible)
    if visible_idx.size == 0:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty(0)
    tree = cKDTree(projected[visible_idx])
    distance, local_idx = tree.query(np.asarray(p2d, dtype=np.float64) + 0.5, k=1)
    keep = np.isfinite(distance) & (distance <= float(radius_px))
    return (
        np.asarray(p2d, dtype=np.float64)[keep],
        bank_xyz[visible_idx[local_idx[keep]]],
        distance[keep],
    )


def balanced_subset(p2d, p3d, scores, K, pose_w2c, width, height, max_count=512):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if p2d.shape[0] <= int(max_count):
        return np.arange(p2d.shape[0], dtype=np.int64)
    _, depth, _ = project_points(p3d, K, pose_w2c)
    boundaries = np.quantile(depth, [0.25, 0.5, 0.75])
    depth_bin = np.digitize(depth, boundaries)
    grid_x = np.clip(np.floor(p2d[:, 0] / max(width, 1) * 4).astype(np.int64), 0, 3)
    grid_y = np.clip(np.floor(p2d[:, 1] / max(height, 1) * 4).astype(np.int64), 0, 3)
    grid_bin = grid_y * 4 + grid_x
    voxel = np.floor(p3d / 0.25).astype(np.int64)
    order = np.argsort(-scores, kind="stable")
    grid_cap = max(1, int(np.ceil(max_count / 16.0 * 1.5)))
    depth_cap = max(1, int(np.ceil(max_count / 4.0 * 1.5)))
    grid_count = np.zeros(16, dtype=np.int64)
    depth_count = np.zeros(4, dtype=np.int64)
    voxel_count = {}
    selected = []
    for idx in order:
        voxel_key = tuple(voxel[idx].tolist())
        if grid_count[grid_bin[idx]] >= grid_cap:
            continue
        if depth_count[depth_bin[idx]] >= depth_cap:
            continue
        if voxel_count.get(voxel_key, 0) >= 2:
            continue
        selected.append(int(idx))
        grid_count[grid_bin[idx]] += 1
        depth_count[depth_bin[idx]] += 1
        voxel_count[voxel_key] = voxel_count.get(voxel_key, 0) + 1
        if len(selected) >= int(max_count):
            break
    if len(selected) < int(max_count):
        selected_set = set(selected)
        selected.extend(int(idx) for idx in order if int(idx) not in selected_set)
    return np.asarray(selected[: int(max_count)], dtype=np.int64)


def pose_result(prefix, p2d, p3d, K, pose_gt):
    pose, success = deterministic_pnp(np.asarray(p2d) + 0.5, p3d, K)
    ae, te = cal_pose_error(pose, pose_gt) if success else (float("inf"), float("inf"))
    result = {
        f"{prefix}_count": int(np.asarray(p2d).shape[0]),
        f"{prefix}_success": bool(success),
        f"{prefix}_ae_deg": float(ae),
        f"{prefix}_te_cm": float(te),
    }
    result.update({f"{prefix}_{key}": value for key, value in pose_information(p3d, K, pose_gt).items()})
    return result


def summarize(per_query):
    summary = {"query_count": len(per_query)}
    keys = sorted({key for item in per_query for key in item if key != "image_name"})
    for key in keys:
        values = [item[key] for item in per_query if isinstance(item.get(key), (int, float))]
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            summary[f"{key}_mean"] = float(finite.mean())
            summary[f"{key}_median"] = float(np.median(finite))
    for prefix in ("o2_detector_oracle", "o3_clean", "o4_balanced"):
        te = np.asarray([item.get(f"{prefix}_te_cm", np.inf) for item in per_query])
        ae = np.asarray([item.get(f"{prefix}_ae_deg", np.inf) for item in per_query])
        summary[f"{prefix}_r2"] = float(np.mean((te <= 2.0) & (ae <= 2.0)))
        summary[f"{prefix}_r5"] = float(np.mean((te <= 5.0) & (ae <= 5.0)))
    return summary


def evaluate(args):
    records = [json.loads(line) for line in Path(args.correspondences).read_text().splitlines() if line]
    if not records:
        raise ValueError("correspondence dump is empty")
    bank_xyz = np.unique(
        np.concatenate([np.asarray(record["p3d"], dtype=np.float32) for record in records], axis=0),
        axis=0,
    ).astype(np.float64)
    per_query = []
    for record in records:
        p2d = np.asarray(record["p2d"], dtype=np.float64)
        p3d = np.asarray(record["p3d"], dtype=np.float64)
        scores = np.asarray(record["scores"], dtype=np.float64)
        K = np.asarray(record["K"], dtype=np.float64)
        pose_gt = np.asarray(record["gt_pose_w2c"], dtype=np.float64)
        projected, depth, valid = project_points(p3d, K, pose_gt)
        error = np.linalg.norm(projected - (p2d + 0.5), axis=1)
        valid &= np.isfinite(error) & (depth > 0)
        positive = valid & (error < float(args.positive_radius_px))
        ambiguous = valid & (error >= float(args.positive_radius_px)) & (
            error <= float(args.negative_radius_px)
        )
        query = {
            "image_name": record["image_name"],
            "candidate_count": int(p2d.shape[0]),
            "candidate_positive_count": int(positive.sum()),
            "candidate_ambiguous_count": int(ambiguous.sum()),
            "candidate_precision_2px": float(positive.mean()) if positive.size else 0.0,
            "candidate_duplicate_landmark_rate": float(
                1.0 - np.unique(p3d, axis=0).shape[0] / max(p3d.shape[0], 1)
            ),
        }
        clean_p2d, clean_p3d, clean_scores = p2d[positive], p3d[positive], scores[positive]
        query.update(pose_result("o3_clean", clean_p2d, clean_p3d, K, pose_gt))
        selected = balanced_subset(
            clean_p2d,
            clean_p3d,
            clean_scores,
            K,
            pose_gt,
            int(record["width"]),
            int(record["height"]),
            max_count=args.balanced_count,
        )
        query.update(
            pose_result(
                "o4_balanced",
                clean_p2d[selected],
                clean_p3d[selected],
                K,
                pose_gt,
            )
        )
        oracle_p2d, oracle_p3d, oracle_error = oracle_assign_detector_points(
            p2d,
            bank_xyz,
            K,
            pose_gt,
            int(record["width"]),
            int(record["height"]),
            args.positive_radius_px,
        )
        query["o2_detector_matchable_rate"] = float(
            oracle_p2d.shape[0] / max(p2d.shape[0], 1)
        )
        query["o2_detector_oracle_reproj_mean_px"] = float(
            oracle_error.mean() if oracle_error.size else np.nan
        )
        query.update(pose_result("o2_detector_oracle", oracle_p2d, oracle_p3d, K, pose_gt))
        per_query.append(query)
    payload = {
        "config": {
            "correspondences": str(args.correspondences),
            "positive_radius_px": float(args.positive_radius_px),
            "negative_radius_px": float(args.negative_radius_px),
            "balanced_count": int(args.balanced_count),
            "oracle_bank_count": int(bank_xyz.shape[0]),
        },
        "summary": summarize(per_query),
        "queries": per_query,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correspondences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive_radius_px", type=float, default=2.0)
    parser.add_argument("--negative_radius_px", type=float, default=6.0)
    parser.add_argument("--balanced_count", type=int, default=512)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
