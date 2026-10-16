"""Shared-observation coordinate control (Table V and Fig. 6)."""

import gc
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from anygsloc.artifacts import SCHEMA, sha256, validate
from anygsloc.mapping.triangulation import (
    _camera_rays,
    _project,
    _weighted_ray_intersection,
)
from anygsloc.pipeline.io import complete, reference

CONFIGS = ("ray", "depth", "center")


def projected_centers(xyz, K, pose, width, height, keypoints, radius=2.0):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    keep = np.isfinite(camera).all(1) & (camera[:, 2] > 1e-6)
    ids = np.flatnonzero(keep)
    uvh = camera[ids] @ K.T
    uv = uvh[:, :2] / uvh[:, 2:3] - 0.5
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, ids = uv[valid], ids[valid]
    result = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    if not len(ids):
        return result
    pixels = np.floor(uv).astype(np.int64)
    bins = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((ids, camera[ids, 2], bins))
    _, first = np.unique(bins[order], return_index=True)
    winners = order[first]
    distance, nearest = cKDTree(uv[winners]).query(
        keypoints, distance_upper_bound=radius
    )
    matched = np.isfinite(distance)
    result[matched] = xyz[ids[winners[nearest[matched]]]]
    return result


def refit_ray(uv, K, poses):
    """Equal base weights; robust perpendicular-ray estimator on ALL input rows."""
    uv, K, poses = (torch.as_tensor(x, dtype=torch.float64) for x in (uv, K, poses))
    centers, directions = _camera_rays(uv, K, poses)
    point, _, _ = _weighted_ray_intersection(
        centers, directions, torch.ones(len(uv), dtype=torch.float64)
    )
    for _ in range(3):
        projected, z = _project(point, K, poses)
        residual = torch.linalg.norm(projected - uv, dim=1)
        weight = torch.minimum(
            torch.ones_like(residual), 2.0 / residual.clamp_min(1e-8)
        ) * (z > 0)
        point, _, _ = _weighted_ray_intersection(centers, directions, weight)
    _, z = _project(point, K, poses)
    if not torch.isfinite(point).all() or int((z > 0).sum()) < 3:
        raise ValueError("unusable common ray geometry")
    return point.numpy().astype(np.float32)


def depth_xyz(uv, depth, K, pose):
    pixel = np.asarray(uv, dtype=np.float64) + 0.5
    z = np.asarray(depth, dtype=np.float64)
    camera = np.column_stack(
        (
            (pixel[:, 0] - K[0, 2]) * z / K[0, 0],
            (pixel[:, 1] - K[1, 2]) * z / K[1, 1],
            z,
        )
    )
    return ((camera - pose[:3, 3]) @ pose[:3, :3]).astype(np.float32)


def coordinate_group(cp, dp, uv, K, poses):
    mask = np.isfinite(cp).all(1) & np.isfinite(dp).all(1)
    if mask.sum() < 3:
        return None
    ray = refit_ray(uv[mask], K[mask], poses[mask])
    center = torch.as_tensor(cp[mask]).median(0).values.numpy()
    depth = torch.as_tensor(dp[mask]).median(0).values.numpy()
    if not all(np.isfinite(x).all() for x in (ray, center, depth)):
        raise ValueError("nonfinite common coordinate")
    return mask, {"ray": ray, "depth": depth, "center": center}


def project_residual(point, uv, K, poses):
    camera = np.einsum("nij,j->ni", poses[:, :3, :3], point) + poses[:, :3, 3]
    p = np.einsum("nij,nj->ni", K, camera)
    valid = np.isfinite(p).all(1) & (camera[:, 2] > 0)
    out = np.full(len(uv), np.inf, dtype=np.float64)
    out[valid] = np.linalg.norm(p[valid, :2] / p[valid, 2, None] - uv[valid], axis=1)
    return out


def build_geometry(state_path, cache_path, ply_path, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    names = state["v6_mapping_query_names"]
    if data["mapping_names"] != names or set(names) & set(data.get("test_names", [])):
        raise ValueError("mapping observation registry differs")
    s = {
        "ply": {"path": str(ply_path)},
        "cache": reference(cache_path),
        "full": reference(state_path),
    }
    scene = out.parent.name
    verts = PlyData.read(s["ply"]["path"])["vertex"]
    gaussian_xyz = np.column_stack([verts[k] for k in ("x", "y", "z")]).astype(
        np.float32
    )
    centers = []
    depths = []
    pixels = []
    Ks = []
    poses = []
    for j, name in enumerate(names):
        r = data["queries"][name]
        uv = np.asarray(r["native_keypoints"], dtype=np.float32)
        K = np.asarray(r["native_K"], dtype=np.float32)
        pose = np.asarray(r["pose_w2c"], dtype=np.float32)
        h, w = map(int, r["native_input_hw"])
        z = np.asarray(r["native_depth_at_keypoints"], dtype=np.float32).reshape(-1)
        alpha = np.asarray(r["native_alpha_at_keypoints"], dtype=np.float32).reshape(-1)
        if len(z) != len(uv) or len(alpha) != len(uv):
            raise ValueError("unaligned sampled buffers")
        valid = np.isfinite(z) & (z > 0) & np.isfinite(alpha) & (alpha >= 0.05)
        dp = depth_xyz(uv, z, K, pose)
        cp = projected_centers(gaussian_xyz, K, pose, w, h, uv)
        dp[~valid] = np.nan
        cp[~valid] = np.nan
        depths.append(dp)
        centers.append(cp)
        pixels.append(uv + 0.5)
        Ks.append(K)
        poses.append(pose)
        if j % 100 == 0:
            print("T5_PROJECT", scene, j + 1, len(names), flush=True)
    del data, verts, gaussian_xyz
    gc.collect()
    Ks = np.stack(Ks)
    poses = np.stack(poses)
    csr = state["projective_anchor_observations"]
    off = np.asarray(csr["observation_offsets"])
    qs = np.asarray(csr["query_indices"])
    ks = np.asarray(csr["keypoint_indices"])
    retained = []
    common_q = []
    common_k = []
    offsets = [0]
    xyz = {k: [] for k in CONFIGS}
    residuals = {k: [] for k in CONFIGS}
    eligible = 0
    ray_rejected = 0
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        q = qs[a:b]
        key = ks[a:b]
        cp = np.stack([centers[v][k] for v, k in zip(q, key)])
        dp = np.stack([depths[v][k] for v, k in zip(q, key)])
        uv = np.stack([pixels[v][k] for v, k in zip(q, key)])
        valid = np.isfinite(cp).all(1) & np.isfinite(dp).all(1)
        if valid.sum() < 3:
            continue
        eligible += 1
        try:
            item = coordinate_group(cp, dp, uv, Ks[q], poses[q])
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            ray_rejected += 1
            continue
        mask, coords = item
        for mode in CONFIGS:
            xyz[mode].append(coords[mode])
            residuals[mode].append(
                project_residual(coords[mode], uv[mask], Ks[q[mask]], poses[q[mask]])
            )
        retained.append(i)
        common_q.extend(q[mask].tolist())
        common_k.extend(key[mask].tolist())
        offsets.append(len(common_q))
        if i % 10000 == 0:
            print("T5_FIT", scene, i, len(off) - 1, flush=True)
    if not retained:
        raise ValueError("empty common support")
    out.mkdir(parents=True)
    with (out / "common_observations.npz").open("xb") as f:
        np.savez(
            f,
            source_rows=np.asarray(retained, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
            query_indices=np.asarray(common_q, dtype=np.int64),
            keypoint_indices=np.asarray(common_k, dtype=np.int64),
        )
    features = state["anchor_features"][retained].float().clone()
    refs = {}
    for mode in CONFIGS:
        folder = out / mode
        folder.mkdir()
        path = folder / "deployment_map.pt"
        candidate = {
            "schema": SCHEMA,
            "anchor_ids": torch.as_tensor(retained, dtype=torch.long),
            "anchor_xyz": torch.as_tensor(np.stack(xyz[mode]), dtype=torch.float32),
            "anchor_features": features.clone(),
            "provenance": {
                "uses_test_queries": False,
                "uses_source_mapping_rgb": False,
                "experiment": "shared_observation_geometry",
                "configuration": mode,
                "protocol_sha256": None,
                "source_full_sha256": s["full"]["sha256"],
                "cache_sha256": s["cache"]["sha256"],
                "common_observations_sha256": sha256(out / "common_observations.npz"),
                "coordinate_only": True,
                "selector_enabled": False,
            },
        }
        validate(candidate)
        torch.save(candidate, path)
        refs[mode] = dict(reference(path), count=len(retained))
        with (folder / "mapping_residuals.npy").open("xb") as f:
            np.save(f, np.concatenate(residuals[mode]))

    complete(
        out,
        ["common_observations.npz"]
        + [f"{m}/deployment_map.pt" for m in CONFIGS]
        + [f"{m}/mapping_residuals.npy" for m in CONFIGS],
        common_count=len(retained),
        common_observation_count=len(common_q),
        common_eligible=eligible,
        shared_ray_rejections=ray_rejected,
    )
