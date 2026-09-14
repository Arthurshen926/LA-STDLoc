"""Mapping-only geometry evidence, not a certificate of real-world accuracy."""
from __future__ import annotations

import numpy as np


def crossfit_geometry(uv, intrinsics, poses, anchor_xyz, *, rendered_depth=None,
                      maximum_prediction_px=2.0, minimum_parallax_deg=1.0):
    """Fit spatially separated camera groups and predict held-out observations.

    No GT, descriptor ranking, map updates, or Gaussian-depth constraints.
    A consistent result cannot rule out a common error in the frozen prior.
    Short/degenerate groups are unknown, never automatically invalid.
    """
    uv, K, T, xyz = [np.asarray(x, dtype=np.float64)
                      for x in (uv, intrinsics, poses, anchor_xyz)]
    n = len(uv)
    if uv.shape != (n, 2) or K.shape != (n, 3, 3) or T.shape != (n, 4, 4) or xyz.shape != (3,):
        raise ValueError("geometry observation shapes do not align")
    if not all(np.isfinite(x).all() for x in (uv, K, T, xyz)):
        raise ValueError("nonfinite geometry input")
    if maximum_prediction_px <= 0 or minimum_parallax_deg <= 0:
        raise ValueError("thresholds must be positive")
    result = {"status": "unknown", "reason": "insufficient_independent_observations",
              "observation_count": n, "absolute_accuracy_certified": False}

    def project(point, rows):
        camera = np.einsum("nij,j->ni", T[rows, :3, :3], point) + T[rows, :3, 3]
        homogeneous = np.einsum("nij,nj->ni", K[rows], camera)
        z = camera[:, 2]
        pixels = homogeneous[:, :2] / np.where(z > 1e-8, homogeneous[:, 2], 1.)[:, None]
        error = np.linalg.norm(pixels - uv[rows], axis=1)
        return np.where(z > 1e-8, error, np.inf), z

    original_error, original_depth = project(xyz, np.arange(n))
    result["original_reprojection_median_px"] = float(np.median(original_error)) if n else None
    if rendered_depth is not None:
        depth = np.asarray(rendered_depth, dtype=np.float64)
        if depth.shape != (n,):
            raise ValueError("render depth rows do not align")
        valid = np.isfinite(depth) & (depth > 0) & (original_depth > 0)
        discrepancy = (original_depth[valid] - depth[valid]) / depth[valid]
        result["surface_evidence_count"] = int(valid.sum())
        result["surface_relative_bias_median"] = float(np.median(discrepancy)) if len(discrepancy) else None
        result["surface_relative_disagreement_mad"] = float(np.median(np.abs(discrepancy - np.median(discrepancy)))) if len(discrepancy) else None
        result["surface_evidence_role"] = "diagnostic_only_not_a_rejection_vote"
    if n < 6:
        return result
    R = T[:, :3, :3]
    centers = -np.einsum("nji,nj->ni", R, T[:, :3, 3])
    # Group by the dominant camera-center extent, not alternating neighboring
    # views (which would put highly correlated cameras into both groups).
    centered = centers - centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(centered @ axis, kind="stable")
    groups = [order[:n // 2], order[n // 2:]]
    bearings = np.linalg.solve(K, np.c_[uv, np.ones(n)][..., None])[..., 0]
    rays = np.einsum("nji,nj->ni", R, bearings)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    points, conditions, parallaxes = [], [], []
    for rows in groups:
        directions = rays[rows]
        projector = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
        normal = projector.sum(axis=0)
        eig = np.linalg.eigvalsh(normal)
        condition = float(eig[-1] / max(eig[0], 1e-15))
        conditions.append(condition)
        cosine = np.clip(directions @ directions.T, -1., 1.)
        angle = np.degrees(np.arccos(cosine[np.triu_indices(len(rows), 1)]))
        parallax = float(np.quantile(angle, .75))
        parallaxes.append(parallax)
        if eig[0] <= 1e-10 or condition > 1e6 or parallax < minimum_parallax_deg:
            result.update(reason="insufficient_group_geometry", group_condition=conditions,
                          group_parallax_p75_deg=parallaxes)
            return result
        points.append(np.linalg.solve(normal, np.einsum("nij,nj->i", projector, centers[rows])))
    heldout = [project(points[0], groups[1])[0], project(points[1], groups[0])[0]]
    medians = [float(np.median(e)) for e in heldout]
    cheirality_ok = all(np.isfinite(e).all() for e in heldout)
    consistent = cheirality_ok and max(medians) <= maximum_prediction_px
    result.update(status="internally_consistent" if consistent else "unstable",
                  reason="cross_group_prediction" if cheirality_ok else "heldout_cheirality",
                  group_condition=conditions, group_parallax_p75_deg=parallaxes,
                  heldout_median_px=medians, group_xyz_disagreement_m=float(np.linalg.norm(points[0] - points[1])),
                  group_sizes=[len(g) for g in groups])
    return result
