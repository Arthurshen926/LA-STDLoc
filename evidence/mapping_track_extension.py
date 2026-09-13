"""Bounded mapping-only geometry extension; descriptors remain frozen."""
import numpy as np
from scipy.optimize import least_squares
from evidence.geometry_crossfit import crossfit_geometry


def extend_geometry(uv, K, T, xyz, added_uv, added_K, added_T):
    """Require additional measured rays to pass cross-group checks.

    Returns a proposal or None. This does not synthesize new pixel observations
    by projecting the old XYZ, and never consumes real query/GT evidence.
    """
    uv, K, T, xyz = [np.asarray(x, dtype=float) for x in (uv, K, T, xyz)]
    a = len(added_uv)
    if a < 1 or a > 2:
        raise ValueError('one or two independent measured observations required')
    all_uv = np.concatenate([uv, np.asarray(added_uv)])
    all_K = np.concatenate([K, np.asarray(added_K)])
    all_T = np.concatenate([T, np.asarray(added_T)])
    before = crossfit_geometry(uv, K, T, xyz)
    if before['status'] != 'unstable': return None
    after = crossfit_geometry(all_uv, all_K, all_T, xyz)
    if after['status'] != 'internally_consistent': return None

    def residual(point):
        cam = np.einsum('nij,j->ni', all_T[:, :3, :3], point) + all_T[:, :3, 3]
        h = np.einsum('nij,nj->ni', all_K, cam)
        return (h[:, :2] / np.maximum(h[:, 2:], 1e-8) - all_uv).reshape(-1)
    fit = least_squares(residual, xyz, loss='huber', f_scale=2., max_nfev=30)
    if not fit.success or not np.isfinite(fit.x).all(): return None
    camera = np.einsum('nij,j->ni', all_T[:, :3, :3], fit.x) + all_T[:, :3, 3]
    if np.any(camera[:, 2] <= 0): return None
    shift = float(np.linalg.norm(fit.x - xyz))
    if shift > .02 * np.median(camera[:, 2]): return None
    original_errors = np.linalg.norm(residual(fit.x).reshape(-1, 2)[:len(uv)], axis=1)
    if np.median(original_errors) > min(2., before['original_reprojection_median_px'] + .25): return None
    normal = fit.jac.T @ fit.jac
    if not np.isfinite(normal).all() or np.linalg.cond(normal) > 1e6: return None
    variance = max(1., float(np.median(original_errors)) ** 2)
    return {'xyz': fit.x, 'covariance': np.linalg.inv(normal) * variance,
            'shift_m': shift, 'before': before, 'after': after}
