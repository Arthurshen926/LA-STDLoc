from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from localization_training.progressive_coreset import (
    build_surface_groups,
    group_representatives,
)


def quaternion_to_rotation_matrix(quaternion):
    quaternion = torch.as_tensor(quaternion)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape [N, 4] in wxyz order")
    q = F.normalize(quaternion, dim=1)
    w, x, y, z = q.unbind(dim=1)
    matrix = q.new_empty((q.shape[0], 3, 3))
    matrix[:, 0, 0] = 1 - 2 * (y.square() + z.square())
    matrix[:, 0, 1] = 2 * (x * y - w * z)
    matrix[:, 0, 2] = 2 * (x * z + w * y)
    matrix[:, 1, 0] = 2 * (x * y + w * z)
    matrix[:, 1, 1] = 1 - 2 * (x.square() + z.square())
    matrix[:, 1, 2] = 2 * (y * z - w * x)
    matrix[:, 2, 0] = 2 * (x * z - w * y)
    matrix[:, 2, 1] = 2 * (y * z + w * x)
    matrix[:, 2, 2] = 1 - 2 * (x.square() + y.square())
    return matrix


def materialize_bounded_surface_anchors(
    base_xyz,
    base_rotation,
    raw_offset,
    *,
    tangent_bound_m,
    normal_bound_m,
):
    """Materialize meter-bounded tangent/tangent/normal localization anchors.

    The tangent bound is a bound on the *total* displacement in the local
    tangent plane, rather than an independent bound on each tangent component.
    This keeps a nominal ``3 mm`` BA update inside a 3 mm disk instead of
    allowing a diagonal displacement of ``sqrt(2) * 3 mm``.
    """
    base_xyz = torch.as_tensor(base_xyz)
    raw_offset = torch.as_tensor(
        raw_offset, device=base_xyz.device, dtype=base_xyz.dtype
    )
    if base_xyz.ndim != 2 or base_xyz.shape[1] != 3:
        raise ValueError("base_xyz must have shape [N, 3]")
    if raw_offset.shape != base_xyz.shape:
        raise ValueError("raw_offset must have shape [N, 3]")
    rotation = quaternion_to_rotation_matrix(
        torch.as_tensor(
            base_rotation, device=base_xyz.device, dtype=base_xyz.dtype
        )
    )
    tangent, normal = bounded_surface_local_offsets(
        raw_offset,
        tangent_bound_m=tangent_bound_m,
        normal_bound_m=normal_bound_m,
    )
    return (
        base_xyz
        + rotation[:, :, 0] * tangent[:, 0:1]
        + rotation[:, :, 1] * tangent[:, 1:2]
        + rotation[:, :, 2] * normal
    )


def bounded_surface_local_offsets(
    raw_offset,
    *,
    tangent_bound_m,
    normal_bound_m,
    eps=1e-8,
):
    """Return local tangent and normal offsets with explicit metric bounds."""
    raw_offset = torch.as_tensor(raw_offset)
    if raw_offset.ndim != 2 or raw_offset.shape[1] != 3:
        raise ValueError("raw_offset must have shape [N, 3]")
    tangent_raw = raw_offset[:, :2]
    tangent_norm = torch.linalg.norm(tangent_raw, dim=1, keepdim=True)
    tangent_radius = torch.tanh(tangent_norm) * float(tangent_bound_m)
    # ``tanh(r) / r`` has a finite limit of one at the origin.  Keeping the
    # clamped denominator alone makes the evaluated scale zero at ``r == 0``
    # and therefore kills both tangent gradients at the usual zero-offset BA
    # initialization.  Materialize that analytic limit explicitly so a
    # tangent reprojection residual can start moving a surface anchor.
    tangent_scale = torch.where(
        tangent_norm > float(eps),
        tangent_radius / tangent_norm.clamp_min(float(eps)),
        torch.full_like(tangent_norm, float(tangent_bound_m)),
    )
    tangent = tangent_raw * tangent_scale
    normal = torch.tanh(raw_offset[:, 2:3]) * float(normal_bound_m)
    return tangent, normal


def bounded_surface_regularization(raw_offset):
    raw_offset = torch.as_tensor(raw_offset)
    if raw_offset.numel() == 0:
        return raw_offset.sum() * 0.0
    return torch.tanh(raw_offset).square().mean()


def validate_surface_anchor_resume_bounds(
    initial_state,
    *,
    tangent_bound_m,
    normal_bound_m,
):
    """Reject a continuation that would reinterpret a saved raw BA offset.

    ``raw_anchor_offset`` is expressed in the unbounded coordinates of the
    tangent/normal ``tanh`` parameterization. It is therefore not a metric
    displacement by itself: changing either bound while reusing a nonzero raw
    tensor moves landmarks even when a descriptor-only stage has no geometry
    loss. A continuation must preserve the parameterization exactly.

    A zero tensor is deliberately exempt. That permits the first bounded-BA
    stage to tighten the default bootstrap bounds before it has moved any
    surface anchor.
    """
    if not isinstance(initial_state, dict):
        return
    if not bool(initial_state.get("_raw_anchor_offset_alignment_valid", False)):
        return
    raw_offset = initial_state.get("raw_anchor_offset")
    if raw_offset is None:
        return
    raw_offset = torch.as_tensor(raw_offset, dtype=torch.float32)
    if raw_offset.numel() == 0 or not bool(torch.isfinite(raw_offset).all().item()):
        return
    if float(raw_offset.abs().max().item()) <= 1e-12:
        return

    prior_config = initial_state.get("config")
    if not isinstance(prior_config, dict):
        raise ValueError(
            "Initial state has nonzero raw_anchor_offset but no surface-anchor "
            "bound metadata; refusing to reinterpret its geometry."
        )
    try:
        prior_tangent_bound = float(prior_config["tangent_bound_m"])
        prior_normal_bound = float(prior_config["normal_bound_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Initial state has nonzero raw_anchor_offset but is missing "
            "tangent_bound_m/normal_bound_m metadata."
        ) from exc

    tangent_bound_m = float(tangent_bound_m)
    normal_bound_m = float(normal_bound_m)
    bounds_match = math.isclose(
        prior_tangent_bound,
        tangent_bound_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        prior_normal_bound,
        normal_bound_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if not bounds_match:
        raise ValueError(
            "Initial state has nonzero raw_anchor_offset encoded with "
            f"tangent_bound_m={prior_tangent_bound:g}, "
            f"normal_bound_m={prior_normal_bound:g}, but this run requested "
            f"tangent_bound_m={tangent_bound_m:g}, "
            f"normal_bound_m={normal_bound_m:g}. Reuse the saved bounds or "
            "explicitly reparameterize the anchor offsets."
        )


@dataclass
class GeometricScaffold:
    indices: torch.Tensor
    voxel_size: float
    group_count: int
    eligible_count: int
    diagnostics: dict


def _geometric_representatives(xyz, normals, voxel_size, normal_bins):
    group_ids, group_count = build_surface_groups(
        xyz,
        normals=normals,
        voxel_size=voxel_size,
        normal_bins=normal_bins,
    )
    centroid_sum = xyz.new_zeros((group_count, 3))
    centroid_count = xyz.new_zeros(group_count)
    centroid_sum.index_add_(0, group_ids, xyz)
    centroid_count.index_add_(0, group_ids, torch.ones_like(group_ids, dtype=xyz.dtype))
    centroid = centroid_sum / centroid_count[:, None].clamp_min(1.0)
    distance = torch.linalg.norm(xyz - centroid[group_ids], dim=1)
    return group_representatives(-distance, group_ids, group_count), group_count


@torch.no_grad()
def build_pure_geometric_scaffold(
    xyz,
    rotations,
    budget,
    *,
    eligible=None,
    normal_bins=6,
    voxel_size=0.0,
    search_steps=14,
    seed=2026,
):
    """Build an exact-size deterministic scaffold using geometry and normals only.

    The adaptive voxel search deliberately does not inspect descriptors, detector
    scores, poses, or localization labels. A fixed random projection is used only
    to spread an over-complete set of surface-patch medoids to the exact budget.
    """
    xyz = torch.as_tensor(xyz).detach().float()
    rotations = torch.as_tensor(rotations, device=xyz.device).detach().float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if rotations.shape != (xyz.shape[0], 4):
        raise ValueError("rotations must have shape [N, 4]")
    finite = torch.isfinite(xyz).all(dim=1) & torch.isfinite(rotations).all(dim=1)
    if eligible is not None:
        finite &= torch.as_tensor(eligible, device=xyz.device, dtype=torch.bool).reshape(-1)
    eligible_indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
    if eligible_indices.numel() == 0:
        raise ValueError("No finite primitives are eligible for the geometric scaffold")
    budget = min(max(int(budget), 1), int(eligible_indices.numel()))
    points = xyz[eligible_indices]
    normals = quaternion_to_rotation_matrix(rotations[eligible_indices])[:, :, 2]

    extent = (points.quantile(0.99, dim=0) - points.quantile(0.01, dim=0)).clamp_min(1e-6)
    scene_scale = float(torch.linalg.norm(extent).item())
    if float(voxel_size) > 0.0:
        selected_voxel_size = float(voxel_size)
        representatives, group_count = _geometric_representatives(
            points, normals, selected_voxel_size, normal_bins
        )
    else:
        lower = scene_scale / 10000.0
        upper = scene_scale
        representatives = torch.arange(points.shape[0], device=points.device)
        group_count = int(representatives.numel())
        selected_voxel_size = lower
        best = None
        for _ in range(max(int(search_steps), 1)):
            trial = (lower * upper) ** 0.5
            trial_representatives, trial_count = _geometric_representatives(
                points, normals, trial, normal_bins
            )
            if trial_count >= budget:
                best = (trial, trial_representatives, trial_count)
                lower = trial
            else:
                upper = trial
        if best is not None:
            selected_voxel_size, representatives, group_count = best

    if representatives.numel() > budget:
        candidate_xyz = points[representatives]
        candidate_normals = normals[representatives]
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        projection = torch.randn((6,), generator=generator)
        projection = projection.to(device=points.device, dtype=points.dtype)
        normalized_xyz = (candidate_xyz - candidate_xyz.median(dim=0).values) / extent
        key = torch.cat([normalized_xyz, candidate_normals], dim=1) @ projection
        order = torch.argsort(key, stable=True)
        positions = torch.div(
            torch.arange(budget, device=points.device, dtype=torch.long)
            * int(order.numel()),
            budget,
            rounding_mode="floor",
        )
        representatives = representatives[order[positions]]
    elif representatives.numel() < budget:
        selected_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        selected_mask[representatives] = True
        remaining = torch.nonzero(~selected_mask, as_tuple=False).reshape(-1)
        fill_count = budget - int(representatives.numel())
        fill_key = (
            (points[remaining] / max(float(selected_voxel_size), 1e-8))
            .sin()
            .sum(dim=1)
        )
        fill = remaining[torch.argsort(fill_key, stable=True)[:fill_count]]
        representatives = torch.cat([representatives, fill])

    indices = eligible_indices[representatives].sort().values
    if indices.numel() != budget or torch.unique(indices).numel() != budget:
        raise RuntimeError("Geometric scaffold did not produce the requested unique budget")
    diagnostics = {
        "mode": "pure_geometry_normal_aware_voxel_medoid",
        "budget": budget,
        "eligible_count": int(eligible_indices.numel()),
        "voxel_size": float(selected_voxel_size),
        "normal_bins": int(normal_bins),
        "surface_group_count": int(group_count),
        "scene_scale": scene_scale,
        "seed": int(seed),
    }
    return GeometricScaffold(
        indices=indices,
        voxel_size=float(selected_voxel_size),
        group_count=int(group_count),
        eligible_count=int(eligible_indices.numel()),
        diagnostics=diagnostics,
    )
