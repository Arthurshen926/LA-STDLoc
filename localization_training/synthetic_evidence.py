"""Artifact-filtered appearance evidence from frozen Gaussian RGB rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SyntheticEvidenceConfig:
    topk_keypoints: int = 2048
    positive_radius_px: float = 4.0
    positives_per_keypoint: int = 4
    minimum_alpha: float = 0.5
    absolute_depth_tolerance: float = 0.08
    relative_depth_tolerance: float = 0.03
    minimum_positive_pairs: int = 32
    minimum_matchable_rate: float = 0.02
    minimum_valid_fraction: float = 0.5
    minimum_valid_keypoint_fraction: float = 0.5
    minimum_alpha_coverage: float = 0.25
    minimum_visible_anchors: int = 64
    require_support_mask: bool = True


@dataclass(frozen=True)
class RenderQualityFilterConfig:
    reference_downsample: int = 4
    maximum_reference_residual: float = 0.18
    minimum_alpha: float = 0.5
    minimum_normal_norm: float = 0.5
    invalid_dilate_radius: int = 2


@dataclass
class RenderQualityMask:
    valid_mask: torch.Tensor
    support_mask: torch.Tensor
    invalid_score: torch.Tensor
    support_score: torch.Tensor
    channel_maps: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def valid_points(self, points_xy):
        return _points_in_mask(self.valid_mask, points_xy)

    def support_points(self, points_xy):
        return _points_in_mask(self.support_mask, points_xy)


def _points_in_mask(mask: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(points_xy, dtype=torch.float32)
    height, width = mask.shape
    x = torch.floor(points[:, 0]).long()
    y = torch.floor(points[:, 1]).long()
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep = torch.zeros(len(points), dtype=torch.bool)
    keep[inside] = mask.cpu()[y[inside], x[inside]]
    return keep


def _as_rgb(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 3 and value.shape[0] != 3:
        value = value.permute(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("render/reference RGB must be shaped 3xHxW")
    return value.clamp(0.0, 1.0)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.bool()
    return (
        F.max_pool2d(
            mask.float()[None, None],
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0]
        > 0
    )


def depth_warped_reference_residual(
    *,
    rendered_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    render_pose_w2c: torch.Tensor,
    render_K: torch.Tensor,
    reference_views: list[dict],
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare a rendered surface point with geometry-warped real views."""
    rendered = _as_rgb(rendered_rgb)
    height, width = rendered.shape[-2:]
    factor = max(int(downsample), 1)
    low_height = max(height // factor, 1)
    low_width = max(width // factor, 1)
    low_rgb = F.interpolate(
        rendered[None],
        size=(low_height, low_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    low_depth = F.interpolate(
        torch.as_tensor(rendered_depth).float().reshape(1, 1, height, width),
        size=(low_height, low_width),
        mode="nearest",
    )[0, 0]
    y, x = torch.meshgrid(
        torch.arange(low_height, dtype=torch.float32),
        torch.arange(low_width, dtype=torch.float32),
        indexing="ij",
    )
    x = (x + 0.5) * float(width) / float(low_width) - 0.5
    y = (y + 0.5) * float(height) / float(low_height) - 0.5
    K = torch.as_tensor(render_K).float()
    camera = torch.stack(
        (
            (x - K[0, 2]) / K[0, 0] * low_depth,
            (y - K[1, 2]) / K[1, 1] * low_depth,
            low_depth,
        ),
        dim=-1,
    ).reshape(-1, 3)
    pose = torch.as_tensor(render_pose_w2c).float()
    world = (camera - pose[:3, 3]) @ pose[:3, :3]
    residuals = []
    validities = []
    for reference in reference_views:
        image = _as_rgb(reference["rgb"])
        reference_height, reference_width = image.shape[-2:]
        reference_pose = torch.as_tensor(reference["pose_w2c"]).float()
        reference_K = torch.as_tensor(reference["K"]).float()
        reference_camera = (
            world @ reference_pose[:3, :3].T
            + reference_pose[:3, 3]
        )
        depth = reference_camera[:, 2]
        u = (
            reference_K[0, 0]
            * reference_camera[:, 0]
            / depth.clamp_min(1e-8)
            + reference_K[0, 2]
        )
        v = (
            reference_K[1, 1]
            * reference_camera[:, 1]
            / depth.clamp_min(1e-8)
            + reference_K[1, 2]
        )
        valid = (
            torch.isfinite(depth)
            & (depth > 0)
            & (u >= 0)
            & (u < reference_width)
            & (v >= 0)
            & (v < reference_height)
            & torch.isfinite(low_depth.reshape(-1))
            & (low_depth.reshape(-1) > 0)
        )
        grid = torch.stack(
            (
                (u + 0.5) / max(reference_width, 1) * 2.0 - 1.0,
                (v + 0.5) / max(reference_height, 1) * 2.0 - 1.0,
            ),
            dim=1,
        ).reshape(1, low_height, low_width, 2)
        sampled = F.grid_sample(
            image[None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        residual = (sampled - low_rgb).abs().mean(dim=0)
        residual[~valid.reshape(low_height, low_width)] = torch.inf
        residuals.append(residual)
        validities.append(valid.reshape(low_height, low_width))
    if not residuals:
        raise ValueError("depth-warped QA requires real reference views")
    stacked = torch.stack(residuals)
    residual = stacked.amin(dim=0)
    valid = torch.stack(validities).any(dim=0)
    residual = F.interpolate(
        residual.nan_to_num(posinf=1.0)[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    valid = (
        F.interpolate(
            valid.float()[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0]
        > 0
    )
    return residual, valid


def build_render_quality_mask(
    *,
    base_mask,
    rendered_rgb: torch.Tensor,
    reference_rgbs: list[torch.Tensor],
    alpha: torch.Tensor,
    rendered_depth: torch.Tensor,
    surface_normal: torch.Tensor,
    config: RenderQualityFilterConfig,
    render_pose_w2c: torch.Tensor | None = None,
    render_K: torch.Tensor | None = None,
    reference_views: list[dict] | None = None,
) -> RenderQualityMask:
    """Filter rendered regions using real-view continuity and raster quality."""
    rendered = _as_rgb(rendered_rgb)
    height, width = rendered.shape[-2:]
    if not reference_rgbs:
        raise ValueError("render quality filtering requires real reference RGB")
    references = []
    for value in reference_rgbs:
        value = _as_rgb(value)
        if value.shape[-2:] != (height, width):
            value = F.interpolate(
                value[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        references.append(value)
    use_warped = (
        render_pose_w2c is not None
        and render_K is not None
        and bool(reference_views)
    )
    if use_warped:
        residual, warp_valid = depth_warped_reference_residual(
            rendered_rgb=rendered,
            rendered_depth=rendered_depth,
            render_pose_w2c=render_pose_w2c,
            render_K=render_K,
            reference_views=list(reference_views or []),
            downsample=config.reference_downsample,
        )
        reference_valid = (
            warp_valid
            & (residual <= float(config.maximum_reference_residual))
        )
    else:
        factor = max(int(config.reference_downsample), 1)
        low_hw = (max(height // factor, 1), max(width // factor, 1))
        low_rendered = F.interpolate(
            rendered[None], size=low_hw, mode="bilinear", align_corners=False
        )[0]
        low_references = F.interpolate(
            torch.stack(references),
            size=low_hw,
            mode="bilinear",
            align_corners=False,
        )
        residual = (
            (low_references - low_rendered).abs().mean(dim=1).amin(dim=0)
        )
        residual = F.interpolate(
            residual[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        reference_valid = residual <= float(
            config.maximum_reference_residual
        )
    alpha_map = torch.as_tensor(alpha).float().squeeze()
    depth_map = torch.as_tensor(rendered_depth).float().squeeze()
    normal = torch.as_tensor(surface_normal).float()
    if normal.ndim == 4:
        normal = normal[0]
    if alpha_map.shape != (height, width):
        alpha_map = F.interpolate(
            alpha_map[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if depth_map.shape != (height, width):
        depth_map = F.interpolate(
            depth_map[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if normal.shape[-2:] != (height, width):
        normal = F.interpolate(
            normal[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
    raster_valid = (
        torch.isfinite(depth_map)
        & (depth_map > 0)
        & (alpha_map >= float(config.minimum_alpha))
        & (
            torch.linalg.vector_norm(normal, dim=0)
            >= float(config.minimum_normal_norm)
        )
    )
    quality_invalid = _dilate(
        ~(reference_valid & raster_valid),
        config.invalid_dilate_radius,
    )
    valid = torch.as_tensor(base_mask.valid_mask).bool() & ~quality_invalid
    support = (
        torch.as_tensor(base_mask.support_mask).bool()
        & valid
    )
    residual_score = (
        residual / max(float(config.maximum_reference_residual), 1e-6)
    ).clamp(0.0, 1.0)
    invalid_score = torch.maximum(
        torch.as_tensor(base_mask.invalid_score).float(),
        torch.maximum(residual_score, quality_invalid.float()),
    )
    summary = {
        **{
            f"rgb_{key}": value
            for key, value in dict(base_mask.summary).items()
        },
        "valid_frac": float(valid.float().mean()),
        "invalid_frac": float((~valid).float().mean()),
        "support_frac": float(support.float().mean()),
        "reference_valid_frac": float(reference_valid.float().mean()),
        "raster_valid_frac": float(raster_valid.float().mean()),
        "reference_residual_mean": float(residual.mean()),
        "reference_residual_p95": float(
            torch.quantile(residual.reshape(-1), 0.95)
        ),
        "reference_alignment": (
            "rendered_depth_warp" if use_warped else "same_pixel_legacy"
        ),
    }
    return RenderQualityMask(
        valid_mask=valid.cpu(),
        support_mask=support.cpu(),
        invalid_score=invalid_score.cpu(),
        support_score=torch.as_tensor(base_mask.support_score).float().cpu(),
        channel_maps={
            **dict(base_mask.channel_maps),
            "reference_residual": residual.cpu(),
            "reference_valid": reference_valid.float().cpu(),
            "raster_valid": raster_valid.float().cpu(),
        },
        summary=summary,
    )


def project_existing_anchors(
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project fixed localization anchors without changing their geometry."""
    xyz = torch.as_tensor(xyz).float()
    pose_w2c = torch.as_tensor(pose_w2c, device=xyz.device).float()
    K = torch.as_tensor(K, device=xyz.device).float()
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    depth = camera[:, 2]
    projected = torch.stack(
        (
            K[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + K[0, 2],
            K[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + K[1, 2],
        ),
        dim=1,
    )
    return projected, depth, depth > 0


def _sample_map_at_xy(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:1][None]
    elif value.ndim != 4:
        raise ValueError("render evidence maps must have 2, 3, or 4 dimensions")
    height, width = value.shape[-2:]
    grid = xy.clone()
    grid[:, 0] = (grid[:, 0] + 0.5) / max(width, 1) * 2.0 - 1.0
    grid[:, 1] = (grid[:, 1] + 0.5) / max(height, 1) * 2.0 - 1.0
    return F.grid_sample(
        value.float(),
        grid.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(-1)


def render_visible_anchor_mask(
    *,
    projected_xy: torch.Tensor,
    anchor_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    alpha: torch.Tensor,
    width: int,
    height: int,
    config: SyntheticEvidenceConfig,
) -> torch.Tensor:
    inside = (
        (projected_xy[:, 0] >= 0)
        & (projected_xy[:, 0] < int(width))
        & (projected_xy[:, 1] >= 0)
        & (projected_xy[:, 1] < int(height))
        & (anchor_depth > 0)
    )
    sampled_depth = _sample_map_at_xy(rendered_depth, projected_xy)
    sampled_alpha = _sample_map_at_xy(alpha, projected_xy)
    tolerance = torch.maximum(
        torch.full_like(anchor_depth, float(config.absolute_depth_tolerance)),
        anchor_depth.abs() * float(config.relative_depth_tolerance),
    )
    return (
        inside
        & (sampled_alpha >= float(config.minimum_alpha))
        & (sampled_depth > 0)
        & ((sampled_depth - anchor_depth).abs() <= tolerance)
    )


def keypoint_positive_csr(
    *,
    keypoints: torch.Tensor,
    projected_xy: torch.Tensor,
    visible_anchor_indices: torch.Tensor,
    config: SyntheticEvidenceConfig,
    distance_chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Associate native keypoints with nearby visible existing anchors."""
    keypoints = torch.as_tensor(keypoints).float()
    projected_xy = torch.as_tensor(
        projected_xy, device=keypoints.device
    ).float()
    visible_anchor_indices = torch.as_tensor(
        visible_anchor_indices, device=keypoints.device
    ).long()
    if not keypoints.numel() or not visible_anchor_indices.numel():
        return (
            torch.arange(keypoints.shape[0], dtype=torch.long),
            torch.zeros(keypoints.shape[0] + 1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    visible_xy = projected_xy[visible_anchor_indices]
    maximum = int(config.positives_per_keypoint)
    radius = float(config.positive_radius_px)
    all_distances = []
    all_anchors = []
    for start in range(0, len(keypoints), max(int(distance_chunk), 1)):
        stop = min(len(keypoints), start + max(int(distance_chunk), 1))
        distances = torch.cdist(keypoints[start:stop], visible_xy)
        values, local = torch.topk(
            distances,
            k=min(maximum, visible_xy.shape[0]),
            largest=False,
            dim=1,
        )
        all_distances.append(values.cpu())
        all_anchors.append(visible_anchor_indices[local].cpu())
    distances = torch.cat(all_distances)
    anchors = torch.cat(all_anchors)
    legal = distances <= radius
    counts = legal.sum(dim=1)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0))
    )
    return (
        torch.arange(len(keypoints), dtype=torch.long),
        offsets,
        anchors[legal],
    )


def build_synthetic_evidence_record(
    *,
    name: str,
    sparse: dict,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    image_hw: tuple[int, int],
    state: dict,
    rendered_depth: torch.Tensor,
    alpha: torch.Tensor,
    valid_mask_result,
    view_bin: int,
    source_query: str,
    config: SyntheticEvidenceConfig,
    device: torch.device,
) -> dict:
    """Create filtered multi-positive appearance evidence for fixed anchors."""
    keypoints = torch.as_tensor(sparse["keypoints"]).float()
    descriptors = F.normalize(
        torch.as_tensor(sparse["descriptors"]).float(), dim=1
    )
    scores = torch.as_tensor(sparse["keypoint_scores"]).float()
    original_keypoint_count = int(keypoints.shape[0])
    valid_points = valid_mask_result.valid_points(keypoints.cpu())
    support_points = valid_mask_result.support_points(keypoints.cpu())
    point_keep = valid_points & (
        support_points if config.require_support_mask else True
    )
    keypoints = keypoints[point_keep.to(keypoints.device)]
    descriptors = descriptors[point_keep.to(descriptors.device)]
    scores = scores[point_keep.to(scores.device)]
    xyz = torch.as_tensor(state["anchor_xyz"]).float().to(device)
    projected, depth, in_front = project_existing_anchors(
        xyz,
        torch.as_tensor(pose_w2c).to(device),
        torch.as_tensor(K).to(device),
    )
    height, width = image_hw
    visible = render_visible_anchor_mask(
        projected_xy=projected,
        anchor_depth=depth,
        rendered_depth=torch.as_tensor(rendered_depth).to(device),
        alpha=torch.as_tensor(alpha).to(device),
        width=width,
        height=height,
        config=config,
    )
    visible &= in_front
    visible_indices = torch.nonzero(
        visible, as_tuple=False
    ).reshape(-1)
    alpha_coverage = float(
        (
            torch.as_tensor(alpha).float()
            >= float(config.minimum_alpha)
        )
        .float()
        .mean()
    )
    rows, offsets, positives = keypoint_positive_csr(
        keypoints=keypoints.to(device),
        projected_xy=projected,
        visible_anchor_indices=visible_indices,
        config=config,
    )
    positive_pair_count = int(positives.numel())
    matchable_count = int(((offsets[1:] - offsets[:-1]) > 0).sum())
    matchable_rate = (
        float(matchable_count / len(rows)) if len(rows) else 0.0
    )
    valid_keypoint_fraction = (
        float(len(rows) / original_keypoint_count)
        if original_keypoint_count
        else 0.0
    )
    accepted = bool(
        positive_pair_count >= int(config.minimum_positive_pairs)
        and matchable_rate >= float(config.minimum_matchable_rate)
        and float(valid_mask_result.summary["valid_frac"])
        >= float(config.minimum_valid_fraction)
        and valid_keypoint_fraction
        >= float(config.minimum_valid_keypoint_fraction)
        and alpha_coverage >= float(config.minimum_alpha_coverage)
        and int(visible_indices.numel())
        >= int(config.minimum_visible_anchors)
    )
    return {
        "query_name": str(name),
        "source_query": str(source_query),
        "view_bin": int(view_bin),
        "accepted": accepted,
        "reason": "ok" if accepted else "insufficient_existing_anchor_support",
        "pose_w2c": torch.as_tensor(pose_w2c).float().cpu(),
        "native_K": torch.as_tensor(K).float().cpu(),
        "native_input_hw": [int(height), int(width)],
        "native_keypoints": keypoints.cpu(),
        "native_descriptors": descriptors.cpu(),
        "native_scores": scores.cpu(),
        "query_rows": rows,
        "positive_offsets": offsets,
        "positive_indices": positives,
        "positive_pair_count": positive_pair_count,
        "matchable_keypoint_count": matchable_count,
        "matchable_rate": matchable_rate,
        "original_keypoint_count": original_keypoint_count,
        "valid_keypoint_fraction": valid_keypoint_fraction,
        "visible_anchor_count": int(visible_indices.numel()),
        "alpha_coverage": alpha_coverage,
        "valid_mask_summary": dict(valid_mask_result.summary),
        "config": asdict(config),
    }


def pack_synthetic_evidence(records: list[dict], *, provenance: dict) -> dict:
    accepted = [record for record in records if bool(record["accepted"])]
    return {
        "schema": "lafgs_artifact_filtered_synthetic_appearance_evidence",
        "version": 1,
        "query_names": [record["query_name"] for record in accepted],
        "records": accepted,
        "rejected_records": [
            {
                key: value
                for key, value in record.items()
                if key
                in {
                    "query_name",
                    "source_query",
                    "reason",
                    "positive_pair_count",
                    "matchable_rate",
                    "visible_anchor_count",
                    "alpha_coverage",
                    "valid_mask_summary",
                }
            }
            for record in records
            if not bool(record["accepted"])
        ],
        "summary": {
            "candidate_view_count": len(records),
            "accepted_view_count": len(accepted),
            "rejected_view_count": len(records) - len(accepted),
            "positive_pair_count": sum(
                int(record["positive_pair_count"]) for record in accepted
            ),
            "matchable_rate_mean": (
                float(
                    sum(record["matchable_rate"] for record in accepted)
                    / len(accepted)
                )
                if accepted
                else 0.0
            ),
        },
        "provenance": dict(provenance),
    }


def synthetic_query_cache_payload(evidence: dict) -> dict:
    queries = {}
    for record in evidence["records"]:
        name = str(record["query_name"])
        queries[name] = {
            "pose_w2c": torch.as_tensor(record["pose_w2c"]).float(),
            "pixel_center_offset": 0.5,
            "native_keypoints": torch.as_tensor(
                record["native_keypoints"]
            ).float(),
            "native_descriptors": torch.as_tensor(
                record["native_descriptors"]
            ).float(),
            "native_scores": torch.as_tensor(record["native_scores"]).float(),
            "native_K": torch.as_tensor(record["native_K"]).float(),
            "native_input_hw": list(record["native_input_hw"]),
        }
    return {
        "schema": "lafgs_artifact_filtered_synthetic_query_cache",
        "version": 1,
        "source": evidence["provenance"],
        "queries": queries,
    }


def synthetic_positive_teacher_payload(
    evidence: dict, *, anchor_count: int
) -> dict:
    records = []
    for query_index, record in enumerate(evidence["records"]):
        rows = torch.as_tensor(record["query_rows"]).long()
        records.append(
            {
                "query_index": query_index,
                "query_name": str(record["query_name"]),
                "query_rows": rows,
                "positive_offsets": torch.as_tensor(
                    record["positive_offsets"]
                ).long(),
                "positive_indices": torch.as_tensor(
                    record["positive_indices"]
                ).long(),
                "ambiguous_offsets": torch.as_tensor(
                    record.get(
                        "ambiguous_offsets",
                        torch.zeros(len(rows) + 1, dtype=torch.long),
                    )
                ).long(),
                "ambiguous_indices": torch.as_tensor(
                    record.get(
                        "ambiguous_indices",
                        torch.empty(0, dtype=torch.long),
                    )
                ).long(),
                "hard_negative_offsets": torch.as_tensor(
                    record.get(
                        "hard_negative_offsets",
                        torch.zeros(len(rows) + 1, dtype=torch.long),
                    )
                ).long(),
                "hard_negative_indices": torch.as_tensor(
                    record.get(
                        "hard_negative_indices",
                        torch.empty(0, dtype=torch.long),
                    )
                ).long(),
                "hard_negative_positive_indices": torch.as_tensor(
                    record.get(
                        "hard_negative_positive_indices",
                        torch.empty(0, dtype=torch.long),
                    )
                ).long(),
                "hard_negative_weights": torch.as_tensor(
                    record.get(
                        "hard_negative_weights",
                        torch.empty(0, dtype=torch.float32),
                    )
                ).float(),
            }
        )
    return {
        "schema": "lafgs_synthetic_existing_anchor_positive_teacher",
        "version": 2,
        "anchor_count": int(anchor_count),
        "query_names": [
            str(record["query_name"]) for record in evidence["records"]
        ],
        "records": records,
        "provenance": evidence["provenance"],
    }


def synthetic_function_graph_payload(
    evidence: dict, *, anchor_count: int
) -> dict:
    return {
        "schema": "lafgs_synthetic_existing_anchor_function_graph",
        "version": 1,
        "anchor_count": int(anchor_count),
        "query_names": [
            str(record["query_name"]) for record in evidence["records"]
        ],
        "records": [
            {
                "query_index": query_index,
                "query_name": str(record["query_name"]),
                "query_rows": torch.as_tensor(record["query_rows"]).long(),
            }
            for query_index, record in enumerate(evidence["records"])
        ],
        "provenance": evidence["provenance"],
    }
