from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _as_float_tensor(value, device=None, dtype=torch.float32):
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=dtype)


def _resize_like(value, target_hw):
    if value is None:
        return None
    if tuple(value.shape[-2:]) == tuple(target_hw):
        return value
    if value.dim() == 2:
        value = value[None, None]
        return F.interpolate(value, size=target_hw, mode="bilinear", align_corners=False)[0, 0]
    if value.dim() == 3:
        return F.interpolate(value[None], size=target_hw, mode="bilinear", align_corners=False)[0]
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(value.shape)}")


def _normalize01(value, start, stop):
    denom = max(float(stop) - float(start), 1e-6)
    return ((value - float(start)) / denom).clamp(0.0, 1.0)


def _entropy_from_weights(weights):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    normalizer = torch.log(torch.tensor(float(max(weights.shape[-1], 2)), dtype=entropy.dtype, device=entropy.device))
    return (entropy / normalizer.clamp_min(1e-8)).clamp(0.0, 1.0)


def _low_detail_map_from_rgb(rendered_rgb, threshold, pool):
    rendered = _as_float_tensor(rendered_rgb)
    if rendered.dim() == 3:
        gray = rendered.float().mean(dim=0)
    else:
        gray = rendered.float()
    tensor = gray[None, None]
    dx = torch.zeros_like(tensor)
    dy = torch.zeros_like(tensor)
    dx[:, :, :, 1:] = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs()
    dy[:, :, 1:, :] = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs()
    grad = torch.sqrt(dx.square() + dy.square()).clamp_min(0.0)
    kernel = max(1, int(pool))
    if kernel % 2 == 0:
        kernel += 1
    local_grad = F.avg_pool2d(grad, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (1.0 - local_grad[0, 0] / max(float(threshold), 1e-6)).clamp(0.0, 1.0)


@dataclass
class ArtifactDetectorConfig:
    rgb_weight: float = 1.0
    feature_weight: float = 1.0
    foundation_weight: float = 0.5
    alpha_weight: float = 0.5
    entropy_weight: float = 0.25
    low_texture_weight: float = 1.0
    rgb_residual_start: float = 0.08
    rgb_residual_stop: float = 0.35
    feature_residual_start: float = 0.15
    feature_residual_stop: float = 0.65
    alpha_threshold: float = 0.05
    low_texture_std_threshold: float = 0.03
    low_texture_grad_threshold: float = 0.002
    low_detail_grad_threshold: float = 0.035
    low_detail_pool: int = 15
    mild_threshold: float = 0.35
    severe_threshold: float = 0.65


@dataclass
class ArtifactEvidence:
    score_map: torch.Tensor
    channel_maps: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    @property
    def confidence_map(self):
        return (1.0 - self.score_map).clamp(0.0, 1.0)


class ArtifactDetector:
    """Combines RGB, feature, alpha and contributor evidence into artifact scores."""

    def __init__(self, config=None):
        self.config = config or ArtifactDetectorConfig()

    def detect(
        self,
        rendered_rgb=None,
        target_rgb=None,
        rendered_feature=None,
        target_feature=None,
        foundation_rendered=None,
        foundation_target=None,
        alpha=None,
        contributor_weights=None,
    ):
        cfg = self.config
        maps = {}
        weighted = []

        target_hw = None
        for value in (target_rgb, rendered_rgb, target_feature, rendered_feature, alpha):
            if value is not None:
                tensor = torch.as_tensor(value)
                target_hw = tuple(tensor.shape[-2:])
                break
        if target_hw is None:
            raise ValueError("At least one image/feature/alpha tensor is required for artifact detection.")

        if rendered_rgb is not None and target_rgb is not None and cfg.rgb_weight > 0:
            rendered = _as_float_tensor(rendered_rgb)
            target = _resize_like(_as_float_tensor(target_rgb, device=rendered.device, dtype=rendered.dtype), rendered.shape[-2:])
            residual = (rendered - target).abs()
            if residual.dim() == 3:
                residual = residual.mean(dim=0)
            rgb_score = _normalize01(residual, cfg.rgb_residual_start, cfg.rgb_residual_stop)
            maps["rgb"] = _resize_like(rgb_score, target_hw)
            weighted.append((maps["rgb"], float(cfg.rgb_weight)))

        if rendered_rgb is not None and target_rgb is None and cfg.low_texture_weight > 0:
            rendered = _as_float_tensor(rendered_rgb)
            if rendered.dim() == 3:
                texture = rendered.float().std(dim=(1, 2)).mean()
                gray = rendered.float().mean(dim=0)
            else:
                texture = rendered.float().std()
                gray = rendered.float()
            low_texture_score = (
                (float(cfg.low_texture_std_threshold) - texture)
                / max(float(cfg.low_texture_std_threshold), 1e-6)
            ).clamp(0.0, 1.0)
            grad_x = (gray[:, 1:] - gray[:, :-1]).abs().mean() if gray.shape[-1] > 1 else gray.new_tensor(0.0)
            grad_y = (gray[1:, :] - gray[:-1, :]).abs().mean() if gray.shape[-2] > 1 else gray.new_tensor(0.0)
            grad = 0.5 * (grad_x + grad_y)
            low_gradient_score = (
                (float(cfg.low_texture_grad_threshold) - grad)
                / max(float(cfg.low_texture_grad_threshold), 1e-6)
            ).clamp(0.0, 1.0)
            low_texture_score = torch.maximum(low_texture_score, low_gradient_score)
            low_detail = _low_detail_map_from_rgb(
                rendered,
                threshold=cfg.low_detail_grad_threshold,
                pool=cfg.low_detail_pool,
            )
            low_texture_global = torch.full(
                target_hw,
                float(low_texture_score.item()),
                dtype=torch.float32,
                device=rendered.device,
            )
            maps["low_detail"] = _resize_like(low_detail, target_hw)
            maps["low_texture"] = torch.maximum(low_texture_global, maps["low_detail"])
            weighted.append((maps["low_texture"], float(cfg.low_texture_weight)))

        if rendered_feature is not None and target_feature is not None and cfg.feature_weight > 0:
            rendered = _as_float_tensor(rendered_feature)
            target = _resize_like(_as_float_tensor(target_feature, device=rendered.device, dtype=rendered.dtype), rendered.shape[-2:])
            rendered_n = F.normalize(rendered, p=2, dim=0)
            target_n = F.normalize(target, p=2, dim=0)
            cosine = (rendered_n * target_n).sum(dim=0)
            feature_score = _normalize01(1.0 - cosine, cfg.feature_residual_start, cfg.feature_residual_stop)
            maps["feature"] = _resize_like(feature_score, target_hw)
            weighted.append((maps["feature"], float(cfg.feature_weight)))

        if foundation_rendered is not None and foundation_target is not None and cfg.foundation_weight > 0:
            rendered = _as_float_tensor(foundation_rendered)
            target = _resize_like(_as_float_tensor(foundation_target, device=rendered.device, dtype=rendered.dtype), rendered.shape[-2:])
            rendered_n = F.normalize(rendered, p=2, dim=0)
            target_n = F.normalize(target, p=2, dim=0)
            foundation_score = (1.0 - (rendered_n * target_n).sum(dim=0)).clamp(0.0, 1.0)
            maps["foundation"] = _resize_like(foundation_score, target_hw)
            weighted.append((maps["foundation"], float(cfg.foundation_weight)))

        if alpha is not None and cfg.alpha_weight > 0:
            alpha = _as_float_tensor(alpha)
            alpha = alpha.squeeze()
            if alpha.dim() == 3:
                if alpha.shape[0] == 1:
                    alpha = alpha[0]
                elif alpha.shape[-1] == 1:
                    alpha = alpha[..., 0]
                else:
                    alpha = alpha.mean(dim=0)
            if alpha.dim() != 2:
                raise ValueError(f"alpha must reduce to a 2D map, got shape {tuple(alpha.shape)}")
            alpha_score = ((float(cfg.alpha_threshold) - alpha) / max(float(cfg.alpha_threshold), 1e-6)).clamp(0.0, 1.0)
            maps["alpha"] = _resize_like(alpha_score, target_hw)
            weighted.append((maps["alpha"], float(cfg.alpha_weight)))

        if contributor_weights is not None and cfg.entropy_weight > 0:
            entropy = _entropy_from_weights(contributor_weights)
            if entropy.dim() == 1:
                maps["contributor_entropy"] = entropy
            else:
                maps["contributor_entropy"] = _resize_like(entropy, target_hw)
                weighted.append((maps["contributor_entropy"], float(cfg.entropy_weight)))

        if not weighted:
            score_map = torch.zeros(target_hw, dtype=torch.float32)
        else:
            total = sum(weight for _, weight in weighted)
            score_map = sum(value * weight for value, weight in weighted) / max(total, 1e-6)
            score_map = score_map.clamp(0.0, 1.0)

        summary = self.summarize(score_map)
        for name, value in maps.items():
            if torch.is_tensor(value) and value.dim() >= 2:
                summary[f"{name}_mean"] = float(value.detach().float().mean().item())
        return ArtifactEvidence(score_map=score_map, channel_maps=maps, summary=summary)

    def summarize(self, score_map):
        score = torch.as_tensor(score_map, dtype=torch.float32).detach().reshape(-1)
        if score.numel() == 0:
            return {
                "artifact_score_mean": 0.0,
                "artifact_score_p95": 0.0,
                "artifact_mild_frac": 0.0,
                "artifact_severe_frac": 0.0,
            }
        return {
            "artifact_score_mean": float(score.mean().item()),
            "artifact_score_p95": float(torch.quantile(score, 0.95).item()),
            "artifact_mild_frac": float((score >= float(self.config.mild_threshold)).float().mean().item()),
            "artifact_severe_frac": float((score >= float(self.config.severe_threshold)).float().mean().item()),
        }

    def gaussian_scores_from_contributors(self, contributor_ids, contributor_weights, anchor_scores, gaussian_count=None):
        ids = torch.as_tensor(contributor_ids, dtype=torch.long)
        weights = torch.as_tensor(contributor_weights, dtype=torch.float32, device=ids.device)
        scores = torch.as_tensor(anchor_scores, dtype=torch.float32, device=ids.device).reshape(-1)
        if ids.dim() == 1:
            ids = ids[:, None]
        if weights.dim() == 1:
            weights = weights[:, None]
        if ids.shape != weights.shape:
            raise ValueError(f"contributor_ids and weights shape mismatch: {tuple(ids.shape)} vs {tuple(weights.shape)}")
        if scores.numel() != ids.shape[0]:
            raise ValueError(f"anchor_scores must have one value per anchor, got {scores.numel()} for {ids.shape[0]}")
        valid = ids >= 0
        if gaussian_count is None:
            gaussian_count = int(ids[valid].max().item()) + 1 if bool(valid.any()) else 0
        out = weights.new_zeros(int(gaussian_count))
        denom = weights.new_zeros(int(gaussian_count))
        if int(gaussian_count) == 0 or not bool(valid.any()):
            return out
        flat_ids = ids[valid]
        flat_weights = weights[valid]
        flat_scores = (scores[:, None].expand_as(weights))[valid]
        in_range = (flat_ids >= 0) & (flat_ids < int(gaussian_count))
        out.scatter_add_(0, flat_ids[in_range], flat_scores[in_range] * flat_weights[in_range])
        denom.scatter_add_(0, flat_ids[in_range], flat_weights[in_range])
        return out / denom.clamp_min(1e-8)

    def gaussian_scores_from_projected_map(self, visible_idx, means2d, score_map, gaussian_count=None):
        visible_idx = torch.as_tensor(visible_idx, dtype=torch.long)
        score_map = torch.as_tensor(score_map, dtype=torch.float32, device=visible_idx.device)
        means2d = torch.as_tensor(means2d, dtype=torch.float32, device=visible_idx.device)
        if means2d.dim() == 3:
            means2d = means2d.squeeze(0)
        means2d = means2d.reshape(-1, 2)
        if means2d.shape[0] != visible_idx.numel():
            raise ValueError(
                "means2d must have one row per visible Gaussian, "
                f"got {means2d.shape[0]} for {visible_idx.numel()}."
            )
        if gaussian_count is None:
            gaussian_count = int(visible_idx.max().item()) + 1 if visible_idx.numel() else 0
        if int(gaussian_count) == 0:
            return score_map.new_zeros(0)
        height, width = score_map.shape[-2:]
        xy = means2d.round().long()
        valid = (
            (visible_idx >= 0)
            & (visible_idx < int(gaussian_count))
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        out = score_map.new_zeros(int(gaussian_count))
        if bool(valid.any()):
            out[visible_idx[valid]] = score_map[xy[valid, 1], xy[valid, 0]]
        return out
