from dataclasses import dataclass

import torch


@dataclass
class ArtifactRepairConfig:
    min_opacity_multiplier: float = 0.15
    suppression_power: float = 1.0
    renormalize_contributors: bool = True
    repair_threshold: float = 0.35
    physical_prune_threshold: float = 0.90


class ArtifactRepair:
    """Turns artifact evidence into non-destructive render-time suppression."""

    def __init__(self, config=None):
        self.config = config or ArtifactRepairConfig()

    def suppression_from_scores(self, scores, min_multiplier=None, power=None):
        scores = torch.as_tensor(scores, dtype=torch.float32)
        min_multiplier = self.config.min_opacity_multiplier if min_multiplier is None else float(min_multiplier)
        power = self.config.suppression_power if power is None else float(power)
        penalty = scores.clamp(0.0, 1.0)
        if power != 1.0:
            penalty = penalty.pow(max(power, 0.0))
        return (1.0 - (1.0 - float(min_multiplier)) * penalty).clamp(float(min_multiplier), 1.0)

    def gaussian_opacity_multiplier(self, gaussian_scores):
        return self.suppression_from_scores(gaussian_scores)

    def suppress_contributor_weights(self, contributor_ids, contributor_weights, gaussian_scores):
        ids = torch.as_tensor(contributor_ids, dtype=torch.long)
        weights = torch.as_tensor(contributor_weights, dtype=torch.float32, device=ids.device)
        scores = torch.as_tensor(gaussian_scores, dtype=torch.float32, device=ids.device).reshape(-1)
        multipliers = torch.ones_like(weights)
        valid = (ids >= 0) & (ids < scores.numel())
        if bool(valid.any()):
            gaussian_multipliers = self.suppression_from_scores(scores)
            multipliers[valid] = gaussian_multipliers[ids[valid]]
        repaired = weights * multipliers
        if self.config.renormalize_contributors:
            repaired = repaired / repaired.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return repaired, {
            "repair_weight_mean": float(repaired.mean().item()) if repaired.numel() else 0.0,
            "repair_multiplier_mean": float(multipliers.mean().item()) if multipliers.numel() else 1.0,
            "repair_suppressed_count": int((multipliers < 0.999).sum().item()),
        }

    def physical_prune_candidates(self, gaussian_scores, good_support=None):
        scores = torch.as_tensor(gaussian_scores, dtype=torch.float32)
        mask = scores >= float(self.config.physical_prune_threshold)
        if good_support is not None:
            support = torch.as_tensor(good_support, dtype=torch.float32, device=scores.device)
            mask = mask & (support <= 0)
        return mask
