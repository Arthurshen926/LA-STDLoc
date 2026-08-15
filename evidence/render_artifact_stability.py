"""Graphics-only raw/clean render stability for frozen Track observations.

The functions in this module never change keypoint rows or Track topology.
They turn the 2DGS distortion buffer into a soft per-pixel risk, transfer that
risk to projected Gaussian centres for a second opacity-suppressed render, and
measure whether the frozen SuperPoint observation survives that intervention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from features.superpoint import batched_nms


def normalized_distortion_risk(
    distortion: torch.Tensor,
    alpha: torch.Tensor,
    *,
    reference_quantile: float = 0.95,
    alpha_minimum: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a finite [H,W] risk normalized by a fixed supported quantile."""

    distortion = torch.as_tensor(distortion).squeeze().float()
    alpha = torch.as_tensor(alpha).squeeze().float()
    if distortion.ndim != 2 or alpha.shape != distortion.shape:
        raise ValueError("distortion and alpha must be aligned [H, W] planes")
    if not 0.0 < float(reference_quantile) < 1.0:
        raise ValueError("reference quantile must lie strictly inside (0, 1)")
    if not 0.0 <= float(alpha_minimum) <= 1.0:
        raise ValueError("alpha minimum must lie in [0, 1]")
    finite = torch.isfinite(distortion) & torch.isfinite(alpha)
    supported = finite & (alpha >= float(alpha_minimum))
    clean_distortion = torch.where(
        torch.isfinite(distortion), distortion.clamp_min(0.0), 0.0
    )
    if not bool(supported.any()):
        return torch.zeros_like(clean_distortion), clean_distortion.new_zeros(())
    scale = torch.quantile(clean_distortion[supported], float(reference_quantile))
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        return torch.zeros_like(clean_distortion), clean_distortion.new_zeros(())
    risk = (clean_distortion / scale).clamp(0.0, 1.0)
    return torch.where(supported, risk, 0.0), scale


def sample_plane_nearest(plane: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Sample a [H,W] plane at native image-grid coordinates."""

    plane = torch.as_tensor(plane).squeeze()
    xy = torch.as_tensor(xy, device=plane.device).float()
    if plane.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("plane and xy must have shapes [H,W] and [N,2]")
    if xy.numel() == 0:
        return plane.new_empty((0,))
    rounded = xy.round().long()
    rounded[:, 0].clamp_(0, plane.shape[1] - 1)
    rounded[:, 1].clamp_(0, plane.shape[0] - 1)
    return plane[rounded[:, 1], rounded[:, 0]]


def gaussian_opacity_multiplier(
    risk: torch.Tensor,
    projected_centres: torch.Tensor,
    visibility: torch.Tensor,
    *,
    suppression_strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer image risk to Gaussian centres and attenuate risky opacity."""

    if not 0.0 <= float(suppression_strength) <= 1.0:
        raise ValueError("suppression strength must lie in [0, 1]")
    centres = torch.as_tensor(projected_centres, device=risk.device)
    if centres.ndim == 3 and centres.shape[0] == 1:
        centres = centres[0]
    visibility = torch.as_tensor(visibility, device=risk.device)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("projected Gaussian centres must have shape [N,2]")
    if visibility.dtype != torch.bool or visibility.shape != (centres.shape[0],):
        raise ValueError("Gaussian visibility must be exact bool [N]")
    gaussian_risk = risk.new_zeros((centres.shape[0],))
    if bool(visibility.any()):
        gaussian_risk[visibility] = sample_plane_nearest(risk, centres[visibility])
    multiplier = 1.0 - float(suppression_strength) * gaussian_risk
    return multiplier.clamp(0.0, 1.0), gaussian_risk


def local_peak_stability(
    score_map: torch.Tensor,
    keypoints: torch.Tensor,
    raw_scores: torch.Tensor,
    *,
    nms_radius: int,
    search_radius: int = 4,
    position_sigma_px: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure score and position stability around frozen keypoint rows."""

    score_map = torch.as_tensor(score_map).squeeze().float()
    keypoints = torch.as_tensor(keypoints, device=score_map.device).float()
    raw_scores = torch.as_tensor(raw_scores, device=score_map.device).float()
    if score_map.ndim != 2:
        raise ValueError("score map must be [H,W]")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must be [N,2]")
    if raw_scores.shape != (keypoints.shape[0],):
        raise ValueError("raw scores must align with keypoints")
    if int(search_radius) < 0 or float(position_sigma_px) <= 0:
        raise ValueError("search radius and position sigma must be positive")

    suppressed = batched_nms(score_map[None], int(nms_radius))[0]
    border = 4
    if suppressed.shape[0] <= 2 * border or suppressed.shape[1] <= 2 * border:
        raise ValueError("score map is too small for the frozen SuperPoint border")
    suppressed[:border] = 0
    suppressed[-border:] = 0
    suppressed[:, :border] = 0
    suppressed[:, -border:] = 0
    radius = int(search_radius)
    offsets = torch.cartesian_prod(
        torch.arange(-radius, radius + 1, device=score_map.device),
        torch.arange(-radius, radius + 1, device=score_map.device),
    )
    # cartesian_prod returns [dy,dx]; store candidate points as native [x,y].
    offset_xy = offsets[:, [1, 0]]
    centre = keypoints.round().long()
    candidate = centre[:, None, :] + offset_xy[None, :, :]
    x = candidate[:, :, 0].clamp(0, score_map.shape[1] - 1)
    y = candidate[:, :, 1].clamp(0, score_map.shape[0] - 1)
    values = suppressed[y, x]
    best_score, best_index = values.max(dim=1)
    best_xy = candidate[
        torch.arange(candidate.shape[0], device=score_map.device), best_index
    ].float()
    displacement = torch.linalg.norm(best_xy - keypoints, dim=1)
    position = torch.exp(-0.5 * displacement.square() / float(position_sigma_px) ** 2)
    epsilon = torch.finfo(score_map.dtype).eps
    score = (
        torch.minimum(best_score, raw_scores).clamp_min(0.0)
        / torch.maximum(best_score, raw_scores).clamp_min(epsilon)
    ).clamp(0.0, 1.0)
    return score, position, displacement


def observation_artifact_reliability(
    raw_descriptors: torch.Tensor,
    clean_descriptors: torch.Tensor,
    detector_score_stability: torch.Tensor,
    position_stability: torch.Tensor,
    artifact_exposure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fuse four preregistered stability terms with a geometric mean."""

    raw = F.normalize(torch.as_tensor(raw_descriptors).float(), dim=1)
    clean = F.normalize(torch.as_tensor(clean_descriptors).float(), dim=1)
    if raw.ndim != 2 or clean.shape != raw.shape:
        raise ValueError("raw and clean descriptors must align as [N,D]")
    count = raw.shape[0]
    terms = [
        torch.as_tensor(value).float().reshape(-1)
        for value in (
            detector_score_stability,
            position_stability,
            artifact_exposure,
        )
    ]
    if any(value.shape != (count,) for value in terms):
        raise ValueError("stability terms must have one value per descriptor")
    descriptor_cosine = (raw * clean).sum(dim=1).clamp(0.0, 1.0)
    score_stability, position, exposure = terms
    exposure_complement = (1.0 - exposure).clamp(0.0, 1.0)
    factors = torch.stack(
        (
            descriptor_cosine,
            score_stability.clamp(0.0, 1.0),
            position.clamp(0.0, 1.0),
            exposure_complement,
        ),
        dim=1,
    )
    reliability = factors.prod(dim=1).clamp_min(0.0).pow(1.0 / 4.0)
    if not bool(torch.isfinite(reliability).all()):
        raise ValueError("artifact reliability is non-finite")
    return {
        "descriptor_cosine": descriptor_cosine,
        "detector_score_stability": score_stability.clamp(0.0, 1.0),
        "position_stability": position.clamp(0.0, 1.0),
        "artifact_exposure": exposure.clamp(0.0, 1.0),
        "reliability": reliability,
    }


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(values).float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("summary values must be non-empty and finite")
    return {
        "minimum": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }
