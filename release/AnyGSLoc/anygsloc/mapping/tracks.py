"""Frozen mainline kernel extracted from evidence.tracks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def robust_fuse_track_descriptors(
    descriptors: torch.Tensor,
    query_bins: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """Fuse observations by view bin, then trim bins farthest from the medoid."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    query_bins = torch.as_tensor(query_bins, dtype=torch.long).reshape(-1)
    if descriptors.shape[0] != query_bins.numel() or descriptors.shape[0] == 0:
        raise ValueError("descriptor/view-bin observations are inconsistent")
    if confidence is None:
        confidence = torch.ones(descriptors.shape[0], dtype=torch.float32)
    confidence = torch.as_tensor(confidence, dtype=torch.float32).reshape(-1)
    if confidence.numel() != descriptors.shape[0]:
        raise ValueError("descriptor/confidence observations are inconsistent")
    bin_prototypes = []
    for view_bin in torch.unique(query_bins, sorted=True).tolist():
        selected = query_bins == int(view_bin)
        weight = confidence[selected].clamp_min(1e-4)
        prototype = (descriptors[selected] * weight[:, None]).sum(dim=0)
        bin_prototypes.append(F.normalize(prototype, dim=0))
    prototypes = torch.stack(bin_prototypes)
    similarity = prototypes @ prototypes.T
    medoid = int(similarity.mean(dim=1).argmax())
    keep_count = max(
        1,
        int(round(prototypes.shape[0] * (1.0 - float(trim_fraction)))),
    )
    keep = torch.topk(similarity[medoid], k=keep_count).indices
    return F.normalize(prototypes[keep].mean(dim=0), dim=0)


def fuse_projective_anchor_observations(
    descriptors: torch.Tensor,
    query_bins: torch.Tensor,
    *,
    detector_weight: torch.Tensor | None = None,
    view_weight: torch.Tensor | None = None,
    visibility_weight: torch.Tensor | None = None,
    sequence_weight: torch.Tensor | None = None,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """GWFF-style fusion for a projective Anchor observation equivalence class.

    Compatibility mode is obtained by passing the historical Track confidence
    as ``detector_weight`` and appearance reliability as
    ``visibility_weight``.  Geometry-view and sequence balancing can then be
    enabled independently without creating a parallel Gaussian landmark map.
    """

    descriptors = torch.as_tensor(descriptors)
    count = int(descriptors.shape[0]) if descriptors.ndim else 0
    if count == 0:
        raise ValueError("projective Anchor fusion requires observations")
    combined = torch.ones(count, dtype=torch.float32, device=descriptors.device)
    for name, value in (
        ("detector_weight", detector_weight),
        ("view_weight", view_weight),
        ("visibility_weight", visibility_weight),
        ("sequence_weight", sequence_weight),
    ):
        if value is None:
            continue
        weight = torch.as_tensor(value, dtype=torch.float32, device=descriptors.device)
        if weight.ndim != 1 or weight.shape[0] != count:
            raise ValueError(f"{name} must have exact shape [{count}]")
        if not torch.isfinite(weight).all() or bool((weight < 0).any()):
            raise ValueError(f"{name} must be finite and non-negative")
        combined = combined * weight
    return robust_fuse_track_descriptors(
        descriptors,
        query_bins,
        combined,
        trim_fraction=trim_fraction,
    )
