from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OneOfKRerankOutput:
    landmark_idx: torch.Tensor
    selected_position: torch.Tensor
    scores: torch.Tensor
    keep: torch.Tensor
    local_peak: torch.Tensor
    local_margin: torch.Tensor
    local_entropy: torch.Tensor
    peak_offset_norm: torch.Tensor
    null_selected: torch.Tensor
    null_margin: torch.Tensor
    ambiguous: torch.Tensor
    global_margin: torch.Tensor


def _local_candidate_correlations(
    query_feature_map,
    keypoint_xy,
    candidate_descriptors,
    image_hw,
    *,
    radius,
    step_px,
):
    feature_map = torch.as_tensor(query_feature_map)
    if feature_map.ndim == 4:
        if feature_map.shape[0] != 1:
            raise ValueError("query_feature_map batch size must be one")
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape CxHxW")
    keypoint_xy = torch.as_tensor(
        keypoint_xy, device=feature_map.device, dtype=feature_map.dtype
    ).reshape(-1, 2)
    candidate_descriptors = torch.as_tensor(
        candidate_descriptors,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    if candidate_descriptors.ndim != 3:
        raise ValueError("candidate_descriptors must have shape NxKxC")
    if (
        candidate_descriptors.shape[0] != keypoint_xy.shape[0]
        or candidate_descriptors.shape[2] != feature_map.shape[0]
    ):
        raise ValueError("candidate descriptors do not align with query features")

    radius = max(int(radius), 0)
    axis = torch.arange(
        -radius,
        radius + 1,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
    physical_xy = keypoint_xy + 0.5
    sample_xy = physical_xy[:, None, :] + offsets[None] * float(step_px)
    height, width = int(image_hw[0]), int(image_hw[1])
    grid = sample_xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(float(width), 1.0) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(float(height), 1.0) - 1.0
    sampled = F.grid_sample(
        feature_map[None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    sampled = F.normalize(sampled, dim=0)
    candidates = F.normalize(candidate_descriptors, dim=2)
    correlations = torch.einsum("cnp,nkc->nkp", sampled, candidates)
    return correlations, offsets


def build_one_of_k_features(
    query_feature_map,
    keypoint_xy,
    topk_landmark_idx,
    topk_scores,
    landmark_descriptors,
    image_hw,
    *,
    radius=2,
    step_px=8.0,
    temperature=0.07,
    landmark_statistics=None,
):
    topk_landmark_idx = torch.as_tensor(topk_landmark_idx, dtype=torch.long)
    topk_scores = torch.as_tensor(
        topk_scores,
        device=topk_landmark_idx.device,
    )
    landmarks = torch.as_tensor(
        landmark_descriptors,
        device=topk_scores.device,
        dtype=topk_scores.dtype,
    )
    candidates = landmarks[topk_landmark_idx]
    correlations, offsets = _local_candidate_correlations(
        query_feature_map,
        keypoint_xy,
        candidates,
        image_hw,
        radius=radius,
        step_px=step_px,
    )
    local_peak, peak_index = correlations.max(dim=2)
    if correlations.shape[2] > 1:
        local_top2 = torch.topk(correlations, 2, dim=2).values
        local_margin = local_top2[:, :, 0] - local_top2[:, :, 1]
    else:
        local_margin = torch.zeros_like(local_peak)
    probability = torch.softmax(
        correlations / max(float(temperature), 1e-6), dim=2
    )
    entropy = -(
        probability * probability.clamp_min(1e-8).log()
    ).sum(dim=2)
    entropy = entropy / max(
        float(torch.log(torch.tensor(correlations.shape[2])).item()), 1.0
    )
    peak_offsets = offsets[peak_index]
    offset_norm = torch.linalg.norm(peak_offsets, dim=2) / max(
        float(max(int(radius), 1)), 1.0
    )
    features = torch.stack(
        [
            topk_scores,
            local_peak,
            local_margin,
            entropy,
            offset_norm,
        ],
        dim=2,
    )
    if landmark_statistics is not None:
        statistics = torch.as_tensor(
            landmark_statistics,
            device=features.device,
            dtype=features.dtype,
        )
        if statistics.ndim != 2 or statistics.shape[0] != landmarks.shape[0]:
            raise ValueError(
                "landmark statistics must have shape landmark_count x channels"
            )
        features = torch.cat([features, statistics[topk_landmark_idx]], dim=2)
    return features


class OneOfKAssignmentHead(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        feature_dim=5,
        global_skip_scale=0.0,
        bounded_residual_max=0.0,
        logit_temperature=1.0,
        null_temperature=1.0,
        null_bias=0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.global_skip_scale = float(global_skip_scale)
        self.bounded_residual_max = float(bounded_residual_max)
        self.logit_temperature = float(logit_temperature)
        self.null_temperature = float(null_temperature)
        self.null_bias = float(null_bias)
        if self.feature_dim < 5:
            raise ValueError("one-of-K feature dimension must be at least five")
        self.candidate = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.null = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features):
        features = torch.as_tensor(features)
        if features.ndim != 3 or features.shape[2] != self.feature_dim:
            raise ValueError(
                "one-of-K features must have shape NxKx"
                f"{self.feature_dim}"
            )
        global_scores = features[:, :, 0]
        residual = self.candidate(features).squeeze(2)
        if self.bounded_residual_max > 0.0:
            candidate_logits = (
                global_scores
                + self.bounded_residual_max * torch.tanh(residual)
            ) / max(self.logit_temperature, 1e-6)
        else:
            candidate_logits = (
                self.global_skip_scale * global_scores + residual
            )
        if global_scores.shape[1] > 1:
            top2 = torch.topk(global_scores, 2, dim=1).values
            global_gap = top2[:, 0] - top2[:, 1]
        else:
            global_gap = torch.zeros_like(global_scores[:, 0])
        null_features = torch.stack(
            [
                global_scores[:, 0],
                global_gap,
                features[:, :, 1].max(dim=1).values,
                features[:, :, 1].mean(dim=1),
                features[:, :, 3].mean(dim=1),
            ],
            dim=1,
        )
        null_logits = (
            self.null(null_features).squeeze(1)
            / max(self.null_temperature, 1e-6)
            + self.null_bias
        )
        return candidate_logits, null_logits

    def export_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "feature_dim": self.feature_dim,
            "global_skip_scale": self.global_skip_scale,
            "bounded_residual_max": self.bounded_residual_max,
            "logit_temperature": self.logit_temperature,
            "null_temperature": self.null_temperature,
            "null_bias": self.null_bias,
        }


def rerank_one_of_k(
    query_feature_map,
    keypoint_xy,
    topk_landmark_idx,
    topk_scores,
    landmark_descriptors,
    image_hw,
    *,
    radius=2,
    step_px=8.0,
    global_weight=1.0,
    local_peak_weight=0.25,
    local_margin_weight=0.10,
    local_entropy_weight=0.02,
    offset_weight=0.02,
    temperature=0.07,
    null_score_threshold=-float("inf"),
    null_margin_threshold=-float("inf"),
    assignment_head=None,
    use_assignment_null=True,
    landmark_statistics=None,
    assignment_global_preserve_scale=0.0,
    ambiguity_margin_threshold=float("inf"),
    max_null_fraction=1.0,
    null_grid_rows=0,
    null_grid_cols=0,
    null_min_kept_per_grid=0,
):
    """Mutually rerank top-K hypotheses and emit one landmark or null per row."""
    topk_landmark_idx = torch.as_tensor(topk_landmark_idx, dtype=torch.long)
    topk_scores = torch.as_tensor(
        topk_scores,
        device=topk_landmark_idx.device,
    )
    if (
        topk_landmark_idx.ndim != 2
        or topk_landmark_idx.shape != topk_scores.shape
    ):
        raise ValueError("top-K indices and scores must have identical NxK shapes")
    features = build_one_of_k_features(
        query_feature_map,
        keypoint_xy,
        topk_landmark_idx,
        topk_scores,
        landmark_descriptors,
        image_hw,
        radius=radius,
        step_px=step_px,
        temperature=temperature,
        landmark_statistics=landmark_statistics,
    )
    local_peak = features[:, :, 1]
    local_margin = features[:, :, 2]
    entropy = features[:, :, 3]
    offset_norm = features[:, :, 4]
    null_logits = None
    if assignment_head is not None:
        logits, null_logits = assignment_head(features)
        logits = (
            logits
            + float(assignment_global_preserve_scale) * topk_scores
        )
        if not bool(use_assignment_null):
            null_logits = None
    else:
        logits = (
            float(global_weight) * topk_scores
            + float(local_peak_weight) * local_peak
            + float(local_margin_weight) * local_margin
            - float(local_entropy_weight) * entropy
            - float(offset_weight) * offset_norm
        )
    global_scores = topk_scores
    if global_scores.shape[1] > 1:
        global_top2 = torch.topk(global_scores, 2, dim=1).values
        global_margin = global_top2[:, 0] - global_top2[:, 1]
    else:
        global_margin = torch.full_like(global_scores[:, 0], float("inf"))
    ambiguous = global_margin < float(ambiguity_margin_threshold)
    best_score, best_position = logits.max(dim=1)
    best_position = torch.where(
        ambiguous, best_position, torch.zeros_like(best_position)
    )
    best_score = logits.gather(1, best_position[:, None]).squeeze(1)
    best_landmark = topk_landmark_idx.gather(
        1, best_position[:, None]
    ).squeeze(1)
    if logits.shape[1] > 1:
        rerank_margin = torch.topk(logits, 2, dim=1).values
        rerank_margin = rerank_margin[:, 0] - rerank_margin[:, 1]
    else:
        rerank_margin = torch.full_like(best_score, float("inf"))
    keep = (
        (best_score > float(null_score_threshold))
        & (rerank_margin > float(null_margin_threshold))
    )
    null_selected = torch.zeros_like(keep)
    null_margin = torch.full_like(best_score, -float("inf"))
    if null_logits is not None:
        null_margin = null_logits - best_score
        null_selected = null_logits >= best_score
        max_null = int(
            max(float(max_null_fraction), 0.0) * null_selected.numel()
        )
        selected_count = int(null_selected.sum().item())
        if selected_count > max_null:
            selected_idx = torch.nonzero(
                null_selected, as_tuple=False
            ).reshape(-1)
            keep_null = selected_idx[
                torch.topk(null_margin[selected_idx], max_null).indices
            ] if max_null > 0 else selected_idx[:0]
            null_selected.zero_()
            null_selected[keep_null] = True
        if (
            int(null_grid_rows) > 0
            and int(null_grid_cols) > 0
            and int(null_min_kept_per_grid) > 0
        ):
            height, width = map(int, image_hw)
            grid_x = torch.clamp(
                (torch.as_tensor(keypoint_xy, device=best_score.device)[:, 0]
                 * int(null_grid_cols) / max(width, 1)).long(),
                0,
                int(null_grid_cols) - 1,
            )
            grid_y = torch.clamp(
                (torch.as_tensor(keypoint_xy, device=best_score.device)[:, 1]
                 * int(null_grid_rows) / max(height, 1)).long(),
                0,
                int(null_grid_rows) - 1,
            )
            cell = grid_y * int(null_grid_cols) + grid_x
            for cell_id in range(int(null_grid_rows) * int(null_grid_cols)):
                rows = torch.nonzero(cell == cell_id, as_tuple=False).reshape(-1)
                kept = rows[~null_selected[rows]]
                missing = int(null_min_kept_per_grid) - int(kept.numel())
                if missing <= 0:
                    continue
                rejected = rows[null_selected[rows]]
                restore = rejected[
                    torch.topk(
                        -null_margin[rejected],
                        min(missing, int(rejected.numel())),
                    ).indices
                ] if rejected.numel() else rejected
                null_selected[restore] = False
        keep &= ~null_selected
    row = torch.arange(best_position.numel(), device=best_position.device)
    return OneOfKRerankOutput(
        landmark_idx=best_landmark,
        selected_position=best_position,
        scores=best_score,
        keep=keep,
        local_peak=local_peak[row, best_position],
        local_margin=local_margin[row, best_position],
        local_entropy=entropy[row, best_position],
        peak_offset_norm=offset_norm[row, best_position],
        null_selected=null_selected,
        null_margin=null_margin,
        ambiguous=ambiguous,
        global_margin=global_margin,
    )
