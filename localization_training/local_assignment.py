from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


ONE_OF_K_BASE_FEATURE_NAMES = (
    "global_cosine",
    "local_peak",
    "local_margin",
    "local_entropy",
    "local_peak_offset_norm",
)

JOINT_ASSIGNMENT_STATISTIC_NAMES = (
    "clean_beta_mean",
    "solver_clean_beta_mean",
    "harmful_beta_reliability",
    "attempt_support_log_p95",
    "trajectory_support_log_p95",
    "temporal_view_bin_support_log_p95",
)

JOINT_ASSIGNMENT_CONTEXT_FEATURE_NAMES = (
    "query_score_z",
    "candidate_rank",
    "keypoint_score_z",
    "keypoint_x",
    "keypoint_y",
    "keypoint_radius",
    "candidate_x_z",
    "candidate_y_z",
    "candidate_z_z",
    "candidate_identity_multiplicity",
    "source_multiplicity",
    "dependency_multiplicity",
)

JOINT_ASSIGNMENT_V1_FEATURE_NAMES = (
    *ONE_OF_K_BASE_FEATURE_NAMES,
    *JOINT_ASSIGNMENT_STATISTIC_NAMES,
    *JOINT_ASSIGNMENT_CONTEXT_FEATURE_NAMES,
)


def validate_joint_assignment_state_contract(state):
    """Reject a trained joint-assignment head with incompatible feature semantics."""

    if state.get("schema") != "lafgs_joint_assignment_loso":
        return False
    if int(state.get("version", -1)) not in (4, 5):
        raise ValueError("joint assignment runtime requires a v4/v5 LOSO artifact")
    config = state.get("config", {})
    if int(config.get("context_version", -1)) != 1:
        raise ValueError("joint assignment runtime requires context_version=1")
    feature_names = tuple(config.get("context_feature_names", ()))
    if feature_names != JOINT_ASSIGNMENT_V1_FEATURE_NAMES:
        raise ValueError(
            "joint assignment feature contract mismatch: expected "
            f"{JOINT_ASSIGNMENT_V1_FEATURE_NAMES}, got {feature_names}"
        )
    head_config = state.get("head_config", {})
    if int(head_config.get("feature_dim", -1)) != len(feature_names):
        raise ValueError("joint assignment head dimension does not match its feature contract")
    statistics = torch.as_tensor(state.get("landmark_statistics", torch.empty(0)))
    if statistics.ndim != 2 or statistics.shape[1] != len(
        JOINT_ASSIGNMENT_STATISTIC_NAMES
    ):
        raise ValueError(
            "joint assignment landmark statistics do not match the v1 feature contract"
        )
    return True


def _normalized_multiplicity(candidate_ids):
    ids = torch.as_tensor(candidate_ids).long()
    shape = ids.shape
    _, inverse, counts = torch.unique(
        ids.reshape(-1), sorted=False, return_inverse=True, return_counts=True
    )
    value = torch.log1p(counts[inverse].float()).reshape(shape)
    return value / value.max().clamp_min(1.0)


def _robust_candidate_xyz(candidate_xyz):
    xyz = torch.as_tensor(candidate_xyz).float()
    flat = xyz.reshape(-1, 3)
    center = torch.quantile(flat, 0.5, dim=0)
    deviation = torch.abs(flat - center)
    scale = torch.quantile(deviation, 0.75, dim=0).clamp_min(1e-4)
    return ((xyz - center) / scale).clamp(-4.0, 4.0)


def build_joint_assignment_context_features(
    topk_landmark_idx,
    topk_scores,
    keypoint_xy,
    image_hw,
    *,
    keypoint_scores,
    landmark_xyz,
    source_groups,
    dependency_groups,
):
    """Build query-normalized 2D/3D and repeated-assignment context."""

    indices = torch.as_tensor(topk_landmark_idx).long()
    scores = torch.as_tensor(topk_scores, device=indices.device).float()
    keypoints = torch.as_tensor(
        keypoint_xy, device=scores.device, dtype=scores.dtype
    ).reshape(-1, 2)
    keypoint_scores = torch.as_tensor(
        keypoint_scores, device=scores.device, dtype=scores.dtype
    ).reshape(-1)
    xyz_bank = torch.as_tensor(
        landmark_xyz, device=scores.device, dtype=scores.dtype
    ).reshape(-1, 3)
    source = torch.as_tensor(source_groups, device=scores.device).long().reshape(-1)
    dependency = torch.as_tensor(
        dependency_groups, device=scores.device
    ).long().reshape(-1)
    if indices.shape != scores.shape or len(indices) != len(keypoints):
        raise ValueError("joint assignment retrieval and keypoints must align")
    if len(keypoint_scores) != len(indices):
        raise ValueError("joint assignment keypoint scores must align")
    if not (len(xyz_bank) == len(source) == len(dependency)):
        raise ValueError("joint assignment map context must align")

    score_z = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)
    rank = torch.linspace(
        0.0,
        1.0,
        scores.shape[1],
        device=scores.device,
        dtype=scores.dtype,
    )[None].expand_as(scores)
    keypoint_score_z = (
        keypoint_scores - keypoint_scores.mean()
    ) / keypoint_scores.std(unbiased=False).clamp_min(1e-6)
    height, width = map(int, image_hw)
    x = keypoints[:, 0] / max(float(width), 1.0)
    y = keypoints[:, 1] / max(float(height), 1.0)
    radius = torch.sqrt((x - 0.5).square() + (y - 0.5).square())
    candidate_xyz = _robust_candidate_xyz(xyz_bank[indices])
    identity_multiplicity = _normalized_multiplicity(indices).to(scores.device)
    source_multiplicity = _normalized_multiplicity(source[indices]).to(scores.device)
    dependency_multiplicity = _normalized_multiplicity(
        dependency[indices]
    ).to(scores.device)
    repeated = lambda value: value[:, None].expand_as(scores)
    return torch.stack(
        (
            score_z,
            rank,
            repeated(keypoint_score_z),
            repeated(x),
            repeated(y),
            repeated(radius),
            candidate_xyz[:, :, 0],
            candidate_xyz[:, :, 1],
            candidate_xyz[:, :, 2],
            identity_multiplicity,
            source_multiplicity,
            dependency_multiplicity,
        ),
        dim=2,
    )


@dataclass
class OneOfKRerankOutput:
    landmark_idx: torch.Tensor
    selected_position: torch.Tensor
    candidate_logits: torch.Tensor
    assignment_score: torch.Tensor
    selected_global_score: torch.Tensor
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
    query_metric=None,
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
    if query_metric is not None:
        channel_count, row_count, sample_count = sampled.shape
        sampled_rows = sampled.permute(1, 2, 0).reshape(-1, channel_count)
        with torch.no_grad():
            sampled_rows, _ = query_metric(sampled_rows)
        sampled = sampled_rows.reshape(
            row_count, sample_count, channel_count
        ).permute(2, 0, 1)
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
    context_version=0,
    keypoint_scores=None,
    landmark_xyz=None,
    source_groups=None,
    dependency_groups=None,
    query_metric=None,
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
        query_metric=query_metric,
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
    if int(context_version) == 1:
        required = (
            keypoint_scores,
            landmark_xyz,
            source_groups,
            dependency_groups,
        )
        if any(value is None for value in required):
            raise ValueError(
                "joint assignment context v1 requires keypoint scores, xyz, "
                "source groups, and dependency groups"
            )
        features = torch.cat(
            (
                features,
                build_joint_assignment_context_features(
                    topk_landmark_idx,
                    topk_scores,
                    keypoint_xy,
                    image_hw,
                    keypoint_scores=keypoint_scores,
                    landmark_xyz=landmark_xyz,
                    source_groups=source_groups,
                    dependency_groups=dependency_groups,
                ),
            ),
            dim=2,
        )
    elif int(context_version) != 0:
        raise ValueError(f"unsupported joint assignment context: {context_version}")
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
        null_feature_mode="base5",
        normalize_candidate_features=False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.global_skip_scale = float(global_skip_scale)
        self.bounded_residual_max = float(bounded_residual_max)
        self.logit_temperature = float(logit_temperature)
        self.null_temperature = float(null_temperature)
        self.null_bias = float(null_bias)
        self.null_feature_mode = str(null_feature_mode)
        self.normalize_candidate_features = bool(normalize_candidate_features)
        if self.feature_dim < 5:
            raise ValueError("one-of-K feature dimension must be at least five")
        self.candidate = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if self.null_feature_mode not in {"base5", "pooled_full"}:
            raise ValueError("unsupported one-of-K null feature mode")
        null_feature_dim = 5 if self.null_feature_mode == "base5" else 2 * self.feature_dim + 1
        self.null = nn.Sequential(
            nn.Linear(null_feature_dim, self.hidden_dim),
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
        network_features = features
        if self.normalize_candidate_features:
            network_features = features.clone()
            # Normalize only local-correlation evidence within each candidate
            # set. Query-wide normalization would see a biased row subset at
            # training time but every detector row at deployment time.
            local = network_features[:, :, 1:5]
            center = local.mean(dim=1, keepdim=True)
            scale = local.std(
                dim=1, unbiased=False, keepdim=True
            ).clamp_min(1e-6)
            network_features[:, :, 1:5] = (local - center) / scale
        residual = self.candidate(network_features).squeeze(2)
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
        if self.null_feature_mode == "pooled_full":
            null_features = torch.cat(
                (
                    network_features[:, 0],
                    network_features.mean(dim=1),
                    global_gap[:, None],
                ),
                dim=1,
            )
        else:
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
            "null_feature_mode": self.null_feature_mode,
            "normalize_candidate_features": self.normalize_candidate_features,
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
    context_version=0,
    keypoint_scores=None,
    landmark_xyz=None,
    source_groups=None,
    dependency_groups=None,
    query_metric=None,
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
        context_version=context_version,
        keypoint_scores=keypoint_scores,
        landmark_xyz=landmark_xyz,
        source_groups=source_groups,
        dependency_groups=dependency_groups,
        query_metric=query_metric,
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
        # Candidate position zero is the protected deployment baseline.  The
        # remaining candidates are not guaranteed to stay score-sorted after
        # that identity is inserted, so compare against the best alternative.
        global_margin = global_scores[:, 0] - global_scores[:, 1:].max(
            dim=1
        ).values
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
        candidate_logits=logits,
        assignment_score=best_score,
        selected_global_score=global_scores[row, best_position],
        # Downstream fixed selectors must preserve their validated score
        # semantics. The learned head chooses an identity (or null); it must
        # not silently become a second learned correspondence ordering.
        scores=global_scores[row, best_position],
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
