"""Training contracts for self-localization-guided top-K assignment."""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from localization_training.local_assignment import JOINT_ASSIGNMENT_STATISTIC_NAMES


def raw_tensor_sha256(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _unit_log_p95(value: torch.Tensor) -> torch.Tensor:
    value = torch.log1p(torch.as_tensor(value).float().clamp_min(0.0))
    positive = value[value > 0]
    scale = torch.quantile(positive, 0.95) if positive.numel() else value.new_tensor(1.0)
    return (value / scale.clamp_min(1e-8)).clamp(0.0, 1.0)


def beta_smoothed_anchor_features(
    anchor_statistics: dict,
    trajectory_support: torch.Tensor,
    temporal_view_bin_support: torch.Tensor,
    *,
    prior_success: float = 1.0,
    prior_failure: float = 1.0,
) -> torch.Tensor:
    """Build scene-normalized map evidence without zero-count overconfidence."""

    required = {"attempts", "clean", "clean_inlier", "harmful_inlier"}
    missing = sorted(required - set(anchor_statistics))
    if missing:
        raise ValueError("anchor statistics miss: " + ", ".join(missing))
    attempts = torch.as_tensor(anchor_statistics["attempts"]).float().reshape(-1)
    clean = torch.as_tensor(anchor_statistics["clean"]).float().reshape(-1)
    solver_clean = torch.as_tensor(anchor_statistics["clean_inlier"]).float().reshape(-1)
    harmful = torch.as_tensor(anchor_statistics["harmful_inlier"]).float().reshape(-1)
    trajectory = torch.as_tensor(trajectory_support).float().reshape(-1)
    temporal = torch.as_tensor(temporal_view_bin_support).float().reshape(-1)
    if not all(
        len(value) == len(attempts)
        for value in (clean, solver_clean, harmful, trajectory, temporal)
    ):
        raise ValueError("anchor statistics and trajectory support must align")
    denominator = attempts + float(prior_success) + float(prior_failure)
    clean_rate = (clean + float(prior_success)) / denominator
    solver_rate = (solver_clean + float(prior_success)) / denominator
    harmful_rate = (harmful + float(prior_failure)) / denominator
    return torch.stack(
        (
            clean_rate.clamp(0.0, 1.0),
            solver_rate.clamp(0.0, 1.0),
            (1.0 - harmful_rate).clamp(0.0, 1.0),
            _unit_log_p95(attempts),
            _unit_log_p95(trajectory),
            _unit_log_p95(temporal),
        ),
        dim=1,
    )


def trajectory_support_from_positive_teacher(teacher: dict, anchor_count: int) -> torch.Tensor:
    names = list(teacher["query_names"])
    records = list(teacher["records"])
    if len(names) != len(records):
        raise ValueError("positive-teacher query registry does not align")
    trajectory_names = sorted({name.split("/", 1)[0] for name in names})
    trajectory_index = {name: index for index, name in enumerate(trajectory_names)}
    observed = torch.zeros((len(trajectory_names), int(anchor_count)), dtype=torch.bool)
    for name, record in zip(names, records):
        indices = torch.as_tensor(record["positive_indices"]).long().unique()
        if indices.numel():
            if int(indices.min()) < 0 or int(indices.max()) >= int(anchor_count):
                raise ValueError("positive teacher contains an invalid anchor index")
            observed[trajectory_index[name.split("/", 1)[0]], indices] = True
    return observed.sum(dim=0)


def temporal_view_bin_support_from_positive_teacher(
    teacher: dict,
    anchor_count: int,
    *,
    bins_per_trajectory: int = 8,
) -> torch.Tensor:
    """Count distinct temporal bins that provide a positive observation."""

    names = list(teacher["query_names"])
    records = list(teacher["records"])
    if len(names) != len(records):
        raise ValueError("positive-teacher query registry does not align")
    grouped = {}
    for index, name in enumerate(names):
        grouped.setdefault(name.split("/", 1)[0], []).append(index)
    bin_ids = [None] * len(names)
    next_bin = 0
    for _, indices in sorted(grouped.items()):
        indices.sort(key=lambda index: names[index])
        local_count = min(max(int(bins_per_trajectory), 1), len(indices))
        for position, index in enumerate(indices):
            local_bin = min(
                position * local_count // len(indices), local_count - 1
            )
            bin_ids[index] = next_bin + local_bin
        next_bin += local_count
    observed = torch.zeros((next_bin, int(anchor_count)), dtype=torch.bool)
    for bin_id, record in zip(bin_ids, records):
        indices = torch.as_tensor(record["positive_indices"]).long().unique()
        if indices.numel():
            if int(indices.min()) < 0 or int(indices.max()) >= int(anchor_count):
                raise ValueError("positive teacher contains an invalid anchor index")
            observed[int(bin_id), indices] = True
    return observed.sum(dim=0)


def _candidate_mask_from_teacher_record(
    record: dict,
    candidate_indices: torch.Tensor,
    *,
    offsets_key: str,
    indices_key: str,
) -> torch.Tensor:
    candidates = torch.as_tensor(candidate_indices).long()
    offsets = torch.as_tensor(record[offsets_key]).long().reshape(-1)
    positives = torch.as_tensor(record[indices_key]).long().reshape(-1)
    if len(offsets) != len(candidates) + 1:
        raise ValueError(f"{offsets_key} rows do not align with candidates")
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(len(candidates)), counts)
    if not len(positives):
        return torch.zeros_like(candidates, dtype=torch.bool)
    anchor_count = max(
        int(candidates.max().item()) + 1,
        int(positives.max().item()) + 1,
    )
    keys = rows * anchor_count + positives
    candidate_rows = torch.arange(len(candidates))[:, None]
    return torch.isin(candidate_rows * anchor_count + candidates, keys)


def positive_mask_from_teacher_record(
    record: dict,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    return _candidate_mask_from_teacher_record(
        record,
        candidate_indices,
        offsets_key="positive_offsets",
        indices_key="positive_indices",
    )


def ambiguous_mask_from_teacher_record(
    record: dict,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Match loose-radius teacher identities without promoting them to positives."""

    candidates = torch.as_tensor(candidate_indices).long()
    if "ambiguous_offsets" not in record or "ambiguous_indices" not in record:
        return torch.zeros_like(candidates, dtype=torch.bool)
    return _candidate_mask_from_teacher_record(
        record,
        candidates,
        offsets_key="ambiguous_offsets",
        indices_key="ambiguous_indices",
    )


def select_balanced_training_rows(
    positive_mask: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    harmful_inlier_mask: torch.Tensor | None = None,
    ignored_row_mask: torch.Tensor | None = None,
    maximum_rows: int = 1024,
    null_to_positive_ratio: float = 2.0,
) -> torch.Tensor:
    """Retain every useful/harmful row and deterministic hard nulls."""

    positive_mask = torch.as_tensor(positive_mask).bool()
    scores = torch.as_tensor(topk_scores).float()
    if positive_mask.shape != scores.shape:
        raise ValueError("positive mask and top-K scores must align")
    useful = positive_mask.any(dim=1)
    harmful = (
        torch.as_tensor(harmful_inlier_mask).bool().reshape(-1)
        if harmful_inlier_mask is not None
        else torch.zeros(len(useful), dtype=torch.bool)
    )
    if len(harmful) != len(useful):
        raise ValueError("harmful-inlier mask must align with candidate rows")
    ignored = (
        torch.as_tensor(ignored_row_mask).bool().reshape(-1)
        if ignored_row_mask is not None
        else torch.zeros(len(useful), dtype=torch.bool)
    )
    if len(ignored) != len(useful):
        raise ValueError("ignored-row mask must align with candidate rows")
    mandatory = (useful | harmful) & ~ignored
    maximum_rows = min(max(int(maximum_rows), int(mandatory.sum())), len(useful))
    null_budget = min(
        maximum_rows - int(mandatory.sum()),
        max(int(round(float(null_to_positive_ratio) * int(useful.sum()))), 0),
    )
    selected = mandatory.clone()
    if null_budget > 0:
        available = torch.nonzero(~mandatory & ~ignored, as_tuple=False).reshape(-1)
        margin = (
            scores[:, 0] - scores[:, 1]
            if scores.shape[1] > 1
            else torch.zeros_like(scores[:, 0])
        )
        # False attractors are high-score but low-margin null rows.
        utility = scores[:, 0] - margin.abs()
        order = torch.argsort(utility[available], descending=True, stable=True)
        selected[available[order[:null_budget]]] = True
    return torch.nonzero(selected, as_tuple=False).reshape(-1)


def select_counterfactual_replacement_rows(
    positive_mask: torch.Tensor,
    harmful_inlier_mask: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    maximum_rows: int = 4,
) -> torch.Tensor:
    """Choose pose-relevant wrong top-1 rows that have a legal top-K repair."""

    positive = torch.as_tensor(positive_mask).bool()
    harmful = torch.as_tensor(harmful_inlier_mask).bool().reshape(-1)
    scores = torch.as_tensor(topk_scores).float()
    if positive.shape != scores.shape or len(harmful) != len(positive):
        raise ValueError("counterfactual row evidence must align")
    repairable = harmful & ~positive[:, 0] & positive[:, 1:].any(dim=1)
    rows = torch.nonzero(repairable, as_tuple=False).reshape(-1)
    if not len(rows) or int(maximum_rows) <= 0:
        return rows[:0]
    margin = scores[:, 0] - scores[:, 1]
    # Prefer confident harmful consensus: these rows are most likely to move a basin.
    order = torch.argsort(margin[rows], descending=True, stable=True)
    return rows[order[: int(maximum_rows)]]


def mask_stale_dynamic_harmful_evidence(
    harmful_inlier_mask: torch.Tensor,
    dynamic_top1_indices: torch.Tensor,
    deployment_top1_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop outcome labels whose landmark identity changed at deployment.

    Dynamic harmful-inlier labels describe a concrete 2D--3D assignment.  A
    numerically tied retrieval may select a different, geometrically distant
    landmark for the same query row, so carrying the old label across that
    identity switch would be incorrect.  GT multi-positive labels remain valid
    and are intentionally handled separately by the caller.
    """

    harmful = torch.as_tensor(harmful_inlier_mask).bool().reshape(-1)
    dynamic = torch.as_tensor(dynamic_top1_indices).long().reshape(-1)
    deployment = torch.as_tensor(deployment_top1_indices).long().reshape(-1)
    if len(harmful) != len(dynamic) or len(dynamic) != len(deployment):
        raise ValueError("dynamic evidence and deployment assignments must align")
    identity_changed = dynamic != deployment
    return harmful & ~identity_changed, identity_changed


def weighted_multi_positive_assignment_loss(
    candidate_logits: torch.Tensor,
    null_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    candidate_target_weights: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    protect_clean_top1: bool = True,
    null_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Multi-positive assignment/null loss with sparse exact-pose evidence."""

    logits = torch.as_tensor(candidate_logits)
    positive = torch.as_tensor(positive_mask, device=logits.device).bool()
    if positive.shape != logits.shape:
        raise ValueError("positive mask and candidate logits must align")
    target_weights = (
        torch.ones_like(logits)
        if candidate_target_weights is None
        else torch.as_tensor(candidate_target_weights, device=logits.device, dtype=logits.dtype)
    )
    if target_weights.shape != logits.shape or bool((target_weights < 0).any()):
        raise ValueError("candidate target weights must be non-negative and aligned")
    row_weights = (
        torch.ones(len(logits), device=logits.device, dtype=logits.dtype)
        if row_weights is None
        else torch.as_tensor(row_weights, device=logits.device, dtype=logits.dtype).reshape(-1)
    )
    if len(row_weights) != len(logits) or bool((row_weights < 0).any()):
        raise ValueError("row weights must be non-negative and aligned")

    protected = positive[:, 0] & bool(protect_clean_top1)
    positive = positive.clone()
    target_weights = target_weights.clone()
    if bool(protected.any()):
        positive[protected] = False
        positive[protected, 0] = True
        target_weights[protected] = 0.0
        target_weights[protected, 0] = 1.0
    positive = positive & (target_weights > 0)
    has_positive = positive.any(dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    candidate_terms = torch.zeros_like(denominator)
    if bool(has_positive.any()):
        # Exact-pose evidence is a relative preference inside the legal
        # positive set, not extra probability mass.  Row-wise normalization
        # keeps every factor in [0, 1], so the positive partition cannot exceed
        # the all-candidate partition and the multi-positive NLL stays valid.
        relative_weights = target_weights / target_weights.max(
            dim=1, keepdim=True
        ).values.clamp_min(1e-12)
        weighted_logits = logits + relative_weights.clamp_min(1e-12).log()
        numerator = torch.logsumexp(
            weighted_logits[has_positive].masked_fill(~positive[has_positive], -torch.inf),
            dim=1,
        )
        candidate_terms[has_positive] = denominator[has_positive] - numerator
    positive_loss = (
        (candidate_terms[has_positive] * row_weights[has_positive]).sum()
        / row_weights[has_positive].sum().clamp_min(1e-8)
        if bool(has_positive.any())
        else logits.sum() * 0.0
    )

    null_target = (~has_positive).to(logits.dtype)
    null_delta = null_logits - logits.max(dim=1).values.detach()
    null_terms = F.binary_cross_entropy_with_logits(
        null_delta, null_target, reduction="none"
    )
    null_loss = (null_terms * row_weights).sum() / row_weights.sum().clamp_min(1e-8)
    loss = positive_loss + float(null_loss_weight) * null_loss
    return loss, {
        "positive_loss": float(positive_loss.detach()),
        "null_loss": float(null_loss.detach()),
        "positive_rows": int(has_positive.sum()),
        "null_rows": int((~has_positive).sum()),
        "protected_top1_rows": int(protected.sum()),
    }
