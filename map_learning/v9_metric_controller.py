"""Bounded shared-metric ranking action trained from no-LOO causal evidence."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from map_learning.metric import SharedLowRankMetric


def train_v9_shared_metric(
    *,
    anchor_features: torch.Tensor,
    query_descriptors: torch.Tensor,
    positive_anchor_rows: torch.Tensor,
    negative_anchor_rows: torch.Tensor,
    sample_weights: torch.Tensor,
    clean_query_descriptors: torch.Tensor,
    clean_positive_anchor_rows: torch.Tensor,
    clean_negative_anchor_rows: torch.Tensor,
    clean_initial_margin: torch.Tensor,
    rank: int = 16,
    maximum_residual_norm: float = 0.05,
    steps: int = 400,
    batch_size: int = 1024,
    learning_rate: float = 2e-4,
    ranking_margin: float = 0.02,
    temperature: float = 0.05,
    clean_protection_weight: float = 1.0,
    seed: int = 20260828,
    device: str | torch.device = "cuda",
) -> tuple[SharedLowRankMetric, dict]:
    """Fit one query/map-shared transform without creating map prototypes."""

    anchors = F.normalize(torch.as_tensor(anchor_features).float(), dim=1)
    query = F.normalize(torch.as_tensor(query_descriptors).float(), dim=1)
    positive = torch.as_tensor(positive_anchor_rows).long().reshape(-1)
    negative = torch.as_tensor(negative_anchor_rows).long().reshape(-1)
    weights = torch.as_tensor(sample_weights).float().reshape(-1)
    clean_query = F.normalize(torch.as_tensor(clean_query_descriptors).float(), dim=1)
    clean_positive = torch.as_tensor(clean_positive_anchor_rows).long().reshape(-1)
    clean_negative = torch.as_tensor(clean_negative_anchor_rows).long().reshape(-1)
    initial_margin = torch.as_tensor(clean_initial_margin).float().reshape(-1)
    if not (query.shape[0] == positive.numel() == negative.numel() == weights.numel()):
        raise ValueError("causal ranking evidence rows do not align")
    if not (
        clean_query.shape[0]
        == clean_positive.numel()
        == clean_negative.numel()
        == initial_margin.numel()
    ):
        raise ValueError("clean-row protection evidence does not align")
    all_rows = torch.cat((positive, negative, clean_positive, clean_negative))
    if all_rows.numel() and (
        int(all_rows.min()) < 0 or int(all_rows.max()) >= anchors.shape[0]
    ):
        raise ValueError("metric evidence is outside the fixed map")
    if query.shape[0] == 0:
        raise ValueError("no cross-family causal ranking evidence was supplied")
    target = torch.device(device)
    generator = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    metric = SharedLowRankMetric(
        descriptor_dim=anchors.shape[1],
        rank=int(rank),
        max_residual_norm=float(maximum_residual_norm),
    ).to(target)
    optimizer = torch.optim.Adam(metric.parameters(), lr=float(learning_rate))
    anchors = anchors.to(target)
    query = query.to(target)
    positive = positive.to(target)
    negative = negative.to(target)
    weights = weights.to(target).clamp_min(1e-3)
    clean_query = clean_query.to(target)
    clean_positive = clean_positive.to(target)
    clean_negative = clean_negative.to(target)
    initial_margin = initial_margin.to(target)
    history = []
    for step in range(int(steps)):
        rows = torch.randint(
            query.shape[0],
            (min(int(batch_size), query.shape[0]),),
            generator=generator,
        ).to(target)
        q, _ = metric(query[rows])
        p, _ = metric(anchors[positive[rows]])
        n, _ = metric(anchors[negative[rows]])
        positive_score = (q * p).sum(1)
        negative_score = (q * n).sum(1)
        ranking = F.softplus(
            (negative_score - positive_score + float(ranking_margin))
            / float(temperature)
        )
        ranking_loss = (ranking * weights[rows]).sum() / weights[rows].sum()
        protection_loss = ranking_loss.new_zeros(())
        if clean_query.shape[0]:
            clean_rows = torch.randint(
                clean_query.shape[0],
                (min(int(batch_size), clean_query.shape[0]),),
                generator=generator,
            ).to(target)
            clean_q, _ = metric(clean_query[clean_rows])
            clean_p, _ = metric(anchors[clean_positive[clean_rows]])
            clean_n, _ = metric(anchors[clean_negative[clean_rows]])
            new_margin = (clean_q * clean_p).sum(1) - (clean_q * clean_n).sum(1)
            protection_loss = F.relu(initial_margin[clean_rows] - new_margin).mean()
        loss = ranking_loss + float(clean_protection_weight) * protection_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, int(steps) - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "ranking_loss": float(ranking_loss.detach()),
                    "clean_protection_loss": float(protection_loss.detach()),
                    "total_loss": float(loss.detach()),
                }
            )
    metric.eval()
    with torch.inference_mode():
        transformed_query, query_residual = metric(query)
        transformed_positive, positive_residual = metric(anchors[positive])
        transformed_negative, negative_residual = metric(anchors[negative])
        final_margin = (transformed_query * transformed_positive).sum(1) - (
            transformed_query * transformed_negative
        ).sum(1)
        original_margin = (query * anchors[positive]).sum(1) - (
            query * anchors[negative]
        ).sum(1)
        maximum_observed_residual = max(
            float(torch.linalg.norm(query_residual, dim=1).max()),
            float(torch.linalg.norm(positive_residual, dim=1).max()),
            float(torch.linalg.norm(negative_residual, dim=1).max()),
        )
    report = {
        "schema": "lafgs_v9_shared_metric_training_report",
        "version": 1,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "training_row_count": int(query.shape[0]),
        "clean_protection_row_count": int(clean_query.shape[0]),
        "initial_pair_accuracy": float((original_margin > 0).float().mean()),
        "final_pair_accuracy": float((final_margin > 0).float().mean()),
        "initial_median_margin": float(original_margin.median()),
        "final_median_margin": float(final_margin.median()),
        "maximum_observed_residual_norm": maximum_observed_residual,
        "maximum_residual_norm": float(maximum_residual_norm),
        "history": history,
    }
    return metric, report


@torch.inference_mode()
def transform_map_anchor_features(
    metric: SharedLowRankMetric,
    anchor_features: torch.Tensor,
    *,
    chunk_size: int = 8192,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Apply the shared metric once to the deployed map descriptor bank."""

    target = torch.device(device)
    metric = metric.to(target).eval()
    features = torch.as_tensor(anchor_features).float()
    output = []
    for start in range(0, features.shape[0], int(chunk_size)):
        transformed, residual = metric(features[start : start + int(chunk_size)].to(target))
        if bool(
            (torch.linalg.norm(residual, dim=1) > metric.max_residual_norm + 1e-6).any()
        ):
            raise RuntimeError("shared metric exceeded its residual trust region")
        output.append(transformed.cpu())
    return torch.cat(output)


def metric_artifact(
    metric: SharedLowRankMetric,
    *,
    anchor_ids: torch.Tensor,
    map_path: str,
    map_sha256: str,
    training_report: Mapping,
) -> dict:
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "protocol": "v9_no_loo_causal_shared_metric",
        "step": int(training_report["history"][-1]["step"]),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            key: value.detach().cpu().clone() for key, value in metric.state_dict().items()
        },
        "landmark_indices": torch.as_tensor(anchor_ids).long().cpu().clone(),
        "map_path": str(map_path),
        "map_sha256": str(map_sha256),
        "photometric_canonicalization_contract": None,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
    }
