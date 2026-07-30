#!/usr/bin/env python3
"""Train a context-vector pair scorer with leave-one-trajectory-out replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.candidate_context_rescue import (
    CandidateConditionedContextScorer,
)
from localization_training.contextual_descriptor import BoundedContextProjector
from localization_training.relational_context import (
    relational_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


SCALAR_NAMES = (
    "local_top1_score",
    "local_top1_margin",
    "challenger_local_gap",
    "context_score_delta",
    "rescued_score_margin",
    "challenger_rank",
    "keypoint_score",
    "keypoint_x_normalized",
    "keypoint_y_normalized",
    "edge_log_occurrences",
    "edge_log_trajectories",
    "edge_harmful_fraction",
    "edge_log_pose_blame",
    "anchor_log_distance",
    "same_source",
    "same_dependency",
)

OBSERVED_SCALAR_NAMES = (
    "correct_observed_available",
    "confusing_observed_available",
    "observed_similarity_delta",
    "correct_log_observations",
    "confusing_log_observations",
)


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _trajectory(name: str) -> str:
    return str(name).split("/", 1)[0]


def _atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _protected_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float,
) -> tuple[float, dict]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order].astype(bool)
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)
    utility = true_positive - float(false_positive_cost) * false_positive
    best = int(np.argmax(utility))
    if float(utility[best]) <= 0.0:
        return 1.0, {
            "selected": 0,
            "precision": 1.0,
            "recall": 0.0,
            "utility": 0.0,
        }
    selected = best + 1
    return float(probabilities[order[best]]), {
        "selected": int(selected),
        "precision": float(true_positive[best] / selected),
        "recall": float(
            true_positive[best] / max(int(labels.sum()), 1)
        ),
        "utility": float(utility[best]),
    }


def _predict(
    model: CandidateConditionedContextScorer,
    arrays: dict[str, torch.Tensor],
    indices: np.ndarray,
    scalar_mean: torch.Tensor,
    scalar_scale: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), int(batch_size)):
            batch = torch.as_tensor(
                indices[start : start + int(batch_size)]
            ).long()
            scalar = (
                arrays["scalar"][batch].float() - scalar_mean
            ) / scalar_scale
            logits = model(
                arrays["query"][batch].to(device).float(),
                arrays["correct"][batch].to(device).float(),
                arrays["confusing"][batch].to(device).float(),
                scalar.to(device),
            )
            output.append(torch.sigmoid(logits).cpu())
    return torch.cat(output).numpy()


def _fit(
    arrays: dict[str, torch.Tensor],
    fit_indices: np.ndarray,
    *,
    context_dim: int,
    scalar_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[CandidateConditionedContextScorer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    scalar_mean = arrays["scalar"][fit_indices].float().mean(dim=0)
    scalar_scale = (
        arrays["scalar"][fit_indices].float().std(dim=0).clamp_min(1e-6)
    )
    model = CandidateConditionedContextScorer(
        context_dim=context_dim,
        scalar_dim=int(scalar_dim),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    labels = arrays["label"][fit_indices].float()
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    positive_weight = math.sqrt(negative / max(positive, 1.0))
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(int(epochs)):
        order = torch.randperm(len(fit_indices), generator=generator)
        model.train()
        for start in range(0, len(order), int(batch_size)):
            local = order[start : start + int(batch_size)]
            batch = torch.as_tensor(fit_indices[local]).long()
            scalar = (
                arrays["scalar"][batch].float() - scalar_mean
            ) / scalar_scale
            logits = model(
                arrays["query"][batch].to(device).float(),
                arrays["correct"][batch].to(device).float(),
                arrays["confusing"][batch].to(device).float(),
                scalar.to(device),
            )
            target = arrays["label"][batch].float().to(device)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=target.new_tensor(positive_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, scalar_mean, scalar_scale


def _model_payload(
    model: CandidateConditionedContextScorer,
    scalar_mean: torch.Tensor,
    scalar_scale: torch.Tensor,
    threshold: float,
    scalar_names: tuple[str, ...],
) -> dict:
    return {
        "model_config": model.export_config(),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "scalar_names": list(scalar_names),
        "scalar_mean": scalar_mean.cpu(),
        "scalar_scale": scalar_scale.cpu(),
        "threshold": float(threshold),
    }


def _best_observed_context(
    query_context: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    heldout_trajectory_index: int,
    prototype_context: torch.Tensor,
    observation_counts: torch.Tensor,
    fallback_context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the best trajectory-excluded observed family per pair."""

    anchors = torch.as_tensor(
        anchor_indices, device=query_context.device
    ).long().reshape(-1)
    if prototype_context.ndim == 3:
        candidates = prototype_context[:, anchors].permute(1, 0, 2)
        counts = observation_counts[:, anchors].permute(1, 0)
    elif prototype_context.ndim == 4:
        candidates = prototype_context[:, :, anchors].permute(2, 0, 1, 3)
        counts = observation_counts[:, :, anchors].permute(2, 0, 1)
    else:
        raise ValueError("observed context must be TxAxD or TxBxAxD")
    valid = counts > 0
    valid[:, int(heldout_trajectory_index)] = False
    flattened_candidates = candidates.flatten(1, candidates.ndim - 2)
    flattened_valid = valid.flatten(1)
    flattened_counts = counts.flatten(1)
    similarities = torch.einsum(
        "nd,nfd->nf", query_context, flattened_candidates
    ).masked_fill(~flattened_valid, -torch.inf)
    best_similarity, best_slot = similarities.max(dim=1)
    selected = torch.gather(
        flattened_candidates,
        1,
        best_slot[:, None, None].expand(
            -1, 1, flattened_candidates.shape[-1]
        ),
    ).squeeze(1)
    selected_counts = torch.gather(
        flattened_counts, 1, best_slot[:, None]
    ).squeeze(1)
    available = torch.isfinite(best_similarity)
    selected = torch.where(
        available[:, None], selected, fallback_context[anchors]
    )
    best_similarity = torch.where(
        available,
        best_similarity,
        torch.einsum(
            "nd,nd->n", query_context, fallback_context[anchors]
        ),
    )
    return selected, best_similarity, available, selected_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--edge-rescue-topk", required=True)
    parser.add_argument("--strict-oracle-topk", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument(
        "--observed-context-families",
        default="",
        help=(
            "Optional observed 2D relation prototypes. A mapping sample cannot "
            "use prototypes from its own trajectory."
        ),
    )
    parser.add_argument("--output-topk", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--false-positive-cost", type=float, default=4.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--calibration-mode",
        choices=("trajectory_block", "random_rows"),
        default="trajectory_block",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    rescue = torch.load(
        args.edge_rescue_topk, map_location="cpu", weights_only=False
    )
    oracle = torch.load(
        args.strict_oracle_topk, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    observed_context = (
        torch.load(
            args.observed_context_families,
            map_location="cpu",
            weights_only=False,
        )
        if args.observed_context_families
        else None
    )
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    if context_state.get("schema") != "lafgs_fixed_3d_context_state":
        raise ValueError("unsupported relational context state")
    if context_state["config"].get(
        "query_representation"
    ) != "relational_sparse_2d_v1":
        raise ValueError("pair scorer requires relational sparse query context")

    baseline_by_name = _records_by_name(baseline)
    rescue_by_name = _records_by_name(rescue)
    oracle_by_name = _records_by_name(oracle)
    edge_metadata = {
        (int(edge["correct_anchor"]), int(edge["confusing_anchor"])): edge
        for edge in graph["edges"]
    }
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    projector = BoundedContextProjector(
        **context_state["query_projector_config"]
    ).to(device)
    projector.load_state_dict(context_state["query_projector_state_dict"])
    projector.eval()
    map_context = F.normalize(
        torch.as_tensor(context_state["anchor_context"]).float(), dim=1
    ).to(device)
    observed_prototypes = None
    observed_counts = None
    observed_trajectory_index = {}
    if observed_context is not None:
        if (
            observed_context.get("schema")
            != "lafgs_observed_context_families"
        ):
            raise ValueError("unsupported observed context family state")
        if not torch.equal(
            torch.as_tensor(observed_context["anchor_ids"]).long(),
            torch.as_tensor(state["anchor_ids"]).long(),
        ):
            raise ValueError("observed context families do not align with map")
        observed_prototypes = F.normalize(
            torch.as_tensor(
                observed_context["prototype_context"]
            ).float(),
            dim=-1,
        ).to(device)
        observed_counts = torch.as_tensor(
            observed_context["observation_counts"]
        ).to(device)
        observed_trajectory_index = {
            str(trajectory): index
            for index, trajectory in enumerate(
                observed_context["trajectories"]
            )
        }
    scalar_names = (
        SCALAR_NAMES + OBSERVED_SCALAR_NAMES
        if observed_context is not None
        else SCALAR_NAMES
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    context_config = context_state["config"]

    query_contexts = []
    correct_contexts = []
    confusing_contexts = []
    scalar_features = []
    labels = []
    sample_queries = []
    sample_slots = []
    with torch.inference_mode():
        for query_index, name in enumerate(baseline["query_names"]):
            before = baseline_by_name[name]
            after = rescue_by_name[name]
            strict = oracle_by_name[name]
            rows = torch.as_tensor(before["query_rows"]).long()
            before_indices = torch.as_tensor(
                before["topk_anchor_indices"]
            ).long()
            before_scores = torch.as_tensor(before["topk_scores"]).float()
            after_indices = torch.as_tensor(
                after["topk_anchor_indices"]
            ).long()
            after_scores = torch.as_tensor(after["topk_scores"]).float()
            strict_indices = torch.as_tensor(
                strict["topk_anchor_indices"]
            ).long()
            changed = torch.nonzero(
                after_indices[:, 0] != before_indices[:, 0],
                as_tuple=False,
            ).reshape(-1)
            if not changed.numel():
                continue
            cached = cache[name]
            all_descriptors = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float().to(device),
                dim=1,
            )
            transformed, _ = metric(all_descriptors)
            query_context = relational_sparse_query_context(
                transformed,
                torch.as_tensor(cached["native_keypoints"]).float().to(device),
                torch.as_tensor(cached["native_scores"]).float().to(device),
                neighbor_count=int(context_config["query_neighbor_count"]),
                chunk_size=int(context_config.get("context_chunk_size", 256)),
            )[rows[changed].to(device)]
            query_context, _ = projector(query_context)
            if observed_context is not None:
                trajectory = _trajectory(name)
                if trajectory not in observed_trajectory_index:
                    raise ValueError(
                        f"observed context misses trajectory {trajectory}"
                    )
                family_kwargs = {
                    "heldout_trajectory_index": observed_trajectory_index[
                        trajectory
                    ],
                    "prototype_context": observed_prototypes,
                    "observation_counts": observed_counts,
                    "fallback_context": map_context,
                }
                (
                    correct_observed,
                    correct_similarity,
                    correct_available,
                    correct_observations,
                ) = _best_observed_context(
                    query_context,
                    after_indices[changed, 0].to(device),
                    **family_kwargs,
                )
                (
                    confusing_observed,
                    confusing_similarity,
                    confusing_available,
                    confusing_observations,
                ) = _best_observed_context(
                    query_context,
                    before_indices[changed, 0].to(device),
                    **family_kwargs,
                )
            keypoint_scores = torch.as_tensor(
                cached["native_scores"]
            ).float()[rows]
            keypoints = torch.as_tensor(
                cached["native_keypoints"]
            ).float()[rows]
            height, width = cached["native_input_hw"]
            for local_index, slot in enumerate(changed.tolist()):
                confusing = int(before_indices[slot, 0])
                correct = int(after_indices[slot, 0])
                metadata = edge_metadata[(correct, confusing)]
                rank = int(
                    torch.nonzero(
                        before_indices[slot] == correct, as_tuple=False
                    ).reshape(-1)[0]
                )
                local_top1 = float(before_scores[slot, 0])
                challenger = float(before_scores[slot, rank])
                rescued_score = float(after_scores[slot, 0])
                occurrences = max(
                    int(metadata.get("occurrences", 0)), 1
                )
                harmful = int(metadata.get("harmful_occurrences", 0))
                scalar = [
                        local_top1,
                        local_top1 - float(before_scores[slot, 1]),
                        local_top1 - challenger,
                        rescued_score - challenger,
                        rescued_score - local_top1,
                        rank / max(before_indices.shape[1] - 1, 1),
                        float(keypoint_scores[slot]),
                        float(keypoints[slot, 0]) / max(float(width), 1.0),
                        float(keypoints[slot, 1]) / max(float(height), 1.0),
                        np.log1p(occurrences),
                        np.log1p(int(metadata.get("trajectory_count", 0))),
                        harmful / occurrences,
                        np.log1p(
                            max(float(metadata.get("pose_blame", 0.0)), 0.0)
                        ),
                        np.log1p(
                            float(
                                torch.linalg.vector_norm(
                                    xyz[correct] - xyz[confusing]
                                )
                            )
                        ),
                        float(source[correct] == source[confusing]),
                        float(dependency[correct] == dependency[confusing]),
                    ]
                if observed_context is not None:
                    scalar.extend(
                        (
                            float(correct_available[local_index]),
                            float(confusing_available[local_index]),
                            float(
                                correct_similarity[local_index]
                                - confusing_similarity[local_index]
                            ),
                            float(
                                np.log1p(
                                    int(correct_observations[local_index])
                                )
                            ),
                            float(
                                np.log1p(
                                    int(confusing_observations[local_index])
                                )
                            ),
                        )
                    )
                scalar_features.append(scalar)
                query_contexts.append(query_context[local_index].half().cpu())
                if observed_context is None:
                    correct_contexts.append(
                        map_context[correct].half().cpu()
                    )
                    confusing_contexts.append(
                        map_context[confusing].half().cpu()
                    )
                else:
                    correct_contexts.append(
                        correct_observed[local_index].half().cpu()
                    )
                    confusing_contexts.append(
                        confusing_observed[local_index].half().cpu()
                    )
                labels.append(int(strict_indices[slot, 0]) == correct)
                sample_queries.append(name)
                sample_slots.append(slot)
            if (query_index + 1) % 100 == 0:
                print(
                    f"context pairs {query_index + 1}/{len(baseline['query_names'])}",
                    flush=True,
                )

    arrays = {
        "query": torch.stack(query_contexts),
        "correct": torch.stack(correct_contexts),
        "confusing": torch.stack(confusing_contexts),
        "scalar": torch.as_tensor(scalar_features).float(),
        "label": torch.as_tensor(labels).bool(),
    }
    labels_np = arrays["label"].numpy()
    trajectories = np.asarray(
        [_trajectory(name) for name in sample_queries]
    )
    unique_trajectories = sorted(set(trajectories.tolist()))
    probabilities = np.zeros(len(labels), dtype=np.float64)
    accepted = np.zeros(len(labels), dtype=bool)
    fold_models = {}
    diagnostics = {}
    all_indices = np.arange(len(labels))
    for fold_index, trajectory in enumerate(unique_trajectories):
        outer_train = all_indices[trajectories != trajectory]
        heldout = all_indices[trajectories == trajectory]
        calibration_trajectory = None
        if args.calibration_mode == "trajectory_block":
            available = [
                value
                for value in unique_trajectories
                if value != trajectory
            ]
            calibration_trajectory = available[fold_index % len(available)]
            calibration = all_indices[
                trajectories == calibration_trajectory
            ]
            fit_indices = outer_train[
                trajectories[outer_train] != calibration_trajectory
            ]
        else:
            generator = np.random.default_rng(args.seed + fold_index)
            shuffled = generator.permutation(outer_train)
            calibration_count = max(
                int(
                    round(
                        len(shuffled)
                        * float(args.calibration_fraction)
                    )
                ),
                1,
            )
            calibration = shuffled[:calibration_count]
            fit_indices = shuffled[calibration_count:]
        model, scalar_mean, scalar_scale = _fit(
            arrays,
            fit_indices,
            context_dim=arrays["query"].shape[1],
            scalar_dim=len(scalar_names),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=args.seed + fold_index,
            device=device,
        )
        calibration_probability = _predict(
            model,
            arrays,
            calibration,
            scalar_mean,
            scalar_scale,
            device=device,
            batch_size=int(args.batch_size),
        )
        threshold, threshold_diagnostics = _protected_threshold(
            labels_np[calibration],
            calibration_probability,
            float(args.false_positive_cost),
        )
        heldout_probability = _predict(
            model,
            arrays,
            heldout,
            scalar_mean,
            scalar_scale,
            device=device,
            batch_size=int(args.batch_size),
        )
        probabilities[heldout] = heldout_probability
        accepted[heldout] = heldout_probability >= threshold
        true_positive = int(
            (accepted[heldout] & labels_np[heldout]).sum()
        )
        false_positive = int(
            (accepted[heldout] & ~labels_np[heldout]).sum()
        )
        fold_models[trajectory] = _model_payload(
            model,
            scalar_mean,
            scalar_scale,
            threshold,
            scalar_names,
        )
        diagnostics[trajectory] = {
            "sample_count": int(len(heldout)),
            "positive_count": int(labels_np[heldout].sum()),
            "auc": _auc(labels_np[heldout], heldout_probability),
            "accepted_count": int(accepted[heldout].sum()),
            "accepted_precision": float(
                true_positive / max(true_positive + false_positive, 1)
            ),
            "accepted_recall": float(
                true_positive / max(int(labels_np[heldout].sum()), 1)
            ),
            "calibration_threshold": threshold_diagnostics,
            "calibration_mode": args.calibration_mode,
            "calibration_trajectory": calibration_trajectory,
        }
        print(json.dumps({trajectory: diagnostics[trajectory]}), flush=True)

    output_records = []
    accepted_by_query: dict[str, set[int]] = {}
    for name, slot, use_rescue in zip(
        sample_queries, sample_slots, accepted
    ):
        if use_rescue:
            accepted_by_query.setdefault(name, set()).add(int(slot))
    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = rescue_by_name[name]
        indices = torch.as_tensor(
            before["topk_anchor_indices"]
        ).long().clone()
        scores = torch.as_tensor(before["topk_scores"]).float().clone()
        selected = sorted(accepted_by_query.get(name, ()))
        if selected:
            selected_tensor = torch.as_tensor(selected).long()
            indices[selected_tensor] = torch.as_tensor(
                after["topk_anchor_indices"]
            ).long()[selected_tensor]
            scores[selected_tensor] = torch.as_tensor(
                after["topk_scores"]
            ).float()[selected_tensor]
        output_records.append(
            {
                "query_name": name,
                "query_rows": torch.as_tensor(before["query_rows"]).long(),
                "topk_anchor_indices": indices,
                "topk_scores": scores,
            }
        )

    full_model, full_mean, full_scale = _fit(
        arrays,
        all_indices,
        context_dim=arrays["query"].shape[1],
        scalar_dim=len(scalar_names),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=args.seed + 1000,
        device=device,
    )
    full_threshold, full_threshold_diagnostics = _protected_threshold(
        labels_np,
        probabilities,
        float(args.false_positive_cost),
    )
    true_positive = int((accepted & labels_np).sum())
    false_positive = int((accepted & ~labels_np).sum())
    summary = {
        "proposal_count": int(len(labels)),
        "strict_positive_count": int(labels_np.sum()),
        "oof_auc": _auc(labels_np, probabilities),
        "oof_accepted_count": int(accepted.sum()),
        "oof_accepted_precision": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "oof_accepted_recall": float(
            true_positive / max(int(labels_np.sum()), 1)
        ),
        "false_positive_cost": float(args.false_positive_cost),
    }
    output_topk = Path(args.output_topk).resolve()
    _atomic_torch(
        output_topk,
        {
            "schema": "lafgs_exact_topk_outcomes",
            "version": 2,
            "query_names": list(baseline["query_names"]),
            "query_start": int(baseline.get("query_start", 0)),
            "topk": int(baseline["topk"]),
            "anchor_count": int(baseline["anchor_count"]),
            "anchor_ids_sha256": baseline["anchor_ids_sha256"],
            "records": output_records,
            "method": "candidate_conditioned_context_oof_vector_pair_scorer",
            "summary": summary,
            "provenance": {
                "baseline_topk": str(Path(args.baseline_topk).resolve()),
                "edge_rescue_topk": str(
                    Path(args.edge_rescue_topk).resolve()
                ),
                "strict_oracle_topk": str(
                    Path(args.strict_oracle_topk).resolve()
                ),
                "context_state": str(Path(args.context_state).resolve()),
                "observed_context_families": str(
                    Path(args.observed_context_families).resolve()
                )
                if args.observed_context_families
                else None,
            },
        },
    )
    output_model = Path(args.output_model).resolve()
    _atomic_torch(
        output_model,
        {
            "schema": "lafgs_candidate_context_pair_scorer",
            "version": 1,
            "fold_contract": "leave_one_trajectory_out",
            "fold_models": fold_models,
            "full_model": _model_payload(
                full_model,
                full_mean,
                full_scale,
                full_threshold,
                scalar_names,
            ),
            "full_threshold_diagnostics": full_threshold_diagnostics,
            "diagnostics": diagnostics,
            "summary": summary,
            "training_config": vars(args),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
