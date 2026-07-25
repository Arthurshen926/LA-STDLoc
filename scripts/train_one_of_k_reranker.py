#!/usr/bin/env python
"""Train a query-level one-of-K/null assignment head on native proposals."""

import argparse
import json
import random
from collections import defaultdict
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.local_assignment import (
    OneOfKAssignmentHead,
    build_one_of_k_features,
)
from train_lafgs_map import _cached_native_observations


def normalized_landmark_statistics(
    statistics_path,
    landmark_indices,
    *,
    global_attractor_path=None,
):
    payload = torch.load(statistics_path, map_location="cpu")
    statistic_indices = torch.as_tensor(payload["landmark_indices"]).reshape(-1)
    landmark_indices = torch.as_tensor(landmark_indices).reshape(-1).cpu()
    if not torch.equal(statistic_indices.cpu(), landmark_indices):
        raise ValueError("landmark statistics do not align with the map state")
    statistics = payload["statistics"]

    def unit_log(value):
        value = torch.log1p(torch.as_tensor(value).float().clamp_min(0.0))
        positive = value[value > 0]
        scale = (
            torch.quantile(positive, 0.95)
            if positive.numel()
            else value.new_tensor(1.0)
        )
        return (value / scale.clamp_min(1e-8)).clamp(0.0, 1.0)

    false_attractor_reliability = (
        1.0 - torch.as_tensor(statistics["false_top1_rate"]).float()
    ).clamp(0.0, 1.0)
    if global_attractor_path:
        global_payload = torch.load(global_attractor_path, map_location="cpu")
        global_indices = torch.as_tensor(
            global_payload["landmark_indices"]
        ).reshape(-1)
        if not torch.equal(global_indices.cpu(), landmark_indices):
            raise ValueError(
                "global attractor statistics do not align with the map state"
            )
        global_statistics = global_payload["statistics"]
        false_attractor_reliability = torch.reciprocal(
            1.0 + torch.as_tensor(global_statistics["score"]).float()
        )
    features = torch.stack(
        [
            torch.as_tensor(statistics["matchability"]).float(),
            false_attractor_reliability,
            (
                1.0
                - torch.as_tensor(
                    statistics["cross_view_top1_harmful_switch_rate"]
                ).float()
            ).clamp(0.0, 1.0),
            unit_log(statistics["rescue_utility"]),
            unit_log(statistics["effective_observation_count"]),
        ],
        dim=1,
    )
    return features, [
        "matchability",
        "false_attractor_reliability",
        "harmful_switch_reliability",
        "rescue_utility_log_p95",
        "observation_count_log_p95",
    ]


def multi_positive_assignment_loss(
    candidate_logits,
    null_logits,
    positive_mask,
    *,
    protect_clean_top1=True,
    null_loss_weight=1.0,
):
    positive_mask = torch.as_tensor(
        positive_mask, device=candidate_logits.device, dtype=torch.bool
    )
    if positive_mask.shape != candidate_logits.shape:
        raise ValueError("positive mask and candidate logits must align")
    protected_top1 = positive_mask[:, 0] & bool(protect_clean_top1)
    target_positive_mask = positive_mask.clone()
    if bool(protected_top1.any()):
        target_positive_mask[protected_top1] = False
        target_positive_mask[protected_top1, 0] = True
    candidate_denominator = torch.logsumexp(candidate_logits, dim=1)
    has_positive = target_positive_mask.any(dim=1)
    positive_loss = candidate_logits.sum() * 0.0
    if bool(has_positive.any().item()):
        positive_numerator = torch.logsumexp(
            candidate_logits[has_positive].masked_fill(
                ~target_positive_mask[has_positive], -torch.inf
            ),
            dim=1,
        )
        positive_loss = (
            candidate_denominator[has_positive] - positive_numerator
        ).mean()
    null_rows = ~has_positive
    null_loss = candidate_logits.sum() * 0.0
    null_delta = null_logits - candidate_logits.max(dim=1).values.detach()
    null_terms = []
    if bool(null_rows.any().item()):
        null_terms.append(
            F.binary_cross_entropy_with_logits(
                null_delta[null_rows], torch.ones_like(null_delta[null_rows])
            )
        )
    if bool(has_positive.any().item()):
        null_terms.append(
            F.binary_cross_entropy_with_logits(
                null_delta[has_positive], torch.zeros_like(null_delta[has_positive])
            )
        )
    if null_terms:
        null_loss = sum(null_terms) / len(null_terms)
    loss = positive_loss + float(null_loss_weight) * null_loss
    return loss, {
        "positive_loss": float(positive_loss.detach().item()),
        "null_loss": float(null_loss.detach().item()),
        "positive_rows": int(has_positive.sum().item()),
        "null_rows": int(null_rows.sum().item()),
        "protected_top1_rows": int(protected_top1.sum().item()),
    }


def candidate_positive_mask(observations, candidate_indices, landmark_count):
    candidate_indices = torch.as_tensor(
        candidate_indices,
        device=observations.query_uv.device,
        dtype=torch.long,
    )
    offsets = observations.positive_offsets
    indices = observations.positive_indices
    if offsets is None or indices is None or indices.numel() == 0:
        return torch.zeros_like(candidate_indices, dtype=torch.bool)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
    positive_keys = rows * int(landmark_count) + indices
    candidate_rows = torch.arange(
        candidate_indices.shape[0], device=candidate_indices.device
    )[:, None]
    return torch.isin(
        candidate_rows * int(landmark_count) + candidate_indices,
        positive_keys,
    )


def ambiguity_gated_positive_mask(positive_mask, global_scores, threshold):
    positive_mask = torch.as_tensor(positive_mask, dtype=torch.bool).clone()
    global_scores = torch.as_tensor(
        global_scores, device=positive_mask.device
    )
    if global_scores.shape[1] < 2 or threshold == float("inf"):
        return positive_mask
    margin = global_scores[:, 0] - global_scores[:, 1]
    protected = margin >= float(threshold)
    if bool(protected.any()):
        protected_top1 = positive_mask[protected, 0].clone()
        positive_mask[protected] = False
        positive_mask[protected, 0] = protected_top1
    return positive_mask


def assignment_error_breakdown(
    candidate_logits,
    null_logits,
    positive_mask,
    global_scores,
    *,
    ambiguity_margin_threshold=float("inf"),
):
    candidate_logits = torch.as_tensor(candidate_logits)
    positive_mask = torch.as_tensor(
        positive_mask, device=candidate_logits.device, dtype=torch.bool
    )
    global_scores = torch.as_tensor(
        global_scores, device=candidate_logits.device
    )
    if global_scores.shape[1] > 1:
        global_margin = global_scores[:, 0] - global_scores[:, 1]
    else:
        global_margin = torch.full_like(global_scores[:, 0], float("inf"))
    ambiguous = global_margin < float(ambiguity_margin_threshold)
    selected = candidate_logits.argmax(dim=1)
    selected = torch.where(ambiguous, selected, torch.zeros_like(selected))
    best = candidate_logits.gather(1, selected[:, None]).squeeze(1)
    null_selected = null_logits >= best
    row = torch.arange(selected.numel(), device=selected.device)
    selected_positive = positive_mask[row, selected] & ~null_selected
    top1_positive = positive_mask[:, 0]
    has_positive = positive_mask.any(dim=1)
    swapped = selected != 0
    beneficial = swapped & ~top1_positive & selected_positive
    harmful = swapped & top1_positive & ~selected_positive
    clean_retained = top1_positive & ~swapped & ~null_selected
    true_null = ~has_positive
    return {
        "rows": int(selected.numel()),
        "positive_in_topk": int(has_positive.sum().item()),
        "selected_positive": int(selected_positive.sum().item()),
        "clean_top1": int(top1_positive.sum().item()),
        "clean_top1_retained": int(clean_retained.sum().item()),
        "swaps": int(swapped.sum().item()),
        "beneficial_swaps": int(beneficial.sum().item()),
        "harmful_swaps": int(harmful.sum().item()),
        "null_selected": int(null_selected.sum().item()),
        "true_null": int(true_null.sum().item()),
        "true_null_selected": int((null_selected & true_null).sum().item()),
        "matchable_rejected": int((null_selected & has_positive).sum().item()),
        "ambiguous": int(ambiguous.sum().item()),
    }


def summarize_assignment_counts(counts):
    rows = max(int(counts["rows"]), 1)
    positive = max(int(counts["positive_in_topk"]), 1)
    clean = max(int(counts["clean_top1"]), 1)
    null_selected = max(int(counts["null_selected"]), 1)
    true_null = max(int(counts["true_null"]), 1)
    return {
        **{key: int(value) for key, value in counts.items()},
        "positive_in_topk_rate": counts["positive_in_topk"] / rows,
        "conditional_positive_accuracy": counts["selected_positive"] / positive,
        "clean_top1_retention": counts["clean_top1_retained"] / clean,
        "swap_rate": counts["swaps"] / rows,
        "beneficial_swap_rate": counts["beneficial_swaps"] / rows,
        "harmful_swap_rate": counts["harmful_swaps"] / rows,
        "null_selection_rate": counts["null_selected"] / rows,
        "null_precision": counts["true_null_selected"] / null_selected,
        "null_recall": counts["true_null_selected"] / true_null,
        "matchable_false_rejection_rate": counts["matchable_rejected"] / positive,
        "ambiguous_rate": counts["ambiguous"] / rows,
    }


def calibrate_null_bias(
    records, head, min_precision, assignment_global_preserve_scale=0.0
):
    deltas = []
    labels = []
    head.eval()
    with torch.no_grad():
        for _, features, positives in records:
            features = features.to(next(head.parameters()).device)
            candidate_logits, null_logits = head(features)
            candidate_logits = (
                candidate_logits
                + float(assignment_global_preserve_scale)
                * features[:, :, 0]
            )
            deltas.append((null_logits - candidate_logits.max(dim=1).values).cpu())
            labels.append((~positives.any(dim=1)).cpu())
    delta = torch.cat(deltas)
    true_null = torch.cat(labels)
    order = torch.argsort(delta, descending=True)
    sorted_true = true_null[order].float()
    precision = sorted_true.cumsum(0) / torch.arange(
        1, sorted_true.numel() + 1, dtype=torch.float32
    )
    valid = torch.nonzero(
        precision >= float(min_precision), as_tuple=False
    ).reshape(-1)
    if valid.numel() == 0:
        threshold = float(delta.max().item() + 1.0)
    else:
        threshold = float(delta[order[valid[-1]]].item())
    head.null_bias -= threshold
    return {
        "min_precision": float(min_precision),
        "delta_threshold": threshold,
        "calibrated_null_bias": float(head.null_bias),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--map_state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=4, choices=[4, 8])
    parser.add_argument("--patch_radius", type=int, default=2)
    parser.add_argument("--patch_step_px", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument(
        "--global_skip_temperature",
        type=float,
        default=0.0,
        help="If positive, add cosine/temperature as a protected top-1 skip.",
    )
    parser.add_argument("--bounded_residual_max", type=float, default=0.0)
    parser.add_argument("--assignment_logit_temperature", type=float, default=1.0)
    parser.add_argument(
        "--ambiguity_margin_threshold", type=float, default=float("inf")
    )
    parser.add_argument("--calibrate_null", action="store_true")
    parser.add_argument("--null_min_precision", type=float, default=0.90)
    parser.add_argument("--null_loss_weight", type=float, default=1.0)
    parser.add_argument("--initial_state", default="")
    parser.add_argument("--freeze_candidate", action="store_true")
    parser.add_argument(
        "--assignment_global_preserve_scale", type=float, default=0.0
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--landmark_statistics", default="")
    parser.add_argument("--global_attractor_statistics", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.map_state, map_location="cpu")
    bank_xyz = torch.as_tensor(state["landmark_xyz"]).float().to(device)
    bank_features = F.normalize(
        torch.as_tensor(state["landmark_features"]).float().to(device), dim=1
    )
    landmark_statistics = None
    landmark_statistic_names = []
    if args.landmark_statistics:
        (
            landmark_statistics,
            landmark_statistic_names,
        ) = normalized_landmark_statistics(
            args.landmark_statistics,
            state["landmark_indices"],
            global_attractor_path=(
                args.global_attractor_statistics or None
            ),
        )
        landmark_statistics = landmark_statistics.to(device)
    cache_blob = torch.load(args.query_cache, map_location="cpu")
    visibility_blob = torch.load(args.visibility_cache, map_location="cpu")
    cache = cache_blob["queries"]
    visibility = visibility_blob["visibility"]
    query_names = sorted(set(cache) & set(visibility))
    observation_args = SimpleNamespace(
        grid_rows=8,
        grid_cols=8,
        native_association_radius_px=2.0,
        native_unmatched_fraction=0.5,
        native_sampling_mode="detector_grid",
    )

    feature_cache = []
    diagnostics = []
    for name in tqdm(query_names, desc="One-of-K feature cache"):
        observations = _cached_native_observations(
            cache[name],
            bank_xyz,
            observation_args,
            max_observations=2048,
            bank_visibility_mask=visibility[name].to(device),
            prediction_bank_xyz=bank_xyz,
        )
        if observations.query_features.numel() == 0:
            continue
        query = F.normalize(observations.query_features, dim=1)
        scores = query @ bank_features.T
        top_scores, top_indices = torch.topk(scores, args.topk, dim=1)
        local_features = build_one_of_k_features(
            observations.query_feature_map,
            observations.query_uv,
            top_indices,
            top_scores,
            bank_features,
            observations.query_feature_image_size,
            radius=args.patch_radius,
            step_px=args.patch_step_px,
            temperature=args.temperature,
            landmark_statistics=landmark_statistics,
        )
        positives = candidate_positive_mask(
            observations, top_indices, bank_features.shape[0]
        )
        feature_cache.append(
            (name, local_features.detach().cpu(), positives.detach().cpu())
        )
        diagnostics.append(
            {
                "rows": int(positives.shape[0]),
                "topk_positive_rows": int(positives.any(dim=1).sum().item()),
            }
        )

    feature_dim = 5 + (
        0
        if landmark_statistics is None
        else int(landmark_statistics.shape[1])
    )
    head = OneOfKAssignmentHead(
        hidden_dim=args.hidden_dim,
        feature_dim=feature_dim,
        global_skip_scale=(
            1.0 / args.global_skip_temperature
            if args.global_skip_temperature > 0.0
            else 0.0
        ),
        bounded_residual_max=args.bounded_residual_max,
        logit_temperature=args.assignment_logit_temperature,
    ).to(device)
    if args.initial_state:
        initial_state = torch.load(args.initial_state, map_location="cpu")
        initial_config = initial_state["head_config"]
        if int(initial_config["feature_dim"]) != feature_dim:
            raise ValueError("initial reranker feature dimension mismatch")
        head.load_state_dict(initial_state["head_state_dict"])
    if args.freeze_candidate:
        for parameter in head.candidate.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(feature_cache)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_records = []
        head.train()
        for index in tqdm(order, desc=f"One-of-K train {epoch + 1}/{args.epochs}"):
            _, features, positives = feature_cache[index]
            features = features.to(device)
            positives = positives.to(device)
            candidate_logits, null_logits = head(features)
            candidate_logits = (
                candidate_logits
                + args.assignment_global_preserve_scale
                * features[:, :, 0]
            )
            training_positives = ambiguity_gated_positive_mask(
                positives,
                features[:, :, 0],
                args.ambiguity_margin_threshold,
            )
            loss, loss_diag = multi_positive_assignment_loss(
                candidate_logits,
                null_logits,
                training_positives,
                null_loss_weight=args.null_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                all_logits = torch.cat(
                    [candidate_logits, null_logits[:, None]], dim=1
                )
                prediction = all_logits.argmax(dim=1)
                candidate_selected = prediction < args.topk
                selected_positive = torch.zeros_like(candidate_selected)
                selected_positive[candidate_selected] = positives[
                    candidate_selected, prediction[candidate_selected]
                ]
                target_has_positive = training_positives.any(dim=1)
                correct = selected_positive | (
                    ~candidate_selected & ~target_has_positive
                )
            epoch_records.append(
                {
                    "loss": float(loss.detach().item()),
                    "accuracy": float(correct.float().mean().item()),
                    "positive_recall": float(
                        selected_positive[target_has_positive].float().mean().item()
                        if bool(target_has_positive.any())
                        else 0.0
                    ),
                    "null_recall": float(
                        (~candidate_selected[~target_has_positive]).float().mean().item()
                        if bool((~target_has_positive).any())
                        else 0.0
                    ),
                    **loss_diag,
                }
            )
        summary = {
            key: float(sum(row[key] for row in epoch_records) / len(epoch_records))
            for key in ("loss", "accuracy", "positive_recall", "null_recall")
        }
        summary["epoch"] = epoch + 1
        history.append(summary)

    null_calibration = {}
    if args.calibrate_null:
        null_calibration = calibrate_null_bias(
            feature_cache,
            head,
            args.null_min_precision,
            args.assignment_global_preserve_scale,
        )

    aggregate = defaultdict(int)
    sequence_counts = defaultdict(lambda: defaultdict(int))
    head.eval()
    with torch.no_grad():
        for name, features, positives in feature_cache:
            features = features.to(device)
            positives = positives.to(device)
            candidate_logits, null_logits = head(features)
            candidate_logits = (
                candidate_logits
                + args.assignment_global_preserve_scale
                * features[:, :, 0]
            )
            counts = assignment_error_breakdown(
                candidate_logits,
                null_logits,
                positives,
                features[:, :, 0],
                ambiguity_margin_threshold=args.ambiguity_margin_threshold,
            )
            sequence = Path(name).parts[0] if Path(name).parts else "all"
            for key, value in counts.items():
                aggregate[key] += int(value)
                sequence_counts[sequence][key] += int(value)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": 1,
        "head_config": head.export_config(),
        "head_state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
        "landmark_indices": torch.as_tensor(
            state["landmark_indices"]
        ).reshape(-1).cpu(),
        **(
            {
                "landmark_statistics": landmark_statistics.detach().cpu(),
            }
            if landmark_statistics is not None
            else {}
        ),
        "config": {
            "topk": args.topk,
            "patch_radius": args.patch_radius,
            "patch_step_px": args.patch_step_px,
            "temperature": args.temperature,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "global_skip_temperature": args.global_skip_temperature,
            "bounded_residual_max": args.bounded_residual_max,
            "assignment_logit_temperature": args.assignment_logit_temperature,
            "ambiguity_margin_threshold": args.ambiguity_margin_threshold,
            "calibrate_null": args.calibrate_null,
            "null_min_precision": args.null_min_precision,
            "null_loss_weight": args.null_loss_weight,
            "initial_state": (
                str(Path(args.initial_state).resolve())
                if args.initial_state
                else ""
            ),
            "freeze_candidate": args.freeze_candidate,
            "assignment_global_preserve_scale": (
                args.assignment_global_preserve_scale
            ),
            "seed": args.seed,
            "query_count": len(feature_cache),
            "map_state": str(Path(args.map_state).resolve()),
            "landmark_statistics": (
                str(Path(args.landmark_statistics).resolve())
                if args.landmark_statistics
                else ""
            ),
            "global_attractor_statistics": (
                str(Path(args.global_attractor_statistics).resolve())
                if args.global_attractor_statistics
                else ""
            ),
            "landmark_statistic_names": landmark_statistic_names,
        },
        "history": history,
        "null_calibration": null_calibration,
        "assignment_error_breakdown": summarize_assignment_counts(aggregate),
        "assignment_error_breakdown_by_sequence": {
            key: summarize_assignment_counts(value)
            for key, value in sequence_counts.items()
        },
        "feature_cache_diagnostics": {
            "rows": int(sum(row["rows"] for row in diagnostics)),
            "topk_positive_rows": int(
                sum(row["topk_positive_rows"] for row in diagnostics)
            ),
        },
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                key: value
                for key, value in artifact.items()
                if key not in {
                    "head_state_dict",
                    "landmark_indices",
                    "landmark_statistics",
                }
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(artifact["history"][-1], indent=2))


if __name__ == "__main__":
    main()
