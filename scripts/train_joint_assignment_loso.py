#!/usr/bin/env python3
"""Train a scene-agnostic top-K/null assignment head with LOSO supervision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.joint_assignment_training import (
    weighted_multi_positive_assignment_loss,
)
from localization_training.local_assignment import OneOfKAssignmentHead


def _trajectory_block_split(graph, seed, block_count=5):
    """Hold out contiguous mapping blocks, including one-trajectory scenes."""

    grouped = defaultdict(list)
    for record in graph["records"]:
        grouped[record["trajectory"]].append(record["query_name"])
    if not grouped:
        raise ValueError("joint-assignment graph has no records")

    calibration = set()
    training = set()
    blocks = []
    for trajectory, names in sorted(grouped.items()):
        names = sorted(names)
        if len(names) < 2:
            training.update(names)
            blocks.append(
                {
                    "trajectory": trajectory,
                    "block_count": 1,
                    "calibration_block": None,
                    "query_count": len(names),
                }
            )
            continue
        local_block_count = min(max(int(block_count), 2), len(names))
        generator = random.Random(
            int(seed)
            + sum(map(ord, graph["scene"]))
            + 1009 * sum(map(ord, trajectory))
        )
        calibration_block = generator.randrange(local_block_count)
        for position, name in enumerate(names):
            block = min(
                position * local_block_count // len(names),
                local_block_count - 1,
            )
            (calibration if block == calibration_block else training).add(name)
        blocks.append(
            {
                "trajectory": trajectory,
                "block_count": local_block_count,
                "calibration_block": calibration_block,
                "query_count": len(names),
            }
        )
    if not calibration or not training:
        raise ValueError(
            f"temporal-block split is empty for {graph['scene']}"
        )
    return calibration, training, blocks


def _forward_record(head, record, device):
    features, positive, target_weights, row_weights = _record_tensors(
        record, device
    )
    candidate_logits, null_logits = head(features)
    return features, positive, target_weights, row_weights, candidate_logits, null_logits


def _record_tensors(record, device):
    staged = record.get("_staged_device_tensors")
    if staged is not None:
        if any(value.device != device for value in staged):
            raise ValueError("joint-assignment record was staged on another device")
        return staged
    staged = (
        torch.as_tensor(record["features"]).float().to(device),
        torch.as_tensor(record["positive_mask"]).bool().to(device),
        torch.as_tensor(record["candidate_target_weights"]).float().to(device),
        torch.as_tensor(record["row_weights"]).float().to(device),
    )
    record["_staged_device_tensors"] = staged
    return staged


def _stage_records(records, device):
    for record in records:
        _record_tensors(record, device)


def _forward_record_batch(head, records, device):
    tensors = [_record_tensors(record, device) for record in records]
    lengths = [len(values[0]) for values in tensors]
    features = torch.cat([values[0] for values in tensors], dim=0)
    candidate, null = head(features)
    outputs = []
    offset = 0
    for values, length in zip(tensors, lengths):
        outputs.append(
            (
                values[1],
                values[2],
                values[3],
                candidate[offset : offset + length],
                null[offset : offset + length],
            )
        )
        offset += length
    return outputs


def _calibrate_null_bias(
    records,
    head,
    device,
    minimum_precision,
    maximum_matchable_rejection_rate=0.01,
):
    deltas = []
    labels = []
    head.eval()
    with torch.no_grad():
        for record in records:
            _, positive, target_weights, _, candidate, null = _forward_record(
                head, record, device
            )
            positive = positive & (target_weights > 0)
            deltas.append((null - candidate.max(dim=1).values).cpu())
            labels.append((~positive.any(dim=1)).cpu())
    delta = torch.cat(deltas)
    true_null = torch.cat(labels)
    order = torch.argsort(delta, descending=True, stable=True)
    precision = true_null[order].float().cumsum(0) / torch.arange(
        1, len(order) + 1, dtype=torch.float32
    )
    false_rejections = (~true_null[order]).float().cumsum(0)
    false_rejection_rate = false_rejections / (~true_null).sum().clamp_min(1)
    valid = torch.nonzero(
        (precision >= float(minimum_precision))
        & (false_rejection_rate <= float(maximum_matchable_rejection_rate)),
        as_tuple=False,
    ).reshape(-1)
    threshold = (
        float(delta[order[valid[-1]]])
        if len(valid)
        else float(delta.max() + 1.0)
    )
    head.null_bias -= threshold
    return {
        "minimum_precision": float(minimum_precision),
        "maximum_matchable_rejection_rate": float(
            maximum_matchable_rejection_rate
        ),
        "raw_delta_threshold": threshold,
        "calibrated_null_bias": float(head.null_bias),
        "calibration_rows": len(delta),
    }


def _assignment_counts(records, head, device, ambiguity_threshold):
    totals = defaultdict(int)
    head.eval()
    with torch.no_grad():
        for record in records:
            features, positive, target_weights, _, candidate, null = _forward_record(
                head, record, device
            )
            positive = positive & (target_weights > 0)
            margin = features[:, 0, 0] - features[:, 1:, 0].max(dim=1).values
            ambiguous = margin < float(ambiguity_threshold)
            selected = candidate.argmax(dim=1)
            selected = torch.where(ambiguous, selected, torch.zeros_like(selected))
            row = torch.arange(len(selected), device=device)
            best = candidate[row, selected]
            null_selected = null >= best
            selected_positive = positive[row, selected] & ~null_selected
            top1_positive = positive[:, 0]
            has_positive = positive.any(dim=1)
            swapped = selected != 0
            values = {
                "rows": len(selected),
                "positive_rows": int(has_positive.sum()),
                "selected_positive": int(selected_positive.sum()),
                "clean_top1": int(top1_positive.sum()),
                "clean_top1_retained": int(
                    (top1_positive & ~swapped & ~null_selected).sum()
                ),
                "beneficial_swaps": int(
                    (swapped & ~top1_positive & selected_positive).sum()
                ),
                "harmful_swaps": int(
                    (swapped & top1_positive & ~selected_positive).sum()
                ),
                "null_selected": int(null_selected.sum()),
                "true_null": int((~has_positive).sum()),
                "true_null_selected": int((null_selected & ~has_positive).sum()),
                "matchable_rejected": int((null_selected & has_positive).sum()),
                "ambiguous": int(ambiguous.sum()),
            }
            for key, value in values.items():
                totals[key] += int(value)
    result = dict(totals)
    result.update(
        {
            "conditional_positive_accuracy": totals["selected_positive"]
            / max(totals["positive_rows"], 1),
            "clean_top1_retention": totals["clean_top1_retained"]
            / max(totals["clean_top1"], 1),
            "beneficial_swap_rate": totals["beneficial_swaps"]
            / max(totals["rows"], 1),
            "harmful_swap_rate": totals["harmful_swaps"]
            / max(totals["rows"], 1),
            "null_precision": totals["true_null_selected"]
            / max(totals["null_selected"], 1),
            "null_recall": totals["true_null_selected"]
            / max(totals["true_null"], 1),
            "matchable_false_rejection_rate": totals["matchable_rejected"]
            / max(totals["positive_rows"], 1),
        }
    )
    return result


def _calibrate_ambiguity_threshold(records, head, device):
    candidates = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, float("inf"))
    evaluations = {
        str(value): _assignment_counts(records, head, device, value)
        for value in candidates
    }
    feasible = []
    for threshold in candidates:
        row = evaluations[str(threshold)]
        if (
            row["clean_top1_retention"] >= 0.995
            and row["matchable_false_rejection_rate"] <= 0.01
        ):
            utility = (
                row["beneficial_swaps"]
                - 8.0 * row["harmful_swaps"]
                - 4.0 * row["matchable_rejected"]
            )
            feasible.append((utility, -float(threshold), threshold))
    selected = max(feasible)[2] if feasible else 0.005
    return float(selected), evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="append", required=True)
    parser.add_argument("--heldout-scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--bounded-residual-max", type=float, default=0.05)
    parser.add_argument("--null-loss-weight", type=float, default=0.5)
    parser.add_argument("--null-minimum-precision", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    graphs = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.graph
    ]
    by_scene = {graph["scene"]: graph for graph in graphs}
    if len(by_scene) != len(graphs) or args.heldout_scene not in by_scene:
        raise ValueError("graphs must contain unique scenes including heldout")
    graph_contracts = {
        (graph.get("schema"), int(graph.get("version", -1))) for graph in graphs
    }
    if graph_contracts != {("lafgs_joint_assignment_scene_graph", 5)}:
        raise ValueError(
            "cross-scene training requires ambiguity-clean v5 scene graphs"
        )
    topks = {int(graph["topk"]) for graph in graphs}
    feature_names = {tuple(graph["feature_names"]) for graph in graphs}
    if len(topks) != 1 or len(feature_names) != 1:
        raise ValueError("cross-scene graphs must share top-K and feature contracts")
    topk = next(iter(topks))
    names = list(next(iter(feature_names)))

    training_by_scene = {}
    calibration = []
    split_summary = {}
    for scene, graph in by_scene.items():
        if scene == args.heldout_scene:
            continue
        calibration_queries, training_queries, temporal_blocks = (
            _trajectory_block_split(graph, args.seed)
        )
        training = [
            record
            for record in graph["records"]
            if record["query_name"] in training_queries
        ]
        local_calibration = [
            record
            for record in graph["records"]
            if record["query_name"] in calibration_queries
        ]
        if not training or not local_calibration:
            raise ValueError(f"trajectory-block split is empty for {scene}")
        training_by_scene[scene] = training
        calibration.extend(local_calibration)
        split_summary[scene] = {
            "mode": "contiguous_temporal_blocks",
            "temporal_blocks": temporal_blocks,
            "training_queries": len(training),
            "calibration_queries": len(local_calibration),
        }

    for records in training_by_scene.values():
        _stage_records(records, device)
    _stage_records(calibration, device)

    head = OneOfKAssignmentHead(
        hidden_dim=args.hidden_dim,
        feature_dim=len(names),
        bounded_residual_max=args.bounded_residual_max,
        null_feature_mode="pooled_full",
        normalize_candidate_features=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    scene_names = sorted(training_by_scene)
    maximum_queries = max(len(training_by_scene[scene]) for scene in scene_names)
    for epoch in range(args.epochs):
        scene_orders = {}
        for scene in scene_names:
            order = list(range(len(training_by_scene[scene])))
            random.Random(args.seed + epoch * 1009 + sum(map(ord, scene))).shuffle(order)
            scene_orders[scene] = order
        head.train()
        totals = defaultdict(float)
        steps = 0
        query_steps = 0
        for position in range(maximum_queries):
            batch_records = []
            for scene in scene_names:
                order = scene_orders[scene]
                batch_records.append(
                    training_by_scene[scene][order[position % len(order)]]
                )
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for positive, target_weights, row_weights, candidate, null in (
                _forward_record_batch(head, batch_records, device)
            ):
                loss, diagnostics = weighted_multi_positive_assignment_loss(
                    candidate,
                    null,
                    positive,
                    candidate_target_weights=target_weights,
                    row_weights=row_weights,
                    null_loss_weight=args.null_loss_weight,
                )
                losses.append(loss)
                totals["positive_loss"] += diagnostics["positive_loss"]
                totals["null_loss"] += diagnostics["null_loss"]
                query_steps += 1
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += sum(float(loss.detach()) for loss in losses)
            steps += 1
        history.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "queries": query_steps,
                **{
                    key: value / max(query_steps, 1)
                    for key, value in totals.items()
                },
            }
        )

    null_calibration = _calibrate_null_bias(
        calibration, head, device, args.null_minimum_precision
    )
    ambiguity_threshold, ambiguity_grid = _calibrate_ambiguity_threshold(
        calibration, head, device
    )
    calibration_summary = _assignment_counts(
        calibration, head, device, ambiguity_threshold
    )
    heldout_graph = by_scene[args.heldout_scene]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "lafgs_joint_assignment_loso",
        "version": 5,
        "head_config": head.export_config(),
        "head_state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
        "landmark_indices": torch.load(
            heldout_graph["config"]["map"], map_location="cpu", weights_only=False
        )["anchor_ids"],
        "landmark_statistics": heldout_graph["anchor_statistics"],
        "config": {
            "topk": topk,
            "patch_radius": int(heldout_graph["config"]["patch_radius"]),
            "patch_step_px": float(heldout_graph["config"]["patch_step_px"]),
            "temperature": 0.07,
            "context_version": 1,
            "context_feature_names": names,
            "ambiguity_margin_threshold": ambiguity_threshold,
            "metric_state": heldout_graph["config"]["metric_state"],
            "heldout_scene": args.heldout_scene,
            "training_scenes": scene_names,
            "trajectory_block_calibration": True,
            "candidate_top1_reference": "deployment_chunked_exact_top1",
            "ambiguous_candidate_policy": (
                "exclude_loose_radius_only_rows_from_assignment_and_null_supervision"
            ),
            "seed": args.seed,
            "maximum_null_fraction": 0.25,
            "null_grid_rows": 4,
            "null_grid_cols": 4,
            "null_minimum_kept_per_grid": 8,
        },
        "training": vars(args),
        "split_summary": split_summary,
        "history": history,
        "null_calibration": null_calibration,
        "ambiguity_calibration": ambiguity_grid,
        "calibration_summary": calibration_summary,
    }
    torch.save(artifact, output)
    report = {
        "output": str(output),
        "heldout_scene": args.heldout_scene,
        "training_scenes": scene_names,
        "history": history,
        "null_calibration": null_calibration,
        "ambiguity_threshold": ambiguity_threshold,
        "calibration_summary": calibration_summary,
        "split_summary": split_summary,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
