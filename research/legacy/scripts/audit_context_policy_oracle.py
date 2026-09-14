#!/usr/bin/env python3
"""Audit the multi-seed PoseLib oracle between A1 and shared context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from map_learning.context_metric import context_from_cached_query
from map_learning.context_policy_oracle import summarize_policy_oracle
from map_learning.context_score_expert import ContextScoreExpert, protocol_name
from map_learning.metric_context_uplift import load_frozen_metric_state
from map_learning.repeated_assignment_audit import _solve_assignments


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    return seeds


def _uniform_subset(values: list[int], requested: int) -> list[int]:
    if requested <= 0 or requested >= len(values):
        return list(values)
    positions = (
        torch.linspace(0, len(values) - 1, steps=int(requested))
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    return [values[position] for position in positions]


def _load_expert(checkpoint: dict, device: torch.device) -> ContextScoreExpert:
    config = checkpoint["expert_config"]
    expert = ContextScoreExpert(
        descriptor_dim=int(config["descriptor_dim"]),
        code_dim=int(config["code_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        context_kernels=tuple(config["context_kernels"]),
        context_mode=str(config["context_mode"]),
        input_scope=str(config["input_scope"]),
        learned_query_gate=bool(config.get("learned_query_gate", False)),
    ).to(device)
    expert.load_state_dict(checkpoint["expert_state_dict"])
    return expert.eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-weight", type=float, default=0.01)
    parser.add_argument("--seeds", type=_parse_seeds, default=(2026, 2027, 2028))
    parser.add_argument("--query-count-per-direction", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.context_weight <= 0.0 or args.query_count_per_direction < 0:
        raise ValueError("context weight must be positive and query count non-negative")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")

    report = json.loads(args.crossfit_report.read_text())
    if bool(report.get("uses_test_queries", True)):
        raise ValueError("policy oracle accepts mapping-only crossfit reports")
    inputs = report["inputs"]
    state = _load(Path(inputs["map"]))
    metric_state = _load(Path(inputs["metric_state"]))
    teacher = _load(Path(inputs["complete_positive_teacher"]))
    query_cache = _load(Path(inputs["query_cache"]))
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    device = torch.device(args.device)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    metric = load_frozen_metric_state(
        metric_state, anchor_ids=anchor_ids, device=device
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    context_name = protocol_name(float(args.context_weight))
    reprojection_px = float(report["config"]["ransac_reprojection_px"])
    pose_rows = []
    direction_reports = []
    for direction in report["directions"]:
        checkpoint_path = Path(direction["checkpoint"])
        checkpoint = _load(checkpoint_path)
        expert = _load_expert(checkpoint, device)
        support = set(int(value) for value in checkpoint["support_query_indices"])
        held_out = [index for index in range(len(names)) if index not in support]
        held_out = _uniform_subset(held_out, int(args.query_count_per_direction))
        anchor_indices = torch.as_tensor(
            checkpoint["anchor_indices"], device=device
        ).long()
        base_bank = torch.as_tensor(
            checkpoint["base_anchor_features"], device=device
        ).float()
        context_bank = torch.as_tensor(
            checkpoint["context_anchor_codes"], device=device
        ).float()
        direction_rows = 0
        for completed, query_index in enumerate(held_out, start=1):
            name = names[query_index]
            record = teacher["records"][query_index]
            rows = torch.as_tensor(record["query_rows"]).long()
            if not rows.numel():
                continue
            cached = cache[name]
            raw, tokens = context_from_cached_query(
                cached,
                rows,
                device=device,
                kernels=expert.context_kernels,
                context_mode=expert.context_mode,
            )
            base, _ = metric(raw)
            context_codes, _ = expert.query(base, tokens)
            base_scores = base @ base_bank.T
            context_scores = base_scores + float(args.context_weight) * (
                context_codes @ context_bank.T
            )
            winners = {
                "a1": anchor_indices[base_scores.argmax(dim=1)].cpu(),
                context_name: anchor_indices[context_scores.argmax(dim=1)].cpu(),
            }
            keypoints = (
                torch.as_tensor(cached["native_keypoints"]).float()[rows]
                + float(cached.get("pixel_center_offset", 0.5))
            ).cpu()
            intrinsic = torch.as_tensor(cached["native_K"]).float().cpu()
            gt_pose = torch.as_tensor(cached["pose_w2c"]).float().cpu()
            for seed in args.seeds:
                pose_row = {
                    "query_index": query_index,
                    "image_name": name,
                    "direction": direction["direction"],
                    "seed": int(seed),
                }
                for policy, assignments in winners.items():
                    result = _solve_assignments(
                        assignments,
                        keypoints=keypoints,
                        xyz=xyz,
                        intrinsic=intrinsic,
                        gt_pose=gt_pose,
                        reprojection_error_px=reprojection_px,
                        seed=int(seed),
                    )
                    pose_row.update(
                        {f"{policy}_{key}": value for key, value in result.items()}
                    )
                pose_rows.append(pose_row)
                direction_rows += 1
            if args.progress_interval > 0 and (
                completed % int(args.progress_interval) == 0
                or completed == len(held_out)
            ):
                print(
                    {
                        "event": "context_policy_oracle",
                        "direction": direction["direction"],
                        "queries_complete": completed,
                        "query_count": len(held_out),
                    },
                    flush=True,
                )
        direction_reports.append(
            {
                "direction": direction["direction"],
                "checkpoint": str(checkpoint_path.resolve()),
                "support_query_count": len(support),
                "held_out_query_count": len(held_out),
                "pose_row_count": direction_rows,
            }
        )
        del expert, base_bank, context_bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loss_config = {
        "translation_scale_cm": 5.0,
        "rotation_scale_deg": 5.0,
        "catastrophe_cm": 100.0,
        "catastrophe_weight": 2.0,
        "hypothesis_scale": 1000.0,
        "hypothesis_weight": 0.05,
    }
    summary = summarize_policy_oracle(
        pose_rows,
        context_protocol=context_name,
        seeds=args.seeds,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=2026,
        loss_config=loss_config,
    )
    output = {
        "schema": "lafgs_context_policy_oracle",
        "version": 1,
        "uses_test_queries": False,
        "crossfit_report": str(args.crossfit_report.resolve()),
        "context_weight": float(args.context_weight),
        "context_protocol": context_name,
        "seeds": list(args.seeds),
        "query_count_per_direction": int(args.query_count_per_direction),
        "reprojection_error_px": reprojection_px,
        "directions": direction_reports,
        "summary": summary,
        "pose_rows": pose_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        {
            "event": "context_policy_oracle_complete",
            "output": str(args.output),
            "summary": {key: value for key, value in summary.items() if key != "queries"},
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
