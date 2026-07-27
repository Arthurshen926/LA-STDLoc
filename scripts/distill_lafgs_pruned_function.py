#!/usr/bin/env python3
"""Short multi-positive functional distillation after Active Map pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _student_positions(
    full_indices: torch.Tensor,
    full_to_student: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student = full_to_student[full_indices.long()]
    retained = student >= 0
    return student, retained


@torch.no_grad()
def _mine_training_data(
    original: dict,
    student: dict,
    graph: dict,
    query_cache: dict,
    *,
    max_positives: int,
):
    full_count = int(original["anchor_xyz"].shape[0])
    student_rows = torch.as_tensor(
        student["functional_pruning"]["selected_source_rows"]
    ).long()
    full_to_student = torch.full((full_count,), -1, dtype=torch.long)
    full_to_student[student_rows] = torch.arange(student_rows.numel())
    records = []
    train_student_rows = []
    names = graph["query_names"]
    for record in graph["records"]:
        query_index = int(record["query_index"])
        indices = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"])
        solver_inlier = torch.as_tensor(record["solver_inlier"]).bool()
        query_rows = torch.as_tensor(record["query_rows"]).long()
        student_indices, retained = _student_positions(
            indices, full_to_student
        )
        retained_positions = torch.where(
            retained,
            torch.arange(indices.shape[1])[None],
            torch.full_like(indices, indices.shape[1]),
        )
        first_position = retained_positions.min(dim=1).values
        has_student = first_position < indices.shape[1]
        student_top1 = student_indices.gather(
            1, first_position.clamp_max(indices.shape[1] - 1)[:, None]
        ).squeeze(1)
        teacher_deleted = full_to_student[indices[:, 0]] < 0
        legal4 = retained & ((flags & 4) != 0)
        recoverable = teacher_deleted & legal4.any(dim=1) & has_student
        legal2_teacher_winner = ((flags[:, 0] & 2) != 0)
        guard = (
            (~teacher_deleted)
            & solver_inlier
            & legal2_teacher_winner
        )
        selected = recoverable | guard
        if not selected.any():
            continue
        selected_rows = torch.nonzero(
            selected, as_tuple=False
        ).reshape(-1)
        positive_lists = []
        harmful_competitors = []
        for row in selected_rows.tolist():
            positives = student_indices[row][legal4[row]]
            positives = torch.unique(positives, sorted=False)[
                :max_positives
            ]
            positive_lists.append(positives)
            if recoverable[row]:
                competitor = student_top1[row]
                if competitor >= 0 and not torch.isin(
                    competitor, positives
                ):
                    harmful_competitors.append(competitor)
        train_student_rows.extend(
            positive
            for positives in positive_lists
            for positive in positives.tolist()
        )
        train_student_rows.extend(
            int(value) for value in harmful_competitors
        )
        records.append(
            {
                "query_index": query_index,
                "cache_rows": query_rows[selected_rows],
                "recoverable": recoverable[selected_rows],
                "guard": guard[selected_rows],
                "student_top1": student_top1[selected_rows],
                "positive_lists": positive_lists,
            }
        )
    train_rows = torch.unique(
        torch.as_tensor(train_student_rows, dtype=torch.long)
    )
    row_to_local = torch.full(
        (student_rows.numel(),), -1, dtype=torch.long
    )
    row_to_local[train_rows] = torch.arange(train_rows.numel())
    descriptors = []
    positive_targets = []
    recoverable_mask = []
    guard_mask = []
    guard_targets = []
    for record in records:
        cached = query_cache[names[record["query_index"]]]
        descriptors.append(
            F.normalize(
                torch.as_tensor(cached["native_descriptors"])[
                    record["cache_rows"]
                ].float(),
                dim=1,
            )
        )
        targets = torch.full(
            (len(record["positive_lists"]), max_positives),
            -1,
            dtype=torch.long,
        )
        for row, positives in enumerate(record["positive_lists"]):
            local = row_to_local[positives]
            local = local[local >= 0]
            targets[row, : local.numel()] = local[:max_positives]
        positive_targets.append(targets)
        recoverable_mask.append(record["recoverable"])
        guard_mask.append(record["guard"])
        guard_targets.append(row_to_local[record["student_top1"]])
    return {
        "train_rows": train_rows,
        "descriptors": torch.cat(descriptors),
        "positive_targets": torch.cat(positive_targets),
        "recoverable": torch.cat(recoverable_mask),
        "guard": torch.cat(guard_mask),
        "guard_target": torch.cat(guard_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-map", required=True)
    parser.add_argument("--student-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", default="50,100")
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--max-residual-norm", type=float, default=0.015)
    parser.add_argument("--guard-weight", type=float, default=8.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=0.5)
    parser.add_argument("--guard-margin", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Functional distillation requires CUDA")
    torch.manual_seed(args.seed)
    original = torch.load(
        args.original_map, map_location="cpu", weights_only=False
    )
    student = torch.load(
        args.student_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = payload.get("queries", payload)
    data = _mine_training_data(
        original,
        student,
        graph,
        query_cache,
        max_positives=args.max_positives,
    )
    device = torch.device("cuda")
    all_features = F.normalize(
        torch.as_tensor(student["anchor_features"]).float(), dim=1
    )
    train_rows = data["train_rows"]
    initial = all_features[train_rows].to(device)
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.AdamW(
        [residual], lr=args.learning_rate, weight_decay=0.0
    )
    descriptors = data["descriptors"]
    positive_targets = data["positive_targets"]
    recoverable = data["recoverable"]
    guard = data["guard"]
    guard_target = data["guard_target"]
    sample_weight = torch.where(
        guard,
        torch.full_like(guard, 4.0, dtype=torch.float32),
        torch.ones_like(guard, dtype=torch.float32),
    )
    sample_weight /= sample_weight.sum()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    checkpoints.add(args.steps)
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        batch = torch.multinomial(
            sample_weight,
            num_samples=args.batch_size,
            replacement=True,
            generator=generator,
        )
        query = descriptors[batch].to(device)
        targets_positive = positive_targets[batch].to(device)
        positives = torch.zeros(
            batch.numel(),
            train_rows.numel(),
            dtype=torch.bool,
            device=device,
        )
        valid_positive = targets_positive >= 0
        positive_rows = torch.arange(
            batch.numel(), device=device
        )[:, None].expand_as(targets_positive)
        positives[
            positive_rows[valid_positive],
            targets_positive[valid_positive],
        ] = True
        batch_guard = guard[batch].to(device)
        batch_recoverable = recoverable[batch].to(device)
        targets = guard_target[batch].to(device)
        candidate = F.normalize(initial + residual, dim=1)
        logits = query @ candidate.T / float(args.temperature)
        initial_logits = query @ initial.T / float(args.temperature)
        positive_logits = logits.masked_fill(
            ~positives, float("-inf")
        )
        has_positive = positives.any(dim=1) & batch_recoverable
        positive_loss = torch.zeros((), device=device)
        if has_positive.any():
            positive_loss = -(
                torch.logsumexp(positive_logits[has_positive], dim=1)
                - torch.logsumexp(logits[has_positive], dim=1)
            ).mean()
        valid_guard_target = batch_guard & (targets >= 0)
        guard_loss = torch.zeros((), device=device)
        guard_violation = torch.zeros((), device=device)
        if valid_guard_target.any():
            guard_logits = logits[valid_guard_target]
            guard_initial = initial_logits[valid_guard_target]
            local_target = targets[valid_guard_target]
            target_new = guard_logits.gather(
                1, local_target[:, None]
            ).squeeze(1)
            target_old = guard_initial.gather(
                1, local_target[:, None]
            ).squeeze(1)
            competitor = guard_logits.clone()
            competitor.scatter_(
                1, local_target[:, None], float("-inf")
            )
            competitor = competitor.max(dim=1).values
            violation = F.relu(
                competitor - target_old + float(args.guard_margin)
            )
            preservation = F.relu(target_old - target_new)
            guard_loss = (violation + preservation).mean()
            guard_violation = (violation > 0).float().mean()
        kd_loss = F.smooth_l1_loss(logits, initial_logits.detach())
        trust_loss = (
            1.0
            - F.cosine_similarity(
                F.normalize(initial + residual, dim=1), initial, dim=1
            )
        ).mean()
        loss = (
            positive_loss
            + float(args.guard_weight) * guard_loss
            + float(args.kd_weight) * kd_loss
            + float(args.trust_weight) * trust_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = torch.linalg.norm(residual, dim=1, keepdim=True)
            residual.mul_(
                (
                    float(args.max_residual_norm)
                    / norm.clamp_min(1e-8)
                ).clamp(max=1.0)
            )
        if step == 1 or step % 25 == 0 or step in checkpoints:
            item = {
                "step": step,
                "loss": float(loss),
                "positive_loss": float(positive_loss),
                "guard_loss": float(guard_loss),
                "guard_violation_rate": float(guard_violation),
                "kd_loss": float(kd_loss),
                "trust_loss": float(trust_loss),
                "residual_norm_mean": float(
                    torch.linalg.norm(residual, dim=1).mean()
                ),
                "residual_norm_max": float(
                    torch.linalg.norm(residual, dim=1).max()
                ),
            }
            history.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
        if step in checkpoints:
            output = dict(student)
            features = all_features.clone()
            features[train_rows] = F.normalize(
                initial + residual, dim=1
            ).detach().cpu()
            output["anchor_features"] = features
            output["functional_distillation"] = {
                "step": step,
                "train_row_count": int(train_rows.numel()),
                "sample_count": int(descriptors.shape[0]),
                "recoverable_count": int(recoverable.sum()),
                "guard_count": int(guard.sum()),
                "config": vars(args),
                "history": history,
            }
            torch.save(
                output, output_dir / f"anchor_map_step_{step:04d}.pt"
            )
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_functional_distillation",
                "version": 1,
                "train_row_count": int(train_rows.numel()),
                "sample_count": int(descriptors.shape[0]),
                "recoverable_count": int(recoverable.sum()),
                "guard_count": int(guard.sum()),
                "history": history,
                "config": vars(args),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
