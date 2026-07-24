#!/usr/bin/env python3
"""Distill a protected one-of-K teacher into landmark descriptors."""

import argparse
import json
import random
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
from localization_training.detector_free_map import (
    materialize_descriptor_residual,
)
from scripts.train_one_of_k_reranker import candidate_positive_mask
from train_lafgs_map import _cached_native_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--map_state", required=True)
    parser.add_argument("--reranker_state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--max_residual_norm", type=float, default=0.05)
    parser.add_argument("--trust_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(args.map_state, map_location="cpu")
    teacher_state = torch.load(args.reranker_state, map_location="cpu")
    source_indices = torch.as_tensor(source["landmark_indices"]).reshape(-1)
    teacher_indices = torch.as_tensor(
        teacher_state["landmark_indices"]
    ).reshape(-1)
    if not torch.equal(source_indices.cpu(), teacher_indices.cpu()):
        raise ValueError("reranker and descriptor field landmark IDs differ")
    initial = F.normalize(
        torch.as_tensor(source["landmark_features"]).float().to(device),
        dim=1,
    )
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.AdamW([residual], lr=args.learning_rate)
    bank_xyz = torch.as_tensor(source["landmark_xyz"]).float().to(device)
    map_statistics = teacher_state.get("landmark_statistics")
    if map_statistics is not None:
        map_statistics = torch.as_tensor(map_statistics).float().to(device)
    head_config = teacher_state["head_config"]
    head = OneOfKAssignmentHead(
        hidden_dim=int(head_config["hidden_dim"]),
        feature_dim=int(head_config["feature_dim"]),
        global_skip_scale=float(head_config.get("global_skip_scale", 0.0)),
    ).to(device)
    head.load_state_dict(teacher_state["head_state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    teacher_config = teacher_state["config"]
    topk = int(teacher_config["topk"])

    cache = torch.load(args.query_cache, map_location="cpu")["queries"]
    visibility = torch.load(
        args.visibility_cache, map_location="cpu"
    )["visibility"]
    names = sorted(set(cache) & set(visibility))
    if args.max_queries > 0:
        names = names[: args.max_queries]
    observation_args = SimpleNamespace(
        grid_rows=8,
        grid_cols=8,
        native_association_radius_px=2.0,
        native_unmatched_fraction=0.5,
        native_sampling_mode="detector_grid",
    )
    history = []
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(names)
        records = []
        for name in tqdm(names, desc=f"Reranker field distill {epoch + 1}"):
            features = materialize_descriptor_residual(
                initial,
                residual,
                residual_scale=1.0,
                max_residual_norm=args.max_residual_norm,
            )
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
            full_scores = query @ features.T
            top_scores, top_indices = torch.topk(full_scores, topk, dim=1)
            positives = candidate_positive_mask(
                observations, top_indices, features.shape[0]
            )
            has_positive = positives.any(dim=1)
            if not bool(has_positive.any()):
                continue
            local = build_one_of_k_features(
                observations.query_feature_map,
                observations.query_uv,
                top_indices,
                top_scores,
                features,
                observations.query_feature_image_size,
                radius=int(teacher_config["patch_radius"]),
                step_px=float(teacher_config["patch_step_px"]),
                temperature=float(teacher_config["temperature"]),
                landmark_statistics=map_statistics,
            )
            with torch.no_grad():
                teacher_logits, _ = head(local)
                masked_teacher = teacher_logits.masked_fill(
                    ~positives, -torch.inf
                )
                target = masked_teacher.argmax(dim=1)
                clean_top1 = positives[:, 0]
                target[clean_top1] = 0
            rows = torch.nonzero(has_positive, as_tuple=False).reshape(-1)
            selected = top_scores[rows, target[rows]]
            negative = top_scores[rows].clone()
            negative[
                torch.arange(rows.numel(), device=device), target[rows]
            ] = -torch.inf
            hardest_negative = negative.max(dim=1).values
            ranking = F.relu(
                hardest_negative - selected + float(args.margin)
            ).mean()
            trust = residual.square().sum(dim=1).mean()
            loss = ranking + float(args.trust_weight) * trust
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([residual], 1.0)
            optimizer.step()
            records.append(
                {
                    "loss": float(loss.detach().item()),
                    "ranking": float(ranking.detach().item()),
                    "target_rows": int(rows.numel()),
                    "keep_rows": int(clean_top1.sum().item()),
                    "swap_rows": int(
                        (has_positive & ~clean_top1).sum().item()
                    ),
                }
            )
        history.append(
            {
                key: (
                    float(sum(record[key] for record in records) / len(records))
                    if key in {"loss", "ranking"}
                    else int(sum(record[key] for record in records))
                )
                for key in (
                    "loss",
                    "ranking",
                    "target_rows",
                    "keep_rows",
                    "swap_rows",
                )
            }
        )

    distilled = materialize_descriptor_residual(
        initial,
        residual,
        residual_scale=1.0,
        max_residual_norm=args.max_residual_norm,
    ).detach().cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source)
    result["landmark_features"] = distilled
    result["config"] = {
        **dict(source.get("config", {})),
        "one_of_k_teacher_distillation": {
            "teacher": str(Path(args.reranker_state).resolve()),
            "epochs": args.epochs,
            "margin": args.margin,
            "max_residual_norm": args.max_residual_norm,
            "protect_clean_top1": True,
            "gt_positive_gate": True,
        },
    }
    result["one_of_k_distillation_history"] = history
    torch.save(result, output)
    summary = {
        "output": str(output.resolve()),
        "history": history,
        "raw_residual_norm_mean": float(
            torch.linalg.norm(residual.detach(), dim=1).mean().item()
        ),
        "raw_residual_norm_max": float(
            torch.linalg.norm(residual.detach(), dim=1).max().item()
        ),
        "effective_descriptor_delta_mean": float(
            torch.linalg.norm(distilled - initial.detach().cpu(), dim=1)
            .mean()
            .item()
        ),
        "effective_descriptor_delta_max": float(
            torch.linalg.norm(distilled - initial.detach().cpu(), dim=1)
            .max()
            .item()
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
