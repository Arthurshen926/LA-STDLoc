#!/usr/bin/env python
"""Train a low-rank query descriptor adapter on a fixed landmark bank."""

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
from localization_training.local_context_adapter import (
    LocalContextMetricAdapter,
    pool_local_query_context,
)
from scripts.train_one_of_k_reranker import (
    candidate_positive_mask,
    normalized_landmark_statistics,
)
from train_lafgs_map import _cached_native_observations


def _first_positive_indices(observations):
    offsets = observations.positive_offsets
    indices = observations.positive_indices
    output = torch.full(
        (observations.query_uv.shape[0],),
        -1,
        dtype=torch.long,
        device=observations.query_uv.device,
    )
    if offsets is None or indices is None or indices.numel() == 0:
        return output
    has_positive = offsets[1:] > offsets[:-1]
    output[has_positive] = indices[offsets[:-1][has_positive]]
    return output


def multi_positive_metric_loss(scores, positive_mask):
    positive_mask = torch.as_tensor(
        positive_mask, device=scores.device, dtype=torch.bool
    )
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return scores.sum() * 0.0
    numerator = torch.logsumexp(
        scores[valid].masked_fill(~positive_mask[valid], -torch.inf), dim=1
    )
    denominator = torch.logsumexp(scores[valid], dim=1)
    return (denominator - numerator).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--map_state", required=True)
    parser.add_argument("--teacher_state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph_cache", default="")
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--query_end", type=int, default=0)
    parser.add_argument("--graph_only", action="store_true")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--patch_radius", type=int, default=2)
    parser.add_argument("--patch_step_px", type=float, default=8.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max_residual_norm", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--metric_temperature", type=float, default=0.07)
    parser.add_argument("--gt_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--trust_weight", type=float, default=1.0)
    parser.add_argument("--clean_trust_multiplier", type=float, default=4.0)
    parser.add_argument(
        "--teacher_global_preserve_scale", type=float, default=30.0
    )
    parser.add_argument("--teacher_use_null", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_state = torch.load(args.map_state, map_location="cpu")
    bank_features = F.normalize(
        torch.as_tensor(map_state["landmark_features"]).float().to(device), dim=1
    )
    bank_xyz = torch.as_tensor(map_state["landmark_xyz"]).float().to(device)

    teacher_state = torch.load(args.teacher_state, map_location="cpu")
    teacher_config = teacher_state.get("config", {})
    head_config = teacher_state["head_config"]
    teacher = OneOfKAssignmentHead(**head_config).to(device)
    teacher.load_state_dict(teacher_state["head_state_dict"])
    teacher.eval()
    landmark_statistics = teacher_state.get("landmark_statistics")
    if landmark_statistics is not None:
        teacher_indices = torch.as_tensor(
            teacher_state["landmark_indices"]
        ).reshape(-1)
        map_indices = torch.as_tensor(map_state["landmark_indices"]).reshape(-1)
        if not torch.equal(teacher_indices.cpu(), map_indices.cpu()):
            raise ValueError("teacher statistics do not align with map state")
        landmark_statistics = torch.as_tensor(
            landmark_statistics, device=device, dtype=bank_features.dtype
        )

    cache = torch.load(args.query_cache, map_location="cpu")["queries"]
    visibility = torch.load(
        args.visibility_cache, map_location="cpu"
    )["visibility"]
    query_names = sorted(set(cache) & set(visibility))
    query_names = query_names[
        max(args.query_start, 0) : (
            args.query_end if args.query_end > 0 else None
        )
    ]
    observation_args = SimpleNamespace(
        grid_rows=8,
        grid_cols=8,
        native_association_radius_px=2.0,
        native_unmatched_fraction=0.5,
        native_sampling_mode="detector_grid",
    )

    graph = []
    graph_diagnostics = {"rows": 0, "positive_in_topk": 0, "matchable": 0}
    graph_cache_path = Path(args.graph_cache) if args.graph_cache else None
    if graph_cache_path is not None and graph_cache_path.exists():
        graph_payload = torch.load(graph_cache_path, map_location="cpu")
        graph = graph_payload["graph"]
        graph_diagnostics = graph_payload["graph_diagnostics"]
        query_names = []
    for name in tqdm(query_names, desc="Fixed adapter graph"):
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
        sparse = F.normalize(observations.query_features, dim=1)
        global_scores = sparse @ bank_features.T
        top_scores, top_indices = torch.topk(
            global_scores, args.topk, dim=1
        )
        positive_mask = candidate_positive_mask(
            observations, top_indices, bank_features.shape[0]
        )
        local_features = build_one_of_k_features(
            observations.query_feature_map,
            observations.query_uv,
            top_indices,
            top_scores,
            bank_features,
            observations.query_feature_image_size,
            radius=args.patch_radius,
            step_px=args.patch_step_px,
            temperature=float(teacher_config.get("temperature", 0.07)),
            landmark_statistics=landmark_statistics,
        )
        with torch.no_grad():
            teacher_candidate, teacher_null = teacher(local_features)
            teacher_candidate = (
                teacher_candidate
                + float(args.teacher_global_preserve_scale) * top_scores
            )
            if args.teacher_use_null:
                teacher_logits = torch.cat(
                    [teacher_candidate, teacher_null[:, None]], dim=1
                )
            else:
                teacher_logits = teacher_candidate
            teacher_probability = torch.softmax(teacher_logits, dim=1)
        context = pool_local_query_context(
            observations.query_feature_map,
            observations.query_uv,
            observations.query_feature_image_size,
            radius=args.patch_radius,
            step_px=args.patch_step_px,
        )
        first_positive = _first_positive_indices(observations)
        graph.append(
            {
                "name": name,
                "sparse": sparse.detach().cpu().half(),
                "context": context.detach().cpu().half(),
                "top_indices": top_indices.detach().cpu(),
                "positive_mask": positive_mask.detach().cpu(),
                "first_positive": first_positive.detach().cpu(),
                "teacher_probability": teacher_probability.detach().cpu().half(),
            }
        )
        graph_diagnostics["rows"] += int(sparse.shape[0])
        graph_diagnostics["positive_in_topk"] += int(
            positive_mask.any(dim=1).sum().item()
        )
        graph_diagnostics["matchable"] += int(
            (first_positive >= 0).sum().item()
        )
    if (
        graph_cache_path is not None
        and not graph_cache_path.exists()
    ):
        graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graph": graph,
                "graph_diagnostics": graph_diagnostics,
                "map_state": str(Path(args.map_state).resolve()),
                "teacher_state": str(Path(args.teacher_state).resolve()),
                "topk": args.topk,
            },
            graph_cache_path,
        )
    if args.graph_only:
        print(json.dumps(graph_diagnostics, indent=2))
        return

    adapter = LocalContextMetricAdapter(
        descriptor_dim=int(bank_features.shape[1]),
        rank=args.rank,
        max_residual_norm=args.max_residual_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(graph)))
        random.Random(args.seed + epoch).shuffle(order)
        records = []
        adapter.train()
        for index in tqdm(order, desc=f"Adapter train {epoch + 1}/{args.epochs}"):
            record = graph[index]
            sparse = record["sparse"].float().to(device)
            context = record["context"].float().to(device)
            top_indices = record["top_indices"].to(device)
            positive_mask = record["positive_mask"].to(device)
            first_positive = record["first_positive"].to(device)
            teacher_probability = record["teacher_probability"].float().to(device)
            adapted, residual = adapter(sparse, context)
            candidate_scores = torch.einsum(
                "nd,nkd->nk", adapted, bank_features[top_indices]
            ) / max(args.metric_temperature, 1e-6)
            has_positive = first_positive >= 0
            safe_positive = first_positive.clamp_min(0)
            positive_score = (
                adapted * bank_features[safe_positive]
            ).sum(dim=1, keepdim=True) / max(args.metric_temperature, 1e-6)
            metric_scores = torch.cat([candidate_scores, positive_score], dim=1)
            metric_positive = torch.cat(
                [positive_mask, has_positive[:, None]], dim=1
            )
            gt_loss = multi_positive_metric_loss(
                metric_scores, metric_positive
            )
            student_log_probability = torch.log_softmax(
                candidate_scores, dim=1
            )
            teacher_candidate_probability = teacher_probability[
                :, : args.topk
            ]
            teacher_candidate_probability = (
                teacher_candidate_probability
                / teacher_candidate_probability.sum(dim=1, keepdim=True).clamp_min(
                    1e-8
                )
            )
            kd_loss = F.kl_div(
                student_log_probability,
                teacher_candidate_probability,
                reduction="batchmean",
            )
            clean_top1 = positive_mask[:, 0]
            trust_per_row = residual.square().sum(dim=1)
            trust_scale = torch.where(
                clean_top1,
                torch.full_like(
                    trust_per_row, args.clean_trust_multiplier
                ),
                torch.ones_like(trust_per_row),
            )
            trust_loss = (trust_per_row * trust_scale).mean()
            loss = (
                args.gt_weight * gt_loss
                + args.kd_weight * kd_loss
                + args.trust_weight * trust_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            records.append(
                {
                    "loss": float(loss.detach().item()),
                    "gt_loss": float(gt_loss.detach().item()),
                    "kd_loss": float(kd_loss.detach().item()),
                    "trust_loss": float(trust_loss.detach().item()),
                    "residual_norm": float(
                        torch.linalg.norm(residual, dim=1).mean().detach().item()
                    ),
                }
            )
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    key: sum(row[key] for row in records) / len(records)
                    for key in records[0]
                },
            }
        )
        torch.save(
            {
                "version": 1,
                "adapter_config": adapter.export_config(),
                "adapter_state_dict": {
                    key: value.detach().cpu()
                    for key, value in adapter.state_dict().items()
                },
                "landmark_indices": torch.as_tensor(
                    map_state["landmark_indices"]
                ).reshape(-1).cpu(),
                "config": {
                    **vars(args),
                    "map_state": str(Path(args.map_state).resolve()),
                    "teacher_state": str(Path(args.teacher_state).resolve()),
                },
                "graph_diagnostics": graph_diagnostics,
                "history": history,
            },
            output.with_name(
                f"{output.stem}.epoch{epoch + 1}{output.suffix}"
            ),
        )

    artifact = {
        "version": 1,
        "adapter_config": adapter.export_config(),
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        },
        "landmark_indices": torch.as_tensor(
            map_state["landmark_indices"]
        ).reshape(-1).cpu(),
        "config": {
            **vars(args),
            "map_state": str(Path(args.map_state).resolve()),
            "teacher_state": str(Path(args.teacher_state).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
            "visibility_cache": str(Path(args.visibility_cache).resolve()),
        },
        "graph_diagnostics": graph_diagnostics,
        "history": history,
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                key: value
                for key, value in artifact.items()
                if key not in {"adapter_state_dict", "landmark_indices"}
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
