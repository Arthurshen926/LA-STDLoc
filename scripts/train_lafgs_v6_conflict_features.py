#!/usr/bin/env python
"""Refine only cached harmful-winner/clean-positive descriptor conflicts."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _first_mask_position(mask):
    positions = torch.arange(mask.shape[1])[None].expand_as(mask)
    sentinel = torch.full_like(positions, mask.shape[1])
    chosen = torch.where(mask, positions, sentinel).amin(dim=1)
    return chosen, chosen < mask.shape[1]


def _collect_pairs(graph, cache, state, max_conflict, max_replay):
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    canonical_count = int(graph["anchor_count"])
    selected = torch.as_tensor(
        state["functional_pruning"]["selected_source_rows"]
    ).long()
    if selected.numel() != count:
        raise ValueError(
            "selected_source_rows must map every materialized anchor"
        )
    if selected.min() < 0 or selected.max() >= canonical_count:
        raise ValueError("selected_source_rows exceeds canonical graph")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("selected_source_rows contains duplicates")
    canonical_to_local = torch.full(
        (canonical_count,), -1, dtype=torch.long
    )
    canonical_to_local[selected] = torch.arange(count)
    active = canonical_to_local >= 0
    conflict_query = []
    conflict_positive = []
    conflict_negative = []
    replay_query = []
    replay_positive = []
    replay_negative = []
    for name, record in zip(graph["query_names"], graph["records"]):
        indices = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)
        row_ids = torch.as_tensor(record["query_rows"]).long()
        active_candidates = active[indices]
        winner_position, winner_valid = _first_mask_position(
            active_candidates
        )
        positive_position, positive_valid = _first_mask_position(
            active_candidates & ((flags & 2) != 0)
        )
        row = torch.arange(indices.shape[0])
        safe_winner_position = winner_position.clamp_max(
            indices.shape[1] - 1
        )
        winner_legal2 = (
            flags[row, safe_winner_position] & 2
        ) != 0
        winner_legal4 = (
            flags[row, safe_winner_position] & 4
        ) != 0
        conflict = winner_valid & positive_valid & ~winner_legal4
        conflict_rows = torch.nonzero(conflict).reshape(-1)[
            : int(max_conflict)
        ]
        if conflict_rows.numel():
            positive = indices[
                conflict_rows, positive_position[conflict_rows]
            ]
            negative = indices[
                conflict_rows, winner_position[conflict_rows]
            ]
            query = torch.as_tensor(
                cache[name]["native_descriptors"]
            ).float()[row_ids[conflict_rows]]
            conflict_query.append(query)
            conflict_positive.append(canonical_to_local[positive])
            conflict_negative.append(canonical_to_local[negative])

        negative_position, negative_valid = _first_mask_position(
            active_candidates & ((flags & 4) == 0)
        )
        replay = winner_valid & winner_legal2 & negative_valid
        replay_rows = torch.nonzero(replay).reshape(-1)[
            : int(max_replay)
        ]
        if replay_rows.numel():
            positive = indices[
                replay_rows, winner_position[replay_rows]
            ]
            negative = indices[
                replay_rows, negative_position[replay_rows]
            ]
            query = torch.as_tensor(
                cache[name]["native_descriptors"]
            ).float()[row_ids[replay_rows]]
            replay_query.append(query)
            replay_positive.append(canonical_to_local[positive])
            replay_negative.append(canonical_to_local[negative])

    def merge(values, *, dim=0):
        if not values:
            raise RuntimeError("conflict feature training set is empty")
        return torch.cat(values, dim=dim)

    return {
        "conflict_query": F.normalize(merge(conflict_query), dim=1),
        "conflict_positive": merge(conflict_positive),
        "conflict_negative": merge(conflict_negative),
        "replay_query": F.normalize(merge(replay_query), dim=1),
        "replay_positive": merge(replay_positive),
        "replay_negative": merge(replay_negative),
    }


def _rank_loss(features, query, positive, negative, margin, temperature):
    normalized = F.normalize(features, dim=1)
    positive_score = (query * normalized[positive]).sum(dim=1)
    negative_score = (query * normalized[negative]).sum(dim=1)
    return (
        F.softplus(
            (negative_score - positive_score + float(margin))
            / float(temperature)
        )
        * float(temperature)
    ).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--function_graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--replay_weight", type=float, default=1.0)
    parser.add_argument("--trust_weight", type=float, default=0.05)
    parser.add_argument("--max_conflict_per_query", type=int, default=256)
    parser.add_argument("--max_replay_per_query", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        graph["query_cache"], map_location="cpu", weights_only=False
    )["queries"]
    pairs = _collect_pairs(
        graph,
        cache,
        state,
        args.max_conflict_per_query,
        args.max_replay_per_query,
    )
    del cache, graph
    device = torch.device("cuda")
    original = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    features = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([features], lr=float(args.lr))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    conflict_count = int(pairs["conflict_query"].shape[0])
    replay_count = int(pairs["replay_query"].shape[0])
    touched = torch.unique(
        torch.cat(
            [
                pairs["conflict_positive"],
                pairs["conflict_negative"],
                pairs["replay_positive"],
                pairs["replay_negative"],
            ]
        )
    ).to(device)
    log = []
    for step in range(1, int(args.steps) + 1):
        conflict_index = torch.randint(
            conflict_count,
            (min(int(args.batch_size), conflict_count),),
            generator=generator,
        )
        replay_index = torch.randint(
            replay_count,
            (min(int(args.batch_size), replay_count),),
            generator=generator,
        )
        conflict_loss = _rank_loss(
            features,
            pairs["conflict_query"][conflict_index].to(device),
            pairs["conflict_positive"][conflict_index].to(device),
            pairs["conflict_negative"][conflict_index].to(device),
            args.margin,
            args.temperature,
        )
        replay_loss = _rank_loss(
            features,
            pairs["replay_query"][replay_index].to(device),
            pairs["replay_positive"][replay_index].to(device),
            pairs["replay_negative"][replay_index].to(device),
            args.margin,
            args.temperature,
        )
        trust = (
            features[touched] - original[touched]
        ).square().sum(dim=1).mean()
        loss = (
            conflict_loss
            + float(args.replay_weight) * replay_loss
            + float(args.trust_weight) * trust
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            features.copy_(F.normalize(features, dim=1))
        if step == 1 or step % 50 == 0 or step == int(args.steps):
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "conflict_loss": float(conflict_loss.detach()),
                "replay_loss": float(replay_loss.detach()),
                "trust_loss": float(trust.detach()),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
    output = dict(state)
    output["anchor_features"] = F.normalize(
        features.detach(), dim=1
    ).cpu()
    output["conflict_feature_refinement"] = {
        "schema": "lafgs_v6_conflict_feature_refinement",
        "version": 1,
        "conflict_pair_count": conflict_count,
        "replay_pair_count": replay_count,
        "touched_anchor_count": int(touched.numel()),
        "descriptor_drift": {
            "mean": float(
                (output["anchor_features"] - original.cpu())
                .norm(dim=1)
                .mean()
            ),
            "p95": float(
                torch.quantile(
                    (output["anchor_features"] - original.cpu()).norm(dim=1),
                    0.95,
                )
            ),
            "max": float(
                (output["anchor_features"] - original.cpu())
                .norm(dim=1)
                .max()
            ),
            "touched_mean": float(
                (
                    output["anchor_features"][touched.cpu()]
                    - original.cpu()[touched.cpu()]
                )
                .norm(dim=1)
                .mean()
            ),
        },
        "config": vars(args),
        "log": log,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    (path.parent / f"{path.stem}_report.json").write_text(
        json.dumps(output["conflict_feature_refinement"], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
