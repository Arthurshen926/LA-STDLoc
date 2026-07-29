#!/usr/bin/env python3
"""Build same-geometry appearance families for sparse localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.appearance_family import (
    build_appearance_mode_pool,
    build_cross_view_families,
    collapse_duplicate_conflicts,
    materialize_family_artifact,
)
from localization_training.shared_metric import SharedLowRankMetric


def _query_bins(payload: dict) -> dict[str, int]:
    return {
        str(name): int(group)
        for name, group in zip(
            payload["query_names"],
            torch.as_tensor(payload["query_bins"]).tolist(),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("collapse_duplicate", "cross_view_stable", "appearance_pool"),
        required=True,
    )
    parser.add_argument("--conflict-map", default="")
    parser.add_argument("--basin-teacher", default="")
    parser.add_argument("--query-cache", default="")
    parser.add_argument("--track-payload", default="")
    parser.add_argument("--metric-state", default="")
    parser.add_argument("--complete-positive-teacher", default="")
    parser.add_argument("--dynamic-outcomes", default="")
    parser.add_argument("--synthetic-evidence", default="")
    parser.add_argument(
        "--maximum-synthetic-to-real-ratio", type=float, default=0.25
    )
    parser.add_argument("--synthetic-targets-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-modes-per-anchor", type=int, default=3)
    parser.add_argument("--minimum-separation", type=float, default=0.08)
    parser.add_argument("--initial-bias", type=float, default=-0.01)
    parser.add_argument("--maximum-prototypes", type=int, default=0)
    parser.add_argument("--minimum-observations", type=int, default=4)
    parser.add_argument("--minimum-trajectories", type=int, default=2)
    parser.add_argument("--minimum-view-bins", type=int, default=3)
    parser.add_argument(
        "--minimum-counterfactual-observations", type=int, default=1
    )
    parser.add_argument("--minimum-oof-gain", type=float, default=0.01)
    parser.add_argument(
        "--minimum-oof-positive-fraction", type=float, default=0.6
    )
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument(
        "--maximum-primary-similarity", type=float, default=0.98
    )
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    observation_payload = None
    if args.mode == "collapse_duplicate":
        if not args.conflict_map:
            raise ValueError("collapse_duplicate requires --conflict-map")
        candidates = collapse_duplicate_conflicts(
            state,
            torch.load(
                args.conflict_map, map_location="cpu", weights_only=False
            ),
        )
    else:
        required = (
            args.query_cache,
            args.track_payload,
            args.metric_state,
        )
        if not all(required):
            raise ValueError(f"{args.mode} inputs are incomplete")
        cache_payload = torch.load(
            args.query_cache, map_location="cpu", weights_only=False
        )
        cache = cache_payload.get("queries", cache_payload)
        payload = torch.load(
            args.track_payload, map_location="cpu", weights_only=False
        )
        metric_payload = torch.load(
            args.metric_state, map_location="cpu", weights_only=False
        )
        device = torch.device(args.device)
        metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(
            device
        )
        metric.load_state_dict(metric_payload["metric_state_dict"])
        metric.eval()
        if args.mode == "cross_view_stable":
            if not args.basin_teacher:
                raise ValueError(
                    "cross_view_stable requires --basin-teacher"
                )
            candidates = build_cross_view_families(
                state,
                torch.load(
                    args.basin_teacher,
                    map_location="cpu",
                    weights_only=False,
                ),
                cache,
                _query_bins(payload),
                metric,
                minimum_observations=args.minimum_observations,
                minimum_trajectories=args.minimum_trajectories,
                minimum_view_bins=args.minimum_view_bins,
                minimum_counterfactual_observations=(
                    args.minimum_counterfactual_observations
                ),
                minimum_oof_gain=args.minimum_oof_gain,
                minimum_oof_positive_fraction=(
                    args.minimum_oof_positive_fraction
                ),
                trim_fraction=args.trim_fraction,
                maximum_primary_similarity=args.maximum_primary_similarity,
            )
        else:
            if not args.complete_positive_teacher:
                raise ValueError(
                    "appearance_pool requires --complete-positive-teacher"
                )
            synthetic_evidence = (
                torch.load(
                    args.synthetic_evidence,
                    map_location="cpu",
                    weights_only=False,
                )
                if args.synthetic_evidence
                else None
            )
            if args.synthetic_targets_only and synthetic_evidence is None:
                raise ValueError(
                    "--synthetic-targets-only requires --synthetic-evidence"
                )
            anchor_filter = (
                torch.unique(
                    torch.cat(
                        [
                            torch.as_tensor(
                                record["positive_indices"]
                            ).long()
                            for record in synthetic_evidence["records"]
                            if torch.as_tensor(
                                record["positive_indices"]
                            ).numel()
                        ]
                    )
                )
                if args.synthetic_targets_only
                else None
            )
            candidates, observation_payload = build_appearance_mode_pool(
                state,
                torch.load(
                    args.complete_positive_teacher,
                    map_location="cpu",
                    weights_only=False,
                ),
                cache,
                _query_bins(payload),
                metric,
                dynamic=(
                    torch.load(
                        args.dynamic_outcomes,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if args.dynamic_outcomes
                    else None
                ),
                basin_teacher=(
                    torch.load(
                        args.basin_teacher,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if args.basin_teacher
                    else None
                ),
                maximum_modes_per_anchor=args.maximum_modes_per_anchor,
                minimum_observations=args.minimum_observations,
                minimum_trajectories=args.minimum_trajectories,
                minimum_view_bins=args.minimum_view_bins,
                minimum_separation=args.minimum_separation,
                maximum_primary_similarity=args.maximum_primary_similarity,
                trim_fraction=args.trim_fraction,
                initial_bias=args.initial_bias,
                device=device,
                synthetic_evidence=synthetic_evidence,
                maximum_synthetic_to_real_ratio=(
                    args.maximum_synthetic_to_real_ratio
                ),
                anchor_filter=anchor_filter,
            )
    if int(args.maximum_prototypes) > 0:
        candidates = candidates[: int(args.maximum_prototypes)]
        if observation_payload is not None:
            offsets = observation_payload["offsets"]
            stop = int(offsets[len(candidates)])
            observation_payload = {
                **observation_payload,
                "offsets": offsets[: len(candidates) + 1].clone(),
                "query_indices": observation_payload["query_indices"][
                    :stop
                ].clone(),
                "query_rows": observation_payload["query_rows"][:stop].clone(),
                "provenance": observation_payload["provenance"][:stop].clone(),
            }
    output = materialize_family_artifact(
        state,
        candidates,
        config=vars(args),
        observation_payload=observation_payload,
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    report = {
        "schema": output["schema"],
        "prototype_count": len(candidates),
        "geometry_anchor_count": int(
            torch.as_tensor(state["anchor_xyz"]).shape[0]
        ),
        "unique_parent_count": len(
            {value["source_anchor"] for value in candidates}
        ),
        "config": vars(args),
        "families": output["families"],
    }
    path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
