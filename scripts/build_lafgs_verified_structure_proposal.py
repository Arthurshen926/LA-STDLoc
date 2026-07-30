#!/usr/bin/env python3
"""Build add-only or add/swap maps from verified coverage evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.artifact_contract import sha256_file
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.verified_structure_update import (
    VerifiedStructureConfig,
    collect_coverage_evidence,
    robust_structure_descriptor,
    safe_retirement_candidates,
    serialize_config,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _materialize_map(
    state: dict,
    canonical: dict,
    additions: list[dict],
    addition_features: torch.Tensor,
    retirements: list[int],
    report: dict,
) -> tuple[dict, torch.Tensor]:
    active_count = len(state["anchor_ids"])
    keep = torch.ones(active_count, dtype=torch.bool)
    if retirements:
        keep[torch.as_tensor(retirements).long()] = False
    old_to_new = torch.full((active_count,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(int(keep.sum()))
    canonical_rows = torch.as_tensor(
        [value["canonical_index"] for value in additions]
    ).long()
    output = dict(state)
    canonical_fields = (
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_type",
    )
    for key in canonical_fields:
        output[key] = torch.cat(
            (
                torch.as_tensor(state[key])[keep],
                torch.as_tensor(canonical[key])[canonical_rows],
            )
        )
    output["anchor_features"] = torch.cat(
        (
            torch.as_tensor(state["anchor_features"])[keep],
            addition_features.cpu(),
        )
    )
    new_count = len(output["anchor_features"])
    output["anchor_ids"] = torch.arange(new_count, dtype=torch.long)
    dependency_start = int(
        torch.as_tensor(
            state.get(
                "coarse_dependency_group_ids",
                state["dependency_group_ids"],
            )
        ).max()
    ) + 1
    new_dependency = torch.arange(
        dependency_start,
        dependency_start + len(additions),
        dtype=torch.long,
    )
    for key in ("dependency_group_ids", "coarse_dependency_group_ids"):
        source = torch.as_tensor(
            state.get(key, state["dependency_group_ids"])
        ).long()
        output[key] = torch.cat((source[keep], new_dependency))
    fine = torch.as_tensor(
        state.get("fine_identity_ids", state["track_cluster_ids"])
    ).long()
    output["fine_identity_ids"] = torch.cat(
        (
            fine[keep],
            torch.as_tensor(canonical["track_cluster_ids"])[canonical_rows]
            .long(),
        )
    )
    if "v7_metric_raw_features" in state:
        output["v7_metric_raw_features"] = torch.cat(
            (
                torch.as_tensor(state["v7_metric_raw_features"])[keep],
                addition_features.cpu(),
            )
        )
    output["canonical_anchor_count"] = new_count
    output["micro_anchor_count"] = (
        new_count - int(output["base_anchor_count"])
    )
    output["requested_micro_anchor_budget"] = int(
        output["micro_anchor_count"]
    )
    output["verified_structure_update"] = report
    return output, old_to_new


def _remap_family(
    family: dict,
    *,
    old_to_new: torch.Tensor,
    new_anchor_count: int,
    report: dict,
) -> dict:
    parents = torch.as_tensor(
        family["prototype_anchor_indices"]
    ).long()
    keep = old_to_new[parents] >= 0
    output = dict(family)
    for key in (
        "prototype_features",
        "prototype_bias",
        "prototype_temperature",
    ):
        output[key] = torch.as_tensor(family[key])[keep]
    output["prototype_anchor_indices"] = old_to_new[parents[keep]]
    output["landmark_indices"] = torch.arange(
        new_anchor_count, dtype=torch.long
    )
    if (
        isinstance(family.get("families"), list)
        and len(family["families"]) == len(parents)
    ):
        output["families"] = [
            value
            for value, selected in zip(
                family["families"], keep.tolist()
            )
            if selected
        ]
    output["verified_structure_update"] = report
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--family-prototype-state", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-family", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-trajectories", type=int, default=3)
    parser.add_argument("--minimum-events", type=int, default=3)
    parser.add_argument("--maximum-additions", type=int, default=128)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--enable-retire", action="store_true")
    parser.add_argument("--maximum-retirements", type=int, default=128)
    args = parser.parse_args()
    paths = {
        "map": Path(args.map).resolve(),
        "canonical_map": Path(args.canonical_map).resolve(),
        "family_prototype_state": Path(
            args.family_prototype_state
        ).resolve(),
        "metric_state": Path(args.metric_state).resolve(),
        "triage": Path(args.triage).resolve(),
        "dynamic_outcomes": Path(args.dynamic_outcomes).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
    }
    payload = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    state = payload["map"]
    canonical = payload["canonical_map"]
    family = payload["family_prototype_state"]
    triage = payload["triage"]
    dynamic = payload["dynamic_outcomes"]
    cache = payload["query_cache"].get(
        "queries", payload["query_cache"]
    )
    if int(triage["active_anchor_count"]) != len(state["anchor_ids"]):
        raise ValueError("structure triage does not align with map")
    config = VerifiedStructureConfig(
        minimum_trajectories=args.minimum_trajectories,
        minimum_events=args.minimum_events,
        maximum_additions=args.maximum_additions,
        descriptor_trim_fraction=args.descriptor_trim_fraction,
        maximum_retirements=args.maximum_retirements,
    )
    evidence_config = VerifiedStructureConfig(
        **{
            **serialize_config(config),
            "maximum_additions": len(canonical["anchor_ids"]),
        }
    )
    additions, diagnostics = collect_coverage_evidence(
        triage, config=evidence_config
    )
    active_identities = set(
        zip(
            torch.as_tensor(state["source_primitive_ids"]).long().tolist(),
            torch.as_tensor(state["track_cluster_ids"]).long().tolist(),
        )
    )
    additions = [
        value
        for value in additions
        if (
            int(
                canonical["source_primitive_ids"][
                    value["canonical_index"]
                ]
            ),
            int(
                canonical["track_cluster_ids"][
                    value["canonical_index"]
                ]
            ),
        )
        not in active_identities
    ][: max(int(config.maximum_additions), 0)]
    device = torch.device(args.device)
    metric_payload = payload["metric_state"]
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(
        device
    )
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    query_names = list(triage["query_names"])
    transformed_by_query = {}
    with torch.inference_mode():
        for query_index in sorted(
            {
                event["query_index"]
                for value in additions
                for event in value["events"]
            }
        ):
            name = query_names[query_index]
            rows = sorted(
                {
                    event["query_row"]
                    for value in additions
                    for event in value["events"]
                    if event["query_index"] == query_index
                }
            )
            row_tensor = torch.as_tensor(rows).long()
            raw = F.normalize(
                torch.as_tensor(cache[name]["native_descriptors"])
                .float()[row_tensor]
                .to(device),
                dim=1,
            )
            transformed, _ = metric(raw)
            transformed_by_query[query_index] = {
                row: feature.cpu()
                for row, feature in zip(
                    rows, F.normalize(transformed, dim=1)
                )
            }
    fused = []
    for value in additions:
        observations = torch.stack(
            [
                transformed_by_query[event["query_index"]][
                    event["query_row"]
                ]
                for event in value["events"]
            ]
        )
        weights = torch.as_tensor(
            [
                event["contribution_mass"]
                / (1.0 + event["reprojection_error_px"] ** 2)
                for event in value["events"]
            ]
        ).float()
        fused.append(
            robust_structure_descriptor(
                observations,
                weights,
                canonical["anchor_features"][
                    value["canonical_index"]
                ],
                trim_fraction=config.descriptor_trim_fraction,
            )
        )
    addition_features = (
        torch.stack(fused)
        if fused
        else torch.empty(
            0,
            torch.as_tensor(state["anchor_features"]).shape[1],
        )
    )
    retirements = (
        safe_retirement_candidates(
            active_count=len(state["anchor_ids"]),
            base_anchor_count=int(state["base_anchor_count"]),
            triage=triage,
            dynamic_outcomes=dynamic,
            family_parent_indices=family["prototype_anchor_indices"],
            config=config,
        )
        if args.enable_retire
        else []
    )
    retirements = retirements[: len(additions)]
    report = {
        "schema": "lafgs_verified_structure_update",
        "version": 1,
        "mode": "add_swap" if args.enable_retire else "add_only",
        "config": serialize_config(config),
        "candidate_diagnostics": diagnostics,
        "added_canonical_indices": [
            value["canonical_index"] for value in additions
        ],
        "added_count": len(additions),
        "retired_anchor_indices": retirements,
        "retired_count": len(retirements),
        "provenance": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    updated_map, old_to_new = _materialize_map(
        state,
        canonical,
        additions,
        addition_features,
        retirements,
        report,
    )
    updated_family = _remap_family(
        family,
        old_to_new=old_to_new,
        new_anchor_count=len(updated_map["anchor_ids"]),
        report=report,
    )
    _atomic_torch(Path(args.output_map).resolve(), updated_map)
    _atomic_torch(Path(args.output_family).resolve(), updated_family)
    report_path = Path(args.report_output).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
