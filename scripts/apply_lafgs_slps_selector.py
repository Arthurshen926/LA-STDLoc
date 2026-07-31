#!/usr/bin/env python3
"""Apply a trained SLPS selector to an exact single-descriptor top-K graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from localization_training.pose_sufficient_selector import (
    build_pose_sufficient_features,
)
from localization_training.slps_selector import (
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    build_relation_groups,
    build_slps_features,
    slps_from_state,
)
from localization_training.slps_residual_signatures import (
    residual_signature_features,
)


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fixed-budget",
        type=int,
        default=0,
        help="Use one learned-order budget instead of adaptive risk stopping.",
    )
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    selector_state = torch.load(
        args.selector, map_location="cpu", weights_only=False
    )
    if topk.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported top-K outcomes")
    if selector_state.get("schema") != "lafgs_slps_selector":
        raise ValueError("unsupported selector state")
    if topk["provenance"].get("family_prototype_state_sha256") is not None:
        raise ValueError("SLPS application requires single descriptors")
    if (
        dict(topk["provenance"])
        != dict(selector_state["candidate_graph_contract"])
    ):
        raise ValueError("SLPS candidate graph contract differs")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if (
        int(topk["anchor_count"]) != len(anchor_ids)
        or topk["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
        or selector_state["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
    ):
        raise ValueError("SLPS map identity differs")

    device = torch.device(args.device)
    model = slps_from_state(selector_state, device=device)
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    track = torch.as_tensor(
        state.get("track_cluster_ids", state["dependency_group_ids"])
    ).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_type = torch.as_tensor(
        state.get("anchor_type", torch.zeros(len(anchor_ids)))
    ).long()
    statistics = {
        name: torch.as_tensor(value).float()
        for name, value in selector_state["anchor_statistics"].items()
    }
    stability = torch.as_tensor(
        selector_state["anchor_track_stability"]
    ).float()
    residual_state = selector_state.get("residual_signature_state")
    needs_residual_signatures = list(
        selector_state.get("feature_names", ())
    ) == list(SLPS_BIAS_AWARE_FEATURE_NAMES)
    if needs_residual_signatures and residual_state is None:
        raise ValueError("bias-aware selector misses residual signature state")
    selector_config = selector_state["selector_config"]
    output_records = []
    selected_counts = []
    fallback_count = 0
    runtimes = []
    with torch.inference_mode():
        for record in topk["records"]:
            name = str(record["query_name"])
            rows = torch.as_tensor(record["query_rows"]).long()
            topk_scores = torch.as_tensor(record["topk_scores"]).float()
            topk_indices = torch.as_tensor(
                record["topk_anchor_indices"]
            ).long()
            top1 = topk_indices[:, 0]
            cached = cache[name]
            keypoints = torch.as_tensor(
                cached["native_keypoints"]
            ).float()[rows]
            base = build_pose_sufficient_features(
                topk_scores,
                topk_indices,
                keypoints=keypoints,
                keypoint_scores=torch.as_tensor(
                    cached["native_scores"]
                ).float()[rows],
                image_hw=cached["native_input_hw"],
                source_groups=source,
                dependency_groups=dependency,
                anchor_statistics=statistics,
                entropy_temperature=float(
                    selector_state.get("entropy_temperature", 0.05)
                ),
                prior_strength=float(
                    selector_state.get("prior_strength", 12.0)
                ),
            )
            residual_features = None
            if residual_state is not None:
                residual_config = residual_state["config"]
                residual_features = residual_signature_features(
                    residual_state["statistics"],
                    anchor_indices=top1,
                    keypoints=keypoints,
                    image_hw=cached["native_input_hw"],
                    grid_size=int(residual_config["grid_size"]),
                    clip_px=float(residual_config["clip_px"]),
                    anchor_prior=float(residual_config["anchor_prior"]),
                    cell_prior=float(residual_config["cell_prior"]),
                    rate_prior=float(residual_config["rate_prior"]),
                )
            features = build_slps_features(
                base,
                xyz=xyz[top1],
                anchor_type=anchor_type[top1],
                track_groups=track[top1],
                track_stability=stability[top1],
                anchor_map_support=statistics["attempts"][top1],
                residual_signature_features=residual_features,
            )
            relation_groups = build_relation_groups(
                keypoints=keypoints,
                image_hw=cached["native_input_hw"],
                xyz=xyz[top1],
                dependency_groups=dependency[top1],
                source_groups=source[top1],
                track_groups=track[top1],
            )
            start = time.perf_counter()
            if int(args.fixed_budget) > 0:
                encoded = model.encode(features, relation_groups)
                ordering = model.greedy_order(
                    encoded,
                    relation_groups,
                    maximum_count=min(int(args.fixed_budget), len(features)),
                )
                selected = torch.zeros(len(features), dtype=torch.bool)
                selected[ordering.cpu()] = True
                diagnostics = {
                    "selected_budget": float(selected.sum()),
                    "fallback": 0.0,
                }
                safe_probability = float("nan")
                catastrophic_probability = float("nan")
                expected_hypotheses = float("nan")
                relative_utility_lcb = float("nan")
                relative_utility_median = float("nan")
                used_fallback = False
            else:
                encoded = model.encode(features, relation_groups)
                selection = model.select(
                    features,
                    relation_groups,
                    anchor_indices=top1,
                    query_name=name,
                    encoded=encoded,
                    **selector_config,
                )
                selected = selection.selected_mask
                diagnostics = selection.diagnostics
                safe_probability = selection.safe_probability
                catastrophic_probability = (
                    selection.catastrophic_probability
                )
                expected_hypotheses = selection.expected_hypotheses
                relative_utility_lcb = selection.relative_utility_lcb
                relative_utility_median = (
                    selection.relative_utility_median
                )
                used_fallback = selection.used_fallback
            runtime_ms = 1000.0 * (time.perf_counter() - start)
            runtimes.append(runtime_ms)
            selected_counts.append(int(selected.sum()))
            fallback_count += int(used_fallback)
            output_records.append(
                {
                    "query_name": name,
                    "query_rows": rows,
                    "topk_anchor_indices": topk_indices,
                    "topk_scores": topk_scores,
                    "selected_row_mask": selected,
                    "slps_safe_probability": safe_probability,
                    "slps_catastrophic_probability": (
                        catastrophic_probability
                    ),
                    "slps_expected_hypotheses": expected_hypotheses,
                    "slps_relative_utility_lcb": relative_utility_lcb,
                    "slps_relative_utility_median": (
                        relative_utility_median
                    ),
                    "slps_runtime_ms": runtime_ms,
                    "slps_diagnostics": diagnostics,
                }
            )
    summary = {
        "query_count": len(output_records),
        "selected_count_mean": float(
            torch.as_tensor(selected_counts).float().mean()
        ),
        "selected_count_median": float(
            torch.as_tensor(selected_counts).float().median()
        ),
        "fallback_rate": float(
            fallback_count / max(len(output_records), 1)
        ),
        "selector_runtime_ms_mean": float(
            torch.as_tensor(runtimes).float().mean()
        ),
    }
    output = Path(args.output).resolve()
    _atomic_torch(
        output,
        {
            "schema": "lafgs_exact_topk_outcomes",
            "version": 5,
            "query_names": list(topk["query_names"]),
            "query_start": int(topk.get("query_start", 0)),
            "topk": int(topk["topk"]),
            "anchor_count": int(topk["anchor_count"]),
            "anchor_ids_sha256": topk["anchor_ids_sha256"],
            "records": output_records,
            "method": (
                f"slps_fixed_{int(args.fixed_budget)}"
                if int(args.fixed_budget) > 0
                else "slps_adaptive_risk"
            ),
            "summary": summary,
            "provenance": dict(topk["provenance"]),
            "selector_state": str(Path(args.selector).resolve()),
        },
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
