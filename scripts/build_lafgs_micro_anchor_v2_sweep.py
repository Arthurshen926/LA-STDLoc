#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from localization_training.micro_anchors import (
    build_v2_materialized_anchor_map,
    truncate_materialized_anchor_map,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--visibility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="1500,2000,2500,3000")
    parser.add_argument(
        "--variants", default="m1_cluster,m2_visible_balanced,m3_identity"
    )
    parser.add_argument("--cluster-radius-m", type=float, default=0.015)
    parser.add_argument(
        "--cluster-min-descriptor-cosine", type=float, default=0.85
    )
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    args = parser.parse_args()

    base_path = Path(args.base_state).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    visibility_path = Path(args.visibility_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(
        {int(value) for value in args.budgets.split(",") if value.strip()}
    )
    variants = [value.strip() for value in args.variants.split(",")]
    known_variants = {
        "m1_cluster": {
            "visibility": False,
            "include_identity_split": False,
            "score_mode": "coverage_count",
        },
        "m2_visible_balanced": {
            "visibility": True,
            "include_identity_split": False,
            "score_mode": "balanced",
        },
        "m3_identity": {
            "visibility": True,
            "include_identity_split": True,
            "score_mode": "balanced",
        },
    }
    unknown = sorted(set(variants) - set(known_variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    base_state = torch.load(base_path, map_location="cpu", weights_only=False)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    print(f"Loading query cache: {query_path}", flush=True)
    query_payload = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    visibility_payload = torch.load(
        visibility_path, map_location="cpu", weights_only=False
    )
    visibility = visibility_payload.get("visibility", visibility_payload)
    provenance = {
        "base_state_path": str(base_path),
        "base_state_sha256": _sha256(base_path),
        "track_payload_path": str(payload_path),
        "track_payload_sha256": _sha256(payload_path),
        "query_cache_path": str(query_path),
        "query_cache_signature": query_payload.get("signature"),
        "visibility_cache_path": str(visibility_path),
        "visibility_cache_sha256": _sha256(visibility_path),
        "visibility_cache_signature": visibility_payload.get("signature"),
    }
    summary = {}
    for variant in variants:
        config = known_variants[variant]
        print(f"Building {variant}: {json.dumps(config)}", flush=True)
        full_state, diagnostics = build_v2_materialized_anchor_map(
            base_state=base_state,
            payload=payload,
            query_cache=query_cache,
            budget=10**9,
            visibility_cache=visibility if config["visibility"] else None,
            include_identity_split=config["include_identity_split"],
            score_mode=config["score_mode"],
            cluster_radius_m=args.cluster_radius_m,
            cluster_min_descriptor_cosine=(
                args.cluster_min_descriptor_cosine
            ),
            descriptor_trim_fraction=args.descriptor_trim_fraction,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        full_state["provenance"] = provenance
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        summary[variant] = {"full_diagnostics": diagnostics, "budgets": {}}
        for budget in budgets:
            state = truncate_materialized_anchor_map(full_state, budget)
            state["provenance"] = provenance
            output_path = variant_dir / f"micro_anchor_{budget:04d}.pt"
            torch.save(state, output_path)
            anchor_type = torch.as_tensor(state["anchor_type"])
            quality = state["micro_anchor_quality"]
            budget_diagnostics = {
                "state": str(output_path),
                "micro_anchor_count": int(state["micro_anchor_count"]),
                "coverage_anchor_count": int((anchor_type == 1).sum()),
                "identity_split_anchor_count": int((anchor_type == 2).sum()),
                "coverage_gain_sum": int(
                    torch.as_tensor(quality["coverage_gain"]).sum()
                ),
                "functional_gain_sum": int(
                    torch.as_tensor(quality["functional_gain"]).sum()
                ),
                "positive_margin_mean": float(
                    torch.as_tensor(
                        quality["positive_hardnegative_margin"]
                    ).float().mean()
                ),
                "false_attractor_incoming_sum": int(
                    torch.as_tensor(
                        quality["false_attractor_incoming_count"]
                    ).sum()
                ),
            }
            summary[variant]["budgets"][str(budget)] = budget_diagnostics
            print(json.dumps({variant: budget_diagnostics}), flush=True)
    (output_dir / "v2_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
