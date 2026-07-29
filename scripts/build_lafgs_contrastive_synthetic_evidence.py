#!/usr/bin/env python3
"""Relabel accepted renders with confusion-conditioned contrastive evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.confusion_evidence import (
    ContrastiveEvidenceConfig,
    build_contrastive_synthetic_record,
    pack_contrastive_synthetic_evidence,
)
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.synthetic_evidence import (
    SyntheticEvidenceConfig,
    synthetic_function_graph_payload,
    synthetic_positive_teacher_payload,
    synthetic_query_cache_payload,
)


def _load_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--synthetic-evidence", required=True)
    parser.add_argument(
        "--render-manifest",
        default="",
        help="Optional confusion view manifest used to preserve target edges.",
    )
    parser.add_argument(
        "--include-positive-only-records",
        action="store_true",
        help=(
            "Keep accepted strong/ambiguous records without a target pair for "
            "the positive-only R3 appearance-pool control."
        ),
    )
    parser.add_argument(
        "--target-positive-only",
        action="store_true",
        help=(
            "Expose only the planned correct family as a strong positive; "
            "other geometric neighbors remain ambiguous context."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strong-radius-px", type=float, default=2.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=6.0)
    parser.add_argument(
        "--maximum-positives-per-keypoint", type=int, default=4
    )
    parser.add_argument(
        "--maximum-negatives-per-keypoint", type=int, default=4
    )
    parser.add_argument(
        "--maximum-negative-score-gap", type=float, default=0.12
    )
    parser.add_argument(
        "--minimum-hard-negative-pairs", type=int, default=8
    )
    parser.add_argument("--minimum-edge-occurrences", type=int, default=2)
    parser.add_argument(
        "--require-negative-visibility",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    source = torch.load(
        args.synthetic_evidence, map_location="cpu", weights_only=False
    )
    if source.get("schema") != (
        "lafgs_artifact_filtered_synthetic_appearance_evidence"
    ):
        raise ValueError("source evidence is not artifact-filtered rendering")
    target_by_name = {}
    if args.render_manifest:
        for manifest_record in _load_jsonl(args.render_manifest):
            meta = manifest_record.get("meta", {})
            required = {
                "edge_index",
                "correct_anchor",
                "confusing_anchor",
            }
            if not required.issubset(meta):
                raise ValueError(
                    "confusion render manifest misses target edge metadata"
                )
            target_by_name[str(manifest_record["query_id"])] = {
                key: meta[key]
                for key in (
                    "edge_index",
                    "correct_anchor",
                    "confusing_anchor",
                    "image_cell",
                    "source_query",
                    "neighbor_query",
                    "cross_trajectory",
                    "acquisition",
                )
                if key in meta
            }
    records = []
    config = ContrastiveEvidenceConfig(
        strong_radius_px=args.strong_radius_px,
        ambiguous_radius_px=args.ambiguous_radius_px,
        maximum_positives_per_keypoint=args.maximum_positives_per_keypoint,
        maximum_negatives_per_keypoint=args.maximum_negatives_per_keypoint,
        maximum_negative_score_gap=args.maximum_negative_score_gap,
        minimum_hard_negative_pairs=args.minimum_hard_negative_pairs,
        minimum_edge_occurrences=args.minimum_edge_occurrences,
        require_negative_visibility=args.require_negative_visibility,
        restrict_strong_to_active_target=args.target_positive_only,
    )
    device = torch.device(args.device)
    for index, record in enumerate(source["records"]):
        record = dict(record)
        if target_by_name:
            name = str(record["query_name"])
            if name not in target_by_name:
                raise ValueError(
                    f"accepted render {name} has no planned target edge"
                )
            record["active_evidence_target"] = target_by_name[name]
        render = torch.load(
            record["render_evidence_path"],
            map_location="cpu",
            weights_only=False,
        )
        visibility_config = SyntheticEvidenceConfig(
            **{
                key: value
                for key, value in record["config"].items()
                if key in SyntheticEvidenceConfig.__dataclass_fields__
            }
        )
        relabeled = build_contrastive_synthetic_record(
            record=record,
            state=state,
            metric=metric,
            family=family,
            confusion_graph=graph,
            rendered_depth=render["depth"],
            alpha=render["alpha"],
            visibility_config=visibility_config,
            config=config,
            device=device,
        )
        records.append(relabeled)
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(source["records"]),
                    "strong": relabeled["strong_positive_pair_count"],
                    "ambiguous": relabeled["ambiguous_pair_count"],
                    "hard_negative": relabeled["hard_negative_pair_count"],
                    "accepted": relabeled["contrastive_accepted"],
                }
            ),
            flush=True,
        )
    evidence = pack_contrastive_synthetic_evidence(
        records,
        source={
            "synthetic_evidence": str(
                Path(args.synthetic_evidence).resolve()
            ),
            "confusion_graph": str(Path(args.confusion_graph).resolve()),
            "family_state": str(Path(args.family_state).resolve()),
            "map": str(Path(args.map).resolve()),
            "metric_state": str(Path(args.metric_state).resolve()),
            "config": vars(args),
        },
        confusion_graph=graph,
        include_positive_only_records=args.include_positive_only_records,
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(evidence, path)
    torch.save(
        synthetic_query_cache_payload(evidence),
        path.with_name(path.stem + "_query_cache.pt"),
    )
    torch.save(
        synthetic_positive_teacher_payload(
            evidence,
            anchor_count=int(torch.as_tensor(state["anchor_xyz"]).shape[0]),
        ),
        path.with_name(path.stem + "_positive_teacher.pt"),
    )
    torch.save(
        synthetic_function_graph_payload(
            evidence,
            anchor_count=int(torch.as_tensor(state["anchor_xyz"]).shape[0]),
        ),
        path.with_name(path.stem + "_function_graph.pt"),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": evidence["schema"],
                "version": evidence["version"],
                "summary": evidence["summary"],
                "source": evidence["source"],
                "confusion_graph_summary": evidence[
                    "confusion_graph_summary"
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
