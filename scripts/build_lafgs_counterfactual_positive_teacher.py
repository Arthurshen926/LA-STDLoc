#!/usr/bin/env python3
"""Build counterfactual pose-validated descriptor repair targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.counterfactual_positive_teacher import (
    CounterfactualTeacherConfig,
    build_counterfactual_positive_teacher,
)
from localization_training.selection_aware_reconstruction import build_mode_table
from localization_training.shared_metric import SharedLowRankMetric


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-prototype-state", required=True)
    parser.add_argument("--selected-outcomes", required=True)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--source-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--translation-scale-m", type=float, default=0.07160573943725686
    )
    parser.add_argument("--rotation-scale-degrees", type=float, default=2.0)
    parser.add_argument("--residual-scale-px", type=float, default=4.0)
    parser.add_argument("--residual-clip-px", type=float, default=12.0)
    parser.add_argument(
        "--minimum-primary-trajectory-support", type=int, default=4
    )
    parser.add_argument(
        "--minimum-family-trajectory-support", type=int, default=2
    )
    args = parser.parse_args()
    paths = {
        "map": args.map,
        "metric_state": args.metric_state,
        "family_prototype_state": args.family_prototype_state,
        "selected_outcomes": args.selected_outcomes,
        "triage": args.triage,
        "source_positive_teacher": args.source_positive_teacher,
        "query_cache": args.query_cache,
    }
    payload = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    device = torch.device(args.device)
    metric_payload = payload["metric_state"]
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    family = payload["family_prototype_state"]
    table = build_mode_table(
        payload["map"]["anchor_features"],
        family["prototype_features"],
        family["prototype_anchor_indices"],
        family.get("prototype_bias", torch.zeros(len(family["prototype_features"]))),
        family.get(
            "prototype_temperature",
            torch.ones(len(family["prototype_features"])),
        ),
    )
    config = CounterfactualTeacherConfig(
        translation_scale_m=args.translation_scale_m,
        rotation_scale_degrees=args.rotation_scale_degrees,
        residual_scale_px=args.residual_scale_px,
        residual_clip_px=args.residual_clip_px,
        minimum_primary_trajectory_support=(
            args.minimum_primary_trajectory_support
        ),
        minimum_family_trajectory_support=(
            args.minimum_family_trajectory_support
        ),
    )

    def progress(completed: int, total: int, summary: dict) -> None:
        if completed % 25 == 0 or completed == total:
            print(json.dumps({"completed": completed, "total": total, **summary}), flush=True)

    audit, teacher = build_counterfactual_positive_teacher(
        state=payload["map"],
        selected_outcomes=payload["selected_outcomes"],
        triage=payload["triage"],
        source_positive_teacher=payload["source_positive_teacher"],
        query_cache=payload["query_cache"],
        metric=metric,
        mode_table=table,
        config=config,
        device=device,
        progress=progress,
    )
    provenance = {
        name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    audit["provenance"] = provenance
    teacher["provenance"] = {
        **dict(teacher.get("provenance", {})),
        "counterfactual_positive_teacher": provenance,
    }
    output = Path(args.output).resolve()
    teacher_output = Path(args.teacher_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(output, audit)
    _atomic_torch(teacher_output, teacher)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": audit["schema"],
                "version": audit["version"],
                "summary": audit["summary"],
                "config": audit["config"],
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(audit["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
