#!/usr/bin/env python3
"""Audit descriptor ambiguity and top-K pose headroom on mapping images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from localization.localizer import load_shared_metric
from map_learning.repeated_assignment_audit import (
    DEFAULT_TOPKS,
    audit_repeated_assignments,
)


def _uniform_indices(total: int, requested: int) -> list[int]:
    if requested <= 0 or requested >= total:
        return list(range(total))
    return (
        torch.linspace(0, total - 1, steps=int(requested))
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )


def _load_mmap(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _sharpness(dataset: Path, images: str, names: list[str]) -> dict[str, float]:
    output = {}
    for name in names:
        path = dataset / images / name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        output[name] = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return output


def _parse_topks(value: str) -> tuple[int, ...]:
    topks = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    if not topks or topks[0] < 1:
        raise argparse.ArgumentTypeError("top-K list must contain positive integers")
    return topks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument(
        "--ransac-reprojection-px",
        type=float,
        help="Mapping-only fixed fallback for legacy runs without a calibration artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topks",
        type=_parse_topks,
        default=DEFAULT_TOPKS,
        help="Comma-separated descriptor ranks (default: 1,2,4,8,16,32).",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=0,
        help="Uniform mapping-only retrieval sample; zero evaluates all mapping queries.",
    )
    parser.add_argument(
        "--oracle-query-count",
        type=int,
        default=96,
        help="Uniform subset for current/oracle PoseLib replay; zero uses all audited queries.",
    )
    parser.add_argument(
        "--skip-oracle-pnp",
        action="store_true",
        help="Run descriptor retrieval only, without current/oracle PoseLib solves.",
    )
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if int(args.query_count) < 0 or int(args.oracle_query_count) < 0:
        raise ValueError("query counts must be non-negative")
    state = _load_mmap(args.map)
    teacher = _load_mmap(args.complete_positive_teacher)
    cache = _load_mmap(args.query_cache)
    calibration = None
    if args.scene_calibration is not None:
        calibration = json.loads(args.scene_calibration.read_text())
        calibration_is_mapping_only = (
            calibration.get("uses_test_queries") is False
            or (
                calibration.get("schema") == "lafgs_mapping_only_scene_calibration"
                and calibration.get("sources", {}).get("uses_test_queries") is False
            )
        )
        if not calibration_is_mapping_only:
            raise ValueError("scene calibration must explicitly be mapping-only")
    if calibration is None and args.ransac_reprojection_px is None:
        raise ValueError(
            "provide a mapping-only scene calibration or --ransac-reprojection-px"
        )
    ransac_reprojection_px = (
        float(args.ransac_reprojection_px)
        if args.ransac_reprojection_px is not None
        else float(calibration["parameters"]["ransac_reprojection_px"])
    )
    if ransac_reprojection_px <= 0:
        raise ValueError("RANSAC reprojection threshold must be positive")
    names = list(teacher["query_names"])
    selected = _uniform_indices(len(names), int(args.query_count))
    oracle_positions = (
        []
        if args.skip_oracle_pnp
        else _uniform_indices(len(selected), int(args.oracle_query_count))
    )
    oracle = [selected[position] for position in oracle_positions]
    selected_names = [names[index] for index in selected]
    sharpness = (
        _sharpness(args.dataset.resolve(), args.images, selected_names)
        if args.dataset is not None
        else None
    )
    device = torch.device(args.device)
    metric = load_shared_metric(
        args.metric_state,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    report = audit_repeated_assignments(
        state=state,
        metric=metric,
        teacher=teacher,
        query_cache=cache,
        device=device,
        topks=args.topks,
        query_indices=selected,
        oracle_query_indices=oracle,
        deployment_row_limit=int(args.deployment_row_limit),
        ransac_reprojection_px=ransac_reprojection_px,
        seed=int(args.seed),
        sharpness_by_name=sharpness,
    )
    report.update(
        {
            "map": str(args.map.resolve()),
            "metric_state": str(args.metric_state.resolve()),
            "complete_positive_teacher": str(
                args.complete_positive_teacher.resolve()
            ),
            "query_cache": str(args.query_cache.resolve()),
            "scene_calibration": (
                str(args.scene_calibration.resolve())
                if args.scene_calibration is not None
                else None
            ),
            "ransac_reprojection_px": ransac_reprojection_px,
            "query_selection": (
                "all_mapping" if len(selected) == len(names) else "uniform_mapping"
            ),
            "oracle_query_selection": (
                "all_audited" if len(oracle) == len(selected) else "uniform_audited"
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_count": report["query_count"],
                "oracle_query_count": report["oracle_query_count"],
                "positive_recall_at_k": report["positive_recall_at_k"],
                "oracle_pose_summaries": report["oracle_pose_summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
