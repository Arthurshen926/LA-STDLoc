#!/usr/bin/env python3
"""Build detached pose-critical weights for a map-specific positive teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.pose_information import (
    compute_pose_information,
    conditional_add_gain,
    fisher_contributions,
    normalize_information_scores,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


def _project_errors(xyz, keypoints, K, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2].clamp_min(1e-8)
    uv = torch.stack(
        (
            K[0, 0] * camera[:, 0] / depth + K[0, 2],
            K[1, 1] * camera[:, 1] / depth + K[1, 2],
        ),
        dim=1,
    )
    return torch.linalg.norm(uv - keypoints, dim=1)


def _pair_rows(offsets):
    offsets = torch.as_tensor(offsets).long()
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(counts.numel()), counts)


def _identity(args) -> str:
    values = {
        key: str(Path(getattr(args, key.replace("-", "_"))).resolve())
        for key in ("map", "positive-teacher", "query-cache", "dynamic-outcomes")
    }
    values.update(
        {
            "protect_threshold_cm": args.protect_threshold_cm,
            "near_threshold_cm": args.near_threshold_cm,
            "tail_threshold_cm": args.tail_threshold_cm,
        }
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protect-threshold-cm", type=float, default=7.0)
    parser.add_argument("--near-threshold-cm", type=float, default=15.0)
    parser.add_argument("--tail-threshold-cm", type=float, default=50.0)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.positive_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    outcomes = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = list(teacher["query_names"])
    if list(outcomes["query_names"]) != names:
        raise ValueError("dynamic outcomes and positive teacher queries differ")
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    if int(outcomes["anchor_count"]) != xyz.shape[0]:
        raise ValueError("dynamic outcomes do not align with active map")
    dependency = torch.as_tensor(state["dependency_group_ids"]).long()
    family_size = torch.bincount(dependency).float().clamp_min(1)
    dependency_weight = family_size[dependency].rsqrt()
    dependency_weight /= torch.quantile(
        dependency_weight, 0.95
    ).clamp_min(1e-8)
    dependency_weight = dependency_weight.clamp(0.5, 1.5)

    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    run_identity = _identity(args)
    output_records = []
    component_sums = {
        "row_weight": 0.0,
        "reprojection": 0.0,
        "dependency": 0.0,
        "schur": 0.0,
        "lgo": 0.0,
    }
    pair_count = 0
    if partial.is_file():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        if saved["run_identity"] != run_identity:
            raise ValueError("pose-critical partial identity mismatch")
        output_records = saved["records"]
        component_sums = saved["component_sums"]
        pair_count = int(saved["pair_count"])
    for query_index, (record, dynamic) in enumerate(
        zip(teacher["records"], outcomes["records"])
    ):
        if query_index < len(output_records):
            continue
        name = names[query_index]
        if dynamic["query_name"] != name:
            raise ValueError("dynamic outcome query order mismatch")
        query_rows = torch.as_tensor(record["query_rows"]).long()
        if not torch.equal(
            query_rows, torch.as_tensor(dynamic["query_rows"]).long()
        ):
            raise ValueError("dynamic outcome rows do not align with teacher")
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        positives = torch.as_tensor(record["positive_indices"]).long()
        pair_rows = _pair_rows(offsets)
        cached = cache[name]
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[
            query_rows
        ]
        K = torch.as_tensor(cached["native_K"]).float()
        pose = torch.as_tensor(cached["pose_w2c"]).float()
        pair_xyz = xyz[positives]
        pair_keypoints = keypoints[pair_rows]
        reprojection_error = _project_errors(pair_xyz, pair_keypoints, K, pose)
        reprojection = torch.exp(
            -0.5 * (reprojection_error / 4.0).square()
        ).clamp(0.25, 1.0)

        clean = torch.as_tensor(dynamic["clean_inlier_mask"]).bool()
        top1 = torch.as_tensor(dynamic["top1_anchor_indices"]).long()
        clean_xyz = xyz[top1[clean]]
        eye = torch.eye(6, dtype=torch.float64) * 1e-6
        if clean_xyz.shape[0] >= 4:
            clean_info = compute_pose_information(
                clean_xyz.double(),
                K.double(),
                pose.double(),
                damping=1e-6,
                translation_scale=0.07160573943725686,
                rotation_scale=torch.deg2rad(torch.tensor(2.0)).item(),
            )
            base_information = clean_info.matrix
            lgo_by_anchor = {}
            lgo_values = normalize_information_scores(
                clean_info.translation_scores, floor=0.0, mode="quantile"
            )
            for anchor, value in zip(top1[clean].tolist(), lgo_values.tolist()):
                lgo_by_anchor[int(anchor)] = max(
                    lgo_by_anchor.get(int(anchor), 0.0), float(value)
                )
        else:
            base_information = eye
            lgo_by_anchor = {}
        jacobian = pose_jacobian_analytic(pair_xyz.double(), K.double(), pose.double())
        jacobian = task_scaled_pose_jacobian(
            jacobian,
            translation_scale=0.07160573943725686,
            rotation_scale=torch.deg2rad(torch.tensor(2.0)).item(),
        )
        contribution = fisher_contributions(jacobian)
        schur = conditional_add_gain(
            base_information[None].expand(contribution.shape[0], -1, -1),
            contribution,
            objective="translation",
        ).clamp_min(0)
        schur = normalize_information_scores(
            schur, floor=0.25, mode="quantile"
        )
        lgo = torch.as_tensor(
            [lgo_by_anchor.get(int(anchor), 0.0) for anchor in positives]
        )
        positive_weights = (
            reprojection
            * dependency_weight[positives]
            * (0.5 + 1.5 * schur.float())
            * (1.0 + lgo)
        )
        counts = offsets[1:] - offsets[:-1]
        row_sum = torch.zeros(query_rows.numel())
        row_sum.index_add_(0, pair_rows, positive_weights)
        row_mean = row_sum / counts.float().clamp_min(1)
        positive_weights = positive_weights / row_mean[pair_rows].clamp_min(1e-8)
        positive_weights = positive_weights.clamp(0.25, 4.0)

        te = float(dynamic["te_cm"])
        if te <= float(args.protect_threshold_cm):
            query_risk = 2.0
        elif te <= float(args.near_threshold_cm):
            query_risk = 1.75
        elif te > float(args.tail_threshold_cm):
            query_risk = 2.5
        else:
            query_risk = 1.0
        row_weights = torch.full((query_rows.numel(),), query_risk)
        harmful = torch.as_tensor(dynamic["harmful_inlier_mask"]).bool()
        row_weights[harmful] *= 1.5
        row_weights = row_weights.clamp(0.5, 4.0)
        output_records.append(
            {
                "query_name": name,
                "query_rows": query_rows,
                "row_weights": row_weights,
                "positive_weights": positive_weights,
            }
        )
        count = int(positives.numel())
        pair_count += count
        component_sums["row_weight"] += float(row_weights.sum())
        component_sums["reprojection"] += float(reprojection.sum())
        component_sums["dependency"] += float(
            dependency_weight[positives].sum()
        )
        component_sums["schur"] += float(schur.sum())
        component_sums["lgo"] += float(lgo.sum())
        if (query_index + 1) % 50 == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch(
                partial,
                {
                    "run_identity": run_identity,
                    "records": output_records,
                    "component_sums": component_sums,
                    "pair_count": pair_count,
                },
            )
            print(f"{query_index + 1}/{len(names)}", flush=True)

    payload = {
        "schema": "lafgs_pose_critical_positive_teacher",
        "version": 1,
        "query_names": names,
        "anchor_count": int(xyz.shape[0]),
        "records": output_records,
        "diagnostics": {
            "positive_pair_count": pair_count,
            **{
                f"{key}_mean": value / max(
                    pair_count if key != "row_weight" else sum(
                        len(record["row_weights"]) for record in output_records
                    ),
                    1,
                )
                for key, value in component_sums.items()
            },
        },
        "config": vars(args),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(output, payload)
    partial.unlink(missing_ok=True)
    (output.with_suffix(".json")).write_text(
        json.dumps(payload["diagnostics"], indent=2) + "\n"
    )
    print(json.dumps(payload["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
