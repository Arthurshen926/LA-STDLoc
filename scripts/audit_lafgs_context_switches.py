#!/usr/bin/env python3
"""Audit assignment switches, candidate recall, and context collisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _positive_sets(record: dict) -> dict[int, set[int]]:
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    return {
        int(row): set(
            int(value)
            for value in indices[offsets[index] : offsets[index + 1]]
        )
        for index, row in enumerate(rows.tolist())
    }


def _project_residuals(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera[:, :2] / camera[:, 2:3].clamp_min(1e-8)
    projected = projected @ K[:2, :2].T + K[:2, 2]
    residual = projected - keypoints
    residual[camera[:, 2] <= 1e-6] = torch.inf
    return residual


def _numeric_summary(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def _load_eval_results(paths: list[str]) -> dict[str, dict]:
    output = {}
    for path in paths:
        if not path:
            continue
        payload = json.loads(Path(path).read_text())
        for record in payload["results"]:
            name = str(record["query"])
            if name in output:
                raise ValueError(f"duplicate evaluation query {name}")
            output[name] = record
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--context-topk", required=True)
    parser.add_argument("--baseline-dynamic", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-eval", action="append", default=[])
    parser.add_argument("--context-eval", action="append", default=[])
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    context = torch.load(
        args.context_topk, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.baseline_dynamic, map_location="cpu", weights_only=False
    )
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    if baseline.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported baseline top-K state")
    if context.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported context top-K state")
    if baseline["query_names"] != context["query_names"]:
        raise ValueError("top-K query registries differ")
    if baseline["anchor_ids_sha256"] != context["anchor_ids_sha256"]:
        raise ValueError("top-K maps differ")

    teacher_by_name = {
        str(record["query_name"]): record
        for record in teacher["records"]
    }
    baseline_by_name = _records_by_name(baseline)
    context_by_name = _records_by_name(context)
    dynamic_by_name = _records_by_name(dynamic)
    targeted = {
        (str(event["query_name"]), int(event["query_row"]))
        for event in graph["events"]
    }
    switches = Counter()
    subsets = {
        "all": Counter(),
        "targeted": Counter(),
        "non_targeted": Counter(),
        "baseline_ransac": Counter(),
        "baseline_harmful_inlier": Counter(),
    }
    errors_before = []
    errors_after = []
    errors_by_switch = {}
    signed_by_switch = {}
    positive_topk = {
        "baseline": Counter(),
        "context": Counter(),
    }
    positive_row_count = 0
    per_query_switch_count = Counter()

    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = context_by_name[name]
        if not torch.equal(
            torch.as_tensor(before["query_rows"]).long(),
            torch.as_tensor(after["query_rows"]).long(),
        ):
            raise ValueError(f"top-K rows differ for {name}")
        rows = torch.as_tensor(before["query_rows"]).long()
        before_topk = torch.as_tensor(
            before["topk_anchor_indices"]
        ).long()
        after_topk = torch.as_tensor(
            after["topk_anchor_indices"]
        ).long()
        positives = _positive_sets(teacher_by_name[name])
        dynamic_record = dynamic_by_name[name]
        dynamic_rows = torch.as_tensor(
            dynamic_record["query_rows"]
        ).long()
        if not torch.equal(rows, dynamic_rows):
            raise ValueError(f"dynamic rows differ for {name}")
        ransac = torch.as_tensor(
            dynamic_record["ransac_inlier_mask"]
        ).bool()
        harmful = torch.as_tensor(
            dynamic_record["harmful_inlier_mask"]
        ).bool()
        cached = cache[name]
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        K = torch.as_tensor(cached["native_K"]).float()
        pose = torch.as_tensor(cached["pose_w2c"]).float()
        before_xyz = torch.as_tensor(state["anchor_xyz"]).float()[
            before_topk[:, 0]
        ]
        after_xyz = torch.as_tensor(state["anchor_xyz"]).float()[
            after_topk[:, 0]
        ]
        before_residual = _project_residuals(
            before_xyz, keypoints, K, pose
        )
        after_residual = _project_residuals(
            after_xyz, keypoints, K, pose
        )
        for local_index, row in enumerate(rows.tolist()):
            row_positives = positives.get(int(row), set())
            if not row_positives:
                continue
            positive_row_count += 1
            before_anchor = int(before_topk[local_index, 0])
            after_anchor = int(after_topk[local_index, 0])
            before_positive = before_anchor in row_positives
            after_positive = after_anchor in row_positives
            if before_positive and after_positive:
                category = (
                    "positive_to_same_positive"
                    if before_anchor == after_anchor
                    else "positive_to_alternative_positive"
                )
            elif before_positive:
                category = "positive_to_wrong"
            elif after_positive:
                category = "wrong_to_positive"
            else:
                category = (
                    "wrong_to_same_wrong"
                    if before_anchor == after_anchor
                    else "wrong_to_wrong"
                )
            switches[category] += 1
            subsets["all"][category] += 1
            subset = (
                "targeted"
                if (name, int(row)) in targeted
                else "non_targeted"
            )
            subsets[subset][category] += 1
            if bool(ransac[local_index]):
                subsets["baseline_ransac"][category] += 1
            if bool(harmful[local_index]):
                subsets["baseline_harmful_inlier"][category] += 1
            if before_anchor != after_anchor:
                per_query_switch_count[name] += 1
            before_error = float(
                torch.linalg.vector_norm(before_residual[local_index])
            )
            after_error = float(
                torch.linalg.vector_norm(after_residual[local_index])
            )
            errors_before.append(before_error)
            errors_after.append(after_error)
            errors_by_switch.setdefault(category, []).append(
                after_error - before_error
            )
            signed_by_switch.setdefault(category, []).append(
                (
                    float(
                        after_residual[local_index, 0]
                        - before_residual[local_index, 0]
                    ),
                    float(
                        after_residual[local_index, 1]
                        - before_residual[local_index, 1]
                    ),
                )
            )
            for width in (1, 4, 8, 16):
                for label, topk in (
                    ("baseline", before_topk),
                    ("context", after_topk),
                ):
                    if row_positives.intersection(
                        int(value)
                        for value in topk[
                            local_index, : min(width, topk.shape[1])
                        ].tolist()
                    ):
                        positive_topk[label][width] += 1

    local = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    )
    map_context = F.normalize(
        torch.as_tensor(context_state["anchor_context"]).float(), dim=1
    )
    edge_correct = torch.as_tensor(
        [int(edge["correct_anchor"]) for edge in graph["edges"]]
    ).long()
    edge_confusing = torch.as_tensor(
        [int(edge["confusing_anchor"]) for edge in graph["edges"]]
    ).long()
    local_cosine = (
        local[edge_correct] * local[edge_confusing]
    ).sum(dim=1)
    context_cosine = (
        map_context[edge_correct] * map_context[edge_confusing]
    ).sum(dim=1)

    baseline_eval = _load_eval_results(args.baseline_eval)
    context_eval = _load_eval_results(args.context_eval)
    pose_audit = None
    if baseline_eval and set(baseline_eval) == set(context_eval):
        pose_delta = {
            name: float(context_eval[name]["te_cm"])
            - float(baseline_eval[name]["te_cm"])
            for name in baseline_eval
        }
        switched = [
            pose_delta[name]
            for name, count in per_query_switch_count.items()
            if count > 0 and name in pose_delta
        ]
        unchanged = [
            delta
            for name, delta in pose_delta.items()
            if per_query_switch_count[name] == 0
        ]
        pose_audit = {
            "all_te_delta_cm": _numeric_summary(pose_delta.values()),
            "switched_query_te_delta_cm": _numeric_summary(switched),
            "unchanged_query_te_delta_cm": _numeric_summary(unchanged),
        }

    output = {
        "schema": "lafgs_context_switch_audit",
        "version": 1,
        "positive_row_count": int(positive_row_count),
        "switch_counts": dict(switches),
        "switch_fractions_percent": {
            key: 100.0 * value / max(positive_row_count, 1)
            for key, value in switches.items()
        },
        "subsets": {
            key: dict(value) for key, value in subsets.items()
        },
        "positive_recall_percent": {
            label: {
                f"top{width}": (
                    100.0
                    * counter[width]
                    / max(positive_row_count, 1)
                )
                for width in (1, 4, 8, 16)
            }
            for label, counter in positive_topk.items()
        },
        "gt_reprojection_before_px": _numeric_summary(errors_before),
        "gt_reprojection_after_px": _numeric_summary(errors_after),
        "gt_reprojection_delta_by_switch_px": {
            key: _numeric_summary(value)
            for key, value in errors_by_switch.items()
        },
        "signed_residual_delta_by_switch_px": {
            key: {
                "dx": _numeric_summary(value[0] for value in values),
                "dy": _numeric_summary(value[1] for value in values),
            }
            for key, values in signed_by_switch.items()
        },
        "map_context_collision": {
            "edge_count": len(graph["edges"]),
            "local_cosine": _numeric_summary(local_cosine.tolist()),
            "context_cosine": _numeric_summary(context_cosine.tolist()),
            "context_minus_local": _numeric_summary(
                (context_cosine - local_cosine).tolist()
            ),
            "context_more_similar_fraction_percent": float(
                100.0 * (context_cosine > local_cosine).float().mean()
            ),
        },
        "pose_audit": pose_audit,
        "provenance": vars(args),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
