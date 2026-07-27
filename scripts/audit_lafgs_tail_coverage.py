#!/usr/bin/env python3
"""Audit whether localization tail errors are explained by mapping-view coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _camera_geometry(pose_w2c) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    center = -(rotation.T @ pose[:3, 3])
    forward = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    forward /= max(np.linalg.norm(forward), 1e-12)
    return center, forward


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    rx, ry = _rankdata(x[keep]), _rankdata(y[keep])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _summary(records: list[dict], label: str) -> dict:
    if not records:
        return {"label": label, "count": 0}
    te = np.asarray([item["te_cm"] for item in records])
    return {
        "label": label,
        "count": len(records),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(te.mean()),
        "p90_te_cm": float(np.quantile(te, 0.9)),
        "r5": float(np.mean(te <= 5.0)),
        "nearest_position_m_median": float(
            np.median([item["nearest_position_m"] for item in records])
        ),
        "nearest_view_angle_deg_median": float(
            np.median([item["nearest_view_angle_deg"] for item in records])
        ),
        "matchable_4px_mean": float(
            np.mean([item["matchable_rate_4px"] for item in records])
        ),
        "positive_top4_4px_mean": float(
            np.mean([item["positive_top4_rate_4px"] for item in records])
        ),
        "positive_top16_4px_mean": float(
            np.mean([item["positive_top16_rate_4px"] for item in records])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-cache", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--position-bin-m", type=float, default=1.0)
    parser.add_argument("--view-bin-deg", type=float, default=15.0)
    args = parser.parse_args()

    payload = torch.load(
        args.mapping_cache, map_location="cpu", weights_only=False
    )
    mapping = payload.get("queries", payload)
    mapping_geometry = [
        _camera_geometry(item["pose_w2c"]) for item in mapping.values()
    ]
    mapping_centers = np.stack([item[0] for item in mapping_geometry])
    mapping_forwards = np.stack([item[1] for item in mapping_geometry])

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    records = []
    for result in results:
        sparse = result["sparse"]
        center, forward = _camera_geometry(result["gt_pose_w2c"])
        distances = np.linalg.norm(mapping_centers - center[None], axis=1)
        angles = np.degrees(
            np.arccos(
                np.clip(mapping_forwards @ forward, -1.0, 1.0)
            )
        )
        joint = distances / max(args.position_bin_m, 1e-6) + (
            angles / max(args.view_bin_deg, 1e-6)
        )
        nearest = int(np.argmin(joint))
        local_counts = {}
        for distance_limit, angle_limit in ((0.5, 15), (1.0, 20), (2.0, 30)):
            local_counts[
                f"support_views_{distance_limit:g}m_{angle_limit:d}deg"
            ] = int(
                np.sum(
                    (distances <= distance_limit)
                    & (angles <= angle_limit)
                )
            )
        record = {
            "image_name": result["image_name"],
            "sequence": result["image_name"].split("/", 1)[0],
            "te_cm": float(result["sparse_TE"]),
            "ae_deg": float(result["sparse_AE"]),
            "nearest_position_m": float(distances[nearest]),
            "nearest_view_angle_deg": float(angles[nearest]),
            "pose_view_bin_occupied": bool(
                np.any(
                    (distances <= args.position_bin_m)
                    & (angles <= args.view_bin_deg)
                )
            ),
            "matchable_rate_2px": float(
                sparse.get("sparse_diag_matchable_rate_2px", np.nan)
            ),
            "matchable_rate_4px": float(
                sparse.get("sparse_diag_matchable_rate_4px", np.nan)
            ),
            "matchable_rate_8px": float(
                sparse.get("sparse_diag_matchable_rate_8px", np.nan)
            ),
            "inlier_ratio": float(
                sparse.get("sparse_diag_ransac_inlier_ratio_solver", np.nan)
            ),
            "ransac_hypotheses": float(
                sparse.get("sparse_diag_ransac_actual_hypotheses", np.nan)
            ),
            **local_counts,
        }
        for topk in (1, 4, 16):
            conditional = float(
                sparse.get(
                    "sparse_diag_conditional_recall_at_"
                    f"{topk}_given_matchable_4px",
                    np.nan,
                )
            )
            record[f"conditional_top{topk}_recall_4px"] = conditional
            record[f"positive_top{topk}_rate_4px"] = (
                record["matchable_rate_4px"] * conditional
            )
        records.append(record)

    fields = list(records[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tail_coverage_per_query.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    correlation_fields = (
        "nearest_position_m",
        "nearest_view_angle_deg",
        "support_views_0.5m_15deg",
        "support_views_1m_20deg",
        "support_views_2m_30deg",
        "matchable_rate_2px",
        "matchable_rate_4px",
        "matchable_rate_8px",
        "positive_top1_rate_4px",
        "positive_top4_rate_4px",
        "positive_top16_rate_4px",
        "inlier_ratio",
        "ransac_hypotheses",
    )
    correlations = {
        key: _spearman(
            [item[key] for item in records],
            [item["te_cm"] for item in records],
        )
        for key in correlation_fields
    }
    sequences = sorted({item["sequence"] for item in records})
    sequence_correlations = {
        sequence: {
            key: _spearman(
                [
                    item[key]
                    for item in records
                    if item["sequence"] == sequence
                ],
                [
                    item["te_cm"]
                    for item in records
                    if item["sequence"] == sequence
                ],
            )
            for key in correlation_fields
        }
        for sequence in sequences
    }
    directional_signals = {
        "nearest_position_m": 1.0,
        "nearest_view_angle_deg": 1.0,
        "support_views_1m_20deg": -1.0,
        "matchable_rate_4px": -1.0,
        "positive_top16_rate_4px": -1.0,
    }
    sequence_summaries = {
        sequence: _summary(
            [item for item in records if item["sequence"] == sequence],
            sequence,
        )
        for sequence in sequences
    }
    easiest = min(
        sequence_summaries.values(), key=lambda item: item["mean_te_cm"]
    )
    hardest = max(
        sequence_summaries.values(), key=lambda item: item["mean_te_cm"]
    )
    topk_sequence_contradiction = (
        hardest["positive_top16_4px_mean"]
        >= easiest["positive_top16_4px_mean"]
    )
    directional_coverage_signal = any(
        correlations[key] is not None
        and directional_signals[key] * correlations[key] >= 0.30
        for key in directional_signals
    )
    report = {
        "schema": "lafgs_tail_coverage_audit",
        "version": 1,
        "mapping_query_count": len(mapping),
        "development_query_count": len(records),
        "protocol": {
            "position_bin_m": args.position_bin_m,
            "view_bin_deg": args.view_bin_deg,
            "topk_rates": (
                "matchable@4px multiplied by conditional recall@K"
            ),
        },
        "overall": _summary(records, "all"),
        "sequences": sequence_summaries,
        "tail": _summary(
            sorted(records, key=lambda item: item["te_cm"], reverse=True)[
                : max(1, int(np.ceil(0.2 * len(records))))
            ],
            "worst_20_percent",
        ),
        "spearman_with_te": correlations,
        "spearman_with_te_by_sequence": sequence_correlations,
        "online_rendering_gate": {
            "coverage_signal_threshold_abs_rho": 0.30,
            "coverage_signal": directional_coverage_signal,
            "topk_sequence_contradiction": topk_sequence_contradiction,
            "stats_only_authorized": directional_coverage_signal,
            "training_authorized": (
                directional_coverage_signal
                and not topk_sequence_contradiction
            ),
            "next_stage": (
                "R1_stats_only"
                if directional_coverage_signal
                and topk_sequence_contradiction
                else (
                    "R2_hard_negative"
                    if directional_coverage_signal
                    else "stop"
                )
            ),
            "requires_proxy_tail_render_ablation": True,
            "interpretation": (
                "A directional signal licenses a low-ratio held-out proxy-tail "
                "rendering ablation; it does not establish rendering as geometry "
                "supervision."
            ),
        },
        "limitations": [
            "Existing evaluation artifacts do not contain per-query primitive visibility sets, so visible-primitive Jaccard is not inferred.",
            "Feature dispersion and base/new micro unique support require a correspondence dump and are not approximated from aggregate diagnostics.",
        ],
    }
    (output_dir / "tail_coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
