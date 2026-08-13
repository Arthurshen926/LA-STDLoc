#!/usr/bin/env python3
"""Train-only PnP oracle over complete geometric teacher positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm


def _sequence_name(image_name: str) -> str:
    return str(image_name).split("/", maxsplit=1)[0]


def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if cache_payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("oracle cache is not rendered-RGB-only")
    if cache_payload.get("uses_test_queries") is not False:
        raise ValueError("oracle cache contains test queries")
    cache = cache_payload["queries"]
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    rows = []
    for query_index, record in enumerate(teacher["records"]):
        name = teacher["query_names"][query_index]
        cached = cache[name]
        query_rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        indices = torch.as_tensor(record["positive_indices"]).long()
        counts = offsets[1:] - offsets[:-1]
        eligible = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
        # Deterministic first legal anchor is sufficient to test geometry
        # coverage; no descriptor score or test-tuned choice is involved.
        selected = indices[offsets[eligible]]
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[
            query_rows[eligible]
        ] + float(cached.get("pixel_center_offset", 0.5))
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[selected].numpy(),
            torch.as_tensor(cached["native_K"]).float().numpy(),
            reprojection_error_px=args.ransac_reprojection_px,
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=args.seed,
        )
        ae_deg, _ = pose_error(
            estimate.pose_w2c,
            torch.as_tensor(cached["pose_w2c"]).numpy(),
        )
        te_cm = _pose_error_cm(estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"]))
        rows.append(
            {
                "query_index": query_index,
                "image_name": name,
                "sequence": _sequence_name(name),
                "positive_correspondence_count": int(eligible.numel()),
                "te_cm": float(te_cm),
                "ae_deg": float(ae_deg),
                "inliers": int(len(estimate.inliers)),
            }
        )
        if (query_index + 1) % 100 == 0 or query_index + 1 == len(teacher["records"]):
            print(
                json.dumps(
                    {
                        "completed_queries": query_index + 1,
                        "query_count": len(teacher["records"]),
                    }
                ),
                flush=True,
            )

    def summary(selected_rows):
        te = np.asarray([row["te_cm"] for row in selected_rows])
        ae = np.asarray([row["ae_deg"] for row in selected_rows])
        return {
            "query_count": len(selected_rows),
            "median_te_cm": float(np.median(te)),
            "mean_te_cm": float(np.mean(te)),
            "p90_te_cm": float(np.percentile(te, 90)),
            "median_ae_deg": float(np.median(ae)),
            "mean_ae_deg": float(np.mean(ae)),
            "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
            "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
            "positive_correspondence_median": float(
                np.median(
                    [row["positive_correspondence_count"] for row in selected_rows]
                )
            ),
        }

    sequences = sorted({row["sequence"] for row in rows})
    output = {
        "schema": "lafgs_rendered_track_train_only_geometry_oracle",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "selection_uses_descriptors": False,
        "combined_summary": summary(rows),
        "sequence_summaries": {
            sequence: summary([row for row in rows if row["sequence"] == sequence])
            for sequence in sequences
        },
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for field in ("anchor_map", "teacher", "query_cache", "output"):
        setattr(args, field, getattr(args, field).resolve())
    output = run(args)
    print(
        json.dumps(
            {
                "combined_summary": output["combined_summary"],
                "sequence_summaries": output["sequence_summaries"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
