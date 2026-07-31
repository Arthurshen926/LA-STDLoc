#!/usr/bin/env python
"""Select deterministic A2 tail failures and matched control queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_seed_results(scene_root, label, seeds):
    scene_root = Path(scene_root)
    rows_by_seed = {}
    for seed in seeds:
        pointer = scene_root / "evaluation" / label / f"seed{seed}" / "result.path"
        if not pointer.is_file():
            raise FileNotFoundError(pointer)
        result_dir = Path(pointer.read_text().strip())
        rows = json.loads((result_dir / "results.json").read_text())
        rows_by_seed[int(seed)] = {
            str(row["image_name"]).replace("\\", "/"): {
                "te_cm": float(row["sparse_TE"]),
                "ae_deg": float(row["sparse_AE"]),
            }
            for row in rows
        }
    names = sorted(next(iter(rows_by_seed.values())))
    expected = set(names)
    for seed, rows in rows_by_seed.items():
        if set(rows) != expected:
            raise ValueError(f"query registry differs for seed {seed}")
    return names, rows_by_seed


def select_queries(
    names,
    rows_by_seed,
    *,
    failure_count,
    control_count,
):
    seeds = sorted(rows_by_seed)
    mean_te = np.asarray(
        [
            np.mean([rows_by_seed[seed][name]["te_cm"] for seed in seeds])
            for name in names
        ],
        dtype=np.float64,
    )
    mean_ae = np.asarray(
        [
            np.mean([rows_by_seed[seed][name]["ae_deg"] for seed in seeds])
            for name in names
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(names), dtype=np.int64)
    descending = np.lexsort((np.asarray(names), -mean_te))
    failure_count = min(max(int(failure_count), 1), len(names))
    failure_idx = descending[:failure_count]
    failure_set = set(failure_idx.tolist())

    remaining = np.asarray(
        [index for index in positions if index not in failure_set],
        dtype=np.int64,
    )
    control_count = min(max(int(control_count), 0), remaining.size)
    if control_count:
        remaining = remaining[
            np.lexsort((np.asarray(names)[remaining], mean_te[remaining]))
        ]
        targets = np.linspace(0, remaining.size - 1, control_count)
        selected_positions = np.rint(targets).astype(np.int64)
        control_idx = remaining[selected_positions]
        control_idx = np.asarray(list(dict.fromkeys(control_idx.tolist())))
        if control_idx.size < control_count:
            selected = set(control_idx.tolist())
            fill = [idx for idx in remaining if int(idx) not in selected]
            control_idx = np.concatenate(
                [control_idx, np.asarray(fill[: control_count - control_idx.size])]
            )
    else:
        control_idx = np.empty(0, dtype=np.int64)

    records = []
    for role, selected in (("failure", failure_idx), ("control", control_idx)):
        for index in selected:
            records.append(
                {
                    "image_name": names[int(index)],
                    "role": role,
                    "mean_te_cm": float(mean_te[index]),
                    "mean_ae_deg": float(mean_ae[index]),
                    "per_seed": {
                        str(seed): rows_by_seed[seed][names[int(index)]]
                        for seed in seeds
                    },
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--label", default="A2_family_all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--failure-count", type=int, default=16)
    parser.add_argument("--control-count", type=int, default=12)
    parser.add_argument("--output-list", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    names, rows_by_seed = load_seed_results(
        args.scene_root, args.label, args.seeds
    )
    records = select_queries(
        names,
        rows_by_seed,
        failure_count=args.failure_count,
        control_count=args.control_count,
    )
    output_list = Path(args.output_list)
    output_report = Path(args.output_report)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    image_names = [record["image_name"] for record in records]
    output_list.write_text(json.dumps(image_names, indent=2) + "\n")
    payload = {
        "schema": "lafgs_group_consensus_query_selection_v1",
        "scene": Path(args.scene_root).name,
        "source_label": args.label,
        "seeds": args.seeds,
        "query_count": len(records),
        "failure_count": sum(row["role"] == "failure" for row in records),
        "control_count": sum(row["role"] == "control" for row in records),
        "queries": records,
    }
    output_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
