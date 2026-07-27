#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from localization_training.micro_anchors import (
    select_function_preserving_base_rows,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-map", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--visibility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-budgets", default="50048,48000")
    args = parser.parse_args()

    map_path = Path(args.expanded_map).resolve()
    payload_path = Path(args.track_payload).resolve()
    visibility_path = Path(args.visibility_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(
        {
            int(value)
            for value in args.target_budgets.split(",")
            if value.strip()
        },
        reverse=True,
    )

    state = torch.load(map_path, map_location="cpu", weights_only=False)
    payload = torch.load(
        payload_path, map_location="cpu", weights_only=False
    )
    visibility_payload = torch.load(
        visibility_path, map_location="cpu", weights_only=False
    )
    visibility = visibility_payload.get("visibility", visibility_payload)
    base_count = int(state["base_anchor_count"])
    canonical_count = int(state["canonical_anchor_count"])
    total_count = int(state["anchor_ids"].numel())
    if base_count != int(
        payload["assignment"]["landmark_best_track_index"].numel()
    ):
        raise ValueError("Track payload does not align with the original base")
    visibility_counts = torch.zeros(base_count, dtype=torch.long)
    for value in visibility.values():
        mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
        if mask.numel() != base_count:
            raise ValueError("visibility cache must align with original base")
        visibility_counts += mask.long()

    summary = {
        "expanded_map": str(map_path),
        "expanded_map_sha256": _sha256(map_path),
        "track_payload": str(payload_path),
        "track_payload_sha256": _sha256(payload_path),
        "visibility_cache": str(visibility_path),
        "visibility_cache_sha256": _sha256(visibility_path),
        "source_total_count": total_count,
        "base_count": base_count,
        "canonical_extension_count": canonical_count - base_count,
        "m4_extension_count": total_count - canonical_count,
        "targets": {},
    }
    row_fields = (
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_features",
        "anchor_type",
    )
    extension_rows = torch.arange(base_count, total_count)
    for target in targets:
        if target < total_count - base_count:
            raise ValueError("target budget cannot preserve all extensions")
        remove_count = total_count - target
        kept_base, removed, diagnostics = (
            select_function_preserving_base_rows(
                base_source_primitive_ids=state["source_primitive_ids"][
                    :base_count
                ],
                extension_source_primitive_ids=state[
                    "source_primitive_ids"
                ][base_count:],
                landmark_best_track_indices=payload["assignment"][
                    "landmark_best_track_index"
                ],
                visibility_counts=visibility_counts,
                remove_count=remove_count,
            )
        )
        rows = torch.cat((kept_base, extension_rows))
        output = dict(state)
        for key in row_fields:
            output[key] = torch.as_tensor(state[key])[rows].clone()
        output["anchor_ids"] = torch.arange(rows.numel(), dtype=torch.long)
        output["base_anchor_count"] = int(kept_base.numel())
        output["canonical_anchor_count"] = int(
            kept_base.numel() + canonical_count - base_count
        )
        output["micro_anchor_count"] = int(
            rows.numel() - kept_base.numel()
        )
        output["fixed_budget_compression"] = {
            "method": "track_first_zero_support_parent_visibility_v1",
            "target_anchor_count": int(target),
            "original_base_rows_preserved": True,
            "all_canonical_extension_rows_preserved": True,
            "all_m4_extension_rows_preserved": True,
            "removed_original_base_rows": removed,
            **diagnostics,
        }
        output["provenance"] = {
            **state.get("provenance", {}),
            "fixed_budget_source_map": str(map_path),
            "fixed_budget_source_map_sha256": _sha256(map_path),
            "fixed_budget_statistics_split": "all_895_mapping_train",
        }
        path = output_dir / f"fixed_budget_{target:05d}.pt"
        torch.save(output, path)
        summary["targets"][str(target)] = {
            "state": str(path),
            **diagnostics,
        }
    (output_dir / "fixed_budget_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
