#!/usr/bin/env python3
"""Build mapping-only signed residual evidence for an SLPS candidate graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.slps_residual_signatures import (
    RESIDUAL_SIGNATURE_FEATURE_NAMES,
    add_residual_statistics,
    empty_residual_statistics,
    residual_statistics_contribution,
    signed_reprojection_residual,
)
from scripts.train_lafgs_pose_sufficient_selector import (
    _records_by_name,
    _sha256_tensor,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--clip-px", type=float, default=12.0)
    parser.add_argument("--strict-px", type=float, default=2.0)
    parser.add_argument("--anchor-prior", type=float, default=4.0)
    parser.add_argument("--cell-prior", type=float, default=8.0)
    parser.add_argument("--rate-prior", type=float, default=12.0)
    args = parser.parse_args()

    map_path = Path(args.map).resolve()
    cache_path = Path(args.query_cache).resolve()
    topk_path = Path(args.topk_outcomes).resolve()
    output = Path(args.output).resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(topk_path, map_location="cpu", weights_only=False)
    if topk.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported top-K outcomes")
    if topk["provenance"].get("family_prototype_state_sha256") is not None:
        raise ValueError("residual signatures require a single-descriptor graph")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if (
        int(topk["anchor_count"]) != len(anchor_ids)
        or topk["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
    ):
        raise ValueError("residual signature graph and map do not align")
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    records = _records_by_name(topk)
    if set(records) - set(cache):
        raise ValueError("residual signature queries are absent from the cache")

    statistics = empty_residual_statistics(
        len(anchor_ids), grid_size=int(args.grid_size)
    )
    row_count = 0
    valid_count = 0
    solver_count = 0
    strict_count = 0
    residual_norm_sum = 0.0
    for query_index, name in enumerate(topk["query_names"]):
        record = records[name]
        cached = cache[name]
        rows = torch.as_tensor(record["query_rows"]).long()
        anchors = torch.as_tensor(
            record["topk_anchor_indices"]
        ).long()[:, 0]
        selector_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        observed = selector_keypoints + float(
            cached.get("pixel_center_offset", 0.5)
        )
        residual, valid = signed_reprojection_residual(
            xyz[anchors],
            observed,
            cached["native_K"],
            cached["pose_w2c"],
        )
        contribution = residual_statistics_contribution(
            anchor_indices=anchors,
            keypoints=selector_keypoints,
            image_hw=cached["native_input_hw"],
            signed_residual=residual,
            valid=valid,
            anchor_count=len(anchor_ids),
            grid_size=int(args.grid_size),
            clip_px=float(args.clip_px),
            strict_px=float(args.strict_px),
        )
        add_residual_statistics(statistics, contribution)
        norm = torch.linalg.norm(residual, dim=1)
        row_count += len(anchors)
        valid_count += int(valid.sum())
        solver_count += int((valid & (norm <= float(args.clip_px))).sum())
        strict_count += int((valid & (norm <= float(args.strict_px))).sum())
        residual_norm_sum += float(norm[valid].sum())
        if (query_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "completed_queries": query_index + 1,
                        "total_queries": len(topk["query_names"]),
                    }
                ),
                flush=True,
            )

    summary = {
        "query_count": len(topk["query_names"]),
        "row_count": row_count,
        "valid_rate": valid_count / max(row_count, 1),
        "strict_rate": strict_count / max(valid_count, 1),
        "solver_rate": solver_count / max(valid_count, 1),
        "mean_residual_px": residual_norm_sum / max(valid_count, 1),
        "observed_anchor_count": int(
            (statistics["attempts"].sum(dim=1) > 0).sum()
        ),
    }
    payload = {
        "schema": "lafgs_slps_residual_signatures",
        "version": 1,
        "feature_names": list(RESIDUAL_SIGNATURE_FEATURE_NAMES),
        "anchor_count": len(anchor_ids),
        "anchor_ids_sha256": _sha256_tensor(anchor_ids),
        "query_names": list(topk["query_names"]),
        "statistics": statistics,
        "config": {
            "grid_size": int(args.grid_size),
            "clip_px": float(args.clip_px),
            "strict_px": float(args.strict_px),
            "anchor_prior": float(args.anchor_prior),
            "cell_prior": float(args.cell_prior),
            "rate_prior": float(args.rate_prior),
        },
        "candidate_graph_contract": dict(topk["provenance"]),
        "source": {
            "map": str(map_path),
            "map_sha256": _sha256_file(map_path),
            "query_cache": str(cache_path),
            "query_cache_sha256": _sha256_file(cache_path),
            "topk_outcomes": str(topk_path),
            "topk_outcomes_sha256": _sha256_file(topk_path),
        },
        "summary": summary,
    }
    _atomic_torch(output, payload)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
