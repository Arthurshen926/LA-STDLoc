#!/usr/bin/env python3
"""Audit accelerated 2DGS provenance against the unscreened full prior.

This audit is intentionally independent from descriptor retrieval.  For every
sampled render keypoint it compares the Gaussian identities and normalized
composition returned by a radius-screened evaluation with an evaluation over
the complete frozen Gaussian prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from common.hashing import sha256_file
from priors.models import GaussianModel2D
from priors.rasterizer import bank_splat_provenance_2dgs
from priors.rendering import render_from_pose_gsplat


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    *,
    keypoints: torch.Tensor,
    primitive_ids: torch.Tensor,
    rgb_meta: dict,
    rendered_depth: torch.Tensor,
    topk: int,
    candidate_topk: int,
    chunk_size: int,
    prefilter_topk: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, float]:
    _synchronize(keypoints.device)
    started = time.perf_counter()
    indices, weights, valid, diagnostics = bank_splat_provenance_2dgs(
        keypoints,
        primitive_ids,
        rgb_meta,
        rendered_depth=rendered_depth,
        topk=topk,
        candidate_topk=candidate_topk,
        chunk_size=chunk_size,
        prefilter_topk=prefilter_topk,
        return_diagnostics=True,
    )
    _synchronize(keypoints.device)
    return (
        indices.cpu(),
        weights.cpu(),
        valid.cpu(),
        {key: value.cpu() for key, value in diagnostics.items()},
        time.perf_counter() - started,
    )


def _retained_summary(diagnostics: dict) -> dict:
    retained = torch.as_tensor(diagnostics["retained_composition_fraction"]).float()
    return {
        "retained_composition_fraction_mean": float(retained.mean()),
        "retained_composition_fraction_p05": float(torch.quantile(retained, 0.05)),
        "retained_composition_fraction_min": float(retained.min()),
        "retained_composition_fraction_ge_0p95": float((retained >= 0.95).float().mean()),
    }


def _compare(
    exact_indices: torch.Tensor,
    exact_weights: torch.Tensor,
    exact_valid: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_weights: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> dict:
    row_count = int(exact_indices.shape[0])
    positive_exact = exact_weights > 1e-7
    positive_candidate = candidate_weights > 1e-7
    exact_top1 = exact_indices[:, 0]
    candidate_top1 = candidate_indices[:, 0]
    valid_union = exact_valid | candidate_valid
    valid_rows = int(valid_union.sum())
    top1_agreement = (~valid_union) | (
        exact_valid & candidate_valid & (exact_top1 == candidate_top1)
    )
    exact_support_recall = []
    support_exact = []
    l1_errors = []
    for row in range(row_count):
        exact = {
            int(identity): float(weight)
            for identity, weight in zip(
                exact_indices[row][positive_exact[row]].tolist(),
                exact_weights[row][positive_exact[row]].tolist(),
            )
        }
        candidate = {
            int(identity): float(weight)
            for identity, weight in zip(
                candidate_indices[row][positive_candidate[row]].tolist(),
                candidate_weights[row][positive_candidate[row]].tolist(),
            )
        }
        if exact:
            exact_support_recall.append(len(exact.keys() & candidate.keys()) / len(exact))
        else:
            exact_support_recall.append(1.0 if not candidate else 0.0)
        support_exact.append(exact.keys() == candidate.keys())
        union = exact.keys() | candidate.keys()
        l1_errors.append(sum(abs(exact.get(key, 0.0) - candidate.get(key, 0.0)) for key in union))
    l1 = torch.tensor(l1_errors, dtype=torch.float64)
    return {
        "row_count": row_count,
        "valid_row_count": valid_rows,
        "valid_agreement_fraction": float((exact_valid == candidate_valid).float().mean()),
        "top1_agreement_fraction": float(torch.as_tensor(top1_agreement).float().mean()),
        "positive_support_exact_fraction": float(torch.tensor(support_exact).float().mean()),
        "exact_positive_support_recall_mean": float(torch.tensor(exact_support_recall).mean()),
        "composition_l1_mean": float(l1.mean()),
        "composition_l1_p95": float(torch.quantile(l1, 0.95)),
        "composition_l1_max": float(l1.max()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-indices", type=int, nargs="+", default=[7])
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--reference-candidate-topk", type=int, default=0)
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--prefilter-topk", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--exact-chunk-size", type=int, default=16)
    parser.add_argument("--screened-chunk-size", type=int, default=64)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if int(args.sample_count) < 1:
        parser.error("sample count must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal V18 provenance audit requires CUDA")

    design = json.loads(args.design_batch.read_text())
    by_index = {int(item["query_index"]): item for item in design["records"]}
    gaussians = GaussianModel2D(args.sh_degree, device=device)
    gaussians.load_ply(args.gaussian_ply.resolve(), loc_feature_dim=0)
    gaussians = gaussians.to(device).eval()
    primitive_ids = torch.arange(gaussians.get_xyz.shape[0], device=device)
    query_reports = []
    for query_index in args.query_indices:
        item = by_index.get(int(query_index))
        if item is None:
            raise KeyError(f"query {query_index} is absent from the design split")
        observed = torch.load(item["path"], map_location="cpu", weights_only=False)
        if observed["certificate_decision"] != "ACCEPT":
            raise ValueError(f"query {query_index} is not V2 ACCEPT")
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError(f"query {query_index} source SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        source_rows = torch.as_tensor(observed["source_query_rows"]).long()
        available = int(source_rows.numel())
        sample_count = min(int(args.sample_count), available)
        # Fixed evenly spaced rows cover the full detector-score ordering without
        # making the audit depend on a random seed or descriptor similarity.
        sample_positions = torch.linspace(0, available - 1, sample_count).round().long()
        selected_rows = source_rows[sample_positions]
        keypoints = torch.as_tensor(source["keypoints"]).float()[selected_rows]
        intrinsic = torch.as_tensor(source["intrinsics"]).float()
        pose = torch.as_tensor(source["pose_w2c"]).float()
        height, width = map(int, torch.as_tensor(source["image_hw"]).tolist())
        fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsic[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsic[1, 1])))
        package = render_from_pose_gsplat(
            gaussians,
            pose.to(device),
            fov_x,
            fov_y,
            width,
            height,
            bg_color=torch.zeros(3, device=device),
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        exact_indices, exact_weights, exact_valid, exact_diagnostics, exact_seconds = _run(
            keypoints=keypoints.to(device),
            primitive_ids=primitive_ids,
            rgb_meta=package["rgb_meta"],
            rendered_depth=package["depth"],
            topk=int(args.topk),
            candidate_topk=int(args.reference_candidate_topk),
            chunk_size=int(args.exact_chunk_size),
            prefilter_topk=None,
        )
        (
            truncated_indices,
            truncated_weights,
            truncated_valid,
            truncated_diagnostics,
            truncated_seconds,
        ) = _run(
            keypoints=keypoints.to(device),
            primitive_ids=primitive_ids,
            rgb_meta=package["rgb_meta"],
            rendered_depth=package["depth"],
            topk=int(args.topk),
            candidate_topk=int(args.candidate_topk),
            chunk_size=int(args.screened_chunk_size),
            prefilter_topk=None,
        )
        candidate_truncation = {
            "candidate_topk": int(args.candidate_topk),
            "seconds": truncated_seconds,
            "speedup_over_exact": exact_seconds / max(truncated_seconds, 1e-12),
            **_retained_summary(truncated_diagnostics),
            **_compare(
                exact_indices,
                exact_weights,
                exact_valid,
                truncated_indices,
                truncated_weights,
                truncated_valid,
            ),
        }
        screened_reports = []
        for prefilter_topk in args.prefilter_topk:
            indices, weights, valid, diagnostics, seconds = _run(
                keypoints=keypoints.to(device),
                primitive_ids=primitive_ids,
                rgb_meta=package["rgb_meta"],
                rendered_depth=package["depth"],
                topk=int(args.topk),
                candidate_topk=int(args.candidate_topk),
                chunk_size=int(args.screened_chunk_size),
                prefilter_topk=int(prefilter_topk),
            )
            screened_reports.append(
                {
                    "prefilter_topk": int(prefilter_topk),
                    "seconds": seconds,
                    "speedup_over_exact": exact_seconds / max(seconds, 1e-12),
                    **_retained_summary(diagnostics),
                    **_compare(
                        exact_indices,
                        exact_weights,
                        exact_valid,
                        indices,
                        weights,
                        valid,
                    ),
                }
            )
        query_reports.append(
            {
                "query_index": int(query_index),
                "pose_family_id": int(observed["pose_family_id"]),
                "sample_count": sample_count,
                "sample_policy": "evenly_spaced_over_v2_accept_source_rows",
                "exact_seconds": exact_seconds,
                "exact_valid_count": int(exact_valid.sum()),
                "exact_retained_composition": _retained_summary(exact_diagnostics),
                "candidate_truncation": candidate_truncation,
                "screened": screened_reports,
            }
        )
        del package
        print(json.dumps(query_reports[-1], sort_keys=True), flush=True)
    artifact = {
        "schema": "lafgs_v18_provenance_prefilter_exact_audit",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "descriptor_independent": True,
        "primitive_count": int(primitive_ids.numel()),
        "config": vars(args) | {"design_batch": str(args.design_batch.resolve()), "gaussian_ply": str(args.gaussian_ply.resolve()), "output": str(args.output.resolve())},
        "inputs": {
            "design_batch_sha256": sha256_file(args.design_batch),
            "gaussian_ply_sha256": sha256_file(args.gaussian_ply),
        },
        "queries": query_reports,
    }
    _atomic_json(artifact, args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": sha256_file(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
