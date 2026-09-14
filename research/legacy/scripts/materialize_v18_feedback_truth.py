#!/usr/bin/env python3
"""Build descriptor-independent provenance truth for feedback render queries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_UNIQUE,
    TruthAssignmentThresholds,
    assign_provenance_truth,
    backproject_query_surface,
    provenance_candidate_graph,
    query_anchor_geometry_evidence,
    transport_candidate_graph,
    truth_membership_mask,
)
from priors.models import GaussianModel2D
from priors.rasterizer import bank_splat_provenance_2dgs
from priors.rendering import render_from_pose_gsplat


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_raster(raster: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(raster).squeeze()
    pixels = torch.floor(keypoints).long()
    x = pixels[:, 0].clamp(0, value.shape[1] - 1)
    y = pixels[:, 1].clamp(0, value.shape[0] - 1)
    return value[y, x]


def _relative_depth_spread(
    depth_raster: torch.Tensor, keypoints: torch.Tensor
) -> torch.Tensor:
    depth = torch.as_tensor(depth_raster).squeeze()
    pixels = torch.floor(keypoints).long()
    samples = []
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            x = (pixels[:, 0] + offset_x).clamp(0, depth.shape[1] - 1)
            y = (pixels[:, 1] + offset_y).clamp(0, depth.shape[0] - 1)
            samples.append(depth[y, x])
    local = torch.stack(samples, dim=1)
    valid = torch.isfinite(local) & (local > 0)
    minimum = local.masked_fill(~valid, float("inf")).amin(1)
    maximum = local.masked_fill(~valid, -float("inf")).amax(1)
    center = _sample_raster(depth, keypoints).abs().clamp_min(1e-6)
    spread = (maximum - minimum) / center
    spread[~valid.any(1)] = float("inf")
    return spread


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--truth-validation", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--topk-primitives", type=int, default=64)
    parser.add_argument(
        "--minimum-composition-mass", type=float, default=0.95
    )
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=0,
        help="0 performs full depth-ordered Gaussian compositing",
    )
    parser.add_argument(
        "--prefilter-topk",
        type=int,
        default=0,
        help="0 evaluates the complete Gaussian prior (required for formal truth)",
    )
    parser.add_argument(
        "--maximum-anchor-candidates",
        type=int,
        default=0,
        help="0 enumerates every provenance-linked Anchor",
    )
    parser.add_argument("--progress-interval", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V18 feedback truth materialization requires CUDA")
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    mapping = torch.load(args.mapping_provenance, map_location="cpu", weights_only=False)
    validation = torch.load(args.truth_validation, map_location="cpu", weights_only=False)
    design = json.loads(args.design_batch.read_text())
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
    ):
        raise ValueError("V18 feedback truth requires the frozen no-test design split")
    if not (
        mapping.get("schema") == "lafgs_v18_mapping_observation_gaussian_provenance"
        and mapping.get("full_gaussian_prior_evaluated") is True
        and mapping.get("full_depth_ordered_compositing") is True
        and float(mapping.get("minimum_retained_composition_mass", 0.0)) >= 0.95
        and validation.get("schema") == "lafgs_v18_provenance_truth_validation"
        and validation.get("uses_test_queries") is False
        and validation.get("loo_used") is False
        and validation.get("full_provenance_anchor_enumeration") is True
    ):
        raise ValueError("V18 mapping truth artifacts differ")
    if int(args.candidate_topk) > 0 or int(args.prefilter_topk) > 0:
        raise ValueError("formal V18 feedback truth requires complete compositing")
    if float(args.minimum_composition_mass) < 0.95:
        raise ValueError("formal V18 feedback truth requires at least 95% mass")
    if int(args.maximum_anchor_candidates) != 0:
        raise ValueError(
            "formal V18 feedback truth requires complete Anchor enumeration"
        )
    observations = state["projective_anchor_observations"]
    anchor_offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    design_observation = torch.as_tensor(
        validation["signature_design_observation_mask"]
    ).bool()
    if design_observation.numel() != observation_queries.numel():
        raise ValueError("V18 signature split differs from the Anchor observation CSR")
    mapping_offsets = torch.as_tensor(mapping["mapping_pixel_center_offset"]).float()
    mapping_keypoints = [
        torch.as_tensor(value).float() + float(mapping_offsets[index])
        for index, value in enumerate(mapping["mapping_keypoints"])
    ]
    mapping_intrinsics = torch.as_tensor(mapping["mapping_intrinsics"]).float()
    mapping_poses = torch.as_tensor(mapping["mapping_poses_w2c"]).float()
    mapping_families = torch.as_tensor(mapping["mapping_view_family_ids"]).long()
    inverse = validation["primitive_anchor_index"]
    thresholds = TruthAssignmentThresholds(**validation["selected_thresholds"])
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchor_count))
    ).long()

    gaussians = GaussianModel2D(args.sh_degree, device=device)
    gaussians.load_ply(args.gaussian_ply.resolve(), loc_feature_dim=0)
    gaussians = gaussians.to(device).eval()
    primitive_count = int(gaussians.get_xyz.shape[0])
    primitive_universe = torch.arange(primitive_count, device=device)
    records = []
    diagnostics = {
        "truth_row_count": 0,
        "unique_or_equivalent_count": 0,
        "descriptor_retrieval_miss_count": 0,
        "top1_competition_failure_count": 0,
        "correct_top1_count": 0,
    }
    accepted_items = []
    for item in design["records"]:
        observed = torch.load(item["path"], map_location="cpu", weights_only=False)
        if observed["certificate_decision"] == "ACCEPT":
            accepted_items.append((item, observed))
    for completed, (item, observed) in enumerate(accepted_items, start=1):
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("V18 feedback source record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        source_rows = torch.as_tensor(observed["source_query_rows"]).long()
        keypoints_grid = torch.as_tensor(source["keypoints"]).float()[source_rows]
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
        local_ids, weights, provenance_valid, provenance_diagnostics = (
            bank_splat_provenance_2dgs(
                keypoints_grid.to(device),
                primitive_universe,
                package["rgb_meta"],
                rendered_depth=package.get("depth"),
                topk=int(args.topk_primitives),
                candidate_topk=int(args.candidate_topk),
                prefilter_topk=(
                    int(args.prefilter_topk)
                    if int(args.prefilter_topk) > 0
                    else None
                ),
                return_diagnostics=True,
                minimum_composition_mass=float(args.minimum_composition_mass),
            )
        )
        graph = provenance_candidate_graph(
            query_primitive_ids=local_ids.cpu(),
            query_weights=weights.cpu(),
            primitive_anchor_index=inverse,
            query_valid=provenance_valid.cpu(),
            query_composition_entropy=(
                -(weights * weights.clamp_min(1e-12).log()).sum(1).cpu()
            ),
            query_relative_depth_spread=_relative_depth_spread(
                package["depth"], keypoints_grid.to(device)
            ).cpu(),
            query_retained_composition_fraction=provenance_diagnostics[
                "retained_composition_fraction"
            ].cpu(),
            maximum_candidates_per_row=int(args.maximum_anchor_candidates),
        )
        depth = _sample_raster(package["depth"], keypoints_grid.to(device)).cpu()
        surface, surface_valid = backproject_query_surface(
            keypoints_grid + 0.5,
            depth,
            intrinsic,
            pose,
        )
        graph["query_valid"] &= surface_valid
        geometry = query_anchor_geometry_evidence(
            candidate_graph=graph,
            query_keypoints=keypoints_grid + 0.5,
            query_depth=depth,
            query_indices=torch.zeros(keypoints_grid.shape[0], dtype=torch.long),
            anchor_xyz=state["anchor_xyz"],
            anchor_covariance=state["anchor_position_covariance"],
            query_intrinsics=intrinsic[None],
            query_poses_w2c=pose[None],
            device=device,
        )
        transport = transport_candidate_graph(
            candidate_graph=graph,
            query_surface_xyz=surface,
            anchor_observation_offsets=anchor_offsets,
            observation_query_indices=observation_queries,
            observation_keypoint_indices=observation_keypoints,
            observation_enabled=design_observation,
            mapping_keypoints=mapping_keypoints,
            mapping_intrinsics=mapping_intrinsics,
            mapping_poses_w2c=mapping_poses,
            mapping_view_family_ids=mapping_families,
            inlier_residual_px=float(validation["selected_thresholds"]["maximum_transport_median_residual_px"]),
            minimum_candidate_overlap_to_evaluate=float(
                validation["selected_thresholds"]["minimum_provenance_overlap"]
            ),
        )
        truth = assign_provenance_truth(
            candidate_graph=graph,
            transport_evidence=transport,
            geometry_evidence=geometry,
            equivalence_class_ids=equivalence,
            thresholds=thresholds,
        )
        topk = torch.as_tensor(observed["topk_anchor_rows"]).long()
        membership = truth_membership_mask(truth, topk)
        decisive = (torch.as_tensor(truth["truth_status"]) == TRUTH_UNIQUE) | (
            torch.as_tensor(truth["truth_status"]) == TRUTH_EQUIVALENT
        )
        retrieved = membership.any(1) & decisive
        top1_correct = membership[:, 0] & decisive
        retrieval_miss = decisive & ~retrieved
        competition_failure = retrieved & ~top1_correct
        diagnostics["truth_row_count"] += int(truth["row_count"])
        diagnostics["unique_or_equivalent_count"] += int(decisive.sum())
        diagnostics["descriptor_retrieval_miss_count"] += int(retrieval_miss.sum())
        diagnostics["top1_competition_failure_count"] += int(competition_failure.sum())
        diagnostics["correct_top1_count"] += int(top1_correct.sum())
        records.append(
            {
                "query_index": int(observed["query_index"]),
                "pose_family_id": int(observed["pose_family_id"]),
                "source_record": str(source_path),
                "source_record_sha256": observed["source_record_sha256"],
                "source_query_rows": source_rows,
                "truth": truth,
                "top64_truth_membership": membership,
                "descriptor_retrieval_miss": retrieval_miss,
                "top1_competition_failure": competition_failure,
                "correct_top1": top1_correct,
            }
        )
        del package, local_ids, weights, provenance_valid, graph, transport
        if completed % max(int(args.progress_interval), 1) == 0 or completed == len(accepted_items):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "accepted_queries": len(accepted_items),
                        **diagnostics,
                    }
                ),
                flush=True,
            )
    decisive_count = diagnostics["unique_or_equivalent_count"]
    artifact = {
        "schema": "lafgs_v18_feedback_provenance_truth_batch",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "descriptor_independent_truth": True,
        "full_gaussian_prior_evaluated": int(args.prefilter_topk) <= 0,
        "full_depth_ordered_compositing": int(args.candidate_topk) <= 0,
        "minimum_retained_composition_mass": float(args.minimum_composition_mass),
        "full_provenance_anchor_enumeration": int(args.maximum_anchor_candidates)
        == 0,
        "top64_role": "competition_graph_only",
        "controller_replacement_authorized_by_mapping_validation": bool(
            validation["controller_replacement_authorized"]
        ),
        "accepted_query_count": len(records),
        "diagnostics": {
            **diagnostics,
            "unique_or_equivalent_coverage": float(
                decisive_count / max(diagnostics["truth_row_count"], 1)
            ),
            "descriptor_retrieval_miss_fraction_of_truth": float(
                diagnostics["descriptor_retrieval_miss_count"] / max(decisive_count, 1)
            ),
            "top1_competition_failure_fraction_of_truth": float(
                diagnostics["top1_competition_failure_count"] / max(decisive_count, 1)
            ),
            "correct_top1_fraction_of_truth": float(
                diagnostics["correct_top1_count"] / max(decisive_count, 1)
            ),
        },
        "selected_thresholds": validation["selected_thresholds"],
        "records": records,
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "design_batch": str(args.design_batch.resolve()),
            "design_batch_sha256": sha256_file(args.design_batch),
            "mapping_provenance": str(args.mapping_provenance.resolve()),
            "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
            "truth_validation": str(args.truth_validation.resolve()),
            "truth_validation_sha256": sha256_file(args.truth_validation),
            "gaussian_ply": str(args.gaussian_ply.resolve()),
            "gaussian_ply_sha256": sha256_file(args.gaussian_ply),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    summary = {key: value for key, value in artifact.items() if key != "records"}
    summary["output"] = str(args.output.resolve())
    summary["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
