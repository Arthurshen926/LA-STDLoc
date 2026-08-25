"""Evaluate observer-only virtual probes against one fixed deployment map."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from evidence.observation_provider import GaussianRenderObservationProvider
from localization.matcher import global_owner_prototype_top1
from localization.pose_solver import solve_absolute_pose
from map_learning.v6_feedback_evaluator import (
    _aligned_keypoint_surface_depth,
    _depth_certified_pose_valid_edges,
    _layer_edges,
    _project,
)
from topology.layered_sufficiency import visibility_image_cells


SCHEMA = "lafgs_v6_fixed_map_virtual_probe_evaluation"
VERSION = 2


def _pose_errors(estimated: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    estimated = torch.as_tensor(estimated).float()
    truth = torch.as_tensor(truth).float()
    rotation = estimated[:3, :3] @ truth[:3, :3].T
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angular = float(torch.rad2deg(torch.acos(cosine)))
    estimated_center = -(estimated[:3, :3].T @ estimated[:3, 3])
    truth_center = -(truth[:3, :3].T @ truth[:3, 3])
    translation = float(torch.linalg.norm(estimated_center - truth_center) * 100.0)
    return translation, angular


def _summary(records: list[dict], prefix: str = "") -> dict:
    te = np.asarray([record[f"{prefix}te_cm"] for record in records], dtype=np.float64)
    ae = np.asarray([record[f"{prefix}ae_deg"] for record in records], dtype=np.float64)
    tail = max(int(math.ceil(0.05 * len(records))), 1)
    return {
        "query_count": len(records),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "recall_5cm_5deg_percent": float(np.mean((te < 5.0) & (ae < 5.0)) * 100.0),
        "catastrophic_100cm_count": int((te >= 100.0).sum()),
    }


@torch.inference_mode()
def evaluate_fixed_map_virtual_probes(
    state: Mapping,
    probe_cache: Mapping,
    *,
    map_sha256: str,
    probe_cache_sha256: str,
    source_map_sha256: str | None = None,
    validation_probe_indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    positive_radius_px: float = 2.0,
    alpha_minimum: float = 0.05,
    ransac_reprojection_px: float,
    seed: int = 2026,
    device: str = "cuda",
    solver: Callable = solve_absolute_pose,
) -> dict:
    if (
        probe_cache.get("schema")
        != "lafgs_v6_fixed_map_observer_probe_cache"
        or int(probe_cache.get("version", -1)) != 1
        or probe_cache.get("uses_source_mapping_rgb") is not False
        or probe_cache.get("uses_test_queries") is not False
    ):
        raise ValueError("virtual probe cache contract differs")
    inputs = probe_cache.get("inputs", {})
    source_map_sha256 = (
        str(map_sha256).lower()
        if source_map_sha256 is None
        else str(source_map_sha256).lower()
    )
    if inputs.get("source_map_sha256") != source_map_sha256:
        raise ValueError("virtual probe cache source map SHA differs")
    if probe_cache.get("virtual_probes_added_to_map") is not False or probe_cache.get(
        "virtual_probes_added_to_anchor_observations"
    ) is not False:
        raise ValueError("virtual probes are not observer-only")
    provider = GaussianRenderObservationProvider(
        dict(probe_cache), query_names=list(probe_cache["query_names"])
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    extra_features = torch.as_tensor(
        state.get(
            "anchor_extra_prototype_features",
            torch.empty((0, bank.shape[1])),
        )
    ).float()
    extra_owners = torch.as_tensor(
        state.get("anchor_extra_prototype_owner_rows", torch.empty(0))
    ).long().reshape(-1)
    if (extra_features.shape[0] != extra_owners.numel()) or (
        extra_features.ndim != 2 or extra_features.shape[1] != bank.shape[1]
    ):
        raise ValueError("virtual probe map prototype extension is invalid")
    if extra_features.numel():
        extra_features = F.normalize(extra_features, dim=1)
    probe_indices = sorted(
        {
            int(probe_cache["queries"][name]["probe_index"])
            for name in probe_cache["query_names"]
        }
    )
    if len(probe_indices) < 2:
        raise ValueError("virtual-probe control validation needs at least two poses")
    if validation_probe_indices is None:
        held_out_count = max(1, int(math.ceil(0.25 * len(probe_indices))))
        validation_probes = probe_indices[-held_out_count:]
    else:
        validation_probes = sorted(
            set(torch.as_tensor(validation_probe_indices).long().reshape(-1).tolist())
        )
    if not validation_probes or not set(validation_probes) < set(probe_indices):
        raise ValueError("virtual-probe validation is not a proper pose partition")
    training_probes = sorted(set(probe_indices) - set(validation_probes))
    active = torch.ones(xyz.shape[0], dtype=torch.bool)
    records = []
    for query_index in range(len(provider)):
        view = provider.build_view(query_index)
        started = time.perf_counter()
        projected, anchor_depth = _project(
            xyz, view.intrinsics.float(), view.pose_w2c.float()
        )
        height, width = view.image_hw
        visible = (
            active
            & torch.isfinite(projected).all(1)
            & torch.isfinite(anchor_depth)
            & (anchor_depth > 0.0)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        if view.alpha is not None:
            x = projected[:, 0].round().long().clamp(0, width - 1)
            y = projected[:, 1].round().long().clamp(0, height - 1)
            visible &= torch.isfinite(view.alpha[y, x]) & (
                view.alpha[y, x] >= float(alpha_minimum)
            )
        visible_rows = torch.nonzero(visible, as_tuple=False).reshape(-1)
        keypoints = view.physical_keypoints.float()
        geometry_edges = _layer_edges(
            keypoints, projected, visible_rows, float(positive_radius_px)
        )
        keypoint_depth, depth_source = _aligned_keypoint_surface_depth(
            view, alpha_minimum=float(alpha_minimum)
        )
        certified_edges, depth_available = _depth_certified_pose_valid_edges(
            geometry_edges,
            anchor_depth=anchor_depth,
            keypoint_depth=keypoint_depth,
        )
        query_descriptor = F.normalize(view.descriptors.float(), dim=1).to(device)
        matches = global_owner_prototype_top1(
            query_descriptor,
            bank.to(device),
            extra_features.to(device),
            extra_owners.to(device),
            anchor_descriptors_normalized=True,
        )
        winners = matches.anchor_indices.cpu()
        winner_scores = matches.scores.cpu()
        winner_valid = torch.tensor(
            [int(winner) in certified_edges[row] for row, winner in enumerate(winners)],
            dtype=torch.bool,
        )
        estimate = solver(
            keypoints.numpy(),
            xyz[winners].numpy(),
            view.intrinsics.float().numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        te_cm, ae_deg = _pose_errors(estimate.pose_w2c, view.pose_w2c)
        valid_rows = []
        oracle_anchors = []
        certified_pairs = []
        descriptor_triplets = []
        descriptor_triplet_pose_weights = []
        top1_negative = torch.zeros(winners.numel(), dtype=torch.bool)
        for row, candidates in enumerate(certified_edges):
            if not candidates:
                continue
            candidate_tensor = torch.tensor(candidates, dtype=torch.long)
            scores = query_descriptor[row].cpu() @ bank[candidate_tensor].T
            best = int(candidate_tensor[int(torch.argmax(scores))])
            valid_rows.append(row)
            oracle_anchors.append(best)
            certified_pairs.extend((row, int(anchor)) for anchor in candidates)
            geometry = set(geometry_edges[row])
            top1_negative[row] = int(winners[row]) not in geometry
            if bool(top1_negative[row]):
                descriptor_triplets.append((row, best, int(winners[row]), 0))
                positive_error = torch.linalg.norm(
                    projected[best] - keypoints[row]
                )
                winner_error = torch.linalg.norm(
                    projected[int(winners[row])] - keypoints[row]
                )
                descriptor_triplet_pose_weights.append(
                    float((winner_error - positive_error).clamp_min(0.0))
                )
        if len(valid_rows) >= 4:
            oracle_rows = torch.tensor(valid_rows, dtype=torch.long)
            oracle_anchor_tensor = torch.tensor(oracle_anchors, dtype=torch.long)
            oracle_estimate = solver(
                keypoints[oracle_rows].numpy(),
                xyz[oracle_anchor_tensor].numpy(),
                view.intrinsics.float().numpy(),
                reprojection_error_px=float(ransac_reprojection_px),
                confidence=0.99999,
                max_iterations=100000,
                min_iterations=1000,
                seed=int(seed),
            )
            oracle_te_cm, oracle_ae_deg = _pose_errors(
                oracle_estimate.pose_w2c, view.pose_w2c
            )
            oracle_available = True
        else:
            oracle_te_cm, oracle_ae_deg = te_cm, ae_deg
            oracle_available = False
        valid_winner_rows = torch.nonzero(winner_valid, as_tuple=False).reshape(-1)
        valid_winner_anchors = winners[valid_winner_rows]
        unique_valid = torch.unique(valid_winner_anchors)
        valid_cells = visibility_image_cells(
            keypoints[valid_winner_rows], image_hw=view.image_hw
        )
        records.append(
            {
                "query_index": query_index,
                "image_name": view.image_name,
                "probe_index": int(
                    probe_cache["queries"][view.image_name]["probe_index"]
                ),
                "sensor_variant": str(
                    probe_cache["queries"][view.image_name]["sensor_variant"]
                ),
                "control_split": (
                    "validation"
                    if int(probe_cache["queries"][view.image_name]["probe_index"])
                    in validation_probes
                    else "training"
                ),
                "te_cm": te_cm,
                "ae_deg": ae_deg,
                "pose_success": bool(te_cm < 5.0 and ae_deg < 5.0),
                "correspondence_count": int(winners.numel()),
                "pose_valid_winner_count": int(winner_valid.sum()),
                "pose_valid_winner_fraction": float(winner_valid.float().mean()),
                "pose_valid_unique_anchor_count": int(unique_valid.numel()),
                "pose_valid_image_cell_count": int(torch.unique(valid_cells).numel())
                if valid_cells.numel()
                else 0,
                "certified_pose_valid_row_count": len(valid_rows),
                "pose_valid_depth_available": bool(depth_available),
                "pose_valid_depth_source": depth_source,
                "inlier_count": len(estimate.inliers),
                "inlier_ratio": len(estimate.inliers) / max(int(winners.numel()), 1),
                "oracle_available": oracle_available,
                "oracle_te_cm": oracle_te_cm,
                "oracle_ae_deg": oracle_ae_deg,
                "oracle_pose_success": bool(
                    oracle_available and oracle_te_cm < 5.0 and oracle_ae_deg < 5.0
                ),
                "winner_anchor_ids": winners.tolist(),
                "winner_scores": winner_scores.tolist(),
                "top1_negative_mask": top1_negative.tolist(),
                "certified_pose_valid_alternative_pairs": certified_pairs,
                "descriptor_triplets": descriptor_triplets,
                "descriptor_triplet_pose_weights": descriptor_triplet_pose_weights,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    clean = [record for record in records if record["sensor_variant"] == "clean"]
    stressed = [record for record in records if record["sensor_variant"] != "clean"]
    oracle_records = [record for record in records if record["oracle_available"]]
    training_records = [
        record for record in records if record["control_split"] == "training"
    ]
    validation_records = [
        record for record in records if record["control_split"] == "validation"
    ]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {
            "map_sha256": str(map_sha256).lower(),
            "source_map_sha256": source_map_sha256,
            "probe_cache_sha256": str(probe_cache_sha256).lower(),
            "probe_plan_sha256": inputs["probe_plan_sha256"],
            "gaussian_ply_sha256": inputs["gaussian_ply_sha256"],
        },
        "fixed_map_plant": True,
        "virtual_probes_added_to_map": False,
        "online_protocol": "native_superpoint_global_top1_one_standard_poselib",
        "configuration": {
            "positive_radius_px": float(positive_radius_px),
            "alpha_minimum": float(alpha_minimum),
            "ransac_reprojection_px": float(ransac_reprojection_px),
            "seed": int(seed),
        },
        "control_split": {
            "policy": "pose_grouped_last_quarter_holdout"
            if validation_probe_indices is None
            else "explicit_pose_group_holdout",
            "training_probe_indices": training_probes,
            "validation_probe_indices": validation_probes,
            "training_query_indices": [
                int(record["query_index"]) for record in training_records
            ],
            "validation_query_indices": [
                int(record["query_index"]) for record in validation_records
            ],
            "sensor_variants_share_their_pose_partition": True,
            "validation_used_by_controller": False,
        },
        "summary": _summary(records),
        "control_training_summary": _summary(training_records),
        "independent_probe_validation_summary": _summary(validation_records),
        "clean_pose_probe_summary": _summary(clean) if clean else None,
        "sensor_stress_probe_summary": _summary(stressed) if stressed else None,
        "pose_valid_oracle_summary": _summary(oracle_records, "oracle_")
        if oracle_records
        else None,
        "records": records,
    }
