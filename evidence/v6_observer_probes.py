"""Fixed-map virtual probe planning for V6 observer excitation."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from common.v6_contracts import FEEDBACK_SCHEMA, require_mapping_only, require_schema
from evidence.observation_provider import ObservationProvider
from evidence.virtual_render_planner import PlannerPolicy, camera_centers, generate_candidate_poses
from topology.layered_sufficiency import visibility_image_cells


SCHEMA = "lafgs_v6_fixed_map_observer_probe_plan"
VERSION = 2
SENSOR_VARIANTS = (
    "clean",
    "exposure_down",
    "exposure_up",
    "gamma_low",
    "gamma_high",
    "motion_blur_mild",
    "sensor_noise_mild",
    "resize_compression_mild",
    "local_occlusion_mild",
    "white_balance_warm",
    "white_balance_cool",
    "contrast_low",
    "motion_blur_vertical_mild",
    "defocus_blur_mild",
)


def _select_diverse_candidates(
    *,
    utility: list[float],
    kinds: list[str],
    pose_families: torch.Tensor,
    budget: int,
) -> list[int]:
    """Cover distinct excitation mechanisms before greedily filling utility."""

    if len(utility) != len(kinds) or len(utility) != int(pose_families.numel()):
        raise ValueError("observer candidate score registry is not aligned")
    ordered = sorted(
        range(len(utility)),
        key=lambda index: (-utility[index], kinds[index], index),
    )
    selected: list[int] = []
    used_family: set[int] = set()
    available_kinds = sorted(set(kinds))
    # The kind order itself follows each kind's best attainable utility.  This
    # avoids an alphabetical preference when the budget is smaller than the
    # number of excitation mechanisms.
    available_kinds.sort(
        key=lambda kind: (
            -max(utility[index] for index in ordered if kinds[index] == kind),
            kind,
        )
    )
    for kind in available_kinds:
        for candidate_index in ordered:
            family = int(pose_families[candidate_index])
            if kinds[candidate_index] == kind and family not in used_family:
                selected.append(candidate_index)
                used_family.add(family)
                break
        if len(selected) == int(budget):
            return selected
    for candidate_index in ordered:
        family = int(pose_families[candidate_index])
        if candidate_index in selected or family in used_family:
            continue
        selected.append(candidate_index)
        used_family.add(family)
        if len(selected) == int(budget):
            break
    return selected


def _project(
    xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    homogeneous = camera @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8), camera[:, 2]


def _ambiguity_anchor_rows(feedback: Mapping, anchor_count: int) -> torch.Tensor:
    rows = []
    for record in feedback["records"]:
        for field in (
            "confusion_pairs",
            "certified_pose_valid_alternative_pairs",
        ):
            pairs = torch.as_tensor(record.get(field, ())).long()
            if pairs.numel() == 0:
                continue
            if pairs.ndim != 2 or pairs.shape[1] not in {2, 3}:
                raise ValueError(f"observer probe {field} has an invalid shape")
            rows.append(pairs[:, 1:].reshape(-1))
        for field in ("harmful_inlier_anchor_ids", "ambiguous_inlier_anchor_ids"):
            rows.append(torch.as_tensor(record.get(field, ())).long().reshape(-1))
    if not rows:
        return torch.empty(0, dtype=torch.long)
    result = torch.unique(torch.cat(rows), sorted=True)
    if result.numel() and (int(result.min()) < 0 or int(result.max()) >= anchor_count):
        raise ValueError("observer probe ambiguity registry has an invalid Anchor")
    return result


def build_fixed_map_observer_probe_plan(
    state: Mapping,
    observations: ObservationProvider,
    feedback: Mapping,
    *,
    map_sha256: str,
    observation_cache_sha256: str,
    feedback_sha256: str,
    selected_pose_budget: int = 32,
    maximum_candidates: int = 512,
    anchor_projection_stride: int = 16,
    sensor_variants_per_pose: int = 4,
) -> dict:
    """Plan observer-only pose/sensor probes without mutating the map.

    Candidate visibility is a geometry proxy used only for experiment design.
    A selected pose becomes an evaluable probe only after the immutable
    Gaussian prior renders RGB plus alpha/depth and passes z-buffer checks.
    """

    require_mapping_only(state.get("provenance", {}), label="observer source map")
    require_schema(feedback, FEEDBACK_SCHEMA, label="observer source feedback")
    if list(feedback["query_names"]) != list(observations.names):
        raise ValueError("observer probe feedback and camera registries differ")
    if int(selected_pose_budget) < 1 or int(maximum_candidates) < 1:
        raise ValueError("observer probe budgets must be positive")
    if int(anchor_projection_stride) < 1:
        raise ValueError("observer probe Anchor stride must be positive")
    if not 1 <= int(sensor_variants_per_pose) <= len(SENSOR_VARIANTS):
        raise ValueError("sensor variants per pose is outside the registered range")
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_count = int(xyz.shape[0])
    ambiguity_rows = _ambiguity_anchor_rows(feedback, anchor_count)
    ambiguity_xyz = xyz[ambiguity_rows]
    poses = torch.stack(
        [observations.build_view(index).pose_w2c.float() for index in range(len(observations))]
    )
    policy = PlannerPolicy(
        maximum_candidates=int(maximum_candidates),
        selected_view_budget=int(selected_pose_budget),
    )
    candidates = generate_candidate_poses(poses, ambiguity_xyz, policy)
    candidate_count = int(candidates["pose_w2c"].shape[0])
    sampled_xyz = xyz[:: int(anchor_projection_stride)]
    source_centers = camera_centers(poses.double())
    diagnostics = []
    ambiguity_counts = []
    pose_cell_counts = []
    novelty_values = []
    for candidate_index in range(candidate_count):
        parent = int(candidates["parent_camera_index"][candidate_index])
        view = observations.build_view(parent)
        pose = candidates["pose_w2c"][candidate_index].float()
        uv, depth = _project(sampled_xyz, view.intrinsics.float(), pose)
        height, width = view.image_hw
        visible = (
            torch.isfinite(uv).all(1)
            & torch.isfinite(depth)
            & (depth > 0.0)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < height)
        )
        cells = visibility_image_cells(uv[visible], image_hw=view.image_hw)
        pose_cells = int(torch.unique(cells).numel()) if cells.numel() else 0
        if ambiguity_xyz.numel():
            ambiguity_uv, ambiguity_depth = _project(
                ambiguity_xyz, view.intrinsics.float(), pose
            )
            ambiguity_visible = (
                torch.isfinite(ambiguity_uv).all(1)
                & torch.isfinite(ambiguity_depth)
                & (ambiguity_depth > 0.0)
                & (ambiguity_uv[:, 0] >= 0.0)
                & (ambiguity_uv[:, 0] < width)
                & (ambiguity_uv[:, 1] >= 0.0)
                & (ambiguity_uv[:, 1] < height)
            )
            ambiguity_count = int(ambiguity_visible.sum())
        else:
            ambiguity_count = 0
        center = camera_centers(pose[None].double())[0]
        translation = float(torch.linalg.norm(center - source_centers[parent]))
        forward = pose[:3, :3].T[:, 2]
        parent_forward = poses[parent, :3, :3].T[:, 2]
        angle = math.degrees(
            math.acos(float(torch.dot(forward, parent_forward).clamp(-1.0, 1.0)))
        )
        novelty = min(translation / 0.25, 1.0) + min(angle / 20.0, 1.0)
        ambiguity_counts.append(ambiguity_count)
        pose_cell_counts.append(pose_cells)
        novelty_values.append(novelty)
        diagnostics.append(
            {
                "candidate_index": candidate_index,
                "parent_camera_index": parent,
                "kind": candidates["kind"][candidate_index],
                "pose_family": int(candidates["pose_family"][candidate_index]),
                "geometry_proxy_visible_anchor_count": int(visible.sum()),
                "geometry_proxy_pose_cell_count": pose_cells,
                "geometry_proxy_ambiguity_anchor_count": ambiguity_count,
                "viewpoint_novelty": novelty,
                "artifact_risk": float(candidates["artifact_risk"][candidate_index]),
            }
        )
    maximum_ambiguity = max(ambiguity_counts, default=1)
    maximum_cells = max(pose_cell_counts, default=1)
    utility = []
    for row in diagnostics:
        view_score = 0.25 * min(float(row["viewpoint_novelty"]) / 2.0, 1.0)
        ambiguity_score = 0.45 * float(row["geometry_proxy_ambiguity_anchor_count"]) / max(
            maximum_ambiguity, 1
        )
        pose_score = 0.30 * float(row["geometry_proxy_pose_cell_count"]) / max(
            maximum_cells, 1
        )
        risk_cost = 0.15 * float(row["artifact_risk"])
        value = view_score + ambiguity_score + pose_score - risk_cost
        row["observer_design_utility"] = value
        utility.append(value)
    selected = _select_diverse_candidates(
        utility=utility,
        kinds=list(candidates["kind"]),
        pose_families=candidates["pose_family"],
        budget=int(selected_pose_budget),
    )
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    selected_records = []
    for order, candidate_index in enumerate(selected):
        parent = int(candidates["parent_camera_index"][candidate_index])
        stressed = SENSOR_VARIANTS[1:]
        offset = (order * max(int(sensor_variants_per_pose) - 1, 1)) % len(stressed)
        variants = ["clean"]
        variants.extend(
            stressed[(offset + index) % len(stressed)]
            for index in range(int(sensor_variants_per_pose) - 1)
        )
        selected_records.append(
            {
                **diagnostics[candidate_index],
                "probe_index": order,
                "pose_w2c": candidates["pose_w2c"][candidate_index],
                "native_K": observations.build_view(parent).intrinsics.float(),
                "native_input_hw": list(observations.build_view(parent).image_hw),
                "sensor_variants": variants,
                "render_status": "planned_not_yet_zbuffer_certified",
            }
        )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {
            "map_sha256": str(map_sha256),
            "observation_cache_sha256": str(observation_cache_sha256),
            "feedback_sha256": str(feedback_sha256),
        },
        "observer_role": "fixed_map_experiment_design_only",
        "plant_map_is_fixed_for_every_probe": True,
        "virtual_probes_added_to_map": False,
        "virtual_probes_added_to_anchor_observations": False,
        "virtual_probes_increase_track_view_count": False,
        "candidate_pose_count": candidate_count,
        "selected_pose_count": len(selected),
        "selected_candidate_indices": selected_tensor,
        "ambiguity_anchor_count": int(ambiguity_rows.numel()),
        "anchor_projection_stride": int(anchor_projection_stride),
        "utility_definition": {
            "viewpoint_novelty_weight": 0.25,
            "ambiguity_co_visibility_weight": 0.45,
            "pose_cell_coverage_weight": 0.30,
            "artifact_risk_cost": 0.15,
        },
        "selection_policy": "excitation_kind_coverage_then_family_unique_utility",
        "sensor_variant_registry": list(SENSOR_VARIANTS),
        "sensor_variants_per_pose": int(sensor_variants_per_pose),
        "selected_probes": selected_records,
        "render_acceptance_contract": {
            "immutable_gaussian_prior_required": True,
            "rgb_required": True,
            "alpha_required": True,
            "expected_depth_required": True,
            "median_depth_requested": True,
            "contribution_entropy_requested": True,
            "depth_consistency_requested": True,
            "zbuffer_certification_required_before_observer_evaluation": True,
            "geometry_proxy_is_not_pose_valid_certificate": True,
        },
    }
