#!/usr/bin/env python
import argparse
import json
import os

import yaml


def make_stdloc_eval_cfg(
    base_cfg,
    output,
    artifact_model_path,
    detector_folder=None,
    detector_iters=None,
    landmark_path=None,
    landmark_meta_path=None,
    candidate_teacher_state_path=None,
    pair_scorer_state_path=None,
    landmark_feature_override_path=None,
    override_landmark_features=None,
    materialized_anchor_map_path=None,
    pair_measurement_state_path=None,
    detect_num=None,
    reprojection_error=None,
    nms=None,
    match_threshold=None,
    match_topk=None,
    unique_landmark_matches=None,
    max_matches_per_keypoint=None,
    max_matches_per_landmark=None,
    use_candidate_dustbin=None,
    use_candidate_pair_scorer=None,
    pair_scorer_threshold=None,
    use_candidate_pair_scorer_calibrated_threshold=None,
    use_pair_measurement=None,
    pair_measurement_threshold=None,
    use_pair_measurement_calibrated_threshold=None,
    use_pair_measurement_offset=None,
    use_pair_measurement_covariance_refinement=None,
    pair_measurement_refinement_iterations=None,
    pair_measurement_mahalanobis_threshold=None,
    pair_measurement_robust_delta=None,
    pair_measurement_covariance_model_floor_px=None,
    use_pair_measurement_progressive_sampling=None,
    pair_measurement_max_prosac_iterations=None,
    pair_measurement_fixed_candidate_count=None,
    pair_measurement_refill_mode=None,
    pair_measurement_refill_grid_rows=None,
    pair_measurement_refill_grid_cols=None,
    pair_measurement_refill_voxel_size=None,
    pair_measurement_refill_spatial_weight=None,
    pair_measurement_refill_voxel_weight=None,
    min_candidate_matches=None,
    candidate_refill_trigger_count=None,
    use_detector_matchability=None,
    detector_matchability_mode=None,
    use_detector_offset=None,
    detector_max_offset=None,
    use_native_matchability=None,
    native_matchability_state_path=None,
    native_matchability_max_prosac_iterations=None,
    candidate_frontend_match_policy=None,
    diagnostics=None,
    diagnostics_dump_correspondences=None,
    diagnostics_dump_inliers_only=None,
    diagnostics_dump_discrete_oracle=None,
    diagnostics_oracle_topk=None,
    diagnostics_grid_rows=None,
    diagnostics_grid_cols=None,
    diagnostics_voxel_size=None,
    diagnostics_task_translation_scale_m=None,
    diagnostics_task_rotation_scale_degrees=None,
    use_two_stage_pose_refinement=None,
    two_stage_tight_reprojection_error=None,
    two_stage_min_inliers=None,
    two_stage_refinement_iterations=None,
    two_stage_robust_delta=None,
    two_stage_damping=None,
    geometry_balance=None,
    geometry_balance_grid_rows=4,
    geometry_balance_grid_cols=4,
    geometry_balance_max_per_cell=64,
    geometry_balance_voxel_size=0.25,
    geometry_balance_max_per_voxel=64,
    geometry_balance_max_matches=0,
    full_primitive_retrieval=None,
    full_primitive_retrieval_topk=None,
    full_primitive_chunk_size=None,
    full_primitive_surface_suppression=None,
    full_primitive_voxel_size=None,
    full_primitive_max_per_surface=None,
    sparse_query_feature_contract=None,
    sparse_frontend=None,
    metric_state_path=None,
    family_prototype_state_path=None,
    pose_sufficient_selector_state_path=None,
    pose_sufficient_budget=None,
    rerank_topk=None,
    rerank_patch_radius=None,
    rerank_patch_step_px=None,
    rerank_global_weight=None,
    rerank_local_peak_weight=None,
    rerank_local_margin_weight=None,
    rerank_local_entropy_weight=None,
    rerank_offset_weight=None,
    rerank_local_temperature=None,
    rerank_null_score_threshold=None,
    rerank_null_margin_threshold=None,
    rerank_state_path=None,
    rerank_use_learned_null=None,
    rerank_assignment_global_preserve_scale=None,
    sparse_ransac_seed=None,
    sparse_solver=None,
    group_saturated_cap=None,
    group_saturated_surface_voxel_scale_ratio=None,
    group_saturated_surface_minimum_voxel_size=None,
    group_saturated_surface_normal_angle_degrees=None,
    preemptive_verification_order=None,
    preemptive_check_interval=None,
):
    with open(base_cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["model_path"] = artifact_model_path
    sparse = cfg.setdefault("sparse", {})
    explicit_detector_artifact = (
        detector_folder is not None or detector_iters is not None
    )
    if explicit_detector_artifact:
        if detector_folder is None or detector_iters is None:
            raise ValueError(
                "detector_folder and detector_iters must be provided together"
            )
        detector_folder = str(detector_folder).strip("/")
        detector_iters = int(detector_iters)
        sparse["detector_path"] = (
            f"{detector_folder}/{detector_iters}_detector.pth"
        )
        sparse["landmark_path"] = (
            str(landmark_path)
            if landmark_path
            else f"{detector_folder}/sampled_idx.pkl"
        )
        sparse["landmark_meta_path"] = (
            str(landmark_meta_path)
            if landmark_meta_path
            else f"{detector_folder}/landmark_meta.pt"
        )
    else:
        # Preserve the base configuration's artifact binding.  Falling back to
        # the historical detector is only valid when the base config has no
        # binding at all; silently replacing an explicit native frontend or
        # external landmark bank makes validation non-reproducible.
        if not sparse.get("detector_path"):
            detector_folder = "detector"
            detector_iters = 30000
            sparse["detector_path"] = (
                f"{detector_folder}/{detector_iters}_detector.pth"
            )
        else:
            detector_folder = os.path.dirname(
                str(sparse["detector_path"])
            ).strip("/") or "."
        if landmark_path:
            sparse["landmark_path"] = str(landmark_path)
        else:
            sparse.setdefault(
                "landmark_path", f"{detector_folder}/sampled_idx.pkl"
            )
        if landmark_meta_path:
            sparse["landmark_meta_path"] = str(landmark_meta_path)
        else:
            sparse.setdefault(
                "landmark_meta_path", f"{detector_folder}/landmark_meta.pt"
            )
    sparse["detector_model_path"] = artifact_model_path
    sparse["landmark_model_path"] = artifact_model_path
    sparse["landmark_meta_model_path"] = artifact_model_path
    sparse["use_landmark_prior"] = False
    if sparse_query_feature_contract is not None:
        sparse["query_feature_contract"] = str(sparse_query_feature_contract)
    if sparse_frontend is not None:
        sparse["sparse_frontend"] = str(sparse_frontend)
    if sparse_ransac_seed is not None:
        sparse["ransac_seed"] = int(sparse_ransac_seed)
    if sparse_solver is not None:
        sparse["solver"] = str(sparse_solver)
    group_saturated_values = {
        "group_saturated_cap": group_saturated_cap,
        "group_saturated_surface_voxel_scale_ratio": (
            group_saturated_surface_voxel_scale_ratio
        ),
        "group_saturated_surface_minimum_voxel_size": (
            group_saturated_surface_minimum_voxel_size
        ),
        "group_saturated_surface_normal_angle_degrees": (
            group_saturated_surface_normal_angle_degrees
        ),
    }
    for key, value in group_saturated_values.items():
        if value is not None:
            sparse[key] = float(value)
    if preemptive_verification_order is not None:
        sparse["preemptive_verification_order"] = str(
            preemptive_verification_order
        )
    if preemptive_check_interval is not None:
        sparse["preemptive_check_interval"] = int(
            preemptive_check_interval
        )
    if metric_state_path is not None:
        sparse["metric_state_path"] = str(metric_state_path)
        sparse["metric_state_model_path"] = artifact_model_path
    if family_prototype_state_path is not None:
        sparse["family_prototype_state_path"] = str(
            family_prototype_state_path
        )
        sparse["family_prototype_state_model_path"] = artifact_model_path
    if pose_sufficient_selector_state_path is not None:
        sparse["use_pose_sufficient_selector"] = True
        sparse["pose_sufficient_selector_state_path"] = str(
            pose_sufficient_selector_state_path
        )
        sparse["pose_sufficient_selector_state_model_path"] = (
            artifact_model_path
        )
    if pose_sufficient_budget is not None:
        sparse["pose_sufficient_budget"] = int(pose_sufficient_budget)
    if rerank_state_path is not None:
        sparse["rerank_state_path"] = str(rerank_state_path)
        sparse["rerank_state_model_path"] = artifact_model_path
    detector_summary_path = os.path.join(
        artifact_model_path,
        detector_folder,
        "candidate_teacher_training_summary.json",
    )
    if (
        str(sparse.get("sparse_frontend", "detector")) == "detector"
        and os.path.isfile(detector_summary_path)
    ):
        with open(detector_summary_path, "r", encoding="utf-8") as handle:
            detector_summary = json.load(handle)
        trained_contract = detector_summary.get("config", {}).get(
            "query_feature_contract"
        )
        if trained_contract:
            configured_contract = sparse.get(
                "query_feature_contract", "legacy_full_then_resized_map"
            )
            if str(trained_contract) != str(configured_contract):
                raise ValueError(
                    "Detector training/evaluation query feature contracts differ: "
                    f"trained={trained_contract!r} configured={configured_contract!r}"
                )
            sparse["detector_training_query_feature_contract"] = str(
                trained_contract
            )
            sparse["detector_training_summary_path"] = detector_summary_path
    if candidate_teacher_state_path:
        sparse["candidate_teacher_state_path"] = str(candidate_teacher_state_path)
        sparse["candidate_teacher_state_model_path"] = artifact_model_path
    if pair_scorer_state_path:
        sparse["pair_scorer_state_path"] = str(pair_scorer_state_path)
        sparse["pair_scorer_state_model_path"] = artifact_model_path
    if landmark_feature_override_path:
        sparse["landmark_feature_override_path"] = str(
            landmark_feature_override_path
        )
        sparse["landmark_feature_override_model_path"] = artifact_model_path
    if override_landmark_features is not None:
        sparse["override_landmark_features"] = bool(override_landmark_features)
    if materialized_anchor_map_path is not None:
        sparse["materialized_anchor_map_path"] = str(
            materialized_anchor_map_path
        )
        sparse["materialized_anchor_map_model_path"] = artifact_model_path
    if pair_measurement_state_path:
        sparse["pair_measurement_state_path"] = str(
            pair_measurement_state_path
        )
        sparse["pair_measurement_state_model_path"] = artifact_model_path
    if detect_num is not None:
        sparse["detect_num"] = int(detect_num)
    if reprojection_error is not None:
        sparse["reprojection_error"] = float(reprojection_error)
    if nms is not None:
        sparse["nms"] = int(nms)
    if match_threshold is not None:
        sparse["threshold"] = float(match_threshold)
    if match_topk is not None:
        sparse["topk"] = int(match_topk)
    if unique_landmark_matches is not None:
        sparse["unique_landmark_matches"] = bool(unique_landmark_matches)
    if max_matches_per_keypoint is not None:
        sparse["max_matches_per_keypoint"] = int(max_matches_per_keypoint)
    else:
        sparse.setdefault("max_matches_per_keypoint", 0)
    if max_matches_per_landmark is not None:
        sparse["max_matches_per_landmark"] = int(max_matches_per_landmark)
    else:
        sparse.setdefault("max_matches_per_landmark", 0)
    optional_sparse = {
        "use_candidate_dustbin": (use_candidate_dustbin, bool),
        "use_candidate_pair_scorer": (use_candidate_pair_scorer, bool),
        "pair_scorer_threshold": (pair_scorer_threshold, float),
        "use_candidate_pair_scorer_calibrated_threshold": (
            use_candidate_pair_scorer_calibrated_threshold,
            bool,
        ),
        "use_pair_measurement": (use_pair_measurement, bool),
        "pair_measurement_threshold": (pair_measurement_threshold, float),
        "use_pair_measurement_calibrated_threshold": (
            use_pair_measurement_calibrated_threshold,
            bool,
        ),
        "use_pair_measurement_offset": (use_pair_measurement_offset, bool),
        "use_pair_measurement_covariance_refinement": (
            use_pair_measurement_covariance_refinement,
            bool,
        ),
        "pair_measurement_refinement_iterations": (
            pair_measurement_refinement_iterations,
            int,
        ),
        "pair_measurement_mahalanobis_threshold": (
            pair_measurement_mahalanobis_threshold,
            float,
        ),
        "pair_measurement_robust_delta": (pair_measurement_robust_delta, float),
        "pair_measurement_covariance_model_floor_px": (
            pair_measurement_covariance_model_floor_px,
            float,
        ),
        "use_pair_measurement_progressive_sampling": (
            use_pair_measurement_progressive_sampling,
            bool,
        ),
        "pair_measurement_max_prosac_iterations": (
            pair_measurement_max_prosac_iterations,
            int,
        ),
        "pair_measurement_fixed_candidate_count": (
            pair_measurement_fixed_candidate_count,
            int,
        ),
        "pair_measurement_refill_mode": (pair_measurement_refill_mode, str),
        "pair_measurement_refill_grid_rows": (
            pair_measurement_refill_grid_rows,
            int,
        ),
        "pair_measurement_refill_grid_cols": (
            pair_measurement_refill_grid_cols,
            int,
        ),
        "pair_measurement_refill_voxel_size": (
            pair_measurement_refill_voxel_size,
            float,
        ),
        "pair_measurement_refill_spatial_weight": (
            pair_measurement_refill_spatial_weight,
            float,
        ),
        "pair_measurement_refill_voxel_weight": (
            pair_measurement_refill_voxel_weight,
            float,
        ),
        "min_candidate_matches": (min_candidate_matches, int),
        "candidate_refill_trigger_count": (candidate_refill_trigger_count, int),
        "use_two_stage_pose_refinement": (
            use_two_stage_pose_refinement,
            bool,
        ),
        "two_stage_tight_reprojection_error": (
            two_stage_tight_reprojection_error,
            float,
        ),
        "two_stage_min_inliers": (two_stage_min_inliers, int),
        "two_stage_refinement_iterations": (
            two_stage_refinement_iterations,
            int,
        ),
        "two_stage_robust_delta": (two_stage_robust_delta, float),
        "two_stage_damping": (two_stage_damping, float),
        "use_detector_matchability": (use_detector_matchability, bool),
        "detector_matchability_mode": (detector_matchability_mode, str),
        "use_detector_offset": (use_detector_offset, bool),
        "detector_max_offset": (detector_max_offset, float),
        "use_native_matchability": (use_native_matchability, bool),
        "native_matchability_state_path": (
            native_matchability_state_path,
            str,
        ),
        "native_matchability_max_prosac_iterations": (
            native_matchability_max_prosac_iterations,
            int,
        ),
        "full_primitive_retrieval": (full_primitive_retrieval, bool),
        "full_primitive_retrieval_topk": (full_primitive_retrieval_topk, int),
        "full_primitive_chunk_size": (full_primitive_chunk_size, int),
        "full_primitive_surface_suppression": (
            full_primitive_surface_suppression,
            bool,
        ),
        "full_primitive_voxel_size": (full_primitive_voxel_size, float),
        "full_primitive_max_per_surface": (
            full_primitive_max_per_surface,
            int,
        ),
        "rerank_topk": (rerank_topk, int),
        "rerank_patch_radius": (rerank_patch_radius, int),
        "rerank_patch_step_px": (rerank_patch_step_px, float),
        "rerank_global_weight": (rerank_global_weight, float),
        "rerank_local_peak_weight": (rerank_local_peak_weight, float),
        "rerank_local_margin_weight": (rerank_local_margin_weight, float),
        "rerank_local_entropy_weight": (
            rerank_local_entropy_weight,
            float,
        ),
        "rerank_offset_weight": (rerank_offset_weight, float),
        "rerank_local_temperature": (rerank_local_temperature, float),
        "rerank_null_score_threshold": (
            rerank_null_score_threshold,
            float,
        ),
        "rerank_null_margin_threshold": (
            rerank_null_margin_threshold,
            float,
        ),
        "rerank_use_learned_null": (rerank_use_learned_null, bool),
        "rerank_assignment_global_preserve_scale": (
            rerank_assignment_global_preserve_scale,
            float,
        ),
    }
    for key, (value, converter) in optional_sparse.items():
        if value is not None:
            sparse[key] = converter(value)
    if candidate_frontend_match_policy is not None:
        sparse["candidate_frontend_match_policy"] = str(
            candidate_frontend_match_policy
        )
    else:
        sparse.setdefault("candidate_frontend_match_policy", "warn")
    diagnostic_values = {
        "dump_correspondences": (diagnostics_dump_correspondences, bool),
        "dump_inliers_only": (diagnostics_dump_inliers_only, bool),
        "dump_discrete_oracle": (diagnostics_dump_discrete_oracle, bool),
        "oracle_topk": (diagnostics_oracle_topk, int),
        "grid_rows": (diagnostics_grid_rows, int),
        "grid_cols": (diagnostics_grid_cols, int),
        "voxel_size": (diagnostics_voxel_size, float),
        "task_translation_scale_m": (
            diagnostics_task_translation_scale_m,
            float,
        ),
        "task_rotation_scale_degrees": (
            diagnostics_task_rotation_scale_degrees,
            float,
        ),
    }
    if diagnostics is not None or any(
        value is not None for value, _ in diagnostic_values.values()
    ):
        diagnostic_cfg = sparse.setdefault("diagnostics", {})
        if diagnostics is not None:
            diagnostic_cfg["enabled"] = bool(diagnostics)
            diagnostic_cfg["gt_metrics"] = bool(diagnostics)
            diagnostic_cfg.setdefault("dump_pre_selector", True)
        elif "enabled" not in diagnostic_cfg:
            diagnostic_cfg["enabled"] = True
            diagnostic_cfg["gt_metrics"] = True
            diagnostic_cfg["dump_pre_selector"] = True
        for key, (value, converter) in diagnostic_values.items():
            if value is not None:
                diagnostic_cfg[key] = converter(value)
    if geometry_balance is True:
        sparse["geometry_balance"] = {
            "enabled": True,
            "grid_rows": int(geometry_balance_grid_rows),
            "grid_cols": int(geometry_balance_grid_cols),
            "max_per_cell": int(geometry_balance_max_per_cell),
            "voxel_size": float(geometry_balance_voxel_size),
            "max_per_voxel": int(geometry_balance_max_per_voxel),
            "max_matches": int(geometry_balance_max_matches),
        }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(cfg, f)
    return {
        "output": output,
        "artifact_model_path": artifact_model_path,
        "detector_path": sparse["detector_path"],
        "landmark_path": sparse["landmark_path"],
        "landmark_meta_path": sparse["landmark_meta_path"],
        "candidate_teacher_state_path": sparse.get("candidate_teacher_state_path"),
        "pair_scorer_state_path": sparse.get("pair_scorer_state_path"),
        "landmark_feature_override_path": sparse.get(
            "landmark_feature_override_path"
        ),
        "override_landmark_features": sparse.get(
            "override_landmark_features", False
        ),
        "materialized_anchor_map_path": sparse.get(
            "materialized_anchor_map_path"
        ),
        "solver": sparse.get("solver"),
        "preemptive_verification_order": sparse.get(
            "preemptive_verification_order"
        ),
        "preemptive_check_interval": sparse.get(
            "preemptive_check_interval"
        ),
        "group_saturated_cap": sparse.get("group_saturated_cap"),
        "pair_measurement_state_path": sparse.get(
            "pair_measurement_state_path"
        ),
        "detect_num": sparse.get("detect_num"),
        "reprojection_error": sparse.get("reprojection_error"),
        "nms": sparse.get("nms"),
        "match_threshold": sparse.get("threshold"),
        "match_topk": sparse.get("topk"),
        "unique_landmark_matches": sparse.get("unique_landmark_matches", False),
        "max_matches_per_keypoint": sparse.get("max_matches_per_keypoint", 0),
        "max_matches_per_landmark": sparse.get("max_matches_per_landmark", 0),
        "use_candidate_dustbin": sparse.get("use_candidate_dustbin", False),
        "use_candidate_pair_scorer": sparse.get("use_candidate_pair_scorer", False),
        "pair_scorer_threshold": sparse.get("pair_scorer_threshold", 0.0),
        "use_candidate_pair_scorer_calibrated_threshold": sparse.get(
            "use_candidate_pair_scorer_calibrated_threshold", False
        ),
        "use_pair_measurement": sparse.get("use_pair_measurement", False),
        "pair_measurement_threshold": sparse.get(
            "pair_measurement_threshold", 0.0
        ),
        "use_pair_measurement_calibrated_threshold": sparse.get(
            "use_pair_measurement_calibrated_threshold", False
        ),
        "use_pair_measurement_offset": sparse.get(
            "use_pair_measurement_offset", True
        ),
        "use_pair_measurement_covariance_refinement": sparse.get(
            "use_pair_measurement_covariance_refinement", False
        ),
        "pair_measurement_covariance_model_floor_px": sparse.get(
            "pair_measurement_covariance_model_floor_px", 1.0
        ),
        "use_pair_measurement_progressive_sampling": sparse.get(
            "use_pair_measurement_progressive_sampling", False
        ),
        "pair_measurement_max_prosac_iterations": sparse.get(
            "pair_measurement_max_prosac_iterations", 100000
        ),
        "pair_measurement_fixed_candidate_count": sparse.get(
            "pair_measurement_fixed_candidate_count", 0
        ),
        "pair_measurement_refill_mode": sparse.get(
            "pair_measurement_refill_mode", "score"
        ),
        "pair_measurement_refill_grid_rows": sparse.get(
            "pair_measurement_refill_grid_rows", 4
        ),
        "pair_measurement_refill_grid_cols": sparse.get(
            "pair_measurement_refill_grid_cols", 4
        ),
        "pair_measurement_refill_voxel_size": sparse.get(
            "pair_measurement_refill_voxel_size", 0.25
        ),
        "pair_measurement_refill_spatial_weight": sparse.get(
            "pair_measurement_refill_spatial_weight", 0.25
        ),
        "pair_measurement_refill_voxel_weight": sparse.get(
            "pair_measurement_refill_voxel_weight", 0.25
        ),
        "min_candidate_matches": sparse.get("min_candidate_matches", 0),
        "candidate_refill_trigger_count": sparse.get("candidate_refill_trigger_count", 0),
        "use_two_stage_pose_refinement": sparse.get(
            "use_two_stage_pose_refinement", False
        ),
        "two_stage_tight_reprojection_error": sparse.get(
            "two_stage_tight_reprojection_error", 4.0
        ),
        "two_stage_min_inliers": sparse.get("two_stage_min_inliers", 6),
        "two_stage_refinement_iterations": sparse.get(
            "two_stage_refinement_iterations", 10
        ),
        "two_stage_robust_delta": sparse.get("two_stage_robust_delta", 1.5),
        "two_stage_damping": sparse.get("two_stage_damping", 1e-6),
        "use_detector_matchability": sparse.get("use_detector_matchability", False),
        "detector_matchability_mode": sparse.get(
            "detector_matchability_mode", "combined_nms"
        ),
        "use_detector_offset": sparse.get("use_detector_offset", False),
        "detector_max_offset": sparse.get("detector_max_offset", 2.0),
        "use_native_matchability": sparse.get("use_native_matchability", False),
        "native_matchability_state_path": sparse.get(
            "native_matchability_state_path", ""
        ),
        "native_matchability_max_prosac_iterations": sparse.get(
            "native_matchability_max_prosac_iterations",
            sparse.get("max_iterations", 100000),
        ),
        "sparse_frontend": sparse.get("sparse_frontend", "detector"),
        "family_prototype_state_path": sparse.get(
            "family_prototype_state_path", ""
        ),
        "use_pose_sufficient_selector": sparse.get(
            "use_pose_sufficient_selector", False
        ),
        "pose_sufficient_selector_state_path": sparse.get(
            "pose_sufficient_selector_state_path", ""
        ),
        "pose_sufficient_budget": sparse.get(
            "pose_sufficient_budget", 0
        ),
        "candidate_frontend_match_policy": sparse.get(
            "candidate_frontend_match_policy", "warn"
        ),
        "diagnostics": sparse.get("diagnostics"),
        "geometry_balance": sparse.get("geometry_balance"),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a STDLoc eval config with explicit artifact paths.")
    parser.add_argument("--base_cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact_model_path", required=True)
    parser.add_argument(
        "--detector_folder",
        default=None,
        help="Override the base config's detector folder; requires --detector_iters.",
    )
    parser.add_argument(
        "--detector_iters",
        type=int,
        default=None,
        help="Override the base config's detector iteration; requires --detector_folder.",
    )
    parser.add_argument(
        "--landmark_path",
        default="",
        help="Explicit sparse landmark ID artifact, independent of detector_folder.",
    )
    parser.add_argument(
        "--landmark_meta_path",
        default="",
        help="Explicit landmark metadata artifact, independent of detector_folder.",
    )
    parser.add_argument("--candidate_teacher_state_path", default="")
    parser.add_argument("--pair_scorer_state_path", default="")
    parser.add_argument("--landmark_feature_override_path", default="")
    parser.add_argument(
        "--override_landmark_features", action="store_true", default=None
    )
    parser.add_argument("--materialized_anchor_map_path", default=None)
    parser.add_argument("--pair_measurement_state_path", default="")
    parser.add_argument("--detect_num", type=int, default=None)
    parser.add_argument("--reprojection_error", type=float, default=None)
    parser.add_argument("--nms", type=int, default=None)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--match_topk", type=int, default=None)
    parser.add_argument("--unique_landmark_matches", action="store_true", default=None)
    parser.add_argument("--max_matches_per_keypoint", type=int, default=None)
    parser.add_argument("--max_matches_per_landmark", type=int, default=None)
    parser.add_argument("--use_candidate_dustbin", action="store_true", default=None)
    parser.add_argument("--use_candidate_pair_scorer", action="store_true", default=None)
    parser.add_argument("--pair_scorer_threshold", type=float, default=None)
    parser.add_argument(
        "--use_candidate_pair_scorer_calibrated_threshold", action="store_true", default=None
    )
    parser.add_argument("--use_pair_measurement", action="store_true", default=None)
    parser.add_argument("--pair_measurement_threshold", type=float, default=None)
    parser.add_argument(
        "--use_pair_measurement_calibrated_threshold", action="store_true", default=None
    )
    parser.add_argument(
        "--disable_pair_measurement_offset", action="store_true", default=None
    )
    parser.add_argument(
        "--use_pair_measurement_covariance_refinement", action="store_true", default=None
    )
    parser.add_argument(
        "--pair_measurement_refinement_iterations", type=int, default=None
    )
    parser.add_argument(
        "--pair_measurement_mahalanobis_threshold", type=float, default=None
    )
    parser.add_argument(
        "--pair_measurement_robust_delta", type=float, default=None
    )
    parser.add_argument(
        "--pair_measurement_covariance_model_floor_px", type=float, default=None
    )
    parser.add_argument(
        "--use_pair_measurement_progressive_sampling", action="store_true", default=None
    )
    parser.add_argument(
        "--pair_measurement_max_prosac_iterations", type=int, default=None
    )
    parser.add_argument(
        "--pair_measurement_fixed_candidate_count", type=int, default=None
    )
    parser.add_argument(
        "--pair_measurement_refill_mode",
        choices=["score", "geometry"],
        default=None,
    )
    parser.add_argument("--pair_measurement_refill_grid_rows", type=int, default=None)
    parser.add_argument("--pair_measurement_refill_grid_cols", type=int, default=None)
    parser.add_argument(
        "--pair_measurement_refill_voxel_size", type=float, default=None
    )
    parser.add_argument(
        "--pair_measurement_refill_spatial_weight", type=float, default=None
    )
    parser.add_argument(
        "--pair_measurement_refill_voxel_weight", type=float, default=None
    )
    parser.add_argument("--min_candidate_matches", type=int, default=None)
    parser.add_argument("--candidate_refill_trigger_count", type=int, default=None)
    parser.add_argument(
        "--use_two_stage_pose_refinement", action="store_true", default=None,
        help="Refine the wide-RANSAC pose on the same tight sparse candidate set.",
    )
    parser.add_argument("--two_stage_tight_reprojection_error", type=float, default=None)
    parser.add_argument("--two_stage_min_inliers", type=int, default=None)
    parser.add_argument("--two_stage_refinement_iterations", type=int, default=None)
    parser.add_argument("--two_stage_robust_delta", type=float, default=None)
    parser.add_argument("--two_stage_damping", type=float, default=None)
    parser.add_argument("--use_detector_matchability", action="store_true", default=None)
    parser.add_argument(
        "--detector_matchability_mode",
        choices=["combined_nms", "proposal_rerank"],
        default=None,
    )
    parser.add_argument("--use_detector_offset", action="store_true", default=None)
    parser.add_argument("--detector_max_offset", type=float, default=None)
    parser.add_argument("--use_native_matchability", action="store_true", default=None)
    parser.add_argument("--native_matchability_state_path", default=None)
    parser.add_argument(
        "--native_matchability_max_prosac_iterations", type=int, default=None
    )
    parser.add_argument(
        "--candidate_frontend_match_policy",
        choices=["error", "warn", "ignore"],
        default=None,
    )
    diagnostics_group = parser.add_mutually_exclusive_group()
    diagnostics_group.add_argument(
        "--diagnostics",
        action="store_true",
        default=None,
        help="Enable GT correspondence and pose-information diagnostics.",
    )
    diagnostics_group.add_argument(
        "--no_diagnostics",
        action="store_true",
        default=None,
        help="Explicitly disable sparse localization diagnostics.",
    )
    parser.add_argument("--diagnostics_dump_correspondences", action="store_true", default=None)
    parser.add_argument("--diagnostics_dump_all", action="store_true", default=None)
    parser.add_argument("--diagnostics_dump_discrete_oracle", action="store_true", default=None)
    parser.add_argument("--diagnostics_oracle_topk", type=int, default=None)
    parser.add_argument("--diagnostics_grid_rows", type=int, default=None)
    parser.add_argument("--diagnostics_grid_cols", type=int, default=None)
    parser.add_argument("--diagnostics_voxel_size", type=float, default=None)
    parser.add_argument(
        "--diagnostics_task_translation_scale_m", type=float, default=None
    )
    parser.add_argument(
        "--diagnostics_task_rotation_scale_degrees", type=float, default=None
    )
    parser.add_argument("--geometry_balance", action="store_true", default=None)
    parser.add_argument("--geometry_balance_grid_rows", type=int, default=4)
    parser.add_argument("--geometry_balance_grid_cols", type=int, default=4)
    parser.add_argument("--geometry_balance_max_per_cell", type=int, default=64)
    parser.add_argument("--geometry_balance_voxel_size", type=float, default=0.25)
    parser.add_argument("--geometry_balance_max_per_voxel", type=int, default=64)
    parser.add_argument("--geometry_balance_max_matches", type=int, default=0)
    parser.add_argument("--full_primitive_retrieval", action="store_true", default=None)
    parser.add_argument("--full_primitive_retrieval_topk", type=int, default=None)
    parser.add_argument("--full_primitive_chunk_size", type=int, default=None)
    parser.add_argument(
        "--full_primitive_surface_suppression", action="store_true", default=None
    )
    parser.add_argument("--full_primitive_voxel_size", type=float, default=None)
    parser.add_argument("--full_primitive_max_per_surface", type=int, default=None)
    parser.add_argument(
        "--sparse_query_feature_contract",
        choices=["legacy_full_then_resized_map", "native_resized_input"],
        default=None,
    )
    parser.add_argument(
        "--sparse_frontend",
        choices=[
            "detector",
            "ulfloc_native",
            "ulfloc_native_rerank",
            "ulfloc_native_adapter",
            "ulfloc_native_metric",
        ],
        default=None,
    )
    parser.add_argument("--metric_state_path", default=None)
    parser.add_argument("--family_prototype_state_path", default=None)
    parser.add_argument(
        "--pose_sufficient_selector_state_path", default=None
    )
    parser.add_argument("--pose_sufficient_budget", type=int, default=None)
    parser.add_argument("--sparse_ransac_seed", type=int, default=None)
    parser.add_argument(
        "--sparse_solver",
        choices=[
            "opencv",
            "poselib",
            "poselib_preemptive",
            "poselib_dependency",
            "poselib_group_saturated",
        ],
        default=None,
    )
    parser.add_argument("--group_saturated_cap", type=float, default=None)
    parser.add_argument(
        "--preemptive_verification_order",
        choices=["input", "low_similarity", "low_margin", "low_confidence"],
        default=None,
    )
    parser.add_argument(
        "--preemptive_check_interval", type=int, default=None
    )
    parser.add_argument(
        "--group_saturated_surface_voxel_scale_ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--group_saturated_surface_minimum_voxel_size",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--group_saturated_surface_normal_angle_degrees",
        type=float,
        default=None,
    )
    parser.add_argument("--rerank_topk", type=int, default=None)
    parser.add_argument("--rerank_patch_radius", type=int, default=None)
    parser.add_argument("--rerank_patch_step_px", type=float, default=None)
    parser.add_argument("--rerank_global_weight", type=float, default=None)
    parser.add_argument("--rerank_local_peak_weight", type=float, default=None)
    parser.add_argument("--rerank_local_margin_weight", type=float, default=None)
    parser.add_argument("--rerank_local_entropy_weight", type=float, default=None)
    parser.add_argument("--rerank_offset_weight", type=float, default=None)
    parser.add_argument("--rerank_local_temperature", type=float, default=None)
    parser.add_argument("--rerank_null_score_threshold", type=float, default=None)
    parser.add_argument("--rerank_null_margin_threshold", type=float, default=None)
    parser.add_argument("--rerank_state_path", default=None)
    parser.add_argument(
        "--rerank_use_learned_null",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rerank_assignment_global_preserve_scale",
        type=float,
        default=None,
    )
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    summary = make_stdloc_eval_cfg(
        args.base_cfg,
        args.output,
        args.artifact_model_path,
        detector_folder=args.detector_folder,
        detector_iters=args.detector_iters,
        landmark_path=args.landmark_path,
        landmark_meta_path=args.landmark_meta_path,
        candidate_teacher_state_path=args.candidate_teacher_state_path,
        pair_scorer_state_path=args.pair_scorer_state_path,
        landmark_feature_override_path=args.landmark_feature_override_path,
        override_landmark_features=args.override_landmark_features,
        materialized_anchor_map_path=args.materialized_anchor_map_path,
        pair_measurement_state_path=args.pair_measurement_state_path,
        detect_num=args.detect_num,
        reprojection_error=args.reprojection_error,
        nms=args.nms,
        match_threshold=args.match_threshold,
        match_topk=args.match_topk,
        unique_landmark_matches=args.unique_landmark_matches,
        max_matches_per_keypoint=args.max_matches_per_keypoint,
        max_matches_per_landmark=args.max_matches_per_landmark,
        use_candidate_dustbin=args.use_candidate_dustbin,
        use_candidate_pair_scorer=args.use_candidate_pair_scorer,
        pair_scorer_threshold=args.pair_scorer_threshold,
        use_candidate_pair_scorer_calibrated_threshold=(
            args.use_candidate_pair_scorer_calibrated_threshold
        ),
        use_pair_measurement=args.use_pair_measurement,
        pair_measurement_threshold=args.pair_measurement_threshold,
        use_pair_measurement_calibrated_threshold=(
            args.use_pair_measurement_calibrated_threshold
        ),
        use_pair_measurement_offset=(
            None
            if args.disable_pair_measurement_offset is None
            else not args.disable_pair_measurement_offset
        ),
        use_pair_measurement_covariance_refinement=(
            args.use_pair_measurement_covariance_refinement
        ),
        pair_measurement_refinement_iterations=(
            args.pair_measurement_refinement_iterations
        ),
        pair_measurement_mahalanobis_threshold=(
            args.pair_measurement_mahalanobis_threshold
        ),
        pair_measurement_robust_delta=args.pair_measurement_robust_delta,
        pair_measurement_covariance_model_floor_px=(
            args.pair_measurement_covariance_model_floor_px
        ),
        use_pair_measurement_progressive_sampling=(
            args.use_pair_measurement_progressive_sampling
        ),
        pair_measurement_max_prosac_iterations=(
            args.pair_measurement_max_prosac_iterations
        ),
        pair_measurement_fixed_candidate_count=(
            args.pair_measurement_fixed_candidate_count
        ),
        pair_measurement_refill_mode=args.pair_measurement_refill_mode,
        pair_measurement_refill_grid_rows=(
            args.pair_measurement_refill_grid_rows
        ),
        pair_measurement_refill_grid_cols=(
            args.pair_measurement_refill_grid_cols
        ),
        pair_measurement_refill_voxel_size=(
            args.pair_measurement_refill_voxel_size
        ),
        pair_measurement_refill_spatial_weight=(
            args.pair_measurement_refill_spatial_weight
        ),
        pair_measurement_refill_voxel_weight=(
            args.pair_measurement_refill_voxel_weight
        ),
        min_candidate_matches=args.min_candidate_matches,
        candidate_refill_trigger_count=args.candidate_refill_trigger_count,
        use_two_stage_pose_refinement=args.use_two_stage_pose_refinement,
        two_stage_tight_reprojection_error=args.two_stage_tight_reprojection_error,
        two_stage_min_inliers=args.two_stage_min_inliers,
        two_stage_refinement_iterations=args.two_stage_refinement_iterations,
        two_stage_robust_delta=args.two_stage_robust_delta,
        two_stage_damping=args.two_stage_damping,
        use_detector_matchability=args.use_detector_matchability,
        detector_matchability_mode=args.detector_matchability_mode,
        use_detector_offset=args.use_detector_offset,
        detector_max_offset=args.detector_max_offset,
        use_native_matchability=args.use_native_matchability,
        native_matchability_state_path=args.native_matchability_state_path,
        native_matchability_max_prosac_iterations=(
            args.native_matchability_max_prosac_iterations
        ),
        candidate_frontend_match_policy=args.candidate_frontend_match_policy,
        diagnostics=(
            args.diagnostics
            if args.diagnostics is not None
            else (
                None
                if args.no_diagnostics is None
                else not args.no_diagnostics
            )
        ),
        diagnostics_dump_correspondences=args.diagnostics_dump_correspondences,
        diagnostics_dump_inliers_only=(
            None
            if args.diagnostics_dump_all is None
            else not args.diagnostics_dump_all
        ),
        diagnostics_dump_discrete_oracle=args.diagnostics_dump_discrete_oracle,
        diagnostics_oracle_topk=args.diagnostics_oracle_topk,
        diagnostics_grid_rows=args.diagnostics_grid_rows,
        diagnostics_grid_cols=args.diagnostics_grid_cols,
        diagnostics_voxel_size=args.diagnostics_voxel_size,
        diagnostics_task_translation_scale_m=(
            args.diagnostics_task_translation_scale_m
        ),
        diagnostics_task_rotation_scale_degrees=(
            args.diagnostics_task_rotation_scale_degrees
        ),
        geometry_balance=args.geometry_balance,
        geometry_balance_grid_rows=args.geometry_balance_grid_rows,
        geometry_balance_grid_cols=args.geometry_balance_grid_cols,
        geometry_balance_max_per_cell=args.geometry_balance_max_per_cell,
        geometry_balance_voxel_size=args.geometry_balance_voxel_size,
        geometry_balance_max_per_voxel=args.geometry_balance_max_per_voxel,
        geometry_balance_max_matches=args.geometry_balance_max_matches,
        full_primitive_retrieval=args.full_primitive_retrieval,
        full_primitive_retrieval_topk=args.full_primitive_retrieval_topk,
        full_primitive_chunk_size=args.full_primitive_chunk_size,
        full_primitive_surface_suppression=(
            args.full_primitive_surface_suppression
        ),
        full_primitive_voxel_size=args.full_primitive_voxel_size,
        full_primitive_max_per_surface=args.full_primitive_max_per_surface,
        sparse_query_feature_contract=args.sparse_query_feature_contract,
        sparse_frontend=args.sparse_frontend,
        sparse_ransac_seed=args.sparse_ransac_seed,
        sparse_solver=args.sparse_solver,
        group_saturated_cap=args.group_saturated_cap,
        group_saturated_surface_voxel_scale_ratio=(
            args.group_saturated_surface_voxel_scale_ratio
        ),
        group_saturated_surface_minimum_voxel_size=(
            args.group_saturated_surface_minimum_voxel_size
        ),
        group_saturated_surface_normal_angle_degrees=(
            args.group_saturated_surface_normal_angle_degrees
        ),
        preemptive_verification_order=args.preemptive_verification_order,
        preemptive_check_interval=args.preemptive_check_interval,
        metric_state_path=args.metric_state_path,
        family_prototype_state_path=args.family_prototype_state_path,
        pose_sufficient_selector_state_path=(
            args.pose_sufficient_selector_state_path
        ),
        pose_sufficient_budget=args.pose_sufficient_budget,
        rerank_topk=args.rerank_topk,
        rerank_patch_radius=args.rerank_patch_radius,
        rerank_patch_step_px=args.rerank_patch_step_px,
        rerank_global_weight=args.rerank_global_weight,
        rerank_local_peak_weight=args.rerank_local_peak_weight,
        rerank_local_margin_weight=args.rerank_local_margin_weight,
        rerank_local_entropy_weight=args.rerank_local_entropy_weight,
        rerank_offset_weight=args.rerank_offset_weight,
        rerank_local_temperature=args.rerank_local_temperature,
        rerank_null_score_threshold=args.rerank_null_score_threshold,
        rerank_null_margin_threshold=args.rerank_null_margin_threshold,
        rerank_state_path=args.rerank_state_path,
        rerank_use_learned_null=args.rerank_use_learned_null,
        rerank_assignment_global_preserve_scale=(
            args.rerank_assignment_global_preserve_scale
        ),
    )
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
