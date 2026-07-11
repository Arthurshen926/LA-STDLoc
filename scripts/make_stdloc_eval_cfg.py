#!/usr/bin/env python
import argparse
import json
import os

import yaml


def make_stdloc_eval_cfg(
    base_cfg,
    output,
    artifact_model_path,
    detector_folder="detector",
    detector_iters=30000,
    candidate_teacher_state_path=None,
    pair_scorer_state_path=None,
    landmark_feature_override_path=None,
    override_landmark_features=False,
    pair_measurement_state_path=None,
    detect_num=None,
    reprojection_error=None,
    nms=None,
    match_threshold=None,
    match_topk=None,
    unique_landmark_matches=False,
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    use_candidate_dustbin=False,
    use_candidate_pair_scorer=False,
    pair_scorer_threshold=0.0,
    use_candidate_pair_scorer_calibrated_threshold=False,
    use_pair_measurement=False,
    pair_measurement_threshold=0.0,
    use_pair_measurement_calibrated_threshold=False,
    use_pair_measurement_offset=True,
    use_pair_measurement_covariance_refinement=False,
    pair_measurement_refinement_iterations=10,
    pair_measurement_mahalanobis_threshold=3.0,
    pair_measurement_robust_delta=2.5,
    pair_measurement_covariance_model_floor_px=1.0,
    use_pair_measurement_progressive_sampling=False,
    pair_measurement_max_prosac_iterations=100000,
    pair_measurement_fixed_candidate_count=0,
    pair_measurement_refill_mode="score",
    pair_measurement_refill_grid_rows=4,
    pair_measurement_refill_grid_cols=4,
    pair_measurement_refill_voxel_size=0.25,
    pair_measurement_refill_spatial_weight=0.25,
    pair_measurement_refill_voxel_weight=0.25,
    min_candidate_matches=0,
    candidate_refill_trigger_count=0,
    use_detector_matchability=False,
    detector_matchability_mode="combined_nms",
    use_detector_offset=False,
    detector_max_offset=2.0,
    candidate_frontend_match_policy="warn",
    diagnostics=True,
    diagnostics_dump_correspondences=False,
    diagnostics_dump_inliers_only=True,
    diagnostics_grid_rows=4,
    diagnostics_grid_cols=4,
    diagnostics_voxel_size=0.25,
    geometry_balance=False,
    geometry_balance_grid_rows=4,
    geometry_balance_grid_cols=4,
    geometry_balance_max_per_cell=64,
    geometry_balance_voxel_size=0.25,
    geometry_balance_max_per_voxel=64,
    geometry_balance_max_matches=0,
):
    with open(base_cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["model_path"] = artifact_model_path
    sparse = cfg.setdefault("sparse", {})
    detector_folder = str(detector_folder).strip("/")
    detector_iters = int(detector_iters)
    sparse["detector_path"] = f"{detector_folder}/{detector_iters}_detector.pth"
    sparse["landmark_path"] = f"{detector_folder}/sampled_idx.pkl"
    sparse["landmark_meta_path"] = f"{detector_folder}/landmark_meta.pt"
    sparse["detector_model_path"] = artifact_model_path
    sparse["landmark_model_path"] = artifact_model_path
    sparse["landmark_meta_model_path"] = artifact_model_path
    sparse["use_landmark_prior"] = False
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
    sparse["override_landmark_features"] = bool(override_landmark_features)
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
    sparse["unique_landmark_matches"] = bool(unique_landmark_matches)
    sparse["max_matches_per_keypoint"] = int(max_matches_per_keypoint)
    sparse["max_matches_per_landmark"] = int(max_matches_per_landmark)
    sparse["use_candidate_dustbin"] = bool(use_candidate_dustbin)
    sparse["use_candidate_pair_scorer"] = bool(use_candidate_pair_scorer)
    sparse["pair_scorer_threshold"] = float(pair_scorer_threshold)
    sparse["use_candidate_pair_scorer_calibrated_threshold"] = bool(
        use_candidate_pair_scorer_calibrated_threshold
    )
    sparse["use_pair_measurement"] = bool(use_pair_measurement)
    sparse["pair_measurement_threshold"] = float(pair_measurement_threshold)
    sparse["use_pair_measurement_calibrated_threshold"] = bool(
        use_pair_measurement_calibrated_threshold
    )
    sparse["use_pair_measurement_offset"] = bool(use_pair_measurement_offset)
    sparse["use_pair_measurement_covariance_refinement"] = bool(
        use_pair_measurement_covariance_refinement
    )
    sparse["pair_measurement_refinement_iterations"] = int(
        pair_measurement_refinement_iterations
    )
    sparse["pair_measurement_mahalanobis_threshold"] = float(
        pair_measurement_mahalanobis_threshold
    )
    sparse["pair_measurement_robust_delta"] = float(
        pair_measurement_robust_delta
    )
    sparse["pair_measurement_covariance_model_floor_px"] = float(
        pair_measurement_covariance_model_floor_px
    )
    sparse["use_pair_measurement_progressive_sampling"] = bool(
        use_pair_measurement_progressive_sampling
    )
    sparse["pair_measurement_max_prosac_iterations"] = int(
        pair_measurement_max_prosac_iterations
    )
    sparse["pair_measurement_fixed_candidate_count"] = int(
        pair_measurement_fixed_candidate_count
    )
    sparse["pair_measurement_refill_mode"] = str(pair_measurement_refill_mode)
    sparse["pair_measurement_refill_grid_rows"] = int(
        pair_measurement_refill_grid_rows
    )
    sparse["pair_measurement_refill_grid_cols"] = int(
        pair_measurement_refill_grid_cols
    )
    sparse["pair_measurement_refill_voxel_size"] = float(
        pair_measurement_refill_voxel_size
    )
    sparse["pair_measurement_refill_spatial_weight"] = float(
        pair_measurement_refill_spatial_weight
    )
    sparse["pair_measurement_refill_voxel_weight"] = float(
        pair_measurement_refill_voxel_weight
    )
    sparse["min_candidate_matches"] = int(min_candidate_matches)
    sparse["candidate_refill_trigger_count"] = int(candidate_refill_trigger_count)
    sparse["use_detector_matchability"] = bool(use_detector_matchability)
    sparse["detector_matchability_mode"] = str(detector_matchability_mode)
    sparse["use_detector_offset"] = bool(use_detector_offset)
    sparse["detector_max_offset"] = float(detector_max_offset)
    sparse["candidate_frontend_match_policy"] = str(candidate_frontend_match_policy)
    sparse["diagnostics"] = {
        "enabled": bool(diagnostics),
        "gt_metrics": bool(diagnostics),
        "dump_correspondences": bool(diagnostics_dump_correspondences),
        "dump_inliers_only": bool(diagnostics_dump_inliers_only),
        "dump_pre_selector": True,
        "grid_rows": int(diagnostics_grid_rows),
        "grid_cols": int(diagnostics_grid_cols),
        "voxel_size": float(diagnostics_voxel_size),
    }
    if geometry_balance:
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
        "candidate_teacher_state_path": sparse.get("candidate_teacher_state_path"),
        "pair_scorer_state_path": sparse.get("pair_scorer_state_path"),
        "landmark_feature_override_path": sparse.get(
            "landmark_feature_override_path"
        ),
        "override_landmark_features": sparse.get(
            "override_landmark_features", False
        ),
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
        "use_detector_matchability": sparse.get("use_detector_matchability", False),
        "detector_matchability_mode": sparse.get(
            "detector_matchability_mode", "combined_nms"
        ),
        "use_detector_offset": sparse.get("use_detector_offset", False),
        "detector_max_offset": sparse.get("detector_max_offset", 2.0),
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
    parser.add_argument("--detector_folder", default="detector")
    parser.add_argument("--detector_iters", type=int, default=30000)
    parser.add_argument("--candidate_teacher_state_path", default="")
    parser.add_argument("--pair_scorer_state_path", default="")
    parser.add_argument("--landmark_feature_override_path", default="")
    parser.add_argument("--override_landmark_features", action="store_true")
    parser.add_argument("--pair_measurement_state_path", default="")
    parser.add_argument("--detect_num", type=int, default=None)
    parser.add_argument("--reprojection_error", type=float, default=None)
    parser.add_argument("--nms", type=int, default=None)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--match_topk", type=int, default=None)
    parser.add_argument("--unique_landmark_matches", action="store_true")
    parser.add_argument("--max_matches_per_keypoint", type=int, default=0)
    parser.add_argument("--max_matches_per_landmark", type=int, default=0)
    parser.add_argument("--use_candidate_dustbin", action="store_true")
    parser.add_argument("--use_candidate_pair_scorer", action="store_true")
    parser.add_argument("--pair_scorer_threshold", type=float, default=0.0)
    parser.add_argument(
        "--use_candidate_pair_scorer_calibrated_threshold", action="store_true"
    )
    parser.add_argument("--use_pair_measurement", action="store_true")
    parser.add_argument("--pair_measurement_threshold", type=float, default=0.0)
    parser.add_argument(
        "--use_pair_measurement_calibrated_threshold", action="store_true"
    )
    parser.add_argument(
        "--disable_pair_measurement_offset", action="store_true"
    )
    parser.add_argument(
        "--use_pair_measurement_covariance_refinement", action="store_true"
    )
    parser.add_argument(
        "--pair_measurement_refinement_iterations", type=int, default=10
    )
    parser.add_argument(
        "--pair_measurement_mahalanobis_threshold", type=float, default=3.0
    )
    parser.add_argument(
        "--pair_measurement_robust_delta", type=float, default=2.5
    )
    parser.add_argument(
        "--pair_measurement_covariance_model_floor_px", type=float, default=1.0
    )
    parser.add_argument(
        "--use_pair_measurement_progressive_sampling", action="store_true"
    )
    parser.add_argument(
        "--pair_measurement_max_prosac_iterations", type=int, default=100000
    )
    parser.add_argument(
        "--pair_measurement_fixed_candidate_count", type=int, default=0
    )
    parser.add_argument(
        "--pair_measurement_refill_mode",
        choices=["score", "geometry"],
        default="score",
    )
    parser.add_argument("--pair_measurement_refill_grid_rows", type=int, default=4)
    parser.add_argument("--pair_measurement_refill_grid_cols", type=int, default=4)
    parser.add_argument(
        "--pair_measurement_refill_voxel_size", type=float, default=0.25
    )
    parser.add_argument(
        "--pair_measurement_refill_spatial_weight", type=float, default=0.25
    )
    parser.add_argument(
        "--pair_measurement_refill_voxel_weight", type=float, default=0.25
    )
    parser.add_argument("--min_candidate_matches", type=int, default=0)
    parser.add_argument("--candidate_refill_trigger_count", type=int, default=0)
    parser.add_argument("--use_detector_matchability", action="store_true")
    parser.add_argument(
        "--detector_matchability_mode",
        choices=["combined_nms", "proposal_rerank"],
        default="combined_nms",
    )
    parser.add_argument("--use_detector_offset", action="store_true")
    parser.add_argument("--detector_max_offset", type=float, default=2.0)
    parser.add_argument(
        "--candidate_frontend_match_policy",
        choices=["error", "warn", "ignore"],
        default="warn",
    )
    parser.add_argument("--no_diagnostics", action="store_true")
    parser.add_argument("--diagnostics_dump_correspondences", action="store_true")
    parser.add_argument("--diagnostics_dump_all", action="store_true")
    parser.add_argument("--diagnostics_grid_rows", type=int, default=4)
    parser.add_argument("--diagnostics_grid_cols", type=int, default=4)
    parser.add_argument("--diagnostics_voxel_size", type=float, default=0.25)
    parser.add_argument("--geometry_balance", action="store_true")
    parser.add_argument("--geometry_balance_grid_rows", type=int, default=4)
    parser.add_argument("--geometry_balance_grid_cols", type=int, default=4)
    parser.add_argument("--geometry_balance_max_per_cell", type=int, default=64)
    parser.add_argument("--geometry_balance_voxel_size", type=float, default=0.25)
    parser.add_argument("--geometry_balance_max_per_voxel", type=int, default=64)
    parser.add_argument("--geometry_balance_max_matches", type=int, default=0)
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    summary = make_stdloc_eval_cfg(
        args.base_cfg,
        args.output,
        args.artifact_model_path,
        detector_folder=args.detector_folder,
        detector_iters=args.detector_iters,
        candidate_teacher_state_path=args.candidate_teacher_state_path,
        pair_scorer_state_path=args.pair_scorer_state_path,
        landmark_feature_override_path=args.landmark_feature_override_path,
        override_landmark_features=args.override_landmark_features,
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
        use_pair_measurement_offset=not args.disable_pair_measurement_offset,
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
        use_detector_matchability=args.use_detector_matchability,
        detector_matchability_mode=args.detector_matchability_mode,
        use_detector_offset=args.use_detector_offset,
        detector_max_offset=args.detector_max_offset,
        candidate_frontend_match_policy=args.candidate_frontend_match_policy,
        diagnostics=not args.no_diagnostics,
        diagnostics_dump_correspondences=args.diagnostics_dump_correspondences,
        diagnostics_dump_inliers_only=not args.diagnostics_dump_all,
        diagnostics_grid_rows=args.diagnostics_grid_rows,
        diagnostics_grid_cols=args.diagnostics_grid_cols,
        diagnostics_voxel_size=args.diagnostics_voxel_size,
        geometry_balance=args.geometry_balance,
        geometry_balance_grid_rows=args.geometry_balance_grid_rows,
        geometry_balance_grid_cols=args.geometry_balance_grid_cols,
        geometry_balance_max_per_cell=args.geometry_balance_max_per_cell,
        geometry_balance_voxel_size=args.geometry_balance_voxel_size,
        geometry_balance_max_per_voxel=args.geometry_balance_max_per_voxel,
        geometry_balance_max_matches=args.geometry_balance_max_matches,
    )
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
