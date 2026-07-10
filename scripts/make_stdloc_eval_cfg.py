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
    detect_num=None,
    reprojection_error=None,
    nms=None,
    diagnostics=True,
    diagnostics_dump_correspondences=False,
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
    if detect_num is not None:
        sparse["detect_num"] = int(detect_num)
    if reprojection_error is not None:
        sparse["reprojection_error"] = float(reprojection_error)
    if nms is not None:
        sparse["nms"] = int(nms)
    sparse["diagnostics"] = {
        "enabled": bool(diagnostics),
        "gt_metrics": bool(diagnostics),
        "dump_correspondences": bool(diagnostics_dump_correspondences),
        "dump_inliers_only": True,
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
        "detect_num": sparse.get("detect_num"),
        "reprojection_error": sparse.get("reprojection_error"),
        "nms": sparse.get("nms"),
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
    parser.add_argument("--detect_num", type=int, default=None)
    parser.add_argument("--reprojection_error", type=float, default=None)
    parser.add_argument("--nms", type=int, default=None)
    parser.add_argument("--no_diagnostics", action="store_true")
    parser.add_argument("--diagnostics_dump_correspondences", action="store_true")
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
        detect_num=args.detect_num,
        reprojection_error=args.reprojection_error,
        nms=args.nms,
        diagnostics=not args.no_diagnostics,
        diagnostics_dump_correspondences=args.diagnostics_dump_correspondences,
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
