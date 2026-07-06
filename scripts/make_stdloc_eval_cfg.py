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
    )
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
