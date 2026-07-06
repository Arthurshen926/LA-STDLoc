#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_from_pose_gsplat
from la_diagnostics.teacher_stage import (
    build_teacher_stage_records,
    selected_image_names_from_sample_flow,
    summarize_stage_records,
    write_stage_csv,
)
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from scripts.diagnose_sparse_inliers import _detect_and_match, _match_reprojection_correct
from stdloc import STDLoc, get_intrinsic
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose


def _load_gaussians(dataset):
    if dataset.gaussian_type == "3dgs":
        return GaussianModel(dataset.sh_degree)
    if dataset.gaussian_type == "2dgs":
        return GaussianModel_2dgs(dataset.sh_degree)
    raise ValueError(f"Unsupported gaussian_type: {dataset.gaussian_type}")


def _load_config(dataset, cfg_path):
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    config.setdefault("sparse", {})["sparse_only"] = False
    config["sparse_only"] = False
    return config


def _camera_map(scene, split):
    cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    return {camera.image_name: camera for camera in cameras}


def _read_image_names(args):
    names = []
    image_names = getattr(args, "image_names", None) or []
    image_names_file = getattr(args, "image_names_file", None)
    sample_flow_csv = getattr(args, "sample_flow_csv", None)
    sample_flow_groups = getattr(args, "sample_flow_groups", ["final_worst", "final_regressed"])
    max_images = int(getattr(args, "max_images", 0) or 0)
    if image_names:
        names.extend(image_names)
    if image_names_file:
        for line in Path(image_names_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                names.append(line)
    if sample_flow_csv:
        names.extend(
            selected_image_names_from_sample_flow(
                Path(sample_flow_csv),
                sample_flow_groups,
                limit=max_images,
            )
        )
    deduped = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    if max_images > 0:
        deduped = deduped[:max_images]
    return deduped


def _diagnose_sparse(stdloc, fine_feature_map, camera):
    matches = _detect_and_match(stdloc, fine_feature_map)
    if matches is None:
        return {
            "matches": 0,
            "correct_matches": 0,
            "inliers": 0,
            "sparse_match_debug": None,
        }

    gt_w2c = camera.world_view_transform.transpose(0, 1).cuda()
    height, width = fine_feature_map.shape[-2:]
    local_landmark_ids = matches["local_landmark_ids"]
    points3d = stdloc.landmarks.get_xyz[local_landmark_ids]
    query_p2d = matches["query_p2d"]
    correct_mask = _match_reprojection_correct(
        points3d,
        query_p2d,
        gt_w2c,
        camera.FoVx,
        camera.FoVy,
        height,
        width,
        stdloc.config["sparse"]["reprojection_error"],
    )

    K = get_intrinsic(camera.FoVx, camera.FoVy, width, height)
    pose_w2c, inliers = solve_pose(
        (query_p2d + 0.5).detach().cpu().numpy(),
        points3d.detach().cpu().numpy(),
        K,
        stdloc.config["sparse"]["solver"],
        stdloc.config["sparse"]["reprojection_error"],
        stdloc.config["sparse"]["confidence"],
        stdloc.config["sparse"]["max_iterations"],
        stdloc.config["sparse"]["min_iterations"],
    )
    inliers = np.asarray(inliers).reshape(-1).astype(np.int64)
    inlier_mask = torch.zeros(local_landmark_ids.numel(), dtype=torch.bool)
    if inliers.size > 0:
        valid = inliers[(inliers >= 0) & (inliers < local_landmark_ids.numel())]
        inlier_mask[torch.as_tensor(valid, dtype=torch.long)] = True

    ae, te = cal_pose_error(pose_w2c, gt_w2c.detach().cpu().numpy())
    return {
        "matches": int(local_landmark_ids.numel()),
        "correct_matches": int(correct_mask.sum().item()),
        "inliers": int(inlier_mask.sum().item()),
        "diagnostic_sparse_AE": float(ae),
        "diagnostic_sparse_TE": float(te),
        "sparse_match_debug": {
            "query_p2d": query_p2d.detach().cpu().numpy(),
            "points3d": points3d.detach().cpu().numpy(),
            "correct_mask": correct_mask.detach().cpu().numpy().astype(bool),
            "inlier_mask": inlier_mask.detach().cpu().numpy().astype(bool),
            "scores": matches["scores"].detach().cpu().numpy() if "scores" in matches else None,
            "height": int(height),
            "width": int(width),
        },
    }


def _render_rgb(gaussians, config, pose_w2c, camera, longest_edge):
    src = camera.original_image
    height, width = get_resolution_from_longest_edge(src.shape[-2], src.shape[-1], longest_edge)
    with torch.no_grad():
        pkg = render_from_pose_gsplat(
            gaussians,
            torch.as_tensor(pose_w2c, device="cuda", dtype=torch.float32),
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            render_mode="RGB+ED",
            norm_feat_bf_render=config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )
    return torch.clamp(pkg["render"][:3].detach().cpu(), 0.0, 1.0)


def _resize_query(query_image, height, width):
    image = query_image.detach().cpu().float()[None]
    image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)[0]
    return torch.clamp(image, 0.0, 1.0)


def _tensor_to_pil(image):
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _residual_to_pil(rendered, query):
    diff = torch.mean(torch.abs(rendered - query), dim=0, keepdim=True).repeat(3, 1, 1)
    diff = torch.clamp(diff * 4.0, 0.0, 1.0)
    return _tensor_to_pil(diff)


def _project_points(points3d, pose_w2c, K, height, width):
    points = np.asarray(points3d, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    xyz_h = np.concatenate([points, ones], axis=1)
    cam = (np.asarray(pose_w2c, dtype=np.float64) @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-8
    uv_h = (np.asarray(K, dtype=np.float64) @ cam.T).T
    uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-8)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid


def _draw_sparse_matches(query_image, debug, camera, max_matches):
    height, width = int(debug["height"]), int(debug["width"])
    resized = _resize_query(query_image, height, width)
    image = _tensor_to_pil(resized)
    draw = ImageDraw.Draw(image)
    query_p2d = np.asarray(debug["query_p2d"], dtype=np.float32)
    points3d = np.asarray(debug["points3d"], dtype=np.float32)
    correct = np.asarray(debug["correct_mask"], dtype=bool)
    inlier = np.asarray(debug["inlier_mask"], dtype=bool)
    scores = debug.get("scores")
    if scores is not None:
        order = np.argsort(np.asarray(scores).reshape(-1))[::-1]
    else:
        order = np.arange(query_p2d.shape[0])
    order = order[: int(max_matches)]
    K = get_intrinsic(camera.FoVx, camera.FoVy, width, height)
    gt_pose = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    uv_gt, valid_gt = _project_points(points3d, gt_pose, K, height, width)

    for idx in order:
        if idx >= query_p2d.shape[0] or not valid_gt[idx]:
            continue
        qx, qy = query_p2d[idx]
        gx, gy = uv_gt[idx]
        color = (30, 210, 80) if correct[idx] else (235, 60, 45)
        width_px = 2 if inlier[idx] else 1
        draw.line((float(qx), float(qy), float(gx), float(gy)), fill=color, width=width_px)
        radius = 3 if inlier[idx] else 2
        draw.ellipse((qx - radius, qy - radius, qx + radius, qy + radius), outline=color, width=width_px)
        draw.rectangle((gx - 2, gy - 2, gx + 2, gy + 2), outline=(255, 255, 255), width=1)
    return image


def _fit_panel(image, size):
    panel = Image.new("RGB", size, (238, 238, 238))
    image = image.convert("RGB")
    image.thumbnail(size)
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def _write_sample_sheet(path, record, query_image, sparse_overlay, sparse_render, dense_render):
    panel_size = (300, 170)
    query = _tensor_to_pil(_resize_query(query_image, sparse_render.shape[-2], sparse_render.shape[-1]))
    sparse_residual = _residual_to_pil(sparse_render, _resize_query(query_image, sparse_render.shape[-2], sparse_render.shape[-1]))
    dense_residual = _residual_to_pil(dense_render, _resize_query(query_image, dense_render.shape[-2], dense_render.shape[-1]))
    panels = [
        ("query RGB", query),
        ("sparse match", sparse_overlay),
        ("sparse render", _tensor_to_pil(sparse_render)),
        ("sparse residual", sparse_residual),
        ("dense render", _tensor_to_pil(dense_render)),
        ("dense residual", dense_residual),
    ]
    sheet = Image.new("RGB", (panel_size[0] * 3, panel_size[1] * 2 + 72), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{record['image_name']} {record['failure_stage']} "
        f"S_TE={record['sparse_te']:.3g} D_TE={record['dense_te']:.3g} "
        f"d={record['dense_delta_te']:.3g}"
    )
    draw.text((10, 8), title, fill=(10, 10, 10))
    for idx, (label, image) in enumerate(panels):
        x = (idx % 3) * panel_size[0]
        y = 36 + (idx // 3) * panel_size[1]
        draw.text((x + 8, y + 4), label, fill=(20, 20, 20))
        sheet.paste(_fit_panel(image, (panel_size[0] - 10, panel_size[1] - 28)), (x + 5, y + 24))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _jsonable_result(row):
    return {
        key: _to_jsonable(value)
        for key, value in row.items()
        if key not in {"sparse_match_debug"}
    }


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(nested) for nested in value]
    return value


def _record_from_localization(image_name, loc_res, gt_w2c, sparse_diag):
    sparse_ae, sparse_te = cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
    dense_final = loc_res["dense"][-1] if loc_res.get("dense") else loc_res["sparse"]
    dense_ae, dense_te = cal_pose_error(dense_final["pose_w2c"], gt_w2c)
    return {
        "image_name": image_name,
        "sparse": {"pose_w2c": loc_res["sparse"]["pose_w2c"], "inliers": loc_res["sparse"]["inliers"]},
        "dense": loc_res.get("dense", []),
        "sparse_AE": float(sparse_ae),
        "sparse_TE": float(sparse_te),
        "dense_AE": float(dense_ae),
        "dense_TE": float(dense_te),
        "matches": sparse_diag.get("matches", 0),
        "correct_matches": sparse_diag.get("correct_matches", 0),
        "diagnostic_inliers": sparse_diag.get("inliers", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose STDLoc sparse-vs-dense teacher stages on selected query images.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_names", nargs="*", default=[])
    parser.add_argument("--image_names_file", default=None)
    parser.add_argument("--sample_flow_csv", default=None)
    parser.add_argument("--sample_flow_groups", nargs="+", default=["final_worst", "final_regressed"])
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--render_longest_edge", type=int, default=640)
    parser.add_argument("--max_match_draw", type=int, default=180)
    parser.add_argument("--sparse_bad_te", type=float, default=20.0)
    parser.add_argument("--good_te", type=float, default=5.0)
    parser.add_argument("--dense_worse_margin", type=float, default=5.0)
    parser.add_argument("--sparse_correct_rate_bad", type=float, default=0.05)
    parser.add_argument("--min_sparse_inliers", type=int, default=16)
    args = get_combined_args(parser)
    args.eval = args.split == "test"

    image_names = _read_image_names(args)
    if not image_names:
        raise ValueError("No image names provided. Use --image_names, --image_names_file, or --sample_flow_csv.")

    dataset = model.extract(args)
    gaussians = _load_gaussians(dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=False)
    config = _load_config(dataset, args.cfg)
    stdloc = STDLoc(gaussians, config)
    cameras = _camera_map(scene, args.split)
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    sparse_diag_by_image = {}
    for image_name in tqdm(image_names, desc="Teacher-stage diagnostics"):
        camera = cameras.get(image_name)
        if camera is None:
            print(f"[skip] Missing camera in split={args.split}: {image_name}")
            continue
        gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        query_image = camera.original_image.cuda()
        fine_feature_map, _ = stdloc.get_feature_map(query_image)
        sparse_diag = _diagnose_sparse(stdloc, fine_feature_map, camera)
        sparse_diag_by_image[image_name] = {
            "image_name": image_name,
            "matches": sparse_diag.get("matches", 0),
            "correct_matches": sparse_diag.get("correct_matches", 0),
            "inliers": sparse_diag.get("inliers", 0),
        }
        loc_res = stdloc.localize(query_image, camera.FoVx, camera.FoVy)
        row = _record_from_localization(image_name, loc_res, gt_w2c, sparse_diag)
        result_rows.append(row)

        dense_final = loc_res["dense"][-1] if loc_res.get("dense") else loc_res["sparse"]
        sparse_render = _render_rgb(gaussians, config, loc_res["sparse"]["pose_w2c"], camera, args.render_longest_edge)
        dense_render = _render_rgb(gaussians, config, dense_final["pose_w2c"], camera, args.render_longest_edge)
        debug = sparse_diag.get("sparse_match_debug")
        if debug is None:
            sparse_overlay = _tensor_to_pil(_resize_query(query_image, sparse_render.shape[-2], sparse_render.shape[-1]))
        else:
            sparse_overlay = _draw_sparse_matches(query_image, debug, camera, args.max_match_draw)

        stage_record = build_teacher_stage_records(
            [row],
            sparse_diag_by_image,
            sparse_bad_te=args.sparse_bad_te,
            good_te=args.good_te,
            dense_worse_margin=args.dense_worse_margin,
            sparse_correct_rate_bad=args.sparse_correct_rate_bad,
            min_sparse_inliers=args.min_sparse_inliers,
        )[0]
        safe_name = image_name.replace("/", "__").replace(".png", "")
        _write_sample_sheet(sample_dir / f"{safe_name}.png", stage_record, query_image.detach().cpu(), sparse_overlay, sparse_render, dense_render)

    stage_records = build_teacher_stage_records(
        result_rows,
        sparse_diag_by_image,
        sparse_bad_te=args.sparse_bad_te,
        good_te=args.good_te,
        dense_worse_margin=args.dense_worse_margin,
        sparse_correct_rate_bad=args.sparse_correct_rate_bad,
        min_sparse_inliers=args.min_sparse_inliers,
    )
    summary = summarize_stage_records(stage_records)
    payload = {
        "summary": summary,
        "image_names": image_names,
        "cfg": args.cfg,
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "split": args.split,
        "records": stage_records,
        "raw_results": [_jsonable_result(row) for row in result_rows],
    }
    write_stage_csv(output_dir / "teacher_stage.csv", stage_records)
    (output_dir / "teacher_stage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "sparse_match_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "matches", "correct_matches", "inliers"])
        writer.writeheader()
        writer.writerows(sparse_diag_by_image.values())
    print(json.dumps(summary, indent=2))
    print(f"Saved teacher-stage diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
