#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision


def _record_get(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_payload(record, index):
    return {
        "index": int(index),
        "query_id": str(_record_get(record, "query_id", "")),
        "image_name": str(_record_get(record, "image_name", "")),
        "image_path": str(_record_get(record, "image_path", "")),
        "pose_w2c": _record_get(record, "pose_w2c"),
        "fovx": float(_record_get(record, "fovx")),
        "fovy": float(_record_get(record, "fovy")),
        "width": int(_record_get(record, "width")),
        "height": int(_record_get(record, "height")),
    }


def _load_manifest_records(path):
    path = Path(path)
    text = path.read_text()
    stripped = text.lstrip()
    rows = []
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "records" in payload:
            return list(payload.get("records", []))
        if isinstance(payload, dict):
            return [payload]
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_resolution(value):
    value = str(value or "").strip().lower()
    if not value:
        return None
    parts = value.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must use WIDTHxHEIGHT format, got: {value}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got: {value}")
    return width, height


def _add_matcha_to_path(matcha_root):
    root = Path(matcha_root)
    gs_root = root / "2d-gaussian-splatting"
    if not gs_root.exists():
        raise FileNotFoundError(f"MAtCha 2DGS root not found: {gs_root}")
    sys.path.insert(0, os.fspath(gs_root))


def _make_minicam(record, resolution=None):
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    width = int(record["width"])
    height = int(record["height"])
    if resolution is not None:
        width, height = int(resolution[0]), int(resolution[1])
    fovx = float(record["fovx"])
    fovy = float(record["fovy"])
    znear = 0.01
    zfar = 100.0
    world_view_transform = torch.as_tensor(record["pose_w2c"], dtype=torch.float32, device="cuda").transpose(0, 1)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)


def render_records(manifest, model_path, output_dir, matcha_root="/root/MAtCha", iteration=30000, resolution=""):
    _add_matcha_to_path(matcha_root)
    from gaussian_renderer import GaussianModel, render

    records = [_record_payload(row, index) for index, row in enumerate(_load_manifest_records(manifest))]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution_hw = _parse_resolution(resolution)

    model_path = Path(model_path)
    point_cloud = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not point_cloud.exists():
        raise FileNotFoundError(f"MAtCha point cloud not found: {point_cloud}")

    gaussians = GaussianModel(3)
    gaussians.load_ply(os.fspath(point_cloud))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    summary = {
        "manifest": os.path.abspath(os.fspath(manifest)),
        "model_path": os.path.abspath(os.fspath(model_path)),
        "point_cloud": os.path.abspath(os.fspath(point_cloud)),
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "count": len(records),
        "iteration": int(iteration),
        "resolution": str(resolution or ""),
        "frames": [],
    }
    with torch.no_grad():
        for record in records:
            cam = _make_minicam(record, resolution=resolution_hw)
            pkg = render(cam, gaussians, pipe, background)
            frame_path = output_dir / f"{int(record['index']):06d}.png"
            torchvision.utils.save_image(pkg["render"].detach().clamp(0.0, 1.0), frame_path)
            summary["frames"].append(
                {
                    "index": int(record["index"]),
                    "query_id": record["query_id"],
                    "path": os.path.abspath(os.fspath(frame_path)),
                    "width": int(cam.image_width),
                    "height": int(cam.image_height),
                }
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render pseudo-query records with a MAtCha/2DGS model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matcha_root", default=os.environ.get("MATCHA_ROOT", "/root/MAtCha"))
    parser.add_argument("--iteration", type=int, default=int(os.environ.get("MATCHA_ITERATION", "30000")))
    parser.add_argument("--resolution", default=os.environ.get("MATCHA_RENDER_RESOLUTION", ""))
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    summary = render_records(
        args.manifest,
        args.model_path,
        args.output_dir,
        matcha_root=args.matcha_root,
        iteration=args.iteration,
        resolution=args.resolution,
    )
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
