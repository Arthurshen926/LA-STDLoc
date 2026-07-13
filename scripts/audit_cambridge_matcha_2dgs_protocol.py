#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary


SCENE_RUNS = {
    "GreatCourt": "GreatCourt_n20_long_masked_retrain_retry",
    "KingsCollege": "KingsCollege_n20_long_masked_retrain",
    "OldHospital": "OldHospital_n20_long_masked_retrain_retry",
    "ShopFacade": "ShopFacade_n20_long_masked_retrain",
    "StMarysChurch": "StMarysChurch_n20_long_masked_retrain",
}


def normalize_image_name(value):
    value = str(value).replace("\\", "/")
    name = Path(value).name
    stem = Path(name).stem
    if "__" in stem:
        sequence, frame = stem.split("__", 1)
        return f"{sequence}/{frame}.png"
    if value.startswith("seq") and "/" in value:
        return value if value.endswith(".png") else value + ".png"
    if not value.endswith(".png"):
        value += ".png"
    return value


def read_dataset_positions(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        fields = line.strip().split()
        if not fields or not fields[0].startswith("seq"):
            continue
        rows[normalize_image_name(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
    return rows


def read_colmap_cameras(sparse_dir):
    cameras = {}
    for image in read_extrinsics_binary(os.fspath(Path(sparse_dir) / "images.bin")).values():
        rotation_w2c = qvec2rotmat(image.qvec)
        rotation_c2w = rotation_w2c.T
        position = -rotation_c2w @ np.asarray(image.tvec, dtype=np.float64)
        cameras[normalize_image_name(image.name)] = {
            "position": position,
            "rotation": rotation_c2w,
        }
    return cameras


def read_ply_header(path):
    data = Path(path).read_bytes()[:65536]
    marker = b"end_header"
    if marker not in data:
        raise ValueError(f"PLY header is longer than 64 KiB: {path}")
    text = data.split(marker, 1)[0].decode("ascii", errors="strict")
    properties = []
    vertex_count = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        if len(fields) == 3 and fields[0] == "property":
            properties.append(fields[2])
    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    return vertex_count, properties


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_error_deg(left, right):
    relative = np.asarray(left).T @ np.asarray(right)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def audit_scene(scene, run_dir, data_root, matcha_sfm_root):
    model_path = Path(run_dir) / "free_gaussians"
    ply_path = model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    vertex_count, properties = read_ply_header(ply_path)
    rgb_metrics_path = model_path / "train" / "ours_30000" / "rgb_metrics.json"
    rgb_metrics = json.loads(rgb_metrics_path.read_text()) if rgb_metrics_path.is_file() else {}
    output_cameras = json.loads((model_path / "cameras.json").read_text())
    output_by_name = {
        normalize_image_name(row["img_name"]): row for row in output_cameras
    }
    selected_dir = Path(matcha_sfm_root) / f"{scene}_n20_long_masked" / "mast3r_sfm" / "images"
    selected_names = sorted(normalize_image_name(path.name) for path in selected_dir.glob("*.png"))
    selected_colmap_cameras = read_colmap_cameras(selected_dir.parent / "sparse" / "0")
    dataset_dir = Path(data_root) / scene
    dataset_positions = read_dataset_positions(dataset_dir / "dataset_train.txt")
    colmap_cameras = read_colmap_cameras(dataset_dir / "sparse" / "0")
    usable_training_names = set(dataset_positions) & set(colmap_cameras)

    selected_position_errors = []
    selected_rotation_errors = []
    selected_missing = []
    for name in selected_names:
        row = selected_colmap_cameras.get(name)
        gt = colmap_cameras.get(name)
        if row is None or gt is None or name not in dataset_positions:
            selected_missing.append(name)
            continue
        selected_position_errors.append(
            np.linalg.norm(row["position"] - dataset_positions[name])
        )
        selected_rotation_errors.append(
            rotation_error_deg(row["rotation"], gt["rotation"])
        )

    all_position_errors = []
    all_rotation_errors = []
    for name, row in output_by_name.items():
        gt = colmap_cameras.get(name)
        position = dataset_positions.get(name)
        if gt is None or position is None:
            continue
        all_position_errors.append(
            np.linalg.norm(np.asarray(row["position"], dtype=np.float64) - position)
        )
        all_rotation_errors.append(
            rotation_error_deg(np.asarray(row["rotation"], dtype=np.float64), gt["rotation"])
        )

    scale_properties = sorted(name for name in properties if name.startswith("scale_"))
    loc_properties = sorted(name for name in properties if name.startswith("loc_"))
    return {
        "scene": scene,
        "model_path": os.path.abspath(model_path),
        "point_cloud": os.path.abspath(ply_path),
        "point_cloud_bytes": ply_path.stat().st_size,
        "point_cloud_sha256": sha256_file(ply_path),
        "vertex_count": vertex_count,
        "property_count": len(properties),
        "scale_properties": scale_properties,
        "has_mip_filter": "mip_filter" in properties,
        "loc_feature_count": len(loc_properties),
        "is_native_2dgs_schema": scale_properties == ["scale_0", "scale_1"],
        "selected_camera_count": len(selected_names),
        "dataset_training_camera_count": len(dataset_positions),
        "full_training_camera_count": len(usable_training_names),
        "selected_training_fraction": (
            len(selected_names) / len(usable_training_names)
            if usable_training_names
            else 0.0
        ),
        "output_camera_count": len(output_cameras),
        "selected_missing": selected_missing,
        "selected_position_error_m": quantiles(selected_position_errors),
        "selected_rotation_error_deg": quantiles(selected_rotation_errors),
        "all_output_position_error_m": quantiles(all_position_errors),
        "all_output_rotation_error_deg": quantiles(all_rotation_errors),
        "rgb_geometry_only": len(loc_properties) == 0,
        "rgb_evaluation": {
            "path": os.path.abspath(rgb_metrics_path),
            "num_images": rgb_metrics.get("num_images"),
            **rgb_metrics.get("mean", {}),
        },
        "can_seed_fixed_geometry_baseline": True,
        "uses_full_cambridge_training_split": (
            set(selected_names) == usable_training_names
        ),
        "strict_from_sfm_iteration0_equivalent": False,
        "protocol_role": "external_rgb_2dgs_fixed_geometry_baseline",
    }


def ensure_symlink(link, target):
    link = Path(link)
    target = Path(target).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Symlink target does not exist: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            if not link.exists():
                link.unlink()
                link.symlink_to(target, target_is_directory=target.is_dir())
                return
            raise FileExistsError(f"Existing symlink points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_wrapper(scene, scene_summary, wrapper_root, data_root):
    source_model = Path(scene_summary["model_path"])
    model_path = Path(wrapper_root) / scene
    model_path.mkdir(parents=True, exist_ok=True)
    point_cloud_root = model_path / "point_cloud"
    point_cloud_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(
        point_cloud_root / "iteration_30000",
        source_model / "point_cloud" / "iteration_30000",
    )
    if (source_model / "input.ply").exists():
        ensure_symlink(model_path / "input.ply", source_model / "input.ply")
    cfg = (
        "Namespace(sh_degree=3, "
        f"source_path={os.fspath(Path(data_root) / scene)!r}, "
        "feature_type='sp', gaussian_type='2dgs', "
        f"model_path={os.fspath(model_path.resolve())!r}, images='processed', "
        "resolution=1, white_background=True, longest_edge=640, "
        "data_device='cpu', eval=False, speedup=False, norm_before_render=True, "
        "render_items=['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map'])"
    )
    (model_path / "cfg_args").write_text(cfg)
    provenance = {
        "scene": scene,
        "role": "matcha_rgb_2dgs_geometry_wrapper",
        "source_model_path": scene_summary["model_path"],
        "source_point_cloud": scene_summary["point_cloud"],
        "source_point_cloud_sha256": scene_summary["point_cloud_sha256"],
        "source_iteration": 30000,
        "rgb_geometry_only": scene_summary["rgb_geometry_only"],
        "protocol_role": scene_summary["protocol_role"],
        "selected_camera_count": scene_summary["selected_camera_count"],
        "full_training_camera_count": scene_summary["full_training_camera_count"],
        "uses_full_cambridge_training_split": scene_summary[
            "uses_full_cambridge_training_split"
        ],
        "strict_from_sfm_iteration0_equivalent": scene_summary[
            "strict_from_sfm_iteration0_equivalent"
        ],
        "localization_protocol_source": os.fspath(Path(data_root) / scene),
    }
    (model_path / "artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True)
    )
    return os.path.abspath(model_path)


def markdown_report(summary):
    lines = [
        "# MAtCha Cambridge 2DGS Protocol Audit",
        "",
        "| Scene | Vertices | Native 2DGS | loc dims | Selected/full cams | Full split | RGB PSNR | RGB SSIM | Position max (m) | Rotation max (deg) | Output camera anomaly max (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in summary["scenes"]:
        row = summary["scenes"][scene]
        lines.append(
            f"| {scene} | {row['vertex_count']} | {row['is_native_2dgs_schema']} | "
            f"{row['loc_feature_count']} | {row['selected_camera_count']}/"
            f"{row['full_training_camera_count']} | "
            f"{row['uses_full_cambridge_training_split']} | "
            f"{row.get('rgb_evaluation', {}).get('psnr', float('nan')):.4f} | "
            f"{row.get('rgb_evaluation', {}).get('ssim', float('nan')):.4f} | "
            f"{row['selected_position_error_m'].get('max', float('nan')):.6g} | "
            f"{row['selected_rotation_error_deg'].get('max', float('nan')):.6g} | "
            f"{row['all_output_position_error_m'].get('max', float('nan')):.6g} |"
        )
    lines.extend(
        [
            "",
            "The checkpoints use a native 2DGS surfel schema (`scale_0`, `scale_1`) and calibrated Cambridge coordinates. They are RGB/geometry-only checkpoints: no `loc_*` descriptor field or STDLoc detector is present. A localization experiment must therefore train or initialize those artifacts before evaluation.",
            "",
            "These MAtCha runs use an `n20` camera subset rather than the full Cambridge training split. They are valid fixed-geometry native-2DGS baselines, but they are not strict replacements for the LaFGS main line that starts from sparse SfM points at iteration 0 and trains on the full split.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", required=True, type=Path)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc", type=Path)
    parser.add_argument("--matcha_sfm_root", default="/root/MAtCha/output_cambridge/runs", type=Path)
    parser.add_argument("--scenes", nargs="*", default=list(SCENE_RUNS))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    parser.add_argument("--prepare_wrapper_root", type=Path)
    args = parser.parse_args()

    scenes = {}
    for scene in args.scenes:
        if scene not in SCENE_RUNS:
            raise ValueError(f"Unsupported Cambridge scene: {scene}")
        run_dir = args.runs_root / SCENE_RUNS[scene]
        scenes[scene] = audit_scene(scene, run_dir, args.data_root, args.matcha_sfm_root)
    summary = {
        "runs_root": os.path.abspath(args.runs_root),
        "data_root": os.path.abspath(args.data_root),
        "scenes": scenes,
    }
    if args.prepare_wrapper_root:
        summary["wrappers"] = {
            scene: prepare_wrapper(scene, row, args.prepare_wrapper_root, args.data_root)
            for scene, row in scenes.items()
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.output_markdown.write_text(markdown_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
