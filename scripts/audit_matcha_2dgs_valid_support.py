#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.audit_valid_support_masks import audit_images


SCENE_RE = re.compile(r"^(?P<scene>.+?)_n\d+_")


def discover_matcha_scene_runs(runs_root, scenes=None):
    runs_root = Path(runs_root)
    allowed = {str(item) for item in scenes} if scenes else None
    found = {}
    for child in sorted(runs_root.iterdir() if runs_root.exists() else []):
        if not child.is_dir():
            continue
        match = SCENE_RE.match(child.name)
        scene = match.group("scene") if match else child.name.split("_")[0]
        if allowed is not None and scene not in allowed:
            continue
        model_path = child / "free_gaussians"
        if (model_path / "cameras.json").exists():
            found[scene] = model_path
    return found


def focal_to_fov(focal, pixels):
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def matcha_camera_rows_to_render_records(rows, scene, max_records=0):
    sorted_rows = sorted(list(rows), key=lambda row: str(row.get("img_name", row.get("id", ""))))
    if int(max_records or 0) > 0 and len(sorted_rows) > int(max_records):
        indices = np.linspace(0, len(sorted_rows) - 1, int(max_records)).round().astype(int).tolist()
        sorted_rows = [sorted_rows[idx] for idx in indices]
    records = []
    for index, row in enumerate(sorted_rows):
        width = int(row["width"])
        height = int(row["height"])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.asarray(row["rotation"], dtype=np.float64).reshape(3, 3)
        c2w[:3, 3] = np.asarray(row["position"], dtype=np.float64).reshape(3)
        pose_w2c = np.linalg.inv(c2w).astype(np.float32)
        image_name = str(row.get("img_name", f"camera_{index:06d}"))
        records.append(
            {
                "index": int(index),
                "query_id": f"matcha_2dgs:{scene}:{image_name}",
                "image_name": image_name,
                "image_path": "",
                "pose_w2c": pose_w2c.tolist(),
                "fovx": float(focal_to_fov(row["fx"], width)),
                "fovy": float(focal_to_fov(row["fy"], height)),
                "width": width,
                "height": height,
            }
        )
    return records


def _write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _load_cameras_json(model_path):
    return json.loads((Path(model_path) / "cameras.json").read_text())


def matcha_subprocess_env(matcha_python, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    python_path = Path(matcha_python or sys.executable)
    conda_lib = python_path.parent.parent / "lib"
    if conda_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [os.fspath(conda_lib)]
        if existing:
            parts.append(existing)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def render_matcha_records(
    records,
    model_path,
    output_dir,
    matcha_root,
    matcha_python,
    iteration=30000,
    resolution="",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "render_records.jsonl"
    summary_path = output_dir / "render_summary.json"
    frames_dir = output_dir / "frames"
    _write_jsonl(records, manifest_path)
    command = [
        str(matcha_python or sys.executable),
        os.fspath(Path(__file__).resolve().parent / "render_matcha_records.py"),
        "--manifest",
        os.path.abspath(os.fspath(manifest_path)),
        "--model_path",
        os.path.abspath(os.fspath(model_path)),
        "--output_dir",
        os.path.abspath(os.fspath(frames_dir)),
        "--matcha_root",
        os.path.abspath(os.fspath(matcha_root)),
        "--iteration",
        str(int(iteration)),
        "--summary_json",
        os.path.abspath(os.fspath(summary_path)),
    ]
    if resolution:
        command.extend(["--resolution", str(resolution)])
    subprocess.run(command, check=True, env=matcha_subprocess_env(matcha_python))
    frame_paths = sorted(frames_dir.glob("*.png"))
    return {
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "frames_dir": str(frames_dir),
        "frame_paths": [str(path) for path in frame_paths],
    }


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def audit_matcha_2dgs(
    runs_root,
    output_dir,
    matcha_root,
    matcha_python=sys.executable,
    scenes=None,
    max_records_per_scene=0,
    iteration=30000,
    resolution="",
    image_scale=0.5,
    visual_max=24,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_matcha_scene_runs(runs_root, scenes=scenes)
    scene_summaries = {}
    for scene, model_path in runs.items():
        rows = _load_cameras_json(model_path)
        records = matcha_camera_rows_to_render_records(rows, scene=scene, max_records=max_records_per_scene)
        scene_dir = output_dir / scene
        render_info = render_matcha_records(
            records,
            model_path=model_path,
            output_dir=scene_dir / "render",
            matcha_root=matcha_root,
            matcha_python=matcha_python,
            iteration=iteration,
            resolution=resolution,
        )
        audit_summary = audit_images(
            render_info["frame_paths"],
            scene_dir / "valid_support_audit",
            image_scale=image_scale,
            visual_max=visual_max,
            sort_by="invalid_frac_desc",
        )
        scene_summaries[scene] = {
            "model_path": os.path.abspath(os.fspath(model_path)),
            "camera_count": len(rows),
            "rendered_count": len(render_info["frame_paths"]),
            "render": render_info,
            "audit": audit_summary,
        }
    summary = {
        "runs_root": os.path.abspath(os.fspath(runs_root)),
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "matcha_root": os.path.abspath(os.fspath(matcha_root)),
        "iteration": int(iteration),
        "resolution": str(resolution or ""),
        "scene_count": len(scene_summaries),
        "scenes": scene_summaries,
    }
    summary_path = output_dir / "matcha_2dgs_valid_support_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render MAtCha 2DGS checkpoints and audit no-reference valid/support masks.")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matcha_root", default=os.environ.get("MATCHA_ROOT", "/root/MAtCha"))
    parser.add_argument("--matcha_python", default=os.environ.get("MATCHA_PYTHON", sys.executable))
    parser.add_argument("--scenes", default="")
    parser.add_argument("--max_records_per_scene", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=int(os.environ.get("MATCHA_ITERATION", "30000")))
    parser.add_argument("--resolution", default=os.environ.get("MATCHA_RENDER_RESOLUTION", ""))
    parser.add_argument("--image_scale", type=float, default=0.5)
    parser.add_argument("--visual_max", type=int, default=24)
    args = parser.parse_args()
    summary = audit_matcha_2dgs(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        matcha_root=args.matcha_root,
        matcha_python=args.matcha_python,
        scenes=_comma_list(args.scenes),
        max_records_per_scene=args.max_records_per_scene,
        iteration=args.iteration,
        resolution=args.resolution,
        image_scale=args.image_scale,
        visual_max=args.visual_max,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
