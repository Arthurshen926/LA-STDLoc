"""Indoor paper profile: alpha-gated extraction and the current Compact selector."""

import os
import subprocess
import sys
from pathlib import Path

from anygsloc.artifacts import export_map
from anygsloc.pipeline.io import complete


def build_indoor(config, scene, device):
    spec = config["scenes"][scene]
    root = Path(config["output"]) / scene
    build_root = root / "indoor_construction"
    if not device.startswith("cuda:"):
        raise ValueError("indoor rendering requires an explicit cuda:N device")
    # The isolated renderer subprocess uses cuda:0 in its visible-device namespace.
    index = int(device.split(":")[1])
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
    command = [
        sys.executable,
        "-m",
        "anygsloc",
        "build-map",
        "--image-domain",
        "rectified",
        "--gpu",
        env["CUDA_VISIBLE_DEVICES"],
        "--dataset",
        spec["dataset"],
        "--images",
        spec.get("images", "processed"),
        "--gaussian-ply",
        spec["prior"],
        "--gaussian-type",
        spec["gaussian_type"],
        "--sh-degree",
        str(spec["sh_degree"]),
        "--output",
        str(build_root),
    ]
    if spec.get("white_background"):
        command.append("--white-background")
    subprocess.run(command, env=env, check=True)
    cell = root / "pure_gs/filtered"
    cell.mkdir(parents=True, exist_ok=True)
    export_map(
        build_root / "full/projective_anchor_map.pt",
        build_root / "full/identity_metric.pt",
        cell / "full.pt",
    )
    target = cell / "selection/compact.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to((build_root / "compact/deployment_map.pt").resolve())
    complete(cell, ["full.pt", "selection/compact.pt"])
