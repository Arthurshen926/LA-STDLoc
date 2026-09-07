#!/usr/bin/env python3
"""Run released Long-LRM inference on a strict mapping-only prepared scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement


def _save_gaussian_ply_vectorized(
    unused_model,
    gaussian_dict,
    save_path,
    opacity_threshold=None,
) -> None:
    """Numerically equivalent replacement for the released tuple-based writer."""

    import torch  # noqa: PLC0415

    opacity_tensor = gaussian_dict["opacity"].detach().cpu().float().squeeze(-1)
    if opacity_threshold is None:
        keep = torch.ones_like(opacity_tensor, dtype=torch.bool)
    else:
        keep = opacity_tensor.sigmoid() > float(opacity_threshold)
    if not bool(keep.any()):
        raise ValueError("Long-LRM opacity threshold removed every primitive")
    device_keep = keep.to(gaussian_dict["xyz"].device)
    xyz = gaussian_dict["xyz"].detach()[device_keep].cpu().float().numpy()
    feature = gaussian_dict["feature"].detach()[device_keep].cpu().float().numpy()
    scales = gaussian_dict["scale"].detach()[device_keep].cpu().float().numpy()
    rotations = gaussian_dict["rotation"].detach()[device_keep].cpu().float().numpy()
    opacity = opacity_tensor[keep].cpu().numpy()
    rows = int(xyz.shape[0])
    if feature.shape[1:] != (16, 3):
        raise ValueError(f"expected degree-3 Long-LRM SH, got {feature.shape}")
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{index}" for index in range(3)]
    fields += [f"f_rest_{index}" for index in range(45)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2"]
    fields += [f"rot_{index}" for index in range(4)]
    dtype = [
        (name, "u1" if name in {"nx", "ny", "nz"} else "<f4")
        for name in fields
    ]
    vertices = np.empty(rows, dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0
    for index in range(3):
        vertices[f"f_dc_{index}"] = feature[:, 0, index]
    rest = feature[:, 1:].transpose(0, 2, 1).reshape(rows, 45)
    for index in range(45):
        vertices[f"f_rest_{index}"] = rest[:, index]
    vertices["opacity"] = opacity
    for index in range(3):
        vertices[f"scale_{index}"] = scales[:, index]
    for index in range(4):
        vertices[f"rot_{index}"] = rotations[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(save_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-lrm-root", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")

    root = Path(args.long_lrm_root).expanduser().resolve()
    prepared_root = Path(args.prepared).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = prepared_root / "anygsloc_long_lrm_input.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "anygsloc_long_lrm_prepared_mapping"
        or manifest.get("version") != 1
        or manifest.get("role") != "mapping_only"
        or manifest.get("uses_test_queries") is not False
        or manifest.get("view_count") != 32
    ):
        raise ValueError("invalid or non-mapping-only Long-LRM prepared input")
    camera_json = Path(manifest["camera_json"])
    if _sha256(camera_json) != manifest["camera_json_sha256"]:
        raise ValueError("prepared Long-LRM camera JSON hash mismatch")
    for row in manifest["records"]:
        image_path = prepared_root / "images" / row["prepared_name"]
        if _sha256(image_path) != row["prepared_image_sha256"]:
            raise ValueError(f"prepared image changed: {image_path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Long-LRM output: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(root))
    import torch  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    from model.llrm import LongLRM  # noqa: PLC0415

    original_forward = LongLRM.forward
    forward_seconds = []

    def timed_forward(model, *forward_args, **forward_kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_forward(model, *forward_args, **forward_kwargs)
        torch.cuda.synchronize()
        forward_seconds.append(time.perf_counter() - start)
        return result

    LongLRM.forward = timed_forward

    # The public helper always emits a costly interpolation video.  It is not
    # an input to reconstruction, audit, or localization, so suppress it while
    # preserving the released model forward and opacity filtering semantics.
    LongLRM.save_input_video = lambda *unused_args, **unused_kwargs: None
    LongLRM.save_gaussian_ply = _save_gaussian_ply_vectorized
    from inference import get_gaussian_reconstruction  # noqa: PLC0415

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    started = time.perf_counter()
    try:
        result = get_gaussian_reconstruction(
            str(root),
            str(camera_json),
            str(temporary),
            str(checkpoint),
            height=540,
            fix_aspect_ratio=True,
            input_num=32,
            sample_method="uniform",
        )
        if not result or not (temporary / "gaussians.ply").is_file():
            raise RuntimeError("Long-LRM did not produce a Gaussian PLY")
        elapsed = time.perf_counter() - started
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        run = {
            "schema": "anygsloc_long_lrm_reconstruction",
            "version": 1,
            "source_method": "released_long_lrm_32view",
            "mapping_only": True,
            "uses_test_queries": False,
            "long_lrm_root": str(root),
            "long_lrm_commit": commit,
            "prepared_manifest": str(manifest_path),
            "prepared_manifest_sha256": _sha256(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "elapsed_seconds": elapsed,
            "seed": args.seed,
            "cpu_threads": args.cpu_threads,
            "synchronized_forward_seconds": sum(forward_seconds),
            "reported_model_forward_seconds": float(result["inference_time"]),
            "reported_model_forward_clock": "upstream_unsynchronized_wall_clock",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch_version": torch.__version__,
            "ply_writer": "vectorized_equivalent_to_released_opacity_threshold_writer",
            "normalized_ply": str(output / "gaussians.ply"),
            "normalized_ply_sha256": _sha256(temporary / "gaussians.ply"),
        }
        (temporary / "anygsloc_long_lrm_run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True)
        )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
