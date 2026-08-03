#!/usr/bin/env python3
"""Select mapping-only, pose-diverse AnySplat feed-forward windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prior_reconstruction.anysplat import (
    colmap_qvec_to_rotation,
    select_trajectory_windows,
)
from scripts.prepare_cambridge_mapping_only_colmap import read_images_binary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-trajectory", type=int, default=24)
    parser.add_argument("--trajectory-segment-size", type=int, default=96)
    parser.add_argument("--complete-coverage", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    images = read_images_binary(dataset / "sparse" / "0" / "images.bin")
    by_name = {image.name: image for image in images.values()}
    mapping_names = [
        name.strip()
        for name in (dataset / "mapping_names.txt").read_text().splitlines()
        if name.strip()
    ]
    centers = {}
    for name in mapping_names:
        image = by_name[name]
        rotation_w2c = colmap_qvec_to_rotation(image.qvec)
        centers[name] = -(rotation_w2c.T @ image.tvec)
    windows = select_trajectory_windows(
        mapping_names,
        centers,
        args.views_per_trajectory,
        args.trajectory_segment_size,
        complete_coverage=args.complete_coverage,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    image_root = dataset / "images"
    for window in windows:
        window_dir = args.output / "images" / str(window["window_id"])
        window_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(window["image_names"]):
            source = (image_root / str(name)).resolve()
            destination = window_dir / f"{index:03d}_{Path(str(name)).name}"
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            os.symlink(source, destination)
        window["image_sha256"] = {
            name: _sha256((image_root / str(name)).resolve())
            for name in window["image_names"]
        }
    manifest = {
        "schema": "lafgs_anysplat_mapping_only_windows",
        "version": 1,
        "dataset": str(dataset),
        "selection": (
            "deterministic camera-center FPS within local trajectory segments"
        ),
        "views_per_trajectory": args.views_per_trajectory,
        "trajectory_segment_size": args.trajectory_segment_size,
        "complete_coverage": args.complete_coverage,
        "mapping_image_count": len(mapping_names),
        "selected_image_count": sum(int(w["selected_view_count"]) for w in windows),
        "trajectory_count": len(windows),
        "test_rgb_used": False,
        "windows": windows,
    }
    path = args.output / "windows_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
