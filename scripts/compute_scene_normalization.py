#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import shlex

import numpy as np
from PIL import Image
from plyfile import PlyData

from localization_training.scene_normalization import compute_scene_normalization


def read_camera_positions(dataset_root):
    positions = []
    image_names = []
    for line in (Path(dataset_root) / "dataset_train.txt").read_text().splitlines():
        fields = line.strip().split()
        if len(fields) < 4 or not fields[0].startswith("seq"):
            continue
        image_names.append(fields[0])
        positions.append([float(value) for value in fields[1:4]])
    return np.asarray(positions, dtype=np.float64), image_names


def read_surfel_radii(point_cloud, max_samples=200000):
    ply = PlyData.read(point_cloud, mmap="r")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names)
    scale_names = sorted(
        (name for name in names if name.startswith("scale_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if len(scale_names) != 2:
        raise ValueError(
            f"Expected native 2DGS scale_0/scale_1 fields, got {scale_names}"
        )
    count = len(vertex)
    stride = max(1, int(np.ceil(count / int(max_samples))))
    log_scale = np.stack(
        [np.asarray(vertex[name])[::stride] for name in scale_names], axis=1
    )
    effective_radius = np.exp(np.mean(log_scale, axis=1))
    return count, effective_radius


def find_image_size(dataset_root, images_dir, image_names):
    for image_name in image_names:
        path = Path(dataset_root) / images_dir / image_name
        if path.is_file():
            with Image.open(path) as image:
                return image.size
    raise FileNotFoundError(
        f"No training image found under {Path(dataset_root) / images_dir}"
    )


def shell_name(name):
    return name.upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--point_cloud", required=True, type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--target_longest_edge", type=int, default=640)
    parser.add_argument("--field_steps", type=int, default=30000)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    positions, image_names = read_camera_positions(args.dataset_root)
    point_count, surfel_radii = read_surfel_radii(args.point_cloud)
    image_size = find_image_size(args.dataset_root, args.images, image_names)
    config = compute_scene_normalization(
        positions,
        point_count,
        surfel_radii,
        image_size,
        target_longest_edge=args.target_longest_edge,
        field_steps=args.field_steps,
    ).to_dict()
    config.update(
        {
            "dataset_root": str(args.dataset_root.resolve()),
            "point_cloud": str(args.point_cloud.resolve()),
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
        }
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if args.shell:
        for key, value in sorted(config.items()):
            print(f"{shell_name(key)}={shlex.quote(str(value))}")
    else:
        print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
