#!/usr/bin/env python3
"""Create a zero-copy flat-image view for the official 2DGS loader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.prepare_cambridge_mapping_only_colmap import (
        read_images_binary,
        sha256,
        write_images_binary,
    )
except ModuleNotFoundError:
    from prepare_cambridge_mapping_only_colmap import (
        read_images_binary,
        sha256,
        write_images_binary,
    )


def _flat_name(name: str) -> str:
    return name.replace("\\", "/").replace("/", "__")


def stage_flat_image_view(*, source: Path, output: Path) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite flat COLMAP view: {output}")
    parent_manifest_path = source / "mapping_only_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if parent_manifest.get("semantic_mask_used") is not False:
        raise ValueError("flat view requires a mask-free mapping-only parent")
    if parent_manifest.get("undistortion_used") is not True:
        raise ValueError("flat view requires an undistorted parent")

    sparse = source / "sparse" / "0"
    images = read_images_binary(sparse / "images.bin")
    flattened = {image.name: _flat_name(image.name) for image in images.values()}
    if len(flattened.values()) != len(set(flattened.values())):
        raise ValueError("flattened COLMAP image names are not unique")

    output_sparse = output / "sparse" / "0"
    output_images = output / "images"
    output_sparse.mkdir(parents=True)
    output_images.mkdir(parents=True)
    rewritten = {
        image_id: image._replace(name=flattened[image.name])
        for image_id, image in images.items()
    }
    write_images_binary(rewritten, output_sparse / "images.bin")
    for filename in ("cameras.bin", "points3D.bin", "points3D.ply"):
        source_path = sparse / filename
        if source_path.exists():
            os.symlink(source_path, output_sparse / filename)
    for source_name, target_name in flattened.items():
        source_image = source / "images" / source_name
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        os.symlink(source_image, output_images / target_name)

    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema": "lafgs_off_the_shelf_flat_colmap_view",
            "version": 1,
            "parent_mapping_dataset": str(source),
            "parent_mapping_manifest_sha256": sha256(parent_manifest_path),
            "image_name_policy": "replace path separators with double underscores",
            "image_pixels_copied": False,
            "image_pixels_changed": False,
            "flattened_image_count": len(rewritten),
        }
    )
    (output / "mapping_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "image_name_map.json").write_text(
        json.dumps(flattened, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = stage_flat_image_view(source=args.source, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
