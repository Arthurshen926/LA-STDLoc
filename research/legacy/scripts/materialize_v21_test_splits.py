#!/usr/bin/env python3
"""Materialize the frozen V21 real-test adaptation protocol split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from map_learning.v21_test_protocol import (
    build_test_protocol_manifest,
    validate_test_protocol_manifest,
)



_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


def _is_sequence_image_name(value: str) -> bool:
    path = PurePosixPath(str(value).replace("\\", "/"))
    return (
        len(path.parts) >= 2
        and path.name not in {"", ".", ".."}
        and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _test_registry(dataset_root: Path) -> tuple[Path, list[str]]:
    cambridge = dataset_root / "dataset_test.txt"
    scene_list = dataset_root / "sparse/0/list_test.txt"
    if cambridge.is_file():
        path = cambridge
        names = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8 or not _is_sequence_image_name(fields[0]):
                continue
            try:
                tuple(float(value) for value in fields[1:8])
            except ValueError:
                continue
            names.append(fields[0])
    elif scene_list.is_file():
        path = scene_list
        names = [
            line.strip()
            for line in path.read_text().splitlines()
            if _is_sequence_image_name(line.strip())
        ]
    else:
        raise FileNotFoundError("V21 requires dataset_test.txt or sparse/0/list_test.txt")
    if not names or len(set(names)) != len(names):
        raise ValueError("V21 test registry must be nonempty and duplicate-free")
    return path.resolve(), names


def _write_json_exclusive(payload: dict[str, Any], output: Path) -> None:
    """Publish JSON atomically while preserving the no-overwrite contract."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 split output already exists: {output}")
    dataset_root = Path(args.dataset).expanduser().resolve()
    registry_path, registry_names = _test_registry(dataset_root)
    stable_map = Path(args.stable_map).expanduser().resolve()
    if not stable_map.is_file():
        raise FileNotFoundError(f"missing V21 stable map: {stable_map}")

    dataset = ColmapDataset(dataset_root, images=str(args.images))
    cameras = dataset.split("test")
    camera_names = [str(camera.image_name).replace("\\", "/") for camera in cameras]
    normalized_registry_names = [name.replace("\\", "/") for name in registry_names]
    if len(camera_names) != len(normalized_registry_names) or set(camera_names) != set(
        normalized_registry_names
    ):
        raise ValueError("V21 loaded test cameras differ from the dataset test registry")

    manifest = build_test_protocol_manifest(
        cameras,
        dataset_root=dataset_root,
        images=str(args.images),
        dataset_registry_path=registry_path,
        stable_map_path=stable_map,
        block_size=int(args.block_size),
        embargo_frames=int(args.embargo_frames),
        minimum_confirmation_queries=int(args.minimum_confirmation_queries),
        minimum_confirmation_blocks=int(args.minimum_confirmation_blocks),
        require_image_content=True,
    )
    script_path = Path(__file__).resolve()
    manifest["producer"] = {
        "path": str(script_path),
        "sha256": sha256_file(script_path),
    }
    validate_test_protocol_manifest(manifest)
    _write_json_exclusive(manifest, output)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output": str(output),
                "counts": manifest["counts"],
                "stable_map_sha256": manifest["stable_map_sha256"],
                "ordered_test_camera_sha256": manifest["dataset_registry"][
                    "ordered_test_camera_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--embargo-frames", type=int, default=1)
    parser.add_argument("--minimum-confirmation-queries", type=int, default=160)
    parser.add_argument("--minimum-confirmation-blocks", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    materialize(_parser().parse_args())


if __name__ == "__main__":
    main()
