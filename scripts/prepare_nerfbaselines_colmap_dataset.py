#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from scene.colmap_loader import read_extrinsics_binary


def _remove_path(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _safe_symlink(src, dst, force=False):
    src = Path(src).resolve()
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src:
            return
        if not force:
            raise FileExistsError(f"Refusing to replace existing path: {dst}")
        _remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=src.is_dir())


def _target_image_size(size, image_downscale_factor=1.0, max_image_width=0):
    width, height = int(size[0]), int(size[1])
    image_downscale_factor = float(image_downscale_factor or 1.0)
    max_image_width = int(max_image_width or 0)
    uses_factor = image_downscale_factor != 1.0
    uses_width = max_image_width > 0
    if uses_factor and uses_width:
        raise ValueError("--image_downscale_factor and --max_image_width are mutually exclusive.")
    if image_downscale_factor <= 0:
        raise ValueError("--image_downscale_factor must be positive.")
    if max_image_width < 0:
        raise ValueError("--max_image_width must be non-negative.")
    if uses_factor:
        return max(1, int(round(width / image_downscale_factor))), max(
            1, int(round(height / image_downscale_factor))
        )
    if uses_width and width > max_image_width:
        scale = float(max_image_width) / float(width)
        return max(1, max_image_width), max(1, int(round(float(height) * scale)))
    return width, height


def _top_level_components(image_names):
    components = set()
    flat_names = []
    for name in image_names:
        parts = Path(name).parts
        if len(parts) <= 1:
            flat_names.append(name)
        else:
            components.add(parts[0])
    return sorted(components), flat_names


def _read_cambridge_split(path):
    path = Path(path)
    if not path.exists():
        return []
    names = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual Landmark") or line.startswith("ImageFile"):
                continue
            name = line.split()[0]
            if "/" in name:
                names.append(name)
    return names


def _stage_resized_images(
    image_names,
    images_source,
    images_out,
    image_downscale_factor=1.0,
    max_image_width=0,
    force=False,
):
    from PIL import Image

    images_source = Path(images_source)
    images_out = Path(images_out)
    if images_out.exists() or images_out.is_symlink():
        if not force:
            raise FileExistsError(f"Refusing to replace existing image directory: {images_out}")
        _remove_path(images_out)
    images_out.mkdir(parents=True, exist_ok=True)

    resized = 0
    copied = 0
    first_original_size = None
    first_staged_size = None
    for name in image_names:
        src = images_source / name
        dst = images_out / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source image for resize staging: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            original_size = image.size
            target_size = _target_image_size(
                original_size,
                image_downscale_factor=image_downscale_factor,
                max_image_width=max_image_width,
            )
            if first_original_size is None:
                first_original_size = list(original_size)
                first_staged_size = list(target_size)
            if target_size == original_size:
                shutil.copy2(src, dst)
                copied += 1
                continue
            image = image.convert("RGB")
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image.save(dst)
            resized += 1
    return {
        "resized_image_count": resized,
        "copied_image_count": copied,
        "first_original_size": first_original_size,
        "first_staged_size": first_staged_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a COLMAP dataset layout accepted by NerfBaselines."
    )
    parser.add_argument("--source_path", required=True, help="Original STDLoc/Cambridge scene path.")
    parser.add_argument("--output", required=True, help="Output staging COLMAP dataset path.")
    parser.add_argument(
        "--images_source",
        default=".",
        help="Image root inside source_path. Use '.' for original seq*/ images or 'processed'.",
    )
    parser.add_argument(
        "--no_split_lists",
        action="store_true",
        help="Do not write NerfBaselines train_list.txt/test_list.txt from Cambridge dataset split files.",
    )
    parser.add_argument(
        "--image_downscale_factor",
        type=float,
        default=1.0,
        help="If >1, materialize resized images at original_size / factor instead of symlinking images.",
    )
    parser.add_argument(
        "--max_image_width",
        type=int,
        default=0,
        help="If >0, materialize resized images capped to this width, preserving aspect ratio.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing symlinks/directories in output.")
    args = parser.parse_args()

    source = Path(args.source_path).resolve()
    output = Path(args.output).resolve()
    images_source = (source / args.images_source).resolve()
    sparse_source = source / "sparse"
    images_bin = source / "sparse" / "0" / "images.bin"

    if not images_bin.exists():
        raise FileNotFoundError(f"Missing COLMAP images.bin: {images_bin}")
    if not images_source.exists():
        raise FileNotFoundError(f"Missing image source root: {images_source}")

    output.mkdir(parents=True, exist_ok=True)
    _safe_symlink(sparse_source, output / "sparse", force=args.force)

    extrinsics = read_extrinsics_binary(str(images_bin))
    image_names = [item.name for item in extrinsics.values()]
    components, flat_names = _top_level_components(image_names)

    images_out = output / "images"
    resize_requested = float(args.image_downscale_factor or 1.0) != 1.0 or int(args.max_image_width or 0) > 0
    resize_manifest = {
        "resized_image_count": 0,
        "copied_image_count": 0,
        "first_original_size": None,
        "first_staged_size": None,
    }
    if resize_requested:
        resize_manifest = _stage_resized_images(
            image_names,
            images_source,
            images_out,
            image_downscale_factor=args.image_downscale_factor,
            max_image_width=args.max_image_width,
            force=args.force,
        )
    else:
        images_out.mkdir(parents=True, exist_ok=True)
        for component in components:
            _safe_symlink(images_source / component, images_out / component, force=args.force)
        for name in flat_names:
            _safe_symlink(images_source / name, images_out / name, force=args.force)

    missing = [name for name in image_names if not (images_out / name).exists()]
    if missing:
        sample = ", ".join(missing[:8])
        raise FileNotFoundError(f"{len(missing)} COLMAP images are missing under {images_out}: {sample}")

    train_names = []
    test_names = []
    if not args.no_split_lists:
        image_name_set = set(image_names)
        train_names = _read_cambridge_split(source / "dataset_train.txt")
        test_names = _read_cambridge_split(source / "dataset_test.txt")
        missing_train = [name for name in train_names if name not in image_name_set]
        missing_test = [name for name in test_names if name not in image_name_set]
        if missing_train or missing_test:
            sample = ", ".join((missing_train + missing_test)[:8])
            raise FileNotFoundError(f"Split list references images missing from COLMAP images.bin: {sample}")
        if train_names:
            (output / "train_list.txt").write_text("\n".join(train_names) + "\n")
        if test_names:
            (output / "test_list.txt").write_text("\n".join(test_names) + "\n")

    manifest = {
        "source_path": str(source),
        "output": str(output),
        "images_source": str(images_source),
        "sparse": str(sparse_source),
        "image_count": len(image_names),
        "linked_components": components,
        "flat_image_count": len(flat_names),
        "image_downscale_factor": float(args.image_downscale_factor or 1.0),
        "max_image_width": int(args.max_image_width or 0),
        "resize_requested": bool(resize_requested),
        **resize_manifest,
        "train_split_count": len(train_names),
        "test_split_count": len(test_names),
    }
    manifest_path = output / "nerfbaselines_dataset_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote NerfBaselines COLMAP dataset: {output}")
    print(
        f"image_count={len(image_names)} train_split={len(train_names)} "
        f"test_split={len(test_names)} linked_components={','.join(components) or '<flat>'}"
    )
    if resize_requested:
        print(
            f"resized={resize_manifest['resized_image_count']} copied={resize_manifest['copied_image_count']} "
            f"first_size={resize_manifest['first_original_size']}->{resize_manifest['first_staged_size']}"
        )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
