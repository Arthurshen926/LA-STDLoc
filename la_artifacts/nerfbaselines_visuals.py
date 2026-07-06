import math
import tarfile
from pathlib import Path

from PIL import Image


def _image_from_tar(tar, name):
    member = tar.getmember(name)
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(name)
    with handle:
        return Image.open(handle).convert("RGB").copy()


def _prediction_pairs(tar):
    names = set(tar.getnames())
    gt_names = sorted(name for name in names if name.startswith("gt-color/") and name.lower().endswith(".png"))
    pairs = []
    for gt_name in gt_names:
        rel = gt_name[len("gt-color/") :]
        color_name = f"color/{rel}"
        if color_name in names:
            pairs.append((gt_name, color_name))
    return pairs


def build_predictions_grid(predictions_tar, output_path, sample_count=24, columns=4):
    predictions_tar = Path(predictions_tar)
    output_path = Path(output_path)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")

    with tarfile.open(predictions_tar, "r:gz") as tar:
        pairs = _prediction_pairs(tar)
        if not pairs:
            raise ValueError(f"No gt-color/color PNG pairs found in {predictions_tar}")
        selected = pairs[:sample_count]
        images = [(_image_from_tar(tar, gt), _image_from_tar(tar, color)) for gt, color in selected]

    tile_w, tile_h = images[0][0].size
    pair_w = tile_w * 2
    rows = int(math.ceil(len(images) / columns))
    grid = Image.new("RGB", (pair_w * columns, tile_h * rows), color=(255, 255, 255))

    for index, (gt, color) in enumerate(images):
        if gt.size != (tile_w, tile_h):
            gt = gt.resize((tile_w, tile_h), Image.BILINEAR)
        if color.size != (tile_w, tile_h):
            color = color.resize((tile_w, tile_h), Image.BILINEAR)
        x = (index % columns) * pair_w
        y = (index // columns) * tile_h
        grid.paste(gt, (x, y))
        grid.paste(color, (x + tile_w, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return {
        "pairs": len(selected),
        "available_pairs": len(pairs),
        "output_path": str(output_path),
        "width": grid.size[0],
        "height": grid.size[1],
    }
