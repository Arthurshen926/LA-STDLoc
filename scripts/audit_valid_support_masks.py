#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from valid_support_mask import (
    NoReferenceValidSupportMaskBuilder,
    NoReferenceValidSupportMaskConfig,
    save_mask_bundle_pngs,
)


def _open_rgb(path):
    return Image.open(path).convert("RGB")


def _image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _resize_image(image, scale):
    scale = float(scale or 1.0)
    if abs(scale - 1.0) < 1e-6:
        return image
    width = max(8, int(round(image.width * scale)))
    height = max(8, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _font(size=12):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(image, text):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0))
    draw.text((6, 4), str(text), fill=(255, 255, 255), font=_font(12))
    return out


def _fit_canvas(image, size, bg=(24, 24, 24)):
    canvas = Image.new("RGB", size, bg)
    work = image.convert("RGB").copy()
    work.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(work, ((size[0] - work.width) // 2, (size[1] - work.height) // 2))
    return canvas


def _heatmap01(score_map):
    score = torch.as_tensor(score_map, dtype=torch.float32).detach().cpu().clamp(0.0, 1.0).numpy()
    red = np.clip(2.0 * score, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(score - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - score), 0.0, 1.0)
    rgb = np.stack([red, green * 0.65, blue * 0.45], axis=-1)
    return Image.fromarray((rgb * 255.0).round().astype("uint8"))


def _overlay_mask(image, mask, color=(0, 215, 90), invalid_color=(220, 40, 35), alpha=0.36):
    base = image.convert("RGB")
    mask_img = Image.fromarray(torch.as_tensor(mask).detach().cpu().bool().numpy().astype(np.uint8) * 255, mode="L")
    mask_img = mask_img.resize(base.size, Image.Resampling.NEAREST)
    valid_layer = Image.new("RGB", base.size, color)
    invalid_layer = Image.new("RGB", base.size, invalid_color)
    overlay = Image.composite(valid_layer, invalid_layer, mask_img)
    return Image.blend(base, overlay, float(alpha))


def _topk_support_points(score_map, max_points=512, threshold=0.05, nms_radius=4):
    score = torch.as_tensor(score_map, dtype=torch.float32).detach().cpu()
    if score.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    radius = int(max(1, nms_radius))
    pooled = F.max_pool2d(score[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0]
    keep = (score >= pooled) & (score > float(threshold))
    ids = torch.nonzero(keep.reshape(-1), as_tuple=False).squeeze(1)
    if ids.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    values = score.reshape(-1)[ids]
    order = torch.topk(values, min(int(max_points), int(values.numel()))).indices
    ids = ids[order]
    width = int(score.shape[1])
    return torch.stack([ids % width, ids // width], dim=1).float()


def _draw_points(image, points_xy, valid_flags, support_flags, max_points=512):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    points = torch.as_tensor(points_xy, dtype=torch.float32).detach().cpu()
    valid = torch.as_tensor(valid_flags, dtype=torch.bool).detach().cpu()
    support = torch.as_tensor(support_flags, dtype=torch.bool).detach().cpu()
    if points.numel() == 0:
        return out
    for (x, y), is_valid, is_support in zip(points[:max_points].tolist(), valid[:max_points].tolist(), support[:max_points].tolist()):
        color = (240, 45, 35) if not is_valid else ((0, 240, 85) if is_support else (255, 205, 35))
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)
    return out


def _text_panel(lines, size, title=""):
    panel = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(panel)
    y = 8
    if title:
        draw.text((8, y), str(title), fill=(12, 12, 12), font=_font(15))
        y += 24
    max_chars = max(12, int((size[0] - 16) / 7.2))
    for line in lines:
        text = str(line)
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for chunk in chunks:
            draw.text((8, y), chunk, fill=(32, 32, 32), font=_font(12))
            y += 16
            if y > size[1] - 18:
                draw.text((8, size[1] - 18), "...", fill=(32, 32, 32), font=_font(12))
                return panel
    return panel


def _safe_id(path_or_id):
    text = str(path_or_id).replace("\\", "/").strip("/")
    return text.replace("/", "__").replace(":", "_").replace(" ", "_")


def _image_id(path):
    path = Path(path)
    return path.stem or _safe_id(path)


def _point_metrics(result, points):
    valid = result.valid_points(points)
    support = result.support_points(points)
    total = int(points.shape[0]) if points.numel() else 0
    valid_count = int(valid.sum().item()) if valid.numel() else 0
    support_count = int((valid & support).sum().item()) if valid.numel() else 0
    return {
        "point_count": total,
        "valid_point_count": valid_count,
        "support_point_count": support_count,
        "valid_point_frac": float(valid_count / total) if total else 0.0,
        "support_point_frac": float(support_count / total) if total else 0.0,
    }, valid, support


def _aggregate(rows):
    if not rows:
        return {"count": 0}

    def mean(name):
        values = [float(row["metrics"].get(name, 0.0)) for row in rows]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "count": len(rows),
        "valid_frac_mean": mean("valid_frac"),
        "invalid_frac_mean": mean("invalid_frac"),
        "support_frac_mean": mean("support_frac"),
        "valid_point_frac_mean": mean("valid_point_frac"),
        "support_point_frac_mean": mean("support_point_frac"),
        "invalid_component_count_mean": mean("invalid_component_count"),
        "largest_invalid_component_frac_mean": mean("largest_invalid_component_frac"),
    }


def _sort_rows(rows, sort_by):
    if sort_by == "invalid_frac_desc":
        return sorted(rows, key=lambda row: (-row["metrics"].get("invalid_frac", 0.0), row["image_id"]))
    if sort_by == "support_frac_asc":
        return sorted(rows, key=lambda row: (row["metrics"].get("support_frac", 1.0), row["image_id"]))
    if sort_by == "valid_frac_asc":
        return sorted(rows, key=lambda row: (row["metrics"].get("valid_frac", 1.0), row["image_id"]))
    return sorted(rows, key=lambda row: row["image_id"])


def _contact_sheet(visuals, output_path, max_records=24):
    visuals = list(visuals)[: int(max_records)]
    cell = (220, 140)
    text_cell = (320, 140)
    panels_per_row = 6
    sheet = Image.new("RGB", (cell[0] * panels_per_row + text_cell[0], cell[1] * max(1, len(visuals))), (232, 232, 228))
    for idx, row in enumerate(visuals):
        y = idx * cell[1]
        image = _open_rgb(row["image_path"])
        result = row["result"]
        points = row["points"]
        valid_flags = row["valid_flags"]
        support_flags = row["support_flags"]
        panels = [
            _label(_fit_canvas(image, cell), "RGB render"),
            _label(_fit_canvas(_overlay_mask(image, result.valid_mask), cell), "valid green"),
            _label(_fit_canvas(_overlay_mask(image, result.support_mask, color=(0, 160, 255), invalid_color=(80, 80, 80), alpha=0.42), cell), "support blue"),
            _label(_fit_canvas(_heatmap01(result.support_score), cell), "support score"),
            _label(_fit_canvas(_heatmap01(result.invalid_score), cell), "invalid score"),
            _label(_fit_canvas(_draw_points(image, points, valid_flags, support_flags), cell), "points"),
        ]
        for col, panel in enumerate(panels):
            sheet.paste(panel, (col * cell[0], y))
        metrics = row["metrics"]
        text = _text_panel(
            [
                f"id: {row['image_id']}",
                f"valid_frac: {metrics.get('valid_frac', 0.0):.3f}",
                f"support_frac: {metrics.get('support_frac', 0.0):.3f}",
                f"invalid_cc: {metrics.get('invalid_component_count', 0)}",
                f"largest_invalid: {metrics.get('largest_invalid_component_frac', 0.0):.3f}",
                f"points valid/support: {metrics.get('valid_point_count', 0)}/{metrics.get('support_point_count', 0)}",
            ],
            text_cell,
            title=f"{idx + 1}. no-ref QA",
        )
        sheet.paste(text, (cell[0] * panels_per_row, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return str(output_path)


def audit_images(
    image_paths,
    output_dir,
    image_scale=0.5,
    visual_max=24,
    sort_by="invalid_frac_desc",
    support_threshold=0.22,
    support_dilate_radius=5,
    invalid_min_area=96,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    builder = NoReferenceValidSupportMaskBuilder(
        NoReferenceValidSupportMaskConfig(
            support_threshold=float(support_threshold),
            support_dilate_radius=int(support_dilate_radius),
            invalid_min_area=int(invalid_min_area),
        )
    )
    rows = []
    visuals = []
    for path in [Path(item) for item in image_paths]:
        image = _resize_image(_open_rgb(path), image_scale)
        result = builder.build(_image_to_tensor(image))
        points = _topk_support_points(result.support_score)
        point_metrics, valid_flags, support_flags = _point_metrics(result, points)
        metrics = {**result.summary, **point_metrics}
        image_id = _image_id(path)
        prefix = mask_dir / _safe_id(image_id)
        mask_paths = save_mask_bundle_pngs(result, prefix)
        row = {
            "image_id": image_id,
            "image_path": os.path.abspath(os.fspath(path)),
            "mask_paths": mask_paths,
            "metrics": metrics,
        }
        rows.append(row)
        visuals.append(
            {
                **row,
                "result": result,
                "points": points,
                "valid_flags": valid_flags,
                "support_flags": support_flags,
            }
        )

    sorted_rows = _sort_rows(rows, sort_by)
    visual_by_id = {row["image_id"]: row for row in visuals}
    visual_records = [visual_by_id[row["image_id"]] for row in sorted_rows[: int(visual_max)]]
    contact_sheet = _contact_sheet(visual_records, output_dir / f"valid_support_mask_{sort_by}.jpg", max_records=visual_max)
    records_jsonl = output_dir / "valid_support_mask_records.jsonl"
    with records_jsonl.open("w") as f:
        for row in sorted_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "records_jsonl": str(records_jsonl),
        "contact_sheet": contact_sheet,
        "image_scale": float(image_scale),
        "sort_by": sort_by,
        "mask_config": {
            "support_threshold": float(support_threshold),
            "support_dilate_radius": int(support_dilate_radius),
            "invalid_min_area": int(invalid_min_area),
        },
        "aggregate": _aggregate(rows),
        "top_visual_image_ids": [row["image_id"] for row in visual_records],
    }
    summary_path = output_dir / "valid_support_mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def collect_image_paths(image_dir="", images=None, glob="*.png"):
    paths = []
    if image_dir:
        root = Path(image_dir)
        patterns = [item.strip() for item in str(glob or "*.png").split(",") if item.strip()]
        for pattern in patterns:
            paths.extend(sorted(root.rglob(pattern)))
    for item in images or []:
        paths.append(Path(item))
    unique = []
    seen = set()
    for path in paths:
        key = os.path.abspath(os.fspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def main():
    parser = argparse.ArgumentParser(description="Audit RGB renders with portable no-reference valid/support masks.")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--glob", default="*.png,*.jpg,*.jpeg")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_scale", type=float, default=0.5)
    parser.add_argument("--visual_max", type=int, default=24)
    parser.add_argument("--sort_by", choices=["invalid_frac_desc", "support_frac_asc", "valid_frac_asc", "manifest"], default="invalid_frac_desc")
    parser.add_argument("--support_threshold", type=float, default=0.22)
    parser.add_argument("--support_dilate_radius", type=int, default=5)
    parser.add_argument("--invalid_min_area", type=int, default=96)
    args = parser.parse_args()
    image_paths = collect_image_paths(args.image_dir, args.images, args.glob)
    if not image_paths:
        raise SystemExit("No images found for audit.")
    summary = audit_images(
        image_paths,
        args.output_dir,
        image_scale=args.image_scale,
        visual_max=args.visual_max,
        sort_by=args.sort_by,
        support_threshold=args.support_threshold,
        support_dilate_radius=args.support_dilate_radius,
        invalid_min_area=args.invalid_min_area,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
