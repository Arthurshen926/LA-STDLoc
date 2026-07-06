#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder, NoReferenceValidMaskConfig, save_no_reference_valid_mask_pngs
from la_artifacts.pseudo_query import PseudoQueryManifest


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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


def _font(size=13):
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


def _draw_points(image, points_xy, valid_flags, support_flags, max_points=800):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    points = torch.as_tensor(points_xy, dtype=torch.float32).detach().cpu()
    valid = torch.as_tensor(valid_flags, dtype=torch.bool).detach().cpu()
    support = torch.as_tensor(support_flags, dtype=torch.bool).detach().cpu()
    if points.numel() == 0:
        return out
    if points.shape[0] > int(max_points):
        points = points[: int(max_points)]
        valid = valid[: int(max_points)]
        support = support[: int(max_points)]
    for (x, y), is_valid, is_support in zip(points.tolist(), valid.tolist(), support.tolist()):
        if not is_valid:
            color = (240, 45, 35)
        elif is_support:
            color = (0, 240, 85)
        else:
            color = (255, 205, 35)
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
        for chunk in [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]:
            draw.text((8, y), chunk, fill=(32, 32, 32), font=_font(12))
            y += 16
            if y > size[1] - 18:
                draw.text((8, size[1] - 18), "...", fill=(32, 32, 32), font=_font(12))
                return panel
    return panel


def _nms_topk_points(score_map, max_points=2048, threshold=0.0, nms_radius=4):
    from scene.kpdetector import simple_nms

    score = torch.as_tensor(score_map, dtype=torch.float32)
    if score.dim() == 3:
        score = score.squeeze()
    nms = simple_nms(score[None, None], int(nms_radius))[0, 0].reshape(-1)
    keep = nms > float(threshold)
    if not keep.any():
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
    values = nms[ids]
    order = torch.topk(values, min(int(max_points), int(values.numel()))).indices
    ids = ids[order]
    values = values[order]
    width = score.shape[1]
    return torch.stack([ids % width, ids // width], dim=1).float(), values


class KeypointExtractor:
    def __init__(self, backend="score", device="auto", max_points=2048, threshold=0.0001, nms_radius=4):
        self.backend = str(backend or "score").lower()
        self.max_points = int(max_points)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.device = torch.device("cuda" if str(device).lower() == "auto" and torch.cuda.is_available() else ("cpu" if str(device).lower() == "auto" else device))
        self.model = None
        if self.backend == "superpoint":
            from encoders.sp_encoder.export_image_embeddings import SuperPoint

            self.model = SuperPoint().to(self.device).eval()
        elif self.backend not in {"score", "none"}:
            raise ValueError(f"Unknown keypoint backend: {backend}")

    @torch.no_grad()
    def extract(self, image, support_score=None):
        if self.backend == "none":
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
        if self.backend == "superpoint":
            tensor = _image_to_tensor(image)[None].to(device=self.device, dtype=torch.float32)
            _, scores = self.model(tensor)
            score = scores[0].detach().cpu()
            if tuple(score.shape[-2:]) != (image.height, image.width):
                score = torch.nn.functional.interpolate(score[None, None], size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
        else:
            if support_score is not None:
                score = torch.as_tensor(support_score, dtype=torch.float32).detach().cpu()
            else:
                gray = _image_to_tensor(image).mean(dim=0)
                dx = torch.zeros_like(gray)
                dy = torch.zeros_like(gray)
                dx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
                dy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
                score = torch.sqrt(dx.square() + dy.square())
        return _nms_topk_points(score, max_points=self.max_points, threshold=self.threshold, nms_radius=self.nms_radius)


def summarize_point_masks(valid_mask, support_mask, points_xy):
    points = torch.as_tensor(points_xy, dtype=torch.float32)
    total = int(points.reshape(-1, 2).shape[0]) if points.numel() else 0
    valid = _points_in_mask(valid_mask, points)
    support = _points_in_mask(support_mask, points)
    valid_count = int(valid.sum().item()) if valid.numel() else 0
    support_count = int((valid & support).sum().item()) if valid.numel() else 0
    return {
        "point_count": total,
        "valid_point_count": valid_count,
        "invalid_point_count": int(total - valid_count),
        "support_point_count": support_count,
        "valid_point_frac": float(valid_count / total) if total else 0.0,
        "support_point_frac": float(support_count / total) if total else 0.0,
    }


def _points_in_mask(mask, points_xy):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    points = torch.as_tensor(points_xy, dtype=torch.float32)
    if points.numel() == 0:
        return torch.zeros(points.shape[:-1], dtype=torch.bool)
    flat = points.reshape(-1, 2)
    x = torch.floor(flat[:, 0]).long()
    y = torch.floor(flat[:, 1]).long()
    height, width = mask.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep = torch.zeros(flat.shape[0], dtype=torch.bool)
    if inside.any():
        keep[inside] = mask[y[inside], x[inside]]
    return keep.reshape(points.shape[:-1])


@dataclass
class NoReferenceVisualRecord:
    query_id: str
    image_path: str
    valid_mask: torch.Tensor
    support_mask: torch.Tensor
    invalid_score: torch.Tensor
    support_score: torch.Tensor
    points_xy: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.float32))
    valid_points: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.bool))
    support_points: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.bool))
    metrics: dict = field(default_factory=dict)


def build_no_reference_contact_sheet(records, output_path, max_records=24):
    records = list(records)[: int(max_records)]
    cell = (220, 140)
    text_cell = (320, 140)
    panels_per_row = 6
    sheet = Image.new("RGB", (cell[0] * panels_per_row + text_cell[0], cell[1] * max(1, len(records))), (232, 232, 228))
    for idx, row in enumerate(records):
        y = idx * cell[1]
        rgb = _open_rgb(row.image_path)
        panels = [
            _label(_fit_canvas(rgb, cell), "synthetic RGB"),
            _label(_fit_canvas(_overlay_mask(rgb, row.valid_mask), cell), "valid: green"),
            _label(_fit_canvas(_overlay_mask(rgb, row.support_mask, color=(0, 160, 255), invalid_color=(80, 80, 80), alpha=0.42), cell), "support: blue"),
            _label(_fit_canvas(_heatmap01(row.support_score), cell), "support score"),
            _label(_fit_canvas(_heatmap01(row.invalid_score), cell), "invalid score"),
            _label(_fit_canvas(_draw_points(rgb, row.points_xy, row.valid_points, row.support_points), cell), "points green/yellow/red"),
        ]
        for col, panel in enumerate(panels):
            sheet.paste(panel, (col * cell[0], y))
        metrics = row.metrics
        text = _text_panel(
            [
                f"id: {row.query_id}",
                f"valid_frac: {metrics.get('valid_frac', 0.0):.3f}",
                f"support_frac: {metrics.get('support_frac', 0.0):.3f}",
                f"points valid/support: {metrics.get('valid_point_count', 0)}/{metrics.get('support_point_count', 0)} of {metrics.get('point_count', 0)}",
                f"support_score_mean: {metrics.get('support_score_mean', 0.0):.3f}",
                f"invalid_score_mean: {metrics.get('invalid_score_mean', 0.0):.3f}",
            ],
            text_cell,
            title=f"{idx + 1}. no-ref",
        )
        sheet.paste(text, (cell[0] * panels_per_row, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return str(output_path)


def _sort_rows(rows, sort_by):
    def key(row):
        metrics = row["metrics"]
        if sort_by == "support_frac_asc":
            return (metrics.get("support_frac", 1.0), row["query_id"])
        if sort_by == "invalid_frac_desc":
            return (-metrics.get("invalid_frac", 0.0), row["query_id"])
        if sort_by == "support_points_asc":
            return (metrics.get("support_point_frac", 1.0), row["query_id"])
        return (row["query_id"],)

    return sorted(rows, key=key)


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
        "point_count_mean": mean("point_count"),
        "support_score_mean": mean("support_score_mean"),
        "invalid_score_mean": mean("invalid_score_mean"),
    }


def evaluate_no_reference_valid_masks(
    manifest_path,
    output_dir,
    sources=("synthetic_rgb",),
    include_rejected=False,
    max_records=0,
    image_scale=0.5,
    keypoint_backend="score",
    keypoint_device="auto",
    keypoint_max=2048,
    keypoint_threshold=0.0001,
    keypoint_nms=4,
    support_threshold=0.22,
    support_dilate_radius=5,
    invalid_min_area=96,
    sort_by="support_points_asc",
    visual_max=24,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    manifest = PseudoQueryManifest.load(manifest_path)
    source_set = {str(item) for item in sources}
    records = [row for row in manifest.records if row.source in source_set and (include_rejected or row.accepted)]
    if int(max_records or 0) > 0:
        records = records[: int(max_records)]
    builder = NoReferenceValidMaskBuilder(
        NoReferenceValidMaskConfig(
            support_threshold=float(support_threshold),
            support_dilate_radius=int(support_dilate_radius),
            invalid_min_area=int(invalid_min_area),
        )
    )
    keypoints = KeypointExtractor(
        backend=keypoint_backend,
        device=keypoint_device,
        max_points=keypoint_max,
        threshold=keypoint_threshold,
        nms_radius=keypoint_nms,
    )
    rows = []
    visuals = []
    for record in records:
        image = _resize_image(_open_rgb(record.image_path), image_scale)
        result = builder.build(_image_to_tensor(image))
        points, point_scores = keypoints.extract(image, support_score=result.support_score)
        valid_flags = result.valid_points(points)
        support_flags = result.support_points(points)
        point_metrics = summarize_point_masks(result.valid_mask, result.support_mask, points)
        prefix = mask_dir / record.query_id.replace(":", "__").replace("/", "_")
        mask_paths = save_no_reference_valid_mask_pngs(result, prefix)
        metrics = {
            **result.summary,
            **point_metrics,
            "point_score_mean": float(point_scores.mean().item()) if point_scores.numel() else 0.0,
        }
        row = {
            "query_id": record.query_id,
            "source": record.source,
            "image_path": record.image_path,
            "mask_paths": mask_paths,
            "metrics": metrics,
        }
        rows.append(row)
        visuals.append(
            NoReferenceVisualRecord(
                query_id=record.query_id,
                image_path=record.image_path,
                valid_mask=result.valid_mask,
                support_mask=result.support_mask,
                invalid_score=result.invalid_score,
                support_score=result.support_score,
                points_xy=points,
                valid_points=valid_flags,
                support_points=support_flags,
                metrics=metrics,
            )
        )
    sorted_rows = _sort_rows(rows, sort_by)
    visual_by_id = {row.query_id: row for row in visuals}
    visual_records = [visual_by_id[row["query_id"]] for row in sorted_rows[: int(visual_max)]]
    contact_sheet = build_no_reference_contact_sheet(visual_records, output_dir / f"no_reference_valid_mask_{sort_by}.jpg", max_records=visual_max)
    records_jsonl = output_dir / "no_reference_valid_mask_records.jsonl"
    with records_jsonl.open("w") as f:
        for row in sorted_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "records_jsonl": str(records_jsonl),
        "contact_sheet": contact_sheet,
        "sources": sorted(source_set),
        "include_rejected": bool(include_rejected),
        "image_scale": float(image_scale),
        "keypoint_backend": keypoint_backend,
        "keypoint_device": str(keypoints.device),
        "mask_config": {
            "support_threshold": float(support_threshold),
            "support_dilate_radius": int(support_dilate_radius),
            "invalid_min_area": int(invalid_min_area),
        },
        "aggregate": _aggregate(rows),
        "top_visual_query_ids": [row.query_id for row in visual_records],
    }
    summary_path = output_dir / "no_reference_valid_mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate no-reference synthetic RGB valid/support masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sources", default="synthetic_rgb")
    parser.add_argument("--include_rejected", action="store_true", default=False)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--image_scale", type=float, default=0.5)
    parser.add_argument("--keypoint_backend", choices=["score", "superpoint", "none"], default="score")
    parser.add_argument("--keypoint_device", default="auto")
    parser.add_argument("--keypoint_max", type=int, default=2048)
    parser.add_argument("--keypoint_threshold", type=float, default=0.0001)
    parser.add_argument("--keypoint_nms", type=int, default=4)
    parser.add_argument("--support_threshold", type=float, default=0.22)
    parser.add_argument("--support_dilate_radius", type=int, default=5)
    parser.add_argument("--invalid_min_area", type=int, default=96)
    parser.add_argument("--sort_by", choices=["support_points_asc", "support_frac_asc", "invalid_frac_desc", "manifest"], default="support_points_asc")
    parser.add_argument("--visual_max", type=int, default=24)
    args = parser.parse_args()
    evaluate_no_reference_valid_masks(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sources=_comma_list(args.sources),
        include_rejected=args.include_rejected,
        max_records=args.max_records,
        image_scale=args.image_scale,
        keypoint_backend=args.keypoint_backend,
        keypoint_device=args.keypoint_device,
        keypoint_max=args.keypoint_max,
        keypoint_threshold=args.keypoint_threshold,
        keypoint_nms=args.keypoint_nms,
        support_threshold=args.support_threshold,
        support_dilate_radius=args.support_dilate_radius,
        invalid_min_area=args.invalid_min_area,
        sort_by=args.sort_by,
        visual_max=args.visual_max,
    )


if __name__ == "__main__":
    main()
