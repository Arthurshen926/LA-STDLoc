#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from la_artifacts.detector import ArtifactDetector
from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache
from la_artifacts.valid_mask import ArtifactValidMaskBuilder, ArtifactValidMaskConfig, save_valid_mask_png


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


def _heatmap01(score_map):
    score = torch.as_tensor(score_map, dtype=torch.float32).detach().cpu().clamp(0.0, 1.0)
    array = score.numpy()
    red = np.clip(2.0 * array, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(array - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - array), 0.0, 1.0)
    rgb = np.stack([red, green * 0.6, blue * 0.4], axis=-1)
    return Image.fromarray((rgb * 255.0).round().astype("uint8"))


def _mask_overlay(image, mask, color=(0, 210, 90), invalid_color=(215, 30, 30), alpha=0.38):
    base = image.convert("RGB")
    mask_img = Image.fromarray(torch.as_tensor(mask).detach().cpu().bool().numpy().astype(np.uint8) * 255, mode="L")
    mask_img = mask_img.resize(base.size, Image.Resampling.NEAREST)
    valid_layer = Image.new("RGB", base.size, color)
    invalid_layer = Image.new("RGB", base.size, invalid_color)
    overlay = Image.composite(valid_layer, invalid_layer, mask_img)
    return Image.blend(base, overlay, float(alpha))


def _fit_canvas(image, size, bg=(24, 24, 24)):
    canvas = Image.new("RGB", size, bg)
    work = image.copy()
    work.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - work.width) // 2
    y = (size[1] - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def _font(size=14):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(image, label):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = _font(12)
    draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0))
    draw.text((6, 4), str(label), fill=(255, 255, 255), font=font)
    return out


def _text_panel(lines, size, title=""):
    panel = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(panel)
    title_font = _font(15)
    body_font = _font(12)
    y = 8
    if title:
        draw.text((8, y), str(title), fill=(12, 12, 12), font=title_font)
        y += 24
    max_chars = max(12, int((size[0] - 16) / 7.2))
    for line in lines:
        text = str(line)
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for chunk in chunks:
            draw.text((8, y), chunk, fill=(32, 32, 32), font=body_font)
            y += 16
            if y > size[1] - 18:
                draw.text((8, size[1] - 18), "...", fill=(32, 32, 32), font=body_font)
                return panel
    return panel


def _draw_keypoints(image, points_xy, valid_flags=None, max_points=600):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    points = torch.as_tensor(points_xy, dtype=torch.float32).detach().cpu()
    if points.numel() == 0:
        return out
    if valid_flags is None:
        valid_flags = torch.ones(points.shape[0], dtype=torch.bool)
    valid_flags = torch.as_tensor(valid_flags, dtype=torch.bool).detach().cpu()
    if points.shape[0] > int(max_points):
        points = points[: int(max_points)]
        valid_flags = valid_flags[: int(max_points)]
    for (x, y), keep in zip(points.tolist(), valid_flags.tolist()):
        color = (0, 255, 80) if keep else (255, 60, 40)
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)
    return out


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _cache_metric(item, name):
    if not item:
        return None
    return _safe_float(item.get(name))


def _resolve_reference(record, scene_root, images="processed"):
    if record.source == "synthetic_rgb" and record.nearest_train_image:
        name = str(record.nearest_train_image)
        base = name.split("__", 1)[0]
        candidates = [name, base]
        for prefix in ("synthetic/", "synthetic_interpolate/", "synthetic_spatial_offset/"):
            if base.startswith(prefix):
                candidates.append(base[len(prefix) :])
        for candidate in candidates:
            path = Path(scene_root) / images / candidate
            if path.exists():
                return str(path)
    if record.source == "train_rgb" and record.image_path:
        return str(record.image_path)
    return ""


def _nms_topk_points(score_map, max_points=2048, threshold=0.0, nms_radius=4):
    from scene.kpdetector import simple_nms

    score = torch.as_tensor(score_map, dtype=torch.float32)
    if score.dim() == 3:
        score = score.squeeze()
    if score.dim() != 2:
        raise ValueError(f"Expected 2D keypoint score map, got {tuple(score.shape)}")
    nms = simple_nms(score[None, None], int(nms_radius))[0, 0].reshape(-1)
    keep = nms > float(threshold)
    if not keep.any():
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
    values = nms[ids]
    count = min(int(max_points), int(values.numel()))
    order = torch.topk(values, count).indices
    ids = ids[order]
    values = values[order]
    width = score.shape[1]
    points = torch.stack([ids % width, ids // width], dim=1).float()
    return points, values


class KeypointExtractor:
    def __init__(self, backend="superpoint", device="auto", max_points=2048, threshold=0.0001, nms_radius=4):
        self.backend = str(backend or "none").lower()
        self.max_points = int(max_points)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.device = self._resolve_device(device)
        self.model = None
        if self.backend == "superpoint":
            from encoders.sp_encoder.export_image_embeddings import SuperPoint

            self.model = SuperPoint().to(self.device).eval()
        elif self.backend not in {"none", "score"}:
            raise ValueError(f"Unknown keypoint backend: {backend}")

    def _resolve_device(self, device):
        if str(device or "auto").lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @torch.no_grad()
    def extract(self, image):
        if self.backend == "none":
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
        tensor = _image_to_tensor(image)[None].to(device=self.device, dtype=torch.float32)
        if self.backend == "superpoint":
            _, scores = self.model(tensor)
            score = scores[0].detach().cpu()
            if tuple(score.shape[-2:]) != (image.height, image.width):
                score = torch.nn.functional.interpolate(
                    score[None, None],
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        else:
            gray = tensor.mean(dim=1)[0].detach().cpu()
            dx = torch.zeros_like(gray)
            dy = torch.zeros_like(gray)
            dx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
            dy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
            score = torch.sqrt(dx.square() + dy.square())
        return _nms_topk_points(score, max_points=self.max_points, threshold=self.threshold, nms_radius=self.nms_radius)


def _summarize_keypoint_retention(valid_mask, points_xy):
    points = torch.as_tensor(points_xy, dtype=torch.float32)
    keep = valid_mask.valid_points(points)
    total = int(points.reshape(-1, 2).shape[0]) if points.numel() else 0
    valid = int(keep.sum().item()) if keep.numel() else 0
    return {
        "keypoint_count": total,
        "valid_keypoint_count": valid,
        "invalid_keypoint_count": int(total - valid),
        "valid_keypoint_frac": float(valid / total) if total else 0.0,
    }


@dataclass
class ValidMaskVisualRecord:
    query_id: str
    source: str
    image_path: str
    reference_path: str = ""
    score_map: torch.Tensor = None
    valid_mask: torch.Tensor = None
    keypoints_xy: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.float32))
    valid_keypoints: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.bool))
    metrics: dict = field(default_factory=dict)


def build_valid_mask_contact_sheet(records, output_path, max_records=24):
    records = list(records)[: int(max_records)]
    cell = (240, 150)
    text_cell = (330, 150)
    width = cell[0] * 5 + text_cell[0]
    height = max(cell[1], cell[1] * max(1, len(records)))
    sheet = Image.new("RGB", (width, height), (232, 232, 228))

    for idx, row in enumerate(records):
        y = idx * cell[1]
        rgb = _open_rgb(row.image_path)
        ref = _open_rgb(row.reference_path) if row.reference_path and Path(row.reference_path).exists() else Image.new("RGB", rgb.size, (32, 32, 32))
        heat = _heatmap01(row.score_map)
        overlay = _mask_overlay(rgb, row.valid_mask)
        points = _draw_keypoints(rgb, row.keypoints_xy, row.valid_keypoints)
        metrics = row.metrics
        text = _text_panel(
            [
                f"id: {row.query_id}",
                f"valid_frac: {metrics.get('valid_frac', 0.0):.3f}",
                f"largest_cc: {metrics.get('largest_component_frac', 0.0):.3f}",
                f"kpts: {metrics.get('valid_keypoint_count', 0)}/{metrics.get('keypoint_count', 0)}",
                f"valid_kpt_frac: {metrics.get('valid_keypoint_frac', 0.0):.3f}",
                f"artifact_mean: {metrics.get('artifact_score_mean', 0.0):.3f}",
                f"artifact_p95: {metrics.get('artifact_score_p95', 0.0):.3f}",
                f"stage: {metrics.get('failure_stage', '<missing>')}",
                f"sparse/dense te: {metrics.get('te', '<na>')} / {metrics.get('dense_te', '<na>')}",
            ],
            text_cell,
            title=f"{idx + 1}. {row.source}",
        )
        panels = [
            _label(_fit_canvas(rgb, cell), "synthetic/train RGB"),
            _label(_fit_canvas(ref, cell), "nearest train/source"),
            _label(_fit_canvas(heat, cell), "artifact score"),
            _label(_fit_canvas(overlay, cell), "valid mask green"),
            _label(_fit_canvas(points, cell), "kpts green=kept red=masked"),
        ]
        for col, panel in enumerate(panels):
            sheet.paste(panel, (col * cell[0], y))
        sheet.paste(text, (cell[0] * len(panels), y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return str(output_path)


def _sort_rows(rows, sort_by):
    def key(row):
        metrics = row["metrics"]
        if sort_by == "invalid_kpoints_desc":
            return (-metrics.get("invalid_keypoint_count", 0), row["query_id"])
        if sort_by == "valid_frac_asc":
            return (metrics.get("valid_frac", 1.0), row["query_id"])
        if sort_by == "artifact_desc":
            return (-metrics.get("artifact_score_mean", 0.0), row["query_id"])
        if sort_by == "dense_te_desc":
            dense = metrics.get("dense_te")
            return (dense is None, -(dense or -1.0), row["query_id"])
        return (row["query_id"],)

    return sorted(rows, key=key)


def _aggregate(rows):
    if not rows:
        return {"count": 0}
    metrics = [row["metrics"] for row in rows]

    def mean(name):
        values = [_safe_float(item.get(name)) for item in metrics]
        values = [value for value in values if value is not None]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "count": len(rows),
        "valid_frac_mean": mean("valid_frac"),
        "valid_frac_min": min(float(item.get("valid_frac", 0.0)) for item in metrics),
        "valid_keypoint_frac_mean": mean("valid_keypoint_frac"),
        "keypoint_count_mean": mean("keypoint_count"),
        "valid_keypoint_count_mean": mean("valid_keypoint_count"),
        "artifact_score_mean": mean("artifact_score_mean"),
        "artifact_score_p95_mean": mean("artifact_score_p95"),
    }


def evaluate_valid_masks(
    manifest_path,
    output_dir,
    teacher_cache_path="",
    scene_root="",
    images="processed",
    sources=("synthetic_rgb",),
    include_rejected=False,
    max_records=0,
    image_scale=0.5,
    keypoint_backend="superpoint",
    keypoint_device="auto",
    keypoint_max=2048,
    keypoint_threshold=0.0001,
    keypoint_nms=4,
    mask_max_artifact_score=0.45,
    mask_erosion_radius=3,
    mask_min_component_area=64,
    mask_min_component_area_frac=0.0,
    mask_min_valid_frac=0.0,
    sort_by="invalid_kpoints_desc",
    visual_max=24,
):
    output_dir = Path(output_dir)
    mask_dir = output_dir / "valid_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = PseudoQueryManifest.load(manifest_path)
    cache = PseudoTeacherCache.load(teacher_cache_path) if teacher_cache_path else PseudoTeacherCache()
    detector = ArtifactDetector()
    mask_builder = ArtifactValidMaskBuilder(
        ArtifactValidMaskConfig(
            max_artifact_score=float(mask_max_artifact_score),
            erosion_radius=int(mask_erosion_radius),
            min_component_area=int(mask_min_component_area),
            min_component_area_frac=float(mask_min_component_area_frac),
            min_valid_frac=float(mask_min_valid_frac),
        )
    )
    keypoints = KeypointExtractor(
        backend=keypoint_backend,
        device=keypoint_device,
        max_points=keypoint_max,
        threshold=keypoint_threshold,
        nms_radius=keypoint_nms,
    )
    source_set = {str(item) for item in sources}
    records = [row for row in manifest.records if row.source in source_set and (include_rejected or row.accepted)]
    if int(max_records or 0) > 0:
        records = records[: int(max_records)]

    rows = []
    visuals = []
    for record in records:
        image = _resize_image(_open_rgb(record.image_path), image_scale)
        evidence = detector.detect(rendered_rgb=_image_to_tensor(image))
        valid = mask_builder.build(evidence)
        points, point_scores = keypoints.extract(image)
        kpt_summary = _summarize_keypoint_retention(valid, points)
        valid_flags = valid.valid_points(points)
        item = cache.get(record.teacher_cache_key or record.query_id)
        mask_path = mask_dir / f"{record.query_id.replace(':', '__').replace('/', '_')}.valid_mask.png"
        save_valid_mask_png(valid, mask_path)
        metrics = {
            **valid.summary,
            **evidence.summary,
            **kpt_summary,
            "point_score_mean": float(point_scores.mean().item()) if point_scores.numel() else 0.0,
            "failure_stage": str(item.get("failure_stage", "<missing>")) if item else "<missing>",
            "te": _cache_metric(item, "te"),
            "dense_te": _cache_metric(item, "dense_te"),
        }
        row = {
            "query_id": record.query_id,
            "source": record.source,
            "image_path": record.image_path,
            "reference_path": _resolve_reference(record, scene_root, images=images) if scene_root else "",
            "mask_path": str(mask_path),
            "metrics": metrics,
        }
        rows.append(row)
        visuals.append(
            ValidMaskVisualRecord(
                query_id=record.query_id,
                source=record.source,
                image_path=record.image_path,
                reference_path=row["reference_path"],
                score_map=evidence.score_map,
                valid_mask=valid.mask,
                keypoints_xy=points,
                valid_keypoints=valid_flags,
                metrics=metrics,
            )
        )

    sorted_rows = _sort_rows(rows, sort_by)
    visual_by_id = {row.query_id: row for row in visuals}
    visual_records = [visual_by_id[row["query_id"]] for row in sorted_rows[: int(visual_max)]]
    contact_sheet = build_valid_mask_contact_sheet(visual_records, output_dir / f"valid_mask_contact_sheet_{sort_by}.jpg", max_records=visual_max)
    records_jsonl = output_dir / "valid_mask_eval_records.jsonl"
    with records_jsonl.open("w") as f:
        for row in sorted_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "teacher_cache": os.path.abspath(os.fspath(teacher_cache_path)) if teacher_cache_path else "",
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "records_jsonl": str(records_jsonl),
        "contact_sheet": contact_sheet,
        "sources": sorted(source_set),
        "include_rejected": bool(include_rejected),
        "image_scale": float(image_scale),
        "keypoint_backend": keypoint_backend,
        "keypoint_device": str(keypoints.device),
        "mask_config": {
            "max_artifact_score": float(mask_max_artifact_score),
            "erosion_radius": int(mask_erosion_radius),
            "min_component_area": int(mask_min_component_area),
            "min_component_area_frac": float(mask_min_component_area_frac),
            "min_valid_frac": float(mask_min_valid_frac),
        },
        "aggregate": _aggregate(rows),
        "top_visual_query_ids": [row.query_id for row in visual_records],
    }
    summary_path = output_dir / "valid_mask_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate artifact valid masks on pseudo-query RGB renders.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--sources", default="synthetic_rgb")
    parser.add_argument("--include_rejected", action="store_true", default=False)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--image_scale", type=float, default=0.5)
    parser.add_argument("--keypoint_backend", choices=["superpoint", "score", "none"], default="superpoint")
    parser.add_argument("--keypoint_device", default="auto")
    parser.add_argument("--keypoint_max", type=int, default=2048)
    parser.add_argument("--keypoint_threshold", type=float, default=0.0001)
    parser.add_argument("--keypoint_nms", type=int, default=4)
    parser.add_argument("--mask_max_artifact_score", type=float, default=0.45)
    parser.add_argument("--mask_erosion_radius", type=int, default=3)
    parser.add_argument("--mask_min_component_area", type=int, default=64)
    parser.add_argument("--mask_min_component_area_frac", type=float, default=0.0)
    parser.add_argument("--mask_min_valid_frac", type=float, default=0.0)
    parser.add_argument(
        "--sort_by",
        choices=["invalid_kpoints_desc", "valid_frac_asc", "artifact_desc", "dense_te_desc", "manifest"],
        default="invalid_kpoints_desc",
    )
    parser.add_argument("--visual_max", type=int, default=24)
    args = parser.parse_args()
    evaluate_valid_masks(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        teacher_cache_path=args.teacher_cache,
        scene_root=args.scene_root,
        images=args.images,
        sources=_comma_list(args.sources),
        include_rejected=args.include_rejected,
        max_records=args.max_records,
        image_scale=args.image_scale,
        keypoint_backend=args.keypoint_backend,
        keypoint_device=args.keypoint_device,
        keypoint_max=args.keypoint_max,
        keypoint_threshold=args.keypoint_threshold,
        keypoint_nms=args.keypoint_nms,
        mask_max_artifact_score=args.mask_max_artifact_score,
        mask_erosion_radius=args.mask_erosion_radius,
        mask_min_component_area=args.mask_min_component_area,
        mask_min_component_area_frac=args.mask_min_component_area_frac,
        mask_min_valid_frac=args.mask_min_valid_frac,
        sort_by=args.sort_by,
        visual_max=args.visual_max,
    )


if __name__ == "__main__":
    main()
