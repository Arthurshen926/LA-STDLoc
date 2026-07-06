#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from la_artifacts.detector import ArtifactDetector
from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache


def _comma_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _open_rgb(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def _image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _heatmap01(score_map):
    score = torch.as_tensor(score_map, dtype=torch.float32).detach().cpu().clamp(0.0, 1.0)
    array = score.numpy()
    red = np.clip(2.0 * array, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(array - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - array), 0.0, 1.0)
    rgb = np.stack([red, green * 0.6, blue * 0.4], axis=-1)
    return Image.fromarray((rgb * 255.0).round().astype("uint8"))


def _low_detail_risk(image, threshold=0.035, pool=15):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(gray)[None, None]
    dx = torch.zeros_like(tensor)
    dy = torch.zeros_like(tensor)
    dx[:, :, :, 1:] = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs()
    dy[:, :, 1:, :] = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs()
    grad = torch.sqrt(dx.square() + dy.square()).clamp_min(0.0)
    pad = int(pool) // 2
    local = torch.nn.functional.avg_pool2d(grad, kernel_size=int(pool), stride=1, padding=pad)
    risk = (1.0 - local[0, 0] / max(float(threshold), 1e-6)).clamp(0.0, 1.0)
    flat = risk.reshape(-1)
    summary = {
        "low_detail_mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "low_detail_p95": float(torch.quantile(flat, 0.95).item()) if flat.numel() else 0.0,
    }
    return risk, summary


def _fit_canvas(image, size, bg=(24, 24, 24)):
    canvas = Image.new("RGB", size, bg)
    work = image.copy()
    work.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - work.width) // 2
    y = (size[1] - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def _font(size=14):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_panel(lines, size, title=None):
    panel = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(panel)
    title_font = _font(15)
    body_font = _font(12)
    y = 8
    if title:
        draw.text((8, y), title, fill=(12, 12, 12), font=title_font)
        y += 24
    for line in lines:
        text = str(line)
        while text:
            max_chars = max(12, int((size[0] - 16) / 7.2))
            chunk, text = text[:max_chars], text[max_chars:]
            draw.text((8, y), chunk, fill=(32, 32, 32), font=body_font)
            y += 16
            if y > size[1] - 18:
                draw.text((8, size[1] - 18), "...", fill=(32, 32, 32), font=body_font)
                return panel
    return panel


def _label(image, label):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = _font(12)
    box_h = 22
    draw.rectangle((0, 0, out.width, box_h), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255), font=font)
    return out


def _resolve_reference(record, scene_root, images="processed"):
    if record.source == "synthetic_rgb" and record.nearest_train_image:
        name = str(record.nearest_train_image)
        candidates = [name]
        base = name.split("__", 1)[0]
        candidates.append(base)
        synthetic_prefixes = (
            "synthetic/",
            "synthetic_interpolate/",
            "synthetic_spatial_offset/",
        )
        for prefix in synthetic_prefixes:
            if base.startswith(prefix):
                candidates.append(base[len(prefix) :])
        for candidate in candidates:
            path = Path(scene_root) / images / candidate
            if path.exists():
                return path
        return Path(scene_root) / images / candidates[0]
    if record.source == "train_rgb":
        return Path(record.image_path)
    return None


def _cache_metric(item, name):
    if not item:
        return None
    value = item.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_records(records, cache, sort_by):
    def key(row):
        item = cache.get(row.teacher_cache_key or row.query_id) if cache else None
        dense_te = _cache_metric(item, "dense_te")
        sparse_te = _cache_metric(item, "te")
        if sort_by == "dense_te_desc":
            return (dense_te is None, -(dense_te or -1.0))
        if sort_by == "sparse_te_desc":
            return (sparse_te is None, -(sparse_te or -1.0))
        if sort_by == "artifact_desc":
            return (False, -float(row.artifact_score))
        if sort_by == "stage":
            return (str(item.get("failure_stage", "")) if item else "", row.query_id)
        return (False, row.query_id)

    return sorted(records, key=key)


def _build_contact_sheet(records, cache, scene_root, images, output_path, max_records=16, sort_by="dense_te_desc"):
    detector = ArtifactDetector()
    records = _sort_records(records, cache, sort_by)[: int(max_records)]
    cell = (240, 160)
    text_cell = (300, 160)
    row_h = cell[1]
    width = cell[0] * 4 + text_cell[0]
    height = max(row_h, row_h * max(1, len(records)))
    sheet = Image.new("RGB", (width, height), (232, 232, 228))

    for row_idx, record in enumerate(records):
        y = row_idx * row_h
        query_path = Path(record.image_path)
        item = cache.get(record.teacher_cache_key or record.query_id) if cache else None
        query = _open_rgb(query_path)
        tensor = _image_to_tensor(query)
        evidence = detector.detect(rendered_rgb=tensor)
        heatmap = _heatmap01(evidence.score_map)
        detail_risk, detail_summary = _low_detail_risk(query)
        detail_heatmap = _heatmap01(detail_risk)
        ref_path = _resolve_reference(record, scene_root, images=images)
        ref = _open_rgb(ref_path) if ref_path and Path(ref_path).exists() else Image.new("RGB", cell, (40, 40, 40))
        query_panel = _label(_fit_canvas(query, cell), "query/rgb teacher render" if record.source == "synthetic_rgb" else "train rgb query")
        ref_panel = _label(_fit_canvas(ref, cell), "nearest train rgb" if record.source == "synthetic_rgb" else "source train rgb")
        heat_panel = _label(_fit_canvas(heatmap, cell), "artifact score heatmap")
        detail_panel = _label(_fit_canvas(detail_heatmap, cell), "low-detail risk map")

        sparse_te = _cache_metric(item, "te")
        dense_te = _cache_metric(item, "dense_te")
        lines = [
            f"id: {record.query_id}",
            f"source: {record.source}",
            f"accepted: {record.accepted} reason={record.reason}",
            f"artifact: manifest={record.artifact_score:.4f} p95={evidence.summary['artifact_score_p95']:.4f}",
            f"low_detail: mean={detail_summary['low_detail_mean']:.4f} p95={detail_summary['low_detail_p95']:.4f}",
            f"repair: {record.repair_action}",
            f"stage: {item.get('failure_stage', '<missing>') if item else '<missing cache>'}",
            f"sparse_te_cm: {sparse_te:.2f}" if sparse_te is not None else "sparse_te_cm: <missing>",
            f"dense_te_cm: {dense_te:.2f}" if dense_te is not None else "dense_te_cm: <missing>",
            f"nearest: {record.nearest_train_image or '<none>'}",
        ]
        text = _text_panel(lines, text_cell, title=f"{row_idx + 1}. {record.source}")

        sheet.paste(query_panel, (0, y))
        sheet.paste(ref_panel, (cell[0], y))
        sheet.paste(heat_panel, (cell[0] * 2, y))
        sheet.paste(detail_panel, (cell[0] * 3, y))
        sheet.paste(text, (cell[0] * 4, y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return str(output_path)


def _gate_summary(manifest, cache, max_sparse_te, max_dense_te, allowed_stages=None):
    accepted = manifest.accepted()
    gated = accepted.filter_by_teacher_cache(
        cache,
        max_sparse_te=max_sparse_te,
        max_dense_te=max_dense_te,
        allowed_stages=allowed_stages,
    )
    cache_stage_counts = Counter()
    source_stage_counts = defaultdict(Counter)
    missing_cache = []
    for row in accepted.records:
        key = row.teacher_cache_key or row.query_id
        item = cache.get(key)
        if not item:
            missing_cache.append(key)
            continue
        stage = str(item.get("failure_stage", "<missing>"))
        cache_stage_counts[stage] += 1
        source_stage_counts[row.source][stage] += 1
    return {
        "manifest_counts": manifest.source_counts(),
        "accepted_counts": accepted.source_counts(),
        "gated_counts": gated.source_counts(),
        "cache_items": len(cache.items),
        "cache_stage_counts": dict(sorted(cache_stage_counts.items())),
        "source_stage_counts": {k: dict(sorted(v.items())) for k, v in sorted(source_stage_counts.items())},
        "missing_cache_count": len(missing_cache),
        "missing_cache_examples": missing_cache[:20],
        "gate": {
            "max_sparse_te_cm": max_sparse_te,
            "max_dense_te_cm": max_dense_te,
            "allowed_stages": allowed_stages or [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize RGB teacher / artifact / pseudo-query cache flow.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sources", default="train_rgb,synthetic_rgb")
    parser.add_argument("--max_records_per_source", type=int, default=12)
    parser.add_argument("--sort_by", choices=["dense_te_desc", "sparse_te_desc", "artifact_desc", "stage", "manifest"], default="dense_te_desc")
    parser.add_argument("--max_sparse_te", type=float, default=100.0)
    parser.add_argument("--max_dense_te", type=float, default=100.0)
    parser.add_argument("--allowed_stages", default="")
    parser.add_argument("--include_rejected", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = PseudoQueryManifest.load(args.manifest)
    cache = PseudoTeacherCache.load(args.teacher_cache) if args.teacher_cache else PseudoTeacherCache()
    allowed_stages = _comma_list(args.allowed_stages)
    sources = _comma_list(args.sources)

    summary = _gate_summary(
        manifest,
        cache,
        max_sparse_te=float(args.max_sparse_te),
        max_dense_te=float(args.max_dense_te),
        allowed_stages=allowed_stages or None,
    )
    outputs = []
    for source in sources:
        records = [
            row
            for row in manifest.records
            if row.source == source and (args.include_rejected or row.accepted)
        ]
        if not records:
            continue
        suffix = "all" if args.include_rejected else "accepted"
        output = out_dir / f"contact_sheet_{source}_{suffix}_{args.sort_by}.jpg"
        outputs.append(
            _build_contact_sheet(
                records,
                cache,
                scene_root=args.scene_root,
                images=args.images,
                output_path=output,
                max_records=args.max_records_per_source,
                sort_by=args.sort_by,
            )
        )

    summary["outputs"] = outputs
    summary_path = out_dir / "visual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
