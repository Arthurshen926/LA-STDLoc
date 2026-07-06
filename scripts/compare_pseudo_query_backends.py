#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache


METRICS = (
    "inliers",
    "dense_inliers",
    "matches",
    "detected_keypoints_raw",
    "detected_keypoints",
    "sparse_valid_mask_filtered_keypoints",
    "sparse_support_score_prior_score_mean",
    "te",
    "dense_te",
    "ae",
    "dense_ae",
)


def _record_key(record):
    return record.teacher_cache_key or record.query_id


def _record_map(path, source):
    manifest = PseudoQueryManifest.load(path)
    rows = {}
    for record in manifest.records:
        if source and record.source != source:
            continue
        rows[_record_key(record)] = record
    return rows


def _to_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _metric_value(item, metric):
    if not item:
        return None
    if metric == "support_frac":
        mask = item.get("sparse_valid_mask") or {}
        return _to_float(mask.get("support_frac"))
    return _to_float(item.get(metric))


def _summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0}
    mid = len(values) // 2
    if len(values) % 2:
        median = values[mid]
    else:
        median = 0.5 * (values[mid - 1] + values[mid])
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(median),
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def _pose_abs_diff(left_record, right_record):
    left = torch.as_tensor(left_record.pose_w2c, dtype=torch.float32)
    right = torch.as_tensor(right_record.pose_w2c, dtype=torch.float32)
    return float((left - right).abs().max().item())


def _backend_summary(label, keys, cache):
    items = {key: cache.get(key) for key in keys if cache.get(key) is not None}
    stage_counts = Counter(str(item.get("failure_stage", "missing")) for item in items.values())
    failed_count = sum(1 for item in items.values() if bool(item.get("failed", False)))
    metrics = {}
    for metric in METRICS + ("support_frac",):
        metrics[metric] = _summary([_metric_value(item, metric) for item in items.values()])
    return {
        "label": label,
        "count": len(items),
        "missing_cache_count": int(len(keys) - len(items)),
        "failed_count": int(failed_count),
        "stage_counts": dict(sorted(stage_counts.items())),
        "metrics": metrics,
    }


def _pairwise_deltas(keys, left_cache, right_cache):
    deltas = {}
    for metric in METRICS + ("support_frac",):
        values = []
        for key in keys:
            left_value = _metric_value(left_cache.get(key), metric)
            right_value = _metric_value(right_cache.get(key), metric)
            if left_value is not None and right_value is not None:
                values.append(float(right_value - left_value))
        deltas[metric] = _summary(values)
    return deltas


def _pairwise_record_summary(key, left_cache, right_cache):
    left_item = left_cache.get(key) or {}
    right_item = right_cache.get(key) or {}
    metrics = {}
    for metric in METRICS + ("support_frac",):
        left_value = _metric_value(left_item, metric)
        right_value = _metric_value(right_item, metric)
        if left_value is None or right_value is None:
            continue
        metrics[metric] = {
            "left": float(left_value),
            "right": float(right_value),
            "delta": float(right_value - left_value),
        }
    return {
        "key": key,
        "left_stage": str(left_item.get("failure_stage", "missing")),
        "right_stage": str(right_item.get("failure_stage", "missing")),
        "metrics": metrics,
    }


def _example_from_record(record, metric):
    metric_values = record["metrics"].get(metric) or {}
    return {
        "key": record["key"],
        "left_stage": record["left_stage"],
        "right_stage": record["right_stage"],
        "left": metric_values.get("left"),
        "right": metric_values.get("right"),
        "delta": metric_values.get("delta"),
    }


def _top_pairwise_examples(records, limit=8):
    def examples(metric, reverse):
        eligible = [record for record in records if metric in record["metrics"]]
        eligible.sort(key=lambda record: record["metrics"][metric]["delta"], reverse=reverse)
        return [_example_from_record(record, metric) for record in eligible[: int(limit)]]

    return {
        "right_te_improves_most": examples("te", reverse=False),
        "right_te_worsens_most": examples("te", reverse=True),
        "right_dense_te_improves_most": examples("dense_te", reverse=False),
        "right_dense_te_worsens_most": examples("dense_te", reverse=True),
        "right_inliers_improves_most": examples("inliers", reverse=True),
        "right_inliers_worsens_most": examples("inliers", reverse=False),
        "right_dense_inliers_improves_most": examples("dense_inliers", reverse=True),
        "right_dense_inliers_worsens_most": examples("dense_inliers", reverse=False),
    }


def compare_backend_runs(
    left_label,
    left_manifest,
    left_cache,
    right_label,
    right_manifest,
    right_cache,
    source="synthetic_rgb",
    pose_tolerance=1e-5,
):
    left_records = _record_map(left_manifest, source)
    right_records = _record_map(right_manifest, source)
    left_cache_items = PseudoTeacherCache.load(left_cache).items
    right_cache_items = PseudoTeacherCache.load(right_cache).items
    common_keys = sorted(set(left_records) & set(right_records))
    pose_diffs = [_pose_abs_diff(left_records[key], right_records[key]) for key in common_keys]
    max_pose_diff = max(pose_diffs) if pose_diffs else 0.0
    cached_common_keys = [
        key for key in common_keys if key in left_cache_items and key in right_cache_items
    ]
    pairwise_records = [
        _pairwise_record_summary(key, left_cache_items, right_cache_items)
        for key in cached_common_keys
    ]
    return {
        "source": source,
        "left_label": left_label,
        "right_label": right_label,
        "manifest_counts": {
            left_label: len(left_records),
            right_label: len(right_records),
            "common": len(common_keys),
            f"missing_in_{left_label}": len(set(right_records) - set(left_records)),
            f"missing_in_{right_label}": len(set(left_records) - set(right_records)),
        },
        "pose_alignment": {
            "same_pose_all": bool(all(diff <= float(pose_tolerance) for diff in pose_diffs)),
            "pose_tolerance": float(pose_tolerance),
            "max_pose_abs_diff": float(max_pose_diff),
            "mean_pose_abs_diff": float(sum(pose_diffs) / len(pose_diffs)) if pose_diffs else 0.0,
            "mismatch_count": int(sum(diff > float(pose_tolerance) for diff in pose_diffs)),
        },
        "cache_common_count": len(cached_common_keys),
        "backends": {
            left_label: _backend_summary(left_label, common_keys, left_cache_items),
            right_label: _backend_summary(right_label, common_keys, right_cache_items),
        },
        "pairwise_delta_right_minus_left": _pairwise_deltas(cached_common_keys, left_cache_items, right_cache_items),
        "pairwise_top_examples": _top_pairwise_examples(pairwise_records),
    }


def _image_or_placeholder(path, size, label):
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", size, color=(245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"missing\n{label}", fill=(20, 20, 20))
        return image
    image.thumbnail(size)
    canvas = Image.new("RGB", size, color=(255, 255, 255))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_contact_sheet(
    left_label,
    left_manifest,
    left_cache,
    right_label,
    right_manifest,
    right_cache,
    output,
    source="synthetic_rgb",
    max_rows=16,
    cell_size=(360, 220),
):
    left_records = _record_map(left_manifest, source)
    right_records = _record_map(right_manifest, source)
    left_cache_items = PseudoTeacherCache.load(left_cache).items
    right_cache_items = PseudoTeacherCache.load(right_cache).items
    keys = sorted(set(left_records) & set(right_records))[: int(max_rows)]
    header_h = 36
    label_h = 64
    gap = 12
    width = cell_size[0] * 2 + gap
    row_h = header_h + cell_size[1] + label_h + gap
    height = max(row_h, row_h * len(keys))
    sheet = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row_idx, key in enumerate(keys):
        y0 = row_idx * row_h
        draw.text((6, y0 + 6), key, fill=(0, 0, 0))
        for col, (label, records, cache) in enumerate(
            (
                (left_label, left_records, left_cache_items),
                (right_label, right_records, right_cache_items),
            )
        ):
            x0 = col * (cell_size[0] + gap)
            image = _image_or_placeholder(records[key].image_path, cell_size, label)
            sheet.paste(image, (x0, y0 + header_h))
            item = cache.get(key) or {}
            text = (
                f"{label} | {item.get('failure_stage', 'missing')}\n"
                f"inliers {item.get('inliers', 'na')} dense {item.get('dense_inliers', 'na')} "
                f"te {item.get('te', 'na')} dense_te {item.get('dense_te', 'na')}"
            )
            draw.text((x0 + 6, y0 + header_h + cell_size[1] + 6), text, fill=(0, 0, 0))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return os.path.abspath(os.fspath(output))


def main():
    parser = argparse.ArgumentParser(description="Compare two pseudo-query RGB backends on identical records.")
    parser.add_argument("--left_label", required=True)
    parser.add_argument("--left_manifest", required=True)
    parser.add_argument("--left_cache", required=True)
    parser.add_argument("--right_label", required=True)
    parser.add_argument("--right_manifest", required=True)
    parser.add_argument("--right_cache", required=True)
    parser.add_argument("--source", default="synthetic_rgb")
    parser.add_argument("--pose_tolerance", type=float, default=1e-5)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--contact_sheet", default="")
    parser.add_argument("--max_visual_rows", type=int, default=16)
    args = parser.parse_args()
    summary = compare_backend_runs(
        left_label=args.left_label,
        left_manifest=args.left_manifest,
        left_cache=args.left_cache,
        right_label=args.right_label,
        right_manifest=args.right_manifest,
        right_cache=args.right_cache,
        source=args.source,
        pose_tolerance=args.pose_tolerance,
    )
    if args.contact_sheet:
        summary["contact_sheet"] = write_contact_sheet(
            left_label=args.left_label,
            left_manifest=args.left_manifest,
            left_cache=args.left_cache,
            right_label=args.right_label,
            right_manifest=args.right_manifest,
            right_cache=args.right_cache,
            output=args.contact_sheet,
            source=args.source,
            max_rows=args.max_visual_rows,
        )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
