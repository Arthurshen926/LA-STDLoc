import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


SAMPLE_FLOW_FIELDS = [
    "group",
    "stage",
    "batch_name",
    "scene",
    "split",
    "image_name",
    "image_path",
    "sparse_te",
    "sparse_ae",
    "baseline_sparse_te",
    "baseline_sparse_ae",
    "delta_te",
    "delta_ae",
    "dense_te",
    "dense_ae",
    "inliers",
    "gate_severity",
    "psnr",
    "psnr_mean_matched",
    "ssim",
    "residual_frac_025",
    "alpha_cov_05",
    "mean_abs_bias",
    "region_weight_path",
    "region_weight_min",
    "region_weight_mean",
    "region_weight_weighted_frac",
]


@dataclass
class BatchInputs:
    batch_name: str
    scene: str
    current_results: Path
    output_dir: Path
    baseline_results: Optional[Path] = None
    artifact_audit_csv: Optional[Path] = None
    region_manifest_csv: Optional[Path] = None
    region_weight_root: Optional[Path] = None
    image_root: Optional[Path] = None
    registry_path: Optional[Path] = None
    top_k: int = 8
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)


def generate_qualitative_report(inputs: BatchInputs) -> Dict[str, object]:
    """Generate a qualitative LA-STDLoc diagnostics report for one batch."""
    inputs = _normalize_inputs(inputs)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    current_rows = _load_results(inputs.current_results)
    baseline_rows = _load_results(inputs.baseline_results) if inputs.baseline_results else []
    selected_final_rows, final_summary = _select_final_test_rows(inputs, current_rows, baseline_rows)

    audit_rows = _read_csv(inputs.artifact_audit_csv) if inputs.artifact_audit_csv else []
    region_index = _load_region_manifest(inputs.region_manifest_csv) if inputs.region_manifest_csv else {}
    selected_artifact_rows, artifact_summary = _select_artifact_rows(inputs, audit_rows, region_index)

    sample_rows = selected_final_rows + selected_artifact_rows
    sample_flow_path = inputs.output_dir / "sample_flow.csv"
    _write_sample_flow(sample_flow_path, sample_rows)

    final_worst_path = inputs.output_dir / "final_test_worst.png"
    final_delta_path = inputs.output_dir / "final_test_improved_regressed.png"
    artifact_path = inputs.output_dir / "artifact_teacher_severe.png"
    artifact_flagged_path = inputs.output_dir / "artifact_teacher_flagged.png"
    _write_contact_sheet(
        [row for row in sample_rows if row["group"] == "final_worst"],
        final_worst_path,
        title="Final test worst sparse pose",
        inputs=inputs,
    )
    _write_contact_sheet(
        [row for row in sample_rows if row["group"] in {"final_improved", "final_regressed"}],
        final_delta_path,
        title="Final test improved / regressed",
        inputs=inputs,
    )
    _write_contact_sheet(
        [row for row in sample_rows if row["group"] == "artifact_severe"],
        artifact_path,
        title="Teacher render severe artifacts",
        inputs=inputs,
        include_region_heatmap=True,
    )
    _write_contact_sheet(
        [row for row in sample_rows if row["group"] in {"artifact_severe", "artifact_mild"}],
        artifact_flagged_path,
        title="Teacher render flagged artifacts",
        inputs=inputs,
        include_region_heatmap=True,
    )

    summary = {
        "batch_name": inputs.batch_name,
        "scene": inputs.scene,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": inputs.notes,
        "metadata": dict(inputs.metadata),
        "inputs": {
            "current_results": _path_str(inputs.current_results),
            "baseline_results": _path_str(inputs.baseline_results),
            "artifact_audit_csv": _path_str(inputs.artifact_audit_csv),
            "region_manifest_csv": _path_str(inputs.region_manifest_csv),
            "region_weight_root": _path_str(inputs.region_weight_root),
            "image_root": _path_str(inputs.image_root),
        },
        "outputs": {
            "output_dir": _path_str(inputs.output_dir),
            "sample_flow_csv": _path_str(sample_flow_path),
            "index_md": _path_str(inputs.output_dir / "index.md"),
            "summary_json": _path_str(inputs.output_dir / "summary.json"),
            "final_test_worst_png": _path_str(final_worst_path),
            "final_test_improved_regressed_png": _path_str(final_delta_path),
            "artifact_teacher_severe_png": _path_str(artifact_path),
            "artifact_teacher_flagged_png": _path_str(artifact_flagged_path),
            "registry_jsonl": _path_str(inputs.registry_path),
        },
        "final_test": final_summary,
        "artifact_teacher": artifact_summary,
        "sample_rows": len(sample_rows),
    }
    _write_json(inputs.output_dir / "summary.json", summary)
    _write_index(inputs.output_dir / "index.md", summary, sample_rows)
    if inputs.registry_path:
        _append_registry(inputs.registry_path, summary)
    return summary


def _normalize_inputs(inputs: BatchInputs) -> BatchInputs:
    return BatchInputs(
        batch_name=inputs.batch_name,
        scene=inputs.scene,
        current_results=Path(inputs.current_results),
        output_dir=Path(inputs.output_dir),
        baseline_results=Path(inputs.baseline_results) if inputs.baseline_results else None,
        artifact_audit_csv=Path(inputs.artifact_audit_csv) if inputs.artifact_audit_csv else None,
        region_manifest_csv=Path(inputs.region_manifest_csv) if inputs.region_manifest_csv else None,
        region_weight_root=Path(inputs.region_weight_root) if inputs.region_weight_root else None,
        image_root=Path(inputs.image_root) if inputs.image_root else None,
        registry_path=Path(inputs.registry_path) if inputs.registry_path else Path(inputs.output_dir) / "registry.jsonl",
        top_k=max(1, int(inputs.top_k)),
        notes=inputs.notes,
        metadata=inputs.metadata,
    )


def _path_str(path: Optional[Path]) -> Optional[str]:
    return str(path) if path else None


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_results(path: Optional[Path]) -> List[Mapping[str, object]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a result list in {path}.")
    return data


def _read_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_region_manifest(path: Optional[Path]) -> Dict[Tuple[str, str], Mapping[str, str]]:
    rows = _read_csv(path)
    index: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        index[(row.get("split", ""), row.get("image_name", ""))] = row
    return index


def _select_final_test_rows(
    inputs: BatchInputs,
    current_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    baseline_by_name = {str(row.get("image_name", "")): row for row in baseline_rows}
    enriched = []
    for row in current_rows:
        image_name = str(row.get("image_name", ""))
        if not image_name:
            continue
        baseline = baseline_by_name.get(image_name)
        sparse_te = _metric(row, "sparse_TE", "sparse_te")
        sparse_ae = _metric(row, "sparse_AE", "sparse_ae")
        baseline_te = _metric(baseline, "sparse_TE", "sparse_te") if baseline else None
        baseline_ae = _metric(baseline, "sparse_AE", "sparse_ae") if baseline else None
        enriched.append(
            {
                "image_name": image_name,
                "row": row,
                "baseline": baseline,
                "sparse_te": sparse_te,
                "sparse_ae": sparse_ae,
                "baseline_sparse_te": baseline_te,
                "baseline_sparse_ae": baseline_ae,
                "delta_te": baseline_te - sparse_te if baseline_te is not None and sparse_te is not None else None,
                "delta_ae": baseline_ae - sparse_ae if baseline_ae is not None and sparse_ae is not None else None,
            }
        )

    selected: List[Dict[str, object]] = []
    worst = _take_sorted(enriched, key=lambda item: _sort_float(item["sparse_te"]), reverse=True, limit=inputs.top_k)
    selected.extend(_final_record(inputs, item, "final_worst") for item in worst)

    paired = [item for item in enriched if item["delta_te"] is not None]
    improved = _take_sorted(paired, key=lambda item: _sort_float(item["delta_te"]), reverse=True, limit=inputs.top_k)
    regressed = _take_sorted(paired, key=lambda item: _sort_float(item["delta_te"]), reverse=False, limit=inputs.top_k)
    selected.extend(_final_record(inputs, item, "final_improved") for item in improved if item["delta_te"] and item["delta_te"] > 0)
    selected.extend(_final_record(inputs, item, "final_regressed") for item in regressed if item["delta_te"] and item["delta_te"] < 0)

    te_values = [item["sparse_te"] for item in enriched if item["sparse_te"] is not None]
    ae_values = [item["sparse_ae"] for item in enriched if item["sparse_ae"] is not None]
    delta_values = [item["delta_te"] for item in enriched if item["delta_te"] is not None]
    summary = {
        "result_count": len(enriched),
        "paired_count": len(paired),
        "median_sparse_te": _median_or_none(te_values),
        "median_sparse_ae": _median_or_none(ae_values),
        "median_delta_te": _median_or_none(delta_values),
        "worst_count": len(worst),
        "improved_count": sum(1 for item in improved if item["delta_te"] and item["delta_te"] > 0),
        "regressed_count": sum(1 for item in regressed if item["delta_te"] and item["delta_te"] < 0),
    }
    return selected, summary


def _select_artifact_rows(
    inputs: BatchInputs,
    audit_rows: Sequence[Mapping[str, str]],
    region_index: Mapping[Tuple[str, str], Mapping[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    severity_counts: Dict[str, int] = {}
    enriched = []
    for row in audit_rows:
        severity = row.get("gate_severity", "") or "unknown"
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        split = row.get("split", "")
        image_name = row.get("image_name", "")
        region_row = region_index.get((split, image_name), {})
        merged = dict(row)
        merged.update({k: v for k, v in region_row.items() if v not in (None, "")})
        enriched.append(merged)

    severe = [row for row in enriched if row.get("gate_severity") == "severe"]
    severe = _take_sorted(
        severe,
        key=lambda row: (
            _sort_float(_float(row.get("psnr_mean_matched") or row.get("psnr")), missing_high=True),
            -_sort_float(_float(row.get("residual_frac_025")), missing_high=False),
        ),
        reverse=False,
        limit=inputs.top_k,
    )
    mild = [row for row in enriched if row.get("gate_severity") == "mild"]
    mild = _take_sorted(
        mild,
        key=lambda row: (
            _sort_float(_float(row.get("psnr_mean_matched") or row.get("psnr")), missing_high=True),
            -_sort_float(_float(row.get("residual_frac_025")), missing_high=False),
        ),
        reverse=False,
        limit=inputs.top_k,
    )
    selected = [_artifact_record(inputs, row, "artifact_severe") for row in severe]
    selected.extend(_artifact_record(inputs, row, "artifact_mild") for row in mild)
    summary = {
        "audit_count": len(audit_rows),
        "severity_counts": severity_counts,
        "selected_severe_count": len(severe),
        "selected_mild_count": len(mild),
    }
    return selected, summary


def _final_record(inputs: BatchInputs, item: Mapping[str, object], group: str) -> Dict[str, object]:
    row = item["row"]
    image_name = str(item["image_name"])
    sparse = row.get("sparse") if isinstance(row, Mapping) else None
    inliers = sparse.get("inliers") if isinstance(sparse, Mapping) else row.get("inliers")
    return _blank_record(
        {
            "group": group,
            "stage": "final_test",
            "batch_name": inputs.batch_name,
            "scene": inputs.scene,
            "split": "final_test",
            "image_name": image_name,
            "image_path": _path_str(_resolve_image_path(inputs.image_root, inputs.scene, image_name)),
            "sparse_te": item["sparse_te"],
            "sparse_ae": item["sparse_ae"],
            "baseline_sparse_te": item["baseline_sparse_te"],
            "baseline_sparse_ae": item["baseline_sparse_ae"],
            "delta_te": item["delta_te"],
            "delta_ae": item["delta_ae"],
            "dense_te": _metric(row, "dense_TE", "dense_te"),
            "dense_ae": _metric(row, "dense_AE", "dense_ae"),
            "inliers": inliers,
        }
    )


def _artifact_record(inputs: BatchInputs, row: Mapping[str, str], group: str) -> Dict[str, object]:
    image_name = row.get("image_name", "")
    split = row.get("split", "")
    region_weight_path = row.get("region_weight_path", "")
    stats = _region_stats(inputs.region_weight_root, region_weight_path)
    return _blank_record(
        {
            "group": group,
            "stage": "artifact_teacher",
            "batch_name": inputs.batch_name,
            "scene": inputs.scene,
            "split": split,
            "image_name": image_name,
            "image_path": _path_str(_resolve_image_path(inputs.image_root, inputs.scene, image_name)),
            "gate_severity": row.get("gate_severity", ""),
            "psnr": _float(row.get("psnr")),
            "psnr_mean_matched": _float(row.get("psnr_mean_matched")),
            "ssim": _float(row.get("ssim")),
            "residual_frac_025": _float(row.get("residual_frac_025")),
            "alpha_cov_05": _float(row.get("alpha_cov_05")),
            "mean_abs_bias": _float(row.get("mean_abs_bias")),
            "region_weight_path": region_weight_path,
            "region_weight_min": _float(row.get("region_weight_min")) if row.get("region_weight_min") else stats.get("min"),
            "region_weight_mean": _float(row.get("region_weight_mean")) if row.get("region_weight_mean") else stats.get("mean"),
            "region_weight_weighted_frac": _float(row.get("region_weight_weighted_frac"))
            if row.get("region_weight_weighted_frac")
            else stats.get("weighted_frac"),
        }
    )


def _blank_record(values: Mapping[str, object]) -> Dict[str, object]:
    record = {field: "" for field in SAMPLE_FLOW_FIELDS}
    for key, value in values.items():
        if key in record and value is not None:
            record[key] = value
    return record


def _write_sample_flow(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FLOW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in SAMPLE_FLOW_FIELDS})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9g}"
    return value


def _write_index(path: Path, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        f"# LA Qualitative Diagnostics: {summary['batch_name']}",
        "",
        f"- Scene: `{summary['scene']}`",
        f"- Notes: {summary.get('notes') or ''}",
        f"- Sample rows: {summary['sample_rows']}",
        f"- Final test paired count: {summary['final_test']['paired_count']}",
        f"- Artifact severity counts: `{summary['artifact_teacher']['severity_counts']}`",
        "",
        "## Outputs",
        "",
    ]
    for name, value in summary["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Selected Samples", ""])
    lines.append(
        "| group | stage | image | TE | baseline TE | delta TE | severity | psnr | region mean |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---:|---:|")
    for row in rows[:80]:
        lines.append(
            "| {group} | {stage} | `{image}` | {te} | {bte} | {dte} | {sev} | {psnr} | {rmean} |".format(
                group=row.get("group", ""),
                stage=row.get("stage", ""),
                image=row.get("image_name", ""),
                te=_md_value(row.get("sparse_te", "")),
                bte=_md_value(row.get("baseline_sparse_te", "")),
                dte=_md_value(row.get("delta_te", "")),
                sev=row.get("gate_severity", ""),
                psnr=_md_value(row.get("psnr_mean_matched") or row.get("psnr") or ""),
                rmean=_md_value(row.get("region_weight_mean", "")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_registry(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "batch_name": summary["batch_name"],
        "scene": summary["scene"],
        "created_at": summary["created_at"],
        "notes": summary["notes"],
        "metadata": summary["metadata"],
        "inputs": summary["inputs"],
        "outputs": summary["outputs"],
        "final_test": summary["final_test"],
        "artifact_teacher": summary["artifact_teacher"],
    }
    records = []
    replaced = False
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if existing.get("batch_name") == record["batch_name"]:
                records.append(record)
                replaced = True
            else:
                records.append(existing)
    if not replaced:
        records.append(record)
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n", encoding="utf-8")


def _write_contact_sheet(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    title: str,
    inputs: BatchInputs,
    include_region_heatmap: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 360, 260
    cols = 2 if len(rows) <= 4 else 3
    cols = max(1, cols)
    rows_n = max(1, math.ceil(max(1, len(rows)) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h + 36), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(10, 10, 10))
    if not rows:
        draw.text((10, 50), "No samples selected.", fill=(80, 80, 80))
        sheet.save(output_path)
        return

    for idx, row in enumerate(rows):
        x = (idx % cols) * tile_w
        y = 36 + (idx // cols) * tile_h
        _draw_tile(sheet, x, y, tile_w, tile_h, row, inputs, include_region_heatmap)
    sheet.save(output_path)


def _draw_tile(
    sheet: Image.Image,
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
    row: Mapping[str, object],
    inputs: BatchInputs,
    include_region_heatmap: bool,
) -> None:
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x + 4, y + 4, x + tile_w - 4, y + tile_h - 4), outline=(210, 210, 210), width=1)
    image_path = row.get("image_path") or _path_str(_resolve_image_path(inputs.image_root, inputs.scene, str(row.get("image_name", ""))))
    panel_w = 160 if include_region_heatmap else 330
    panel_h = 150
    if image_path and Path(str(image_path)).exists():
        image = Image.open(str(image_path)).convert("RGB")
        image.thumbnail((panel_w, panel_h))
        sheet.paste(image, (x + 12, y + 12))
    else:
        draw.rectangle((x + 12, y + 12, x + 12 + panel_w, y + 12 + panel_h), fill=(225, 225, 225))
        draw.text((x + 20, y + 72), "image missing", fill=(100, 100, 100))

    if include_region_heatmap:
        heatmap = _load_region_heatmap(inputs.region_weight_root, str(row.get("region_weight_path", "")))
        if heatmap is not None:
            heatmap.thumbnail((160, panel_h))
            sheet.paste(heatmap, (x + 188, y + 12))
        else:
            draw.rectangle((x + 188, y + 12, x + 348, y + 12 + panel_h), fill=(235, 235, 235))
            draw.text((x + 198, y + 72), "heatmap missing", fill=(100, 100, 100))

    text_y = y + 170
    label = _sample_label(row)
    for line in _wrap_text(label, width=44)[:5]:
        draw.text((x + 12, text_y), line, fill=(20, 20, 20))
        text_y += 15


def _sample_label(row: Mapping[str, object]) -> str:
    if row.get("stage") == "final_test":
        return (
            f"{row.get('group')} {row.get('image_name')} "
            f"TE={_label_value(row.get('sparse_te'))} "
            f"base={_label_value(row.get('baseline_sparse_te'))} "
            f"dTE={_label_value(row.get('delta_te'))}"
        )
    return (
        f"{row.get('group')} {row.get('image_name')} "
        f"sev={row.get('gate_severity')} psnr={_label_value(row.get('psnr_mean_matched') or row.get('psnr'))} "
        f"res={_label_value(row.get('residual_frac_025'))} rw={_label_value(row.get('region_weight_mean'))}"
    )


def _wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _resolve_image_path(image_root: Optional[Path], scene: str, image_name: str) -> Optional[Path]:
    if not image_root or not image_name:
        return None
    candidates = [
        image_root / image_name,
        image_root / scene / image_name,
        image_root / "processed" / image_name,
        image_root / "processed_ulfrepro1920" / image_name,
        image_root / scene / "processed" / image_name,
        image_root / scene / "processed_ulfrepro1920" / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _region_stats(region_weight_root: Optional[Path], region_weight_path: str) -> Dict[str, float]:
    array = _load_region_array(region_weight_root, region_weight_path)
    if array is None:
        return {}
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "weighted_frac": float(np.mean(array < 0.999)),
    }


def _load_region_heatmap(region_weight_root: Optional[Path], region_weight_path: str) -> Optional[Image.Image]:
    array = _load_region_array(region_weight_root, region_weight_path)
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        return None
    array = np.clip(array, 0.0, 1.0)
    red = ((1.0 - array) * 255).astype(np.uint8)
    green = (array * 200).astype(np.uint8)
    blue = np.full_like(red, 60, dtype=np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB").resize((320, 180), resample=Image.NEAREST)


def _load_region_array(region_weight_root: Optional[Path], region_weight_path: str) -> Optional[np.ndarray]:
    if not region_weight_root or not region_weight_path:
        return None
    path = region_weight_root / region_weight_path
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".pt":
        try:
            import torch

            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(path, map_location="cpu")
            return _tensor_payload_to_array(value)
        except Exception:
            return None
    return None


def _tensor_payload_to_array(value: object) -> Optional[np.ndarray]:
    if isinstance(value, Mapping):
        for key in ("weight", "weights", "region_weight", "artifact_weight"):
            if key in value:
                return _tensor_payload_to_array(value[key])
        for nested in value.values():
            array = _tensor_payload_to_array(nested)
            if array is not None:
                return array
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        return None
    return array


def _metric(row: Optional[Mapping[str, object]], *keys: str) -> Optional[float]:
    if not row:
        return None
    for key in keys:
        if key in row:
            return _float(row.get(key))
    return None


def _float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _sort_float(value: object, missing_high: bool = False) -> float:
    number = _float(value)
    if number is None:
        return float("inf") if missing_high else float("-inf")
    return number


def _take_sorted(items: Iterable[object], key, reverse: bool, limit: int) -> List[object]:
    return list(sorted(items, key=key, reverse=reverse))[:limit]


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def _md_value(value: object) -> str:
    if value in (None, ""):
        return ""
    number = _float(value)
    if number is None:
        return str(value)
    return f"{number:.4g}"


def _label_value(value: object) -> str:
    if value in (None, ""):
        return "na"
    number = _float(value)
    if number is None:
        return str(value)
    return f"{number:.3g}"
