#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _float_or_default(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _float_or_none(value):
    return _float_or_default(value, None)


def _artifact_score(record):
    meta = getattr(record, "meta", {}) or {}
    summary = meta.get("artifact_summary") or {}
    return _float_or_default(summary.get("artifact_score_mean", record.artifact_score), float("inf"))


def _cache_item(cache, record):
    if cache is None:
        return None
    return cache.get(record.teacher_cache_key or record.query_id)


def _support_score(item):
    item = item or {}
    mask = item.get("sparse_valid_mask") or {}
    return (
        _float_or_default(item.get("sparse_support_score_prior_score_mean"), float("-inf")),
        _float_or_default(mask.get("support_frac"), float("-inf")),
    )


def _below_support_threshold(record, cache, min_support_frac=0.0, min_support_score=-1.0):
    min_support_frac = float(min_support_frac or 0.0)
    min_support_score = float(min_support_score if min_support_score is not None else -1.0)
    if min_support_frac <= 0.0 and min_support_score < 0.0:
        return False
    item = _cache_item(cache, record)
    if not item:
        return False
    support_mean, support_frac = _support_score(item)
    support_mean = _float_or_none(support_mean)
    support_frac = _float_or_none(support_frac)
    if min_support_frac > 0.0 and support_frac is not None and support_frac < min_support_frac:
        return True
    if min_support_score >= 0.0 and support_mean is not None and support_mean < min_support_score:
        return True
    return False


def _sort_key(record, cache, sort_by):
    item = _cache_item(cache, record) or {}
    dense_te = _float_or_default(item.get("dense_te"), float("inf"))
    sparse_te = _float_or_default(item.get("te"), float("inf"))
    artifact = _artifact_score(record)
    if sort_by == "artifact":
        primary = (artifact,)
    elif sort_by == "support":
        support_mean, support_frac = _support_score(item)
        primary = (-support_mean, -support_frac, artifact)
    elif sort_by == "sparse_te":
        primary = (sparse_te, dense_te, artifact)
    else:
        primary = (dense_te, sparse_te, artifact)
    return (*primary, str(record.image_name))


def _selection_meta(record, cache, selected, rank=None):
    item = _cache_item(cache, record) or {}
    support_mean, support_frac = _support_score(item)
    return {
        "selected": bool(selected),
        "rank": int(rank) if rank is not None else None,
        "artifact_score_mean": _artifact_score(record),
        "support_score_mean": _float_or_none(support_mean),
        "support_frac": _float_or_none(support_frac),
        "sparse_te_cm": _float_or_none(item.get("te")),
        "dense_te_cm": _float_or_none(item.get("dense_te")),
        "failure_stage": str(item.get("failure_stage", "")),
    }


def select_pseudo_query_pool(
    manifest,
    cache=None,
    max_synthetic=0,
    synthetic_sources=("synthetic_rgb",),
    sort_by="artifact",
    min_support_frac=0.0,
    min_support_score=-1.0,
):
    synthetic_sources = {str(item) for item in synthetic_sources if str(item)}
    max_synthetic = int(max_synthetic or 0)
    input_counts = manifest.source_counts()
    synthetic = []
    support_rejected_synthetic = []
    already_rejected_synthetic = []
    for row in manifest.records:
        if row.source not in synthetic_sources:
            continue
        row.meta = getattr(row, "meta", {}) or {}
        if not row.accepted:
            already_rejected_synthetic.append(row)
            continue
        if _below_support_threshold(
            row,
            cache,
            min_support_frac=min_support_frac,
            min_support_score=min_support_score,
        ):
            row.accepted = False
            row.reason = "synthetic_pool_low_support"
            row.meta["synthetic_pool_selection"] = _selection_meta(row, cache, selected=False)
            support_rejected_synthetic.append(row)
            continue
        synthetic.append(row)
    selected_synthetic = sorted(synthetic, key=lambda row: _sort_key(row, cache, sort_by))
    if max_synthetic > 0:
        selected_synthetic = selected_synthetic[:max_synthetic]
    selected_ids = {row.query_id for row in selected_synthetic}

    non_synthetic = [row for row in manifest.records if row.source not in synthetic_sources]
    rejected_synthetic = []
    for row in synthetic:
        if row.query_id in selected_ids:
            continue
        row.accepted = False
        row.reason = "synthetic_pool_not_selected"
        row.meta["synthetic_pool_selection"] = _selection_meta(row, cache, selected=False)
        rejected_synthetic.append(row)

    for rank, row in enumerate(selected_synthetic, start=1):
        row.meta = getattr(row, "meta", {}) or {}
        row.meta["synthetic_pool_selection"] = _selection_meta(row, cache, selected=True, rank=rank)

    records = (
        list(non_synthetic)
        + selected_synthetic
        + rejected_synthetic
        + support_rejected_synthetic
        + already_rejected_synthetic
    )
    selected = PseudoQueryManifest(version=manifest.version, records=records)
    summary = {
        "input_counts": input_counts,
        "output_counts": selected.source_counts(),
        "accepted_counts": selected.accepted().source_counts(),
        "max_synthetic": max_synthetic,
        "sort_by": sort_by,
        "min_support_frac": float(min_support_frac or 0.0),
        "min_support_score": float(min_support_score if min_support_score is not None else -1.0),
        "synthetic_sources": sorted(synthetic_sources),
        "synthetic_candidates": len(synthetic) + len(support_rejected_synthetic),
        "synthetic_selected": len(selected_synthetic),
        "synthetic_rejected_by_cap": len(rejected_synthetic),
        "synthetic_rejected_by_support": len(support_rejected_synthetic),
        "synthetic_already_rejected": len(already_rejected_synthetic),
    }
    return selected, summary


def select_pseudo_query_pool_file(
    manifest_path,
    output_path,
    summary_json="",
    teacher_cache_path="",
    max_synthetic=0,
    synthetic_sources=("synthetic_rgb",),
    sort_by="artifact",
    min_support_frac=0.0,
    min_support_score=-1.0,
):
    manifest = PseudoQueryManifest.load(manifest_path)
    cache = PseudoTeacherCache.load(teacher_cache_path) if teacher_cache_path else PseudoTeacherCache()
    selected, summary = select_pseudo_query_pool(
        manifest,
        cache=cache,
        max_synthetic=max_synthetic,
        synthetic_sources=synthetic_sources,
        sort_by=sort_by,
        min_support_frac=min_support_frac,
        min_support_score=min_support_score,
    )
    selected.save_jsonl(output_path)
    summary = {
        **summary,
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "output": os.path.abspath(os.fspath(output_path)),
        "teacher_cache": os.path.abspath(os.fspath(teacher_cache_path)) if teacher_cache_path else "",
    }
    if summary_json:
        path = Path(summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Select a capped pseudo-query training pool after render QA.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--max_synthetic", type=int, default=0, help="0 keeps all accepted synthetic records.")
    parser.add_argument("--synthetic_sources", default="synthetic_rgb")
    parser.add_argument("--sort_by", choices=["artifact", "support", "dense_te", "sparse_te"], default="artifact")
    parser.add_argument("--min_support_frac", type=float, default=float(os.environ.get("PSEUDO_QUERY_MIN_SUPPORT_FRAC", "0.0")))
    parser.add_argument("--min_support_score", type=float, default=float(os.environ.get("PSEUDO_QUERY_MIN_SUPPORT_SCORE", "-1.0")))
    args = parser.parse_args()
    summary = select_pseudo_query_pool_file(
        args.manifest,
        args.output,
        summary_json=args.summary_json,
        teacher_cache_path=args.teacher_cache,
        max_synthetic=args.max_synthetic,
        synthetic_sources=_comma_list(args.synthetic_sources),
        sort_by=args.sort_by,
        min_support_frac=args.min_support_frac,
        min_support_score=args.min_support_score,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
