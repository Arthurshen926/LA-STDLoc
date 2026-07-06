#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache
from la_artifacts.quality_gate import (
    SyntheticQualityGate,
    SyntheticQualityGateConfig,
    TeacherCacheGate,
    TeacherCacheGateConfig,
    summarize_gate_decisions,
)


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _limit(value):
    if value is None:
        return None
    value = float(value)
    return None if value < 0 else value


def _record_key(record):
    return record.teacher_cache_key or record.query_id


def _image_artifact_summary(path):
    import numpy as np
    import torch
    from PIL import Image

    from la_artifacts.detector import ArtifactDetector

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ArtifactDetector().detect(rendered_rgb=tensor).summary


def _artifact_summary_for_record(record, recompute_missing=True):
    meta = getattr(record, "meta", {}) or {}
    summary = dict(meta.get("artifact_summary") or {})
    if summary.get("artifact_score_mean") is not None:
        return summary
    if recompute_missing and record.image_path and os.path.exists(record.image_path):
        return _image_artifact_summary(record.image_path)
    if record.artifact_score is not None:
        summary["artifact_score_mean"] = float(record.artifact_score)
    return summary


def gate_manifest_file(
    manifest_path,
    output_path,
    summary_json="",
    teacher_cache_path="",
    synthetic_qa=True,
    synthetic_qa_max_mean=0.60,
    synthetic_qa_max_p95=-1.0,
    synthetic_qa_max_mild_frac=0.85,
    synthetic_qa_max_severe_frac=0.58,
    synthetic_qa_max_low_detail_mean=0.60,
    recompute_missing_artifact=True,
    teacher_gate=False,
    teacher_max_sparse_te=100.0,
    teacher_max_dense_te=100.0,
    teacher_allowed_stages=None,
    teacher_gate_sources=None,
):
    manifest = PseudoQueryManifest.load(manifest_path)
    initial_counts = manifest.source_counts()
    cache = PseudoTeacherCache.load(teacher_cache_path) if teacher_cache_path else PseudoTeacherCache()
    teacher_gate = bool(teacher_gate and teacher_cache_path)
    teacher_allowed_stages = list(teacher_allowed_stages or ["teacher_ok"])
    teacher_gate_sources = {str(item) for item in (teacher_gate_sources or []) if str(item)}
    synthetic_gate = SyntheticQualityGate(
        SyntheticQualityGateConfig(
            max_artifact_mean=_limit(synthetic_qa_max_mean),
            max_artifact_p95=_limit(synthetic_qa_max_p95),
            max_artifact_mild_frac=_limit(synthetic_qa_max_mild_frac),
            max_artifact_severe_frac=_limit(synthetic_qa_max_severe_frac),
            max_low_detail_mean=_limit(synthetic_qa_max_low_detail_mean),
        )
    )
    teacher_cache_gate = TeacherCacheGate(
        TeacherCacheGateConfig(
            max_sparse_te=_limit(teacher_max_sparse_te),
            max_dense_te=_limit(teacher_max_dense_te),
            allowed_stages=tuple(teacher_allowed_stages),
        )
    )

    synthetic_decisions = []
    teacher_decisions = []
    output_records = []
    for record in manifest.records:
        record.meta = getattr(record, "meta", {}) or {}
        if record.accepted and synthetic_qa and record.source == "synthetic_rgb":
            artifact_summary = _artifact_summary_for_record(
                record,
                recompute_missing=bool(recompute_missing_artifact),
            )
            decision = synthetic_gate.apply_to_record(record, artifact_summary)
            synthetic_decisions.append(decision)
        should_teacher_gate = teacher_gate and (not teacher_gate_sources or record.source in teacher_gate_sources)
        if record.accepted and should_teacher_gate:
            item = cache.get(_record_key(record))
            decision = teacher_cache_gate.evaluate(item)
            record.meta["teacher_cache_gate"] = decision.to_dict()
            teacher_decisions.append(decision)
            if not decision.accepted:
                record.accepted = False
                record.reason = decision.reason
        output_records.append(record)

    gated = PseudoQueryManifest(version=manifest.version, records=output_records)
    gated.save_jsonl(output_path)
    summary = {
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "output": os.path.abspath(os.fspath(output_path)),
        "teacher_cache": os.path.abspath(os.fspath(teacher_cache_path)) if teacher_cache_path else "",
        "initial_counts": initial_counts,
        "final_counts": gated.source_counts(),
        "accepted_counts": gated.accepted().source_counts(),
        "synthetic_quality_gate": {
            **summarize_gate_decisions(synthetic_decisions),
            "enabled": bool(synthetic_qa),
            "thresholds": {
                "max_artifact_mean": _limit(synthetic_qa_max_mean),
                "max_artifact_p95": _limit(synthetic_qa_max_p95),
                "max_artifact_mild_frac": _limit(synthetic_qa_max_mild_frac),
                "max_artifact_severe_frac": _limit(synthetic_qa_max_severe_frac),
                "max_low_detail_mean": _limit(synthetic_qa_max_low_detail_mean),
            },
        },
        "teacher_cache_gate": {
            **summarize_gate_decisions(teacher_decisions),
            "enabled": bool(teacher_gate),
            "thresholds": {
                "max_sparse_te_cm": _limit(teacher_max_sparse_te),
                "max_dense_te_cm": _limit(teacher_max_dense_te),
                "allowed_stages": teacher_allowed_stages,
                "sources": sorted(teacher_gate_sources),
            },
        },
    }
    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply strict synthetic QA and teacher-cache gates to a pseudo-query manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--skip_synthetic_qa", action="store_true", default=False)
    parser.add_argument("--synthetic_qa_max_mean", type=float, default=0.60)
    parser.add_argument("--synthetic_qa_max_p95", type=float, default=-1.0, help="Negative disables this hard gate.")
    parser.add_argument("--synthetic_qa_max_mild_frac", type=float, default=0.85)
    parser.add_argument("--synthetic_qa_max_severe_frac", type=float, default=0.58)
    parser.add_argument("--synthetic_qa_max_low_detail_mean", type=float, default=0.60)
    parser.add_argument("--no_recompute_missing_artifact", action="store_true", default=False)
    parser.add_argument("--teacher_gate", action="store_true", default=False)
    parser.add_argument("--no_teacher_gate", action="store_false", dest="teacher_gate")
    parser.add_argument("--teacher_max_sparse_te", type=float, default=100.0)
    parser.add_argument("--teacher_max_dense_te", type=float, default=100.0)
    parser.add_argument("--teacher_allowed_stages", default="teacher_ok")
    parser.add_argument("--teacher_gate_sources", default="", help="Comma-separated sources to teacher-gate. Empty gates all sources.")
    args = parser.parse_args()
    summary = gate_manifest_file(
        args.manifest,
        args.output,
        summary_json=args.summary_json,
        teacher_cache_path=args.teacher_cache,
        synthetic_qa=not args.skip_synthetic_qa,
        synthetic_qa_max_mean=args.synthetic_qa_max_mean,
        synthetic_qa_max_p95=args.synthetic_qa_max_p95,
        synthetic_qa_max_mild_frac=args.synthetic_qa_max_mild_frac,
        synthetic_qa_max_severe_frac=args.synthetic_qa_max_severe_frac,
        synthetic_qa_max_low_detail_mean=args.synthetic_qa_max_low_detail_mean,
        recompute_missing_artifact=not args.no_recompute_missing_artifact,
        teacher_gate=args.teacher_gate,
        teacher_max_sparse_te=args.teacher_max_sparse_te,
        teacher_max_dense_te=args.teacher_max_dense_te,
        teacher_allowed_stages=_comma_list(args.teacher_allowed_stages),
        teacher_gate_sources=_comma_list(args.teacher_gate_sources),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
