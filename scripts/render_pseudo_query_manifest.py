#!/usr/bin/env python3
import argparse
import copy
import os
from pathlib import Path

from la_artifacts.pseudo_query import PseudoQueryManifest
from scripts.build_pseudo_query_manifest import (
    _render_synthetic_records_matcha,
    _render_synthetic_records_wildgaussians,
    _synthetic_quality_gate_from_args,
)


def prepare_records_for_backend(records, synthetic_image_root):
    """Clone a manifest and rewrite only synthetic image paths for backend rendering."""
    root = os.fspath(synthetic_image_root)
    all_records = []
    render_records = []
    for record in records:
        row = copy.deepcopy(record)
        if row.source == "synthetic_rgb":
            row.image_path = os.path.join(root, row.image_name)
            row.accepted = False
            row.reason = "not_rendered"
            row.repair_action = "none"
            row.meta = dict(row.meta or {})
            render_records.append(row)
        all_records.append(row)
    return all_records, render_records


def render_manifest(
    input_manifest,
    output_manifest,
    synthetic_image_root,
    backend,
    rgb_teacher_checkpoint="",
    wildgaussians_render_root="",
    nerfbaselines_bin="nerfbaselines",
    nerfbaselines_backend="conda",
    wildgaussians_output_names="color",
    wildgaussians_render_scale=1.0,
    wildgaussians_render_resolution="",
    wildgaussians_appearance_mode="auto",
    matcha_model_path="",
    matcha_root="/root/MAtCha",
    matcha_python=None,
    matcha_render_root="",
    matcha_iteration=30000,
    matcha_render_resolution="",
    quality_gate=None,
    valid_mask_root="",
):
    manifest = PseudoQueryManifest.load(input_manifest)
    all_records, render_records = prepare_records_for_backend(manifest.records, synthetic_image_root)
    backend = str(backend)
    if backend == "wildgaussians":
        render_root = wildgaussians_render_root or os.path.join(os.fspath(synthetic_image_root), "_wildgaussians_render")
        _render_synthetic_records_wildgaussians(
            render_records,
            checkpoint=rgb_teacher_checkpoint,
            render_root=render_root,
            nerfbaselines_bin=nerfbaselines_bin,
            nerfbaselines_backend=nerfbaselines_backend,
            output_names=wildgaussians_output_names,
            image_scale=wildgaussians_render_scale,
            resolution=wildgaussians_render_resolution,
            appearance_mode=wildgaussians_appearance_mode,
            quality_gate=quality_gate,
            valid_mask_root=valid_mask_root,
        )
    elif backend == "matcha":
        render_root = matcha_render_root or os.path.join(os.fspath(synthetic_image_root), "_matcha_render")
        _render_synthetic_records_matcha(
            render_records,
            model_path=matcha_model_path,
            render_root=render_root,
            matcha_root=matcha_root,
            matcha_python=matcha_python,
            iteration=matcha_iteration,
            resolution=matcha_render_resolution,
            quality_gate=quality_gate,
            valid_mask_root=valid_mask_root,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    rendered = PseudoQueryManifest(version=manifest.version, records=all_records)
    rendered.save_jsonl(output_manifest)
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Render synthetic records from an existing pseudo-query manifest.")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--synthetic_image_root", required=True)
    parser.add_argument("--backend", choices=["wildgaussians", "matcha"], required=True)
    parser.add_argument("--rgb_teacher_checkpoint", default=os.environ.get("RGB_TEACHER_CHECKPOINT", ""))
    parser.add_argument("--nerfbaselines_bin", default=os.environ.get("NERFBASELINES_BIN", "nerfbaselines"))
    parser.add_argument("--nerfbaselines_backend", default=os.environ.get("NERFBASELINES_BACKEND", "conda"))
    parser.add_argument("--wildgaussians_render_root", default="")
    parser.add_argument("--wildgaussians_output_names", default=os.environ.get("RGB_TEACHER_RENDER_OUTPUT_NAMES", "color"))
    parser.add_argument("--wildgaussians_render_scale", type=float, default=float(os.environ.get("WILDGAUSSIANS_RENDER_SCALE", "1.0")))
    parser.add_argument("--wildgaussians_render_resolution", default=os.environ.get("WILDGAUSSIANS_RENDER_RESOLUTION", ""))
    parser.add_argument("--wildgaussians_appearance_mode", choices=["auto", "none", "record"], default=os.environ.get("WILDGAUSSIANS_APPEARANCE_MODE", "auto"))
    parser.add_argument("--matcha_model_path", default=os.environ.get("MATCHA_MODEL_PATH", ""))
    parser.add_argument("--matcha_root", default=os.environ.get("MATCHA_ROOT", "/root/MAtCha"))
    parser.add_argument("--matcha_python", default=os.environ.get("MATCHA_PYTHON", ""))
    parser.add_argument("--matcha_render_root", default="")
    parser.add_argument("--matcha_iteration", type=int, default=int(os.environ.get("MATCHA_ITERATION", "30000")))
    parser.add_argument("--matcha_render_resolution", default=os.environ.get("MATCHA_RENDER_RESOLUTION", ""))
    parser.add_argument("--skip_synthetic_quality_gate", action="store_true", default=False)
    parser.add_argument("--synthetic_accept_score", type=float, default=0.65)
    parser.add_argument("--synthetic_qa_max_mean", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_MEAN", "-1.0")))
    parser.add_argument("--synthetic_qa_max_p95", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_P95", "-1.0")))
    parser.add_argument("--synthetic_qa_max_mild_frac", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_MILD_FRAC", "-1.0")))
    parser.add_argument("--synthetic_qa_max_severe_frac", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_SEVERE_FRAC", "-1.0")))
    parser.add_argument("--synthetic_qa_max_low_detail_mean", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN", "-1.0")))
    parser.add_argument("--synthetic_valid_mask_root", default=os.environ.get("SYNTHETIC_VALID_MASK_ROOT", ""))
    args = parser.parse_args()

    quality_gate = _synthetic_quality_gate_from_args(args)
    render_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        synthetic_image_root=args.synthetic_image_root,
        backend=args.backend,
        rgb_teacher_checkpoint=args.rgb_teacher_checkpoint,
        wildgaussians_render_root=args.wildgaussians_render_root,
        nerfbaselines_bin=args.nerfbaselines_bin,
        nerfbaselines_backend=args.nerfbaselines_backend,
        wildgaussians_output_names=args.wildgaussians_output_names,
        wildgaussians_render_scale=args.wildgaussians_render_scale,
        wildgaussians_render_resolution=args.wildgaussians_render_resolution,
        wildgaussians_appearance_mode=args.wildgaussians_appearance_mode,
        matcha_model_path=args.matcha_model_path,
        matcha_root=args.matcha_root,
        matcha_python=args.matcha_python or None,
        matcha_render_root=args.matcha_render_root,
        matcha_iteration=args.matcha_iteration,
        matcha_render_resolution=args.matcha_render_resolution,
        quality_gate=quality_gate,
        valid_mask_root=args.synthetic_valid_mask_root,
    )
    print(f"Wrote rendered pseudo-query manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
