#!/usr/bin/env python3
"""Extract artifact-filtered native SP evidence from frozen Gaussian renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from encoders.sp_encoder.export_image_embeddings import SuperPoint
from localization_training.synthetic_evidence import (
    RenderQualityFilterConfig,
    SyntheticEvidenceConfig,
    build_render_quality_mask,
    build_synthetic_evidence_record,
    pack_synthetic_evidence,
    synthetic_function_graph_payload,
    synthetic_positive_teacher_payload,
    synthetic_query_cache_payload,
)
from valid_support_mask import (
    NoReferenceValidSupportMaskBuilder,
    NoReferenceValidSupportMaskConfig,
    save_mask_bundle_pngs,
)


def _load_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _image_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(image).permute(2, 0, 1) / 255.0


def _scaled_intrinsics(record: dict, width: int, height: int) -> torch.Tensor:
    K = torch.as_tensor(record["meta"]["K"]).float().clone()
    scale_x = float(width) / max(int(record["width"]), 1)
    scale_y = float(height) / max(int(record["height"]), 1)
    K[0] *= scale_x
    K[1] *= scale_y
    return K


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--render-evidence-dir", required=True)
    parser.add_argument("--real-image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk-keypoints", type=int, default=2048)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--positives-per-keypoint", type=int, default=4)
    parser.add_argument("--minimum-alpha", type=float, default=0.5)
    parser.add_argument(
        "--absolute-depth-tolerance", type=float, default=0.08
    )
    parser.add_argument(
        "--relative-depth-tolerance", type=float, default=0.03
    )
    parser.add_argument("--minimum-positive-pairs", type=int, default=32)
    parser.add_argument("--minimum-matchable-rate", type=float, default=0.02)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.5)
    parser.add_argument(
        "--minimum-valid-keypoint-fraction", type=float, default=0.5
    )
    parser.add_argument("--minimum-alpha-coverage", type=float, default=0.25)
    parser.add_argument("--minimum-visible-anchors", type=int, default=64)
    parser.add_argument("--require-support-mask", action="store_true")
    parser.add_argument("--support-threshold", type=float, default=0.22)
    parser.add_argument("--support-dilate-radius", type=int, default=5)
    parser.add_argument("--invalid-min-area", type=int, default=96)
    parser.add_argument("--reference-downsample", type=int, default=4)
    parser.add_argument(
        "--maximum-reference-residual", type=float, default=0.18
    )
    parser.add_argument("--minimum-normal-norm", type=float, default=0.5)
    parser.add_argument("--quality-invalid-dilate-radius", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    manifest = _load_jsonl(args.render_manifest)
    extractor = SuperPoint().to(device).eval()
    mask_builder = NoReferenceValidSupportMaskBuilder(
        NoReferenceValidSupportMaskConfig(
            support_threshold=args.support_threshold,
            support_dilate_radius=args.support_dilate_radius,
            invalid_min_area=args.invalid_min_area,
        )
    )
    config = SyntheticEvidenceConfig(
        topk_keypoints=args.topk_keypoints,
        positive_radius_px=args.positive_radius_px,
        positives_per_keypoint=args.positives_per_keypoint,
        minimum_alpha=args.minimum_alpha,
        absolute_depth_tolerance=args.absolute_depth_tolerance,
        relative_depth_tolerance=args.relative_depth_tolerance,
        minimum_positive_pairs=args.minimum_positive_pairs,
        minimum_matchable_rate=args.minimum_matchable_rate,
        minimum_valid_fraction=args.minimum_valid_fraction,
        minimum_valid_keypoint_fraction=(
            args.minimum_valid_keypoint_fraction
        ),
        minimum_alpha_coverage=args.minimum_alpha_coverage,
        minimum_visible_anchors=args.minimum_visible_anchors,
        require_support_mask=args.require_support_mask,
    )
    frames_dir = Path(args.frames_dir)
    render_evidence_dir = Path(args.render_evidence_dir)
    real_image_root = Path(args.real_image_root)
    mask_output = Path(args.mask_output_dir) if args.mask_output_dir else None
    records = []
    for index, manifest_record in enumerate(manifest):
        frame_path = frames_dir / f"{index:06d}.png"
        evidence_path = render_evidence_dir / f"{index:06d}.pt"
        image = _image_tensor(frame_path)
        height, width = image.shape[-2:]
        valid_mask = mask_builder.build(image)
        render_evidence = torch.load(
            evidence_path, map_location="cpu", weights_only=False
        )
        meta = manifest_record.get("meta", {})
        references = [
            _image_tensor(real_image_root / str(meta[key]))
            for key in ("source_query", "neighbor_query")
        ]
        valid_mask = build_render_quality_mask(
            base_mask=valid_mask,
            rendered_rgb=image,
            reference_rgbs=references,
            alpha=render_evidence["alpha"],
            rendered_depth=render_evidence["depth"],
            surface_normal=render_evidence["surface_normal"],
            config=RenderQualityFilterConfig(
                reference_downsample=args.reference_downsample,
                maximum_reference_residual=(
                    args.maximum_reference_residual
                ),
                minimum_alpha=args.minimum_alpha,
                minimum_normal_norm=args.minimum_normal_norm,
                invalid_dilate_radius=args.quality_invalid_dilate_radius,
            ),
        )
        if mask_output is not None:
            save_mask_bundle_pngs(
                valid_mask, mask_output / f"{index:06d}"
            )
        with torch.inference_mode():
            sparse = extractor.detectAndCompute(
                image[None].to(device),
                top_k=args.topk_keypoints,
            )[0]
        record = build_synthetic_evidence_record(
            name=str(manifest_record["query_id"]),
            sparse=sparse,
            pose_w2c=torch.as_tensor(manifest_record["pose_w2c"]),
            K=_scaled_intrinsics(manifest_record, width, height),
            image_hw=(height, width),
            state=state,
            rendered_depth=render_evidence["depth"],
            alpha=render_evidence["alpha"],
            valid_mask_result=valid_mask,
            view_bin=int(meta["view_bin"]),
            source_query=str(meta["source_query"]),
            config=config,
            device=device,
        )
        record["frame_path"] = str(frame_path.resolve())
        record["render_evidence_path"] = str(evidence_path.resolve())
        records.append(record)
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "view_count": len(manifest),
                    "accepted": record["accepted"],
                    "positive_pairs": record["positive_pair_count"],
                    "matchable_rate": record["matchable_rate"],
                    "visible_anchors": record["visible_anchor_count"],
                    "alpha_coverage": record["alpha_coverage"],
                    "valid_fraction": record["valid_mask_summary"][
                        "valid_frac"
                    ],
                    "valid_keypoint_fraction": record[
                        "valid_keypoint_fraction"
                    ],
                }
            ),
            flush=True,
        )
    output = pack_synthetic_evidence(
        records,
        provenance={
            "map": str(Path(args.map).resolve()),
            "render_manifest": str(Path(args.render_manifest).resolve()),
            "frames_dir": str(frames_dir.resolve()),
            "render_evidence_dir": str(render_evidence_dir.resolve()),
            "geometry_policy": (
                "existing Track-First anchors only; rendered evidence cannot "
                "create or move geometry"
            ),
            "mask_policy": (
                "NoReferenceValidSupportMaskBuilder plus adjacent-real RGB "
                "continuity and raster alpha/depth/normal consistency"
            ),
            "config": vars(args),
        },
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    torch.save(
        synthetic_query_cache_payload(output),
        path.with_name(path.stem + "_query_cache.pt"),
    )
    torch.save(
        synthetic_positive_teacher_payload(
            output,
            anchor_count=int(
                torch.as_tensor(state["anchor_xyz"]).shape[0]
            ),
        ),
        path.with_name(path.stem + "_positive_teacher.pt"),
    )
    torch.save(
        synthetic_function_graph_payload(
            output,
            anchor_count=int(
                torch.as_tensor(state["anchor_xyz"]).shape[0]
            ),
        ),
        path.with_name(path.stem + "_function_graph.pt"),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "summary": output["summary"],
                "rejected_records": output["rejected_records"],
                "provenance": output["provenance"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
