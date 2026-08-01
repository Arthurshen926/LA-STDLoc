#!/usr/bin/env python3
"""Extract artifact-filtered native SP evidence from frozen Gaussian renders."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from encoders.sp_encoder.export_image_embeddings import SuperPoint
from gaussian_renderer import render_from_pose_gsplat
from localization_training.splat_provenance import (
    anchor_source_csr,
    bank_splat_provenance_2dgs,
    bank_splat_provenance_3dgs,
)
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
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
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


def _save_partial(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--render-evidence-dir", required=True)
    parser.add_argument("--real-image-root", required=True)
    parser.add_argument("--real-query-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk-keypoints", type=int, default=2048)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=6.0)
    parser.add_argument("--positives-per-keypoint", type=int, default=4)
    parser.add_argument("--association-candidates-per-keypoint", type=int, default=8)
    parser.add_argument("--minimum-source-provenance-mass", type=float, default=0.05)
    parser.add_argument("--gaussian-ply", default="")
    parser.add_argument(
        "--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs"
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--track-payload", default="")
    parser.add_argument("--full-prior-pool", default="")
    parser.add_argument(
        "--require-raster-provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--provenance-topk-primitives", type=int, default=16)
    parser.add_argument("--provenance-candidate-topk", type=int, default=64)
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
    parser.add_argument(
        "--require-support-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-unsafe-no-support-mask",
        action="store_true",
        help="Explicitly allow non-canonical evidence without support masking.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.22)
    parser.add_argument("--support-dilate-radius", type=int, default=5)
    parser.add_argument("--invalid-min-area", type=int, default=96)
    parser.add_argument("--reference-downsample", type=int, default=4)
    parser.add_argument(
        "--maximum-reference-residual", type=float, default=0.18
    )
    parser.add_argument("--minimum-normal-norm", type=float, default=0.5)
    parser.add_argument("--quality-invalid-dilate-radius", type=int, default=2)
    parser.add_argument(
        "--require-warped-reference-qa",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.require_support_mask and not args.allow_unsafe_no_support_mask:
        raise ValueError(
            "canonical synthetic evidence requires --require-support-mask; "
            "use the explicit unsafe override only for a diagnostic ablation"
        )
    if args.require_warped_reference_qa and not args.real_query_cache:
        raise ValueError(
            "canonical rendering QA requires --real-query-cache for "
            "rendered-depth warping"
        )
    if args.require_raster_provenance and not args.gaussian_ply:
        raise ValueError(
            "canonical synthetic evidence requires --gaussian-ply for "
            "anchor/source-specific raster provenance"
        )

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    track_payload = (
        torch.load(args.track_payload, map_location="cpu", weights_only=False)
        if args.track_payload
        else None
    )
    full_prior_pool = (
        torch.load(args.full_prior_pool, map_location="cpu", weights_only=False)
        if args.full_prior_pool
        else None
    )
    anchor_source = anchor_source_csr(state, track_payload, full_prior_pool)
    source_universe = torch.unique(anchor_source[1]).sort().values
    gaussians = None
    provenance_function = None
    if args.require_raster_provenance:
        gaussians = (
            GaussianModel_2dgs(args.sh_degree)
            if args.gaussian_type == "2dgs"
            else GaussianModel(args.sh_degree)
        )
        gaussians.load_ply(args.gaussian_ply)
        gaussians = gaussians.cuda().eval()
        if source_universe.numel() == 0:
            raise ValueError("synthetic anchor source universe is empty")
        if int(source_universe.max()) >= int(gaussians.get_xyz.shape[0]):
            raise ValueError("synthetic anchor source IDs exceed Gaussian prior")
        provenance_function = (
            bank_splat_provenance_2dgs
            if args.gaussian_type == "2dgs"
            else bank_splat_provenance_3dgs
        )
    manifest = _load_jsonl(args.render_manifest)
    output_path = Path(args.output).resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    manifest_query_ids = [
        str(record["query_id"]) for record in manifest
    ]
    partial_config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"device"}
    }
    completed_records = []
    if partial_path.exists():
        partial = torch.load(
            partial_path, map_location="cpu", weights_only=False
        )
        if partial.get("manifest_query_ids") != manifest_query_ids:
            raise ValueError(
                "synthetic evidence partial does not align with manifest"
            )
        if partial.get("config") != partial_config:
            raise ValueError(
                "synthetic evidence partial uses a different resolved config"
            )
        completed_records = list(partial["records"])
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
        ambiguous_radius_px=args.ambiguous_radius_px,
        positives_per_keypoint=args.positives_per_keypoint,
        association_candidates_per_keypoint=(
            args.association_candidates_per_keypoint
        ),
        minimum_source_provenance_mass=args.minimum_source_provenance_mass,
        require_raster_provenance=args.require_raster_provenance,
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
    real_cache_payload = (
        torch.load(
            args.real_query_cache, map_location="cpu", weights_only=False
        )
        if args.real_query_cache
        else {}
    )
    real_cache = real_cache_payload.get("queries", real_cache_payload)
    mask_output = Path(args.mask_output_dir) if args.mask_output_dir else None
    records = list(completed_records)
    for index, manifest_record in enumerate(manifest):
        if index < len(completed_records):
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "view_count": len(manifest),
                        "resumed": True,
                    }
                ),
                flush=True,
            )
            continue
        frame_path = frames_dir / f"{index:06d}.png"
        evidence_path = render_evidence_dir / f"{index:06d}.pt"
        image = _image_tensor(frame_path)
        height, width = image.shape[-2:]
        valid_mask = mask_builder.build(image)
        render_evidence = torch.load(
            evidence_path, map_location="cpu", weights_only=False
        )
        meta = manifest_record.get("meta", {})
        reference_names = [
            str(meta[key]) for key in ("source_query", "neighbor_query")
        ]
        references = [
            _image_tensor(real_image_root / name)
            for name in reference_names
        ]
        synthetic_K = _scaled_intrinsics(
            manifest_record, width, height
        )
        reference_views = []
        if args.real_query_cache:
            for reference_name, reference_rgb in zip(
                reference_names, references
            ):
                cached = real_cache[reference_name]
                reference_K = torch.as_tensor(
                    cached["native_K"]
                ).float().clone()
                cached_height, cached_width = cached["native_input_hw"]
                reference_height, reference_width = reference_rgb.shape[-2:]
                reference_K[0] *= float(reference_width) / max(
                    int(cached_width), 1
                )
                reference_K[1] *= float(reference_height) / max(
                    int(cached_height), 1
                )
                reference_views.append(
                    {
                        "rgb": reference_rgb,
                        "pose_w2c": cached["pose_w2c"],
                        "K": reference_K,
                    }
                )
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
            render_pose_w2c=manifest_record["pose_w2c"],
            render_K=synthetic_K,
            reference_views=reference_views,
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
        keypoint_provenance = None
        if args.require_raster_provenance:
            fovx = 2.0 * math.atan(
                float(width) / (2.0 * float(synthetic_K[0, 0]))
            )
            fovy = 2.0 * math.atan(
                float(height) / (2.0 * float(synthetic_K[1, 1]))
            )
            package = render_from_pose_gsplat(
                gaussians,
                torch.as_tensor(manifest_record["pose_w2c"]).to(device).float(),
                fovx,
                fovy,
                width,
                height,
                bg_color=torch.zeros(3, device=device),
                render_mode="RGB+ED",
                rgb_only=True,
                return_rgb_meta=True,
                rasterize_mode="antialiased",
            )
            local_ids, contribution_mass, provenance_valid = provenance_function(
                torch.as_tensor(sparse["keypoints"]).to(device).float(),
                source_universe.to(device),
                package["rgb_meta"],
                rendered_depth=torch.as_tensor(render_evidence["depth"]).to(device),
                topk=args.provenance_topk_primitives,
                candidate_topk=args.provenance_candidate_topk,
                depth_abs_tolerance=args.absolute_depth_tolerance,
                depth_rel_tolerance=args.relative_depth_tolerance,
            )
            keypoint_provenance = {
                "primitive_ids": source_universe[local_ids.cpu()],
                "contribution_mass": contribution_mass.cpu(),
                "valid": provenance_valid.cpu(),
            }
            del package, local_ids, contribution_mass, provenance_valid
        record = build_synthetic_evidence_record(
            name=str(manifest_record["query_id"]),
            sparse=sparse,
            pose_w2c=torch.as_tensor(manifest_record["pose_w2c"]),
            K=synthetic_K,
            image_hw=(height, width),
            state=state,
            rendered_depth=render_evidence["depth"],
            alpha=render_evidence["alpha"],
            valid_mask_result=valid_mask,
            view_bin=int(
                meta["view_bin"]
                if "view_bin" in meta
                else meta["source_view_bin"]
            ),
            source_query=str(meta["source_query"]),
            config=config,
            device=device,
            keypoint_provenance=keypoint_provenance,
            anchor_source=anchor_source,
        )
        record["frame_path"] = str(frame_path.resolve())
        record["render_evidence_path"] = str(evidence_path.resolve())
        records.append(record)
        _save_partial(
            partial_path,
            {
                "schema": "lafgs_synthetic_evidence_partial",
                "version": 1,
                "manifest_query_ids": manifest_query_ids,
                "config": partial_config,
                "records": records,
            },
        )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "view_count": len(manifest),
                    "accepted": record["accepted"],
                    "positive_pairs": record["positive_pair_count"],
                    "ambiguous_pairs": record["ambiguous_pair_count"],
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
                "NoReferenceValidSupportMaskBuilder plus rendered-depth "
                "warped real-view consistency and raster alpha/depth/normal"
            ),
            "config": vars(args),
        },
    )
    path = output_path
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
    partial_path.unlink(missing_ok=True)
    print(path)


if __name__ == "__main__":
    main()
