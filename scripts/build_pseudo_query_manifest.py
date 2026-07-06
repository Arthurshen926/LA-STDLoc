#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams
from gaussian_renderer import render_from_pose_gsplat
from la_artifacts.detector import ArtifactDetector
from la_artifacts.quality_gate import SyntheticQualityGate, SyntheticQualityGateConfig
from la_artifacts.repair import ArtifactRepair
from la_artifacts.valid_mask import ArtifactValidMaskBuilder, ArtifactValidMaskConfig, save_valid_mask_png
from la_artifacts.rgb_teacher import (
    normalize_render_resolution,
    resolve_wildgaussians_appearance_mode,
    save_nerfbaselines_trajectory,
)
from la_artifacts.rgb_teacher_health import check_wildgaussians_checkpoint
from la_artifacts.pseudo_query import (
    PseudoQueryManifest,
    PseudoQueryRecord,
    apply_wildgaussians_appearance_strategy,
    synthetic_records_from_cameras,
)
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import seed_everything


def _tensor_to_image(tensor, path):
    array = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray((array * 255.0).round().astype("uint8"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _image_to_tensor(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _limit(value):
    if value is None:
        return None
    value = float(value)
    return None if value < 0 else value


def _synthetic_quality_gate_from_args(args):
    if bool(getattr(args, "skip_synthetic_quality_gate", False)):
        return None
    max_mean = _limit(getattr(args, "synthetic_qa_max_mean", -1.0))
    if max_mean is None:
        max_mean = float(args.synthetic_accept_score)
    return SyntheticQualityGate(
        SyntheticQualityGateConfig(
            max_artifact_mean=max_mean,
            max_artifact_p95=_limit(getattr(args, "synthetic_qa_max_p95", -1.0)),
            max_artifact_mild_frac=_limit(getattr(args, "synthetic_qa_max_mild_frac", -1.0)),
            max_artifact_severe_frac=_limit(getattr(args, "synthetic_qa_max_severe_frac", -1.0)),
            max_low_detail_mean=_limit(getattr(args, "synthetic_qa_max_low_detail_mean", -1.0)),
        )
    )


def _apply_synthetic_quality_gate(record, artifact_summary, gate):
    if gate is None:
        record.accepted = True
        record.reason = "ok"
        record.artifact_score = 0.0
        meta = getattr(record, "meta", {}) or {}
        meta["synthetic_quality_gate"] = {"enabled": False, "reason": "disabled"}
        record.meta = meta
        return None
    decision = gate.apply_to_record(record, artifact_summary)
    return decision


def _valid_mask_path(mask_root, record):
    image_name = Path(record.image_name)
    suffix = "".join(image_name.suffixes) or ".png"
    stem = str(image_name)
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return Path(mask_root) / f"{stem}.valid_mask.png"


def _write_synthetic_valid_mask(
    record,
    evidence,
    mask_root,
    max_artifact_score=0.45,
    erosion_radius=3,
    min_component_area=64,
    min_component_area_frac=0.0,
    min_valid_frac=0.0,
):
    if not mask_root:
        return None
    builder = ArtifactValidMaskBuilder(
        ArtifactValidMaskConfig(
            max_artifact_score=float(max_artifact_score),
            erosion_radius=int(erosion_radius),
            min_component_area=int(min_component_area),
            min_component_area_frac=float(min_component_area_frac),
            min_valid_frac=float(min_valid_frac),
        )
    )
    result = builder.build(evidence)
    path = _valid_mask_path(mask_root, record)
    save_valid_mask_png(result, path)
    record.meta.setdefault("artifact_valid_mask", {})
    record.meta["artifact_valid_mask"].update(
        {
            "mask_path": os.path.abspath(os.fspath(path)),
            "valid_frac": float(result.summary["valid_frac"]),
            "invalid_frac": float(result.summary["invalid_frac"]),
            "component_count": int(result.summary["component_count"]),
            "largest_component_frac": float(result.summary["largest_component_frac"]),
            "components": result.components,
            "thresholds": {
                "max_artifact_score": float(max_artifact_score),
                "erosion_radius": int(erosion_radius),
                "min_component_area": int(min_component_area),
                "min_component_area_frac": float(min_component_area_frac),
                "min_valid_frac": float(min_valid_frac),
            },
        }
    )
    return result


def _render_synthetic_records(
    records,
    gaussians,
    background,
    detector,
    repair=None,
    norm_feat_bf_render=True,
    accept_score=0.65,
    repair_threshold=0.35,
    quality_gate=None,
    valid_mask_root="",
    valid_mask_max_artifact_score=0.45,
    valid_mask_erosion_radius=3,
    valid_mask_min_component_area=64,
    valid_mask_min_component_area_frac=0.0,
    valid_mask_min_valid_frac=0.0,
):
    if quality_gate is None:
        detector_required = bool(repair is not None or valid_mask_root)
    else:
        detector_required = True
    accepted = []
    for record in tqdm(records, desc="Synthetic RGB render"):
        pose = torch.as_tensor(record.pose_w2c, dtype=torch.float32, device="cuda")
        width = int(record.width)
        height = int(record.height)
        with torch.no_grad():
            render_pkg = render_from_pose_gsplat(
                gaussians,
                pose,
                record.fovx,
                record.fovy,
                width,
                height,
                bg_color=background,
                render_mode="RGB+ED",
                rgb_only=True,
                norm_feat_bf_render=norm_feat_bf_render,
                rasterize_mode="antialiased",
            )
            rendered = render_pkg["render"]
            evidence = None
            before_score = 0.0
            if detector_required:
                evidence = detector.detect(rendered_rgb=rendered, alpha=render_pkg.get("alphas"))
                before_score = float(evidence.summary["artifact_score_mean"])
            repair_action = "none"
            suppressed = 0
            if repair is not None and before_score >= float(repair_threshold):
                loc_render_pkg = render_from_pose_gsplat(
                    gaussians,
                    pose,
                    record.fovx,
                    record.fovy,
                    width,
                    height,
                    bg_color=background,
                    render_mode="RGB+ED",
                    rgb_only=False,
                    return_loc_meta=True,
                    norm_feat_bf_render=norm_feat_bf_render,
                    rasterize_mode="antialiased",
                )
                gaussian_scores = detector.gaussian_scores_from_projected_map(
                    loc_render_pkg["loc_visible_idx"],
                    loc_render_pkg["loc_viewspace_points"],
                    evidence.score_map,
                    gaussian_count=gaussians.get_xyz.shape[0],
                )
                opacity_multiplier = repair.gaussian_opacity_multiplier(gaussian_scores).to(device=gaussians.get_xyz.device)
                suppressed = int((opacity_multiplier < 0.999).sum().item())
                repaired_pkg = render_from_pose_gsplat(
                    gaussians,
                    pose,
                    record.fovx,
                    record.fovy,
                    width,
                    height,
                    bg_color=background,
                    render_mode="RGB+ED",
                    rgb_only=True,
                    norm_feat_bf_render=norm_feat_bf_render,
                    opacity_multiplier=opacity_multiplier,
                    loc_opacity_multiplier=opacity_multiplier,
                    rasterize_mode="antialiased",
                )
                repaired_evidence = detector.detect(
                    rendered_rgb=repaired_pkg["render"],
                    alpha=repaired_pkg.get("alphas"),
                )
                rendered = repaired_pkg["render"]
                evidence = repaired_evidence
                repair_action = "opacity_suppression"
        _tensor_to_image(rendered, record.image_path)
        record.repair_action = repair_action
        record.meta.update(
            {
                "artifact_score_before_repair": before_score,
                "artifact_score_after_repair": float(evidence.summary["artifact_score_mean"]) if evidence is not None else 0.0,
                "repair_suppressed_gaussians": suppressed,
            }
        )
        if evidence is not None:
            _write_synthetic_valid_mask(
                record,
                evidence,
                valid_mask_root,
                max_artifact_score=valid_mask_max_artifact_score,
                erosion_radius=valid_mask_erosion_radius,
                min_component_area=valid_mask_min_component_area,
                min_component_area_frac=valid_mask_min_component_area_frac,
                min_valid_frac=valid_mask_min_valid_frac,
            )
            _apply_synthetic_quality_gate(record, evidence.summary, quality_gate)
        else:
            _apply_synthetic_quality_gate(record, {}, quality_gate)
        accepted.append(record)
    return accepted


def _wildgaussians_frame_path(output_dir, index):
    direct = Path(output_dir) / f"{index:05d}.png"
    if direct.exists():
        return direct
    nested = Path(output_dir) / "color" / f"{index:05d}.png"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Missing WildGaussians rendered frame {index:05d}.png under {output_dir}")


def _sequential_frame_path(output_dir, index, backend_name):
    direct = Path(output_dir) / f"{index:06d}.png"
    if direct.exists():
        return direct
    direct_5 = Path(output_dir) / f"{index:05d}.png"
    if direct_5.exists():
        return direct_5
    raise FileNotFoundError(f"Missing {backend_name} rendered frame {index:06d}.png under {output_dir}")


def _render_synthetic_records_wildgaussians(
    records,
    checkpoint,
    render_root,
    nerfbaselines_bin="nerfbaselines",
    nerfbaselines_backend="conda",
    output_names="color",
    accept_score=0.65,
    image_scale=1.0,
    resolution="",
    appearance_mode="auto",
    quality_gate=None,
    valid_mask_root="",
    valid_mask_max_artifact_score=0.45,
    valid_mask_erosion_radius=3,
    valid_mask_min_component_area=64,
    valid_mask_min_component_area_frac=0.0,
    valid_mask_min_valid_frac=0.0,
):
    if not checkpoint:
        raise ValueError("--rgb_teacher_checkpoint is required for --render_synthetic_backend wildgaussians")
    records = list(records)
    if not records:
        return records
    detector_required = bool(quality_gate is not None or valid_mask_root)
    health = check_wildgaussians_checkpoint(checkpoint)
    if not health.ok:
        raise RuntimeError(
            "Refusing to render synthetic RGB from an unhealthy WildGaussians checkpoint: "
            f"{health.reason}"
        )
    render_root = Path(render_root)
    frames_dir = render_root / "frames"
    trajectory_path = render_root / "trajectory.json"
    resolution = normalize_render_resolution(resolution)
    resolved_appearance_mode = resolve_wildgaussians_appearance_mode(
        appearance_mode,
        checkpoint,
        records=records,
    )
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    render_root.mkdir(parents=True, exist_ok=True)
    save_nerfbaselines_trajectory(
        records,
        trajectory_path,
        image_scale=image_scale,
        appearance_mode=resolved_appearance_mode,
    )

    command = [
        nerfbaselines_bin,
        "render-trajectory",
        "--checkpoint",
        os.path.abspath(checkpoint),
        "--trajectory",
        os.path.abspath(trajectory_path),
        "--output",
        os.path.abspath(frames_dir),
        "--output-names",
        str(output_names),
    ]
    if resolution:
        command.extend(["--resolution", resolution])
    if nerfbaselines_backend:
        command.extend(["--backend", nerfbaselines_backend])
    subprocess.run(command, check=True)

    detector = ArtifactDetector() if detector_required else None
    for idx, record in enumerate(records):
        src = _wildgaussians_frame_path(frames_dir, idx)
        dst = Path(record.image_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rendered = _image_to_tensor(dst)
        record.height = int(rendered.shape[1])
        record.width = int(rendered.shape[2])
        record.repair_action = "wildgaussians_render"
        record.meta.update(
            {
                "render_backend": "wildgaussians",
                "rgb_teacher_checkpoint": os.path.abspath(checkpoint),
                "rgb_teacher_health": health.to_dict(),
                "nerfbaselines_trajectory": os.path.abspath(trajectory_path),
                "nerfbaselines_output_dir": os.path.abspath(frames_dir),
                "nerfbaselines_output_names": str(output_names),
                "nerfbaselines_render_scale": float(image_scale or 1.0),
                "nerfbaselines_render_resolution": resolution,
                "nerfbaselines_appearance_mode": str(resolved_appearance_mode),
                "nerfbaselines_requested_appearance_mode": str(appearance_mode),
                "artifact_detector_inputs": ["rendered_rgb"] if detector_required else [],
                "artifact_score_note": (
                    "Synthetic quality gate disabled; no reference RGB exists for this synthetic view."
                    if not detector_required
                    else "No target residual is available for synthetic RGB; score covers only detector channels present in the render bundle."
                ),
            }
        )
        if detector is not None:
            evidence = detector.detect(rendered_rgb=rendered)
            _write_synthetic_valid_mask(
                record,
                evidence,
                valid_mask_root,
                max_artifact_score=valid_mask_max_artifact_score,
                erosion_radius=valid_mask_erosion_radius,
                min_component_area=valid_mask_min_component_area,
                min_component_area_frac=valid_mask_min_component_area_frac,
                min_valid_frac=valid_mask_min_valid_frac,
            )
            _apply_synthetic_quality_gate(record, evidence.summary, quality_gate)
        else:
            _apply_synthetic_quality_gate(record, {}, quality_gate)
    return records


def _render_synthetic_records_matcha(
    records,
    model_path,
    render_root,
    matcha_root="/root/MAtCha",
    matcha_python=None,
    iteration=30000,
    resolution="",
    quality_gate=None,
    valid_mask_root="",
    valid_mask_max_artifact_score=0.45,
    valid_mask_erosion_radius=3,
    valid_mask_min_component_area=64,
    valid_mask_min_component_area_frac=0.0,
    valid_mask_min_valid_frac=0.0,
):
    if not model_path:
        raise ValueError("--matcha_model_path is required for --render_synthetic_backend matcha")
    records = list(records)
    if not records:
        return records
    detector_required = bool(quality_gate is not None or valid_mask_root)
    render_root = Path(render_root)
    frames_dir = render_root / "frames"
    manifest_path = render_root / "records.jsonl"
    summary_path = render_root / "render_summary.json"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    render_root.mkdir(parents=True, exist_ok=True)
    PseudoQueryManifest(version=1, records=records).save_jsonl(manifest_path)

    renderer = Path(__file__).resolve().parent / "render_matcha_records.py"
    command = [
        str(matcha_python or sys.executable),
        os.fspath(renderer),
        "--manifest",
        os.path.abspath(os.fspath(manifest_path)),
        "--model_path",
        os.path.abspath(os.fspath(model_path)),
        "--output_dir",
        os.path.abspath(os.fspath(frames_dir)),
        "--matcha_root",
        os.path.abspath(os.fspath(matcha_root)),
        "--iteration",
        str(int(iteration)),
        "--summary_json",
        os.path.abspath(os.fspath(summary_path)),
    ]
    resolution = normalize_render_resolution(resolution)
    if resolution:
        command.extend(["--resolution", resolution])
    subprocess.run(command, check=True)

    detector = ArtifactDetector() if detector_required else None
    for idx, record in enumerate(records):
        src = _sequential_frame_path(frames_dir, idx, "MAtCha")
        dst = Path(record.image_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rendered = _image_to_tensor(dst)
        record.height = int(rendered.shape[1])
        record.width = int(rendered.shape[2])
        record.repair_action = "matcha_render"
        record.meta.update(
            {
                "render_backend": "matcha",
                "matcha_model_path": os.path.abspath(os.fspath(model_path)),
                "matcha_root": os.path.abspath(os.fspath(matcha_root)),
                "matcha_iteration": int(iteration),
                "matcha_manifest": os.path.abspath(os.fspath(manifest_path)),
                "matcha_output_dir": os.path.abspath(os.fspath(frames_dir)),
                "matcha_render_summary": os.path.abspath(os.fspath(summary_path)),
                "matcha_render_resolution": resolution,
                "artifact_detector_inputs": ["rendered_rgb"] if detector_required else [],
                "artifact_score_note": (
                    "Synthetic quality gate disabled; no reference RGB exists for this synthetic view."
                    if not detector_required
                    else "No target residual is available for synthetic RGB; score covers only detector channels present in the render bundle."
                ),
            }
        )
        if detector is not None:
            evidence = detector.detect(rendered_rgb=rendered)
            _write_synthetic_valid_mask(
                record,
                evidence,
                valid_mask_root,
                max_artifact_score=valid_mask_max_artifact_score,
                erosion_radius=valid_mask_erosion_radius,
                min_component_area=valid_mask_min_component_area,
                min_component_area_frac=valid_mask_min_component_area_frac,
                min_valid_frac=valid_mask_min_valid_frac,
            )
            _apply_synthetic_quality_gate(record, evidence.summary, quality_gate)
        else:
            _apply_synthetic_quality_gate(record, {}, quality_gate)
    return records


def main():
    parser = argparse.ArgumentParser(description="Build all-train + synthetic RGB pseudo-query manifest.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic_count", type=int, default=0)
    parser.add_argument("--synthetic_image_root", default="")
    parser.add_argument("--synthetic_seed", type=int, default=2026)
    parser.add_argument(
        "--synthetic_pose_sampler",
        choices=["adjacent_interpolate", "adjacent", "interpolate", "spatial_offset", "spatial"],
        default=os.environ.get("SYNTHETIC_POSE_SAMPLER", "spatial_offset"),
    )
    parser.add_argument("--synthetic_alpha_min", type=float, default=0.35)
    parser.add_argument("--synthetic_alpha_max", type=float, default=0.65)
    parser.add_argument("--synthetic_spatial_min_offset_ratio", type=float, default=float(os.environ.get("SYNTHETIC_SPATIAL_MIN_OFFSET_RATIO", "1.0")))
    parser.add_argument("--synthetic_spatial_max_offset_ratio", type=float, default=float(os.environ.get("SYNTHETIC_SPATIAL_MAX_OFFSET_RATIO", "3.0")))
    parser.add_argument("--synthetic_spatial_yaw_deg", type=float, default=float(os.environ.get("SYNTHETIC_SPATIAL_YAW_DEG", "20.0")))
    parser.add_argument("--synthetic_spatial_height_offset_ratio", type=float, default=float(os.environ.get("SYNTHETIC_SPATIAL_HEIGHT_OFFSET_RATIO", "0.15")))
    parser.add_argument("--render_synthetic_backend", choices=["none", "inrepo", "wildgaussians", "matcha"], default="none")
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
    parser.add_argument("--matcha_python", default=os.environ.get("MATCHA_PYTHON", sys.executable))
    parser.add_argument("--matcha_render_root", default="")
    parser.add_argument("--matcha_iteration", type=int, default=int(os.environ.get("MATCHA_ITERATION", "30000")))
    parser.add_argument("--matcha_render_resolution", default=os.environ.get("MATCHA_RENDER_RESOLUTION", ""))
    parser.add_argument("--synthetic_appearance_strategy", choices=["blend", "nearest", "none", "endpoint_a", "endpoint_b"], default=os.environ.get("SYNTHETIC_APPEARANCE_STRATEGY", "nearest"))
    parser.add_argument("--synthetic_accept_score", type=float, default=0.65)
    parser.add_argument("--synthetic_qa_max_mean", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_MEAN", "-1.0")))
    parser.add_argument("--synthetic_qa_max_p95", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_P95", "-1.0")))
    parser.add_argument("--synthetic_qa_max_mild_frac", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_MILD_FRAC", "-1.0")))
    parser.add_argument("--synthetic_qa_max_severe_frac", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_SEVERE_FRAC", "-1.0")))
    parser.add_argument("--synthetic_qa_max_low_detail_mean", type=float, default=float(os.environ.get("SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN", "-1.0")))
    parser.add_argument("--skip_synthetic_quality_gate", action="store_true", default=False)
    parser.add_argument("--synthetic_valid_mask_root", default=os.environ.get("SYNTHETIC_VALID_MASK_ROOT", ""))
    parser.add_argument("--synthetic_valid_mask_max_artifact_score", type=float, default=float(os.environ.get("SYNTHETIC_VALID_MASK_MAX_ARTIFACT_SCORE", "0.45")))
    parser.add_argument("--synthetic_valid_mask_erosion_radius", type=int, default=int(os.environ.get("SYNTHETIC_VALID_MASK_EROSION_RADIUS", "3")))
    parser.add_argument("--synthetic_valid_mask_min_component_area", type=int, default=int(os.environ.get("SYNTHETIC_VALID_MASK_MIN_COMPONENT_AREA", "64")))
    parser.add_argument("--synthetic_valid_mask_min_component_area_frac", type=float, default=float(os.environ.get("SYNTHETIC_VALID_MASK_MIN_COMPONENT_AREA_FRAC", "0.0")))
    parser.add_argument("--synthetic_valid_mask_min_valid_frac", type=float, default=float(os.environ.get("SYNTHETIC_VALID_MASK_MIN_VALID_FRAC", "0.0")))
    parser.add_argument("--repair_synthetic_artifacts", action="store_true", default=False)
    parser.add_argument("--synthetic_repair_threshold", type=float, default=0.35)
    parser.add_argument("--min_opacity_multiplier", type=float, default=0.15)
    args = parser.parse_args()
    args.eval = False
    seed_everything(args.synthetic_seed)

    dataset = model.extract(args)
    scene_name = args.scene_name or os.path.basename(os.path.normpath(dataset.source_path))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()

    image_root = os.path.join(dataset.source_path, dataset.images)
    records = [
        PseudoQueryRecord.from_camera(
            camera,
            scene=scene_name,
            image_root=image_root,
            source="train_rgb",
            train_index=index,
        )
        for index, camera in enumerate(train_cameras)
    ]

    if int(args.synthetic_count) > 0:
        if not args.synthetic_image_root:
            raise ValueError("--synthetic_image_root is required when --synthetic_count > 0")
        synthetic_records = synthetic_records_from_cameras(
            train_cameras,
            scene=scene_name,
            count=args.synthetic_count,
            image_root=args.synthetic_image_root,
            seed=args.synthetic_seed,
            alpha_min=args.synthetic_alpha_min,
            alpha_max=args.synthetic_alpha_max,
            pose_sampler=args.synthetic_pose_sampler,
            spatial_min_offset_ratio=args.synthetic_spatial_min_offset_ratio,
            spatial_max_offset_ratio=args.synthetic_spatial_max_offset_ratio,
            spatial_yaw_deg=args.synthetic_spatial_yaw_deg,
            spatial_height_offset_ratio=args.synthetic_spatial_height_offset_ratio,
        )
        synthetic_records = apply_wildgaussians_appearance_strategy(
            synthetic_records,
            args.synthetic_appearance_strategy,
        )
        synthetic_quality_gate = _synthetic_quality_gate_from_args(args)
        if args.render_synthetic_backend == "inrepo":
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            repair = ArtifactRepair() if args.repair_synthetic_artifacts else None
            if repair is not None:
                repair.config.min_opacity_multiplier = float(args.min_opacity_multiplier)
            synthetic_records = _render_synthetic_records(
                synthetic_records,
                gaussians,
                background,
                ArtifactDetector(),
                repair=repair,
                norm_feat_bf_render=dataset.norm_before_render,
                accept_score=args.synthetic_accept_score,
                repair_threshold=args.synthetic_repair_threshold,
                quality_gate=synthetic_quality_gate,
                valid_mask_root=args.synthetic_valid_mask_root,
                valid_mask_max_artifact_score=args.synthetic_valid_mask_max_artifact_score,
                valid_mask_erosion_radius=args.synthetic_valid_mask_erosion_radius,
                valid_mask_min_component_area=args.synthetic_valid_mask_min_component_area,
                valid_mask_min_component_area_frac=args.synthetic_valid_mask_min_component_area_frac,
                valid_mask_min_valid_frac=args.synthetic_valid_mask_min_valid_frac,
            )
        elif args.render_synthetic_backend == "wildgaussians":
            render_root = args.wildgaussians_render_root or os.path.join(args.synthetic_image_root, "_wildgaussians_render")
            synthetic_records = _render_synthetic_records_wildgaussians(
                synthetic_records,
                checkpoint=args.rgb_teacher_checkpoint,
                render_root=render_root,
                nerfbaselines_bin=args.nerfbaselines_bin,
                nerfbaselines_backend=args.nerfbaselines_backend,
                output_names=args.wildgaussians_output_names,
                accept_score=args.synthetic_accept_score,
                image_scale=args.wildgaussians_render_scale,
                resolution=args.wildgaussians_render_resolution,
                appearance_mode=args.wildgaussians_appearance_mode,
                quality_gate=synthetic_quality_gate,
                valid_mask_root=args.synthetic_valid_mask_root,
                valid_mask_max_artifact_score=args.synthetic_valid_mask_max_artifact_score,
                valid_mask_erosion_radius=args.synthetic_valid_mask_erosion_radius,
                valid_mask_min_component_area=args.synthetic_valid_mask_min_component_area,
                valid_mask_min_component_area_frac=args.synthetic_valid_mask_min_component_area_frac,
                valid_mask_min_valid_frac=args.synthetic_valid_mask_min_valid_frac,
            )
        elif args.render_synthetic_backend == "matcha":
            render_root = args.matcha_render_root or os.path.join(args.synthetic_image_root, "_matcha_render")
            synthetic_records = _render_synthetic_records_matcha(
                synthetic_records,
                model_path=args.matcha_model_path,
                render_root=render_root,
                matcha_root=args.matcha_root,
                matcha_python=args.matcha_python,
                iteration=args.matcha_iteration,
                resolution=args.matcha_render_resolution,
                quality_gate=synthetic_quality_gate,
                valid_mask_root=args.synthetic_valid_mask_root,
                valid_mask_max_artifact_score=args.synthetic_valid_mask_max_artifact_score,
                valid_mask_erosion_radius=args.synthetic_valid_mask_erosion_radius,
                valid_mask_min_component_area=args.synthetic_valid_mask_min_component_area,
                valid_mask_min_component_area_frac=args.synthetic_valid_mask_min_component_area_frac,
                valid_mask_min_valid_frac=args.synthetic_valid_mask_min_valid_frac,
            )
        records.extend(synthetic_records)

    manifest = PseudoQueryManifest(version=1, records=records)
    manifest.save_jsonl(args.output)
    print(f"Wrote pseudo-query manifest: {args.output}")
    print("summary:", manifest.source_counts())


if __name__ == "__main__":
    main()
