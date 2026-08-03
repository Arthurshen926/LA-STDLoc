"""Held-out RGB quality evaluation for off-the-shelf Gaussian priors."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def summarize_quality(rows: list[dict]) -> dict:
    output = {"query_count": len(rows)}
    for key in ("psnr_db", "ssim", "lpips"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
    return output


def _safe_image_name(name: str) -> str:
    return name.replace("\\", "/").replace("/", "__")


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@torch.inference_mode()
def evaluate_prior_quality(
    args,
    *,
    include_views: Iterable[str] | None = None,
    save_render_dir: str | Path | None = None,
    save_ground_truth: bool = False,
) -> dict:
    """Evaluate or export selected held-out views without training-side effects."""

    from gaussian_renderer import render_gsplat
    from scene import Scene
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from tqdm import tqdm
    from utils.loss_utils import ssim

    model_root = Path(args.model_root).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    manifest_path = model_root / "rgb_prior_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["gaussian_type"] != args.gaussian_type:
        raise ValueError("manifest Gaussian type does not match evaluation")
    if int(manifest["sh_degree"]) != int(args.sh_degree):
        raise ValueError("manifest SH degree does not match evaluation")
    if bool(manifest.get("prior_training_used_feature_loss", True)):
        raise ValueError("quality evaluation requires an RGB-only prior")
    scene_manifest = json.loads(
        (source / "evaluation_scene_manifest.json").read_text()
    )
    if scene_manifest.get("evaluation_only") is not True:
        raise ValueError("source is not an evaluation-only camera scene")
    if scene_manifest.get("used_for_prior_training") is not False:
        raise ValueError("evaluation source was used for prior training")

    dataset = Namespace(
        model_path=str(model_root),
        source_path=str(source),
        images="images",
        eval=True,
        feature_type="sp",
        gaussian_type=args.gaussian_type,
        sh_degree=int(args.sh_degree),
        resolution=1,
        data_device="cpu",
        longest_edge=0,
        white_background=bool(manifest.get("white_background", False)),
        speedup=False,
    )
    gaussians = (
        GaussianModel_2dgs(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel(args.sh_degree)
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=True,
    )
    background = torch.full(
        (3,),
        1.0 if dataset.white_background else 0.0,
        dtype=torch.float32,
        device="cuda",
    )
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).cuda().eval()
    selected = (
        {str(name).replace("\\", "/") for name in include_views}
        if include_views is not None
        else None
    )
    render_dir = (
        Path(save_render_dir).expanduser().resolve()
        if save_render_dir is not None
        else None
    )
    rows = []
    cameras = scene.getTestCameras()
    for index, camera in enumerate(tqdm(cameras, desc="prior quality")):
        if args.max_views and index >= args.max_views:
            break
        image_name = str(camera.image_name).replace("\\", "/")
        if selected is not None and image_name not in selected:
            continue
        gt = camera.original_image[:3].cuda(non_blocking=True).clamp(0, 1)
        longest_edge = max(int(camera.image_width), int(camera.image_height))
        rendered = render_gsplat(
            camera,
            gaussians,
            background,
            rgb_only=True,
            longest_edge=longest_edge,
        )["render"].clamp(0, 1)
        mse = torch.mean((rendered - gt) ** 2).clamp_min(1e-12)
        psnr_db = -10.0 * torch.log10(mse)
        ssim_value = ssim(rendered[None], gt[None])
        perceptual.reset()
        perceptual.update(rendered[None], gt[None])
        lpips_value = perceptual.compute()
        row = {
            "image_name": image_name,
            "psnr_db": float(psnr_db.item()),
            "ssim": float(ssim_value.item()),
            "lpips": float(lpips_value.item()),
        }
        if render_dir is not None:
            stem = _safe_image_name(image_name)
            render_path = render_dir / f"{stem}.render.png"
            _save_tensor_image(rendered, render_path)
            row["render_path"] = str(render_path)
            if save_ground_truth:
                gt_path = render_dir / f"{stem}.gt.png"
                _save_tensor_image(gt, gt_path)
                row["ground_truth_path"] = str(gt_path)
        rows.append(row)
    if selected is not None:
        found = {row["image_name"] for row in rows}
        missing = selected - found
        if missing:
            raise ValueError(f"requested evaluation views were not found: {sorted(missing)}")
    expected = int(scene_manifest["test_image_count"])
    if selected is None and not args.max_views and len(rows) != expected:
        raise ValueError(f"expected {expected} test views, evaluated {len(rows)}")
    report = {
        "schema": "lafgs_off_the_shelf_prior_quality",
        "version": 2,
        "model_root": str(model_root),
        "evaluation_source": str(source),
        "gaussian_type": args.gaussian_type,
        "iteration": int(args.iteration),
        "full_resolution": True,
        "semantic_mask_used": False,
        "background": "white" if dataset.white_background else "black",
        "selected_view_export": selected is not None,
        "summary": summarize_quality(rows),
        "per_view": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
