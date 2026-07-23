#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint, random

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_from_pose_gsplat, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from localization_training.direct_landmark_teacher import gaussian_localization_xyz
from localization_training.episode_sampler import (
    sample_interpolated_novel_view,
    split_support_query_cameras,
)
from localization_training.landmark_distill import (
    coverage_preserving_sample,
    localization_aware_sample,
    save_landmark_meta,
)
from localization_training.pair_scorer import SparsePairScorer
from localization_training.pair_measurement import PairMeasurementHead
from localization_training.hard_candidate_teacher import (
    HardCandidateTeacherCache,
    hard_candidate_preservation_loss,
)
from localization_training.sparse_candidate_teacher import (
    build_sparse_candidate_batch,
    calibrate_binary_threshold,
    sparse_candidate_losses,
)
from localization_training.sparse_frontend import (
    SparseMatchResult,
    limit_matches_per_keypoint,
    rank_keypoint_proposals,
)


def _artifact_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _detector_git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def write_detector_reproducibility_manifest(dataset, args):
    output_dir = Path(dataset.model_path) / str(args.detector_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve(path):
        if not path:
            return ""
        path = Path(path)
        if not path.is_absolute():
            path = Path(dataset.model_path) / path
        return str(path.resolve())

    inputs = {
        "precomputed_landmark_path": resolve(args.precomputed_landmark_path),
        "candidate_teacher_state_init_path": resolve(
            args.candidate_teacher_state_init_path
        ),
        "candidate_teacher_detector_init_path": resolve(
            args.candidate_teacher_detector_init_path
        ),
        "candidate_teacher_pair_scorer_init_path": resolve(
            args.candidate_teacher_pair_scorer_init_path
        ),
        "candidate_teacher_pair_measurement_init_path": resolve(
            args.candidate_teacher_pair_measurement_init_path
        ),
    }
    git_diff = _detector_git_output("diff", "--binary")
    manifest = {
        "version": 1,
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "dataset": {
            "model_path": os.path.abspath(dataset.model_path),
            "source_path": os.path.abspath(dataset.source_path),
            "images": str(dataset.images),
            "resolution": int(dataset.resolution),
            "longest_edge": int(dataset.longest_edge),
            "gaussian_type": str(dataset.gaussian_type),
            "feature_type": str(dataset.feature_type),
        },
        "inputs": {
            key: {
                "path": path,
                "sha256": _artifact_sha256(path) if path else None,
            }
            for key, path in inputs.items()
        },
        "effective_objective": (
            "detector_match_plus_offset"
            if bool(args.candidate_teacher_detector_only)
            else "configured_candidate_teacher_objective"
        ),
        "git": {
            "commit": _detector_git_output("rev-parse", "HEAD"),
            "branch": _detector_git_output("branch", "--show-current"),
            "status_porcelain": _detector_git_output("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(git_diff.encode("utf-8")).hexdigest(),
        },
    }
    path = output_dir / "reproducibility_manifest.json"
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path
from localization_training.splat_provenance import compress_2dgs_rgb_meta_to_bank
from scene.kpdetector import KpDetector


def _parameter_gradient_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter is None or parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return float(torch.sqrt(squared).item()) if squared is not None else 0.0


def _isolated_loss_gradient_norm(loss, parameters):
    parameters = [
        parameter
        for parameter in parameters
        if parameter is not None and parameter.requires_grad
    ]
    if not parameters or not bool(loss.requires_grad):
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = None
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return float(torch.sqrt(squared).item()) if squared is not None else 0.0


def partition_candidate_teacher_cameras(
    cameras,
    *,
    support_query_split=False,
    query_ratio=0.2,
    validation_ratio=0.0,
    split_mode="temporal_block",
    split_seed=2026,
):
    """Build candidate-train and held-out validation camera sets.

    Candidate-map training, direct held-out evaluation, and detector fitting
    must operate on the same canonical camera order.  ``Scene`` may shuffle
    camera objects for conventional detector training, so partitioning an
    arbitrary input order makes a temporal-block split semantically unstable.
    """
    training_cameras = sorted(
        list(cameras),
        key=lambda camera: str(getattr(camera, "image_name", "")).replace("\\", "/"),
    )
    validation_cameras = []
    support_camera_count = len(training_cameras)
    if bool(support_query_split):
        support_cameras, training_cameras = split_support_query_cameras(
            training_cameras,
            query_ratio=query_ratio,
            seed=split_seed,
            mode=split_mode,
        )
        support_camera_count = len(support_cameras)
    if float(validation_ratio) > 0.0:
        training_cameras, validation_cameras = split_support_query_cameras(
            training_cameras,
            query_ratio=validation_ratio,
            seed=split_seed + 1,
            mode=split_mode,
        )
    return training_cameras, validation_cameras, support_camera_count


def candidate_teacher_camera_names_sha256(cameras):
    """Hash the canonical candidate-camera set for split-audit artifacts."""
    names = sorted(
        str(getattr(camera, "image_name", "")).replace("\\", "/")
        for camera in cameras
    )
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def write_candidate_teacher_partition_manifest(
    output_dir,
    support_cameras,
    training_cameras,
    validation_cameras,
    *,
    split_mode,
    split_seed,
    validation_ratio,
):
    """Persist exact camera identities so a direct holdout is auditable."""
    def names(cameras):
        return sorted(
            str(getattr(camera, "image_name", "")).replace("\\", "/")
            for camera in cameras
        )

    payload = {
        "version": 1,
        "camera_order": "image_name_lexicographic",
        "split_mode": str(split_mode),
        "split_seed": int(split_seed),
        "validation_ratio": float(validation_ratio),
        "support_camera_names": names(support_cameras),
        "candidate_train_camera_names": names(training_cameras),
        "candidate_validation_camera_names": names(validation_cameras),
        "support_camera_names_sha256": candidate_teacher_camera_names_sha256(
            support_cameras
        ),
        "candidate_train_camera_names_sha256": (
            candidate_teacher_camera_names_sha256(training_cameras)
        ),
        "candidate_validation_camera_names_sha256": (
            candidate_teacher_camera_names_sha256(validation_cameras)
        ),
    }
    output_path = Path(output_dir) / "candidate_teacher_partition.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def training_requires_test_cameras(test_iterations, total_iterations):
    """Return whether detector training schedules any test-set evaluation."""
    total_iterations = int(total_iterations)
    return any(
        1 <= int(iteration) <= total_iterations
        for iteration in (test_iterations or ())
    )


def _resize_hard_valid_mask(valid_mask, target_hw, device):
    """Resize a binary query-support mask without introducing soft borders."""
    if valid_mask is None:
        return None
    mask = torch.as_tensor(valid_mask, dtype=torch.float32, device=device)
    while mask.ndim > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            raise ValueError(
                "A query validity mask must reduce to [H,W], got "
                f"shape {tuple(mask.shape)}"
            )
    if mask.ndim != 2:
        raise ValueError(f"A query validity mask must be [H,W], got {tuple(mask.shape)}")
    return F.interpolate(
        mask[None, None], size=tuple(map(int, target_hw)), mode="nearest"
    )[0, 0].bool()


def camera_valid_mask(masks, camera, device="cuda"):
    """Return the common object/sky/distortion support for a real query."""
    if masks is None:
        return None
    raw_name = str(camera.image_name)
    name = raw_name.replace("\\", "/")
    if name not in masks and raw_name in masks:
        name = raw_name
    if name not in masks:
        return None
    channels = masks[name]
    if len(channels) < 3:
        raise ValueError(
            f"Mask entry for {name!r} must contain object, sky and distortion masks"
        )
    target_hw = tuple(map(int, camera.original_image.shape[-2:]))
    return (
        _resize_hard_valid_mask(channels[0], target_hw, device)
        & _resize_hard_valid_mask(channels[1], target_hw, device)
        & _resize_hard_valid_mask(channels[2], target_hw, device)
    )


def extract_normalized_feature_map(
    feature_extractor,
    image,
    size,
    *,
    query_feature_contract="legacy_full_then_resized_map",
    valid_mask=None,
):
    """Run the fixed encoder under the chosen sparse feature contract."""
    with torch.no_grad():
        if query_feature_contract == "native_resized_input":
            resized_image = F.interpolate(
                image[None], size=size, mode="bilinear", align_corners=False
            )
            input_mask = _resize_hard_valid_mask(
                valid_mask, size, resized_image.device
            )
            if input_mask is not None:
                resized_image = resized_image * input_mask[None, None].to(
                    dtype=resized_image.dtype
                )
            gt_feature_map = feature_extractor(resized_image)["feature_map"]
            expected_hw = (max(int(size[0]) // 8, 1), max(int(size[1]) // 8, 1))
            if tuple(gt_feature_map.shape[-2:]) != expected_hw:
                raise RuntimeError(
                    "Native SuperPoint detector target has an unexpected "
                    f"stride-8 shape: got={tuple(gt_feature_map.shape[-2:])} "
                    f"expected={expected_hw}"
                )
            gt_feature_map = gt_feature_map.squeeze(0)
        elif query_feature_contract == "legacy_full_then_resized_map":
            input_mask = _resize_hard_valid_mask(
                valid_mask, image.shape[-2:], image.device
            )
            if input_mask is not None:
                image = image * input_mask[None].to(dtype=image.dtype)
            gt_feature_map = feature_extractor(image[None])["feature_map"]
            gt_feature_map = F.interpolate(
                gt_feature_map,
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise ValueError(
                f"Unknown query feature contract: {query_feature_contract!r}"
            )
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
        output_mask = _resize_hard_valid_mask(
            valid_mask, gt_feature_map.shape[-2:], gt_feature_map.device
        )
        if output_mask is not None:
            gt_feature_map = gt_feature_map * output_mask[None].to(
                dtype=gt_feature_map.dtype
            )
    return gt_feature_map.detach()


def scheduled_online_render_ratio(
    iteration,
    total_iterations,
    ratio_start=0.0,
    ratio_end=0.0,
    ramp_start_fraction=0.0,
    ramp_end_fraction=1.0,
):
    """Linear synthetic-query curriculum, expressed independently of scene size."""
    total = max(int(total_iterations), 1)
    progress = max(0.0, min(1.0, float(iteration) / float(total)))
    start = max(0.0, min(1.0, float(ramp_start_fraction)))
    end = max(start, min(1.0, float(ramp_end_fraction)))
    if end <= start:
        alpha = 1.0 if progress >= end else 0.0
    else:
        alpha = max(0.0, min(1.0, (progress - start) / (end - start)))
    ratio = (1.0 - alpha) * float(ratio_start) + alpha * float(ratio_end)
    return max(0.0, min(1.0, ratio))


def candidate_teacher_trust_alpha(
    visible_count,
    correct_count,
    *,
    alpha_min=0.25,
    view_prior=3.0,
    warmup_active=False,
    warmup_blend=None,
):
    """Derive a bounded descriptor-update scale from deployment-edge evidence."""
    visible_count = torch.as_tensor(visible_count)
    correct_count = torch.as_tensor(correct_count, device=visible_count.device)
    if visible_count.shape != correct_count.shape:
        raise ValueError("visible_count and correct_count must have the same shape")
    if visible_count.ndim != 1:
        raise ValueError("candidate teacher trust counts must be one-dimensional")
    alpha_min = max(0.0, min(1.0, float(alpha_min)))
    view_prior = max(0.0, float(view_prior))
    if bool(warmup_active) and warmup_blend is None:
        return torch.ones_like(visible_count, dtype=torch.float32)
    visible = visible_count.to(dtype=torch.float32).clamp_min(0.0)
    correct = correct_count.to(dtype=torch.float32).clamp_min(0.0)
    correct = torch.minimum(correct, visible)
    reliability = correct / visible.clamp_min(1.0)
    # ``visible`` is the number of *unique real cameras* that observed this
    # landmark.  It is deliberately not the number of SGD visits: otherwise
    # a camera sampled more often would make a descriptor look better
    # supported than it is geometrically.
    view_support = visible / (visible + view_prior).clamp_min(1e-6)
    evidence_alpha = (reliability * view_support).clamp(
        min=alpha_min,
        max=1.0,
    )
    if warmup_blend is None:
        return evidence_alpha
    warmup_blend = max(0.0, min(1.0, float(warmup_blend)))
    return evidence_alpha + warmup_blend * (1.0 - evidence_alpha)


def candidate_teacher_effective_features(
    initial_features,
    raw_features,
    trust_alpha,
):
    """Apply the frozen-bank residual parameterization used at inference time."""
    if initial_features.shape != raw_features.shape:
        raise ValueError("initial_features and raw_features must have the same shape")
    trust_alpha = torch.as_tensor(
        trust_alpha,
        device=raw_features.device,
        dtype=raw_features.dtype,
    ).reshape(-1)
    if trust_alpha.numel() != raw_features.shape[0]:
        raise ValueError("trust_alpha must have one value per landmark")
    return F.normalize(
        initial_features + trust_alpha[:, None] * (raw_features - initial_features),
        dim=1,
    )


@torch.no_grad()
def update_candidate_teacher_trust_evidence(
    visible_count,
    correct_count,
    visible_mask,
    candidate_landmark_idx,
    candidate_correct_mask,
    candidate_retained_mask,
    *,
    camera_index=None,
    visible_view_mask=None,
    correct_view_mask=None,
    report=True,
    validate_indices=True,
):
    """Accumulate one visibility/correctness observation per landmark and query.

    The caller supplies only candidates that survive the deployment-equivalent
    top-k/threshold/quota frontend. Recovered training positives are deliberately
    excluded, so they cannot make a landmark appear more trustworthy than it is
    under inference.
    """
    visible_count = torch.as_tensor(visible_count)
    correct_count = torch.as_tensor(correct_count, device=visible_count.device)
    if visible_count.shape != correct_count.shape or visible_count.ndim != 1:
        raise ValueError("candidate teacher trust counts must be matching one-dimensional tensors")
    visible_mask = torch.as_tensor(
        visible_mask,
        device=visible_count.device,
        dtype=torch.bool,
    ).reshape(-1)
    if visible_mask.numel() != visible_count.numel():
        raise ValueError("visible_mask must have one value per landmark")
    candidate_landmark_idx = torch.as_tensor(
        candidate_landmark_idx,
        device=visible_count.device,
        dtype=torch.long,
    ).reshape(-1)
    candidate_correct_mask = torch.as_tensor(
        candidate_correct_mask,
        device=visible_count.device,
        dtype=torch.bool,
    ).reshape(-1)
    candidate_retained_mask = torch.as_tensor(
        candidate_retained_mask,
        device=visible_count.device,
        dtype=torch.bool,
    ).reshape(-1)
    if not (
        candidate_landmark_idx.numel()
        == candidate_correct_mask.numel()
        == candidate_retained_mask.numel()
    ):
        raise ValueError("candidate trust tensors must have matching lengths")
    use_unique_camera_evidence = (
        visible_view_mask is not None or correct_view_mask is not None
    )
    if use_unique_camera_evidence:
        if visible_view_mask is None or correct_view_mask is None:
            raise ValueError(
                "unique trust evidence requires both visible_view_mask and "
                "correct_view_mask"
            )
        visible_view_mask = torch.as_tensor(
            visible_view_mask,
            device=visible_count.device,
            dtype=torch.bool,
        )
        correct_view_mask = torch.as_tensor(
            correct_view_mask,
            device=visible_count.device,
            dtype=torch.bool,
        )
        if visible_view_mask.ndim != 2 or correct_view_mask.shape != visible_view_mask.shape:
            raise ValueError(
                "unique trust evidence masks must have matching "
                "[landmark, camera] shapes"
            )
        if visible_view_mask.shape[0] != visible_count.numel():
            raise ValueError(
                "unique trust evidence masks must have one row per landmark"
            )
        if camera_index is None:
            raise ValueError("unique trust evidence requires camera_index")
        camera_index = int(camera_index)
        if not 0 <= camera_index < visible_view_mask.shape[1]:
            raise ValueError("camera_index is outside the trust evidence cameras")
        visible_seen = visible_view_mask[:, camera_index]
        visible_added_mask = visible_mask & ~visible_seen
        visible_count.add_(visible_added_mask.to(dtype=visible_count.dtype))
        visible_view_mask[visible_mask, camera_index] = True
    else:
        visible_added_mask = visible_mask
        visible_count.add_(visible_mask.to(dtype=visible_count.dtype))
    valid = candidate_correct_mask & candidate_retained_mask
    correct_landmarks = candidate_landmark_idx[valid]
    correct_observed_mask = torch.zeros_like(visible_mask)
    if correct_landmarks.numel() > 0:
        if bool(validate_indices) and (
            int(correct_landmarks.min().item()) < 0
            or int(correct_landmarks.max().item()) >= visible_count.numel()
        ):
            raise ValueError("candidate landmark index is outside the trust bank")
        correct_observed_mask.index_fill_(0, correct_landmarks, True)
        if use_unique_camera_evidence:
            correct_seen = correct_view_mask[:, camera_index]
            correct_added_mask = correct_observed_mask & ~correct_seen
            correct_count.add_(correct_added_mask.to(dtype=correct_count.dtype))
            correct_view_mask[correct_observed_mask, camera_index] = True
        else:
            correct_added_mask = correct_observed_mask
            correct_count.add_(correct_observed_mask.to(dtype=correct_count.dtype))
    else:
        correct_added_mask = correct_observed_mask
    if not bool(report):
        return {}
    return {
        "trust_visible_added": float(visible_added_mask.sum().item()),
        "trust_correct_landmarks_added": float(correct_added_mask.sum().item()),
        "trust_unique_camera_evidence": 1.0 if use_unique_camera_evidence else 0.0,
    }


@torch.no_grad()
def candidate_teacher_trust_diagnostics(
    visible_count,
    correct_count,
    trust_alpha,
    *,
    warmup_active=False,
):
    """Summarize trust calibration without exposing per-landmark training state."""
    visible = torch.as_tensor(visible_count, dtype=torch.float32).reshape(-1)
    correct = torch.as_tensor(correct_count, dtype=torch.float32, device=visible.device).reshape(-1)
    alpha = torch.as_tensor(trust_alpha, dtype=torch.float32, device=visible.device).reshape(-1)
    if not (visible.numel() == correct.numel() == alpha.numel()):
        raise ValueError("candidate teacher trust diagnostic tensors must have matching lengths")
    observed = visible > 0
    ratio = correct[observed] / visible[observed].clamp_min(1.0)
    quantiles = torch.quantile(alpha, torch.tensor([0.1, 0.5, 0.9], device=alpha.device))
    return {
        "trust_enabled": 1.0,
        "trust_warmup_active": 1.0 if warmup_active else 0.0,
        "trust_observed_landmark_count": float(observed.sum().item()),
        "trust_correct_landmark_count": float((correct > 0).sum().item()),
        "trust_visible_count_mean": float(visible[observed].mean().item()) if bool(observed.any()) else 0.0,
        "trust_correct_visible_ratio_mean": float(ratio.mean().item()) if ratio.numel() else 0.0,
        "trust_alpha_min": float(alpha.min().item()) if alpha.numel() else 0.0,
        "trust_alpha_p10": float(quantiles[0].item()) if alpha.numel() else 0.0,
        "trust_alpha_median": float(quantiles[1].item()) if alpha.numel() else 0.0,
        "trust_alpha_p90": float(quantiles[2].item()) if alpha.numel() else 0.0,
        "trust_alpha_max": float(alpha.max().item()) if alpha.numel() else 0.0,
    }


def failure_guided_pair_weights(camera_failure_scores, temperature=1.0, uniform_floor=0.1):
    """Convert per-camera failure EMAs into adjacent-pair sampling weights."""
    scores = torch.as_tensor(camera_failure_scores, dtype=torch.float32, device="cpu").reshape(-1)
    if scores.numel() < 2:
        return torch.empty(0, dtype=torch.float32)
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    pair_scores = 0.5 * (scores[:-1] + scores[1:])
    scale = pair_scores.std(unbiased=False).clamp_min(1e-6)
    logits = (pair_scores - pair_scores.mean()) / (scale * max(float(temperature), 1e-3))
    guided = torch.softmax(logits.clamp(-20.0, 20.0), dim=0)
    floor = max(0.0, min(1.0, float(uniform_floor)))
    uniform = torch.full_like(guided, 1.0 / max(guided.numel(), 1))
    return (1.0 - floor) * guided + floor * uniform


def update_camera_failure_ema(scores, index, failure, decay=0.9):
    failure = float(max(0.0, min(20.0, failure)))
    decay = max(0.0, min(1.0, float(decay)))
    scores[index] = decay * scores[index] + (1.0 - decay) * failure


def candidate_query_failure_score(teacher_losses):
    """Detached query-level failure used only to choose future rendered views."""
    terms = (
        teacher_losses.assignment,
        teacher_losses.hard_negative,
        teacher_losses.map_cleanliness,
        teacher_losses.map_bias,
        teacher_losses.matcher_reprojection_assignment,
        teacher_losses.map_directional_bias,
    )
    value = sum(float(term.detach().item()) for term in terms)
    return math.log1p(max(value, 0.0))


def render_online_candidate_query(
    cameras,
    gaussians,
    feature_extractor,
    longest_edge,
    background,
    *,
    alpha_min=0.35,
    alpha_max=0.65,
    return_provenance=False,
    provenance_landmark_indices=None,
    pair_weights=None,
    query_feature_contract="legacy_full_then_resized_map",
):
    """Render current RGB geometry and encode it without candidate-loss gradients."""
    candidate = sample_interpolated_novel_view(
        cameras,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        pair_weights=pair_weights,
    )
    reference = cameras[max(0, int(candidate.train_index_a))]
    fine_resolution = get_resolution_from_longest_edge(
        reference.original_image.shape[1],
        reference.original_image.shape[2],
        longest_edge,
    )
    pose_w2c = candidate.world_view_transform.transpose(0, 1).cuda()
    with torch.no_grad():
        render_pkg = render_from_pose_gsplat(
            gaussians,
            pose_w2c,
            candidate.FoVx,
            candidate.FoVy,
            fine_resolution[1],
            fine_resolution[0],
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=bool(return_provenance),
            rasterize_mode="antialiased",
        )
        feature_map = extract_normalized_feature_map(
            feature_extractor,
            render_pkg["render"].clamp(0.0, 1.0),
            size=fine_resolution,
            query_feature_contract=query_feature_contract,
        )
    if return_provenance:
        render_pkg["rgb_meta"]["rendered_depth"] = render_pkg.get("depth")
        if provenance_landmark_indices is not None:
            render_pkg["rgb_meta"] = compress_2dgs_rgb_meta_to_bank(
                render_pkg["rgb_meta"], provenance_landmark_indices
            )
    return candidate, feature_map, pose_w2c, render_pkg


def store_render_visible_mask(render_visible_masks, image_name, visible_mask):
    render_visible_masks[image_name] = visible_mask.detach().to(device="cpu", dtype=torch.bool)


def render_visible_mask_from_cache(render_visible_masks, image_name, device):
    visible_mask = render_visible_masks.get(image_name, None)
    if visible_mask is None:
        return None
    return visible_mask.to(device=device, dtype=torch.bool, non_blocking=True)


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = gaussians.__class__(gaussians.max_sh_degree)
    sampled_gaussians.active_sh_degree = gaussians.active_sh_degree
    sampled_gaussians.spatial_lr_scale = gaussians.spatial_lr_scale
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians.materialized_loc_feature(idx_sampled)
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    if torch.is_tensor(getattr(gaussians, "_loc_opacity", None)) and gaussians._loc_opacity.shape[0] == gaussians.get_xyz.shape[0]:
        sampled_gaussians._loc_opacity = gaussians._loc_opacity[idx_sampled]
    if torch.is_tensor(getattr(gaussians, "_loc_anchor_offset", None)) and gaussians._loc_anchor_offset.shape[0] == gaussians.get_xyz.shape[0]:
        sampled_gaussians._loc_anchor_offset = gaussians._loc_anchor_offset[idx_sampled]
    sampled_gaussians.surfel_loc_tangent_bound = float(getattr(gaussians, "surfel_loc_tangent_bound", 0.0) or 0.0)
    sampled_gaussians.surfel_loc_normal_bound = float(getattr(gaussians, "surfel_loc_normal_bound", 0.0) or 0.0)
    sampled_gaussians.max_radii2D = torch.zeros(
        sampled_gaussians.get_xyz.shape[0],
        dtype=torch.float32,
        device=sampled_gaussians.get_xyz.device,
    )
    return sampled_gaussians


@torch.no_grad()
def calculate_match_score(
    gaussians: GaussianModel,
    gt_feature_map,
    pose,
    K,
    render_visible_mask=None,
    img_mask=None,
):
    xyz = gaussian_localization_xyz(gaussians)
    feat = gaussians.get_loc_feature.squeeze()

    # project gaussians to image space
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    feat = feat[visible_mask]

    gs_feats = F.normalize(feat, p=2, dim=1)
    im_feats = gt_feature_map[:, xy[1], xy[0]].T
    score = (gs_feats * im_feats).sum(-1)
    return score, visible_mask


def generate_gt_map(
    gaussians: GaussianModel,
    gt_feature_map,
    idx_sampled,
    pose,
    K,
    render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussian_localization_xyz(gaussians)[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )
    
    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths

    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]

    gt_map[:, xy_pos[1], xy_pos[0]] = 1

    return gt_map


def _project_xyz_to_feature(xyz, pose, K, height, width):
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
    xyz_cam = (pose.to(device=xyz.device, dtype=xyz.dtype) @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths.clamp_min(1e-8)
    xy = (K.to(device=xyz.device, dtype=xyz.dtype) @ xyz_cam_norm)[:2]
    valid = (
        (depths > 1e-8)
        & (xy[0] >= 0)
        & (xy[0] <= width - 1)
        & (xy[1] >= 0)
        & (xy[1] <= height - 1)
    )
    return xy, valid


def _project_xyz_to_feature_with_depth(xyz, pose, K, height, width):
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
    xyz_cam = (pose.to(device=xyz.device, dtype=xyz.dtype) @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths.clamp_min(1e-8)
    xy = (K.to(device=xyz.device, dtype=xyz.dtype) @ xyz_cam_norm)[:2]
    valid = (
        (depths > 1e-8)
        & (xy[0] >= 0)
        & (xy[0] <= width - 1)
        & (xy[1] >= 0)
        & (xy[1] <= height - 1)
    )
    return xy, valid, depths


def _calibrated_utility_weights(utility, min_weight=1.0, max_weight=2.0):
    if utility is None:
        return None
    utility = utility.float().reshape(-1)
    if utility.numel() == 0:
        return utility
    center = utility.median()
    scale = (utility - center).abs().median().clamp_min(1e-6)
    normalized = (utility - center) / scale
    return min_weight + (max_weight - min_weight) * torch.sigmoid(normalized)


def _meta_vector(landmark_meta, key, count):
    if landmark_meta is None or key not in landmark_meta:
        return None
    value = torch.as_tensor(landmark_meta[key], dtype=torch.float32).reshape(-1)
    if value.numel() == 1:
        value = value.expand(count)
    if value.numel() < count:
        pad = value[-1].expand(count - value.numel()) if value.numel() else torch.zeros(count)
        value = torch.cat([value, pad], dim=0)
    return value[:count]


def _first_meta_vector(landmark_meta, keys, count):
    for key in keys:
        value = _meta_vector(landmark_meta, key, count)
        if value is not None:
            return value
    return None


def _quality_factor(values, floor=0.25):
    values = values.float()
    finite = torch.isfinite(values)
    factor = torch.zeros_like(values)
    if finite.any():
        finite_values = values[finite].clamp_min(0.0)
        if finite_values.numel() > 0 and float(finite_values.max().item()) > 1.0:
            finite_values = finite_values / finite_values.max().clamp_min(1e-6)
        factor[finite] = finite_values.clamp(0.0, 1.0)
    floor = max(0.0, min(float(floor), 1.0))
    return floor + (1.0 - floor) * factor


def _has_any_meta_key(landmark_meta, keys):
    return landmark_meta is not None and any(key in landmark_meta for key in keys)


def _error_cleanliness(values, scale=4.0):
    values = values.float()
    scale = max(float(scale), 1e-6)
    clean = torch.exp(-values.clamp_min(0.0) / scale).clamp(0.0, 1.0)
    clean[~torch.isfinite(clean)] = 0.0
    return clean


def _inverse_count_balance(ids, count):
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    device = ids.device
    balance = torch.ones(count, dtype=torch.float32, device=device)
    if ids.numel() < count:
        pad = torch.full((count - ids.numel(),), -1, dtype=torch.long, device=device)
        ids = torch.cat([ids, pad], dim=0)
    ids = ids[:count]
    valid = ids >= 0
    if not bool(valid.any().item()):
        return balance
    unique, counts = torch.unique(ids[valid], return_counts=True)
    count_map = torch.zeros(int(unique.max().item()) + 1, dtype=torch.float32, device=device)
    count_map[unique] = counts.to(dtype=torch.float32)
    balance[valid] = torch.rsqrt(count_map[ids[valid]].clamp_min(1.0))
    return balance / balance.mean().clamp_min(1e-6)


def _coverage_spatial_balance_from_meta(landmark_meta, count):
    uv = landmark_meta.get("coverage_uv") if landmark_meta is not None else None
    if uv is None:
        return None
    uv = torch.as_tensor(uv, dtype=torch.float32).reshape(-1, 2)
    if uv.numel() == 0:
        return None
    if uv.shape[0] < count:
        pad = uv[-1:].expand(count - uv.shape[0], -1) if uv.shape[0] else torch.zeros(count, 2)
        uv = torch.cat([uv, pad], dim=0)
    uv = uv[:count]
    grid_size_value = landmark_meta.get("coverage_grid_size", 4)
    grid_size = int(torch.as_tensor(grid_size_value).reshape(-1)[0].item()) if grid_size_value is not None else 4
    if grid_size <= 1:
        return None
    image_size = landmark_meta.get("coverage_image_size")
    if image_size is not None:
        image_size = torch.as_tensor(image_size, dtype=torch.float32).reshape(-1)
    if image_size is not None and image_size.numel() >= 2 and float(image_size[0]) > 0 and float(image_size[1]) > 0:
        height = float(image_size[0])
        width = float(image_size[1])
        x = torch.floor(uv[:, 0].clamp(0, width - 1) / max(width, 1.0) * grid_size)
        y = torch.floor(uv[:, 1].clamp(0, height - 1) / max(height, 1.0) * grid_size)
    else:
        finite = torch.isfinite(uv).all(dim=1)
        if not bool(finite.any().item()):
            return None
        min_xy = uv[finite].min(dim=0).values
        span = (uv[finite].max(dim=0).values - min_xy).clamp_min(1e-6)
        normalized = (uv - min_xy) / span
        x = torch.floor(normalized[:, 0].clamp(0, 1) * grid_size)
        y = torch.floor(normalized[:, 1].clamp(0, 1) * grid_size)
    x = x.to(dtype=torch.long).clamp(0, grid_size - 1)
    y = y.to(dtype=torch.long).clamp(0, grid_size - 1)
    return _inverse_count_balance(y * grid_size + x, count)


def _coverage_depth_balance_from_meta(landmark_meta, count):
    depth = landmark_meta.get("coverage_depth") if landmark_meta is not None else None
    if depth is None:
        return None
    depth = torch.as_tensor(depth, dtype=torch.float32).reshape(-1)
    if depth.numel() == 0:
        return None
    if depth.numel() < count:
        pad = depth[-1].expand(count - depth.numel())
        depth = torch.cat([depth, pad], dim=0)
    depth = depth[:count]
    bins_value = landmark_meta.get("coverage_depth_bins", 4)
    bins = int(torch.as_tensor(bins_value).reshape(-1)[0].item()) if bins_value is not None else 4
    if bins <= 1:
        return None
    finite = torch.isfinite(depth)
    if not bool(finite.any().item()):
        return None
    selected = depth[finite]
    span = (selected.max() - selected.min()).clamp_min(1e-6)
    ids = torch.full((count,), -1, dtype=torch.long, device=depth.device)
    ids[finite] = torch.floor((selected - selected.min()) / span * bins).to(dtype=torch.long).clamp(0, bins - 1)
    return _inverse_count_balance(ids, count)


def final_candidate_quality_from_meta(
    landmark_meta,
    count,
    reprojection_error_scale=4.0,
    cleanliness_weight=1.0,
    pose_info_weight=1.0,
    balance_weight=1.0,
    reliability_weight=0.25,
    utility_weight=0.0,
):
    """Compose detector supervision from final-candidate localization quality signals."""
    device = None
    if landmark_meta is not None:
        for value in landmark_meta.values():
            if torch.is_tensor(value):
                device = value.device
                break
    if device is None:
        device = torch.device("cpu")
    count = int(count)
    ones = torch.ones(count, dtype=torch.float32, device=device)
    eps = 1e-6

    raw_precision = _first_meta_vector(
        landmark_meta,
        ("raw_gt_precision_2px", "all_gt_precision_2px", "gt_precision_2px"),
        count,
    )
    inlier_precision = _first_meta_vector(
        landmark_meta,
        ("inlier_gt_precision_2px", "pnp_inlier_precision_2px"),
        count,
    )
    explicit_cleanliness = _first_meta_vector(
        landmark_meta,
        ("candidate_cleanliness", "gt_cleanliness", "gt_precision_6px", "raw_gt_precision_4px", "all_gt_precision_4px"),
        count,
    )
    clean_parts = []
    if raw_precision is not None:
        clean_parts.append(_quality_factor(raw_precision.to(device), floor=0.0))
    if inlier_precision is not None:
        clean_parts.append(_quality_factor(inlier_precision.to(device), floor=0.0))
    if explicit_cleanliness is not None:
        clean_parts.append(_quality_factor(explicit_cleanliness.to(device), floor=0.0))
    reproj_error = _first_meta_vector(
        landmark_meta,
        ("reproj_error", "gt_reproj_error", "gt_reproj_px"),
        count,
    )
    if reproj_error is not None:
        clean_parts.append(_error_cleanliness(reproj_error.to(device), scale=reprojection_error_scale))
    if clean_parts:
        cleanliness = torch.stack(clean_parts, dim=0).clamp_min(eps).log().mean(dim=0).exp()
    else:
        cleanliness = ones

    pose = _first_meta_vector(
        landmark_meta,
        ("pose_info_contribution", "pose_min_eig", "information", "pose_information"),
        count,
    )
    if pose is not None:
        pose_info = _quality_factor(pose.to(device), floor=0.05)
    else:
        pose_info = ones

    spatial_balance = _first_meta_vector(
        landmark_meta,
        ("spatial_balance", "geometry_balance"),
        count,
    )
    if spatial_balance is None and landmark_meta is not None:
        spatial_balance = _coverage_spatial_balance_from_meta(landmark_meta, count)
    if spatial_balance is not None:
        spatial_balance = _quality_factor(spatial_balance.to(device), floor=0.05)
    else:
        spatial_balance = ones

    depth_balance = _meta_vector(landmark_meta, "depth_balance", count)
    if depth_balance is None and landmark_meta is not None:
        depth_balance = _coverage_depth_balance_from_meta(landmark_meta, count)
    if depth_balance is not None:
        depth_balance = _quality_factor(depth_balance.to(device), floor=0.05)
    else:
        depth_balance = ones
    balance = (spatial_balance.clamp_min(eps) * depth_balance.clamp_min(eps)).sqrt()

    reliability_parts = []
    repeatability = _meta_vector(landmark_meta, "repeatability", count)
    if repeatability is not None:
        reliability_parts.append(_quality_factor(repeatability.to(device), floor=0.05))
    positive_prob = _meta_vector(landmark_meta, "positive_prob", count)
    if positive_prob is not None:
        reliability_parts.append(_quality_factor(positive_prob.to(device), floor=0.05))
    margin = _meta_vector(landmark_meta, "margin", count)
    if margin is not None:
        reliability_parts.append(_quality_factor(margin.to(device), floor=0.05))
    outlier = _meta_vector(landmark_meta, "outlier", count)
    if outlier is not None:
        reliability_parts.append((1.0 - _quality_factor(outlier.to(device), floor=0.0)).clamp(0.0, 1.0))
    if reliability_parts:
        reliability = torch.stack(reliability_parts, dim=0).clamp_min(eps).log().mean(dim=0).exp()
    else:
        reliability = ones

    utility = _meta_vector(landmark_meta, "utility", count)
    if utility is not None:
        utility_quality = _quality_factor(utility.to(device), floor=0.05)
    else:
        utility_quality = ones

    weighted_logs = []
    weight_sum = 0.0
    for value, weight in (
        (cleanliness, cleanliness_weight),
        (pose_info, pose_info_weight),
        (balance, balance_weight),
        (reliability, reliability_weight),
        (utility_quality, utility_weight),
    ):
        weight = max(0.0, float(weight))
        if weight <= 0.0:
            continue
        weighted_logs.append(weight * value.clamp_min(eps).log())
        weight_sum += weight

    if not weighted_logs:
        quality = ones
    else:
        quality = (torch.stack(weighted_logs, dim=0).sum(dim=0) / max(weight_sum, eps)).exp()
    quality[~torch.isfinite(quality)] = 0.0
    components = {
        "candidate_quality": quality.clamp(0.0, 1.0),
        "candidate_cleanliness": cleanliness.clamp(0.0, 1.0),
        "pose_info_contribution": pose_info.clamp(0.0, 1.0),
        "spatial_balance": spatial_balance.clamp(0.0, 1.0),
        "depth_balance": depth_balance.clamp(0.0, 1.0),
        "candidate_balance": balance.clamp(0.0, 1.0),
        "candidate_reliability": reliability.clamp(0.0, 1.0),
    }
    return components["candidate_quality"], components


def detector_landmark_quality_from_meta(landmark_meta, count, reprojection_error_scale=8.0):
    if landmark_meta is None:
        return None
    candidate_quality = _meta_vector(landmark_meta, "candidate_quality", count)
    if candidate_quality is not None:
        candidate_quality[~torch.isfinite(candidate_quality)] = 0.0
        return candidate_quality.clamp_min(0.0)
    final_signal_keys = (
        "candidate_cleanliness",
        "gt_cleanliness",
        "raw_gt_precision_2px",
        "all_gt_precision_2px",
        "gt_precision_2px",
        "inlier_gt_precision_2px",
        "pnp_inlier_precision_2px",
        "pose_info_contribution",
        "pose_min_eig",
        "reproj_error",
        "gt_reproj_error",
        "gt_reproj_px",
        "depth_balance",
        "spatial_balance",
        "geometry_balance",
        "coverage_uv",
        "coverage_depth",
    )
    if _has_any_meta_key(landmark_meta, final_signal_keys):
        quality, _ = final_candidate_quality_from_meta(
            landmark_meta,
            count,
            reprojection_error_scale=reprojection_error_scale,
        )
        return quality
    quality = _meta_vector(landmark_meta, "utility", count)
    if quality is None:
        quality = torch.ones(count, dtype=torch.float32)
    else:
        quality = quality.float().clamp_min(0.0)

    information = _first_meta_vector(landmark_meta, ("pose_min_eig", "information", "pose_information"), count)
    if information is not None:
        quality = quality * _quality_factor(information, floor=0.25)

    raw_precision = _first_meta_vector(
        landmark_meta,
        ("raw_gt_precision_2px", "all_gt_precision_2px", "gt_precision_2px"),
        count,
    )
    if raw_precision is not None:
        quality = quality * _quality_factor(raw_precision, floor=0.1)

    inlier_precision = _first_meta_vector(
        landmark_meta,
        ("inlier_gt_precision_2px", "pnp_inlier_precision_2px"),
        count,
    )
    if inlier_precision is not None:
        quality = quality * _quality_factor(inlier_precision, floor=0.1)

    precision = _first_meta_vector(
        landmark_meta,
        ("gt_precision_6px", "raw_gt_precision_4px", "all_gt_precision_4px"),
        count,
    )
    if precision is not None:
        quality = quality * _quality_factor(precision, floor=0.25)

    for balance_key in ("depth_balance", "spatial_balance", "geometry_balance"):
        balance = _meta_vector(landmark_meta, balance_key, count)
        if balance is not None:
            quality = quality * _quality_factor(balance, floor=0.25)
    if not any(key in landmark_meta for key in ("spatial_balance", "geometry_balance")):
        spatial_balance = _coverage_spatial_balance_from_meta(landmark_meta, count)
        if spatial_balance is not None:
            quality = quality * _quality_factor(spatial_balance, floor=0.25)
    if "depth_balance" not in landmark_meta:
        depth_balance = _coverage_depth_balance_from_meta(landmark_meta, count)
        if depth_balance is not None:
            quality = quality * _quality_factor(depth_balance, floor=0.25)

    reproj_error = _meta_vector(landmark_meta, "reproj_error", count)
    if reproj_error is None:
        reproj_error = _meta_vector(landmark_meta, "gt_reproj_error", count)
    if reproj_error is None:
        reproj_error = _meta_vector(landmark_meta, "gt_reproj_px", count)
    if reproj_error is not None:
        scale = max(float(reprojection_error_scale), 1e-6)
        reproj_quality = torch.exp(-reproj_error.float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
        quality = quality * reproj_quality

    quality[~torch.isfinite(quality)] = 0.0
    return quality.clamp_min(0.0)


def generate_weighted_hard_gt_map(
    xyz,
    gt_feature_map,
    pose,
    K,
    utility=None,
    render_visible_mask=None,
):
    """Generate hard detector peaks and a calibrated utility loss-weight map."""
    height, width = gt_feature_map.shape[1], gt_feature_map.shape[2]
    device = gt_feature_map.device
    dtype = gt_feature_map.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    xy, valid = _project_xyz_to_feature(xyz, pose, K, height, width)
    if render_visible_mask is not None:
        valid = valid & render_visible_mask.to(device=device, dtype=torch.bool)

    gt_flat = torch.zeros(height * width, device=device, dtype=dtype)
    weight_flat = torch.ones(height * width, device=device, dtype=dtype)
    if valid.sum() == 0:
        return gt_flat.view(height, width)[None], weight_flat.view(height, width)[None]

    xy_int = xy[:, valid].to(dtype=torch.long)
    flat_idx = xy_int[1] * width + xy_int[0]
    gt_flat[flat_idx] = 1.0

    utility_weights = _calibrated_utility_weights(utility)
    if utility_weights is not None:
        utility_weights = utility_weights.to(device=device, dtype=dtype)[valid]
        weight_flat.scatter_reduce_(
            0,
            flat_idx,
            utility_weights,
            reduce="amax",
            include_self=True,
        )
    return gt_flat.view(height, width)[None].detach(), weight_flat.view(height, width)[None].detach()


def generate_soft_gt_map(
    xyz,
    gt_feature_map,
    pose,
    K,
    utility=None,
    render_visible_mask=None,
    soft_sigma=1.5,
):
    """Generate local Gaussian detector targets without allocating a full image meshgrid per landmark."""
    height, width = gt_feature_map.shape[1], gt_feature_map.shape[2]
    device = gt_feature_map.device
    dtype = gt_feature_map.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    xy, valid = _project_xyz_to_feature(xyz, pose, K, height, width)
    if render_visible_mask is not None:
        valid = valid & render_visible_mask.to(device=device, dtype=torch.bool)

    gt_flat = torch.zeros(height * width, device=device, dtype=dtype)
    weight_flat = torch.ones(height * width, device=device, dtype=dtype)
    if valid.sum() == 0:
        return gt_flat.view(height, width)[None], weight_flat.view(height, width)[None]

    sigma = max(float(soft_sigma), 1e-6)
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    offsets = torch.arange(-radius, radius + 1, device=device)
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    off_x = off_x.reshape(1, -1)
    off_y = off_y.reshape(1, -1)

    centers = xy[:, valid].T
    centers_int = centers.round().to(dtype=torch.long)
    px = centers_int[:, 0:1] + off_x
    py = centers_int[:, 1:2] + off_y
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    dist2 = (px.to(dtype=dtype) - centers[:, 0:1]).pow(2) + (py.to(dtype=dtype) - centers[:, 1:2]).pow(2)
    values = torch.exp(-0.5 * dist2 / (sigma * sigma)).to(dtype=dtype)
    flat_idx = py * width + px

    gt_flat.scatter_reduce_(
        0,
        flat_idx[in_image].to(dtype=torch.long),
        values[in_image],
        reduce="amax",
        include_self=True,
    )

    utility_weights = _calibrated_utility_weights(utility)
    if utility_weights is not None:
        utility_weights = utility_weights.to(device=device, dtype=dtype)[valid]
        weight_values = 1.0 + (utility_weights[:, None] - 1.0) * values
        weight_flat.scatter_reduce_(
            0,
            flat_idx[in_image].to(dtype=torch.long),
            weight_values[in_image],
            reduce="amax",
            include_self=True,
        )
    return gt_flat.view(height, width)[None].detach(), weight_flat.view(height, width)[None].detach()


def utility_weighted_detector_loss(pred, target, weight_map=None, gamma=2.0, alpha=0.25):
    """Focal BCE for detector targets with optional calibrated utility weights."""
    target = target.float()
    pred = pred.float()
    if pred.min() < 0 or pred.max() > 1:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
    else:
        prob = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(prob, target, reduction="none")
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * (1.0 - pt).pow(gamma) * bce
    weights = weight_map.float() if weight_map is not None else 1.0 + target
    return (focal * weights).sum() / weights.sum().clamp_min(1e-6)


def build_detector_target_map(
    gaussians,
    gt_feature_map,
    sampled_idx,
    pose,
    K,
    render_visible_mask=None,
    detector_target_mode="hard",
    landmark_meta=None,
    soft_sigma=1.5,
    landmark_xyz=None,
):
    target_xyz = (
        gaussian_localization_xyz(gaussians)[sampled_idx]
        if landmark_xyz is None
        else torch.as_tensor(
            landmark_xyz,
            device=gt_feature_map.device,
            dtype=gt_feature_map.dtype,
        ).reshape(sampled_idx.numel(), 3)
    )
    if detector_target_mode == "soft":
        utility = detector_landmark_quality_from_meta(landmark_meta, int(sampled_idx.numel()))
        sampled_visible = None
        if render_visible_mask is not None:
            sampled_visible = render_visible_mask[sampled_idx]
        gt_map, weight_map = generate_soft_gt_map(
            target_xyz,
            gt_feature_map,
            pose,
            K,
            utility=utility,
            render_visible_mask=sampled_visible,
            soft_sigma=soft_sigma,
        )
        return gt_map, True, weight_map
    if detector_target_mode == "weighted_hard":
        utility = detector_landmark_quality_from_meta(landmark_meta, int(sampled_idx.numel()))
        sampled_visible = None
        if render_visible_mask is not None:
            sampled_visible = render_visible_mask[sampled_idx]
        gt_map, weight_map = generate_weighted_hard_gt_map(
            target_xyz,
            gt_feature_map,
            pose,
            K,
            utility=utility,
            render_visible_mask=sampled_visible,
        )
        return gt_map, True, weight_map
    if detector_target_mode != "hard":
        raise ValueError(f"Unknown detector_target_mode: {detector_target_mode}")
    if landmark_xyz is None:
        gt_map = generate_gt_map(
            gaussians,
            gt_feature_map,
            sampled_idx,
            pose,
            K,
            render_visible_mask,
        )
    else:
        sampled_visible = (
            None
            if render_visible_mask is None
            else render_visible_mask[sampled_idx]
        )
        gt_map, _ = generate_weighted_hard_gt_map(
            target_xyz,
            gt_feature_map,
            pose,
            K,
            render_visible_mask=sampled_visible,
        )
    return gt_map, False, None


def detector_target_loss(heat_map, gt_map, soft_target=False, weight_map=None):
    if soft_target:
        return utility_weighted_detector_loss(heat_map, gt_map, weight_map=weight_map)
    return score_map_bce_loss(heat_map, gt_map)


def detector_proposal_preservation_loss(
    student_keypoint,
    student_matchability,
    teacher_keypoint,
    teacher_matchability,
    valid_mask=None,
):
    """Keep the deployed combined proposal map close to a frozen detector snapshot."""
    if student_keypoint.shape != teacher_keypoint.shape:
        raise ValueError("student and teacher keypoint heatmaps must have the same shape")
    if student_matchability.shape != teacher_matchability.shape:
        raise ValueError(
            "student and teacher matchability heatmaps must have the same shape"
        )
    student = torch.sqrt((student_keypoint * student_matchability).clamp_min(0.0))
    teacher = torch.sqrt(
        (teacher_keypoint.detach() * teacher_matchability.detach()).clamp_min(0.0)
    )
    squared_error = (student - teacher).square()
    if valid_mask is None:
        return squared_error.mean()
    mask = valid_mask.to(device=squared_error.device, dtype=squared_error.dtype)
    try:
        mask = torch.broadcast_to(mask, squared_error.shape)
    except RuntimeError as exc:
        raise ValueError(
            "detector preservation mask is not broadcastable to the proposal map"
        ) from exc
    if not bool(torch.any(mask > 0).item()):
        return squared_error.sum() * 0.0
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def validate_detector_only_candidate_teacher_configuration(
    *,
    enabled,
    sparse_candidate_teacher,
    optimize_features=False,
    freeze_detector=False,
    dustbin_weight=0.0,
    pair_scorer_weight=0.0,
    pair_scorer_assignment_weight=0.0,
    pair_measurement_inlier_weight=0.0,
    pair_measurement_nll_weight=0.0,
    pair_measurement_bias_weight=0.0,
    pair_measurement_covariance_weight=0.0,
):
    """Reject ambiguous detector-only stages before optimizer construction."""
    if not bool(enabled):
        return
    if not bool(sparse_candidate_teacher):
        raise ValueError(
            "candidate_teacher_detector_only requires --sparse_candidate_teacher"
        )

    conflicts = []
    if bool(optimize_features):
        conflicts.append("candidate_teacher_optimize_features")
    if bool(freeze_detector):
        conflicts.append("candidate_teacher_freeze_detector")
    if float(dustbin_weight) != 0.0:
        conflicts.append("candidate_teacher_dustbin_weight")
    if float(pair_scorer_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_scorer_weight")
    if float(pair_scorer_assignment_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_scorer_assignment_weight")
    if float(pair_measurement_inlier_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_measurement_inlier_weight")
    if float(pair_measurement_nll_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_measurement_nll_weight")
    if float(pair_measurement_bias_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_measurement_bias_weight")
    if float(pair_measurement_covariance_weight) != 0.0:
        conflicts.append("candidate_teacher_pair_measurement_covariance_weight")
    if conflicts:
        raise ValueError(
            "candidate_teacher_detector_only permits only detector parameters; "
            "disable: " + ", ".join(conflicts)
        )


@torch.no_grad()
def random_knn_score(points, npoints, score, k=32, query_chunk=512, point_chunk=65536):
    points = points.detach()
    device = points.device
    dtype = torch.float32 if not points.is_floating_point() else points.dtype
    points = points.to(device=device, dtype=dtype)
    score = score.to(device=device, dtype=torch.float32).reshape(-1)
    total = int(points.shape[0])
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    npoints = min(int(npoints), total)
    k = max(1, min(int(k), total))
    sampled_idx = torch.randperm(total, device=device)[:npoints]
    selected = []
    selected_set = set()

    for q_start in range(0, npoints, int(query_chunk)):
        q_end = min(q_start + int(query_chunk), npoints)
        query = points[sampled_idx[q_start:q_end]]
        q_count = query.shape[0]
        best_dist = torch.full((q_count, k), float("inf"), dtype=dtype, device=device)
        best_idx = torch.full((q_count, k), -1, dtype=torch.long, device=device)

        for p_start in range(0, total, int(point_chunk)):
            p_end = min(p_start + int(point_chunk), total)
            dist = torch.cdist(query, points[p_start:p_end])
            local_k = min(k, p_end - p_start)
            local_dist, local_idx = torch.topk(dist, local_k, largest=False, dim=-1)
            local_idx = local_idx + p_start

            merged_dist = torch.cat([best_dist, local_dist], dim=1)
            merged_idx = torch.cat([best_idx, local_idx], dim=1)
            best_dist, order = torch.topk(merged_dist, k, largest=False, dim=-1)
            best_idx = torch.gather(merged_idx, 1, order)
            del dist, local_dist, local_idx, merged_dist, merged_idx, order

        knn_score = score[best_idx.clamp_min(0)]
        score_order = torch.argsort(knn_score, descending=True, dim=-1)
        best_idx_cpu = best_idx.detach().cpu()
        score_order_cpu = score_order.detach().cpu()
        fallback_cpu = sampled_idx[q_start:q_end].detach().cpu()
        for row in range(q_count):
            chosen = None
            for col in score_order_cpu[row].tolist():
                idx = int(best_idx_cpu[row, col].item())
                if idx >= 0 and idx not in selected_set:
                    chosen = idx
                    break
            if chosen is None:
                chosen = int(fallback_cpu[row].item())
            if chosen not in selected_set:
                selected_set.add(chosen)
                selected.append(chosen)
        del best_dist, best_idx, knn_score, score_order

    return torch.tensor(selected, dtype=torch.long, device=device)


def matching_oriented_sample(
    scene,
    gaussians,
    feature_extractor,
    render_visible_masks,
    masks=None,
    num=16384,
    k=32,
    return_coverage_stats=False,
    query_feature_contract="legacy_full_then_resized_map",
):
    viewpoint_stack = scene.getTrainCameras().copy()
    loc_xyz = gaussian_localization_xyz(gaussians)
    score_sum = torch.zeros(
        loc_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    score_num = torch.zeros(loc_xyz.shape[0], dtype=torch.int, device="cuda")
    uv_sum = torch.zeros((loc_xyz.shape[0], 2), dtype=torch.float32, device="cuda")
    depth_sum = torch.zeros(loc_xyz.shape[0], dtype=torch.float32, device="cuda")
    for viewpoint_cam in tqdm(viewpoint_stack, desc="Match Score"):
        fine_resolution = get_resolution_from_longest_edge(
            viewpoint_cam.original_image.shape[1],
            viewpoint_cam.original_image.shape[2],
            scene.longest_edge,
        )
        gt_image = viewpoint_cam.original_image.cuda()
        query_valid_mask = camera_valid_mask(masks, viewpoint_cam, gt_image.device)
        gt_feature_map = extract_normalized_feature_map(
            feature_extractor,
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
            query_feature_contract=query_feature_contract,
            valid_mask=query_valid_mask,
        )

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        # print("focal:", focalX, focalY)
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        render_visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            viewpoint_cam.image_name,
            gt_feature_map.device,
        )
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            store_render_visible_mask(
                render_visible_masks,
                viewpoint_cam.image_name,
                render_visible_mask,
            )
        img_mask = (
            _resize_hard_valid_mask(
                query_valid_mask,
                gt_feature_map.shape[-2:],
                gt_feature_map.device,
            )[None]
            if query_valid_mask is not None
            else None
        )

        score, mask = calculate_match_score(
            gaussians,
            gt_feature_map,
            viewmat,
            K,
            render_visible_mask=render_visible_mask,
            img_mask=img_mask,
        )
        score_num[mask] += 1
        score_sum[mask] += score
        if return_coverage_stats and bool(mask.any().item()):
            xy, project_valid, depths = _project_xyz_to_feature_with_depth(
                loc_xyz,
                viewmat,
                K,
                gt_feature_map.shape[1],
                gt_feature_map.shape[2],
            )
            stat_mask = mask & project_valid
            uv_sum[stat_mask] += xy[:, stat_mask].T.float()
            depth_sum[stat_mask] += depths[stat_mask].float()

    observation_count = score_num.clone()
    score_num[score_num == 0] = 1  # avoid divide by zero
    score_avg = score_sum / score_num

    sampled_idx = random_knn_score(loc_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    if return_coverage_stats:
        denom = observation_count.clamp_min(1).to(dtype=torch.float32)
        coverage_stats = {
            "uv": uv_sum / denom[:, None],
            "depth": depth_sum / denom,
            "observed": observation_count > 0,
            "image_size": torch.tensor(
                [fine_resolution[0], fine_resolution[1]],
                dtype=torch.long,
                device=loc_xyz.device,
            ),
        }
        return sampled_idx, score_avg, score_num, coverage_stats
    return sampled_idx, score_avg, score_num


def validate_detector_sampled_indices(
    sampled_idx,
    sampling_mode="baseline",
    min_loc_observations=0,
    point_count=None,
):
    sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1)
    if sampled_idx.numel() == 0:
        raise ValueError(
            "sampled 0 detector landmarks; "
            f"sampling_mode={sampling_mode}, min_loc_observations={min_loc_observations}. "
            "Check that the loaded Gaussian map has localization observations for localization-aware sampling, "
            "or lower min_loc_observations/use baseline sampling for detector-only ablations."
        )
    if point_count is not None:
        min_idx = int(sampled_idx.min().item())
        max_idx = int(sampled_idx.max().item())
        if min_idx < 0 or max_idx >= int(point_count):
            raise ValueError(
                "detector landmark indices are outside the Gaussian point cloud; "
                f"point_count={int(point_count)}, min_idx={min_idx}, max_idx={max_idx}"
            )
    return sampled_idx


def detector_sampling_observed_mask(loc_observation_count, min_loc_observations=1, coverage_stats=None):
    observed = torch.as_tensor(loc_observation_count) >= int(min_loc_observations)
    if coverage_stats is None or "observed" not in coverage_stats:
        return observed
    coverage_observed = torch.as_tensor(
        coverage_stats["observed"],
        device=observed.device,
        dtype=torch.bool,
    )
    if coverage_observed.shape != observed.shape:
        raise ValueError(
            "coverage_stats['observed'] shape does not match localization observation count: "
            f"{tuple(coverage_observed.shape)} vs {tuple(observed.shape)}"
        )
    return observed & coverage_observed


def load_precomputed_detector_landmarks(path, point_count=None, device=None):
    with open(path, "rb") as handle:
        sampled_idx = pickle.load(handle)
    sampled_idx = validate_detector_sampled_indices(
        sampled_idx,
        sampling_mode="precomputed",
        point_count=point_count,
    )
    if device is not None:
        sampled_idx = sampled_idx.to(device=device, dtype=torch.long)
    return sampled_idx


def save_detector_sampled_indices(path, sampled_idx):
    sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).detach().cpu()
    with open(path, "wb") as handle:
        pickle.dump(sampled_idx, handle)


def validation_hypothesis_indices_per_keypoint(
    keypoint_idx,
    scores,
    max_matches_per_keypoint,
):
    """Apply validation-only keypoint quotas on CPU to avoid CUDA peak memory."""
    keypoint_idx = keypoint_idx.detach().to(device="cpu", dtype=torch.long)
    scores = scores.detach().to(device="cpu")
    hypothesis_ids = torch.arange(scores.numel(), dtype=torch.long)
    limited = limit_matches_per_keypoint(
        SparseMatchResult(keypoint_idx, hypothesis_ids, scores),
        max_matches_per_keypoint,
    )
    return limited.landmark_idx


def load_precomputed_landmark_meta(path, device="cuda"):
    meta_path = os.path.join(os.path.dirname(path), "landmark_meta.pt")
    if not os.path.exists(meta_path):
        return None
    return torch.load(meta_path, map_location=device)


def _gaussian_localization_vector(gaussians, name, count, default=0.0):
    value = getattr(gaussians, name, None)
    if torch.is_tensor(value):
        value = value.detach().float().reshape(-1)
        if value.numel() >= count:
            return value[:count]
        if value.numel() > 0:
            return torch.cat([value, value[-1].expand(count - value.numel())], dim=0)
    device = gaussians.get_xyz.device
    return torch.full((count,), float(default), dtype=torch.float32, device=device)


def final_candidate_quality_from_gaussians(
    gaussians,
    min_observations=4,
    coverage_stats=None,
    reprojection_error_scale=4.0,
    cleanliness_weight=1.0,
    pose_info_weight=1.0,
    balance_weight=1.0,
    reliability_weight=0.25,
    utility_weight=0.0,
):
    count = int(gaussians.get_xyz.shape[0])
    legacy_utility = gaussians.compute_localization_utility(min_observations=min_observations)
    meta = {
        "utility": legacy_utility.detach(),
        "repeatability": _gaussian_localization_vector(gaussians, "loc_repeatability_ema", count),
        "positive_prob": _gaussian_localization_vector(gaussians, "loc_positive_prob_ema", count),
        "margin": _gaussian_localization_vector(gaussians, "loc_margin_ema", count),
        "outlier": _gaussian_localization_vector(gaussians, "loc_outlier_ema", count),
        "reproj_error": _gaussian_localization_vector(gaussians, "loc_reproj_error_ema", count),
        "information": _gaussian_localization_vector(gaussians, "loc_information_ema", count),
    }
    if coverage_stats is not None:
        if coverage_stats.get("uv") is not None:
            meta["coverage_uv"] = coverage_stats["uv"]
        if coverage_stats.get("depth") is not None:
            meta["coverage_depth"] = coverage_stats["depth"]
        if coverage_stats.get("image_size") is not None:
            meta["coverage_image_size"] = coverage_stats["image_size"]
        if "coverage_grid_size" in coverage_stats:
            meta["coverage_grid_size"] = coverage_stats["coverage_grid_size"]
        if "coverage_depth_bins" in coverage_stats:
            meta["coverage_depth_bins"] = coverage_stats["coverage_depth_bins"]
    quality, components = final_candidate_quality_from_meta(
        meta,
        count,
        reprojection_error_scale=reprojection_error_scale,
        cleanliness_weight=cleanliness_weight,
        pose_info_weight=pose_info_weight,
        balance_weight=balance_weight,
        reliability_weight=reliability_weight,
        utility_weight=utility_weight,
    )
    observed = detector_sampling_observed_mask(
        gaussians.loc_observation_count,
        min_loc_observations=min_observations,
        coverage_stats=coverage_stats,
    )
    quality = quality.to(device=gaussians.get_xyz.device, dtype=torch.float32)
    quality = quality.masked_fill(~observed.to(device=quality.device, dtype=torch.bool), 0.0)
    components["candidate_quality"] = quality
    components["legacy_utility"] = legacy_utility.detach()
    return quality, components


def evaluate_detector(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    scene,
    masks=None,
    render_visible_masks=None,
    tb_writer=None,
    iteration=0,
    query_feature_contract="legacy_full_then_resized_map",
):
    torch.cuda.empty_cache()

    landmarks = get_sampled_gaussian(gaussians, sampled_idx)

    bg_color = [1, 1, 1] if scene.args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            fine_resolution = get_resolution_from_longest_edge(
                config["cameras"][0].original_image.shape[1],
                config["cameras"][0].original_image.shape[2],
                scene.longest_edge,
            )
            loss_sum = 0.0
            for idx, viewpoint_cam in enumerate(config["cameras"]):
                gt_image = viewpoint_cam.original_image.cuda()
                query_valid_mask = camera_valid_mask(
                    masks, viewpoint_cam, gt_image.device
                )
                gt_feature_map = extract_normalized_feature_map(
                    feature_extractor,
                    gt_image,
                    size=(fine_resolution[0], fine_resolution[1]),
                    query_feature_contract=query_feature_contract,
                    valid_mask=query_valid_mask,
                )

                viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
                focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
                focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
                # print("focal:", focalX, focalY)
                K = torch.tensor(
                    [
                        [focalX, 0.0, gt_feature_map.shape[2] / 2],
                        [0.0, focalY, gt_feature_map.shape[1] / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                    device="cuda",
                )
                visible_mask = render_visible_mask_from_cache(
                    render_visible_masks,
                    viewpoint_cam.image_name,
                    gt_feature_map.device,
                )
                if visible_mask is None:
                    visible_mask = get_render_visible_mask(
                        gaussians,
                        viewpoint_cam,
                        gt_feature_map.shape[2],
                        gt_feature_map.shape[1],
                    )
                    store_render_visible_mask(
                        render_visible_masks,
                        viewpoint_cam.image_name,
                        visible_mask,
                    )

                gt_map = generate_gt_map(
                    gaussians,
                    gt_feature_map,
                    sampled_idx,
                    viewmat,
                    K,
                    visible_mask,
                )

                if query_valid_mask is not None:
                    gt_map_mask = _resize_hard_valid_mask(
                        query_valid_mask,
                        gt_map.shape[-2:],
                        gt_map.device,
                    )[None]
                    gt_map = gt_map * gt_map_mask

                # Loss
                heat_map = detector(gt_feature_map)
                loss = score_map_bce_loss(heat_map, gt_map)

                loss_sum += loss.item()
                if tb_writer and idx < 5:
                    render = render_gsplat(
                        viewpoint_cam, gaussians, background, rgb_only=True
                    )["render"]
                    sampled_render = render_gsplat(
                        viewpoint_cam, landmarks, background, rgb_only=True
                    )["render"]
                    heat_map = (heat_map - heat_map.min()) / (
                        heat_map.max() - heat_map.min()
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/gt_map_{idx}",
                        gt_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/heat_map{idx}",
                        heat_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/render_{idx}",
                        render[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/sampled_render_{idx}",
                        sampled_render[None],
                        iteration,
                    )

            loss_sum /= len(config["cameras"])
            print(
                f"\n[ITER {iteration}] Evaluating detector: {config['name']} loss {loss_sum}"
            )
            if tb_writer:
                tb_writer.add_scalar(
                f"detector_loss_patches/{config['name']}_loss",
                loss_sum,
                iteration,
            )


def _resolve_detector_artifact_path(scene_model_path, path):
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    candidate = os.path.join(scene_model_path, path)
    return candidate if os.path.exists(candidate) else path


def save_sparse_candidate_teacher_state(
    path,
    sampled_idx,
    landmark_features,
    iteration,
    config,
    landmark_xyz=None,
    diagnostics=None,
    dustbin_score=None,
    pair_scorer=None,
    pair_scorer_threshold=None,
    pair_measurement_head=None,
    pair_measurement_threshold=None,
    adaptive_trust_state=None,
    normalize_features=True,
):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exported_features = landmark_features.detach().reshape(
        landmark_features.shape[0], -1
    ).float()
    if bool(normalize_features):
        exported_features = F.normalize(exported_features, dim=1)
    state = {
        "version": 4,
        "iteration": int(iteration),
        "landmark_indices": torch.as_tensor(sampled_idx, dtype=torch.long).detach().cpu(),
        "landmark_features": exported_features.cpu(),
        "config": dict(config),
        "diagnostics": dict(diagnostics or {}),
    }
    if landmark_xyz is not None:
        exported_xyz = torch.as_tensor(landmark_xyz).detach().float().cpu()
        if exported_xyz.shape != (exported_features.shape[0], 3):
            raise ValueError(
                "candidate teacher landmark_xyz must have shape "
                f"({exported_features.shape[0]}, 3), got {tuple(exported_xyz.shape)}"
            )
        if not bool(torch.isfinite(exported_xyz).all()):
            raise ValueError("candidate teacher landmark_xyz contains non-finite values")
        state["landmark_xyz"] = exported_xyz
    if dustbin_score is not None:
        state["dustbin_score"] = float(torch.as_tensor(dustbin_score).detach().item())
    if pair_scorer is not None:
        state["pair_scorer_config"] = pair_scorer.export_config()
        state["pair_scorer_state_dict"] = {
            key: value.detach().cpu() for key, value in pair_scorer.state_dict().items()
        }
    if pair_scorer_threshold is not None:
        state["pair_scorer_threshold"] = float(pair_scorer_threshold)
    if pair_measurement_head is not None:
        state["pair_measurement_config"] = pair_measurement_head.export_config()
        state["pair_measurement_state_dict"] = {
            key: value.detach().cpu()
            for key, value in pair_measurement_head.state_dict().items()
        }
    if pair_measurement_threshold is not None:
        state["pair_measurement_threshold"] = float(pair_measurement_threshold)
    if adaptive_trust_state is not None:
        trust_state = dict(adaptive_trust_state)
        required_keys = (
            "initial_features",
            "raw_features",
            "visible_count",
            "correct_count",
            "update_steps",
        )
        missing_keys = [key for key in required_keys if key not in trust_state]
        if missing_keys:
            raise ValueError(
                "adaptive trust state is missing required keys: "
                + ", ".join(missing_keys)
            )
        expected_shape = state["landmark_features"].shape
        initial_features = torch.as_tensor(
            trust_state["initial_features"], dtype=torch.float32
        ).detach().cpu()
        raw_features = torch.as_tensor(
            trust_state["raw_features"], dtype=torch.float32
        ).detach().cpu()
        visible_count = torch.as_tensor(
            trust_state["visible_count"], dtype=torch.float32
        ).detach().cpu().reshape(-1)
        correct_count = torch.as_tensor(
            trust_state["correct_count"], dtype=torch.float32
        ).detach().cpu().reshape(-1)
        if initial_features.shape != expected_shape or raw_features.shape != expected_shape:
            raise ValueError(
                "adaptive trust feature shape does not match landmark features: "
                f"initial={tuple(initial_features.shape)} raw={tuple(raw_features.shape)} "
                f"expected={tuple(expected_shape)}"
            )
        if visible_count.numel() != expected_shape[0] or correct_count.numel() != expected_shape[0]:
            raise ValueError("adaptive trust evidence count does not match landmark count")
        if not bool(torch.isfinite(initial_features).all().item()) or not bool(torch.isfinite(raw_features).all().item()):
            raise ValueError("adaptive trust features contain non-finite values")
        if not bool(torch.isfinite(visible_count).all().item()) or not bool(torch.isfinite(correct_count).all().item()):
            raise ValueError("adaptive trust evidence contains non-finite values")
        if bool((visible_count < 0).any().item()) or bool((correct_count < 0).any().item()):
            raise ValueError("adaptive trust evidence counts must be non-negative")
        if bool((correct_count > visible_count).any().item()):
            raise ValueError("adaptive trust correct count cannot exceed visible count")
        update_steps = int(trust_state["update_steps"])
        if update_steps < 0:
            raise ValueError("adaptive trust update_steps must be non-negative")
        unique_evidence_keys = (
            "visible_view_mask",
            "correct_view_mask",
            "evidence_camera_names",
        )
        present_unique_evidence = [
            key in trust_state for key in unique_evidence_keys
        ]
        if any(present_unique_evidence) and not all(present_unique_evidence):
            raise ValueError(
                "adaptive trust unique-camera evidence must provide "
                + ", ".join(unique_evidence_keys)
            )
        serialized_trust_state = {
            "version": 1,
            "initial_features": F.normalize(initial_features, dim=1),
            "raw_features": raw_features,
            "visible_count": visible_count,
            "correct_count": correct_count,
            "update_steps": update_steps,
        }
        if all(present_unique_evidence):
            visible_view_mask = torch.as_tensor(
                trust_state["visible_view_mask"], dtype=torch.bool
            ).detach().cpu()
            correct_view_mask = torch.as_tensor(
                trust_state["correct_view_mask"], dtype=torch.bool
            ).detach().cpu()
            evidence_camera_names = tuple(
                str(name) for name in trust_state["evidence_camera_names"]
            )
            if (
                visible_view_mask.ndim != 2
                or correct_view_mask.shape != visible_view_mask.shape
                or visible_view_mask.shape[0] != expected_shape[0]
            ):
                raise ValueError(
                    "adaptive trust unique-camera masks must have matching "
                    "[landmark, camera] shapes"
                )
            if visible_view_mask.shape[1] != len(evidence_camera_names):
                raise ValueError(
                    "adaptive trust evidence_camera_names does not match "
                    "unique-camera mask width"
                )
            if len(set(evidence_camera_names)) != len(evidence_camera_names):
                raise ValueError("adaptive trust evidence camera names must be unique")
            visible_from_mask = visible_view_mask.sum(dim=1).to(torch.float32)
            correct_from_mask = correct_view_mask.sum(dim=1).to(torch.float32)
            if not torch.equal(visible_from_mask, visible_count):
                raise ValueError(
                    "adaptive trust visible_count must equal unique-camera "
                    "evidence mask counts"
                )
            if not torch.equal(correct_from_mask, correct_count):
                raise ValueError(
                    "adaptive trust correct_count must equal unique-camera "
                    "evidence mask counts"
                )
            serialized_trust_state.update(
                {
                    "version": 2,
                    "visible_view_mask": visible_view_mask,
                    "correct_view_mask": correct_view_mask,
                    "evidence_camera_names": evidence_camera_names,
                }
            )
        state["adaptive_trust_state"] = serialized_trust_state
    torch.save(state, path)
    return state


def load_sparse_candidate_teacher_features(path, sampled_idx, device="cuda"):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "landmark_features" not in state:
        raise ValueError(f"Invalid sparse candidate teacher state: {path}")
    expected = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    actual = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1).cpu()
    if not torch.equal(actual, expected):
        raise ValueError(
            "sparse candidate teacher landmark indices do not match sampled_idx: "
            f"state_count={actual.numel()} expected_count={expected.numel()}"
        )
    features = torch.as_tensor(state["landmark_features"], dtype=torch.float32)
    if features.ndim < 2 or features.shape[0] != expected.numel():
        raise ValueError(
            "sparse candidate teacher feature count does not match sampled_idx: "
            f"features={features.shape[0] if features.ndim else 0} expected={expected.numel()}"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("sparse candidate teacher features contain non-finite values")
    return features.to(device=device)


def load_sparse_candidate_teacher_adaptive_trust_state(
    state_or_path,
    sampled_idx,
    device="cuda",
):
    """Load resumable adaptive-trust state when a checkpoint provides it.

    Older candidate-teacher checkpoints intentionally return ``None`` so they
    remain valid initializations for a fresh adaptive-trust run.
    """
    if isinstance(state_or_path, (str, os.PathLike)):
        state = torch.load(state_or_path, map_location="cpu")
    else:
        state = state_or_path
    if not isinstance(state, dict) or "adaptive_trust_state" not in state:
        return None
    expected = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    actual = torch.as_tensor(
        state.get("landmark_indices"), dtype=torch.long
    ).reshape(-1).cpu()
    if not torch.equal(actual, expected):
        raise ValueError("adaptive trust state landmark indices do not match sampled_idx")
    trust_state = state["adaptive_trust_state"]
    if not isinstance(trust_state, dict):
        raise ValueError("adaptive trust state must be a dictionary")
    required_keys = (
        "initial_features",
        "raw_features",
        "visible_count",
        "correct_count",
        "update_steps",
    )
    missing_keys = [key for key in required_keys if key not in trust_state]
    if missing_keys:
        raise ValueError(
            "adaptive trust state is missing required keys: " + ", ".join(missing_keys)
        )
    initial_features = torch.as_tensor(
        trust_state["initial_features"], dtype=torch.float32
    )
    raw_features = torch.as_tensor(trust_state["raw_features"], dtype=torch.float32)
    visible_count = torch.as_tensor(
        trust_state["visible_count"], dtype=torch.float32
    ).reshape(-1)
    correct_count = torch.as_tensor(
        trust_state["correct_count"], dtype=torch.float32
    ).reshape(-1)
    if initial_features.ndim != 2 or initial_features.shape[0] != expected.numel():
        raise ValueError("adaptive trust initial feature count does not match sampled_idx")
    if raw_features.shape != initial_features.shape:
        raise ValueError("adaptive trust raw features do not match initial features")
    if visible_count.numel() != expected.numel() or correct_count.numel() != expected.numel():
        raise ValueError("adaptive trust evidence count does not match sampled_idx")
    tensors = (initial_features, raw_features, visible_count, correct_count)
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("adaptive trust state contains non-finite values")
    if bool((visible_count < 0).any().item()) or bool((correct_count < 0).any().item()):
        raise ValueError("adaptive trust evidence counts must be non-negative")
    if bool((correct_count > visible_count).any().item()):
        raise ValueError("adaptive trust correct count cannot exceed visible count")
    update_steps = int(trust_state["update_steps"])
    if update_steps < 0:
        raise ValueError("adaptive trust update_steps must be non-negative")
    resumed = {
        "initial_features": F.normalize(initial_features, dim=1).to(device=device),
        "raw_features": raw_features.to(device=device),
        "visible_count": visible_count.to(device=device),
        "correct_count": correct_count.to(device=device),
        "update_steps": update_steps,
    }
    unique_evidence_keys = (
        "visible_view_mask",
        "correct_view_mask",
        "evidence_camera_names",
    )
    present_unique_evidence = [key in trust_state for key in unique_evidence_keys]
    if any(present_unique_evidence) and not all(present_unique_evidence):
        raise ValueError(
            "adaptive trust unique-camera evidence must provide "
            + ", ".join(unique_evidence_keys)
        )
    if all(present_unique_evidence):
        visible_view_mask = torch.as_tensor(
            trust_state["visible_view_mask"], dtype=torch.bool
        )
        correct_view_mask = torch.as_tensor(
            trust_state["correct_view_mask"], dtype=torch.bool
        )
        evidence_camera_names = tuple(
            str(name) for name in trust_state["evidence_camera_names"]
        )
        if (
            visible_view_mask.ndim != 2
            or correct_view_mask.shape != visible_view_mask.shape
            or visible_view_mask.shape[0] != expected.numel()
        ):
            raise ValueError(
                "adaptive trust unique-camera masks must have matching "
                "[landmark, camera] shapes"
            )
        if visible_view_mask.shape[1] != len(evidence_camera_names):
            raise ValueError(
                "adaptive trust evidence_camera_names does not match "
                "unique-camera mask width"
            )
        if len(set(evidence_camera_names)) != len(evidence_camera_names):
            raise ValueError("adaptive trust evidence camera names must be unique")
        if not torch.equal(
            visible_view_mask.sum(dim=1).to(torch.float32), visible_count
        ):
            raise ValueError(
                "adaptive trust visible_count must equal unique-camera "
                "evidence mask counts"
            )
        if not torch.equal(
            correct_view_mask.sum(dim=1).to(torch.float32), correct_count
        ):
            raise ValueError(
                "adaptive trust correct_count must equal unique-camera "
                "evidence mask counts"
            )
        resumed.update(
            {
                "visible_view_mask": visible_view_mask.to(device=device),
                "correct_view_mask": correct_view_mask.to(device=device),
                "evidence_camera_names": evidence_camera_names,
            }
        )
    return resumed


def _numeric_teacher_diagnostics(diagnostics):
    result = {}
    for key, value in diagnostics.items():
        if isinstance(value, bool):
            result[key] = bool(value)
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().item())
    return result


@torch.no_grad()
def evaluate_sparse_candidate_teacher(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    landmark_features,
    landmark_xyz,
    dustbin_score,
    pair_scorer,
    pair_measurement_head,
    cameras,
    render_visible_masks,
    masks,
    scene,
    candidate_kwargs,
    assignment_mode,
    assignment_temperature,
    assignment_margin,
    reprojection_sigma_px=1.0,
    scorer_min_recall=0.75,
    scorer_max_matches_per_keypoint=1,
    set_risk_residual_clip_px=32.0,
    set_risk_reference_translation_m=0.01,
    map_fisher_translation_scale=0.02,
    map_fisher_rotation_scale_degrees=2.0,
    map_fisher_measurement_sigma_px=1.0,
    map_fisher_residual_clip_px=12.0,
    map_fisher_inlier_sigma_px=4.0,
    map_fisher_condition_target=100.0,
    map_bias_huber_delta=1.0,
    map_bias_clip=4.0,
    map_directional_temperature=0.05,
    map_directional_residual_clip_px=24.0,
    map_directional_robust_scale_px=12.0,
    map_directional_robust_quality_floor=0.01,
    query_feature_contract="legacy_full_then_resized_map",
):
    if not cameras:
        return {}
    was_training = detector.training
    detector.eval()
    measurement_was_training = (
        pair_measurement_head.training
        if pair_measurement_head is not None
        else False
    )
    if pair_measurement_head is not None:
        pair_measurement_head.eval()
    records = []
    scorer_logits = []
    scorer_labels = []
    scorer_valid = []
    measurement_logits = []
    measurement_labels = []
    measurement_valid = []
    measurement_reranked_correct_count = 0
    measurement_reranked_valid_count = 0
    reranked_correct_count = 0
    reranked_valid_count = 0
    for camera in cameras:
        fine_resolution = get_resolution_from_longest_edge(
            camera.original_image.shape[1],
            camera.original_image.shape[2],
            scene.longest_edge,
        )
        feature_map = extract_normalized_feature_map(
            feature_extractor,
            camera.original_image.cuda(),
            size=(fine_resolution[0], fine_resolution[1]),
            query_feature_contract=query_feature_contract,
            valid_mask=camera_valid_mask(
                masks, camera, camera.original_image.device
            ),
        )
        pose_w2c = camera.world_view_transform.transpose(0, 1).cuda()
        focal_x = fov2focal(camera.FoVx, feature_map.shape[2])
        focal_y = fov2focal(camera.FoVy, feature_map.shape[1])
        K = torch.tensor(
            [
                [focal_x, 0.0, feature_map.shape[2] / 2],
                [0.0, focal_y, feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=feature_map.device,
        )
        visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            camera.image_name,
            feature_map.device,
        )
        if visible_mask is None:
            with torch.enable_grad():
                visible_mask = get_render_visible_mask(
                    gaussians,
                    camera,
                    feature_map.shape[2],
                    feature_map.shape[1],
                )
            store_render_visible_mask(
                render_visible_masks,
                camera.image_name,
                visible_mask,
            )
        keypoint_heatmap, matchability_heatmap, offset_heatmap = detector.forward_all(
            feature_map
        )
        heatmap = rank_keypoint_proposals(
            keypoint_heatmap,
            matchability_heatmap,
            candidate_kwargs["nms_radius"],
        )
        valid_mask = camera_valid_mask(masks, camera, feature_map.device)
        if valid_mask is not None:
            valid_mask = _resize_hard_valid_mask(
                valid_mask,
                feature_map.shape[-2:],
                feature_map.device,
            )
            heatmap = heatmap * valid_mask
            matchability_heatmap = matchability_heatmap * valid_mask
        validation_candidate_kwargs = dict(candidate_kwargs)
        validation_candidate_kwargs["nms_radius"] = 0
        batch = build_sparse_candidate_batch(
            feature_map,
            heatmap,
            landmark_features,
            landmark_xyz,
            K,
            pose_w2c,
            visible_mask=visible_mask[sampled_idx],
            dustbin_score=dustbin_score,
            pair_scorer=pair_scorer,
            pair_measurement_head=pair_measurement_head,
            detector_supervision_heatmap=matchability_heatmap,
            keypoint_offset_map=offset_heatmap,
            **validation_candidate_kwargs,
        )
        losses = sparse_candidate_losses(
            batch,
            assignment_mode=assignment_mode,
            assignment_temperature=assignment_temperature,
            assignment_margin=assignment_margin,
            reprojection_sigma_px=reprojection_sigma_px,
            set_risk_residual_clip_px=set_risk_residual_clip_px,
            set_risk_reference_translation_m=set_risk_reference_translation_m,
            map_fisher_translation_scale=map_fisher_translation_scale,
            map_fisher_rotation_scale_degrees=map_fisher_rotation_scale_degrees,
            map_fisher_measurement_sigma_px=map_fisher_measurement_sigma_px,
            map_fisher_residual_clip_px=map_fisher_residual_clip_px,
            map_fisher_inlier_sigma_px=map_fisher_inlier_sigma_px,
            map_fisher_condition_target=map_fisher_condition_target,
            map_bias_huber_delta=map_bias_huber_delta,
            map_bias_clip=map_bias_clip,
            map_directional_temperature=map_directional_temperature,
            map_directional_residual_clip_px=(
                map_directional_residual_clip_px
            ),
            map_directional_robust_scale_px=map_directional_robust_scale_px,
            map_directional_robust_quality_floor=(
                map_directional_robust_quality_floor
            ),
        )
        record = _numeric_teacher_diagnostics(batch.diagnostics)
        record.update(
            {
                "loss_pair": float(losses.pair.item()),
                "loss_hard_negative": float(losses.hard_negative.item()),
                "loss_assignment": float(losses.assignment.item()),
                "loss_counterfactual_assignment": float(
                    losses.counterfactual_assignment.item()
                ),
                "loss_dustbin_assignment": float(losses.dustbin_assignment.item()),
                "loss_matcher_assignment": float(losses.matcher_assignment.item()),
                "loss_matcher_reprojection_assignment": float(
                    losses.matcher_reprojection_assignment.item()
                ),
                "loss_pair_scorer": float(losses.pair_scorer.item()),
                "loss_pair_scorer_assignment": float(
                    losses.pair_scorer_assignment.item()
                ),
                "loss_pair_measurement_inlier": float(
                    losses.pair_measurement_inlier.item()
                ),
                "loss_pair_measurement_nll": float(
                    losses.pair_measurement_nll.item()
                ),
                "loss_pair_measurement_translation_bias": float(
                    losses.pair_measurement_translation_bias.item()
                ),
                "loss_pair_measurement_translation_covariance": float(
                    losses.pair_measurement_translation_covariance.item()
                ),
                "loss_matcher_translation_info": float(
                    losses.matcher_translation_info.item()
                ),
                "loss_translation_info": float(losses.translation_info.item()),
                "loss_detector_match": float(losses.detector_match.item()),
                "loss_detector_offset": float(losses.detector_offset.item()),
                "loss_geometry_set": float(losses.geometry_set.item()),
                "loss_coverage": float(losses.coverage.item()),
                "loss_map_cleanliness": float(losses.map_cleanliness.item()),
                "loss_map_full_information": float(
                    losses.map_full_information.item()
                ),
                "loss_map_translation_information": float(
                    losses.map_translation_information.item()
                ),
                "loss_map_translation_trace": float(
                    losses.map_translation_trace.item()
                ),
                "loss_map_translation_condition": float(
                    losses.map_translation_condition.item()
                ),
                "loss_map_bias": float(losses.map_bias.item()),
                "loss_map_directional_bias": float(
                    losses.map_directional_bias.item()
                ),
                "loss_map_capacity": float(losses.map_capacity.item()),
            }
        )
        records.append(record)
        if batch.pair_scorer_logits.numel() > 0:
            selected = validation_hypothesis_indices_per_keypoint(
                batch.pair_scorer_keypoint_idx,
                batch.pair_scorer_logits,
                scorer_max_matches_per_keypoint,
            )
            selected_valid = batch.pair_scorer_valid_mask.detach().cpu()[selected]
            selected_correct = (
                (batch.pair_scorer_labels.detach().cpu()[selected] > 0.5)
                & selected_valid
            )
            reranked_correct_count += int(selected_correct.sum().item())
            reranked_valid_count += int(selected_valid.sum().item())
            scorer_logits.append(batch.pair_scorer_logits.detach().cpu()[selected])
            scorer_labels.append(batch.pair_scorer_labels.detach().cpu()[selected])
            scorer_valid.append(selected_valid)
        if batch.pair_measurement_inlier_logits.numel() > 0:
            selected = validation_hypothesis_indices_per_keypoint(
                batch.pair_scorer_keypoint_idx,
                batch.pair_measurement_inlier_logits,
                scorer_max_matches_per_keypoint,
            )
            selected_valid = batch.pair_scorer_valid_mask.detach().cpu()[selected]
            selected_correct = (
                (batch.pair_scorer_labels.detach().cpu()[selected] > 0.5)
                & selected_valid
            )
            measurement_reranked_correct_count += int(selected_correct.sum().item())
            measurement_reranked_valid_count += int(selected_valid.sum().item())
            measurement_logits.append(
                batch.pair_measurement_inlier_logits.detach().cpu()[selected]
            )
            measurement_labels.append(
                batch.pair_scorer_labels.detach().cpu()[selected]
            )
            measurement_valid.append(selected_valid)
    if was_training:
        detector.train()
    if pair_measurement_head is not None and measurement_was_training:
        pair_measurement_head.train()
    keys = sorted({key for record in records for key in record})
    result = {"camera_count": float(len(records))}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if values:
            tensor = torch.as_tensor(values, dtype=torch.float64)
            result[f"{key}_mean"] = float(tensor.mean().item())
            result[f"{key}_median"] = float(tensor.median().item())
    if scorer_logits:
        result["pair_scorer_reranked_correct_count_mean"] = float(
            reranked_correct_count / len(records)
        )
        result["pair_scorer_reranked_valid_count_mean"] = float(
            reranked_valid_count / len(records)
        )
        result["pair_scorer_reranked_gt_precision"] = float(
            reranked_correct_count / max(reranked_valid_count, 1)
        )
        calibrated = calibrate_binary_threshold(
            torch.cat(scorer_logits),
            torch.cat(scorer_labels),
            torch.cat(scorer_valid),
            min_recall=scorer_min_recall,
        )
        result.update(
            {
                "pair_scorer_calibrated_threshold": calibrated["threshold"],
                "pair_scorer_calibrated_precision": calibrated["precision"],
                "pair_scorer_calibrated_recall": calibrated["recall"],
                "pair_scorer_calibrated_accepted_count": float(
                    calibrated["accepted_count"]
                ),
                "pair_scorer_calibrated_correct_count": float(
                    calibrated["correct_count"]
                ),
            }
        )
    if measurement_logits:
        result["pair_measurement_reranked_correct_count_mean"] = float(
            measurement_reranked_correct_count / len(records)
        )
        result["pair_measurement_reranked_valid_count_mean"] = float(
            measurement_reranked_valid_count / len(records)
        )
        result["pair_measurement_reranked_gt_precision"] = float(
            measurement_reranked_correct_count
            / max(measurement_reranked_valid_count, 1)
        )
        calibrated = calibrate_binary_threshold(
            torch.cat(measurement_logits),
            torch.cat(measurement_labels),
            torch.cat(measurement_valid),
            min_recall=scorer_min_recall,
        )
        result.update(
            {
                "pair_measurement_calibrated_threshold": calibrated["threshold"],
                "pair_measurement_calibrated_precision": calibrated["precision"],
                "pair_measurement_calibrated_recall": calibrated["recall"],
                "pair_measurement_calibrated_accepted_count": float(
                    calibrated["accepted_count"]
                ),
                "pair_measurement_calibrated_correct_count": float(
                    calibrated["correct_count"]
                ),
            }
        )
    return result


def training_detector(
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    tb_writer,
    train_iteration=30000,
    detector_folder="",
    landmark_num=16384,
    landmark_k=32,
    sampling_mode="baseline",
    utility_weight=1.0,
    pnp_voxel_size=0.25,
    pnp_max_per_voxel=8,
    pnp_preserve_ratio=0.5,
    min_loc_observations=1,
    detector_target_mode="hard",
    soft_sigma=1.5,
    coverage_preserve_ratio=0.5,
    coverage_utility_ratio=0.25,
    coverage_high_confidence_ratio=0.0,
    coverage_grid_size=0,
    coverage_max_per_grid=0,
    coverage_depth_bins=0,
    coverage_max_per_depth_bin=0,
    coverage_allow_unbalanced_fallback=False,
    candidate_reprojection_error_scale=4.0,
    candidate_cleanliness_weight=1.0,
    candidate_pose_info_weight=1.0,
    candidate_balance_weight=1.0,
    candidate_reliability_weight=0.25,
    candidate_utility_weight=0.0,
    landmark_only=False,
    precomputed_landmark_path="",
    sparse_candidate_teacher=False,
    candidate_teacher_detector_init_path="",
    candidate_teacher_state_init_path="",
    candidate_teacher_pair_scorer_init_path="",
    candidate_teacher_pair_measurement_init_path="",
    candidate_teacher_optimize_features=False,
    candidate_teacher_freeze_detector=False,
    candidate_teacher_detector_only=False,
    candidate_teacher_detector_lr=1e-4,
    candidate_teacher_feature_lr=5e-5,
    candidate_teacher_dustbin_lr=0.0,
    candidate_teacher_pair_scorer_lr=1e-3,
    candidate_teacher_pair_measurement_lr=1e-3,
    candidate_teacher_pair_scorer_architecture="auto",
    candidate_teacher_detect_num=2048,
    candidate_teacher_nms_radius=2,
    candidate_teacher_query_feature_contract="legacy_full_then_resized_map",
    candidate_teacher_match_mode="topk",
    candidate_teacher_match_topk=1,
    candidate_teacher_match_threshold=0.0,
    candidate_teacher_dual_softmax=False,
    candidate_teacher_dual_softmax_temperature=0.1,
    candidate_teacher_positive_radius_px=2.0,
    candidate_teacher_negative_radius_px=6.0,
    candidate_teacher_max_positives=4,
    candidate_teacher_hard_negatives=8,
    candidate_teacher_match_temperature=0.1,
    candidate_teacher_match_margin=0.5,
    candidate_teacher_assignment_mode="single_nearest",
    candidate_teacher_assignment_temperature=0.05,
    candidate_teacher_assignment_margin=0.05,
    candidate_teacher_assignment_pose_information_mode="none",
    candidate_teacher_assignment_pose_information_weight=0.0,
    candidate_teacher_assignment_pose_information_floor=0.05,
    candidate_teacher_assignment_pose_information_normalization="quantile",
    candidate_teacher_assignment_fisher_translation_scale=0.02,
    candidate_teacher_assignment_fisher_rotation_scale_degrees=2.0,
    candidate_teacher_assignment_fisher_measurement_sigma=1.0,
    candidate_teacher_assignment_fisher_use_matchability=False,
    candidate_teacher_assignment_fisher_matchability_floor=0.05,
    candidate_teacher_assignment_fisher_matchability_power=1.0,
    candidate_teacher_assignment_fisher_uncertainty_entropy_scale=0.0,
    candidate_teacher_map_cleanliness_weight=0.0,
    candidate_teacher_map_full_information_weight=0.0,
    candidate_teacher_map_translation_information_weight=0.0,
    candidate_teacher_map_translation_trace_weight=0.0,
    candidate_teacher_map_translation_condition_weight=0.0,
    candidate_teacher_map_bias_weight=0.0,
    candidate_teacher_map_directional_bias_weight=0.0,
    candidate_teacher_map_capacity_weight=0.0,
    candidate_teacher_map_fisher_translation_scale=0.02,
    candidate_teacher_map_fisher_rotation_scale_degrees=2.0,
    candidate_teacher_map_fisher_measurement_sigma_px=1.0,
    candidate_teacher_map_fisher_residual_clip_px=12.0,
    candidate_teacher_map_fisher_inlier_sigma_px=4.0,
    candidate_teacher_map_fisher_condition_target=100.0,
    candidate_teacher_map_bias_huber_delta=1.0,
    candidate_teacher_map_bias_clip=4.0,
    candidate_teacher_map_max_matches_per_landmark=0,
    candidate_teacher_map_directional_topk=0,
    candidate_teacher_map_directional_temperature=0.05,
    candidate_teacher_map_directional_residual_clip_px=24.0,
    candidate_teacher_map_directional_robust_scale_px=12.0,
    candidate_teacher_map_directional_robust_quality_floor=0.01,
    candidate_teacher_counterfactual_bias_utility_weight=1.0,
    candidate_teacher_counterfactual_translation_utility_weight=0.0,
    candidate_teacher_counterfactual_utility_floor=0.1,
    candidate_teacher_counterfactual_target_mode="all_false",
    candidate_teacher_counterfactual_require_current_retained=False,
    candidate_teacher_counterfactual_exact_decision_set=False,
    candidate_teacher_counterfactual_require_positive_bias_gain=False,
    candidate_teacher_counterfactual_require_nonnegative_translation_gain=False,
    candidate_teacher_hard_preservation_weight=0.0,
    candidate_teacher_hard_preservation_refresh_visits=2,
    candidate_teacher_hard_preservation_solver="poselib",
    candidate_teacher_hard_preservation_reprojection_error=8.0,
    candidate_teacher_hard_preservation_ransac_seed=0,
    candidate_teacher_hard_preservation_min_inliers=4,
    candidate_teacher_hard_preservation_max_pose_error_cm=100.0,
    candidate_teacher_hard_preservation_max_useful=96,
    candidate_teacher_hard_preservation_max_harmful=96,
    candidate_teacher_hard_preservation_temperature=0.05,
    candidate_teacher_hard_preservation_margin=0.05,
    candidate_teacher_hard_preservation_harmful_mode="all_false",
    candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m=0.0,
    candidate_teacher_hard_preservation_exact_replay_max_candidates=8,
    candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm=0.0,
    candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree=0.0,
    candidate_teacher_grid_rows=4,
    candidate_teacher_grid_cols=4,
    candidate_teacher_depth_bins=4,
    candidate_teacher_pair_weight=1.0,
    candidate_teacher_hard_negative_weight=0.5,
    candidate_teacher_assignment_weight=1.0,
    candidate_teacher_counterfactual_assignment_weight=0.0,
    candidate_teacher_dustbin_weight=0.0,
    candidate_teacher_matcher_assignment_weight=0.0,
    candidate_teacher_matcher_reprojection_weight=0.0,
    candidate_teacher_reprojection_sigma_px=1.0,
    candidate_teacher_dustbin_init=0.5,
    candidate_teacher_pair_scorer_weight=0.0,
    candidate_teacher_pair_scorer_assignment_weight=0.0,
    candidate_teacher_pair_measurement_inlier_weight=0.0,
    candidate_teacher_pair_measurement_nll_weight=0.0,
    candidate_teacher_pair_measurement_bias_weight=0.0,
    candidate_teacher_pair_measurement_covariance_weight=0.0,
    candidate_teacher_pair_measurement_residual_clip_px=32.0,
    candidate_teacher_pair_measurement_reference_translation_m=0.01,
    candidate_teacher_matcher_translation_info_weight=0.0,
    candidate_teacher_translation_info_weight=0.0,
    candidate_teacher_pair_scorer_hidden_dim=16,
    candidate_teacher_pair_measurement_hidden_dim=64,
    candidate_teacher_pair_measurement_patch_radius=2,
    candidate_teacher_pair_measurement_max_offset=2.0,
    candidate_teacher_pair_measurement_covariance_floor=0.1,
    candidate_teacher_pair_measurement_set_context=False,
    candidate_teacher_pair_measurement_geometry_context=False,
    candidate_teacher_freeze_pair_measurement=False,
    candidate_teacher_pair_context_topk=8,
    candidate_teacher_scorer_min_recall=0.75,
    candidate_teacher_scorer_max_matches_per_keypoint=1,
    candidate_teacher_matchability_head=False,
    candidate_teacher_matchability_only=False,
    candidate_teacher_offset_head=False,
    candidate_teacher_offset_only=False,
    candidate_teacher_max_offset=2.0,
    candidate_teacher_offset_target_source="geometric_nearest",
    candidate_teacher_selection_source="combined",
    candidate_teacher_detector_target_source="geometric",
    candidate_teacher_detector_binary_target=False,
    candidate_teacher_detector_match_weight=1.0,
    candidate_teacher_detector_offset_weight=0.0,
    candidate_teacher_geometry_weight=0.1,
    candidate_teacher_coverage_weight=0.1,
    candidate_teacher_base_detector_weight=0.1,
    candidate_teacher_detector_preservation_weight=0.0,
    candidate_teacher_feature_anchor_weight=0.01,
    candidate_teacher_adaptive_trust=False,
    candidate_teacher_trust_alpha_min=0.25,
    candidate_teacher_trust_view_prior=3.0,
    candidate_teacher_trust_warmup_passes=1.0,
    candidate_teacher_support_query_split=False,
    candidate_teacher_query_ratio=0.2,
    candidate_teacher_validation_ratio=0.0,
    candidate_teacher_split_mode="temporal_block",
    candidate_teacher_split_seed=2026,
    candidate_teacher_online_render_ratio_start=0.0,
    candidate_teacher_online_render_ratio_end=0.0,
    candidate_teacher_online_render_ramp_start=0.0,
    candidate_teacher_online_render_ramp_end=1.0,
    candidate_teacher_online_render_alpha_min=0.35,
    candidate_teacher_online_render_alpha_max=0.65,
    candidate_teacher_online_render_provenance_mode="none",
    candidate_teacher_online_render_provenance_weight=0.0,
    candidate_teacher_online_render_provenance_topk=4,
    candidate_teacher_online_render_provenance_temperature=0.05,
    candidate_teacher_online_render_sampling_mode="uniform",
    candidate_teacher_online_render_failure_ema=0.9,
    candidate_teacher_online_render_failure_temperature=1.0,
    candidate_teacher_online_render_uniform_floor=0.1,
):
    if candidate_teacher_query_feature_contract not in {
        "legacy_full_then_resized_map",
        "native_resized_input",
    }:
        raise ValueError(
            "candidate_teacher_query_feature_contract must be "
            "'legacy_full_then_resized_map' or 'native_resized_input'"
        )
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    feature_extractor = FeatureExtractor(scene.feature_type).cuda().eval()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if scene.args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    render_visible_masks = {}

    save_path = os.path.join(scene.model_path, detector_folder)
    os.makedirs(save_path, exist_ok=True)
    landmark_meta = None
    if precomputed_landmark_path:
        precomputed_landmark_path = _resolve_detector_artifact_path(
            scene.model_path,
            precomputed_landmark_path,
        )
        print(f"Loading precomputed detector landmarks from {precomputed_landmark_path}")
        sampled_idx = validate_detector_sampled_indices(
            load_precomputed_detector_landmarks(
                precomputed_landmark_path,
                point_count=gaussians.get_xyz.shape[0],
                device=gaussians.get_xyz.device,
            ),
            sampling_mode="precomputed",
            min_loc_observations=min_loc_observations,
            point_count=gaussians.get_xyz.shape[0],
        )
        landmark_meta = load_precomputed_landmark_meta(precomputed_landmark_path)
        if landmark_meta is not None:
            save_landmark_meta(os.path.join(save_path, "landmark_meta.pt"), landmark_meta)
    else:
        # M.O. sampling
        print("Matching oriented sampling...")
        sample_result = matching_oriented_sample(
            scene,
            gaussians,
            feature_extractor,
            render_visible_masks,
            masks=masks,
            num=landmark_num,
            k=landmark_k,
            return_coverage_stats=sampling_mode == "coverage_preserving",
            query_feature_contract=candidate_teacher_query_feature_contract,
        )
        if sampling_mode == "coverage_preserving":
            sampled_idx, score_avg, score_num, coverage_stats = sample_result
        else:
            sampled_idx, score_avg, score_num = sample_result
            coverage_stats = None
        if sampling_mode in {
            "localization_aware",
            "localization_aware_spatial",
            "localization_aware_global",
            "localization_aware_pnp",
            "coverage_preserving",
        }:
            if not hasattr(gaussians, "compute_localization_utility"):
                raise ValueError("localization_aware sampling requires Gaussian localization state")
            observed = detector_sampling_observed_mask(
                gaussians.loc_observation_count,
                min_loc_observations=min_loc_observations,
                coverage_stats=coverage_stats if sampling_mode == "coverage_preserving" else None,
            )
            candidate_quality, candidate_components = final_candidate_quality_from_gaussians(
                gaussians,
                min_observations=min_loc_observations,
                coverage_stats=coverage_stats if sampling_mode == "coverage_preserving" else None,
                reprojection_error_scale=candidate_reprojection_error_scale,
                cleanliness_weight=candidate_cleanliness_weight,
                pose_info_weight=candidate_pose_info_weight,
                balance_weight=candidate_balance_weight,
                reliability_weight=candidate_reliability_weight,
                utility_weight=candidate_utility_weight,
            )
            utility = candidate_quality
            if sampling_mode == "coverage_preserving":
                high_confidence = candidate_components.get("candidate_cleanliness", utility) * candidate_components.get(
                    "pose_info_contribution", utility.new_ones(utility.shape)
                )
                coverage_uv = coverage_stats.get("uv") if coverage_stats is not None else None
                coverage_depth = coverage_stats.get("depth") if coverage_stats is not None else None
                image_size_tensor = coverage_stats.get("image_size") if coverage_stats is not None else None
                image_size = (
                    tuple(int(v) for v in image_size_tensor.detach().cpu().tolist())
                    if image_size_tensor is not None
                    else None
                )
                sampled_idx, landmark_meta = coverage_preserving_sample(
                    gaussian_localization_xyz(gaussians),
                    score_avg,
                    utility,
                    num=landmark_num,
                    k=landmark_k,
                    min_observations=observed,
                    utility_weight=utility_weight,
                    base_preserve_ratio=coverage_preserve_ratio,
                    utility_preserve_ratio=coverage_utility_ratio,
                    high_confidence=high_confidence,
                    high_confidence_ratio=coverage_high_confidence_ratio,
                    voxel_size=pnp_voxel_size,
                    max_per_voxel=pnp_max_per_voxel,
                    uv=coverage_uv,
                    image_size=image_size,
                    grid_size=coverage_grid_size,
                    max_per_grid=coverage_max_per_grid,
                    depth=coverage_depth,
                    depth_bins=coverage_depth_bins,
                    max_per_depth_bin=coverage_max_per_depth_bin,
                    allow_unbalanced_fallback=coverage_allow_unbalanced_fallback,
                )
            else:
                pnp_balance = sampling_mode == "localization_aware_pnp"
                use_spatial_sampling = sampling_mode != "localization_aware_global"
                sampled_idx, landmark_meta = localization_aware_sample(
                    gaussian_localization_xyz(gaussians),
                    score_avg,
                    utility,
                    num=landmark_num,
                    k=landmark_k,
                    min_observations=observed,
                    utility_weight=utility_weight,
                    spatial=use_spatial_sampling,
                    pnp_balance=pnp_balance,
                    pnp_voxel_size=pnp_voxel_size,
                    pnp_max_per_voxel=pnp_max_per_voxel,
                    pnp_preserve_ratio=pnp_preserve_ratio,
                )
            sampled_idx = validate_detector_sampled_indices(
                sampled_idx,
                sampling_mode=sampling_mode,
                min_loc_observations=min_loc_observations,
                point_count=gaussians.get_xyz.shape[0],
            )
            landmark_meta["repeatability"] = gaussians.loc_repeatability_ema[sampled_idx]
            landmark_meta["margin"] = gaussians.loc_margin_ema[sampled_idx]
            landmark_meta["information"] = gaussians.loc_information_ema[sampled_idx]
            landmark_meta["reproj_error"] = gaussians.loc_reproj_error_ema[sampled_idx]
            landmark_meta["prototype"] = gaussians.loc_prototype[sampled_idx]
            landmark_meta["legacy_utility"] = candidate_components["legacy_utility"][sampled_idx]
            for key, value in candidate_components.items():
                if key == "legacy_utility":
                    continue
                landmark_meta[key] = value[sampled_idx]
            landmark_meta["full_candidate_quality"] = candidate_quality.detach().clone()
            landmark_meta["landmark_indices"] = sampled_idx.detach().clone()
            save_landmark_meta(os.path.join(save_path, "landmark_meta.pt"), landmark_meta)
        elif sampling_mode != "baseline":
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        else:
            sampled_idx = validate_detector_sampled_indices(
                sampled_idx,
                sampling_mode=sampling_mode,
                min_loc_observations=min_loc_observations,
                point_count=gaussians.get_xyz.shape[0],
            )
    save_detector_sampled_indices(
        os.path.join(save_path, "sampled_idx.pkl"),
        sampled_idx,
    )
    if sparse_candidate_teacher:
        requested_landmarks = int(landmark_num)
        unique_landmarks = int(torch.unique(sampled_idx).numel())
        if sampled_idx.numel() != requested_landmarks or unique_landmarks != requested_landmarks:
            raise ValueError(
                "sparse candidate teacher requires an exact, duplicate-free landmark bank: "
                f"requested={requested_landmarks} actual={sampled_idx.numel()} unique={unique_landmarks}"
            )
    if landmark_only:
        print(
            "Detector landmark-only bootstrap complete: "
            f"path={save_path} landmarks={sampled_idx.numel()} sampling_mode={sampling_mode}"
        )
        return
    if "score_avg" in locals():
        del score_avg, score_num
    if "utility" in locals():
        del utility
    if "observed" in locals():
        del observed
    torch.cuda.empty_cache()

    # training scene-specific detector
    print("Training detector...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    training_cameras = scene.getTrainCameras().copy()
    input_cameras = list(training_cameras)
    validation_cameras = []
    support_camera_count = len(training_cameras)
    camera_partition = None
    if sparse_candidate_teacher:
        training_cameras, validation_cameras, support_camera_count = (
            partition_candidate_teacher_cameras(
                training_cameras,
                support_query_split=candidate_teacher_support_query_split,
                query_ratio=candidate_teacher_query_ratio,
                validation_ratio=candidate_teacher_validation_ratio,
                split_mode=candidate_teacher_split_mode,
                split_seed=candidate_teacher_split_seed,
            )
        )
        camera_partition = write_candidate_teacher_partition_manifest(
            save_path,
            input_cameras,
            training_cameras,
            validation_cameras,
            split_mode=candidate_teacher_split_mode,
            split_seed=candidate_teacher_split_seed,
            validation_ratio=candidate_teacher_validation_ratio,
        )
    if sparse_candidate_teacher and (
        candidate_teacher_support_query_split or validation_cameras
    ):
        print(
            "Sparse candidate teacher camera partition: "
            f"support={support_camera_count} candidate_train={len(training_cameras)} "
            f"candidate_val={len(validation_cameras)} "
            f"mode={candidate_teacher_split_mode}"
        )
    teacher_trust_camera_names = tuple(
        str(camera.image_name) for camera in training_cameras
    )
    if len(set(teacher_trust_camera_names)) != len(teacher_trust_camera_names):
        raise ValueError(
            "candidate-teacher training cameras must have unique image names "
            "for adaptive-trust evidence"
        )
    teacher_trust_camera_index = {
        name: index for index, name in enumerate(teacher_trust_camera_names)
    }

    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iteration), desc="Scene-Specific Detector")
    first_iter = 1

    detector = KpDetector(
        feature_extractor.feature_dim,
        matchability_head=candidate_teacher_matchability_head,
        offset_head=candidate_teacher_offset_head,
        max_offset=candidate_teacher_max_offset,
    ).cuda().train()
    detector_init_path = _resolve_detector_artifact_path(
        scene.model_path,
        candidate_teacher_detector_init_path,
    )
    if detector_init_path:
        print(f"Loading detector initialization from {detector_init_path}")
        detector_state = torch.load(detector_init_path, map_location="cuda")
        has_optional_head = bool(
            candidate_teacher_matchability_head or candidate_teacher_offset_head
        )
        incompatible = detector.load_state_dict(
            detector_state,
            strict=not has_optional_head,
        )
        if has_optional_head:
            allowed_missing = set()
            if candidate_teacher_matchability_head:
                allowed_missing.update(
                    {"matchability_head.weight", "matchability_head.bias"}
                )
            if candidate_teacher_offset_head:
                allowed_missing.update({"offset_head.weight", "offset_head.bias"})
            unexpected = set(incompatible.unexpected_keys)
            missing = set(incompatible.missing_keys)
            if unexpected or not missing.issubset(allowed_missing):
                raise ValueError(
                    "detector initialization is incompatible with optional heads: "
                    f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
                )
            if missing & {"matchability_head.weight", "matchability_head.bias"}:
                detector.initialize_matchability_from_keypoint()
            if missing & {"offset_head.weight", "offset_head.bias"}:
                detector.initialize_offset_to_zero()

    detector_preservation_teacher = None
    if (
        sparse_candidate_teacher
        and float(candidate_teacher_detector_preservation_weight) > 0.0
    ):
        detector_preservation_teacher = copy.deepcopy(detector).eval()
        detector_preservation_teacher.requires_grad_(False)

    teacher_landmark_features = None
    teacher_initial_features = None
    teacher_landmark_xyz = None
    teacher_trust_visible_count = None
    teacher_trust_correct_count = None
    teacher_trust_visible_view_mask = None
    teacher_trust_correct_view_mask = None
    teacher_trust_update_steps = 0
    teacher_history = []
    teacher_validation_history = []
    teacher_last_diagnostics = {}
    calibrated_pair_scorer_threshold = None
    calibrated_pair_measurement_threshold = None
    grad_accum = 8
    grad_clip_norm = 10.0
    if bool(candidate_teacher_adaptive_trust) and not bool(candidate_teacher_optimize_features):
        raise ValueError("adaptive candidate-teacher trust requires feature optimization")
    if not 0.0 <= float(candidate_teacher_trust_alpha_min) <= 1.0:
        raise ValueError("candidate_teacher_trust_alpha_min must be in [0, 1]")
    if float(candidate_teacher_trust_view_prior) < 0.0:
        raise ValueError("candidate_teacher_trust_view_prior must be non-negative")
    if float(candidate_teacher_detector_preservation_weight) < 0.0:
        raise ValueError(
            "candidate_teacher_detector_preservation_weight must be non-negative"
        )
    validate_detector_only_candidate_teacher_configuration(
        enabled=candidate_teacher_detector_only,
        sparse_candidate_teacher=sparse_candidate_teacher,
        optimize_features=candidate_teacher_optimize_features,
        freeze_detector=candidate_teacher_freeze_detector,
        dustbin_weight=candidate_teacher_dustbin_weight,
        pair_scorer_weight=candidate_teacher_pair_scorer_weight,
        pair_scorer_assignment_weight=candidate_teacher_pair_scorer_assignment_weight,
        pair_measurement_inlier_weight=candidate_teacher_pair_measurement_inlier_weight,
        pair_measurement_nll_weight=candidate_teacher_pair_measurement_nll_weight,
        pair_measurement_bias_weight=candidate_teacher_pair_measurement_bias_weight,
        pair_measurement_covariance_weight=(
            candidate_teacher_pair_measurement_covariance_weight
        ),
    )
    if float(candidate_teacher_trust_warmup_passes) < 0.0:
        raise ValueError("candidate_teacher_trust_warmup_passes must be non-negative")
    if float(candidate_teacher_hard_preservation_weight) < 0.0:
        raise ValueError("candidate_teacher_hard_preservation_weight must be non-negative")
    if int(candidate_teacher_hard_preservation_refresh_visits) < 1:
        raise ValueError("candidate_teacher_hard_preservation_refresh_visits must be >= 1")
    if str(candidate_teacher_hard_preservation_solver) not in {"poselib", "opencv"}:
        raise ValueError("candidate_teacher_hard_preservation_solver must be 'poselib' or 'opencv'")
    if float(candidate_teacher_hard_preservation_reprojection_error) <= 0.0:
        raise ValueError("candidate_teacher_hard_preservation_reprojection_error must be positive")
    if int(candidate_teacher_hard_preservation_ransac_seed) < 0:
        raise ValueError("candidate_teacher_hard_preservation_ransac_seed must be non-negative")
    if int(candidate_teacher_hard_preservation_min_inliers) < 4:
        raise ValueError("candidate_teacher_hard_preservation_min_inliers must be >= 4")
    if float(candidate_teacher_hard_preservation_max_pose_error_cm) <= 0.0:
        raise ValueError("candidate_teacher_hard_preservation_max_pose_error_cm must be positive")
    if int(candidate_teacher_hard_preservation_max_useful) < 1:
        raise ValueError("candidate_teacher_hard_preservation_max_useful must be >= 1")
    if int(candidate_teacher_hard_preservation_max_harmful) < 1:
        raise ValueError("candidate_teacher_hard_preservation_max_harmful must be >= 1")
    if float(candidate_teacher_hard_preservation_temperature) <= 0.0:
        raise ValueError("candidate_teacher_hard_preservation_temperature must be positive")
    if candidate_teacher_hard_preservation_harmful_mode not in {
        "all_false",
        "translation_delete",
        "exact_pose_delete",
    }:
        raise ValueError(
            "candidate_teacher_hard_preservation_harmful_mode must be "
            "'all_false', 'translation_delete', or 'exact_pose_delete'"
        )
    if (
        float(
            candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m
        )
        < 0.0
    ):
        raise ValueError(
            "candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m "
            "must be non-negative"
        )
    if int(candidate_teacher_hard_preservation_exact_replay_max_candidates) < 0:
        raise ValueError(
            "candidate_teacher_hard_preservation_exact_replay_max_candidates "
            "must be non-negative"
        )
    if float(candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm) < 0.0:
        raise ValueError(
            "candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm "
            "must be non-negative"
        )
    if (
        float(
            candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree
        )
        < 0.0
    ):
        raise ValueError(
            "candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree "
            "must be non-negative"
        )
    candidate_teacher_trust_warmup_steps = int(
        math.ceil(
            float(candidate_teacher_trust_warmup_passes)
            * max(len(training_cameras), 1)
        )
    )
    teacher_config = {
        "enabled": bool(sparse_candidate_teacher),
        "optimize_features": bool(candidate_teacher_optimize_features),
        "landmark_features_frozen": not bool(candidate_teacher_optimize_features),
        "freeze_detector": bool(candidate_teacher_freeze_detector),
        "detector_only": bool(candidate_teacher_detector_only),
        "effective_objective": (
            "detector_match_plus_offset"
            if bool(candidate_teacher_detector_only)
            else "configured_candidate_teacher_objective"
        ),
        "detector_init_path": detector_init_path,
        "state_init_path": candidate_teacher_state_init_path,
        "pair_scorer_init_path": candidate_teacher_pair_scorer_init_path,
        "pair_measurement_init_path": candidate_teacher_pair_measurement_init_path,
        "detector_lr": float(candidate_teacher_detector_lr),
        "feature_lr": float(candidate_teacher_feature_lr),
        "dustbin_lr": float(
            candidate_teacher_dustbin_lr
            if float(candidate_teacher_dustbin_lr) > 0.0
            else candidate_teacher_feature_lr
        ),
        "optimizer": "AdamW",
        "optimizer_weight_decay": 1e-4,
        "gradient_accumulation": int(grad_accum),
        "gradient_clip_norm": float(grad_clip_norm),
        "online_render_ratio_start": float(
            candidate_teacher_online_render_ratio_start
        ),
        "online_render_ratio_end": float(candidate_teacher_online_render_ratio_end),
        "online_render_ramp_start": float(
            candidate_teacher_online_render_ramp_start
        ),
        "online_render_ramp_end": float(candidate_teacher_online_render_ramp_end),
        "online_render_alpha_min": float(candidate_teacher_online_render_alpha_min),
        "online_render_alpha_max": float(candidate_teacher_online_render_alpha_max),
        "online_render_provenance_mode": str(
            candidate_teacher_online_render_provenance_mode
        ),
        "online_render_provenance_weight": float(
            candidate_teacher_online_render_provenance_weight
        ),
        "online_render_provenance_topk": int(
            candidate_teacher_online_render_provenance_topk
        ),
        "online_render_provenance_temperature": float(
            candidate_teacher_online_render_provenance_temperature
        ),
        "online_render_sampling_mode": str(candidate_teacher_online_render_sampling_mode),
        "online_render_failure_ema": float(candidate_teacher_online_render_failure_ema),
        "online_render_failure_temperature": float(
            candidate_teacher_online_render_failure_temperature
        ),
        "online_render_uniform_floor": float(candidate_teacher_online_render_uniform_floor),
        "landmark_num": int(landmark_num),
        "sampling_mode": str(sampling_mode),
        "precomputed_landmark_path": str(precomputed_landmark_path),
        "detector_target_mode": str(detector_target_mode),
        "soft_sigma": float(soft_sigma),
        "detect_num": int(candidate_teacher_detect_num),
        "nms_radius": int(candidate_teacher_nms_radius),
        "query_feature_contract": str(candidate_teacher_query_feature_contract),
        "match_mode": str(candidate_teacher_match_mode),
        "match_topk": int(candidate_teacher_match_topk),
        "match_threshold": float(candidate_teacher_match_threshold),
        "dual_softmax": bool(candidate_teacher_dual_softmax),
        "dual_softmax_temperature": float(candidate_teacher_dual_softmax_temperature),
        "positive_radius_px": float(candidate_teacher_positive_radius_px),
        "negative_radius_px": float(candidate_teacher_negative_radius_px),
        "max_positives": int(candidate_teacher_max_positives),
        "hard_negatives": int(candidate_teacher_hard_negatives),
        "match_temperature": float(candidate_teacher_match_temperature),
        "match_margin": float(candidate_teacher_match_margin),
        "assignment_mode": str(candidate_teacher_assignment_mode),
        "assignment_temperature": float(candidate_teacher_assignment_temperature),
        "assignment_margin": float(candidate_teacher_assignment_margin),
        "assignment_pose_information_mode": str(
            candidate_teacher_assignment_pose_information_mode
        ),
        "assignment_pose_information_weight": float(
            candidate_teacher_assignment_pose_information_weight
        ),
        "assignment_pose_information_floor": float(
            candidate_teacher_assignment_pose_information_floor
        ),
        "assignment_pose_information_normalization": str(
            candidate_teacher_assignment_pose_information_normalization
        ),
        "assignment_fisher_translation_scale": float(
            candidate_teacher_assignment_fisher_translation_scale
        ),
        "assignment_fisher_rotation_scale_degrees": float(
            candidate_teacher_assignment_fisher_rotation_scale_degrees
        ),
        "assignment_fisher_measurement_sigma": float(
            candidate_teacher_assignment_fisher_measurement_sigma
        ),
        "assignment_fisher_use_matchability": bool(
            candidate_teacher_assignment_fisher_use_matchability
        ),
        "assignment_fisher_matchability_floor": float(
            candidate_teacher_assignment_fisher_matchability_floor
        ),
        "assignment_fisher_matchability_power": float(
            candidate_teacher_assignment_fisher_matchability_power
        ),
        "assignment_fisher_uncertainty_entropy_scale": float(
            candidate_teacher_assignment_fisher_uncertainty_entropy_scale
        ),
        "map_cleanliness_weight": float(candidate_teacher_map_cleanliness_weight),
        "map_full_information_weight": float(
            candidate_teacher_map_full_information_weight
        ),
        "map_translation_information_weight": float(
            candidate_teacher_map_translation_information_weight
        ),
        "map_translation_trace_weight": float(
            candidate_teacher_map_translation_trace_weight
        ),
        "map_translation_condition_weight": float(
            candidate_teacher_map_translation_condition_weight
        ),
        "map_bias_weight": float(candidate_teacher_map_bias_weight),
        "map_directional_bias_weight": float(
            candidate_teacher_map_directional_bias_weight
        ),
        "map_capacity_weight": float(candidate_teacher_map_capacity_weight),
        "map_fisher_translation_scale": float(
            candidate_teacher_map_fisher_translation_scale
        ),
        "map_fisher_rotation_scale_degrees": float(
            candidate_teacher_map_fisher_rotation_scale_degrees
        ),
        "map_fisher_measurement_sigma_px": float(
            candidate_teacher_map_fisher_measurement_sigma_px
        ),
        "map_fisher_residual_clip_px": float(
            candidate_teacher_map_fisher_residual_clip_px
        ),
        "map_fisher_inlier_sigma_px": float(
            candidate_teacher_map_fisher_inlier_sigma_px
        ),
        "map_fisher_condition_target": float(
            candidate_teacher_map_fisher_condition_target
        ),
        "map_bias_huber_delta": float(candidate_teacher_map_bias_huber_delta),
        "map_bias_clip": float(candidate_teacher_map_bias_clip),
        "map_max_matches_per_landmark": int(
            candidate_teacher_map_max_matches_per_landmark
        ),
        "map_directional_topk": int(candidate_teacher_map_directional_topk),
        "map_directional_temperature": float(
            candidate_teacher_map_directional_temperature
        ),
        "map_directional_residual_clip_px": float(
            candidate_teacher_map_directional_residual_clip_px
        ),
        "map_directional_robust_scale_px": float(
            candidate_teacher_map_directional_robust_scale_px
        ),
        "map_directional_robust_quality_floor": float(
            candidate_teacher_map_directional_robust_quality_floor
        ),
        "counterfactual_bias_utility_weight": float(
            candidate_teacher_counterfactual_bias_utility_weight
        ),
        "counterfactual_enabled": bool(
            float(candidate_teacher_counterfactual_assignment_weight) > 0.0
        ),
        "counterfactual_translation_utility_weight": float(
            candidate_teacher_counterfactual_translation_utility_weight
        ),
        "counterfactual_utility_floor": float(
            candidate_teacher_counterfactual_utility_floor
        ),
        "counterfactual_target_mode": str(
            candidate_teacher_counterfactual_target_mode
        ),
        "counterfactual_require_current_retained": bool(
            candidate_teacher_counterfactual_require_current_retained
        ),
        "counterfactual_exact_decision_set": bool(
            candidate_teacher_counterfactual_exact_decision_set
        ),
        "counterfactual_require_positive_bias_gain": bool(
            candidate_teacher_counterfactual_require_positive_bias_gain
        ),
        "counterfactual_require_nonnegative_translation_gain": bool(
            candidate_teacher_counterfactual_require_nonnegative_translation_gain
        ),
        "hard_preservation_enabled": bool(
            float(candidate_teacher_hard_preservation_weight) > 0.0
        ),
        "hard_preservation_weight": float(candidate_teacher_hard_preservation_weight),
        "hard_preservation_refresh_visits": int(
            candidate_teacher_hard_preservation_refresh_visits
        ),
        "hard_preservation_solver": str(candidate_teacher_hard_preservation_solver),
        "hard_preservation_reprojection_error": float(
            candidate_teacher_hard_preservation_reprojection_error
        ),
        "hard_preservation_ransac_seed": int(
            candidate_teacher_hard_preservation_ransac_seed
        ),
        "hard_preservation_confidence": 0.99999,
        "hard_preservation_max_iterations": 100000,
        "hard_preservation_min_iterations": 1000,
        "hard_preservation_min_inliers": int(
            candidate_teacher_hard_preservation_min_inliers
        ),
        "hard_preservation_max_pose_error_cm": float(
            candidate_teacher_hard_preservation_max_pose_error_cm
        ),
        "hard_preservation_max_useful": int(
            candidate_teacher_hard_preservation_max_useful
        ),
        "hard_preservation_max_harmful": int(
            candidate_teacher_hard_preservation_max_harmful
        ),
        "hard_preservation_temperature": float(
            candidate_teacher_hard_preservation_temperature
        ),
        "hard_preservation_margin": float(candidate_teacher_hard_preservation_margin),
        "hard_preservation_harmful_mode": str(
            candidate_teacher_hard_preservation_harmful_mode
        ),
        "hard_preservation_harmful_min_translation_delete_gain_m": float(
            candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m
        ),
        "hard_preservation_exact_replay_max_candidates": int(
            candidate_teacher_hard_preservation_exact_replay_max_candidates
        ),
        "hard_preservation_exact_replay_min_pose_gain_cm": float(
            candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm
        ),
        "hard_preservation_exact_replay_rotation_weight_cm_per_degree": float(
            candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree
        ),
        "hard_preservation_exact_replay_requires_harmful": bool(
            candidate_teacher_hard_preservation_harmful_mode == "exact_pose_delete"
        ),
        "hard_preservation_score_target": float(candidate_teacher_match_margin),
        "hard_preservation_deployment_graph": "match_score_threshold_plus_landmark_quota",
        "grid_rows": int(candidate_teacher_grid_rows),
        "grid_cols": int(candidate_teacher_grid_cols),
        "depth_bins": int(candidate_teacher_depth_bins),
        "pair_weight": float(candidate_teacher_pair_weight),
        "hard_negative_weight": float(candidate_teacher_hard_negative_weight),
        "assignment_weight": float(candidate_teacher_assignment_weight),
        "counterfactual_assignment_weight": float(
            candidate_teacher_counterfactual_assignment_weight
        ),
        "dustbin_weight": float(candidate_teacher_dustbin_weight),
        "matcher_assignment_weight": float(
            candidate_teacher_matcher_assignment_weight
        ),
        "matcher_reprojection_weight": float(
            candidate_teacher_matcher_reprojection_weight
        ),
        "reprojection_sigma_px": float(candidate_teacher_reprojection_sigma_px),
        "dustbin_init": float(candidate_teacher_dustbin_init),
        "pair_scorer_weight": float(candidate_teacher_pair_scorer_weight),
        "pair_scorer_assignment_weight": float(
            candidate_teacher_pair_scorer_assignment_weight
        ),
        "pair_measurement_inlier_weight": float(
            candidate_teacher_pair_measurement_inlier_weight
        ),
        "pair_measurement_nll_weight": float(
            candidate_teacher_pair_measurement_nll_weight
        ),
        "pair_measurement_bias_weight": float(
            candidate_teacher_pair_measurement_bias_weight
        ),
        "pair_measurement_covariance_weight": float(
            candidate_teacher_pair_measurement_covariance_weight
        ),
        "pair_measurement_residual_clip_px": float(
            candidate_teacher_pair_measurement_residual_clip_px
        ),
        "pair_measurement_reference_translation_m": float(
            candidate_teacher_pair_measurement_reference_translation_m
        ),
        "matcher_translation_info_weight": float(
            candidate_teacher_matcher_translation_info_weight
        ),
        "translation_info_weight": float(candidate_teacher_translation_info_weight),
        "pair_scorer_lr": float(candidate_teacher_pair_scorer_lr),
        "pair_measurement_lr": float(candidate_teacher_pair_measurement_lr),
        "pair_scorer_architecture": str(candidate_teacher_pair_scorer_architecture),
        "pair_scorer_hidden_dim": int(candidate_teacher_pair_scorer_hidden_dim),
        "pair_measurement_hidden_dim": int(
            candidate_teacher_pair_measurement_hidden_dim
        ),
        "pair_measurement_patch_radius": int(
            candidate_teacher_pair_measurement_patch_radius
        ),
        "pair_measurement_max_offset": float(
            candidate_teacher_pair_measurement_max_offset
        ),
        "pair_measurement_covariance_floor": float(
            candidate_teacher_pair_measurement_covariance_floor
        ),
        "pair_measurement_set_context": bool(
            candidate_teacher_pair_measurement_set_context
        ),
        "pair_measurement_geometry_context": bool(
            candidate_teacher_pair_measurement_geometry_context
        ),
        "freeze_pair_measurement": bool(
            candidate_teacher_freeze_pair_measurement
        ),
        "pair_context_topk": int(candidate_teacher_pair_context_topk),
        "scorer_min_recall": float(candidate_teacher_scorer_min_recall),
        "scorer_max_matches_per_keypoint": int(
            candidate_teacher_scorer_max_matches_per_keypoint
        ),
        "matchability_head": bool(candidate_teacher_matchability_head),
        "matchability_only": bool(candidate_teacher_matchability_only),
        "offset_head": bool(candidate_teacher_offset_head),
        "offset_only": bool(candidate_teacher_offset_only),
        "max_offset": float(candidate_teacher_max_offset),
        "offset_target_source": str(candidate_teacher_offset_target_source),
        "selection_source": str(candidate_teacher_selection_source),
        "detector_target_source": str(candidate_teacher_detector_target_source),
        "detector_binary_target": bool(candidate_teacher_detector_binary_target),
        "detector_match_weight": float(candidate_teacher_detector_match_weight),
        "detector_offset_weight": float(candidate_teacher_detector_offset_weight),
        "geometry_weight": float(candidate_teacher_geometry_weight),
        "coverage_weight": float(candidate_teacher_coverage_weight),
        "base_detector_weight": float(candidate_teacher_base_detector_weight),
        "detector_preservation_weight": float(
            candidate_teacher_detector_preservation_weight
        ),
        "detector_preservation_target": "frozen_combined_proposal_heatmap",
        "feature_anchor_weight": float(candidate_teacher_feature_anchor_weight),
        "feature_parameterization": (
            "adaptive_trust_residual"
            if bool(candidate_teacher_adaptive_trust)
            else "direct"
        ),
        "adaptive_trust": bool(candidate_teacher_adaptive_trust),
        "trust_alpha_min": float(candidate_teacher_trust_alpha_min),
        "trust_view_prior": float(candidate_teacher_trust_view_prior),
        "trust_warmup_passes": float(candidate_teacher_trust_warmup_passes),
        "trust_warmup_steps": int(candidate_teacher_trust_warmup_steps),
        "trust_warmup_schedule": "linear_blend_to_evidence",
        "trust_evidence": "unique_real_camera_deployment_predicted_gt_correct_after_quota",
        "trust_resume_state_version": 2,
        "support_query_split": bool(candidate_teacher_support_query_split),
        "support_camera_count": int(support_camera_count),
        "query_camera_count": int(len(training_cameras)),
        "validation_camera_count": int(len(validation_cameras)),
        "camera_order": (
            str(camera_partition["camera_order"])
            if camera_partition is not None
            else "scene_order"
        ),
        "input_camera_names_sha256": (
            str(camera_partition["support_camera_names_sha256"])
            if camera_partition is not None
            else candidate_teacher_camera_names_sha256(input_cameras)
        ),
        "query_camera_names_sha256": candidate_teacher_camera_names_sha256(
            training_cameras
        ),
        "validation_camera_names_sha256": candidate_teacher_camera_names_sha256(
            validation_cameras
        ),
        "query_ratio": float(candidate_teacher_query_ratio),
        "validation_ratio": float(candidate_teacher_validation_ratio),
        "split_mode": str(candidate_teacher_split_mode),
        "split_seed": int(candidate_teacher_split_seed),
    }

    if sparse_candidate_teacher:
        teacher_initial_features = gaussians.materialized_loc_feature(sampled_idx).reshape(
            sampled_idx.numel(), -1
        ).detach().float().clone()
        state_init_path = _resolve_detector_artifact_path(
            scene.model_path,
            candidate_teacher_state_init_path,
        )
        teacher_init_state = None
        if state_init_path:
            print(f"Loading sparse candidate teacher feature initialization from {state_init_path}")
            teacher_init_state = torch.load(state_init_path, map_location="cpu")
            teacher_initial_features = load_sparse_candidate_teacher_features(
                state_init_path,
                sampled_idx,
                device=teacher_initial_features.device,
            )
        if bool(candidate_teacher_optimize_features):
            teacher_initial_features = F.normalize(teacher_initial_features, dim=1)
        teacher_raw_features = teacher_initial_features
        if bool(candidate_teacher_adaptive_trust):
            resumed_trust_state = load_sparse_candidate_teacher_adaptive_trust_state(
                teacher_init_state,
                sampled_idx,
                device=teacher_initial_features.device,
            )
            if resumed_trust_state is not None:
                if (
                    "visible_view_mask" not in resumed_trust_state
                    or "correct_view_mask" not in resumed_trust_state
                    or "evidence_camera_names" not in resumed_trust_state
                ):
                    raise ValueError(
                        "Cannot resume legacy visit-count adaptive trust. Start a "
                        "fresh trust run from a non-trust candidate-teacher state."
                    )
                if tuple(resumed_trust_state["evidence_camera_names"]) != teacher_trust_camera_names:
                    raise ValueError(
                        "adaptive-trust resume cameras do not match the current "
                        "candidate-training split"
                    )
                print(
                    "Resuming sparse candidate adaptive-trust evidence: "
                    f"updates={resumed_trust_state['update_steps']}"
                )
                teacher_initial_features = resumed_trust_state["initial_features"]
                teacher_raw_features = resumed_trust_state["raw_features"]
                teacher_trust_visible_count = resumed_trust_state["visible_count"]
                teacher_trust_correct_count = resumed_trust_state["correct_count"]
                teacher_trust_visible_view_mask = resumed_trust_state[
                    "visible_view_mask"
                ]
                teacher_trust_correct_view_mask = resumed_trust_state[
                    "correct_view_mask"
                ]
                teacher_trust_update_steps = int(
                    resumed_trust_state["update_steps"]
                )
            else:
                teacher_trust_visible_count = torch.zeros(
                    sampled_idx.numel(),
                    dtype=torch.float32,
                    device=teacher_initial_features.device,
                )
                teacher_trust_correct_count = torch.zeros_like(
                    teacher_trust_visible_count
                )
                evidence_shape = (
                    sampled_idx.numel(),
                    len(teacher_trust_camera_names),
                )
                teacher_trust_visible_view_mask = torch.zeros(
                    evidence_shape,
                    dtype=torch.bool,
                    device=teacher_initial_features.device,
                )
                teacher_trust_correct_view_mask = torch.zeros_like(
                    teacher_trust_visible_view_mask
                )
        teacher_landmark_features = torch.nn.Parameter(
            teacher_raw_features.clone(),
            requires_grad=bool(candidate_teacher_optimize_features),
        )
        teacher_landmark_xyz = gaussian_localization_xyz(gaussians)[sampled_idx].detach().float()
        if isinstance(teacher_init_state, dict) and "landmark_xyz" in teacher_init_state:
            state_indices = torch.as_tensor(
                teacher_init_state.get("landmark_indices", []), dtype=torch.long
            ).reshape(-1)
            if not torch.equal(state_indices.cpu(), sampled_idx.detach().cpu()):
                raise ValueError(
                    "candidate teacher landmark_xyz is not aligned with detector landmarks"
                )
            state_xyz = torch.as_tensor(
                teacher_init_state["landmark_xyz"],
                device=teacher_landmark_xyz.device,
                dtype=teacher_landmark_xyz.dtype,
            ).reshape(-1, 3)
            if state_xyz.shape != teacher_landmark_xyz.shape or not bool(
                torch.isfinite(state_xyz).all().item()
            ):
                raise ValueError("candidate teacher landmark_xyz is invalid")
            teacher_landmark_xyz = state_xyz
        initial_dustbin_score = float(candidate_teacher_dustbin_init)
        if isinstance(teacher_init_state, dict) and "dustbin_score" in teacher_init_state:
            initial_dustbin_score = float(teacher_init_state["dustbin_score"])
        teacher_dustbin_score = torch.nn.Parameter(
            teacher_initial_features.new_tensor(initial_dustbin_score),
            requires_grad=float(candidate_teacher_dustbin_weight) > 0.0,
        )
        scorer_init_state = teacher_init_state
        scorer_init_path = _resolve_detector_artifact_path(
            scene.model_path,
            candidate_teacher_pair_scorer_init_path,
        )
        if scorer_init_path:
            print(f"Loading pair scorer initialization from {scorer_init_path}")
            scorer_init_state = torch.load(scorer_init_path, map_location="cpu")
        scorer_state = (
            scorer_init_state.get("pair_scorer_state_dict")
            if isinstance(scorer_init_state, dict)
            else None
        )
        scorer_config = (
            scorer_init_state.get("pair_scorer_config", {})
            if isinstance(scorer_init_state, dict)
            else {}
        )
        optimize_pair_scorer = (
            float(candidate_teacher_pair_scorer_weight) > 0.0
            or float(candidate_teacher_pair_scorer_assignment_weight) > 0.0
        )
        if optimize_pair_scorer or scorer_state is not None:
            source_architecture = scorer_config.get(
                "architecture", "cosine_residual_v1"
            )
            scorer_architecture = str(candidate_teacher_pair_scorer_architecture)
            if scorer_architecture == "auto":
                scorer_architecture = source_architecture
            descriptor_dim = (
                int(teacher_initial_features.shape[1])
                if scorer_architecture == "descriptor_set_residual_v2"
                else 0
            )
            teacher_pair_scorer = SparsePairScorer(
                input_dim=int(scorer_config.get("input_dim", 6)),
                hidden_dim=int(
                    scorer_config.get(
                        "hidden_dim",
                        candidate_teacher_pair_scorer_hidden_dim,
                    )
                ),
                cosine_bias=float(candidate_teacher_dustbin_init),
                architecture=scorer_architecture,
                descriptor_dim=descriptor_dim,
            ).to(device=teacher_initial_features.device)
            if scorer_state is not None:
                upgrading_to_descriptor = (
                    source_architecture == "cosine_residual_v1"
                    and scorer_architecture == "descriptor_set_residual_v2"
                )
                incompatible = teacher_pair_scorer.load_state_dict(
                    scorer_state,
                    strict=not upgrading_to_descriptor,
                )
                if upgrading_to_descriptor:
                    allowed_missing = {
                        "descriptor_network.0.weight",
                        "descriptor_network.0.bias",
                        "descriptor_network.2.weight",
                        "descriptor_network.2.bias",
                    }
                    if (
                        set(incompatible.missing_keys) != allowed_missing
                        or incompatible.unexpected_keys
                    ):
                        raise ValueError(
                            "incompatible v1-to-v2 pair scorer upgrade: "
                            f"missing={incompatible.missing_keys} "
                            f"unexpected={incompatible.unexpected_keys}"
                        )
            teacher_pair_scorer.requires_grad_(
                optimize_pair_scorer
            )
        else:
            teacher_pair_scorer = None

        measurement_init_state = teacher_init_state
        measurement_init_path = _resolve_detector_artifact_path(
            scene.model_path,
            candidate_teacher_pair_measurement_init_path,
        )
        if measurement_init_path:
            print(
                "Loading pair measurement initialization from "
                f"{measurement_init_path}"
            )
            measurement_init_state = torch.load(
                measurement_init_path, map_location="cpu"
            )
        measurement_state = (
            measurement_init_state.get("pair_measurement_state_dict")
            if isinstance(measurement_init_state, dict)
            else None
        )
        measurement_config = (
            measurement_init_state.get("pair_measurement_config", {})
            if isinstance(measurement_init_state, dict)
            else {}
        )
        teacher_pair_measurement_accept_threshold = float(
            measurement_init_state.get("pair_measurement_threshold", 0.0)
            if isinstance(measurement_init_state, dict)
            else 0.0
        )
        teacher_config["pair_measurement_accept_threshold"] = (
            teacher_pair_measurement_accept_threshold
        )
        optimize_pair_measurement = (
            float(candidate_teacher_pair_measurement_inlier_weight) > 0.0
            or float(candidate_teacher_pair_measurement_nll_weight) > 0.0
            or float(candidate_teacher_pair_measurement_bias_weight) > 0.0
            or float(candidate_teacher_pair_measurement_covariance_weight) > 0.0
        )
        if optimize_pair_measurement or measurement_state is not None:
            descriptor_dim = int(teacher_initial_features.shape[1])
            configured_descriptor_dim = int(
                measurement_config.get("descriptor_dim", descriptor_dim)
            )
            if configured_descriptor_dim != descriptor_dim:
                raise ValueError(
                    "pair measurement descriptor dimension does not match map: "
                    f"state={configured_descriptor_dim} map={descriptor_dim}"
                )
            teacher_pair_measurement_head = PairMeasurementHead(
                descriptor_dim=descriptor_dim,
                pair_feature_dim=int(
                    measurement_config.get("pair_feature_dim", 6)
                ),
                patch_radius=int(
                    measurement_config.get(
                        "patch_radius",
                        candidate_teacher_pair_measurement_patch_radius,
                    )
                ),
                hidden_dim=int(
                    measurement_config.get(
                        "hidden_dim",
                        candidate_teacher_pair_measurement_hidden_dim,
                    )
                ),
                max_offset=float(
                    measurement_config.get(
                        "max_offset",
                        candidate_teacher_pair_measurement_max_offset,
                    )
                ),
                covariance_floor=float(
                    measurement_config.get(
                        "covariance_floor",
                        candidate_teacher_pair_measurement_covariance_floor,
                    )
                ),
                cosine_bias=float(candidate_teacher_dustbin_init),
                use_set_context=bool(
                    measurement_config.get("use_set_context", False)
                    or candidate_teacher_pair_measurement_set_context
                    or candidate_teacher_pair_measurement_geometry_context
                ),
                use_geometry_context=bool(
                    measurement_config.get("use_geometry_context", False)
                    or candidate_teacher_pair_measurement_geometry_context
                ),
            ).to(device=teacher_initial_features.device)
            if measurement_state is not None:
                upgrading_set_context = (
                    teacher_pair_measurement_head.use_set_context
                    and not bool(measurement_config.get("use_set_context", False))
                )
                upgrading_geometry_context = (
                    teacher_pair_measurement_head.use_geometry_context
                    and not bool(
                        measurement_config.get("use_geometry_context", False)
                    )
                )
                incompatible = teacher_pair_measurement_head.load_state_dict(
                    measurement_state,
                    strict=not (
                        upgrading_set_context or upgrading_geometry_context
                    ),
                )
                if upgrading_set_context or upgrading_geometry_context:
                    allowed_missing = set()
                    if upgrading_set_context:
                        allowed_missing.update(
                            {
                                "set_network.0.weight",
                                "set_network.0.bias",
                                "set_network.2.weight",
                                "set_network.2.bias",
                            }
                        )
                    if upgrading_geometry_context:
                        allowed_missing.update(
                            {
                                "geometry_token_network.0.weight",
                                "geometry_token_network.0.bias",
                                "geometry_token_network.2.weight",
                                "geometry_token_network.2.bias",
                                "geometry_set_network.0.weight",
                                "geometry_set_network.0.bias",
                                "geometry_set_network.2.weight",
                                "geometry_set_network.2.bias",
                            }
                        )
                    if (
                        set(incompatible.missing_keys) != allowed_missing
                        or incompatible.unexpected_keys
                    ):
                        raise ValueError(
                            "incompatible pair measurement set-context upgrade: "
                            f"missing={incompatible.missing_keys} "
                            f"unexpected={incompatible.unexpected_keys}"
                        )
            teacher_pair_measurement_head.requires_grad_(
                optimize_pair_measurement
                and not candidate_teacher_freeze_pair_measurement
            )
        else:
            teacher_pair_measurement_head = None
    else:
        teacher_dustbin_score = None
        teacher_pair_scorer = None
        teacher_pair_measurement_head = None

    if candidate_teacher_matchability_only and candidate_teacher_offset_only:
        raise ValueError("matchability-only and offset-only training are mutually exclusive")
    if sparse_candidate_teacher and candidate_teacher_detector_only:
        # Keep descriptor/map-side objects immutable even when an initialized state
        # contains optional pair modules.
        teacher_landmark_features.requires_grad_(False)
        teacher_dustbin_score.requires_grad_(False)
        if teacher_pair_scorer is not None:
            teacher_pair_scorer.requires_grad_(False)
        if teacher_pair_measurement_head is not None:
            teacher_pair_measurement_head.requires_grad_(False)
    if sparse_candidate_teacher and candidate_teacher_freeze_detector:
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
    elif sparse_candidate_teacher and candidate_teacher_offset_only:
        if detector.offset_head is None:
            raise ValueError("offset-only training requires --candidate_teacher_offset_head")
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        for parameter in detector.offset_head.parameters():
            parameter.requires_grad_(True)
    elif sparse_candidate_teacher and candidate_teacher_matchability_only:
        if detector.matchability_head is None:
            raise ValueError("matchability-only training requires --candidate_teacher_matchability_head")
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        for parameter in detector.matchability_head.parameters():
            parameter.requires_grad_(True)

    if sparse_candidate_teacher:
        parameter_groups = []
        detector_parameters = [parameter for parameter in detector.parameters() if parameter.requires_grad]
        if detector_parameters:
            parameter_groups.append(
                {"params": detector_parameters, "lr": float(candidate_teacher_detector_lr), "name": "detector"}
            )
        if teacher_landmark_features is not None and teacher_landmark_features.requires_grad:
            parameter_groups.append(
                {
                    "params": [teacher_landmark_features],
                    "lr": float(candidate_teacher_feature_lr),
                    "weight_decay": 0.0,
                    "name": "landmark_features",
                }
            )
        if teacher_dustbin_score is not None and teacher_dustbin_score.requires_grad:
            parameter_groups.append(
                {
                    "params": [teacher_dustbin_score],
                    "lr": float(
                        candidate_teacher_dustbin_lr
                        if float(candidate_teacher_dustbin_lr) > 0.0
                        else candidate_teacher_feature_lr
                    ),
                    "weight_decay": 0.0,
                    "name": "dustbin_score",
                }
            )
        if teacher_pair_scorer is not None:
            scorer_parameters = [
                parameter for parameter in teacher_pair_scorer.parameters() if parameter.requires_grad
            ]
            if scorer_parameters:
                parameter_groups.append(
                    {
                        "params": scorer_parameters,
                        "lr": float(candidate_teacher_pair_scorer_lr),
                        "weight_decay": 1e-4,
                        "name": "pair_scorer",
                    }
                )
        if teacher_pair_measurement_head is not None:
            measurement_parameters = [
                parameter
                for parameter in teacher_pair_measurement_head.parameters()
                if parameter.requires_grad
            ]
            if measurement_parameters:
                parameter_groups.append(
                    {
                        "params": measurement_parameters,
                        "lr": float(candidate_teacher_pair_measurement_lr),
                        "weight_decay": 1e-4,
                        "name": "pair_measurement",
                    }
                )
        if not parameter_groups:
            raise ValueError(
                "sparse candidate teacher has no trainable parameters; enable detector training or "
                "--candidate_teacher_optimize_features"
            )
        if candidate_teacher_detector_only:
            group_names = [group["name"] for group in parameter_groups]
            if group_names != ["detector"]:
                raise RuntimeError(
                    "candidate_teacher_detector_only unexpectedly created "
                    f"trainable groups: {group_names}"
                )
            teacher_config["trainable_parameter_groups"] = group_names
            print(
                "Sparse candidate teacher detector-only stage: "
                f"{sum(parameter.numel() for parameter in detector_parameters)} "
                "detector parameters"
            )
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_iteration // grad_accum),
        eta_min=0.0 if sparse_candidate_teacher else 0.0005,
    )
    optimizer.zero_grad()
    hard_candidate_teacher = None
    if (
        sparse_candidate_teacher
        and float(candidate_teacher_hard_preservation_weight) > 0.0
    ):
        hard_candidate_teacher = HardCandidateTeacherCache(
            refresh_visits=candidate_teacher_hard_preservation_refresh_visits,
            solver=candidate_teacher_hard_preservation_solver,
            reprojection_error=candidate_teacher_hard_preservation_reprojection_error,
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            ransac_seed=candidate_teacher_hard_preservation_ransac_seed,
            min_inliers=candidate_teacher_hard_preservation_min_inliers,
            max_pose_error_cm=candidate_teacher_hard_preservation_max_pose_error_cm,
            max_useful=candidate_teacher_hard_preservation_max_useful,
            max_harmful=candidate_teacher_hard_preservation_max_harmful,
            translation_scale=candidate_teacher_map_fisher_translation_scale,
            rotation_scale_degrees=(
                candidate_teacher_map_fisher_rotation_scale_degrees
            ),
            harmful_mode=candidate_teacher_hard_preservation_harmful_mode,
            harmful_min_translation_delete_gain_m=(
                candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m
            ),
            exact_replay_max_candidates=(
                candidate_teacher_hard_preservation_exact_replay_max_candidates
            ),
            exact_replay_min_pose_gain_cm=(
                candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm
            ),
            exact_replay_rotation_weight_cm_per_degree=(
                candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree
            ),
            # ``match_score_matrix`` has already applied
            # candidate_teacher_match_threshold before the candidate tensor is
            # built. The deployed quota mask below therefore replays all
            # remaining candidates with a -inf selector threshold.
            exact_replay_selection_threshold=-float("inf"),
            exact_replay_max_matches_per_landmark=(
                candidate_teacher_map_max_matches_per_landmark
            ),
        )
    camera_failure_scores = torch.ones(len(training_cameras), dtype=torch.float32)
    camera_failure_index = {
        str(camera.image_name): index for index, camera in enumerate(training_cameras)
    }
    online_render_stats = {
        "episodes": 0,
        "failure_guided_episodes": 0,
        "provenance_episodes": 0,
        "provenance_valid_count_total": 0.0,
        "provenance_effective_targets_total": 0.0,
        "pair_weight_max": 0.0,
    }

    for iteration in range(first_iter, train_iteration + 1):
        teacher_gradient_diagnostics = {}
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = training_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        online_render_ratio = scheduled_online_render_ratio(
            iteration,
            train_iteration,
            candidate_teacher_online_render_ratio_start,
            candidate_teacher_online_render_ratio_end,
            candidate_teacher_online_render_ramp_start,
            candidate_teacher_online_render_ramp_end,
        )
        online_render_used = bool(
            sparse_candidate_teacher
            and len(training_cameras) >= 2
            and online_render_ratio > 0.0
            and random() < online_render_ratio
        )
        online_render_pkg = None
        pair_weights = None
        query_valid_mask = None
        if online_render_used:
            if candidate_teacher_online_render_sampling_mode == "failure_guided":
                pair_weights = failure_guided_pair_weights(
                    camera_failure_scores,
                    temperature=candidate_teacher_online_render_failure_temperature,
                    uniform_floor=candidate_teacher_online_render_uniform_floor,
                )
            (
                viewpoint_cam,
                gt_feature_map,
                viewmat,
                online_render_pkg,
            ) = render_online_candidate_query(
                training_cameras,
                gaussians,
                feature_extractor,
                scene.longest_edge,
                background,
                alpha_min=candidate_teacher_online_render_alpha_min,
                alpha_max=candidate_teacher_online_render_alpha_max,
                return_provenance=(
                    candidate_teacher_online_render_provenance_mode != "none"
                ),
                provenance_landmark_indices=sampled_idx,
                pair_weights=pair_weights,
                query_feature_contract=candidate_teacher_query_feature_contract,
            )
        else:
            fine_resolution = get_resolution_from_longest_edge(
                viewpoint_cam.original_image.shape[1],
                viewpoint_cam.original_image.shape[2],
                scene.longest_edge,
            )
            gt_image = viewpoint_cam.original_image.cuda()
            query_valid_mask = camera_valid_mask(
                masks, viewpoint_cam, gt_image.device
            )
            gt_feature_map = extract_normalized_feature_map(
                feature_extractor,
                gt_image,
                size=(fine_resolution[0], fine_resolution[1]),
                query_feature_contract=candidate_teacher_query_feature_contract,
                valid_mask=query_valid_mask,
            )
            viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()

        # get K for either a real or an online-rendered query
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        # get render visible mask
        if online_render_pkg is not None:
            render_visible_mask = online_render_pkg["visibility_filter"].detach()
        else:
            render_visible_mask = render_visible_mask_from_cache(
                render_visible_masks,
                viewpoint_cam.image_name,
                gt_feature_map.device,
            )
            if render_visible_mask is None:
                render_visible_mask = get_render_visible_mask(
                    gaussians,
                    viewpoint_cam,
                    gt_feature_map.shape[2],
                    gt_feature_map.shape[1],
                )
                store_render_visible_mask(
                    render_visible_masks,
                    viewpoint_cam.image_name,
                    render_visible_mask,
                )

        need_base_target = (not sparse_candidate_teacher) or float(candidate_teacher_base_detector_weight) > 0.0
        gt_map = None
        soft_target = False
        weight_map = None
        if need_base_target:
            gt_map, soft_target, weight_map = build_detector_target_map(
                gaussians,
                gt_feature_map,
                sampled_idx,
                viewmat,
                K,
                render_visible_mask=render_visible_mask,
                detector_target_mode=detector_target_mode,
                landmark_meta=landmark_meta,
                soft_sigma=soft_sigma,
                landmark_xyz=teacher_landmark_xyz,
            )

        # use mask to filter out object
        gt_map_mask = None
        if query_valid_mask is not None and not online_render_used:
            gt_map_mask = _resize_hard_valid_mask(
                query_valid_mask,
                gt_feature_map.shape[-2:],
                gt_feature_map.device,
            )[None]
            if gt_map is not None:
                gt_map = gt_map * gt_map_mask
                if weight_map is not None:
                    weight_map = torch.where(gt_map_mask, weight_map, torch.ones_like(weight_map))

        # Loss
        keypoint_heat_map, matchability_heat_map, offset_heat_map = detector.forward_all(
            gt_feature_map
        )
        heat_map = keypoint_heat_map
        candidate_heat_map = torch.sqrt(
            (keypoint_heat_map * matchability_heat_map).clamp_min(0.0)
        )
        base_detector_loss = (
            detector_target_loss(
                heat_map,
                gt_map,
                soft_target=soft_target,
                weight_map=weight_map,
            )
            if gt_map is not None
            else heat_map.sum() * 0.0
        )
        detector_preservation_loss = heat_map.sum() * 0.0
        if detector_preservation_teacher is not None:
            with torch.no_grad():
                teacher_keypoint_heat_map, teacher_matchability_heat_map, _ = (
                    detector_preservation_teacher.forward_all(gt_feature_map)
                )
            detector_preservation_loss = detector_proposal_preservation_loss(
                keypoint_heat_map,
                matchability_heat_map,
                teacher_keypoint_heat_map,
                teacher_matchability_heat_map,
                valid_mask=gt_map_mask,
            )
        teacher_losses = None
        feature_anchor_loss = heat_map.sum() * 0.0
        if sparse_candidate_teacher:
            if candidate_teacher_selection_source == "keypoint_teacher":
                # Phase-D labels must come from a fixed proposal distribution.  The
                # keypoint head is frozen in matchability-only training, so detach it
                # here while retaining gradients through the sampled matchability
                # scores below.
                candidate_selection_heat_map = keypoint_heat_map.detach()
            elif candidate_teacher_selection_source == "combined":
                candidate_selection_heat_map = candidate_heat_map
            else:
                raise ValueError(
                    "candidate_teacher_selection_source must be 'combined' or "
                    f"'keypoint_teacher', got {candidate_teacher_selection_source!r}"
                )
            teacher_heat_map = (
                candidate_selection_heat_map
                if gt_map_mask is None
                else candidate_selection_heat_map * gt_map_mask
            )
            detector_supervision_heatmap = (
                matchability_heat_map
                if gt_map_mask is None
                else matchability_heat_map * gt_map_mask
            )
            teacher_features_for_matching = teacher_landmark_features
            teacher_trust_alpha = None
            teacher_trust_warmup_active = False
            if bool(candidate_teacher_adaptive_trust):
                teacher_trust_warmup_active = (
                    teacher_trust_update_steps < candidate_teacher_trust_warmup_steps
                )
                teacher_trust_warmup_blend = (
                    max(
                        0.0,
                        1.0
                        - float(teacher_trust_update_steps)
                        / float(max(candidate_teacher_trust_warmup_steps, 1)),
                    )
                    if candidate_teacher_trust_warmup_steps > 0
                    else 0.0
                )
                teacher_trust_alpha = candidate_teacher_trust_alpha(
                    teacher_trust_visible_count,
                    teacher_trust_correct_count,
                    alpha_min=candidate_teacher_trust_alpha_min,
                    view_prior=candidate_teacher_trust_view_prior,
                    warmup_active=teacher_trust_warmup_active,
                    warmup_blend=teacher_trust_warmup_blend,
                )
                teacher_features_for_matching = candidate_teacher_effective_features(
                    teacher_initial_features,
                    teacher_landmark_features,
                    teacher_trust_alpha,
                )
            candidate_batch = build_sparse_candidate_batch(
                gt_feature_map,
                teacher_heat_map,
                teacher_features_for_matching,
                teacher_landmark_xyz,
                K,
                viewmat,
                visible_mask=render_visible_mask[sampled_idx],
                detect_num=candidate_teacher_detect_num,
                nms_radius=candidate_teacher_nms_radius,
                match_mode=candidate_teacher_match_mode,
                match_topk=candidate_teacher_match_topk,
                match_threshold=candidate_teacher_match_threshold,
                dual_softmax=candidate_teacher_dual_softmax,
                dual_softmax_temperature=candidate_teacher_dual_softmax_temperature,
                positive_radius_px=candidate_teacher_positive_radius_px,
                negative_radius_px=candidate_teacher_negative_radius_px,
                max_positives=candidate_teacher_max_positives,
                hard_negatives=candidate_teacher_hard_negatives,
                match_temperature=candidate_teacher_match_temperature,
                match_margin=candidate_teacher_match_margin,
                assignment_pose_information_mode=(
                    candidate_teacher_assignment_pose_information_mode
                ),
                assignment_pose_information_weight=(
                    candidate_teacher_assignment_pose_information_weight
                ),
                assignment_pose_information_floor=(
                    candidate_teacher_assignment_pose_information_floor
                ),
                assignment_pose_information_normalization=(
                    candidate_teacher_assignment_pose_information_normalization
                ),
                assignment_fisher_translation_scale=(
                    candidate_teacher_assignment_fisher_translation_scale
                ),
                assignment_fisher_rotation_scale_degrees=(
                    candidate_teacher_assignment_fisher_rotation_scale_degrees
                ),
                assignment_fisher_measurement_sigma=(
                    candidate_teacher_assignment_fisher_measurement_sigma
                ),
                assignment_fisher_use_matchability=(
                    candidate_teacher_assignment_fisher_use_matchability
                ),
                assignment_fisher_matchability_floor=(
                    candidate_teacher_assignment_fisher_matchability_floor
                ),
                assignment_fisher_matchability_power=(
                    candidate_teacher_assignment_fisher_matchability_power
                ),
                assignment_fisher_uncertainty_entropy_scale=(
                    candidate_teacher_assignment_fisher_uncertainty_entropy_scale
                ),
                grid_rows=candidate_teacher_grid_rows,
                grid_cols=candidate_teacher_grid_cols,
                depth_bins=candidate_teacher_depth_bins,
                dustbin_score=teacher_dustbin_score,
                pair_scorer=teacher_pair_scorer,
                pair_measurement_head=teacher_pair_measurement_head,
                pair_context_topk=candidate_teacher_pair_context_topk,
                detector_supervision_heatmap=detector_supervision_heatmap,
                keypoint_offset_map=offset_heat_map,
                pair_measurement_accept_threshold=(
                    teacher_pair_measurement_accept_threshold
                ),
                detector_offset_target_source=(
                    candidate_teacher_offset_target_source
                ),
                detector_target_source=candidate_teacher_detector_target_source,
                detector_binary_target=candidate_teacher_detector_binary_target,
                map_max_matches_per_landmark=(
                    candidate_teacher_map_max_matches_per_landmark
                ),
                directional_candidate_topk=(
                    candidate_teacher_map_directional_topk
                ),
                counterfactual_enabled=(
                    float(candidate_teacher_counterfactual_assignment_weight) > 0.0
                ),
                counterfactual_bias_utility_weight=(
                    candidate_teacher_counterfactual_bias_utility_weight
                ),
                counterfactual_translation_utility_weight=(
                    candidate_teacher_counterfactual_translation_utility_weight
                ),
                counterfactual_utility_floor=(
                    candidate_teacher_counterfactual_utility_floor
                ),
                counterfactual_target_mode=(
                    candidate_teacher_counterfactual_target_mode
                ),
                counterfactual_require_current_retained=(
                    candidate_teacher_counterfactual_require_current_retained
                ),
                counterfactual_exact_decision_set=(
                    candidate_teacher_counterfactual_exact_decision_set
                ),
                counterfactual_require_positive_bias_gain=(
                    candidate_teacher_counterfactual_require_positive_bias_gain
                ),
                counterfactual_require_nonnegative_translation_gain=(
                    candidate_teacher_counterfactual_require_nonnegative_translation_gain
                ),
                counterfactual_translation_scale=(
                    candidate_teacher_map_fisher_translation_scale
                ),
                counterfactual_rotation_scale_degrees=(
                    candidate_teacher_map_fisher_rotation_scale_degrees
                ),
                counterfactual_measurement_sigma_px=(
                    candidate_teacher_map_fisher_measurement_sigma_px
                ),
                counterfactual_residual_clip_px=(
                    candidate_teacher_map_fisher_residual_clip_px
                ),
                counterfactual_inlier_sigma_px=(
                    candidate_teacher_map_fisher_inlier_sigma_px
                ),
                splat_provenance_meta=(
                    online_render_pkg.get("rgb_meta")
                    if online_render_pkg is not None
                    and candidate_teacher_online_render_provenance_mode != "none"
                    else None
                ),
                landmark_global_indices=(
                    torch.arange(
                        sampled_idx.numel(),
                        device=sampled_idx.device,
                        dtype=torch.long,
                    )
                    if online_render_pkg is not None
                    and candidate_teacher_online_render_provenance_mode != "none"
                    else None
                ),
                splat_provenance_mode=(
                    candidate_teacher_online_render_provenance_mode
                    if online_render_pkg is not None
                    else "none"
                ),
                splat_provenance_topk=(
                    candidate_teacher_online_render_provenance_topk
                ),
                splat_provenance_temperature=(
                    candidate_teacher_online_render_provenance_temperature
                ),
            )
            hard_preservation_loss = heat_map.sum() * 0.0
            if hard_candidate_teacher is not None and not online_render_used:
                hard_targets = hard_candidate_teacher.build(
                    str(viewpoint_cam.image_name),
                    keypoint_ids=candidate_batch.keypoint_ids,
                    candidate_keypoint_idx=candidate_batch.pair_scorer_keypoint_idx,
                    candidate_landmark_idx=candidate_batch.pair_scorer_landmark_idx,
                    candidate_scores=(
                        candidate_batch.matcher_deployment_score.detach()
                    ),
                    keypoint_xy=candidate_batch.keypoint_xy,
                    deployment_mask=candidate_batch.map_candidate_quota_mask,
                    gt_correct_mask=(candidate_batch.pair_scorer_labels > 0.5),
                    landmark_xyz=teacher_landmark_xyz,
                    K=K,
                    pose_gt_w2c=viewmat,
                )
                hard_preservation_loss, hard_diagnostics = (
                    hard_candidate_preservation_loss(
                        candidate_batch.matcher_similarity,
                        hard_targets,
                        temperature=candidate_teacher_hard_preservation_temperature,
                        margin=candidate_teacher_hard_preservation_margin,
                        score_target=candidate_teacher_match_margin,
                        require_harmful=(
                            candidate_teacher_hard_preservation_harmful_mode
                            == "exact_pose_delete"
                        ),
                    )
                )
                candidate_batch.diagnostics.update(hard_targets.diagnostics)
                candidate_batch.diagnostics.update(hard_diagnostics)
            else:
                candidate_batch.diagnostics.update(
                    {
                        "hard_teacher_refreshed": 0.0,
                        "hard_teacher_cached_target_count": 0.0,
                        "hard_teacher_loss_useful": 0.0,
                        "hard_teacher_loss_harmful": 0.0,
                        "hard_teacher_loss_pairwise": 0.0,
                        "hard_teacher_loss_skipped_no_harmful": 0.0,
                    }
                )
            teacher_losses = sparse_candidate_losses(
                candidate_batch,
                assignment_mode=candidate_teacher_assignment_mode,
                assignment_temperature=candidate_teacher_assignment_temperature,
                assignment_margin=candidate_teacher_assignment_margin,
                reprojection_sigma_px=candidate_teacher_reprojection_sigma_px,
                set_risk_residual_clip_px=(
                    candidate_teacher_pair_measurement_residual_clip_px
                ),
                set_risk_reference_translation_m=(
                    candidate_teacher_pair_measurement_reference_translation_m
                ),
                map_fisher_translation_scale=(
                    candidate_teacher_map_fisher_translation_scale
                ),
                map_fisher_rotation_scale_degrees=(
                    candidate_teacher_map_fisher_rotation_scale_degrees
                ),
                map_fisher_measurement_sigma_px=(
                    candidate_teacher_map_fisher_measurement_sigma_px
                ),
                map_fisher_residual_clip_px=(
                    candidate_teacher_map_fisher_residual_clip_px
                ),
                map_fisher_inlier_sigma_px=(
                    candidate_teacher_map_fisher_inlier_sigma_px
                ),
                map_fisher_condition_target=(
                    candidate_teacher_map_fisher_condition_target
                ),
                map_bias_huber_delta=candidate_teacher_map_bias_huber_delta,
                map_bias_clip=candidate_teacher_map_bias_clip,
                map_directional_temperature=(
                    candidate_teacher_map_directional_temperature
                ),
                map_directional_residual_clip_px=(
                    candidate_teacher_map_directional_residual_clip_px
                ),
                map_directional_robust_scale_px=(
                    candidate_teacher_map_directional_robust_scale_px
                ),
                map_directional_robust_quality_floor=(
                    candidate_teacher_map_directional_robust_quality_floor
                ),
            )
            if not online_render_used:
                failure_index = camera_failure_index.get(str(viewpoint_cam.image_name))
                if failure_index is not None:
                    update_camera_failure_ema(
                        camera_failure_scores,
                        failure_index,
                        candidate_query_failure_score(teacher_losses),
                        decay=candidate_teacher_online_render_failure_ema,
                    )
            teacher_last_diagnostics = _numeric_teacher_diagnostics(candidate_batch.diagnostics)
            if hard_candidate_teacher is not None:
                teacher_last_diagnostics.update(hard_candidate_teacher.diagnostics())
            if bool(candidate_teacher_adaptive_trust) and not online_render_used:
                trust_camera_index = teacher_trust_camera_index.get(
                    str(viewpoint_cam.image_name)
                )
                if trust_camera_index is None:
                    raise ValueError(
                        "real candidate-teacher query is absent from the "
                        "adaptive-trust camera index"
                    )
                update_candidate_teacher_trust_evidence(
                    teacher_trust_visible_count,
                    teacher_trust_correct_count,
                    render_visible_mask[sampled_idx],
                    candidate_batch.pair_scorer_landmark_idx,
                    candidate_batch.pair_scorer_labels > 0.5,
                    candidate_batch.map_candidate_classification_valid_mask,
                    camera_index=trust_camera_index,
                    visible_view_mask=teacher_trust_visible_view_mask,
                    correct_view_mask=teacher_trust_correct_view_mask,
                    report=False,
                    validate_indices=False,
                )
                teacher_trust_update_steps += 1
                teacher_last_diagnostics["trust_evidence_updates"] = float(
                    teacher_trust_update_steps
                )
            elif bool(candidate_teacher_adaptive_trust):
                teacher_last_diagnostics["trust_synthetic_query_skipped"] = 1.0
            if candidate_teacher_optimize_features:
                feature_anchor_drift = (
                    1.0
                    - (
                        F.normalize(teacher_features_for_matching, dim=1)
                        * teacher_initial_features
                    ).sum(dim=1)
                ).clamp_min(0.0)
                if teacher_trust_alpha is None:
                    feature_anchor_loss = feature_anchor_drift.mean()
                else:
                    trust_weight = (teacher_trust_alpha + 1e-6).reciprocal()
                    feature_anchor_loss = (
                        feature_anchor_drift * trust_weight
                    ).sum() / trust_weight.sum().clamp_min(1e-6)
            loss = (
                float(candidate_teacher_pair_weight) * teacher_losses.pair
                + float(candidate_teacher_hard_negative_weight) * teacher_losses.hard_negative
                + float(candidate_teacher_assignment_weight) * teacher_losses.assignment
                + float(candidate_teacher_online_render_provenance_weight)
                * teacher_losses.provenance_assignment
                + float(candidate_teacher_counterfactual_assignment_weight)
                * teacher_losses.counterfactual_assignment
                + float(candidate_teacher_hard_preservation_weight)
                * hard_preservation_loss
                + float(candidate_teacher_dustbin_weight) * teacher_losses.dustbin_assignment
                + float(candidate_teacher_matcher_assignment_weight)
                * teacher_losses.matcher_assignment
                + float(candidate_teacher_matcher_reprojection_weight)
                * teacher_losses.matcher_reprojection_assignment
                + float(candidate_teacher_pair_scorer_weight) * teacher_losses.pair_scorer
                + float(candidate_teacher_pair_scorer_assignment_weight)
                * teacher_losses.pair_scorer_assignment
                + float(candidate_teacher_pair_measurement_inlier_weight)
                * teacher_losses.pair_measurement_inlier
                + float(candidate_teacher_pair_measurement_nll_weight)
                * teacher_losses.pair_measurement_nll
                + float(candidate_teacher_pair_measurement_bias_weight)
                * teacher_losses.pair_measurement_translation_bias
                + float(candidate_teacher_pair_measurement_covariance_weight)
                * teacher_losses.pair_measurement_translation_covariance
                + float(candidate_teacher_matcher_translation_info_weight)
                * teacher_losses.matcher_translation_info
                + float(candidate_teacher_translation_info_weight) * teacher_losses.translation_info
                + float(candidate_teacher_detector_match_weight) * teacher_losses.detector_match
                + float(candidate_teacher_detector_offset_weight)
                * teacher_losses.detector_offset
                + float(candidate_teacher_geometry_weight) * teacher_losses.geometry_set
                + float(candidate_teacher_coverage_weight) * teacher_losses.coverage
                + float(candidate_teacher_map_cleanliness_weight)
                * teacher_losses.map_cleanliness
                + float(candidate_teacher_map_full_information_weight)
                * teacher_losses.map_full_information
                + float(candidate_teacher_map_translation_information_weight)
                * teacher_losses.map_translation_information
                + float(candidate_teacher_map_translation_trace_weight)
                * teacher_losses.map_translation_trace
                + float(candidate_teacher_map_translation_condition_weight)
                * teacher_losses.map_translation_condition
                + float(candidate_teacher_map_bias_weight) * teacher_losses.map_bias
                + float(candidate_teacher_map_directional_bias_weight)
                * teacher_losses.map_directional_bias
                + float(candidate_teacher_map_capacity_weight)
                * teacher_losses.map_capacity
                + float(candidate_teacher_base_detector_weight) * base_detector_loss
                + float(candidate_teacher_detector_preservation_weight)
                * detector_preservation_loss
                + float(candidate_teacher_feature_anchor_weight) * feature_anchor_loss
            )
            if candidate_teacher_detector_only:
                loss = (
                    float(candidate_teacher_detector_match_weight)
                    * teacher_losses.detector_match
                    + float(candidate_teacher_detector_offset_weight)
                    * teacher_losses.detector_offset
                )
            if online_render_used:
                online_render_stats["episodes"] += 1
                if pair_weights is not None:
                    online_render_stats["failure_guided_episodes"] += 1
                    online_render_stats["pair_weight_max"] = max(
                        online_render_stats["pair_weight_max"],
                        float(pair_weights.max().item()),
                    )
                provenance_valid = float(
                    teacher_last_diagnostics.get("splat_provenance_valid_count", 0.0)
                )
                provenance_effective = float(
                    teacher_last_diagnostics.get("splat_provenance_effective_targets", 0.0)
                )
                if provenance_valid > 0.0:
                    online_render_stats["provenance_episodes"] += 1
                online_render_stats["provenance_valid_count_total"] += provenance_valid
                online_render_stats["provenance_effective_targets_total"] += provenance_effective
            teacher_last_diagnostics.update(
                {
                    "online_render_used": 1.0 if online_render_used else 0.0,
                    "online_render_ratio": float(online_render_ratio),
                    "online_render_alpha": float(
                        getattr(viewpoint_cam, "alpha", 0.0)
                        if online_render_used
                        else 0.0
                    ),
                    "online_render_failure_mean": float(camera_failure_scores.mean().item()),
                    "online_render_failure_max": float(camera_failure_scores.max().item()),
                    "online_render_pair_weight_max": float(
                        pair_weights.max().item() if pair_weights is not None and pair_weights.numel() else 0.0
                    ),
                }
            )
        else:
            loss = base_detector_loss

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite detector loss at iteration {iteration}: {float(loss.detach().item())}"
            )

        gradient_audit = sparse_candidate_teacher and (
            iteration == 1 or iteration % 50 == 0 or iteration == train_iteration
        )
        if gradient_audit:
            teacher_gradient_diagnostics.update(
                {
                    "grad_provenance_loc_feature": _isolated_loss_gradient_norm(
                        float(candidate_teacher_online_render_provenance_weight)
                        * teacher_losses.provenance_assignment,
                        [teacher_landmark_features],
                    ),
                    "grad_provenance_dustbin": _isolated_loss_gradient_norm(
                        float(candidate_teacher_online_render_provenance_weight)
                        * teacher_losses.provenance_assignment,
                        [teacher_dustbin_score],
                    ),
                    "grad_counterfactual_loc_feature": _isolated_loss_gradient_norm(
                        float(candidate_teacher_counterfactual_assignment_weight)
                        * teacher_losses.counterfactual_assignment,
                        [teacher_landmark_features],
                    ),
                    "grad_counterfactual_dustbin": _isolated_loss_gradient_norm(
                        float(candidate_teacher_counterfactual_assignment_weight)
                        * teacher_losses.counterfactual_assignment,
                        [teacher_dustbin_score],
                    ),
                    "grad_hard_preservation_loc_feature": _isolated_loss_gradient_norm(
                        float(candidate_teacher_hard_preservation_weight)
                        * hard_preservation_loss,
                        [teacher_landmark_features],
                    ),
                    "grad_feature_anchor_loc_feature": _isolated_loss_gradient_norm(
                        float(candidate_teacher_feature_anchor_weight)
                        * feature_anchor_loss,
                        [teacher_landmark_features],
                    ),
                    "grad_detector_preservation_detector_trunk": (
                        _isolated_loss_gradient_norm(
                            float(candidate_teacher_detector_preservation_weight)
                            * detector_preservation_loss,
                            detector.cnn.parameters(),
                        )
                    ),
                    "grad_detector_preservation_matchability": (
                        _isolated_loss_gradient_norm(
                            float(candidate_teacher_detector_preservation_weight)
                            * detector_preservation_loss,
                            detector.matchability_head.parameters()
                            if detector.matchability_head is not None
                            else [],
                        )
                    ),
                }
            )

        loss.backward()
        if gradient_audit:
            teacher_gradient_diagnostics.update(
                {
                    "grad_total_loc_feature": _parameter_gradient_norm(
                        [teacher_landmark_features]
                    ),
                    "grad_total_detector_trunk": _parameter_gradient_norm(
                        detector.cnn.parameters()
                    ),
                    "grad_total_matchability": _parameter_gradient_norm(
                        detector.matchability_head.parameters()
                        if detector.matchability_head is not None
                        else []
                    ),
                    "grad_total_detector_offset": _parameter_gradient_norm(
                        detector.offset_head.parameters()
                        if detector.offset_head is not None
                        else []
                    ),
                    "grad_total_pair_scorer": _parameter_gradient_norm(
                        teacher_pair_scorer.parameters()
                        if teacher_pair_scorer is not None
                        else []
                    ),
                    "grad_total_pair_measurement": _parameter_gradient_norm(
                        teacher_pair_measurement_head.parameters()
                        if teacher_pair_measurement_head is not None
                        else []
                    ),
                    "grad_total_dustbin": _parameter_gradient_norm(
                        [teacher_dustbin_score]
                    ),
                    "grad_total_geometry_anchor": _parameter_gradient_norm(
                        [getattr(gaussians, "_loc_anchor_offset", None)]
                    ),
                }
            )
            teacher_last_diagnostics.update(teacher_gradient_diagnostics)
        if iteration % grad_accum == 0 or iteration == train_iteration:
            if sparse_candidate_teacher:
                trainable_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if trainable_parameters:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=grad_clip_norm,
                    )
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                postfix = {"Loss": f"{loss_val:.7f}"}
                if sparse_candidate_teacher:
                    postfix.update(
                        {
                            "Pair": f"{float(teacher_losses.pair.detach().item()):.4f}",
                            "Rank": f"{float(teacher_losses.assignment.detach().item()):.4f}",
                            "Swap": f"{float(teacher_losses.counterfactual_assignment.detach().item()):.4f}",
                            "Hard": f"{float(hard_preservation_loss.detach().item()):.4f}",
                            "Reproj": f"{float(teacher_losses.matcher_reprojection_assignment.detach().item()):.4f}",
                            "Dust": f"{float(teacher_losses.dustbin_assignment.detach().item()):.4f}",
                            "Score": f"{float(teacher_losses.pair_scorer.detach().item()):.4f}",
                            "Trans": f"{float(teacher_losses.translation_info.detach().item()):.4f}",
                            "MapB": f"{float(teacher_losses.map_bias.detach().item()):.4f}",
                            "DirB": f"{float(teacher_losses.map_directional_bias.detach().item()):.4f}",
                            "Offset": f"{float(teacher_losses.detector_offset.detach().item()):.4f}",
                            "Keep": f"{float(detector_preservation_loss.detach().item()):.4f}",
                            "Prec": f"{teacher_last_diagnostics.get('predicted_gt_precision', 0.0):.3f}",
                            "FN": f"{teacher_last_diagnostics.get('false_negative_rate', 0.0):.3f}",
                        }
                    )
                progress_bar.set_postfix(
                    postfix
                )
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )
                if sparse_candidate_teacher:
                    component_values = {
                        "pair": teacher_losses.pair,
                        "hard_negative": teacher_losses.hard_negative,
                        "assignment": teacher_losses.assignment,
                        "provenance_assignment": (
                            teacher_losses.provenance_assignment
                        ),
                        "counterfactual_assignment": (
                            teacher_losses.counterfactual_assignment
                        ),
                        "hard_preservation": hard_preservation_loss,
                        "dustbin_assignment": teacher_losses.dustbin_assignment,
                        "matcher_assignment": teacher_losses.matcher_assignment,
                        "matcher_reprojection_assignment": (
                            teacher_losses.matcher_reprojection_assignment
                        ),
                        "pair_scorer": teacher_losses.pair_scorer,
                        "pair_scorer_assignment": teacher_losses.pair_scorer_assignment,
                        "pair_measurement_inlier": teacher_losses.pair_measurement_inlier,
                        "pair_measurement_nll": teacher_losses.pair_measurement_nll,
                        "pair_measurement_translation_bias": (
                            teacher_losses.pair_measurement_translation_bias
                        ),
                        "pair_measurement_translation_covariance": (
                            teacher_losses.pair_measurement_translation_covariance
                        ),
                        "matcher_translation_info": teacher_losses.matcher_translation_info,
                        "translation_info": teacher_losses.translation_info,
                        "detector_match": teacher_losses.detector_match,
                        "detector_offset": teacher_losses.detector_offset,
                        "geometry_set": teacher_losses.geometry_set,
                        "coverage": teacher_losses.coverage,
                        "map_cleanliness": teacher_losses.map_cleanliness,
                        "map_full_information": teacher_losses.map_full_information,
                        "map_translation_information": (
                            teacher_losses.map_translation_information
                        ),
                        "map_translation_trace": teacher_losses.map_translation_trace,
                        "map_translation_condition": (
                            teacher_losses.map_translation_condition
                        ),
                        "map_bias": teacher_losses.map_bias,
                        "map_directional_bias": teacher_losses.map_directional_bias,
                        "map_capacity": teacher_losses.map_capacity,
                        "base_detector": base_detector_loss,
                        "detector_preservation": detector_preservation_loss,
                        "feature_anchor": feature_anchor_loss,
                    }
                    for name, value in component_values.items():
                        tb_writer.add_scalar(
                            f"sparse_candidate_teacher/loss_{name}",
                            float(value.detach().item()),
                            iteration,
                        )
                    del component_values
                    for name, value in teacher_last_diagnostics.items():
                        tb_writer.add_scalar(
                            f"sparse_candidate_teacher/{name}",
                            value,
                            iteration,
                        )

            if sparse_candidate_teacher and (
                iteration == 1 or iteration % 50 == 0 or iteration == train_iteration
            ):
                history_item = {
                    "iteration": int(iteration),
                    "loss_total": float(loss.detach().item()),
                    "loss_pair": float(teacher_losses.pair.detach().item()),
                    "loss_hard_negative": float(teacher_losses.hard_negative.detach().item()),
                    "loss_assignment": float(teacher_losses.assignment.detach().item()),
                    "loss_counterfactual_assignment": float(
                        teacher_losses.counterfactual_assignment.detach().item()
                    ),
                    "loss_hard_preservation": float(
                        hard_preservation_loss.detach().item()
                    ),
                    "loss_dustbin_assignment": float(
                        teacher_losses.dustbin_assignment.detach().item()
                    ),
                    "loss_matcher_assignment": float(
                        teacher_losses.matcher_assignment.detach().item()
                    ),
                    "loss_matcher_reprojection_assignment": float(
                        teacher_losses.matcher_reprojection_assignment.detach().item()
                    ),
                    "loss_pair_scorer": float(teacher_losses.pair_scorer.detach().item()),
                    "loss_pair_scorer_assignment": float(
                        teacher_losses.pair_scorer_assignment.detach().item()
                    ),
                    "loss_pair_measurement_inlier": float(
                        teacher_losses.pair_measurement_inlier.detach().item()
                    ),
                    "loss_pair_measurement_nll": float(
                        teacher_losses.pair_measurement_nll.detach().item()
                    ),
                    "loss_pair_measurement_translation_bias": float(
                        teacher_losses.pair_measurement_translation_bias.detach().item()
                    ),
                    "loss_pair_measurement_translation_covariance": float(
                        teacher_losses.pair_measurement_translation_covariance.detach().item()
                    ),
                    "loss_matcher_translation_info": float(
                        teacher_losses.matcher_translation_info.detach().item()
                    ),
                    "loss_translation_info": float(
                        teacher_losses.translation_info.detach().item()
                    ),
                    "loss_detector_match": float(teacher_losses.detector_match.detach().item()),
                    "loss_detector_offset": float(
                        teacher_losses.detector_offset.detach().item()
                    ),
                    "loss_geometry_set": float(teacher_losses.geometry_set.detach().item()),
                    "loss_coverage": float(teacher_losses.coverage.detach().item()),
                    "loss_map_cleanliness": float(
                        teacher_losses.map_cleanliness.detach().item()
                    ),
                    "loss_map_full_information": float(
                        teacher_losses.map_full_information.detach().item()
                    ),
                    "loss_map_translation_information": float(
                        teacher_losses.map_translation_information.detach().item()
                    ),
                    "loss_map_translation_trace": float(
                        teacher_losses.map_translation_trace.detach().item()
                    ),
                    "loss_map_translation_condition": float(
                        teacher_losses.map_translation_condition.detach().item()
                    ),
                    "loss_map_bias": float(teacher_losses.map_bias.detach().item()),
                    "loss_map_directional_bias": float(
                        teacher_losses.map_directional_bias.detach().item()
                    ),
                    "loss_map_capacity": float(
                        teacher_losses.map_capacity.detach().item()
                    ),
                    "loss_base_detector": float(base_detector_loss.detach().item()),
                    "loss_detector_preservation": float(
                        detector_preservation_loss.detach().item()
                    ),
                    "loss_feature_anchor": float(feature_anchor_loss.detach().item()),
                }
                history_item.update(teacher_last_diagnostics)
                teacher_history.append(history_item)

        checkpoint_iteration = (
            iteration in testing_iterations or iteration in saving_iterations
        )
        if checkpoint_iteration:
            # Validation builds another full candidate matrix. Release the completed
            # training graph first so both matrices never coexist on CUDA.
            if sparse_candidate_teacher:
                del candidate_batch
                del candidate_selection_heat_map
                del teacher_heat_map
                del detector_supervision_heatmap
                del teacher_features_for_matching
            del loss
            del teacher_losses
            del base_detector_loss
            del detector_preservation_loss
            del feature_anchor_loss
            del keypoint_heat_map
            del matchability_heat_map
            del offset_heat_map
            del heat_map
            del candidate_heat_map
            del gt_feature_map
            torch.cuda.empty_cache()

        if iteration in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration))
            detector.eval()
            evaluate_detector(
                detector,
                feature_extractor,
                gaussians,
                sampled_idx,
                scene,
                masks,
                render_visible_masks,
                tb_writer,
                iteration,
                query_feature_contract=candidate_teacher_query_feature_contract,
            )
            detector.train()

        if iteration in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration))
            teacher_features_for_export = teacher_landmark_features
            normalize_exported_features = bool(candidate_teacher_optimize_features)
            adaptive_trust_state_for_export = None
            if sparse_candidate_teacher and bool(candidate_teacher_adaptive_trust):
                with torch.no_grad():
                    export_trust_warmup_active = (
                        teacher_trust_update_steps < candidate_teacher_trust_warmup_steps
                    )
                    export_trust_warmup_blend = (
                        max(
                            0.0,
                            1.0
                            - float(teacher_trust_update_steps)
                            / float(max(candidate_teacher_trust_warmup_steps, 1)),
                        )
                        if candidate_teacher_trust_warmup_steps > 0
                        else 0.0
                    )
                    export_trust_alpha = candidate_teacher_trust_alpha(
                        teacher_trust_visible_count,
                        teacher_trust_correct_count,
                        alpha_min=candidate_teacher_trust_alpha_min,
                        view_prior=candidate_teacher_trust_view_prior,
                        warmup_active=export_trust_warmup_active,
                        warmup_blend=export_trust_warmup_blend,
                    )
                    teacher_features_for_export = candidate_teacher_effective_features(
                        teacher_initial_features,
                        teacher_landmark_features,
                        export_trust_alpha,
                    )
                    normalize_exported_features = True
                    teacher_last_diagnostics.update(
                        candidate_teacher_trust_diagnostics(
                            teacher_trust_visible_count,
                            teacher_trust_correct_count,
                            export_trust_alpha,
                            warmup_active=export_trust_warmup_active,
                        )
                    )
                    teacher_last_diagnostics["trust_evidence_updates"] = float(
                        teacher_trust_update_steps
                    )
                    teacher_last_diagnostics["trust_warmup_blend"] = float(
                        export_trust_warmup_blend
                    )
                    adaptive_trust_state_for_export = {
                        "initial_features": teacher_initial_features,
                        "raw_features": teacher_landmark_features,
                        "visible_count": teacher_trust_visible_count,
                        "correct_count": teacher_trust_correct_count,
                        "visible_view_mask": teacher_trust_visible_view_mask,
                        "correct_view_mask": teacher_trust_correct_view_mask,
                        "evidence_camera_names": teacher_trust_camera_names,
                        "update_steps": teacher_trust_update_steps,
                    }
            if sparse_candidate_teacher and validation_cameras:
                validation_metrics = evaluate_sparse_candidate_teacher(
                    detector,
                    feature_extractor,
                    gaussians,
                    sampled_idx,
                    teacher_features_for_export,
                    teacher_landmark_xyz,
                    teacher_dustbin_score,
                    teacher_pair_scorer,
                    teacher_pair_measurement_head,
                    validation_cameras,
                    render_visible_masks,
                    masks,
                    scene,
                    candidate_kwargs={
                        "detect_num": candidate_teacher_detect_num,
                        "nms_radius": candidate_teacher_nms_radius,
                        "match_mode": candidate_teacher_match_mode,
                        "match_topk": candidate_teacher_match_topk,
                        "match_threshold": candidate_teacher_match_threshold,
                        "dual_softmax": candidate_teacher_dual_softmax,
                        "dual_softmax_temperature": candidate_teacher_dual_softmax_temperature,
                        "positive_radius_px": candidate_teacher_positive_radius_px,
                        "negative_radius_px": candidate_teacher_negative_radius_px,
                        "max_positives": candidate_teacher_max_positives,
                        "hard_negatives": candidate_teacher_hard_negatives,
                        "match_temperature": candidate_teacher_match_temperature,
                        "match_margin": candidate_teacher_match_margin,
                        "assignment_pose_information_mode": (
                            candidate_teacher_assignment_pose_information_mode
                        ),
                        "assignment_pose_information_weight": (
                            candidate_teacher_assignment_pose_information_weight
                        ),
                        "assignment_pose_information_floor": (
                            candidate_teacher_assignment_pose_information_floor
                        ),
                        "assignment_pose_information_normalization": (
                            candidate_teacher_assignment_pose_information_normalization
                        ),
                        "assignment_fisher_translation_scale": (
                            candidate_teacher_assignment_fisher_translation_scale
                        ),
                        "assignment_fisher_rotation_scale_degrees": (
                            candidate_teacher_assignment_fisher_rotation_scale_degrees
                        ),
                        "assignment_fisher_measurement_sigma": (
                            candidate_teacher_assignment_fisher_measurement_sigma
                        ),
                        "assignment_fisher_use_matchability": (
                            candidate_teacher_assignment_fisher_use_matchability
                        ),
                        "assignment_fisher_matchability_floor": (
                            candidate_teacher_assignment_fisher_matchability_floor
                        ),
                        "assignment_fisher_matchability_power": (
                            candidate_teacher_assignment_fisher_matchability_power
                        ),
                        "assignment_fisher_uncertainty_entropy_scale": (
                            candidate_teacher_assignment_fisher_uncertainty_entropy_scale
                        ),
                        "grid_rows": candidate_teacher_grid_rows,
                        "grid_cols": candidate_teacher_grid_cols,
                        "depth_bins": candidate_teacher_depth_bins,
                        "pair_context_topk": candidate_teacher_pair_context_topk,
                        "pair_measurement_accept_threshold": (
                            teacher_pair_measurement_accept_threshold
                        ),
                        "detector_offset_target_source": (
                            candidate_teacher_offset_target_source
                        ),
                        "detector_target_source": candidate_teacher_detector_target_source,
                        "detector_binary_target": candidate_teacher_detector_binary_target,
                        "map_max_matches_per_landmark": (
                            candidate_teacher_map_max_matches_per_landmark
                        ),
                        "directional_candidate_topk": (
                            candidate_teacher_map_directional_topk
                        ),
                        "counterfactual_enabled": (
                            float(candidate_teacher_counterfactual_assignment_weight)
                            > 0.0
                        ),
                        "counterfactual_bias_utility_weight": (
                            candidate_teacher_counterfactual_bias_utility_weight
                        ),
                        "counterfactual_translation_utility_weight": (
                            candidate_teacher_counterfactual_translation_utility_weight
                        ),
                        "counterfactual_utility_floor": (
                            candidate_teacher_counterfactual_utility_floor
                        ),
                        "counterfactual_target_mode": (
                            candidate_teacher_counterfactual_target_mode
                        ),
                        "counterfactual_require_current_retained": (
                            candidate_teacher_counterfactual_require_current_retained
                        ),
                        "counterfactual_exact_decision_set": (
                            candidate_teacher_counterfactual_exact_decision_set
                        ),
                        "counterfactual_require_positive_bias_gain": (
                            candidate_teacher_counterfactual_require_positive_bias_gain
                        ),
                        "counterfactual_require_nonnegative_translation_gain": (
                            candidate_teacher_counterfactual_require_nonnegative_translation_gain
                        ),
                        "counterfactual_translation_scale": (
                            candidate_teacher_map_fisher_translation_scale
                        ),
                        "counterfactual_rotation_scale_degrees": (
                            candidate_teacher_map_fisher_rotation_scale_degrees
                        ),
                        "counterfactual_measurement_sigma_px": (
                            candidate_teacher_map_fisher_measurement_sigma_px
                        ),
                        "counterfactual_residual_clip_px": (
                            candidate_teacher_map_fisher_residual_clip_px
                        ),
                        "counterfactual_inlier_sigma_px": (
                            candidate_teacher_map_fisher_inlier_sigma_px
                        ),
                    },
                    assignment_mode=candidate_teacher_assignment_mode,
                    assignment_temperature=candidate_teacher_assignment_temperature,
                    assignment_margin=candidate_teacher_assignment_margin,
                    reprojection_sigma_px=candidate_teacher_reprojection_sigma_px,
                    scorer_min_recall=candidate_teacher_scorer_min_recall,
                    scorer_max_matches_per_keypoint=(
                        candidate_teacher_scorer_max_matches_per_keypoint
                    ),
                    set_risk_residual_clip_px=(
                        candidate_teacher_pair_measurement_residual_clip_px
                    ),
                    set_risk_reference_translation_m=(
                        candidate_teacher_pair_measurement_reference_translation_m
                    ),
                    map_fisher_translation_scale=(
                        candidate_teacher_map_fisher_translation_scale
                    ),
                    map_fisher_rotation_scale_degrees=(
                        candidate_teacher_map_fisher_rotation_scale_degrees
                    ),
                    map_fisher_measurement_sigma_px=(
                        candidate_teacher_map_fisher_measurement_sigma_px
                    ),
                    map_fisher_residual_clip_px=(
                        candidate_teacher_map_fisher_residual_clip_px
                    ),
                    map_fisher_inlier_sigma_px=(
                        candidate_teacher_map_fisher_inlier_sigma_px
                    ),
                    map_fisher_condition_target=(
                        candidate_teacher_map_fisher_condition_target
                    ),
                    map_bias_huber_delta=candidate_teacher_map_bias_huber_delta,
                    map_bias_clip=candidate_teacher_map_bias_clip,
                    map_directional_temperature=(
                        candidate_teacher_map_directional_temperature
                    ),
                    map_directional_residual_clip_px=(
                        candidate_teacher_map_directional_residual_clip_px
                    ),
                    map_directional_robust_scale_px=(
                        candidate_teacher_map_directional_robust_scale_px
                    ),
                    map_directional_robust_quality_floor=(
                        candidate_teacher_map_directional_robust_quality_floor
                    ),
                    query_feature_contract=candidate_teacher_query_feature_contract,
                )
                calibrated_pair_scorer_threshold = validation_metrics.get(
                    "pair_scorer_calibrated_threshold"
                )
                calibrated_pair_measurement_threshold = validation_metrics.get(
                    "pair_measurement_calibrated_threshold"
                )
                validation_item = {"iteration": int(iteration)}
                validation_item.update(validation_metrics)
                teacher_validation_history.append(validation_item)
                print(
                    "Sparse candidate validation: "
                    f"AP={validation_metrics.get('pair_ap_mean', 0.0):.4f} "
                    f"scorer_AP={validation_metrics.get('pair_scorer_ap_mean', 0.0):.4f} "
                    f"scorer_thr={validation_metrics.get('pair_scorer_calibrated_threshold', 0.0):.4f} "
                    f"measurement_AP={validation_metrics.get('pair_measurement_ap_mean', 0.0):.4f} "
                    f"measurement_EPE={validation_metrics.get('pair_measurement_offset_epe_mean_mean', 0.0):.4f} "
                    f"accepted_precision={validation_metrics.get('dustbin_accepted_gt_precision_mean', 0.0):.4f} "
                    f"reject={validation_metrics.get('dustbin_unmatched_reject_accuracy_mean', 0.0):.4f}"
                )
            torch.save(detector.state_dict(), save_path + f"/{iteration}_detector.pth")
            if sparse_candidate_teacher:
                state_path = os.path.join(
                    save_path,
                    f"{iteration}_candidate_teacher_state.pt",
                )
                save_sparse_candidate_teacher_state(
                    state_path,
                    sampled_idx,
                    teacher_features_for_export,
                    iteration,
                    teacher_config,
                    landmark_xyz=teacher_landmark_xyz,
                    diagnostics=teacher_last_diagnostics,
                    dustbin_score=teacher_dustbin_score,
                    pair_scorer=teacher_pair_scorer,
                    pair_scorer_threshold=calibrated_pair_scorer_threshold,
                    pair_measurement_head=teacher_pair_measurement_head,
                    pair_measurement_threshold=(
                        calibrated_pair_measurement_threshold
                    ),
                    adaptive_trust_state=adaptive_trust_state_for_export,
                    normalize_features=normalize_exported_features,
                )
                save_sparse_candidate_teacher_state(
                    os.path.join(save_path, "candidate_teacher_state.pt"),
                    sampled_idx,
                    teacher_features_for_export,
                    iteration,
                    teacher_config,
                    landmark_xyz=teacher_landmark_xyz,
                    diagnostics=teacher_last_diagnostics,
                    dustbin_score=teacher_dustbin_score,
                    pair_scorer=teacher_pair_scorer,
                    pair_scorer_threshold=calibrated_pair_scorer_threshold,
                    pair_measurement_head=teacher_pair_measurement_head,
                    pair_measurement_threshold=(
                        calibrated_pair_measurement_threshold
                    ),
                    adaptive_trust_state=adaptive_trust_state_for_export,
                    normalize_features=normalize_exported_features,
                )

    if sparse_candidate_teacher:
        final_pair_weights = failure_guided_pair_weights(
            camera_failure_scores,
            temperature=candidate_teacher_online_render_failure_temperature,
            uniform_floor=candidate_teacher_online_render_uniform_floor,
        )
        if final_pair_weights.numel() > 0:
            pair_entropy = -(
                final_pair_weights * final_pair_weights.clamp_min(1e-12).log()
            ).sum()
            online_render_stats.update(
                {
                    "failure_mean_final": float(camera_failure_scores.mean().item()),
                    "failure_std_final": float(camera_failure_scores.std(unbiased=False).item()),
                    "failure_max_final": float(camera_failure_scores.max().item()),
                    "pair_weight_entropy_final": float(pair_entropy.item()),
                    "pair_weight_effective_count_final": float(pair_entropy.exp().item()),
                }
            )
        summary = {
            "version": 3,
            "iterations": int(train_iteration),
            "landmark_count": int(sampled_idx.numel()),
            "config": teacher_config,
            "final": {
                **(teacher_history[-1] if teacher_history else {}),
                **teacher_last_diagnostics,
            },
            "history": teacher_history,
            "validation_history": teacher_validation_history,
            "online_render_stats": online_render_stats,
            "hard_candidate_teacher_stats": (
                hard_candidate_teacher.diagnostics()
                if hard_candidate_teacher is not None
                else {}
            ),
        }
        summary_path = os.path.join(save_path, "candidate_teacher_training_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Saved sparse candidate teacher summary: {summary_path}")


def prepare_output_and_logger(args, folder=None):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if folder:
        output_path = os.path.join(args.model_path, folder)
    else:
        output_path = args.model_path
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def fill_missing_model_defaults(args):
    defaults = {
        "sh_degree": 3,
        "feature_type": "",
        "gaussian_type": "3dgs",
        "images": "images",
        "resolution": -1,
        "white_background": True,
        "longest_edge": 640,
        "data_device": "cuda",
        "eval": False,
        "speedup": False,
        "norm_before_render": True,
        "render_items": ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    }
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_arg_parser(with_components=False):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "localization_aware",
            "localization_aware_spatial",
            "localization_aware_global",
            "localization_aware_pnp",
            "coverage_preserving",
        ],
    )
    parser.add_argument("--utility_weight", type=float, default=1.0)
    parser.add_argument("--pnp_voxel_size", type=float, default=0.25)
    parser.add_argument("--pnp_max_per_voxel", type=int, default=8)
    parser.add_argument("--pnp_preserve_ratio", type=float, default=0.5)
    parser.add_argument("--min_loc_observations", type=int, default=1)
    parser.add_argument("--detector_target_mode", type=str, default="hard", choices=["hard", "soft", "weighted_hard"])
    parser.add_argument("--soft_sigma", type=float, default=1.5)
    parser.add_argument("--coverage_preserve_ratio", type=float, default=0.5)
    parser.add_argument("--coverage_utility_ratio", type=float, default=0.25)
    parser.add_argument("--coverage_high_confidence_ratio", type=float, default=0.0)
    parser.add_argument("--coverage_grid_size", type=int, default=0)
    parser.add_argument("--coverage_max_per_grid", type=int, default=0)
    parser.add_argument("--coverage_depth_bins", type=int, default=0)
    parser.add_argument("--coverage_max_per_depth_bin", type=int, default=0)
    parser.add_argument("--coverage_allow_unbalanced_fallback", action="store_true")
    parser.add_argument("--candidate_reprojection_error_scale", type=float, default=4.0)
    parser.add_argument("--candidate_cleanliness_weight", type=float, default=1.0)
    parser.add_argument("--candidate_pose_info_weight", type=float, default=1.0)
    parser.add_argument("--candidate_balance_weight", type=float, default=1.0)
    parser.add_argument("--candidate_reliability_weight", type=float, default=0.25)
    parser.add_argument("--candidate_utility_weight", type=float, default=0.0)
    parser.add_argument("--landmark_only", action="store_true", default=False)
    parser.add_argument("--precomputed_landmark_path", type=str, default="")
    parser.add_argument("--sparse_candidate_teacher", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_detector_init_path", type=str, default="")
    parser.add_argument("--candidate_teacher_state_init_path", type=str, default="")
    parser.add_argument("--candidate_teacher_pair_scorer_init_path", type=str, default="")
    parser.add_argument(
        "--candidate_teacher_pair_measurement_init_path", type=str, default=""
    )
    parser.add_argument("--candidate_teacher_optimize_features", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_freeze_detector", action="store_true", default=False)
    parser.add_argument(
        "--candidate_teacher_detector_only",
        action="store_true",
        default=False,
        help="Freeze all candidate-map-side parameters and train only the detector.",
    )
    parser.add_argument("--candidate_teacher_detector_lr", type=float, default=1e-4)
    parser.add_argument("--candidate_teacher_feature_lr", type=float, default=5e-5)
    parser.add_argument("--candidate_teacher_dustbin_lr", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_pair_scorer_lr", type=float, default=1e-3)
    parser.add_argument(
        "--candidate_teacher_pair_measurement_lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--candidate_teacher_pair_scorer_architecture",
        choices=["auto", "cosine_residual_v1", "descriptor_set_residual_v2"],
        default="auto",
    )
    parser.add_argument("--candidate_teacher_detect_num", type=int, default=2048)
    parser.add_argument("--candidate_teacher_nms_radius", type=int, default=2)
    parser.add_argument(
        "--candidate_teacher_query_feature_contract",
        choices=["legacy_full_then_resized_map", "native_resized_input"],
        default="legacy_full_then_resized_map",
        help="Sparse query encoder contract shared with LaFGS map training.",
    )
    parser.add_argument(
        "--candidate_teacher_match_mode",
        choices=["topk", "mnn"],
        default="topk",
    )
    parser.add_argument("--candidate_teacher_match_topk", type=int, default=1)
    parser.add_argument("--candidate_teacher_match_threshold", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_dual_softmax", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_dual_softmax_temperature", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_positive_radius_px", type=float, default=2.0)
    parser.add_argument("--candidate_teacher_negative_radius_px", type=float, default=6.0)
    parser.add_argument("--candidate_teacher_max_positives", type=int, default=4)
    parser.add_argument("--candidate_teacher_hard_negatives", type=int, default=8)
    parser.add_argument("--candidate_teacher_match_temperature", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_match_margin", type=float, default=0.5)
    parser.add_argument(
        "--candidate_teacher_assignment_mode",
        choices=["single_nearest", "multi_positive"],
        default="single_nearest",
    )
    parser.add_argument("--candidate_teacher_assignment_temperature", type=float, default=0.05)
    parser.add_argument("--candidate_teacher_assignment_margin", type=float, default=0.05)
    parser.add_argument(
        "--candidate_teacher_assignment_pose_information_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "point_jacobian",
            "full_set_leverage",
            "conditional_full",
            "conditional_translation",
        ],
    )
    parser.add_argument(
        "--candidate_teacher_assignment_pose_information_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_pose_information_floor",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_pose_information_normalization",
        type=str,
        default="quantile",
        choices=["max", "quantile", "rank"],
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_translation_scale",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_rotation_scale_degrees",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_measurement_sigma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_use_matchability",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_matchability_floor",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_matchability_power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate_teacher_assignment_fisher_uncertainty_entropy_scale",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_map_cleanliness_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_map_full_information_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_map_translation_information_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_map_translation_trace_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_map_translation_condition_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_map_bias_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_map_directional_bias_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_map_capacity_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_map_fisher_translation_scale", type=float, default=0.02
    )
    parser.add_argument(
        "--candidate_teacher_map_fisher_rotation_scale_degrees",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--candidate_teacher_map_fisher_measurement_sigma_px",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate_teacher_map_fisher_residual_clip_px", type=float, default=12.0
    )
    parser.add_argument(
        "--candidate_teacher_map_fisher_inlier_sigma_px", type=float, default=4.0
    )
    parser.add_argument(
        "--candidate_teacher_map_fisher_condition_target", type=float, default=100.0
    )
    parser.add_argument(
        "--candidate_teacher_map_bias_huber_delta", type=float, default=1.0
    )
    parser.add_argument(
        "--candidate_teacher_map_bias_clip", type=float, default=4.0
    )
    parser.add_argument(
        "--candidate_teacher_map_max_matches_per_landmark", type=int, default=0
    )
    parser.add_argument(
        "--candidate_teacher_map_directional_topk", type=int, default=0
    )
    parser.add_argument(
        "--candidate_teacher_map_directional_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate_teacher_map_directional_residual_clip_px",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--candidate_teacher_map_directional_robust_scale_px",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--candidate_teacher_map_directional_robust_quality_floor",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_bias_utility_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_translation_utility_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_utility_floor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_target_mode",
        choices=("all_false", "assignment_missed"),
        default="all_false",
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_require_current_retained",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_exact_decision_set",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_require_positive_bias_gain",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_counterfactual_require_nonnegative_translation_gain",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_refresh_visits",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_solver",
        choices=["poselib", "opencv"],
        default="poselib",
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_reprojection_error",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_ransac_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_min_inliers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_max_pose_error_cm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_max_useful", type=int, default=96
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_max_harmful", type=int, default=96
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_margin", type=float, default=0.05
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_harmful_mode",
        choices=["all_false", "translation_delete", "exact_pose_delete"],
        default="all_false",
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_exact_replay_max_candidates",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_grid_rows", type=int, default=4)
    parser.add_argument("--candidate_teacher_grid_cols", type=int, default=4)
    parser.add_argument("--candidate_teacher_depth_bins", type=int, default=4)
    parser.add_argument("--candidate_teacher_pair_weight", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_hard_negative_weight", type=float, default=0.5)
    parser.add_argument("--candidate_teacher_assignment_weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate_teacher_counterfactual_assignment_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_dustbin_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_matcher_assignment_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_matcher_reprojection_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_reprojection_sigma_px", type=float, default=1.0
    )
    parser.add_argument("--candidate_teacher_dustbin_init", type=float, default=0.5)
    parser.add_argument("--candidate_teacher_pair_scorer_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_pair_scorer_assignment_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_inlier_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_nll_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_bias_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_covariance_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_residual_clip_px",
        type=float,
        default=32.0,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_reference_translation_m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--candidate_teacher_matcher_translation_info_weight", type=float, default=0.0
    )
    parser.add_argument("--candidate_teacher_translation_info_weight", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_pair_scorer_hidden_dim", type=int, default=16)
    parser.add_argument(
        "--candidate_teacher_pair_measurement_hidden_dim", type=int, default=64
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_patch_radius", type=int, default=2
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_max_offset", type=float, default=2.0
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_covariance_floor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_set_context",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_pair_measurement_geometry_context",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_teacher_freeze_pair_measurement",
        action="store_true",
    )
    parser.add_argument("--candidate_teacher_pair_context_topk", type=int, default=8)
    parser.add_argument("--candidate_teacher_scorer_min_recall", type=float, default=0.75)
    parser.add_argument(
        "--candidate_teacher_scorer_max_matches_per_keypoint", type=int, default=1
    )
    parser.add_argument("--candidate_teacher_matchability_head", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_matchability_only", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_offset_head", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_offset_only", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_max_offset", type=float, default=2.0)
    parser.add_argument(
        "--candidate_teacher_offset_target_source",
        choices=["geometric_nearest", "matched_top1"],
        default="geometric_nearest",
    )
    parser.add_argument(
        "--candidate_teacher_selection_source",
        choices=["combined", "keypoint_teacher"],
        default="combined",
    )
    parser.add_argument(
        "--candidate_teacher_detector_target_source",
        choices=[
            "geometric",
            "predicted_correct",
            "scorer_accepted_correct",
            "measurement_accepted_correct",
            "final_accepted_correct",
            "final_or_geometric",
        ],
        default="geometric",
    )
    parser.add_argument("--candidate_teacher_detector_binary_target", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_detector_match_weight", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_detector_offset_weight", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_geometry_weight", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_coverage_weight", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_base_detector_weight", type=float, default=0.1)
    parser.add_argument(
        "--candidate_teacher_detector_preservation_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--candidate_teacher_feature_anchor_weight", type=float, default=0.01)
    parser.add_argument("--candidate_teacher_adaptive_trust", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_trust_alpha_min", type=float, default=0.25)
    parser.add_argument("--candidate_teacher_trust_view_prior", type=float, default=3.0)
    parser.add_argument("--candidate_teacher_trust_warmup_passes", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_support_query_split", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_query_ratio", type=float, default=0.2)
    parser.add_argument("--candidate_teacher_validation_ratio", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_split_mode",
        choices=[
            "random",
            "sequence_block",
            "temporal_block",
            "stratified_temporal_block",
        ],
        default="temporal_block",
    )
    parser.add_argument("--candidate_teacher_split_seed", type=int, default=2026)
    parser.add_argument(
        "--candidate_teacher_online_render_ratio_start", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_online_render_ratio_end", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_online_render_ramp_start", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_online_render_ramp_end", type=float, default=1.0
    )
    parser.add_argument(
        "--candidate_teacher_online_render_alpha_min", type=float, default=0.35
    )
    parser.add_argument(
        "--candidate_teacher_online_render_alpha_max", type=float, default=0.65
    )
    parser.add_argument(
        "--candidate_teacher_online_render_provenance_mode",
        choices=["none", "hard", "soft"],
        default="none",
    )
    parser.add_argument(
        "--candidate_teacher_online_render_provenance_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate_teacher_online_render_provenance_topk", type=int, default=4
    )
    parser.add_argument(
        "--candidate_teacher_online_render_provenance_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate_teacher_online_render_sampling_mode",
        choices=["uniform", "failure_guided"],
        default="uniform",
    )
    parser.add_argument(
        "--candidate_teacher_online_render_failure_ema", type=float, default=0.9
    )
    parser.add_argument(
        "--candidate_teacher_online_render_failure_temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--candidate_teacher_online_render_uniform_floor", type=float, default=0.1
    )
    if with_components:
        return parser, lp, op
    return parser


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser, lp, op = build_arg_parser(with_components=True)
    args = get_combined_args(parser)
    fill_missing_model_defaults(args)
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    write_detector_reproducibility_manifest(dataset, args)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs
        gaussians = GaussianModel_2dgs(dataset.sh_degree)

    masks = None
    for mask_path in (
        os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        os.path.join(dataset.source_path, "masks.pkl"),
    ):
        if os.path.exists(mask_path):
            import pickle
            masks = pickle.load(open(mask_path, "rb"))
            break

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        # Candidate-teacher direct holdouts are defined on lexical camera
        # order, matching train_lafgs_map.py and stdloc.py.  Conventional
        # detector training retains the historical shuffled order.
        shuffle=not bool(args.sparse_candidate_teacher),
        load_test_cameras=training_requires_test_cameras(
            args.test_iterations,
            args.iterations,
        ),
    )

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(dataset.model_path, args.detector_folder)
        )
    else:
        tb_writer = None

    training_detector(
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=args.iterations,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
        sampling_mode=args.sampling_mode,
        utility_weight=args.utility_weight,
        pnp_voxel_size=args.pnp_voxel_size,
        pnp_max_per_voxel=args.pnp_max_per_voxel,
        pnp_preserve_ratio=args.pnp_preserve_ratio,
        min_loc_observations=args.min_loc_observations,
        detector_target_mode=args.detector_target_mode,
        soft_sigma=args.soft_sigma,
        coverage_preserve_ratio=args.coverage_preserve_ratio,
        coverage_utility_ratio=args.coverage_utility_ratio,
        coverage_high_confidence_ratio=args.coverage_high_confidence_ratio,
        coverage_grid_size=args.coverage_grid_size,
        coverage_max_per_grid=args.coverage_max_per_grid,
        coverage_depth_bins=args.coverage_depth_bins,
        coverage_max_per_depth_bin=args.coverage_max_per_depth_bin,
        coverage_allow_unbalanced_fallback=args.coverage_allow_unbalanced_fallback,
        candidate_reprojection_error_scale=args.candidate_reprojection_error_scale,
        candidate_cleanliness_weight=args.candidate_cleanliness_weight,
        candidate_pose_info_weight=args.candidate_pose_info_weight,
        candidate_balance_weight=args.candidate_balance_weight,
        candidate_reliability_weight=args.candidate_reliability_weight,
        candidate_utility_weight=args.candidate_utility_weight,
        landmark_only=args.landmark_only,
        precomputed_landmark_path=args.precomputed_landmark_path,
        sparse_candidate_teacher=args.sparse_candidate_teacher,
        candidate_teacher_detector_init_path=args.candidate_teacher_detector_init_path,
        candidate_teacher_state_init_path=args.candidate_teacher_state_init_path,
        candidate_teacher_pair_scorer_init_path=(
            args.candidate_teacher_pair_scorer_init_path
        ),
        candidate_teacher_pair_measurement_init_path=(
            args.candidate_teacher_pair_measurement_init_path
        ),
        candidate_teacher_optimize_features=args.candidate_teacher_optimize_features,
        candidate_teacher_freeze_detector=args.candidate_teacher_freeze_detector,
        candidate_teacher_detector_only=args.candidate_teacher_detector_only,
        candidate_teacher_detector_lr=args.candidate_teacher_detector_lr,
        candidate_teacher_feature_lr=args.candidate_teacher_feature_lr,
        candidate_teacher_dustbin_lr=args.candidate_teacher_dustbin_lr,
        candidate_teacher_pair_scorer_lr=args.candidate_teacher_pair_scorer_lr,
        candidate_teacher_pair_measurement_lr=(
            args.candidate_teacher_pair_measurement_lr
        ),
        candidate_teacher_pair_scorer_architecture=(
            args.candidate_teacher_pair_scorer_architecture
        ),
        candidate_teacher_detect_num=args.candidate_teacher_detect_num,
        candidate_teacher_nms_radius=args.candidate_teacher_nms_radius,
        candidate_teacher_query_feature_contract=(
            args.candidate_teacher_query_feature_contract
        ),
        candidate_teacher_match_mode=args.candidate_teacher_match_mode,
        candidate_teacher_match_topk=args.candidate_teacher_match_topk,
        candidate_teacher_match_threshold=args.candidate_teacher_match_threshold,
        candidate_teacher_dual_softmax=args.candidate_teacher_dual_softmax,
        candidate_teacher_dual_softmax_temperature=args.candidate_teacher_dual_softmax_temperature,
        candidate_teacher_positive_radius_px=args.candidate_teacher_positive_radius_px,
        candidate_teacher_negative_radius_px=args.candidate_teacher_negative_radius_px,
        candidate_teacher_max_positives=args.candidate_teacher_max_positives,
        candidate_teacher_hard_negatives=args.candidate_teacher_hard_negatives,
        candidate_teacher_match_temperature=args.candidate_teacher_match_temperature,
        candidate_teacher_match_margin=args.candidate_teacher_match_margin,
        candidate_teacher_assignment_mode=args.candidate_teacher_assignment_mode,
        candidate_teacher_assignment_temperature=args.candidate_teacher_assignment_temperature,
        candidate_teacher_assignment_margin=args.candidate_teacher_assignment_margin,
        candidate_teacher_assignment_pose_information_mode=(
            args.candidate_teacher_assignment_pose_information_mode
        ),
        candidate_teacher_assignment_pose_information_weight=(
            args.candidate_teacher_assignment_pose_information_weight
        ),
        candidate_teacher_assignment_pose_information_floor=(
            args.candidate_teacher_assignment_pose_information_floor
        ),
        candidate_teacher_assignment_pose_information_normalization=(
            args.candidate_teacher_assignment_pose_information_normalization
        ),
        candidate_teacher_assignment_fisher_translation_scale=(
            args.candidate_teacher_assignment_fisher_translation_scale
        ),
        candidate_teacher_assignment_fisher_rotation_scale_degrees=(
            args.candidate_teacher_assignment_fisher_rotation_scale_degrees
        ),
        candidate_teacher_assignment_fisher_measurement_sigma=(
            args.candidate_teacher_assignment_fisher_measurement_sigma
        ),
        candidate_teacher_assignment_fisher_use_matchability=(
            args.candidate_teacher_assignment_fisher_use_matchability
        ),
        candidate_teacher_assignment_fisher_matchability_floor=(
            args.candidate_teacher_assignment_fisher_matchability_floor
        ),
        candidate_teacher_assignment_fisher_matchability_power=(
            args.candidate_teacher_assignment_fisher_matchability_power
        ),
        candidate_teacher_assignment_fisher_uncertainty_entropy_scale=(
            args.candidate_teacher_assignment_fisher_uncertainty_entropy_scale
        ),
        candidate_teacher_map_cleanliness_weight=(
            args.candidate_teacher_map_cleanliness_weight
        ),
        candidate_teacher_map_full_information_weight=(
            args.candidate_teacher_map_full_information_weight
        ),
        candidate_teacher_map_translation_information_weight=(
            args.candidate_teacher_map_translation_information_weight
        ),
        candidate_teacher_map_translation_trace_weight=(
            args.candidate_teacher_map_translation_trace_weight
        ),
        candidate_teacher_map_translation_condition_weight=(
            args.candidate_teacher_map_translation_condition_weight
        ),
        candidate_teacher_map_bias_weight=args.candidate_teacher_map_bias_weight,
        candidate_teacher_map_directional_bias_weight=(
            args.candidate_teacher_map_directional_bias_weight
        ),
        candidate_teacher_map_capacity_weight=(
            args.candidate_teacher_map_capacity_weight
        ),
        candidate_teacher_map_fisher_translation_scale=(
            args.candidate_teacher_map_fisher_translation_scale
        ),
        candidate_teacher_map_fisher_rotation_scale_degrees=(
            args.candidate_teacher_map_fisher_rotation_scale_degrees
        ),
        candidate_teacher_map_fisher_measurement_sigma_px=(
            args.candidate_teacher_map_fisher_measurement_sigma_px
        ),
        candidate_teacher_map_fisher_residual_clip_px=(
            args.candidate_teacher_map_fisher_residual_clip_px
        ),
        candidate_teacher_map_fisher_inlier_sigma_px=(
            args.candidate_teacher_map_fisher_inlier_sigma_px
        ),
        candidate_teacher_map_fisher_condition_target=(
            args.candidate_teacher_map_fisher_condition_target
        ),
        candidate_teacher_map_bias_huber_delta=(
            args.candidate_teacher_map_bias_huber_delta
        ),
        candidate_teacher_map_bias_clip=args.candidate_teacher_map_bias_clip,
        candidate_teacher_map_max_matches_per_landmark=(
            args.candidate_teacher_map_max_matches_per_landmark
        ),
        candidate_teacher_map_directional_topk=(
            args.candidate_teacher_map_directional_topk
        ),
        candidate_teacher_map_directional_temperature=(
            args.candidate_teacher_map_directional_temperature
        ),
        candidate_teacher_map_directional_residual_clip_px=(
            args.candidate_teacher_map_directional_residual_clip_px
        ),
        candidate_teacher_map_directional_robust_scale_px=(
            args.candidate_teacher_map_directional_robust_scale_px
        ),
        candidate_teacher_map_directional_robust_quality_floor=(
            args.candidate_teacher_map_directional_robust_quality_floor
        ),
        candidate_teacher_counterfactual_bias_utility_weight=(
            args.candidate_teacher_counterfactual_bias_utility_weight
        ),
        candidate_teacher_counterfactual_translation_utility_weight=(
            args.candidate_teacher_counterfactual_translation_utility_weight
        ),
        candidate_teacher_counterfactual_utility_floor=(
            args.candidate_teacher_counterfactual_utility_floor
        ),
        candidate_teacher_counterfactual_target_mode=(
            args.candidate_teacher_counterfactual_target_mode
        ),
        candidate_teacher_counterfactual_require_current_retained=(
            args.candidate_teacher_counterfactual_require_current_retained
        ),
        candidate_teacher_counterfactual_exact_decision_set=(
            args.candidate_teacher_counterfactual_exact_decision_set
        ),
        candidate_teacher_counterfactual_require_positive_bias_gain=(
            args.candidate_teacher_counterfactual_require_positive_bias_gain
        ),
        candidate_teacher_counterfactual_require_nonnegative_translation_gain=(
            args.candidate_teacher_counterfactual_require_nonnegative_translation_gain
        ),
        candidate_teacher_hard_preservation_weight=(
            args.candidate_teacher_hard_preservation_weight
        ),
        candidate_teacher_hard_preservation_refresh_visits=(
            args.candidate_teacher_hard_preservation_refresh_visits
        ),
        candidate_teacher_hard_preservation_solver=(
            args.candidate_teacher_hard_preservation_solver
        ),
        candidate_teacher_hard_preservation_reprojection_error=(
            args.candidate_teacher_hard_preservation_reprojection_error
        ),
        candidate_teacher_hard_preservation_ransac_seed=(
            args.candidate_teacher_hard_preservation_ransac_seed
        ),
        candidate_teacher_hard_preservation_min_inliers=(
            args.candidate_teacher_hard_preservation_min_inliers
        ),
        candidate_teacher_hard_preservation_max_pose_error_cm=(
            args.candidate_teacher_hard_preservation_max_pose_error_cm
        ),
        candidate_teacher_hard_preservation_max_useful=(
            args.candidate_teacher_hard_preservation_max_useful
        ),
        candidate_teacher_hard_preservation_max_harmful=(
            args.candidate_teacher_hard_preservation_max_harmful
        ),
        candidate_teacher_hard_preservation_temperature=(
            args.candidate_teacher_hard_preservation_temperature
        ),
        candidate_teacher_hard_preservation_margin=(
            args.candidate_teacher_hard_preservation_margin
        ),
        candidate_teacher_hard_preservation_harmful_mode=(
            args.candidate_teacher_hard_preservation_harmful_mode
        ),
        candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m=(
            args.candidate_teacher_hard_preservation_harmful_min_translation_delete_gain_m
        ),
        candidate_teacher_hard_preservation_exact_replay_max_candidates=(
            args.candidate_teacher_hard_preservation_exact_replay_max_candidates
        ),
        candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm=(
            args.candidate_teacher_hard_preservation_exact_replay_min_pose_gain_cm
        ),
        candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree=(
            args.candidate_teacher_hard_preservation_exact_replay_rotation_weight_cm_per_degree
        ),
        candidate_teacher_grid_rows=args.candidate_teacher_grid_rows,
        candidate_teacher_grid_cols=args.candidate_teacher_grid_cols,
        candidate_teacher_depth_bins=args.candidate_teacher_depth_bins,
        candidate_teacher_pair_weight=args.candidate_teacher_pair_weight,
        candidate_teacher_hard_negative_weight=args.candidate_teacher_hard_negative_weight,
        candidate_teacher_assignment_weight=args.candidate_teacher_assignment_weight,
        candidate_teacher_counterfactual_assignment_weight=(
            args.candidate_teacher_counterfactual_assignment_weight
        ),
        candidate_teacher_dustbin_weight=args.candidate_teacher_dustbin_weight,
        candidate_teacher_matcher_assignment_weight=(
            args.candidate_teacher_matcher_assignment_weight
        ),
        candidate_teacher_matcher_reprojection_weight=(
            args.candidate_teacher_matcher_reprojection_weight
        ),
        candidate_teacher_reprojection_sigma_px=(
            args.candidate_teacher_reprojection_sigma_px
        ),
        candidate_teacher_dustbin_init=args.candidate_teacher_dustbin_init,
        candidate_teacher_pair_scorer_weight=args.candidate_teacher_pair_scorer_weight,
        candidate_teacher_pair_scorer_assignment_weight=(
            args.candidate_teacher_pair_scorer_assignment_weight
        ),
        candidate_teacher_pair_measurement_inlier_weight=(
            args.candidate_teacher_pair_measurement_inlier_weight
        ),
        candidate_teacher_pair_measurement_nll_weight=(
            args.candidate_teacher_pair_measurement_nll_weight
        ),
        candidate_teacher_pair_measurement_bias_weight=(
            args.candidate_teacher_pair_measurement_bias_weight
        ),
        candidate_teacher_pair_measurement_covariance_weight=(
            args.candidate_teacher_pair_measurement_covariance_weight
        ),
        candidate_teacher_pair_measurement_residual_clip_px=(
            args.candidate_teacher_pair_measurement_residual_clip_px
        ),
        candidate_teacher_pair_measurement_reference_translation_m=(
            args.candidate_teacher_pair_measurement_reference_translation_m
        ),
        candidate_teacher_matcher_translation_info_weight=(
            args.candidate_teacher_matcher_translation_info_weight
        ),
        candidate_teacher_translation_info_weight=args.candidate_teacher_translation_info_weight,
        candidate_teacher_pair_scorer_hidden_dim=args.candidate_teacher_pair_scorer_hidden_dim,
        candidate_teacher_pair_measurement_hidden_dim=(
            args.candidate_teacher_pair_measurement_hidden_dim
        ),
        candidate_teacher_pair_measurement_patch_radius=(
            args.candidate_teacher_pair_measurement_patch_radius
        ),
        candidate_teacher_pair_measurement_max_offset=(
            args.candidate_teacher_pair_measurement_max_offset
        ),
        candidate_teacher_pair_measurement_covariance_floor=(
            args.candidate_teacher_pair_measurement_covariance_floor
        ),
        candidate_teacher_pair_measurement_set_context=(
            args.candidate_teacher_pair_measurement_set_context
        ),
        candidate_teacher_pair_measurement_geometry_context=(
            args.candidate_teacher_pair_measurement_geometry_context
        ),
        candidate_teacher_freeze_pair_measurement=(
            args.candidate_teacher_freeze_pair_measurement
        ),
        candidate_teacher_pair_context_topk=args.candidate_teacher_pair_context_topk,
        candidate_teacher_scorer_min_recall=args.candidate_teacher_scorer_min_recall,
        candidate_teacher_scorer_max_matches_per_keypoint=(
            args.candidate_teacher_scorer_max_matches_per_keypoint
        ),
        candidate_teacher_matchability_head=args.candidate_teacher_matchability_head,
        candidate_teacher_matchability_only=args.candidate_teacher_matchability_only,
        candidate_teacher_offset_head=args.candidate_teacher_offset_head,
        candidate_teacher_offset_only=args.candidate_teacher_offset_only,
        candidate_teacher_max_offset=args.candidate_teacher_max_offset,
        candidate_teacher_offset_target_source=(
            args.candidate_teacher_offset_target_source
        ),
        candidate_teacher_selection_source=args.candidate_teacher_selection_source,
        candidate_teacher_detector_target_source=args.candidate_teacher_detector_target_source,
        candidate_teacher_detector_binary_target=args.candidate_teacher_detector_binary_target,
        candidate_teacher_detector_match_weight=args.candidate_teacher_detector_match_weight,
        candidate_teacher_detector_offset_weight=(
            args.candidate_teacher_detector_offset_weight
        ),
        candidate_teacher_geometry_weight=args.candidate_teacher_geometry_weight,
        candidate_teacher_coverage_weight=args.candidate_teacher_coverage_weight,
        candidate_teacher_base_detector_weight=args.candidate_teacher_base_detector_weight,
        candidate_teacher_detector_preservation_weight=(
            args.candidate_teacher_detector_preservation_weight
        ),
        candidate_teacher_feature_anchor_weight=args.candidate_teacher_feature_anchor_weight,
        candidate_teacher_adaptive_trust=args.candidate_teacher_adaptive_trust,
        candidate_teacher_trust_alpha_min=args.candidate_teacher_trust_alpha_min,
        candidate_teacher_trust_view_prior=args.candidate_teacher_trust_view_prior,
        candidate_teacher_trust_warmup_passes=(
            args.candidate_teacher_trust_warmup_passes
        ),
        candidate_teacher_support_query_split=args.candidate_teacher_support_query_split,
        candidate_teacher_query_ratio=args.candidate_teacher_query_ratio,
        candidate_teacher_validation_ratio=args.candidate_teacher_validation_ratio,
        candidate_teacher_split_mode=args.candidate_teacher_split_mode,
        candidate_teacher_split_seed=args.candidate_teacher_split_seed,
        candidate_teacher_online_render_ratio_start=(
            args.candidate_teacher_online_render_ratio_start
        ),
        candidate_teacher_online_render_ratio_end=(
            args.candidate_teacher_online_render_ratio_end
        ),
        candidate_teacher_online_render_ramp_start=(
            args.candidate_teacher_online_render_ramp_start
        ),
        candidate_teacher_online_render_ramp_end=(
            args.candidate_teacher_online_render_ramp_end
        ),
        candidate_teacher_online_render_alpha_min=(
            args.candidate_teacher_online_render_alpha_min
        ),
        candidate_teacher_online_render_alpha_max=(
            args.candidate_teacher_online_render_alpha_max
        ),
        candidate_teacher_online_render_provenance_mode=(
            args.candidate_teacher_online_render_provenance_mode
        ),
        candidate_teacher_online_render_provenance_weight=(
            args.candidate_teacher_online_render_provenance_weight
        ),
        candidate_teacher_online_render_provenance_topk=(
            args.candidate_teacher_online_render_provenance_topk
        ),
        candidate_teacher_online_render_provenance_temperature=(
            args.candidate_teacher_online_render_provenance_temperature
        ),
        candidate_teacher_online_render_sampling_mode=(
            args.candidate_teacher_online_render_sampling_mode
        ),
        candidate_teacher_online_render_failure_ema=(
            args.candidate_teacher_online_render_failure_ema
        ),
        candidate_teacher_online_render_failure_temperature=(
            args.candidate_teacher_online_render_failure_temperature
        ),
        candidate_teacher_online_render_uniform_floor=(
            args.candidate_teacher_online_render_uniform_floor
        ),
    )

    # All done
    print("\n Scene-specific detector training complete.")
