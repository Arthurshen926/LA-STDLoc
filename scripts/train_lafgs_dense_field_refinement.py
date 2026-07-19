#!/usr/bin/env python
"""Train a frozen-geometry rendered LaFGS dense feature field.

This is a standalone dense-refinement experiment, not a replacement for the
LaFGS V2 sparse candidate map.  It starts from a geometry-only support set
whose descriptors were initialized from real multi-view image features, then
optimizes the *composited* feature render against frozen real-query features.

The important distinction from the sparse map trainer is that supervision is
applied after 2DGS splatting.  A feature vector can no longer be good only as
an isolated landmark: its contribution must make the rendered field locally
peak at the geometrically correct query pixel under small pose perturbations.
RGB, 2DGS geometry, and feature opacity support are frozen throughout.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, get_combined_args
from localization_training.dense_teacher import dense_localization_teacher
from localization_training.pose_refiner import se3_exp
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.general_utils import safe_state, seed_everything


def _file_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _make_gaussians(dataset):
    gaussian_type = str(dataset.gaussian_type).lower()
    if gaussian_type == "2dgs":
        return GaussianModel_2dgs(dataset.sh_degree)
    if gaussian_type == "3dgs":
        return GaussianModel(dataset.sh_degree)
    raise ValueError(f"Unsupported gaussian type: {dataset.gaussian_type}")


def _load_indexed_state(path, point_count, expected_dim):
    path = Path(path)
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid dense-field state: {path}")
    indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)
    features = torch.as_tensor(state.get("landmark_features"), dtype=torch.float32)
    if indices.numel() == 0 or features.ndim < 2:
        raise ValueError(f"State has no indexed descriptors: {path}")
    features = features.reshape(indices.numel(), -1)
    if features.shape[1] != int(expected_dim):
        raise ValueError(
            f"Descriptor dim mismatch for {path}: {features.shape[1]} != {expected_dim}"
        )
    if int(indices.min()) < 0 or int(indices.max()) >= int(point_count):
        raise ValueError(f"State indices are outside frozen map range: {path}")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError(f"State contains duplicate primitive indices: {path}")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"State descriptors are non-finite: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "state": state,
        "indices": indices,
        "features": F.normalize(features, p=2, dim=1),
    }


def _merge_sparse_initialization(support, sparse, point_count):
    """Use sparse-map descriptors where they overlap dense support IDs."""
    features = support["features"].clone()
    if sparse is None:
        return features, 0
    lookup = torch.full((int(point_count),), -1, dtype=torch.long)
    lookup[support["indices"]] = torch.arange(support["indices"].numel())
    positions = lookup[sparse["indices"]]
    matched = positions >= 0
    if bool(matched.any().item()):
        features[positions[matched]] = sparse["features"][matched]
    return F.normalize(features, p=2, dim=1), int(matched.sum().item())


def _install_dense_support(gaussians, indices, initial_features, non_support_logit):
    """Keep RGB geometry intact while enabling only selected loc primitives."""
    point_count = int(gaussians.get_xyz.shape[0])
    device_indices = indices.to(device=gaussians._loc_feature.device, dtype=torch.long)
    with torch.no_grad():
        gaussians._loc_feature.data[device_indices] = initial_features.to(
            device=gaussians._loc_feature.device,
            dtype=gaussians._loc_feature.dtype,
        ).reshape(device_indices.numel(), *gaussians._loc_feature.shape[1:])
        loc_opacity = torch.full_like(
            gaussians._opacity.detach(), float(non_support_logit)
        )
        loc_opacity[device_indices] = gaussians._opacity.detach()[device_indices]
    gaussians._loc_opacity = torch.nn.Parameter(loc_opacity, requires_grad=False)
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)
    gaussians._loc_feature.requires_grad_(True)
    if int(gaussians._loc_feature.shape[0]) != point_count:
        raise RuntimeError("Frozen map loc-feature cardinality changed unexpectedly")
    return device_indices


def _cache_queries(path, dataset, iteration, feature_dim):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), dict):
        raise ValueError("Query cache must contain a 'queries' dictionary")
    signature = payload.get("signature_payload", {})
    checks = {
        "model_path": os.path.abspath(dataset.model_path),
        "source_path": os.path.abspath(dataset.source_path),
        "load_iteration": int(iteration),
    }
    for key, expected in checks.items():
        actual = signature.get(key)
        if actual is None:
            continue
        if key.endswith("_path"):
            actual = os.path.abspath(str(actual))
        if actual != expected:
            raise ValueError(
                f"Query cache {key} mismatch: cache={actual!r} expected={expected!r}"
            )
    for name, entry in payload["queries"].items():
        feature_map = torch.as_tensor(entry.get("feature_map"))
        if feature_map.ndim != 3 or int(feature_map.shape[0]) != int(feature_dim):
            raise ValueError(f"Invalid cached feature map for {name!r}")
        for key in ("pose_w2c", "valid_mask"):
            if key not in entry:
                raise ValueError(f"Cached query {name!r} is missing {key}")
    return payload["queries"], payload.get("signature"), signature


def _excluded_query_names(path):
    """Read a saved results list whose cameras must not train the field."""
    if not path:
        return set(), None
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("exclude_query_results must be a results.json list")
    names = set()
    for record in payload:
        if not isinstance(record, dict) or not record.get("image_name"):
            raise ValueError("every excluded result must contain image_name")
        names.add(str(record["image_name"]).replace("\\", "/"))
    if not names:
        raise ValueError("exclude_query_results is empty")
    return names, _file_sha256(path)


def _normalize_image_name(value):
    return str(value).replace("\\", "/")


def _load_sparse_seed_poses(path, *, allowed_names, excluded_names):
    """Load frozen sparse pose estimates for seed-conditioned episodes.

    The dense field is trained against the same kind of initial pose it will
    receive at inference.  The result artifact is intentionally kept outside
    the query cache so that its source, coverage, and holdout relationship can
    be recorded and audited independently.
    """
    if not path:
        return {}, {
            "path": None,
            "sha256": None,
            "result_count": 0,
            "train_seed_count": 0,
            "ignored_result_count": 0,
        }
    path = Path(path)
    with path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("sparse_seed_results must be a non-empty results.json list")

    allowed_names = {_normalize_image_name(name) for name in allowed_names}
    excluded_names = {_normalize_image_name(name) for name in excluded_names}
    poses = {}
    result_names = set()
    for record in records:
        if not isinstance(record, dict) or not record.get("image_name"):
            raise ValueError("every sparse seed result must contain image_name")
        name = _normalize_image_name(record["image_name"])
        if name in result_names:
            raise ValueError(f"sparse_seed_results contains duplicate image {name!r}")
        result_names.add(name)
        sparse = record.get("sparse")
        pose = sparse.get("pose_w2c") if isinstance(sparse, dict) else None
        if pose is None:
            pose = record.get("sparse_pose_w2c", record.get("pose_w2c"))
        if pose is None:
            raise ValueError(f"sparse seed result {name!r} has no sparse pose_w2c")
        pose = torch.as_tensor(pose, dtype=torch.float32)
        if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all().item()):
            raise ValueError(f"sparse seed result {name!r} has an invalid pose_w2c")
        if name in allowed_names:
            poses[name] = pose.contiguous()

    leakage = sorted(result_names.intersection(excluded_names))
    if leakage:
        raise ValueError(
            "sparse_seed_results overlaps excluded holdout queries: "
            f"{leakage[:3]}"
        )
    return poses, {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "result_count": int(len(records)),
        "train_seed_count": int(len(poses)),
        "ignored_result_count": int(len(result_names.difference(allowed_names))),
    }


def _pose_seed_error(pose_w2c, pose_gt_w2c):
    """Return camera-center translation and geodesic rotation error."""
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64).reshape(4, 4)
    target = torch.as_tensor(pose_gt_w2c, dtype=torch.float64).reshape(4, 4)
    pose_c2w = torch.linalg.inv(pose)
    target_c2w = torch.linalg.inv(target)
    translation = torch.linalg.vector_norm(pose_c2w[:3, 3] - target_c2w[:3, 3])
    relative = pose[:3, :3] @ target[:3, :3].T
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation = torch.rad2deg(torch.acos(cosine))
    return float(translation.item()), float(rotation.item())


def _filter_sparse_seed_poses(seed_poses, cache, *, max_translation_m, max_rotation_deg):
    """Keep only real seed poses inside an explicitly declared training basin."""
    filtered = {}
    translations = []
    rotations = []
    rejected = 0
    for name, pose in seed_poses.items():
        if name not in cache:
            continue
        translation, rotation = _pose_seed_error(pose, cache[name]["pose_w2c"])
        is_too_far = (
            (float(max_translation_m) > 0.0 and translation > float(max_translation_m))
            or (float(max_rotation_deg) > 0.0 and rotation > float(max_rotation_deg))
        )
        if is_too_far:
            rejected += 1
            continue
        filtered[name] = pose
        translations.append(translation)
        rotations.append(rotation)

    def _summary(values, prefix):
        if not values:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_median": None,
                f"{prefix}_p95": None,
            }
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            f"{prefix}_mean": float(tensor.mean().item()),
            f"{prefix}_median": float(tensor.median().item()),
            f"{prefix}_p95": float(torch.quantile(tensor, 0.95).item()),
        }

    return filtered, {
        "eligible_count": int(len(filtered)),
        "rejected_outside_basin_count": int(rejected),
        **_summary(translations, "translation_m"),
        **_summary(rotations, "rotation_deg"),
    }


def _pose_perturbation(
    pose_gt_w2c,
    *,
    translation_max_m,
    rotation_max_deg,
    exact_probability,
    generator,
):
    """Sample a bounded camera-frame left perturbation around GT pose."""
    if float(exact_probability) > 0.0:
        keep_exact = torch.rand((), device=pose_gt_w2c.device, generator=generator)
        if float(keep_exact.item()) < float(exact_probability):
            return pose_gt_w2c

    def direction():
        value = torch.randn(3, device=pose_gt_w2c.device, generator=generator)
        return value / torch.linalg.norm(value).clamp_min(1e-8)

    translation = direction() * torch.rand(
        (), device=pose_gt_w2c.device, generator=generator
    ) * float(max(translation_max_m, 0.0))
    axis = direction()
    angle = (
        (2.0 * torch.rand((), device=pose_gt_w2c.device, generator=generator) - 1.0)
        * math.radians(float(max(rotation_max_deg, 0.0)))
    )
    twist = torch.cat([translation, axis * angle])
    return se3_exp(twist) @ pose_gt_w2c


def _curriculum(step, args):
    progress = min(max(float(step) / max(int(args.jitter_warmup_steps), 1), 0.0), 1.0)
    translation = (
        float(args.jitter_translation_start_m)
        + progress
        * (float(args.jitter_translation_end_m) - float(args.jitter_translation_start_m))
    )
    rotation = (
        float(args.jitter_rotation_start_deg)
        + progress
        * (float(args.jitter_rotation_end_deg) - float(args.jitter_rotation_start_deg))
    )
    return max(translation, 0.0), max(rotation, 0.0)


@torch.no_grad()
def _manual_adam_step(
    parameter,
    indices,
    exp_avg,
    exp_avg_sq,
    *,
    step,
    lr,
    beta1,
    beta2,
    eps,
    gradient_clip_norm,
):
    if parameter.grad is None:
        return False, 0.0
    gradient = parameter.grad[indices].reshape_as(exp_avg)
    if not bool(torch.isfinite(gradient).all().item()):
        return False, float("nan")
    grad_norm = torch.linalg.vector_norm(gradient)
    if float(gradient_clip_norm) > 0.0 and float(grad_norm.item()) > float(gradient_clip_norm):
        gradient = gradient * (float(gradient_clip_norm) / grad_norm.clamp_min(1e-12))
    exp_avg.mul_(float(beta1)).add_(gradient, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(gradient, gradient, value=1.0 - float(beta2))
    bias_1 = 1.0 - float(beta1) ** int(step)
    bias_2 = 1.0 - float(beta2) ** int(step)
    update = (exp_avg / bias_1) / ((exp_avg_sq / bias_2).sqrt() + float(eps))
    current = parameter.data[indices].reshape_as(exp_avg)
    updated = F.normalize(current - float(lr) * update, p=2, dim=1)
    parameter.data[indices] = updated.reshape(
        indices.numel(), *parameter.shape[1:]
    )
    return True, float(grad_norm.item())


def _recent_mean(records, count):
    selected = records[-max(int(count), 1) :]
    keys = sorted({key for record in selected for key in record})
    out = {}
    for key in keys:
        values = [record[key] for record in selected if isinstance(record.get(key), (int, float))]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def _save_state(path, *, step, indices, gaussians, config, diagnostics):
    features = F.normalize(
        gaussians._loc_feature.detach()[indices].reshape(indices.numel(), -1).float(),
        p=2,
        dim=1,
    ).cpu()
    state = {
        "version": 1,
        "iteration": int(step),
        "landmark_indices": indices.detach().cpu(),
        "landmark_features": features,
        "landmark_xyz": gaussians.get_xyz.detach()[indices].float().cpu(),
        "config": dict(config),
        "diagnostics": dict(diagnostics),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(args, dataset):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussians = _make_gaussians(dataset)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=False,
    )
    if not scene.loaded_iter:
        raise ValueError("A pretrained external 2DGS/3DGS map is required")

    point_count = int(gaussians.get_xyz.shape[0])
    feature_dim = int(gaussians.get_loc_feature.reshape(point_count, -1).shape[1])
    support = _load_indexed_state(args.support_state, point_count, feature_dim)
    sparse = (
        _load_indexed_state(args.sparse_state, point_count, feature_dim)
        if args.sparse_state
        else None
    )
    initial_features_cpu, sparse_merge_count = _merge_sparse_initialization(
        support, sparse, point_count
    )
    support_indices = _install_dense_support(
        gaussians,
        support["indices"],
        initial_features_cpu,
        args.non_support_opacity_logit,
    )
    initial_features = initial_features_cpu.to(device=gaussians._loc_feature.device)
    exp_avg = torch.zeros_like(initial_features)
    exp_avg_sq = torch.zeros_like(initial_features)

    cache, cache_signature, cache_signature_payload = _cache_queries(
        args.query_cache, dataset, args.iteration, feature_dim
    )
    camera_info = {
        str(camera.image_name).replace("\\", "/"): camera
        for camera in scene.scene_info.train_cameras
    }
    names = sorted(set(cache).intersection(camera_info))
    excluded_names, excluded_sha256 = _excluded_query_names(args.exclude_query_results)
    excluded_from_training = sorted(set(names).intersection(excluded_names))
    if args.exclude_query_results and not excluded_from_training:
        raise ValueError("exclude_query_results has no overlap with the dense training cache")
    names = [name for name in names if name not in excluded_names]
    if int(args.max_train_views) > 0 and len(names) > int(args.max_train_views):
        positions = torch.linspace(0, len(names) - 1, int(args.max_train_views)).round().long().tolist()
        names = [names[position] for position in positions]
    if not names:
        raise ValueError("No training cameras overlap the readonly real-query cache")

    sparse_seed_poses, sparse_seed_manifest = _load_sparse_seed_poses(
        args.sparse_seed_results,
        allowed_names=names,
        excluded_names=excluded_names,
    )
    sparse_seed_poses, sparse_seed_stats = _filter_sparse_seed_poses(
        sparse_seed_poses,
        cache,
        max_translation_m=args.sparse_seed_max_translation_m,
        max_rotation_deg=args.sparse_seed_max_rotation_deg,
    )
    sparse_seed_coverage = float(len(sparse_seed_poses)) / float(len(names))
    if (
        float(args.sparse_seed_probability) > 0.0
        and not sparse_seed_poses
    ):
        raise ValueError(
            "sparse_seed_probability is positive but no eligible sparse seed poses exist"
        )
    if sparse_seed_coverage < float(args.min_sparse_seed_coverage):
        raise ValueError(
            "sparse seed coverage is below min_sparse_seed_coverage: "
            f"{sparse_seed_coverage:.4f} < {float(args.min_sparse_seed_coverage):.4f}"
        )

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device=gaussians.get_xyz.device,
    )
    config = {
        "method": "lafgs_dense_composite_field_refinement_v1",
        "standalone_experimental": True,
        "geometry_frozen": True,
        "rgb_appearance_frozen": True,
        "loc_opacity_frozen": True,
        "field_depth_source": "field_expected",
        "fine_window_center": "render",
        "pixel_center_offset": 0.5,
        "query_source": "real_train_cache",
        "training_query_count": int(len(names)),
        "excluded_query_count": int(len(excluded_from_training)),
        "exclude_query_results": (
            str(Path(args.exclude_query_results).resolve())
            if args.exclude_query_results
            else None
        ),
        "exclude_query_results_sha256": excluded_sha256,
        "support_state": support["path"],
        "support_state_sha256": support["sha256"],
        "sparse_state": None if sparse is None else sparse["path"],
        "sparse_state_sha256": None if sparse is None else sparse["sha256"],
        "support_count": int(support_indices.numel()),
        "sparse_merge_count": int(sparse_merge_count),
        "point_count": point_count,
        "feature_dim": feature_dim,
        "sparse_seed": {
            **sparse_seed_manifest,
            **sparse_seed_stats,
            "coverage": sparse_seed_coverage,
            "probability": float(args.sparse_seed_probability),
            "max_translation_m": float(args.sparse_seed_max_translation_m),
            "max_rotation_deg": float(args.sparse_seed_max_rotation_deg),
        },
        "cache_path": str(Path(args.query_cache).resolve()),
        "cache_signature": cache_signature,
        "cache_signature_payload": cache_signature_payload,
        "model_path": os.path.abspath(dataset.model_path),
        "source_path": os.path.abspath(dataset.source_path),
        "map_iteration": int(args.iteration),
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"source_path", "model_path"}
        },
    }
    _write_json(output_dir / "manifest.json", config)

    generator = torch.Generator(device=gaussians.get_xyz.device).manual_seed(args.train_seed)
    camera_rng = random.Random(args.train_seed)
    order = list(names)
    camera_rng.shuffle(order)
    position = 0
    save_steps = sorted({int(step) for step in args.save_steps if int(step) >= 0} | {int(args.steps)})
    history = []
    started = time.time()

    def save_checkpoint(step):
        current = F.normalize(
            gaussians._loc_feature.detach()[support_indices].reshape(support_indices.numel(), -1),
            p=2,
            dim=1,
        )
        drift = 1.0 - (current * initial_features).sum(dim=1)
        diagnostics = {
            **_recent_mean(history, args.log_interval),
            "step": int(step),
            "feature_drift_cosine_mean": float((1.0 - drift).mean().item()),
            "feature_drift_l2_mean": float(
                torch.linalg.norm(current - initial_features, dim=1).mean().item()
            ),
            "feature_drift_l2_p95": float(
                torch.quantile(torch.linalg.norm(current - initial_features, dim=1), 0.95).item()
            ),
            "elapsed_seconds": float(time.time() - started),
        }
        _save_state(
            output_dir / f"{int(step)}_lafgs_dense_field_state.pt",
            step=step,
            indices=support_indices,
            gaussians=gaussians,
            config=config,
            diagnostics=diagnostics,
        )
        _write_json(output_dir / "training_history.json", history)
        _write_json(output_dir / "training_summary.json", {"config": config, "latest": diagnostics})

    if 0 in save_steps:
        save_checkpoint(0)

    progress = tqdm(range(1, int(args.steps) + 1), desc="LaFGS dense field")
    for step in progress:
        if position >= len(order):
            camera_rng.shuffle(order)
            position = 0
        name = order[position]
        position += 1
        cached = cache[name]
        query_feature = F.normalize(
            torch.as_tensor(cached["feature_map"], device=gaussians.get_xyz.device, dtype=torch.float32),
            p=2,
            dim=0,
        )
        query_valid_mask = torch.as_tensor(
            cached["valid_mask"], device=gaussians.get_xyz.device, dtype=torch.bool
        )
        pose_gt = torch.as_tensor(
            cached["pose_w2c"], device=gaussians.get_xyz.device, dtype=torch.float32
        )
        translation_jitter, rotation_jitter = _curriculum(step, args)
        sparse_seed_probability = float(args.sparse_seed_probability)
        if int(args.sparse_seed_warmup_steps) > 0:
            sparse_seed_probability *= min(
                float(step) / float(args.sparse_seed_warmup_steps), 1.0
            )
        use_sparse_seed = (
            name in sparse_seed_poses
            and sparse_seed_probability > 0.0
            and float(torch.rand((), device=pose_gt.device, generator=generator).item())
            < sparse_seed_probability
        )
        if use_sparse_seed:
            pose_init = sparse_seed_poses[name].to(
                device=pose_gt.device, dtype=pose_gt.dtype
            )
            pose_source = "sparse_seed"
        else:
            pose_init = _pose_perturbation(
                pose_gt,
                translation_max_m=translation_jitter,
                rotation_max_deg=rotation_jitter,
                exact_probability=args.exact_pose_probability,
                generator=generator,
            )
            pose_source = "synthetic_jitter"
        seed_translation_m, seed_rotation_deg = _pose_seed_error(pose_init, pose_gt)
        camera = camera_info[name]
        teacher = dense_localization_teacher(
            gaussians,
            query_feature,
            pose_init,
            pose_gt,
            float(camera.FovX),
            float(camera.FovY),
            int(query_feature.shape[2]),
            int(query_feature.shape[1]),
            background,
            anchor_count=args.anchor_count,
            alpha_threshold=args.alpha_threshold,
            desc_temperature=args.desc_temperature,
            fine_temperature=args.fine_temperature,
        fine_window_radius=args.fine_window_radius,
        fine_peak_weight=args.fine_peak_weight,
        fine_target_sigma=args.fine_target_sigma,
        fine_window_center="render",
            norm_feat_bf_render=True,
            use_loc_opacity=True,
            field_depth_source="field_expected",
            query_valid_mask=query_valid_mask,
            min_anchors=args.min_anchors,
            pose_refinement_weight=args.pose_refinement_weight,
            pose_refinement_iterations=args.pose_refinement_iterations,
            pose_refinement_damping=args.pose_refinement_damping,
            pose_refinement_translation_scale_m=args.pose_refinement_translation_scale_m,
            pose_refinement_rotation_scale_deg=args.pose_refinement_rotation_scale_deg,
            pose_refinement_max_anchors=args.pose_refinement_max_anchors,
            rasterize_args={"rasterize_mode": "antialiased"},
        )
        current = F.normalize(
            gaussians._loc_feature[support_indices].reshape(support_indices.numel(), -1),
            p=2,
            dim=1,
        )
        prototype = (1.0 - (current * initial_features).sum(dim=1)).mean()
        pose_refinement = (
            teacher.pose_loss
            if isinstance(teacher.pose_loss, torch.Tensor)
            else query_feature.new_tensor(0.0)
        )
        loss = (
            float(args.descriptor_weight) * teacher.desc_loss
            + float(args.reprojection_weight) * teacher.reproj_loss
            + float(args.prototype_weight) * prototype
            + float(args.pose_refinement_weight) * pose_refinement
        )
        gaussians._loc_feature.grad = None
        update_applied = False
        grad_norm = 0.0
        if bool(torch.isfinite(loss).item()) and int(teacher.anchor_count) >= int(args.min_anchors):
            loss.backward()
            update_applied, grad_norm = _manual_adam_step(
                gaussians._loc_feature,
                support_indices,
                exp_avg,
                exp_avg_sq,
                step=step,
                lr=args.feature_lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        gaussians._loc_feature.grad = None
        record = {
            "step": int(step),
            "loss": float(loss.detach().item()),
            "descriptor_loss": float(teacher.desc_loss.detach().item()),
            "reprojection_loss": float(teacher.reproj_loss.detach().item()),
            "prototype_loss": float(prototype.detach().item()),
            "pose_refinement_loss": float(pose_refinement.detach().item()),
            "anchor_count": int(teacher.anchor_count),
            "update_applied": float(update_applied),
            "grad_norm": float(grad_norm),
            "jitter_translation_m": float(translation_jitter),
            "jitter_rotation_deg": float(rotation_jitter),
            "pose_source": pose_source,
            "pose_source_sparse_seed": float(use_sparse_seed),
            "sparse_seed_probability_effective": float(sparse_seed_probability),
            "pose_seed_translation_m": float(seed_translation_m),
            "pose_seed_rotation_deg": float(seed_rotation_deg),
            "positive_prob_mean": float(
                teacher.diagnostics.get("anchor_positive_prob_mean", 0.0)
            ),
            "reproj_error_mean_px": float(
                teacher.diagnostics.get("anchor_reproj_error_mean_px", 0.0)
            ),
            "target_nll_mean": float(
                teacher.diagnostics.get("anchor_target_nll_mean", 0.0)
            ),
            "query_name_hash": hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
            **{
                key: value
                for key, value in teacher.diagnostics.items()
                if isinstance(value, (int, float))
            },
        }
        history.append(record)
        if step % max(int(args.log_interval), 1) == 0:
            recent = _recent_mean(history, args.log_interval)
            progress.set_postfix(
                loss=f"{recent.get('loss', 0.0):.4f}",
                desc=f"{recent.get('descriptor_loss', 0.0):.3f}",
                reproj=f"{recent.get('reprojection_loss', 0.0):.3f}",
                pos=f"{recent.get('positive_prob_mean', 0.0):.3f}",
            )
        if step in save_steps:
            save_checkpoint(step)
        del query_feature, query_valid_mask, pose_gt, pose_init, teacher, current, prototype, loss

    print(json.dumps(_recent_mean(history, args.log_interval), indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone composited dense-feature field refinement"
    )
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--support_state", required=True)
    parser.add_argument("--sparse_state", default="")
    parser.add_argument("--query_cache", required=True)
    parser.add_argument(
        "--exclude_query_results",
        default="",
        help="Saved results.json cameras excluded from dense-field training.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, nargs="*", default=[100, 500, 1000])
    parser.add_argument("--max_train_views", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=2026)
    parser.add_argument("--feature_lr", type=float, default=5e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--descriptor_weight", type=float, default=0.25)
    parser.add_argument("--reprojection_weight", type=float, default=1.0)
    parser.add_argument("--prototype_weight", type=float, default=0.02)
    parser.add_argument("--anchor_count", type=int, default=512)
    parser.add_argument("--min_anchors", type=int, default=64)
    parser.add_argument("--alpha_threshold", type=float, default=0.25)
    parser.add_argument("--desc_temperature", type=float, default=0.07)
    parser.add_argument("--fine_temperature", type=float, default=0.05)
    parser.add_argument("--fine_window_radius", type=int, default=4)
    parser.add_argument("--fine_peak_weight", type=float, default=0.5)
    parser.add_argument("--fine_target_sigma", type=float, default=0.75)
    parser.add_argument("--non_support_opacity_logit", type=float, default=-20.0)
    parser.add_argument("--jitter_translation_start_m", type=float, default=0.02)
    parser.add_argument("--jitter_translation_end_m", type=float, default=0.15)
    parser.add_argument("--jitter_rotation_start_deg", type=float, default=0.10)
    parser.add_argument("--jitter_rotation_end_deg", type=float, default=0.75)
    parser.add_argument("--jitter_warmup_steps", type=int, default=250)
    parser.add_argument("--exact_pose_probability", type=float, default=0.25)
    parser.add_argument(
        "--pose_refinement_weight",
        type=float,
        default=0.0,
        help="Weight for differentiable local GN pose supervision; zero disables it.",
    )
    parser.add_argument("--pose_refinement_iterations", type=int, default=1)
    parser.add_argument("--pose_refinement_damping", type=float, default=1e-2)
    parser.add_argument("--pose_refinement_translation_scale_m", type=float, default=0.05)
    parser.add_argument("--pose_refinement_rotation_scale_deg", type=float, default=0.5)
    parser.add_argument("--pose_refinement_max_anchors", type=int, default=128)
    parser.add_argument(
        "--sparse_seed_results",
        default="",
        help="Optional sparse results.json used as real dense-refinement pose seeds.",
    )
    parser.add_argument(
        "--sparse_seed_probability",
        type=float,
        default=0.0,
        help="Probability of a real sparse seed episode; the remainder uses jitter.",
    )
    parser.add_argument(
        "--min_sparse_seed_coverage",
        type=float,
        default=0.0,
        help="Minimum eligible sparse-seed fraction of dense training views.",
    )
    parser.add_argument(
        "--sparse_seed_warmup_steps",
        type=int,
        default=0,
        help="Linearly ramp real sparse-seed episodes over this many steps.",
    )
    parser.add_argument(
        "--sparse_seed_max_translation_m",
        type=float,
        default=0.0,
        help="Reject real sparse seeds beyond this camera-center error; <=0 disables.",
    )
    parser.add_argument(
        "--sparse_seed_max_rotation_deg",
        type=float,
        default=0.0,
        help="Reject real sparse seeds beyond this rotation error; <=0 disables.",
    )
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    return parser, model


def main():
    parser, model = build_parser()
    args = get_combined_args(parser)
    if int(args.steps) <= 0 or int(args.anchor_count) <= 0 or int(args.min_anchors) <= 0:
        raise ValueError("steps, anchor_count, and min_anchors must be positive")
    if int(args.min_anchors) > int(args.anchor_count):
        raise ValueError("min_anchors cannot exceed anchor_count")
    if not (0.0 <= float(args.alpha_threshold) <= 1.0):
        raise ValueError("alpha_threshold must be in [0, 1]")
    if not (0.0 <= float(args.exact_pose_probability) <= 1.0):
        raise ValueError("exact_pose_probability must be in [0, 1]")
    if not (0.0 <= float(args.sparse_seed_probability) <= 1.0):
        raise ValueError("sparse_seed_probability must be in [0, 1]")
    if not (0.0 <= float(args.min_sparse_seed_coverage) <= 1.0):
        raise ValueError("min_sparse_seed_coverage must be in [0, 1]")
    if min(
        float(args.sparse_seed_max_translation_m),
        float(args.sparse_seed_max_rotation_deg),
    ) < 0.0:
        raise ValueError("sparse seed basin limits must be non-negative")
    if int(args.sparse_seed_warmup_steps) < 0:
        raise ValueError("sparse_seed_warmup_steps must be non-negative")
    if int(args.pose_refinement_iterations) <= 0 or int(args.pose_refinement_max_anchors) < 4:
        raise ValueError("pose refinement iterations/max anchors are invalid")
    if float(args.pose_refinement_weight) < 0.0:
        raise ValueError("pose_refinement_weight must be non-negative")
    if float(args.pose_refinement_weight) > 0.0 and min(
        float(args.pose_refinement_damping),
        float(args.pose_refinement_translation_scale_m),
        float(args.pose_refinement_rotation_scale_deg),
    ) <= 0.0:
        raise ValueError("enabled pose refinement requires positive damping/scales")
    if min(float(args.feature_lr), float(args.desc_temperature), float(args.fine_temperature)) <= 0.0:
        raise ValueError("feature_lr and temperatures must be positive")
    safe_state(args.quiet)
    seed_everything(args.train_seed)
    dataset = model.extract(args)
    train(args, dataset)


if __name__ == "__main__":
    main()
