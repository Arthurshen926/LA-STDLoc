import os
import pickle
import sys
import uuid
import argparse
import copy
import csv
import json
import math
import random as py_random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import random, randint

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat, render_gsplat
from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder, NoReferenceValidMaskConfig
from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQuerySampler, PseudoTeacherCache
from la_artifacts.pseudo_query_training import (
    pseudo_query_reliability_decision as _pseudo_query_reliability_decision,
    pseudo_teacher_cache_reliability_stats as _pseudo_teacher_cache_reliability_stats,
)
from localization_training.dense_teacher import dense_localization_teacher
from localization_training.direct_landmark_teacher import (
    LandmarkObservationMemory,
    direct_landmark_teacher,
    gaussian_localization_xyz,
    make_intrinsics_from_fov,
    stable_landmark_memory_indices,
)
from localization_training.episode_sampler import (
    EpisodeSampler,
    SparsePoseCache,
    sample_interpolated_novel_view,
    split_support_query_cameras,
)
from localization_training.render_artifacts import (
    comma_set as artifact_comma_set,
    filter_cameras_by_artifacts,
    load_artifact_filter_names,
    load_artifact_region_weight_lookup,
    load_artifact_weight_lookup,
    weighted_mean,
)
from localization_training.losses import (
    geometry_anchor_loss,
    hard_negative_ranking_loss,
    localization_opacity_regularizer,
    prototype_loss,
)
from localization_training.lafgs_reconstruction import (
    apply_multiview_initialization,
    bounded_geometry_residual_loss,
    build_multiview_initialization,
    DifferentiablePnPConfig,
    MultiViewInitConfig,
    differentiable_pnp_pose_loss,
    lafgs_curriculum_step,
    lafgs_phase_from_starts,
    lafgs_should_sample_synthetic_view,
    pnp_output_to_landmark_stats,
    select_multiview_init_cameras,
    update_diff_pnp_training_summary,
)
from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import cal_pose_error

try:
    import numpy as np
except ImportError:
    np = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


def _json_safe(value):
    if isinstance(value, Namespace):
        return _json_safe(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_training_args_snapshot(dataset, opt, args, argv=None):
    model_path = Path(getattr(args, "model_path", "") or getattr(dataset, "model_path", ""))
    model_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "argv": list(sys.argv if argv is None else argv),
        "dataset": _json_safe(vars(dataset)),
        "opt": _json_safe(vars(opt)),
        "args": _json_safe(vars(args)),
    }
    path = model_path / "lafgs_training_args.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def append_unique_iteration(values, iteration):
    iteration = int(iteration)
    if iteration not in values:
        values.append(iteration)
    return values


def should_save_locaware_full_checkpoint(args, iteration):
    mode = str(getattr(args, "loc_full_checkpoint_mode", "save_iterations") or "save_iterations")
    iteration = int(iteration)
    if mode == "none":
        return False
    if mode == "save_iterations":
        return iteration in set(int(value) for value in getattr(args, "save_iterations", []) or [])
    if mode == "final":
        return iteration == int(getattr(args, "iterations", 0) or 0)
    if mode == "explicit":
        return iteration in set(
            int(value) for value in getattr(args, "loc_full_checkpoint_iterations", []) or []
        )
    raise ValueError(f"Unsupported loc_full_checkpoint_mode: {mode}")


def _load_masks(dataset):
    candidates = [
        os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        os.path.join(dataset.source_path, "masks.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print("Loading masks from", path)
            return pickle.load(open(path, "rb"))
    return None


def _comma_set(value):
    return artifact_comma_set(value, lower=False)


def _normalize_image_name(name):
    return str(name).replace("\\", "/").lstrip("./")


def _iter_artifact_filter_rows(path):
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError(f"Artifact filter JSON must contain a list of rows: {path}")
        for row in payload:
            if isinstance(row, dict):
                yield row
        return

    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def _load_query_artifact_filter_names(path, scene_name=None, severities="mild,severe", splits="heldout_query_sample"):
    return load_artifact_filter_names(
        path,
        scene_name=scene_name,
        severities=severities,
        splits=splits,
    )


def _filter_query_cameras_by_artifacts(cameras, artifact_names):
    return filter_cameras_by_artifacts(cameras, artifact_names)


def _resize_bool_mask(mask, target_hw):
    if mask.shape[-2:] == target_hw:
        return mask.bool()
    return (
        F.interpolate(mask[None].float(), size=target_hw, mode="nearest")
        .squeeze(0)
        .bool()
    )


def _is_locaware_checkpoint(checkpoint_data):
    return isinstance(checkpoint_data, dict) and "model_params" in checkpoint_data


def _split_checkpoint_payload(checkpoint_data):
    if _is_locaware_checkpoint(checkpoint_data):
        return (
            checkpoint_data["model_params"],
            checkpoint_data.get("iteration", 0),
            checkpoint_data.get("localization_state"),
        )
    model_params, first_iter = checkpoint_data
    return model_params, first_iter, None


def _restore_checkpoint(gaussians, opt, checkpoint):
    first_iter = 0
    if not checkpoint:
        gaussians.training_setup(opt)
        return first_iter

    checkpoint_data = torch.load(checkpoint)
    model_params, first_iter, loc_state = _split_checkpoint_payload(checkpoint_data)
    gaussians.restore(model_params, opt)
    if loc_state is not None:
        gaussians.restore_localization_state(loc_state)
    else:
        gaussians.init_localization_state(from_rgb_opacity=True)
    return first_iter


def _restore_external_localization_state(gaussians, state_or_path):
    if not state_or_path:
        return False
    if isinstance(state_or_path, (str, os.PathLike)):
        point_tensor = getattr(gaussians, "get_xyz", None)
        device = point_tensor.device if torch.is_tensor(point_tensor) else "cpu"
        state = torch.load(os.fspath(state_or_path), map_location=device)
    else:
        state = state_or_path
    gaussians.restore_localization_state(state)
    return True


def _gaussian_model_for_type(gaussian_type, sh_degree):
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs

    gaussian_type = str(gaussian_type or "3dgs").lower()
    if gaussian_type == "3dgs":
        return GaussianModel(sh_degree)
    if gaussian_type == "2dgs":
        return GaussianModel_2dgs(sh_degree)
    raise ValueError(f"Unsupported gaussian_type for LaFGS training: {gaussian_type}")


def _pseudo_teacher_cache_required(args):
    if bool(getattr(args, "pseudo_query_filter_teacher_cache", False)):
        return True
    if getattr(args, "pseudo_query_manifest", "") and bool(getattr(args, "pseudo_query_require_teacher_cache", True)):
        return True
    if str(getattr(args, "pseudo_query_reliability_mode", "none") or "none").lower() != "none":
        return True
    if str(getattr(args, "pseudo_query_stage_objective_mode", "none") or "none").lower() != "none":
        return True
    if not getattr(args, "pseudo_query_manifest", ""):
        return False
    query_mode = str(getattr(args, "query_mode", "noise") or "noise").lower()
    if query_mode == "sparse":
        return True
    if query_mode == "mixed" and float(getattr(args, "mixed_sparse_probability", 0.0) or 0.0) > 0.0:
        return True
    return False


def _scale_loc_loss_by_pseudo_reliability(loc_loss, pseudo_query_reliability, args):
    weight = float((pseudo_query_reliability or {}).get("weight", 1.0))
    if str(getattr(args, "pseudo_query_reliability_loss_mode", "none") or "none").lower() != "soft":
        return loc_loss
    if weight >= 0.999999:
        return loc_loss
    return loc_loss * loc_loss.new_tensor(weight)


def _pseudo_query_stage_direct_loss_policy(pseudo_query_reliability, args):
    mode = str(getattr(args, "pseudo_query_stage_objective_mode", "none") or "none").lower()
    reliability = pseudo_query_reliability or {}
    base_update_memory = bool(reliability.get("update_memory", True))
    base_update_stats = bool(reliability.get("update_stats", True))
    stage = str(reliability.get("stage", "unknown") or "unknown")
    policy = {
        "enabled": False,
        "stage": stage,
        "desc": 1.0,
        "multiview": 1.0,
        "full_bank": 1.0,
        "anchor": 1.0,
        "update_memory": base_update_memory,
        "update_stats": base_update_stats,
    }
    if mode == "none":
        return policy
    if mode != "direct":
        raise ValueError(f"Unsupported pseudo_query_stage_objective_mode: {mode}")

    stage_weights = {
        "teacher_ok": (1.0, 1.0, 1.0, 1.0, True, True),
        "dense_improves_sparse": (1.0, 1.0, 1.0, 1.0, True, True),
        "mixed_or_uncertain": (0.70, 0.70, 0.50, 1.0, True, True),
        "dense_rescues_sparse": (0.55, 1.0, 0.85, 1.0, True, True),
        "sparse_failure": (0.25, 0.0, 0.75, 1.0, False, False),
        "dense_regression_after_good_sparse": (0.35, 0.25, 0.50, 1.0, False, False),
        "unknown": (0.60, 0.50, 0.50, 1.0, True, True),
    }
    desc, multiview, full_bank, anchor, stage_update_memory, stage_update_stats = stage_weights.get(
        stage,
        stage_weights["unknown"],
    )
    stats_policy = str(getattr(args, "pseudo_query_stage_stats_policy", "hard") or "hard").lower()
    if stats_policy not in {"hard", "soft"}:
        raise ValueError(f"Unsupported pseudo_query_stage_stats_policy: {stats_policy}")
    update_stats = bool(base_update_stats and (stage_update_stats or stats_policy == "soft"))
    policy.update(
        {
            "enabled": True,
            "desc": float(desc),
            "multiview": float(multiview),
            "full_bank": float(full_bank),
            "anchor": float(anchor),
            "update_memory": bool(base_update_memory and stage_update_memory),
            "update_stats": update_stats,
        }
    )
    return policy


def _compose_direct_loc_loss(
    loc_desc_loss,
    loc_multiview_loss,
    loc_full_bank_loss,
    loc_anchor_loss,
    pseudo_query_reliability,
    args,
    stage_policy=None,
    full_bank_weight_scale=1.0,
    loc_clean_hard_negative_loss=None,
    clean_hard_negative_weight=None,
):
    policy = (
        stage_policy
        if stage_policy is not None
        else _pseudo_query_stage_direct_loss_policy(pseudo_query_reliability, args)
    )
    if loc_clean_hard_negative_loss is None:
        loc_clean_hard_negative_loss = loc_full_bank_loss.new_tensor(0.0)
    if clean_hard_negative_weight is None:
        clean_hard_negative_weight = _float_arg(args, "loc_clean_hard_negative_weight", -1.0)
        if clean_hard_negative_weight < 0.0:
            clean_hard_negative_weight = _float_arg(args, "loc_full_bank_clean_hard_negative_weight", 0.0)
    loc_loss = (
        float(args.loc_direct_weight) * float(policy.get("desc", 1.0)) * loc_desc_loss
        + float(args.loc_multiview_weight) * float(policy.get("multiview", 1.0)) * loc_multiview_loss
        + float(args.loc_full_bank_weight)
        * float(max(0.0, full_bank_weight_scale))
        * float(policy.get("full_bank", 1.0))
        * loc_full_bank_loss
        + float(max(0.0, clean_hard_negative_weight)) * loc_clean_hard_negative_loss
        + float(args.loc_anchor_weight) * float(policy.get("anchor", 1.0)) * loc_anchor_loss
    )
    return loc_loss, policy


def _clean_field_stage_controls(args, lafgs_step):
    start_iter = int(getattr(args, "loc_clean_field_start_iter", 0) or 0)
    active = int(lafgs_step) >= max(0, start_iter)
    full_bank_scale = 1.0
    clean_hn_scale = 1.0
    diff_pnp_scale = 1.0
    clean_hn_base_weight = _float_arg(args, "loc_clean_hard_negative_weight", -1.0)
    if clean_hn_base_weight < 0.0:
        clean_hn_base_weight = _float_arg(args, "loc_full_bank_clean_hard_negative_weight")
    balance_weight = _float_arg(args, "loc_full_bank_balance_weight")
    pose_info_weight = _float_arg(args, "loc_full_bank_pose_information_weight")
    if active:
        full_bank_scale = max(0.0, _float_arg(args, "loc_clean_field_full_bank_weight_scale", 1.0))
        clean_hn_scale = max(0.0, _float_arg(args, "loc_clean_field_clean_hn_weight_scale", 1.0))
        diff_pnp_scale = max(0.0, _float_arg(args, "loc_clean_field_diff_pnp_weight_scale", 1.0))
        override_balance = _float_arg(args, "loc_clean_field_balance_weight", -1.0)
        override_pose_info = _float_arg(args, "loc_clean_field_pose_information_weight", -1.0)
        if override_balance >= 0.0:
            balance_weight = override_balance
        if override_pose_info >= 0.0:
            pose_info_weight = override_pose_info
    return {
        "active": bool(active),
        "start_iter": start_iter,
        "full_bank_weight_scale": full_bank_scale,
        "clean_hn_weight": clean_hn_base_weight * clean_hn_scale,
        "clean_hn_weight_scale": clean_hn_scale,
        "balance_weight": balance_weight,
        "pose_information_weight": pose_info_weight,
        "diff_pnp_weight": _float_arg(args, "lafgs_diff_pnp_weight") * diff_pnp_scale,
        "diff_pnp_weight_scale": diff_pnp_scale,
    }


def _pseudo_query_stage_source_diagnostics(record, pseudo_query_reliability):
    source = str(getattr(record, "source", "") or "")
    known_sources = ("train_rgb", "synthetic_rgb", "other")
    source_key = source if source in {"train_rgb", "synthetic_rgb"} else "other"
    stage = str((pseudo_query_reliability or {}).get("stage", "unknown") or "unknown")
    known_stages = (
        "teacher_ok",
        "dense_improves_sparse",
        "mixed_or_uncertain",
        "dense_rescues_sparse",
        "sparse_failure",
        "dense_regression_after_good_sparse",
        "unknown",
    )
    if stage not in known_stages:
        stage = "unknown"

    diagnostics = {
        "pseudo_query_source_train_rgb": 1.0 if source_key == "train_rgb" else 0.0,
        "pseudo_query_source_synthetic_rgb": 1.0 if source_key == "synthetic_rgb" else 0.0,
        "pseudo_query_source_other": 1.0 if source_key == "other" else 0.0,
    }
    for stage_name in known_stages:
        diagnostics[f"pseudo_query_stage_{stage_name}"] = 1.0 if stage == stage_name else 0.0
    for source_name in known_sources:
        for stage_name in known_stages:
            diagnostics[f"pseudo_query_source_stage_{source_name}_{stage_name}"] = (
                1.0 if source_key == source_name and stage == stage_name else 0.0
            )
    return diagnostics


def _load_training_pose_cache(args):
    if getattr(args, "sparse_pose_cache", None):
        return SparsePoseCache(args.sparse_pose_cache).load()
    cache_path = getattr(args, "pseudo_teacher_cache", "") or ""
    if not cache_path:
        if _pseudo_teacher_cache_required(args):
            raise FileNotFoundError("Pseudo teacher cache is required by the current pseudo-query training options.")
        return None
    if not os.path.exists(cache_path):
        if _pseudo_teacher_cache_required(args):
            raise FileNotFoundError(
                "Pseudo teacher cache is required by the current pseudo-query training options "
                f"but was not found: {cache_path}"
            )
        print(f"Skipping missing optional pseudo teacher cache: {cache_path}")
        return None
    cache = PseudoTeacherCache.load(cache_path)
    print(f"Loaded pseudo teacher cache: {cache_path} items={len(cache.items)}")
    return cache


def _pseudo_teacher_cache_get_for_record(cache, record):
    if cache is None or record is None:
        return None
    key = getattr(record, "teacher_cache_key", "") or getattr(record, "query_id", "")
    item = cache.get(key) if key else None
    image_name = getattr(record, "image_name", "")
    if item is None and image_name and key != image_name:
        item = cache.get(image_name)
    return item


def _align_pseudo_manifest_to_teacher_cache(pseudo_manifest, sparse_pose_cache, enabled=True):
    before = len(getattr(pseudo_manifest, "records", []) or [])
    summary = {
        "enabled": bool(enabled),
        "before": int(before),
        "after": int(before),
        "dropped_missing_teacher_cache": 0,
        "quality_filtered": 0,
    }
    if not enabled:
        return pseudo_manifest, summary
    if sparse_pose_cache is None:
        summary["enabled"] = False
        summary["reason"] = "no_teacher_cache"
        return pseudo_manifest, summary

    rows = []
    for record in pseudo_manifest.records:
        if _pseudo_teacher_cache_get_for_record(sparse_pose_cache, record) is None:
            summary["dropped_missing_teacher_cache"] += 1
            continue
        rows.append(record)
    summary["after"] = len(rows)
    summary["source_counts_before"] = pseudo_manifest.source_counts()
    aligned = PseudoQueryManifest(version=pseudo_manifest.version, records=rows)
    summary["source_counts_after"] = aligned.source_counts()
    return aligned, summary


def _capture_geometry_anchor(gaussians):
    xyz = gaussians._xyz.detach()
    node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(node_ids) and node_ids.numel() == xyz.shape[0]:
        node_ids = node_ids.detach().clone().to(dtype=torch.long, device=xyz.device)
    else:
        node_ids = torch.arange(xyz.shape[0], dtype=torch.long, device=xyz.device)
    return {
        "node_ids": node_ids,
        "xyz": gaussians._xyz.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
    }


def _refresh_geometry_anchor_if_point_count_changed(gaussians, geometry_anchor):
    current = _current_geometry_state(gaussians)
    if "node_ids" not in geometry_anchor:
        if geometry_anchor["xyz"].shape[0] == current["xyz"].shape[0]:
            return geometry_anchor
        return _capture_geometry_anchor(gaussians)

    anchor_node_ids = geometry_anchor["node_ids"].detach().to(
        dtype=torch.long,
        device=current["xyz"].device,
    ).reshape(-1)
    current_node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(current_node_ids) and current_node_ids.numel() == current["xyz"].shape[0]:
        current_node_ids = current_node_ids.detach().to(
            dtype=torch.long,
            device=current["xyz"].device,
        ).reshape(-1)
    else:
        current_node_ids = torch.arange(
            current["xyz"].shape[0],
            dtype=torch.long,
            device=current["xyz"].device,
        )
    if (
        anchor_node_ids.shape == current_node_ids.shape
        and torch.equal(anchor_node_ids, current_node_ids)
    ):
        return geometry_anchor
    return _capture_geometry_anchor(gaussians)


def _current_geometry_state(gaussians):
    return {
        "xyz": gaussians._xyz,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }


def _capture_feature_anchor(gaussians):
    features = gaussians.get_loc_feature.detach().clone()
    node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(node_ids) and node_ids.numel() == features.shape[0]:
        node_ids = node_ids.detach().clone().to(dtype=torch.long, device=features.device)
    else:
        node_ids = torch.arange(features.shape[0], dtype=torch.long, device=features.device)
    return {"node_ids": node_ids, "features": features}


def _feature_anchor_tensor(feature_anchor):
    if feature_anchor is None:
        return None
    if isinstance(feature_anchor, dict):
        return feature_anchor["features"]
    return feature_anchor


def _refresh_feature_anchor_if_point_count_changed(gaussians, feature_anchor):
    if feature_anchor is None:
        return None
    current = gaussians.get_loc_feature.detach()
    if not isinstance(feature_anchor, dict):
        if feature_anchor.shape[0] == current.shape[0]:
            return feature_anchor
        if feature_anchor.shape[0] < current.shape[0]:
            return torch.cat([feature_anchor, current[feature_anchor.shape[0] :].clone()], dim=0)
        return feature_anchor[: current.shape[0]].clone()

    anchor_features = feature_anchor["features"].detach()
    anchor_node_ids = feature_anchor["node_ids"].detach().to(dtype=torch.long, device=current.device).reshape(-1)
    current_node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(current_node_ids) and current_node_ids.numel() == current.shape[0]:
        current_node_ids = current_node_ids.detach().to(dtype=torch.long, device=current.device).reshape(-1)
    else:
        current_node_ids = torch.arange(current.shape[0], dtype=torch.long, device=current.device)
    if (
        anchor_features.shape[0] == current.shape[0]
        and anchor_node_ids.shape[0] == current_node_ids.shape[0]
        and torch.equal(anchor_node_ids, current_node_ids)
    ):
        return feature_anchor

    anchor_features = anchor_features.to(device=current.device, dtype=current.dtype)
    aligned = current.clone()
    anchor_pos = {int(node_id): idx for idx, node_id in enumerate(anchor_node_ids.detach().cpu().tolist())}
    for row, node_id in enumerate(current_node_ids.detach().cpu().tolist()):
        idx = anchor_pos.get(int(node_id))
        if idx is not None:
            aligned[row] = anchor_features[idx]
    return {"node_ids": current_node_ids.detach().clone(), "features": aligned.detach().clone()}


def _load_landmark_indices(model_path, landmark_path, device="cpu", point_count=None):
    if str(landmark_path or "").lower() in {"", "__all__", "all"}:
        if point_count is None:
            raise ValueError("point_count is required when landmark_path selects all current points.")
        return torch.arange(int(point_count), dtype=torch.long, device=device)
    path = landmark_path
    if not os.path.isabs(path):
        path = os.path.join(model_path, path)
    with open(path, "rb") as f:
        indices = pickle.load(f)
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _current_landmark_indices_from_source_index(source_landmark_indices, gaussians):
    source_landmark_indices = torch.as_tensor(source_landmark_indices, dtype=torch.long).reshape(-1)
    source_index = getattr(gaussians, "loc_source_index", None)
    point_count = int(gaussians.get_xyz.shape[0])
    if not torch.is_tensor(source_index) or source_index.numel() != point_count:
        return source_landmark_indices.detach().cpu()
    if source_landmark_indices.numel() == 0:
        return source_landmark_indices.detach().cpu()
    source_index = source_index.to(dtype=torch.long)
    wanted = source_landmark_indices.to(device=source_index.device)
    current_mask = torch.isin(source_index, wanted)
    current = torch.nonzero(current_mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
    if current.numel() == 0:
        return source_landmark_indices.detach().cpu()
    return current.detach().cpu()


def _flatten_render_map(value):
    if value is None:
        return None
    while value.dim() > 2:
        value = value.squeeze(0)
    return value


def _flatten_render_alpha(value):
    if value is None:
        return None
    value = value.squeeze()
    if value.dim() == 3:
        value = value[..., 0]
    return value


def _set_phase_lrs(gaussians, phase, args):
    for group in gaussians.optimizer.param_groups:
        group.setdefault("la_base_lr", group["lr"])
        group["lr"] = group["la_base_lr"]

    overlay_mode = getattr(args, "loc_overlay_mode", "none")
    is_2dgs = _gaussian_type_is_2dgs(args)
    raw_xyz_trainable = (not is_2dgs) or _allow_raw_xyz_geometry_grad(args)
    anchor_trainable = _surface_loc_anchor_active(args)
    loc_trainable = {"loc_feature"}
    if anchor_trainable:
        loc_trainable.add("loc_anchor_offset")
    if getattr(args, "use_loc_opacity", False):
        loc_trainable.add("loc_opacity")
    if overlay_mode == "descriptor":
        loc_trainable = {"loc_overlay_feature", "loc_overlay_active_logit"}
        if anchor_trainable:
            loc_trainable.add("loc_anchor_offset")
        if getattr(args, "use_loc_opacity", False):
            loc_trainable.add("loc_opacity")

    diff_pnp_geometry_grad = _diff_pnp_allows_geometry_grad(args, phase)
    rgb_scaffold_trainable = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
    sfm_from_zero = str(getattr(args, "lafgs_stage_schedule", "none") or "none") == "sfm_from_zero"
    if sfm_from_zero:
        if phase == "mv_init":
            trainable = set(rgb_scaffold_trainable)
        elif phase in {"feature", "locrec", "diff_pnp"}:
            trainable = set(rgb_scaffold_trainable) | loc_trainable
        elif phase in {"geometry", "topology", "closed_loop"}:
            trainable = set(rgb_scaffold_trainable) | loc_trainable
        else:
            trainable = {group["name"] for group in gaussians.optimizer.param_groups}
        if is_2dgs and not raw_xyz_trainable:
            trainable.discard("xyz")
    else:
        if phase == "mv_init":
            trainable = set()
        elif phase in {"feature", "locrec", "diff_pnp"}:
            trainable = loc_trainable
            if diff_pnp_geometry_grad and raw_xyz_trainable:
                trainable = trainable | {"xyz"}
        elif phase in {"geometry", "topology", "closed_loop"}:
            trainable = {"scaling", "rotation", "loc_opacity"} | loc_trainable
            if raw_xyz_trainable:
                trainable.add("xyz")
        else:
            trainable = {group["name"] for group in gaussians.optimizer.param_groups}
            if is_2dgs and not raw_xyz_trainable:
                trainable.discard("xyz")

    for group in gaussians.optimizer.param_groups:
        if group["name"] not in trainable:
            group["lr"] = 0.0
        elif phase in {"geometry", "topology", "closed_loop"} or diff_pnp_geometry_grad:
            if group["name"] == "xyz":
                diff_pnp_xyz_lr = float(getattr(args, "lafgs_diff_pnp_geometry_xyz_lr", 0.0) or 0.0)
                if diff_pnp_geometry_grad and diff_pnp_xyz_lr > 0.0:
                    group["lr"] = diff_pnp_xyz_lr
                else:
                    group["lr"] = group["la_base_lr"] * args.geometry_xyz_lr_mult
            elif phase in {"geometry", "topology", "closed_loop"} and group["name"] == "scaling":
                group["lr"] = group["la_base_lr"] * args.geometry_scale_lr_mult
            elif phase in {"geometry", "topology", "closed_loop"} and group["name"] == "rotation":
                group["lr"] = group["la_base_lr"] * args.geometry_rotation_lr_mult


def _diff_pnp_allows_geometry_grad(args, phase):
    return bool(getattr(args, "lafgs_diff_pnp_allow_geometry_grad", False)) and phase in {
        "diff_pnp",
        "geometry",
        "topology",
        "closed_loop",
        "full",
    }


def _diff_pnp_needs_projected_uv(args):
    return (
        float(getattr(args, "lafgs_diff_pnp_local_window_radius", 0.0) or 0.0) > 0.0
        or float(getattr(args, "lafgs_diff_pnp_geometry_local_window_radius", 0.0) or 0.0) > 0.0
    )


def full_bank_nearby_as_positive_active(args, iteration, lafgs_step=None):
    if not bool(getattr(args, "loc_full_bank_nearby_as_positive", False)):
        return False
    until = int(getattr(args, "loc_full_bank_nearby_as_positive_until", 0) or 0)
    if until <= 0:
        return True
    step = int(lafgs_step if lafgs_step is not None else iteration)
    return step <= until


def lafgs_curriculum_base_iteration(args, scene_loaded_iter=0):
    if str(getattr(args, "lafgs_stage_schedule", "none") or "none") == "sfm_from_zero":
        return 0
    return int(scene_loaded_iter or 0)


def lafgs_should_run_multiview_initialization(args, first_iter=0):
    if not bool(getattr(args, "lafgs_mvinit_enabled", False)):
        return False
    if int(getattr(args, "lafgs_mvinit_max_views", 0) or 0) == 0:
        return False
    if str(getattr(args, "lafgs_stage_schedule", "none") or "none") == "sfm_from_zero" and int(first_iter) > 0:
        return False
    return True


def lafgs_stage_loss_weights(args, lafgs_step):
    base = float(getattr(args, "base_loss_weight", 1.0))
    loc = float(getattr(args, "loc_loss_weight", 1.0))
    geometry_anchor = float(getattr(args, "geometry_anchor_weight", 0.0))
    schedule = str(getattr(args, "lafgs_stage_schedule", "none") or "none")
    if schedule != "sfm_from_zero":
        return {
            "stage": "legacy",
            "base": base,
            "loc": loc,
            "geometry_anchor": geometry_anchor,
        }

    step = int(lafgs_step)
    bootstrap_until = int(getattr(args, "lafgs_stage_bootstrap_until", 3000) or 3000)
    joint_until = int(getattr(args, "lafgs_stage_joint_until", 15000) or 15000)
    if step <= bootstrap_until:
        return {
            "stage": "bootstrap",
            "base": float(getattr(args, "lafgs_stage_bootstrap_base_weight", 1.0)),
            "loc": float(getattr(args, "lafgs_stage_bootstrap_loc_weight", 0.15)),
            "geometry_anchor": float(getattr(args, "lafgs_stage_bootstrap_geometry_anchor_weight", geometry_anchor)),
        }
    if step <= joint_until:
        return {
            "stage": "joint",
            "base": float(getattr(args, "lafgs_stage_joint_base_weight", 0.5)),
            "loc": float(getattr(args, "lafgs_stage_joint_loc_weight", 1.0)),
            "geometry_anchor": float(getattr(args, "lafgs_stage_joint_geometry_anchor_weight", geometry_anchor)),
        }
    return {
        "stage": "refine",
        "base": float(getattr(args, "lafgs_stage_refine_base_weight", 0.15)),
        "loc": float(getattr(args, "lafgs_stage_refine_loc_weight", 1.5)),
        "geometry_anchor": float(getattr(args, "lafgs_stage_refine_geometry_anchor_weight", geometry_anchor)),
    }


def lafgs_rgb_densify_active(args, lafgs_step):
    if not bool(getattr(args, "lafgs_rgb_densify", False)):
        return False
    until = int(getattr(args, "lafgs_rgb_densify_until_iter", 0) or 0)
    if until <= 0:
        return True
    return int(lafgs_step) < until


def _prune_lafgs_rgb_densify_child_outliers(gaussians, max_source_drift):
    max_source_drift = float(max_source_drift or 0.0)
    if max_source_drift <= 0.0:
        return {"pruned": 0}
    xyz = getattr(gaussians, "get_xyz", None)
    source_xyz = getattr(gaussians, "loc_source_xyz", None)
    birth_iteration = getattr(gaussians, "loc_birth_iteration", None)
    if not (
        torch.is_tensor(xyz)
        and torch.is_tensor(source_xyz)
        and torch.is_tensor(birth_iteration)
        and hasattr(gaussians, "prune_points")
    ):
        return {"pruned": 0, "skipped": 1}
    count = min(int(xyz.shape[0]), int(source_xyz.shape[0]), int(birth_iteration.numel()))
    if count <= 0:
        return {"pruned": 0}
    xyz_head = xyz[:count].detach()
    source_head = source_xyz[:count].to(device=xyz_head.device, dtype=xyz_head.dtype)
    birth_head = birth_iteration[:count].to(device=xyz_head.device, dtype=torch.long).reshape(-1)
    child_mask = birth_head > 0
    if not bool(child_mask.any().item()):
        return {"pruned": 0, "child_count": 0}
    drift = torch.linalg.norm(xyz_head - source_head, dim=-1)
    prune_head = child_mask & ((~torch.isfinite(drift)) | (drift > max_source_drift))
    pruned = int(prune_head.sum().item())
    stats = {
        "pruned": pruned,
        "child_count": int(child_mask.sum().item()),
        "max_source_drift": float(drift[child_mask].detach().max().item()),
        "mean_source_drift": float(drift[child_mask].detach().mean().item()),
    }
    if pruned <= 0:
        return stats
    prune_mask = torch.zeros((int(xyz.shape[0]),), dtype=torch.bool, device=xyz_head.device)
    prune_mask[:count] = prune_head
    gaussians.prune_points(prune_mask)
    return stats


def _float_arg(args, name, default=0.0):
    value = getattr(args, name, default)
    if value is None:
        value = default
    return float(value)


def _gaussian_type_is_2dgs(args):
    return str(getattr(args, "gaussian_type", "") or "").lower() == "2dgs"


def _surface_loc_bounds_active(args):
    return (
        _float_arg(args, "surfel_loc_tangent_bound") > 0.0
        or _float_arg(args, "surfel_loc_normal_bound") > 0.0
    )


def _surface_loc_anchor_active(args):
    return _float_arg(args, "loc_anchor_lr") > 0.0 and _surface_loc_bounds_active(args)


def _allow_raw_xyz_geometry_grad(args):
    return bool(getattr(args, "allow_raw_xyz_geometry_grad", False))


def _geometry_feedback_requested(args):
    return any(
        _float_arg(args, name) > 0.0
        for name in (
            "lafgs_diff_pnp_geometry_reproj_weight",
            "lafgs_diff_pnp_geometry_depth_anchor_weight",
            "lafgs_diff_pnp_geometry_match_reproj_weight",
        )
    )


def _validate_lafgs_surface_geometry_config(gaussians, args):
    if _float_arg(args, "loc_anchor_lr") > 0.0 and not _surface_loc_bounds_active(args):
        raise ValueError(
            "--loc_anchor_lr is positive but both surfel localization bounds are zero; "
            "set --surfel_loc_tangent_bound and/or --surfel_loc_normal_bound so get_loc_xyz can move."
        )
    if _gaussian_type_is_2dgs(args) and bool(getattr(args, "lafgs_diff_pnp_allow_geometry_grad", False)):
        geometry_xyz_lr = _float_arg(args, "lafgs_diff_pnp_geometry_xyz_lr")
        if geometry_xyz_lr > 0.0 and not _allow_raw_xyz_geometry_grad(args):
            raise ValueError(
                "2DGS raw surfel-center geometry updates are disabled by default. "
                "Set --allow_raw_xyz_geometry_grad to explicitly allow "
                "--lafgs_diff_pnp_geometry_xyz_lr > 0."
            )
        if (
            not _allow_raw_xyz_geometry_grad(args)
            and not _surface_loc_anchor_active(args)
            and _geometry_feedback_requested(args)
        ):
            print(
                "[LaFGS config warning] 2DGS geometry feedback is enabled, but neither "
                "surface localization anchors nor raw xyz updates are trainable. Geometry losses "
                "will mainly act on descriptors/scores."
            )
    if (
        _float_arg(args, "lafgs_diff_pnp_weight") > 0.0
        and _float_arg(args, "lafgs_diff_pnp_local_window_radius") <= 0.0
    ):
        print(
            "[LaFGS config warning] --lafgs_diff_pnp_local_window_radius <= 0 uses global "
            "soft matching for PnP feedback; this is noisy for localization-aware geometry."
        )
    if _float_arg(args, "lafgs_diff_pnp_point_weight_floor") > 0.5:
        print(
            "[LaFGS config warning] --lafgs_diff_pnp_point_weight_floor is high; low-utility "
            "landmarks will still receive strong PnP weights."
        )


def _configure_surface_localization_anchor(gaussians, args, opt=None):
    if opt is not None:
        setattr(opt, "loc_anchor_lr", float(getattr(args, "loc_anchor_lr", 0.0) or 0.0))
    if hasattr(gaussians, "detach_loc_anchor_base"):
        gaussians.detach_loc_anchor_base = _gaussian_type_is_2dgs(args) and not _allow_raw_xyz_geometry_grad(args)
    if hasattr(gaussians, "surfel_loc_tangent_bound"):
        gaussians.surfel_loc_tangent_bound = float(getattr(args, "surfel_loc_tangent_bound", 0.0) or 0.0)
    if hasattr(gaussians, "surfel_loc_normal_bound"):
        gaussians.surfel_loc_normal_bound = float(getattr(args, "surfel_loc_normal_bound", 0.0) or 0.0)
    if hasattr(gaussians, "surfel_loc_radius_floor"):
        gaussians.surfel_loc_radius_floor = float(getattr(args, "surfel_loc_radius_floor", 0.0) or 0.0)
    if hasattr(gaussians, "_ensure_loc_anchor_state"):
        gaussians._ensure_loc_anchor_state()


def _phase_allows_geometry_update(args, phase):
    return phase in {"geometry", "topology", "closed_loop", "full"} or _diff_pnp_allows_geometry_grad(args, phase)


def _optimizer_lr_for_group(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group.get("lr", 0.0))
    return 0.0


def _record_geometry_optimizer_diagnostics(
    summary,
    gaussians,
    phase,
    xyz_before=None,
    record_lr_grad=True,
    geometry_active=None,
):
    if geometry_active is None:
        geometry_active = phase in {"geometry", "topology", "closed_loop", "full"}
    if not bool(geometry_active):
        return

    if record_lr_grad:
        xyz_lr = _optimizer_lr_for_group(gaussians.optimizer, "xyz")
        loc_anchor_lr = _optimizer_lr_for_group(gaussians.optimizer, "loc_anchor_offset")
        summary["geometry_optimizer_episodes"] = summary.get("geometry_optimizer_episodes", 0) + 1
        summary["geometry_xyz_lr_total"] = summary.get("geometry_xyz_lr_total", 0.0) + xyz_lr
        summary["geometry_xyz_lr_max"] = max(summary.get("geometry_xyz_lr_max", 0.0), xyz_lr)
        if xyz_lr > 0.0:
            summary["geometry_xyz_lr_nonzero_episodes"] = summary.get("geometry_xyz_lr_nonzero_episodes", 0) + 1
        summary["geometry_loc_anchor_lr_total"] = summary.get("geometry_loc_anchor_lr_total", 0.0) + loc_anchor_lr
        summary["geometry_loc_anchor_lr_max"] = max(
            summary.get("geometry_loc_anchor_lr_max", 0.0),
            loc_anchor_lr,
        )
        if loc_anchor_lr > 0.0:
            summary["geometry_loc_anchor_lr_nonzero_episodes"] = (
                summary.get("geometry_loc_anchor_lr_nonzero_episodes", 0) + 1
            )

        xyz_grad = getattr(gaussians._xyz, "grad", None)
        grad_max = 0.0
        if xyz_grad is not None:
            finite_grad = torch.nan_to_num(xyz_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            if finite_grad.numel() > 0:
                grad_max = float(finite_grad.abs().max().item())
        summary["geometry_xyz_grad_abs_max"] = max(summary.get("geometry_xyz_grad_abs_max", 0.0), grad_max)
        if grad_max > 0.0:
            summary["geometry_xyz_grad_nonzero_episodes"] = (
                summary.get("geometry_xyz_grad_nonzero_episodes", 0) + 1
            )
        loc_anchor_grad = getattr(getattr(gaussians, "_loc_anchor_offset", None), "grad", None)
        loc_anchor_grad_max = 0.0
        if loc_anchor_grad is not None:
            finite_grad = torch.nan_to_num(loc_anchor_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            if finite_grad.numel() > 0:
                loc_anchor_grad_max = float(finite_grad.abs().max().item())
        summary["geometry_loc_anchor_grad_abs_max"] = max(
            summary.get("geometry_loc_anchor_grad_abs_max", 0.0),
            loc_anchor_grad_max,
        )
        if loc_anchor_grad_max > 0.0:
            summary["geometry_loc_anchor_grad_nonzero_episodes"] = (
                summary.get("geometry_loc_anchor_grad_nonzero_episodes", 0) + 1
            )

    if xyz_before is not None:
        with torch.no_grad():
            xyz_after = gaussians._xyz.detach()
            before_count = int(xyz_before.shape[0])
            after_count = int(xyz_after.shape[0])
            if before_count != after_count:
                summary["geometry_xyz_step_point_count_changed"] = (
                    summary.get("geometry_xyz_step_point_count_changed", 0) + 1
                )
                summary["geometry_xyz_step_delta_skipped_point_count_changed"] = (
                    summary.get("geometry_xyz_step_delta_skipped_point_count_changed", 0) + 1
                )
                return
            count = min(before_count, after_count)
            delta = torch.linalg.norm(
                xyz_after[:count] - xyz_before[:count].to(xyz_after.device),
                dim=-1,
            )
            step_max = float(delta.max().item()) if delta.numel() else 0.0
        summary["geometry_xyz_step_delta_max"] = max(
            summary.get("geometry_xyz_step_delta_max", 0.0),
            step_max,
        )
        if step_max > 0.0:
            summary["geometry_xyz_step_nonzero_episodes"] = (
                summary.get("geometry_xyz_step_nonzero_episodes", 0) + 1
            )


def _record_direct_teacher_diagnostics(summary, diagnostics, prefix="direct_diag"):
    if not diagnostics:
        return
    recorded = 0
    for name, value in diagnostics.items():
        if not isinstance(value, (int, float)):
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        key = f"{prefix}_{name}"
        summary[f"{key}_total"] = summary.get(f"{key}_total", 0.0) + value
        summary[f"{key}_max"] = max(summary.get(f"{key}_max", value), value)
        summary[f"{key}_min"] = min(summary.get(f"{key}_min", value), value)
        recorded += 1
    if recorded > 0:
        summary[f"{prefix}_episodes"] = summary.get(f"{prefix}_episodes", 0) + 1


def _record_lafgs_geometry_residual_diagnostics(summary, teacher_out, loss, stats):
    diagnostics = {
        "lafgs_geometry_residual_loss": float(loss.detach().item()),
        "lafgs_geometry_residual_over_limit_count": float(stats["over_limit_count"]),
        "lafgs_geometry_residual_max_norm": float(stats["max_residual_norm"]),
        "lafgs_geometry_residual_max_allowed": float(stats["max_allowed_norm"]),
    }
    if teacher_out is not None:
        teacher_out.diagnostics.update(diagnostics)
    _record_direct_teacher_diagnostics(summary, diagnostics)


def _clip_lafgs_geometry_gradients(gaussians, max_abs, summary=None):
    max_abs = float(max_abs or 0.0)
    if max_abs <= 0.0:
        return 0

    clipped = 0
    for name in ("_xyz", "_loc_anchor_offset", "_scaling", "_rotation"):
        param = getattr(gaussians, name, None)
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        finite = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if finite.numel() == 0:
            continue
        before = float(finite.abs().max().item())
        key = name[1:] if name.startswith("_") else name
        if summary is not None:
            summary[f"geometry_grad_clip_{key}_before_abs_max"] = max(
                summary.get(f"geometry_grad_clip_{key}_before_abs_max", 0.0),
                before,
            )
        if before > max_abs:
            grad.detach().clamp_(min=-max_abs, max=max_abs)
            clipped += 1
            if summary is not None:
                summary[f"geometry_grad_clip_{key}_events"] = (
                    summary.get(f"geometry_grad_clip_{key}_events", 0) + 1
                )
    if summary is not None:
        summary["geometry_grad_clip_abs_config"] = max_abs
        if clipped > 0:
            summary["geometry_grad_clip_events"] = summary.get("geometry_grad_clip_events", 0) + 1
            summary["geometry_grad_clip_param_events"] = (
                summary.get("geometry_grad_clip_param_events", 0) + clipped
            )
    return clipped


def _tensor_abs_max(value):
    if value is None:
        return 0.0
    finite = torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(finite.abs().max().item()) if finite.numel() else 0.0


def _record_gradient_source_diagnostics(summary, gaussians, loss, source_name, scale=1.0):
    if not (torch.is_tensor(loss) and bool(loss.requires_grad)):
        return
    targets = [
        ("raw_xyz", getattr(gaussians, "_xyz", None)),
        ("loc_anchor_offset", getattr(gaussians, "_loc_anchor_offset", None)),
        ("loc_feature", getattr(gaussians, "_loc_feature", None)),
    ]
    params = [
        (name, param)
        for name, param in targets
        if torch.is_tensor(param) and bool(getattr(param, "requires_grad", False))
    ]
    if not params:
        return
    scaled_loss = loss * loss.new_tensor(float(scale))
    grads = torch.autograd.grad(
        scaled_loss,
        [param for _, param in params],
        retain_graph=True,
        allow_unused=True,
    )
    for (target_name, _), grad in zip(params, grads):
        prefix = f"diff_pnp_grad_{source_name}_{target_name}"
        summary[f"{prefix}_episodes"] = summary.get(f"{prefix}_episodes", 0) + 1
        abs_max = _tensor_abs_max(grad)
        norm = 0.0
        if grad is not None:
            finite = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            norm = float(torch.linalg.norm(finite.reshape(-1)).item()) if finite.numel() else 0.0
        summary[f"{prefix}_abs_max_max"] = max(summary.get(f"{prefix}_abs_max_max", 0.0), abs_max)
        summary[f"{prefix}_norm_total"] = summary.get(f"{prefix}_norm_total", 0.0) + norm
        summary[f"{prefix}_norm_max"] = max(summary.get(f"{prefix}_norm_max", 0.0), norm)
        if abs_max > 0.0:
            summary[f"{prefix}_nonzero_episodes"] = summary.get(f"{prefix}_nonzero_episodes", 0) + 1


def _record_diff_pnp_gradient_diagnostics(summary, gaussians, pnp_out, args, effective_pnp_weight=None):
    pnp_weight = (
        _float_arg(args, "lafgs_diff_pnp_weight", 1.0)
        if effective_pnp_weight is None
        else float(effective_pnp_weight)
    )
    base_scale = _float_arg(args, "loc_loss_weight", 1.0) * pnp_weight
    if base_scale == 0.0:
        return
    components = (
        ("total_loss", pnp_out.loss, 1.0),
        ("pose_loss", pnp_out.pose_loss, _float_arg(args, "lafgs_diff_pnp_pose_weight", 1.0)),
        (
            "reprojection_loss",
            pnp_out.reprojection_loss,
            _float_arg(args, "lafgs_diff_pnp_reproj_weight", 1.0),
        ),
        (
            "gt_reprojection_loss",
            pnp_out.gt_reprojection_loss,
            _float_arg(args, "lafgs_diff_pnp_gt_reproj_weight", 1.0),
        ),
        (
            "geometry_reproj_loss",
            pnp_out.geometry_reprojection_loss,
            _float_arg(args, "lafgs_diff_pnp_geometry_reproj_weight", 1.0),
        ),
        (
            "geometry_depth_anchor_loss",
            pnp_out.geometry_depth_anchor_loss,
            _float_arg(args, "lafgs_diff_pnp_geometry_depth_anchor_weight", 1.0),
        ),
        (
            "geometry_match_loss",
            pnp_out.geometry_match_reprojection_loss,
            _float_arg(args, "lafgs_diff_pnp_geometry_match_reproj_weight", 1.0),
        ),
    )
    for name, component_loss, component_scale in components:
        if component_scale == 0.0:
            continue
        _record_gradient_source_diagnostics(
            summary,
            gaussians,
            component_loss,
            name,
            scale=base_scale * component_scale,
        )


def _record_norm_stats(summary, prefix, values):
    if values is None:
        return
    values = torch.nan_to_num(values.detach().reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        summary[f"{prefix}_mean"] = 0.0
        summary[f"{prefix}_median"] = 0.0
        summary[f"{prefix}_p95"] = 0.0
        summary[f"{prefix}_max"] = 0.0
        return
    summary[f"{prefix}_mean"] = float(values.mean().item())
    summary[f"{prefix}_median"] = float(torch.quantile(values, 0.5).item())
    summary[f"{prefix}_p95"] = float(torch.quantile(values, 0.95).item())
    summary[f"{prefix}_max"] = float(values.max().item())


def _record_lafgs_static_config(summary, args, gaussians):
    summary.update(
        {
            "gaussian_type": str(getattr(args, "gaussian_type", "")),
            "loc_anchor_lr_config": _float_arg(args, "loc_anchor_lr"),
            "loc_anchor_active": bool(_surface_loc_anchor_active(args)),
            "loc_anchor_bounds_active": bool(_surface_loc_bounds_active(args)),
            "surfel_loc_tangent_bound_config": _float_arg(args, "surfel_loc_tangent_bound"),
            "surfel_loc_normal_bound_config": _float_arg(args, "surfel_loc_normal_bound"),
            "surfel_loc_radius_floor_config": _float_arg(args, "surfel_loc_radius_floor"),
            "surfel_loc_anchor_reg_weight_config": _float_arg(args, "surfel_loc_anchor_reg_weight"),
            "detach_loc_anchor_base": bool(getattr(gaussians, "detach_loc_anchor_base", False)),
            "allow_raw_xyz_geometry_grad": bool(_allow_raw_xyz_geometry_grad(args)),
            "geometry_xyz_lr_config": _float_arg(args, "lafgs_diff_pnp_geometry_xyz_lr"),
            "diff_pnp_local_window_radius_config": _float_arg(args, "lafgs_diff_pnp_local_window_radius"),
            "diff_pnp_geometry_local_window_radius_config": _float_arg(
                args,
                "lafgs_diff_pnp_geometry_local_window_radius",
            ),
            "diff_pnp_point_weight_floor_config": _float_arg(args, "lafgs_diff_pnp_point_weight_floor"),
            "diff_pnp_max_condition_number_config": _float_arg(args, "lafgs_diff_pnp_max_condition_number", -1.0),
            "diff_pnp_detach_pnp_points_config": bool(getattr(args, "lafgs_diff_pnp_detach_pnp_points", False)),
            "diff_pnp_allow_geometry_grad_config": bool(
                getattr(args, "lafgs_diff_pnp_allow_geometry_grad", False)
            ),
            "diff_pnp_geometry_reproj_weight_config": _float_arg(
                args,
                "lafgs_diff_pnp_geometry_reproj_weight",
            ),
            "diff_pnp_geometry_depth_anchor_weight_config": _float_arg(
                args,
                "lafgs_diff_pnp_geometry_depth_anchor_weight",
            ),
            "diff_pnp_geometry_match_reproj_weight_config": _float_arg(
                args,
                "lafgs_diff_pnp_geometry_match_reproj_weight",
            ),
            "loc_full_bank_balance_weight_config": _float_arg(args, "loc_full_bank_balance_weight"),
            "loc_full_bank_balance_grid_size_config": int(
                getattr(args, "loc_full_bank_balance_grid_size", 0) or 0
            ),
            "loc_full_bank_balance_depth_bins_config": int(
                getattr(args, "loc_full_bank_balance_depth_bins", 0) or 0
            ),
            "loc_full_bank_clean_hard_negative_weight_config": _float_arg(
                args,
                "loc_full_bank_clean_hard_negative_weight",
            ),
            "loc_clean_hard_negative_weight_config": _float_arg(
                args,
                "loc_clean_hard_negative_weight",
                -1.0,
            ),
            "loc_full_bank_clean_reproj_radius_config": _float_arg(
                args,
                "loc_full_bank_clean_reproj_radius",
            ),
            "loc_clean_field_start_iter_config": int(
                getattr(args, "loc_clean_field_start_iter", 0) or 0
            ),
            "loc_clean_field_full_bank_weight_scale_config": _float_arg(
                args,
                "loc_clean_field_full_bank_weight_scale",
                1.0,
            ),
            "loc_clean_field_clean_hn_weight_scale_config": _float_arg(
                args,
                "loc_clean_field_clean_hn_weight_scale",
                1.0,
            ),
            "loc_clean_field_balance_weight_config": _float_arg(
                args,
                "loc_clean_field_balance_weight",
                -1.0,
            ),
            "loc_clean_field_pose_information_weight_config": _float_arg(
                args,
                "loc_clean_field_pose_information_weight",
                -1.0,
            ),
            "loc_clean_field_diff_pnp_weight_scale_config": _float_arg(
                args,
                "loc_clean_field_diff_pnp_weight_scale",
                1.0,
            ),
            "diff_pnp_geometry_use_all_correspondences_config": bool(
                getattr(args, "lafgs_diff_pnp_geometry_use_all_correspondences", False)
            ),
            "lafgs_stage_schedule": str(getattr(args, "lafgs_stage_schedule", "none")),
            "lafgs_stage_bootstrap_until": int(getattr(args, "lafgs_stage_bootstrap_until", 0) or 0),
            "lafgs_stage_joint_until": int(getattr(args, "lafgs_stage_joint_until", 0) or 0),
            "lafgs_rgb_densify_config": bool(getattr(args, "lafgs_rgb_densify", False)),
            "lafgs_rgb_densify_until_iter_config": int(
                getattr(args, "lafgs_rgb_densify_until_iter", 0) or 0
            ),
            "lafgs_rgb_densify_child_max_source_drift_config": _float_arg(
                args,
                "lafgs_rgb_densify_child_max_source_drift",
                0.0,
            ),
            "lafgs_geometry_grad_clip_abs_config": _float_arg(
                args,
                "lafgs_geometry_grad_clip_abs",
                0.0,
            ),
        }
    )


def _capture_geometry_delta_reference(gaussians):
    with torch.no_grad():
        return {
            "raw_xyz": gaussians.get_xyz.detach().clone(),
            "loc_xyz": gaussian_localization_xyz(gaussians).detach().clone(),
        }


def _record_anchor_component_stats(summary, gaussians):
    raw_offset = getattr(gaussians, "_loc_anchor_offset", None)
    if not torch.is_tensor(raw_offset) or raw_offset.numel() == 0:
        return
    tangent_bound = float(getattr(gaussians, "surfel_loc_tangent_bound", 0.0) or 0.0)
    normal_bound = float(getattr(gaussians, "surfel_loc_normal_bound", 0.0) or 0.0)
    with torch.no_grad():
        raw_offset = raw_offset.detach()
        _record_norm_stats(summary, "loc_anchor_raw_tanh_norm", torch.linalg.norm(torch.tanh(raw_offset), dim=-1))
        scales = gaussians.get_scaling.detach()
        if scales.numel() == 0:
            return
        if scales.shape[1] >= 2:
            radius = scales[:, :2].mean(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            radius = scales.reshape(scales.shape[0], -1).mean(dim=1, keepdim=True).clamp_min(1e-8)
        radius_floor = float(getattr(gaussians, "surfel_loc_radius_floor", 0.0) or 0.0)
        if radius_floor > 0.0:
            radius = radius.clamp_min(radius_floor)
        tangent_delta = torch.tanh(raw_offset[:, :2]) * (tangent_bound * radius)
        normal_delta = torch.tanh(raw_offset[:, 2:3]) * (normal_bound * radius)
        _record_norm_stats(summary, "loc_anchor_tangent_delta_norm", torch.linalg.norm(tangent_delta, dim=-1))
        _record_norm_stats(summary, "loc_anchor_normal_delta_abs", normal_delta.abs().reshape(-1))


def _record_final_geometry_delta_summary(summary, gaussians, reference):
    if not reference:
        return

    def _source_aligned_delta(current_xyz, ref_xyz, source_index, mask=None):
        if not torch.is_tensor(ref_xyz) or ref_xyz.numel() == 0:
            return None
        if not torch.is_tensor(source_index) or source_index.numel() != current_xyz.shape[0]:
            return None
        source_index = source_index.to(device=current_xyz.device, dtype=torch.long).reshape(-1)
        valid = (source_index >= 0) & (source_index < int(ref_xyz.shape[0]))
        if mask is not None:
            valid = valid & mask.to(device=current_xyz.device, dtype=torch.bool).reshape(-1)
        if not bool(valid.any().item()):
            return current_xyz.new_zeros((0,))
        aligned_ref = ref_xyz.to(device=current_xyz.device, dtype=current_xyz.dtype)[source_index[valid]]
        return torch.linalg.norm(current_xyz[valid] - aligned_ref, dim=-1)

    with torch.no_grad():
        raw_xyz = gaussians.get_xyz.detach()
        loc_xyz = gaussian_localization_xyz(gaussians).detach()
        raw_ref = reference.get("raw_xyz")
        loc_ref = reference.get("loc_xyz")
        source_index = getattr(gaussians, "loc_source_index", None)
        birth_iteration = getattr(gaussians, "loc_birth_iteration", None)
        source_aligned = (
            torch.is_tensor(source_index)
            and source_index.numel() == raw_xyz.shape[0]
            and torch.is_tensor(raw_ref)
        )
        birth0_mask = None
        child_mask = None
        if source_aligned:
            source_index = source_index.to(device=raw_xyz.device, dtype=torch.long).reshape(-1)
            valid_source = (source_index >= 0) & (source_index < int(raw_ref.shape[0]))
            summary["geometry_source_aligned_delta_count"] = int(valid_source.sum().item())
            if torch.is_tensor(birth_iteration) and birth_iteration.numel() == raw_xyz.shape[0]:
                birth_iteration = birth_iteration.to(device=raw_xyz.device, dtype=torch.long).reshape(-1)
                birth0_mask = valid_source & (birth_iteration <= 0)
                child_mask = valid_source & (birth_iteration > 0)
            else:
                birth0_mask = valid_source
                child_mask = valid_source & torch.zeros_like(valid_source)
            summary["geometry_birth0_delta_count"] = int(birth0_mask.sum().item())
            summary["geometry_child_delta_count"] = int(child_mask.sum().item())
        summary["geometry_initial_point_count"] = int(raw_ref.shape[0]) if torch.is_tensor(raw_ref) else 0
        summary["geometry_final_point_count"] = int(raw_xyz.shape[0])
        summary["geometry_point_count_changed"] = bool(
            torch.is_tensor(raw_ref) and raw_ref.shape[0] != raw_xyz.shape[0]
        )
        if torch.is_tensor(raw_ref):
            if source_aligned:
                all_delta = _source_aligned_delta(raw_xyz, raw_ref, source_index)
                if all_delta is not None:
                    _record_norm_stats(summary, "raw_xyz_delta_from_initial_all_sources", all_delta)
                birth0_delta = _source_aligned_delta(raw_xyz, raw_ref, source_index, birth0_mask)
                if birth0_delta is not None:
                    _record_norm_stats(summary, "raw_xyz_delta_from_initial", birth0_delta)
                child_delta = _source_aligned_delta(raw_xyz, raw_ref, source_index, child_mask)
                if child_delta is not None and child_delta.numel() > 0:
                    _record_norm_stats(summary, "raw_xyz_child_delta_from_source", child_delta)
            else:
                count = min(int(raw_ref.shape[0]), int(raw_xyz.shape[0]))
                _record_norm_stats(
                    summary,
                    "raw_xyz_delta_from_initial",
                    torch.linalg.norm(raw_xyz[:count] - raw_ref[:count].to(raw_xyz.device), dim=-1),
                )
        if torch.is_tensor(loc_ref):
            if source_aligned:
                all_delta = _source_aligned_delta(loc_xyz, loc_ref, source_index)
                if all_delta is not None:
                    _record_norm_stats(summary, "loc_xyz_delta_from_initial_all_sources", all_delta)
                birth0_delta = _source_aligned_delta(loc_xyz, loc_ref, source_index, birth0_mask)
                if birth0_delta is not None:
                    _record_norm_stats(summary, "loc_xyz_delta_from_initial", birth0_delta)
                child_delta = _source_aligned_delta(loc_xyz, loc_ref, source_index, child_mask)
                if child_delta is not None and child_delta.numel() > 0:
                    _record_norm_stats(summary, "loc_xyz_child_delta_from_source", child_delta)
            else:
                count = min(int(loc_ref.shape[0]), int(loc_xyz.shape[0]))
                _record_norm_stats(
                    summary,
                    "loc_xyz_delta_from_initial",
                    torch.linalg.norm(loc_xyz[:count] - loc_ref[:count].to(loc_xyz.device), dim=-1),
                )
        count = min(int(raw_xyz.shape[0]), int(loc_xyz.shape[0]))
        _record_norm_stats(
            summary,
            "loc_xyz_minus_raw_xyz",
            torch.linalg.norm(loc_xyz[:count] - raw_xyz[:count], dim=-1),
        )
    _record_anchor_component_stats(summary, gaussians)


def _backward_with_optional_isolated_xyz_grad(
    total_loss,
    xyz_only_loss,
    gaussians,
    isolate_xyz_grad=False,
    isolated_xyz_scaffold_loss=None,
    isolated_xyz_regularizer_loss=None,
    summary=None,
):
    if not bool(isolate_xyz_grad):
        total_loss.backward()
        return

    xyz_param = getattr(gaussians, "_xyz", None)
    if xyz_param is None:
        total_loss.backward()
        return

    isolated_xyz_terms = []
    if torch.is_tensor(isolated_xyz_scaffold_loss) and bool(isolated_xyz_scaffold_loss.requires_grad):
        isolated_xyz_terms.append(isolated_xyz_scaffold_loss)
    if torch.is_tensor(xyz_only_loss) and bool(xyz_only_loss.requires_grad):
        isolated_xyz_terms.append(xyz_only_loss)
    if torch.is_tensor(isolated_xyz_regularizer_loss) and bool(isolated_xyz_regularizer_loss.requires_grad):
        isolated_xyz_terms.append(isolated_xyz_regularizer_loss)
    isolated_xyz_loss = None
    if isolated_xyz_terms:
        isolated_xyz_loss = isolated_xyz_terms[0]
        for term in isolated_xyz_terms[1:]:
            isolated_xyz_loss = isolated_xyz_loss + term

    xyz_loss_requires_grad = isolated_xyz_loss is not None
    total_loss.backward(retain_graph=xyz_loss_requires_grad)
    full_grad = getattr(xyz_param, "grad", None)
    full_grad_max = _tensor_abs_max(full_grad)

    isolated_grad = None
    if isolated_xyz_loss is not None:
        isolated_grad = torch.autograd.grad(
            isolated_xyz_loss,
            xyz_param,
            retain_graph=False,
            allow_unused=True,
        )[0]

    if isolated_grad is None:
        xyz_param.grad = None
        isolated_grad_max = 0.0
    else:
        if xyz_param.grad is None:
            xyz_param.grad = isolated_grad.detach().clone()
        else:
            xyz_param.grad.detach().copy_(isolated_grad.detach())
        isolated_grad_max = _tensor_abs_max(isolated_grad)

    if summary is not None:
        summary["geometry_xyz_isolated_grad_episodes"] = (
            summary.get("geometry_xyz_isolated_grad_episodes", 0) + 1
        )
        summary["geometry_xyz_full_grad_abs_max"] = max(
            summary.get("geometry_xyz_full_grad_abs_max", 0.0),
            full_grad_max,
        )
        summary["geometry_xyz_isolated_grad_abs_max"] = max(
            summary.get("geometry_xyz_isolated_grad_abs_max", 0.0),
            isolated_grad_max,
        )


def _configure_descriptor_overlay(gaussians, args, direct_landmark_indices=None):
    mode = getattr(args, "loc_overlay_mode", "none")
    if mode == "none":
        return False
    if mode != "descriptor":
        raise ValueError(f"Unsupported loc_overlay_mode: {mode}")
    if direct_landmark_indices is None:
        raise ValueError("descriptor overlay requires direct landmark source indices")
    gaussians.init_descriptor_overlay(
        direct_landmark_indices,
        init_active_logit=getattr(args, "loc_overlay_active_logit", 0.0),
        max_residual_norm=getattr(args, "loc_overlay_max_residual_norm", 0.0),
        normalize=getattr(args, "loc_overlay_normalize", False),
    )
    gaussians.add_descriptor_overlay_to_optimizer(lr=getattr(args, "loc_overlay_lr", 0.0))
    return True


def _descriptor_overlay_regularizer(gaussians):
    if not getattr(gaussians, "_has_descriptor_overlay", lambda: False)():
        feature = getattr(gaussians, "_loc_feature", None)
        if torch.is_tensor(feature):
            return feature.new_tensor(0.0)
        return torch.tensor(0.0)
    feature = gaussians._loc_overlay_feature
    active = torch.sigmoid(gaussians._loc_overlay_active_logit.to(device=feature.device, dtype=feature.dtype))
    residual = feature * active
    return residual.reshape(residual.shape[0], -1).pow(2).sum(dim=1).mean()


def _mask_frozen_child_loc_feature_gradients(gaussians, iteration, freeze_steps):
    if int(freeze_steps) <= 0:
        return 0
    loc_feature = getattr(gaussians, "_loc_feature", None)
    grad = getattr(loc_feature, "grad", None)
    if grad is None:
        return 0
    parent_node_id = getattr(gaussians, "loc_parent_node_id", None)
    last_topology_iteration = getattr(gaussians, "last_topology_iteration", None)
    if not torch.is_tensor(parent_node_id) or not torch.is_tensor(last_topology_iteration):
        return 0
    if parent_node_id.shape[0] != grad.shape[0] or last_topology_iteration.shape[0] != grad.shape[0]:
        return 0
    parent_node_id = parent_node_id.to(device=grad.device, dtype=torch.long).reshape(-1)
    last_topology_iteration = last_topology_iteration.to(device=grad.device, dtype=torch.long).reshape(-1)
    age = int(iteration) - last_topology_iteration
    frozen = (parent_node_id >= 0) & (age >= 0) & (age <= int(freeze_steps))
    if not frozen.any():
        return 0
    grad[frozen] = 0
    return int(frozen.sum().item())


def add_locaware_training_args(parser):
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--localization_state_path", type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument(
        "--loc_full_checkpoint_mode",
        choices=["save_iterations", "final", "explicit", "none"],
        default="save_iterations",
        help=(
            "Controls writing heavyweight chkpnt_locaware_*.pth training checkpoints. "
            "Map artifacts are still saved by --save_iterations."
        ),
    )
    parser.add_argument("--loc_full_checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--localization_enabled", action="store_true", default=True)
    parser.add_argument("--feature_only", action="store_true", default=False)
    parser.add_argument("--train_phase", type=str, default="feature", choices=["feature", "geometry", "topology", "closed_loop", "full"])
    parser.add_argument("--base_loss_weight", type=float, default=1.0)
    parser.add_argument("--base_feature_weight", type=float, default=1.0)
    parser.add_argument("--loc_loss_weight", type=float, default=1.0)
    parser.add_argument("--loc_start_iter", type=int, default=1)
    parser.add_argument("--loc_interval", type=int, default=8)
    parser.add_argument("--loc_anchors", type=int, default=512)
    parser.add_argument("--loc_alpha_threshold", type=float, default=0.2)
    parser.add_argument("--loc_desc_temperature", type=float, default=0.07)
    parser.add_argument("--loc_fine_temperature", type=float, default=0.05)
    parser.add_argument("--loc_fine_window_radius", type=int, default=4)
    parser.add_argument("--loc_desc_weight", type=float, default=1.0)
    parser.add_argument("--loc_reproj_weight", type=float, default=0.1)
    parser.add_argument("--loc_dense_kl_weight", type=float, default=0.0)
    parser.add_argument("--loc_dense_kl_temperature", type=float, default=0.07)
    parser.add_argument("--loc_dense_rank_weight", type=float, default=0.0)
    parser.add_argument("--loc_dense_rank_margin", type=float, default=0.2)
    parser.add_argument("--loc_dense_rank_teacher_confidence", type=float, default=0.0)
    parser.add_argument("--loc_dense_rank_miss_topk", type=int, default=1)
    parser.add_argument("--loc_responsibility_topk", type=int, default=32)
    parser.add_argument("--loc_responsibility_opacity_weight", type=float, default=0.0)
    parser.add_argument("--loc_responsibility_depth_weight", type=float, default=0.0)
    parser.add_argument("--loc_dense_pose_gate", action="store_true", default=False)
    parser.add_argument("--loc_dense_pose_gate_min_te", type=float, default=0.0)
    parser.add_argument("--loc_dense_pose_gate_min_ae", type=float, default=0.0)
    parser.add_argument("--loc_dense_advantage_gate", action="store_true", default=False)
    parser.add_argument("--loc_dense_advantage_min_te", type=float, default=0.0)
    parser.add_argument("--loc_dense_advantage_min_ae", type=float, default=0.0)
    parser.add_argument("--loc_dense_advantage_te_scale", type=float, default=10.0)
    parser.add_argument("--loc_dense_advantage_ae_scale", type=float, default=1.0)
    parser.add_argument("--loc_dense_attr_cosine_threshold", type=float, default=-1.0)
    parser.add_argument("--loc_dense_attr_entropy_threshold", type=float, default=-1.0)
    parser.add_argument("--loc_dense_min_positive_prob", type=float, default=-1.0)
    parser.add_argument("--loc_dense_max_reproj_error", type=float, default=-1.0)
    parser.add_argument("--loc_dense_min_eligible_anchors", type=int, default=1)
    parser.add_argument("--loc_teacher", type=str, default="dense", choices=["dense", "direct"])
    parser.add_argument("--loc_direct_weight", type=float, default=0.1)
    parser.add_argument("--loc_multiview_weight", type=float, default=0.05)
    parser.add_argument("--loc_multiview_temperature", type=float, default=0.07)
    parser.add_argument("--loc_multiview_slots", type=int, default=4)
    parser.add_argument("--loc_multiview_ignore_radius", type=float, default=2.0)
    parser.add_argument("--loc_full_bank_weight", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_temperature", type=float, default=0.07)
    parser.add_argument("--loc_full_bank_hard_negatives", type=int, default=32)
    parser.add_argument("--loc_full_bank_margin", type=float, default=0.2)
    parser.add_argument("--loc_full_bank_stats_chunk_size", type=int, default=256)
    parser.add_argument("--loc_full_bank_ignore_3d_radius", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_ignore_uv_radius", type=float, default=0.0)
    parser.add_argument(
        "--loc_full_bank_source_mode",
        type=str,
        default="ignore",
        choices=["ignore", "positive", "responsibility"],
    )
    parser.add_argument("--loc_full_bank_nearby_as_positive", action="store_true", default=False)
    parser.add_argument("--loc_full_bank_nearby_as_positive_until", type=int, default=0)
    parser.add_argument("--loc_full_bank_pose_information_weight", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_pose_information_floor", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_balance_weight", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_balance_grid_size", type=int, default=0)
    parser.add_argument("--loc_full_bank_balance_depth_bins", type=int, default=0)
    parser.add_argument("--loc_full_bank_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--loc_full_bank_clean_hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--loc_clean_hard_negative_weight", type=float, default=-1.0)
    parser.add_argument("--loc_full_bank_clean_reproj_radius", type=float, default=4.0)
    parser.add_argument("--loc_full_bank_clean_hard_negatives", type=int, default=16)
    parser.add_argument("--loc_clean_field_start_iter", type=int, default=0)
    parser.add_argument("--loc_clean_field_full_bank_weight_scale", type=float, default=1.0)
    parser.add_argument("--loc_clean_field_clean_hn_weight_scale", type=float, default=1.0)
    parser.add_argument("--loc_clean_field_balance_weight", type=float, default=-1.0)
    parser.add_argument("--loc_clean_field_pose_information_weight", type=float, default=-1.0)
    parser.add_argument("--loc_clean_field_diff_pnp_weight_scale", type=float, default=1.0)
    parser.add_argument("--loc_child_feature_freeze_steps", type=int, default=0)
    parser.add_argument("--loc_child_responsibility_mode", type=str, default="none", choices=["none", "feature"])
    parser.add_argument("--loc_child_responsibility_start_iter", type=int, default=0)
    parser.add_argument("--loc_overlay_mode", type=str, default="none", choices=["none", "descriptor"])
    parser.add_argument("--loc_overlay_lr", type=float, default=0.0)
    parser.add_argument("--loc_overlay_active_logit", type=float, default=0.0)
    parser.add_argument("--loc_overlay_max_residual_norm", type=float, default=0.0)
    parser.add_argument("--loc_overlay_normalize", action="store_true", default=False)
    parser.add_argument("--loc_overlay_reg_weight", type=float, default=0.0)
    parser.add_argument("--loc_anchor_weight", type=float, default=0.0)
    parser.add_argument("--loc_anchor_lr", type=float, default=0.0)
    parser.add_argument("--surfel_loc_tangent_bound", type=float, default=0.0)
    parser.add_argument("--surfel_loc_normal_bound", type=float, default=0.0)
    parser.add_argument("--surfel_loc_radius_floor", type=float, default=0.0)
    parser.add_argument("--surfel_loc_anchor_reg_weight", type=float, default=0.0)
    parser.add_argument("--landmark_path", type=str, default="detector/sampled_idx.pkl")
    parser.add_argument("--direct_depth_check", action="store_true", default=False)
    parser.add_argument("--direct_depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--direct_depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--loc_proto_weight", type=float, default=0.0)
    parser.add_argument("--loc_rank_weight", type=float, default=0.0)
    parser.add_argument("--loc_rank_margin", type=float, default=0.2)
    parser.add_argument("--loc_opacity_weight", type=float, default=0.0)
    parser.add_argument("--loc_opacity_target", type=float, default=0.5)
    parser.add_argument("--loc_ema_decay", type=float, default=0.95)
    parser.add_argument("--lafgs_diff_pnp_weight", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_start_iter", type=int, default=0)
    parser.add_argument("--lafgs_diff_pnp_temperature", type=float, default=0.07)
    parser.add_argument("--lafgs_diff_pnp_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_min_correspondences", type=int, default=6)
    parser.add_argument("--lafgs_diff_pnp_iterations", type=int, default=3)
    parser.add_argument("--lafgs_diff_pnp_pose_weight", type=float, default=1.0)
    parser.add_argument("--lafgs_diff_pnp_reproj_weight", type=float, default=0.1)
    parser.add_argument("--lafgs_diff_pnp_gt_reproj_weight", type=float, default=1.0)
    parser.add_argument("--lafgs_diff_pnp_entropy_weight", type=float, default=0.0)
    parser.add_argument(
        "--lafgs_diff_pnp_reprojection_loss_type",
        choices=["smooth_l1", "huber", "cauchy"],
        default="smooth_l1",
    )
    parser.add_argument("--lafgs_diff_pnp_reprojection_loss_delta", type=float, default=1.0)
    parser.add_argument("--lafgs_diff_pnp_local_window_radius", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_max_correspondences", type=int, default=0)
    parser.add_argument("--lafgs_diff_pnp_spatial_grid_size", type=int, default=0)
    parser.add_argument("--lafgs_diff_pnp_min_spatial_span", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_min_spatial_area", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_point_weight_floor", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_utility_pose_loss_scale", type=float, default=1.0)
    parser.add_argument("--lafgs_diff_pnp_utility_reprojection_error_scale", type=float, default=4.0)
    parser.add_argument("--lafgs_diff_pnp_allow_geometry_grad", action="store_true", default=False)
    parser.add_argument("--allow_raw_xyz_geometry_grad", dest="allow_raw_xyz_geometry_grad", action="store_true")
    parser.add_argument("--disallow_raw_xyz_geometry_grad", dest="allow_raw_xyz_geometry_grad", action="store_false")
    parser.set_defaults(allow_raw_xyz_geometry_grad=False)
    parser.add_argument("--lafgs_diff_pnp_isolate_geometry_grad", action="store_true", default=False)
    parser.add_argument("--lafgs_diff_pnp_geometry_xyz_lr", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_reproj_weight", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_depth_anchor_weight", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_reproj_weight", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_margin_threshold", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_peak_probability_threshold", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_max_entropy", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_max_reproj_error", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_margin_threshold", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_peak_probability_threshold", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_max_entropy", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_max_reproj_error", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_use_all_correspondences", action="store_true", default=False)
    parser.add_argument("--lafgs_diff_pnp_geometry_local_window_radius", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_max_condition_number", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_max_loss", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_softness", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_min_scale", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_max_loss", type=float, default=-1.0)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_softness", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_min_scale", type=float, default=0.0)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", action="store_true", default=False)
    parser.add_argument("--lafgs_diff_pnp_detach_pnp_points", action="store_true", default=False)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--lafgs_diff_pnp_detach_gt_reprojection_points",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
    else:
        parser.add_argument(
            "--lafgs_diff_pnp_detach_gt_reprojection_points",
            dest="lafgs_diff_pnp_detach_gt_reprojection_points",
            action="store_true",
        )
        parser.add_argument(
            "--no-lafgs_diff_pnp_detach_gt_reprojection_points",
            dest="lafgs_diff_pnp_detach_gt_reprojection_points",
            action="store_false",
        )
        parser.set_defaults(lafgs_diff_pnp_detach_gt_reprojection_points=False)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--lafgs_diff_pnp_use_loc_opacity_weight",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
    else:
        parser.add_argument("--lafgs_diff_pnp_use_loc_opacity_weight", dest="lafgs_diff_pnp_use_loc_opacity_weight", action="store_true")
        parser.add_argument("--no-lafgs_diff_pnp_use_loc_opacity_weight", dest="lafgs_diff_pnp_use_loc_opacity_weight", action="store_false")
        parser.set_defaults(lafgs_diff_pnp_use_loc_opacity_weight=False)
    parser.add_argument("--lafgs_mvinit_enabled", action="store_true", default=False)
    parser.add_argument("--lafgs_mv_init_until", type=int, default=0)
    parser.add_argument("--lafgs_mvinit_max_views", type=int, default=0)
    parser.add_argument("--lafgs_mvinit_view_selection", choices=["first", "uniform"], default="first")
    parser.add_argument("--lafgs_mvinit_min_observations", type=int, default=1)
    parser.add_argument("--lafgs_mvinit_chunk_size", type=int, default=0)
    parser.add_argument("--lafgs_mvinit_feature_scale", type=float, default=1.0)
    parser.add_argument("--lafgs_curriculum", action="store_true", default=False)
    parser.add_argument("--lafgs_locrec_start_iter", type=int, default=1)
    parser.add_argument("--lafgs_geometry_start_iter", type=int, default=10000)
    parser.add_argument("--lafgs_topology_start_iter", type=int, default=15000)
    parser.add_argument("--lafgs_stage_schedule", choices=["none", "sfm_from_zero"], default="none")
    parser.add_argument("--lafgs_stage_bootstrap_until", type=int, default=3000)
    parser.add_argument("--lafgs_stage_joint_until", type=int, default=15000)
    parser.add_argument("--lafgs_stage_bootstrap_base_weight", type=float, default=1.0)
    parser.add_argument("--lafgs_stage_bootstrap_loc_weight", type=float, default=0.15)
    parser.add_argument("--lafgs_stage_bootstrap_geometry_anchor_weight", type=float, default=0.05)
    parser.add_argument("--lafgs_stage_joint_base_weight", type=float, default=0.5)
    parser.add_argument("--lafgs_stage_joint_loc_weight", type=float, default=1.0)
    parser.add_argument("--lafgs_stage_joint_geometry_anchor_weight", type=float, default=0.05)
    parser.add_argument("--lafgs_stage_refine_base_weight", type=float, default=0.15)
    parser.add_argument("--lafgs_stage_refine_loc_weight", type=float, default=1.5)
    parser.add_argument("--lafgs_stage_refine_geometry_anchor_weight", type=float, default=0.02)
    parser.add_argument("--lafgs_rgb_densify", action="store_true", default=False)
    parser.add_argument("--lafgs_rgb_densify_until_iter", type=int, default=0)
    parser.add_argument("--lafgs_rgb_densify_child_max_source_drift", type=float, default=0.0)
    parser.add_argument("--lafgs_geometry_residual", action="store_true", default=False)
    parser.add_argument("--lafgs_geometry_residual_weight", type=float, default=0.0)
    parser.add_argument("--lafgs_geometry_residual_max_scale_ratio", type=float, default=0.2)
    parser.add_argument("--lafgs_geometry_grad_clip_abs", type=float, default=0.0)
    parser.add_argument(
        "--lafgs_synthetic_feature_source",
        type=str,
        default="loc_feature",
        choices=["loc_feature", "rgb"],
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--use_loc_opacity", action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument("--use_loc_opacity", dest="use_loc_opacity", action="store_true")
        parser.add_argument("--no-use_loc_opacity", dest="use_loc_opacity", action="store_false")
        parser.set_defaults(use_loc_opacity=False)
    parser.add_argument("--query_mode", type=str, default="noise", choices=["noise", "sparse", "mixed"])
    parser.add_argument("--pose_noise_quantile", type=float, default=0.5)
    parser.add_argument("--pose_noise_sampling", type=str, default="empirical", choices=["empirical", "quantile"])
    parser.add_argument("--mixed_sparse_probability", type=float, default=0.5)
    parser.add_argument("--sparse_pose_cache", type=str, default=None)
    parser.add_argument("--synthetic_view_ratio", type=float, default=0.0)
    parser.add_argument("--synthetic_view_candidates", type=int, default=1)
    parser.add_argument("--synthetic_view_alpha_min", type=float, default=0.35)
    parser.add_argument("--synthetic_view_alpha_max", type=float, default=0.65)
    parser.add_argument("--synthetic_view_min_observability", type=float, default=0.0)
    parser.add_argument("--synthetic_view_desc_weight", type=float, default=0.0)
    parser.add_argument("--synthetic_view_reproj_weight", type=float, default=0.0)
    parser.add_argument("--pseudo_query_manifest", type=str, default="")
    parser.add_argument("--pseudo_teacher_cache", type=str, default="")
    parser.add_argument("--pseudo_query_real_weight", type=float, default=2.0)
    parser.add_argument("--pseudo_query_synthetic_weight", type=float, default=1.0)
    parser.add_argument(
        "--pseudo_query_sampling_mode",
        type=str,
        default="record_proportional",
        choices=["source_balanced", "record_proportional"],
    )
    parser.add_argument("--pseudo_query_max_synthetic", type=int, default=0)
    parser.add_argument("--pseudo_query_sources", type=str, default="train_rgb")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--pseudo_query_require_teacher_cache", action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument("--pseudo_query_require_teacher_cache", dest="pseudo_query_require_teacher_cache", action="store_true")
        parser.add_argument("--no-pseudo_query_require_teacher_cache", dest="pseudo_query_require_teacher_cache", action="store_false")
        parser.set_defaults(pseudo_query_require_teacher_cache=True)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--pseudo_query_filter_teacher_cache", action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument("--pseudo_query_filter_teacher_cache", dest="pseudo_query_filter_teacher_cache", action="store_true")
        parser.add_argument("--no-pseudo_query_filter_teacher_cache", dest="pseudo_query_filter_teacher_cache", action="store_false")
        parser.set_defaults(pseudo_query_filter_teacher_cache=False)
    parser.add_argument("--pseudo_query_exclude_sparse_failure_stages", action="store_true", default=False)
    parser.add_argument("--pseudo_query_teacher_max_sparse_te", type=float, default=100.0)
    parser.add_argument("--pseudo_query_teacher_max_dense_te", type=float, default=100.0)
    parser.add_argument("--pseudo_query_teacher_allowed_stages", type=str, default="")
    parser.add_argument(
        "--pseudo_query_reliability_mode",
        type=str,
        default="none",
        choices=["none", "soft"],
    )
    parser.add_argument(
        "--pseudo_query_reliability_loss_mode",
        type=str,
        default="none",
        choices=["none", "soft"],
    )
    parser.add_argument(
        "--pseudo_query_stage_objective_mode",
        type=str,
        default="none",
        choices=["none", "direct"],
        help="Use pseudo teacher cache stages to reweight direct localization loss components.",
    )
    parser.add_argument(
        "--pseudo_query_stage_stats_policy",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Use hard stage gates for stats updates, or keep soft-weighted stages in stats updates.",
    )
    parser.add_argument("--pseudo_query_reliability_min_weight", type=float, default=0.20)
    parser.add_argument("--pseudo_query_reliability_real_min_weight", type=float, default=0.50)
    parser.add_argument("--pseudo_query_reliability_synthetic_min_weight", type=float, default=0.25)
    parser.add_argument("--pseudo_query_reliability_memory_min_weight", type=float, default=0.75)
    parser.add_argument("--pseudo_query_reliability_stats_min_weight", type=float, default=None)
    parser.add_argument("--pseudo_query_reliability_error_scale", type=float, default=2.0)
    parser.add_argument("--pseudo_query_reliability_inlier_power", type=float, default=0.5)
    parser.add_argument("--pseudo_query_reliability_teacher_ok_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_query_reliability_dense_improves_weight", type=float, default=0.90)
    parser.add_argument("--pseudo_query_reliability_mixed_weight", type=float, default=0.70)
    parser.add_argument("--pseudo_query_reliability_dense_rescues_weight", type=float, default=0.55)
    parser.add_argument("--pseudo_query_reliability_sparse_failure_weight", type=float, default=0.30)
    parser.add_argument("--pseudo_query_reliability_dense_regression_weight", type=float, default=0.35)
    parser.add_argument("--pseudo_query_reliability_unknown_weight", type=float, default=0.60)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--pseudo_query_no_reference_region_weight", action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument("--pseudo_query_no_reference_region_weight", dest="pseudo_query_no_reference_region_weight", action="store_true")
        parser.add_argument("--no-pseudo_query_no_reference_region_weight", dest="pseudo_query_no_reference_region_weight", action="store_false")
        parser.set_defaults(pseudo_query_no_reference_region_weight=False)
    parser.add_argument("--pseudo_query_no_reference_region_weight_sources", type=str, default="synthetic_rgb")
    parser.add_argument("--pseudo_query_no_reference_region_weight_min", type=float, default=0.25)
    parser.add_argument("--pseudo_query_no_reference_region_weight_support_power", type=float, default=1.0)
    parser.add_argument("--pseudo_query_no_reference_region_weight_image_scale", type=float, default=0.25)
    parser.add_argument("--pseudo_query_no_reference_support_threshold", type=float, default=0.22)
    parser.add_argument("--pseudo_query_no_reference_support_dilate_radius", type=int, default=5)
    parser.add_argument("--pseudo_query_no_reference_support_min_area", type=int, default=24)
    parser.add_argument("--pseudo_query_no_reference_invalid_min_area", type=int, default=96)
    parser.add_argument("--support_query_split", action="store_true", default=False)
    parser.add_argument("--query_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--query_split_seed", type=int, default=2025)
    parser.add_argument("--query_split_mode", type=str, default="random", choices=["random", "sequence_block", "temporal_block"])
    parser.add_argument("--support_query_sort_by_name", action="store_true", default=False)
    parser.add_argument("--query_artifact_filter_path", type=str, default="")
    parser.add_argument("--query_artifact_filter_severities", type=str, default="mild,severe")
    parser.add_argument("--query_artifact_filter_splits", type=str, default="heldout_query_sample")
    parser.add_argument("--render_artifact_weight_path", type=str, default="")
    parser.add_argument("--render_artifact_weight_splits", type=str, default="heldout_query_sample")
    parser.add_argument("--render_artifact_weight_severities", type=str, default="severe")
    parser.add_argument("--render_artifact_weight_mode", type=str, default="severity", choices=["severity", "continuous"])
    parser.add_argument("--render_artifact_weight_targets", type=str, default="teacher")
    parser.add_argument("--render_artifact_weight_default", type=float, default=1.0)
    parser.add_argument("--render_artifact_weight_mild", type=float, default=1.0)
    parser.add_argument("--render_artifact_weight_severe", type=float, default=0.70)
    parser.add_argument("--render_artifact_weight_continuous_min", type=float, default=0.70)
    parser.add_argument("--render_artifact_weight_continuous_power", type=float, default=1.0)
    parser.add_argument(
        "--render_artifact_direct_weight_combine_mode",
        type=str,
        default="product",
        choices=["product", "min", "none"],
    )
    parser.add_argument(
        "--render_artifact_direct_loss_scale_mode",
        type=str,
        default="none",
        choices=["none", "region_mean", "combined_mean"],
    )
    parser.add_argument("--render_artifact_region_weight_path", type=str, default="")
    parser.add_argument("--render_artifact_region_weight_root", type=str, default="")
    parser.add_argument("--render_artifact_region_weight_splits", type=str, default="heldout_query_sample")
    parser.add_argument("--render_artifact_region_weight_severities", type=str, default="severe")
    parser.add_argument("--render_artifact_region_weight_targets", type=str, default="direct")
    parser.add_argument("--render_artifact_region_weight_default", type=float, default=1.0)
    parser.add_argument("--loc_anchor_grid_size", type=int, default=8)
    parser.add_argument("--geometry_anchor_weight", type=float, default=0.0)
    parser.add_argument("--geometry_anchor_scale_weight", type=float, default=0.1)
    parser.add_argument("--geometry_anchor_rotation_weight", type=float, default=0.1)
    parser.add_argument("--geometry_xyz_lr_mult", type=float, default=0.05)
    parser.add_argument("--geometry_scale_lr_mult", type=float, default=0.1)
    parser.add_argument("--geometry_rotation_lr_mult", type=float, default=0.1)
    parser.add_argument("--enable_topology", action="store_true", default=False)
    parser.add_argument("--topology_stats_warmup", type=int, default=1000)
    parser.add_argument("--topology_update_interval", type=int, default=200)
    parser.add_argument("--topology_min_observations", type=int, default=8)
    parser.add_argument("--topology_split_quantile", type=float, default=0.95)
    parser.add_argument("--topology_ambiguity_quantile", type=float, default=0.90)
    parser.add_argument("--topology_growth_cap_per_event", type=float, default=0.03)
    parser.add_argument("--topology_total_point_budget_ratio", type=float, default=1.25)
    parser.add_argument("--topology_cooldown_iterations", type=int, default=300)
    parser.add_argument("--topology_disable_split", action="store_true", default=False)
    parser.add_argument("--topology_min_repeatability", type=float, default=0.25)
    parser.add_argument("--topology_min_radius", type=float, default=4.0)
    parser.add_argument("--topology_enable_soft_prune", action="store_true", default=False)
    parser.add_argument("--topology_enable_physical_prune", action="store_true", default=False)
    parser.add_argument("--topology_protect_landmarks", action="store_true", default=False)
    parser.add_argument("--topology_soft_prune_threshold", type=float, default=-1.0)
    parser.add_argument("--topology_soft_prune_step", type=float, default=1.0)
    parser.add_argument("--topology_physical_rgb_threshold", type=float, default=0.005)
    parser.add_argument("--topology_physical_loc_threshold", type=float, default=0.005)
    parser.add_argument("--topology_physical_utility_threshold", type=float, default=-3.0)
    parser.add_argument("--topology_allow_untrained_loc_opacity_prune", action="store_true", default=False)
    parser.add_argument("--topology_max_mutation_events", type=int, default=0)
    parser.add_argument(
        "--topology_risk_commit_policy",
        type=str,
        default="off",
        choices=["off", "accept_all", "reject_all", "heldout_descriptor", "heldout_pose"],
    )
    parser.add_argument("--topology_risk_holdout_size", type=int, default=4)
    parser.add_argument(
        "--topology_risk_holdout_selection",
        choices=["prefix", "strided", "pose_stratified"],
        default="prefix",
    )
    parser.add_argument("--topology_risk_epsilon", type=float, default=0.0)
    parser.add_argument("--topology_risk_ci_z", type=float, default=0.0)
    parser.add_argument("--topology_risk_min_ci_samples", type=int, default=2)
    parser.add_argument("--topology_risk_desc_weight", type=float, default=1.0)
    parser.add_argument("--topology_risk_full_bank_weight", type=float, default=1.0)
    parser.add_argument("--topology_risk_reproj_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_anchors", type=int, default=256)
    parser.add_argument("--topology_risk_pose_cfg", type=str, default="")
    parser.add_argument("--topology_risk_pose_ae_weight", type=float, default=1.0)
    parser.add_argument("--topology_risk_pose_te_weight", type=float, default=1.0)
    parser.add_argument("--topology_risk_pose_inlier_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_ae_scale", type=float, default=5.0)
    parser.add_argument("--topology_risk_pose_te_scale", type=float, default=200.0)
    parser.add_argument("--topology_risk_pose_inlier_scale", type=float, default=100.0)
    parser.add_argument("--topology_risk_pose_r5_miss_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_r2_miss_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_tail_fail_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_cvar_weight", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_cvar_fraction", type=float, default=0.25)
    parser.add_argument(
        "--topology_risk_pose_veto_mode",
        choices=["off", "r5", "r5_r2", "r5_r2_tail"],
        default="off",
    )
    parser.add_argument("--topology_risk_pose_r5_ae_threshold", type=float, default=5.0)
    parser.add_argument("--topology_risk_pose_r5_te_threshold", type=float, default=5.0)
    parser.add_argument("--topology_risk_pose_r2_ae_threshold", type=float, default=2.0)
    parser.add_argument("--topology_risk_pose_r2_te_threshold", type=float, default=2.0)
    parser.add_argument("--topology_risk_pose_tail_ae_threshold", type=float, default=10.0)
    parser.add_argument("--topology_risk_pose_tail_te_threshold", type=float, default=500.0)
    parser.add_argument("--topology_risk_pose_r2_tolerance", type=float, default=0.0)
    parser.add_argument("--topology_risk_pose_tail_tolerance", type=float, default=0.0)
    return parser


def _dense_pose_improvement_weight(meta, min_te=0.0, min_ae=0.0):
    if not meta:
        return 0.0
    required = ("te", "ae", "dense_te", "dense_ae")
    if any(meta.get(key) is None for key in required):
        return 0.0
    te_improved = float(meta["te"]) - float(meta["dense_te"]) > float(min_te)
    ae_improved = float(meta["ae"]) - float(meta["dense_ae"]) > float(min_ae)
    return 1.0 if te_improved and ae_improved else 0.0


def _dense_pose_advantage_weight(
    meta,
    min_te=0.0,
    min_ae=0.0,
    te_scale=10.0,
    ae_scale=1.0,
):
    if not meta:
        return 0.0
    required = ("te", "ae", "dense_te", "dense_ae")
    if any(meta.get(key) is None for key in required):
        return 0.0
    te_advantage = float(meta["te"]) - float(meta["dense_te"]) - float(min_te)
    ae_advantage = float(meta["ae"]) - float(meta["dense_ae"]) - float(min_ae)
    if te_advantage <= 0.0 or ae_advantage <= 0.0:
        return 0.0
    weights = []
    if float(te_scale) > 0.0:
        weights.append(te_advantage / float(te_scale))
    if float(ae_scale) > 0.0:
        weights.append(ae_advantage / float(ae_scale))
    if not weights:
        return 1.0
    return float(max(0.0, min(1.0, min(weights))))


def _dense_loss_weights_for_episode(args, synthetic_view_used=False):
    desc_weight = float(args.loc_desc_weight)
    reproj_weight = float(args.loc_reproj_weight)
    if synthetic_view_used:
        desc_weight = float(args.synthetic_view_desc_weight)
        reproj_weight = float(args.synthetic_view_reproj_weight)
    return {
        "desc": desc_weight,
        "reproj": reproj_weight,
        "kl": float(args.loc_dense_kl_weight),
        "rank": float(args.loc_dense_rank_weight),
    }


def _clone_tensor_tree(value):
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(value.detach().clone(), requires_grad=value.requires_grad)
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    return copy.deepcopy(value)


def _capture_locaware_training_state(gaussians):
    return {
        "model_params": _clone_tensor_tree(gaussians.capture()),
        "localization_state": _clone_tensor_tree(gaussians.capture_localization_state()),
        "loc_opacity_grad_seen": bool(getattr(gaussians, "loc_opacity_grad_seen", False)),
    }


def _restore_locaware_training_state(gaussians, opt, state):
    gaussians.restore(state["model_params"], opt)
    gaussians.restore_localization_state(state["localization_state"])
    gaussians.loc_opacity_grad_seen = bool(state.get("loc_opacity_grad_seen", False))


def _apply_split_proposal_trial(gaussians, proposal, scene_extent):
    if torch.as_tensor(proposal.physical_prune_mask).bool().any():
        raise RuntimeError("held-out descriptor risk currently supports split-only proposals")
    split = torch.as_tensor(
        proposal.split_mask,
        dtype=torch.bool,
        device=gaussians.get_xyz.device,
    ).reshape(-1)
    split_count = int(split.sum().item())
    if split_count == 0:
        return 0
    point_count_before = int(gaussians.get_xyz.shape[0])
    gaussians.densify_and_split_selected(
        split,
        scene_extent=scene_extent,
        N=int(getattr(proposal, "num_children_per_parent", 2)),
    )
    point_count_after = int(gaussians.get_xyz.shape[0])
    new_clone_count = point_count_after - point_count_before + split_count
    if new_clone_count > 0:
        gaussians.last_topology_iteration[-new_clone_count:] = int(proposal.iteration)
        if hasattr(gaussians, "loc_birth_iteration"):
            gaussians.loc_birth_iteration[-new_clone_count:] = int(proposal.iteration)
    return split_count


def _normalize_risk_score(score):
    metrics = {}
    if isinstance(score, dict):
        raw_metrics = score.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
        raw_values = score.get("values", score.get("risks", score.get("samples", None)))
        if raw_values is not None:
            values = _risk_score_values(raw_values)
            risk = float(score.get("risk", sum(values) / len(values) if values else float("inf")))
            return risk, values, metrics
        return float(score.get("risk", float("inf"))), [], metrics
    values = _risk_score_values(score)
    if values:
        return float(sum(values) / len(values)), values, metrics
    return float(score), [], metrics


def _risk_score_values(score):
    if torch.is_tensor(score):
        score = score.detach().flatten().cpu().tolist()
    if isinstance(score, (list, tuple)):
        values = []
        for value in score:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return []
            values.append(value)
        return values
    return []


def _paired_delta_upper_confidence_bound(baseline_values, trial_values, z):
    if len(baseline_values) != len(trial_values) or not baseline_values:
        return float("nan")
    deltas = [float(t) - float(b) for b, t in zip(baseline_values, trial_values)]
    if any(not math.isfinite(delta) for delta in deltas):
        return float("nan")
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return mean_delta
    variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (len(deltas) - 1)
    stderr = math.sqrt(max(variance, 0.0)) / math.sqrt(len(deltas))
    return mean_delta + float(z) * stderr


def _capture_rng_state():
    state = {
        "python": py_random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": None,
        "numpy": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if np is not None:
        state["numpy"] = np.random.get_state()
    return state


def _restore_rng_state(state):
    py_random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if np is not None and state.get("numpy") is not None:
        np.random.set_state(state["numpy"])


class HeldoutRiskCommitEvaluator:
    def __init__(
        self,
        score_fn,
        apply_trial_fn,
        capture_state_fn,
        restore_state_fn,
        epsilon=0.0,
        ci_z=0.0,
        min_ci_samples=2,
        reason_prefix="heldout_descriptor",
        metric_gate_fn=None,
    ):
        self.score_fn = score_fn
        self.apply_trial_fn = apply_trial_fn
        self.capture_state_fn = capture_state_fn
        self.restore_state_fn = restore_state_fn
        self.epsilon = float(epsilon)
        self.ci_z = float(ci_z)
        self.min_ci_samples = max(1, int(min_ci_samples))
        self.reason_prefix = str(reason_prefix)
        self.metric_gate_fn = metric_gate_fn

    def __call__(self, proposal, gaussians):
        rng_state = _capture_rng_state()
        state = None
        try:
            baseline_risk, baseline_values, baseline_metrics = _normalize_risk_score(self.score_fn(gaussians))
            state = self.capture_state_fn(gaussians)
            trial_risk = float("nan")
            trial_values = []
            trial_metrics = {}
            try:
                self.apply_trial_fn(gaussians, proposal)
                trial_risk, trial_values, trial_metrics = _normalize_risk_score(self.score_fn(gaussians))
            finally:
                self.restore_state_fn(gaussians, state)
        finally:
            _restore_rng_state(rng_state)
        delta = trial_risk - baseline_risk
        finite = math.isfinite(baseline_risk) and math.isfinite(trial_risk) and math.isfinite(delta)
        if not finite:
            accepted = False
            reason = f"{self.reason_prefix}_nonfinite"
            delta_ucb = float("nan")
        elif self.ci_z > 0.0:
            sample_count = min(len(baseline_values), len(trial_values))
            delta_ucb = _paired_delta_upper_confidence_bound(baseline_values, trial_values, self.ci_z)
            if sample_count < self.min_ci_samples or not math.isfinite(delta_ucb):
                accepted = False
                reason = f"{self.reason_prefix}_ci_insufficient"
            else:
                accepted = delta_ucb <= -self.epsilon
                reason = f"{self.reason_prefix}_{'ucb_decreased' if accepted else 'ucb_not_decreased'}"
        else:
            delta_ucb = float("nan")
            accepted = delta <= -self.epsilon
            reason = f"{self.reason_prefix}_{'decreased' if accepted else 'not_decreased'}"
        metric_details = {}
        if self.metric_gate_fn is not None:
            metric_ok, metric_reason, metric_details = self.metric_gate_fn(baseline_metrics, trial_metrics)
            if accepted and not metric_ok:
                accepted = False
                metric_reason = metric_reason or "metric_veto"
                reason = f"{self.reason_prefix}_{metric_reason}"
        metric_log = {}
        for key, value in (metric_details or {}).items():
            metric_log[key if key.startswith("risk_") else f"risk_{key}"] = value
        return {
            "accepted": accepted,
            "reason": reason,
            "baseline_risk": baseline_risk,
            "trial_risk": trial_risk,
            "delta_risk": delta,
            "delta_risk_ucb": delta_ucb,
            "epsilon": self.epsilon,
            "ci_z": self.ci_z,
            "risk_sample_count": int(min(len(baseline_values), len(trial_values))),
            "candidate_count": int(getattr(proposal, "candidate_count", 0)),
            "requested_split_count": int(torch.as_tensor(proposal.split_mask).bool().sum().item()),
            **metric_log,
        }


def _heldout_query_feature_map(viewpoint_cam, feature_extractor, masks=None):
    original_image = viewpoint_cam.original_image.cuda()
    with torch.no_grad():
        feature_map = feature_extractor(original_image[None])["feature_map"][0]
        feature_map = _normalize_feature_map_inplace(feature_map)
    if masks is not None:
        obj_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], feature_map.shape[-2:])
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], feature_map.shape[-2:])
        feature_map = feature_map * (obj_mask & distort_mask)
    return feature_map


def _normalize_feature_map_inplace(feature_map, eps=1e-12):
    denom = torch.linalg.vector_norm(feature_map, ord=2, dim=0, keepdim=True).clamp_min_(float(eps))
    return feature_map.div_(denom)


def _score_heldout_direct_descriptor_risk(
    gaussians,
    heldout_cameras,
    feature_extractor,
    args,
    masks,
    direct_landmark_indices,
    return_values=False,
):
    risks = []
    with torch.no_grad():
        current_direct_landmark_indices = _current_landmark_indices_from_source_index(
            direct_landmark_indices,
            gaussians,
        )
        if current_direct_landmark_indices.numel() == 0:
            return [float("inf")] if return_values else float("inf")
        for query_cam in heldout_cameras:
            query_feature_map = _heldout_query_feature_map(query_cam, feature_extractor, masks=masks)
            pose_gt = query_cam.world_view_transform.transpose(0, 1).cuda()
            teacher_out = direct_landmark_teacher(
                gaussians,
                query_feature_map,
                pose_gt,
                query_cam.FoVx,
                query_cam.FoVy,
                current_direct_landmark_indices,
                alpha_threshold=args.loc_alpha_threshold,
                max_landmarks=args.topology_risk_anchors,
                full_bank_indices=(
                    current_direct_landmark_indices
                    if float(args.topology_risk_full_bank_weight) > 0.0
                    else None
                ),
                full_bank_temperature=args.loc_full_bank_temperature,
                full_bank_hard_negative_topk=args.loc_full_bank_hard_negatives,
                full_bank_hard_negative_margin=args.loc_full_bank_margin,
                full_bank_ignore_3d_radius=args.loc_full_bank_ignore_3d_radius,
                full_bank_ignore_uv_radius=args.loc_full_bank_ignore_uv_radius,
                full_bank_source_mode=args.loc_full_bank_source_mode,
                full_bank_stats_chunk_size=getattr(args, "loc_full_bank_stats_chunk_size", 256),
                full_bank_pose_information_weight=args.loc_full_bank_pose_information_weight,
                full_bank_pose_information_floor=args.loc_full_bank_pose_information_floor,
                full_bank_balance_weight=args.loc_full_bank_balance_weight,
                full_bank_balance_grid_size=args.loc_full_bank_balance_grid_size,
                full_bank_balance_depth_bins=args.loc_full_bank_balance_depth_bins,
                full_bank_balance_max_weight=args.loc_full_bank_balance_max_weight,
                full_bank_clean_hard_negative_weight=args.loc_full_bank_clean_hard_negative_weight,
                full_bank_clean_reproj_radius=args.loc_full_bank_clean_reproj_radius,
                full_bank_clean_hard_negatives=args.loc_full_bank_clean_hard_negatives,
                sampling_grid_size=args.loc_anchor_grid_size,
                child_responsibility_mode=args.loc_child_responsibility_mode,
            )
            risk = (
                float(args.topology_risk_desc_weight) * teacher_out.desc_loss
                + float(args.topology_risk_full_bank_weight) * teacher_out.full_bank_loss
                + float(args.topology_risk_reproj_weight) * teacher_out.reproj_loss
            )
            risks.append(float(risk.detach().item()))
    if not risks:
        return [float("inf")] if return_values else float("inf")
    if return_values:
        return risks
    return float(sum(risks) / len(risks))


def _make_heldout_descriptor_risk_evaluator(
    args,
    gaussians,
    opt,
    heldout_cameras,
    feature_extractor,
    masks,
    direct_landmark_indices,
    scene_extent,
):
    risk_cameras = _select_risk_cameras(
        heldout_cameras,
        args.topology_risk_holdout_size,
        args.topology_risk_holdout_selection,
    )
    if not risk_cameras:
        raise ValueError("heldout_descriptor risk requires at least one held-out query camera")

    def score_fn(model):
        return _score_heldout_direct_descriptor_risk(
            model,
            risk_cameras,
            feature_extractor,
            args,
            masks,
            direct_landmark_indices,
            return_values=float(args.topology_risk_ci_z) > 0.0,
        )

    return HeldoutRiskCommitEvaluator(
        score_fn=score_fn,
        apply_trial_fn=lambda model, proposal: _apply_split_proposal_trial(model, proposal, scene_extent),
        capture_state_fn=_capture_locaware_training_state,
        restore_state_fn=lambda model, state: _restore_locaware_training_state(model, opt, state),
        epsilon=args.topology_risk_epsilon,
        ci_z=args.topology_risk_ci_z,
        min_ci_samples=args.topology_risk_min_ci_samples,
        reason_prefix="heldout_descriptor",
    )


def _strided_sample_indices(count, sample_count):
    if count <= 0:
        return []
    sample_count = max(1, int(sample_count))
    if sample_count >= count:
        return list(range(count))
    if sample_count == 1:
        return [count // 2]
    last = count - 1
    return [int(round(i * last / (sample_count - 1))) for i in range(sample_count)]


def _camera_center_tensor(camera):
    center = getattr(camera, "camera_center", None)
    if center is None:
        return None
    try:
        tensor = torch.as_tensor(center).detach().cpu().float().flatten()
    except Exception:
        return None
    if tensor.numel() < 3:
        return None
    tensor = tensor[:3]
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def _pose_stratified_sample_indices(cameras, sample_count):
    centers = [_camera_center_tensor(camera) for camera in cameras]
    if any(center is None for center in centers):
        return _strided_sample_indices(len(cameras), sample_count)
    center_tensor = torch.stack(centers, dim=0)
    centered = center_tensor - center_tensor.mean(dim=0, keepdim=True)
    if float(centered.norm().item()) <= 1e-12:
        return _strided_sample_indices(len(cameras), sample_count)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        major_axis = int(torch.argmax(torch.abs(axis)).item())
        if float(axis[major_axis].item()) < 0.0:
            axis = -axis
        scores = torch.matmul(centered, axis).tolist()
    except Exception:
        return _strided_sample_indices(len(cameras), sample_count)
    sorted_indices = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (item[1], item[0]))]
    sampled_positions = _strided_sample_indices(len(sorted_indices), sample_count)
    return [sorted_indices[pos] for pos in sampled_positions]


def _select_risk_cameras(heldout_cameras, holdout_size, selection):
    cameras = list(heldout_cameras)
    if not cameras:
        return []
    holdout_size = max(1, int(holdout_size))
    if holdout_size >= len(cameras):
        return cameras
    mode = str(selection)
    if mode == "prefix":
        return cameras[:holdout_size]
    if mode == "strided":
        indices = _strided_sample_indices(len(cameras), holdout_size)
        return [cameras[idx] for idx in indices]
    if mode == "pose_stratified":
        indices = _pose_stratified_sample_indices(cameras, holdout_size)
        return [cameras[idx] for idx in indices]
    raise ValueError(f"unknown topology risk holdout selection: {selection}")


def _pose_risk_from_sparse_metrics(ae_deg, te_cm, inliers, args):
    ae = float(ae_deg)
    te = float(te_cm)
    inlier_count = float(inliers)
    if not (math.isfinite(ae) and math.isfinite(te) and math.isfinite(inlier_count)):
        return float("inf")
    ae_scale = max(float(args.topology_risk_pose_ae_scale), 1e-6)
    te_scale = max(float(args.topology_risk_pose_te_scale), 1e-6)
    inlier_scale = max(float(args.topology_risk_pose_inlier_scale), 1e-6)
    inlier_reward = min(max(inlier_count, 0.0) / inlier_scale, 1.0)
    risk = (
        float(args.topology_risk_pose_ae_weight) * (ae / ae_scale)
        + float(args.topology_risk_pose_te_weight) * (te / te_scale)
        - float(args.topology_risk_pose_inlier_weight) * inlier_reward
    )
    r5_ae = float(getattr(args, "topology_risk_pose_r5_ae_threshold", 5.0))
    r5_te = float(getattr(args, "topology_risk_pose_r5_te_threshold", 5.0))
    r2_ae = float(getattr(args, "topology_risk_pose_r2_ae_threshold", 2.0))
    r2_te = float(getattr(args, "topology_risk_pose_r2_te_threshold", 2.0))
    tail_ae = float(getattr(args, "topology_risk_pose_tail_ae_threshold", 10.0))
    tail_te = float(getattr(args, "topology_risk_pose_tail_te_threshold", 500.0))
    r5_miss_weight = max(float(getattr(args, "topology_risk_pose_r5_miss_weight", 0.0)), 0.0)
    r2_miss_weight = max(float(getattr(args, "topology_risk_pose_r2_miss_weight", 0.0)), 0.0)
    tail_fail_weight = max(float(getattr(args, "topology_risk_pose_tail_fail_weight", 0.0)), 0.0)
    if r5_miss_weight > 0.0 and not (ae <= r5_ae and te <= r5_te):
        risk += r5_miss_weight
    if r2_miss_weight > 0.0 and not (ae <= r2_ae and te <= r2_te):
        risk += r2_miss_weight
    if tail_fail_weight > 0.0 and (ae > tail_ae or te > tail_te):
        risk += tail_fail_weight
    return float(risk)


def _aggregate_pose_risk_values(risks, args, weights=None):
    values = [float(v) for v in risks]
    if not values or any(not math.isfinite(v) for v in values):
        return float("inf")
    mean_risk = weighted_mean(values, weights)
    cvar_weight = max(float(getattr(args, "topology_risk_pose_cvar_weight", 0.0)), 0.0)
    if cvar_weight <= 0.0:
        return mean_risk
    cvar_fraction = float(getattr(args, "topology_risk_pose_cvar_fraction", 0.25))
    cvar_fraction = min(max(cvar_fraction, 1.0 / len(values)), 1.0)
    tail_count = max(1, int(math.ceil(len(values) * cvar_fraction)))
    if weights is None:
        tail_values = sorted(values, reverse=True)[:tail_count]
        tail_mean = float(sum(tail_values) / len(tail_values))
    else:
        weights = [max(0.0, float(weight)) for weight in weights[: len(values)]]
        pairs = sorted(zip(values, weights), key=lambda item: item[0], reverse=True)[:tail_count]
        tail_values = [value for value, _ in pairs]
        tail_weights = [weight for _, weight in pairs]
        tail_mean = weighted_mean(tail_values, tail_weights)
    return float(mean_risk + cvar_weight * tail_mean)


def _metric_rate(count, total):
    total = int(total)
    if total <= 0:
        return float("nan")
    return float(count) / float(total)


def _pose_metric_summary(ae_values, te_values, inlier_values, args):
    count = min(len(ae_values), len(te_values), len(inlier_values))
    ae_values = [float(v) for v in ae_values[:count]]
    te_values = [float(v) for v in te_values[:count]]
    inlier_values = [float(v) for v in inlier_values[:count]]
    r5_ae = float(getattr(args, "topology_risk_pose_r5_ae_threshold", 5.0))
    r5_te = float(getattr(args, "topology_risk_pose_r5_te_threshold", 5.0))
    r2_ae = float(getattr(args, "topology_risk_pose_r2_ae_threshold", 2.0))
    r2_te = float(getattr(args, "topology_risk_pose_r2_te_threshold", 2.0))
    tail_ae = float(getattr(args, "topology_risk_pose_tail_ae_threshold", 10.0))
    tail_te = float(getattr(args, "topology_risk_pose_tail_te_threshold", 500.0))
    r5_count = sum(1 for ae, te in zip(ae_values, te_values) if ae <= r5_ae and te <= r5_te)
    r2_count = sum(1 for ae, te in zip(ae_values, te_values) if ae <= r2_ae and te <= r2_te)
    tail_fail_count = sum(1 for ae, te in zip(ae_values, te_values) if ae > tail_ae or te > tail_te)
    avg_inliers = sum(inlier_values) / count if count > 0 else float("nan")
    return {
        "count": int(count),
        "r5_count": int(r5_count),
        "r2_count": int(r2_count),
        "tail_fail_count": int(tail_fail_count),
        "r5_rate": _metric_rate(r5_count, count),
        "r2_rate": _metric_rate(r2_count, count),
        "tail_fail_rate": _metric_rate(tail_fail_count, count),
        "avg_inliers": float(avg_inliers),
    }


def _pose_recall_tail_veto(baseline_metrics, trial_metrics, args):
    mode = str(getattr(args, "topology_risk_pose_veto_mode", "off"))
    if mode == "off":
        return True, "", {}
    baseline_count = int(baseline_metrics.get("count", 0) or 0)
    trial_count = int(trial_metrics.get("count", 0) or 0)
    count = min(baseline_count, trial_count)
    details = {
        "metric_count": int(count),
        "r5_delta": int(trial_metrics.get("r5_count", 0) or 0) - int(baseline_metrics.get("r5_count", 0) or 0),
        "r2_delta": int(trial_metrics.get("r2_count", 0) or 0) - int(baseline_metrics.get("r2_count", 0) or 0),
        "tail_fail_delta": int(trial_metrics.get("tail_fail_count", 0) or 0)
        - int(baseline_metrics.get("tail_fail_count", 0) or 0),
    }
    if count <= 0:
        return False, "metrics_missing", details
    baseline_r5 = _metric_rate(baseline_metrics.get("r5_count", 0), baseline_count)
    trial_r5 = _metric_rate(trial_metrics.get("r5_count", 0), trial_count)
    baseline_r2 = _metric_rate(baseline_metrics.get("r2_count", 0), baseline_count)
    trial_r2 = _metric_rate(trial_metrics.get("r2_count", 0), trial_count)
    baseline_tail = _metric_rate(baseline_metrics.get("tail_fail_count", 0), baseline_count)
    trial_tail = _metric_rate(trial_metrics.get("tail_fail_count", 0), trial_count)
    details.update(
        {
            "r5_rate_delta": trial_r5 - baseline_r5,
            "r2_rate_delta": trial_r2 - baseline_r2,
            "tail_fail_rate_delta": trial_tail - baseline_tail,
        }
    )
    if mode in {"r5", "r5_r2", "r5_r2_tail"} and trial_r5 < baseline_r5 - 1e-12:
        return False, "r5_decreased", details
    r2_tolerance = max(float(getattr(args, "topology_risk_pose_r2_tolerance", 0.0)), 0.0)
    if mode in {"r5_r2", "r5_r2_tail"} and trial_r2 < baseline_r2 - r2_tolerance - 1e-12:
        return False, "r2_decreased", details
    tail_tolerance = max(float(getattr(args, "topology_risk_pose_tail_tolerance", 0.0)), 0.0)
    if mode == "r5_r2_tail" and trial_tail > baseline_tail + tail_tolerance + 1e-12:
        return False, "tail_increased", details
    return True, "", details


def _make_sparse_pose_risk_config(args, dataset):
    if not args.topology_risk_pose_cfg:
        raise ValueError("heldout_pose risk requires --topology_risk_pose_cfg")
    with open(args.topology_risk_pose_cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config.setdefault("sparse", {})["sparse_only"] = True
    config["sparse"]["use_landmark_prior"] = False
    config.setdefault("dense", {})["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    return config


def _refresh_stdloc_sparse_landmarks(stdloc, gaussians, source_landmark_indices):
    from stdloc import sample_gaussians

    current_indices = _current_landmark_indices_from_source_index(
        source_landmark_indices,
        gaussians,
    )
    if current_indices.numel() == 0:
        return 0
    stdloc.landmark_indices = current_indices.detach().cpu()
    stdloc.landmarks = sample_gaussians(gaussians, stdloc.landmark_indices)
    stdloc.landmark_meta = None
    return int(current_indices.numel())


def _score_heldout_sparse_pose_risk(
    gaussians,
    stdloc,
    heldout_cameras,
    args,
    direct_landmark_indices,
    artifact_weight_lookup=None,
    return_values=False,
    return_metrics=False,
):
    if _refresh_stdloc_sparse_landmarks(stdloc, gaussians, direct_landmark_indices) == 0:
        if return_values or return_metrics:
            output = {"risk": float("inf")}
            if return_values:
                output["values"] = [float("inf")]
            if return_metrics:
                output["metrics"] = _pose_metric_summary([], [], [], args)
            return output
        return float("inf")
    risks = []
    ae_values = []
    te_values = []
    inlier_values = []
    risk_weights = []
    with torch.no_grad():
        for query_cam in heldout_cameras:
            query_image = query_cam.original_image.to("cuda")
            loc_res = stdloc.localize(query_image, query_cam.FoVx, query_cam.FoVy)
            gt_w2c = query_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
            sparse_res = loc_res["sparse"]
            ae, te = cal_pose_error(sparse_res["pose_w2c"], gt_w2c)
            inliers = sparse_res.get("inliers", 0)
            risk = _pose_risk_from_sparse_metrics(
                ae,
                te,
                inliers,
                args,
            )
            ae_values.append(float(ae))
            te_values.append(float(te))
            inlier_values.append(float(inliers))
            risks.append(float(risk))
            if artifact_weight_lookup is not None:
                risk_weights.append(artifact_weight_lookup.weight_for_camera(query_cam))
    if not risks:
        if return_values or return_metrics:
            output = {"risk": float("inf")}
            if return_values:
                output["values"] = [float("inf")]
            if return_metrics:
                output["metrics"] = _pose_metric_summary([], [], [], args)
            return output
        return float("inf")
    risk_mean = _aggregate_pose_risk_values(
        risks,
        args,
        weights=risk_weights if artifact_weight_lookup is not None else None,
    )
    if return_values or return_metrics:
        output = {"risk": risk_mean}
        if return_values:
            output["values"] = risks
        if return_metrics:
            output["metrics"] = _pose_metric_summary(ae_values, te_values, inlier_values, args)
        return output
    return risk_mean


def _make_heldout_pose_risk_evaluator(
    args,
    dataset,
    gaussians,
    opt,
    heldout_cameras,
    direct_landmark_indices,
    scene_extent,
    artifact_weight_lookup=None,
):
    from stdloc import STDLoc

    risk_cameras = _select_risk_cameras(
        heldout_cameras,
        args.topology_risk_holdout_size,
        args.topology_risk_holdout_selection,
    )
    if not risk_cameras:
        raise ValueError("heldout_pose risk requires at least one held-out query camera")
    config = _make_sparse_pose_risk_config(args, dataset)
    stdloc = STDLoc(gaussians, config)
    use_metric_veto = str(args.topology_risk_pose_veto_mode) != "off"

    def score_fn(model):
        return _score_heldout_sparse_pose_risk(
            model,
            stdloc,
            risk_cameras,
            args,
            direct_landmark_indices,
            artifact_weight_lookup=artifact_weight_lookup,
            return_values=float(args.topology_risk_ci_z) > 0.0,
            return_metrics=use_metric_veto,
        )

    return HeldoutRiskCommitEvaluator(
        score_fn=score_fn,
        apply_trial_fn=lambda model, proposal: _apply_split_proposal_trial(model, proposal, scene_extent),
        capture_state_fn=_capture_locaware_training_state,
        restore_state_fn=lambda model, state: _restore_locaware_training_state(model, opt, state),
        epsilon=args.topology_risk_epsilon,
        ci_z=args.topology_risk_ci_z,
        min_ci_samples=args.topology_risk_min_ci_samples,
        reason_prefix="heldout_pose",
        metric_gate_fn=(
            (lambda baseline, trial: _pose_recall_tail_veto(baseline, trial, args))
            if use_metric_veto
            else None
        ),
    )


def _base_losses(viewpoint_cam, render_pkg, feature_extractor, dataset, masks=None):
    image = render_pkg["render"]
    feature_map = render_pkg["feature_map"]
    original_image = viewpoint_cam.original_image.cuda()
    gt_image = F.interpolate(
        original_image.unsqueeze(0),
        size=(image.shape[1], image.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mask = None
    sky_mask = None
    if masks is not None:
        obj_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], image.shape[-2:])
        sky_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][1].cuda()[None], image.shape[-2:])
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], image.shape[-2:])
        mask = obj_mask & distort_mask
        image = image * mask
        gt_image = gt_image * mask
        gt_image[sky_mask.repeat(3, 1, 1) == False] = 1

    Ll1 = l1_loss(image, gt_image)
    if feature_map is None:
        Ll1_feature = image.new_tensor(0.0)
        gt_feature_map = None
    else:
        with torch.no_grad():
            gt_feature_map = feature_extractor(original_image[None])["feature_map"][0]
            gt_feature_map = F.interpolate(
                gt_feature_map.unsqueeze(0),
                size=(feature_map.shape[1], feature_map.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
        if mask is not None:
            feature_map_mask = _resize_bool_mask(mask, (gt_feature_map.shape[1], gt_feature_map.shape[2]))
            feature_map = feature_map * feature_map_mask
            gt_feature_map = gt_feature_map * feature_map_mask
        Ll1_feature = l1_loss(feature_map, gt_feature_map)

    return {
        "image": image,
        "gt_image": gt_image,
        "gt_feature_map": gt_feature_map,
        "Ll1": Ll1,
        "Ll1_feature": Ll1_feature,
    }


def _query_feature_map(viewpoint_cam, feature_extractor, target_hw, masks=None):
    original_image = viewpoint_cam.original_image.cuda()
    with torch.no_grad():
        gt_feature_map = feature_extractor(original_image[None])["feature_map"][0]
        gt_feature_map = F.interpolate(
            gt_feature_map.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = _normalize_feature_map_inplace(gt_feature_map)
    if masks is not None and getattr(viewpoint_cam, "image_name", "") in masks:
        obj_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], target_hw)
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], target_hw)
        gt_feature_map = gt_feature_map * (obj_mask & distort_mask)
    return gt_feature_map


def _scale_chw_image(image, scale):
    scale = float(scale or 1.0)
    if abs(scale - 1.0) < 1e-6:
        return image
    height, width = image.shape[-2:]
    target_hw = (
        max(8, int(round(height * scale))),
        max(8, int(round(width * scale))),
    )
    return F.interpolate(
        image.float()[None],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]


def _combine_region_weight_maps(first, second):
    if first is None:
        return second
    if second is None:
        return first
    first = torch.as_tensor(first, dtype=torch.float32)
    second = torch.as_tensor(second, dtype=torch.float32, device=first.device)
    while first.dim() > 2:
        first = first.squeeze(0)
    while second.dim() > 2:
        second = second.squeeze(0)
    if first.shape != second.shape:
        first = F.interpolate(
            first[None, None],
            size=second.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return (first.to(device=second.device, dtype=second.dtype) * second).clamp(0.0, 1.0)


def _pseudo_query_no_reference_region_weight(
    record,
    query_image,
    enabled=False,
    allowed_sources=None,
    min_weight=0.25,
    support_power=1.0,
    image_scale=1.0,
    builder=None,
):
    if not enabled:
        return None, {"enabled": False, "reason": "disabled"}
    allowed_sources = {str(item) for item in (allowed_sources or []) if str(item)}
    if allowed_sources and getattr(record, "source", "") not in allowed_sources:
        return None, {"enabled": False, "reason": "source_not_enabled"}
    builder = builder or NoReferenceValidMaskBuilder()
    image = torch.as_tensor(query_image).detach().cpu().float()
    if image.dim() == 4 and image.shape[0] == 1:
        image = image[0]
    image = _scale_chw_image(image, image_scale)
    result = builder.build(image)
    support = result.support_score.detach().float().clamp(0.0, 1.0)
    power = float(support_power or 1.0)
    if abs(power - 1.0) > 1e-6:
        support = support.pow(power)
    min_weight = max(0.0, min(float(min_weight), 1.0))
    weight_map = min_weight + (1.0 - min_weight) * support
    weight_map = weight_map * result.valid_mask.detach().float()
    weight_map = weight_map.clamp(0.0, 1.0)
    summary = {
        "enabled": True,
        "reason": "ok",
        "mode": "no_reference_region_weight",
        **result.summary,
        "region_weight_min": float(weight_map.min().item()) if weight_map.numel() else 1.0,
        "region_weight_mean": float(weight_map.mean().item()) if weight_map.numel() else 1.0,
        "region_weight_max": float(weight_map.max().item()) if weight_map.numel() else 1.0,
        "region_weight_min_config": min_weight,
        "support_power": power,
        "image_scale": float(image_scale or 1.0),
    }
    return weight_map, summary


def _pseudo_record_to_camera(record, train_camera_by_name):
    if record.source == "train_rgb":
        camera = train_camera_by_name.get(_normalize_image_name(record.image_name))
        if camera is not None:
            camera.teacher_cache_key = record.teacher_cache_key or record.query_id
            camera.pseudo_query_source = record.source
            camera.pseudo_query_artifact_score = float(record.artifact_score)
            return camera
    camera = record.to_camera(device="cpu")
    camera.teacher_cache_key = record.teacher_cache_key or record.query_id
    camera.pseudo_query_source = record.source
    camera.pseudo_query_artifact_score = float(record.artifact_score)
    return camera


def _synthetic_query_feature_map(
    gaussians,
    pose_w2c,
    fovx,
    fovy,
    target_hw,
    background,
    feature_extractor=None,
    feature_source="loc_feature",
    norm_feat_bf_render=True,
    alpha_threshold=0.2,
):
    feature_source = str(feature_source)
    with torch.no_grad():
        render_pkg = render_from_pose_gsplat(
            gaussians,
            pose_w2c,
            fovx,
            fovy,
            target_hw[1],
            target_hw[0],
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=feature_source == "rgb",
            norm_feat_bf_render=norm_feat_bf_render,
            rasterize_mode="antialiased",
        )
        if feature_source == "rgb":
            if feature_extractor is None:
                return None, 0.0
            feature_map = feature_extractor(render_pkg["render"].clamp(0.0, 1.0)[None])["feature_map"][0]
            feature_map = F.interpolate(
                feature_map.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            feature_map = _normalize_feature_map_inplace(feature_map.detach())
        else:
            feature_map = render_pkg.get("feature_map")
            if feature_map is None:
                return None, 0.0
            feature_map = _normalize_feature_map_inplace(feature_map.detach())
        alpha = _flatten_render_alpha(render_pkg.get("alphas"))
        if alpha is None:
            return feature_map, 1.0
        alpha = alpha.to(device=feature_map.device, dtype=feature_map.dtype)
        valid = alpha > float(alpha_threshold)
        observability = float(valid.float().mean().item()) if valid.numel() else 0.0
        if valid.shape == feature_map.shape[-2:]:
            feature_map = feature_map * valid.to(dtype=feature_map.dtype).unsqueeze(0)
        return feature_map, observability


def _sample_synthetic_query(
    cameras,
    gaussians,
    target_hw,
    background,
    args,
    feature_extractor=None,
    norm_feat_bf_render=True,
):
    best = None
    best_score = -1.0
    for _ in range(max(1, int(args.synthetic_view_candidates))):
        try:
            candidate = sample_interpolated_novel_view(
                cameras,
                alpha_min=args.synthetic_view_alpha_min,
                alpha_max=args.synthetic_view_alpha_max,
            )
        except ValueError:
            return None, None, {}
        pose_gt = candidate.world_view_transform.transpose(0, 1).cuda()
        feature_map, observability = _synthetic_query_feature_map(
            gaussians,
            pose_gt,
            candidate.FoVx,
            candidate.FoVy,
            target_hw,
            background,
            feature_extractor=feature_extractor,
            feature_source=args.lafgs_synthetic_feature_source,
            norm_feat_bf_render=norm_feat_bf_render,
            alpha_threshold=args.loc_alpha_threshold,
        )
        if feature_map is None:
            continue
        score = float(candidate.difficulty * candidate.coverage * observability)
        if observability >= float(args.synthetic_view_min_observability) and score > best_score:
            best = candidate
            best_score = score
            diagnostics = {
                "synthetic_view_used": 1.0,
                "synthetic_view_observability": observability,
                "synthetic_view_score": score,
                "synthetic_view_alpha": float(candidate.alpha),
            }
            best = (candidate, feature_map, diagnostics)
    if best is None:
        return None, None, {"synthetic_view_used": 0.0}
    return best


def training(dataset, opt, args):
    print(opt)
    tb_writer = prepare_output_and_logger(dataset)
    training_args_path = write_training_args_snapshot(dataset, opt, args)
    print("Feature type:", dataset.feature_type)
    print("Gaussian type:", dataset.gaussian_type)
    gaussians = _gaussian_model_for_type(dataset.gaussian_type, dataset.sh_degree)
    _configure_surface_localization_anchor(gaussians, args, opt)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)
    masks = _load_masks(dataset)
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    first_iter = _restore_checkpoint(gaussians, opt, args.start_checkpoint)
    _configure_surface_localization_anchor(gaussians, args, opt)
    if args.localization_state_path:
        _restore_external_localization_state(gaussians, args.localization_state_path)
        _configure_surface_localization_anchor(gaussians, args, opt)
        print(f"Loaded external localization state from {args.localization_state_path}")
    _validate_lafgs_surface_geometry_config(gaussians, args)
    if first_iter == 0 and scene.loaded_iter:
        first_iter = scene.loaded_iter
    geometry_anchor = _capture_geometry_anchor(gaussians)
    loc_feature_anchor = _capture_feature_anchor(gaussians) if args.loc_anchor_weight > 0 else None
    sparse_pose_cache = _load_training_pose_cache(args)
    episode_sampler = EpisodeSampler(
        sparse_pose_cache=sparse_pose_cache,
        query_mode=args.query_mode,
        noise_quantile=args.pose_noise_quantile,
        mixed_sparse_probability=args.mixed_sparse_probability,
        noise_sampling=args.pose_noise_sampling,
        exclude_sparse_failure_stages=args.pseudo_query_exclude_sparse_failure_stages,
    )
    direct_landmark_indices = None
    direct_observation_memory = None
    if args.loc_teacher == "direct":
        direct_landmark_indices = _load_landmark_indices(
            dataset.model_path,
            args.landmark_path,
            device="cpu",
            point_count=gaussians.get_xyz.shape[0],
        )
        print(f"Loaded {direct_landmark_indices.numel()} direct teacher landmarks from {args.landmark_path}")
        if args.loc_multiview_weight > 0:
            feature_dim = gaussians.get_loc_feature.reshape(gaussians.get_xyz.shape[0], -1).shape[1]
            memory_landmark_indices = torch.unique(
                stable_landmark_memory_indices(gaussians, direct_landmark_indices),
                sorted=True,
            )
            direct_observation_memory = LandmarkObservationMemory(
                memory_landmark_indices,
                feature_dim=feature_dim,
                slots=args.loc_multiview_slots,
                device=gaussians.get_xyz.device,
            )
            print(
                "Initialized direct multi-view memory: "
                f"landmarks={direct_landmark_indices.numel()} "
                f"stable_sources={memory_landmark_indices.numel()} "
                f"slots={args.loc_multiview_slots}"
            )
    if args.loc_overlay_mode == "descriptor":
        if direct_landmark_indices is None:
            direct_landmark_indices = _load_landmark_indices(
                dataset.model_path,
                args.landmark_path,
                device="cpu",
                point_count=gaussians.get_xyz.shape[0],
            )
            print(f"Loaded {direct_landmark_indices.numel()} descriptor overlay landmarks from {args.landmark_path}")
        _configure_descriptor_overlay(gaussians, args, direct_landmark_indices=direct_landmark_indices)
        print(
            "Initialized descriptor overlay: "
            f"sources={gaussians.loc_overlay_source_index.numel()} "
            f"lr={args.loc_overlay_lr if args.loc_overlay_lr > 0 else opt.loc_feature_lr}"
        )
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras().copy()
    if args.support_query_sort_by_name:
        train_cameras = sorted(
            train_cameras,
            key=lambda camera: _normalize_image_name(getattr(camera, "image_name", "")),
        )
    if args.support_query_split:
        support_cameras, query_cameras = split_support_query_cameras(
            train_cameras,
            query_ratio=args.query_holdout_ratio,
            seed=args.query_split_seed,
            mode=args.query_split_mode,
        )
        print(
            "Support/query split enabled: "
            f"support={len(support_cameras)} query={len(query_cameras)} "
            f"query_ratio={args.query_holdout_ratio} "
            f"query_split_seed={args.query_split_seed} "
            f"query_split_mode={args.query_split_mode} "
            f"sort_by_name={args.support_query_sort_by_name}"
        )
    else:
        support_cameras = train_cameras
        query_cameras = train_cameras
    mvinit_summary = {
        "mvinit_enabled": bool(getattr(args, "lafgs_mvinit_enabled", False)),
        "mvinit_requested_max_views": int(getattr(args, "lafgs_mvinit_max_views", 0)),
        "mvinit_view_selection": str(getattr(args, "lafgs_mvinit_view_selection", "first")),
        "mvinit_feature_scale": float(getattr(args, "lafgs_mvinit_feature_scale", 1.0) or 1.0),
        "mvinit_used_views": 0,
        "mvinit_observed_gaussians": 0,
        "mvinit_mean_observations": 0.0,
        "mvinit_skipped_resume": 0,
    }
    if lafgs_should_run_multiview_initialization(args, first_iter=first_iter):
        mvinit_cameras = select_multiview_init_cameras(
            support_cameras,
            max_views=args.lafgs_mvinit_max_views,
            mode=args.lafgs_mvinit_view_selection,
        )
        print(
            "Running LaFGS MVInit: "
            f"views={len(mvinit_cameras)} "
            f"selection={args.lafgs_mvinit_view_selection} "
            f"min_observations={args.lafgs_mvinit_min_observations}"
        )

        mvinit_feature_scale = max(float(getattr(args, "lafgs_mvinit_feature_scale", 1.0) or 1.0), 1e-3)

        def _mvinit_feature_map(camera):
            target_hw = (
                max(8, int(round(camera.image_height * mvinit_feature_scale))),
                max(8, int(round(camera.image_width * mvinit_feature_scale))),
            )
            return _query_feature_map(
                camera,
                feature_extractor,
                target_hw=target_hw,
                masks=masks,
            )

        mvinit_result = build_multiview_initialization(
            gaussians,
            mvinit_cameras,
            _mvinit_feature_map,
            config=MultiViewInitConfig(
                min_observations=args.lafgs_mvinit_min_observations,
                chunk_size=args.lafgs_mvinit_chunk_size,
            ),
        )
        apply_multiview_initialization(gaussians, mvinit_result)
        if args.loc_anchor_weight > 0:
            loc_feature_anchor = _capture_feature_anchor(gaussians)
        print(
            "LaFGS MVInit complete: "
            f"observed={mvinit_result.diagnostics.get('observed_gaussians', 0)} "
            f"mean_obs={mvinit_result.diagnostics.get('mean_observations', 0.0):.3f}"
        )
        mvinit_summary.update(
            {
                "mvinit_used_views": len(mvinit_cameras),
                "mvinit_observed_gaussians": int(mvinit_result.diagnostics.get("observed_gaussians", 0)),
                "mvinit_mean_observations": float(mvinit_result.diagnostics.get("mean_observations", 0.0)),
            }
        )
    elif (
        bool(getattr(args, "lafgs_mvinit_enabled", False))
        and int(getattr(args, "lafgs_mvinit_max_views", 0) or 0) != 0
    ):
        mvinit_summary["mvinit_skipped_resume"] = 1
        print(
            "Skipping LaFGS MVInit on resume: "
            f"first_iter={first_iter} "
            f"stage_schedule={getattr(args, 'lafgs_stage_schedule', 'none')}"
        )
    geometry_delta_reference = _capture_geometry_delta_reference(gaussians)
    pseudo_query_sampler = None
    pseudo_query_reliability_stats = None
    train_camera_by_name = {
        _normalize_image_name(getattr(camera, "image_name", "")): camera
        for camera in train_cameras
    }
    if args.pseudo_query_manifest:
        pseudo_manifest = PseudoQueryManifest.load(args.pseudo_query_manifest).accepted(
            sources=artifact_comma_set(args.pseudo_query_sources, lower=False)
        )
        pseudo_manifest, cache_alignment_summary = _align_pseudo_manifest_to_teacher_cache(
            pseudo_manifest,
            sparse_pose_cache,
            enabled=bool(args.pseudo_query_require_teacher_cache),
        )
        if cache_alignment_summary.get("enabled", False):
            print(
                "Pseudo-query teacher-cache alignment: "
                f"before={cache_alignment_summary['before']} "
                f"after={cache_alignment_summary['after']} "
                f"dropped_missing={cache_alignment_summary['dropped_missing_teacher_cache']}"
            )
        elif cache_alignment_summary.get("reason"):
            print(
                "Pseudo-query teacher-cache alignment skipped: "
                f"reason={cache_alignment_summary['reason']} "
                f"records={cache_alignment_summary['before']}"
            )
        if sparse_pose_cache is not None and bool(args.pseudo_query_filter_teacher_cache):
            before_counts = pseudo_manifest.source_counts()
            allowed_stages = artifact_comma_set(args.pseudo_query_teacher_allowed_stages, lower=False)
            pseudo_manifest = pseudo_manifest.filter_by_teacher_cache(
                sparse_pose_cache,
                max_sparse_te=float(args.pseudo_query_teacher_max_sparse_te),
                max_dense_te=float(args.pseudo_query_teacher_max_dense_te),
                allowed_stages=allowed_stages or None,
            )
            print(
                "Pseudo-query teacher-cache filter: "
                f"before={before_counts} after={pseudo_manifest.source_counts()} "
                f"max_sparse_te={args.pseudo_query_teacher_max_sparse_te} "
                f"max_dense_te={args.pseudo_query_teacher_max_dense_te} "
                f"allowed_stages={','.join(allowed_stages) if allowed_stages else '<any>'}"
            )
        if int(args.pseudo_query_max_synthetic) > 0:
            kept = []
            synthetic_seen = 0
            for record in pseudo_manifest.records:
                if record.source == "synthetic_rgb":
                    if synthetic_seen >= int(args.pseudo_query_max_synthetic):
                        continue
                    synthetic_seen += 1
                kept.append(record)
            pseudo_manifest = PseudoQueryManifest(version=pseudo_manifest.version, records=kept)
        pseudo_query_sampler = PseudoQuerySampler(
            pseudo_manifest.records,
            real_weight=args.pseudo_query_real_weight,
            synthetic_weight=args.pseudo_query_synthetic_weight,
            seed=args.train_seed,
            sampling_mode=args.pseudo_query_sampling_mode,
        )
        print(
            "Pseudo-query manifest enabled: "
            f"path={args.pseudo_query_manifest} "
            f"counts={pseudo_manifest.source_counts()} "
            f"real_weight={args.pseudo_query_real_weight} "
            f"synthetic_weight={args.pseudo_query_synthetic_weight} "
            f"sampling_mode={args.pseudo_query_sampling_mode}"
        )
        if sparse_pose_cache is not None and args.pseudo_query_reliability_mode != "none":
            pseudo_query_reliability_stats = _pseudo_teacher_cache_reliability_stats(sparse_pose_cache)
            global_stats = pseudo_query_reliability_stats.get("__global__", {})
            print(
                "Pseudo-query reliability weighting enabled: "
                f"mode={args.pseudo_query_reliability_mode} "
                f"global_median_te={global_stats.get('median_final_te', 0.0):.3f} "
                f"global_median_inliers={global_stats.get('median_inliers', 0.0):.1f} "
                f"memory_min_weight={args.pseudo_query_reliability_memory_min_weight:.3f}"
            )
    pseudo_no_reference_region_weight_sources = artifact_comma_set(
        args.pseudo_query_no_reference_region_weight_sources,
        lower=False,
    )
    pseudo_no_reference_region_weight_builder = None
    if bool(args.pseudo_query_no_reference_region_weight):
        pseudo_no_reference_region_weight_builder = NoReferenceValidMaskBuilder(
            NoReferenceValidMaskConfig(
                support_threshold=float(args.pseudo_query_no_reference_support_threshold),
                support_dilate_radius=int(args.pseudo_query_no_reference_support_dilate_radius),
                support_min_area=int(args.pseudo_query_no_reference_support_min_area),
                invalid_min_area=int(args.pseudo_query_no_reference_invalid_min_area),
            )
        )
        print(
            "Pseudo-query no-reference region weighting enabled: "
            f"sources={','.join(sorted(pseudo_no_reference_region_weight_sources))} "
            f"min={args.pseudo_query_no_reference_region_weight_min:.3f} "
            f"support_power={args.pseudo_query_no_reference_region_weight_support_power:.3f} "
            f"image_scale={args.pseudo_query_no_reference_region_weight_image_scale:.3f}"
        )
    if args.query_artifact_filter_path:
        scene_name = os.path.basename(os.path.normpath(dataset.source_path))
        artifact_names = _load_query_artifact_filter_names(
            args.query_artifact_filter_path,
            scene_name=scene_name,
            severities=args.query_artifact_filter_severities,
            splits=args.query_artifact_filter_splits,
        )
        original_query_count = len(query_cameras)
        query_cameras, removed_query_artifacts = _filter_query_cameras_by_artifacts(query_cameras, artifact_names)
        print(
            "Query artifact filter enabled: "
            f"path={args.query_artifact_filter_path} "
            f"scene={scene_name} "
            f"severities={args.query_artifact_filter_severities} "
            f"splits={args.query_artifact_filter_splits} "
            f"matched={len(artifact_names)} "
            f"removed={len(removed_query_artifacts)} "
            f"query={len(query_cameras)}/{original_query_count}"
        )
    artifact_weight_lookup = None
    artifact_weight_targets = {item.lower() for item in _comma_set(args.render_artifact_weight_targets)}
    if args.render_artifact_weight_path:
        scene_name = os.path.basename(os.path.normpath(dataset.source_path))
        artifact_weight_lookup = load_artifact_weight_lookup(
            args.render_artifact_weight_path,
            scene_name=scene_name,
            splits=args.render_artifact_weight_splits,
            severities=args.render_artifact_weight_severities,
            default_weight=args.render_artifact_weight_default,
            mild_weight=args.render_artifact_weight_mild,
            severe_weight=args.render_artifact_weight_severe,
            mode=args.render_artifact_weight_mode,
            continuous_min_weight=args.render_artifact_weight_continuous_min,
            continuous_power=args.render_artifact_weight_continuous_power,
        )
        print(
            "Render artifact weight enabled: "
            f"path={args.render_artifact_weight_path} "
            f"scene={scene_name} "
            f"mode={args.render_artifact_weight_mode} "
            f"targets={','.join(sorted(artifact_weight_targets))} "
            f"splits={args.render_artifact_weight_splits} "
            f"severities={args.render_artifact_weight_severities} "
            f"default={args.render_artifact_weight_default:.3f} "
            f"mild={args.render_artifact_weight_mild:.3f} "
            f"severe={args.render_artifact_weight_severe:.3f} "
            f"continuous_min={args.render_artifact_weight_continuous_min:.3f} "
            f"continuous_power={args.render_artifact_weight_continuous_power:.3f} "
            f"direct_combine={args.render_artifact_direct_weight_combine_mode} "
            f"direct_loss_scale={args.render_artifact_direct_loss_scale_mode} "
            f"matched={len(artifact_weight_lookup.weights_by_name)} "
            f"summary={artifact_weight_lookup.summary()}"
        )
    teacher_artifact_weight_lookup = (
        artifact_weight_lookup if artifact_weight_lookup is not None and "teacher" in artifact_weight_targets else None
    )
    risk_artifact_weight_lookup = (
        artifact_weight_lookup if artifact_weight_lookup is not None and "risk" in artifact_weight_targets else None
    )
    artifact_region_weight_lookup = None
    artifact_region_weight_targets = {
        item.lower() for item in _comma_set(args.render_artifact_region_weight_targets)
    }
    if args.render_artifact_region_weight_path:
        scene_name = os.path.basename(os.path.normpath(dataset.source_path))
        artifact_region_weight_lookup = load_artifact_region_weight_lookup(
            args.render_artifact_region_weight_path,
            scene_name=scene_name,
            splits=args.render_artifact_region_weight_splits,
            severities=args.render_artifact_region_weight_severities,
            default_weight=args.render_artifact_region_weight_default,
            root=args.render_artifact_region_weight_root,
        )
        print(
            "Render artifact region weight enabled: "
            f"path={args.render_artifact_region_weight_path} "
            f"root={args.render_artifact_region_weight_root} "
            f"scene={scene_name} "
            f"targets={','.join(sorted(artifact_region_weight_targets))} "
            f"splits={args.render_artifact_region_weight_splits} "
            f"severities={args.render_artifact_region_weight_severities} "
            f"default={args.render_artifact_region_weight_default:.3f} "
            f"matched={len(artifact_region_weight_lookup.maps_by_name)} "
            f"summary={artifact_region_weight_lookup.summary()}"
        )
    teacher_artifact_region_weight_lookup = (
        artifact_region_weight_lookup
        if artifact_region_weight_lookup is not None
        and ("direct" in artifact_region_weight_targets or "teacher" in artifact_region_weight_targets)
        else None
    )
    topology_controller = None
    if args.enable_topology or args.train_phase in {"topology", "closed_loop"} or bool(getattr(args, "lafgs_curriculum", False)):
        protected_source_indices = None
        if args.topology_protect_landmarks:
            protected_source_indices = _load_landmark_indices(
                dataset.model_path,
                args.landmark_path,
                device="cpu",
                point_count=gaussians.get_xyz.shape[0],
            )
            print(f"Protecting {protected_source_indices.numel()} sparse landmark source ids from physical prune")
        risk_evaluator = None
        if args.topology_risk_commit_policy in {"heldout_descriptor", "heldout_pose"}:
            if direct_landmark_indices is None:
                direct_landmark_indices = _load_landmark_indices(
                    dataset.model_path,
                    args.landmark_path,
                    device="cpu",
                    point_count=gaussians.get_xyz.shape[0],
                )
                print(f"Loaded {direct_landmark_indices.numel()} held-out risk landmarks from {args.landmark_path}")
        if args.topology_risk_commit_policy == "heldout_descriptor":
            risk_evaluator = _make_heldout_descriptor_risk_evaluator(
                args,
                gaussians,
                opt,
                query_cameras,
                feature_extractor,
                masks,
                direct_landmark_indices,
                scene.cameras_extent,
            )
            print(
                "Initialized held-out descriptor topology risk: "
                f"holdout={min(max(1, int(args.topology_risk_holdout_size)), len(query_cameras))} "
                f"selection={args.topology_risk_holdout_selection} "
                f"epsilon={args.topology_risk_epsilon} "
                f"ci_z={args.topology_risk_ci_z} "
                f"min_ci_samples={args.topology_risk_min_ci_samples}"
            )
        elif args.topology_risk_commit_policy == "heldout_pose":
            risk_evaluator = _make_heldout_pose_risk_evaluator(
                args,
                dataset,
                gaussians,
                opt,
                query_cameras,
                direct_landmark_indices,
                scene.cameras_extent,
                artifact_weight_lookup=risk_artifact_weight_lookup,
            )
            print(
                "Initialized held-out sparse pose topology risk: "
                f"holdout={min(max(1, int(args.topology_risk_holdout_size)), len(query_cameras))} "
                f"selection={args.topology_risk_holdout_selection} "
                f"epsilon={args.topology_risk_epsilon} "
                f"ci_z={args.topology_risk_ci_z} "
                f"min_ci_samples={args.topology_risk_min_ci_samples} "
                f"cfg={args.topology_risk_pose_cfg}"
            )
        topology_controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=args.topology_stats_warmup,
                update_interval=args.topology_update_interval,
                min_observations=args.topology_min_observations,
                split_quantile=args.topology_split_quantile,
                ambiguity_quantile=args.topology_ambiguity_quantile,
                growth_cap_per_event=args.topology_growth_cap_per_event,
                total_point_budget_ratio=args.topology_total_point_budget_ratio,
                cooldown_iterations=args.topology_cooldown_iterations,
                enable_split=not args.topology_disable_split,
                min_repeatability=args.topology_min_repeatability,
                min_radius=args.topology_min_radius,
                enable_loc_clone=False,
                enable_soft_prune=args.topology_enable_soft_prune,
                enable_physical_prune=args.topology_enable_physical_prune,
                soft_prune_threshold=args.topology_soft_prune_threshold,
                soft_prune_step=args.topology_soft_prune_step,
                physical_rgb_threshold=args.topology_physical_rgb_threshold,
                physical_loc_threshold=args.topology_physical_loc_threshold,
                physical_utility_threshold=args.topology_physical_utility_threshold,
                require_loc_opacity_trained_for_physical_prune=not args.topology_allow_untrained_loc_opacity_prune,
                max_mutation_events=args.topology_max_mutation_events,
                risk_commit_policy=args.topology_risk_commit_policy,
            ),
            initial_points=gaussians.get_xyz.shape[0],
            protected_source_indices=protected_source_indices,
            risk_evaluator=risk_evaluator,
        )
    viewpoint_stack = None
    query_viewpoint_stack = None
    ema_loss_for_log = 0.0
    loc_opacity_grad_seen = False
    loc_training_summary = {
        "episodes": 0,
        "pseudo_query_episodes": 0,
        "direct_episodes": 0,
        "direct_visible_episodes": 0,
        "direct_visible_total": 0,
        "direct_visible_max": 0,
        "direct_nonzero_loss_episodes": 0,
        "stats_candidate_episodes": 0,
        "stats_update_episodes": 0,
        "stats_update_points_total": 0,
        "stats_skip_reliability_episodes": 0,
        "stats_skip_stage_episodes": 0,
        "stats_skip_no_visible_episodes": 0,
        "diff_pnp_episodes": 0,
        "diff_pnp_used_correspondences_total": 0,
        "training_args_path": str(training_args_path),
    }
    _record_lafgs_static_config(loc_training_summary, args, gaussians)
    loc_training_summary.update(mvinit_summary)
    lafgs_curriculum_base_iter = lafgs_curriculum_base_iteration(
        args,
        scene_loaded_iter=scene.loaded_iter,
    )
    loc_training_summary["lafgs_curriculum_base_iter"] = int(lafgs_curriculum_base_iter)
    loc_training_summary["lafgs_resume_first_iter"] = int(first_iter)
    if first_iter > 0:
        print(
            "LaFGS resume schedule: "
            f"first_iter={first_iter} "
            f"scene_loaded_iter={int(scene.loaded_iter or 0)} "
            f"curriculum_base_iter={lafgs_curriculum_base_iter}"
        )
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="LA Feature Gaussian")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        lafgs_step = lafgs_curriculum_step(iteration, base_iteration=lafgs_curriculum_base_iter)
        phase = "feature" if args.feature_only else args.train_phase
        if bool(getattr(args, "lafgs_curriculum", False)) and not args.feature_only:
            phase = lafgs_phase_from_starts(
                lafgs_step,
                args.lafgs_locrec_start_iter,
                args.lafgs_diff_pnp_start_iter,
                args.lafgs_geometry_start_iter,
                args.lafgs_topology_start_iter,
            )
        _set_phase_lrs(gaussians, phase, args)
        geometry_update_active = _phase_allows_geometry_update(args, phase)
        geometry_xyz_before = None
        if geometry_update_active:
            geometry_xyz_before = gaussians._xyz.detach().clone()
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = support_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_gsplat(
            viewpoint_cam,
            gaussians,
            background,
            rgb_only=False,
            norm_feat_bf_render=dataset.norm_before_render,
            longest_edge=dataset.longest_edge,
            rasterize_mode="antialiased",
        )
        losses = _base_losses(viewpoint_cam, render_pkg, feature_extractor, dataset, masks=masks)
        image = losses["image"]
        base_rgb_loss = (
            (1.0 - opt.lambda_dssim) * losses["Ll1"]
            + opt.lambda_dssim * (1.0 - ssim(image, losses["gt_image"]))
        )
        base_loss = base_rgb_loss + args.base_feature_weight * losses["Ll1_feature"]

        loc_loss = image.new_tensor(0.0)
        loc_desc_loss = image.new_tensor(0.0)
        loc_multiview_loss = image.new_tensor(0.0)
        loc_full_bank_loss = image.new_tensor(0.0)
        loc_clean_hard_negative_loss = image.new_tensor(0.0)
        loc_anchor_loss = image.new_tensor(0.0)
        loc_reproj_loss = image.new_tensor(0.0)
        loc_pnp_loss = image.new_tensor(0.0)
        effective_pnp_weight = float(args.lafgs_diff_pnp_weight)
        loc_dense_kl_loss = image.new_tensor(0.0)
        loc_dense_rank_loss = image.new_tensor(0.0)
        loc_proto_loss = image.new_tensor(0.0)
        loc_rank_loss = image.new_tensor(0.0)
        loc_opacity_loss = image.new_tensor(0.0)
        loc_overlay_reg_loss = image.new_tensor(0.0)
        loc_surface_anchor_loss = image.new_tensor(0.0)
        geom_anchor_loss = image.new_tensor(0.0)
        loc_geometry_residual_loss = image.new_tensor(0.0)
        loc_grad = None
        teacher_out = None
        loc_stats_update_allowed = False
        run_loc_episode = (
            args.localization_enabled
            and losses["gt_feature_map"] is not None
            and iteration >= args.loc_start_iter
            and iteration % args.loc_interval == 0
        )
        pseudo_query_reliability = {
            "enabled": False,
            "weight": 1.0,
            "update_memory": True,
            "update_stats": True,
        }
        pseudo_query_stage_direct_policy = _pseudo_query_stage_direct_loss_policy(
            pseudo_query_reliability,
            args,
        )

        if run_loc_episode:
            loc_training_summary["episodes"] += 1
            synthetic_diagnostics = {}
            query_record = None
            query_cam = None
            query_feature_map = None
            pseudo_query_region_weight_map = None
            if pseudo_query_sampler is not None:
                query_record = pseudo_query_sampler.sample_record()
                if query_record is None:
                    raise RuntimeError("Pseudo-query manifest has no accepted records.")
                query_cam = _pseudo_record_to_camera(query_record, train_camera_by_name)
                query_feature_map = _query_feature_map(
                    query_cam,
                    feature_extractor,
                    target_hw=losses["gt_feature_map"].shape[-2:],
                    masks=masks if query_record.source == "train_rgb" else None,
                )
                synthetic_diagnostics = {
                    "pseudo_query_used": 1.0,
                    "pseudo_query_is_synthetic": 1.0 if query_record.source == "synthetic_rgb" else 0.0,
                    "pseudo_query_artifact_score": float(query_record.artifact_score),
                }
                pseudo_query_region_weight_map, region_weight_summary = _pseudo_query_no_reference_region_weight(
                    query_record,
                    query_cam.original_image,
                    enabled=bool(args.pseudo_query_no_reference_region_weight),
                    allowed_sources=pseudo_no_reference_region_weight_sources,
                    min_weight=args.pseudo_query_no_reference_region_weight_min,
                    support_power=args.pseudo_query_no_reference_region_weight_support_power,
                    image_scale=args.pseudo_query_no_reference_region_weight_image_scale,
                    builder=pseudo_no_reference_region_weight_builder,
                )
                if region_weight_summary:
                    synthetic_diagnostics.update(
                        {
                            "pseudo_query_region_weight_enabled": 1.0
                            if region_weight_summary.get("enabled", False)
                            else 0.0,
                            "pseudo_query_region_weight_mean": float(
                                region_weight_summary.get("region_weight_mean", 1.0)
                            ),
                            "pseudo_query_region_weight_min": float(
                                region_weight_summary.get("region_weight_min", 1.0)
                            ),
                            "pseudo_query_region_weight_valid_frac": float(
                                region_weight_summary.get("valid_frac", 1.0)
                            ),
                            "pseudo_query_region_weight_support_frac": float(
                                region_weight_summary.get("support_frac", 1.0)
                            ),
                        }
                    )
            else:
                use_synthetic_view = (
                    lafgs_should_sample_synthetic_view(
                        args.loc_teacher,
                        args.synthetic_view_ratio,
                        query_camera_count=len(query_cameras),
                        random_value=random(),
                    )
                )
                if use_synthetic_view:
                    query_cam, query_feature_map, synthetic_diagnostics = _sample_synthetic_query(
                        query_cameras,
                        gaussians,
                        losses["gt_feature_map"].shape[-2:],
                        background,
                        args,
                        feature_extractor=feature_extractor,
                        norm_feat_bf_render=dataset.norm_before_render,
                    )
                if query_cam is None or query_feature_map is None:
                    if float(args.synthetic_view_ratio) > 0.0 and args.loc_teacher in {"dense", "direct"}:
                        synthetic_diagnostics = {"synthetic_view_used": 0.0}
                    query_cam = viewpoint_cam
                    query_feature_map = losses["gt_feature_map"]
                if args.support_query_split and query_cam is viewpoint_cam:
                    if not query_viewpoint_stack:
                        query_viewpoint_stack = query_cameras.copy()
                    query_cam = query_viewpoint_stack.pop(randint(0, len(query_viewpoint_stack) - 1))
                    query_feature_map = _query_feature_map(
                        query_cam,
                        feature_extractor,
                        target_hw=losses["gt_feature_map"].shape[-2:],
                        masks=masks,
                    )
            episode = episode_sampler.sample(query_cam)
            if query_record is not None:
                loc_training_summary["pseudo_query_episodes"] += 1
                source_key = str(query_record.source or "unknown").replace("/", "_")
                source_metric = f"source_{source_key}_episodes"
                loc_training_summary[source_metric] = loc_training_summary.get(source_metric, 0) + 1
                pseudo_query_reliability = _pseudo_query_reliability_decision(
                    query_record,
                    episode.sparse_meta,
                    pseudo_query_reliability_stats,
                    args,
                )
                synthetic_diagnostics.update(
                    {
                        "pseudo_query_reliability_enabled": 1.0
                        if pseudo_query_reliability.get("enabled", False)
                        else 0.0,
                        "pseudo_query_reliability_weight": float(
                            pseudo_query_reliability.get("weight", 1.0)
                        ),
                        "pseudo_query_reliability_stage_weight": float(
                            pseudo_query_reliability.get("stage_weight", 1.0)
                        ),
                        "pseudo_query_reliability_error_weight": float(
                            pseudo_query_reliability.get("error_weight", 1.0)
                        ),
                        "pseudo_query_reliability_inlier_weight": float(
                            pseudo_query_reliability.get("inlier_weight", 1.0)
                        ),
                        "pseudo_query_reliability_support_weight": float(
                            pseudo_query_reliability.get("support_weight", 1.0)
                        ),
                        "pseudo_query_reliability_update_memory": 1.0
                        if pseudo_query_reliability.get("update_memory", True)
                        else 0.0,
                        "pseudo_query_reliability_update_stats": 1.0
                        if pseudo_query_reliability.get("update_stats", True)
                        else 0.0,
                        "pseudo_query_reliability_loss_scaled": 1.0
                        if args.pseudo_query_reliability_loss_mode == "soft"
                        else 0.0,
                    }
                )
                synthetic_diagnostics.update(
                    _pseudo_query_stage_source_diagnostics(query_record, pseudo_query_reliability)
                )
            pose_gt = episode.pose_gt_w2c.cuda()
            pose_init = episode.pose_init_w2c.cuda()
            if args.loc_teacher == "direct":
                current_direct_landmark_indices = _current_landmark_indices_from_source_index(
                    direct_landmark_indices,
                    gaussians,
                )
                pseudo_query_stage_direct_policy = _pseudo_query_stage_direct_loss_policy(
                    pseudo_query_reliability,
                    args,
                )
                synthetic_diagnostics.update(
                    {
                        "pseudo_query_stage_objective_enabled": 1.0
                        if pseudo_query_stage_direct_policy.get("enabled", False)
                        else 0.0,
                        "pseudo_query_stage_objective_desc_weight": float(
                            pseudo_query_stage_direct_policy.get("desc", 1.0)
                        ),
                        "pseudo_query_stage_objective_multiview_weight": float(
                            pseudo_query_stage_direct_policy.get("multiview", 1.0)
                        ),
                        "pseudo_query_stage_objective_full_bank_weight": float(
                            pseudo_query_stage_direct_policy.get("full_bank", 1.0)
                        ),
                        "pseudo_query_stage_objective_anchor_weight": float(
                            pseudo_query_stage_direct_policy.get("anchor", 1.0)
                        ),
                        "pseudo_query_stage_objective_update_memory": 1.0
                        if pseudo_query_stage_direct_policy.get("update_memory", True)
                        else 0.0,
                        "pseudo_query_stage_objective_update_stats": 1.0
                        if pseudo_query_stage_direct_policy.get("update_stats", True)
                        else 0.0,
                    }
                )
                target_depth = None
                target_alpha = None
                if args.direct_depth_check:
                    with torch.no_grad():
                        gt_render = render_from_pose_gsplat(
                            gaussians,
                            pose_gt,
                            query_cam.FoVx,
                            query_cam.FoVy,
                            query_feature_map.shape[2],
                            query_feature_map.shape[1],
                            bg_color=background,
                            render_mode="RGB+ED",
                            rgb_only=True,
                            norm_feat_bf_render=dataset.norm_before_render,
                            rasterize_mode="antialiased",
                        )
                    target_depth = _flatten_render_map(gt_render.get("depth"))
                    target_alpha = _flatten_render_alpha(gt_render.get("alphas"))
                child_responsibility_mode = args.loc_child_responsibility_mode
                if (
                    args.loc_child_responsibility_start_iter > 0
                    and iteration < args.loc_child_responsibility_start_iter
                ):
                    child_responsibility_mode = "none"
                artifact_region_weight_map = None
                if teacher_artifact_region_weight_lookup is not None:
                    artifact_region_weight_map = teacher_artifact_region_weight_lookup.map_for_camera(
                        query_cam,
                        device=query_feature_map.device,
                        dtype=query_feature_map.dtype,
                    )
                if pseudo_query_region_weight_map is not None:
                    pseudo_query_region_weight_map = pseudo_query_region_weight_map.to(
                        device=query_feature_map.device,
                        dtype=query_feature_map.dtype,
                    )
                    artifact_region_weight_map = _combine_region_weight_maps(
                        artifact_region_weight_map,
                        pseudo_query_region_weight_map,
                    )
                artifact_image_weight = 1.0
                if teacher_artifact_weight_lookup is not None:
                    artifact_image_weight = float(teacher_artifact_weight_lookup.weight_for_camera(query_cam))
                clean_field_controls = _clean_field_stage_controls(args, lafgs_step)
                teacher_out = direct_landmark_teacher(
                    gaussians,
                    query_feature_map,
                    pose_gt,
                    query_cam.FoVx,
                    query_cam.FoVy,
                    current_direct_landmark_indices,
                    target_depth=target_depth,
                    target_alpha=target_alpha,
                    alpha_threshold=args.loc_alpha_threshold,
                    depth_abs_tolerance=args.direct_depth_abs_tolerance,
                    depth_rel_tolerance=args.direct_depth_rel_tolerance,
                    max_landmarks=args.loc_anchors,
                    multiview_memory=direct_observation_memory,
                    multiview_temperature=args.loc_multiview_temperature,
                    multiview_ignore_radius=args.loc_multiview_ignore_radius,
                    update_multiview_memory=bool(pseudo_query_stage_direct_policy.get("update_memory", True)),
                    full_bank_indices=current_direct_landmark_indices if args.loc_full_bank_weight > 0 else None,
                    full_bank_temperature=args.loc_full_bank_temperature,
                    full_bank_hard_negative_topk=args.loc_full_bank_hard_negatives,
                    full_bank_hard_negative_margin=args.loc_full_bank_margin,
                    full_bank_ignore_3d_radius=args.loc_full_bank_ignore_3d_radius,
                    full_bank_ignore_uv_radius=args.loc_full_bank_ignore_uv_radius,
                    full_bank_source_mode=args.loc_full_bank_source_mode,
                    full_bank_nearby_as_positive=full_bank_nearby_as_positive_active(
                        args,
                        iteration=iteration,
                        lafgs_step=lafgs_step,
                    ),
                    full_bank_stats_chunk_size=getattr(args, "loc_full_bank_stats_chunk_size", 256),
                    full_bank_pose_information_weight=clean_field_controls["pose_information_weight"],
                    full_bank_pose_information_floor=args.loc_full_bank_pose_information_floor,
                    full_bank_balance_weight=clean_field_controls["balance_weight"],
                    full_bank_balance_grid_size=args.loc_full_bank_balance_grid_size,
                    full_bank_balance_depth_bins=args.loc_full_bank_balance_depth_bins,
                    full_bank_balance_max_weight=args.loc_full_bank_balance_max_weight,
                    full_bank_clean_hard_negative_weight=clean_field_controls["clean_hn_weight"],
                    full_bank_clean_reproj_radius=args.loc_full_bank_clean_reproj_radius,
                    full_bank_clean_hard_negatives=args.loc_full_bank_clean_hard_negatives,
                    sampling_grid_size=args.loc_anchor_grid_size,
                    anchor_features=_feature_anchor_tensor(loc_feature_anchor) if args.loc_anchor_weight > 0 else None,
                    child_responsibility_mode=child_responsibility_mode,
                    artifact_weight_map=artifact_region_weight_map,
                    artifact_image_weight=artifact_image_weight,
                    artifact_weight_combine_mode=args.render_artifact_direct_weight_combine_mode,
                    artifact_loss_scale_mode=args.render_artifact_direct_loss_scale_mode,
                )
                loc_desc_loss = teacher_out.desc_loss
                loc_multiview_loss = teacher_out.multiview_loss
                loc_full_bank_loss = teacher_out.full_bank_loss
                loc_clean_hard_negative_loss = teacher_out.clean_hard_negative_loss
                loc_anchor_loss = teacher_out.anchor_loss
                loc_reproj_loss = teacher_out.reproj_loss
                loc_loss, pseudo_query_stage_direct_policy = _compose_direct_loc_loss(
                    loc_desc_loss,
                    loc_multiview_loss,
                    loc_full_bank_loss,
                    loc_anchor_loss,
                    pseudo_query_reliability,
                    args,
                    stage_policy=pseudo_query_stage_direct_policy,
                    full_bank_weight_scale=clean_field_controls["full_bank_weight_scale"],
                    loc_clean_hard_negative_loss=loc_clean_hard_negative_loss,
                    clean_hard_negative_weight=clean_field_controls["clean_hn_weight"],
                )
                teacher_out.diagnostics.update(
                    {
                        "clean_field_stage_active": 1.0 if clean_field_controls["active"] else 0.0,
                        "clean_field_stage_start_iter": float(clean_field_controls["start_iter"]),
                        "clean_field_full_bank_weight_scale": float(
                            clean_field_controls["full_bank_weight_scale"]
                        ),
                        "clean_field_clean_hn_weight": float(clean_field_controls["clean_hn_weight"]),
                        "clean_field_clean_hn_weight_scale": float(
                            clean_field_controls["clean_hn_weight_scale"]
                        ),
                        "clean_field_balance_weight": float(clean_field_controls["balance_weight"]),
                        "clean_field_pose_information_weight": float(
                            clean_field_controls["pose_information_weight"]
                        ),
                        "clean_field_diff_pnp_weight": float(clean_field_controls["diff_pnp_weight"]),
                        "clean_field_diff_pnp_weight_scale": float(
                            clean_field_controls["diff_pnp_weight_scale"]
                        ),
                    }
                )
                if bool(synthetic_diagnostics.get("synthetic_view_used", 0.0) > 0.0):
                    loc_loss = loc_loss * float(args.synthetic_view_desc_weight)
                visible_count = 0
                if teacher_out.loc_visible_idx is not None:
                    visible_count = int(teacher_out.loc_visible_idx.numel())
                loc_training_summary["direct_episodes"] += 1
                loc_training_summary["direct_visible_total"] += visible_count
                loc_training_summary["direct_visible_max"] = max(
                    loc_training_summary["direct_visible_max"],
                    visible_count,
                )
                if visible_count > 0:
                    loc_training_summary["direct_visible_episodes"] += 1
                if (
                    args.lafgs_diff_pnp_weight > 0.0
                    and lafgs_step >= args.lafgs_diff_pnp_start_iter
                    and teacher_out.loc_visible_idx is not None
                    and teacher_out.loc_visible_idx.numel() >= args.lafgs_diff_pnp_min_correspondences
                ):
                    loc_xyz_all = gaussian_localization_xyz(gaussians)
                    pnp_indices = teacher_out.loc_visible_idx.to(device=loc_xyz_all.device, dtype=torch.long)
                    pnp_points_world = loc_xyz_all[pnp_indices]
                    K = make_intrinsics_from_fov(
                        query_cam.FoVx,
                        query_cam.FoVy,
                        query_feature_map.shape[2],
                        query_feature_map.shape[1],
                        device=query_feature_map.device,
                        dtype=query_feature_map.dtype,
                    )
                    allow_geometry_grad = _diff_pnp_allows_geometry_grad(args, phase)
                    pnp_point_weights = None
                    if bool(args.lafgs_diff_pnp_use_loc_opacity_weight) and args.use_loc_opacity:
                        pnp_point_weights = gaussians.get_loc_opacity[pnp_indices].reshape(-1)
                    pnp_geometry_anchor_points = None
                    if (
                        float(args.lafgs_diff_pnp_geometry_depth_anchor_weight) > 0.0
                        and hasattr(gaussians, "loc_source_xyz")
                    ):
                        pnp_geometry_anchor_points = gaussians.loc_source_xyz[pnp_indices]
                    pnp_out = differentiable_pnp_pose_loss(
                        gaussians.get_loc_feature[pnp_indices].reshape(pnp_indices.numel(), -1),
                        query_feature_map,
                        pnp_points_world,
                        K,
                        pose_gt,
                        pose_init_w2c=pose_init,
                        projected_uv=teacher_out.target_uv if _diff_pnp_needs_projected_uv(args) else None,
                        geometry_anchor_points_world=pnp_geometry_anchor_points,
                        point_weights=pnp_point_weights,
                        config=DifferentiablePnPConfig(
                            temperature=args.lafgs_diff_pnp_temperature,
                            min_correspondences=args.lafgs_diff_pnp_min_correspondences,
                            confidence_threshold=args.lafgs_diff_pnp_confidence_threshold,
                            pnp_iterations=args.lafgs_diff_pnp_iterations,
                            pose_weight=args.lafgs_diff_pnp_pose_weight,
                            reprojection_weight=args.lafgs_diff_pnp_reproj_weight,
                            gt_reprojection_weight=args.lafgs_diff_pnp_gt_reproj_weight,
                            entropy_weight=args.lafgs_diff_pnp_entropy_weight,
                            reprojection_loss_type=args.lafgs_diff_pnp_reprojection_loss_type,
                            reprojection_loss_delta=args.lafgs_diff_pnp_reprojection_loss_delta,
                            max_condition_number=args.lafgs_diff_pnp_max_condition_number,
                            geometry_reprojection_weight=args.lafgs_diff_pnp_geometry_reproj_weight,
                            geometry_depth_anchor_weight=args.lafgs_diff_pnp_geometry_depth_anchor_weight,
                            geometry_match_reprojection_weight=(
                                args.lafgs_diff_pnp_geometry_match_reproj_weight
                            ),
                            geometry_match_confidence_threshold=(
                                args.lafgs_diff_pnp_geometry_match_confidence_threshold
                            ),
                            geometry_match_margin_threshold=(
                                args.lafgs_diff_pnp_geometry_match_margin_threshold
                            ),
                            geometry_match_peak_probability_threshold=(
                                args.lafgs_diff_pnp_geometry_match_peak_probability_threshold
                            ),
                            geometry_match_max_entropy=args.lafgs_diff_pnp_geometry_match_max_entropy,
                            geometry_match_max_reprojection_error=(
                                args.lafgs_diff_pnp_geometry_match_max_reproj_error
                            ),
                            geometry_confidence_threshold=args.lafgs_diff_pnp_geometry_confidence_threshold,
                            geometry_margin_threshold=args.lafgs_diff_pnp_geometry_margin_threshold,
                            geometry_peak_probability_threshold=(
                                args.lafgs_diff_pnp_geometry_peak_probability_threshold
                            ),
                            geometry_max_entropy=args.lafgs_diff_pnp_geometry_max_entropy,
                            geometry_max_reprojection_error=args.lafgs_diff_pnp_geometry_max_reproj_error,
                            geometry_use_all_correspondences=(
                                args.lafgs_diff_pnp_geometry_use_all_correspondences
                            ),
                            geometry_local_window_radius=args.lafgs_diff_pnp_geometry_local_window_radius,
                            geometry_pose_guard_max_loss_increase=(
                                args.lafgs_diff_pnp_geometry_pose_guard_max_loss_increase
                            ),
                            geometry_pose_guard_max_loss=args.lafgs_diff_pnp_geometry_pose_guard_max_loss,
                            geometry_pose_guard_softness=args.lafgs_diff_pnp_geometry_pose_guard_softness,
                            geometry_pose_guard_min_scale=args.lafgs_diff_pnp_geometry_pose_guard_min_scale,
                            feedback_pose_guard_max_loss_increase=(
                                args.lafgs_diff_pnp_feedback_pose_guard_max_loss_increase
                            ),
                            feedback_pose_guard_max_loss=args.lafgs_diff_pnp_feedback_pose_guard_max_loss,
                            feedback_pose_guard_softness=args.lafgs_diff_pnp_feedback_pose_guard_softness,
                            feedback_pose_guard_min_scale=args.lafgs_diff_pnp_feedback_pose_guard_min_scale,
                            feedback_pose_guard_keep_gt_reprojection=(
                                args.lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection
                            ),
                            detach_gt_reprojection_points=args.lafgs_diff_pnp_detach_gt_reprojection_points,
                            detach_pnp_points=args.lafgs_diff_pnp_detach_pnp_points,
                            allow_geometry_grad=allow_geometry_grad,
                            local_window_radius=args.lafgs_diff_pnp_local_window_radius,
                            max_correspondences=args.lafgs_diff_pnp_max_correspondences,
                            spatial_grid_size=args.lafgs_diff_pnp_spatial_grid_size,
                            min_spatial_span=args.lafgs_diff_pnp_min_spatial_span,
                            min_spatial_area=args.lafgs_diff_pnp_min_spatial_area,
                            point_weight_floor=args.lafgs_diff_pnp_point_weight_floor,
                        ),
                    )
                    loc_pnp_loss = pnp_out.loss
                    effective_pnp_weight = clean_field_controls["diff_pnp_weight"]
                    weighted_loc_pnp_loss = effective_pnp_weight * loc_pnp_loss
                    loc_loss = loc_loss + weighted_loc_pnp_loss
                    pnp_landmark_stats = pnp_output_to_landmark_stats(
                        pnp_out,
                        pnp_points_world,
                        K,
                        pose_gt,
                        full_bank_positive_prob=teacher_out.stats.get("full_bank_positive_prob"),
                        full_bank_margin=teacher_out.stats.get("margin"),
                        pose_loss_scale=args.lafgs_diff_pnp_utility_pose_loss_scale,
                        reprojection_error_scale=args.lafgs_diff_pnp_utility_reprojection_error_scale,
                    )
                    teacher_out.stats.update(pnp_landmark_stats)
                    teacher_out.diagnostics.update(
                        {
                            "lafgs_diff_pnp_loss": float(loc_pnp_loss.detach().item()),
                            "lafgs_diff_pnp_weight": float(effective_pnp_weight),
                            "lafgs_diff_pnp_base_weight": float(args.lafgs_diff_pnp_weight),
                            "lafgs_diff_pnp_weighted_loss": float(weighted_loc_pnp_loss.detach().item()),
                            "lafgs_diff_pnp_pose_loss": float(pnp_out.pose_loss.detach().item()),
                            "lafgs_diff_pnp_reprojection_loss": float(pnp_out.reprojection_loss.detach().item()),
                            "lafgs_diff_pnp_gt_reprojection_loss": float(
                                pnp_out.gt_reprojection_loss.detach().item()
                            ),
                            "lafgs_diff_pnp_geometry_reprojection_loss": float(
                                pnp_out.geometry_reprojection_loss.detach().item()
                            ),
                            "lafgs_diff_pnp_geometry_depth_anchor_loss": float(
                                pnp_out.geometry_depth_anchor_loss.detach().item()
                            ),
                            "lafgs_diff_pnp_geometry_match_reprojection_loss": float(
                                pnp_out.geometry_match_reprojection_loss.detach().item()
                            ),
                            "lafgs_diff_pnp_entropy_loss": float(pnp_out.entropy_loss.detach().item()),
                            "lafgs_diff_pnp_used_correspondences": float(pnp_out.used_correspondences),
                            "lafgs_diff_pnp_geometry_correspondences": float(
                                pnp_out.diagnostics.get("geometry_correspondences", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_candidate_count": float(
                                pnp_out.diagnostics.get("geometry_candidate_count", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_depth_anchor_correspondences": float(
                                pnp_out.diagnostics.get("geometry_depth_anchor_correspondences", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_depth_anchor_candidate_count": float(
                                pnp_out.diagnostics.get("geometry_depth_anchor_candidate_count", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_depth_anchor_weight": float(
                                args.lafgs_diff_pnp_geometry_depth_anchor_weight
                            ),
                            "lafgs_diff_pnp_geometry_match_correspondences": float(
                                pnp_out.diagnostics.get("geometry_match_correspondences", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_candidate_count": float(
                                pnp_out.diagnostics.get("geometry_match_candidate_count", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_reproj_weight": float(
                                args.lafgs_diff_pnp_geometry_match_reproj_weight
                            ),
                            "lafgs_diff_pnp_geometry_match_confidence_threshold": float(
                                pnp_out.diagnostics.get("geometry_match_confidence_threshold", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_margin_threshold": float(
                                pnp_out.diagnostics.get("geometry_match_margin_threshold", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_peak_probability_threshold": float(
                                pnp_out.diagnostics.get("geometry_match_peak_probability_threshold", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_max_entropy": float(
                                pnp_out.diagnostics.get("geometry_match_max_entropy", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_match_max_reproj_error": float(
                                pnp_out.diagnostics.get("geometry_match_max_reprojection_error", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_use_all_correspondences": (
                                1.0 if args.lafgs_diff_pnp_geometry_use_all_correspondences else 0.0
                            ),
                            "lafgs_diff_pnp_geometry_local_window_radius": float(
                                args.lafgs_diff_pnp_geometry_local_window_radius
                            ),
                            "lafgs_diff_pnp_geometry_peak_probability_threshold": float(
                                args.lafgs_diff_pnp_geometry_peak_probability_threshold
                            ),
                            "lafgs_diff_pnp_geometry_max_entropy": float(
                                args.lafgs_diff_pnp_geometry_max_entropy
                            ),
                            "lafgs_diff_pnp_condition_guard_scale": float(
                                pnp_out.diagnostics.get("condition_guard_scale", 0.0)
                            ),
                            "lafgs_diff_pnp_condition_guard_passed": float(
                                pnp_out.diagnostics.get("condition_guard_passed", 0.0)
                            ),
                            "lafgs_diff_pnp_geometry_pose_guard_scale": float(
                                pnp_out.diagnostics.get("geometry_pose_guard_scale", 0.0)
                            ),
                            "lafgs_diff_pnp_reprojection_loss_type": str(
                                args.lafgs_diff_pnp_reprojection_loss_type
                            ),
                            "lafgs_diff_pnp_reprojection_loss_delta": float(
                                args.lafgs_diff_pnp_reprojection_loss_delta
                            ),
                            "lafgs_diff_pnp_allow_geometry_grad": 1.0 if allow_geometry_grad else 0.0,
                            "lafgs_diff_pnp_use_loc_opacity_weight": (
                                1.0 if pnp_point_weights is not None else 0.0
                            ),
                            "lafgs_diff_pnp_selected_spatial_cells": float(
                                pnp_out.diagnostics.get("selected_spatial_cells", 0.0)
                            ),
                            "lafgs_diff_pnp_spatial_min_span": float(
                                pnp_out.diagnostics.get("spatial_min_span", 0.0)
                            ),
                            "lafgs_diff_pnp_spatial_area": float(
                                pnp_out.diagnostics.get("spatial_area", 0.0)
                            ),
                            "lafgs_diff_pnp_point_weight_mean": float(
                                pnp_out.diagnostics.get("point_weight_mean", 0.0)
                            ),
                            "lafgs_diff_pnp_point_weight_floor": float(
                                args.lafgs_diff_pnp_point_weight_floor
                            ),
                            "lafgs_diff_pnp_utility_pose_loss_scale": float(
                                args.lafgs_diff_pnp_utility_pose_loss_scale
                            ),
                            "lafgs_diff_pnp_utility_reprojection_error_scale": float(
                                args.lafgs_diff_pnp_utility_reprojection_error_scale
                            ),
                            "lafgs_diff_pnp_loc_utility_mean": float(
                                pnp_landmark_stats["loc_utility"].detach().mean().item()
                            ),
                        }
                    )
                    loc_training_summary["diff_pnp_weighted_loss_total"] = (
                        loc_training_summary.get("diff_pnp_weighted_loss_total", 0.0)
                        + float(weighted_loc_pnp_loss.detach().item())
                    )
                    loc_training_summary["diff_pnp_weight_max"] = max(
                        loc_training_summary.get("diff_pnp_weight_max", 0.0),
                        float(effective_pnp_weight),
                    )
                    update_diff_pnp_training_summary(
                        loc_training_summary,
                        pnp_out,
                        loc_pnp_loss,
                        allow_geometry_grad=allow_geometry_grad,
                    )
                    _record_diff_pnp_gradient_diagnostics(
                        loc_training_summary,
                        gaussians,
                        pnp_out,
                        args,
                        effective_pnp_weight=effective_pnp_weight,
                    )
            else:
                dense_pose_weight = 1.0
                if args.loc_dense_advantage_gate:
                    dense_pose_weight = _dense_pose_advantage_weight(
                        episode.sparse_meta,
                        min_te=args.loc_dense_advantage_min_te,
                        min_ae=args.loc_dense_advantage_min_ae,
                        te_scale=args.loc_dense_advantage_te_scale,
                        ae_scale=args.loc_dense_advantage_ae_scale,
                    )
                elif args.loc_dense_pose_gate:
                    dense_pose_weight = _dense_pose_improvement_weight(
                        episode.sparse_meta,
                        min_te=args.loc_dense_pose_gate_min_te,
                        min_ae=args.loc_dense_pose_gate_min_ae,
                    )
                teacher_out = dense_localization_teacher(
                    gaussians,
                    query_feature_map,
                    pose_init,
                    pose_gt,
                    query_cam.FoVx,
                    query_cam.FoVy,
                    query_feature_map.shape[2],
                    query_feature_map.shape[1],
                    background,
                    anchor_count=args.loc_anchors,
                    alpha_threshold=args.loc_alpha_threshold,
                    desc_temperature=args.loc_desc_temperature,
                    fine_temperature=args.loc_fine_temperature,
                    fine_window_radius=args.loc_fine_window_radius,
                    dense_kl_weight=args.loc_dense_kl_weight,
                    dense_kl_temperature=args.loc_dense_kl_temperature,
                    dense_rank_weight=args.loc_dense_rank_weight,
                    dense_rank_margin=args.loc_dense_rank_margin,
                    dense_rank_teacher_confidence=args.loc_dense_rank_teacher_confidence,
                    dense_rank_miss_topk=args.loc_dense_rank_miss_topk,
                    responsibility_topk=args.loc_responsibility_topk,
                    responsibility_opacity_weight=args.loc_responsibility_opacity_weight,
                    responsibility_depth_weight=args.loc_responsibility_depth_weight,
                    dense_pose_weight=dense_pose_weight,
                    attr_cosine_threshold=args.loc_dense_attr_cosine_threshold,
                    attr_entropy_threshold=args.loc_dense_attr_entropy_threshold,
                    min_positive_prob=args.loc_dense_min_positive_prob,
                    max_reproj_error=args.loc_dense_max_reproj_error,
                    min_eligible_anchors=args.loc_dense_min_eligible_anchors,
                    norm_feat_bf_render=dataset.norm_before_render,
                    use_loc_opacity=args.use_loc_opacity,
                    rasterize_args={"rasterize_mode": "antialiased"},
                )
                loc_desc_loss = teacher_out.desc_loss
                loc_reproj_loss = teacher_out.reproj_loss
                loc_dense_kl_loss = teacher_out.kl_loss
                loc_dense_rank_loss = getattr(teacher_out, "rank_loss", None)
                if loc_dense_rank_loss is None:
                    loc_dense_rank_loss = image.new_tensor(0.0)
                dense_loss_weights = _dense_loss_weights_for_episode(
                    args,
                    synthetic_view_used=bool(synthetic_diagnostics.get("synthetic_view_used", 0.0) > 0.0),
                )
                loc_loss = (
                    dense_loss_weights["desc"] * loc_desc_loss
                    + dense_loss_weights["reproj"] * loc_reproj_loss
                    + dense_loss_weights["kl"] * loc_dense_kl_loss
                    + dense_loss_weights["rank"] * loc_dense_rank_loss
                )

            if teacher_out is not None and synthetic_diagnostics:
                teacher_out.diagnostics.update(synthetic_diagnostics)

            direct_consumed_artifact_image_weight = (
                args.loc_teacher == "direct"
                and args.render_artifact_direct_loss_scale_mode == "combined_mean"
            )
            if teacher_artifact_weight_lookup is not None:
                artifact_weight = float(teacher_artifact_weight_lookup.weight_for_camera(query_cam))
                if not direct_consumed_artifact_image_weight:
                    loc_loss = loc_loss * artifact_weight
                teacher_out.diagnostics["render_artifact_weight"] = artifact_weight
                teacher_out.diagnostics["render_artifact_is_weighted"] = 1.0 if artifact_weight < 1.0 else 0.0
                teacher_out.diagnostics["render_artifact_outer_scale_applied"] = (
                    0.0 if direct_consumed_artifact_image_weight else 1.0
                )

            loc_loss = _scale_loc_loss_by_pseudo_reliability(loc_loss, pseudo_query_reliability, args)
            if args.loc_teacher == "direct" and float(loc_loss.detach().item()) > 0.0:
                loc_training_summary["direct_nonzero_loss_episodes"] += 1

            visible_idx = teacher_out.loc_visible_idx
            if visible_idx is not None and visible_idx.numel() > 0:
                seen = gaussians.loc_prototype_count[visible_idx] > 0
                if seen.any():
                    loc_features = gaussians.get_loc_feature[visible_idx][seen].reshape(seen.sum(), -1)
                    prototypes = gaussians.loc_prototype[visible_idx][seen]
                    loc_proto_loss = prototype_loss(loc_features, prototypes)
                    loc_loss = loc_loss + args.loc_proto_weight * loc_proto_loss
                    if args.loc_rank_weight > 0 and loc_features.shape[0] > 1:
                        loc_rank_loss = hard_negative_ranking_loss(loc_features, prototypes, margin=args.loc_rank_margin)
                        loc_loss = loc_loss + args.loc_rank_weight * loc_rank_loss

            if args.use_loc_opacity and args.loc_opacity_weight > 0:
                loc_opacity_loss = localization_opacity_regularizer(
                    gaussians.get_loc_opacity,
                    target_density=args.loc_opacity_target,
                    sparsity_weight=1.0,
                    density_weight=1.0,
                )
                loc_loss = loc_loss + args.loc_opacity_weight * loc_opacity_loss

            if args.loc_overlay_reg_weight > 0:
                loc_overlay_reg_loss = _descriptor_overlay_regularizer(gaussians)
                loc_loss = loc_loss + args.loc_overlay_reg_weight * loc_overlay_reg_loss

            if (
                float(getattr(args, "surfel_loc_anchor_reg_weight", 0.0) or 0.0) > 0.0
                and hasattr(gaussians, "loc_anchor_offset_regularization")
            ):
                loc_surface_anchor_loss = gaussians.loc_anchor_offset_regularization()
                loc_loss = loc_loss + args.surfel_loc_anchor_reg_weight * loc_surface_anchor_loss
                if teacher_out is not None:
                    teacher_out.diagnostics["surfel_loc_anchor_reg_loss"] = float(
                        loc_surface_anchor_loss.detach().item()
                    )
                    teacher_out.diagnostics["surfel_loc_tangent_bound"] = float(
                        getattr(gaussians, "surfel_loc_tangent_bound", 0.0) or 0.0
                    )
                    teacher_out.diagnostics["surfel_loc_normal_bound"] = float(
                        getattr(gaussians, "surfel_loc_normal_bound", 0.0) or 0.0
                    )
                    teacher_out.diagnostics["surfel_loc_radius_floor"] = float(
                        getattr(gaussians, "surfel_loc_radius_floor", 0.0) or 0.0
                    )

            if teacher_out is not None:
                _record_direct_teacher_diagnostics(
                    loc_training_summary,
                    getattr(teacher_out, "diagnostics", {}),
                )

            if teacher_out.loc_viewspace_points is not None and loc_loss.requires_grad:
                loc_grad = torch.autograd.grad(
                    loc_loss,
                    teacher_out.loc_viewspace_points,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            loc_stats_update_allowed = (
                teacher_out is not None
                and teacher_out.loc_visible_idx is not None
                and teacher_out.loc_visible_idx.numel() > 0
                and bool(pseudo_query_reliability.get("update_stats", True))
                and bool(pseudo_query_stage_direct_policy.get("update_stats", True))
            )

        if geometry_update_active and args.geometry_anchor_weight > 0:
            geometry_anchor = _refresh_geometry_anchor_if_point_count_changed(gaussians, geometry_anchor)
            geom_anchor_loss = geometry_anchor_loss(
                _current_geometry_state(gaussians),
                geometry_anchor,
                xyz_weight=1.0,
                scale_weight=args.geometry_anchor_scale_weight,
                rotation_weight=args.geometry_anchor_rotation_weight,
            )
        geometry_residual_weight = (
            float(getattr(args, "lafgs_geometry_residual_weight", 0.0))
            if bool(getattr(args, "lafgs_geometry_residual", False))
            else 0.0
        )
        if (
            geometry_update_active
            and geometry_residual_weight > 0.0
            and hasattr(gaussians, "loc_source_xyz")
        ):
            loc_geometry_residual_loss, loc_geometry_residual_stats = bounded_geometry_residual_loss(
                gaussians.get_xyz,
                gaussians.loc_source_xyz,
                gaussians.get_scaling,
                max_scale_ratio=args.lafgs_geometry_residual_max_scale_ratio,
            )
            _record_lafgs_geometry_residual_diagnostics(
                loc_training_summary,
                teacher_out,
                loc_geometry_residual_loss,
                loc_geometry_residual_stats,
            )

        stage_loss_weights = lafgs_stage_loss_weights(args, lafgs_step)
        total_loss = (
            stage_loss_weights["base"] * base_loss
            + stage_loss_weights["loc"] * loc_loss
            + stage_loss_weights["geometry_anchor"] * geom_anchor_loss
            + geometry_residual_weight * loc_geometry_residual_loss
        )
        sfm_from_zero_raw_xyz = (
            str(getattr(args, "lafgs_stage_schedule", "none") or "none") == "sfm_from_zero"
            and _allow_raw_xyz_geometry_grad(args)
        )
        isolate_diff_pnp_geometry_grad = bool(
            getattr(args, "lafgs_diff_pnp_isolate_geometry_grad", False)
        ) and (sfm_from_zero_raw_xyz or _diff_pnp_allows_geometry_grad(args, phase))
        isolated_xyz_scaffold_loss = (
            stage_loss_weights["base"] * base_rgb_loss if sfm_from_zero_raw_xyz else None
        )
        isolated_xyz_loss = (
            stage_loss_weights["loc"] * effective_pnp_weight * loc_pnp_loss
            if _diff_pnp_allows_geometry_grad(args, phase)
            else None
        )
        isolated_xyz_regularizer_loss = (
            stage_loss_weights["geometry_anchor"] * geom_anchor_loss
            + geometry_residual_weight * loc_geometry_residual_loss
        )
        _backward_with_optional_isolated_xyz_grad(
            total_loss,
            isolated_xyz_loss,
            gaussians,
            isolate_xyz_grad=isolate_diff_pnp_geometry_grad,
            isolated_xyz_scaffold_loss=isolated_xyz_scaffold_loss,
            isolated_xyz_regularizer_loss=isolated_xyz_regularizer_loss,
            summary=loc_training_summary,
        )
        _record_geometry_optimizer_diagnostics(
            loc_training_summary,
            gaussians,
            phase,
            geometry_active=geometry_update_active,
        )
        frozen_child_feature_count = _mask_frozen_child_loc_feature_gradients(
            gaussians,
            iteration=iteration,
            freeze_steps=args.loc_child_feature_freeze_steps,
        )
        loc_opacity_grad = getattr(getattr(gaussians, "_loc_opacity", None), "grad", None)
        if loc_opacity_grad is not None:
            loc_opacity_grad_seen = loc_opacity_grad_seen or bool(
                torch.isfinite(loc_opacity_grad).any().item()
                and (loc_opacity_grad.detach().abs().max() > 0).item()
            )
        gaussians.loc_opacity_grad_seen = loc_opacity_grad_seen

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            gaussians.update_screen_radii(render_pkg.get("visibility_filter"), render_pkg.get("radii"))
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Loc": f"{loc_loss.item():.6f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar("train_loss/base", base_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/base_rgb", base_rgb_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc", loc_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/base_weight", stage_loss_weights["base"], iteration)
                tb_writer.add_scalar("train_loss/loc_weight", stage_loss_weights["loc"], iteration)
                tb_writer.add_scalar(
                    "train_loss/geometry_anchor_weight",
                    stage_loss_weights["geometry_anchor"],
                    iteration,
                )
                tb_writer.add_scalar("train_loss/loc_desc", loc_desc_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_multiview", loc_multiview_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_full_bank", loc_full_bank_loss.item(), iteration)
                tb_writer.add_scalar(
                    "train_loss/loc_clean_hard_negative",
                    loc_clean_hard_negative_loss.item(),
                    iteration,
                )
                tb_writer.add_scalar("train_loss/loc_anchor", loc_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_reproj", loc_reproj_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_pnp", loc_pnp_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_dense_kl", loc_dense_kl_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_dense_rank", loc_dense_rank_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_proto", loc_proto_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_rank", loc_rank_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_opacity", loc_opacity_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_overlay_reg", loc_overlay_reg_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/surfel_loc_anchor_reg", loc_surface_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/geometry_anchor", geom_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/lafgs_geometry_residual", loc_geometry_residual_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/total", total_loss.item(), iteration)
                tb_writer.add_scalar("train/points", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("train/frozen_child_feature_count", frozen_child_feature_count, iteration)
                if teacher_out is not None:
                    for name, value in getattr(teacher_out, "diagnostics", {}).items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(f"train_diagnostics/{name}", float(value), iteration)

            if (
                teacher_out is not None
                and teacher_out.loc_visible_idx is not None
            ):
                has_visible_stats = teacher_out.loc_visible_idx.numel() > 0
                if has_visible_stats:
                    loc_training_summary["stats_candidate_episodes"] += 1
                else:
                    loc_training_summary["stats_skip_no_visible_episodes"] += 1
                if not bool(pseudo_query_reliability.get("update_stats", True)):
                    loc_training_summary["stats_skip_reliability_episodes"] += 1
                if not bool(pseudo_query_stage_direct_policy.get("update_stats", True)):
                    loc_training_summary["stats_skip_stage_episodes"] += 1
                if loc_stats_update_allowed:
                    gaussians.add_localization_stats(
                        full_idx=teacher_out.loc_visible_idx,
                        means2d_grad=loc_grad,
                        radii=teacher_out.loc_radii,
                        episode_stats=teacher_out.stats,
                        ema_decay=args.loc_ema_decay,
                    )
                    loc_training_summary["stats_update_episodes"] += 1
                    loc_training_summary["stats_update_points_total"] += int(
                        teacher_out.loc_visible_idx.numel()
                    )

            stage_name = str(stage_loss_weights["stage"])
            loc_training_summary[f"stage_{stage_name}_episodes"] = (
                loc_training_summary.get(f"stage_{stage_name}_episodes", 0) + 1
            )
            loc_training_summary["stage_last"] = stage_name
            loc_training_summary["stage_base_weight_last"] = float(stage_loss_weights["base"])
            loc_training_summary["stage_loc_weight_last"] = float(stage_loss_weights["loc"])
            loc_training_summary["stage_geometry_anchor_weight_last"] = float(
                stage_loss_weights["geometry_anchor"]
            )

            rgb_densify_active = lafgs_rgb_densify_active(args, lafgs_step) and (
                int(lafgs_step) < int(opt.densify_until_iter)
            )
            if rgb_densify_active:
                viewspace_points = render_pkg.get("viewspace_points")
                visibility_filter = render_pkg.get("visibility_filter")
                viewspace_grad = getattr(viewspace_points, "grad", None)
                if viewspace_points is not None and visibility_filter is not None and viewspace_grad is not None:
                    gaussians.add_densification_stats_gsplat(
                        viewspace_points,
                        visibility_filter,
                        image.shape[2],
                        image.shape[1],
                    )
                    loc_training_summary["rgb_densify_stats_episodes"] = (
                        loc_training_summary.get("rgb_densify_stats_episodes", 0) + 1
                    )

                    if (
                        int(lafgs_step) > int(opt.densify_from_iter)
                        and int(lafgs_step) % int(opt.densification_interval) == 0
                    ):
                        before_count = int(gaussians.get_xyz.shape[0])
                        size_threshold = 20 if int(lafgs_step) > int(opt.opacity_reset_interval) else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            loc_birth_iteration=int(lafgs_step),
                        )
                        child_prune_stats = _prune_lafgs_rgb_densify_child_outliers(
                            gaussians,
                            args.lafgs_rgb_densify_child_max_source_drift,
                        )
                        after_count = int(gaussians.get_xyz.shape[0])
                        loc_training_summary["rgb_densify_mutation_events"] = (
                            loc_training_summary.get("rgb_densify_mutation_events", 0) + 1
                        )
                        child_pruned = int(child_prune_stats.get("pruned", 0) or 0)
                        loc_training_summary["rgb_densify_child_outlier_pruned_total"] = (
                            loc_training_summary.get("rgb_densify_child_outlier_pruned_total", 0)
                            + child_pruned
                        )
                        if child_pruned > 0:
                            loc_training_summary["rgb_densify_child_outlier_prune_events"] = (
                                loc_training_summary.get("rgb_densify_child_outlier_prune_events", 0) + 1
                            )
                        if "max_source_drift" in child_prune_stats:
                            loc_training_summary["rgb_densify_child_source_drift_max"] = max(
                                loc_training_summary.get("rgb_densify_child_source_drift_max", 0.0),
                                float(child_prune_stats["max_source_drift"]),
                            )
                        if "mean_source_drift" in child_prune_stats:
                            loc_training_summary["rgb_densify_child_source_drift_mean_last"] = float(
                                child_prune_stats["mean_source_drift"]
                            )
                        loc_training_summary["rgb_densify_point_count_before_last"] = before_count
                        loc_training_summary["rgb_densify_point_count_after_last"] = after_count
                        loc_training_summary["rgb_densify_point_count_delta_total"] = (
                            loc_training_summary.get("rgb_densify_point_count_delta_total", 0)
                            + after_count
                            - before_count
                        )

                    if int(lafgs_step) % int(opt.opacity_reset_interval) == 0 or (
                        dataset.white_background and int(lafgs_step) == int(opt.densify_from_iter)
                    ):
                        gaussians.reset_opacity()
                        loc_training_summary["rgb_opacity_reset_events"] = (
                            loc_training_summary.get("rgb_opacity_reset_events", 0) + 1
                        )
                else:
                    loc_training_summary["rgb_densify_skip_no_grad"] = (
                        loc_training_summary.get("rgb_densify_skip_no_grad", 0) + 1
                    )

            _clip_lafgs_geometry_gradients(
                gaussians,
                getattr(args, "lafgs_geometry_grad_clip_abs", 0.0),
                summary=loc_training_summary,
            )
            gaussians.optimizer.step()
            _record_geometry_optimizer_diagnostics(
                loc_training_summary,
                gaussians,
                phase,
                xyz_before=geometry_xyz_before,
                record_lr_grad=False,
                geometry_active=geometry_update_active,
            )
            gaussians.optimizer.zero_grad(set_to_none=True)

            if (
                topology_controller is not None
                and phase in {"topology", "closed_loop", "full"}
                and topology_controller.should_update(iteration)
            ):
                topology_controller.update(gaussians, scene.cameras_extent, iteration)

            loc_feature_anchor = _refresh_feature_anchor_if_point_count_changed(gaussians, loc_feature_anchor)

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving LA Gaussians")
                scene.save(iteration)
                if should_save_locaware_full_checkpoint(args, iteration):
                    torch.save(
                        {
                            "version": 2,
                            "iteration": iteration,
                            "model_params": gaussians.capture(),
                            "localization_state": gaussians.capture_localization_state(),
                            "config": vars(args),
                        },
                        os.path.join(dataset.model_path, f"chkpnt_locaware_{iteration}.pth"),
                    )

            if iteration in args.test_iterations:
                psnr_val = psnr(image, losses["gt_image"]).mean().item()
                print(
                    f"\n[ITER {iteration}] base {base_loss.item():.6f} "
                    f"loc {loc_loss.item():.6f} psnr {psnr_val:.3f}"
                )

    if tb_writer:
        tb_writer.close()
    with torch.no_grad():
        obs = getattr(gaussians, "loc_observation_count", None)
        if torch.is_tensor(obs):
            obs_cpu = obs.detach().cpu()
            loc_training_summary.update(
                {
                    "observed_points": int((obs_cpu > 0).sum().item()),
                    "observed_points_ge_1": int((obs_cpu >= 1).sum().item()),
                    "observed_points_ge_2": int((obs_cpu >= 2).sum().item()),
                    "observed_points_ge_4": int((obs_cpu >= 4).sum().item()),
                    "observed_points_max": int(obs_cpu.max().item()) if obs_cpu.numel() else 0,
                }
            )
        _record_final_geometry_delta_summary(loc_training_summary, gaussians, geometry_delta_reference)
    loc_summary_path = os.path.join(dataset.model_path, "loc_training_summary.json")
    with open(loc_summary_path, "w") as f:
        json.dump(loc_training_summary, f, indent=2, sort_keys=True)
    print(f"Saved LA localization training summary: {loc_summary_path} {loc_training_summary}")


if __name__ == "__main__":
    parser = ArgumentParser(description="LA-STDLoc training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    add_locaware_training_args(parser)

    args = parser.parse_args(sys.argv[1:])
    append_unique_iteration(args.save_iterations, args.iterations)
    append_unique_iteration(args.test_iterations, args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.train_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), args)
    print("\nLA-STDLoc training complete.")
