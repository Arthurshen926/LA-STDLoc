#!/usr/bin/env python
"""Fit a train-only confidence model for an unchanged native candidate graph.

This is deliberately not a descriptor-training script.  It replays native
SuperPoint descriptors from the LaFGS training split, labels the *existing*
cosine top-1 pair with rasterizer visibility plus GT reprojection, and fits a
six-feature logistic regressor.  Candidate-validation and test views are
never read for fitting or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.direct_landmark_teacher import project_landmarks_to_query
from localization_training.episode_sampler import split_support_query_cameras
from localization_training.full_primitive_retrieval import chunked_exact_topk
from localization_training.native_matchability import (
    FEATURE_NAMES,
    build_native_matchability_features,
    calibrated_native_matchability,
    validate_native_matchability_state,
)


def _tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_names_sha256(names):
    normalized = sorted(str(name).replace("\\", "/") for name in names)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def _uniform_subsample(names, maximum):
    maximum = int(maximum)
    if maximum <= 0 or len(names) <= maximum:
        return list(names)
    positions = torch.linspace(0, len(names) - 1, maximum).round().long().tolist()
    return [names[position] for position in positions]


def _split_names(names, *, ratio, seed, mode):
    cameras = [SimpleNamespace(image_name=name) for name in names]
    support, query = split_support_query_cameras(
        cameras,
        query_ratio=float(ratio),
        seed=int(seed),
        mode=str(mode),
    )
    return [camera.image_name for camera in support], [camera.image_name for camera in query]


def _select_native_keypoints(keypoint_scores, maximum):
    count = int(keypoint_scores.numel())
    maximum = int(maximum)
    if maximum <= 0 or count <= maximum:
        return torch.arange(count, dtype=torch.long)
    # Stable sorting keeps the replay deterministic when detector scores tie.
    return torch.argsort(keypoint_scores.float(), descending=True, stable=True)[:maximum]


def _binary_metrics(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64).reshape(-1).cpu()
    labels = torch.as_tensor(labels, dtype=torch.bool).reshape(-1).cpu()
    probabilities = torch.sigmoid(logits)
    positive = int(labels.sum().item())
    total = int(labels.numel())
    negative = total - positive
    result = {
        "count": total,
        "positive_count": positive,
        "positive_rate": float(positive / max(total, 1)),
        "brier": float(((probabilities - labels.double()) ** 2).mean().item())
        if total
        else 0.0,
        "nll": float(
            F.binary_cross_entropy(probabilities.clamp(1e-8, 1.0 - 1e-8), labels.double()).item()
        )
        if total
        else 0.0,
    }
    if positive == 0 or negative == 0:
        result.update({"auroc": None, "average_precision": None})
        return result
    order = torch.argsort(logits, descending=False, stable=True)
    ranks = torch.empty(total, dtype=torch.float64)
    ranks[order] = torch.arange(1, total + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels].sum()
    result["auroc"] = float(
        ((positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)).item()
    )
    descending = torch.argsort(logits, descending=True, stable=True)
    sorted_labels = labels[descending].double()
    precision = torch.cumsum(sorted_labels, dim=0) / torch.arange(
        1, total + 1, dtype=torch.float64
    )
    result["average_precision"] = float(
        ((precision * sorted_labels).sum() / positive).item()
    )
    return result


def _fit_logistic(features, labels, *, l2, max_iterations):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("invalid native matchability training features")
    if features.shape[0] != labels.numel() or labels.numel() == 0:
        raise ValueError("native matchability training labels are empty or misaligned")
    if int(labels.sum().item()) == 0 or int((1.0 - labels).sum().item()) == 0:
        raise ValueError("native matchability requires both clean and false candidates")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    weights = torch.nn.Parameter(torch.zeros(normalized.shape[1], device=features.device))
    base_rate = labels.mean().clamp(1e-5, 1.0 - 1e-5)
    bias = torch.nn.Parameter(torch.logit(base_rate).detach().clone())
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=max(int(max_iterations), 1),
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        logits = normalized @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if float(l2) > 0.0:
            loss = loss + 0.5 * float(l2) * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    logits = (normalized @ weights + bias).detach()
    return {
        "feature_mean": mean.detach(),
        "feature_std": std.detach(),
        "weights": weights.detach(),
        "bias": bias.detach(),
        "training_metrics": _binary_metrics(logits, labels),
    }


def _model_logits(features, model):
    return (
        (features - model["feature_mean"]) / model["feature_std"]
    ) @ model["weights"] + model["bias"]


def _load_payload(path, *, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Train-only calibrated solver confidence for native LaFGS matching."
    )
    parser.add_argument("--field_state", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--attractor_prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--max_keypoints_per_view", type=int, default=512)
    parser.add_argument("--max_train_views", type=int, default=0)
    parser.add_argument("--positive_radius_px", type=float, default=2.0)
    parser.add_argument("--entropy_temperature", type=float, default=0.05)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--inner_validation_ratio", type=float, default=0.2)
    parser.add_argument("--inner_split_seed_offset", type=int, default=101)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--lbfgs_iterations", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.topk) < 2:
        raise ValueError("--topk must be at least 2")
    if not 0.0 <= float(args.confidence_floor) < 1.0:
        raise ValueError("--confidence_floor must be in [0, 1)")
    if float(args.positive_radius_px) <= 0.0:
        raise ValueError("--positive_radius_px must be positive")
    if not 0.0 < float(args.inner_validation_ratio) < 1.0:
        raise ValueError("--inner_validation_ratio must be in (0, 1)")

    field_path = Path(args.field_state).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing calibrator: {output_path}")
    field = _load_payload(field_path, label="field state")
    field_config = field.get("config", {})
    field_features = F.normalize(
        torch.as_tensor(field["landmark_features"], dtype=torch.float32).reshape(
            -1, torch.as_tensor(field["landmark_features"]).shape[-1]
        ),
        dim=1,
    )
    field_xyz = torch.as_tensor(field["landmark_xyz"], dtype=torch.float32).reshape(-1, 3)
    landmark_indices = torch.as_tensor(field["landmark_indices"], dtype=torch.long).reshape(-1)
    if field_features.shape[0] != field_xyz.shape[0] or field_features.shape[0] != landmark_indices.numel():
        raise ValueError("field state has misaligned feature, geometry, or landmark IDs")

    cache_payload = _load_payload(args.query_cache, label="query cache")
    raw_query_cache = cache_payload.get("queries", {})
    if not isinstance(raw_query_cache, dict) or not raw_query_cache:
        raise ValueError("query cache does not contain queries")
    query_cache = {
        str(name).replace("\\", "/"): value
        for name, value in raw_query_cache.items()
    }
    if len(query_cache) != len(raw_query_cache):
        raise ValueError("query cache contains duplicate normalized camera names")
    all_names = sorted(query_cache)
    split_mode = str(field_config.get("split_mode", "stratified_temporal_block"))
    split_seed = int(field_config.get("split_seed", 2026))
    validation_ratio = float(field_config.get("validation_ratio", 0.2))
    train_names, heldout_names = _split_names(
        all_names,
        ratio=validation_ratio,
        seed=split_seed + 1,
        mode=split_mode,
    )
    expected_train_hash = field_config.get("train_camera_names_sha256")
    expected_heldout_hash = field_config.get("validation_camera_names_sha256")
    if expected_train_hash and _camera_names_sha256(train_names) != expected_train_hash:
        raise ValueError("query-cache train split does not match the field-state contract")
    if expected_heldout_hash and _camera_names_sha256(heldout_names) != expected_heldout_hash:
        raise ValueError("query-cache holdout split does not match the field-state contract")
    selected_train_names = _uniform_subsample(train_names, args.max_train_views)

    visibility_payload = _load_payload(args.visibility_cache, label="visibility cache")
    raw_visibility = visibility_payload.get("visibility", {})
    visibility = {
        str(name).replace("\\", "/"): value
        for name, value in raw_visibility.items()
    }
    missing_visibility = [name for name in selected_train_names if name not in visibility]
    if missing_visibility:
        raise ValueError(
            "visibility cache is missing selected training views: "
            f"{missing_visibility[:3]}"
        )
    prior = _load_payload(args.attractor_prior, label="false-attractor prior")
    if str(prior.get("split", "")) != "train_only":
        raise ValueError("native matchability requires a train_only false-attractor prior")
    if not torch.equal(
        torch.as_tensor(prior.get("landmark_indices", []), dtype=torch.long).reshape(-1),
        landmark_indices.cpu(),
    ):
        raise ValueError("false-attractor prior landmark IDs do not match field state")
    if expected_train_hash and prior.get("train_camera_names_sha256") != expected_train_hash:
        raise ValueError("false-attractor prior train split does not match field state")
    statistics = prior.get("statistics", {})
    false_rate = torch.as_tensor(statistics.get("false_rate", []), dtype=torch.float32).reshape(-1)
    incoming_count = torch.as_tensor(statistics.get("incoming_count", []), dtype=torch.float32).reshape(-1)
    if false_rate.numel() != landmark_indices.numel() or incoming_count.numel() != landmark_indices.numel():
        raise ValueError("false-attractor prior statistics do not match field bank")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    map_features = field_features.to(device=device)
    map_xyz = field_xyz.to(device=device)
    false_rate_device = false_rate.to(device=device)
    incoming_device = incoming_count.to(device=device)
    batches = {}
    feature_count = 0
    clean_count = 0
    required_cache_keys = {
        "native_keypoints",
        "native_descriptors",
        "native_scores",
        "native_K",
        "pose_w2c",
        "native_input_hw",
    }
    for name in tqdm(selected_train_names, desc="Train-only native matchability replay"):
        cached = query_cache[name]
        missing = sorted(required_cache_keys - set(cached))
        if missing:
            raise ValueError(f"query cache {name} is missing {', '.join(missing)}")
        keypoints_cpu = torch.as_tensor(cached["native_keypoints"], dtype=torch.float32)
        descriptors_cpu = torch.as_tensor(cached["native_descriptors"], dtype=torch.float32)
        scores_cpu = torch.as_tensor(cached["native_scores"], dtype=torch.float32).reshape(-1)
        if keypoints_cpu.ndim != 2 or keypoints_cpu.shape[1] != 2:
            raise ValueError(f"invalid native keypoints in {name}")
        if descriptors_cpu.ndim != 2 or descriptors_cpu.shape[0] != keypoints_cpu.shape[0]:
            raise ValueError(f"invalid native descriptors in {name}")
        if scores_cpu.numel() != keypoints_cpu.shape[0]:
            raise ValueError(f"invalid native scores in {name}")
        selected = _select_native_keypoints(scores_cpu, args.max_keypoints_per_view)
        if selected.numel() == 0:
            continue
        keypoints = keypoints_cpu[selected].to(device=device)
        descriptors = F.normalize(descriptors_cpu[selected].to(device=device), dim=1)
        keypoint_scores = scores_cpu[selected].to(device=device)
        retrieval = chunked_exact_topk(
            descriptors,
            map_features,
            topk=args.topk,
            chunk_size=args.chunk_size,
        )
        height, width = map(int, cached["native_input_hw"])
        K = torch.as_tensor(cached["native_K"], device=device, dtype=torch.float32)
        pose = torch.as_tensor(cached["pose_w2c"], device=device, dtype=torch.float32)
        projected_uv, _, projected = project_landmarks_to_query(
            map_xyz,
            K,
            pose,
            height=height,
            width=width,
            pixel_center_offset=0.5,
        )
        visible = torch.as_tensor(visibility[name], device=device, dtype=torch.bool).reshape(-1)
        if visible.numel() != map_xyz.shape[0]:
            raise ValueError(f"visibility bank size mismatch for {name}")
        top1 = retrieval.indices[:, 0]
        reprojection_error = torch.linalg.norm(
            projected_uv[top1] - keypoints, dim=1
        )
        clean = (
            projected[top1]
            & visible[top1]
            & torch.isfinite(reprojection_error)
            & (reprojection_error <= float(args.positive_radius_px))
        )
        features = build_native_matchability_features(
            retrieval.scores,
            retrieval.indices,
            false_attractor_rate=false_rate_device,
            incoming_count=incoming_device,
            keypoint_scores=keypoint_scores,
            entropy_temperature=args.entropy_temperature,
        )
        batches[name] = (features.detach().cpu(), clean.detach().cpu())
        feature_count += int(clean.numel())
        clean_count += int(clean.sum().item())

    if not batches:
        raise RuntimeError("no native candidates were available for calibrator fitting")
    used_names = [name for name in selected_train_names if name in batches]
    inner_train_names, inner_validation_names = _split_names(
        used_names,
        ratio=args.inner_validation_ratio,
        seed=split_seed + int(args.inner_split_seed_offset),
        mode=split_mode,
    )
    train_features = torch.cat([batches[name][0] for name in inner_train_names], dim=0).to(device)
    train_labels = torch.cat([batches[name][1] for name in inner_train_names], dim=0).to(device)
    validation_features = torch.cat(
        [batches[name][0] for name in inner_validation_names], dim=0
    ).to(device)
    validation_labels = torch.cat(
        [batches[name][1] for name in inner_validation_names], dim=0
    ).to(device)
    inner_model = _fit_logistic(
        train_features,
        train_labels,
        l2=args.l2,
        max_iterations=args.lbfgs_iterations,
    )
    inner_validation_metrics = _binary_metrics(
        _model_logits(validation_features, inner_model).detach(), validation_labels
    )

    all_features = torch.cat([batches[name][0] for name in used_names], dim=0).to(device)
    all_labels = torch.cat([batches[name][1] for name in used_names], dim=0).to(device)
    final_model = _fit_logistic(
        all_features,
        all_labels,
        l2=args.l2,
        max_iterations=args.lbfgs_iterations,
    )
    provisional_state = {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "topk": int(args.topk),
        "entropy_temperature": float(args.entropy_temperature),
        "confidence_floor": float(args.confidence_floor),
        "feature_mean": final_model["feature_mean"].detach().cpu(),
        "feature_std": final_model["feature_std"].detach().cpu(),
        "weights": final_model["weights"].detach().cpu(),
        "bias": final_model["bias"].detach().cpu(),
        "landmark_false_attractor_rate": false_rate.cpu(),
        "landmark_incoming_count": incoming_count.cpu(),
    }
    validate_native_matchability_state(
        provisional_state, landmark_count=int(landmark_indices.numel())
    )
    # This is a monotonic transform of the logistic probability.  Record it
    # explicitly so an evaluator cannot silently turn it into a threshold.
    final_confidence = calibrated_native_matchability(
        all_features, provisional_state
    ).detach()
    state = {
        **provisional_state,
        "provenance": {
            "field_state_path": str(field_path),
            "field_state_file_sha256": _file_sha256(field_path),
            "landmark_indices_sha256": _tensor_sha256(landmark_indices),
            "landmark_features_sha256": _tensor_sha256(field_features),
            "landmark_xyz_sha256": _tensor_sha256(field_xyz),
            "attractor_prior_path": str(Path(args.attractor_prior).resolve()),
            "attractor_prior_file_sha256": _file_sha256(args.attractor_prior),
            "query_cache_path": str(Path(args.query_cache).resolve()),
            "query_cache_signature": cache_payload.get("signature"),
            "visibility_cache_path": str(Path(args.visibility_cache).resolve()),
            "visibility_cache_signature": visibility_payload.get("signature"),
            "train_camera_names_sha256": _camera_names_sha256(train_names),
            "field_validation_camera_names_sha256": _camera_names_sha256(heldout_names),
            "split_mode": split_mode,
            "split_seed": split_seed,
            "field_validation_ratio": validation_ratio,
            "fit_uses_candidate_validation": False,
            "fit_uses_test": False,
        },
        "training": {
            "label_definition": "rasterizer_visible_and_gt_reprojection_le_2px_v1",
            "positive_radius_px": float(args.positive_radius_px),
            "max_keypoints_per_view": int(args.max_keypoints_per_view),
            "selected_train_view_count": len(selected_train_names),
            "used_train_view_count": len(used_names),
            "candidate_validation_view_count": len(heldout_names),
            "candidate_count": feature_count,
            "clean_count": clean_count,
            "inner_train_view_count": len(inner_train_names),
            "inner_validation_view_count": len(inner_validation_names),
            "inner_training": inner_model["training_metrics"],
            "inner_validation": inner_validation_metrics,
            "refit_training": final_model["training_metrics"],
            "refit_confidence": {
                "mean": float(final_confidence.mean().item()),
                "p10": float(torch.quantile(final_confidence, 0.10).item()),
                "p90": float(torch.quantile(final_confidence, 0.90).item()),
            },
            "l2": float(args.l2),
            "lbfgs_iterations": int(args.lbfgs_iterations),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, output_path)
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output": str(output_path),
                "provenance": state["provenance"],
                "training": state["training"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps({"output": str(output_path), "training": state["training"]}, indent=2))


if __name__ == "__main__":
    main()
