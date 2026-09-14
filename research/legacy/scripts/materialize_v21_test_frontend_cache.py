#!/usr/bin/env python3
"""Materialize one role/shard of the V21 real-test frontend baseline cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from common.config import (
    load_mainline_config,
    load_scene_calibration,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from features.superpoint import (
    SUPERPOINT_WEIGHT_SHA256,
    resolve_superpoint_weights,
)
from localization.localizer import SparseLocalizer
from localization.pose_solver import pose_error
from map_learning.metric import validate_zero_identity_metric
from map_learning.v8_feedback_controller import task_error
from map_learning.v21_test_cache import (
    ALLOWED_ROLES,
    CACHE_SCHEMA,
    CACHE_VERSION,
    VALID_MASK_SEMANTICS,
    atomic_torch_save_fresh,
    build_query_record,
    build_shard_registry,
    pose_w2c_sha256,
    records_for_shard,
    sha256_json,
    training_consumer_policy,
    validate_split_manifest,
)


IDENTITY_PROTOCOLS = {
    "v6_identity_shared_metric",
    "rendered_track_map_bound_identity",
    "v20_sparse_anchor_native_query_identity",
}
PRODUCER_SOURCES = (
    "data/datasets.py",
    "features/superpoint.py",
    "localization/frontend.py",
    "localization/localizer.py",
    "localization/matcher.py",
    "localization/pose_solver.py",
    "map_learning/v21_test_cache.py",
    "scripts/materialize_v21_test_frontend_cache.py",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"V21 JSON input must be a mapping: {path}")
    return value


def _source(path: Path) -> dict[str, str | int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _verify_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        path = Path(str(source["path"]))
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(source["size_bytes"])
            or sha256_file(path) != source["sha256"]
        ):
            raise RuntimeError(f"V21 cache source changed while running: {path}")


def _unique_sources(paths: list[Path]) -> list[dict[str, Any]]:
    unique = {path.expanduser().resolve() for path in paths}
    return [_source(path) for path in sorted(unique, key=str)]


def _dataset_geometry_sources(root: Path) -> list[Path]:
    sparse = root / "sparse/0"
    sources = []
    for stem in ("images", "cameras"):
        candidates = (sparse / f"{stem}.bin", sparse / f"{stem}.txt")
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            raise FileNotFoundError(f"missing COLMAP {stem} registry in {sparse}")
        sources.append(selected)
    return sources


def _mask_source(dataset_root: Path, images: str) -> Path | None:
    return next(
        (
            path
            for path in (
                dataset_root / images / "masks.pkl",
                dataset_root / "masks.pkl",
            )
            if path.is_file()
        ),
        None,
    )


def _resolve_calibration(
    requested: Path | None, *, stable_map: Path
) -> tuple[Path | None, dict | None]:
    path = requested
    if path is None:
        inferred = stable_map.parent / "scene_calibration.json"
        path = inferred if inferred.is_file() else None
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    return resolved, load_scene_calibration(resolved)


def _validate_dataset_registry(
    *,
    manifest: dict,
    manifest_path: Path,
    dataset: ColmapDataset,
    dataset_root: Path,
    images: str,
) -> dict[int, tuple[dict, Any]]:
    registry = manifest["dataset_registry"]
    if (
        Path(str(registry.get("dataset_root", ""))).expanduser().resolve()
        != dataset_root
        or str(registry.get("images")) != str(images)
    ):
        raise ValueError("V21 manifest dataset root/images differ from the CLI")
    registry_path = Path(str(registry.get("path", ""))).expanduser().resolve()
    if (
        not registry_path.is_file()
        or sha256_file(registry_path) != registry.get("sha256")
    ):
        raise ValueError("V21 dataset test registry source differs from the manifest")

    cameras = list(dataset.split("test"))
    records = sorted(manifest["records"], key=lambda row: int(row["query_index"]))
    if len(cameras) != len(records):
        raise ValueError("V21 manifest does not cover the exact dataset test split")
    bound: dict[int, tuple[dict, Any]] = {}
    for query_index, (record, camera) in enumerate(zip(records, cameras)):
        image_path = Path(camera.image_path).expanduser().resolve()
        if (
            int(record["query_index"]) != query_index
            or str(record["image_name"]).replace("\\", "/")
            != str(camera.image_name).replace("\\", "/")
            or Path(str(record["image_path"])).expanduser().resolve() != image_path
            or pose_w2c_sha256(camera.pose_w2c) != record["pose_w2c_sha256"]
        ):
            raise ValueError("V21 manifest query/image/GT registry differs from COLMAP")
        if record["image_sha256"] is not None:
            if (
                not image_path.is_file()
                or sha256_file(image_path) != record["image_sha256"]
            ):
                raise ValueError("V21 real test RGB differs from the split manifest")
        bound[query_index] = (record, camera)
    # Retain the manifest path in the exception context for malformed callers.
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return bound


def materialize(args: argparse.Namespace) -> dict:
    """Materialize one immutable cache shard and return its payload."""

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 cache output already exists: {output}")
    role = str(args.role)
    if role not in ALLOWED_ROLES:
        raise ValueError(f"V21 role must be one of {sorted(ALLOWED_ROLES)}")
    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("V21 shard coordinates are invalid")

    manifest_path = Path(args.split_manifest).expanduser().resolve()
    stable_map = Path(args.stable_map).expanduser().resolve()
    identity_metric = Path(args.identity_metric).expanduser().resolve()
    dataset_root = Path(args.dataset).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    manifest = _load_json(manifest_path)
    selected_records = validate_split_manifest(manifest, role=role)
    manifest_sha = sha256_file(manifest_path)
    map_sha = sha256_file(stable_map)
    metric_sha = sha256_file(identity_metric)
    if (
        manifest.get("stable_map_sha256") != map_sha
        or Path(str(manifest.get("stable_map", ""))).expanduser().resolve()
        != stable_map
    ):
        raise ValueError("V21 split manifest is not bound to the requested stable map")

    config = load_mainline_config(config_path)
    deployment = config.values["deployment"]
    dataset = ColmapDataset(dataset_root, images=args.images)
    bound_cameras = _validate_dataset_registry(
        manifest=manifest,
        manifest_path=manifest_path,
        dataset=dataset,
        dataset_root=dataset_root,
        images=args.images,
    )
    calibration_cameras = dataset.split("mapping")
    keypoint_count = resolve_keypoint_count(deployment, calibration_cameras)
    calibration_path, scene_calibration = _resolve_calibration(
        args.scene_calibration, stable_map=stable_map
    )
    reprojection_error_px = resolve_reprojection_error_px(
        deployment, calibration_cameras, scene_calibration
    )

    map_payload = torch.load(stable_map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        identity_metric, map_location="cpu", weights_only=False
    )
    if map_payload.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("V21 stable map has an unsupported schema")
    anchor_ids = torch.as_tensor(map_payload.get("anchor_ids")).long().reshape(-1)
    anchor_features = torch.as_tensor(map_payload.get("anchor_features")).float()
    if (
        anchor_ids.numel() == 0
        or anchor_features.ndim != 2
        or anchor_features.shape[0] != anchor_ids.numel()
        or torch.unique(anchor_ids).numel() != anchor_ids.numel()
    ):
        raise ValueError("V21 stable map Anchor registry is invalid")
    validate_zero_identity_metric(
        metric_payload,
        descriptor_dim=int(anchor_features.shape[1]),
        landmark_indices=anchor_ids,
        map_path=str(stable_map),
        map_sha256=map_sha,
        allowed_protocols=IDENTITY_PROTOCOLS,
    )
    if (
        metric_payload.get("photometric_canonicalization_contract")
        != map_payload.get("photometric_canonicalization_contract")
    ):
        raise ValueError("V21 identity metric/map photometric contracts differ")

    weights_path = resolve_superpoint_weights()
    weights_sha = sha256_file(weights_path)
    if weights_sha != SUPERPOINT_WEIGHT_SHA256:
        raise RuntimeError("V21 SuperPoint weights differ from the frozen frontend")
    localizer = SparseLocalizer(
        stable_map,
        identity_metric,
        device=args.device,
        keypoint_count=keypoint_count,
        nms_radius=int(deployment["nms"]),
        reprojection_error_px=reprojection_error_px,
        confidence=deployment["confidence"],
        max_iterations=deployment["maximum_iterations"],
        min_iterations=deployment["minimum_iterations"],
        seed=int(args.seed),
        suppress_duplicate_anchors=False,
        guided_sampling=False,
        group_aware_pose=False,
        assignment_topk=0,
        profile_mode=True,
    )
    if localizer.anchor_extra_prototype_features.numel():
        raise ValueError(
            "V21 plain global-Top1 baseline forbids extra owner prototypes"
        )
    if localizer.frontend.context_adapter is not None:
        raise ValueError("V21 baseline requires native descriptors and identity metric")

    registry = build_shard_registry(
        selected_records,
        role=role,
        shard_count=shard_count,
        split_manifest_sha256=manifest_sha,
    )
    registry_rows = records_for_shard(registry, shard_index=shard_index)
    selected_by_query = {
        int(record["query_index"]): record for record in selected_records
    }
    mask_path = _mask_source(dataset_root, args.images)
    dataset_registry_path = Path(manifest["dataset_registry"]["path"])
    source_paths = [
        manifest_path,
        stable_map,
        identity_metric,
        config_path,
        weights_path,
        dataset_registry_path,
        *_dataset_geometry_sources(dataset_root),
        *(REPOSITORY_ROOT / path for path in PRODUCER_SOURCES),
    ]
    if calibration_path is not None:
        source_paths.append(calibration_path)
    if mask_path is not None:
        source_paths.append(mask_path)
    source_paths.extend(
        Path(bound_cameras[row["query_index"]][1].image_path)
        for row in registry_rows
    )
    sources = _unique_sources(source_paths)
    source_by_path = {source["path"]: source for source in sources}

    preprocessing_config = {
        "schema": "lafgs_v21_native_superpoint_preprocessing",
        "version": 1,
        "dataset_loader": "data.datasets.ColmapDataset.load_image",
        "dataset_images": str(args.images),
        "rgb_value_range": "uint8_to_float32_[0,1]",
        "resize": "bilinear_align_corners_false_to_colmap_camera_hw",
        "channel_policy": "grayscale_expand_then_first_three_rgb_channels",
        "photometric_canonicalization_contract": map_payload.get(
            "photometric_canonicalization_contract"
        ),
        "frontend": "localization.frontend.NativeSuperPointFrontend",
        "frontend_api": "SuperPoint.detectAndCompute",
        "keypoint_count": int(keypoint_count),
        "nms_radius": int(deployment["nms"]),
        "descriptor_normalization": "l2_then_zero_identity_metric_then_l2",
        "valid_mask_semantics": dict(VALID_MASK_SEMANTICS),
        "superpoint_weights_sha256": weights_sha,
        "mainline_config_sha256": config.file_sha256,
        "resolved_mainline_config_sha256": config.resolved_sha256,
    }
    preprocessing_sha = sha256_json(preprocessing_config)

    records = []
    for registry_row in registry_rows:
        query_index = int(registry_row["query_index"])
        split_record = selected_by_query[query_index]
        _, camera = bound_cameras[query_index]
        image = dataset.load_image(camera)
        valid_mask = dataset.valid_mask(camera)
        result = localizer.localize(
            image,
            fov_x=float(camera.fov_x),
            fov_y=float(camera.fov_y),
            valid_mask=valid_mask,
        )
        sparse = result.sparse_features
        matches = result.matches
        expected_query_rows = torch.arange(
            sparse.keypoints.shape[0], device=matches.keypoint_indices.device
        )
        if not torch.equal(matches.keypoint_indices, expected_query_rows):
            raise RuntimeError("V21 baseline did not preserve one Top1 per native row")
        rotation_deg, translation_cm = pose_error(
            result.pose.pose_w2c, camera.pose_w2c
        )
        records.append(
            build_query_record(
                split_record=split_record,
                keypoints=sparse.keypoints,
                descriptors=sparse.descriptors,
                scores=sparse.scores,
                image_hw=sparse.image_hw,
                valid_mask=valid_mask,
                intrinsics=torch.from_numpy(result.intrinsic),
                pose_w2c=torch.from_numpy(camera.pose_w2c),
                winner_anchor_rows=matches.anchor_indices,
                winner_anchor_ids=localizer.anchor_ids[
                    matches.anchor_indices.detach().cpu()
                ],
                winner_scores=matches.scores,
                baseline_pose_w2c=torch.from_numpy(result.pose.pose_w2c),
                baseline_inliers=torch.from_numpy(result.pose.inliers),
                rotation_error_deg=rotation_deg,
                translation_error_cm=translation_cm,
                task_error=task_error(translation_cm, rotation_deg),
            )
        )

    _verify_sources(sources)
    if (
        source_by_path[str(manifest_path)]["sha256"] != manifest_sha
        or source_by_path[str(stable_map)]["sha256"] != map_sha
        or source_by_path[str(identity_metric)]["sha256"] != metric_sha
        or source_by_path[str(config_path)]["sha256"] != config.file_sha256
        or source_by_path[str(weights_path)]["sha256"] != weights_sha
    ):
        raise RuntimeError("V21 primary cache input changed while running")
    policy = training_consumer_policy(role)
    payload = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": role,
        "split_manifest_sha256": manifest_sha,
        "training_consumer_allowed": policy["training_consumer_allowed"],
        "training_consumers_allowed": policy["training_consumers_allowed"],
        "consumer_policy": policy,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "query_count": len(records),
        "role_query_count": int(registry["role_query_count"]),
        "anchor_count": int(anchor_ids.numel()),
        "descriptor_dim": int(anchor_features.shape[1]),
        "shard_registry": registry,
        "frontend_contract": preprocessing_config,
        "preprocessing_config_sha256": preprocessing_sha,
        "baseline_contract": {
            "matching": "exact_global_cosine_top1_lower_anchor_row_tie_break",
            "pose_solver": "single_standard_poselib_absolute_pose",
            "cached_keypoints": "native_integer_grid_without_pixel_center_offset",
            "pose_solver_points_2d": "cached_keypoints_plus_0.5",
            "pixel_center_offset": 0.5,
            "reprojection_error_px": float(reprojection_error_px),
            "confidence": float(deployment["confidence"]),
            "maximum_iterations": int(deployment["maximum_iterations"]),
            "minimum_iterations": int(deployment["minimum_iterations"]),
            "seed": int(args.seed),
            "r5": "translation_cm_strictly_below_5_and_rotation_deg_strictly_below_5",
        },
        "inputs": {
            "split_manifest": dict(source_by_path[str(manifest_path)]),
            "stable_map": dict(source_by_path[str(stable_map)]),
            "identity_metric": dict(source_by_path[str(identity_metric)]),
            "frontend_weights": dict(source_by_path[str(weights_path)]),
            "mainline_config": {
                **source_by_path[str(config_path)],
                "resolved_sha256": config.resolved_sha256,
            },
            "scene_calibration": (
                dict(source_by_path[str(calibration_path)])
                if calibration_path is not None
                else None
            ),
            "dataset_registry": dict(source_by_path[str(dataset_registry_path)]),
            "valid_mask_source": (
                dict(source_by_path[str(mask_path)])
                if mask_path is not None
                else None
            ),
            "all_source_files": sources,
        },
        "records": records,
    }
    atomic_torch_save_fresh(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(ALLOWED_ROLES), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--identity-metric", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="configs/paper_mainline.yaml")
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
