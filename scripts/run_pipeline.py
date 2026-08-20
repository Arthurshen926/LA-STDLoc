#!/usr/bin/env python3
"""Run the frozen LaFGS reconstruction pipeline from an imported RGB prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Sequence

import torch

from common.anchor_registry_artifact import materialize_anchor_registry
from common.content_addressed_dag import (
    ContentAddressedStore,
    node_spec,
    path_content_record,
    runtime_identity,
    source_identity,
)
from common.hashing import sha256_file
from common.pipeline_completion import write_pipeline_completion
from data.datasets import ColmapDataset
from evaluation.evaluator import evaluate_dataset
from features.superpoint import resolve_superpoint_weights
from localization.localizer import SparseLocalizer
from common.config import (
    load_scene_calibration,
    load_mainline_config,
    materialize_keypoint_factor_config,
    materialize_mapping_keypoint_config,
    materialize_surface_track_config,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from map_learning.pipeline import (
    build_bootstrap_and_tracks,
    build_evidence,
    distill_compact_map,
    resolve_prior_ply,
    train_compact_map,
    write_pipeline_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    parser.add_argument("--valid-masks", default="")
    parser.add_argument("--function-graph-shards", type=int, default=1)
    parser.add_argument("--provenance-shards", type=int, default=1)
    parser.add_argument("--observation-shards", type=int, default=1)
    parser.add_argument("--pose-scoring-shards", type=int, default=1)
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly opt into reading and evaluating the test split.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--artifact-cache",
        type=Path,
        help=(
            "Optional bounded content-addressed cache for the mapping Observation, "
            "Track, and Geometry node. No cache is written unless this is explicit."
        ),
    )
    parser.add_argument("--artifact-cache-max-node-gib", type=float, default=8.0)
    parser.add_argument("--artifact-cache-max-total-gib", type=float, default=20.0)
    parser.add_argument(
        "--keypoints",
        type=int,
        choices=(1024, 2048),
        help=(
            "Lock detector density for mapping observations, reconstruction, "
            "and sparse deployment in a materialized factor config."
        ),
    )
    parser.add_argument(
        "--surface-supported-tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the cross-fitted weak-axis surface geometry factor.",
    )
    parser.add_argument(
        "--mapping-keypoints",
        type=int,
        choices=(1024, 2048),
        help=(
            "Lock mapping-observation density only; sparse deployment keeps "
            "the config's independent density policy."
        ),
    )
    return parser


def _validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, object]:
    experimental_factors = {
        "joint_keypoints": args.keypoints,
        "mapping_keypoints": args.mapping_keypoints,
        "surface_supported_tracks": bool(args.surface_supported_tracks),
    }
    if args.keypoints is not None and args.mapping_keypoints is not None:
        parser.error("--mapping-keypoints cannot be combined with joint --keypoints")
    if args.evaluate and any(bool(value) for value in experimental_factors.values()):
        parser.error(
            "experimental factor flags and --evaluate are mutually exclusive"
        )
    if (
        args.artifact_cache_max_node_gib <= 0
        or args.artifact_cache_max_total_gib <= 0
        or args.artifact_cache_max_node_gib > args.artifact_cache_max_total_gib
    ):
        parser.error("artifact cache limits must be positive and node <= total")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(
            "--output must be a fresh, nonexistent root; quarantine partial or stale runs"
        )
    return experimental_factors


_BOOTSTRAP_DAG_SOURCES = (
    "common/anchor_registry_artifact.py",
    "common/artifact_contract.py",
    "common/calibration.py",
    "common/cli.py",
    "common/config.py",
    "common/content_addressed_dag.py",
    "common/geometry.py",
    "common/hashing.py",
    "common/pipeline_completion.py",
    "common/producer_identity.py",
    "common/runtime.py",
    "common/tensor_identity.py",
    "data/cameras.py",
    "data/colmap.py",
    "data/datasets.py",
    "data/images.py",
    "data/scene.py",
    "data/splits.py",
    "evidence/camera_pair_policy.py",
    "evidence/parallax_stratified_pair_policy.py",
    "evidence/parallel_triangulation.py",
    "evidence/track_provenance_assignment.py",
    "evidence/tracks.py",
    "evidence/triangulation.py",
    "evidence/visibility.py",
    "features/extractor.py",
    "features/matching.py",
    "features/multiview_fusion.py",
    "features/sampling.py",
    "features/superpoint.py",
    "localization/pose_solver.py",
    "map_learning/alias_teacher.py",
    "map_learning/bootstrap.py",
    "map_learning/metric.py",
    "map_learning/pipeline.py",
    "map_learning/soft_pose.py",
    "map_learning/stage_a_loss.py",
    "map_learning/trainer.py",
    "map_learning/trust.py",
    "priors/geometry.py",
    "priors/models.py",
    "priors/rasterizer.py",
    "priors/rendering.py",
    "scripts/run_pipeline.py",
    "topology/anchor_covariance.py",
    "topology/anchor_registry.py",
    "topology/geometry_materializer.py",
    "topology/pose_information.py",
    "topology/sampling.py",
)

_BOOTSTRAP_DAG_FILES = {
    "base_state": "base_state.pt",
    "track_payload": "track_payload.pt",
    "query_cache": "query_cache.pt",
    "visibility_cache": "visibility_cache.pt",
    "landmark_ids": "landmark_ids.pkl",
    "mapping_frontend_contract": "mapping_frontend_contract.json",
    "scene_calibration": "scene_calibration.json",
}
_BOOTSTRAP_DAG_PATH_BINDINGS = "path_bindings.json"


def _mapping_mask_record(dataset_root: Path, names: list[str]) -> dict:
    mask_path = next(
        (
            path
            for path in (
                dataset_root / "processed/masks.pkl",
                dataset_root / "masks.pkl",
            )
            if path.is_file()
        ),
        None,
    )
    if mask_path is None:
        return {"kind": "mapping_mask_manifest", "present": False}
    with mask_path.open("rb") as handle:
        masks = pickle.load(handle)
    if not isinstance(masks, dict):
        raise ValueError("mapping mask artifact must contain a dictionary")
    digest = hashlib.sha256()
    selected = 0
    for name in names:
        if name not in masks:
            continue
        channels = masks[name]
        if not isinstance(channels, (list, tuple)) or len(channels) < 3:
            raise ValueError(
                f"Mask entry for {name!r} must contain three validity channels"
            )
        selected += 1
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        for channel in channels[:3]:
            value = torch.as_tensor(channel).detach().cpu().contiguous()
            for encoded in (
                str(value.dtype).encode("ascii"),
                json.dumps(list(value.shape)).encode("ascii"),
                value.numpy().tobytes(),
            ):
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return {
        "kind": "mapping_mask_manifest",
        "present": True,
        "mapping_mask_count": selected,
        "sha256": digest.hexdigest(),
    }


def _mapping_dataset_records(dataset_root: Path) -> tuple[dict, dict, dict]:
    """Hash mapping RGB and its camera schedule without opening test RGB."""
    # Build only the camera registry here.  The regular constructor also
    # deserializes optional all-scene masks, which are irrelevant to this key
    # and may contain test entries.
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.root = dataset_root.expanduser().resolve()
    dataset.images = "processed"
    dataset._test_names = dataset._load_test_names()
    dataset.cameras = dataset._load_cameras()
    cameras = dataset.split("mapping")
    if not cameras:
        raise ValueError("DAG cache key requires at least one mapping camera")
    schedule = []
    images = []
    for camera in cameras:
        image_path = camera.image_path.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        schedule.append(
            {
                "image_name": camera.image_name,
                "fov_x": float(camera.fov_x),
                "fov_y": float(camera.fov_y),
                "rotation_c2w": camera.rotation_c2w.tolist(),
                "translation_w2c": camera.translation_w2c.tolist(),
                "width": int(camera.width),
                "height": int(camera.height),
            }
        )
        images.append(
            {
                "image_name": camera.image_name,
                "sha256": sha256_file(image_path),
                "size_bytes": image_path.stat().st_size,
            }
        )
    schedule_json = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    images_json = json.dumps(images, sort_keys=True, separators=(",", ":"))
    return (
        {
            "kind": "mapping_camera_schedule",
            "camera_count": len(schedule),
            "sha256": hashlib.sha256(schedule_json.encode()).hexdigest(),
        },
        {
            "kind": "mapping_rgb_manifest",
            "image_count": len(images),
            "sha256": hashlib.sha256(images_json.encode()).hexdigest(),
            "size_bytes": sum(record["size_bytes"] for record in images),
        },
        _mapping_mask_record(dataset.root, [camera.image_name for camera in cameras]),
    )


def _bootstrap_dag_spec(args: argparse.Namespace, config: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    prior_ply = resolve_prior_ply(args.prior)
    weights = resolve_superpoint_weights()
    camera_schedule, mapping_rgb, mapping_masks = _mapping_dataset_records(
        Path(args.dataset)
    )
    upstream = {
        "mapping_rgb": mapping_rgb,
        "mapping_masks": mapping_masks,
        "camera_schedule": camera_schedule,
        "gaussian_checkpoint": path_content_record(prior_ply),
        "frontend_checkpoint": path_content_record(weights),
        "config_file": path_content_record(config),
    }
    for name in ("prior_manifest.json", "rgb_prior_manifest.json"):
        candidate = Path(args.prior) / name
        if candidate.is_file():
            upstream[f"gaussian_{name}"] = path_content_record(candidate)
    return node_spec(
        node="mapping_observation_track_geometry",
        config={
            "gaussian_type": args.gaussian_type,
            "sh_degree": int(args.sh_degree),
            "mainline": load_mainline_config(config).values,
        },
        upstream=upstream,
        producer={
            **source_identity(root, _BOOTSTRAP_DAG_SOURCES),
            "runtime": runtime_identity(),
        },
    )


def _rebound_json(
    *, source: Path, target: Path, replacements: dict[str, str]
) -> Path:
    payload = json.loads(source.read_text())
    sources = payload.get("sources")
    if isinstance(sources, dict):
        for name, value in replacements.items():
            if name in sources:
                sources[name] = value
    if "query_cache" in payload:
        payload["query_cache"] = replacements["query_cache"]
    if target.exists():
        target.chmod(0o644)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def _rebind_value_paths(value, replacements: dict[str, str]):
    if isinstance(value, Path):
        rebound, changed = _rebind_value_paths(str(value), replacements)
        return Path(rebound), changed
    if isinstance(value, str):
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            if value == old:
                return new, True
            if value.startswith(old + os.sep):
                return new + value[len(old) :], True
        return value, False
    if isinstance(value, dict):
        output = {}
        changed = False
        for key, item in value.items():
            new_key, key_changed = _rebind_value_paths(key, replacements)
            new_item, item_changed = _rebind_value_paths(item, replacements)
            if new_key in output and new_key != key:
                raise ValueError("path rebinding would collide dictionary keys")
            output[new_key] = new_item
            changed = changed or key_changed or item_changed
        return output, changed
    if isinstance(value, list):
        values = [_rebind_value_paths(item, replacements) for item in value]
        return [item for item, _ in values], any(changed for _, changed in values)
    if isinstance(value, tuple):
        values = [_rebind_value_paths(item, replacements) for item in value]
        return tuple(item for item, _ in values), any(changed for _, changed in values)
    if isinstance(value, (set, frozenset)):
        values = [_rebind_value_paths(item, replacements) for item in value]
        rebound = type(value)(item for item, _ in values)
        if len(rebound) != len(value):
            raise ValueError("path rebinding would collide set entries")
        return rebound, any(changed for _, changed in values)
    return value, False


def _rebind_torch_artifact_paths(path: Path, replacements: dict[str, str]) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rebound, changed = _rebind_value_paths(payload, replacements)
    if not changed:
        return False
    temporary = path.with_name(f".{path.name}.rebind.{os.getpid()}")
    try:
        torch.save(rebound, temporary)
        path.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _build_or_reuse_bootstrap(
    *, args: argparse.Namespace, config: Path
) -> tuple[dict[str, Path], dict | None]:
    if args.artifact_cache is None:
        return (
            build_bootstrap_and_tracks(
                dataset=args.dataset,
                prior=args.prior,
                output=args.output,
                gaussian_type=args.gaussian_type,
                sh_degree=args.sh_degree,
                config=config,
            ),
            None,
        )
    gib = 1 << 30
    store = ContentAddressedStore(
        args.artifact_cache,
        maximum_node_bytes=round(args.artifact_cache_max_node_gib * gib),
        maximum_store_bytes=round(args.artifact_cache_max_total_gib * gib),
    )
    key_started = time.perf_counter()
    spec = _bootstrap_dag_spec(args, config)
    key_seconds = time.perf_counter() - key_started
    lookup_started = time.perf_counter()
    cached = store.load(spec)
    lookup_seconds = time.perf_counter() - lookup_started
    cache_hit = cached is not None
    publish_seconds = 0.0
    if cached is None:
        built = build_bootstrap_and_tracks(
            dataset=args.dataset,
            prior=args.prior,
            output=args.output,
            gaussian_type=args.gaussian_type,
            sh_degree=args.sh_degree,
            config=config,
        )
        missing = sorted(set(_BOOTSTRAP_DAG_FILES) - set(built))
        if missing:
            raise RuntimeError(f"bootstrap DAG node is incomplete: {missing}")
        rebound_dir = args.output / "bootstrap/dag_publish"
        rebound_dir.mkdir(parents=True, exist_ok=True)
        publish_sources = {
            filename: Path(built[name])
            for name, filename in _BOOTSTRAP_DAG_FILES.items()
        }
        replacements = {
            "query_cache": "@dag/query_cache.pt",
            "track_payload": "@dag/track_payload.pt",
        }
        publish_sources[_BOOTSTRAP_DAG_FILES["scene_calibration"]] = _rebound_json(
            source=Path(built["scene_calibration"]),
            target=rebound_dir / "scene_calibration.json",
            replacements=replacements,
        )
        publish_sources[_BOOTSTRAP_DAG_FILES["mapping_frontend_contract"]] = (
            _rebound_json(
                source=Path(built["mapping_frontend_contract"]),
                target=rebound_dir / "mapping_frontend_contract.json",
                replacements=replacements,
            )
        )
        binding_source = rebound_dir / _BOOTSTRAP_DAG_PATH_BINDINGS
        binding_source.write_text(
            json.dumps(
                {
                    "schema": "lafgs_dag_origin_path_bindings",
                    "version": 1,
                    "output": str(Path(args.output).resolve()),
                    "dataset": str(Path(args.dataset).expanduser().resolve()),
                    "prior": str(Path(args.prior).expanduser().resolve()),
                    "config": str(Path(config).expanduser().resolve()),
                    "artifacts": {
                        name: str(Path(path).expanduser().resolve())
                        for name, path in built.items()
                        if isinstance(path, (str, Path))
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        publish_sources[_BOOTSTRAP_DAG_PATH_BINDINGS] = binding_source
        publish_started = time.perf_counter()
        final_spec = _bootstrap_dag_spec(args, config)
        if final_spec != spec:
            raise RuntimeError(
                "mapping inputs, config, runtime, or producer changed during DAG build"
            )
        cached = store.publish(spec, publish_sources)
        publish_seconds = time.perf_counter() - publish_started
    materialize_started = time.perf_counter()
    local, materialize_modes = store.materialize(
        spec, args.output / "bootstrap_dag_materialized"
    )
    artifacts = {name: local[filename] for name, filename in _BOOTSTRAP_DAG_FILES.items()}
    local_replacements = {
        "query_cache": str(artifacts["query_cache"]),
        "track_payload": str(artifacts["track_payload"]),
    }
    _rebound_json(
        source=artifacts["scene_calibration"],
        target=artifacts["scene_calibration"],
        replacements=local_replacements,
    )
    bindings_path = local[_BOOTSTRAP_DAG_PATH_BINDINGS]
    origin = json.loads(bindings_path.read_text())
    embedded_replacements = {
        str(origin["output"]): "@dag-origin-output",
        str(origin["dataset"]): str(Path(args.dataset).expanduser().resolve()),
        str(origin["prior"]): str(Path(args.prior).expanduser().resolve()),
        str(origin["config"]): str(Path(config).expanduser().resolve()),
    }
    for name, old_path in origin.get("artifacts", {}).items():
        if name in artifacts:
            embedded_replacements[str(old_path)] = str(artifacts[name])
    rebound_torch = {}
    for name in (
        "base_state",
        "track_payload",
        "query_cache",
        "visibility_cache",
    ):
        rebound_torch[name] = _rebind_torch_artifact_paths(
            artifacts[name], embedded_replacements
        )
    origin.update(
        {
            "schema": "lafgs_run_local_path_bindings",
            "origin_cache_paths_removed": True,
            "output": str(Path(args.output).resolve()),
            "dataset": str(Path(args.dataset).expanduser().resolve()),
            "prior": str(Path(args.prior).expanduser().resolve()),
            "config": str(Path(config).expanduser().resolve()),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }
    )
    bindings_path.chmod(0o644)
    bindings_path.write_text(json.dumps(origin, indent=2, sort_keys=True) + "\n")
    _rebound_json(
        source=artifacts["mapping_frontend_contract"],
        target=artifacts["mapping_frontend_contract"],
        replacements=local_replacements,
    )
    materialize_seconds = time.perf_counter() - materialize_started
    report = {
        "schema": "lafgs_pipeline_dag_reuse_report",
        "version": 1,
        "node": spec["node"],
        "key_sha256": spec["key_sha256"],
        "cache_hit": cache_hit,
        "cache_root": str(store.root),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "materialization_copy_mode": materialize_modes,
        "run_local_torch_path_rebound": rebound_torch,
        "timing_seconds": {
            "cache_key_validation": key_seconds,
            "cache_lookup_and_sha_verification": lookup_seconds,
            "cache_publish": publish_seconds,
            "hit_materialization": materialize_seconds,
        },
    }
    report_path = args.output / "bootstrap_dag_reuse.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifacts["bootstrap_dag_reuse"] = report_path
    return artifacts, report


def run(args: argparse.Namespace, *, experimental_factors: dict[str, object]) -> dict:
    # Atomically claim the run root.  This closes the preflight/write TOCTOU
    # window and makes direct API callers obey the same fresh-root contract.
    args.output.mkdir(parents=True, exist_ok=False)
    config = Path(args.config)
    if args.keypoints is not None:
        config = materialize_keypoint_factor_config(
            config,
            args.output / f"factor_config_k{args.keypoints}.yaml",
            args.keypoints,
        )
    if args.mapping_keypoints is not None:
        config = materialize_mapping_keypoint_config(
            config,
            args.output / f"factor_config_mapping_k{args.mapping_keypoints}.yaml",
            args.mapping_keypoints,
        )
    if args.surface_supported_tracks:
        config = materialize_surface_track_config(
            config,
            args.output / "factor_config_surface_tracks.yaml",
        )
    artifacts, _ = _build_or_reuse_bootstrap(args=args, config=Path(config))
    prior_ply = resolve_prior_ply(args.prior)
    evidence = build_evidence(
        base_state=artifacts["base_state"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        prior_ply=prior_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        visibility_cache=artifacts["visibility_cache"],
        output=args.output / "evidence",
        config=config,
        valid_masks=args.valid_masks or None,
        function_graph_shards=args.function_graph_shards,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
        scene_calibration=artifacts.get("scene_calibration"),
    )
    compact_map = distill_compact_map(
        canonical_map=evidence["canonical_map"],
        function_graph=evidence["function_graph"],
        positive_teacher=evidence["positive_teacher"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        output=args.output / "topology",
        config=config,
        pose_scoring_shards=args.pose_scoring_shards,
    )
    trained = train_compact_map(
        compact_map=compact_map,
        function_graph=evidence["function_graph"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        prior_ply=prior_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        output=args.output / "map_learning",
        config=config,
        valid_masks=args.valid_masks or None,
        rebuild_function_graph=True,
        function_graph_shards=args.function_graph_shards,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
        scene_calibration=artifacts.get("scene_calibration"),
    )
    outputs = {
        **artifacts,
        **evidence,
        **trained,
        "prior_ply": prior_ply,
        "compact_map": compact_map,
    }
    if args.valid_masks:
        outputs["valid_masks"] = Path(args.valid_masks).expanduser().resolve()
    if args.evaluate:
        deployment = load_mainline_config(config).values["deployment"]
        dataset = ColmapDataset(args.dataset, images="processed")
        test_cameras = dataset.split("test")
        mapping_cameras = dataset.split("mapping")
        calibration_path = (
            Path(trained["trained_map"]).parent / "scene_calibration.json"
        )
        scene_calibration = load_scene_calibration(calibration_path)
        keypoint_count = resolve_keypoint_count(deployment, mapping_cameras)
        reprojection_error_px = resolve_reprojection_error_px(
            deployment, mapping_cameras, scene_calibration
        )
        localizer = SparseLocalizer(
            trained["trained_map"],
            trained["metric_state"],
            device=args.device,
            keypoint_count=keypoint_count,
            nms_radius=int(deployment["nms"]),
            reprojection_error_px=reprojection_error_px,
            confidence=deployment["confidence"],
            max_iterations=deployment["maximum_iterations"],
            min_iterations=deployment["minimum_iterations"],
            seed=2026,
        )
        result = evaluate_dataset(
            dataset=dataset,
            localizer=localizer,
            cameras=test_cameras,
            output=args.output / "evaluation",
        )
        (args.output / "evaluation" / "deployment_contract.json").write_text(
            json.dumps(
                {
                    "schema": "lafgs_sparse_deployment_contract",
                    "version": 1,
                    "keypoint_count": int(keypoint_count),
                    "nms_radius": int(deployment["nms"]),
                    "ransac_reprojection_px": float(reprojection_error_px),
                    "scene_calibration": str(calibration_path.resolve()),
                    "calibration_split": "mapping",
                    "evaluated_split": "test",
                    "pose_solves": 1,
                    "duplicate_anchor_suppression": False,
                    "guided_sampling": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(result["summary"], indent=2))
        outputs["evaluation"] = args.output / "evaluation"
    selection_provenance = args.output / "topology/adaptive_selection_provenance.pt"
    scene_calibration = trained.get("scene_calibration")
    if scene_calibration is None:
        raise RuntimeError(
            "new pipeline completion requires an explicit mapping-only calibration"
        )
    registry_parents = {
        "trained_map": Path(trained["trained_map"]),
        "compact_map": Path(compact_map),
        "positive_teacher": Path(trained["compact_positive_teacher"]),
        "track_payload": Path(artifacts["track_payload"]),
        "query_cache": Path(artifacts["query_cache"]),
        "raster_provenance": Path(trained["compact_provenance"]),
        "selection_provenance": selection_provenance,
        "scene_calibration": Path(scene_calibration),
        "metric_state": Path(trained["metric_state"]),
        "config": Path(config).expanduser().resolve(),
        "gaussian_ply": Path(prior_ply),
    }
    locked_parents = {
        name: (path, sha256_file(path)) for name, path in registry_parents.items()
    }
    registry_result = materialize_anchor_registry(
        parents=locked_parents,
        output=Path(trained["trained_map"]).parent / "neutral_anchor_registry.pt",
        contract_output=(
            Path(trained["trained_map"]).parent
            / "neutral_anchor_registry.contract.json"
        ),
        require_pipeline_parents=True,
    )
    outputs["anchor_registry"] = registry_result["registry"]
    outputs["anchor_registry_contract"] = registry_result["contract"]
    outputs["selection_provenance"] = selection_provenance
    outputs["config"] = Path(config).expanduser().resolve()
    manifest = write_pipeline_manifest(args.output, outputs)
    completion = write_pipeline_completion(
        output=args.output,
        artifacts=outputs,
        pipeline_manifest=manifest,
        anchor_registry_contract=registry_result["contract"],
        config=config,
        evaluation_requested=bool(args.evaluate),
        experimental_factors=experimental_factors,
    )
    print(
        json.dumps(
            {
                "pipeline_completion": completion["path"],
                "pipeline_completion_sha256": completion["sha256"],
                "uses_test_queries": completion["uses_test_queries"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return completion


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    experimental_factors = _validate_arguments(args, parser)
    run(args, experimental_factors=experimental_factors)


if __name__ == "__main__":
    main()
