"""Canonical end-to-end LaFGS paper pipeline orchestration.

Every subprocess is a root-package module with its own artifact contract.  No
method logic lives in shell runners or environment-variable overrides.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from common.calibration import (
    calibrate_scene,
    validate_frozen_numeric_scene_calibration,
)
from common.config import (
    load_mainline_config,
    mapping_keypoint_policy_source,
    resolve_mapping_keypoint_count,
    resolve_mapping_nms_radius,
)
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from map_learning.trainer import full_refresh_interval, train


def _read_exact_scene_calibration(
    *,
    query_cache: str | Path,
    track_payload: str | Path,
    policy: dict,
    cached_path: str | Path | None,
) -> dict | None:
    query_cache = Path(query_cache).expanduser().resolve()
    track_payload = Path(track_payload).expanduser().resolve()
    if cached_path is not None:
        cached_path = Path(cached_path).expanduser().resolve()
        if cached_path.is_file():
            cached = json.loads(cached_path.read_text())
            sources = cached.get("sources", {})
            if (
                cached.get("schema") == "lafgs_mapping_only_scene_calibration"
                and int(cached.get("version", 0)) >= 2
                and cached.get("policy") == dict(policy)
                and sources.get("query_cache") == str(query_cache)
                and sources.get("track_payload") == str(track_payload)
                and sources.get("uses_test_queries") is False
            ):
                if cached.get("lineage") or any(
                    name in sources
                    for name in ("query_cache_sha256", "track_payload_sha256")
                ):
                    validate_frozen_numeric_scene_calibration(
                        cached,
                        calibration_path=cached_path,
                        query_cache_path=query_cache,
                        track_payload_path=track_payload,
                        policy=policy,
                    )
                return cached
    return None


def _load_or_compute_scene_calibration(
    *,
    query_cache: str | Path,
    track_payload: str | Path,
    policy: dict,
    cached_path: str | Path | None = None,
) -> dict:
    """Reuse an exact mapping-only calibration instead of reloading a large cache."""
    cached = _read_exact_scene_calibration(
        query_cache=query_cache,
        track_payload=track_payload,
        policy=policy,
        cached_path=cached_path,
    )
    if cached is not None:
        return cached
    return calibrate_scene(query_cache, track_payload, policy=policy)


def _run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def _run_parallel(module: str, argument_sets: list[list[object]]) -> None:
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                module,
                *(str(value) for value in arguments),
            ],
            env=(
                {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": visible_devices[
                        index % len(visible_devices)
                    ],
                }
                if visible_devices
                else None
            ),
        )
        for index, arguments in enumerate(argument_sets)
    ]
    try:
        pending = set(processes)
        while pending:
            for process in tuple(pending):
                status = process.poll()
                if status is None:
                    continue
                pending.remove(process)
                if status:
                    raise subprocess.CalledProcessError(status, process.args)
            if pending:
                time.sleep(0.1)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise


def _run_query_shards(
    *,
    module: str,
    merge_module: str,
    arguments: list[object],
    output: Path,
    shard_count: int,
) -> None:
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("query shard count must be positive")
    if shard_count == 1:
        _run(module, *arguments, "--output", output)
        return
    shard_paths = [
        output.parent
        / (f"{output.stem}.shard_{index:03d}_of_{shard_count:03d}{output.suffix}")
        for index in range(shard_count)
    ]
    _run_parallel(
        module,
        [
            [
                *arguments,
                "--output",
                shard_paths[index],
                "--num-shards",
                shard_count,
                "--shard-index",
                index,
            ]
            for index in range(shard_count)
        ],
    )
    _run(
        merge_module,
        "--inputs",
        *shard_paths,
        "--output",
        output,
    )
    for shard_path in shard_paths:
        shard_path.unlink()


def _assert_adaptive_threshold_contract(
    *,
    graph: Path | None,
    provenance: Path,
    teacher: Path,
    parameters: dict,
) -> None:
    """Reject stale artifacts that mix incompatible scene thresholds."""
    import torch

    provenance_payload = torch.load(provenance, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher, map_location="cpu", weights_only=False)
    expected = {
        "depth_abs_tolerance_m": float(parameters["evidence_depth_abs_tolerance_m"])
    }
    actual = {
        "provenance.depth_abs_tolerance_m": float(
            provenance_payload["config"]["depth_abs_tolerance_m"]
        ),
        "teacher.depth_abs_tolerance_m": float(
            teacher_payload["config"]["depth_abs_tolerance_m"]
        ),
    }
    expected.update(
        {
            "strong_radius_px": float(parameters["positive_radius_px"]),
            "ambiguous_radius_px": float(parameters["negative_radius_px"]),
        }
    )
    actual.update(
        {
            "teacher.strong_radius_px": float(
                teacher_payload["config"]["strong_radius_px"]
            ),
            "teacher.ambiguous_radius_px": float(
                teacher_payload["config"]["ambiguous_radius_px"]
            ),
        }
    )
    if graph is not None:
        graph_payload = torch.load(graph, map_location="cpu", weights_only=False)
        thresholds = graph_payload.get("resolved_thresholds")
        if thresholds is None:
            raise ValueError("adaptive function graph lacks resolved thresholds")
        for name in (
            "strong_radius_px",
            "clean_radius_px",
            "ambiguous_radius_px",
            "pnp_reprojection_error_px",
            "harm_radius_px",
            "depth_abs_tolerance_m",
        ):
            actual[f"graph.{name}"] = float(thresholds[name])
        expected.update(
            {
                "clean_radius_px": float(parameters["clean_radius_px"]),
                "pnp_reprojection_error_px": float(
                    parameters["ransac_reprojection_px"]
                ),
                "harm_radius_px": float(parameters["harm_radius_px"]),
            }
        )
    expected_by_suffix = {key: value for key, value in expected.items()}
    for qualified, value in actual.items():
        suffix = qualified.split(".", 1)[1]
        target = expected_by_suffix[suffix]
        if not math.isclose(value, target, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"adaptive threshold mismatch for {qualified}: {value} != {target}"
            )


def _assert_compact_evidence_path_contract(
    *,
    compact_map: Path,
    graph_v2: Path,
    provenance: Path,
    graph: Path,
    teacher: Path,
) -> None:
    """Reject resumed compact evidence built for a different anchor universe."""
    import torch

    expected_map = compact_map.expanduser().resolve()
    expected_graph_v2 = graph_v2.expanduser().resolve()
    expected_provenance = provenance.expanduser().resolve()
    checks = [
        (graph_v2, "anchor_map", expected_map, True),
        (provenance, "anchor_map", expected_map, True),
        (provenance, "config.function_graph", expected_graph_v2, False),
        (graph, "anchor_map", expected_map, True),
        (graph, "raster_provenance", expected_provenance, False),
        (teacher, "anchor_map", expected_map, True),
        (teacher, "raster_provenance", expected_provenance, False),
    ]
    mismatches = []
    for artifact, key, expected, allow_content_equivalence in checks:
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        value = payload
        for component in key.split("."):
            value = value.get(component, "") if isinstance(value, dict) else ""
        actual = Path(value).expanduser().resolve() if value else None
        equivalent = actual == expected
        if (
            not equivalent
            and allow_content_equivalence
            and actual is not None
            and actual.is_file()
            and expected.is_file()
        ):
            equivalent = sha256_file(actual) == sha256_file(expected)
        if not equivalent:
            mismatches.append(f"{artifact.name}:{key}={actual!s} expected {expected}")
        del payload
    if mismatches:
        raise RuntimeError(
            "Compact evidence artifacts do not share one anchor universe: "
            + "; ".join(mismatches)
        )


def _assert_compact_training_threshold_contract(
    report: Path,
    parameters: dict,
) -> None:
    """Prevent a resumed adaptive run from accepting a stale metric refresh."""
    if not report.is_file():
        raise ValueError("adaptive compact training report is missing")
    payload = json.loads(report.read_text())
    config = payload.get("config", {})
    expected = {
        "ransac_reprojection_px": float(parameters["ransac_reprojection_px"]),
        "clean_reprojection_px": float(parameters["clean_radius_px"]),
    }
    for name, target in expected.items():
        actual = float(config.get(name, float("nan")))
        if not math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"adaptive compact training threshold mismatch for {name}: "
                f"expected {target}, found {actual}"
            )


def resolve_prior_ply(prior: str | Path) -> Path:
    """Resolve the normalized Gaussian PLY without depending on a producer repo."""
    prior = Path(prior).expanduser().resolve()
    manifest_path = prior / "prior_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        relative = manifest.get("gaussians")
        if relative:
            path = (prior / relative).resolve()
            if path.is_file():
                return path
    legacy_manifest = prior / "rgb_prior_manifest.json"
    if legacy_manifest.is_file():
        path = Path(json.loads(legacy_manifest.read_text())["exported_ply"])
        if path.is_file():
            return path.resolve()
    candidates = sorted(prior.glob("point_cloud/iteration_*/point_cloud.ply"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise FileNotFoundError(
        f"Could not resolve one normalized Gaussian PLY below {prior}"
    )


def _common_bootstrap_arguments(
    *,
    dataset: Path,
    prior: Path,
    output: Path,
    gaussian_type: str,
    sh_degree: int,
    query_cache: Path,
    visibility_cache: Path,
    native_keypoint_count: int = 2048,
    native_nms_radius: int = 4,
    native_association_radius_px: float = 2.0,
) -> list[object]:
    return [
        "--model_path",
        prior,
        "--source_path",
        dataset,
        "--images",
        "processed",
        "--data_device",
        "cpu",
        "--gaussian_type",
        gaussian_type,
        "--sh_degree",
        sh_degree,
        "--feature_type",
        "sp",
        "--resolution",
        1,
        "--longest_edge",
        0,
        "--norm_before_render",
        "--load_iteration",
        30000,
        "--require_rgb_prior_manifest",
        "--rgb_prior_manifest_path",
        prior / "rgb_prior_manifest.json",
        "--query_feature_contract",
        "native_resized_input",
        "--query_cache_path",
        query_cache,
        "--visibility_cache_path",
        visibility_cache,
        "--visibility_mode",
        "rasterizer",
        "--objective",
        "hard",
        "--observation_source",
        "native",
        "--native_keypoint_count",
        native_keypoint_count,
        "--native_nms_radius",
        native_nms_radius,
        "--max_observations",
        native_keypoint_count,
        "--validation_observations",
        native_keypoint_count,
        "--native_sampling_mode",
        "detector_grid",
        "--native_association_radius_px",
        native_association_radius_px,
        "--validation_ratio",
        0,
        "--split_mode",
        "stratified_temporal_block",
        "--split_seed",
        2026,
        "--train_seed",
        2026,
        "--max_train_views",
        0,
    ]


def build_bootstrap_and_tracks(
    *,
    dataset: str | Path,
    prior: str | Path,
    output: str | Path,
    gaussian_type: str,
    sh_degree: int,
    config: str | Path,
) -> dict[str, Path]:
    dataset, prior, output = map(
        lambda value: Path(value).expanduser().resolve(), (dataset, prior, output)
    )
    cfg = load_mainline_config(config).values
    initialization = cfg["initialization"]
    reconstruction = cfg["reconstruction"]
    stage = reconstruction["stage_a"]
    run = output / "bootstrap"
    init_dir, stage_dir, track_dir = (
        run / "initialization",
        run / "stage_a",
        run / "tracks",
    )
    for directory in (init_dir, stage_dir, track_dir):
        directory.mkdir(parents=True, exist_ok=True)
    query_cache = run / "query_cache.pt"
    visibility = run / "visibility.pt"
    adaptive = cfg.get("adaptive")
    sparse = dataset / "sparse/0"
    has_camera_model = any(
        (sparse / name).is_file() for name in ("images.bin", "images.txt")
    )
    if adaptive and has_camera_model:
        mapping_cameras = ColmapDataset(dataset, images="processed").split("mapping")
        if not mapping_cameras:
            raise ValueError("adaptive calibration requires mapping cameras")
        focals = sorted(
            math.sqrt(
                (camera.width / (2.0 * math.tan(camera.fov_x / 2.0)))
                * (camera.height / (2.0 * math.tan(camera.fov_y / 2.0)))
            )
            for camera in mapping_cameras
        )
        prebootstrap_angular_scale = focals[len(focals) // 2] / float(
            adaptive.get("reference_focal_px", 1672.028076171875)
        )
        pose_bins = round(
            (len(mapping_cameras) / float(adaptive["queries_per_pose_bin_squared"]))
            ** 0.5
        )
        pose_bins = max(int(adaptive["pose_bins_minimum"]), pose_bins)
        pose_bins = min(int(adaptive["pose_bins_maximum"]), pose_bins)
    else:
        prebootstrap_angular_scale = 1.0
        pose_bins = int(initialization["kcs_view_bins"])
    native_keypoint_count = (
        resolve_mapping_keypoint_count(cfg, mapping_cameras)
        if adaptive and has_camera_model
        else int(
            cfg.get("mapping", {}).get(
                "keypoints", initialization["kcs_keypoints"]
            )
        )
    )
    native_nms_radius = resolve_mapping_nms_radius(cfg)
    native_association_radius = max(0.5, 2.0 * prebootstrap_angular_scale)
    kcs_radius = max(
        0.5,
        float(initialization["kcs_radius_px"]) * prebootstrap_angular_scale,
    )
    common = _common_bootstrap_arguments(
        dataset=dataset,
        prior=prior,
        output=output,
        gaussian_type=gaussian_type,
        sh_degree=sh_degree,
        query_cache=query_cache,
        visibility_cache=visibility,
        native_keypoint_count=native_keypoint_count,
        native_nms_radius=native_nms_radius,
        native_association_radius_px=native_association_radius,
    )
    bootstrap_state = init_dir / "0_lafgs_map_state.pt"
    landmark_ids = init_dir / "sampled_idx.pkl"
    if not bootstrap_state.is_file():
        _run(
            "map_learning.bootstrap",
            *common,
            "--query_cache_policy",
            "reuse_or_build",
            "--output_dir",
            init_dir,
            "--scaffold_mode",
            "ulf_robust_consensus",
            "--generated_landmark_path",
            init_dir / "robust_ids.pkl",
            (
                "--no-regenerate_scaffold"
                if (init_dir / "robust_ids.pkl").is_file()
                else "--regenerate_scaffold"
            ),
            "--scaffold_budget",
            (
                adaptive["scaffold_safety_cap"]
                if adaptive
                else initialization["scaffold_budget"]
            ),
            "--scaffold_min_opacity",
            0,
            "--scaffold_opacity_keep_quantile",
            0.1,
            "--initialization_mode",
            "ulf_robust_geometry",
            "--ulf_consensus_keypoints",
            native_keypoint_count,
            "--ulf_consensus_radius_px",
            kcs_radius,
            "--ulf_consensus_min_visible_views",
            initialization["kcs_min_visible_views"],
            "--ulf_consensus_min_votes",
            initialization["kcs_min_votes"],
            "--ulf_consensus_min_rate",
            initialization["kcs_min_consensus_rate"],
            "--ulf_consensus_view_bins",
            pose_bins,
            "--ulf_consensus_min_distinct_view_bins",
            initialization["kcs_min_distinct_view_bins"],
            "--ulf_consensus_trajectory_bins",
            pose_bins,
            "--ulf_consensus_min_distinct_trajectory_bins",
            initialization["kcs_min_distinct_trajectory_bins"],
            "--ulf_consensus_independent_bin_scoring",
            (
                "--ulf_consensus_allow_underfill"
                if adaptive
                else "--ulf_consensus_allow_nonconsensus_fallback"
            ),
            "--ulf_consensus_extent_quantile",
            0.01,
            "--ulf_support_view_sampling",
            "uniform",
            "--ulf_support_mask_policy",
            initialization["support_mask_policy"],
            "--ulf_consensus_max_views",
            0,
            "--ulf_fusion_max_views",
            0,
            "--ulf_fusion_min_cosine",
            0,
            "--ulf_fusion_view_bins",
            pose_bins,
            "--ulf_fusion_descriptor_trim_fraction",
            initialization["gwff_trim_fraction"],
            "--ulf_fusion_descriptor_min_cosine",
            -1,
            "--ulf_fusion_trim_histogram_bins",
            64,
            "--no-ulf_fusion_exact_bin_balance",
            "--no-native_outcome_mode",
            "--retrieval_weight",
            0,
            "--trust_weight",
            0,
            "--steps",
            0,
            "--save_steps",
            0,
        )
    preliminary_calibration = (
        calibrate_scene(query_cache, policy=adaptive)
        if adaptive and query_cache.is_file()
        else None
    )
    parameters = (
        preliminary_calibration["parameters"]
        if preliminary_calibration is not None
        else None
    )
    if preliminary_calibration is not None:
        (run / "preliminary_scene_calibration.json").write_text(
            json.dumps(preliminary_calibration, indent=2, sort_keys=True) + "\n"
        )
    steps = (
        int(parameters["stage_a_steps"])
        if parameters is not None
        else (
            1000
            if adaptive and reconstruction["stage_a_steps"] == "adaptive"
            else int(reconstruction["stage_a_steps"])
        )
    )
    positive_radius = (
        parameters["positive_radius_px"]
        if parameters is not None
        else stage["positive_radius_px"]
    )
    negative_radius = (
        parameters["negative_radius_px"]
        if parameters is not None
        else stage["negative_radius_px"]
    )
    stage_midpoint = max(1, round(0.5 * steps))
    stage_state = stage_dir / f"{steps}_lafgs_map_state.pt"
    if not stage_state.is_file():
        _run(
            "map_learning.bootstrap",
            *common,
            "--query_cache_policy",
            "readonly",
            "--output_dir",
            stage_dir,
            "--scaffold_mode",
            "file",
            "--landmark_path",
            landmark_ids,
            "--initial_state_path",
            bootstrap_state,
            "--initial_state_blend",
            1,
            "--initialization_mode",
            "ulf_robust_geometry",
            "--native_outcome_mode",
            "--native_keep_weight",
            stage["keep_weight"],
            "--native_keep_margin",
            stage["keep_margin"],
            "--native_swap_weight",
            stage["swap_weight"],
            "--native_swap_margin",
            stage["swap_margin"],
            "--native_miss_weight",
            stage["miss_weight"],
            "--native_miss_margin",
            stage["miss_margin"],
            "--native_global_attractor_weight",
            stage["global_attractor_weight"],
            "--native_global_attractor_min_incoming",
            4,
            "--native_global_attractor_support_power",
            0.5,
            "--native_global_attractor_max_score",
            4,
            "--native_semidense_weight",
            stage["local_peak_weight"],
            "--native_semidense_start_step",
            (
                stage_midpoint
                if parameters is not None
                else stage["local_peak_start_step"]
            ),
            "--native_semidense_interval",
            stage["local_peak_interval"],
            "--native_semidense_max_anchors",
            stage["local_peak_max_anchors"],
            "--native_semidense_neighbors",
            stage["local_peak_neighbors"],
            "--native_semidense_local_radius_px",
            (
                parameters["semidense_local_radius_px"]
                if parameters is not None
                else stage["local_peak_radius_px"]
            ),
            "--native_semidense_target_sigma_px",
            (
                parameters["semidense_sigma_px"]
                if parameters is not None
                else stage["local_peak_sigma_px"]
            ),
            "--native_semidense_temperature",
            stage["local_peak_temperature"],
            "--native_semidense_protected_v2",
            "--native_semidense_measurement_min_reprojection_px",
            positive_radius,
            "--native_semidense_measurement_max_reprojection_px",
            negative_radius,
            "--native_semidense_surface_point_plane_m",
            (parameters["surface_point_plane_m"] if parameters is not None else 0.03),
            "--native_semidense_surface_max_distance_m",
            (parameters["surface_max_distance_m"] if parameters is not None else 0.15),
            "--native_semidense_surface_normal_cosine",
            0.95,
            "--native_semidense_projected_neighbor_radius_px",
            (
                parameters["projected_neighbor_radius_px"]
                if parameters is not None
                else 64
            ),
            "--native_semidense_local_identity_weight",
            stage["local_identity_weight"],
            "--native_semidense_margin_preservation_weight",
            stage["margin_preservation_weight"],
            "--native_semidense_reference_refresh_steps",
            (stage_midpoint if parameters is not None else 500),
            "--native_semidense_alternate_global",
            "--native_semidense_max_gradient_ratio",
            0.25,
            "--native_protected_set_weight",
            stage["high_precision_weight"],
            "--native_protected_set_start_step",
            (
                stage_midpoint
                if parameters is not None
                else stage["high_precision_start_step"]
            ),
            "--native_protected_set_interval",
            stage["high_precision_interval"],
            "--native_protected_set_refresh_visits",
            1,
            "--native_protected_set_ransac_seed",
            0,
            "--native_protected_set_ransac_reprojection_px",
            (parameters["ransac_reprojection_px"] if parameters is not None else 12),
            "--native_protected_set_ransac_max_iterations",
            5000,
            "--native_protected_set_ransac_min_iterations",
            100,
            "--native_protected_set_max_pose_error_cm",
            100,
            "--native_protected_set_max_useful",
            (parameters["matching_rows_target"] if parameters is not None else 96),
            "--native_protected_set_max_harmful",
            (parameters["matching_rows_target"] if parameters is not None else 96),
            "--native_protected_set_grid_rows",
            4,
            "--native_protected_set_grid_cols",
            4,
            "--native_protected_set_depth_bins",
            4,
            "--native_protected_set_surface_voxel_m",
            (parameters["surface_group_voxel_m"] if parameters is not None else 0.25),
            "--native_protected_set_max_per_surface_group",
            2,
            "--retrieval_weight",
            1,
            "--trust_weight",
            stage["trust_weight"],
            "--feature_lr",
            stage["feature_learning_rate"],
            "--weight_decay",
            stage["weight_decay"],
            "--hypothesis_topk",
            stage["hypothesis_topk"],
            "--positive_radius_px",
            positive_radius,
            "--negative_radius_px",
            negative_radius,
            "--steps",
            steps,
            "--save_steps",
            steps,
            "--log_interval",
            100,
        )
    identity = "track_first_provenance" if gaussian_type == "2dgs" else "track_first"

    def build_track_payload(directory: Path, resolved_parameters) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload_path = directory / "track_micro_anchor_payload.pt"
        if payload_path.is_file():
            return payload_path
        track_args: list[object] = [
            *common,
            "--query_cache_policy",
            "readonly",
            "--output_dir",
            directory,
            "--scaffold_mode",
            "file",
            "--landmark_path",
            landmark_ids,
            "--initial_state_path",
            stage_state,
            "--initial_state_blend",
            1,
            "--initialization_mode",
            "ulf_robust_geometry",
            "--native_outcome_mode",
            "--retrieval_weight",
            0,
            "--trust_weight",
            0,
            "--positive_radius_px",
            (
                resolved_parameters["positive_radius_px"]
                if resolved_parameters is not None
                else positive_radius
            ),
            "--negative_radius_px",
            (
                resolved_parameters["negative_radius_px"]
                if resolved_parameters is not None
                else negative_radius
            ),
            "--save_independent_geometry_teacher",
            "--geometry_teacher_identity_mode",
            identity,
            "--geometry_teacher_min_views",
            3,
            "--geometry_teacher_view_bins",
            (
                resolved_parameters["view_bin_count"]
                if resolved_parameters is not None
                else 8
            ),
            "--geometry_teacher_min_view_bins",
            2,
            "--geometry_teacher_min_parallax_deg",
            1,
            "--geometry_teacher_parallax_quantile",
            0.75,
            "--geometry_teacher_max_reprojection_px",
            (
                resolved_parameters["positive_radius_px"]
                if resolved_parameters is not None
                else positive_radius
            ),
            "--geometry_teacher_max_covariance_trace_m2",
            (
                0.01 * resolved_parameters["covariance_scale"]
                if resolved_parameters is not None
                else 0.01
            ),
            "--geometry_teacher_max_rendered_depth_residual_m",
            (
                resolved_parameters["depth_residual_m"]
                if resolved_parameters is not None
                else 0.15
            ),
            "--geometry_teacher_min_rendered_depth_observations",
            2,
            "--geometry_teacher_track_pair_neighbors",
            6,
            "--geometry_teacher_track_min_similarity",
            0.65,
            "--geometry_teacher_track_min_margin",
            0.01,
            "--geometry_teacher_track_max_epipolar_error_px",
            (
                resolved_parameters["positive_radius_px"]
                if resolved_parameters is not None
                else positive_radius
            ),
            "--geometry_teacher_track_epipolar_candidate_topk",
            4,
            "--geometry_teacher_track_epipolar_recovered_min_similarity",
            -1,
            "--geometry_teacher_track_epipolar_recovered_min_margin",
            -1,
            "--geometry_teacher_track_allow_chain_tracks",
            "--geometry_teacher_track_assignment_max_distance_m",
            (
                resolved_parameters["assignment_distance_m"]
                if resolved_parameters is not None
                else 0.2
            ),
            "--geometry_teacher_track_assignment_min_margin_m",
            0,
            "--save_track_micro_anchor_payload",
            "--steps",
            0,
            "--save_steps",
            0,
        ]
        geometry_policy = cfg.get("geometry", {})
        if bool(geometry_policy.get("surface_supported_tracks", False)):
            depth_sigma = float(
                geometry_policy.get(
                    "surface_covariance_sigma_m",
                    resolved_parameters["depth_residual_m"]
                    if resolved_parameters is not None
                    else 0.02,
                )
            )
            correction_scale = float(
                geometry_policy.get("surface_max_correction_depth_sigmas", 4.0)
            )
            track_args += [
                "--geometry_teacher_surface_support",
                "--geometry_teacher_surface_huber_m",
                depth_sigma,
                "--geometry_teacher_surface_covariance_sigma_m",
                depth_sigma,
                "--geometry_teacher_surface_max_correction_m",
                depth_sigma * correction_scale,
                "--geometry_teacher_surface_max_weak_information_ratio",
                geometry_policy.get("surface_max_weak_information_ratio", 0.25),
                "--geometry_teacher_surface_min_depth_improvement_fraction",
                geometry_policy.get("surface_min_depth_improvement_fraction", 0.10),
                "--geometry_teacher_surface_max_reprojection_increase_px",
                geometry_policy.get("surface_max_reprojection_increase_px", 0.05),
            ]
        if gaussian_type == "2dgs":
            track_args += [
                "--geometry_teacher_provenance_topk",
                4,
                "--geometry_teacher_provenance_min_consensus_rate",
                0.35,
                "--geometry_teacher_provenance_min_views",
                2,
                "--geometry_teacher_provenance_group_max_landmarks",
                4,
                "--geometry_teacher_provenance_group_min_relative_mass",
                0.25,
                "--geometry_teacher_provenance_group_min_consensus_rate",
                0.10,
            ]
        _run("map_learning.bootstrap", *track_args)
        return payload_path

    track_payload = build_track_payload(track_dir, parameters)
    if adaptive and query_cache.is_file() and track_payload.is_file():
        cached_calibration_path = run / "scene_calibration.json"
        cached_track_payload = None
        if cached_calibration_path.is_file():
            cached_sources = json.loads(cached_calibration_path.read_text()).get(
                "sources", {}
            )
            cached_track_value = cached_sources.get("track_payload")
            if cached_track_value:
                candidate = Path(cached_track_value).expanduser().resolve()
                if candidate.is_file():
                    cached_track_payload = candidate
        if cached_track_payload is not None:
            full_calibration = _read_exact_scene_calibration(
                query_cache=query_cache,
                track_payload=cached_track_payload,
                policy=adaptive,
                cached_path=cached_calibration_path,
            )
        else:
            full_calibration = None
        if full_calibration is not None:
            track_payload = cached_track_payload
        else:
            full_calibration = calibrate_scene(
                query_cache, track_payload, policy=adaptive
            )
            preliminary_scale = float(parameters["metric_scale"])
            full_scale = float(full_calibration["parameters"]["metric_scale"])
            scale_ratio = full_scale / max(preliminary_scale, 1e-12)
            relative_drift = max(scale_ratio, 1.0 / max(scale_ratio, 1e-12)) - 1.0
            threshold = float(adaptive.get("calibration_rebuild_relative_drift", 0.25))
            rebuilt = relative_drift > threshold
            if rebuilt:
                track_payload = build_track_payload(
                    run / "tracks_refined", full_calibration["parameters"]
                )
                refined = calibrate_scene(query_cache, track_payload, policy=adaptive)
                refined["refinement"] = {
                    "preliminary_to_full_scale_ratio": scale_ratio,
                    "relative_drift": relative_drift,
                    "rebuild_threshold": threshold,
                    "track_evidence_rebuilt": True,
                    "first_pass_track_payload": str(
                        (track_dir / "track_micro_anchor_payload.pt").resolve()
                    ),
                }
                full_calibration = refined
            else:
                full_calibration["refinement"] = {
                    "preliminary_to_full_scale_ratio": scale_ratio,
                    "relative_drift": relative_drift,
                    "rebuild_threshold": threshold,
                    "track_evidence_rebuilt": False,
                }
            cached_calibration_path.write_text(
                json.dumps(full_calibration, indent=2, sort_keys=True) + "\n"
            )
    artifacts = {
        "base_state": stage_state,
        "track_payload": track_payload,
        "query_cache": query_cache,
        "visibility_cache": visibility,
        "landmark_ids": landmark_ids,
    }
    mapping_contract = run / "mapping_frontend_contract.json"
    mapping_contract.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_frontend_contract",
                "version": 1,
                "uses_test_queries": False,
                "sparse_frontend": "ulfloc_native_metric",
                "keypoint_count": int(native_keypoint_count),
                "keypoint_source": mapping_keypoint_policy_source(cfg),
                "nms_radius": int(native_nms_radius),
                "deployment_keypoint_policy_unchanged": dict(cfg["deployment"]),
                "query_cache": str(query_cache.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifacts["mapping_frontend_contract"] = mapping_contract
    if adaptive:
        artifacts["preliminary_scene_calibration"] = (
            run / "preliminary_scene_calibration.json"
        )
        artifacts["scene_calibration"] = run / "scene_calibration.json"
    return artifacts


def build_evidence(
    *,
    base_state: str | Path,
    track_payload: str | Path,
    query_cache: str | Path,
    prior_ply: str | Path,
    gaussian_type: str,
    sh_degree: int,
    visibility_cache: str | Path,
    output: str | Path,
    config: str | Path = "configs/paper_mainline.yaml",
    valid_masks: str | Path | None = None,
    function_graph_shards: int = 1,
    provenance_shards: int = 1,
    observation_shards: int = 1,
    scene_calibration: str | Path | None = None,
) -> dict[str, Path]:
    """Build the frozen canonical map and real-image localization evidence."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = load_mainline_config(config).values
    calibration = (
        _load_or_compute_scene_calibration(
            query_cache=query_cache,
            track_payload=track_payload,
            policy=resolved_config["adaptive"],
            cached_path=scene_calibration,
        )
        if int(resolved_config["version"]) >= 2
        else None
    )
    parameters = calibration["parameters"] if calibration is not None else None
    if calibration is not None:
        (output / "scene_calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n"
        )
    canonical = output / "canonical_map.pt"
    graph_v2 = output / "function_graph_v2.pt"
    provenance = output / "raster_provenance.pt"
    graph = output / "function_graph.pt"
    teacher = output / "complete_positive_teacher.pt"
    contract = output / "evidence_contract.json"
    if not canonical.is_file():
        _run(
            "topology.candidates",
            "--base_state",
            base_state,
            "--track_payload",
            track_payload,
            "--query_cache",
            query_cache,
            "--output",
            canonical,
            "--budget",
            0,
        )
    if not graph_v2.is_file():
        arguments: list[object] = [
            "--anchor-map",
            canonical,
            "--query-cache",
            query_cache,
            "--topk",
            64,
        ]
        if parameters is not None:
            arguments += [
                "--strong-radius-px",
                parameters["positive_radius_px"],
                "--clean-radius-px",
                parameters["clean_radius_px"],
                "--ambiguous-radius-px",
                parameters["negative_radius_px"],
                "--pnp-reprojection-error-px",
                parameters["ransac_reprojection_px"],
                "--harm-radius-px",
                parameters["harm_radius_px"],
                "--depth-abs-tolerance-m",
                parameters["evidence_depth_abs_tolerance_m"],
            ]
        if valid_masks:
            arguments += ["--deployment-mask-cache", valid_masks]
        if visibility_cache:
            arguments += ["--raster-visibility-cache", visibility_cache]
        _run_query_shards(
            module="evidence.function_graph",
            merge_module="evidence.merge_function_graph",
            arguments=arguments,
            output=graph_v2,
            shard_count=function_graph_shards,
        )
    if not provenance.is_file():
        arguments = [
            "--anchor-map",
            canonical,
            "--query-cache",
            query_cache,
            "--gaussian-ply",
            prior_ply,
            "--gaussian-type",
            gaussian_type,
            "--sh-degree",
            sh_degree,
            "--function-graph",
            graph_v2,
            "--track-payload",
            track_payload,
        ]
        if parameters is not None:
            arguments += [
                "--depth-abs-tolerance-m",
                parameters["evidence_depth_abs_tolerance_m"],
            ]
        if valid_masks:
            arguments += ["--deployment-mask-cache", valid_masks]
        _run_query_shards(
            module="priors.provenance",
            merge_module="priors.merge_provenance",
            arguments=arguments,
            output=provenance,
            shard_count=provenance_shards,
        )
    if not graph.is_file():
        _run(
            "evidence.evidence_graph",
            "--function-graph-v2",
            graph_v2,
            "--raster-provenance",
            provenance,
            "--output",
            graph,
        )
    if not teacher.is_file():
        _run_query_shards(
            module="map_learning.observations",
            merge_module="map_learning.merge_observations",
            arguments=[
                "--anchor-map",
                canonical,
                "--query-cache",
                query_cache,
                "--raster-provenance",
                provenance,
                "--track-payload",
                track_payload,
                *(
                    [
                        "--strong-radius-px",
                        parameters["positive_radius_px"],
                        "--ambiguous-radius-px",
                        parameters["negative_radius_px"],
                        "--depth-abs-tolerance-m",
                        parameters["evidence_depth_abs_tolerance_m"],
                    ]
                    if parameters is not None
                    else []
                ),
            ],
            output=teacher,
            shard_count=observation_shards,
        )
    if parameters is not None:
        _assert_adaptive_threshold_contract(
            graph=graph,
            provenance=provenance,
            teacher=teacher,
            parameters=parameters,
        )
    if not contract.is_file():
        _run(
            "common.evidence_contract",
            "--query-cache",
            query_cache,
            "--track-payload",
            track_payload,
            "--primitive-prior",
            prior_ply,
            "--anchor-map",
            canonical,
            "--function-graph",
            graph,
            "--raster-provenance",
            provenance,
            "--positive-teacher",
            teacher,
            "--output",
            contract,
        )
    result = {
        "canonical_map": canonical,
        "function_graph": graph,
        "positive_teacher": teacher,
        "raster_provenance": provenance,
        "evidence_contract": contract,
    }
    if calibration is not None:
        result["scene_calibration"] = output / "scene_calibration.json"
    return result


def distill_compact_map(
    *,
    canonical_map: str | Path,
    function_graph: str | Path,
    positive_teacher: str | Path,
    track_payload: str | Path,
    query_cache: str | Path,
    output: str | Path,
    config: str | Path,
    pose_scoring_shards: int = 1,
    scene_calibration: str | Path | None = None,
) -> Path:
    """Materialize the Track core plus Gaussian-supported coverage reserve."""
    resolved_config = load_mainline_config(config).values
    cfg = resolved_config["reconstruction"]
    output = Path(output).expanduser().resolve()
    if int(resolved_config["version"]) >= 2:
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "adaptive_distillation_build.json"
        if not report_path.is_file():
            arguments: list[object] = [
                "topology.adaptive_distillation",
                "--canonical-map",
                canonical_map,
                "--function-graph",
                function_graph,
                "--complete-positive-teacher",
                positive_teacher,
                "--track-payload",
                track_payload,
                "--query-cache",
                query_cache,
                "--output-dir",
                output,
                "--config",
                config,
            ]
            if scene_calibration is not None:
                calibration_path = Path(scene_calibration).expanduser().resolve()
                arguments += [
                    "--frozen-scene-calibration",
                    calibration_path,
                    "--expected-frozen-scene-calibration-sha256",
                    sha256_file(calibration_path),
                ]
            _run(*arguments)
        report = json.loads(report_path.read_text())
        final = Path(report["map"])
        if not final.is_file():
            raise RuntimeError(f"Adaptive compact map was not produced: {final}")
        return final
    core_dir, reserve_dir = output / "track_core", output / "coverage_reserve"
    core_dir.mkdir(parents=True, exist_ok=True)
    reserve_dir.mkdir(parents=True, exist_ok=True)
    report_path = core_dir / "minimum_sufficient_build.json"
    if not report_path.is_file():
        _run(
            "topology.distillation",
            "--canonical-map",
            canonical_map,
            "--function-graph",
            function_graph,
            "--complete-positive-teacher",
            positive_teacher,
            "--track-payload",
            track_payload,
            "--query-cache",
            query_cache,
            "--output-dir",
            core_dir,
            "--track-cores",
            f"{cfg['track_core']}:medium",
            "--minimum-rows-per-query",
            cfg["minimum_rows_per_mapping_query"],
            "--maximum-reserve",
            cfg["maximum_reserve"],
        )
    report = json.loads(report_path.read_text())
    prefix = f"core{int(cfg['track_core']):05d}_medium"
    matches = [
        Path(value["path"])
        for key, value in report["maps"].items()
        if key.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix} map, found {matches}")
    final = reserve_dir / (
        f"pose_sufficient_add{int(cfg['pose_reserve_additions']):04d}.pt"
    )
    if not final.is_file():
        arguments: list[object] = [
            "--core-map",
            matches[0],
            "--canonical-map",
            canonical_map,
            "--function-graph",
            function_graph,
            "--complete-positive-teacher",
            positive_teacher,
            "--track-payload",
            track_payload,
            "--query-cache",
            query_cache,
            "--reserve-additions",
            cfg["pose_reserve_additions"],
        ]
        shard_count = int(pose_scoring_shards)
        if shard_count < 1:
            raise ValueError("pose_scoring_shards must be positive")
        if shard_count == 1:
            _run(
                "topology.coverage_reserve",
                *arguments,
                "--output-dir",
                reserve_dir,
            )
        else:
            shard_root = reserve_dir / "scoring_shards"
            shard_dirs = [
                shard_root / f"shard_{index:03d}_of_{shard_count:03d}"
                for index in range(shard_count)
            ]
            for shard_dir in shard_dirs:
                shard_dir.mkdir(parents=True, exist_ok=True)
            _run_parallel(
                "topology.coverage_reserve",
                [
                    [
                        *arguments,
                        "--output-dir",
                        shard_dirs[index],
                        "--score-only",
                        "--scoring-num-shards",
                        shard_count,
                        "--scoring-shard-index",
                        index,
                    ]
                    for index in range(shard_count)
                ],
            )
            shard_paths = [
                shard_dir / "pose_reserve_scoring.pt" for shard_dir in shard_dirs
            ]
            merged_scoring = reserve_dir / "pose_reserve_scoring_merged.pt"
            _run(
                "topology.merge_pose_scoring",
                "--inputs",
                *shard_paths,
                "--output",
                merged_scoring,
            )
            _run(
                "topology.coverage_reserve",
                *arguments,
                "--output-dir",
                reserve_dir,
                "--pose-scoring-cache",
                merged_scoring,
            )
            for shard_path, shard_dir in zip(shard_paths, shard_dirs):
                shard_path.unlink()
                shard_dir.rmdir()
            shard_root.rmdir()
    if not final.is_file():
        raise RuntimeError(f"Compact map was not produced: {final}")
    return final


def train_compact_map(
    *,
    compact_map: str | Path,
    function_graph: str | Path,
    track_payload: str | Path,
    query_cache: str | Path,
    prior_ply: str | Path,
    gaussian_type: str,
    sh_degree: int,
    output: str | Path,
    config: str | Path,
    valid_masks: str | Path | None = None,
    rebuild_function_graph: bool = False,
    function_graph_shards: int = 1,
    provenance_shards: int = 1,
    observation_shards: int = 1,
    scene_calibration: str | Path | None = None,
    refresh_all_ransac_shards: bool = False,
    refresh_shards: int = 7,
    deployment_row_limit: int = 0,
    soft_pose_weight: float = 0.0,
) -> dict[str, Path]:
    """Rebuild compact-map evidence, then run frozen A1 reconstruction.

    ``function_graph`` remains the compatibility fallback.  The formal V4 path
    sets ``rebuild_function_graph`` so graph rows, raster provenance, positive
    observations, and the trained map all refer to the same compact anchor
    universe.
    """
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = load_mainline_config(config).values
    calibration = (
        _load_or_compute_scene_calibration(
            query_cache=query_cache,
            track_payload=track_payload,
            policy=resolved_config["adaptive"],
            cached_path=scene_calibration,
        )
        if int(resolved_config["version"]) >= 2
        else None
    )
    parameters = calibration["parameters"] if calibration is not None else None
    if calibration is not None:
        (output / "scene_calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n"
        )
    provenance = output / "raster_provenance.pt"
    teacher = output / "complete_positive_teacher.pt"
    compact_graph_v2 = output / "compact_function_graph_v2.pt"
    compact_graph = output / "compact_function_graph.pt"
    effective_function_graph = Path(function_graph).expanduser().resolve()
    if rebuild_function_graph and not compact_graph_v2.is_file():
        arguments: list[object] = [
            "--anchor-map",
            compact_map,
            "--query-cache",
            query_cache,
            "--topk",
            64,
        ]
        if parameters is not None:
            arguments += [
                "--strong-radius-px",
                parameters["positive_radius_px"],
                "--clean-radius-px",
                parameters["clean_radius_px"],
                "--ambiguous-radius-px",
                parameters["negative_radius_px"],
                "--pnp-reprojection-error-px",
                parameters["ransac_reprojection_px"],
                "--harm-radius-px",
                parameters["harm_radius_px"],
                "--depth-abs-tolerance-m",
                parameters["evidence_depth_abs_tolerance_m"],
            ]
        if valid_masks:
            arguments += ["--deployment-mask-cache", valid_masks]
        _run_query_shards(
            module="evidence.function_graph",
            merge_module="evidence.merge_function_graph",
            arguments=arguments,
            output=compact_graph_v2,
            shard_count=function_graph_shards,
        )
    if not provenance.is_file():
        arguments: list[object] = [
            "--anchor-map",
            compact_map,
            "--query-cache",
            query_cache,
            "--gaussian-ply",
            prior_ply,
            "--gaussian-type",
            gaussian_type,
            "--sh-degree",
            sh_degree,
            "--track-payload",
            track_payload,
        ]
        if rebuild_function_graph:
            arguments += ["--function-graph", compact_graph_v2]
        if parameters is not None:
            arguments += [
                "--depth-abs-tolerance-m",
                parameters["evidence_depth_abs_tolerance_m"],
            ]
        if valid_masks:
            arguments += ["--deployment-mask-cache", valid_masks]
        _run_query_shards(
            module="priors.provenance",
            merge_module="priors.merge_provenance",
            arguments=arguments,
            output=provenance,
            shard_count=provenance_shards,
        )
    if rebuild_function_graph:
        if not compact_graph.is_file():
            _run(
                "evidence.evidence_graph",
                "--function-graph-v2",
                compact_graph_v2,
                "--raster-provenance",
                provenance,
                "--output",
                compact_graph,
            )
        effective_function_graph = compact_graph
    if not teacher.is_file():
        _run_query_shards(
            module="map_learning.observations",
            merge_module="map_learning.merge_observations",
            arguments=[
                "--anchor-map",
                compact_map,
                "--query-cache",
                query_cache,
                "--raster-provenance",
                provenance,
                "--track-payload",
                track_payload,
                *(
                    [
                        "--strong-radius-px",
                        parameters["positive_radius_px"],
                        "--ambiguous-radius-px",
                        parameters["negative_radius_px"],
                        "--depth-abs-tolerance-m",
                        parameters["evidence_depth_abs_tolerance_m"],
                    ]
                    if parameters is not None
                    else []
                ),
            ],
            output=teacher,
            shard_count=observation_shards,
        )
    if rebuild_function_graph:
        _assert_compact_evidence_path_contract(
            compact_map=Path(compact_map),
            graph_v2=compact_graph_v2,
            provenance=provenance,
            graph=compact_graph,
            teacher=teacher,
        )
    if parameters is not None:
        _assert_adaptive_threshold_contract(
            graph=effective_function_graph,
            provenance=provenance,
            teacher=teacher,
            parameters=parameters,
        )
    reconstruction = resolved_config["reconstruction"]
    steps = (
        int(parameters["metric_steps"])
        if parameters is not None
        else int(reconstruction["metric_steps"])
    )
    trained_map = output / f"anchor_map_step_{steps:04d}.pt"
    metric_state = output / f"metric_state_step_{steps:04d}.pt"
    if not trained_map.is_file() or not metric_state.is_file():
        train(
            map_path=compact_map,
            function_graph_path=effective_function_graph,
            track_payload_path=track_payload,
            query_cache_path=query_cache,
            positive_teacher_path=teacher,
            output_dir=output,
            steps=steps,
            checkpoint_steps=(steps,),
            rank=int(reconstruction["metric_rank"]),
            metric_residual=float(reconstruction["metric_residual"]),
            learning_rate=float(reconstruction["learning_rate"]),
            temperature=float(reconstruction["temperature"]),
            harmful_weight=float(reconstruction["harmful_weight"]),
            trust_weight=float(reconstruction["trust_weight"]),
            group_dro_eta=float(reconstruction["group_dro_eta"]),
            group_dro_max_weight_ratio=float(
                reconstruction["group_dro_max_weight_ratio"]
            ),
            refresh_interval=(
                full_refresh_interval(steps, refresh_shards)
                if refresh_all_ransac_shards
                else 0
            ),
            refresh_shards=refresh_shards,
            deployment_row_limit=deployment_row_limit,
            soft_pose_weight=soft_pose_weight,
            task_translation_m=(
                float(parameters["task_translation_m"])
                if parameters is not None
                else 0.05
            ),
            task_rotation_deg=(
                float(parameters["task_rotation_deg"])
                if parameters is not None
                else 5.0
            ),
            ransac_reprojection_px=(
                float(parameters["ransac_reprojection_px"])
                if parameters is not None
                else float(resolved_config["deployment"]["reprojection_error_px"])
            ),
            clean_reprojection_px=(
                float(parameters["clean_radius_px"]) if parameters is not None else 4.0
            ),
            seed=2026,
        )
    if parameters is not None:
        _assert_compact_training_threshold_contract(
            output / "training_report.json", parameters
        )
    results = {
        "compact_provenance": provenance,
        "compact_positive_teacher": teacher,
        "trained_map": trained_map,
        "metric_state": metric_state,
    }
    if rebuild_function_graph:
        results.update(
            {
                "compact_function_graph_v2": compact_graph_v2,
                "compact_function_graph": compact_graph,
            }
        )
    return results


def write_pipeline_manifest(output: Path, values: dict) -> None:
    serializable = {key: str(value) for key, value in values.items()}
    (output / "pipeline_manifest.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )
