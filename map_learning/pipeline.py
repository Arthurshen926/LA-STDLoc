"""Canonical end-to-end LaFGS paper pipeline orchestration.

Every subprocess is a root-package module with its own artifact contract.  No
method logic lives in shell runners or environment-variable overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from common.config import load_mainline_config
from map_learning.trainer import train


def _run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def _run_parallel(module: str, argument_sets: list[list[object]]) -> None:
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                module,
                *(str(value) for value in arguments),
            ]
        )
        for arguments in argument_sets
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
                    raise subprocess.CalledProcessError(
                        status, process.args
                    )
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
        / (
            f"{output.stem}.shard_{index:03d}_of_"
            f"{shard_count:03d}{output.suffix}"
        )
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
) -> list[object]:
    return [
        "--model_path", prior,
        "--source_path", dataset,
        "--images", "processed",
        "--data_device", "cpu",
        "--gaussian_type", gaussian_type,
        "--sh_degree", sh_degree,
        "--feature_type", "sp",
        "--resolution", 1,
        "--longest_edge", 0,
        "--norm_before_render",
        "--load_iteration", 30000,
        "--require_rgb_prior_manifest",
        "--rgb_prior_manifest_path", prior / "rgb_prior_manifest.json",
        "--query_feature_contract", "native_resized_input",
        "--query_cache_path", query_cache,
        "--visibility_cache_path", visibility_cache,
        "--visibility_mode", "rasterizer",
        "--objective", "hard",
        "--observation_source", "native",
        "--native_keypoint_count", 2048,
        "--max_observations", 2048,
        "--validation_observations", 2048,
        "--native_sampling_mode", "detector_grid",
        "--native_association_radius_px", 2,
        "--validation_ratio", 0,
        "--split_mode", "stratified_temporal_block",
        "--split_seed", 2026,
        "--train_seed", 2026,
        "--max_train_views", 0,
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
    dataset, prior, output = map(lambda value: Path(value).expanduser().resolve(), (dataset, prior, output))
    cfg = load_mainline_config(config).values
    initialization = cfg["initialization"]
    reconstruction = cfg["reconstruction"]
    stage = reconstruction["stage_a"]
    run = output / "bootstrap"
    init_dir, stage_dir, track_dir = run / "initialization", run / "stage_a", run / "tracks"
    for directory in (init_dir, stage_dir, track_dir):
        directory.mkdir(parents=True, exist_ok=True)
    query_cache = run / "query_cache.pt"
    visibility = run / "visibility.pt"
    common = _common_bootstrap_arguments(
        dataset=dataset,
        prior=prior,
        output=output,
        gaussian_type=gaussian_type,
        sh_degree=sh_degree,
        query_cache=query_cache,
        visibility_cache=visibility,
    )
    bootstrap_state = init_dir / "0_lafgs_map_state.pt"
    landmark_ids = init_dir / "sampled_idx.pkl"
    if not bootstrap_state.is_file():
        _run(
            "map_learning.bootstrap",
            *common,
            "--query_cache_policy", "reuse_or_build",
            "--output_dir", init_dir,
            "--scaffold_mode", "ulf_robust_consensus",
            "--generated_landmark_path", init_dir / "robust_ids.pkl",
            "--regenerate_scaffold",
            "--scaffold_budget", initialization["scaffold_budget"],
            "--scaffold_min_opacity", 0,
            "--scaffold_opacity_keep_quantile", 0.1,
            "--initialization_mode", "ulf_robust_geometry",
            "--ulf_consensus_keypoints", initialization["kcs_keypoints"],
            "--ulf_consensus_radius_px", initialization["kcs_radius_px"],
            "--ulf_consensus_min_visible_views", initialization["kcs_min_visible_views"],
            "--ulf_consensus_min_votes", initialization["kcs_min_votes"],
            "--ulf_consensus_min_rate", initialization["kcs_min_consensus_rate"],
            "--ulf_consensus_view_bins", initialization["kcs_view_bins"],
            "--ulf_consensus_min_distinct_view_bins", initialization["kcs_min_distinct_view_bins"],
            "--ulf_consensus_trajectory_bins", initialization["kcs_trajectory_bins"],
            "--ulf_consensus_min_distinct_trajectory_bins", initialization["kcs_min_distinct_trajectory_bins"],
            "--ulf_consensus_independent_bin_scoring",
            "--ulf_consensus_allow_nonconsensus_fallback",
            "--ulf_consensus_extent_quantile", 0.01,
            "--ulf_support_view_sampling", "uniform",
            "--ulf_support_mask_policy", initialization["support_mask_policy"],
            "--ulf_consensus_max_views", 0,
            "--ulf_fusion_max_views", 0,
            "--ulf_fusion_min_cosine", 0,
            "--ulf_fusion_view_bins", 4,
            "--ulf_fusion_descriptor_trim_fraction", initialization["gwff_trim_fraction"],
            "--ulf_fusion_descriptor_min_cosine", -1,
            "--ulf_fusion_trim_histogram_bins", 64,
            "--no-ulf_fusion_exact_bin_balance",
            "--no-native_outcome_mode",
            "--retrieval_weight", 0,
            "--trust_weight", 0,
            "--steps", 0,
            "--save_steps", 0,
        )
    steps = int(reconstruction["stage_a_steps"])
    stage_state = stage_dir / f"{steps}_lafgs_map_state.pt"
    if not stage_state.is_file():
        _run(
            "map_learning.bootstrap",
            *common,
            "--query_cache_policy", "readonly",
            "--output_dir", stage_dir,
            "--scaffold_mode", "file",
            "--landmark_path", landmark_ids,
            "--initial_state_path", bootstrap_state,
            "--initial_state_blend", 1,
            "--initialization_mode", "ulf_robust_geometry",
            "--native_outcome_mode",
            "--native_keep_weight", stage["keep_weight"],
            "--native_keep_margin", stage["keep_margin"],
            "--native_swap_weight", stage["swap_weight"],
            "--native_swap_margin", stage["swap_margin"],
            "--native_miss_weight", stage["miss_weight"],
            "--native_miss_margin", stage["miss_margin"],
            "--native_global_attractor_weight", stage["global_attractor_weight"],
            "--native_global_attractor_min_incoming", 4,
            "--native_global_attractor_support_power", 0.5,
            "--native_global_attractor_max_score", 4,
            "--native_semidense_weight", stage["local_peak_weight"],
            "--native_semidense_start_step", stage["local_peak_start_step"],
            "--native_semidense_interval", stage["local_peak_interval"],
            "--native_semidense_max_anchors", stage["local_peak_max_anchors"],
            "--native_semidense_neighbors", stage["local_peak_neighbors"],
            "--native_semidense_local_radius_px", stage["local_peak_radius_px"],
            "--native_semidense_target_sigma_px", stage["local_peak_sigma_px"],
            "--native_semidense_temperature", stage["local_peak_temperature"],
            "--native_semidense_protected_v2",
            "--native_semidense_measurement_min_reprojection_px", 2,
            "--native_semidense_measurement_max_reprojection_px", 8,
            "--native_semidense_surface_point_plane_m", 0.03,
            "--native_semidense_surface_max_distance_m", 0.15,
            "--native_semidense_surface_normal_cosine", 0.95,
            "--native_semidense_projected_neighbor_radius_px", 64,
            "--native_semidense_local_identity_weight", stage["local_identity_weight"],
            "--native_semidense_margin_preservation_weight", stage["margin_preservation_weight"],
            "--native_semidense_reference_refresh_steps", 500,
            "--native_semidense_alternate_global",
            "--native_semidense_max_gradient_ratio", 0.25,
            "--native_protected_set_weight", stage["high_precision_weight"],
            "--native_protected_set_start_step", stage["high_precision_start_step"],
            "--native_protected_set_interval", stage["high_precision_interval"],
            "--native_protected_set_refresh_visits", 1,
            "--native_protected_set_ransac_seed", 0,
            "--native_protected_set_ransac_reprojection_px", 8,
            "--native_protected_set_ransac_max_iterations", 5000,
            "--native_protected_set_ransac_min_iterations", 100,
            "--native_protected_set_max_pose_error_cm", 100,
            "--native_protected_set_max_useful", 96,
            "--native_protected_set_max_harmful", 96,
            "--native_protected_set_grid_rows", 4,
            "--native_protected_set_grid_cols", 4,
            "--native_protected_set_depth_bins", 4,
            "--native_protected_set_surface_voxel_m", 0.25,
            "--native_protected_set_max_per_surface_group", 2,
            "--retrieval_weight", 1,
            "--trust_weight", stage["trust_weight"],
            "--feature_lr", stage["feature_learning_rate"],
            "--weight_decay", stage["weight_decay"],
            "--hypothesis_topk", stage["hypothesis_topk"],
            "--positive_radius_px", stage["positive_radius_px"],
            "--negative_radius_px", stage["negative_radius_px"],
            "--steps", steps,
            "--save_steps", steps,
            "--log_interval", 100,
        )
    identity = "track_first_provenance" if gaussian_type == "2dgs" else "track_first"
    track_payload = track_dir / "track_micro_anchor_payload.pt"
    if not track_payload.is_file():
        track_args: list[object] = [
            *common,
            "--query_cache_policy", "readonly",
            "--output_dir", track_dir,
            "--scaffold_mode", "file",
            "--landmark_path", landmark_ids,
            "--initial_state_path", stage_state,
            "--initial_state_blend", 1,
            "--initialization_mode", "ulf_robust_geometry",
            "--native_outcome_mode",
            "--retrieval_weight", 0,
            "--trust_weight", 0,
            "--positive_radius_px", 2,
            "--negative_radius_px", 8,
            "--save_independent_geometry_teacher",
            "--geometry_teacher_identity_mode", identity,
            "--geometry_teacher_min_views", 3,
            "--geometry_teacher_min_view_bins", 2,
            "--geometry_teacher_min_parallax_deg", 1,
            "--geometry_teacher_parallax_quantile", 0.75,
            "--geometry_teacher_max_reprojection_px", 2,
            "--geometry_teacher_max_covariance_trace_m2", 0.01,
            "--geometry_teacher_max_rendered_depth_residual_m", 0.15,
            "--geometry_teacher_min_rendered_depth_observations", 2,
            "--geometry_teacher_track_pair_neighbors", 6,
            "--geometry_teacher_track_min_similarity", 0.65,
            "--geometry_teacher_track_min_margin", 0.01,
            "--geometry_teacher_track_max_epipolar_error_px", 2,
            "--geometry_teacher_track_epipolar_candidate_topk", 4,
            "--geometry_teacher_track_epipolar_recovered_min_similarity", -1,
            "--geometry_teacher_track_epipolar_recovered_min_margin", -1,
            "--geometry_teacher_track_allow_chain_tracks",
            "--geometry_teacher_track_assignment_max_distance_m", 0.2,
            "--geometry_teacher_track_assignment_min_margin_m", 0,
            "--save_track_micro_anchor_payload",
            "--steps", 0,
            "--save_steps", 0,
        ]
        if gaussian_type == "2dgs":
            track_args += [
                "--geometry_teacher_provenance_topk", 4,
                "--geometry_teacher_provenance_min_consensus_rate", 0.35,
                "--geometry_teacher_provenance_min_views", 2,
                "--geometry_teacher_provenance_group_max_landmarks", 4,
                "--geometry_teacher_provenance_group_min_relative_mass", 0.25,
                "--geometry_teacher_provenance_group_min_consensus_rate", 0.10,
            ]
        _run("map_learning.bootstrap", *track_args)
    return {
        "base_state": stage_state,
        "track_payload": track_payload,
        "query_cache": query_cache,
        "visibility_cache": visibility,
        "landmark_ids": landmark_ids,
    }


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
    valid_masks: str | Path | None = None,
    function_graph_shards: int = 1,
    provenance_shards: int = 1,
    observation_shards: int = 1,
) -> dict[str, Path]:
    """Build the frozen canonical map and real-image localization evidence."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonical = output / "canonical_map.pt"
    graph_v2 = output / "function_graph_v2.pt"
    provenance = output / "raster_provenance.pt"
    graph = output / "function_graph.pt"
    teacher = output / "complete_positive_teacher.pt"
    contract = output / "evidence_contract.json"
    if not canonical.is_file():
        _run(
            "topology.candidates",
            "--base_state", base_state,
            "--track_payload", track_payload,
            "--query_cache", query_cache,
            "--output", canonical,
            "--budget", 0,
        )
    if not graph_v2.is_file():
        arguments: list[object] = [
            "--anchor-map", canonical,
            "--query-cache", query_cache,
            "--topk", 64,
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
            "--anchor-map", canonical,
            "--query-cache", query_cache,
            "--gaussian-ply", prior_ply,
            "--gaussian-type", gaussian_type,
            "--sh-degree", sh_degree,
            "--function-graph", graph_v2,
            "--track-payload", track_payload,
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
            "--function-graph-v2", graph_v2,
            "--raster-provenance", provenance,
            "--output", graph,
        )
    if not teacher.is_file():
        _run_query_shards(
            module="map_learning.observations",
            merge_module="map_learning.merge_observations",
            arguments=[
            "--anchor-map", canonical,
            "--query-cache", query_cache,
            "--raster-provenance", provenance,
            "--track-payload", track_payload,
            ],
            output=teacher,
            shard_count=observation_shards,
        )
    if not contract.is_file():
        _run(
            "common.evidence_contract",
            "--query-cache", query_cache,
            "--track-payload", track_payload,
            "--primitive-prior", prior_ply,
            "--anchor-map", canonical,
            "--function-graph", graph,
            "--raster-provenance", provenance,
            "--positive-teacher", teacher,
            "--output", contract,
        )
    return {
        "canonical_map": canonical,
        "function_graph": graph,
        "positive_teacher": teacher,
        "raster_provenance": provenance,
        "evidence_contract": contract,
    }


def distill_compact_map(
    *,
    canonical_map: str | Path,
    function_graph: str | Path,
    positive_teacher: str | Path,
    track_payload: str | Path,
    query_cache: str | Path,
    output: str | Path,
    config: str | Path,
) -> Path:
    """Materialize the Track core plus Gaussian-supported coverage reserve."""
    cfg = load_mainline_config(config).values["reconstruction"]
    output = Path(output).expanduser().resolve()
    core_dir, reserve_dir = output / "track_core", output / "coverage_reserve"
    core_dir.mkdir(parents=True, exist_ok=True)
    reserve_dir.mkdir(parents=True, exist_ok=True)
    report_path = core_dir / "minimum_sufficient_build.json"
    if not report_path.is_file():
        _run(
            "topology.distillation",
            "--canonical-map", canonical_map,
            "--function-graph", function_graph,
            "--complete-positive-teacher", positive_teacher,
            "--track-payload", track_payload,
            "--query-cache", query_cache,
            "--output-dir", core_dir,
            "--track-cores", f"{cfg['track_core']}:medium",
            "--minimum-rows-per-query", cfg["minimum_rows_per_mapping_query"],
            "--maximum-reserve", cfg["maximum_reserve"],
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
        _run(
            "topology.coverage_reserve",
            "--core-map", matches[0],
            "--canonical-map", canonical_map,
            "--function-graph", function_graph,
            "--complete-positive-teacher", positive_teacher,
            "--track-payload", track_payload,
            "--query-cache", query_cache,
            "--output-dir", reserve_dir,
            "--reserve-additions", cfg["pose_reserve_additions"],
        )
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
    provenance_shards: int = 1,
    observation_shards: int = 1,
) -> dict[str, Path]:
    """Rebuild compact-map labels, then run frozen A1 reconstruction."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "raster_provenance.pt"
    teacher = output / "complete_positive_teacher.pt"
    if not provenance.is_file():
        arguments: list[object] = [
            "--anchor-map", compact_map,
            "--query-cache", query_cache,
            "--gaussian-ply", prior_ply,
            "--gaussian-type", gaussian_type,
            "--sh-degree", sh_degree,
            "--track-payload", track_payload,
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
    if not teacher.is_file():
        _run_query_shards(
            module="map_learning.observations",
            merge_module="map_learning.merge_observations",
            arguments=[
            "--anchor-map", compact_map,
            "--query-cache", query_cache,
            "--raster-provenance", provenance,
            "--track-payload", track_payload,
            ],
            output=teacher,
            shard_count=observation_shards,
        )
    reconstruction = load_mainline_config(config).values["reconstruction"]
    steps = int(reconstruction["metric_steps"])
    trained_map = output / f"anchor_map_step_{steps:04d}.pt"
    metric_state = output / f"metric_state_step_{steps:04d}.pt"
    if not trained_map.is_file() or not metric_state.is_file():
        train(
            map_path=compact_map,
            function_graph_path=function_graph,
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
            seed=2026,
        )
    return {
        "compact_provenance": provenance,
        "compact_positive_teacher": teacher,
        "trained_map": trained_map,
        "metric_state": metric_state,
    }


def write_pipeline_manifest(output: Path, values: dict) -> None:
    serializable = {key: str(value) for key, value in values.items()}
    (output / "pipeline_manifest.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )
