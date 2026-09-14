#!/usr/bin/env python3


"""Frozen mainline kernel extracted from scripts.materialize_v8_v2_projective_map."""

from __future__ import annotations


import argparse


import json


import os


from pathlib import Path


import time


import torch


from anygsloc.utils.hashing import sha256_file


from anygsloc.mapping.contracts import (
    ASSOCIATION_GRAPH_SCHEMA,
    RENDER_OBSERVATION_SCHEMA,
    require_schema,
)


import anygsloc.mapping.cycle_core_reserve as cycle_core_reserve_module


import anygsloc.mapping.projective_reconstruction as projective_reconstruction_module


import anygsloc.mapping.v2_filtered_observations as v2_filtered_observations_module


from anygsloc.mapping.cycle_core_reserve import select_cycle_core_coverage_reserve


from anygsloc.mapping.projective_association import build_projective_association_graph


from anygsloc.mapping.projective_completion import build_projective_completion


from anygsloc.mapping.projective_reconstruction import reconstruct_projective_anchors


from anygsloc.mapping.v2_filtered_observations import (
    build_v2_filtered_provider,
    remap_candidate_rows_to_source,
)


from anygsloc.mapping.view_support import build_anchor_view_support


from anygsloc.mapping.anchor_map import (
    identity_metric_state,
    materialize_projective_anchor_map,
    merge_projective_candidates,
)


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _implementation_sha256() -> dict[str, str]:
    """Hash implementation modules, never decorator wrapper bytecode."""

    return {
        "materializer": sha256_file(Path(__file__)),
        "projective_reconstruction": sha256_file(
            Path(projective_reconstruction_module.__file__)
        ),
        "v2_filtered_observations": sha256_file(
            Path(v2_filtered_observations_module.__file__)
        ),
        "cycle_core_reserve": sha256_file(Path(cycle_core_reserve_module.__file__)),
    }


def _load_v2_rows(
    paths: list[Path],
    names: list[str],
    *,
    observation_cache: Path,
    observation_cache_sha256: str,
    cache_input: dict | None = None,
    cache_configuration: dict | None = None,
) -> list[torch.Tensor]:
    records = {}
    shard_indices = set()
    expected_shard_count = len(paths)
    prior_signatures = set()
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard_index = int(shard.get("shard_index", -1))
        shard_input = dict(shard.get("input", {}))
        prior_signatures.add(
            tuple(
                shard_input.get(k)
                for k in [
                    "gaussian_ply_sha256",
                    "gaussian_type",
                    "sh_degree",
                    "background",
                ]
            )
        )
        if cache_input and cache_input.get("gaussian_ply_sha256"):
            from anygsloc.mapping.render_input_contract import validate_render_input

            validate_render_input(
                {"input": cache_input, "configuration": cache_configuration or {}},
                prior_sha256=shard_input.get("gaussian_ply_sha256"),
                gaussian_type=shard_input.get("gaussian_type"),
                sh_degree=shard_input.get("sh_degree", -1),
                background=shard_input.get("background"),
            )
        if (
            shard.get("schema") != "lafgs_v7_mapping_render_quality_audit_shard"
            or shard.get("status") != "PASS"
            or shard.get("uses_source_mapping_rgb") is not False
            or shard.get("uses_test_queries") is not False
            or int(shard.get("shard_count", -1)) != expected_shard_count
            or shard_index < 0
            or shard_index >= expected_shard_count
            or shard_index in shard_indices
            or Path(str(shard_input.get("observation_cache", ""))).resolve()
            != observation_cache.resolve()
            or shard_input.get("observation_cache_sha256") != observation_cache_sha256
            or int(shard.get("mapping_query_count", -1)) != len(names)
        ):
            raise ValueError("V2 audit shard is outside mapping-only scope")
        shard_indices.add(shard_index)
        for record in shard["records"]:
            index = int(record["query_index"])
            if index in records:
                raise ValueError("duplicate V2 query row")
            if record["query_name"] != names[index]:
                raise ValueError("V2 query registry differs from observation cache")
            records[index] = torch.as_tensor(record["row_valid"]).bool()
    if len(prior_signatures) != 1:
        raise ValueError("V2 shards use inconsistent Gaussian/render sources")
    if shard_indices != set(range(expected_shard_count)) or sorted(records) != list(
        range(len(names))
    ):
        raise ValueError("V2 audit shards do not exactly cover mapping queries")
    return [records[index] for index in range(len(names))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--v2-audit-shards", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--disable-v2-row-filter",
        action="store_true",
        help="Explicit no-V2 ablation: retain cached detector rows; audits are lineage-checked but not used to filter.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disable-cycle-consistency", action="store_true")
    parser.add_argument("--disable-robust-ray-reweighting", action="store_true")
    parser.add_argument(
        "--reuse-association-graph",
        type=Path,
        help=(
            "Reuse an exactly lineage-bound association graph when evaluating "
            "a reconstruction-only factor such as Track purification."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--triangulation-workers", type=int, default=4)
    parser.add_argument(
        "--maximum-valid-rows-per-query",
        type=int,
        default=None,
        help=(
            "Optional association budget applied after V2 filtering. This "
            "supports validity-compensated detection: detect more raw rows, "
            "then retain the strongest valid rows up to this fixed budget."
        ),
    )
    parser.add_argument(
        "--joint-track-purification",
        action="store_true",
        help=(
            "Reweight Track observations using mapping-only reprojection, "
            "descriptor, and relative rendered-depth consistency before a "
            "second pure-ray reconstruction."
        ),
    )
    parser.add_argument("--purification-minimum-weight", type=float, default=0.05)
    parser.add_argument(
        "--purification-maximum-adjusted-fraction-per-track",
        type=float,
        default=0.20,
        help=(
            "Maximum fraction of observations that Track purification may "
            "downweight inside any one Track."
        ),
    )
    parser.add_argument("--cycle-core-coverage-reserve", action="store_true")
    parser.add_argument("--reserve-spatial-voxel-size-m", type=float, default=0.50)
    parser.add_argument("--reserve-image-grid-y", type=int, default=6)
    parser.add_argument("--reserve-image-grid-x", type=int, default=8)
    args = parser.parse_args()
    if args.disable_v2_row_filter and args.reuse_association_graph is not None:
        parser.error("no-V2 ablation must rebuild associations")
    if args.disable_cycle_consistency and args.reuse_association_graph is not None:
        parser.error("association ablation must rebuild associations")
    if not 0.0 < float(args.purification_minimum_weight) <= 1.0:
        parser.error("--purification-minimum-weight must lie inside (0,1]")
    if not 0.0 <= float(args.purification_maximum_adjusted_fraction_per_track) <= 0.5:
        parser.error(
            "--purification-maximum-adjusted-fraction-per-track must lie inside [0,0.5]"
        )
    if (
        float(args.reserve_spatial_voxel_size_m) <= 0
        or int(args.reserve_image_grid_y) < 1
        or int(args.reserve_image_grid_x) < 1
    ):
        parser.error("cycle-core coverage parameters must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    started = time.perf_counter()
    cache = torch.load(args.observation_cache, map_location="cpu", weights_only=False)
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="V8 observations")
    names = list(cache["queries"])
    observation_cache_sha256 = sha256_file(args.observation_cache)
    audit_shard_sha256 = [sha256_file(path) for path in args.v2_audit_shards]
    rows = _load_v2_rows(
        args.v2_audit_shards,
        names,
        observation_cache=args.observation_cache,
        observation_cache_sha256=observation_cache_sha256,
        cache_input=cache.get("input"),
        cache_configuration=cache.get("configuration"),
    )
    original_v2_valid_count = sum(int(row.sum()) for row in rows)
    if args.disable_v2_row_filter:
        rows = [torch.ones_like(row, dtype=torch.bool) for row in rows]
    provider, source_rows, filter_report = build_v2_filtered_provider(
        cache,
        rows_by_query=rows,
        allow_empty_queries=True,
        maximum_rows_per_query=args.maximum_valid_rows_per_query,
    )
    if args.disable_v2_row_filter:
        filter_report = {
            **filter_report,
            "v2_row_filter_applied": False,
            "row_filter_policy": "all_cached_detector_rows_no_v2_ablation",
            "v2_valid_row_count_before_budget": original_v2_valid_count,
            "audit_role": "lineage_checked_not_used_for_row_selection",
            "filter_stage": "disabled_no_v2_ablation",
            "row_budget_selection": (
                "all_cached_detector_rows"
                if args.maximum_valid_rows_per_query is None
                else "stable_detector_score_topk_without_v2"
            ),
        }
        provider.payload["v2_preassociation_filter"] = filter_report
    # Release the unfiltered sparse tensors. Dense alpha/depth tensors remain
    # referenced by the filtered records for completion proposals.
    del cache

    association_started = time.perf_counter()
    reused_association_sha256 = None
    if args.reuse_association_graph is None:
        association = build_projective_association_graph(
            provider,
            pair_neighbors=6,
            minimum_similarity=0.65,
            minimum_margin=0.01,
            maximum_epipolar_error_px=2.0,
            minimum_track_views=3,
            device=args.device,
            require_cycle=not args.disable_cycle_consistency,
        )
        association["input_sha256"] = {
            "observation_cache": observation_cache_sha256,
            "v2_audit_shards": audit_shard_sha256,
        }
        if args.disable_v2_row_filter:
            association["input_sha256"]["v2_row_filter_disabled"] = True
        association["v2_preassociation_filter"] = filter_report
        association["source_keypoint_indices_by_query"] = source_rows
    else:
        reuse_path = args.reuse_association_graph.resolve()
        reused_association_sha256 = sha256_file(reuse_path)
        association = torch.load(reuse_path, map_location="cpu", weights_only=False)
        expected_input = {
            "observation_cache": observation_cache_sha256,
            "v2_audit_shards": audit_shard_sha256,
        }
        source = association.get("source_keypoint_indices_by_query")
        source_matches = (
            isinstance(source, list)
            and len(source) == len(source_rows)
            and all(
                torch.equal(torch.as_tensor(left), torch.as_tensor(right))
                for left, right in zip(source, source_rows)
            )
        )
        prior_filter = dict(association.get("v2_preassociation_filter", {}))
        filter_matches = all(
            prior_filter.get(key) == filter_report.get(key)
            for key in ("raw_row_count", "retained_row_count")
        ) and all(
            key not in prior_filter or prior_filter[key] == filter_report.get(key)
            for key in (
                "v2_valid_row_count_before_budget",
                "maximum_rows_per_query",
                "row_budget_selection",
            )
        )
        if (
            association.get("schema") != ASSOCIATION_GRAPH_SCHEMA
            or association.get("uses_source_mapping_rgb") is not False
            or association.get("uses_test_queries") is not False
            or list(association.get("query_names", ())) != names
            or association.get("input_sha256") != expected_input
            or not source_matches
            or not filter_matches
        ):
            raise ValueError("reused association graph differs from exact V2 inputs")
    association_seconds = time.perf_counter() - association_started

    reconstruction_started = time.perf_counter()
    base = reconstruct_projective_anchors(
        provider,
        association,
        minimum_views=3,
        minimum_view_bins=1,
        minimum_parallax_deg=1.0,
        maximum_reprojection_px=2.0,
        parallel_workers=args.triangulation_workers,
        parallel_minimum_tracks=5000,
        robust_reweighting=not args.disable_robust_ray_reweighting,
        joint_track_purification=args.joint_track_purification,
        purification_minimum_weight=args.purification_minimum_weight,
        purification_maximum_adjusted_fraction_per_track=(
            args.purification_maximum_adjusted_fraction_per_track
        ),
    )
    cycle_core_reserve_report = None
    if args.cycle_core_coverage_reserve:
        mapping_views = [provider.build_view(index) for index in range(len(provider))]
        base, cycle_core_reserve_report = select_cycle_core_coverage_reserve(
            base,
            association,
            keypoints_by_query=[view.keypoints for view in mapping_views],
            image_hw_by_query=[view.image_hw for view in mapping_views],
            spatial_voxel_size_m=args.reserve_spatial_voxel_size_m,
            image_grid_shape=(args.reserve_image_grid_y, args.reserve_image_grid_x),
        )
    base["candidate_kind"] = (
        "unfiltered_projective_track"
        if args.disable_v2_row_filter
        else "v2_projective_track"
    )
    completion = None
    completion_unavailable_reason = None
    try:
        completion = build_projective_completion(
            provider,
            association,
            voxel_size_m=0.05,
            alpha_minimum=0.05,
            minimum_similarity=0.7,
            minimum_margin=0.01,
            robust_reweighting=not args.disable_robust_ray_reweighting,
            maximum_epipolar_error_px=2.0,
            minimum_observations=3,
            minimum_camera_families=2,
            maximum_rows_per_view=256,
            safety_maximum_components=100000,
            device=args.device,
        )
    except ValueError as error:
        unavailable = {
            "no unused render-valid observations for completion": (
                "no_unused_render_valid_completion_observations"
            ),
            "depth proposals produced no descriptor-consistent component": (
                "no_descriptor_consistent_projective_completion"
            ),
            "association graph contains no ray-triangulated Anchor": (
                "no_ray_triangulated_completion_anchor"
            ),
        }
        if str(error) not in unavailable:
            raise
        completion_unavailable_reason = unavailable[str(error)]
    base = remap_candidate_rows_to_source(base, source_rows)
    candidate_parts = [base]
    if completion is not None:
        completion = remap_candidate_rows_to_source(completion, source_rows)
        candidate_parts.append(completion)
    candidates = merge_projective_candidates(candidate_parts)
    if args.disable_v2_row_filter:
        candidates["contract"]["v2_row_filter_applied"] = False
    reconstruction_seconds = time.perf_counter() - reconstruction_started

    association_path = args.output_dir / "association_graph.pt"
    candidates_path = args.output_dir / "projective_anchor_candidates.pt"
    map_path = args.output_dir / "projective_anchor_map.pt"
    metric_path = args.output_dir / "identity_metric.pt"
    _save(association, association_path)
    _save(candidates, candidates_path)
    lineage = {
        "implementation_sha256": _implementation_sha256(),
        "v8_observation_cache": str(args.observation_cache.resolve()),
        "v8_observation_cache_sha256": observation_cache_sha256,
        "v8_v2_audit_shards": [str(path.resolve()) for path in args.v2_audit_shards],
        "v8_v2_audit_shard_sha256": audit_shard_sha256,
        "v8_association_graph": str(association_path.resolve()),
        "v8_association_graph_sha256": sha256_file(association_path),
        "reused_association_graph": (
            None
            if args.reuse_association_graph is None
            else str(args.reuse_association_graph.resolve())
        ),
        "reused_association_graph_sha256": reused_association_sha256,
        "v8_filter_stage": (
            "disabled_no_v2_ablation"
            if args.disable_v2_row_filter
            else "after_detection_before_pair_association"
        ),
        "validity_compensated_detection": {
            "enabled": args.maximum_valid_rows_per_query is not None,
            "raw_keypoints_per_query": int(
                provider.payload.get("configuration", {}).get("keypoints", -1)
            ),
            "maximum_valid_rows_per_query": args.maximum_valid_rows_per_query,
            "selection": filter_report["row_budget_selection"],
        },
        "joint_track_purification": {
            "enabled": bool(args.joint_track_purification),
            "minimum_weight": float(args.purification_minimum_weight),
            "maximum_adjusted_fraction_per_track": float(
                args.purification_maximum_adjusted_fraction_per_track
            ),
            "report": base.get("joint_track_purification"),
        },
        "cycle_core_coverage_reserve": cycle_core_reserve_report,
        "projective_completion_attempted": True,
        "projective_completion_enabled": completion is not None,
        "projective_completion_unavailable_reason": completion_unavailable_reason,
    }
    state = materialize_projective_anchor_map(candidates, lineage=lineage)
    # The candidate CSR has already been remapped from filtered-local row ids
    # to the immutable source-cache row ids.  Preserve that contract in the
    # final map instead of relying on implicit knowledge in downstream audits.
    state["projective_anchor_observations"]["keypoint_index_semantics"] = (
        "original_unfiltered_observation_cache_row"
    )
    state["projective_anchor_construction"] = {
        **dict(state["projective_anchor_construction"]),
        "schema": "lafgs_v8_v2_projective_anchor_construction",
        "v2_preassociation_filter": not args.disable_v2_row_filter,
    }
    state["provenance"] = {
        **dict(state["provenance"]),
        "mapping_source": (
            "gaussian_render_no_v2_ablation"
            if args.disable_v2_row_filter
            else "gaussian_render_v2_filtered_before_projective_association"
        ),
    }
    mapping_poses = torch.stack(
        [
            provider.build_view(index).pose_w2c.float().cpu()
            for index in range(len(provider))
        ]
    )
    observation_csr = state["projective_anchor_observations"]
    state["anchor_view_support"] = build_anchor_view_support(
        anchor_xyz=state["anchor_xyz"],
        observation_offsets=observation_csr["observation_offsets"],
        observation_query_indices=observation_csr["query_indices"],
        mapping_pose_w2c=mapping_poses,
    )
    state["provenance"]["v24_mapping_only_anchor_view_support"] = True
    _save(state, map_path)
    metric = identity_metric_state(
        state, map_path=str(map_path.resolve()), map_sha256=sha256_file(map_path)
    )
    _save(metric, metric_path)
    report = {
        "schema": "lafgs_v8_v2_projective_map_materialization_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "filter": filter_report,
        "counts": {
            "mapping_views": len(provider),
            "association_components": int(association["diagnostics"]["track_count"]),
            "base_projective_anchors": int(base["anchor_xyz"].shape[0]),
            "completion_anchors": int(
                0 if completion is None else completion["anchor_xyz"].shape[0]
            ),
            "total_anchors": int(state["anchor_xyz"].shape[0]),
        },
        "association_diagnostics": dict(association["diagnostics"]),
        "output": {
            "association_graph": str(association_path.resolve()),
            "association_graph_sha256": sha256_file(association_path),
            "candidates": str(candidates_path.resolve()),
            "candidates_sha256": sha256_file(candidates_path),
            "map": str(map_path.resolve()),
            "map_sha256": sha256_file(map_path),
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
        },
        "contracts": {
            "superpoint_complete_unmasked_rgb": True,
            "v2_before_pair_association": not args.disable_v2_row_filter,
            "association_reused_exactly": args.reuse_association_graph is not None,
            "validity_compensated_detection": (
                args.maximum_valid_rows_per_query is not None
            ),
            "maximum_valid_rows_per_query": args.maximum_valid_rows_per_query,
            "joint_track_purification": bool(args.joint_track_purification),
            "purification_maximum_adjusted_fraction_per_track": float(
                args.purification_maximum_adjusted_fraction_per_track
            ),
            "gaussian_depth_role": (
                "observation_consistency_only"
                if args.joint_track_purification
                else "unused"
            ),
            "cycle_core_coverage_reserve": bool(args.cycle_core_coverage_reserve),
            "all_tracks_rebuilt": args.reuse_association_graph is None,
            "tracks_reused_exactly": args.reuse_association_graph is not None,
            "all_geometry_retriangulated": True,
            "all_descriptors_fused_from_v2_valid_observations": not args.disable_v2_row_filter,
            "query_detector_used": False,
            "feedback_used": False,
            "mapping_only_anchor_view_support": True,
            "empty_v2_query_policy": "retain_zero_row_mapping_view",
            "projective_completion_attempted": True,
            "projective_completion_available": completion is not None,
            "projective_completion_unavailable_reason": (completion_unavailable_reason),
        },
        "timing_seconds": {
            "association": association_seconds,
            "reconstruction_and_completion": reconstruction_seconds,
            "total": time.perf_counter() - started,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
