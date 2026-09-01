#!/usr/bin/env python3
"""Materialize V21 adaptation correspondence truth with fail-closed actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from common.hashing import sha256_file
from map_learning.v18_provenance_truth import backproject_query_surface
from map_learning.v19_track_extension_teacher import (
    TrackExtensionTier,
    assign_track_extension_truth,
    full_map_projection_candidate_graph,
    prepare_track_observation_bank,
    track_observation_consensus,
)
from map_learning.v21_correspondence_truth import (
    ROLE,
    SCHEMA,
    SEMANTICS,
    VERSION,
    atomic_torch_save_fresh,
    build_query_truth_record,
    gaussian_row_validity,
    resolve_teacher_action,
    sha256_json,
    status_counts,
    validate_frontend_support_alignment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SOURCES = (
    "map_learning/v18_provenance_truth.py",
    "map_learning/v19_track_extension_teacher.py",
    "map_learning/v21_correspondence_truth.py",
    "map_learning/v21_gaussian_support.py",
    "map_learning/v21_test_cache.py",
    "scripts/materialize_v21_correspondence_truth.py",
)


def _source(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"V21 source SHA256 differs: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def _verify_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        path = Path(str(source["path"]))
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(source["size_bytes"])
            or sha256_file(path) != source["sha256"]
        ):
            raise RuntimeError(f"V21 correspondence source changed: {path}")


def _resolve(path: object) -> Path:
    return Path(str(path)).expanduser().resolve()


def _validate_mapping_teacher_inputs(
    *,
    state: MappingLike,
    provenance: MappingLike,
    feature_cache: MappingLike,
    teacher: MappingLike,
    map_source: dict,
    provenance_source: dict,
    cache_source: dict,
) -> dict:
    """Validate the exact stable-map observation lineage used by V19."""

    names = list(state.get("v6_mapping_query_names", ()))
    feature_records = feature_cache.get("queries", feature_cache)
    observations = state.get("projective_anchor_observations")
    if not (
        state.get("schema") == "lafgs_materialized_anchor_map"
        and isinstance(observations, dict)
        and observations.get("schema") == "lafgs_projective_anchor_observations"
        and names
        and feature_cache.get("uses_test_queries") is False
        and feature_cache.get("uses_source_mapping_rgb") is False
        and provenance.get("schema")
        == "lafgs_v18_mapping_observation_gaussian_provenance"
        and provenance.get("uses_test_queries") is False
        and provenance.get("descriptor_independent") is True
        and provenance.get("full_gaussian_prior_evaluated") is True
        and provenance.get("full_depth_ordered_compositing") is True
        and names == list(feature_records)
        and names == list(provenance.get("mapping_query_names", ()))
    ):
        raise ValueError("V21 stable-map mapping observation contract differs")
    map_provenance = state.get("provenance", {})
    provenance_inputs = provenance.get("inputs", {})
    teacher_inputs = teacher.get("inputs", {})
    if not (
        map_provenance.get("v8_observation_cache_sha256")
        == cache_source["sha256"]
        and _resolve(map_provenance.get("v8_observation_cache"))
        == _resolve(cache_source["path"])
        and provenance_inputs.get("anchor_map_sha256") == map_source["sha256"]
        and _resolve(provenance_inputs.get("anchor_map"))
        == _resolve(map_source["path"])
        and provenance_inputs.get("observation_cache_sha256")
        == cache_source["sha256"]
        and _resolve(provenance_inputs.get("observation_cache"))
        == _resolve(cache_source["path"])
        and teacher_inputs.get("anchor_map_sha256") == map_source["sha256"]
        and _resolve(teacher_inputs.get("anchor_map"))
        == _resolve(map_source["path"])
        and teacher_inputs.get("mapping_provenance_sha256")
        == provenance_source["sha256"]
        and _resolve(teacher_inputs.get("mapping_provenance"))
        == _resolve(provenance_source["path"])
        and teacher_inputs.get("mapping_feature_cache_sha256")
        == cache_source["sha256"]
        and _resolve(teacher_inputs.get("mapping_feature_cache"))
        == _resolve(cache_source["path"])
    ):
        raise ValueError("V21 map/provenance/cache/teacher lineage differs")

    anchor_ids = torch.as_tensor(state.get("anchor_ids")).long().cpu()
    anchor_xyz = torch.as_tensor(state.get("anchor_xyz")).float().cpu()
    anchor_covariance = torch.as_tensor(
        state.get("anchor_position_covariance")
    ).float().cpu()
    anchor_count = int(anchor_ids.numel())
    offsets = torch.as_tensor(observations.get("observation_offsets")).long().cpu()
    observation_queries = torch.as_tensor(
        observations.get("query_indices")
    ).long().cpu()
    observation_keypoints = torch.as_tensor(
        observations.get("keypoint_indices")
    ).long().cpu()
    edge_count = int(observation_queries.numel())
    if not (
        anchor_count > 0
        and anchor_xyz.shape == (anchor_count, 3)
        and anchor_covariance.shape == (anchor_count, 3, 3)
        and offsets.shape == (anchor_count + 1,)
        and int(offsets[0]) == 0
        and int(offsets[-1]) == edge_count
        and not bool((offsets[1:] < offsets[:-1]).any())
        and observation_keypoints.shape == observation_queries.shape
        and edge_count == int(provenance.get("global_observation_count", -1))
        and int(provenance.get("anchor_count", -1)) == anchor_count
    ):
        raise ValueError("V21 Anchor/mapping observation registries do not align")
    provenance_rows = torch.as_tensor(provenance.get("observation_rows")).long().cpu()
    provenance_valid = torch.as_tensor(
        provenance.get("observation_valid")
    ).bool().cpu()
    if (
        provenance_rows.shape != (edge_count,)
        or provenance_valid.shape != (edge_count,)
        or edge_count
        and (
            int(provenance_rows.min()) != 0
            or int(provenance_rows.max()) != edge_count - 1
            or torch.unique(provenance_rows).numel() != edge_count
        )
    ):
        raise ValueError("V21 mapping provenance does not fully cover Anchor CSR")
    observation_valid = torch.zeros(edge_count, dtype=torch.bool)
    observation_valid[provenance_rows] = provenance_valid
    mapping_families = torch.as_tensor(
        provenance.get("mapping_view_family_ids")
    ).long().cpu()
    family_roles = {
        int(key): str(value) for key, value in teacher.get("family_roles", {}).items()
    }
    if mapping_families.shape != (len(names),) or any(
        int(family) not in family_roles for family in mapping_families.tolist()
    ):
        raise ValueError("V21 mapping view-family registry differs from teacher")
    track_bank_mask = torch.tensor(
        [
            family_roles[int(mapping_families[int(query)])] == "track_bank"
            for query in observation_queries
        ],
        dtype=torch.bool,
    ) & observation_valid
    if not bool(track_bank_mask.any()):
        raise ValueError("V21 teacher Track bank is empty")
    mapping_offsets = torch.as_tensor(
        provenance.get("mapping_pixel_center_offset")
    ).float().cpu()
    if mapping_offsets.shape != (len(names),):
        raise ValueError("V21 mapping pixel-centre registry differs")
    mapping_keypoints = []
    mapping_descriptors = []
    descriptor_dim = None
    for index, name in enumerate(names):
        record = feature_records[name]
        keypoints = torch.as_tensor(record["native_keypoints"]).float().cpu()
        descriptors = torch.as_tensor(record["native_descriptors"]).float().cpu()
        if keypoints.ndim != 2 or keypoints.shape[1] != 2 or descriptors.ndim != 2:
            raise ValueError("V21 mapping feature row is malformed")
        if keypoints.shape[0] != descriptors.shape[0]:
            raise ValueError("V21 mapping keypoints/descriptors do not align")
        descriptor_dim = descriptor_dim or int(descriptors.shape[1])
        if int(descriptors.shape[1]) != descriptor_dim:
            raise ValueError("V21 mapping descriptor dimensions differ")
        mapping_keypoints.append(keypoints + float(mapping_offsets[index]))
        mapping_descriptors.append(descriptors)
    return {
        "names": names,
        "anchor_count": anchor_count,
        "descriptor_dim": int(descriptor_dim),
        "anchor_offsets": offsets,
        "observation_queries": observation_queries,
        "observation_keypoints": observation_keypoints,
        "track_bank_mask": track_bank_mask,
        "mapping_families": mapping_families,
        "mapping_keypoints": mapping_keypoints,
        "mapping_descriptors": mapping_descriptors,
        "mapping_intrinsics": torch.as_tensor(
            provenance.get("mapping_intrinsics")
        ).float().cpu(),
        "mapping_poses_w2c": torch.as_tensor(
            provenance.get("mapping_poses_w2c")
        ).float().cpu(),
        "anchor_xyz": anchor_xyz,
        "anchor_covariance": anchor_covariance,
        "observation_count": offsets[1:] - offsets[:-1],
        "equivalence": torch.as_tensor(
            state.get("fine_identity_ids", torch.arange(anchor_count))
        ).long().cpu(),
    }


MappingLike = dict[str, Any]


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    map_source = _source(
        args.stable_map, expected_sha256=args.expected_stable_map_sha256
    )
    support_source = _source(
        args.gaussian_support,
        expected_sha256=args.expected_gaussian_support_sha256,
    )
    provenance_source = _source(
        args.mapping_provenance,
        expected_sha256=args.expected_mapping_provenance_sha256,
    )
    cache_source = _source(
        args.mapping_feature_cache,
        expected_sha256=args.expected_mapping_feature_cache_sha256,
    )
    teacher_source = _source(
        args.teacher_validation,
        expected_sha256=args.expected_teacher_validation_sha256,
    )
    frontend_sources = [_source(path) for path in args.frontend_cache]

    frontend_caches = [
        torch.load(source["path"], map_location="cpu", weights_only=False)
        for source in frontend_sources
    ]
    support = torch.load(
        support_source["path"], map_location="cpu", weights_only=False
    )
    registry, ordered_queries = validate_frontend_support_alignment(
        frontend_caches, support
    )
    support_frontend_sources = support.get("inputs", {}).get(
        "frontend_caches", ()
    )
    expected_frontend = sorted(
        (item["path"], item["sha256"]) for item in support_frontend_sources
    )
    actual_frontend = sorted(
        (item["path"], item["sha256"]) for item in frontend_sources
    )
    if expected_frontend != actual_frontend:
        raise ValueError("V21 Gaussian support uses another frontend cache set")
    if support.get("stable_map_sha256") != map_source["sha256"]:
        raise ValueError("V21 Gaussian support uses another stable map")

    state = torch.load(map_source["path"], map_location="cpu", weights_only=False)
    provenance = torch.load(
        provenance_source["path"], map_location="cpu", weights_only=False
    )
    feature_cache = torch.load(
        cache_source["path"], map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        teacher_source["path"], map_location="cpu", weights_only=False
    )
    decision = resolve_teacher_action(
        teacher,
        tier_name=args.tier,
        requested_action=args.requested_action,
    )
    context = _validate_mapping_teacher_inputs(
        state=state,
        provenance=provenance,
        feature_cache=feature_cache,
        teacher=teacher,
        map_source=map_source,
        provenance_source=provenance_source,
        cache_source=cache_source,
    )
    if int(frontend_caches[0]["anchor_count"]) != context["anchor_count"]:
        raise ValueError("V21 frontend stable-map Anchor count differs")
    if int(frontend_caches[0]["descriptor_dim"]) != context["descriptor_dim"]:
        raise ValueError("V21 frontend/mapping descriptor dimensions differ")
    prepared_bank = prepare_track_observation_bank(
        anchor_observation_offsets=context["anchor_offsets"],
        observation_query_indices=context["observation_queries"],
        observation_keypoint_indices=context["observation_keypoints"],
        observation_enabled=context["track_bank_mask"],
        mapping_keypoints=context["mapping_keypoints"],
        mapping_descriptors=context["mapping_descriptors"],
        mapping_view_family_ids=context["mapping_families"],
        maximum_observations_per_anchor=int(args.maximum_track_observations),
    )
    tier = TrackExtensionTier(**decision["thresholds"])
    geometry_gates = {
        "minimum_alpha": float(args.minimum_alpha),
        "maximum_relative_depth_spread": float(
            args.maximum_relative_depth_spread
        ),
        "minimum_local_valid_fraction": float(
            args.minimum_local_valid_fraction
        ),
        "maximum_projection_candidates_per_row": int(
            args.maximum_projection_candidates
        ),
        "saturated_candidate_rows": "abstain",
    }
    device = torch.device(args.device)
    records = []
    saturated_total = 0
    for completed, (frontend, support_record) in enumerate(
        ordered_queries, start=1
    ):
        geometry_valid = gaussian_row_validity(
            support_record,
            minimum_alpha=geometry_gates["minimum_alpha"],
            maximum_relative_depth_spread=geometry_gates[
                "maximum_relative_depth_spread"
            ],
            minimum_local_valid_fraction=geometry_gates[
                "minimum_local_valid_fraction"
            ],
        )
        keypoints = torch.as_tensor(frontend["keypoints"]).float().cpu()
        descriptors = torch.as_tensor(frontend["descriptors"]).float().cpu()
        depth = torch.as_tensor(
            support_record["gaussian_depth_at_keypoints"]
        ).float().cpu()
        offset = float(support_record["pixel_center_offset"])
        keypoints_centered = keypoints + offset
        surface, surface_valid = backproject_query_surface(
            keypoints_centered,
            depth,
            frontend["intrinsics"],
            frontend["pose_w2c"],
        )
        graph = full_map_projection_candidate_graph(
            keypoints=keypoints_centered,
            rendered_depth=depth,
            query_indices=torch.zeros(keypoints.shape[0], dtype=torch.long),
            anchor_xyz=context["anchor_xyz"],
            anchor_covariance=context["anchor_covariance"],
            observation_count=context["observation_count"],
            query_intrinsics=torch.as_tensor(frontend["intrinsics"])[None],
            query_poses_w2c=torch.as_tensor(frontend["pose_w2c"])[None],
            broad_reprojection_px=float(args.broad_reprojection_px),
            broad_depth_absolute_m=float(args.broad_depth_absolute_m),
            broad_depth_relative=float(args.broad_depth_relative),
            broad_normalized_depth_residual=float(
                args.broad_normalized_depth_residual
            ),
            broad_projection_std_px=float(args.broad_projection_std_px),
            minimum_observations=int(args.minimum_anchor_observations),
            maximum_candidates_per_row=int(args.maximum_projection_candidates),
            row_chunk_size=int(args.row_chunk_size),
            device=device,
        )
        candidate_counts = (
            torch.as_tensor(graph["candidate_offsets"])[1:]
            - torch.as_tensor(graph["candidate_offsets"][:-1])
        )
        saturated = candidate_counts >= int(args.maximum_projection_candidates)
        saturated_total += int(saturated.sum())
        exhaustive = ~saturated
        graph["query_valid"] = (
            graph["query_valid"] & geometry_valid & surface_valid & exhaustive
        )
        consensus = track_observation_consensus(
            candidate_graph=graph,
            query_surface_xyz=surface,
            query_descriptors=descriptors,
            anchor_observation_offsets=context["anchor_offsets"],
            observation_query_indices=context["observation_queries"],
            observation_keypoint_indices=context["observation_keypoints"],
            observation_enabled=context["track_bank_mask"],
            mapping_keypoints=context["mapping_keypoints"],
            mapping_descriptors=context["mapping_descriptors"],
            mapping_intrinsics=context["mapping_intrinsics"],
            mapping_poses_w2c=context["mapping_poses_w2c"],
            mapping_view_family_ids=context["mapping_families"],
            maximum_observations_per_candidate=int(
                args.maximum_track_observations
            ),
            edge_chunk_size=int(args.consensus_edge_chunk_size),
            device=device,
            prepared_observation_bank=prepared_bank,
        )
        truth = assign_track_extension_truth(
            candidate_graph=graph,
            consensus=consensus,
            equivalence_class_ids=context["equivalence"],
            tier=tier,
        )
        records.append(
            build_query_truth_record(
                frontend_record=frontend,
                support_record=support_record,
                v19_truth=truth,
                projection_candidate_offsets=graph["candidate_offsets"],
                geometry_valid=graph["query_valid"],
                action_authorized=bool(decision["action_authorized"]),
                tier_name=args.tier,
                requested_action=args.requested_action,
            )
        )
        print(
            json.dumps(
                {
                    "completed_queries": completed,
                    "query_count": len(ordered_queries),
                    "image_name": frontend["image_name"],
                    "diagnostic_decisive": int(
                        (
                            records[-1]["diagnostic_positive_offsets"][1:]
                            > records[-1]["diagnostic_positive_offsets"][:-1]
                        ).sum()
                    ),
                    "action_authorized": decision["action_authorized"],
                }
            ),
            flush=True,
        )

    producer_sources = [_source(REPOSITORY_ROOT / path) for path in PRODUCER_SOURCES]
    primary_sources = [
        map_source,
        support_source,
        provenance_source,
        cache_source,
        teacher_source,
        *frontend_sources,
        *producer_sources,
    ]
    _verify_sources(primary_sources)
    diagnostic_counts = status_counts(records, diagnostic=True)
    certified_counts = status_counts(records, diagnostic=False)
    diagnostic_positive_rows = diagnostic_counts["UNIQUE"] + diagnostic_counts[
        "EQUIVALENT"
    ]
    diagnostic_positive_edges = sum(
        int(torch.as_tensor(record["diagnostic_positive_anchor_rows"]).numel())
        for record in records
    )
    positive_edges = sum(
        int(torch.as_tensor(record["positive_anchor_rows"]).numel())
        for record in records
    )
    blocked_rows = (
        diagnostic_positive_rows if not decision["action_authorized"] else 0
    )
    blocked_edges = (
        diagnostic_positive_edges if not decision["action_authorized"] else 0
    )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": ROLE,
        "training_consumers_allowed": bool(decision["action_authorized"]),
        "planner_diagnostic_consumers_allowed": bool(
            decision["planner_diagnostic_authorized"]
        ),
        "control_or_confirmation_forbidden": True,
        "negative_labels_created": False,
        "ambiguous_or_unlabelled_are_negative": False,
        "feedback_enters_mapping_track_registry": False,
        "artifact_writes_map": False,
        "exact_poselib_recovery_is_identity_truth": False,
        "semantics": dict(SEMANTICS),
        "stable_map_sha256": map_source["sha256"],
        "gaussian_support_sha256": support_source["sha256"],
        "mapping_provenance_sha256": provenance_source["sha256"],
        "mapping_feature_cache_sha256": cache_source["sha256"],
        "teacher_validation_sha256": teacher_source["sha256"],
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "anchor_count": context["anchor_count"],
        "mapping_query_count": len(context["names"]),
        "mapping_observation_count": int(context["observation_queries"].numel()),
        "query_count": len(records),
        "teacher_action_decision": decision,
        "action_authorized": bool(decision["action_authorized"]),
        "gaussian_geometry_gates": geometry_gates,
        "gaussian_geometry_gates_sha256": sha256_json(geometry_gates),
        "saturated_projection_row_count": saturated_total,
        "status_counts": certified_counts,
        "diagnostic_status_counts": diagnostic_counts,
        "diagnostic_positive_edge_count": diagnostic_positive_edges,
        "positive_edge_count": positive_edges,
        "blocked_diagnostic_positive_row_count": blocked_rows,
        "blocked_diagnostic_positive_edge_count": blocked_edges,
        "blocked_diagnostic_positive_reason": (
            decision["action_block_reason"] if blocked_rows else None
        ),
        "inputs": {
            "stable_map": map_source,
            "gaussian_support": support_source,
            "mapping_provenance": provenance_source,
            "mapping_feature_cache": cache_source,
            "teacher_validation": teacher_source,
            "frontend_caches": frontend_sources,
            "producer_sources": producer_sources,
        },
        "records": records,
    }
    atomic_torch_save_fresh(payload, output)
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"records", "frontend_shard_registry", "inputs"}
    }
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontend-cache", type=Path, action="append", required=True
    )
    parser.add_argument("--gaussian-support", type=Path, required=True)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--mapping-feature-cache", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--expected-gaussian-support-sha256", required=True)
    parser.add_argument("--expected-mapping-provenance-sha256", required=True)
    parser.add_argument(
        "--expected-mapping-feature-cache-sha256", required=True
    )
    parser.add_argument("--expected-teacher-validation-sha256", required=True)
    parser.add_argument("--tier", choices=("tier_a", "tier_b", "tier_c"), required=True)
    parser.add_argument(
        "--requested-action",
        choices=(
            "destructive_map_control",
            "strong_metric_control",
            "soft_diagnostic",
            "planner_priority",
        ),
        required=True,
    )
    parser.add_argument("--minimum-alpha", type=float, default=0.2)
    parser.add_argument(
        "--maximum-relative-depth-spread", type=float, default=0.05
    )
    parser.add_argument(
        "--minimum-local-valid-fraction", type=float, default=1.0
    )
    parser.add_argument("--broad-reprojection-px", type=float, default=4.0)
    parser.add_argument("--broad-depth-absolute-m", type=float, default=0.25)
    parser.add_argument("--broad-depth-relative", type=float, default=0.05)
    parser.add_argument(
        "--broad-normalized-depth-residual", type=float, default=1.0
    )
    parser.add_argument("--broad-projection-std-px", type=float, default=2.0)
    parser.add_argument("--minimum-anchor-observations", type=int, default=3)
    parser.add_argument(
        "--maximum-projection-candidates", type=int, default=64
    )
    parser.add_argument("--maximum-track-observations", type=int, default=48)
    parser.add_argument("--row-chunk-size", type=int, default=256)
    parser.add_argument("--consensus-edge-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.minimum_anchor_observations,
        args.maximum_projection_candidates,
        args.maximum_track_observations,
        args.row_chunk_size,
        args.consensus_edge_chunk_size,
    ) <= 0:
        parser.error("V21 correspondence integer limits must be positive")
    return args


def main() -> None:
    materialize(parse_args())


if __name__ == "__main__":
    main()
