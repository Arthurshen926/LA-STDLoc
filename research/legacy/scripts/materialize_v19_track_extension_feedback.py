#!/usr/bin/env python3
"""Extend frozen mapping Tracks to certified virtual Queries without map writes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v8_safety_actions import certified_feedback_row_mask
from map_learning.v18_provenance_truth import backproject_query_surface
from map_learning.v19_track_extension_teacher import (
    TrackExtensionTier,
    assign_track_extension_truth,
    full_map_projection_candidate_graph,
    prepare_track_observation_bank,
    track_observation_consensus,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_raster(raster: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(raster).squeeze()
    pixels = torch.floor(torch.as_tensor(keypoints)).long()
    x = pixels[:, 0].clamp(0, value.shape[1] - 1)
    y = pixels[:, 1].clamp(0, value.shape[0] - 1)
    return value[y, x]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--mapping-feature-cache", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--expected-role", choices=("feedback_query", "confirmation_query"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid V19 feedback shard")

    batch = json.loads(args.certified_batch.read_text())
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.mapping_provenance, map_location="cpu", weights_only=False
    )
    validation = torch.load(
        args.teacher_validation, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        args.mapping_feature_cache, map_location="cpu", weights_only=False
    )
    records = cache.get("queries", cache)
    names = list(state["v6_mapping_query_names"])
    if not (
        batch.get("view_role") == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
        and batch.get("schema")
        in {
            "lafgs_v7_certified_clean_render_batch",
            "lafgs_v13_merged_certified_render_batch",
            "lafgs_v14_observer_split_certified_view",
        }
        and validation.get("schema")
        == "lafgs_v19_track_extension_teacher_validation"
        and validation.get("uses_test_queries") is False
        and validation.get("feedback_enters_track_registry") is False
        and validation.get("reference_available_for_novel_query") is False
        and validation.get("selection_uses_validation") is False
        and validation.get("authorization_uses_wilson_lower_bound") is True
        and validation.get("authorization_requires_independent_mapping_families")
        is True
        and provenance.get("uses_test_queries") is False
        and cache.get("uses_test_queries") is False
        and cache.get("uses_source_mapping_rgb") is False
        and names == list(records)
        and names == list(provenance["mapping_query_names"])
    ):
        raise ValueError("V19 novel Track-extension input contract differs")
    attested_cache = validation["inputs"]["mapping_feature_cache"]
    if Path(attested_cache).resolve() != args.mapping_feature_cache.resolve():
        raise ValueError("V19 novel teacher mapping descriptor bank differs")

    observations = state["projective_anchor_observations"]
    anchor_offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    edge_count = int(observation_queries.numel())
    provenance_rows = torch.as_tensor(provenance["observation_rows"]).long()
    if provenance_rows.numel() != edge_count:
        raise ValueError("V19 novel teacher mapping provenance is incomplete")
    observation_valid = torch.zeros(edge_count, dtype=torch.bool)
    observation_valid[provenance_rows] = torch.as_tensor(
        provenance["observation_valid"]
    ).bool()
    mapping_families = torch.as_tensor(
        provenance["mapping_view_family_ids"]
    ).long()
    family_roles = {
        int(key): value for key, value in validation["family_roles"].items()
    }
    track_bank_mask = torch.tensor(
        [
            family_roles[int(mapping_families[int(query)])] == "track_bank"
            for query in observation_queries
        ]
    ) & observation_valid
    mapping_offsets = torch.as_tensor(
        provenance["mapping_pixel_center_offset"]
    ).float()
    mapping_keypoints = [
        torch.as_tensor(records[name]["native_keypoints"]).float()
        + float(mapping_offsets[index])
        for index, name in enumerate(names)
    ]
    mapping_descriptors = [
        torch.as_tensor(records[name]["native_descriptors"]).float()
        for name in names
    ]
    prepared_bank = prepare_track_observation_bank(
        anchor_observation_offsets=anchor_offsets,
        observation_query_indices=observation_queries,
        observation_keypoint_indices=observation_keypoints,
        observation_enabled=track_bank_mask,
        mapping_keypoints=mapping_keypoints,
        mapping_descriptors=mapping_descriptors,
        mapping_view_family_ids=mapping_families,
    )
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchor_count))
    ).long()
    observation_count = anchor_offsets[1:] - anchor_offsets[:-1]
    tiers = {
        name: TrackExtensionTier(**item["thresholds"])
        for name, item in validation["selected_tiers"].items()
    }
    authorization = {
        name: bool(item["authorized_actions"])
        for name, item in validation["selected_tiers"].items()
    }

    selected_items = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    outputs = []
    totals = {
        name: {"rows": 0, "decisive": 0, "ambiguous": 0, "none": 0}
        for name in tiers
    }
    for completed, item in enumerate(selected_items, start=1):
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise ValueError("V19 certified Query SHA256 differs")
        source = torch.load(path, map_location="cpu", weights_only=False)
        if source["certificate"]["decision"] != "ACCEPT":
            continue
        valid = certified_feedback_row_mask(source["certificate"])
        source_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        keypoints_grid = torch.as_tensor(source["keypoints"])[source_rows].float()
        keypoints = keypoints_grid + 0.5
        descriptors = torch.as_tensor(source["descriptors"])[source_rows].float()
        depth = _sample_raster(source["depth_float16"], keypoints_grid).float()
        surface, surface_valid = backproject_query_surface(
            keypoints,
            depth,
            source["intrinsics"],
            source["pose_w2c"],
        )
        graph = full_map_projection_candidate_graph(
            keypoints=keypoints,
            rendered_depth=depth,
            query_indices=torch.zeros(source_rows.numel(), dtype=torch.long),
            anchor_xyz=state["anchor_xyz"],
            anchor_covariance=state["anchor_position_covariance"],
            observation_count=observation_count,
            query_intrinsics=torch.as_tensor(source["intrinsics"])[None],
            query_poses_w2c=torch.as_tensor(source["pose_w2c"])[None],
            device=args.device,
        )
        graph["query_valid"] &= surface_valid
        consensus = track_observation_consensus(
            candidate_graph=graph,
            query_surface_xyz=surface,
            query_descriptors=descriptors,
            anchor_observation_offsets=anchor_offsets,
            observation_query_indices=observation_queries,
            observation_keypoint_indices=observation_keypoints,
            observation_enabled=track_bank_mask,
            mapping_keypoints=mapping_keypoints,
            mapping_descriptors=mapping_descriptors,
            mapping_intrinsics=provenance["mapping_intrinsics"],
            mapping_poses_w2c=provenance["mapping_poses_w2c"],
            mapping_view_family_ids=mapping_families,
            device=args.device,
            prepared_observation_bank=prepared_bank,
        )
        truths = {}
        for name, tier in tiers.items():
            truth = assign_track_extension_truth(
                candidate_graph=graph,
                consensus=consensus,
                equivalence_class_ids=equivalence,
                tier=tier,
            )
            truths[name] = truth
            counts = truth["status_counts"]
            totals[name]["rows"] += int(truth["row_count"])
            totals[name]["decisive"] += int(counts["UNIQUE"] + counts["EQUIVALENT"])
            totals[name]["ambiguous"] += int(counts["AMBIGUOUS"])
            totals[name]["none"] += int(counts["NONE"])
        outputs.append(
            {
                "query_index": int(source["query_index"]),
                "pose_family_id": int(source["pose_family_id"]),
                "source_record": str(path),
                "source_record_sha256": item["sha256"],
                "source_query_rows": source_rows,
                "projection_candidate_graph": graph,
                "track_consensus": consensus,
                "truth_tiers": truths,
            }
        )
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "completed": completed,
                    "selected": len(selected_items),
                    "accepted": len(outputs),
                }
            ),
            flush=True,
        )
    artifact = {
        "schema": "lafgs_v19_novel_track_extension_shard",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "view_role": args.expected_role,
        "feedback_enters_track_registry": False,
        "reference_source": "mapping_observation_track_membership",
        "reference_available_for_novel_query": False,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "tier_action_authorization": authorization,
        "totals": totals,
        "records": outputs,
        "inputs": {
            "certified_batch": str(args.certified_batch.resolve()),
            "certified_batch_sha256": sha256_file(args.certified_batch),
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "mapping_provenance": str(args.mapping_provenance.resolve()),
            "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
            "mapping_feature_cache": str(args.mapping_feature_cache.resolve()),
            "mapping_feature_cache_sha256": validation["inputs"][
                "mapping_feature_cache_sha256"
            ],
            "teacher_validation": str(args.teacher_validation.resolve()),
            "teacher_validation_sha256": sha256_file(args.teacher_validation),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {key: value for key, value in artifact.items() if key != "records"}
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
