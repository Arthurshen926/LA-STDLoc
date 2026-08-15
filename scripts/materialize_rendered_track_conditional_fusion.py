#!/usr/bin/env python3
"""Materialize conditional artifact-aware GWFF descriptors on frozen Tracks.

This is a descriptor-fusion experiment, not a scientific gate.  Track
components, ray geometry, selected membership, row order, map size, query rows,
and the zero-residual identity metric remain fixed.  R1 artifact evidence only
controls whether a jointly descriptor-inconsistent observation may be trimmed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from common.hashing import sha256_file
from common.tensor_identity import recursive_bitwise_equal
from evidence.conditional_track_fusion import (
    MAXIMUM_TRIM_FRACTION,
    conditional_artifact_keep_masks,
)
from evidence.tracks import fuse_track_descriptors
from map_learning.metric import SharedLowRankMetric
from scripts.audit_rendered_track_artifact_cache_equivalence import (
    _ADDED_QUERY_FIELDS as R1_ADDED_QUERY_FIELDS,
    audit_artifact_cache_equivalence,
)


_CONDITIONAL_QUERY_FIELDS = {
    "native_conditional_artifact_trim_eligible",
    "native_conditional_descriptor_medoid_cosine",
    "native_descriptor_fusion_keep_mask",
}
_SOURCE_PATHS = (
    "evidence/conditional_track_fusion.py",
    "evidence/tracks.py",
    "scripts/materialize_rendered_track_conditional_fusion.py",
)


def _producer_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("conditional-fusion producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
    }


def _require_sha(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected).strip().lower():
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        reloaded = _load(temporary)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary conditional artifact did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary conditional report did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _topology_fields(state: Mapping[str, Any]) -> tuple[str, ...]:
    required = (
        "anchor_ids",
        "anchor_xyz",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
        "dependency_group_ids",
        "coarse_dependency_group_ids",
        "fine_identity_ids",
        "source_dependency_group_ids",
        "parent_source_track_ids",
        "repair_child_index",
        "repair_parent_child_count",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"selected map lacks frozen topology fields: {missing}")
    return required


def _audit_conditional_cache(
    source: Mapping[str, Any],
    artifact: Mapping[str, Any],
    conditional: Mapping[str, Any],
    track_payload: Mapping[str, Any],
) -> dict[str, Any]:
    r1_audit = audit_artifact_cache_equivalence(source, artifact, track_payload)
    if (
        conditional.get("schema")
        != "lafgs_rendered_rgb_conditional_artifact_fusion_cache"
        or conditional.get("version") != 1
        or conditional.get("uses_source_mapping_rgb") is not False
        or conditional.get("uses_test_queries") is not False
    ):
        raise ValueError("conditional cache schema or scope is invalid")
    source_queries = source["queries"]
    artifact_queries = artifact["queries"]
    conditional_queries = conditional["queries"]
    if list(source_queries) != list(conditional_queries):
        raise ValueError("conditional cache changed query order")
    allowed_top = {"artifact_stability", "conditional_artifact_fusion"}
    if set(conditional) != set(source) | allowed_top:
        raise ValueError("conditional cache has an unexpected top-level field")
    for key in source:
        if key in {"schema", "queries"}:
            continue
        if not recursive_bitwise_equal(source[key], conditional[key]):
            raise ValueError(f"conditional cache changed top-level field {key}")
    for name in source_queries:
        source_record = source_queries[name]
        artifact_record = artifact_queries[name]
        conditional_record = conditional_queries[name]
        expected_fields = (
            set(source_record) | R1_ADDED_QUERY_FIELDS | _CONDITIONAL_QUERY_FIELDS
        )
        if set(conditional_record) != expected_fields:
            raise ValueError(f"conditional query {name} fields differ")
        for key in source_record:
            if key == "source":
                continue
            if not recursive_bitwise_equal(source_record[key], conditional_record[key]):
                raise ValueError(f"conditional query {name} changed {key}")
        if conditional_record.get("source") != (
            "gaussian_rendered_rgb_conditional_artifact_gwff"
        ):
            raise ValueError(f"conditional query {name} source tag differs")
        for key in R1_ADDED_QUERY_FIELDS:
            if not recursive_bitwise_equal(
                artifact_record[key], conditional_record[key]
            ):
                raise ValueError(f"conditional query {name} changed R1 evidence {key}")
        row_count = int(torch.as_tensor(source_record["native_keypoints"]).shape[0])
        keep = torch.as_tensor(conditional_record["native_descriptor_fusion_keep_mask"])
        eligible = torch.as_tensor(
            conditional_record["native_conditional_artifact_trim_eligible"]
        )
        cosine = torch.as_tensor(
            conditional_record["native_conditional_descriptor_medoid_cosine"]
        )
        if (
            keep.dtype != torch.bool
            or eligible.dtype != torch.bool
            or keep.shape != (row_count,)
            or eligible.shape != (row_count,)
            or cosine.shape != (row_count,)
            or not cosine.is_floating_point()
            or not bool(torch.isfinite(cosine).all())
            or bool((~keep & ~eligible).any())
        ):
            raise ValueError(f"conditional query {name} annotations are invalid")
    return {
        **r1_audit,
        "artifact_evidence_bitwise_reused": True,
        "base_appearance_reliability_bitwise_exact": True,
        "localization_query_rows_bitwise_exact": True,
        "conditional_annotations_valid": True,
        "calibration_numeric_reuse_authorized": True,
        "content_equivalent_track_payload_reuse_authorized": True,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    inputs = {
        "appearance_cache": args.appearance_cache.resolve(),
        "artifact_cache": args.artifact_cache.resolve(),
        "track_payload": args.track_payload.resolve(),
        "selected_map": args.selected_map.resolve(),
    }
    expected = {
        "appearance_cache": args.expected_appearance_cache_sha256,
        "artifact_cache": args.expected_artifact_cache_sha256,
        "track_payload": args.expected_track_payload_sha256,
        "selected_map": args.expected_selected_map_sha256,
    }
    input_sha256 = {
        label: _require_sha(path, expected[label], label=label)
        for label, path in inputs.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    appearance = _load(inputs["appearance_cache"])
    artifact = _load(inputs["artifact_cache"])
    payload = _load(inputs["track_payload"])
    source_map = _load(inputs["selected_map"])
    audit_artifact_cache_equivalence(appearance, artifact, payload)
    if payload.get("rendered_rgb_only") is not True:
        raise ValueError("conditional fusion requires rendered-RGB-only Tracks")
    selected_tracks = torch.as_tensor(source_map["track_cluster_ids"]).long()
    annotations, diagnostics = conditional_artifact_keep_masks(
        payload=payload,
        appearance_cache=appearance,
        artifact_cache=artifact,
        selected_tracks=selected_tracks,
        maximum_trim_fraction=float(args.maximum_trim_fraction),
    )
    records: dict[str, dict[str, Any]] = {}
    for name, source_record in appearance["queries"].items():
        records[name] = {
            **source_record,
            **{key: artifact["queries"][name][key] for key in R1_ADDED_QUERY_FIELDS},
            **annotations[name],
            "source": "gaussian_rendered_rgb_conditional_artifact_gwff",
        }
    conditional_cache = {
        **appearance,
        "schema": "lafgs_rendered_rgb_conditional_artifact_fusion_cache",
        "version": 1,
        "queries": records,
        "artifact_stability": artifact["artifact_stability"],
        "conditional_artifact_fusion": {
            "policy": "artifact_low_and_descriptor_outlier_noncritical_trim_v1",
            "artifact_evidence_used_as_weight": False,
            "artifact_threshold": "strictly_below_within_track_median",
            "descriptor_threshold": "strictly_below_joint_sequence_pose_medoid_median",
            "strong_identity_policy": "identity_positive_certified_never_trim",
            "support_policy": "retain_at_least_one_observation_per_pose_bin_and_sequence",
            "maximum_trim_fraction": float(args.maximum_trim_fraction),
            "final_fusion": "existing_view_balanced_cosine_medoid_trim",
            "diagnostics": diagnostics,
        },
    }
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=conditional_cache,
        track_indices=selected_tracks,
        trim_fraction=float(args.descriptor_trim_fraction),
    )
    output_map = dict(source_map)
    output_map["anchor_features"] = fused.float()
    output_map["v7_metric_raw_features"] = fused.float()
    output_map["provenance"] = {
        **source_map.get("provenance", {}),
        "conditional_artifact_gwff": {
            "source_map": str(inputs["selected_map"]),
            "appearance_cache": str(inputs["appearance_cache"]),
            "artifact_cache": str(inputs["artifact_cache"]),
            "fixed_tracks_xyz_selection_row_order_and_size": True,
            "artifact_evidence_used_as_weight": False,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    for key in _topology_fields(source_map):
        if not recursive_bitwise_equal(source_map[key], output_map[key]):
            raise RuntimeError(f"conditional fusion modified topology field {key}")
    identity = _producer_identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "conditional_fusion_cache.pt"
    map_path = args.output_dir / "conditional_fusion_anchor_map.pt"
    metric_path = args.output_dir / "conditional_fusion_identity_metric.pt"
    equivalence_path = args.output_dir / "conditional_cache_equivalence_v2.json"
    _atomic_torch(conditional_cache, cache_path)
    _atomic_torch(output_map, map_path)
    metric = SharedLowRankMetric(
        descriptor_dim=int(fused.shape[1]), rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_payload = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(fused.shape[0], dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in metric.state_dict().items()
        },
        "map_path": str(map_path),
        "map_sha256": sha256_file(map_path),
        "step": 0,
        "protocol": "rendered_track_conditional_artifact_gwff_identity",
    }
    _atomic_torch(metric_payload, metric_path)
    audit = _audit_conditional_cache(appearance, artifact, conditional_cache, payload)
    checks = {
        "query_order_exact": audit["query_order_exact"],
        "localization_query_rows_bitwise_exact": audit[
            "localization_query_rows_bitwise_exact"
        ],
        "rendered_geometry_samples_bitwise_exact": audit[
            "rendered_geometry_samples_bitwise_exact"
        ],
        "artifact_evidence_bitwise_reused": audit["artifact_evidence_bitwise_reused"],
        "base_appearance_reliability_bitwise_exact": audit[
            "base_appearance_reliability_bitwise_exact"
        ],
        "conditional_annotations_valid": audit["conditional_annotations_valid"],
        "source_track_query_registry_exact": audit["source_track_query_registry_exact"],
        "source_image_free_mapping_only": audit["source_image_free_mapping_only"],
        "content_equivalent_track_payload_reuse_authorized": audit[
            "content_equivalent_track_payload_reuse_authorized"
        ],
        "calibration_numeric_reuse_authorized": audit[
            "calibration_numeric_reuse_authorized"
        ],
    }
    equivalence = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": all(checks.values()),
        "sources": {
            "source_cache": {
                "path": str(inputs["appearance_cache"]),
                "sha256": input_sha256["appearance_cache"],
            },
            "refreshed_cache": {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
            },
            "source_track_payload": {
                "path": str(inputs["track_payload"]),
                "sha256": input_sha256["track_payload"],
            },
        },
        "checks": checks,
        "audit": audit,
        "producer_identity": identity,
    }
    _atomic_json(equivalence, equivalence_path)
    report = {
        "schema": "lafgs_rendered_track_conditional_artifact_fusion_materialization",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "method_role": "experimental_method_enhancement_not_a_gate",
        "cpu_threads": int(args.cpu_threads),
        "inputs": {key: str(value) for key, value in inputs.items()},
        "input_sha256": input_sha256,
        "outputs": {
            "query_cache": str(cache_path),
            "anchor_map": str(map_path),
            "identity_metric": str(metric_path),
            "cache_equivalence": str(equivalence_path),
        },
        "output_sha256": {
            "query_cache": sha256_file(cache_path),
            "anchor_map": sha256_file(map_path),
            "identity_metric": sha256_file(metric_path),
            "cache_equivalence": sha256_file(equivalence_path),
        },
        "diagnostics": diagnostics,
        "producer_identity": identity,
    }
    _atomic_json(report, args.output_dir / "conditional_fusion_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-cache", type=Path, required=True)
    parser.add_argument("--expected-appearance-cache-sha256", required=True)
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--expected-artifact-cache-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--expected-selected-map-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-trim-fraction", type=float, default=MAXIMUM_TRIM_FRACTION
    )
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
