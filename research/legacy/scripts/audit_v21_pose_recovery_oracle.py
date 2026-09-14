#!/usr/bin/env python3
"""Audit exact PoseLib recovery headroom on V21 adaptation queries."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import uuid

import torch

from common.hashing import sha256_file
from map_learning.v21_pose_leverage import (
    GAUSSIAN_GEOMETRY_SOURCE,
    TRACK_CONSENSUS_DIAGNOSTIC,
    analyze_pose_recovery_query,
    summarize_pose_recovery,
)
from map_learning.v21_correspondence_truth import (
    SCHEMA as CORRESPONDENCE_TRUTH_SCHEMA,
    VERSION as CORRESPONDENCE_TRUTH_VERSION,
    validate_correspondence_payload,
)
from map_learning.v21_gaussian_support import (
    SCHEMA as GAUSSIAN_SUPPORT_SCHEMA,
    VERSION as GAUSSIAN_SUPPORT_VERSION,
    validate_support_payload,
)
from map_learning.v21_test_cache import tensor_sha256, validate_cache_payload


CACHE_SCHEMA = "lafgs_v21_test_frontend_baseline_cache"
CACHE_VERSION = 1
OUTPUT_SCHEMA = "lafgs_v21_pose_recovery_oracle_shard"


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"V21 {label} SHA256 differs")
    return actual


def _field(record: dict, *names: str, required: bool = True):
    for name in names:
        if name in record:
            return record[name]
    if required:
        raise ValueError(f"V21 adaptation record lacks required field {names[0]}")
    return None


def _map_binding(cache: dict) -> str | None:
    inputs = cache.get("inputs", {})
    if isinstance(inputs, dict):
        stable = inputs.get("stable_map")
        if isinstance(stable, dict) and stable.get("sha256") is not None:
            return str(stable["sha256"])
    return None


def _optional_baseline(record: dict) -> dict | None:
    baseline = record.get("baseline")
    if isinstance(baseline, dict):
        return baseline
    keys = {
        "pose_w2c": record.get("baseline_pose_w2c"),
        "translation_error_cm": record.get("baseline_translation_error_cm"),
        "rotation_error_deg": record.get("baseline_rotation_error_deg"),
        "task_error": record.get("baseline_task_error"),
        "r5_success": record.get("baseline_r5_success"),
        "inlier_count": record.get("baseline_inlier_count"),
        "inlier_query_rows": record.get("baseline_inliers"),
    }
    if keys["r5_success"] is None:
        keys["r5_success"] = record.get("baseline_r5")
    return keys if any(value is not None for value in keys.values()) else None


def _validate_cached_baseline(cached: dict | None, exact: dict) -> None:
    if cached is None:
        return
    numeric = (
        "translation_error_cm",
        "rotation_error_deg",
        "task_error",
    )
    for key in numeric:
        if cached.get(key) is None:
            continue
        value = float(cached[key])
        if not math.isfinite(value) or abs(value - float(exact[key])) > 1e-4:
            raise ValueError(f"V21 cached baseline {key} differs from exact replay")
    if cached.get("r5_success") is not None and bool(cached["r5_success"]) != bool(
        exact["r5_success"]
    ):
        raise ValueError("V21 cached baseline R5 status differs from exact replay")
    if cached.get("inlier_count") is not None and int(cached["inlier_count"]) != int(
        exact["inlier_count"]
    ):
        raise ValueError("V21 cached baseline inlier count differs from exact replay")
    if cached.get("inlier_query_rows") is not None and not torch.equal(
        torch.as_tensor(cached["inlier_query_rows"]).long().reshape(-1),
        torch.as_tensor(exact["inlier_query_rows"]).long().reshape(-1),
    ):
        raise ValueError("V21 cached baseline inlier rows differ from exact replay")
    if cached.get("pose_w2c") is not None and not torch.equal(
        torch.as_tensor(cached["pose_w2c"]).float().cpu(),
        torch.as_tensor(exact["pose_w2c"]).float().cpu(),
    ):
        raise ValueError("V21 cached baseline pose differs from exact replay")


def _source_identity(value: object, *, label: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"V21 {label} source record is missing")
    raw_path = str(value.get("path", ""))
    digest = str(value.get("sha256", ""))
    if not raw_path or len(digest) != 64:
        raise ValueError(f"V21 {label} source identity is invalid")
    path = str(Path(raw_path).expanduser().resolve())
    return path, digest


def _validate_complete_cache_shards(
    cache_entries: list[tuple[Path, str, dict]],
) -> list[tuple[Path, str, dict]]:
    """Reject a valid-looking subset of the adaptation shard registry."""

    if not cache_entries:
        raise ValueError("V21 adaptation cache set is empty")
    first = cache_entries[0][2]
    shard_count = int(first["shard_count"])
    registry_sha = str(first["shard_registry"]["registry_sha256"])
    split_source = _source_identity(
        first.get("inputs", {}).get("split_manifest"), label="split"
    )
    stable_source = _source_identity(
        first.get("inputs", {}).get("stable_map"), label="stable map"
    )
    coordinates: set[int] = set()
    paths: set[str] = set()
    digests: set[str] = set()
    for path, digest, payload in cache_entries:
        coordinate = int(payload["shard_index"])
        if (
            int(payload["shard_count"]) != shard_count
            or payload["shard_registry"]["registry_sha256"] != registry_sha
            or _source_identity(
                payload.get("inputs", {}).get("split_manifest"), label="split"
            )
            != split_source
            or _source_identity(
                payload.get("inputs", {}).get("stable_map"), label="stable map"
            )
            != stable_source
        ):
            raise ValueError("V21 adaptation cache shard registry/lineage differs")
        resolved = str(path)
        if coordinate in coordinates or resolved in paths or digest in digests:
            raise ValueError("V21 adaptation cache shard set is duplicated")
        coordinates.add(coordinate)
        paths.add(resolved)
        digests.add(digest)
    if coordinates != set(range(shard_count)):
        raise ValueError("V21 adaptation cache shards do not cover the full registry")
    return sorted(cache_entries, key=lambda value: int(value[2]["shard_index"]))


def _ordered_source_records(
    cache_entries: list[tuple[Path, str, dict]],
) -> list[tuple[int, int, dict]]:
    registry = cache_entries[0][2]["shard_registry"]
    ordinal_by_query = {
        int(row["query_index"]): int(row["ordinal"]) for row in registry["rows"]
    }
    records = [
        (cache_index, local_index, record)
        for cache_index, (_, _, payload) in enumerate(cache_entries)
        for local_index, record in enumerate(payload["records"])
    ]
    if (
        len(records) != int(registry["role_query_count"])
        or len(ordinal_by_query) != len(records)
        or {int(record["query_index"]) for _, _, record in records}
        != set(ordinal_by_query)
    ):
        raise ValueError("V21 adaptation caches do not exactly cover registry rows")
    return sorted(
        records,
        key=lambda value: ordinal_by_query[int(value[2]["query_index"])],
    )


def _join_gaussian_support(
    *,
    support_entries: list[tuple[Path, str, dict]],
    cache_entries: list[tuple[Path, str, dict]],
    source_records: list[tuple[int, int, dict]],
    map_path: Path,
    map_sha: str,
) -> dict[int, dict]:
    if not support_entries:
        return {}
    cache_sources = {(str(path), digest) for path, digest, _ in cache_entries}
    split_sources = {
        _source_identity(payload.get("inputs", {}).get("split_manifest"), label="split")
        for _, _, payload in cache_entries
    }
    registry_shas = {
        str(payload.get("shard_registry", {}).get("registry_sha256", ""))
        for _, _, payload in cache_entries
    }
    if len(split_sources) != 1 or len(registry_shas) != 1:
        raise ValueError("V21 frontend cache split/registry lineage differs")
    source_by_query = {
        int(record["query_index"]): (cache_index, local_index, record)
        for cache_index, local_index, record in source_records
    }
    if len(source_by_query) != len(source_records):
        raise ValueError("V21 frontend cache query registry is duplicated")
    joined: dict[int, dict] = {}
    referenced_caches: set[tuple[str, str]] = set()
    for path, digest, payload in support_entries:
        validate_support_payload(payload)
        if not (
            payload.get("schema") == GAUSSIAN_SUPPORT_SCHEMA
            and payload.get("version") == GAUSSIAN_SUPPORT_VERSION
            and payload.get("role") == "adaptation"
            and payload.get("stable_map_sha256") == map_sha
            and payload.get("deployment_authority") is False
            and payload.get("correspondence_truth_claimed") is False
        ):
            raise ValueError("V21 Gaussian support authority contract differs")
        support_inputs = payload["inputs"]
        if _source_identity(
            support_inputs.get("stable_map"), label="Gaussian stable map"
        ) != (str(map_path), map_sha):
            raise ValueError("V21 Gaussian support frozen-map lineage differs")
        if _source_identity(
            support_inputs.get("split_manifest"), label="Gaussian split"
        ) not in split_sources:
            raise ValueError("V21 Gaussian support split lineage differs")
        if payload.get("frontend_shard_registry_sha256") not in registry_shas:
            raise ValueError("V21 Gaussian support frontend registry differs")
        local_sources = {
            _source_identity(value, label="Gaussian frontend cache")
            for value in support_inputs.get("frontend_caches", [])
        }
        if not local_sources or not local_sources.issubset(cache_sources):
            raise ValueError("V21 Gaussian support references another frontend cache")
        referenced_caches.update(local_sources)
        for record in payload["records"]:
            query_index = int(record["query_index"])
            if query_index in joined or query_index not in source_by_query:
                raise ValueError("V21 Gaussian support query registry differs")
            cache_index, _, frontend = source_by_query[query_index]
            cache_path, cache_sha, cache_payload = cache_entries[cache_index]
            if not (
                record["image_name"] == frontend["image_name"]
                and record["image_sha256"] == frontend["image_sha256"]
                and record["sequence_id"] == frontend["sequence_id"]
                and int(record["frame_index"]) == int(frontend["frame_index"])
                and record["block_id"] == frontend["block_id"]
                and record["role"] == frontend["role"] == "adaptation"
                and record["source_record_sha256"]
                == frontend["source_record_sha256"]
                and record["pose_w2c_sha256"] == frontend["pose_w2c_sha256"]
                and int(record["keypoint_count"])
                == int(torch.as_tensor(frontend["keypoints"]).shape[0])
                and record["keypoints_sha256"]
                == tensor_sha256(torch.as_tensor(frontend["keypoints"]).float())
                and record["intrinsics_sha256"]
                == tensor_sha256(frontend["intrinsics"])
                and torch.equal(
                    torch.as_tensor(record["image_hw"]).long(),
                    torch.as_tensor(frontend["image_hw"]).long(),
                )
                and record["frontend_cache_path"] == str(cache_path)
                and record["frontend_cache_sha256"] == cache_sha
                and int(record["frontend_shard_index"])
                == int(cache_payload["shard_index"])
                and float(record["pixel_center_offset"])
                == float(cache_payload["baseline_contract"]["pixel_center_offset"])
                and torch.equal(
                    torch.as_tensor(record["sample_pixel_xy"]).long(),
                    torch.floor(
                        torch.as_tensor(frontend["keypoints"]).float()
                        + float(record["pixel_center_offset"])
                    ).long(),
                )
            ):
                raise ValueError("V21 Gaussian support row binding differs")
            joined[query_index] = record
        # Preserve the support artifact in exception traces and output lineage.
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("V21 Gaussian support changed while joining")
    if referenced_caches != cache_sources or set(joined) != set(source_by_query):
        raise ValueError("V21 Gaussian support does not exactly cover frontend caches")
    return joined


def _support_geometry_kwargs(record: dict | None) -> dict:
    if record is None:
        return {}
    return {
        "gaussian_depth_at_keypoints": record["gaussian_depth_at_keypoints"],
        "gaussian_alpha_at_keypoints": record["gaussian_alpha_at_keypoints"],
        "gaussian_valid_keypoint_mask": record["gaussian_support_valid"],
        "gaussian_relative_depth_spread_3x3": record[
            "gaussian_relative_depth_spread_3x3"
        ],
        "gaussian_local_valid_fraction_3x3": record[
            "gaussian_local_valid_fraction_3x3"
        ],
    }


def _join_correspondence_truth(
    *,
    truth_entry: tuple[Path, str, dict],
    cache_entries: list[tuple[Path, str, dict]],
    support_entries: list[tuple[Path, str, dict]],
    source_records: list[tuple[int, int, dict]],
    map_path: Path,
    map_sha: str,
    anchor_count: int,
) -> tuple[dict[int, dict], dict]:
    """Bind a planner-only Track diagnostic to the exact Query registry."""

    path, digest, payload = truth_entry
    validate_correspondence_payload(payload)
    decision = payload.get("teacher_action_decision", {})
    if not (
        payload.get("schema") == CORRESPONDENCE_TRUTH_SCHEMA
        and payload.get("version") == CORRESPONDENCE_TRUTH_VERSION
        and payload.get("role") == "adaptation"
        and payload.get("action_authorized") is False
        and payload.get("training_consumers_allowed") is False
        and payload.get("planner_diagnostic_consumers_allowed") is True
        and payload.get("artifact_writes_map") is False
        and payload.get("exact_poselib_recovery_is_identity_truth") is False
        and decision.get("planner_diagnostic_authorized") is True
        and decision.get("action_authorized") is False
        and int(payload.get("anchor_count", -1)) == int(anchor_count)
    ):
        raise ValueError("V21 correspondence truth is not planner-only diagnostic")
    if len(support_entries) != 1:
        raise ValueError("V21 Track diagnostic requires one complete Gaussian support")
    support_path, support_sha, _ = support_entries[0]
    inputs = payload["inputs"]
    if _source_identity(
        inputs.get("stable_map"), label="correspondence stable map"
    ) != (str(map_path), map_sha):
        raise ValueError("V21 correspondence truth frozen-map lineage differs")
    if _source_identity(
        inputs.get("gaussian_support"), label="correspondence Gaussian support"
    ) != (str(support_path), support_sha):
        raise ValueError("V21 correspondence truth Gaussian lineage differs")
    cache_sources = {(str(path), digest) for path, digest, _ in cache_entries}
    truth_cache_sources = {
        _source_identity(value, label="correspondence frontend cache")
        for value in inputs.get("frontend_caches", [])
    }
    if truth_cache_sources != cache_sources:
        raise ValueError("V21 correspondence truth frontend lineage differs")
    registry = cache_entries[0][2]["shard_registry"]
    if (
        payload.get("frontend_shard_registry") != registry
        or payload.get("frontend_shard_registry_sha256")
        != registry["registry_sha256"]
    ):
        raise ValueError("V21 correspondence truth registry differs")

    frontend_by_query = {
        int(record["query_index"]): record for _, _, record in source_records
    }
    if len(frontend_by_query) != len(source_records):
        raise ValueError("V21 correspondence frontend registry is duplicated")
    joined = {}
    for record in payload["records"]:
        query_index = int(record["query_index"])
        frontend = frontend_by_query.get(query_index)
        if frontend is None or query_index in joined:
            raise ValueError("V21 correspondence truth query registry differs")
        if not (
            record["image_name"] == frontend["image_name"]
            and record["image_sha256"] == frontend["image_sha256"]
            and record["sequence_id"] == frontend["sequence_id"]
            and int(record["frame_index"]) == int(frontend["frame_index"])
            and record["block_id"] == frontend["block_id"]
            and record["role"] == frontend["role"] == "adaptation"
            and record["source_record_sha256"]
            == frontend["source_record_sha256"]
            and record["pose_w2c_sha256"] == frontend["pose_w2c_sha256"]
            and int(record["keypoint_count"])
            == int(torch.as_tensor(frontend["keypoints"]).shape[0])
            and record["keypoints_sha256"]
            == tensor_sha256(torch.as_tensor(frontend["keypoints"]).float())
            and record["descriptors_sha256"]
            == tensor_sha256(torch.as_tensor(frontend["descriptors"]).float())
            and record["action_authorized"] is False
        ):
            raise ValueError("V21 correspondence truth row binding differs")
        joined[query_index] = record
    if set(joined) != set(frontend_by_query):
        raise ValueError("V21 correspondence truth does not exactly cover Queries")
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("V21 correspondence truth changed while joining")
    return joined, payload


def _resolve_baseline_plant(
    contract: dict, *, requested_reprojection_px: float | None, requested_seed: int | None
) -> tuple[float, int]:
    reprojection_px = float(contract.get("reprojection_error_px", math.nan))
    seed = int(contract.get("seed", -1))
    if not math.isfinite(reprojection_px) or reprojection_px <= 0.0 or seed < 0:
        raise ValueError("V21 cached exact plant parameters are invalid")
    if (
        requested_reprojection_px is not None
        and float(requested_reprojection_px) != reprojection_px
    ):
        raise ValueError("V21 requested reprojection threshold differs from cache")
    if requested_seed is not None and int(requested_seed) != seed:
        raise ValueError("V21 requested seed differs from cache")
    return reprojection_px, seed


def _validate_output_payload(payload: dict) -> None:
    records = payload.get("records")
    summary = payload.get("summary")
    inputs = payload.get("input")
    positive_source = payload.get("positive_source")
    if not (
        payload.get("schema") == OUTPUT_SCHEMA
        and payload.get("version") == 1
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("role") == "adaptation"
        and payload.get("correspondence_identity_authority_present") is False
        and payload.get("controller_authorized_query_count_must_be_zero") is True
        and payload.get("all_pose_recovery_claims_use_exact_poselib") is True
        and payload.get("exact_poselib_is_controller_action_authority") is False
        and positive_source
        in {GAUSSIAN_GEOMETRY_SOURCE, TRACK_CONSENSUS_DIAGNOSTIC}
        and isinstance(inputs, dict)
        and (
            (positive_source == TRACK_CONSENSUS_DIAGNOSTIC)
            == isinstance(inputs.get("correspondence_truth"), dict)
        )
        and isinstance(records, list)
        and isinstance(summary, dict)
        and int(summary.get("query_count", -1)) == len(records)
        and int(summary.get("controller_authorized_query_count", -1)) == 0
        and not any(bool(record.get("controller_authorized")) for record in records)
        and all(record.get("positive_source") == positive_source for record in records)
    ):
        raise ValueError("V21 pose-recovery oracle output contract is invalid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-source",
        choices=(GAUSSIAN_GEOMETRY_SOURCE, TRACK_CONSENSUS_DIAGNOSTIC),
        required=True,
    )
    parser.add_argument(
        "--adaptation-cache", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--expected-adaptation-cache-sha256", action="append", required=True
    )
    parser.add_argument("--gaussian-support", type=Path, action="append", default=[])
    parser.add_argument(
        "--expected-gaussian-support-sha256", action="append", default=[]
    )
    parser.add_argument("--correspondence-truth", type=Path)
    parser.add_argument("--expected-correspondence-truth-sha256")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--positive-reprojection-px", type=float, default=2.0)
    parser.add_argument("--ransac-reprojection-px", type=float)
    parser.add_argument("--minimum-alpha", type=float, default=0.05)
    parser.add_argument("--depth-absolute-m", type=float, default=0.50)
    parser.add_argument("--depth-relative", type=float, default=0.10)
    parser.add_argument("--maximum-relative-depth-spread", type=float, default=0.25)
    parser.add_argument("--minimum-local-valid-fraction", type=float, default=0.50)
    parser.add_argument("--bundle-target-translation-cm", type=float, default=5.0)
    parser.add_argument("--bundle-target-rotation-deg", type=float, default=5.0)
    parser.add_argument("--near-boundary-multiplier", type=float, default=1.5)
    parser.add_argument("--maximum-minimal-candidates", type=int, default=24)
    parser.add_argument("--maximum-minimal-set-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--prefix-initial-set-size", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if args.cpu_threads < 1:
        raise ValueError("V21 CPU thread count must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("V21 shard index/count is invalid")
    torch.set_num_threads(int(args.cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))

    map_path = args.map.resolve()
    map_sha = _require_sha(map_path, args.expected_map_sha256, "frozen map")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if len(args.adaptation_cache) != len(args.expected_adaptation_cache_sha256):
        raise ValueError("V21 adaptation cache paths and expected SHAs do not align")
    cache_entries = []
    for index, (raw_path, expected_sha) in enumerate(
        zip(args.adaptation_cache, args.expected_adaptation_cache_sha256)
    ):
        cache_path = raw_path.resolve()
        cache_sha = _require_sha(
            cache_path, expected_sha, f"adaptation cache {index}"
        )
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_cache_payload(cache)
        role = cache.get("role", cache.get("view_role"))
        if not (
            cache.get("schema") == CACHE_SCHEMA
            and cache.get("version") == CACHE_VERSION
            and cache.get("protocol") == "test_adapted"
            and cache.get("uses_test_queries") is True
            and cache.get("test_adapted") is True
            and role == "adaptation"
            and cache.get("training_consumers_allowed") is True
            and cache.get("loo_used", False) is False
            and isinstance(cache.get("records"), list)
        ):
            raise ValueError(
                "V21 oracle requires the formal adaptation cache contract"
            )
        binding = _map_binding(cache)
        stable_source = _source_identity(
            cache.get("inputs", {}).get("stable_map"), label="stable map"
        )
        if binding != map_sha or stable_source != (str(map_path), map_sha):
            raise ValueError("V21 adaptation cache is not bound to the frozen map")
        cache_entries.append((cache_path, cache_sha, cache))
    cache_entries = _validate_complete_cache_shards(cache_entries)
    cache = cache_entries[0][2]
    common_cache_contract = (
        cache.get("frontend_contract"),
        cache.get("preprocessing_config_sha256"),
        cache.get("baseline_contract"),
        int(cache.get("anchor_count", -1)),
        int(cache.get("descriptor_dim", -1)),
    )
    for _, _, value in cache_entries[1:]:
        if (
            value.get("frontend_contract"),
            value.get("preprocessing_config_sha256"),
            value.get("baseline_contract"),
            int(value.get("anchor_count", -1)),
            int(value.get("descriptor_dim", -1)),
        ) != common_cache_contract:
            raise ValueError("V21 adaptation cache shard contracts differ")
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("V21 oracle requires a materialized frozen Anchor map")
    xyz = torch.as_tensor(state.get("anchor_xyz")).float().cpu()
    features = torch.as_tensor(state.get("anchor_features")).float().cpu()
    anchor_ids = torch.as_tensor(state.get("anchor_ids")).long().cpu().reshape(-1)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or features.ndim != 2
        or features.shape[0] != xyz.shape[0]
        or anchor_ids.shape != (xyz.shape[0],)
        or torch.unique(anchor_ids).numel() != anchor_ids.numel()
        or not bool(torch.isfinite(xyz).all())
        or not bool(torch.isfinite(features).all())
        or int(cache["anchor_count"]) != int(xyz.shape[0])
        or int(cache["descriptor_dim"]) != int(features.shape[1])
    ):
        raise ValueError("V21 frozen map registry is invalid")
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(xyz.shape[0]))
    ).long().cpu().reshape(-1)
    if equivalence.shape != (xyz.shape[0],) or bool((equivalence < 0).any()):
        raise ValueError("V21 frozen map equivalence registry is invalid")
    contract = cache.get("baseline_contract", {})
    if not (
        isinstance(contract, dict)
        and contract.get("matching")
        == "exact_global_cosine_top1_lower_anchor_row_tie_break"
        and contract.get("pose_solver") == "single_standard_poselib_absolute_pose"
        and float(contract.get("pixel_center_offset", math.nan)) == 0.5
        and float(contract.get("confidence", math.nan)) == 0.99999
        and int(contract.get("maximum_iterations", -1)) == 100000
        and int(contract.get("minimum_iterations", -1)) == 1000
    ):
        raise ValueError("V21 oracle parameters differ from the cached exact plant")
    ransac_reprojection_px, plant_seed = _resolve_baseline_plant(
        contract,
        requested_reprojection_px=args.ransac_reprojection_px,
        requested_seed=args.seed,
    )

    source_records = _ordered_source_records(cache_entries)
    if len(args.gaussian_support) != len(args.expected_gaussian_support_sha256):
        raise ValueError("V21 Gaussian support paths and expected SHAs do not align")
    support_entries = []
    for index, (raw_path, expected_sha) in enumerate(
        zip(args.gaussian_support, args.expected_gaussian_support_sha256)
    ):
        support_path = raw_path.resolve()
        support_sha = _require_sha(
            support_path, expected_sha, f"Gaussian support {index}"
        )
        support = torch.load(support_path, map_location="cpu", weights_only=False)
        support_entries.append((support_path, support_sha, support))
    support_by_query = _join_gaussian_support(
        support_entries=support_entries,
        cache_entries=cache_entries,
        source_records=source_records,
        map_path=map_path,
        map_sha=map_sha,
    )
    truth_supplied = args.correspondence_truth is not None
    truth_sha_supplied = args.expected_correspondence_truth_sha256 is not None
    if truth_supplied != truth_sha_supplied:
        raise ValueError("V21 correspondence truth path and expected SHA do not align")
    if args.positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
        if not truth_supplied:
            raise ValueError("V21 Track diagnostic source requires correspondence truth")
        truth_path = args.correspondence_truth.resolve()
        truth_sha = _require_sha(
            truth_path,
            args.expected_correspondence_truth_sha256,
            "correspondence truth",
        )
        truth_payload = torch.load(
            truth_path, map_location="cpu", weights_only=False
        )
        truth_entry = (truth_path, truth_sha, truth_payload)
        truth_by_query, truth_payload = _join_correspondence_truth(
            truth_entry=truth_entry,
            cache_entries=cache_entries,
            support_entries=support_entries,
            source_records=source_records,
            map_path=map_path,
            map_sha=map_sha,
            anchor_count=int(xyz.shape[0]),
        )
    else:
        if truth_supplied:
            raise ValueError("V21 Gaussian geometry source forbids correspondence truth")
        truth_entry = None
        truth_by_query = {}
        truth_payload = None

    seen_queries: set[int] = set()
    selected = []
    for source_index, (cache_index, local_index, record) in enumerate(source_records):
        if not isinstance(record, dict):
            raise ValueError("V21 adaptation cache record is not a mapping")
        query_index = int(_field(record, "query_index"))
        if query_index in seen_queries:
            raise ValueError("V21 adaptation cache repeats a query index")
        seen_queries.add(query_index)
        if source_index % args.shard_count != args.shard_index:
            continue
        keypoints = torch.as_tensor(
            _field(record, "keypoints", "physical_keypoints")
        ).float() + float(contract["pixel_center_offset"])
        descriptors = _field(record, "descriptors", "native_descriptors")
        intrinsic = _field(record, "intrinsics", "intrinsic")
        pose_w2c = _field(record, "pose_w2c", "ground_truth_pose_w2c")
        winners = _field(
            record, "winner_anchor_rows", "baseline_winner_anchor_rows"
        )
        winner_rows = torch.as_tensor(winners).long().reshape(-1)
        winner_ids = torch.as_tensor(_field(record, "winner_anchor_ids")).long()
        if winner_ids.shape != winner_rows.shape or not torch.equal(
            winner_ids, anchor_ids[winner_rows]
        ):
            raise ValueError("V21 cached winner Anchor IDs differ from the map")
        query = torch.as_tensor(descriptors).float().cpu()
        if query.ndim != 2 or query.shape[1] != features.shape[1]:
            raise ValueError("V21 adaptation descriptor dimension differs from the map")
        winner_scores = _field(
            record, "winner_scores", "baseline_winner_scores", required=False
        )
        if winner_scores is not None:
            score = torch.as_tensor(winner_scores).float().reshape(-1)
            if score.shape != (query.shape[0],) or not bool(torch.isfinite(score).all()):
                raise ValueError("V21 cached winner scores do not align")
        topk = _field(
            record,
            "candidate_anchor_rows",
            "topk_candidate_anchor_rows",
            required=False,
        )
        topk_scores = _field(
            record, "candidate_scores", "topk_candidate_scores", required=False
        )
        if topk_scores is not None:
            if topk is None or torch.as_tensor(topk_scores).shape != torch.as_tensor(
                topk
            ).shape:
                raise ValueError("V21 cached Top-K scores do not align")
            if not bool(torch.isfinite(torch.as_tensor(topk_scores).float()).all()):
                raise ValueError("V21 cached Top-K scores are non-finite")
        if args.positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            positive_source_kwargs = {
                "track_consensus_record": truth_by_query[query_index]
            }
        else:
            positive_source_kwargs = {
                **_support_geometry_kwargs(support_by_query.get(query_index)),
                "anchor_visibility_mask": _field(
                    record, "anchor_visibility_mask", required=False
                ),
            }
        result = analyze_pose_recovery_query(
            query_index=query_index,
            keypoints=keypoints,
            winner_anchor_rows=winners,
            anchor_xyz=xyz,
            intrinsic=intrinsic,
            ground_truth_w2c=pose_w2c,
            equivalence_class_ids=equivalence,
            query_descriptors=query,
            anchor_features=features,
            topk_candidate_anchor_rows=topk,
            positive_source=args.positive_source,
            row_valid_mask=_field(
                record,
                "native_valid_keypoint_mask",
                "row_valid_mask",
                "row_valid",
                required=False,
            ),
            positive_reprojection_px=args.positive_reprojection_px,
            minimum_alpha=args.minimum_alpha,
            depth_absolute_m=args.depth_absolute_m,
            depth_relative=args.depth_relative,
            maximum_relative_depth_spread=args.maximum_relative_depth_spread,
            minimum_local_valid_fraction=args.minimum_local_valid_fraction,
            bundle_target_translation_cm=args.bundle_target_translation_cm,
            bundle_target_rotation_deg=args.bundle_target_rotation_deg,
            ransac_reprojection_px=ransac_reprojection_px,
            near_boundary_multiplier=args.near_boundary_multiplier,
            maximum_minimal_candidates=args.maximum_minimal_candidates,
            maximum_minimal_set_size=args.maximum_minimal_set_size,
            beam_width=args.beam_width,
            prefix_initial_set_size=args.prefix_initial_set_size,
            seed=plant_seed,
            **positive_source_kwargs,
        )
        _validate_cached_baseline(_optional_baseline(record), result["baseline"])
        result["source_cache_index"] = int(cache_index)
        result["source_record_index"] = int(local_index)
        result["image_name"] = str(record.get("image_name", ""))
        result["sequence_id"] = str(record.get("sequence_id", ""))
        result["block_id"] = str(record.get("block_id", ""))
        selected.append(result)
        print(
            f"V21 pose oracle shard {args.shard_index}: "
            f"{len(selected)} queries complete",
            flush=True,
        )

    output = {
        "schema": OUTPUT_SCHEMA,
        "version": 1,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "role": "adaptation",
        "positive_source": args.positive_source,
        "loo_used": False,
        "ground_truth_pose_is_feedback_authority": True,
        "topk_is_candidate_mining_only": True,
        "all_pose_recovery_claims_use_exact_poselib": True,
        "exact_poselib_is_controller_action_authority": False,
        "unlabeled_rows_are_negative": False,
        "negative_anchor_label_count": 0,
        "gaussian_support_is_geometry_only": True,
        "correspondence_identity_authority_present": False,
        "track_consensus_identity_evidence_present": (
            args.positive_source == TRACK_CONSENSUS_DIAGNOSTIC
        ),
        "track_consensus_identity_evidence_is_deployment_authority": False,
        "track_consensus_diagnostic_is_action_authority": False,
        "controller_authorized_query_count_must_be_zero": True,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "source_query_count": len(source_records),
        "input": {
            "adaptation_caches": [
                {"path": str(path), "sha256": digest}
                for path, digest, _ in cache_entries
            ],
            "gaussian_support": [
                {"path": str(path), "sha256": digest}
                for path, digest, _ in support_entries
            ],
            "correspondence_truth": (
                {"path": str(truth_entry[0]), "sha256": truth_entry[1]}
                if truth_entry is not None
                else None
            ),
            "frozen_map": str(map_path),
            "frozen_map_sha256": map_sha,
        },
        "parameters": {
            "positive_source_parameter_usage": (
                {
                    "direct_diagnostic_csr": True,
                    "cli_gaussian_geometry_thresholds_used": False,
                    "cli_reprojection_threshold_used": False,
                    "truth_gaussian_geometry_gates": dict(
                        truth_payload["gaussian_geometry_gates"]
                    ),
                }
                if truth_payload is not None
                else {
                    "direct_diagnostic_csr": False,
                    "cli_gaussian_geometry_thresholds_used": bool(
                        support_entries
                    ),
                    "cli_reprojection_threshold_used": True,
                    "truth_gaussian_geometry_gates": None,
                }
            ),
            "positive_reprojection_px": float(args.positive_reprojection_px),
            "ransac_reprojection_px": float(ransac_reprojection_px),
            "minimum_alpha": float(args.minimum_alpha),
            "depth_absolute_m": float(args.depth_absolute_m),
            "depth_relative": float(args.depth_relative),
            "maximum_relative_depth_spread": float(
                args.maximum_relative_depth_spread
            ),
            "minimum_local_valid_fraction": float(
                args.minimum_local_valid_fraction
            ),
            "bundle_target_translation_cm": float(
                args.bundle_target_translation_cm
            ),
            "bundle_target_rotation_deg": float(
                args.bundle_target_rotation_deg
            ),
            "standard_r5_reporting_translation_cm": 5.0,
            "standard_r5_reporting_rotation_deg": 5.0,
            "bundle_target_changes_standard_r5_definition": False,
            "near_boundary_multiplier": float(args.near_boundary_multiplier),
            "maximum_minimal_candidates": int(args.maximum_minimal_candidates),
            "maximum_minimal_set_size": int(args.maximum_minimal_set_size),
            "beam_width": int(args.beam_width),
            "prefix_initial_set_size": int(args.prefix_initial_set_size),
            "seed": int(plant_seed),
            "track_consensus_diagnostic": (
                {
                    "tier_name": str(
                        truth_payload["teacher_action_decision"]["tier_name"]
                    ),
                    "requested_action": str(
                        truth_payload["teacher_action_decision"][
                            "requested_action"
                        ]
                    ),
                    "source_action_authorized": bool(
                        truth_payload["action_authorized"]
                    ),
                    "oracle_action_authorized": False,
                }
                if truth_payload is not None
                else None
            ),
        },
        "records": selected,
        "summary": summarize_pose_recovery(
            selected, positive_source=args.positive_source
        ),
    }
    immutable_entries = [*cache_entries, *support_entries]
    if truth_entry is not None:
        immutable_entries.append(truth_entry)
    for path, digest, _ in immutable_entries:
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("V21 oracle input changed while processing")
    if not map_path.is_file() or sha256_file(map_path) != map_sha:
        raise RuntimeError("V21 frozen map changed while processing")
    _validate_output_payload(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(output, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        _validate_output_payload(reloaded)
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 oracle output appeared while running: {output_path}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
