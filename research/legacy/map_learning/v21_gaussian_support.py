"""Contracts and raster sampling for V21 Gaussian geometry support.

This artifact is deliberately weaker than correspondence truth.  A Gaussian
render supplies depth/alpha evidence that may reject implausible projected
Anchor candidates, but it does not prove an Anchor identity and it never
creates a negative label.  Ground-truth test poses are delayed feedback
authority, so only the V21 adaptation role may be materialized or consumed by
training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
import re
import uuid

import torch

from common.hashing import canonical_json
from map_learning.v21_test_cache import tensor_sha256, validate_shard_registry


SCHEMA = "lafgs_v21_gaussian_geometry_support_cache"
VERSION = 1
ROLE = "adaptation"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EVIDENCE_SEMANTICS = {
    "authority": "delayed_ground_truth_test_pose_feedback",
    "scope": "adaptation_role_only",
    "identity_claim": "none",
    "negative_label_claim": "none",
    "allowed_use": "geometric_candidate_filter_and_oracle_diagnosis",
    "deployment_authority": False,
    "sample_coordinates": "floor(raw_native_keypoint_plus_pixel_center_offset)",
    "depth_discontinuity": "relative_3x3_positive_finite_depth_range_over_center_depth",
}


def sha256_json(value: Mapping) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _source_record(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 Gaussian {label} source record is missing")
    path = str(value.get("path", ""))
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA256")
    size = int(value.get("size_bytes", 0))
    if not path or size <= 0:
        raise ValueError(f"V21 Gaussian {label} source record is empty")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _plane(value: object, *, image_hw: Sequence[int], label: str) -> torch.Tensor:
    plane = torch.as_tensor(value).detach().float().squeeze()
    expected = (int(image_hw[0]), int(image_hw[1]))
    if plane.ndim != 2 or tuple(plane.shape) != expected:
        raise ValueError(f"V21 Gaussian {label} raster differs from image_hw")
    return plane


@torch.inference_mode()
def sample_keypoint_raster_support(
    *,
    keypoints: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    image_hw: Sequence[int],
    pixel_center_offset: float = 0.5,
) -> dict:
    """Sample only sparse rows plus a 3x3 depth-stability diagnostic.

    Values outside the raster or without finite positive centre depth are kept
    as NaN and marked invalid.  Local invalid neighbours do not become zero
    depth; their fraction is recorded independently.
    """

    xy = torch.as_tensor(keypoints).detach().float().cpu()
    if xy.ndim != 2 or xy.shape[1] != 2 or not bool(torch.isfinite(xy).all()):
        raise ValueError("V21 Gaussian keypoints must be finite [N,2]")
    height, width = map(int, image_hw)
    if height <= 0 or width <= 0:
        raise ValueError("V21 Gaussian image dimensions must be positive")
    depth_plane = _plane(depth, image_hw=(height, width), label="depth").cpu()
    alpha_plane = _plane(alpha, image_hw=(height, width), label="alpha").cpu()
    offset = float(pixel_center_offset)
    if not math.isfinite(offset):
        raise ValueError("V21 Gaussian pixel-centre offset must be finite")

    pixel = torch.floor(xy + offset).long()
    inside = (
        (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
    )
    safe = pixel.clone()
    safe[:, 0].clamp_(0, width - 1)
    safe[:, 1].clamp_(0, height - 1)
    center_depth = depth_plane[safe[:, 1], safe[:, 0]]
    center_alpha = alpha_plane[safe[:, 1], safe[:, 0]]

    local_samples = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x = (safe[:, 0] + dx).clamp(0, width - 1)
            y = (safe[:, 1] + dy).clamp(0, height - 1)
            local_samples.append(depth_plane[y, x])
    local = torch.stack(local_samples, dim=1)
    local_valid = torch.isfinite(local) & (local > 0.0) & inside[:, None]
    local_minimum = local.masked_fill(~local_valid, float("inf")).amin(1)
    local_maximum = local.masked_fill(~local_valid, -float("inf")).amax(1)
    local_fraction = local_valid.float().mean(1)
    discontinuity = (local_maximum - local_minimum) / center_depth.abs().clamp_min(
        1e-8
    )
    valid = (
        inside
        & torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_alpha)
        & (center_alpha >= 0.0)
        & (center_alpha <= 1.0 + 1e-4)
        & local_valid.any(1)
        & torch.isfinite(discontinuity)
    )
    sampled_depth = center_depth.clone()
    sampled_alpha = center_alpha.clone()
    sampled_depth[~inside] = float("nan")
    sampled_alpha[~inside] = float("nan")
    discontinuity[~valid] = float("inf")
    local_fraction[~inside] = 0.0
    return {
        "gaussian_depth_at_keypoints": sampled_depth.contiguous(),
        "gaussian_alpha_at_keypoints": sampled_alpha.contiguous(),
        "gaussian_relative_depth_spread_3x3": discontinuity.contiguous(),
        "gaussian_local_valid_fraction_3x3": local_fraction.contiguous(),
        "gaussian_support_valid": valid.contiguous(),
        "sample_pixel_xy": pixel.contiguous(),
    }


def build_support_record(
    *,
    frontend_record: Mapping,
    frontend_cache_path: str | Path,
    frontend_cache_sha256: str,
    frontend_shard_index: int,
    sampled: Mapping,
    pixel_center_offset: float,
) -> dict:
    """Bind sampled support to one immutable frontend row registry."""

    if str(frontend_record.get("role")) != ROLE:
        raise ValueError("V21 Gaussian support only accepts adaptation records")
    keypoints = torch.as_tensor(frontend_record.get("keypoints")).float().cpu()
    image_hw = torch.as_tensor(frontend_record.get("image_hw")).long().cpu()
    if keypoints.ndim != 2 or keypoints.shape[1] != 2 or image_hw.shape != (2,):
        raise ValueError("V21 Gaussian frontend geometry is malformed")
    count = int(keypoints.shape[0])
    output = {
        "query_index": int(frontend_record["query_index"]),
        "image_name": str(frontend_record["image_name"]),
        "image_sha256": str(frontend_record["image_sha256"]),
        "sequence_id": str(frontend_record["sequence_id"]),
        "frame_index": int(frontend_record["frame_index"]),
        "block_id": str(frontend_record["block_id"]),
        "role": ROLE,
        "source_record_sha256": str(frontend_record["source_record_sha256"]),
        "pose_w2c_sha256": str(frontend_record["pose_w2c_sha256"]),
        "keypoint_count": count,
        "keypoints_sha256": tensor_sha256(keypoints),
        "intrinsics_sha256": tensor_sha256(frontend_record["intrinsics"]),
        "image_hw": image_hw.contiguous(),
        "frontend_cache_path": str(Path(frontend_cache_path).expanduser().resolve()),
        "frontend_cache_sha256": _require_sha256(
            frontend_cache_sha256, label="frontend cache SHA256"
        ),
        "frontend_shard_index": int(frontend_shard_index),
        "pixel_center_offset": float(pixel_center_offset),
    }
    for field in (
        "gaussian_depth_at_keypoints",
        "gaussian_alpha_at_keypoints",
        "gaussian_relative_depth_spread_3x3",
        "gaussian_local_valid_fraction_3x3",
        "gaussian_support_valid",
        "sample_pixel_xy",
    ):
        output[field] = torch.as_tensor(sampled[field]).detach().cpu().contiguous()
    validate_support_record(output)
    return output


def validate_support_record(record: Mapping) -> None:
    count = int(record.get("keypoint_count", -1))
    if (
        count < 0
        or int(record.get("query_index", -1)) < 0
        or str(record.get("role")) != ROLE
        or not str(record.get("image_name", ""))
        or not str(record.get("sequence_id", ""))
        or not str(record.get("block_id", ""))
    ):
        raise ValueError("V21 Gaussian support record identity is invalid")
    for name in (
        "image_sha256",
        "source_record_sha256",
        "pose_w2c_sha256",
        "keypoints_sha256",
        "intrinsics_sha256",
        "frontend_cache_sha256",
    ):
        _require_sha256(record.get(name), label=f"support {name}")
    if not str(record.get("frontend_cache_path", "")):
        raise ValueError("V21 Gaussian frontend-cache path is missing")
    if int(record.get("frontend_shard_index", -1)) < 0:
        raise ValueError("V21 Gaussian frontend shard index is invalid")
    if not math.isfinite(float(record.get("pixel_center_offset", math.nan))):
        raise ValueError("V21 Gaussian pixel-centre offset is invalid")
    image_hw = torch.as_tensor(record.get("image_hw"))
    depth = torch.as_tensor(record.get("gaussian_depth_at_keypoints"))
    alpha = torch.as_tensor(record.get("gaussian_alpha_at_keypoints"))
    spread = torch.as_tensor(record.get("gaussian_relative_depth_spread_3x3"))
    fraction = torch.as_tensor(record.get("gaussian_local_valid_fraction_3x3"))
    valid = torch.as_tensor(record.get("gaussian_support_valid"))
    pixels = torch.as_tensor(record.get("sample_pixel_xy"))
    if (
        image_hw.shape != (2,)
        or depth.shape != (count,)
        or alpha.shape != (count,)
        or spread.shape != (count,)
        or fraction.shape != (count,)
        or valid.shape != (count,)
        or valid.dtype != torch.bool
        or pixels.shape != (count, 2)
    ):
        raise ValueError("V21 Gaussian sampled columns do not align")
    if bool(valid.any()):
        rows = valid
        if not bool(
            torch.isfinite(depth[rows]).all()
            and (depth[rows] > 0).all()
            and torch.isfinite(alpha[rows]).all()
            and (alpha[rows] >= 0).all()
            and (alpha[rows] <= 1.0 + 1e-4).all()
            and torch.isfinite(spread[rows]).all()
            and (spread[rows] >= 0).all()
        ):
            raise ValueError("V21 Gaussian valid rows contain invalid evidence")
    if not bool(
        torch.isfinite(fraction).all()
        and (fraction >= 0).all()
        and (fraction <= 1).all()
    ):
        raise ValueError("V21 Gaussian local-valid fractions are invalid")


def validate_support_payload(payload: Mapping) -> None:
    if (
        payload.get("schema") != SCHEMA
        or payload.get("version") != VERSION
        or payload.get("protocol") != "test_adapted"
        or payload.get("uses_test_queries") is not True
        or payload.get("test_adapted") is not True
        or payload.get("role") != ROLE
        or payload.get("training_consumers_allowed") is not True
        or payload.get("ground_truth_pose_is_delayed_feedback_authority") is not True
        or payload.get("control_or_confirmation_forbidden") is not True
        or payload.get("correspondence_truth_claimed") is not False
        or payload.get("negative_labels_created") is not False
        or payload.get("deployment_authority") is not False
        or payload.get("evidence_semantics") != EVIDENCE_SEMANTICS
    ):
        raise ValueError("unsupported V21 Gaussian support contract")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 Gaussian input lineage is missing")
    split = _source_record(inputs.get("split_manifest"), label="split manifest")
    stable_map = _source_record(inputs.get("stable_map"), label="stable map")
    gaussian = _source_record(inputs.get("gaussian_ply"), label="Gaussian PLY")
    caches = inputs.get("frontend_caches")
    producers = inputs.get("producer_sources")
    if not isinstance(caches, list) or not caches:
        raise ValueError("V21 Gaussian frontend-cache lineage is empty")
    if not isinstance(producers, list) or not producers:
        raise ValueError("V21 Gaussian producer lineage is empty")
    normalized_caches = [_source_record(value, label="frontend cache") for value in caches]
    [_source_record(value, label="producer") for value in producers]
    if len({value["path"] for value in normalized_caches}) != len(normalized_caches):
        raise ValueError("V21 Gaussian frontend-cache lineage is duplicated")
    if (
        payload.get("split_manifest_sha256") != split["sha256"]
        or payload.get("stable_map_sha256") != stable_map["sha256"]
        or payload.get("gaussian_ply_sha256") != gaussian["sha256"]
    ):
        raise ValueError("V21 Gaussian primary SHA lineage differs")
    render = payload.get("render_contract")
    if not isinstance(render, Mapping) or payload.get(
        "render_contract_sha256"
    ) != sha256_json(render):
        raise ValueError("V21 Gaussian render contract hash is invalid")
    if (
        render.get("gaussian_type") != "2dgs"
        or render.get("requested_rasterize_mode") != "antialiased"
        or render.get("effective_rasterize_mode")
        != "omitted_unsupported_by_2dgs_wrapper"
        or render.get("rasterize_mode_argument_forwarded") is not False
        or "rasterize_mode" in render
    ):
        raise ValueError("V21 Gaussian effective 2DGS rasterization contract differs")
    registry = payload.get("frontend_shard_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("V21 Gaussian frontend shard registry is missing")
    validate_shard_registry(registry)
    registry_sha = _require_sha256(
        registry.get("registry_sha256"), label="frontend shard registry SHA256"
    )
    if payload.get("frontend_shard_registry_sha256") != registry_sha:
        raise ValueError("V21 Gaussian frontend registry binding differs")
    shard_count = int(registry.get("shard_count", 0))
    source_shards = payload.get("source_frontend_shards")
    if (
        shard_count <= 0
        or not isinstance(source_shards, list)
        or sorted(int(value["shard_index"]) for value in source_shards)
        != list(range(shard_count))
        or len(source_shards) != len(normalized_caches)
    ):
        raise ValueError("V21 Gaussian frontend shard coverage is incomplete")
    source_by_sha = {value["sha256"]: value for value in normalized_caches}
    if len(source_by_sha) != len(normalized_caches):
        raise ValueError("V21 Gaussian frontend cache contents are duplicated")
    for shard in source_shards:
        digest = _require_sha256(shard.get("sha256"), label="source shard SHA256")
        if digest not in source_by_sha or source_by_sha[digest]["path"] != shard.get(
            "path"
        ):
            raise ValueError("V21 Gaussian source shard lineage differs")
    records = payload.get("records")
    if (
        int(payload.get("gaussian_primitive_count", 0)) <= 0
        or not isinstance(records, list)
        or int(payload.get("query_count", -1)) != len(records)
    ):
        raise ValueError("V21 Gaussian support records are missing")
    rows = registry.get("rows")
    if not isinstance(rows, list) or len(rows) != len(records):
        raise ValueError("V21 Gaussian support does not cover the full registry")
    expected = sorted(rows, key=lambda row: int(row["ordinal"]))
    queries: set[int] = set()
    cache_by_shard = {
        int(value["shard_index"]): value for value in source_shards
    }
    for record, row in zip(records, expected):
        validate_support_record(record)
        query = int(record["query_index"])
        shard = int(row["shard_index"])
        source = cache_by_shard.get(shard)
        if (
            query in queries
            or query != int(row["query_index"])
            or record["image_name"] != row["image_name"]
            or record["image_sha256"] != row["image_sha256"]
            or record["source_record_sha256"] != row["source_record_sha256"]
            or int(record["frontend_shard_index"]) != shard
            or source is None
            or record["frontend_cache_sha256"] != source["sha256"]
            or record["frontend_cache_path"] != source["path"]
        ):
            raise ValueError("V21 Gaussian support identity/lineage differs")
        queries.add(query)


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 Gaussian support output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_support_payload(reloaded)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 Gaussian support output appeared while running: {output}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "EVIDENCE_SEMANTICS",
    "ROLE",
    "SCHEMA",
    "VERSION",
    "atomic_torch_save_fresh",
    "build_support_record",
    "sample_keypoint_raster_support",
    "sha256_json",
    "validate_support_payload",
    "validate_support_record",
]
