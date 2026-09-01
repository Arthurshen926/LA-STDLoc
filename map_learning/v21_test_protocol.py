"""Prospective real-test split contracts for V21 test-targeted adaptation.

The splitter is deliberately metadata-only.  Roles are derived from the
ordered test camera registry, sequence names, and frame indices; localization
errors and candidate outputs are not accepted by this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from common.hashing import canonical_json, sha256_file


SCHEMA = "lafgs_v21_test_adaptation_split"
VERSION = 1
PRIMARY_ROLES = ("adaptation", "control", "confirmation")
ROLE_RATIOS = {
    "adaptation": 0.45,
    "control": 0.20,
    "confirmation": 0.35,
}
ORDERING_POLICY = "forward_adaptation_control_confirmation"
_FRAME_PATTERN = re.compile(r"(\d+)(?!.*\d)")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_json(value: dict[str, Any] | list[Any]) -> str:
    return hashlib.sha256(canonical_json({"value": value}).encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return _SHA256_PATTERN.fullmatch(str(value)) is not None


def _pose_sha256(pose_w2c: Any) -> str:
    pose = np.asarray(pose_w2c, dtype="<f4")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("V21 requires a finite 4x4 GT pose for every test camera")
    digest = hashlib.sha256(b"pose_w2c:<f4:4x4:")
    digest.update(np.ascontiguousarray(pose).tobytes())
    return digest.hexdigest()


def _sequence_and_frame(image_name: str) -> tuple[str, int]:
    normalized = str(image_name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if len(path.parts) < 2:
        raise ValueError("V21 test image names must contain a sequence directory")
    match = _FRAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError("V21 test image names must expose a numeric frame index")
    return str(PurePosixPath(*path.parts[:-1])), int(match.group(1))


def _role_counts(block_count: int) -> dict[str, int]:
    if block_count < len(PRIMARY_ROLES):
        raise ValueError("each V21 test sequence requires at least three blocks")
    raw = {role: ROLE_RATIOS[role] * block_count for role in PRIMARY_ROLES}
    counts = {role: int(math.floor(raw[role])) for role in PRIMARY_ROLES}
    remaining = block_count - sum(counts.values())
    order = sorted(
        PRIMARY_ROLES,
        key=lambda role: (raw[role] - counts[role], -PRIMARY_ROLES.index(role)),
        reverse=True,
    )
    for role in order[:remaining]:
        counts[role] += 1
    for role in PRIMARY_ROLES:
        if counts[role] > 0:
            continue
        donor = max(PRIMARY_ROLES, key=lambda candidate: counts[candidate])
        if counts[donor] <= 1:
            raise ValueError("V21 cannot give every role sequence coverage")
        counts[donor] -= 1
        counts[role] += 1
    return counts


def _ordered_registry_sha(records: Sequence[dict[str, Any]]) -> str:
    identity = [
        {
            "query_index": int(record["query_index"]),
            "image_name": str(record["image_name"]),
            "image_path": str(record["image_path"]),
            "image_sha256": record["image_sha256"],
            "pose_w2c_sha256": str(record["pose_w2c_sha256"]),
        }
        for record in sorted(records, key=lambda item: int(item["query_index"]))
    ]
    return _sha256_json(identity)


def _base_records(
    cameras: Sequence[Any], *, require_image_content: bool
) -> list[dict[str, Any]]:
    records = []
    names = set()
    sequence_frames: set[tuple[str, int]] = set()
    for query_index, camera in enumerate(cameras):
        image_name = str(camera.image_name).replace("\\", "/")
        if image_name in names:
            raise ValueError("V21 test camera image names must be unique")
        names.add(image_name)
        sequence_id, frame_index = _sequence_and_frame(image_name)
        if (sequence_id, frame_index) in sequence_frames:
            raise ValueError("V21 frame indices must be unique within each sequence")
        sequence_frames.add((sequence_id, frame_index))
        image_path = Path(camera.image_path).expanduser().resolve()
        image_available = image_path.is_file()
        if require_image_content and not image_available:
            raise FileNotFoundError(f"missing V21 test image: {image_path}")
        records.append(
            {
                "query_index": int(query_index),
                "image_name": image_name,
                "image_path": str(image_path),
                "image_content_available": bool(image_available),
                "image_sha256": sha256_file(image_path) if image_available else None,
                "sequence_id": sequence_id,
                "frame_index": int(frame_index),
                "pose_w2c_sha256": _pose_sha256(camera.pose_w2c),
            }
        )
    if not records:
        raise ValueError("V21 requires a nonempty test camera registry")
    return records


def _sequence_blocks(
    sequence_id: str,
    records: list[dict[str, Any]],
    *,
    block_size: int,
    embargo_frames: int,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: (item["frame_index"], item["image_name"]))
    for sequence_index, record in enumerate(ordered):
        record["sequence_query_index"] = int(sequence_index)
    blocks = [
        {
            "block_id": f"{sequence_id}:trajectory_{index:04d}",
            "sequence_id": sequence_id,
            "role": None,
            "records": ordered[start : start + block_size],
        }
        for index, start in enumerate(range(0, len(ordered), block_size))
    ]
    counts = _role_counts(len(blocks))
    cursor = 0
    for role in PRIMARY_ROLES:
        for block in blocks[cursor : cursor + counts[role]]:
            block["role"] = role
        cursor += counts[role]
    if cursor != len(blocks) or any(block["role"] is None for block in blocks):
        raise RuntimeError("V21 internal block allocation failed")

    embargo = []
    boundary_index = 0
    if embargo_frames:
        for index in range(len(blocks) - 1):
            left, right = blocks[index], blocks[index + 1]
            if left["role"] == right["role"]:
                continue
            if (
                len(left["records"]) <= embargo_frames
                or len(right["records"]) <= embargo_frames
            ):
                raise ValueError("V21 embargo would empty a trajectory block")
            removed = [
                *left["records"][-embargo_frames:],
                *right["records"][:embargo_frames],
            ]
            left["records"] = left["records"][:-embargo_frames]
            right["records"] = right["records"][embargo_frames:]
            embargo.append(
                {
                    "block_id": f"{sequence_id}:embargo_{boundary_index:04d}",
                    "sequence_id": sequence_id,
                    "role": "embargo",
                    "records": removed,
                }
            )
            boundary_index += 1
    return [*blocks, *embargo]


def _block_summary(block: dict[str, Any]) -> dict[str, Any]:
    records = sorted(block["records"], key=lambda item: item["sequence_query_index"])
    if not records:
        raise ValueError("V21 trajectory blocks must be nonempty")
    for block_index, record in enumerate(records):
        record["within_block_index"] = int(block_index)
        record["block_id"] = str(block["block_id"])
        record["role"] = str(block["role"])
    return {
        "block_id": str(block["block_id"]),
        "sequence_id": str(block["sequence_id"]),
        "role": str(block["role"]),
        "query_count": len(records),
        "query_indices": [int(record["query_index"]) for record in records],
        "first_frame_index": int(records[0]["frame_index"]),
        "last_frame_index": int(records[-1]["frame_index"]),
    }


def _role_summary(
    role: str, records: Sequence[dict[str, Any]], blocks: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    selected = [record for record in records if record["role"] == role]
    role_blocks = [block for block in blocks if block["role"] == role]
    return {
        "query_count": len(selected),
        "block_count": len(role_blocks),
        "query_indices": sorted(int(record["query_index"]) for record in selected),
        "block_ids": sorted(str(block["block_id"]) for block in role_blocks),
        "image_names_sha256": _sha256_json(
            [record["image_name"] for record in sorted(selected, key=lambda row: row["query_index"])]
        ),
    }


def build_test_protocol_manifest(
    cameras: Sequence[Any],
    *,
    dataset_root: str | Path,
    images: str,
    dataset_registry_path: str | Path,
    stable_map_path: str | Path,
    block_size: int = 10,
    embargo_frames: int = 1,
    minimum_confirmation_queries: int = 160,
    minimum_confirmation_blocks: int = 15,
    require_image_content: bool = True,
) -> dict[str, Any]:
    """Build a deterministic V21 A/C/F manifest without outcome inputs."""
    if int(block_size) < 3:
        raise ValueError("V21 block size must be at least three frames")
    if not 0 <= int(embargo_frames) < int(block_size) // 2:
        raise ValueError("V21 embargo must lie in [0, block_size/2)")
    if min(int(minimum_confirmation_queries), int(minimum_confirmation_blocks)) < 1:
        raise ValueError("V21 confirmation minimums must be positive")
    dataset_root = Path(dataset_root).expanduser().resolve()
    registry_path = Path(dataset_registry_path).expanduser().resolve()
    map_path = Path(stable_map_path).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"missing V21 dataset test registry: {registry_path}")
    if not map_path.is_file():
        raise FileNotFoundError(f"missing V21 stable map: {map_path}")

    records = _base_records(list(cameras), require_image_content=require_image_content)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sequence[record["sequence_id"]].append(record)
    allocated_blocks = []
    for sequence_id in sorted(by_sequence):
        allocated_blocks.extend(
            _sequence_blocks(
                sequence_id,
                by_sequence[sequence_id],
                block_size=int(block_size),
                embargo_frames=int(embargo_frames),
            )
        )
    blocks = [_block_summary(block) for block in allocated_blocks]
    records.sort(key=lambda item: int(item["query_index"]))
    roles = {
        role: _role_summary(role, records, blocks)
        for role in (*PRIMARY_ROLES, "embargo")
    }
    primary_count = sum(roles[role]["query_count"] for role in PRIMARY_ROLES)
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "test_adapted": True,
        "uses_test_queries": True,
        "split_uses_localization_errors": False,
        "split_uses_candidate_outputs": False,
        "split_policy": "sequence_temporal_indivisible_blocks",
        "ordering_policy": ORDERING_POLICY,
        "block_size_target": int(block_size),
        "embargo_frames_per_role_boundary_side": int(embargo_frames),
        "requested_role_ratios": dict(ROLE_RATIOS),
        "achieved_role_ratios": {
            role: roles[role]["query_count"] / primary_count
            for role in PRIMARY_ROLES
        },
        "minimum_confirmation_queries": int(minimum_confirmation_queries),
        "minimum_confirmation_blocks": int(minimum_confirmation_blocks),
        "stable_map": str(map_path),
        "stable_map_sha256": sha256_file(map_path),
        "dataset_registry": {
            "dataset_root": str(dataset_root),
            "images": str(images),
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "ordered_test_camera_sha256": _ordered_registry_sha(records),
        },
        "counts": {
            "test_queries": len(records),
            "sequences": len(by_sequence),
            "blocks": len(blocks),
            "primary_role_queries": primary_count,
            "embargo_queries": roles["embargo"]["query_count"],
        },
        "roles": roles,
        "blocks": sorted(
            blocks,
            key=lambda block: (
                block["sequence_id"],
                block["first_frame_index"],
                block["block_id"],
            ),
        ),
        "records": records,
    }
    validate_test_protocol_manifest(manifest)
    return manifest


def validate_test_protocol_manifest(manifest: dict[str, Any]) -> None:
    """Validate split identity, exclusivity, coverage, and confirmation size."""
    if not (
        manifest.get("schema") == SCHEMA
        and manifest.get("version") == VERSION
        and manifest.get("protocol") == "test_adapted"
        and manifest.get("test_adapted") is True
        and manifest.get("uses_test_queries") is True
        and manifest.get("split_uses_localization_errors") is False
        and manifest.get("split_uses_candidate_outputs") is False
        and manifest.get("ordering_policy") == ORDERING_POLICY
        and _is_sha256(manifest.get("stable_map_sha256"))
    ):
        raise ValueError("invalid V21 test-adapted manifest header")
    block_size = int(manifest.get("block_size_target", 0))
    embargo_frames = int(
        manifest.get("embargo_frames_per_role_boundary_side", -1)
    )
    if block_size < 3 or not 0 <= embargo_frames < block_size // 2:
        raise ValueError("invalid V21 trajectory block or embargo policy")
    if manifest.get("requested_role_ratios") != ROLE_RATIOS:
        raise ValueError("V21 requested role ratios differ from the frozen policy")
    registry = manifest.get("dataset_registry", {})
    if not (
        isinstance(registry, dict)
        and registry.get("dataset_root")
        and registry.get("images")
        and registry.get("path")
        and _is_sha256(registry.get("sha256"))
        and _is_sha256(registry.get("ordered_test_camera_sha256"))
    ):
        raise ValueError("invalid V21 dataset registry binding")
    records = list(manifest.get("records", []))
    blocks = list(manifest.get("blocks", []))
    if not records or not blocks:
        raise ValueError("V21 split manifest must contain records and blocks")
    indices = [int(record["query_index"]) for record in records]
    names = [str(record["image_name"]) for record in records]
    if sorted(indices) != list(range(len(records))) or len(set(names)) != len(names):
        raise ValueError("V21 query registry is incomplete or duplicated")
    required = {
        "query_index",
        "image_name",
        "image_path",
        "image_sha256",
        "sequence_id",
        "frame_index",
        "pose_w2c_sha256",
        "sequence_query_index",
        "block_id",
        "role",
        "within_block_index",
        "image_content_available",
    }
    if any(not required.issubset(record) for record in records):
        raise ValueError("V21 split record fields are incomplete")
    allowed_roles = {*PRIMARY_ROLES, "embargo"}
    if any(record["role"] not in allowed_roles for record in records):
        raise ValueError("V21 split record has an unknown role")
    if any(not _is_sha256(record["pose_w2c_sha256"]) for record in records):
        raise ValueError("V21 GT pose hashes are invalid")
    if any(
        record["image_sha256"] is not None
        and not _is_sha256(record["image_sha256"])
        for record in records
    ):
        raise ValueError("V21 image content hashes are invalid")
    if any(
        bool(record["image_content_available"])
        != (record["image_sha256"] is not None)
        for record in records
    ):
        raise ValueError("V21 image availability and content hashes differ")
    for record in records:
        sequence_id, frame_index = _sequence_and_frame(record["image_name"])
        if (
            record["sequence_id"] != sequence_id
            or int(record["frame_index"]) != frame_index
        ):
            raise ValueError("V21 sequence/frame metadata differs from the image name")

    block_by_id = {str(block["block_id"]): block for block in blocks}
    if len(block_by_id) != len(blocks):
        raise ValueError("V21 block IDs must be unique")
    block_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        block_members[str(record["block_id"])].append(record)
    if set(block_members) != set(block_by_id):
        raise ValueError("V21 records and block registry differ")
    for block_id, members in block_members.items():
        block = block_by_id[block_id]
        if (
            {record["role"] for record in members} != {block["role"]}
            or {record["sequence_id"] for record in members}
            != {block["sequence_id"]}
            or sorted(record["query_index"] for record in members)
            != sorted(int(value) for value in block["query_indices"])
            or len(members) != int(block["query_count"])
        ):
            raise ValueError("V21 trajectory block is not indivisible")
        chronological = sorted(members, key=lambda row: row["within_block_index"])
        frames = [int(record["frame_index"]) for record in chronological]
        within = [int(record["within_block_index"]) for record in chronological]
        if (
            frames != sorted(frames)
            or len(set(frames)) != len(frames)
            or within != list(range(len(members)))
            or int(block["first_frame_index"]) != frames[0]
            or int(block["last_frame_index"]) != frames[-1]
        ):
            raise ValueError("V21 trajectory block is not chronological")

    rebuilt_records = [dict(record) for record in records]
    rebuilt_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rebuilt_records:
        rebuilt_by_sequence[str(record["sequence_id"])].append(record)
    rebuilt_allocations = []
    for sequence_id in sorted(rebuilt_by_sequence):
        rebuilt_allocations.extend(
            _sequence_blocks(
                sequence_id,
                rebuilt_by_sequence[sequence_id],
                block_size=block_size,
                embargo_frames=embargo_frames,
            )
        )
    rebuilt_blocks = [_block_summary(block) for block in rebuilt_allocations]
    rebuilt_blocks.sort(
        key=lambda block: (
            block["sequence_id"],
            block["first_frame_index"],
            block["block_id"],
        )
    )
    actual_blocks = sorted(
        blocks,
        key=lambda block: (
            block["sequence_id"],
            block["first_frame_index"],
            block["block_id"],
        ),
    )
    if actual_blocks != rebuilt_blocks:
        raise ValueError("V21 blocks differ from the metadata-only frozen allocation")
    assignment_fields = (
        "query_index",
        "sequence_query_index",
        "block_id",
        "role",
        "within_block_index",
    )
    actual_assignment = sorted(
        tuple(record[field] for field in assignment_fields) for record in records
    )
    rebuilt_assignment = sorted(
        tuple(record[field] for field in assignment_fields)
        for record in rebuilt_records
    )
    if actual_assignment != rebuilt_assignment:
        raise ValueError("V21 records differ from the metadata-only frozen allocation")

    role_sets = {
        role: {int(record["query_index"]) for record in records if record["role"] == role}
        for role in PRIMARY_ROLES
    }
    if any(
        role_sets[left] & role_sets[right]
        for position, left in enumerate(PRIMARY_ROLES)
        for right in PRIMARY_ROLES[position + 1 :]
    ):
        raise ValueError("V21 adaptation/control/confirmation roles overlap")
    sequences = {str(record["sequence_id"]) for record in records}
    for sequence in sequences:
        present = {
            record["role"]
            for record in records
            if record["sequence_id"] == sequence
            and record["role"] in PRIMARY_ROLES
        }
        if present != set(PRIMARY_ROLES):
            raise ValueError("every V21 test sequence must cover all three roles")
    expected_roles = {
        role: _role_summary(role, records, blocks)
        for role in (*PRIMARY_ROLES, "embargo")
    }
    if manifest.get("roles") != expected_roles:
        raise ValueError("V21 role summaries do not bind the record table")
    primary_count = sum(
        expected_roles[role]["query_count"] for role in PRIMARY_ROLES
    )
    expected_ratios = {
        role: expected_roles[role]["query_count"] / primary_count
        for role in PRIMARY_ROLES
    }
    if manifest.get("achieved_role_ratios") != expected_ratios:
        raise ValueError("V21 achieved role ratios do not bind the record table")
    confirmation = expected_roles["confirmation"]
    if (
        confirmation["query_count"]
        < int(manifest["minimum_confirmation_queries"])
        or confirmation["block_count"]
        < int(manifest["minimum_confirmation_blocks"])
    ):
        raise ValueError("V21 confirmation split is below its frozen minimum")
    if registry.get("ordered_test_camera_sha256") != _ordered_registry_sha(records):
        raise ValueError("V21 ordered dataset registry hash differs")
    expected_counts = {
        "test_queries": len(records),
        "sequences": len(sequences),
        "blocks": len(blocks),
        "primary_role_queries": sum(len(values) for values in role_sets.values()),
        "embargo_queries": len(records)
        - sum(len(values) for values in role_sets.values()),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("V21 manifest counts do not bind the record table")


__all__ = [
    "ORDERING_POLICY",
    "PRIMARY_ROLES",
    "ROLE_RATIOS",
    "SCHEMA",
    "VERSION",
    "build_test_protocol_manifest",
    "validate_test_protocol_manifest",
]
