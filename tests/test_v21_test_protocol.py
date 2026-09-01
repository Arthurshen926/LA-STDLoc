from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.hashing import sha256_file
from map_learning.v21_test_protocol import (
    PRIMARY_ROLES,
    build_test_protocol_manifest,
    validate_test_protocol_manifest,
)
from scripts import materialize_v21_test_splits as cli


def _cameras(
    dataset_root: Path, *, sequence_count: int = 3, frames_per_sequence: int = 60
) -> list[SimpleNamespace]:
    cameras = []
    for sequence in range(sequence_count):
        for frame in range(frames_per_sequence):
            image_name = f"seq{sequence + 1}/frame{frame:06d}.png"
            image_path = dataset_root / "processed" / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"image:{sequence}:{frame}".encode())
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(sequence)
            pose[1, 3] = float(frame) / 10.0
            cameras.append(
                SimpleNamespace(
                    image_name=image_name,
                    image_path=image_path,
                    pose_w2c=pose,
                )
            )
    return cameras


def _inputs(
    tmp_path: Path, *, sequence_count: int = 3, frames_per_sequence: int = 60
) -> tuple[list[SimpleNamespace], Path, Path]:
    cameras = _cameras(
        tmp_path,
        sequence_count=sequence_count,
        frames_per_sequence=frames_per_sequence,
    )
    registry = tmp_path / "dataset_test.txt"
    registry.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        + "\n".join(
            f"{camera.image_name} 0 0 0 1 0 0 0" for camera in cameras
        )
        + "\n"
    )
    stable_map = tmp_path / "stable_map.pt"
    stable_map.write_bytes(b"frozen stable map")
    return cameras, registry, stable_map


def _build(
    tmp_path: Path,
    *,
    embargo_frames: int = 0,
    frames_per_sequence: int = 60,
) -> dict:
    cameras, registry, stable_map = _inputs(
        tmp_path, frames_per_sequence=frames_per_sequence
    )
    return build_test_protocol_manifest(
        cameras,
        dataset_root=tmp_path,
        images="processed",
        dataset_registry_path=registry,
        stable_map_path=stable_map,
        block_size=10,
        embargo_frames=embargo_frames,
        minimum_confirmation_queries=30,
        minimum_confirmation_blocks=3,
    )


def test_build_is_deterministic_and_binds_all_test_evidence(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    repeated = _build(tmp_path)

    assert manifest == repeated
    assert manifest["schema"] == "lafgs_v21_test_adaptation_split"
    assert manifest["protocol"] == "test_adapted"
    assert manifest["test_adapted"] is True
    assert manifest["uses_test_queries"] is True
    assert manifest["split_uses_localization_errors"] is False
    assert manifest["split_uses_candidate_outputs"] is False
    assert manifest["ordering_policy"] == "forward_adaptation_control_confirmation"
    assert len(manifest["stable_map_sha256"]) == 64
    assert len(manifest["dataset_registry"]["ordered_test_camera_sha256"]) == 64
    assert manifest["counts"]["test_queries"] == 180
    assert manifest["counts"]["embargo_queries"] == 0
    assert {record["query_index"] for record in manifest["records"]} == set(
        range(180)
    )

    for record in manifest["records"]:
        assert len(record["pose_w2c_sha256"]) == 64
        assert record["image_sha256"] == sha256_file(record["image_path"])
    for sequence in {record["sequence_id"] for record in manifest["records"]}:
        sequence_roles = {
            record["role"]
            for record in manifest["records"]
            if record["sequence_id"] == sequence
        }
        assert sequence_roles == set(PRIMARY_ROLES)
        temporal_roles = [
            block["role"]
            for block in sorted(
                (
                    block
                    for block in manifest["blocks"]
                    if block["sequence_id"] == sequence
                    and block["role"] in PRIMARY_ROLES
                ),
                key=lambda block: block["first_frame_index"],
            )
        ]
        compressed_roles = [
            role
            for index, role in enumerate(temporal_roles)
            if index == 0 or temporal_roles[index - 1] != role
        ]
        assert compressed_roles == list(PRIMARY_ROLES)
    for block in manifest["blocks"]:
        members = [
            record
            for record in manifest["records"]
            if record["block_id"] == block["block_id"]
        ]
        assert {record["role"] for record in members} == {block["role"]}
        assert {record["sequence_id"] for record in members} == {
            block["sequence_id"]
        }


def test_cambridge_registry_parser_ignores_non_image_headers(tmp_path: Path) -> None:
    cameras, registry, _ = _inputs(
        tmp_path, sequence_count=1, frames_per_sequence=3
    )

    parsed_path, names = cli._test_registry(tmp_path)

    assert parsed_path == registry.resolve()
    assert names == [camera.image_name for camera in cameras]


def test_embargo_is_explicit_and_preserves_complete_query_registry(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path, embargo_frames=1, frames_per_sequence=90)

    assert manifest["embargo_frames_per_role_boundary_side"] == 1
    assert manifest["counts"]["test_queries"] == 270
    assert manifest["counts"]["embargo_queries"] == 12
    assert manifest["roles"]["embargo"]["query_count"] == 12
    assert manifest["roles"]["embargo"]["block_count"] == 6
    assert sum(role["query_count"] for role in manifest["roles"].values()) == 270
    assert len({record["query_index"] for record in manifest["records"]}) == 270
    assert all(
        block["role"] == "embargo"
        for block in manifest["blocks"]
        if block["block_id"] in manifest["roles"]["embargo"]["block_ids"]
    )
    validate_test_protocol_manifest(manifest)


def test_validator_rejects_role_tampering_and_small_confirmation(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path)
    tampered = deepcopy(manifest)
    original_role = tampered["records"][0]["role"]
    tampered["records"][0]["role"] = next(
        role for role in PRIMARY_ROLES if role != original_role
    )
    with pytest.raises(ValueError):
        validate_test_protocol_manifest(tampered)

    cameras, registry, stable_map = _inputs(
        tmp_path / "small", frames_per_sequence=30
    )
    with pytest.raises(ValueError, match="below its frozen minimum"):
        build_test_protocol_manifest(
            cameras,
            dataset_root=tmp_path / "small",
            images="processed",
            dataset_registry_path=registry,
            stable_map_path=stable_map,
            block_size=10,
            embargo_frames=0,
            minimum_confirmation_queries=31,
            minimum_confirmation_blocks=3,
        )


def test_cli_is_fail_closed_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras, _, stable_map = _inputs(tmp_path, frames_per_sequence=30)

    class FakeDataset:
        def __init__(self, root: Path, *, images: str) -> None:
            assert Path(root) == tmp_path
            assert images == "processed"

        def split(self, role: str) -> list[SimpleNamespace]:
            assert role == "test"
            return cameras

    monkeypatch.setattr(cli, "ColmapDataset", FakeDataset)
    output = tmp_path / "evidence" / "v21_split.json"
    args = argparse.Namespace(
        dataset=tmp_path,
        images="processed",
        stable_map=stable_map,
        block_size=10,
        embargo_frames=0,
        minimum_confirmation_queries=30,
        minimum_confirmation_blocks=3,
        output=output,
    )
    manifest = cli.materialize(args)
    on_disk = json.loads(output.read_text())
    assert on_disk == manifest
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        cli.materialize(args)
    assert output.read_bytes() == original


def test_cli_rejects_registry_camera_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras, _, stable_map = _inputs(tmp_path, frames_per_sequence=30)

    class MissingCameraDataset:
        def __init__(self, root: Path, *, images: str) -> None:
            pass

        def split(self, role: str) -> list[SimpleNamespace]:
            return cameras[:-1]

    monkeypatch.setattr(cli, "ColmapDataset", MissingCameraDataset)
    args = argparse.Namespace(
        dataset=tmp_path,
        images="processed",
        stable_map=stable_map,
        block_size=10,
        embargo_frames=0,
        minimum_confirmation_queries=30,
        minimum_confirmation_blocks=3,
        output=tmp_path / "must_not_exist.json",
    )
    with pytest.raises(ValueError, match="differ from the dataset test registry"):
        cli.materialize(args)
    assert not args.output.exists()
