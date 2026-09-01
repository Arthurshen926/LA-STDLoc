from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.hashing import sha256_file
from localization.frontend import SparseFeatures
from localization.matcher import Top1Matches
from map_learning.metric import SharedLowRankMetric
from map_learning.v21_test_cache import (
    CACHE_SCHEMA,
    atomic_torch_save_fresh,
    build_query_record,
    build_shard_registry,
    records_for_shard,
    sha256_json,
    training_consumer_policy,
    validate_cache_payload,
    validate_split_manifest,
)
from map_learning.v21_test_protocol import build_test_protocol_manifest
from scripts import materialize_v21_test_frontend_cache as cli


def _cameras(
    root: Path, *, sequences: int = 3, frames: int = 9
) -> list[SimpleNamespace]:
    cameras = []
    for sequence in range(sequences):
        for frame in range(frames):
            name = f"seq{sequence}/frame{frame:06d}.png"
            path = root / "processed" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"rgb:{sequence}:{frame}".encode())
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(sequence)
            pose[1, 3] = float(frame) / 10.0
            cameras.append(
                SimpleNamespace(
                    image_name=name,
                    image_path=path,
                    pose_w2c=pose,
                    fov_x=1.0,
                    fov_y=0.8,
                    width=5,
                    height=4,
                )
            )
    return cameras


def _manifest(
    tmp_path: Path,
    *,
    stable_map: Path | None = None,
    embargo_frames: int = 0,
) -> tuple[dict, list[SimpleNamespace], Path, Path]:
    cameras = _cameras(tmp_path, frames=15 if embargo_frames else 9)
    registry = tmp_path / "dataset_test.txt"
    registry.write_text("\n".join(camera.image_name for camera in cameras) + "\n")
    if stable_map is None:
        stable_map = tmp_path / "stable_map.pt"
        stable_map.write_bytes(b"stable-map")
    manifest = build_test_protocol_manifest(
        cameras,
        dataset_root=tmp_path,
        images="processed",
        dataset_registry_path=registry,
        stable_map_path=stable_map,
        block_size=5 if embargo_frames else 3,
        embargo_frames=embargo_frames,
        minimum_confirmation_queries=1,
        minimum_confirmation_blocks=1,
    )
    return manifest, cameras, registry, stable_map


def _query_payload(
    manifest: dict, cameras: list[SimpleNamespace], *, role: str = "adaptation"
) -> dict:
    selected = validate_split_manifest(manifest, role=role)
    manifest_sha = "a" * 64
    registry = build_shard_registry(
        selected,
        role=role,
        shard_count=1,
        split_manifest_sha256=manifest_sha,
    )
    split_record = selected[0]
    camera = cameras[int(split_record["query_index"])]
    record = build_query_record(
        split_record=split_record,
        keypoints=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        descriptors=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        scores=torch.tensor([0.8, 0.7]),
        image_hw=(4, 5),
        valid_mask=torch.ones(4, 5, dtype=torch.bool),
        intrinsics=torch.eye(3),
        pose_w2c=torch.from_numpy(camera.pose_w2c),
        winner_anchor_rows=torch.tensor([0, 1]),
        winner_anchor_ids=torch.tensor([10, 11]),
        winner_scores=torch.tensor([0.9, 0.6]),
        baseline_pose_w2c=torch.from_numpy(camera.pose_w2c),
        baseline_inliers=torch.tensor([0, 1]),
        rotation_error_deg=0.0,
        translation_error_cm=0.0,
        task_error=0.0,
    )
    frontend = {
        "keypoint_count": 2,
        "nms_radius": 4,
        "superpoint_weights_sha256": "d" * 64,
        "mainline_config_sha256": "e" * 64,
        "resolved_mainline_config_sha256": "f" * 64,
    }
    source_specs = {
        "split_manifest": ("/split.json", manifest_sha),
        "stable_map": ("/stable_map.pt", "b" * 64),
        "identity_metric": ("/identity.pt", "c" * 64),
        "frontend_weights": ("/superpoint.pth", "d" * 64),
        "mainline_config": ("/config.yaml", "e" * 64),
        "dataset_registry": ("/dataset_test.txt", "1" * 64),
    }
    input_sources = {
        name: {"path": path, "sha256": digest, "size_bytes": 1}
        for name, (path, digest) in source_specs.items()
    }
    input_sources["mainline_config"] = {
        **input_sources["mainline_config"],
        "resolved_sha256": "f" * 64,
    }
    input_sources.update(
        {
            "scene_calibration": None,
            "valid_mask_source": {
                "path": "/masks.pkl",
                "sha256": "2" * 64,
                "size_bytes": 1,
            },
            "all_source_files": [
                {"path": path, "sha256": digest, "size_bytes": 1}
                for path, digest in source_specs.values()
            ]
            + [
                {"path": "/masks.pkl", "sha256": "2" * 64, "size_bytes": 1},
                {
                    "path": split_record["image_path"],
                    "sha256": split_record["image_sha256"],
                    "size_bytes": 1,
                }
            ],
        }
    )
    return {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": role,
        "split_manifest_sha256": manifest_sha,
        "training_consumer_allowed": role == "adaptation",
        "training_consumers_allowed": role == "adaptation",
        "consumer_policy": training_consumer_policy(role),
        "shard_index": 0,
        "shard_count": 1,
        "query_count": 1,
        "role_query_count": len(selected),
        "anchor_count": 3,
        "descriptor_dim": 4,
        "shard_registry": {
            **registry,
            "rows": [registry["rows"][0]],
            "role_query_count": 1,
        },
        "frontend_contract": frontend,
        "preprocessing_config_sha256": sha256_json(frontend),
        "inputs": input_sources,
        "records": [record],
    }


def test_split_role_selection_and_complete_shard_registry(tmp_path: Path) -> None:
    manifest, _, _, _ = _manifest(tmp_path, embargo_frames=1)
    adaptation = validate_split_manifest(manifest, role="adaptation")
    assert adaptation
    assert {record["role"] for record in adaptation} == {"adaptation"}
    assert all(record["block_id"].startswith("seq") for record in adaptation)
    with pytest.raises(ValueError, match="role must be one of"):
        validate_split_manifest(manifest, role="embargo")

    registry = build_shard_registry(
        adaptation,
        role="adaptation",
        shard_count=3,
        split_manifest_sha256="a" * 64,
    )
    emitted = [
        row
        for shard in range(3)
        for row in records_for_shard(registry, shard_index=shard)
    ]
    assert {row["query_index"] for row in emitted} == {
        record["query_index"] for record in adaptation
    }
    assert len(emitted) == len(adaptation)
    assert len({row["source_record_sha256"] for row in emitted}) == len(emitted)


def test_cache_is_aligned_role_gated_and_never_overwritten(tmp_path: Path) -> None:
    manifest, cameras, _, _ = _manifest(tmp_path)
    payload = _query_payload(manifest, cameras)
    # Keep the complete registry contract while selecting its actual shard rows.
    selected = validate_split_manifest(manifest, role="adaptation")
    registry = build_shard_registry(
        selected,
        role="adaptation",
        shard_count=99,
        split_manifest_sha256="a" * 64,
    )
    first_shard = records_for_shard(registry, shard_index=0)
    assert len(first_shard) == 1
    payload["shard_count"] = 99
    payload["role_query_count"] = len(selected)
    payload["shard_registry"] = registry
    payload["records"][0]["query_index"] = first_shard[0]["query_index"]
    payload["records"][0]["image_name"] = first_shard[0]["image_name"]
    payload["records"][0]["source_record_sha256"] = first_shard[0][
        "source_record_sha256"
    ]
    validate_cache_payload(payload)

    output = tmp_path / "cache" / "adaptation_000.pt"
    atomic_torch_save_fresh(payload, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        atomic_torch_save_fresh(payload, output)
    assert output.read_bytes() == original

    tampered = deepcopy(payload)
    tampered["records"][0]["winner_scores"] = torch.tensor([0.9])
    with pytest.raises(ValueError, match="columns do not align"):
        validate_cache_payload(tampered)

    assert training_consumer_policy("adaptation")["training_consumers_allowed"]
    for role in ("control", "confirmation"):
        policy = training_consumer_policy(role)
        assert policy["training_consumers_allowed"] is False
        assert policy["control_or_confirmation_forbidden_for_training"] is True


def test_materializer_uses_exact_localizer_baseline_and_source_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stable_map = tmp_path / "stable_map.pt"
    anchor_ids = torch.tensor([10, 11, 12])
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_ids": anchor_ids,
            "anchor_xyz": torch.tensor(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
            ),
            "anchor_features": torch.eye(4)[:3],
            "photometric_canonicalization_contract": None,
        },
        stable_map,
    )
    manifest, cameras, registry_path, _ = _manifest(
        tmp_path, stable_map=stable_map
    )
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest))
    metric = SharedLowRankMetric(
        descriptor_dim=4, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_path = tmp_path / "identity.pt"
    torch.save(
        {
            "schema": "lafgs_shared_metric_state",
            "version": 1,
            "protocol": "v6_identity_shared_metric",
            "step": 0,
            "metric_config": metric.export_config(),
            "metric_state_dict": metric.state_dict(),
            "landmark_indices": anchor_ids,
            "map_path": str(stable_map.resolve()),
            "map_sha256": sha256_file(stable_map),
            "photometric_canonicalization_contract": None,
        },
        metric_path,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("frozen config source\n")
    weights_path = tmp_path / "superpoint.pth"
    weights_path.write_bytes(b"frozen-superpoint")
    (tmp_path / "masks.pkl").write_bytes(b"frozen-valid-mask-source")

    mapping_camera = SimpleNamespace(width=5, height=4, fov_x=1.0, fov_y=0.8)

    class FakeDataset:
        def __init__(self, root: Path, *, images: str) -> None:
            assert Path(root).resolve() == tmp_path.resolve()
            assert images == "processed"

        def split(self, split: str):
            return cameras if split == "test" else [mapping_camera]

        @staticmethod
        def load_image(camera):
            return torch.ones(3, camera.height, camera.width)

        @staticmethod
        def valid_mask(camera):
            return torch.ones(camera.height, camera.width, dtype=torch.bool)

    captured = []

    class FakeLocalizer:
        def __init__(self, map_path, state_path, **kwargs) -> None:
            assert Path(map_path) == stable_map
            assert Path(state_path) == metric_path
            assert kwargs["keypoint_count"] == 2
            assert kwargs["nms_radius"] == 4
            assert kwargs["assignment_topk"] == 0
            assert kwargs["suppress_duplicate_anchors"] is False
            captured.append(kwargs)
            self.anchor_ids = anchor_ids
            self.anchor_extra_prototype_features = torch.empty(0, 4)
            self.frontend = SimpleNamespace(context_adapter=None)

        def localize(self, image, *, fov_x, fov_y, valid_mask):
            assert image.shape == (3, 4, 5)
            assert valid_mask.shape == (4, 5)
            sparse = SparseFeatures(
                keypoints=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                scores=torch.tensor([0.8, 0.7]),
                descriptors=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                ),
                image_hw=(4, 5),
            )
            matches = Top1Matches(
                keypoint_indices=torch.arange(2),
                anchor_indices=torch.tensor([0, 1]),
                scores=torch.tensor([0.9, 0.6]),
            )
            pose = SimpleNamespace(
                pose_w2c=np.eye(4, dtype=np.float32),
                inliers=np.asarray([0, 1], dtype=np.int64),
            )
            return SimpleNamespace(
                sparse_features=sparse,
                matches=matches,
                pose=pose,
                intrinsic=np.eye(3, dtype=np.float32),
            )

    config = SimpleNamespace(
        values={
            "deployment": {
                "nms": 4,
                "confidence": 0.99999,
                "maximum_iterations": 100000,
                "minimum_iterations": 1000,
            }
        },
        file_sha256=sha256_file(config_path),
        resolved_sha256="a" * 64,
    )
    monkeypatch.setattr(cli, "ColmapDataset", FakeDataset)
    monkeypatch.setattr(cli, "SparseLocalizer", FakeLocalizer)
    monkeypatch.setattr(cli, "load_mainline_config", lambda path: config)
    monkeypatch.setattr(cli, "resolve_keypoint_count", lambda *args: 2)
    monkeypatch.setattr(cli, "resolve_reprojection_error_px", lambda *args: 12.0)
    monkeypatch.setattr(cli, "resolve_superpoint_weights", lambda: weights_path)
    monkeypatch.setattr(cli, "SUPERPOINT_WEIGHT_SHA256", sha256_file(weights_path))
    monkeypatch.setattr(cli, "_dataset_geometry_sources", lambda root: [registry_path])
    monkeypatch.setattr(cli, "pose_error", lambda predicted, ground_truth: (0.0, 0.0))

    output = tmp_path / "cache.pt"
    payload = cli.materialize(
        argparse.Namespace(
            split_manifest=manifest_path,
            role="adaptation",
            dataset=tmp_path,
            images="processed",
            stable_map=stable_map,
            identity_metric=metric_path,
            config=config_path,
            scene_calibration=None,
            device="cpu",
            seed=2026,
            shard_index=0,
            shard_count=1,
            output=output,
        )
    )
    assert captured
    assert output.is_file()
    assert payload["schema"] == CACHE_SCHEMA
    assert payload["protocol"] == "test_adapted"
    assert payload["uses_test_queries"] is True
    assert payload["test_adapted"] is True
    assert payload["training_consumers_allowed"] is True
    assert payload["records"]
    assert payload["records"][0]["baseline_inliers"].tolist() == [0, 1]
    assert payload["records"][0]["baseline_r5"] is True
    assert payload["records"][0]["keypoints"].tolist() == [
        [0.0, 0.0],
        [1.0, 1.0],
    ]
    assert payload["records"][0]["winner_anchor_rows"].tolist() == [0, 1]
    assert "valid_mask" not in payload["records"][0]
    assert len(payload["records"][0]["valid_mask_sha256"]) == 64
    assert payload["baseline_contract"]["pixel_center_offset"] == 0.5
    assert payload["inputs"]["stable_map"]["sha256"] == sha256_file(stable_map)
