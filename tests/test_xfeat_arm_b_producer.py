from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from map_learning.frontend_upper_bound import (
    file_sha256,
    validate_probe,
)
from map_learning.xfeat_arm_b import (
    XFeatArtifactSpec,
    native_to_xfeat_coordinates,
    sample_xfeat_descriptor_field,
    xfeat_resize_contract,
)
import scripts.materialize_xfeat_arm_b as producer_cli


MODEL_SOURCE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        pooled = F.avg_pool2d(gray, kernel_size=8, stride=8)
        channels = torch.arange(1, 65, dtype=x.dtype, device=x.device)[None, :, None, None]
        features = (pooled + channels / 64.0) * self.scale
        logits = torch.zeros((x.shape[0], 65, *pooled.shape[-2:]), dtype=x.dtype, device=x.device)
        reliability = torch.ones((x.shape[0], 1, *pooled.shape[-2:]), dtype=x.dtype, device=x.device)
        return features, logits, reliability
"""


INTERPOLATOR_SOURCE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolateSparse2d(nn.Module):
    def __init__(self, mode='bicubic', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        return 2.0 * (x / torch.tensor([W-1, H-1], device=x.device, dtype=x.dtype)) - 1.0

    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        value = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return value.permute(0, 2, 3, 1).squeeze(-2)
"""


class OfficialInterpolator(torch.nn.Module):
    def forward(self, value, positions, H, W):
        scale = positions.new_tensor([W - 1, H - 1])
        grid = (2.0 * positions / scale - 1.0).unsqueeze(-2)
        sampled = F.grid_sample(
            value, grid, mode="bicubic", align_corners=False
        )
        return sampled.permute(0, 2, 3, 1).squeeze(-2)


def _run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_external_xfeat(root: Path) -> dict:
    model = root / "encoders/XFeat/modules/model.py"
    interpolator = root / "encoders/XFeat/modules/interpolator.py"
    wrapper = root / "encoders/XFeat/modules/xfeat.py"
    weights = root / "encoders/XFeat/weights/xfeat.pt"
    lighterglue = root / "encoders/XFeat/weights/xfeat-lighterglue.pt"
    license_path = root / "encoders/XFeat/LICENSE"
    model.parent.mkdir(parents=True)
    weights.parent.mkdir(parents=True)
    model.write_text(MODEL_SOURCE, encoding="utf-8")
    interpolator.write_text(INTERPOLATOR_SOURCE, encoding="utf-8")
    wrapper.write_text("# locked sparse XFeat wrapper\n", encoding="utf-8")
    license_path.write_text("synthetic Apache-2.0 fixture\n", encoding="utf-8")
    torch.save({"scale": torch.tensor(1.0)}, weights)
    lighterglue.write_bytes(b"locked pair matcher")
    _run_git(root, "init", "-q")
    _run_git(root, "add", "encoders/XFeat")
    _run_git(
        root,
        "-c",
        "user.name=Arm B Test",
        "-c",
        "user.email=arm-b@example.invalid",
        "commit",
        "-q",
        "-m",
        "locked fake XFeat",
    )
    return {
        "worktree": root,
        "weights": weights,
        "lighterglue": lighterglue,
        "weights_sha": file_sha256(weights),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "tree": _run_git(root, "rev-parse", "HEAD:encoders/XFeat"),
        "model_sha": file_sha256(model),
        "interpolator_sha": file_sha256(interpolator),
        "wrapper_sha": file_sha256(wrapper),
    }


def _write_dataset(root: Path) -> tuple[list[str], str, torch.Tensor]:
    height, width = 33, 65
    mapping_names = ["mapping-000.png", "mapping-001.png"]
    test_name = "test-000.png"
    images = root / "processed"
    images.mkdir(parents=True)
    y, x = np.mgrid[:height, :width]
    for index, name in enumerate([*mapping_names, test_name]):
        rgb = np.stack(
            [
                (x * 3 + index * 7) % 256,
                (y * 5 + index * 11) % 256,
                ((x + y) * 2 + index * 13) % 256,
            ],
            axis=2,
        ).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(images / name)
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 65 33 50 50 32 16\n", encoding="utf-8"
    )
    rows = []
    for index, name in enumerate([*mapping_names, test_name], start=1):
        rows.append(f"{index} 1 0 0 0 0 0 0 1 {name}\n\n")
    (sparse / "images.txt").write_text("".join(rows), encoding="utf-8")
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (sparse / "list_test.txt").write_text(test_name + "\n", encoding="utf-8")
    valid = torch.ones((height, width), dtype=torch.bool)
    valid[0, 0] = False
    masks = {
        name: [valid.numpy(), np.ones_like(valid), np.ones_like(valid)]
        for name in [*mapping_names, test_name]
    }
    with (root / "masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    return mapping_names, test_name, valid


def _signature_payload(dataset: Path) -> dict:
    return {
        "version": 11,
        "query_feature_contract": "native_resized_input",
        "feature_resize_mode": "resize_image_then_native_stride8",
        "descriptor_source": "superpoint_native_dense_resized_input",
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "pixel_center_offset": 0.5,
        "valid_mask_policy": "object_and_sky_and_distortion_v1",
        "model_path": str((dataset / "frozen-prior").resolve()),
        "rgb_prior_fingerprint": {"synthetic": True},
        "source_path": str(dataset.resolve()),
        "load_iteration": 1,
        "feature_type": "sp",
        "images": "processed",
        "resolution": 1,
        "longest_edge": 0,
        "white_background": True,
        "norm_before_render": True,
        "native_sparse_enabled": True,
        "native_sparse_keypoint_count": 4,
        "native_sparse_nms_radius": 4,
        "native_sparse_coordinate_convention": (
            "superpoint_grid_index_then_pnp_plus_half_v1"
        ),
    }


def _make_reference(
    root: Path,
    mapping_names: list[str],
    valid_mask: torch.Tensor,
) -> tuple[Path, Path, dict, dict]:
    keypoints = torch.tensor(
        [[1.0, 1.0], [32.0, 16.0], [63.0, 31.0]], dtype=torch.float32
    )
    payload = _signature_payload(root)
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    queries = {}
    records = []
    for index, name in enumerate(mapping_names):
        queries[name] = {
            "native_keypoints": keypoints.clone(),
            "native_descriptors": torch.ones((3, 256), dtype=torch.float16),
            "native_scores": torch.tensor([0.9, 0.8, 0.7]),
            "native_valid_mask": valid_mask.clone(),
            "native_input_hw": [33, 65],
            "pixel_center_offset": 0.5,
            "native_sparse_metadata": {
                "detect_and_compute": True,
                "detect_num": 4,
                "requested_keypoint_count": 4,
                "nms_radius": 4,
                "keypoint_count_before_mask": 4,
                "keypoint_count_after_mask": 3,
                "coordinate_convention": (
                    "superpoint_grid_index_then_pnp_plus_half_v1"
                ),
            },
        }
        records.append(
            {
                "query_index": index,
                "query_name": name,
                "query_rows": torch.arange(3),
                "positive_offsets": torch.tensor([0, 1, 2, 3]),
                "positive_indices": torch.tensor([0, 1, 2]),
                "ambiguous_offsets": torch.tensor([0, 0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        )
    query_cache = {
        "version": 3,
        "signature": signature,
        "signature_payload": payload,
        "queries": queries,
    }
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": 3,
        "query_names": list(mapping_names),
        "records": records,
        "config": {},
    }
    query_cache_path = root / "query_cache.pt"
    teacher_path = root / "teacher.pt"
    torch.save(query_cache, query_cache_path)
    torch.save(teacher, teacher_path)
    return query_cache_path, teacher_path, query_cache, teacher


@pytest.fixture
def arm_b_bundle(tmp_path: Path) -> dict:
    dataset = tmp_path / "dataset"
    mapping_names, test_name, valid_mask = _write_dataset(dataset)
    query_cache, teacher, query_payload, teacher_payload = _make_reference(
        dataset, mapping_names, valid_mask
    )
    artifact = _make_external_xfeat(tmp_path / "external")
    return {
        "dataset": dataset,
        "mapping_names": mapping_names,
        "test_name": test_name,
        "query_cache": query_cache,
        "teacher": teacher,
        "query_payload": query_payload,
        "teacher_payload": teacher_payload,
        "artifact": artifact,
    }


def _cli_arguments(bundle: dict, output: Path) -> list[str]:
    artifact = bundle["artifact"]
    return [
        "--dataset",
        str(bundle["dataset"]),
        "--query-cache",
        str(bundle["query_cache"]),
        "--expected-query-cache-sha256",
        file_sha256(bundle["query_cache"]),
        "--teacher",
        str(bundle["teacher"]),
        "--expected-teacher-sha256",
        file_sha256(bundle["teacher"]),
        "--xfeat-worktree",
        str(artifact["worktree"]),
        "--weights",
        str(artifact["weights"]),
        "--expected-weights-sha256",
        artifact["weights_sha"],
        "--expected-parent-commit",
        artifact["commit"],
        "--expected-xfeat-tree",
        artifact["tree"],
        "--expected-model-sha256",
        artifact["model_sha"],
        "--expected-interpolator-sha256",
        artifact["interpolator_sha"],
        "--expected-wrapper-sha256",
        artifact["wrapper_sha"],
        "--device",
        "cpu",
        "--output",
        str(output),
    ]


def test_official_bicubic_sampling_preserves_locked_geometry():
    contract = xfeat_resize_contract((33, 65))
    native = torch.tensor(
        [[0.0, 0.0], [32.0, 16.0], [64.0, 32.0], [12.25, 20.75]]
    )
    transformed = native_to_xfeat_coordinates(native, contract)
    scale = transformed.new_tensor([contract["rw"], contract["rh"]])
    assert torch.allclose(transformed * scale, native)
    dense = torch.arange(1, 1 + 64 * 4 * 8, dtype=torch.float32).reshape(
        1, 64, 4, 8
    )
    dense = F.normalize(dense, dim=1)
    sampled = sample_xfeat_descriptor_field(
        dense,
        native,
        contract=contract,
        interpolator=OfficialInterpolator(),
    )
    grid_scale = transformed.new_tensor([63.0, 31.0])
    grid = (2.0 * transformed / grid_scale - 1.0)[None, :, None]
    expected = F.grid_sample(
        dense, grid, mode="bicubic", align_corners=False
    ).permute(0, 2, 3, 1)[0, :, 0]
    expected = F.normalize(expected, dim=1)
    assert sampled.shape == (4, 64)
    assert torch.allclose(sampled, expected)
    assert torch.allclose(torch.linalg.norm(sampled, dim=1), torch.ones(4))


def test_cli_materializes_consumer_valid_cpu_arm_b(
    arm_b_bundle: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "xfeat_arm_b.pt"
    producer_cli.main(_cli_arguments(arm_b_bundle, output))
    probe = torch.load(output, map_location="cpu", weights_only=False)
    assert probe["mapping_only"] is True
    assert probe["uses_test_queries"] is False
    assert probe["capabilities"] == {
        "detector_repeatability": False,
        "descriptor_identity": True,
    }
    assert probe["producer"]["device"] == "cpu"
    assert probe["producer"]["gpu_used"] is False
    assert probe["producer"]["candidate_detector_used"] is False
    assert set(probe["queries"]) == set(arm_b_bundle["mapping_names"])
    assert arm_b_bundle["test_name"] not in probe["queries"]
    for name in arm_b_bundle["mapping_names"]:
        record = probe["queries"][name]
        descriptor = record["descriptor_at_reference_keypoints"]
        assert descriptor.shape == (3, 64)
        assert descriptor.dtype == torch.float32
        assert torch.allclose(
            torch.linalg.norm(descriptor, dim=1), torch.ones(3), atol=1e-5
        )
        assert record["coordinate_lineage"]["native_input_hw"] == [33, 65]
        assert record["coordinate_lineage"]["xfeat_input_hw"] == [32, 64]
        assert record["reference_row_indices"].tolist() == [0, 1, 2]
    validated = validate_probe(
        probe,
        arm_b_bundle["query_payload"],
        arm_b_bundle["teacher_payload"],
        require_descriptor=True,
    )
    assert validated["candidate_descriptor_dim"] == 64
    assert validated["validated_descriptor_rows"] == 6
    assert _run_git(
        arm_b_bundle["artifact"]["worktree"],
        "status",
        "--porcelain",
        "--untracked-files=all",
    ) == ""


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--expected-weights-sha256", "weights SHA256 mismatch"),
        ("--expected-model-sha256", "model SHA256 mismatch"),
    ],
)
def test_cli_rejects_artifact_hash_mismatch(
    arm_b_bundle: dict,
    tmp_path: Path,
    flag: str,
    message: str,
):
    output = tmp_path / f"bad-{flag[11:]}.pt"
    arguments = _cli_arguments(arm_b_bundle, output)
    arguments[arguments.index(flag) + 1] = "0" * 64
    with pytest.raises(ValueError, match=message):
        producer_cli.main(arguments)
    assert not output.exists()


def test_cli_rejects_dirty_external_worktree(
    arm_b_bundle: dict, tmp_path: Path
):
    output = tmp_path / "dirty.pt"
    (arm_b_bundle["artifact"]["worktree"] / "untracked.txt").write_text(
        "dirty", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not clean"):
        producer_cli.main(_cli_arguments(arm_b_bundle, output))
    assert not output.exists()


def test_cli_rejects_reference_artifact_hash_mismatch(
    arm_b_bundle: dict, tmp_path: Path
):
    output = tmp_path / "wrong-reference-sha.pt"
    arguments = _cli_arguments(arm_b_bundle, output)
    flag = "--expected-query-cache-sha256"
    arguments[arguments.index(flag) + 1] = "0" * 64
    with pytest.raises(ValueError, match="query-cache SHA256 mismatch"):
        producer_cli.main(arguments)
    assert not output.exists()


def test_cli_rejects_cache_signature_and_non_mapping_query_set(
    arm_b_bundle: dict, tmp_path: Path
):
    bad_signature = dict(arm_b_bundle["query_payload"])
    bad_signature["signature"] = "0" * 64
    bad_signature_path = tmp_path / "bad-signature.pt"
    torch.save(bad_signature, bad_signature_path)
    arguments = _cli_arguments(arm_b_bundle, tmp_path / "bad-output.pt")
    arguments[arguments.index("--query-cache") + 1] = str(bad_signature_path)
    arguments[arguments.index("--expected-query-cache-sha256") + 1] = (
        file_sha256(bad_signature_path)
    )
    with pytest.raises(ValueError, match="signature does not match"):
        producer_cli.main(arguments)

    extra = dict(arm_b_bundle["query_payload"])
    extra["queries"] = dict(extra["queries"])
    first = arm_b_bundle["mapping_names"][0]
    extra["queries"][arm_b_bundle["test_name"]] = dict(extra["queries"][first])
    extra_path = tmp_path / "test-query-cache.pt"
    torch.save(extra, extra_path)
    arguments[arguments.index("--query-cache") + 1] = str(extra_path)
    arguments[arguments.index("--expected-query-cache-sha256") + 1] = (
        file_sha256(extra_path)
    )
    with pytest.raises(ValueError, match="exact mapping-only query set"):
        producer_cli.main(arguments)


def test_consumer_rejects_corrupted_reference_row_hash(
    arm_b_bundle: dict, tmp_path: Path
):
    output = tmp_path / "valid.pt"
    producer_cli.main(_cli_arguments(arm_b_bundle, output))
    probe = torch.load(output, map_location="cpu", weights_only=False)
    name = arm_b_bundle["mapping_names"][0]
    probe["queries"][name]["reference_keypoints_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry mismatch"):
        validate_probe(
            probe,
            arm_b_bundle["query_payload"],
            arm_b_bundle["teacher_payload"],
            require_descriptor=True,
        )


def test_consumer_binds_exact_serialized_reference_artifacts(
    arm_b_bundle: dict, tmp_path: Path
):
    output = tmp_path / "source-bound.pt"
    producer_cli.main(_cli_arguments(arm_b_bundle, output))
    probe = torch.load(output, map_location="cpu", weights_only=False)
    query_path = arm_b_bundle["query_cache"]
    teacher_path = arm_b_bundle["teacher"]

    copied_query = tmp_path / "copied-query-cache.pt"
    copied_query.write_bytes(query_path.read_bytes())
    with pytest.raises(ValueError, match="query_cache path mismatch"):
        validate_probe(
            probe,
            arm_b_bundle["query_payload"],
            arm_b_bundle["teacher_payload"],
            require_descriptor=True,
            query_cache_path=copied_query,
            teacher_path=teacher_path,
        )

    original_query_bytes = query_path.read_bytes()
    changed_query = dict(arm_b_bundle["query_payload"])
    changed_query["same_signature_different_serialization"] = True
    torch.save(changed_query, query_path)
    with pytest.raises(ValueError, match="query_cache SHA256 mismatch"):
        validate_probe(
            probe,
            changed_query,
            arm_b_bundle["teacher_payload"],
            require_descriptor=True,
            query_cache_path=query_path,
            teacher_path=teacher_path,
        )
    query_path.write_bytes(original_query_bytes)

    changed_teacher = dict(arm_b_bundle["teacher_payload"])
    changed_teacher["same_schema_different_records"] = True
    torch.save(changed_teacher, teacher_path)
    with pytest.raises(ValueError, match="teacher SHA256 mismatch"):
        validate_probe(
            probe,
            arm_b_bundle["query_payload"],
            changed_teacher,
            require_descriptor=True,
            query_cache_path=query_path,
            teacher_path=teacher_path,
        )


def test_artifact_spec_rejects_lighterglue_checkpoint(
    arm_b_bundle: dict,
):
    artifact = arm_b_bundle["artifact"]
    lighterglue = artifact["lighterglue"]
    spec = XFeatArtifactSpec(
        worktree=artifact["worktree"],
        weights=lighterglue,
        expected_weights_sha256=file_sha256(lighterglue),
        expected_parent_commit=artifact["commit"],
        expected_xfeat_tree=artifact["tree"],
        expected_model_sha256=artifact["model_sha"],
        expected_interpolator_sha256=artifact["interpolator_sha"],
        expected_wrapper_sha256=artifact["wrapper_sha"],
    )
    from map_learning.xfeat_arm_b import validate_xfeat_artifact

    with pytest.raises(ValueError, match="accepts only"):
        validate_xfeat_artifact(spec)
