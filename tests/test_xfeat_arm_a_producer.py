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

from map_learning.frontend_upper_bound import file_sha256, validate_probe
from map_learning.xfeat_arm_a import (
    _mask_equivalence_proof,
    extract_xfeat_detector,
)
from map_learning.xfeat_arm_b import _load_module, xfeat_resize_contract
import scripts.materialize_xfeat_arm_a as producer_cli


MODEL_SOURCE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        pooled = F.avg_pool2d(x.mean(dim=1, keepdim=True), kernel_size=8, stride=8)
        channels = torch.arange(1, 65, dtype=x.dtype, device=x.device)[None, :, None, None]
        features = (pooled + channels / 64.0) * self.scale
        logits = torch.full((x.shape[0], 65, *pooled.shape[-2:]), -8.0, dtype=x.dtype, device=x.device)
        logits = logits + F.one_hot(torch.tensor(9, device=x.device), 65).to(x.dtype)[None, :, None, None] * (16.0 * self.scale)
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
    license_path = root / "encoders/XFeat/LICENSE"
    model.parent.mkdir(parents=True)
    weights.parent.mkdir(parents=True)
    model.write_text(MODEL_SOURCE, encoding="utf-8")
    interpolator.write_text(INTERPOLATOR_SOURCE, encoding="utf-8")
    wrapper.write_text("# locked sparse XFeat wrapper\n", encoding="utf-8")
    license_path.write_text("synthetic Apache-2.0 fixture\n", encoding="utf-8")
    torch.save({"scale": torch.tensor(1.0)}, weights)
    _run_git(root, "init", "-q")
    _run_git(root, "add", "encoders/XFeat")
    _run_git(
        root,
        "-c",
        "user.name=Arm A Test",
        "-c",
        "user.email=arm-a@example.invalid",
        "commit",
        "-q",
        "-m",
        "locked fake XFeat",
    )
    return {
        "worktree": root,
        "weights": weights,
        "weights_sha": file_sha256(weights),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "tree": _run_git(root, "rev-parse", "HEAD:encoders/XFeat"),
        "model_sha": file_sha256(model),
        "interpolator_sha": file_sha256(interpolator),
        "wrapper_sha": file_sha256(wrapper),
    }


def _write_dataset(root: Path) -> tuple[list[str], str, torch.Tensor]:
    height = width = 64
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
        "1 PINHOLE 64 64 50 50 31.5 31.5\n", encoding="utf-8"
    )
    rows = [
        f"{index} 1 0 0 0 0 0 0 1 {name}\n\n"
        for index, name in enumerate([*mapping_names, test_name], start=1)
    ]
    (sparse / "images.txt").write_text("".join(rows), encoding="utf-8")
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (sparse / "list_test.txt").write_text(test_name + "\n", encoding="utf-8")
    valid = torch.ones((height, width), dtype=torch.bool)
    # The locked align_corners=False interpolator downweights boundary rows;
    # with the synthetic equal logits, (9,9) is the first stable top-K row.
    valid[9, 9] = False
    masks = {
        name: [valid.numpy(), np.ones_like(valid.numpy()), np.ones_like(valid.numpy())]
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
        [[2.0, 2.0], [30.0, 30.0], [60.0, 60.0]], dtype=torch.float32
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
            "native_input_hw": [64, 64],
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
    query_path = root / "query_cache.pt"
    teacher_path = root / "teacher.pt"
    torch.save(query_cache, query_path)
    torch.save(teacher, teacher_path)
    return query_path, teacher_path, query_cache, teacher


@pytest.fixture
def arm_a_bundle(tmp_path: Path) -> dict:
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
        "valid_mask": valid_mask,
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


def _synthetic_outputs(height: int, width: int) -> tuple[torch.Tensor, ...]:
    coarse_hw = (height // 8, width // 8)
    dense = torch.ones((1, 64, *coarse_hw))
    logits = torch.full((1, 65, *coarse_hw), -8.0)
    logits[:, 9] = 8.0
    reliability = torch.ones((1, 1, *coarse_hw))
    return dense, logits, reliability


def _interpolators(artifact: dict) -> tuple[torch.nn.Module, torch.nn.Module]:
    module = _load_module(
        artifact["worktree"] / "encoders/XFeat/modules/interpolator.py",
        identity="arm_a_test",
    )
    return module.InterpolateSparse2d("nearest"), module.InterpolateSparse2d(
        "bilinear"
    )


def test_detector_matches_locked_single_image_semantics_and_mask_proof(
    arm_a_bundle: dict,
):
    nearest, bilinear = _interpolators(arm_a_bundle["artifact"])
    keypoints, scores, before_mask, lineage = extract_xfeat_detector(
        outputs=_synthetic_outputs(64, 64),
        native_hw=(64, 64),
        valid_mask=arm_a_bundle["valid_mask"],
        requested_keypoint_count=4,
        nearest_interpolator=nearest,
        bilinear_interpolator=bilinear,
    )
    assert before_mask == 4
    assert keypoints.tolist() == [[17.0, 9.0], [25.0, 9.0], [33.0, 9.0]]
    assert scores.shape == (3,)
    assert bool((scores[:-1] >= scores[1:]).all())
    assert lineage["strict_probability_threshold"] == 0.05
    assert lineage["nms_kernel_size"] == 5
    assert lineage["score_semantics"] == (
        "nearest_probability_times_bilinear_reliability"
    )
    proof = lineage["mask_equivalence_proof"]
    assert proof["identity_xfeat_resize"] is True
    assert proof["round_floor_indices_equal"] is True
    assert proof["round_floor_mask_decisions_equal"] is True
    assert proof["checked_pre_mask_rows"] == 4


def test_detector_rejects_nondivisible_native_mask_contract(arm_a_bundle: dict):
    nearest, bilinear = _interpolators(arm_a_bundle["artifact"])
    with pytest.raises(ValueError, match="divisible by 32"):
        extract_xfeat_detector(
            outputs=_synthetic_outputs(32, 64),
            native_hw=(33, 65),
            valid_mask=torch.ones((33, 65), dtype=torch.bool),
            requested_keypoint_count=4,
            nearest_interpolator=nearest,
            bilinear_interpolator=bilinear,
        )


def test_stairs_480x640_round_floor_mask_contract_is_exact():
    height, width = 480, 640
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = ((x + 3 * y) % 7) != 0
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [319.0, 239.0], [639.0, 479.0]]
    )
    keep, proof = _mask_equivalence_proof(
        contract=xfeat_resize_contract((height, width)),
        xfeat_keypoints=coordinates,
        native_keypoints=coordinates.clone(),
        valid_mask=mask,
    )
    assert keep.tolist() == [False, True, False, True]
    assert proof["identity_xfeat_resize"] is True
    assert proof["round_indices_sha256"] == proof["floor_indices_sha256"]
    assert proof["round_mask_keep_sha256"] == proof["floor_mask_keep_sha256"]


def test_cli_materializes_consumer_valid_mapping_only_arm_a(
    arm_a_bundle: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "xfeat_arm_a.pt"
    producer_cli.main(_cli_arguments(arm_a_bundle, output))
    probe = torch.load(output, map_location="cpu", weights_only=False)
    assert probe["mapping_only"] is True
    assert probe["uses_test_queries"] is False
    assert probe["capabilities"] == {
        "detector_repeatability": True,
        "descriptor_identity": False,
    }
    assert probe["producer"]["device"] == "cpu"
    assert probe["producer"]["gpu_used"] is False
    assert probe["producer"]["candidate_detector_used"] is True
    assert probe["producer"]["shared_forward_descriptor_output_used"] is False
    assert probe["producer"]["candidate_descriptor_rows_materialized"] is False
    assert probe["producer"]["pair_matcher_used"] is False
    assert set(probe["queries"]) == set(arm_a_bundle["mapping_names"])
    assert arm_a_bundle["test_name"] not in probe["queries"]
    for record in probe["queries"].values():
        assert "descriptor_at_reference_keypoints" not in record
        assert record["detected_count_before_mask"] == 4
        assert record["detected_count_after_mask"] == 3
        assert record["detector_keypoints"].tolist() == [
            [17.0, 9.0],
            [25.0, 9.0],
            [33.0, 9.0],
        ]
        proof = record["detector_lineage"]["mask_equivalence_proof"]
        assert proof["round_floor_indices_equal"] is True
        assert proof["round_floor_mask_decisions_equal"] is True
    validated = validate_probe(
        probe,
        arm_a_bundle["query_payload"],
        arm_a_bundle["teacher_payload"],
        require_detector=True,
        query_cache_path=arm_a_bundle["query_cache"],
        teacher_path=arm_a_bundle["teacher"],
    )
    assert validated["requested_keypoint_count"] == 4
    assert validated["validated_detector_keypoints"] == 6
    assert _run_git(
        arm_a_bundle["artifact"]["worktree"],
        "status",
        "--porcelain",
        "--untracked-files=all",
    ) == ""


def test_cli_rejects_reference_sha_mismatch(
    arm_a_bundle: dict, tmp_path: Path
):
    arguments = _cli_arguments(arm_a_bundle, tmp_path / "bad.pt")
    flag = "--expected-query-cache-sha256"
    arguments[arguments.index(flag) + 1] = "0" * 64
    with pytest.raises(ValueError, match="query-cache SHA256 mismatch"):
        producer_cli.main(arguments)
