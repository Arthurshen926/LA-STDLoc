#!/usr/bin/env python3
"""Replay exact frozen-2DGS provenance assignment for a Track factor payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from common.hashing import sha256_file
from data.scene import FrozenScene
from evidence.track_provenance_assignment import (
    assign_tracks_by_splat_provenance,
)
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from map_learning.bootstrap import _gaussian_model_for_type


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_factor_contract(
    factor: dict,
    *,
    expected_mapping_keypoints: int,
    expected_pair_budget: int,
) -> None:
    if factor.get("schema") != "lafgs_pair_policy_track_factor":
        raise ValueError("Unexpected pair factor schema")
    if factor.get("version") != 1:
        raise ValueError("Unexpected pair factor version")
    if factor.get("uses_test_queries") is not False:
        raise ValueError("Assignment replay must be mapping-only")
    if "assignment" in factor:
        raise ValueError(
            "Pair factors containing assignment are invalid and must never be "
            "consumed"
        )
    if any(
        factor.get(name) is not False
        for name in (
            "density_factor_mutated",
            "descriptor_factor_mutated",
            "selector_factor_mutated",
        )
    ):
        raise ValueError("Track factor is not pair-policy-only")
    expected_mapping_keypoints = int(expected_mapping_keypoints)
    expected_pair_budget = int(expected_pair_budget)
    if expected_mapping_keypoints <= 0:
        raise ValueError("Expected mapping keypoints must be positive")
    if expected_pair_budget <= 0:
        raise ValueError("Expected pair budget must be positive")
    if int(factor.get("mapping_keypoint_factor", -1)) != (
        expected_mapping_keypoints
    ):
        raise ValueError("Track factor mapping K differs from the replay contract")
    sidecar = factor.get("pair_sidecar")
    policy = sidecar.get("policy") if isinstance(sidecar, dict) else None
    pair = sidecar.get("pair") if isinstance(sidecar, dict) else None
    if not isinstance(policy, dict) or not isinstance(pair, dict):
        raise ValueError("Track factor lacks an exact pair-policy sidecar")
    left = torch.as_tensor(pair.get("left_query_index", [])).reshape(-1)
    right = torch.as_tensor(pair.get("right_query_index", [])).reshape(-1)
    if (
        int(policy.get("exact_pair_budget", -1)) != expected_pair_budget
        or int(left.numel()) != expected_pair_budget
        or int(right.numel()) != expected_pair_budget
    ):
        raise ValueError("Track factor pair budget differs from the replay contract")


def _camera_key(camera) -> str:
    return str(camera.image_name).replace("\\", "/")


def _camera_names_sha256(names) -> str:
    normalized = sorted(str(name).replace("\\", "/") for name in names)
    return hashlib.sha256(
        ("\n".join(normalized) + "\n").encode("utf-8")
    ).hexdigest()


def _assignment_dependent_diagnostics(
    *, tracks: dict, track_geometry: dict, landmark_geometry: dict,
    assignment: dict, query_count: int, provenance_diagnostics: dict,
) -> dict:
    """Reproduce bootstrap diagnostics derived after provenance assignment."""
    track_high_confidence = torch.as_tensor(
        track_geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    track_indices = torch.as_tensor(tracks["track_index"], dtype=torch.long)
    query_indices = torch.as_tensor(tracks["query_index"], dtype=torch.long)
    group_offsets = torch.as_tensor(
        assignment["track_landmark_offsets"], dtype=torch.long
    )
    group_indices = torch.as_tensor(
        assignment["track_landmark_indices"], dtype=torch.long
    )
    support_by_query = [[] for _ in range(int(query_count))]
    for track, query in zip(track_indices.tolist(), query_indices.tolist()):
        if not bool(track_high_confidence[track]):
            continue
        begin = int(group_offsets[track])
        end = int(group_offsets[track + 1])
        support_by_query[query].extend(group_indices[begin:end].tolist())
    support_edge_count = sum(
        int(torch.unique(torch.as_tensor(values, dtype=torch.long)).numel())
        for values in support_by_query
    )
    diagnostics = {
        "geometry_teacher_identity_mode": "track_first_provenance",
        "geometry_teacher_triangulated_track_count": int(
            torch.as_tensor(track_geometry["triangulated"]).sum().item()
        ),
        "geometry_teacher_high_confidence_track_count": int(
            track_high_confidence.sum().item()
        ),
        "geometry_teacher_assigned_landmark_count": int(
            torch.as_tensor(landmark_geometry["track_assigned"]).sum().item()
        ),
        "geometry_teacher_high_confidence_landmark_count": int(
            torch.as_tensor(
                landmark_geometry["triangulation_high_confidence"]
            ).sum().item()
        ),
        "geometry_teacher_query_support_edge_count": int(support_edge_count),
        **dict(provenance_diagnostics),
    }
    if "landmark_track_count" in landmark_geometry:
        counts = torch.as_tensor(landmark_geometry["landmark_track_count"])
        assigned_landmarks = counts > 0
        multi_track = counts > 1
        effective_support = torch.as_tensor(
            landmark_geometry["landmark_effective_track_support"]
        )
        xyz_max_residual = torch.as_tensor(
            landmark_geometry["landmark_track_xyz_max_residual_m"]
        )
        diagnostics.update(
            {
                "geometry_teacher_multi_track_landmark_count": int(
                    multi_track.sum().item()
                ),
                "geometry_teacher_multi_track_fraction": float(
                    multi_track.float().sum().item()
                    / max(int(assigned_landmarks.sum().item()), 1)
                ),
                "geometry_teacher_effective_track_support_mean": float(
                    effective_support[assigned_landmarks].mean().item()
                    if bool(assigned_landmarks.any())
                    else 0.0
                ),
                "geometry_teacher_track_conflict_gt_1cm_count": int(
                    (multi_track & (xyz_max_residual > 0.01)).sum().item()
                ),
                "geometry_teacher_track_conflict_gt_3cm_count": int(
                    (multi_track & (xyz_max_residual > 0.03)).sum().item()
                ),
                "geometry_teacher_track_conflict_gt_5cm_count": int(
                    (multi_track & (xyz_max_residual > 0.05)).sum().item()
                ),
            }
        )
    return diagnostics


def _write_payload(
    *, factor: dict, assignment: dict, diagnostics: dict,
    landmark_indices: torch.Tensor, output: Path, factor_path: Path,
    base_state_path: Path, query_cache_path: Path, bootstrap_manifest_path: Path,
    assignment_parameters: dict, expected_query_cache_sha256: str,
    expected_frozen_bootstrap_manifest_sha256: str,
) -> dict:
    names = [str(value) for value in factor["query_names"]]
    payload = {
        "version": 1,
        "schema": "lafgs_track_first_payload",
        "query_names": names,
        "tracks": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in factor["tracks"].items()
        },
        "track_geometry": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in factor["track_geometry"].items()
        },
        "assignment": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in assignment.items()
        },
        "query_bins": torch.as_tensor(factor["query_bins"]).long(),
        "diagnostics": {**dict(factor["diagnostics"]), **dict(diagnostics)},
        "pair_sidecar": factor["pair_sidecar"],
        "train_camera_names_sha256": _camera_names_sha256(names),
        "landmark_indices": torch.as_tensor(landmark_indices).long().cpu(),
        "provenance": {
            "schema": "lafgs_replayed_track_provenance_assignment",
            "version": 1,
            "uses_test_queries": False,
            "source_factor": str(factor_path.resolve()),
            "source_factor_sha256": sha256_file(factor_path),
            "base_state": str(base_state_path.resolve()),
            "base_state_sha256": sha256_file(base_state_path),
            "query_cache": str(query_cache_path.resolve()),
            "query_cache_sha256": sha256_file(query_cache_path),
            "expected_query_cache_sha256": expected_query_cache_sha256,
            "frozen_bootstrap_manifest": str(
                bootstrap_manifest_path.resolve()
            ),
            "frozen_bootstrap_manifest_sha256": sha256_file(
                bootstrap_manifest_path
            ),
            "expected_frozen_bootstrap_manifest_sha256": (
                expected_frozen_bootstrap_manifest_sha256
            ),
            "assignment_algorithm": "frozen_2dgs_splat_provenance_exact_replay",
            "assignment_parameters": dict(assignment_parameters),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--frozen-bootstrap-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-frozen-bootstrap-manifest-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    factor = _load(args.factor)
    _validate_factor_contract(
        factor,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_pair_budget=args.expected_pair_budget,
    )
    expected_frozen_bootstrap_manifest_sha256 = str(
        args.expected_frozen_bootstrap_manifest_sha256
    ).strip().lower()
    if len(expected_frozen_bootstrap_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_frozen_bootstrap_manifest_sha256
    ):
        raise ValueError(
            "Expected frozen-bootstrap-manifest SHA-256 must be 64 lowercase "
            "hex digits"
        )
    if sha256_file(args.frozen_bootstrap_manifest) != (
        expected_frozen_bootstrap_manifest_sha256
    ):
        raise ValueError(
            "Frozen-bootstrap-manifest SHA-256 differs from the replay contract"
        )
    bootstrap_manifest = json.loads(args.frozen_bootstrap_manifest.read_text())
    frozen = dict(bootstrap_manifest["arguments"])
    if str(frozen.get("geometry_teacher_identity_mode")) != (
        "track_first_provenance"
    ):
        raise ValueError("Frozen bootstrap did not use splat provenance")
    expected_paths = {
        "initial_state_path": args.base_state,
        "query_cache_path": args.query_cache,
    }
    for name, supplied in expected_paths.items():
        if Path(str(frozen[name])).resolve() != supplied.resolve():
            raise ValueError(f"{name} differs from the frozen bootstrap contract")
    frozen_inputs = dict(bootstrap_manifest["inputs"])
    base_contract = dict(frozen_inputs["initial_state_path"])
    if Path(str(base_contract["path"])).resolve() != args.base_state.resolve():
        raise ValueError("Manifest input lineage names a different Stage-A state")
    expected_base_sha256 = str(base_contract.get("sha256") or "")
    if not expected_base_sha256 or sha256_file(args.base_state) != expected_base_sha256:
        raise ValueError("Stage-A state SHA-256 differs from frozen bootstrap")
    expected_query_cache_sha256 = str(
        args.expected_query_cache_sha256
    ).strip().lower()
    if len(expected_query_cache_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_query_cache_sha256
    ):
        raise ValueError("Expected query-cache SHA-256 must be 64 lowercase hex digits")
    if sha256_file(args.query_cache) != expected_query_cache_sha256:
        raise ValueError("Query-cache SHA-256 differs from frozen factor contract")
    expected_mapping_keypoints = int(args.expected_mapping_keypoints)
    if int(factor.get("mapping_keypoint_factor", -1)) != int(
        frozen["native_keypoint_count"]
    ):
        raise ValueError("Factor density differs from the frozen bootstrap contract")
    if int(frozen["native_keypoint_count"]) != expected_mapping_keypoints:
        raise ValueError(
            "Frozen bootstrap mapping K differs from the explicit replay contract"
        )
    assignment_parameters = {
        "topk": int(frozen["geometry_teacher_provenance_topk"]),
        "minimum_consensus_rate": float(
            frozen["geometry_teacher_provenance_min_consensus_rate"]
        ),
        "minimum_views": int(
            frozen["geometry_teacher_provenance_min_views"]
        ),
        "group_maximum_landmarks": int(
            frozen["geometry_teacher_provenance_group_max_landmarks"]
        ),
        "group_minimum_relative_mass": float(
            frozen["geometry_teacher_provenance_group_min_relative_mass"]
        ),
        "group_minimum_consensus_rate": float(
            frozen["geometry_teacher_provenance_group_min_consensus_rate"]
        ),
        "depth_absolute_tolerance_m": float(
            frozen["geometry_teacher_provenance_depth_abs_tolerance_m"]
        ),
        "depth_relative_tolerance": float(
            frozen["geometry_teacher_provenance_depth_rel_tolerance"]
        ),
    }
    base_state = _load(args.base_state)
    cache_payload = _load(args.query_cache)
    cache = cache_payload.get("queries", cache_payload)
    cache_contract = dict(cache_payload.get("signature_payload", {}))
    names = [str(value) for value in factor["query_names"]]
    if len(names) != len(set(names)) or names != list(cache):
        raise ValueError("Query cache does not align with Track factor")
    dataset_path = Path(str(frozen["source_path"])).resolve()
    prior_path = Path(str(frozen["model_path"])).resolve()
    expected_source = Path(str(cache_contract["source_path"])).resolve()
    expected_model = Path(str(cache_contract["model_path"])).resolve()
    if expected_source != dataset_path:
        raise ValueError("Dataset path differs from the frozen query-cache contract")
    if expected_model != prior_path:
        raise ValueError("Prior path differs from the frozen query-cache contract")
    frozen_prior = dict(bootstrap_manifest["rgb_prior"])
    cache_prior = dict(cache_contract["rgb_prior_fingerprint"])
    prior_ply = (
        prior_path
        / "point_cloud"
        / f"iteration_{int(frozen['load_iteration'])}"
        / "point_cloud.ply"
    )
    if Path(str(frozen_prior["exported_ply"])).resolve() != prior_ply.resolve():
        raise ValueError("Frozen manifest names a different Gaussian PLY")
    expected_prior_sha256 = str(frozen_prior["exported_ply_sha256"])
    if sha256_file(prior_ply) != expected_prior_sha256:
        raise ValueError("Gaussian PLY SHA-256 differs from frozen bootstrap")
    for name in ("exported_ply_sha256", "primitive_count"):
        if cache_prior.get(name) != frozen_prior.get(name):
            raise ValueError(f"RGB prior {name} differs from query-cache contract")
    scene_args = SimpleNamespace(
        model_path=str(prior_path),
        source_path=str(dataset_path),
        images=str(frozen["images"]),
        resolution=int(frozen["resolution"]),
        data_device=str(frozen["data_device"]),
        gaussian_type=str(frozen["gaussian_type"]),
        sh_degree=int(frozen["sh_degree"]),
        white_background=bool(frozen["white_background"]),
    )
    for name in ("images", "resolution", "white_background", "load_iteration"):
        if cache_contract.get(name) != frozen.get(name):
            raise ValueError(f"{name} differs from the frozen query-cache contract")
    if int(cache_contract.get("native_sparse_keypoint_count", -1)) != (
        expected_mapping_keypoints
    ):
        raise ValueError(
            "Query-cache mapping K differs from the explicit replay contract"
        )
    if scene_args.gaussian_type != "2dgs":
        raise ValueError("Exact splat-provenance replay requires frozen 2DGS")
    gaussians = _gaussian_model_for_type(
        scene_args.gaussian_type, scene_args.sh_degree
    )
    scene = FrozenScene(
        scene_args,
        gaussians,
        load_iteration=int(frozen["load_iteration"]),
        load_test_cameras=False,
    )
    mapping_cameras = scene.getTrainCameras()
    camera_names = [_camera_key(camera) for camera in mapping_cameras]
    cameras_by_name = dict(zip(camera_names, mapping_cameras))
    if len(camera_names) != len(set(camera_names)) or names != camera_names:
        raise ValueError("Mapping camera scene does not align with Track factor")
    if int(gaussians.get_xyz.shape[0]) != int(frozen_prior["primitive_count"]):
        raise ValueError("Loaded Gaussian primitive count differs from frozen bootstrap")
    for name, camera in zip(names, mapping_cameras):
        cached = cache[name]
        height, width = map(int, cached["native_input_hw"])
        if (height, width) != (camera.image_height, camera.image_width):
            raise ValueError(f"Frozen image size differs for mapping camera {name}")
        K = torch.as_tensor(cached["native_K"], dtype=torch.float64)
        focal_x = width / (2.0 * math.tan(float(camera.FoVx) / 2.0))
        focal_y = height / (2.0 * math.tan(float(camera.FoVy) / 2.0))
        expected_K = torch.tensor(
            [
                [focal_x, 0.0, width / 2.0],
                [0.0, focal_y, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        if not torch.allclose(K, expected_K, rtol=1e-6, atol=1e-5):
            raise ValueError(f"Frozen intrinsics differ for mapping camera {name}")
    landmark_indices = torch.as_tensor(base_state["landmark_indices"]).long()
    bank_xyz = torch.as_tensor(base_state["landmark_xyz"]).float()
    if landmark_indices.shape != (bank_xyz.shape[0],):
        raise ValueError("Stage-A landmark IDs and XYZ do not align")
    if landmark_indices.numel() and (
        int(landmark_indices.min()) < 0
        or int(landmark_indices.max()) >= int(gaussians.get_xyz.shape[0])
    ):
        raise ValueError("Stage-A landmark IDs are outside the frozen prior")
    global_indices = landmark_indices.to(device=gaussians.get_xyz.device)
    keypoints = [
        torch.as_tensor(cache[name]["native_keypoints"]).float()
        + float(PIXEL_CENTER_OFFSET)
        for name in names
    ]
    background_value = 1.0 if scene_args.white_background else 0.0
    background = torch.full(
        (3,), background_value, device=gaussians.get_xyz.device
    )
    landmark_geometry, assignment, provenance_diagnostics = (
        assign_tracks_by_splat_provenance(
        tracks=factor["tracks"],
        track_geometry=factor["track_geometry"],
        keypoints=keypoints,
        query_names=names,
        cache=cache,
        bank_xyz=bank_xyz,
        gaussians=gaussians,
        cameras_by_name=cameras_by_name,
        landmark_global_indices=global_indices,
        background=background,
        **assignment_parameters,
        )
    )
    diagnostics = _assignment_dependent_diagnostics(
        tracks=factor["tracks"],
        track_geometry=factor["track_geometry"],
        landmark_geometry=landmark_geometry,
        assignment=assignment,
        query_count=len(names),
        provenance_diagnostics=provenance_diagnostics,
    )
    payload = _write_payload(
        factor=factor,
        assignment=assignment,
        diagnostics=diagnostics,
        landmark_indices=landmark_indices,
        output=args.output,
        factor_path=args.factor,
        base_state_path=args.base_state,
        query_cache_path=args.query_cache,
        bootstrap_manifest_path=args.frozen_bootstrap_manifest,
        assignment_parameters=assignment_parameters,
        expected_query_cache_sha256=expected_query_cache_sha256,
        expected_frozen_bootstrap_manifest_sha256=(
            expected_frozen_bootstrap_manifest_sha256
        ),
    )
    report = {
        "schema": "lafgs_track_provenance_assignment_replay",
        "version": 1,
        "uses_test_queries": False,
        "pair_policy": str(factor["pair_policy"]),
        "expected_mapping_keypoints": int(args.expected_mapping_keypoints),
        "expected_pair_budget": int(args.expected_pair_budget),
        "assignment_fields": sorted(payload["assignment"]),
        "track_count": int(torch.as_tensor(factor["tracks"]["track_level"]).numel()),
        "high_confidence_track_count": int(
            torch.as_tensor(factor["track_geometry"]["triangulation_high_confidence"]).sum()
        ),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "frozen_bootstrap_manifest_sha256": sha256_file(
            args.frozen_bootstrap_manifest
        ),
        "expected_frozen_bootstrap_manifest_sha256": (
            expected_frozen_bootstrap_manifest_sha256
        ),
        "assignment_parameters": assignment_parameters,
        "expected_query_cache_sha256": expected_query_cache_sha256,
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
