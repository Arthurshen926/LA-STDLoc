import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import probe_track_build_inputs
from scripts import compare_cycle_verified_fisher_mechanism as compare_cli
from scripts import materialize_cycle_verified_pair_probe as probe_cli
from scripts import select_cycle_verified_fisher_pairs as select_cli
from scripts.cycle_verified_fisher_cli_common import SCENE_CONTRACTS, torch_load


def _look_at_pose(center, target):
    center = torch.as_tensor(center, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    forward = torch.nn.functional.normalize(target - center, dim=0)
    right = torch.nn.functional.normalize(
        torch.cross(
            forward,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            dim=0,
        ),
        dim=0,
    )
    down = torch.cross(forward, right, dim=0)
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = torch.stack((right, down, forward))
    pose[:3, 3] = -(pose[:3, :3] @ center)
    return pose


def _project(points, camera_K, pose):
    camera = torch.einsum("ij,pj->pi", pose[:3, :3], points) + pose[:3, 3]
    pixel = torch.einsum("ij,pj->pi", camera_K, camera)
    return pixel[:, :2] / pixel[:, 2:]


def _names_sha256(names):
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def _write_factor(
    path,
    *,
    policy,
    pairs,
    names,
    names_sha256,
    cache_path,
    cache_sha256,
    probe=None,
    selection=None,
):
    precomputed = policy == "cycle_verified_fisher"
    lineage = {
        "query_cache": {"path": str(cache_path), "sha256": cache_sha256}
    }
    if probe is not None:
        lineage["pair_match_probe"] = {
            "path": str(probe["path"]),
            "sha256": probe["sha256"],
            "content_sha256": probe["content_sha256"],
        }
    if selection is not None:
        lineage["pair_selection"] = {
            "path": str(selection["path"]),
            "sha256": selection["sha256"],
            "content_sha256": selection["content_sha256"],
        }
    payload = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 3,
        "mapping_nms_radius": 1,
        "pair_policy": policy,
        "query_names": names,
        "query_names_sha256": names_sha256,
        "input_lineage": lineage,
        "diagnostics": {"track_pair_matches_reused": int(precomputed)},
        "pair_sidecar": {
            "policy": {
                "name": policy,
                "uses_test_queries": False,
                "uses_precomputed_pair_matches": precomputed,
            },
            "pair": {
                "left_query_index": torch.tensor([pair[0] for pair in pairs]),
                "right_query_index": torch.tensor([pair[1] for pair in pairs]),
            },
        },
    }
    torch.save(payload, path)
    return payload


def _write_report(
    path,
    *,
    policy,
    factor_path,
    cache_path,
    cache_sha256,
    names_sha256,
    track_scale,
):
    factor_sha256 = sha256_file(factor_path)
    payload = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 3,
        "mapping_nms_radius": 1,
        "pair_policy": policy,
        "exact_pair_budget": 5,
        "mapping_query_count": 5,
        "query_names_sha256": names_sha256,
        "artifact": str(factor_path),
        "artifact_sha256": factor_sha256,
        "inputs": {
            "query_cache": {"path": str(cache_path), "sha256": cache_sha256}
        },
        "track": {
            "triangulated_track_count": 100 * track_scale,
            "broad_eligible_track_count": 100 * track_scale,
            "high_confidence_track_count": 100 * track_scale,
            "triangulated_covariance_trace_m2": {"p90": 1.0 / track_scale},
            "mapping_query_with_broad_track_fraction": 1.0 * track_scale,
        },
    }
    path.write_text(json.dumps(payload))
    return payload


@pytest.fixture
def p8_cli_artifacts(tmp_path, monkeypatch):
    monkeypatch.setitem(
        SCENE_CONTRACTS,
        "stairs",
        {
            "mapping_keypoints": 3,
            "nms_radius": 1,
            "pair_budget": 5,
            "candidate_pair_count": 6,
            "candidate_component_count": 1,
        },
    )
    centers = torch.tensor(
        [
            [-0.8, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.8, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    poses = torch.stack([_look_at_pose(center, target) for center in centers])
    camera_K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(5, 1, 1)
    points = torch.tensor(
        [[-0.2, 0.0, 4.0], [0.0, 0.15, 4.2], [0.2, -0.1, 3.8]],
        dtype=torch.float64,
    )
    keypoints = [_project(points, camera_K[index], poses[index]) for index in range(5)]
    names = [f"mapping/{index}.png" for index in range(5)]
    names_sha256 = _names_sha256(names)
    cache_path = tmp_path / "query_cache.pt"
    cache = {
        "version": 3,
        "uses_test_queries": False,
        "signature_payload": {
            "native_sparse_keypoint_count": 3,
            "native_sparse_nms_radius": 1,
        },
        "queries": {
            name: {
                "native_descriptors": torch.eye(3),
                "native_keypoints": keypoints[index].float() - 0.5,
                "native_scores": torch.ones(3),
                "native_K": camera_K[index].float(),
                "pose_w2c": poses[index].float(),
                "native_sparse_metadata": {
                    "nms_radius": 1,
                    "requested_keypoint_count": 3,
                },
            }
            for index, name in enumerate(names)
        },
    }
    torch.save(cache, cache_path)
    cache_sha256 = sha256_file(cache_path)
    nearest_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    geometry_pairs = [(0, 1), (0, 2), (1, 2), (2, 4), (3, 4)]
    nearest_path = tmp_path / "nearest.pt"
    geometry_path = tmp_path / "geometry.pt"
    _write_factor(
        nearest_path,
        policy="nearest",
        pairs=nearest_pairs,
        names=names,
        names_sha256=names_sha256,
        cache_path=cache_path,
        cache_sha256=cache_sha256,
    )
    _write_factor(
        geometry_path,
        policy="parallax_diverse",
        pairs=geometry_pairs,
        names=names,
        names_sha256=names_sha256,
        cache_path=cache_path,
        cache_sha256=cache_sha256,
    )
    probe_path = tmp_path / "probe.pt"
    probe_args = [
        "--scene", "stairs",
        "--query-cache", str(cache_path),
        "--expected-query-cache-sha256", cache_sha256,
        "--nearest-factor", str(nearest_path),
        "--expected-nearest-factor-sha256", sha256_file(nearest_path),
        "--geometry-factor", str(geometry_path),
        "--expected-geometry-factor-sha256", sha256_file(geometry_path),
        "--expected-query-names-sha256", names_sha256,
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--minimum-similarity", "0.65",
        "--minimum-margin", "0.01",
        "--maximum-epipolar-error-px", "2.0",
        "--epipolar-candidate-topk", "1",
        "--epipolar-recovered-minimum-similarity", "-1.0",
        "--epipolar-recovered-minimum-margin", "-1.0",
        "--device", "cpu",
        "--output", str(probe_path),
    ]
    probe_cli.main(probe_args)
    probe = torch_load(probe_path)
    selection_path = tmp_path / "selection.pt"
    selection_args = [
        "--scene", "stairs",
        "--query-cache", str(cache_path),
        "--expected-query-cache-sha256", cache_sha256,
        "--probe", str(probe_path),
        "--expected-probe-sha256", sha256_file(probe_path),
        "--expected-probe-content-sha256", probe["content_sha256"],
        "--expected-query-names-sha256", names_sha256,
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--minimum-camera-degree", "1",
        "--maximum-cycle-reprojection-error-px", "2.0",
        "--output", str(selection_path),
    ]
    select_cli.main(selection_args)
    selection = torch_load(selection_path)
    return {
        "tmp_path": tmp_path,
        "cache_path": cache_path,
        "cache_sha256": cache_sha256,
        "cache": cache,
        "names": names,
        "names_sha256": names_sha256,
        "keypoints": keypoints,
        "camera_K": camera_K,
        "poses": poses,
        "nearest_path": nearest_path,
        "nearest_pairs": nearest_pairs,
        "probe_path": probe_path,
        "probe": probe,
        "selection_path": selection_path,
        "selection": selection,
        "selection_args": selection_args,
    }


def test_cli_probe_selection_reaches_tracks_without_second_matcher(
    p8_cli_artifacts, monkeypatch
):
    from evidence import triangulation

    artifact = p8_cli_artifacts

    def forbidden_matcher(*args, **kwargs):
        raise AssertionError("Track construction must reuse selected probe rows")

    monkeypatch.setattr(triangulation, "reciprocal_epipolar_matches", forbidden_matcher)
    tracks, diagnostics, sidecar = triangulation.build_cycle_consistent_tracks(
        descriptors=[torch.eye(3) for _ in range(5)],
        keypoints=artifact["keypoints"],
        detector_scores=[torch.ones(3) for _ in range(5)],
        camera_K=artifact["camera_K"],
        pose_w2c=artifact["poses"],
        pair_policy="cycle_verified_fisher",
        pair_budget=5,
        minimum_track_views=3,
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        device="cpu",
        **probe_track_build_inputs(artifact["probe"], artifact["selection"]),
    )
    assert diagnostics["track_pair_matches_reused"] == 1
    assert diagnostics["track_camera_pair_candidate_count"] == 5
    assert diagnostics["track_count"] == 3
    assert tracks["query_index"].unique().numel() == 5
    assert sidecar["policy"]["uses_precomputed_pair_matches"] is True


def test_lineage_failure_is_exit_one(p8_cli_artifacts):
    arguments = list(p8_cli_artifacts["selection_args"])
    digest_index = arguments.index("--expected-probe-sha256") + 1
    arguments[digest_index] = "0" * 64
    arguments.extend(("--overwrite",))
    with pytest.raises(SystemExit) as error:
        select_cli.entrypoint(arguments)
    assert error.value.code == 1


def test_mechanism_stop_is_persisted_and_exits_two(p8_cli_artifacts):
    artifact = p8_cli_artifacts
    tmp_path = artifact["tmp_path"]
    selected = artifact["selection"]["selected_pair"]
    selected_pairs = list(
        zip(selected["left_query_index"].tolist(), selected["right_query_index"].tolist())
    )
    probe_record = {
        "path": artifact["probe_path"],
        "sha256": sha256_file(artifact["probe_path"]),
        "content_sha256": artifact["probe"]["content_sha256"],
    }
    selection_record = {
        "path": artifact["selection_path"],
        "sha256": sha256_file(artifact["selection_path"]),
        "content_sha256": artifact["selection"]["content_sha256"],
    }
    variant_path = tmp_path / "variant.pt"
    _write_factor(
        variant_path,
        policy="cycle_verified_fisher",
        pairs=selected_pairs,
        names=artifact["names"],
        names_sha256=artifact["names_sha256"],
        cache_path=artifact["cache_path"],
        cache_sha256=artifact["cache_sha256"],
        probe=probe_record,
        selection=selection_record,
    )
    control_report = tmp_path / "control.json"
    variant_report = tmp_path / "variant.json"
    _write_report(
        control_report,
        policy="nearest",
        factor_path=artifact["nearest_path"],
        cache_path=artifact["cache_path"],
        cache_sha256=artifact["cache_sha256"],
        names_sha256=artifact["names_sha256"],
        track_scale=1.0,
    )
    _write_report(
        variant_report,
        policy="cycle_verified_fisher",
        factor_path=variant_path,
        cache_path=artifact["cache_path"],
        cache_sha256=artifact["cache_sha256"],
        names_sha256=artifact["names_sha256"],
        track_scale=0.5,
    )
    output = tmp_path / "mechanism_gate.json"
    arguments = [
        "--scene", "stairs",
        "--query-cache", str(artifact["cache_path"]),
        "--expected-query-cache-sha256", artifact["cache_sha256"],
        "--probe", str(artifact["probe_path"]),
        "--expected-probe-sha256", probe_record["sha256"],
        "--expected-probe-content-sha256", probe_record["content_sha256"],
        "--selection", str(artifact["selection_path"]),
        "--expected-selection-sha256", selection_record["sha256"],
        "--expected-selection-content-sha256", selection_record["content_sha256"],
        "--control-factor", str(artifact["nearest_path"]),
        "--expected-control-factor-sha256", sha256_file(artifact["nearest_path"]),
        "--control-report", str(control_report),
        "--expected-control-report-sha256", sha256_file(control_report),
        "--variant-factor", str(variant_path),
        "--expected-variant-factor-sha256", sha256_file(variant_path),
        "--variant-report", str(variant_report),
        "--expected-variant-report-sha256", sha256_file(variant_report),
        "--expected-query-names-sha256", artifact["names_sha256"],
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--maximum-cycle-reprojection-error-px", "2.0",
        "--output", str(output),
    ]
    with pytest.raises(SystemExit) as error:
        compare_cli.entrypoint(arguments)
    assert error.value.code == 2
    gate = json.loads(output.read_text())
    assert gate["valid"] is True
    assert gate["mechanism_gate_passed"] is False
    assert gate["decision"] == "STOP_BEFORE_FULLCHAIN"
    assert gate["stage_b"]["gates"]["selected_probe_rows_reused"] is True
