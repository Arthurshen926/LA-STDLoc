import json
import sys

import pytest

from common.hashing import sha256_file
from scripts.compare_track_pair_factor import main


PARAMETERS = {
    "minimum_overlap_jaccard": 0.15,
    "minimum_joint_visibility_points": 8,
    "parallax_saturation_deg": 2.0,
    "diversity_weight": 0.2,
    "candidate_pool_per_camera": 48,
    "scene_points_per_camera": 8,
    "maximum_scene_points": 4096,
    "scene_point_voxel_size_m": 0.02,
}


def _report(tmp_path, policy, *, low, broad, triangulated, covariance, support):
    inputs = {}
    for name in ("manifest", "query_cache", "frozen_track_payload"):
        path = tmp_path / name
        if not path.exists():
            path.write_bytes(name.encode())
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    artifact = tmp_path / f"{policy}.pt"
    artifact.write_bytes(policy.encode())
    return {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 2048,
        "mapping_nms_radius": 4,
        "pair_policy": policy,
        "pair_policy_parameters": PARAMETERS,
        "exact_pair_budget": 5254,
        "mapping_query_count": 1531,
        "query_names_sha256": "1" * 64,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "inputs": inputs,
        "pair": {
            "pair_count": 5254,
            "mapping_point_parallax_below_1deg_fraction": low,
            "mapping_point_parallax_median_deg": {"median": 2.0},
        },
        "track": {
            "triangulated_track_count": triangulated,
            "broad_eligible_track_count": broad,
            "triangulated_covariance_trace_m2": {"p90": covariance},
            "broad_track_support_per_mapping_query": {"p10": support},
        },
    }


def _argv(control, variant, output):
    inputs = control["inputs"]
    return [
        "compare_track_pair_factor",
        "--control",
        str(control["path"]),
        "--expected-control-sha256",
        control["sha"],
        "--variant",
        str(variant["path"]),
        "--expected-variant-sha256",
        variant["sha"],
        "--expected-mapping-keypoints",
        "2048",
        "--expected-nms-radius",
        "4",
        "--expected-pair-budget",
        "5254",
        "--expected-query-count",
        "1531",
        "--expected-query-names-sha256",
        "1" * 64,
        "--expected-manifest-sha256",
        inputs["manifest"]["sha256"],
        "--expected-query-cache-sha256",
        inputs["query_cache"]["sha256"],
        "--expected-frozen-track-payload-sha256",
        inputs["frozen_track_payload"]["sha256"],
        "--minimum-overlap-jaccard",
        "0.15",
        "--minimum-joint-visibility-points",
        "8",
        "--parallax-saturation-deg",
        "2.0",
        "--diversity-weight",
        "0.2",
        "--candidate-pool-per-camera",
        "48",
        "--scene-points-per-camera",
        "8",
        "--maximum-scene-points",
        "4096",
        "--scene-point-voxel-size-m",
        "0.02",
        "--output",
        str(output),
    ]


def _write(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return {"path": path, "sha": sha256_file(path), **payload}


def test_mechanism_gate_binds_inputs_and_exits_nonzero_on_stop(tmp_path, monkeypatch):
    control = _write(
        tmp_path,
        "control",
        _report(
            tmp_path,
            "nearest",
            low=0.30,
            broad=100,
            triangulated=100,
            covariance=1.0,
            support=10,
        ),
    )
    variant = _write(
        tmp_path,
        "variant",
        _report(
            tmp_path,
            "parallax_diverse",
            low=0.15,
            broad=100,
            triangulated=100,
            covariance=1.0,
            support=10,
        ),
    )
    output = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", _argv(control, variant, output))
    main()
    assert json.loads(output.read_text())["mechanism_gate_passed"] is True

    failed = dict(variant)
    failed["track"] = dict(variant["track"])
    failed["track"]["broad_eligible_track_count"] = 1
    failed = _write(
        tmp_path,
        "failed",
        {k: v for k, v in failed.items() if k not in {"path", "sha"}},
    )
    monkeypatch.setattr(
        sys, "argv", _argv(control, failed, tmp_path / "failed_gate.json")
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_mechanism_gate_rejects_relabelled_cache(tmp_path, monkeypatch):
    control = _write(
        tmp_path,
        "control",
        _report(
            tmp_path,
            "nearest",
            low=0.3,
            broad=100,
            triangulated=100,
            covariance=1,
            support=10,
        ),
    )
    payload = _report(
        tmp_path,
        "parallax_diverse",
        low=0.1,
        broad=100,
        triangulated=100,
        covariance=1,
        support=10,
    )
    payload["inputs"] = dict(payload["inputs"])
    payload["inputs"]["query_cache"] = dict(payload["inputs"]["query_cache"])
    payload["inputs"]["query_cache"]["sha256"] = "0" * 64
    variant = _write(tmp_path, "variant", payload)
    monkeypatch.setattr(sys, "argv", _argv(control, variant, tmp_path / "gate.json"))
    with pytest.raises(ValueError, match="query_cache input differs"):
        main()
