from types import SimpleNamespace

import torch

import evidence.track_provenance_assignment as provenance
from scripts.replay_track_provenance_assignment import (
    _assignment_dependent_diagnostics,
    _camera_names_sha256,
    _validate_factor_contract,
)
from scripts.audit_track_payload_parity import audit_payload_parity


def test_replayable_provenance_assignment_emits_complete_group_contract(monkeypatch):
    def fake_render(*args, **kwargs):
        return {"rgb_meta": object(), "depth": torch.ones((4, 4))}

    def fake_provenance(keypoints, *args, topk, **kwargs):
        assert topk == 2
        rows = keypoints.shape[0]
        return (
            torch.tensor([[0, 1]], dtype=torch.long).repeat(rows, 1),
            torch.tensor([[0.8, 0.3]], dtype=torch.float32).repeat(rows, 1),
            torch.ones(rows, dtype=torch.bool),
        )

    monkeypatch.setattr(provenance, "render_from_pose_gsplat", fake_render)
    monkeypatch.setattr(provenance, "bank_splat_provenance_2dgs", fake_provenance)
    tracks = {
        "track_index": torch.tensor([0, 0]),
        "query_index": torch.tensor([0, 1]),
        "keypoint_index": torch.tensor([0, 0]),
    }
    track_geometry = {
        "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
        "triangulated": torch.tensor([True]),
        "triangulation_high_confidence": torch.tensor([True]),
        "triangulation_parallax_deg": torch.tensor([2.0]),
    }
    cache = {
        name: {
            "native_input_hw": [4, 4],
            "pose_w2c": torch.eye(4),
        }
        for name in ("q0", "q1")
    }
    cameras = {
        name: SimpleNamespace(FoVx=1.0, FoVy=1.0) for name in cache
    }
    geometry, assignment, diagnostics = (
        provenance.assign_tracks_by_splat_provenance(
            tracks=tracks,
            track_geometry=track_geometry,
            keypoints=[torch.tensor([[1.5, 1.5]])] * 2,
            query_names=["q0", "q1"],
            cache=cache,
            bank_xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            gaussians=object(),
            cameras_by_name=cameras,
            landmark_global_indices=torch.tensor([7, 9]),
            background=torch.ones(3),
            topk=2,
            minimum_consensus_rate=0.35,
            minimum_views=2,
            group_maximum_landmarks=4,
            group_minimum_relative_mass=0.25,
            group_minimum_consensus_rate=0.10,
            depth_absolute_tolerance_m=0.05,
            depth_relative_tolerance=0.02,
        )
    )
    assert set(assignment) == {
        "track_landmark_index",
        "track_assignment_cost",
        "landmark_best_track_index",
        "track_landmark_offsets",
        "track_landmark_indices",
        "track_landmark_costs",
    }
    assert assignment["track_landmark_index"].tolist() == [0]
    assert assignment["track_landmark_offsets"].tolist() == [0, 2]
    assert assignment["track_landmark_indices"].tolist() == [0, 1]
    assert torch.allclose(
        assignment["track_landmark_costs"], torch.tensor([0.2, 0.7])
    )
    assert diagnostics == {
        "geometry_teacher_provenance_valid_observation_count": 2,
        "geometry_teacher_provenance_assigned_track_count": 1,
        "geometry_teacher_provenance_assigned_landmark_count": 2,
        "geometry_teacher_provenance_group_assigned_track_count": 1,
        "geometry_teacher_provenance_group_edge_count": 2,
        "geometry_teacher_provenance_group_size_mean": 2.0,
    }
    assert geometry["track_assigned"].tolist() == [True, True]
    assert torch.allclose(
        geometry["track_provenance_consensus_rate"], torch.tensor([0.8, 0.3])
    )


def test_replay_diagnostics_and_camera_hash_match_bootstrap_semantics():
    tracks = {
        "track_index": torch.tensor([0, 0, 1]),
        "query_index": torch.tensor([0, 1, 1]),
    }
    track_geometry = {
        "triangulated": torch.tensor([True, True]),
        "triangulation_high_confidence": torch.tensor([True, False]),
    }
    landmark_geometry = {
        "track_assigned": torch.tensor([True, True, False]),
        "triangulation_high_confidence": torch.tensor([True, True, False]),
    }
    assignment = {
        "track_landmark_offsets": torch.tensor([0, 2, 2]),
        "track_landmark_indices": torch.tensor([0, 1]),
    }
    diagnostics = _assignment_dependent_diagnostics(
        tracks=tracks,
        track_geometry=track_geometry,
        landmark_geometry=landmark_geometry,
        assignment=assignment,
        query_count=2,
        provenance_diagnostics={
            "geometry_teacher_provenance_group_edge_count": 2
        },
    )
    assert diagnostics["geometry_teacher_query_support_edge_count"] == 4
    assert diagnostics["geometry_teacher_assigned_landmark_count"] == 2
    assert (
        _camera_names_sha256(["b", "a"])
        == _camera_names_sha256(["a", "b"])
    )


def test_control_parity_requires_all_six_assignment_fields_and_query_contract():
    six = {
        "track_landmark_index": torch.tensor([0]),
        "track_assignment_cost": torch.tensor([0.2]),
        "landmark_best_track_index": torch.tensor([0]),
        "track_landmark_offsets": torch.tensor([0, 1]),
        "track_landmark_indices": torch.tensor([0]),
        "track_landmark_costs": torch.tensor([0.2]),
    }
    payload = {
        "schema": "lafgs_track_first_payload",
        "version": 1,
        "assignment": six,
        "tracks": {"track_index": torch.tensor([0])},
        "track_geometry": {"triangulated_xyz": torch.tensor([[0.0, 0.0, 1.0]])},
        "diagnostics": {
            "geometry_teacher_provenance_assigned_track_count": 1,
            "geometry_teacher_query_support_edge_count": 2,
        },
        "query_names": ["q0"],
        "query_bins": torch.tensor([0]),
        "train_camera_names_sha256": "hash",
        "landmark_indices": torch.tensor([9]),
    }
    assert audit_payload_parity(payload, payload, float_atol=1e-7)["valid"]
    incomplete = {**payload, "assignment": dict(six)}
    incomplete["assignment"].pop("track_landmark_costs")
    assert not audit_payload_parity(
        payload, incomplete, float_atol=1e-7
    )["valid"]


def test_replay_rejects_factor_owned_assignment():
    factor = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "density_factor_mutated": False,
        "descriptor_factor_mutated": False,
        "selector_factor_mutated": False,
        "assignment": {},
    }
    try:
        _validate_factor_contract(factor)
    except ValueError as error:
        assert "must never be consumed" in str(error)
    else:
        raise AssertionError("factor-owned assignment must fail closed")
