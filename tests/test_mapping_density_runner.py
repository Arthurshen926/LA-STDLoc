import pytest

from scripts.run_mapping_density_track_factor import density_track_command


def _manifest(extra=()):
    return {
        "command": [
            "/old/python",
            "/old/bootstrap.py",
            "--query_cache_path",
            "/old/cache.pt",
            "--query_cache_policy",
            "readonly",
            "--output_dir",
            "/old/out",
            "--native_keypoint_count",
            "1024",
            "--native_nms_radius",
            "4",
            "--max_observations",
            "1024",
            "--validation_observations",
            "1024",
            "--save_track_micro_anchor_payload",
            "--steps",
            "0",
            *extra,
        ]
    }


def _value(command, flag):
    return command[command.index(flag) + 1]


def test_density_runner_changes_only_density_artifacts_and_limits(tmp_path):
    command = density_track_command(
        _manifest(),
        query_cache=tmp_path / "cache.pt",
        output_dir=tmp_path / "out",
        mapping_keypoints=2048,
        nms_radius=4,
        python="/new/python",
    )
    assert command[0] == "/new/python"
    assert command[1:3] == ["-m", "map_learning.bootstrap"]
    assert _value(command, "--query_cache_policy") == "readonly"
    assert _value(command, "--native_keypoint_count") == "2048"
    assert _value(command, "--native_nms_radius") == "4"
    assert _value(command, "--max_observations") == "2048"
    assert _value(command, "--validation_observations") == "2048"


def test_density_runner_rejects_pair_policy_mutation(tmp_path):
    with pytest.raises(ValueError, match="pair-policy"):
        density_track_command(
            _manifest(("--geometry_teacher_track_pair_policy", "parallax_stratified")),
            query_cache=tmp_path / "cache.pt",
            output_dir=tmp_path / "out",
            mapping_keypoints=2048,
            nms_radius=4,
            python="python",
        )
