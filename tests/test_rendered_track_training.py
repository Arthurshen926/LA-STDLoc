from pathlib import Path

import torch

from scripts.materialize_rendered_track_training import materialize


def _inputs(tmp_path: Path):
    names = ["seq-02/a.png", "seq-03/b.png", "seq-05/c.png"]
    queries = {}
    poses = []
    for index, name in enumerate(names):
        pose = torch.eye(4)
        pose[0, 3] = -0.1 * index
        poses.append(pose)
        queries[name] = {
            "native_keypoints": torch.tensor([[50.0 - 5.0 * index, 50.0]]),
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_K": torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
            "pixel_center_offset": 0.0,
        }
    cache = {
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": queries,
    }
    track_payload = {
        "query_names": names,
        "query_bins": torch.arange(3),
        "tracks": {
            "track_index": torch.zeros(3, dtype=torch.long),
            "query_index": torch.arange(3),
            "keypoint_index": torch.zeros(3, dtype=torch.long),
        },
    }
    anchor_map = {
        "anchor_ids": torch.tensor([0]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0]]),
        "track_cluster_ids": torch.tensor([0]),
    }
    paths = {}
    for key, value in (
        ("cache", cache),
        ("track", track_payload),
        ("map", anchor_map),
    ):
        paths[key] = tmp_path / f"{key}.pt"
        torch.save(value, paths[key])
    return paths


def test_rendered_track_training_materializes_geometry_teacher(tmp_path):
    paths = _inputs(tmp_path)
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "out",
        strong_radius_px=2.0,
        ambiguous_radius_px=8.0,
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    payload = torch.load(
        report["outputs"]["track_payload"],
        map_location="cpu",
        weights_only=False,
    )
    assert teacher["diagnostics"]["positive_rows"] == 3
    assert teacher["diagnostics"]["exact_track_positive_count"] == 3
    assert all(
        record["positive_indices"].tolist() == [0] for record in teacher["records"]
    )
    assert payload["training_sequence_names"] == ["seq-02", "seq-03", "seq-05"]
    assert payload["query_bins"].tolist() == [0, 1, 2]
    assert report["uses_source_mapping_rgb"] is False
    assert report["uses_test_queries"] is False


def test_rendered_track_training_rejects_source_rgb_or_test_cache(tmp_path):
    for field in ("uses_source_mapping_rgb", "uses_test_queries"):
        case = tmp_path / field
        case.mkdir()
        paths = _inputs(case)
        cache = torch.load(paths["cache"], map_location="cpu", weights_only=False)
        cache[field] = True
        torch.save(cache, paths["cache"])
        try:
            materialize(
                anchor_map_path=paths["map"],
                track_payload_path=paths["track"],
                query_cache_path=paths["cache"],
                output_dir=case / "out",
                strong_radius_px=2.0,
                ambiguous_radius_px=8.0,
            )
        except ValueError as error:
            assert "rendered-RGB-only" in str(error) or "test queries" in str(error)
        else:
            raise AssertionError("forbidden training source must fail closed")
