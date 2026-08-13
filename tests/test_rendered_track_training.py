from pathlib import Path

import torch

from scripts.audit_rendered_track_positive_ranks import positive_rank_hits
from scripts.materialize_rendered_track_training import materialize
from scripts.evaluate_rendered_track_multiprototype_crossfit import (
    _fold_map,
    expand_teacher,
)
from scripts.train_rendered_track_crossfit import _training_inputs


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


def test_crossfit_training_inputs_exclude_held_mapping_sequence():
    teacher = {
        "query_names": ["seq-02/a", "seq-03/b", "seq-05/c"],
        "records": [
            {
                "query_index": index,
                "query_name": name,
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([0]),
                "ambiguous_offsets": torch.tensor([0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
            for index, name in enumerate(("seq-02/a", "seq-03/b", "seq-05/c"))
        ],
        "diagnostics": {
            "query_count": 3,
            "positive_rows": 3,
            "strong_pair_count": 3,
            "ambiguous_pair_count": 0,
        },
    }
    revised, graph, payload = _training_inputs(
        held_sequence="seq-03", fold_teacher=teacher, full_payload={}
    )
    assert revised["query_names"] == ["seq-02/a", "seq-05/c"]
    assert graph["query_names"] == revised["query_names"]
    assert payload["query_names"] == revised["query_names"]
    assert payload["query_bins"].tolist() == [0, 1]
    assert payload["uses_test_queries"] is False
    assert revised["crossfit"]["held_mapping_sequence"] == "seq-03"


def test_multi_prototype_teacher_expansion_preserves_row_semantics():
    teacher = {
        "anchor_count": 2,
        "records": [
            {
                "positive_offsets": torch.tensor([0, 2, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 1]),
                "ambiguous_indices": torch.tensor([1]),
            }
        ],
        "diagnostics": {
            "positive_rows": 1,
            "strong_pair_count": 2,
            "ambiguous_pair_count": 1,
        },
    }
    revised = expand_teacher(teacher, [[0, 1], []])
    record = revised["records"][0]
    assert revised["anchor_count"] == 2
    assert record["positive_offsets"].tolist() == [0, 2, 2]
    assert record["positive_indices"].tolist() == [0, 1]
    assert record["ambiguous_offsets"].tolist() == [0, 0, 0]
    assert record["ambiguous_indices"].numel() == 0
    assert revised["diagnostics"]["positive_rows"] == 1
    assert revised["diagnostics"]["strong_pair_count"] == 2


def test_multi_prototype_fold_excludes_held_sequence_descriptors():
    names = ["seq-02/a", "seq-03/b", "seq-05/c", "seq-05/d"]
    cache = {
        "queries": {
            names[0]: {"native_descriptors": torch.tensor([[1.0, 0.0]])},
            names[1]: {"native_descriptors": torch.tensor([[0.0, 1.0]])},
            names[2]: {"native_descriptors": torch.tensor([[-1.0, 0.0]])},
            names[3]: {"native_descriptors": torch.tensor([[0.0, -1.0]])},
        }
    }
    payload = {
        "query_names": names,
        "query_bins": torch.tensor([0, 1, 2, 3]),
        "tracks": {
            "track_index": torch.tensor([0, 0, 0, 1]),
            "query_index": torch.arange(4),
            "keypoint_index": torch.zeros(4, dtype=torch.long),
            "confidence": torch.ones(4),
        },
    }
    state = {
        "anchor_ids": torch.arange(2),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        "anchor_features": torch.eye(2),
        "track_cluster_ids": torch.arange(2),
    }
    fold, rows = _fold_map(
        state=state,
        payload=payload,
        cache_payload=cache,
        held_sequence="seq-05",
        maximum_prototypes=2,
        trim_fraction=0.0,
    )
    assert fold["prototype_source_anchor"].tolist() == [0, 0]
    assert fold["prototype_view_bin"].tolist() == [0, 1]
    assert rows == [[0, 1], []]
    assert torch.equal(fold["anchor_features"], torch.eye(2))
    assert fold["provenance"]["multi_prototype_crossfit"]["uses_test_queries"] is False


def test_positive_rank_hits_uses_row_aligned_csr():
    ranks, has_positive = positive_rank_hits(
        torch.tensor([[5, 2, 1], [4, 3, 2], [1, 0, 2]]),
        torch.tensor([0, 2, 2, 3]),
        torch.tensor([1, 8, 0]),
    )
    assert ranks.tolist() == [3, 4, 2]
    assert has_positive.tolist() == [True, False, True]
