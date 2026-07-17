import pickle
import tempfile

import torch
from types import SimpleNamespace

from train_coreset_v2 import _detect_query, _nearest_reprojection_labels, _save_atom_state


class _FixedDetector(torch.nn.Module):
    def forward(self, feature_map):
        return feature_map[:1]


def test_detect_query_excludes_invalid_feature_cells():
    feature_map = torch.tensor(
        [
            [[0.1, 10.0], [0.8, 0.2]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    valid = torch.tensor([[True, False], [True, False]])
    xy, descriptors, scores = _detect_query(
        feature_map,
        _FixedDetector(),
        count=2,
        nms_radius=0,
        valid_feature_mask=valid,
    )
    assert xy.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    torch.testing.assert_close(scores, torch.tensor([0.8, 0.1]))
    assert descriptors.shape == (2, 2)


def test_reprojection_positive_rejects_occluded_landmark_by_rendered_depth():
    camera = SimpleNamespace(
        FoVx=torch.pi / 2,
        FoVy=torch.pi / 2,
        world_view_transform=torch.eye(4),
    )
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    render_pkg = {
        "depth": torch.ones(1, 10, 10),
        "visibility_filter": torch.tensor([True, True]),
        "rgb_meta": {"depths": torch.tensor([[1.0, 2.0]])},
    }
    raw_ids, valid = _nearest_reprojection_labels(
        xyz,
        torch.tensor([0, 1]),
        camera,
        torch.tensor([[5.0, 5.0]]),
        render_pkg,
        width=10,
        height=10,
        radius=1.0,
    )
    assert valid.tolist() == [True]
    assert raw_ids.tolist() == [0]

    raw_ids, valid = _nearest_reprojection_labels(
        xyz,
        torch.tensor([1]),
        camera,
        torch.tensor([[5.0, 5.0]]),
        render_pkg,
        width=10,
        height=10,
        radius=1.0,
    )
    assert valid.tolist() == [False]
    assert raw_ids.tolist() == [-1]


def test_saved_candidate_quality_aligns_with_active_bank():
    with tempfile.TemporaryDirectory() as output_dir:
        _save_atom_state(
            output_dir,
            torch.nn.Parameter(torch.eye(4)),
            torch.eye(4),
            torch.tensor([10, 20, 30, 40]),
            torch.arange(4),
            torch.arange(4),
            torch.arange(4),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            torch.tensor([1, 3]),
            {},
            [],
            {},
        )
        meta = torch.load(f"{output_dir}/landmark_meta.pt", map_location="cpu")
        assert meta["landmark_indices"].tolist() == [20, 40]
        torch.testing.assert_close(
            meta["candidate_quality"].float(), torch.tensor([0.2, 0.4]), rtol=1e-3, atol=1e-3
        )
        with open(f"{output_dir}/sampled_idx.pkl", "rb") as handle:
            assert pickle.load(handle).tolist() == [20, 40]
