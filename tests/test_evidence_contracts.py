from types import SimpleNamespace

import torch

from data.splits import split_support_query_cameras
from evidence.evidence_graph import _same_anchor_map
from map_learning.bootstrap import _build_query_cache, _cache_payload_compatible
from map_learning.bootstrap import build_parser as build_bootstrap_parser
from map_learning.pipeline import build_bootstrap_and_tracks
from priors.rasterizer import anchor_source_csr


def test_mapping_support_query_split_is_disjoint_and_complete():
    cameras = [
        SimpleNamespace(image_name=f"sequence{index // 5}/frame{index:03d}.png")
        for index in range(20)
    ]
    support, query = split_support_query_cameras(
        cameras,
        query_ratio=0.25,
        seed=2026,
        mode="stratified_temporal_block",
    )
    support_names = {camera.image_name for camera in support}
    query_names = {camera.image_name for camera in query}
    assert support_names.isdisjoint(query_names)
    assert support_names | query_names == {camera.image_name for camera in cameras}
    assert len(query) == 5


def test_query_cache_contract_rejects_prior_or_resolution_change():
    payload = {
        "version": 11,
        "query_feature_contract": "native_resized_input",
        "feature_resize_mode": "resize_image_then_native_stride8",
        "descriptor_source": "superpoint_native_dense_resized_input",
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "pixel_center_offset": 0.5,
        "valid_mask_policy": "object_sky_distortion_and_bounds_v1",
        "model_path": "/prior",
        "rgb_prior_fingerprint": {"geometry_sha256": "a"},
        "source_path": "/dataset",
        "load_iteration": 30000,
        "feature_type": "sp",
        "images": "processed",
        "resolution": 1,
        "longest_edge": 0,
        "white_background": True,
        "norm_before_render": True,
        "native_sparse_enabled": True,
        "native_sparse_keypoint_count": 2048,
        "native_sparse_nms_radius": 4,
        "native_sparse_coordinate_convention": (
            "superpoint_grid_index_then_pnp_plus_half_v1"
        ),
    }
    assert _cache_payload_compatible(dict(payload), payload)
    changed = dict(payload)
    changed["rgb_prior_fingerprint"] = {"geometry_sha256": "b"}
    assert not _cache_payload_compatible(payload, changed)
    changed = dict(payload)
    changed["longest_edge"] = 1024
    assert not _cache_payload_compatible(payload, changed)
    changed = dict(payload)
    changed["native_sparse_nms_radius"] = 2
    assert not _cache_payload_compatible(payload, changed)


def test_query_cache_attests_mapping_k_and_nms_per_query(monkeypatch):
    camera = SimpleNamespace(
        image_name="seq/frame.png",
        FoVx=1.0,
        FoVy=1.0,
        world_view_transform=torch.eye(4),
    )

    class Extractor:
        nms_radius = 4

        def detectAndCompute(self, image, top_k):
            assert top_k == 2048
            return [{
                "keypoints": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                "descriptors": torch.ones(2, 4),
                "keypoint_scores": torch.ones(2),
            }]

    monkeypatch.setattr(
        "map_learning.bootstrap._query_feature_outputs",
        lambda *args, **kwargs: (
            torch.ones(4, 2, 2),
            None,
            torch.ones(2, 2, dtype=torch.bool),
            {},
        ),
    )
    monkeypatch.setattr(
        "map_learning.bootstrap._native_feature_input",
        lambda *args, **kwargs: (
            torch.ones(3, 4, 4),
            torch.ones(4, 4, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        "map_learning.bootstrap._render_depth_alpha",
        lambda *args, **kwargs: (torch.ones(4, 4), torch.ones(4, 4)),
    )
    cache = _build_query_cache(
        [camera],
        object(),
        Extractor(),
        None,
        torch.zeros(3),
        True,
        0,
        include_native_sparse=True,
        native_keypoint_count=2048,
    )
    metadata = cache["seq/frame.png"]["native_sparse_metadata"]
    assert metadata["detect_num"] == 2048
    assert metadata["requested_keypoint_count"] == 2048
    assert metadata["nms_radius"] == 4


def test_anchor_source_csr_preserves_one_source_per_base_anchor():
    state = {
        "anchor_xyz": torch.zeros(3, 3),
        "anchor_ids": torch.tensor([8, 9, 10]),
        "source_primitive_ids": torch.tensor([2, 2, 7]),
        "track_cluster_ids": torch.full((3,), -1),
    }
    offsets, source_ids, weights = anchor_source_csr(state)
    assert offsets.tolist() == [0, 1, 2, 3]
    assert source_ids.tolist() == [2, 2, 7]
    torch.testing.assert_close(weights, torch.ones(3))


def test_pipeline_only_emits_supported_bootstrap_arguments(tmp_path, monkeypatch):
    calls = []

    def capture(module, *arguments):
        calls.append((module, [str(value) for value in arguments]))

    monkeypatch.setattr("map_learning.pipeline._run", capture)
    build_bootstrap_and_tracks(
        dataset=tmp_path / "dataset",
        prior=tmp_path / "prior",
        output=tmp_path / "output",
        gaussian_type="2dgs",
        sh_degree=3,
        config="configs/paper_mainline.yaml",
    )
    bootstrap_calls = [arguments for module, arguments in calls if module == "map_learning.bootstrap"]
    assert len(bootstrap_calls) == 3
    parser, _ = build_bootstrap_parser()
    for arguments in bootstrap_calls:
        parsed = parser.parse_args(arguments)
        assert parsed.query_feature_contract == "native_resized_input"
        assert parsed.initialization_mode == "ulf_robust_geometry"


def test_bootstrap_accepts_fractional_adaptive_pixel_radius():
    parser, _ = build_bootstrap_parser()
    parsed = parser.parse_args([
        "--source_path", "/dataset",
        "--model_path", "/prior",
        "--output_dir", "/output",
        "--gaussian_type", "2dgs",
        "--native_semidense_local_radius_px", "2.525053269801378",
    ])
    assert parsed.native_semidense_local_radius_px == 2.525053269801378


def test_evidence_graph_accepts_relocated_identical_anchor_map(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    left.write_bytes(b"identical anchor map")
    right.write_bytes(left.read_bytes())
    assert _same_anchor_map(str(left), str(right))
    right.write_bytes(b"different anchor map")
    assert not _same_anchor_map(str(left), str(right))
