from pathlib import Path

import torch

from localization_training.shared_metric import (
    NativeNullHead,
    SharedLowRankMetric,
    build_native_null_features,
    select_native_matchable_rows,
)
from scripts.build_lafgs_v7_track_centric_maps import (
    _align_query_values,
    _eligible_tracks,
    _group_balanced_base_utility,
    _normalized_log_score,
    _voxel_diverse_order,
)
from scripts.build_lafgs_v9_complete_positive_teacher import (
    _deduplicated_csr,
    _exact_track_observations,
    _expand_provenance_candidates,
    _source_anchor_lookup,
    _query_index_remap,
)
from scripts.build_lafgs_v9_minimum_sufficient_maps import (
    _event_id,
    _positive_events_by_anchor,
    greedy_query_multicover,
)
from scripts.train_lafgs_v7_online_metric import (
    _build_rotating_shards,
    _bounded_anchor_features,
    _csr_first_k,
    _group_pose_risk,
    _multi_positive_list_loss,
    _replace_refreshed_pairs,
    _save_checkpoint,
    _query_index_remap as _training_query_index_remap,
)


def test_shared_metric_starts_as_identity_and_is_bounded():
    metric = SharedLowRankMetric(
        descriptor_dim=4, rank=2, max_residual_norm=0.05
    )
    value = torch.randn(5, 4)
    output, residual = metric(value)
    torch.testing.assert_close(output, torch.nn.functional.normalize(value))
    assert float(residual.norm(dim=1).max()) == 0.0
    with torch.no_grad():
        metric.up.weight.fill_(10.0)
    _, residual = metric(value)
    assert float(residual.norm(dim=1).max()) <= 0.050001


def test_voxel_order_takes_spatial_representatives_before_duplicates():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    order = _voxel_diverse_order(
        xyz, torch.tensor([4.0, 3.0, 2.0]), voxel_size=1.0
    )
    assert order.tolist() == [0, 2, 1]
    assert torch.isfinite(_normalized_log_score(torch.zeros(3))).all()


def test_relaxed_track_tier_is_a_superset_of_broad():
    geometry = {
        "triangulated": torch.ones(3, dtype=torch.bool),
        "triangulated_xyz": torch.zeros(3, 3),
        "triangulation_distinct_view_bin_count": torch.full((3,), 2),
        "triangulation_reprojection_median_px": torch.tensor([1.0, 5.0, 7.0]),
        "triangulation_reprojection_p90_px": torch.tensor([2.0, 30.0, 50.0]),
        "triangulation_covariance_trace": torch.tensor([0.1, 2.0, 4.0]),
        "triangulation_parallax_deg": torch.tensor([2.0, 0.3, 0.2]),
    }
    broad = _eligible_tracks(geometry, "broad")
    relaxed = _eligible_tracks(geometry, "relaxed")
    assert bool((broad & ~relaxed).any()) is False
    assert int(relaxed.sum()) > int(broad.sum())


def test_group_balanced_base_utility_rewards_rare_group_support():
    graph = {
        "records": [
            {
                "query_index": 0,
                "top_indices": torch.tensor([[0, 1]]),
                "legal_flags": torch.tensor([[0, 2]], dtype=torch.uint8),
            },
            {
                "query_index": 1,
                "top_indices": torch.tensor([[0, 1]]),
                "legal_flags": torch.tensor([[2, 0]], dtype=torch.uint8),
            },
        ]
    }
    score, report = _group_balanced_base_utility(
        graph, torch.tensor([0, 1]), 2, torch.zeros(2)
    )
    assert bool((score > 0).all())
    assert report["group_count"] == 2
    aligned = _align_query_values(
        torch.tensor([10, 20]), ["b", "a"], ["a", "b"]
    )
    assert aligned.tolist() == [20, 10]


def test_complete_teacher_inverts_to_anchor_events():
    teacher = {
        "records": [
            {
                "query_index": 0,
                "query_rows": torch.tensor([3, 7]),
                "positive_offsets": torch.tensor([0, 2, 3]),
                "positive_indices": torch.tensor([0, 1, 1]),
            }
        ]
    }
    events = _positive_events_by_anchor(teacher, 2)
    assert events[0] == {_event_id(0, 3)}
    assert events[1] == {_event_id(0, 3), _event_id(0, 7)}


def test_query_multicover_uses_shared_rescue_and_stops_at_constraint():
    events = [
        {_event_id(0, 0), _event_id(1, 0)},
        {_event_id(0, 1)},
        {_event_id(1, 1)},
    ]
    selected, report = greedy_query_multicover(
        events,
        set(),
        torch.tensor([0, 1]),
        minimum_rows_per_query=1,
        utility=torch.tensor([0.0, 10.0, 10.0]),
    )
    assert selected.tolist() == [0]
    assert report["reserve_count"] == 1
    assert report["unmet_query_count"] == 0


def test_listwise_loss_rewards_any_positive_and_penalizes_harmful_mass():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]])
    positives = torch.tensor([[0, -1]])
    clean, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=1.0,
        harmful_indices=torch.full((1, 1), -1),
    )
    harmful, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=1.0,
        harmful_indices=torch.tensor([[1]]),
    )
    assert float(harmful) > float(clean)


def test_listwise_pose_critical_weight_prefers_critical_positive():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[0.8, 0.6], [0.8, -0.6], [-1.0, 0.0]])
    positives = torch.tensor([[0, 1]])
    first, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=0.0,
        positive_weights=torch.tensor([[4.0, 0.25]]),
    )
    uniform, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=0.0,
        positive_weights=torch.ones(1, 2),
    )
    assert torch.isfinite(first).all()
    assert float(first) < float(uniform)


def test_anchor_residual_is_bounded():
    raw = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    adapted, residual = _bounded_anchor_features(
        raw, torch.full_like(raw, 10.0), maximum=0.02
    )
    assert float(residual.norm(dim=1).max()) <= 0.020001
    torch.testing.assert_close(adapted.norm(dim=1), torch.ones(4))


def test_metric_frontend_uses_native_sparse_dispatch():
    source = (Path(__file__).parents[1] / "stdloc.py").read_text()
    native_dispatch = source.split(
        "sparse_result = self.loc_sparse_ulfloc_native", 1
    )[0].rsplit("def localize(", 1)[-1]
    assert '"ulfloc_native_metric"' in native_dispatch
    assert "select_native_matchable_rows" in source


def test_null_features_and_spatial_floor_preserve_candidates():
    scores = torch.tensor(
        [[0.9, 0.8, 0.1], [0.7, 0.69, 0.2], [0.8, 0.1, 0.0]]
    )
    features = build_native_null_features(
        scores, torch.tensor([0.5, 0.4, 0.3])
    )
    assert features.shape == (3, 4)
    assert NativeNullHead()(features).shape == (3,)
    kept = select_native_matchable_rows(
        torch.tensor([0.1, 0.2, 0.9]),
        torch.tensor([[5.0, 5.0], [6.0, 6.0], [90.0, 90.0]]),
        width=100,
        height=100,
        threshold=0.8,
        minimum_total=2,
        grid_rows=2,
        grid_cols=2,
        minimum_per_cell=1,
    )
    assert set(kept.tolist()) == {1, 2}


def test_rotating_shards_cover_all_groups_once():
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    shards = _build_rotating_shards(groups, 3)
    assert sorted(index for shard in shards for index in shard) == list(
        range(groups.numel())
    )
    assert all(len(shard) > 0 for shard in shards)
    assert _group_pose_risk([1.0, 2.0, 3.0]) < _group_pose_risk(
        [1.0, 2.0, 30.0]
    )


def test_metric_checkpoint_round_trips_null_head(tmp_path):
    metric = SharedLowRankMetric(descriptor_dim=4, rank=2)
    null_head = NativeNullHead()
    raw = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    _save_checkpoint(
        output_dir=tmp_path,
        step=5,
        state={"anchor_features": raw},
        metric=metric,
        null_head=null_head,
        raw_features=raw,
        anchor_residual=torch.zeros_like(raw),
        maximum_anchor_residual=0.02,
        history=[],
        config={
            "null_temperature": 0.05,
            "null_threshold": 0.5,
            "null_minimum_total": 2,
            "null_grid_rows": 2,
            "null_grid_cols": 2,
            "null_minimum_per_cell": 1,
        },
    )
    state = torch.load(
        tmp_path / "metric_state_step_0005.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["null_head_config"]["feature_dim"] == 4
    assert set(state["null_head_state_dict"]) == {
        "linear.weight",
        "linear.bias",
    }


def test_complete_teacher_expands_one_source_to_many_anchors():
    sources, lookup = _source_anchor_lookup(torch.tensor([5, 5, 8]))
    assert sources.tolist() == [5, 8]
    assert lookup.tolist() == [[0, 1], [2, -1]]
    candidates, valid = _expand_provenance_candidates(
        torch.tensor([[5, 8, 9]]),
        torch.tensor([[0.8, 0.2, 1.0]]),
        sources,
        lookup,
        minimum_mass=0.1,
    )
    assert set(candidates[valid].tolist()) == {0, 1, 2}
    offsets, indices = _deduplicated_csr(
        torch.tensor([[2, 1, 1, -1]]),
        torch.tensor([[True, True, True, False]]),
        value_count=3,
    )
    assert offsets.tolist() == [0, 2]
    assert indices.tolist() == [1, 2]
    dense = _csr_first_k(offsets, indices, width=3)
    assert dense.tolist() == [[1, 2, -1]]


def test_exact_track_observations_preserve_multiple_positives():
    payload = {
        "tracks": {
            "track_index": torch.tensor([2, 3]),
            "query_index": torch.tensor([0, 0]),
            "keypoint_index": torch.tensor([7, 7]),
        }
    }
    exact = _exact_track_observations(payload, torch.tensor([2, 3]))
    assert exact[0][7] == [0, 1]


def test_exact_track_observations_use_nonprefix_anchor_rows():
    payload = {
        "tracks": {
            "track_index": torch.tensor([2]),
            "query_index": torch.tensor([0]),
            "keypoint_index": torch.tensor([7]),
        }
    }
    exact = _exact_track_observations(
        payload, torch.tensor([2]), torch.tensor([11])
    )
    assert exact[0][7] == [11]


def test_exact_track_observations_remap_query_order_by_name():
    payload = {
        "tracks": {
            "track_index": torch.tensor([2]),
            "query_index": torch.tensor([0]),
            "keypoint_index": torch.tensor([7]),
        }
    }
    remap = _query_index_remap(["b", "a"], ["a", "b"])
    exact = _exact_track_observations(
        payload,
        torch.tensor([2]),
        query_index_remap=remap,
    )
    assert exact[1][7] == [0]
    torch.testing.assert_close(
        _training_query_index_remap(["b", "a"], ["a", "b"]),
        torch.tensor([1, 0]),
    )


def test_refresh_replaces_stale_pair_labels():
    clean = {0: {1: 2}, 1: {3: 4}}
    harmful = {0: {5: 6}, 1: {7: 8}}
    report = _replace_refreshed_pairs(
        clean,
        harmful,
        [0],
        refreshed_clean={},
        refreshed_harmful={0: {9: 10}},
    )
    assert 0 not in clean
    assert harmful[0] == {9: 10}
    assert clean[1] == {3: 4}
    assert report["old_clean_pair_count"] == 1
    assert report["new_clean_pair_count"] == 0
