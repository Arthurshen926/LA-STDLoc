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
    _select_capacity_limited_tracks,
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
    _basin_hyperedge_losses,
    _basin_set_log_scores,
    _build_rotating_shards,
    _bounded_anchor_features,
    _csr_first_k,
    _csr_topk_by_values,
    _group_pose_risk,
    _multi_positive_list_loss,
    _replace_refreshed_pairs,
    _save_checkpoint,
    _trajectory_stable_promotion_loss,
    _query_index_remap as _training_query_index_remap,
)


def test_basin_set_log_score_prefers_jointly_correct_triplet():
    bank = torch.eye(4)
    query = torch.stack((bank[0], bank[1], bank[2])).reshape(1, 3, 4)
    good = _basin_set_log_scores(
        query, bank, torch.tensor([[0, 1, 2]]), assignment_temperature=0.1
    )
    bad = _basin_set_log_scores(
        query, bank, torch.tensor([[3, 3, 3]]), assignment_temperature=0.1
    )
    assert good.item() > bad.item()


def test_basin_hyperedge_ranks_repaired_child_above_harmful_parent():
    bank = torch.eye(4)
    query = torch.stack(
        (bank[0], bank[1], bank[2], bank[0], bank[1], bank[2])
    ).reshape(2, 3, 4)
    contrastive, counterfactual, tiers = _basin_hyperedge_losses(
        query,
        bank,
        torch.tensor([[3, 3, 3], [0, 1, 2]]),
        torch.tensor([1, 2]),
        torch.tensor([False, True]),
        torch.tensor([100.0, 3.0]),
        torch.tensor([10.0, 1.0]),
        torch.tensor([-1, 0]),
        torch.ones(2),
        assignment_temperature=0.1,
        basin_temperature=1.0,
        counterfactual_temperature=0.25,
        counterfactual_margin=0.1,
        translation_reward_scale_cm=15.0,
        rotation_reward_scale_deg=2.0,
        maximum_inverse_propensity=100.0,
    )
    assert contrastive < 1e-3
    assert counterfactual < 1e-3
    assert tiers == {"coarse": 1, "precision": 1, "strict": 1}


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


def test_track_core_preserves_quality_gate_when_scene_capacity_is_small():
    selected, report = _select_capacity_limited_tracks(
        torch.tensor([0.1, 0.8, 0.4, 0.9]),
        torch.tensor([True, True, False, False]),
        requested_count=3,
    )
    assert selected.tolist() == [1, 0]
    assert report == {
        "requested_track_count": 3,
        "eligible_track_count": 2,
        "realized_track_count": 2,
        "capacity_limited": True,
    }


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


def test_listwise_loss_ignores_loose_radius_ambiguous_anchor():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[0.8, 0.6], [1.0, 0.0], [-1.0, 0.0]])
    positives = torch.tensor([[0]])
    ordinary, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=0.0,
    )
    ignored, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=0.0,
        ignored_indices=torch.tensor([[1]]),
    )
    assert float(ignored) < float(ordinary)


def test_listwise_pose_critical_weight_prefers_critical_positive():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
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


def test_basin_good_set_loss_rewards_joint_triplet_assignments():
    from scripts.train_lafgs_v7_online_metric import _basin_good_set_loss

    bank = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        dim=1,
    )
    anchors = torch.tensor([[0, 1, 2]])
    kwargs = {
        "topk": 4,
        "temperature": 0.1,
        "maximum_inverse_propensity": 10.0,
    }
    good = _basin_good_set_loss(
        bank[anchors], bank, anchors, torch.tensor([0.1]), **kwargs
    )
    bad = _basin_good_set_loss(
        bank[torch.tensor([[3, 3, 3]])],
        bank,
        anchors,
        torch.tensor([0.1]),
        **kwargs,
    )
    assert float(good) < float(bad)


def test_basin_blame_only_penalizes_counterfactual_harmful_edge():
    from scripts.train_lafgs_v7_online_metric import (
        _basin_counterfactual_blame_loss,
    )

    bank = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
    )
    query = bank[[1]]
    clean = _basin_counterfactual_blame_loss(
        query,
        bank,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([2.0]),
        temperature=0.1,
        margin=0.02,
    )
    harmful = _basin_counterfactual_blame_loss(
        query,
        bank,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([2.0]),
        temperature=0.1,
        margin=0.02,
    )
    assert float(clean) < float(harmful)


def test_basin_margin_guard_is_zero_until_a_good_edge_loses_margin():
    from scripts.train_lafgs_v7_online_metric import (
        _basin_good_margin_guard_loss,
    )

    bank = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=1
    )
    anchors = torch.tensor([[0, 1, 2]])
    query = bank[anchors]
    unchanged = _basin_good_margin_guard_loss(
        query,
        bank,
        query,
        bank,
        anchors,
        torch.tensor([1e-8]),
        maximum_inverse_propensity=100,
    )
    damaged_query = query.clone()
    damaged_query[0, 0] = bank[1]
    damaged = _basin_good_margin_guard_loss(
        query,
        bank,
        damaged_query,
        bank,
        anchors,
        torch.tensor([1e-8]),
        maximum_inverse_propensity=100,
    )
    assert float(unchanged) == 0.0
    assert float(damaged) > 0.0


def test_pose_critical_csr_truncation_keeps_highest_weights():
    indices, values = _csr_topk_by_values(
        torch.tensor([0, 3, 5]),
        torch.tensor([10, 11, 12, 20, 21]),
        torch.tensor([0.2, 2.0, 1.0, 0.1, 0.8]),
        width=2,
    )
    assert indices.tolist() == [[11, 12], [21, 20]]
    torch.testing.assert_close(
        values, torch.tensor([[2.0, 1.0], [0.8, 0.1]])
    )


def test_anchor_residual_is_bounded():
    raw = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    adapted, residual = _bounded_anchor_features(
        raw, torch.full_like(raw, 10.0), maximum=0.02
    )
    assert float(residual.norm(dim=1).max()) <= 0.020001
    torch.testing.assert_close(adapted.norm(dim=1), torch.ones(4))


def test_metric_warm_start_uses_deployed_features_unless_explicitly_overridden():
    from scripts.train_lafgs_v7_online_metric import _initial_anchor_features

    state = {
        "anchor_features": torch.tensor([[1.0, 0.0]]),
        "v7_metric_raw_features": torch.tensor([[0.0, 1.0]]),
    }
    current, current_key = _initial_anchor_features(state, "current")
    raw, raw_key = _initial_anchor_features(state, "pre_metric_raw")
    assert current_key == "anchor_features"
    assert raw_key == "v7_metric_raw_features"
    torch.testing.assert_close(current, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(raw, torch.tensor([[0.0, 1.0]]))


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
