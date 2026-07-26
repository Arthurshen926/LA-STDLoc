import torch

from localization_training.map_sanitization import (
    binary_ranking_metrics,
    build_sanitization_scores,
    percentile_rank,
    select_sanitized_landmarks,
)


def _inputs(count=100):
    observations = torch.arange(1, count + 1).float()
    correct = observations * torch.linspace(0.1, 0.95, count)
    statistics = {
        "observation_count": observations,
        "effective_observation_count": observations,
        "correct_count": correct,
        "cross_view_top1_harmful_switch_rate": torch.linspace(
            0.9, 0.0, count
        ),
        "target_incoming_count": observations,
        "target_false_hit_count": observations - correct,
        "margin": torch.linspace(-1.0, 1.0, count),
        "reprojection_error": torch.linspace(10.0, 0.1, count),
        "mean_uv": torch.rand(count, 2),
        "mean_depth": torch.linspace(1.0, 20.0, count),
    }
    geometry = {
        "gaussian_type": "3dgs",
        "raster_visibility_count": observations,
        "mvinit_observation_count": observations,
        "opacity": torch.linspace(0.05, 0.95, count),
        "planarity": torch.linspace(1.0, 0.05, count),
        "scaling": torch.full((count, 3), 0.02),
        "rgb_center_offset_mahalanobis": torch.linspace(20.0, 0.0, count),
        "rgb_center_offset_m": torch.linspace(0.1, 0.0, count),
    }
    return statistics, geometry


def test_reliability_separates_strong_and_weak_landmarks():
    statistics, geometry = _inputs()
    scores = build_sanitization_scores(statistics, geometry)
    assert scores.localization_reliability[-1] > scores.localization_reliability[0]
    assert scores.geometry_reliability[-1] > scores.geometry_reliability[0]
    assert int((scores.state == 1).sum()) > 0
    assert int((scores.state == 3).sum()) > 0


def test_selection_has_exact_budget_and_unique_indices():
    statistics, geometry = _inputs()
    geometry["rgb_center_offset_m"].zero_()
    scores = build_sanitization_scores(statistics, geometry)
    for mode in ("loc", "loc_geo", "loc_geo_coverage"):
        selected = select_sanitized_landmarks(
            scores, statistics, mode=mode, budget=50
        )
        assert selected.numel() == 50
        assert torch.unique(selected).numel() == 50


def test_binary_ranking_metrics_are_perfect_for_separable_scores():
    metrics = binary_ranking_metrics(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([False, False, True, True]),
    )
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_percentile_rank_does_not_leak_input_order_across_ties():
    values = torch.tensor([2.0, 1.0, 2.0, 3.0, 1.0])
    rank = percentile_rank(values)
    assert rank[0] == rank[2]
    assert rank[1] == rank[4]
    assert rank[1] < rank[0] < rank[3]


def test_constant_geometry_evidence_is_neutral_not_index_ranked():
    statistics, geometry = _inputs()
    count = statistics["observation_count"].numel()
    geometry["raster_visibility_count"] = torch.zeros(count)
    geometry["mvinit_observation_count"] = torch.zeros(count)
    geometry["opacity"] = torch.ones(count)
    scores = build_sanitization_scores(statistics, geometry)
    assert torch.unique(scores.components["visibility_support"]).numel() == 1
    assert torch.unique(scores.components["mv_support"]).numel() == 1
    assert torch.unique(scores.components["opacity_quality"]).numel() == 1


def test_unobserved_landmark_is_not_treated_as_perfect_reprojection():
    statistics, geometry = _inputs(count=4)
    statistics["observation_count"][0] = 0
    statistics["effective_observation_count"][0] = 0
    statistics["correct_count"][0] = 0
    statistics["reprojection_error"][0] = 0
    scores = build_sanitization_scores(statistics, geometry)
    assert scores.components["reprojection_quality"][0] == 0


def test_metric_center_drift_is_not_hidden_by_large_covariance():
    statistics, geometry = _inputs(count=4)
    geometry["rgb_center_offset_mahalanobis"] = torch.zeros(4)
    geometry["rgb_center_offset_m"] = torch.tensor([0.1, 0.0, 0.0, 0.0])
    scores = build_sanitization_scores(statistics, geometry)
    consistency = scores.components["rgb_center_metric_consistency"]
    assert consistency[0] < 0.01
    assert torch.all(consistency[1:] == 1)


def test_geometry_selection_cannot_reintroduce_large_anchor_drift():
    statistics, geometry = _inputs(count=20)
    geometry["rgb_center_offset_m"] = torch.zeros(20)
    geometry["rgb_center_offset_m"][:2] = 0.1
    # Make the corrupted anchors maximally attractive on the localization axis.
    statistics["correct_count"][:2] = statistics["observation_count"][:2]
    statistics["target_false_hit_count"][:2] = 0
    scores = build_sanitization_scores(statistics, geometry)
    selected = select_sanitized_landmarks(
        scores, statistics, mode="loc_geo", budget=18
    )
    assert not bool(torch.isin(torch.tensor([0, 1]), selected).any())


def test_independent_geometry_uses_five_states_and_hard_reject_gate():
    statistics, geometry = _inputs(count=20)
    geometry["rgb_center_offset_m"].zero_()
    geometry["triangulation_high_confidence"] = torch.ones(20, dtype=torch.bool)
    geometry["triangulation_current_center_offset_m"] = torch.zeros(20)
    geometry["triangulation_current_center_offset_m"][0] = 0.2
    scores = build_sanitization_scores(statistics, geometry)
    assert scores.state[0] == 3
    selected = select_sanitized_landmarks(
        scores, statistics, mode="hard_geo_loc", budget=19
    )
    assert 0 not in selected


def test_2dgs_surface_residual_ignores_native_patch_extent_but_rejects_drift():
    statistics, geometry = _inputs(count=20)
    geometry["gaussian_type"] = "2dgs"
    geometry["scaling"] = torch.full((20, 2), 0.05)
    geometry["rgb_center_offset_m"].zero_()
    geometry["triangulation_high_confidence"] = torch.ones(20, dtype=torch.bool)
    geometry["triangulation_current_center_offset_m"] = torch.full((20,), 0.2)
    geometry["triangulation_rgb_normal_distance_m"] = torch.zeros(20)
    geometry["triangulation_current_normal_distance_m"] = torch.zeros(20)
    geometry["triangulation_rgb_tangent_normalized"] = torch.full((20,), 4.0)
    geometry["triangulation_current_tangent_normalized"] = torch.full(
        (20,), 4.0
    )
    scores = build_sanitization_scores(statistics, geometry)
    assert scores.components["triangulation_surface_aware"].all()
    assert scores.components["triangulation_geometry_mismatch_score"].max() == 0
    assert not bool(scores.components["triangulation_hard_reject"].any())

    geometry["triangulation_current_normal_distance_m"][0] = 0.1
    scores = build_sanitization_scores(statistics, geometry)
    assert scores.components["triangulation_geometry_mismatch_score"][0] == 5
    assert scores.state[0] == 3


def test_query_level_coverage_reserves_supported_landmarks():
    statistics, geometry = _inputs(count=20)
    geometry["rgb_center_offset_m"].zero_()
    statistics["query_support_offsets"] = torch.tensor([0, 2, 4])
    statistics["query_support_indices"] = torch.tensor([0, 1, 2, 3])
    scores = build_sanitization_scores(statistics, geometry)
    selected = select_sanitized_landmarks(
        scores, statistics, mode="loc_query_coverage", budget=10
    )
    assert bool(torch.isin(torch.tensor([0, 1]), selected).any())
    assert bool(torch.isin(torch.tensor([2, 3]), selected).any())
