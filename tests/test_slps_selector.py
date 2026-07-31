import torch

from localization_training.pose_sufficient_selector import FEATURE_NAMES
from localization_training.slps_selector import (
    RELATION_NAMES,
    SLPS_FEATURE_NAMES,
    SLPSModelConfig,
    SLPSSelector,
    beta_track_stability,
    build_relation_groups,
    build_slps_features,
    slps_from_state,
)
from scripts.augment_lafgs_slps_self_outcomes import _learned_subsets
from scripts.train_lafgs_slps_selector import (
    _calibrate_risk_thresholds,
    _sample_training_examples,
)


def _inputs(count=24):
    generator = torch.Generator().manual_seed(41)
    base = torch.randn(count, len(FEATURE_NAMES), generator=generator)
    xyz = torch.randn(count, 3, generator=generator)
    anchor_type = torch.arange(count) % 3
    track = torch.arange(count) // 2
    dependency = torch.arange(count) // 3
    source = torch.arange(count) // 4
    keypoints = torch.rand(count, 2, generator=generator)
    keypoints[:, 0] *= 640
    keypoints[:, 1] *= 480
    features = build_slps_features(
        base,
        xyz=xyz,
        anchor_type=anchor_type,
        track_groups=track,
        track_stability=torch.linspace(0.2, 0.9, count),
        anchor_map_support=torch.arange(count).float(),
    )
    relations = build_relation_groups(
        keypoints=keypoints,
        image_hw=(480, 640),
        xyz=xyz,
        dependency_groups=dependency,
        source_groups=source,
        track_groups=track,
    )
    return features, relations


def test_slps_feature_and_relation_contracts_align():
    features, relations = _inputs()
    assert features.shape == (24, len(SLPS_FEATURE_NAMES))
    assert relations.shape == (24, len(RELATION_NAMES))
    assert torch.isfinite(features).all()
    for relation in range(relations.shape[1]):
        values = torch.unique(relations[:, relation], sorted=True)
        assert torch.equal(values, torch.arange(len(values)))


def test_unknown_track_ids_are_independent_singletons():
    count = 4
    base = torch.zeros(count, len(FEATURE_NAMES))
    xyz = torch.arange(count * 3).reshape(count, 3).float()
    track = torch.tensor([-1, -1, 7, 7])
    features = build_slps_features(
        base,
        xyz=xyz,
        anchor_type=torch.zeros(count),
        track_groups=track,
        track_stability=torch.full((count,), 0.5),
        anchor_map_support=torch.ones(count),
    )
    multiplicity_column = SLPS_FEATURE_NAMES.index(
        "query_track_multiplicity"
    )
    multiplicity = torch.expm1(features[:, multiplicity_column])
    assert torch.equal(multiplicity, torch.tensor([1.0, 1.0, 2.0, 2.0]))

    relations = build_relation_groups(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        image_hw=(8, 8),
        xyz=xyz,
        dependency_groups=torch.arange(count),
        source_groups=torch.arange(count),
        track_groups=track,
    )
    track_column = RELATION_NAMES.index("track")
    assert relations[0, track_column] != relations[1, track_column]
    assert relations[2, track_column] == relations[3, track_column]


def test_unknown_track_stability_uses_per_anchor_beta_prior():
    stability = beta_track_stability(
        attempts_by_fold=torch.tensor(
            [[10.0, 10.0, 5.0, 5.0], [10.0, 10.0, 5.0, 5.0]]
        ),
        clean_inlier_by_fold=torch.tensor(
            [[10.0, 0.0, 4.0, 4.0], [10.0, 0.0, 4.0, 4.0]]
        ),
        track_groups=torch.tensor([-1, -1, 3, 3]),
        prior_strength=4.0,
    )
    assert stability[0] > stability[1]
    assert torch.equal(stability[2], stability[3])


def test_slps_relation_encoder_is_permutation_equivariant():
    features, relations = _inputs()
    model = SLPSSelector(SLPSModelConfig()).eval()
    permutation = torch.randperm(len(features), generator=torch.Generator().manual_seed(3))
    forward = model.encode(features, relations)["hidden"]
    permuted = model.encode(
        features[permutation], relations[permutation]
    )["hidden"]
    assert torch.allclose(
        forward[permutation], permuted, atol=1e-5, rtol=1e-5
    )


def test_slps_set_utility_has_diminishing_return_for_nested_sets():
    features, relations = _inputs(count=8)
    model = SLPSSelector(SLPSModelConfig()).eval()
    encoded = model.encode(features, relations)
    a = torch.tensor([True, False, False, False, False, False, False, False])
    b = torch.tensor([True, True, False, False, False, False, False, False])
    a_plus = a.clone()
    b_plus = b.clone()
    a_plus[2] = True
    b_plus[2] = True
    marginal_a = model.score_set(encoded, relations, a_plus) - model.score_set(
        encoded, relations, a
    )
    marginal_b = model.score_set(encoded, relations, b_plus) - model.score_set(
        encoded, relations, b
    )
    assert marginal_a >= marginal_b - 1e-5


def test_vectorized_set_utility_matches_scalar_reference():
    features, relations = _inputs(count=16)
    model = SLPSSelector(
        SLPSModelConfig(quality_utility_heads=True)
    ).eval()
    encoded = model.encode(features, relations)
    masks = torch.stack(
        (
            torch.arange(16) < 4,
            torch.arange(16) < 9,
            torch.arange(16) % 2 == 0,
        )
    )
    vectorized = model.score_sets(encoded, relations, masks)
    reference = torch.stack(
        [
            model._score_set_reference(encoded, relations, mask)
            for mask in masks
        ]
    )
    assert torch.allclose(vectorized, reference, atol=1e-5, rtol=1e-5)


def test_slps_risk_gate_uses_smallest_safe_set_and_falls_back():
    features, relations = _inputs(count=16)
    model = SLPSSelector(SLPSModelConfig()).eval()
    final = model.set_outcome_head[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([10.0, -10.0, 7.0]))
    compact = model.select(
        features,
        relations,
        budgets=(8, 12),
        safe_probability_threshold=0.75,
        catastrophic_probability_threshold=0.15,
    )
    assert compact.selected_budget == 8
    assert int(compact.selected_mask.sum()) == 8
    assert not compact.used_fallback

    with torch.no_grad():
        final.bias.copy_(torch.tensor([-10.0, 10.0, 7.0]))
    fallback = model.select(
        features,
        relations,
        budgets=(8, 12),
        safe_probability_threshold=0.75,
        catastrophic_probability_threshold=0.15,
    )
    assert fallback.selected_budget == 16
    assert fallback.selected_mask.all()
    assert fallback.used_fallback

    fail_closed = model.select(
        features,
        relations,
        budgets=(8, 12),
        safe_probability_threshold=1.01,
        catastrophic_probability_threshold=-0.01,
    )
    assert fail_closed.selected_mask.all()
    assert fail_closed.used_fallback
    assert fail_closed.diagnostics["fallback_infeasible_calibration"] == 1.0


def test_slps_serialized_model_roundtrip():
    features, relations = _inputs()
    model = SLPSSelector(SLPSModelConfig()).eval()
    state = {
        "schema": "lafgs_slps_selector",
        "feature_names": list(SLPS_FEATURE_NAMES),
        "relation_names": list(RELATION_NAMES),
        "model_config": model.export_config(),
        "feature_mean": model.feature_mean,
        "feature_scale": model.feature_scale,
        "model_state_dict": model.state_dict(),
    }
    restored = slps_from_state(state)
    expected = model.encode(features, relations)["solver_probability"]
    actual = restored.encode(features, relations)["solver_probability"]
    assert torch.allclose(expected, actual)


def test_slps_old_state_without_relative_head_remains_loadable():
    model = SLPSSelector(SLPSModelConfig()).eval()
    old_state_dict = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("relative_outcome_head.")
    }
    state = {
        "schema": "lafgs_slps_selector",
        "feature_names": list(SLPS_FEATURE_NAMES),
        "relation_names": list(RELATION_NAMES),
        "model_config": model.export_config(),
        "feature_mean": model.feature_mean,
        "feature_scale": model.feature_scale,
        "model_state_dict": old_state_dict,
    }
    restored = slps_from_state(state)
    assert isinstance(restored, SLPSSelector)


def test_quality_heads_contribute_to_set_utility_when_enabled():
    features, relations = _inputs(count=8)
    model = SLPSSelector(
        SLPSModelConfig(quality_utility_heads=True)
    ).eval()
    encoded = model.encode(features, relations)
    mask = torch.tensor(
        [True, True, False, False, False, False, False, False]
    )
    enabled = model.score_set(encoded, relations, mask)
    model.config = SLPSModelConfig(quality_utility_heads=False)
    disabled = model.score_set(encoded, relations, mask)
    assert enabled > disabled


def test_decoupled_risk_encoder_does_not_change_greedy_ordering():
    features, relations = _inputs(count=32)
    model = SLPSSelector(
        SLPSModelConfig(decoupled_risk_encoder=True, greedy_block_size=4)
    ).eval()
    encoded = model.encode(features, relations)
    before = model.greedy_order(encoded, relations, maximum_count=16)
    assert "risk_hidden" in encoded
    with torch.no_grad():
        for parameter in model.risk_row_encoder.parameters():
            parameter.add_(torch.randn_like(parameter))
    changed = model.encode(features, relations)
    after = model.greedy_order(changed, relations, maximum_count=16)
    assert torch.equal(before, after)
    assert not torch.allclose(encoded["risk_hidden"], changed["risk_hidden"])


def test_bounded_residual_utility_starts_as_exact_ordering_identity():
    features, relations = _inputs(count=32)
    base = SLPSSelector(
        SLPSModelConfig(quality_utility_heads=True, greedy_block_size=4)
    ).eval()
    residual = SLPSSelector(
        SLPSModelConfig(
            quality_utility_heads=True,
            greedy_block_size=4,
            bounded_residual_utility_fraction=0.05,
        )
    ).eval()
    incompatible = residual.load_state_dict(base.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "residual_utility_head.weight",
        "residual_utility_head.bias",
    }
    base_encoded = base.encode(features, relations)
    residual_encoded = residual.encode(features, relations)
    assert torch.equal(base_encoded["additive"], residual_encoded["additive"])
    assert torch.equal(
        base.greedy_order(base_encoded, relations, maximum_count=16),
        residual.greedy_order(residual_encoded, relations, maximum_count=16),
    )


def test_bounded_residual_utility_cannot_exceed_query_trust_region():
    features, relations = _inputs(count=32)
    fraction = 0.03
    model = SLPSSelector(
        SLPSModelConfig(bounded_residual_utility_fraction=fraction)
    ).eval()
    with torch.no_grad():
        model.residual_utility_head.weight.fill_(100.0)
        model.residual_utility_head.bias.fill_(100.0)
    encoded = model.encode(features, relations)
    robust_scale = (
        torch.quantile(encoded["additive_base"], 0.90)
        - torch.quantile(encoded["additive_base"], 0.10)
    ).clamp_min(0.1)
    assert encoded["utility_residual"].abs().max() <= (
        fraction * robust_scale + 1e-6
    )


def test_relative_outcome_lcb_controls_compact_set_acceptance():
    features, relations = _inputs(count=16)
    model = SLPSSelector(
        SLPSModelConfig(relative_outcome_heads=True)
    ).eval()
    with torch.no_grad():
        outcome = model.set_outcome_head[-1]
        outcome.weight.zero_()
        outcome.bias.copy_(torch.tensor([10.0, -10.0, 7.0]))
        relative = model.relative_outcome_head[-1]
        relative.weight.zero_()
        relative.bias.copy_(torch.tensor([1.0, -10.0]))
    accepted = model.select(
        features,
        relations,
        budgets=(8,),
        safe_probability_threshold=0.75,
        catastrophic_probability_threshold=0.15,
        relative_utility_lcb_threshold=0.0,
    )
    assert accepted.selected_budget == 8
    assert accepted.relative_utility_lcb > 0.0

    with torch.no_grad():
        relative.bias.copy_(torch.tensor([-1.0, -10.0]))
    rejected = model.select(
        features,
        relations,
        budgets=(8,),
        safe_probability_threshold=0.75,
        catastrophic_probability_threshold=0.15,
        relative_utility_lcb_threshold=0.0,
    )
    assert rejected.selected_budget == 16
    assert rejected.used_fallback


def test_outcome_atlas_transfers_neighbor_risk_and_excludes_self():
    model = SLPSSelector(SLPSModelConfig()).eval()
    model.attach_outcome_atlas(
        {
            "schema": "lafgs_slps_outcome_atlas",
            "support_query_names": ["query-a", "query-b", "query-c"],
            "support_anchor_mask": torch.tensor(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                ],
                dtype=torch.bool,
            ),
            "budgets": [8, 12],
            "safe_probability_targets": torch.tensor(
                [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
            ),
            "catastrophic_probability_targets": torch.tensor(
                [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
            ),
            "relative_utility_targets": torch.tensor(
                [[-1.0, -1.0], [0.5, 1.0], [-1.0, -1.0]]
            ),
            "neighbor_count": 1,
            "similarity_power": 4.0,
        }
    )
    outcome = model.predict_atlas_outcomes(
        torch.tensor([0, 1, 2]), query_name="query-a"
    )
    assert outcome[8]["safe_probability"] == 1.0
    assert outcome[12]["relative_utility"] == 1.0
    assert outcome[8]["maximum_similarity"] > 0.5


def test_outcome_atlas_gate_chooses_highest_expected_utility_budget():
    features, relations = _inputs(count=16)
    model = SLPSSelector(SLPSModelConfig()).eval()
    model.attach_outcome_atlas(
        {
            "schema": "lafgs_slps_outcome_atlas",
            "support_query_names": ["support"],
            "support_anchor_mask": torch.ones((1, 16), dtype=torch.bool),
            "budgets": [8, 12],
            "safe_probability_targets": torch.ones((1, 2)),
            "catastrophic_probability_targets": torch.zeros((1, 2)),
            "relative_utility_targets": torch.tensor([[0.2, 0.7]]),
            "neighbor_count": 1,
            "similarity_power": 4.0,
        }
    )
    selection = model.select(
        features,
        relations,
        budgets=(8, 12),
        risk_gate_mode="atlas",
        atlas_safe_probability_threshold=0.9,
        atlas_catastrophic_probability_threshold=0.1,
        atlas_relative_utility_threshold=0.0,
        anchor_indices=torch.arange(16),
    )
    assert selection.selected_budget == 12
    assert int(selection.selected_mask.sum()) == 12
    assert not selection.used_fallback

    fallback = model.select(
        features,
        relations,
        budgets=(8, 12),
        risk_gate_mode="atlas",
        atlas_minimum_similarity=1.01,
        anchor_indices=torch.arange(16),
    )
    assert fallback.selected_budget == 16
    assert fallback.used_fallback


def test_outcome_atlas_uses_joint_anchor_image_context():
    features, _ = _inputs(count=1)
    x_index = SLPS_FEATURE_NAMES.index("keypoint_x")
    y_index = SLPS_FEATURE_NAMES.index("keypoint_y")
    features[:, x_index] = 0.75
    features[:, y_index] = 0.75
    model = SLPSSelector(SLPSModelConfig()).eval()
    context = torch.zeros((2, 8), dtype=torch.bool)
    context[0, 0] = True
    context[1, 3] = True
    model.attach_outcome_atlas(
        {
            "schema": "lafgs_slps_outcome_atlas",
            "support_query_names": ["bad-view", "good-view"],
            "support_anchor_mask": torch.ones((2, 2), dtype=torch.bool),
            "support_context_mask": context,
            "context_grid_size": 2,
            "context_weight": 1.0,
            "budgets": [1],
            "safe_probability_targets": torch.tensor([[0.0], [1.0]]),
            "catastrophic_probability_targets": torch.tensor([[1.0], [0.0]]),
            "relative_utility_targets": torch.tensor([[-1.0], [1.0]]),
            "neighbor_count": 1,
            "similarity_power": 4.0,
        }
    )
    outcome = model.predict_atlas_outcomes(
        torch.tensor([0]), features=features
    )
    assert outcome[1]["safe_probability"] == 1.0
    assert outcome[1]["catastrophic_probability"] == 0.0
    assert outcome[1]["relative_utility"] == 1.0


def test_outcome_atlas_conditions_risk_on_the_selected_set():
    model = SLPSSelector(SLPSModelConfig()).eval()
    model.attach_outcome_atlas(
        {
            "schema": "lafgs_slps_outcome_atlas",
            "support_query_names": ["bad-set", "good-set"],
            "support_anchor_mask": torch.ones((2, 2), dtype=torch.bool),
            "support_set_anchor_mask": torch.tensor(
                [[[1, 0]], [[0, 1]]], dtype=torch.bool
            ),
            "set_query_context_weight": 0.0,
            "budgets": [1],
            "safe_probability_targets": torch.tensor([[0.0], [1.0]]),
            "catastrophic_probability_targets": torch.tensor([[1.0], [0.0]]),
            "relative_utility_targets": torch.tensor([[-1.0], [1.0]]),
            "neighbor_count": 1,
            "similarity_power": 4.0,
        }
    )
    outcome = model.predict_atlas_outcomes(
        torch.tensor([1]),
        budget_masks={1: torch.tensor([True])},
    )
    assert outcome[1]["safe_probability"] == 1.0
    assert outcome[1]["catastrophic_probability"] == 0.0
    assert outcome[1]["relative_utility"] == 1.0


def test_track_stability_penalizes_inconsistent_track():
    attempts = torch.tensor(
        [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
    )
    clean = torch.tensor(
        [[9.0, 9.0], [9.0, 1.0], [9.0, 9.0]]
    )
    stability = beta_track_stability(
        attempts_by_fold=attempts,
        clean_inlier_by_fold=clean,
        track_groups=torch.tensor([0, 1]),
    )
    assert stability[0] > stability[1]


def test_self_mining_generates_local_learned_set_interventions():
    features, relations = _inputs(count=40)
    model = SLPSSelector(SLPSModelConfig()).eval()
    query = {
        "query_name": "sequence/frame0001.png",
        "features": features,
        "relation_groups": relations,
        "strict_clean": torch.arange(len(features)) % 3 == 0,
    }
    subsets = _learned_subsets(
        query,
        model,
        budgets=(16,),
        replacement_count=4,
        seed=2026,
    )
    names = {subset["name"] for subset in subsets}
    assert "learned_nested_16" in names
    assert "learned_drop_16" in names
    assert "learned_add_16" in names
    assert "learned_random_swap_16" in names
    signatures = {
        tuple(torch.as_tensor(subset["indices"]).tolist())
        for subset in subsets
    }
    assert len(signatures) == len(subsets)


def test_training_sampler_preserves_self_mined_examples():
    subsets = []
    for index in range(24):
        name = (
            "all"
            if index == 0
            else f"learned_nested_{index}"
            if index < 12
            else f"static_{index}"
        )
        subsets.append(
            {
                "name": name,
                "outcomes": [{"seed": 2026, "target_utility": -index}],
            }
        )
    sampled = _sample_training_examples(
        {"subsets": subsets},
        seed=2026,
        generator=__import__("random").Random(7),
        maximum_sets=10,
    )
    names = [subset["name"] for subset, _ in sampled]
    assert "all" in names
    assert sum(name.startswith("learned_") for name in names) >= 5


def test_risk_calibration_fails_closed_without_safe_threshold():
    features, relations = _inputs(count=16)
    model = SLPSSelector(SLPSModelConfig()).eval()
    with torch.no_grad():
        final = model.set_outcome_head[-1]
        final.weight.zero_()
        final.bias.zero_()
    query = {
        "features": features,
        "relation_groups": relations,
        "subsets": [
            {
                "name": "all",
                "indices": torch.arange(len(features)),
                "outcomes": [
                    {
                        "safe_relative_all": False,
                        "catastrophic": True,
                    }
                ],
            }
        ],
    }
    thresholds, calibration = _calibrate_risk_thresholds(model, [query])
    assert not calibration["feasible"]
    assert thresholds["safe_probability_threshold"] > 1.0
    assert thresholds["catastrophic_probability_threshold"] < 0.0


def test_risk_calibration_uses_worst_seed_for_deployment_sets():
    features, relations = _inputs(count=16)
    model = SLPSSelector(SLPSModelConfig()).eval()
    with torch.no_grad():
        final = model.set_outcome_head[-1]
        final.weight.zero_()
        final.bias.copy_(torch.tensor([10.0, -10.0, 7.0]))
    query = {
        "features": features,
        "relation_groups": relations,
        "subsets": [
            {
                "name": "learned_nested_8",
                "indices": torch.arange(8),
                "outcomes": [
                    {
                        "safe_relative_all": True,
                        "catastrophic": False,
                    },
                    {
                        "safe_relative_all": False,
                        "catastrophic": True,
                    },
                ],
            }
        ],
    }
    thresholds, calibration = _calibrate_risk_thresholds(model, [query])
    assert not calibration["feasible"]
    assert calibration["calibration_profile"] == "learned_nested"
    assert calibration["calibration_set_count"] == 1
    assert thresholds["safe_probability_threshold"] > 1.0
