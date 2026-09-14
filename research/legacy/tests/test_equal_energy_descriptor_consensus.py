import math

import numpy as np
import torch

from scripts.audit_equal_energy_descriptor_consensus import (
    FEATURE_NAMES,
    _balanced_accuracy,
    _conformal_lower_bounds,
    _fit_ridge,
    _loso_predictions,
    _rank_auc,
    extract_query_features,
    restricted_agreement_choices,
)


def _unit_rows(rows: int, dimensions: int) -> torch.Tensor:
    values = torch.arange(1, rows * dimensions + 1, dtype=torch.float32).reshape(
        rows, dimensions
    )
    return torch.nn.functional.normalize(values, dim=1)


def test_query_feature_contract_is_fixed_retrieval_before_19d():
    superpoint = _unit_rows(4, 256)
    xfeat = _unit_rows(4, 64).flip(0)
    record = {
        "native_descriptors": torch.cat((superpoint, xfeat), dim=1)
        / math.sqrt(2.0),
        "native_scores": torch.tensor([0.2, 0.4, 0.6, 0.8]),
        "feature_map": torch.arange(256 * 2 * 2, dtype=torch.float32).reshape(
            256, 2, 2
        )
        / 1024.0,
    }
    features = extract_query_features(record, torch.arange(4))
    assert features.shape == (len(FEATURE_NAMES),) == (19,)
    assert np.isfinite(features).all()
    assert features[0] <= 1.0
    assert features[5] <= 1.0
    assert features[-4] == np.mean([0.2, 0.4, 0.6, 0.8])


def test_restricted_agreement_uses_minimum_branch_score_without_weight():
    query_superpoint = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    query_xfeat = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    map_superpoint = torch.tensor(
        [[0.9, 0.0], [0.8, 0.0], [0.95, 0.0], [0.7, 0.0]]
    )
    map_xfeat = torch.tensor(
        [[0.1, 0.0], [0.7, 0.0], [0.2, 0.0], [0.6, 0.0]]
    )
    choices = restricted_agreement_choices(
        query_superpoint=query_superpoint,
        query_xfeat=query_xfeat,
        map_superpoint=map_superpoint,
        map_xfeat=map_xfeat,
        baseline_winners=np.asarray([0, 2]),
        candidate_winners=np.asarray([1, 3]),
    )
    assert choices.tolist() == [True, True]


def test_fixed_ridge_has_intercept_and_alpha_one_solution():
    x = np.asarray([[-1.0], [0.0], [1.0]])
    y = np.asarray([-1.0, 1.0, 3.0])
    model = _fit_ridge(x, y, alpha=1.0)
    assert np.allclose(model(np.asarray([[0.0], [2.0]])), [1.0, 11.0 / 3.0])


def test_loso_and_nested_conformal_never_fit_the_held_out_sequence():
    groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 4)
    feature = np.tile(np.asarray([-1.0, -0.5, 0.5, 1.0]), 4)[:, None]
    advantage = feature[:, 0] * 0.2
    prediction, folds = _loso_predictions(
        feature,
        advantage,
        groups,
        kind="regression",
    )
    conformal_prediction, lower, conformal_folds = _conformal_lower_bounds(
        feature,
        advantage,
        groups,
    )
    assert len(folds) == len(conformal_folds) == 4
    assert all(row["support_query_count"] == 12 for row in folds)
    assert np.corrcoef(prediction, advantage)[0, 1] > 0.99
    assert np.isfinite(conformal_prediction).all()
    assert np.all(lower <= conformal_prediction)


def test_fixed_logistic_loso_and_metric_helpers_are_deterministic():
    groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 4)
    feature = np.tile(np.asarray([-2.0, -1.0, 1.0, 2.0]), 4)[:, None]
    advantage = feature[:, 0] * 0.1
    first, _ = _loso_predictions(
        feature,
        advantage,
        groups,
        kind="classification",
    )
    second, _ = _loso_predictions(
        feature,
        advantage,
        groups,
        kind="classification",
    )
    selected = first > 0.0
    assert np.array_equal(first, second)
    assert _rank_auc(advantage > 0.0, first) == 1.0
    assert _balanced_accuracy(advantage > 0.0, selected) == 1.0
