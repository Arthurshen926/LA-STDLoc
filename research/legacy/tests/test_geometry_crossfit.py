import numpy as np
import pytest
from evidence.geometry_crossfit import crossfit_geometry


def scene(point=(0., 0., 10.), baseline=1.):
    T = np.tile(np.eye(4), (12, 1, 1))
    T[:, 0, 3] = -np.linspace(-3, 3, 12) * baseline
    K = np.tile(np.diag([500., 500., 1.]), (12, 1, 1))
    p = np.asarray(point)
    cam = p + T[:, :3, 3]
    uv = cam[:, :2] / cam[:, 2:] * 500
    return uv, K, T, p


def test_exact_geometry_is_internally_consistent_not_absolute_truth():
    r = crossfit_geometry(*scene())
    assert r['status'] == 'internally_consistent'
    assert max(r['heldout_median_px']) < 1e-8
    assert not r['absolute_accuracy_certified']


def test_disagreeing_view_groups_are_detected():
    uv, K, T, p = scene()
    uv[6:, 1] += 8.
    assert crossfit_geometry(uv, K, T, p)['status'] == 'unstable'


def test_short_or_degenerate_track_is_unknown_not_invalid():
    uv, K, T, p = scene()
    assert crossfit_geometry(uv[:5], K[:5], T[:5], p)['status'] == 'unknown'
    assert crossfit_geometry(*scene(baseline=1e-6))['status'] == 'unknown'


def test_common_wrong_surface_cannot_be_certified_against_reality():
    # The measurements all describe z=12 rather than a hypothetical real z=10.
    r = crossfit_geometry(*scene(point=(0., 0., 12.)), rendered_depth=np.full(12, 10.))
    assert r['status'] == 'internally_consistent'
    assert abs(r['surface_relative_bias_median'] - .2) < 1e-10
    assert r['surface_relative_disagreement_mad'] == 0
    assert not r['absolute_accuracy_certified']


def test_missing_surface_depth_is_unknown_evidence_not_a_geometric_failure():
    r = crossfit_geometry(*scene(), rendered_depth=np.full(12, np.nan))
    assert r['status'] == 'internally_consistent'
    assert r['surface_evidence_count'] == 0
    assert r['surface_relative_bias_median'] is None


def test_invalid_input_is_rejected():
    uv, K, T, p = scene()
    with pytest.raises(ValueError):
        crossfit_geometry(uv[:-1], K, T, p)
    uv[0, 0] = np.nan
    with pytest.raises(ValueError):
        crossfit_geometry(uv, K, T, p)
