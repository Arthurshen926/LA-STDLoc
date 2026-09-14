import numpy as np
import pytest
from evidence.mapping_track_extension import extend_geometry


def inputs():
    T = np.tile(np.eye(4), (9,1,1))
    T[:,0,3] = -np.r_[np.linspace(-3,3,7), -5, 5]
    K = np.tile(np.diag([500.,500.,1.]), (9,1,1))
    p = np.array([0.,0.,10.])
    cam = p + T[:,:3,3]
    uv = cam[:,:2]/cam[:,2:]*500
    noisy = uv[:7] + np.random.default_rng(7).normal(0,1.5,(7,2))
    return noisy, K[:7], T[:7], p, uv[7:], K[7:], T[7:]


def test_independent_measurements_can_resolve_internal_crossfit_failure():
    values = inputs()
    original = [v.copy() for v in values]
    result = extend_geometry(*values)
    assert result is not None
    assert result['before']['status'] == 'unstable'
    assert result['after']['status'] == 'internally_consistent'
    assert np.linalg.eigvalsh(result['covariance']).min() > 0
    for a, b in zip(original, values): np.testing.assert_array_equal(a, b)
    # This is internal evidence, deliberately NOT an assertion of GT improvement.


def test_consistent_track_is_not_modified():
    uv, K, T, p, auv, aK, aT = inputs()
    cam = p + T[:,:3,3]
    uv = cam[:,:2]/cam[:,2:]*500
    assert extend_geometry(uv,K,T,p,auv,aK,aT) is None


def test_extension_requires_new_measurements():
    uv,K,T,p,auv,aK,aT = inputs()
    with pytest.raises(ValueError): extend_geometry(uv,K,T,p,auv[:0],aK[:0],aT[:0])
