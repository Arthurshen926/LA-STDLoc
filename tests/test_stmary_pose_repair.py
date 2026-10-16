import numpy as np
import pytest
from experiments.stmary_pose_repair.prepare import normal_pose
from anygsloc.data.colmap import qvec2rotmat

@pytest.mark.parametrize('q',[[2.,-.2,.3,.7],[-2.,.2,-.3,-.7],[1.,0,0,0]])
def test_pose_preserves_sign_center_and_so3(q):
    C = np.array([11.,-3.,21.])
    unit,t = normal_pose(q,C)
    R = qvec2rotmat(unit)
    assert np.dot(unit,q)>0
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R),1.,atol=1e-14)
    np.testing.assert_allclose(-R.T@t,C,atol=1e-13)

@pytest.mark.parametrize('q',[[0.,0,0,0],[float('nan'),0,0,0],[float('inf'),0,0,0],[1,2,3]])
def test_invalid_quaternion_fails_closed(q):
    with pytest.raises(ValueError):normal_pose(q,[1,2,3])

def test_opposite_sign_same_pose():
    q=np.array([.8,.1,.2,.4]); C=np.array([4.,5.,6.])
    a,t=normal_pose(q,C); b,u=normal_pose(-q,C)
    np.testing.assert_array_equal(a,-b)
    np.testing.assert_array_equal(t,u)

@pytest.mark.parametrize('C',[[float('nan'),0,0],[1,2]])
def test_invalid_center_fails_closed(C):
    with pytest.raises(ValueError):normal_pose([1,0,0,0],C)
