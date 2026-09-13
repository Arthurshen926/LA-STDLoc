import cv2
import numpy as np
import torch
from evidence.radial_compensation import undo_extra_undistortion, refit_fixed_tracks


def test_inverse_extra_undistortion_matches_opencv_remap():
    K = np.array([[150.,0,160.],[0,150.,120.],[0,0,1.]])
    mx, my = cv2.initUndistortRectifyMap(K, np.array([.05,0,0,0]), None, K, (320,240), cv2.CV_32FC1)
    uv = torch.tensor([[0.,0.],[160.,120.],[300.,220.]], dtype=torch.double)
    actual = undo_extra_undistortion(uv, torch.from_numpy(K).expand(3,-1,-1), torch.full((3,), .05, dtype=torch.double))
    expected = np.array([[mx[int(y),int(x)],my[int(y),int(x)]] for x,y in uv])
    assert np.allclose(actual, expected, atol=2e-5)


def test_fixed_camera_refit_recovers_point_without_query_evidence():
    T = torch.eye(4, dtype=torch.double).repeat(5,1,1)
    T[:,0,3] = torch.linspace(-1,1,5)
    K = torch.tensor([[500.,0,320.],[0,500.,240.],[0,0,1.]], dtype=torch.double).repeat(5,1,1)
    truth = torch.tensor([[.2,.1,5.]], dtype=torch.double)
    cam = truth + T[:,:3,3]; h = (K @ cam[...,None])[...,0]
    uv = h[:,:2]/h[:,2:]
    out, valid = refit_fixed_tracks(truth + torch.tensor([[.1,.2,.3]]), torch.zeros(5,dtype=torch.long), uv, K, T)
    assert valid.all()
    assert torch.allclose(out, truth, atol=1e-7)
