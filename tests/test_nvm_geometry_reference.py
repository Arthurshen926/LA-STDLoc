from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts import audit_nvm_geometry_reference as audit


def test_nvm_name_alias_pose_calibration_and_empty_view(tmp_path, monkeypatch):
    path=tmp_path/'reconstruction.nvm'
    path.write_text('NVM_V3\n\n2\na.jpg 100 1 0 0 0 0 0 0 0 0\nb.jpg 100 1 0 0 0 0 0 0 0 0\n\n1\n0 0 5 0 0 0 1 0 0 0 0\n')
    intr={1:NS(model='SIMPLE_RADIAL',params=np.array([100.,50.,40.,0.]))}
    extr={1:NS(name='a.png',camera_id=1),2:NS(name='b.png',camera_id=1)}
    monkeypatch.setattr(audit,'read_intrinsics_binary',lambda _:intr)
    monkeypatch.setattr(audit,'read_extrinsics_binary',lambda _:extr)
    K=np.repeat(np.array([[[100.,0.,50.],[0.,100.,40.],[0.,0.,1.]]]),2,axis=0)
    T=np.repeat(np.eye(4)[None],2,axis=0)
    xyz,views,residual=audit.read_reference(path,['a.png','b.png'],K,T,tmp_path)
    assert np.array_equal(xyz,[[0.,0.,5.]])
    assert np.array_equal(views[0][1],[[50.,40.]])
    assert views[1][1].shape==(0,2)
    assert residual==[0.,0.,0.]
    T[0,0,3]=1.
    with pytest.raises(ValueError,match='pose mismatch'):
        audit.read_reference(path,['a.png','b.png'],K,T,tmp_path)


def test_empty_camera_registry_cannot_silently_pass(tmp_path,monkeypatch):
    path=tmp_path/'reconstruction.nvm'
    path.write_text('NVM_V3\n1\na.jpg 100 1 0 0 0 0 0 0 0 0\n1\n0 0 5 0 0 0 1 0 0 0 0\n')
    monkeypatch.setattr(audit,'read_intrinsics_binary',lambda _: {})
    monkeypatch.setattr(audit,'read_extrinsics_binary',lambda _: {})
    with pytest.raises(ValueError,match='no NVM mapping camera'):
        audit.read_reference(path,[],np.empty((0,3,3)),np.empty((0,4,4)),tmp_path)
