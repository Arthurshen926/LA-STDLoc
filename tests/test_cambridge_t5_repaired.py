import numpy as np
import pytest
import torch
from experiments.cambridge_mainline_ablations import t5

def geometry():
 point=np.array([.2,.1,5.],dtype=np.float32)
 K=np.repeat(np.array([[500,0,320],[0,500,240],[0,0,1]],dtype=np.float32)[None],3,0)
 poses=np.repeat(np.eye(4,dtype=np.float32)[None],3,0)
 poses[:,0,3]=[1,0,-1]
 camera=np.einsum("nij,j->ni",poses[:,:3,:3],point)+poses[:,:3,3]
 p=np.einsum("nij,nj->ni",K,camera)
 uv=p[:,:2]/p[:,2,None]
 return point,K,poses,uv

def test_common_ray_and_proposals():
 point,K,pose,uv=geometry()
 cp=np.repeat(point[None],3,0);dp=cp.copy()
 mask,out=t5.coordinate_group(cp,dp,uv,K,pose)
 assert mask.all()
 for value in out.values():np.testing.assert_allclose(value,point,atol=1e-5)

def test_common_support_applies_to_all_arms():
 point,K,pose,uv=geometry();cp=np.repeat(point[None],3,0);dp=cp.copy();dp[0]=np.nan
 assert t5.coordinate_group(cp,dp,uv,K,pose) is None

def test_depth_pixel_convention():
 point,K,pose,uv=geometry()
 for j in range(3):
  xyz=t5.depth_xyz(uv[j:j+1]-.5,np.array([5.]),K[j],pose[j])
  np.testing.assert_allclose(xyz[0],point,atol=1e-6)

def test_center_frontmost_and_radius():
 K=np.array([[100,0,50],[0,100,50],[0,0,1]],dtype=np.float32)
 pts=np.array([[0,0,3],[0,0,5]],dtype=np.float32)
 cp=t5.projected_centers(pts,K,np.eye(4),100,100,np.array([[49.5,49.5],[1,1]],dtype=np.float32))
 np.testing.assert_array_equal(cp[0],pts[0]);assert np.isnan(cp[1]).all()

def test_mapping_residuals_keep_invalid():
 point,K,pose,uv=geometry()
 np.testing.assert_allclose(t5.project_residual(point,uv,K,pose),0,atol=1e-5)
 assert np.isinf(t5.project_residual(-point,uv,K,pose)).all()

def test_cpu_only():
 assert not torch.cuda.is_initialized()
