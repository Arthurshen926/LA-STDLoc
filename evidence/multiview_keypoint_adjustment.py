"""Bounded one-pass featuremetric adjustment toward OTHER mapping views."""
import torch
from torch.nn import functional as F
from features.superpoint import sample_descriptors


def spatial_prediction_improves(before, after):
    import numpy as np
    old=before.get('heldout_median_px');new=after.get('heldout_median_px')
    if old is None or new is None:return False
    old,new=np.asarray(old,float),np.asarray(new,float)
    if old.shape!=(2,) or new.shape!=(2,) or not np.isfinite(old).all() or not np.isfinite(new).all():return False
    return bool(np.all(new<=old) and np.max(new)<=.95*np.max(old))


def adjust_keypoints(dense, points, other_view_targets, *, radius=1.5, steps=40, trust=.002):
    if radius<=0 or steps<1 or trust<=0:raise ValueError('positive bounds required')
    # Frontend uses inference_mode; clone outside it before grid-coordinate AD.
    dense=dense.detach().clone();points=points.detach();target=F.normalize(other_view_targets.detach(),dim=1)
    delta=torch.zeros_like(points,requires_grad=True)
    optimizer=torch.optim.Adam([delta],lr=.1)
    with torch.no_grad():
        initial=1.-(sample_descriptors(points[None],dense)[0].T*target).sum(1)
    for _ in range(steps):
        optimizer.zero_grad()
        desc=sample_descriptors((points+delta)[None],dense)[0].T
        loss=(1.-(desc*target).sum(1)+trust*delta.square().sum(1)).sum()
        loss.backward();optimizer.step()
        with torch.no_grad():
            delta.mul_((radius/torch.linalg.vector_norm(delta,dim=1).clamp_min(radius))[:,None])
    with torch.no_grad():
        result=points+delta
        desc=sample_descriptors(result[None],dense)[0].T
        final=1.-(desc*target).sum(1)
        improved=final+trust*delta.square().sum(1)<initial-1e-6
        result=torch.where(improved[:,None],result,points)
        desc=sample_descriptors(result[None],dense)[0].T
    return result,desc,initial,1.-(desc*target).sum(1)
