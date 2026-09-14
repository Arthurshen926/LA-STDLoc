import torch
from evidence.surface_geometry_evidence import surface_row_evidence


def test_surface_conflict_needs_both_channels_and_valid_support():
    z=torch.ones(4);m=torch.tensor([.9,.9,1.,.9]);alpha=torch.tensor([1.,1.,1.,.1])
    n=torch.tensor([[1.,0.,0.]]).repeat(4,1);s=torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,1.,0.],[0.,1.,0.]])
    out=surface_row_evidence(z,m,alpha,n,s)
    assert out['conflict'].tolist()==[True,False,False,False]
    assert out['valid'].tolist()==[True,True,True,False]


def test_normal_sign_and_common_depth_bias_do_not_become_certified_error():
    out=surface_row_evidence(torch.tensor([10.]),torch.tensor([10.]),torch.ones(1),
                             torch.tensor([[1.,0.,0.]]),torch.tensor([[-1.,0.,0.]]))
    assert out['valid'].item() and not out['conflict'].item()
    # No external surface is provided; the equal depths may both be wrong.
    assert 'absolute_accuracy' not in out
