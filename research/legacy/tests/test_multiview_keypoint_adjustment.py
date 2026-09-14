import torch
from evidence.multiview_keypoint_adjustment import adjust_keypoints
from features.superpoint import sample_descriptors


def test_featuremetric_moves_toward_other_view_target_with_bound():
    dense=torch.stack([torch.ones(8,8),torch.arange(8).float()[None].expand(8,-1)/8])[None]
    point=torch.tensor([[25.,25.]])
    target=sample_descriptors((point+torch.tensor([[1.,0.]]))[None],dense)[0].T
    revised,desc,before,after=adjust_keypoints(dense,point,target,trust=.0001)
    assert revised[0,0]>point[0,0]
    assert torch.linalg.vector_norm(revised-point)<=1.50001
    assert after[0]<before[0]
    assert torch.allclose(torch.linalg.vector_norm(desc,dim=1),torch.ones(1))


def test_constant_feature_field_does_not_invent_geometry_information():
    dense=torch.ones(1,2,8,8);point=torch.tensor([[25.,25.]])
    revised,_,_,_=adjust_keypoints(dense,point,torch.tensor([[1.,0.]]))
    assert torch.equal(revised,point)


def test_nonfinite_crossfit_is_not_an_improvement():
    from evidence.multiview_keypoint_adjustment import spatial_prediction_improves
    assert not spatial_prediction_improves({'heldout_median_px':[2.,float('inf')]},{'heldout_median_px':[1.,float('inf')]})
    assert not spatial_prediction_improves({'heldout_median_px':[2.,3.]},{})
    assert not spatial_prediction_improves({'heldout_median_px':[2.,3.]},{'heldout_median_px':[2.1,2.]})
    assert spatial_prediction_improves({'heldout_median_px':[2.,3.]},{'heldout_median_px':[2.,2.5]})


def test_frontend_inference_tensor_supports_coordinate_gradients():
    with torch.inference_mode():dense=torch.ones(1,2,8,8)
    point=torch.tensor([[25.,25.]])
    revised,_,_,_=adjust_keypoints(dense,point,torch.tensor([[1.,0.]]),steps=2)
    assert torch.equal(revised,point)
