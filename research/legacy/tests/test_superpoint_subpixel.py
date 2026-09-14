import torch

from features.superpoint import quadratic_subpixel_keypoints


def test_quadratic_subpixel_peak_is_row_stable_and_bounded():
    y, x = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
    score = -((x.float() - 3.25) ** 2) - 2.0 * ((y.float() - 2.2) ** 2)
    keypoints = torch.tensor([[3.0, 2.0], [0.0, 0.0], [6.0, 6.0]])
    refined = quadratic_subpixel_keypoints(keypoints, score)
    torch.testing.assert_close(refined[0], torch.tensor([3.25, 2.2]))
    assert refined[1:].tolist() == keypoints[1:].tolist()
    assert refined.shape == keypoints.shape


def test_quadratic_subpixel_rejects_non_peak_axes_and_caps_offset():
    score = torch.zeros((5, 5))
    score[2, 1:4] = torch.tensor([0.0, 1.0, 0.9])
    keypoints = torch.tensor([[2.0, 2.0]])
    refined = quadratic_subpixel_keypoints(keypoints, score, maximum_offset=0.1)
    assert torch.all((refined - keypoints).abs() <= 0.1)
    assert refined[0, 1] == 2.0
