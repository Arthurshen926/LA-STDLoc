import torch

from localization_training.synthetic_evidence import (
    RenderQualityFilterConfig,
    SyntheticEvidenceConfig,
    build_render_quality_mask,
    keypoint_positive_csr,
    project_existing_anchors,
    render_visible_anchor_mask,
)


class _Mask:
    valid_mask = torch.ones(8, 8, dtype=torch.bool)
    support_mask = torch.ones(8, 8, dtype=torch.bool)
    invalid_score = torch.zeros(8, 8)
    support_score = torch.ones(8, 8)
    channel_maps = {}
    summary = {"valid_frac": 1.0}


def test_render_quality_mask_rejects_reference_inconsistent_region():
    rendered = torch.zeros(3, 8, 8)
    rendered[:, :, 4:] = 1.0
    reference = torch.zeros(3, 8, 8)
    result = build_render_quality_mask(
        base_mask=_Mask(),
        rendered_rgb=rendered,
        reference_rgbs=[reference],
        alpha=torch.ones(1, 8, 8),
        rendered_depth=torch.ones(1, 8, 8),
        surface_normal=torch.ones(3, 8, 8),
        config=RenderQualityFilterConfig(
            reference_downsample=1,
            maximum_reference_residual=0.2,
            invalid_dilate_radius=0,
        ),
    )
    assert result.valid_mask[:, :4].all()
    assert not result.valid_mask[:, 4:].any()
    assert result.summary["reference_valid_frac"] == 0.5


def test_render_visibility_requires_alpha_and_depth_consistency():
    config = SyntheticEvidenceConfig(
        minimum_alpha=0.5,
        absolute_depth_tolerance=0.1,
        relative_depth_tolerance=0.0,
    )
    projected = torch.tensor([[2.0, 2.0], [5.0, 5.0], [7.0, 7.0]])
    anchor_depth = torch.tensor([2.0, 2.0, 2.0])
    depth = torch.full((1, 10, 10), 2.0)
    depth[:, 5, 5] = 3.0
    alpha = torch.ones((1, 10, 10))
    alpha[:, 7, 7] = 0.1
    visible = render_visible_anchor_mask(
        projected_xy=projected,
        anchor_depth=anchor_depth,
        rendered_depth=depth,
        alpha=alpha,
        width=10,
        height=10,
        config=config,
    )
    assert visible.tolist() == [True, False, False]


def test_synthetic_positive_csr_only_uses_existing_anchor_indices():
    config = SyntheticEvidenceConfig(
        positive_radius_px=1.0,
        positives_per_keypoint=2,
    )
    rows, offsets, positives = keypoint_positive_csr(
        keypoints=torch.tensor([[1.1, 1.0], [8.0, 8.0]]),
        projected_xy=torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [8.4, 8.0]]
        ),
        visible_anchor_indices=torch.tensor([0, 2]),
        config=config,
    )
    assert rows.tolist() == [0, 1]
    assert offsets.tolist() == [0, 1, 2]
    assert positives.tolist() == [0, 2]


def test_project_existing_anchors_uses_fixed_world_geometry():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, -1.0]])
    K = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    projected, depth, valid = project_existing_anchors(
        xyz, torch.eye(4), K
    )
    assert torch.allclose(projected[0], torch.tensor([50.0, 40.0]))
    assert depth.tolist() == [2.0, -1.0]
    assert valid.tolist() == [True, False]
