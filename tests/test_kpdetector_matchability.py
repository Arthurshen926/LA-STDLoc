import torch


def test_matchability_head_initialization_preserves_legacy_detector_heatmap():
    from scene.kpdetector import KpDetector

    torch.manual_seed(3)
    legacy = KpDetector(8)
    candidate = KpDetector(8, matchability_head=True)
    missing, unexpected = candidate.load_state_dict(legacy.state_dict(), strict=False)
    candidate.initialize_matchability_from_keypoint()
    feature_map = torch.randn(8, 12, 10)

    legacy_heatmap = legacy(feature_map)
    keypoint, matchability = candidate.forward_heads(feature_map)
    combined = candidate.forward_combined(feature_map)

    assert set(missing) == {"matchability_head.weight", "matchability_head.bias"}
    assert unexpected == []
    assert torch.allclose(keypoint, legacy_heatmap)
    assert torch.allclose(matchability, legacy_heatmap)
    assert torch.allclose(combined, legacy_heatmap, atol=1e-6)


def test_offset_head_initializes_to_zero_and_is_bounded():
    from scene.kpdetector import KpDetector

    detector = KpDetector(8, offset_head=True, max_offset=1.5)
    feature_map = torch.randn(8, 7, 9)
    _, _, offset = detector.forward_all(feature_map)

    assert offset.shape == (2, 7, 9)
    assert torch.count_nonzero(offset) == 0

    with torch.no_grad():
        detector.offset_head.bias.fill_(10.0)
    _, _, offset = detector.forward_all(feature_map)
    assert offset.abs().max() <= 1.5


def test_legacy_detector_load_allows_zero_initialized_offset_head():
    from scene.kpdetector import KpDetector

    legacy = KpDetector(8)
    detector = KpDetector(8, offset_head=True)
    incompatible = detector.load_state_dict(legacy.state_dict(), strict=False)
    detector.initialize_offset_to_zero()

    assert set(incompatible.missing_keys) == {
        "offset_head.weight",
        "offset_head.bias",
    }
    assert incompatible.unexpected_keys == []
