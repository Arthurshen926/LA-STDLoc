import pytest
import torch
import torch.nn.functional as F

from evidence.tracks import (
    LeaveOneQueryOutTrackDescriptorBank,
    fuse_track_descriptors,
)
from map_learning.metric import SharedLowRankMetric
from map_learning.trainer import (
    bounded_anchor_bank,
    bounded_query_anchor_bank,
    track_descriptor_payload_for_loo,
)
from scripts.evaluate_rendered_track_fullmap import _DeviceBankUpdater


def _fixture() -> tuple[dict, dict, torch.Tensor]:
    names = ["seq/q0.png", "seq/q1.png", "seq/q2.png"]
    payload = {
        "query_names": names,
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "track_index": torch.tensor([0, 0, 0, 1, 1]),
            "query_index": torch.tensor([0, 1, 2, 1, 2]),
            "keypoint_index": torch.tensor([0, 0, 0, 1, 1]),
            "confidence": torch.ones(5),
        },
    }
    descriptors = (
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    cache = {
        "queries": {
            name: {
                "native_descriptors": value,
                "native_valid_keypoint_mask": torch.ones(2, dtype=torch.bool),
            }
            for name, value in zip(names, descriptors)
        }
    }
    tracks = torch.tensor([0, 1])
    return payload, cache, tracks


def test_full_mapping_loo_removes_only_current_query_descriptor() -> None:
    payload, cache, tracks = _fixture()
    reference = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        trim_fraction=0.0,
    )
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        reference_features=reference,
        trim_fraction=0.0,
    )
    rows, features = replay.query_update(0)
    assert rows.tolist() == [0]
    assert torch.allclose(features[0], torch.tensor([0.0, 1.0]))
    assert replay.rows_by_query[0] == [0]
    assert replay.rows_by_query[1] == [0, 1]

    unaffected = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=torch.tensor([1]),
        reference_features=reference[1:],
        trim_fraction=0.0,
    )
    rows, features = unaffected.query_update(0)
    assert rows.numel() == 0
    assert features.shape == (0, 2)


def test_device_bank_updater_restores_previous_query_rows() -> None:
    payload, cache, tracks = _fixture()
    reference = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        trim_fraction=0.0,
    )
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        reference_features=reference,
        trim_fraction=0.0,
    )
    updater = _DeviceBankUpdater(replay, torch.device("cpu"))
    bank = F.normalize(reference, dim=1).clone()
    updater(0, bank)
    assert torch.allclose(bank[0], torch.tensor([0.0, 1.0]))
    updater(1, bank)
    # Track 0 is restored first, then recomputed without q1.  The result is
    # no longer the q0 exclusion from the preceding feedback query.
    assert not torch.equal(bank[0], torch.tensor([0.0, 1.0]))
    assert updater.affected_anchor_updates == 3


def test_loo_metric_applies_to_query_conditioned_raw_bank() -> None:
    payload, cache, tracks = _fixture()
    reference = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        trim_fraction=0.0,
    )
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        reference_features=reference,
        trim_fraction=0.0,
    )
    metric = SharedLowRankMetric(descriptor_dim=2, rank=1, max_residual_norm=0.5)
    with torch.no_grad():
        metric.down.weight.copy_(torch.tensor([[1.0, -0.5]]))
        metric.down.bias.fill_(0.2)
        metric.up.weight.copy_(torch.tensor([[0.3], [-0.4]]))
    adapted, _, _ = bounded_anchor_bank(metric, reference, None, 0.0)
    state = {
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
    }
    updater = _DeviceBankUpdater(
        replay,
        torch.device("cpu"),
        metric_state=state,
        adapted_reference_features=adapted,
    )
    bank = adapted.clone()
    rows, loo_raw = replay.query_update(0)
    expected, _, _ = bounded_anchor_bank(metric, loo_raw, None, 0.0)
    updater(0, bank)
    assert torch.allclose(bank[rows], expected)
    assert not torch.allclose(bank[rows], F.normalize(loo_raw, dim=1))

    training_bank, _, _, affected = bounded_query_anchor_bank(
        metric=metric,
        raw_features=reference,
        query_index=0,
        loo_descriptor_bank=replay,
        anchor_residual_parameter=None,
        maximum_norm=0.0,
    )
    assert affected == rows.numel()
    assert torch.allclose(training_bank[rows], expected)
    untouched = torch.ones(reference.shape[0], dtype=torch.bool)
    untouched[rows] = False
    assert torch.equal(training_bank[untouched], adapted[untouched])


def test_training_loo_uses_pose_view_bins_not_group_dro_sequence_bins() -> None:
    payload, cache, tracks = _fixture()
    reference = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        trim_fraction=0.0,
    )
    training_payload = {
        **payload,
        "pose_view_bins": payload["query_bins"].clone(),
        "query_bins": torch.zeros_like(payload["query_bins"]),
    }
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=track_descriptor_payload_for_loo(training_payload),
        query_cache=cache,
        track_indices=tracks,
        reference_features=reference,
        trim_fraction=0.0,
    )
    assert replay.query_update(0)[0].numel() == 1


def test_loo_rejects_non_exact_reference_and_sole_observation() -> None:
    payload, cache, tracks = _fixture()
    reference = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=tracks,
        trim_fraction=0.0,
    )
    with pytest.raises(ValueError, match="exact full-observation"):
        LeaveOneQueryOutTrackDescriptorBank(
            payload=payload,
            query_cache=cache,
            track_indices=tracks,
            reference_features=reference + 0.01,
            trim_fraction=0.0,
        )

    single_payload = {
        "query_names": ["q0"],
        "query_bins": torch.tensor([0]),
        "tracks": {
            "track_index": torch.tensor([0]),
            "query_index": torch.tensor([0]),
            "keypoint_index": torch.tensor([0]),
            "confidence": torch.ones(1),
        },
    }
    single_cache = {
        "queries": {"q0": {"native_descriptors": torch.tensor([[1.0, 0.0]])}}
    }
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=single_payload,
        query_cache=single_cache,
        track_indices=torch.tensor([0]),
        reference_features=torch.tensor([[1.0, 0.0]]),
        trim_fraction=0.0,
    )
    with pytest.raises(ValueError, match="sole observation"):
        replay.query_update(0)
