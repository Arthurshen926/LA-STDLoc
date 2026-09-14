import torch

from evidence.conditional_track_fusion import conditional_artifact_keep_masks
from evidence.tracks import (
    LeaveOneQueryOutTrackDescriptorBank,
    fuse_track_descriptors,
)


def _fixture(*, identity_certified: bool = False, unique_outlier_bin: bool = False):
    names = [
        "seq-a/q0.png",
        "seq-a/q1.png",
        "seq-a/q2.png",
        "seq-b/q3.png",
        "seq-b/q4.png",
    ]
    descriptors = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    ]
    payload = {
        "query_names": names,
        "query_bins": torch.tensor([0, 0, 0 if not unique_outlier_bin else 2, 1, 1]),
        "tracks": {
            "track_index": torch.zeros(5, dtype=torch.long),
            "query_index": torch.arange(5),
            "keypoint_index": torch.zeros(5, dtype=torch.long),
            "confidence": torch.ones(5),
            "identity_positive_certified": torch.tensor(
                [False, False, identity_certified, False, False]
            ),
        },
    }
    appearance = {
        "queries": {
            name: {
                "native_keypoints": torch.tensor([[0.0, 0.0]]),
                "native_descriptors": descriptor[None],
                "native_valid_keypoint_mask": torch.tensor([True]),
                "native_appearance_reliability": torch.tensor([1.0]),
            }
            for name, descriptor in zip(names, descriptors)
        }
    }
    artifact = {
        "queries": {
            name: {"native_artifact_reliability": torch.tensor([value])}
            for name, value in zip(names, [0.9, 0.8, 0.01, 0.7, 0.6])
        }
    }
    return payload, appearance, artifact


def test_conditional_fusion_trims_only_joint_artifact_descriptor_outlier():
    payload, appearance, artifact = _fixture()
    annotations, diagnostics = conditional_artifact_keep_masks(
        payload=payload,
        appearance_cache=appearance,
        artifact_cache=artifact,
        selected_tracks=torch.tensor([0]),
    )
    assert annotations["seq-a/q2.png"][
        "native_conditional_artifact_trim_eligible"
    ].tolist() == [True]
    assert annotations["seq-a/q2.png"][
        "native_descriptor_fusion_keep_mask"
    ].tolist() == [False]
    assert diagnostics["trimmed_observation_count"] == 1
    assert diagnostics["artifact_evidence_used_as_weight"] is False


def test_conditional_fusion_protects_identity_seed_and_unique_view():
    for identity_certified, unique_outlier_bin, diagnostic in (
        (True, False, "protected_identity_certified_count"),
        (False, True, "protected_unique_view_count"),
    ):
        payload, appearance, artifact = _fixture(
            identity_certified=identity_certified,
            unique_outlier_bin=unique_outlier_bin,
        )
        annotations, diagnostics = conditional_artifact_keep_masks(
            payload=payload,
            appearance_cache=appearance,
            artifact_cache=artifact,
            selected_tracks=torch.tensor([0]),
        )
        assert annotations["seq-a/q2.png"][
            "native_descriptor_fusion_keep_mask"
        ].tolist() == [True]
        assert diagnostics[diagnostic] >= 1
        assert diagnostics["trimmed_observation_count"] == 0


def test_fusion_and_loo_honor_conditional_keep_mask():
    payload, appearance, artifact = _fixture()
    annotations, _ = conditional_artifact_keep_masks(
        payload=payload,
        appearance_cache=appearance,
        artifact_cache=artifact,
        selected_tracks=torch.tensor([0]),
    )
    cache = {
        "queries": {
            name: {**appearance["queries"][name], **annotations[name]}
            for name in payload["query_names"]
        }
    }
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=torch.tensor([0]),
        trim_fraction=0.0,
    )
    torch.testing.assert_close(fused[0], torch.tensor([1.0, 0.0]))
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=torch.tensor([0]),
        reference_features=fused,
        trim_fraction=0.0,
    )
    assert replay.rows_by_query[2] == []
    rows, features = replay.query_update(2)
    assert rows.numel() == 0
    assert features.shape == (0, 2)


def test_conditional_fusion_rejects_duplicate_selected_observation():
    payload, appearance, artifact = _fixture()
    for key in (
        "track_index",
        "query_index",
        "keypoint_index",
        "confidence",
        "identity_positive_certified",
    ):
        value = payload["tracks"][key]
        payload["tracks"][key] = torch.cat((value, value[:1]))
    payload["tracks"]["track_index"][-1] = 1
    try:
        conditional_artifact_keep_masks(
            payload=payload,
            appearance_cache=appearance,
            artifact_cache=artifact,
            selected_tracks=torch.tensor([0, 1]),
        )
    except ValueError as error:
        assert "reuse a query/keypoint" in str(error)
    else:
        raise AssertionError("duplicate selected observation was accepted")
