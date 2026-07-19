import json


def _summary(
    path,
    *,
    te,
    ae,
    raw,
    inlier,
    logdet,
    count=46,
    protocol_sha256="pinned-r1-images",
):
    path.write_text(
        json.dumps(
            {
                "evaluation_camera_subset": "candidate_validation",
                "evaluation_camera_count": count,
                "evaluation_protocol": {
                    "protocol_sha256": protocol_sha256,
                },
                "sparse": {
                    "median_te": te,
                    "median_ae": ae,
                },
                "sparse_diagnostics": {
                    "sparse_diag_pre_selector_all_gt_precision_2px_mean": raw,
                    "sparse_diag_pre_selector_inlier_gt_precision_2px_mean": inlier,
                    (
                        "sparse_diag_pre_selector_inlier_pose_info_"
                        "translation_logdet_mean"
                    ): logdet,
                },
            }
        )
    )
    return path


def test_joint_gate_rejects_pose_gain_that_loses_inlier_cleanliness(tmp_path):
    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    candidate = _summary(
        tmp_path / "candidate.json",
        te=2.8,
        ae=0.14,
        raw=0.11,
        inlier=0.39,
        logdet=12.1,
    )
    report = select_checkpoint(
        control,
        tmp_path / "control.pt",
        [("candidate", candidate, tmp_path / "candidate.pt")],
    )
    assert report["candidates"][0]["accepted"] is False
    assert report["selected_tag"] == "control_strong"
    assert report["used_strong_fallback"] is True


def test_joint_gate_accepts_synchronized_improvement(tmp_path):
    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    candidate = _summary(
        tmp_path / "candidate.json",
        te=2.9,
        ae=0.14,
        raw=0.11,
        inlier=0.41,
        logdet=12.1,
    )
    report = select_checkpoint(
        control,
        tmp_path / "control.pt",
        [("candidate", candidate, tmp_path / "candidate.pt")],
    )
    assert report["candidates"][0]["accepted"] is True
    assert report["selected_tag"] == "candidate"
    assert report["used_strong_fallback"] is False


def test_joint_gate_rejects_mismatched_validation_split(tmp_path):
    import pytest

    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    candidate = _summary(
        tmp_path / "candidate.json",
        te=2.9,
        ae=0.14,
        raw=0.11,
        inlier=0.41,
        logdet=12.1,
        count=45,
    )
    with pytest.raises(ValueError, match="same validation protocol"):
        select_checkpoint(
            control,
            tmp_path / "control.pt",
            [("candidate", candidate, tmp_path / "candidate.pt")],
        )


def test_joint_gate_rejects_mismatched_image_protocol(tmp_path):
    import pytest

    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
        protocol_sha256="resolution-1",
    )
    candidate = _summary(
        tmp_path / "candidate.json",
        te=2.9,
        ae=0.14,
        raw=0.11,
        inlier=0.41,
        logdet=12.1,
        protocol_sha256="resolution-minus-1",
    )
    with pytest.raises(ValueError, match="same validation protocol"):
        select_checkpoint(
            control,
            tmp_path / "control.pt",
            [("candidate", candidate, tmp_path / "candidate.pt")],
        )


def test_joint_gate_requires_protocol_even_without_candidates(tmp_path):
    import pytest

    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
        protocol_sha256=None,
    )
    with pytest.raises(ValueError, match="requires evaluation_protocol"):
        select_checkpoint(
            control,
            tmp_path / "control.pt",
            [],
        )
