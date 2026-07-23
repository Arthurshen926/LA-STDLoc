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


def test_selector_preserves_nonbootstrap_control_identity(tmp_path):
    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    control = _summary(
        tmp_path / "residual.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    candidate = _summary(
        tmp_path / "ba.json",
        te=2.8,
        ae=0.20,
        raw=0.10,
        inlier=0.39,
        logdet=12.0,
    )

    report = select_checkpoint(
        control,
        tmp_path / "residual.pt",
        [("ba", candidate, tmp_path / "ba.pt")],
        control_tag="residual_5000",
    )

    assert report["control_tag"] == "residual_5000"
    assert report["selected_tag"] == "residual_5000"
    assert report["used_control_fallback"] is True


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


def test_performance_selector_uses_pose_objective_with_deployment_recall_guard(
    tmp_path,
):
    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    def write_per_query_results(summary_path, errors):
        summary_path.with_name("results.json").write_text(
            json.dumps([{"sparse_TE": error} for error in errors])
        )

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    control_payload = json.loads(control.read_text())
    control_payload["sparse"].update({"recall_2m_5d": 0.95, "recall_5cm_5d": 0.20})
    control.write_text(json.dumps(control_payload))
    write_per_query_results(control, [2.0, 4.0])

    preferred = _summary(
        tmp_path / "preferred.json",
        te=2.8,
        ae=0.16,
        raw=0.09,
        inlier=0.39,
        logdet=11.0,
    )
    preferred_payload = json.loads(preferred.read_text())
    preferred_payload["sparse"].update({"recall_2m_5d": 0.95, "recall_5cm_5d": 0.20})
    preferred.write_text(json.dumps(preferred_payload))
    write_per_query_results(preferred, [2.0, 4.0])

    rejected = _summary(
        tmp_path / "rejected.json",
        te=2.5,
        ae=0.10,
        raw=0.20,
        inlier=0.60,
        logdet=15.0,
    )
    rejected_payload = json.loads(rejected.read_text())
    rejected_payload["sparse"].update({"recall_2m_5d": 0.93, "recall_5cm_5d": 0.15})
    rejected.write_text(json.dumps(rejected_payload))
    write_per_query_results(rejected, [2.0, 3.0])

    report = select_checkpoint(
        control,
        tmp_path / "control.pt",
        [
            ("preferred", preferred, tmp_path / "preferred.pt"),
            ("rejected", rejected, tmp_path / "rejected.pt"),
        ],
        selection_mode="performance",
        mean_te_weight=0.05,
        max_recall_2m_drop=0.01,
        max_recall_5cm_drop=0.01,
    )

    assert report["selected_tag"] == "preferred"
    assert report["candidates"][0]["accepted"] is True
    assert report["candidates"][1]["accepted"] is False
    assert report["selection_protocol"]["test_metrics_used"] is False


def test_performance_selector_keeps_control_when_candidate_loses_primary_score(
    tmp_path,
):
    from scripts.select_lafgs_map_checkpoint import select_checkpoint

    def write_per_query_results(summary_path, errors):
        summary_path.with_name("results.json").write_text(
            json.dumps([{"sparse_TE": error} for error in errors])
        )

    control = _summary(
        tmp_path / "control.json",
        te=3.0,
        ae=0.15,
        raw=0.10,
        inlier=0.40,
        logdet=12.0,
    )
    control_payload = json.loads(control.read_text())
    control_payload["sparse"].update({"recall_2m_5d": 0.95, "recall_5cm_5d": 0.20})
    control.write_text(json.dumps(control_payload))
    write_per_query_results(control, [2.0, 4.0])

    worse = _summary(
        tmp_path / "worse.json",
        te=3.2,
        ae=0.10,
        raw=0.20,
        inlier=0.60,
        logdet=15.0,
    )
    worse_payload = json.loads(worse.read_text())
    worse_payload["sparse"].update({"recall_2m_5d": 0.95, "recall_5cm_5d": 0.20})
    worse.write_text(json.dumps(worse_payload))
    write_per_query_results(worse, [2.0, 4.0])

    report = select_checkpoint(
        control,
        tmp_path / "control.pt",
        [("worse", worse, tmp_path / "worse.pt")],
        selection_mode="performance",
        mean_te_weight=0.05,
        max_recall_2m_drop=0.01,
        max_recall_5cm_drop=0.01,
    )

    assert report["candidates"][0]["gate_checks"]["primary_objective_gain"] is False
    assert report["candidates"][0]["accepted"] is False
    assert report["selected_tag"] == "control_strong"
    assert report["used_strong_fallback"] is True
