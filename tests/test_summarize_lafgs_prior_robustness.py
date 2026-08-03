import json
import os

import pytest
import torch

from scripts.summarize_lafgs_prior_robustness import (
    _tail_diagnostics,
    render_markdown,
    summarize_enhanced_matcha_profile,
    summarize_profile,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_profile_uses_nested_quality_summary(tmp_path):
    prior = tmp_path / "priors" / "Scene" / "vanilla_3dgs"
    lafgs = tmp_path / "lafgs" / "vanilla_3dgs" / "Scene"
    ply = prior / "prior.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply")
    _write(
        prior / "offtheshelf_prior_protocol.json",
        {
            "official_commit": "abc",
            "mapping_input": {"mapping_image_count": 10},
            "controls": {
                "test_rgb_used": False,
                "rgb_prior_semantic_object_sky_mask_used": False,
            },
            "rgb_prior": {"primitive_count": 100, "source_ply": str(ply)},
            "training_seconds": 12,
        },
    )
    _write(
        prior / "prior_quality.json",
        {
            "summary": {
                "query_count": 4,
                "psnr_db": {"mean": 18.5},
                "ssim": {"mean": 0.7},
                "lpips": {"mean": 0.2},
            }
        },
    )
    _write(
        lafgs / "frozen_results.json",
        {
            "deployment_total_bytes": 20,
            "results": {
                "A0_bootstrap": {
                    "seed_aggregate": {"median_te_cm": {"mean": 12.0}}
                },
                "A1_reconstructed": {
                    "seed_aggregate": {"median_te_cm": {"mean": 9.0}}
                },
            },
        },
    )

    record = summarize_profile(tmp_path, "Scene", "vanilla_3dgs")

    assert record["complete"] is True
    assert record["heldout_rgb_quality"]["psnr_db_mean"] == 18.5
    assert record["prior"]["ply_bytes"] == 3
    markdown = render_markdown([record])
    assert "12.00/-/-" in markdown
    assert "9.00/-/-" in markdown


def test_summarize_profile_uses_2dgs_provenance_statistics(tmp_path):
    lafgs = tmp_path / "lafgs" / "vanilla_2dgs" / "Scene"
    statistics = (
        lafgs
        / "runs"
        / "frozen_v1"
        / "statistics_combined_1000_frozen_g3_track_provenance_v1"
    )
    _write(
        statistics / "training_summary.json",
        {
            "landmark_statistics": {
                "track_count": 100,
                "track_observation_count": 500,
                "geometry_teacher_triangulated_track_count": 42,
                "geometry_teacher_high_confidence_track_count": 9,
            }
        },
    )

    record = summarize_profile(tmp_path, "Scene", "vanilla_2dgs")

    assert record["track_first"]["triangulated_track_count"] == 42


def test_summarize_profile_accepts_isolated_lafgs_namespace(tmp_path):
    prior = tmp_path / "priors" / "Scene" / "vanilla_3dgs"
    lafgs = tmp_path / "lafgs_strict_v2" / "vanilla_3dgs" / "Scene"
    _write(prior / "offtheshelf_prior_protocol.json", {"version": 1})
    _write(prior / "prior_quality.json", {"summary": {"query_count": 1}})
    _write(lafgs / "frozen_results.json", {"results": {"complete": True}})

    record = summarize_profile(
        tmp_path,
        "Scene",
        "vanilla_3dgs",
        lafgs_namespace="lafgs_strict_v2",
    )

    assert record["complete"] is True


def test_summarize_profile_reports_feedforward_cost_and_alignment(tmp_path):
    prior = tmp_path / "priors" / "Scene" / "anysplat_ff"
    lafgs = tmp_path / "lafgs" / "anysplat_ff" / "Scene"
    _write(
        prior / "offtheshelf_prior_protocol.json",
        {
            "model_load_seconds": 2.0,
            "feedforward_seconds": 3.0,
            "alignment_and_fusion_seconds": 4.0,
            "total_prior_seconds": 9.0,
            "feedforward_summary": {
                "windows": [
                    {"primitive_count": 20, "peak_vram_bytes": 2 * 1024**3}
                ]
            },
            "mapping_only_sim3_alignment": {
                "primitive_count": 10,
                "windows": [
                    {
                        "camera_center_error_m": {"p90": 0.5},
                        "camera_rotation_error_deg": {"p90": 1.5},
                    }
                ],
            },
        },
    )
    milestones = [
        lafgs / "contracts" / "rgb_prior.json",
        lafgs / "runs" / "frozen_v1" / "query_cache_native_fullres_k2048.pt",
        lafgs
        / "self_localization_reconstruction"
        / "complete_positive_teacher.pt",
        lafgs / "self_localization_reconstruction" / "training_report.json",
    ]
    for index, path in enumerate(milestones):
        _write(path, {"stage": index})
        timestamp = 100.0 + 60.0 * index
        os.utime(path, (timestamp, timestamp))

    record = summarize_profile(tmp_path, "Scene", "anysplat_ff")

    diagnostics = record["feedforward_prior_diagnostics"]
    assert diagnostics["raw_primitive_count"] == 20
    assert diagnostics["fused_primitive_count"] == 10
    assert diagnostics["peak_vram_gib"] == 2.0
    markdown = render_markdown([record])
    assert "2.00/3.00/4.00/9.00" in markdown
    assert "20->10" in markdown
    assert record["lafgs_build_diagnostics"]["lafgs_total_seconds"] == 180.0
    assert record["lafgs_build_diagnostics"]["prior_plus_lafgs_seconds"] == 189.0
    assert "1.00/1.00/1.00/3.00/3.15" in markdown


def test_summarize_profile_reports_fail_closed_feedforward_geometry_gate(tmp_path):
    prior = tmp_path / "priors" / "Scene" / "anysplat_ff_allviews"
    lafgs = tmp_path / "lafgs" / "anysplat_ff_allviews" / "Scene"
    _write(
        prior / "offtheshelf_prior_protocol.json",
        {
            "controls": {"prior_uses_complete_mapping_split": True},
            "feedforward_summary": {"windows": [{"primitive_count": 1}]},
            "mapping_only_sim3_alignment": {"windows": []},
        },
    )
    statistics = (
        lafgs
        / "runs"
        / "frozen_v1"
        / "statistics_combined_1000_frozen_g2_track_first_v1"
    )
    statistics.mkdir(parents=True)
    torch.save(
        {
            "diagnostics": {
                "geometry_teacher_triangulated_track_count": 5,
                "geometry_teacher_high_confidence_track_count": 0,
                "geometry_teacher_assigned_landmark_count": 0,
            },
            "track_geometry": {
                "triangulation_rendered_depth_absolute_median_m": torch.tensor(
                    [0.1, 0.2, 1.0]
                )
            },
        },
        statistics / "track_micro_anchor_payload.pt",
    )

    record = summarize_profile(
        tmp_path,
        "Scene",
        "anysplat_ff_allviews",
        lafgs_namespace="lafgs",
    )

    diagnostics = record["track_geometry_diagnostics"]
    assert diagnostics["geometry_gate_status"] == "fail_zero_high_confidence_tracks"
    assert diagnostics["rendered_depth_absolute_median_m"]["p50"] == pytest.approx(
        0.2
    )
    markdown = render_markdown([record])
    assert "Feed-Forward Geometry Gate" in markdown
    assert "fail_zero_high_confidence_tracks" in markdown


def test_enhanced_matcha_summary_exposes_selected_view_protocol(tmp_path):
    enhanced = tmp_path / "enhanced"
    quality = tmp_path / "quality"
    scene = enhanced / "Scene"
    prior = scene / "prior" / "rgb_matcha_2dgs"
    ply = prior / "point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply")
    _write(
        prior / "rgb_prior_manifest.json",
        {
            "primitive_count": 200,
            "source_ply": str(ply),
        },
    )
    _write(
        scene / "audit" / "matcha_protocol.json",
        {
            "scenes": {
                "Scene": {
                    "selected_camera_count": 20,
                    "dataset_training_camera_count": 100,
                    "uses_full_cambridge_training_split": False,
                }
            }
        },
    )
    _write(
        quality / "Scene.json",
        {"summary": {"query_count": 4, "psnr_db": {"mean": 12.5}}},
    )
    _write(
        scene / "frozen_results.json",
        {"results": {"A0_bootstrap": {}, "A1_reconstructed": {}}},
    )

    record = summarize_enhanced_matcha_profile(enhanced, quality, "Scene")

    assert record["complete"] is True
    assert record["prior"]["mapping_image_count"] == 20
    assert record["prior"]["available_mapping_image_count"] == 100
    assert record["prior"]["semantic_mask_used"] is True
    assert "| 20 | yes | 200 |" in render_markdown([record])


def test_tail_diagnostics_separates_persistent_and_seed_unstable(tmp_path):
    stage = {}
    for seed, errors in {
        "2026": [20.0, 150.0, 300.0],
        "2027": [20.0, 50.0, 320.0],
        "2028": [20.0, 140.0, 310.0],
    }.items():
        result = tmp_path / seed
        _write(
            result / "results.json",
            [
                {"image_name": f"q{index}", "sparse_TE": error}
                for index, error in enumerate(errors)
            ],
        )
        stage[seed] = {"result_path": str(result)}
    frozen = {"results": {"A1_reconstructed": stage}}

    diagnostics = _tail_diagnostics(frozen, "A1_reconstructed")

    assert diagnostics["persistent_queries"] == ["q2"]
    assert diagnostics["seed_unstable_queries"] == ["q1"]
    assert diagnostics["failure_union_count"] == 2
