import json

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
