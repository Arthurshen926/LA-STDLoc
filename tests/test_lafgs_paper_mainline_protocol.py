from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_paper_mainline_is_compact_single_descriptor_one_pose_solve():
    config = yaml.safe_load(
        (ROOT / "configs" / "lafgs_paper_mainline.yaml").read_text()
    )
    assert config["method"]["status"] == "frozen"
    assert config["prior"]["frozen"] is True
    assert config["deployment"]["global_topk"] == 1
    assert config["deployment"]["max_matches_per_keypoint"] == 0
    assert config["deployment"]["max_matches_per_landmark"] == 0
    assert config["deployment"]["pose_solves"] == 1
    assert config["deployment"]["dense_refinement"] is False
    assert config["deployment"]["rendering"] is False
    excluded = set(config["method"]["excluded_by_default"])
    assert {
        "viewpoint_completion",
        "trajectory_stable_candidate_teacher",
        "family_prototypes",
        "selector",
        "pair_lgcv",
        "differentiable_pnp",
        "bounded_bundle_adjustment",
        "test_time_rendering",
    }.issubset(excluded)


def test_frozen_runner_defaults_to_a0_a1_without_legacy_prerequisites():
    runner = (
        ROOT / "scripts" / "run_lafgs_v1_frozen_multiscene.sh"
    ).read_text()
    assert (
        'EVAL_VARIANTS="${LAFGS_EVAL_VARIANTS_OVERRIDE:-A0_bootstrap '
        'A1_reconstructed}"'
    ) in runner
    assert '*" A2_family_all "*) family_refinement ;;' in runner
    assert '*) deployment_eval_prerequisites ;;' in runner
    assert '"$CONTRACTS/reconstructed_map.json"' in runner
    assert "offline query caches that are not read by stdloc.py" in runner


def test_off_the_shelf_matrix_is_mapping_only_and_gpu2_locked():
    runner = (
        ROOT / "scripts" / "run_lafgs_offtheshelf_prior_matrix.sh"
    ).read_text()
    assert 'if [[ "$GPU" != "2" ]]' in runner
    assert 'LAFGS_EVAL_VARIANTS_OVERRIDE="A0_bootstrap A1_reconstructed"' in runner
    assert '--depth_l1_weight_init 0 --depth_l1_weight_final 0' in runner
    assert '"test_rgb_used": False' in runner
    assert '"rgb_prior_semantic_object_sky_mask_used": False' in runner
    assert '"localization_valid_mask_policy": (' in runner
    assert '"lafgs_gradient_to_rgb_gaussian": False' in runner
    assert 'LAFGS_NAMESPACE="${LAFGS_OFFTHESHELF_LAFGS_NAMESPACE:-lafgs_strict_v2}"' in runner
    assert 'local full_cache="$run_root/query_cache_native_fullres_k2048.pt"' in runner
    assert 'local sparse_cache="$run_root/query_cache_native_sparse_teacher.pt"' in runner
    assert "stdloc_lafgs_v1_frozen_multiscene" not in runner
