import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_cambridge_matcha_pretrained_mainline.sh"


def test_matcha_mainline_defaults_to_f0_and_keeps_exact_optional():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'CANDIDATE_OBJECTIVE="${CAMBRIDGE_MATCHA_CANDIDATE_OBJECTIVE:-f0}"' in text
    assert "detector_queryjoint_${CANDIDATE_OBJECTIVE}_normalized${pair_suffix}_" in text
    assert 'if [[ "$CANDIDATE_OBJECTIVE" == "exact" ]]' in text
    assert "--candidate_teacher_counterfactual_exact_decision_set" in text


def test_matcha_mainline_calibrates_final_frontend_without_calibration_recursion():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'USE_CALIBRATED_FINAL_FRONTEND="${CAMBRIDGE_MATCHA_USE_CALIBRATED_FINAL_FRONTEND:-1}"' in text
    assert 'reprojection_error_px="$CALIBRATED_RESIDUAL_CLIP_PX"' in text
    assert '[[ "$variant" == "candidate" ]] || [[ "$subset" == "test" ]]' in text
    assert 'run_eval validation detector "$DETECTOR_STEPS"' in text


def test_matcha_mainline_shell_parses():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_matcha_mainline_selects_validation_checkpoint_before_test():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'run_eval validation candidate "$CANDIDATE_Q1"' in text
    assert 'run_eval validation candidate "$CANDIDATE_Q2"' in text
    assert 'run_eval validation candidate "$CANDIDATE_STEPS"' in text
    assert 'eval "$(select_candidate_checkpoint)"' in text
    assert 'run_eval test candidate "$SELECTED_CANDIDATE_ITERATION"' in text


def test_matcha_mainline_has_bounded_heldout_topology_smoke():
    text = RUNNER.read_text(encoding="utf-8")

    assert "CAMBRIDGE_MATCHA_SMOKE_TOPOLOGY" in text
    assert "--topology_min_radius 0" in text
    assert "--topology_growth_cap_per_event 0.00001" in text
    assert "--topology_max_mutation_events 1" in text
    assert "--topology_risk_commit_policy heldout_descriptor" in text


def test_matcha_mainline_exposes_pair_fixed_budget_and_geometry_refill():
    text = RUNNER.read_text()
    assert "CAMBRIDGE_MATCHA_PAIR_FIXED_CANDIDATE_COUNT" in text
    assert "CAMBRIDGE_MATCHA_PAIR_REFILL_MODE" in text
    assert '--pair_measurement_fixed_candidate_count "$PAIR_FIXED_CANDIDATE_COUNT"' in text
    assert '--pair_measurement_refill_mode "$PAIR_REFILL_MODE"' in text
    assert "--use_pair_measurement_covariance_refinement" in text
    assert "--use_pair_measurement_progressive_sampling" in text


def test_matcha_mainline_has_online_render_curriculum_for_candidate_teacher():
    text = RUNNER.read_text(encoding="utf-8")

    assert "CAMBRIDGE_MATCHA_CANDIDATE_ONLINE_RENDER" in text
    assert "CAMBRIDGE_MATCHA_ONLINE_RENDER_RATIO_START:-0.10" in text
    assert "CAMBRIDGE_MATCHA_ONLINE_RENDER_RATIO_END:-0.30" in text
    assert "CAMBRIDGE_MATCHA_ONLINE_RENDER_PROVENANCE:-none" in text
    assert "--candidate_teacher_online_render_provenance_mode" in text
    load_block, candidate_block = text.split("run_candidate() {", 1)
    assert "--candidate_teacher_online_render_ratio_start" not in load_block
    assert "--candidate_teacher_online_render_ratio_start" in candidate_block
