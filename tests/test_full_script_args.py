import re
import tempfile
import unittest
import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_cambridge_full.sh"
V02_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v02_shopfacade.sh"
V03_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_shopfacade.sh"
V03_MULTISCENE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_multiscene.sh"
V03_TOPOLOGY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_topology_full.sh"
DENSE_KL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_densekl_v03_cambridge.sh"
DENSE_LONG_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_update2_dense_long_worker.sh"
TOPOLOGY_LONG_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_update2_long_worker.sh"
UPDATE3_P0_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_update3_p0_worker.sh"
PREPARE_BASELINES_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cambridge_baseline_artifacts.sh"
GEOM_2X2_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_geometry_balance_2x2_shopfacade.sh"
DESCRIPTOR_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_descriptors.py"
INLIER_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_inliers.py"
REMAP_TOPOLOGY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remap_topology_landmarks.py"
DENSE_RESP_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_dense_responsibility.py"
PSEUDO_QUERY_PIPELINE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_pseudo_query_pipeline.sh"
CLEAN_PSEUDO_QUERY_MAINLINE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_la_clean_real_train_mainline.sh"
)
CLEAN_CONTROL_MATRIX_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_clean_control_matrix.sh"
OLDHOSPITAL_OBJECTIVE_ABLATION_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_la_oldhospital_objective_ablation.sh"
)
CAPACITY_ABLATION_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_capacity_fullchain_ablation.sh"
REFACTORED_MAINLINE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_refactored_mainline.sh"
SELECT_PSEUDO_QUERY_POOL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_pseudo_query_pool.py"
PLAIN_SPARSE_EVAL_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_lafgs_plain_sparse_eval.sh"
)
STDLOC_SCRIPT = Path(__file__).resolve().parents[1] / "stdloc.py"


class FullRunScriptArgsTest(unittest.TestCase):
    def _command_blocks(self, command_name):
        text = SCRIPT.read_text()
        pattern = re.compile(rf'"\$PYTHON" {re.escape(command_name)} \\\n(?P<body>.*?)(?=\n"\$PYTHON"|\nif |\nelse|\nfi|\Z)', re.S)
        return [match.group("body") for match in pattern.finditer(text)]

    def _load_descriptor_diag_module(self):
        spec = importlib.util.spec_from_file_location("descriptor_diag_script", DESCRIPTOR_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_inlier_diag_module(self):
        spec = importlib.util.spec_from_file_location("inlier_diag_script", INLIER_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_dense_resp_diag_module(self):
        spec = importlib.util.spec_from_file_location("dense_resp_diag_script", DENSE_RESP_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_eval_commands_do_not_receive_training_only_args(self):
        for command_name in ("stdloc.py", "cache_sparse_poses.py"):
            with self.subTest(command=command_name):
                blocks = self._command_blocks(command_name)
                self.assertTrue(blocks, f"{command_name} is not invoked by the full run script")
                for block in blocks:
                    self.assertIn('"${DATA_ARGS[@]}"', block)
                    self.assertNotIn('"${TRAIN_ARGS[@]}"', block)
                    self.assertNotIn('"${COMMON_ARGS[@]}"', block)

    def test_la_training_phases_are_resume_safe(self):
        text = SCRIPT.read_text()
        for phase_end in ("FEATURE_END", "GEOMETRY_END", "TOPOLOGY_END", "CLOSED_LOOP_END"):
            with self.subTest(phase_end=phase_end):
                self.assertIn(f'if ! point_cloud_exists "$LA_MODEL" "${phase_end}"; then', text)

    def test_v02_script_keeps_sparse_pipeline_fixed(self):
        text = V02_SCRIPT.read_text()
        self.assertIn("--loc_teacher direct", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertIn("--loc_multiview_weight 0.05", text)
        self.assertIn("--support_query_split", text)
        self.assertIn("--loc_proto_weight 0.0", text)
        self.assertIn("--loc_rank_weight 0.0", text)
        self.assertIn("--direct_depth_check", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)
        self.assertIn("--prefix \"phase-v02-${checkpoint}\"", text)

    def test_v02_script_does_not_default_to_invalid_e3_fixed_baseline_indices(self):
        text = V02_SCRIPT.read_text()
        self.assertIn("RUN_E3=${RUN_E3:-0}", text)
        self.assertNotIn("phase-e3-40k-fixed-baseline-sparse", text)
        self.assertIn("phase-e3-40k-baseline-hard-sparse", text)

    def test_v03_script_runs_feature_only_full_bank_anchor_without_topology(self):
        text = V03_SCRIPT.read_text()
        self.assertIn("--loc_teacher direct", text)
        self.assertIn("--loc_full_bank_weight", text)
        self.assertIn("--loc_full_bank_hard_negatives", text)
        self.assertIn("--loc_full_bank_ignore_3d_radius", text)
        self.assertIn("--loc_full_bank_ignore_uv_radius", text)
        self.assertIn("V03_FULL_BANK_SOURCE_MODE=${V03_FULL_BANK_SOURCE_MODE:-ignore}", text)
        self.assertIn('--loc_full_bank_source_mode "$V03_FULL_BANK_SOURCE_MODE"', text)
        self.assertIn("V03_FULL_BANK_NEARBY_AS_POSITIVE=${V03_FULL_BANK_NEARBY_AS_POSITIVE:-0}", text)
        self.assertIn("--loc_full_bank_nearby_as_positive", text)
        self.assertIn("V03_LOC_OVERLAY_MODE=${V03_LOC_OVERLAY_MODE:-none}", text)
        self.assertIn("V03_LOC_OVERLAY_LR=${V03_LOC_OVERLAY_LR:-0.0}", text)
        self.assertIn("V03_LOC_OVERLAY_ACTIVE_LOGIT=${V03_LOC_OVERLAY_ACTIVE_LOGIT:-0.0}", text)
        self.assertIn("V03_LOC_OVERLAY_MAX_RESIDUAL_NORM=${V03_LOC_OVERLAY_MAX_RESIDUAL_NORM:-0.0}", text)
        self.assertIn("V03_LOC_OVERLAY_NORMALIZE=${V03_LOC_OVERLAY_NORMALIZE:-0}", text)
        self.assertIn("V03_LOC_OVERLAY_REG_WEIGHT=${V03_LOC_OVERLAY_REG_WEIGHT:-0.0}", text)
        self.assertIn("V03_CHILD_RESPONSIBILITY_MODE=${V03_CHILD_RESPONSIBILITY_MODE:-none}", text)
        self.assertIn("V03_CHILD_RESPONSIBILITY_START_ITER=${V03_CHILD_RESPONSIBILITY_START_ITER:-0}", text)
        self.assertIn('--loc_overlay_mode "$V03_LOC_OVERLAY_MODE"', text)
        self.assertIn('--loc_overlay_lr "$V03_LOC_OVERLAY_LR"', text)
        self.assertIn('--loc_overlay_active_logit "$V03_LOC_OVERLAY_ACTIVE_LOGIT"', text)
        self.assertIn('--loc_overlay_max_residual_norm "$V03_LOC_OVERLAY_MAX_RESIDUAL_NORM"', text)
        self.assertIn('"${LOC_OVERLAY_ARGS[@]}"', text)
        self.assertIn('--loc_overlay_reg_weight "$V03_LOC_OVERLAY_REG_WEIGHT"', text)
        self.assertIn('--loc_child_responsibility_mode "$V03_CHILD_RESPONSIBILITY_MODE"', text)
        self.assertIn('--loc_child_responsibility_start_iter "$V03_CHILD_RESPONSIBILITY_START_ITER"', text)
        self.assertIn("--loc_anchor_weight", text)
        self.assertIn("--loc_opacity_weight 0.0", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertIn("--support_query_split", text)
        self.assertNotIn("--enable_topology", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)

    def test_v03_script_accepts_query_split_seed_for_multiseed_runs(self):
        text = V03_SCRIPT.read_text()
        self.assertIn("V03_QUERY_SPLIT_SEED=${V03_QUERY_SPLIT_SEED:-2025}", text)
        self.assertIn("V03_TRAIN_SEED=${V03_TRAIN_SEED:-0}", text)
        self.assertIn("V03_QUERY_SPLIT_MODE=${V03_QUERY_SPLIT_MODE:-random}", text)
        self.assertIn('--train_seed "$V03_TRAIN_SEED"', text)
        self.assertIn('--query_split_seed "$V03_QUERY_SPLIT_SEED"', text)
        self.assertIn('--query_split_mode "$V03_QUERY_SPLIT_MODE"', text)
        self.assertNotIn("--query_split_seed 2025", text)

    def test_v03_multiscene_script_dispatches_scene_seed_matrix_safely(self):
        text = V03_MULTISCENE_SCRIPT.read_text()
        self.assertIn("SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}", text)
        self.assertIn("TRAIN_SEEDS=${TRAIN_SEEDS:-0 1 2}", text)
        self.assertIn("QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-2025 2026 2027}", text)
        self.assertIn('BASELINE_MODEL="$BASELINE_ROOT/${scene}_baseline"', text)
        self.assertIn('if [[ ! -d "$BASELINE_MODEL" ]]; then', text)
        self.assertIn('if [[ ! -f "$BASELINE_MODEL/detector/30000_detector.pth" || ! -f "$BASELINE_MODEL/detector/sampled_idx.pkl" ]]; then', text)
        self.assertIn("run_locaware_v03_shopfacade.sh", text)
        self.assertIn('V03_TRAIN_SEED="$train_seed"', text)
        self.assertIn('V03_QUERY_SPLIT_SEED="$query_split_seed"', text)
        self.assertIn('MODEL_ROOT="$MODEL_ROOT/${scene}/train_seed_${train_seed}/query_split_${query_split_seed}"', text)

    def test_train_locaware_seeds_after_safe_state_from_cli_arg(self):
        text = (Path(__file__).resolve().parents[1] / "train_locaware.py").read_text()

        self.assertIn("--train_seed", text)
        self.assertNotIn("seed_everything(2025)", text)
        self.assertLess(text.index("safe_state(args.quiet)"), text.index("seed_everything(args.train_seed)"))

    def test_clean_real_train_mainline_hard_disables_experimental_branches(self):
        self.assertTrue(CLEAN_PSEUDO_QUERY_MAINLINE_SCRIPT.exists(), "clean real-train mainline script is missing")
        text = CLEAN_PSEUDO_QUERY_MAINLINE_SCRIPT.read_text()
        self.assertIn("export LA_ENABLE_SYNTHETIC=0", text)
        self.assertIn("export SYNTHETIC_COUNT=0", text)
        self.assertIn("export TEACHER_CACHE_SPARSE_VALID_MASK=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_GATE=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_SELECT=0", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_FILTER_TEACHER_CACHE=0", text)
        self.assertIn("export PSEUDO_QUERY_ENABLE_TEACHER_GATE=0", text)
        self.assertIn("export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=0", text)
        self.assertIn("export LA_DIRECT_DEPTH_CHECK=0", text)
        self.assertIn('exec "$SCRIPT_DIR/run_la_pseudo_query_pipeline.sh"', text)

    def test_clean_control_matrix_runs_validated_controls_with_logs(self):
        self.assertTrue(CLEAN_CONTROL_MATRIX_SCRIPT.exists(), "clean control matrix script is missing")
        text = CLEAN_CONTROL_MATRIX_SCRIPT.read_text()
        self.assertIn("LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-2000}", text)
        self.assertIn("RUN_SHOP_8192=${RUN_SHOP_8192:-1}", text)
        self.assertIn("RUN_OLD_8192=${RUN_OLD_8192:-1}", text)
        self.assertIn("RUN_OLD_16384=${RUN_OLD_16384:-1}", text)
        self.assertIn("SEED_SHOP_8192=${SEED_SHOP_8192:-301}", text)
        self.assertIn("SEED_OLD_8192=${SEED_OLD_8192:-302}", text)
        self.assertIn("SEED_OLD_16384=${SEED_OLD_16384:-303}", text)
        self.assertIn(
            'bash "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh" 2>&1 | tee "$LOG_ROOT/$log_name"',
            text,
        )

    def test_oldhospital_objective_ablation_keeps_clean_data_boundary(self):
        self.assertTrue(
            OLDHOSPITAL_OBJECTIVE_ABLATION_SCRIPT.exists(),
            "OldHospital objective ablation script is missing",
        )
        text = OLDHOSPITAL_OBJECTIVE_ABLATION_SCRIPT.read_text()
        self.assertIn("export SCENES=OldHospital", text)
        self.assertIn("LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-500}", text)
        self.assertIn("export LA_ENABLE_SYNTHETIC=0", text)
        self.assertIn("export SYNTHETIC_COUNT=0", text)
        self.assertIn("export TEACHER_CACHE_SOURCES=train_rgb", text)
        self.assertIn("export TEACHER_CACHE_SPARSE_VALID_MASK=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_GATE=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_SELECT=0", text)
        self.assertIn("export PSEUDO_QUERY_FILTER_TEACHER_CACHE=0", text)
        self.assertIn("export PSEUDO_QUERY_ENABLE_TEACHER_GATE=0", text)
        self.assertIn("export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=0", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_MODE=soft", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_LOSS_MODE=soft", text)
        self.assertIn("export PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct", text)
        self.assertIn("export PSEUDO_QUERY_STAGE_STATS_POLICY=${PSEUDO_QUERY_STAGE_STATS_POLICY:-soft}", text)
        self.assertIn(
            "export PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT:-0.0}",
            text,
        )
        self.assertIn('bash "$SCRIPT_DIR/run_la_pseudo_query_pipeline.sh" 2>&1 | tee', text)

    def test_pseudo_query_pipeline_forwards_soft_stats_controls(self):
        self.assertTrue(PSEUDO_QUERY_PIPELINE_SCRIPT.exists(), "pseudo-query pipeline script is missing")
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()
        self.assertIn("PSEUDO_QUERY_STAGE_STATS_POLICY=${PSEUDO_QUERY_STAGE_STATS_POLICY:-hard}", text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT:-}", text)
        self.assertIn("pseudo_reliability_stats_args=()", text)
        self.assertIn("pseudo_reliability_stats_args+=(", text)
        self.assertIn("--pseudo_query_reliability_stats_min_weight", text)
        self.assertIn('"$PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT"', text)
        self.assertIn('"${pseudo_reliability_stats_args[@]}"', text)
        self.assertIn('--pseudo_query_stage_stats_policy "$PSEUDO_QUERY_STAGE_STATS_POLICY"', text)

    def test_pseudo_query_pipeline_forwards_pnp_sampling_controls(self):
        self.assertTrue(PSEUDO_QUERY_PIPELINE_SCRIPT.exists(), "pseudo-query pipeline script is missing")
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()
        self.assertIn("LA_DETECTOR_PNP_VOXEL_SIZE=${LA_DETECTOR_PNP_VOXEL_SIZE:-0.25}", text)
        self.assertIn("LA_DETECTOR_PNP_MAX_PER_VOXEL=${LA_DETECTOR_PNP_MAX_PER_VOXEL:-8}", text)
        self.assertIn("LA_DETECTOR_PNP_PRESERVE_RATIO=${LA_DETECTOR_PNP_PRESERVE_RATIO:-0.5}", text)
        self.assertIn('--pnp_voxel_size "$LA_DETECTOR_PNP_VOXEL_SIZE"', text)
        self.assertIn('--pnp_max_per_voxel "$LA_DETECTOR_PNP_MAX_PER_VOXEL"', text)
        self.assertIn('--pnp_preserve_ratio "$LA_DETECTOR_PNP_PRESERVE_RATIO"', text)

    def test_capacity_fullchain_ablation_reuses_cache_and_runs_clean_mainline(self):
        self.assertTrue(CAPACITY_ABLATION_SCRIPT.exists(), "capacity ablation script is missing")
        text = CAPACITY_ABLATION_SCRIPT.read_text()
        self.assertIn("LANDMARK_NUM=${LANDMARK_NUM:-8192}", text)
        self.assertIn("PSEUDO_QUERY_SOURCE_ROOT=${PSEUDO_QUERY_SOURCE_ROOT:-/mnt/pool/sqy/stdloc_la_mainline_refactor_2000_20260630}", text)
        self.assertIn('cp -a "$source_pseudo_dir/." "$target_pseudo_dir/"', text)
        self.assertIn("export RUN_PSEUDO_QUERY_MANIFEST=0", text)
        self.assertIn("export RUN_TEACHER_CACHE=0", text)
        self.assertIn("export RUN_TEACHER_CACHE_AUDIT=1", text)
        self.assertIn('export LA_BOOTSTRAP_LANDMARK_NUM="$LANDMARK_NUM"', text)
        self.assertIn('export LA_DETECTOR_LANDMARK_NUM="$LANDMARK_NUM"', text)
        self.assertIn('export SCENES="$SCENE"', text)
        self.assertIn('exec "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh"', text)

    def test_refactored_mainline_uses_matcha_synthetic_without_teacher_gate_or_selector(self):
        self.assertTrue(REFACTORED_MAINLINE_SCRIPT.exists(), "refactored LA mainline script is missing")
        text = REFACTORED_MAINLINE_SCRIPT.read_text()
        self.assertIn("export LA_ENABLE_SYNTHETIC=1", text)
        self.assertIn("export RENDER_SYNTHETIC_BACKEND=matcha", text)
        self.assertIn("export PSEUDO_QUERY_POSE_SAMPLER=spatial_offset", text)
        self.assertIn("export SYNTHETIC_QUALITY_GATE=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_GATE=0", text)
        self.assertIn("export RUN_PSEUDO_QUERY_SELECT=0", text)
        self.assertIn("export PSEUDO_QUERY_ENABLE_TEACHER_GATE=0", text)
        self.assertIn("export PSEUDO_QUERY_FILTER_TEACHER_CACHE=0", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=none", text)
        self.assertIn("export PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=0", text)
        self.assertIn("export TEACHER_CACHE_SPARSE_VALID_MASK=1", text)
        self.assertIn("export TEACHER_CACHE_SPARSE_VALID_MASK_MODE=support_mask_score", text)
        self.assertIn("export TEACHER_CACHE_SPARSE_VALID_MASK_SOURCES=synthetic_rgb", text)
        self.assertIn("export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=1", text)
        self.assertIn("export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SOURCES=synthetic_rgb", text)
        self.assertIn("export PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=1", text)
        self.assertIn("export LA_TRAIN_MODE=scratch", text)
        self.assertIn("export RUN_TEACHER_CACHE=1", text)
        self.assertIn("export RUN_TRAIN=1", text)
        self.assertIn("export RUN_EVAL=1", text)
        self.assertIn("export RUN_LA_FRONTEND_REFRESH=1", text)
        self.assertIn('exec "$SCRIPT_DIR/run_la_pseudo_query_pipeline.sh"', text)

    def test_refactored_mainline_lowers_frontend_min_observations_for_short_smoke(self):
        self.assertTrue(REFACTORED_MAINLINE_SCRIPT.exists(), "refactored LA mainline script is missing")
        text = REFACTORED_MAINLINE_SCRIPT.read_text()
        self.assertIn('if [[ -z "${LA_DETECTOR_MIN_LOC_OBSERVATIONS+x}" ]]; then', text)
        self.assertIn("if (( LA_ADAPT_STEPS < 4 )); then", text)
        self.assertIn("export LA_DETECTOR_MIN_LOC_OBSERVATIONS=1", text)
        self.assertIn("export LA_DETECTOR_MIN_LOC_OBSERVATIONS=4", text)
        self.assertIn("export LA_DETECTOR_MIN_LOC_OBSERVATIONS", text)

    def test_v03_topology_script_matches_v03_direct_objective_by_default(self):
        self.assertTrue(V03_TOPOLOGY_SCRIPT.exists(), "v0.3 topology rerun script is missing")
        text = V03_TOPOLOGY_SCRIPT.read_text()
        self.assertIn("diagnose_sparse_inliers.py", text)
        self.assertIn("--label_state_output", text)
        self.assertIn("--label_state_reset", text)
        self.assertIn("--localization_state_path", text)
        self.assertIn("TRAIN_PHASE=${TRAIN_PHASE:-feature}", text)
        self.assertIn("LOC_TEACHER=${LOC_TEACHER:-direct}", text)
        self.assertIn('TOPOLOGY_MUTATION_MODE=${TOPOLOGY_MUTATION_MODE:-split_only}', text)
        self.assertIn("TOPOLOGY_USE_LOC_OPACITY=${TOPOLOGY_USE_LOC_OPACITY:-0}", text)
        self.assertIn("TOPOLOGY_PROTECT_LANDMARKS=${TOPOLOGY_PROTECT_LANDMARKS:-0}", text)
        self.assertIn("TOPOLOGY_REMAP_MODE=${TOPOLOGY_REMAP_MODE:-source_distance}", text)
        self.assertIn("FORCE_TOPOLOGY_TRAIN=${FORCE_TOPOLOGY_TRAIN:-0}", text)
        self.assertIn("strip_future_point_clouds()", text)
        self.assertIn("if (( iteration > V03_ITERATION )); then", text)
        self.assertIn("TOPOLOGY_DENSE_DESC_WEIGHT=${TOPOLOGY_DENSE_DESC_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_DENSE_REPROJ_WEIGHT=${TOPOLOGY_DENSE_REPROJ_WEIGHT:-0.0}", text)
        self.assertIn('--train_phase "$TRAIN_PHASE"', text)
        self.assertIn('--loc_teacher "$LOC_TEACHER"', text)
        self.assertIn("--loc_direct_weight", text)
        self.assertIn("--loc_multiview_weight", text)
        self.assertIn("--loc_full_bank_weight", text)
        self.assertIn("--loc_full_bank_ignore_3d_radius", text)
        self.assertIn("--loc_full_bank_ignore_uv_radius", text)
        self.assertIn("TOPOLOGY_FULL_BANK_SOURCE_MODE=${TOPOLOGY_FULL_BANK_SOURCE_MODE:-ignore}", text)
        self.assertIn('--loc_full_bank_source_mode "$TOPOLOGY_FULL_BANK_SOURCE_MODE"', text)
        self.assertIn("TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE:-0}", text)
        self.assertIn("--loc_full_bank_nearby_as_positive", text)
        self.assertIn("TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL:-0}", text)
        self.assertIn("--loc_full_bank_nearby_as_positive_until", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_MODE=${TOPOLOGY_LOC_OVERLAY_MODE:-none}", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_LR=${TOPOLOGY_LOC_OVERLAY_LR:-0.0}", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT=${TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT:-0.0}", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM=${TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM:-0.0}", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_NORMALIZE=${TOPOLOGY_LOC_OVERLAY_NORMALIZE:-0}", text)
        self.assertIn("TOPOLOGY_LOC_OVERLAY_REG_WEIGHT=${TOPOLOGY_LOC_OVERLAY_REG_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_CHILD_RESPONSIBILITY_MODE=${TOPOLOGY_CHILD_RESPONSIBILITY_MODE:-none}", text)
        self.assertIn("TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=${TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER:-0}", text)
        self.assertIn('--loc_overlay_mode "$TOPOLOGY_LOC_OVERLAY_MODE"', text)
        self.assertIn('--loc_overlay_lr "$TOPOLOGY_LOC_OVERLAY_LR"', text)
        self.assertIn('--loc_overlay_active_logit "$TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT"', text)
        self.assertIn('--loc_overlay_max_residual_norm "$TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM"', text)
        self.assertIn('"${LOC_OVERLAY_ARGS[@]}"', text)
        self.assertIn('--loc_overlay_reg_weight "$TOPOLOGY_LOC_OVERLAY_REG_WEIGHT"', text)
        self.assertIn('--loc_child_responsibility_mode "$TOPOLOGY_CHILD_RESPONSIBILITY_MODE"', text)
        self.assertIn('--loc_child_responsibility_start_iter "$TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER"', text)
        self.assertIn("TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS=${TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS:-0}", text)
        self.assertIn("--loc_child_feature_freeze_steps", text)
        self.assertIn("TOPOLOGY_MAX_MUTATION_EVENTS=${TOPOLOGY_MAX_MUTATION_EVENTS:-0}", text)
        self.assertIn("--topology_max_mutation_events", text)
        self.assertIn("TOPOLOGY_RISK_COMMIT_POLICY=${TOPOLOGY_RISK_COMMIT_POLICY:-off}", text)
        self.assertIn("TOPOLOGY_RISK_HOLDOUT_SIZE=${TOPOLOGY_RISK_HOLDOUT_SIZE:-4}", text)
        self.assertIn("TOPOLOGY_RISK_HOLDOUT_SELECTION=${TOPOLOGY_RISK_HOLDOUT_SELECTION:-prefix}", text)
        self.assertIn("TOPOLOGY_RISK_EPSILON=${TOPOLOGY_RISK_EPSILON:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_DESC_WEIGHT=${TOPOLOGY_RISK_DESC_WEIGHT:-1.0}", text)
        self.assertIn("TOPOLOGY_RISK_FULL_BANK_WEIGHT=${TOPOLOGY_RISK_FULL_BANK_WEIGHT:-1.0}", text)
        self.assertIn("TOPOLOGY_RISK_REPROJ_WEIGHT=${TOPOLOGY_RISK_REPROJ_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_ANCHORS=${TOPOLOGY_RISK_ANCHORS:-256}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_AE_WEIGHT=${TOPOLOGY_RISK_POSE_AE_WEIGHT:-1.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_TE_WEIGHT=${TOPOLOGY_RISK_POSE_TE_WEIGHT:-1.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_INLIER_WEIGHT=${TOPOLOGY_RISK_POSE_INLIER_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_AE_SCALE=${TOPOLOGY_RISK_POSE_AE_SCALE:-5.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_TE_SCALE=${TOPOLOGY_RISK_POSE_TE_SCALE:-200.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_INLIER_SCALE=${TOPOLOGY_RISK_POSE_INLIER_SCALE:-100.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT=${TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT=${TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT=${TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_CVAR_WEIGHT=${TOPOLOGY_RISK_POSE_CVAR_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_CVAR_FRACTION=${TOPOLOGY_RISK_POSE_CVAR_FRACTION:-0.25}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_VETO_MODE=${TOPOLOGY_RISK_POSE_VETO_MODE:-off}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_R2_TOLERANCE=${TOPOLOGY_RISK_POSE_R2_TOLERANCE:-0.0}", text)
        self.assertIn("TOPOLOGY_RISK_POSE_TAIL_TOLERANCE=${TOPOLOGY_RISK_POSE_TAIL_TOLERANCE:-0.0}", text)
        self.assertIn("QUERY_ARTIFACT_FILTER_PATH=${QUERY_ARTIFACT_FILTER_PATH:-}", text)
        self.assertIn("QUERY_ARTIFACT_FILTER_ARGS=()", text)
        self.assertIn('--query_artifact_filter_path "$QUERY_ARTIFACT_FILTER_PATH"', text)
        self.assertIn('--query_artifact_filter_severities "$QUERY_ARTIFACT_FILTER_SEVERITIES"', text)
        self.assertIn('--query_artifact_filter_splits "$QUERY_ARTIFACT_FILTER_SPLITS"', text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_PATH=${RENDER_ARTIFACT_WEIGHT_PATH:-}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_MODE=${RENDER_ARTIFACT_WEIGHT_MODE:-severity}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_SEVERITIES=${RENDER_ARTIFACT_WEIGHT_SEVERITIES:-severe}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_MILD=${RENDER_ARTIFACT_WEIGHT_MILD:-1.0}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_SEVERE=${RENDER_ARTIFACT_WEIGHT_SEVERE:-0.70}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN=${RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN:-0.70}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER=${RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER:-1.0}", text)
        self.assertIn(
            "RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE=${RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE:-product}",
            text,
        )
        self.assertIn(
            "RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE=${RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE:-none}",
            text,
        )
        self.assertIn("RENDER_ARTIFACT_REGION_WEIGHT_PATH=${RENDER_ARTIFACT_REGION_WEIGHT_PATH:-}", text)
        self.assertIn("RENDER_ARTIFACT_REGION_WEIGHT_ROOT=${RENDER_ARTIFACT_REGION_WEIGHT_ROOT:-}", text)
        self.assertIn("RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES=${RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES:-severe}", text)
        self.assertIn("RENDER_ARTIFACT_REGION_WEIGHT_TARGETS=${RENDER_ARTIFACT_REGION_WEIGHT_TARGETS:-direct}", text)
        self.assertIn("RENDER_ARTIFACT_WEIGHT_ARGS=()", text)
        self.assertIn("RENDER_ARTIFACT_REGION_WEIGHT_ARGS=()", text)
        self.assertIn('--render_artifact_weight_path "$RENDER_ARTIFACT_WEIGHT_PATH"', text)
        self.assertIn('--render_artifact_weight_mode "$RENDER_ARTIFACT_WEIGHT_MODE"', text)
        self.assertIn('--render_artifact_weight_targets "$RENDER_ARTIFACT_WEIGHT_TARGETS"', text)
        self.assertIn('--render_artifact_weight_mild "$RENDER_ARTIFACT_WEIGHT_MILD"', text)
        self.assertIn('--render_artifact_weight_severe "$RENDER_ARTIFACT_WEIGHT_SEVERE"', text)
        self.assertIn('--render_artifact_weight_continuous_min "$RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN"', text)
        self.assertIn('--render_artifact_weight_continuous_power "$RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER"', text)
        self.assertIn(
            '--render_artifact_direct_weight_combine_mode "$RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE"',
            text,
        )
        self.assertIn(
            '--render_artifact_direct_loss_scale_mode "$RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE"',
            text,
        )
        self.assertIn('--render_artifact_region_weight_path "$RENDER_ARTIFACT_REGION_WEIGHT_PATH"', text)
        self.assertIn('--render_artifact_region_weight_root "$RENDER_ARTIFACT_REGION_WEIGHT_ROOT"', text)
        self.assertIn('--render_artifact_region_weight_targets "$RENDER_ARTIFACT_REGION_WEIGHT_TARGETS"', text)
        self.assertIn('"${RENDER_ARTIFACT_REGION_WEIGHT_ARGS[@]}"', text)
        self.assertIn("TOPOLOGY_SUPPORT_QUERY_SPLIT=${TOPOLOGY_SUPPORT_QUERY_SPLIT:-1}", text)
        self.assertIn("TOPOLOGY_QUERY_HOLDOUT_RATIO=${TOPOLOGY_QUERY_HOLDOUT_RATIO:-0.2}", text)
        self.assertIn("TOPOLOGY_QUERY_SPLIT_MODE=${TOPOLOGY_QUERY_SPLIT_MODE:-sequence_block}", text)
        self.assertIn("TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME=${TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME:-0}", text)
        self.assertIn("SUPPORT_QUERY_ARGS=()", text)
        self.assertIn("--support_query_split", text)
        self.assertIn('--query_holdout_ratio "$TOPOLOGY_QUERY_HOLDOUT_RATIO"', text)
        self.assertIn('--query_split_seed "$QUERY_SPLIT_SEED"', text)
        self.assertIn('--query_split_mode "$TOPOLOGY_QUERY_SPLIT_MODE"', text)
        self.assertIn("--support_query_sort_by_name", text)
        self.assertIn('"${SUPPORT_QUERY_ARGS[@]}"', text)
        self.assertIn('--topology_risk_commit_policy "$TOPOLOGY_RISK_COMMIT_POLICY"', text)
        self.assertIn('--topology_risk_holdout_size "$TOPOLOGY_RISK_HOLDOUT_SIZE"', text)
        self.assertIn('--topology_risk_holdout_selection "$TOPOLOGY_RISK_HOLDOUT_SELECTION"', text)
        self.assertIn('--topology_risk_epsilon "$TOPOLOGY_RISK_EPSILON"', text)
        self.assertIn('--topology_risk_desc_weight "$TOPOLOGY_RISK_DESC_WEIGHT"', text)
        self.assertIn('--topology_risk_full_bank_weight "$TOPOLOGY_RISK_FULL_BANK_WEIGHT"', text)
        self.assertIn('--topology_risk_reproj_weight "$TOPOLOGY_RISK_REPROJ_WEIGHT"', text)
        self.assertIn('--topology_risk_anchors "$TOPOLOGY_RISK_ANCHORS"', text)
        self.assertIn('--topology_risk_pose_cfg "$BASELINE_CFG"', text)
        self.assertIn('--topology_risk_pose_ae_weight "$TOPOLOGY_RISK_POSE_AE_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_te_weight "$TOPOLOGY_RISK_POSE_TE_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_inlier_weight "$TOPOLOGY_RISK_POSE_INLIER_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_ae_scale "$TOPOLOGY_RISK_POSE_AE_SCALE"', text)
        self.assertIn('--topology_risk_pose_te_scale "$TOPOLOGY_RISK_POSE_TE_SCALE"', text)
        self.assertIn('--topology_risk_pose_inlier_scale "$TOPOLOGY_RISK_POSE_INLIER_SCALE"', text)
        self.assertIn('--topology_risk_pose_r5_miss_weight "$TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_r2_miss_weight "$TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_tail_fail_weight "$TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_cvar_weight "$TOPOLOGY_RISK_POSE_CVAR_WEIGHT"', text)
        self.assertIn('--topology_risk_pose_cvar_fraction "$TOPOLOGY_RISK_POSE_CVAR_FRACTION"', text)
        self.assertIn('--topology_risk_pose_veto_mode "$TOPOLOGY_RISK_POSE_VETO_MODE"', text)
        self.assertIn('--topology_risk_pose_r2_tolerance "$TOPOLOGY_RISK_POSE_R2_TOLERANCE"', text)
        self.assertIn('--topology_risk_pose_tail_tolerance "$TOPOLOGY_RISK_POSE_TAIL_TOLERANCE"', text)
        self.assertIn('"${QUERY_ARTIFACT_FILTER_ARGS[@]}"', text)
        self.assertIn('"${RENDER_ARTIFACT_WEIGHT_ARGS[@]}"', text)
        self.assertIn("--loc_anchor_weight", text)
        self.assertIn("--direct_depth_check", text)
        self.assertIn("LOC_OPACITY_ARGS=(--no-use_loc_opacity --loc_opacity_weight 0.0)", text)
        self.assertIn('"${LOC_OPACITY_ARGS[@]}"', text)
        self.assertIn('--loc_desc_weight "$TOPOLOGY_DENSE_DESC_WEIGHT"', text)
        self.assertIn('--loc_reproj_weight "$TOPOLOGY_DENSE_REPROJ_WEIGHT"', text)
        self.assertIn("--enable_topology", text)
        self.assertIn('case "$TOPOLOGY_MUTATION_MODE" in', text)
        self.assertIn("no_mutation)", text)
        self.assertIn("split_only)", text)
        self.assertIn("one_shot_split)", text)
        self.assertIn("one_shot_split_freeze)", text)
        self.assertIn("soft_prune)", text)
        self.assertIn("soft_prune_only)", text)
        self.assertIn("physical_prune)", text)
        self.assertIn("physical_prune_only)", text)
        self.assertIn("TOPOLOGY_DISABLE_SPLIT=${TOPOLOGY_DISABLE_SPLIT:-0}", text)
        self.assertIn("--topology_disable_split", text)
        self.assertIn("TOPOLOGY_DENSE_DESC_WEIGHT=1.0", text)
        self.assertIn("TOPOLOGY_DENSE_REPROJ_WEIGHT=0.1", text)
        self.assertIn('TOPOLOGY_ARGS+=(--topology_enable_physical_prune)', text)
        self.assertIn("--topology_allow_untrained_loc_opacity_prune", text)
        self.assertIn("remap_topology_landmarks.py", text)
        self.assertIn("--topology_loc_state", text)
        self.assertIn("--output_sampled_idx", text)
        self.assertIn("--remap_mode", text)
        self.assertIn("stdloc.py", text)
        self.assertIn('sparse["landmark_path"] = topology_landmark_path', text)
        self.assertIn('sparse["landmark_model_path"] = topology_model', text)
        self.assertIn("--prefix \"phase-v03-topology-${TOPOLOGY_END}\"", text)
        self.assertIn('if [[ "$FORCE_TOPOLOGY_TRAIN" == "1" ]]; then', text)
        self.assertIn('rm -rf "$TOPOLOGY_MODEL/point_cloud/iteration_${TOPOLOGY_END}"', text)

    def test_la_update3_p0_worker_runs_s0_s3_modes(self):
        self.assertTrue(UPDATE3_P0_WORKER_SCRIPT.exists(), "LA_update3 P0 worker is missing")
        text = UPDATE3_P0_WORKER_SCRIPT.read_text()

        self.assertIn("SPECS=${SPECS:-S0:no_mutation S1:one_shot_split S2:split_only S3:one_shot_split_freeze}", text)
        self.assertIn("run_locaware_v03_topology_full.sh", text)
        self.assertIn('TOPOLOGY_MUTATION_MODE="$mode"', text)
        self.assertIn("local freeze_steps=0", text)
        self.assertIn('if [[ "$mode" == "one_shot_split_freeze" ]]; then', text)
        self.assertIn('freeze_steps="$S3_CHILD_FEATURE_FREEZE_STEPS"', text)
        self.assertIn('TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS="$freeze_steps"', text)
        self.assertIn("TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL", text)
        self.assertIn("tag=\"p0_${label}_${steps}\"", text)

    def test_dense_kl_script_runs_dense_teacher_without_topology(self):
        text = DENSE_KL_SCRIPT.read_text()
        self.assertIn("--loc_teacher dense", text)
        self.assertIn("--loc_dense_kl_weight", text)
        self.assertIn("--loc_dense_rank_weight", text)
        self.assertIn("--loc_dense_rank_margin", text)
        self.assertIn("--loc_dense_rank_teacher_confidence", text)
        self.assertIn("--loc_dense_rank_miss_topk", text)
        self.assertIn("--loc_responsibility_topk", text)
        self.assertIn("--loc_responsibility_opacity_weight", text)
        self.assertIn("--loc_responsibility_depth_weight", text)
        self.assertIn("RUN_DENSE_POSE_CACHE=${RUN_DENSE_POSE_CACHE:-0}", text)
        self.assertIn("DENSEKL_QUERY_MODE=${DENSEKL_QUERY_MODE:-noise}", text)
        self.assertIn("DENSEKL_QUERY_MODE=sparse", text)
        self.assertIn("--include_dense", text)
        self.assertIn('--query_mode "$DENSEKL_QUERY_MODE"', text)
        self.assertIn("--loc_dense_pose_gate", text)
        self.assertIn("--loc_dense_advantage_gate", text)
        self.assertIn("--loc_dense_attr_cosine_threshold", text)
        self.assertIn("--loc_dense_attr_entropy_threshold", text)
        self.assertIn("--loc_dense_min_positive_prob", text)
        self.assertIn("--loc_dense_max_reproj_error", text)
        self.assertIn("--loc_dense_min_eligible_anchors", text)
        self.assertIn("DENSEKL_ADVANTAGE_GATE=${DENSEKL_ADVANTAGE_GATE:-0}", text)
        self.assertIn("DENSEKL_ADVANTAGE_TE_SCALE=${DENSEKL_ADVANTAGE_TE_SCALE:-10.0}", text)
        self.assertIn("DENSEKL_RANK_WEIGHT=${DENSEKL_RANK_WEIGHT:-0.0}", text)
        self.assertIn("DENSEKL_RANK_TEACHER_CONFIDENCE=${DENSEKL_RANK_TEACHER_CONFIDENCE:-0.0}", text)
        self.assertIn("DENSEKL_SYNTHETIC_VIEW_RATIO=${DENSEKL_SYNTHETIC_VIEW_RATIO:-0.0}", text)
        self.assertIn("DENSEKL_SYNTHETIC_VIEW_DESC_WEIGHT=${DENSEKL_SYNTHETIC_VIEW_DESC_WEIGHT:-0.0}", text)
        self.assertIn("DENSEKL_SYNTHETIC_VIEW_REPROJ_WEIGHT=${DENSEKL_SYNTHETIC_VIEW_REPROJ_WEIGHT:-0.0}", text)
        self.assertIn("--synthetic_view_ratio", text)
        self.assertIn("--synthetic_view_candidates", text)
        self.assertIn("--synthetic_view_min_observability", text)
        self.assertIn("--synthetic_view_desc_weight", text)
        self.assertIn("--synthetic_view_reproj_weight", text)
        self.assertIn("ADVANTAGE_GATE_ARGS=()", text)
        self.assertIn("DENSEKL_SAVE_STEPS=${DENSEKL_SAVE_STEPS:-$DENSEKL_STEPS}", text)
        self.assertIn("DENSEKL_EVAL_STEPS=${DENSEKL_EVAL_STEPS:-$DENSEKL_SAVE_STEPS}", text)
        self.assertIn("FORCE_DENSEKL_TRAIN=${FORCE_DENSEKL_TRAIN:-0}", text)
        self.assertIn("steps_to_iterations()", text)
        self.assertIn("strip_future_point_clouds()", text)
        self.assertIn("if (( iteration > LOAD_ITERATION )); then", text)
        self.assertIn("--save_iterations \"${DENSEKL_SAVE_ITERATIONS[@]}\"", text)
        self.assertIn("for eval_iteration in \"${DENSEKL_EVAL_ITERATIONS[@]}\"; do", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertNotIn("--enable_topology", text)
        self.assertIn("diagnose_dense_responsibility.py", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)
        self.assertIn("--sparse_only", text)

    def test_dense_kl_script_accepts_explicit_training_seed(self):
        text = DENSE_KL_SCRIPT.read_text()

        self.assertIn("DENSEKL_TRAIN_SEED=${DENSEKL_TRAIN_SEED:-0}", text)
        self.assertIn('--train_seed "$DENSEKL_TRAIN_SEED"', text)

    def test_dense_kl_script_accepts_support_query_split_controls(self):
        text = DENSE_KL_SCRIPT.read_text()

        self.assertIn("DENSEKL_SUPPORT_QUERY_SPLIT=${DENSEKL_SUPPORT_QUERY_SPLIT:-0}", text)
        self.assertIn("DENSEKL_QUERY_HOLDOUT_RATIO=${DENSEKL_QUERY_HOLDOUT_RATIO:-0.2}", text)
        self.assertIn("DENSEKL_QUERY_SPLIT_SEED=${DENSEKL_QUERY_SPLIT_SEED:-2025}", text)
        self.assertIn("DENSEKL_QUERY_SPLIT_MODE=${DENSEKL_QUERY_SPLIT_MODE:-sequence_block}", text)
        self.assertIn("SUPPORT_QUERY_ARGS=()", text)
        self.assertIn("--support_query_split", text)
        self.assertIn('--query_holdout_ratio "$DENSEKL_QUERY_HOLDOUT_RATIO"', text)
        self.assertIn('--query_split_seed "$DENSEKL_QUERY_SPLIT_SEED"', text)
        self.assertIn('--query_split_mode "$DENSEKL_QUERY_SPLIT_MODE"', text)
        self.assertIn('"${SUPPORT_QUERY_ARGS[@]}"', text)

    def test_la_update2_workers_separate_train_seed_from_query_split_seed(self):
        dense_text = DENSE_LONG_WORKER_SCRIPT.read_text()
        topology_text = TOPOLOGY_LONG_WORKER_SCRIPT.read_text()

        for text in (dense_text, topology_text):
            with self.subTest(script=text[:80]):
                self.assertIn("TRAIN_SEEDS=${TRAIN_SEEDS:-${SEEDS:-0 1 2}}", text)
                self.assertIn("QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-${SEEDS:-2025 2026 2027}}", text)
                self.assertIn("for train_seed in $TRAIN_SEEDS; do", text)
                self.assertIn("for query_split_seed in $QUERY_SPLIT_SEEDS; do", text)
                self.assertIn("train_seed_${train_seed}/query_split_${query_split_seed}", text)
                self.assertIn("seed_${query_split_seed}", text)

        self.assertIn('DENSEKL_TRAIN_SEED="$train_seed"', dense_text)
        self.assertIn('DENSEKL_QUERY_SPLIT_SEED="$query_split_seed"', dense_text)
        self.assertIn("DENSEKL_ADVANTAGE_GATE=${DENSEKL_ADVANTAGE_GATE:-0}", dense_text)
        self.assertIn('DENSEKL_ADVANTAGE_GATE="$DENSEKL_ADVANTAGE_GATE"', dense_text)
        self.assertIn("DENSEKL_RANK_WEIGHT=${DENSEKL_RANK_WEIGHT:-0.0}", dense_text)
        self.assertIn('DENSEKL_RANK_WEIGHT="$DENSEKL_RANK_WEIGHT"', dense_text)
        self.assertIn('TRAIN_SEED="$train_seed"', topology_text)
        self.assertIn('QUERY_SPLIT_SEED="$query_split_seed"', topology_text)

    def test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector(self):
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()

        self.assertIn("SYNTHETIC_CANDIDATE_MULTIPLIER=${SYNTHETIC_CANDIDATE_MULTIPLIER:-1}", text)
        self.assertIn("SYNTHETIC_RENDER_COUNT=$((SYNTHETIC_COUNT * SYNTHETIC_CANDIDATE_MULTIPLIER))", text)
        self.assertIn("RUN_PSEUDO_QUERY_MANIFEST=${RUN_PSEUDO_QUERY_MANIFEST:-1}", text)
        self.assertIn('if [[ "$RUN_PSEUDO_QUERY_MANIFEST" == "1" ]]; then', text)
        self.assertIn("Missing pseudo-query manifest for $scene", text)
        self.assertIn("RUN_TEACHER_CACHE_AUDIT=${RUN_TEACHER_CACHE_AUDIT:-1}", text)
        self.assertIn('if [[ "$RUN_TEACHER_CACHE_AUDIT" == "1" && -f "$manifest" && -f "$teacher_cache" ]]; then', text)
        self.assertIn("scripts/audit_pseudo_teacher_cache.py", text)
        self.assertIn('--synthetic_count "$SYNTHETIC_RENDER_COUNT"', text)
        self.assertIn("PSEUDO_QUERY_POSE_SAMPLER=${PSEUDO_QUERY_POSE_SAMPLER:-spatial_offset}", text)
        self.assertIn('--synthetic_pose_sampler "$PSEUDO_QUERY_POSE_SAMPLER"', text)
        self.assertIn("RUN_PSEUDO_QUERY_SELECT=${RUN_PSEUDO_QUERY_SELECT:-0}", text)
        self.assertIn("scripts/select_pseudo_query_pool.py", text)
        self.assertIn('--max_synthetic "$PSEUDO_QUERY_SELECT_MAX_SYNTHETIC"', text)
        self.assertIn("PSEUDO_QUERY_SELECT_SORT_BY=${PSEUDO_QUERY_SELECT_SORT_BY:-artifact}", text)
        self.assertIn('choices=["artifact", "support", "dense_te", "sparse_te"]', SELECT_PSEUDO_QUERY_POOL_SCRIPT.read_text())
        self.assertIn("LA_TRAIN_MODE=${LA_TRAIN_MODE:-adapt}", text)
        self.assertIn('case "$LA_TRAIN_MODE" in', text)
        self.assertIn("adapt)", text)
        self.assertIn("scratch)", text)
        self.assertIn("LA_BOOTSTRAP_DETECTOR_FOLDER=${LA_BOOTSTRAP_DETECTOR_FOLDER:-detector_bootstrap}", text)
        self.assertIn("LA_BOOTSTRAP_SAMPLING_MODE=${LA_BOOTSTRAP_SAMPLING_MODE:-baseline}", text)
        self.assertIn("FORCE_LA_BOOTSTRAP_LANDMARKS=${FORCE_LA_BOOTSTRAP_LANDMARKS:-0}", text)
        self.assertIn("train_needs_bootstrap_landmarks=1", text)
        self.assertIn("--landmark_only", text)
        self.assertIn('--iteration 0', text)
        self.assertIn('--detector_folder "$LA_BOOTSTRAP_DETECTOR_FOLDER"', text)
        self.assertIn('Scratch LA eval requires RUN_LA_FRONTEND_REFRESH=1', text)
        self.assertIn("LA_LOC_START_ITER=${LA_LOC_START_ITER:-1}", text)
        self.assertIn("LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-${TRAIN_STEPS:-100}}", text)
        self.assertIn("TRAIN_STEPS=\"$LA_ADAPT_STEPS\"", text)
        self.assertIn("end_iter=$((train_start_iter + LA_ADAPT_STEPS))", text)
        self.assertIn("PSEUDO_QUERY_MAX_SYNTHETIC=${PSEUDO_QUERY_MAX_SYNTHETIC:-0}", text)
        self.assertIn("PSEUDO_QUERY_SELECT_MAX_SYNTHETIC=${PSEUDO_QUERY_SELECT_MAX_SYNTHETIC:-0}", text)
        self.assertIn("PSEUDO_QUERY_MIN_SUPPORT_FRAC=${PSEUDO_QUERY_MIN_SUPPORT_FRAC:-0.0}", text)
        self.assertIn("PSEUDO_QUERY_MIN_SUPPORT_SCORE=${PSEUDO_QUERY_MIN_SUPPORT_SCORE:--1.0}", text)
        self.assertIn('--min_support_frac "$PSEUDO_QUERY_MIN_SUPPORT_FRAC"', text)
        self.assertIn('--min_support_score "$PSEUDO_QUERY_MIN_SUPPORT_SCORE"', text)
        self.assertIn("PSEUDO_QUERY_ENABLE_TEACHER_GATE=${PSEUDO_QUERY_ENABLE_TEACHER_GATE:-0}", text)
        self.assertIn("--teacher_gate", text)
        self.assertIn("PSEUDO_QUERY_FILTER_TEACHER_CACHE=${PSEUDO_QUERY_FILTER_TEACHER_CACHE:-0}", text)
        self.assertIn("PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=${PSEUDO_QUERY_REQUIRE_TEACHER_CACHE:-1}", text)
        self.assertIn("PSEUDO_QUERY_TEACHER_ALLOWED_STAGES=${PSEUDO_QUERY_TEACHER_ALLOWED_STAGES:-}", text)
        self.assertIn("PSEUDO_QUERY_REAL_WEIGHT=${PSEUDO_QUERY_REAL_WEIGHT:-2.0}", text)
        self.assertIn("PSEUDO_QUERY_SYNTHETIC_WEIGHT=${PSEUDO_QUERY_SYNTHETIC_WEIGHT:-1.0}", text)
        self.assertIn("PSEUDO_QUERY_SAMPLING_MODE=${PSEUDO_QUERY_SAMPLING_MODE:-record_proportional}", text)
        self.assertIn('--pseudo_query_sampling_mode "$PSEUDO_QUERY_SAMPLING_MODE"', text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_MODE=${PSEUDO_QUERY_RELIABILITY_MODE:-none}", text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_LOSS_MODE=${PSEUDO_QUERY_RELIABILITY_LOSS_MODE:-none}", text)
        self.assertIn("PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=${PSEUDO_QUERY_STAGE_OBJECTIVE_MODE:-none}", text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT:-0.50}", text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_SYNTHETIC_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_SYNTHETIC_MIN_WEIGHT:-0.25}", text)
        self.assertIn("PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT:-0.75}", text)
        self.assertIn("PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=${PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES:-0}", text)
        self.assertIn('if [[ "$PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES" == "1" ]]; then', text)
        self.assertIn("pseudo_stage_gate_args+=(--pseudo_query_exclude_sparse_failure_stages)", text)
        self.assertIn('if [[ "$PSEUDO_QUERY_REQUIRE_TEACHER_CACHE" == "1" ]]; then', text)
        self.assertIn("--pseudo_query_require_teacher_cache", text)
        self.assertIn("--no-pseudo_query_require_teacher_cache", text)
        self.assertIn('--pseudo_query_reliability_mode "$PSEUDO_QUERY_RELIABILITY_MODE"', text)
        self.assertIn('--pseudo_query_reliability_loss_mode "$PSEUDO_QUERY_RELIABILITY_LOSS_MODE"', text)
        self.assertIn('--pseudo_query_stage_objective_mode "$PSEUDO_QUERY_STAGE_OBJECTIVE_MODE"', text)
        self.assertIn('--pseudo_query_reliability_memory_min_weight "$PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT"', text)
        self.assertIn('--loc_start_iter "$LA_LOC_START_ITER"', text)
        self.assertIn('"${pseudo_stage_gate_args[@]}"', text)
        self.assertIn("LA_DIRECT_DEPTH_CHECK=${LA_DIRECT_DEPTH_CHECK:-0}", text)
        self.assertIn('if [[ "$LA_DIRECT_DEPTH_CHECK" == "1" ]]; then', text)
        self.assertIn('direct_depth_check_args+=(--direct_depth_check)', text)
        self.assertIn('"${direct_depth_check_args[@]}"', text)
        self.assertIn("PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT:-$LA_ENABLE_SYNTHETIC}", text)
        self.assertIn("--pseudo_query_no_reference_region_weight", text)
        self.assertIn("RUN_LA_FRONTEND_REFRESH=${RUN_LA_FRONTEND_REFRESH:-0}", text)
        self.assertIn("LA_DETECTOR_FOLDER=${LA_DETECTOR_FOLDER:-detector_la}", text)
        self.assertIn("LA_DETECTOR_SAMPLING_MODE=${LA_DETECTOR_SAMPLING_MODE:-localization_aware}", text)
        self.assertIn("LA_DETECTOR_TARGET_MODE=${LA_DETECTOR_TARGET_MODE:-soft}", text)
        self.assertIn("LA_DETECTOR_MIN_LOC_OBSERVATIONS=${LA_DETECTOR_MIN_LOC_OBSERVATIONS:-4}", text)
        self.assertIn('"$PYTHON" train_detector.py', text)
        self.assertIn('--detector_folder "$LA_DETECTOR_FOLDER"', text)
        self.assertIn('--sampling_mode "$LA_DETECTOR_SAMPLING_MODE"', text)
        self.assertIn('--detector_target_mode "$LA_DETECTOR_TARGET_MODE"', text)
        self.assertIn('--min_loc_observations "$LA_DETECTOR_MIN_LOC_OBSERVATIONS"', text)
        self.assertIn('make_eval_cfg "$scene" "$train_model" "$student_eval_cfg" "$LA_DETECTOR_FOLDER"', text)
        self.assertIn('final_eval_cfg="$student_eval_cfg"', text)
        self.assertIn('--cfg "$final_eval_cfg"', text)

    def test_pseudo_query_pipeline_defaults_to_train_rgb_mainline(self):
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()

        self.assertIn("LA_ENABLE_SYNTHETIC=${LA_ENABLE_SYNTHETIC:-0}", text)
        self.assertIn('if [[ -z "${SYNTHETIC_COUNT+x}" ]]; then', text)
        self.assertIn('if [[ "$LA_ENABLE_SYNTHETIC" == "1" ]]; then', text)
        self.assertIn("SYNTHETIC_COUNT=16", text)
        self.assertIn("SYNTHETIC_COUNT=0", text)
        self.assertIn('if [[ -z "${PSEUDO_QUERY_SOURCES+x}" ]]; then', text)
        self.assertIn("PSEUDO_QUERY_SOURCES=train_rgb,synthetic_rgb", text)
        self.assertIn("PSEUDO_QUERY_SOURCES=train_rgb", text)
        self.assertIn('if [[ -z "${TEACHER_CACHE_SOURCES+x}" ]]; then', text)
        self.assertIn("TEACHER_CACHE_SOURCES=train_rgb,synthetic_rgb", text)
        self.assertIn("TEACHER_CACHE_SOURCES=train_rgb", text)
        self.assertIn("TEACHER_CACHE_SPARSE_VALID_MASK=${TEACHER_CACHE_SPARSE_VALID_MASK:-$LA_ENABLE_SYNTHETIC}", text)
        self.assertIn("PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT:-$LA_ENABLE_SYNTHETIC}", text)
        self.assertIn('elif (( SYNTHETIC_RENDER_COUNT > 0 )) && [[ "$RENDER_SYNTHETIC_BACKEND" == "wildgaussians" ]]; then', text)
        self.assertIn('if (( SYNTHETIC_RENDER_COUNT > 0 )) && [[ "$RENDER_SYNTHETIC_BACKEND" == "matcha" && ! -d "$matcha_model_path" ]]; then', text)

    def test_pseudo_query_pipeline_defaults_to_matcha_synthetic_backend(self):
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()

        self.assertIn("RENDER_SYNTHETIC_BACKEND=${RENDER_SYNTHETIC_BACKEND:-matcha}", text)
        self.assertIn("MATCHA_PYTHON_DEFAULT=${MATCHA_PYTHON_DEFAULT:-/root/miniconda3/envs/cybersim_agent/bin/python}", text)
        self.assertIn('scene_matcha_var="MATCHA_MODEL_PATH_${scene}"', text)
        self.assertIn(
            'matcha_model_path="${!scene_matcha_var:-${MATCHA_MODEL_PATH:-$MATCHA_RUNS_ROOT/${scene}_n20_long_masked/free_gaussians}}"',
            text,
        )
        self.assertIn('if (( SYNTHETIC_RENDER_COUNT > 0 )) && [[ "$RENDER_SYNTHETIC_BACKEND" == "matcha" && ! -d "$matcha_model_path" ]]; then', text)
        self.assertIn('needs_nerfbaselines=0', text)
        self.assertIn('if [[ "$needs_nerfbaselines" == "1" ]]; then', text)
        self.assertIn('MATCHA_ENV_LIB="$(dirname "$(dirname "$MATCHA_PYTHON")")/lib"', text)

    def test_prepare_baseline_artifacts_script_builds_missing_scene_baselines(self):
        text = PREPARE_BASELINES_SCRIPT.read_text()
        self.assertIn("SOURCE_ROOT=${SOURCE_ROOT:-/mnt/pool/sqy/ulfloc_repro_20260607}", text)
        self.assertIn("TARGET_ROOT=${TARGET_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}", text)
        self.assertIn("SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}", text)
        self.assertIn("SKIP_DETECTOR=${SKIP_DETECTOR:-0}", text)
        self.assertIn("REQUIRE_LOC_FEATURE=${REQUIRE_LOC_FEATURE:-1}", text)
        self.assertIn("TRAIN_MISSING_BASELINE=${TRAIN_MISSING_BASELINE:-0}", text)
        self.assertIn("FORCE_BASELINE_TRAIN=${FORCE_BASELINE_TRAIN:-0}", text)
        self.assertIn("ply_has_loc_feature()", text)
        self.assertIn("train_full_baseline()", text)
        self.assertIn('SOURCE_MODEL="$SOURCE_ROOT/$scene"', text)
        self.assertIn('TARGET_MODEL="$TARGET_ROOT/${scene}_baseline"', text)
        self.assertIn('SOURCE_PLY="$SOURCE_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"', text)
        self.assertIn('cp -a "$SOURCE_MODEL" "$TARGET_MODEL"', text)
        self.assertIn('"$PYTHON" train.py', text)
        self.assertIn("--train_detector", text)
        self.assertIn("--test_detector_iterations", text)
        self.assertIn("--save_detector_iterations", text)
        self.assertIn('if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$SOURCE_PLY"; then', text)
        self.assertIn('if [[ "$TRAIN_MISSING_BASELINE" == "1" ]]; then', text)
        self.assertIn("source point cloud lacks loc_* feature fields", text)
        self.assertIn('if [[ "$SKIP_DETECTOR" == "1" ]]; then', text)
        self.assertIn("train_detector.py", text)
        self.assertIn("--sampling_mode baseline", text)
        self.assertIn("--detector_target_mode hard", text)
        self.assertIn("--detector_folder detector", text)
        self.assertIn('TARGET_PLY="$TARGET_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"', text)
        detector_ready_pos = text.index('if [[ -f "$TARGET_MODEL/detector/${DETECTOR_ITERS}_detector.pth" && -f "$TARGET_MODEL/detector/sampled_idx.pkl" ]]; then')
        source_loc_check_pos = text.index('if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$SOURCE_PLY"; then')
        self.assertLess(detector_ready_pos, source_loc_check_pos)
        self.assertIn('if [[ ! -f "$TARGET_PLY" ]]; then', text)

    def test_geometry_balance_2x2_script_runs_original_and_balanced_selectors(self):
        text = GEOM_2X2_SCRIPT.read_text()
        self.assertIn('geometry_balance"', text)
        self.assertIn('"enabled": False', text)
        self.assertIn('"enabled": True', text)
        self.assertIn('"post": {', text)
        self.assertIn('"score_weight": float(post_score_weight)', text)
        self.assertIn("phase-2x2-baseline-original", text)
        self.assertIn("phase-2x2-baseline-balanced", text)
        self.assertIn("phase-2x2-la-original", text)
        self.assertIn("phase-2x2-la-balanced", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)

    def test_train_locaware_applies_topology_before_saving_same_iteration_checkpoint(self):
        text = (Path(__file__).resolve().parents[1] / "train_locaware.py").read_text()

        topology_pos = text.index("topology_controller.update(gaussians, scene.cameras_extent, iteration)")
        save_pos = text.index("scene.save(iteration)")

        self.assertLess(topology_pos, save_pos)

    def test_descriptor_diagnostics_script_reports_level1_metrics(self):
        text = DESCRIPTOR_DIAG_SCRIPT.read_text()
        self.assertIn("descriptor_alignment_metrics", text)
        self.assertIn("--baseline_model_path", text)
        self.assertIn("--max_images", text)
        self.assertIn("positive_cosine_mean", text)
        self.assertIn("mnn_precision", text)

    def test_descriptor_diagnostics_limits_iterable_cameras(self):
        module = self._load_descriptor_diag_module()
        cameras = (idx for idx in range(4))
        self.assertEqual(module._limit_cameras(cameras, 2), [0, 1])

    def test_descriptor_diagnostics_tolerates_missing_optional_overrides(self):
        module = self._load_descriptor_diag_module()

        updated = module._ensure_optional_args(Namespace())

        self.assertIsNone(updated.landmark_model_path)
        self.assertIsNone(updated.baseline_model_path)
        self.assertEqual(updated.baseline_iteration, 30000)

    def test_descriptor_diagnostics_loads_localization_state_when_present(self):
        module = self._load_descriptor_diag_module()

        class DummyGaussians:
            def __init__(self, sh_degree):
                self.sh_degree = sh_degree
                self.loaded_ply = None
                self.loaded_state = None

            def load_ply(self, path):
                self.loaded_ply = path

            def load_localization_state(self, path):
                self.loaded_state = path

        original_gaussian_model = module.GaussianModel
        module.GaussianModel = DummyGaussians
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                iteration_dir = Path(tmpdir) / "point_cloud" / "iteration_7"
                iteration_dir.mkdir(parents=True)
                loc_state_path = iteration_dir / "loc_state.pt"
                module.torch.save({"loc_overlay_feature": module.torch.zeros(1, 1, 2)}, loc_state_path)

                gaussians = module._load_gaussians_from_iteration(
                    Namespace(gaussian_type="3dgs", sh_degree=3),
                    tmpdir,
                    7,
                )
        finally:
            module.GaussianModel = original_gaussian_model

        self.assertEqual(gaussians.loaded_ply, str(Path(tmpdir) / "point_cloud" / "iteration_7" / "point_cloud.ply"))
        self.assertEqual(gaussians.loaded_state, str(loc_state_path))

    def test_sparse_inlier_diagnostics_script_reports_level3_metrics(self):
        text = INLIER_DIAG_SCRIPT.read_text()
        self.assertIn("summarize_landmark_value", text)
        self.assertIn("--reprojection_error", text)
        self.assertIn("spearman_utility_inlier_rate", text)
        self.assertIn("visible_count", text)
        self.assertIn("inlier_count", text)
        self.assertIn("--label_state_output", text)
        self.assertIn("--label_state_reset", text)
        self.assertIn("add_sparse_match_label_stats", text)
        self.assertIn("calibrate_landmark_quality", text)
        self.assertIn("--calibrated_utility_output", text)
        self.assertIn("--calibration_target", text)
        self.assertIn("--calibration_split_mode", text)
        self.assertIn("--calibration_eval_fraction", text)
        self.assertIn("spearman_calibrated_correct_rate", text)
        self.assertIn("calibration_heldout", text)

    def test_sparse_inlier_diagnostics_limits_iterable_cameras(self):
        module = self._load_inlier_diag_module()
        cameras = (idx for idx in range(5))
        self.assertEqual(module._limit_cameras(cameras, 3), [0, 1, 2])

    def test_sparse_inlier_diagnostics_tolerates_missing_optional_overrides(self):
        module = self._load_inlier_diag_module()
        config = {"sparse": {}}

        updated = module._apply_sparse_overrides(config, Namespace())

        self.assertTrue(updated["sparse"]["sparse_only"])

    def test_sparse_inlier_diagnostics_builds_block_calibration_holdout(self):
        module = self._load_inlier_diag_module()

        roles = module._calibration_image_roles(5, mode="block", eval_fraction=0.4)

        self.assertEqual(roles, ["train", "train", "train", "eval", "eval"])

    def test_sparse_inlier_diagnostics_uses_target_specific_calibration_denominator(self):
        module = self._load_inlier_diag_module()
        visible = module.torch.tensor([10, 8])
        matched = module.torch.tensor([4, 5])
        correct = module.torch.tensor([2, 4])
        inlier = module.torch.tensor([1, 3])

        correct_positive, correct_trial, correct_mask = module._calibration_target_counts(
            "correct",
            visible,
            matched,
            correct,
            inlier,
        )
        inlier_positive, inlier_trial, inlier_mask = module._calibration_target_counts(
            "inlier",
            visible,
            matched,
            correct,
            inlier,
        )

        self.assertTrue(module.torch.equal(correct_positive, correct))
        self.assertTrue(module.torch.equal(correct_trial, visible))
        self.assertTrue(module.torch.equal(correct_mask, visible > 0))
        self.assertTrue(module.torch.equal(inlier_positive, inlier))
        self.assertTrue(module.torch.equal(inlier_trial, matched))
        self.assertTrue(module.torch.equal(inlier_mask, matched > 0))

    def test_remap_topology_landmarks_script_uses_source_index_mapping(self):
        text = REMAP_TOPOLOGY_SCRIPT.read_text()
        self.assertIn("remap_sampled_indices_from_source_index", text)
        self.assertIn("--source_sampled_idx", text)
        self.assertIn("--topology_loc_state", text)
        self.assertIn("--output_sampled_idx", text)
        self.assertIn("--remap_score_source", text)
        self.assertIn("--remap_mode", text)
        self.assertIn("--max_source_distance", text)
        self.assertIn("loc_source_xyz", text)
        self.assertIn("loc_current_xyz", text)
        self.assertIn("missing_count", text)

    def test_dense_responsibility_diagnostics_script_reports_reconstruction_metrics(self):
        text = DENSE_RESP_DIAG_SCRIPT.read_text()
        self.assertIn("dense_localization_teacher", text)
        self.assertIn("responsibility_reconstruction_mean_cosine", text)
        self.assertIn("responsibility_reconstruction_p10_cosine", text)
        self.assertIn("--responsibility_topk", text)
        self.assertIn("--responsibility_opacity_weight", text)
        self.assertIn("--responsibility_depth_weight", text)
        self.assertIn("--anchor_count", text)
        self.assertIn("--output", text)

    def test_dense_responsibility_diagnostics_summarizes_images_by_valid_anchor_count(self):
        module = self._load_dense_resp_diag_module()
        images = [
            {
                "anchor_count": 128,
                "responsibility_reconstruction_mean_cosine": 0.5,
                "responsibility_reconstruction_p10_cosine": 0.2,
                "responsibility_reconstruction_min_cosine": -0.1,
                "responsibility_reconstruction_valid_anchor_count": 10,
            },
            {
                "anchor_count": 64,
                "responsibility_reconstruction_mean_cosine": 0.8,
                "responsibility_reconstruction_p10_cosine": 0.6,
                "responsibility_reconstruction_min_cosine": 0.1,
                "responsibility_reconstruction_valid_anchor_count": 30,
            },
        ]

        summary = module._summarize_images(images)

        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(summary["total_valid_anchor_count"], 40)
        self.assertAlmostEqual(summary["mean_anchor_count"], 96.0)
        self.assertAlmostEqual(summary["mean_responsibility_reconstruction_mean_cosine"], 0.725)
        self.assertAlmostEqual(summary["mean_responsibility_reconstruction_p10_cosine"], 0.5)
        self.assertAlmostEqual(summary["min_responsibility_reconstruction_min_cosine"], -0.1)

    def test_pseudo_query_pipeline_uses_shared_eval_cfg_builder(self):
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()

        self.assertIn("EVAL_SPARSE_DETECT_NUM=${EVAL_SPARSE_DETECT_NUM:-}", text)
        self.assertIn("EVAL_SPARSE_REPROJECTION_ERROR=${EVAL_SPARSE_REPROJECTION_ERROR:-}", text)
        self.assertIn("eval_cfg_args=()", text)
        self.assertIn('--detect_num "$EVAL_SPARSE_DETECT_NUM"', text)
        self.assertIn('--reprojection_error "$EVAL_SPARSE_REPROJECTION_ERROR"', text)
        self.assertIn("scripts/make_stdloc_eval_cfg.py", text)
        self.assertNotIn('"$PYTHON" - "$CFG" "$out_cfg"', text)

    def test_pseudo_query_pipeline_checks_cuda_toolchain_before_training(self):
        text = PSEUDO_QUERY_PIPELINE_SCRIPT.read_text()

        self.assertIn("require_cuda_toolchain()", text)
        self.assertIn('"$CUDA_HOME/bin/nvcc"', text)
        self.assertIn("cuda_runtime.h", text)

    def test_plain_sparse_feature_override_is_explicitly_enabled(self):
        text = PLAIN_SPARSE_EVAL_SCRIPT.read_text()

        feature_override_case = text.split("feature_override)", 1)[1].split(
            ";;", 1
        )[0]
        self.assertIn("--landmark_feature_override_path", feature_override_case)
        self.assertIn("--override_landmark_features", feature_override_case)

    def test_plain_sparse_mapping_gate_is_supported_end_to_end(self):
        runner = PLAIN_SPARSE_EVAL_SCRIPT.read_text()
        stdloc = STDLOC_SCRIPT.read_text()

        self.assertIn("LAFGS_EVAL_CAMERA_SUBSET_OVERRIDE", runner)
        self.assertIn('choices=["test", "train", "candidate_validation"]', stdloc)
        self.assertIn(
            'elif args.evaluation_camera_subset == "train":',
            stdloc,
        )
        self.assertIn("test_cameras = scene.getTrainCameras()", stdloc)


if __name__ == "__main__":
    unittest.main()
