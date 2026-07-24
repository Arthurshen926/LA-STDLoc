import torch
from pathlib import Path


def test_ulf_parity_dense_sampler_matches_reference_grid_sample():
    from train_lafgs_map import _sample_ulf_parity_dense_features

    torch.manual_seed(17)
    dense = torch.randn(1, 7, 3, 5)
    pixel_uv = torch.tensor([[0.0, 0.0], [3.0, 7.0], [18.0, 11.0]])
    image_hw = (24, 40)

    sampled = _sample_ulf_parity_dense_features(
        dense, pixel_uv, image_hw, channel_chunk=2
    )
    grid = pixel_uv.clone()
    grid[:, 0] = 2.0 * (grid[:, 0] + 0.5) / image_hw[1] - 1.0
    grid[:, 1] = 2.0 * (grid[:, 1] + 0.5) / image_hw[0] - 1.0
    # This is ULF-Loc's original direct stride-8 sampling expression.
    expected = torch.nn.functional.grid_sample(
        dense,
        grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, :, 0].T
    assert torch.allclose(sampled, expected, atol=1e-6)


def test_nonpositive_longest_edge_preserves_native_resolution():
    from utils.image_utils import get_resolution_from_longest_edge

    assert get_resolution_from_longest_edge(1080, 1920, 0) == (1080, 1920)
    assert get_resolution_from_longest_edge(1080, 1920, -1) == (1080, 1920)


def test_lafgs_parser_accepts_strict_ulf_parity_modes():
    from train_lafgs_map import build_parser

    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "--output_dir", "/tmp/lafgs-parity-test",
            "--scaffold_mode", "ulf_parity",
            "--initialization_mode", "ulf_parity",
            "--ulf_support_view_sampling", "pose_diverse",
            "--ulf_parity_kcs_mask_policy", "deployment_post_filter",
        ]
    )
    assert args.scaffold_mode == "ulf_parity"
    assert args.initialization_mode == "ulf_parity"
    assert args.ulf_support_view_sampling == "pose_diverse"
    assert args.ulf_parity_kcs_mask_policy == "deployment_post_filter"


def test_ulf_parity_kcs_mask_policy_defaults_to_reference_rgb_only():
    from train_lafgs_map import _resolve_ulf_parity_kcs_mask_policy, build_parser

    parser, _ = build_parser()
    args = parser.parse_args(["--output_dir", "/tmp/lafgs-parity-test"])
    assert args.ulf_parity_kcs_mask_policy == "rgb_only"
    assert _resolve_ulf_parity_kcs_mask_policy("rgb_only") == "rgb_only"
    assert (
        _resolve_ulf_parity_kcs_mask_policy("deployment_post_filter")
        == "deployment_post_filter"
    )


def test_pose_diverse_support_sampling_is_deterministic_and_unique():
    from train_lafgs_map import _subsample_ulf_support_cameras

    class Camera:
        def __init__(self, center):
            self.camera_center = torch.tensor(center, dtype=torch.float32)

    cameras = [
        Camera((0.0, 0.0, 0.0)),
        Camera((1.0, 0.0, 0.0)),
        Camera((2.0, 0.0, 0.0)),
        Camera((10.0, 0.0, 0.0)),
    ]
    selected = _subsample_ulf_support_cameras(cameras, 3, "pose_diverse")
    selected_centers = [tuple(camera.camera_center.tolist()) for camera in selected]
    assert selected_centers == [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    ]


def test_native_sparse_descriptor_sampling_matches_native_dense_sampler():
    from encoders.sp_encoder.export_image_embeddings import sample_descriptors
    from localization_training.ulf_initializer import (
        grid_index_to_physical,
        sample_dense_descriptors_at_image_uv,
    )

    torch.manual_seed(7)
    dense = torch.randn(1, 8, 4, 5)
    dense = torch.nn.functional.normalize(dense, dim=1)
    keypoints = torch.tensor([[0.0, 0.0], [7.0, 9.0], [31.0, 24.0]])
    sparse = sample_descriptors(keypoints[None], dense)[0].T
    fused = sample_dense_descriptors_at_image_uv(
        dense,
        grid_index_to_physical(keypoints),
        (32, 40),
    )
    cosine = (sparse * fused).sum(dim=1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)


def test_surface_normal_and_geometry_weight_use_surfel_local_z_axis():
    from localization_training.ulf_initializer import (
        geometry_view_weights,
        surface_normals_from_rotation,
    )

    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    normals = surface_normals_from_rotation(rotation, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(normals, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)
    fronto_parallel = geometry_view_weights(
        torch.tensor([[0.0, 0.0, 0.0]]), normals, torch.tensor([0.0, 0.0, 2.0])
    )
    grazing = geometry_view_weights(
        torch.tensor([[0.0, 0.0, 0.0]]), normals, torch.tensor([2.0, 0.0, 0.0])
    )
    assert torch.allclose(fronto_parallel, torch.ones(1), atol=1e-6)
    assert torch.allclose(grazing, torch.zeros(1), atol=1e-6)


def test_consensus_nearest_keypoint_distance_is_exact_without_faiss():
    from localization_training.ulf_initializer import nearest_keypoint_distance

    projected = torch.tensor([[1.0, 1.0], [3.0, 4.0], [10.0, 0.0]])
    keypoints = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    distance = nearest_keypoint_distance(projected, keypoints, chunk_size=1)
    assert torch.allclose(distance, torch.tensor([0.0, 1.0, (52.0) ** 0.5]))


def test_pixel_center_projection_returns_feature_grid_index():
    from localization_training.direct_landmark_teacher import project_landmarks_to_query

    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    K = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)
    uv, _, valid = project_landmarks_to_query(
        xyz,
        K,
        pose,
        height=10,
        width=10,
        pixel_center_offset=0.5,
    )
    assert valid.tolist() == [True]
    assert torch.allclose(uv, torch.tensor([[4.5, 4.5]]))


def test_robust_kcs_requires_visible_rate_and_distinct_vote_bins():
    from localization_training.ulf_initializer import consensus_eligibility

    eligible, rate = consensus_eligibility(
        torch.tensor([3, 3, 1, 4]),
        torch.tensor([4, 30, 4, 4]),
        minimum_votes=2,
        minimum_visible_views=4,
        minimum_rate=0.5,
        distinct_view_bins=torch.tensor([2, 2, 1, 1]),
        minimum_distinct_view_bins=2,
        distinct_trajectory_bins=torch.tensor([2, 2, 1, 2]),
        minimum_distinct_trajectory_bins=2,
    )
    assert eligible.tolist() == [True, False, False, False]
    assert torch.allclose(rate, torch.tensor([0.75, 0.1, 0.25, 1.0]))


def test_streaming_cosine_histogram_trim_drops_bottom_quantile_conservatively():
    from localization_training.ulf_initializer import (
        accumulate_cosine_histogram,
        cosine_histogram_trim_thresholds,
    )

    histogram = torch.zeros((2, 10), dtype=torch.int32)
    histogram = accumulate_cosine_histogram(
        histogram,
        torch.tensor([0, 0, 0, 0, 1, 1]),
        torch.tensor([-0.9, -0.7, 0.4, 0.8, -0.2, 0.9]),
    )
    thresholds = cosine_histogram_trim_thresholds(histogram, 0.25)
    # Landmark zero's bottom item is discarded; its retained threshold is the
    # lower boundary of the next bin. A two-observation row rounds down to no
    # explicit removal, but its threshold is still the first observed bin so a
    # newly recomputed lower-cosine sample cannot enter the final fusion.
    assert thresholds[0] > -0.9
    assert thresholds[0] <= -0.6
    assert torch.isclose(thresholds[1], torch.tensor(-0.2), atol=1e-6)


def test_adaptive_histogram_trim_only_targets_supported_low_agreement_tails():
    from localization_training.ulf_initializer import (
        adaptive_cosine_histogram_trim_fractions,
    )

    # Landmark 0 is a stable high-cosine prototype; landmark 1 has a broad
    # incompatible tail; landmark 2 does not have enough observations to
    # estimate a landmark-specific trimming schedule.
    histogram = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 3, 4],
            [3, 2, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    fractions, tail_rate = adaptive_cosine_histogram_trim_fractions(
        histogram,
        tail_cosine=0.5,
        min_fraction=0.0,
        max_fraction=0.2,
        min_observations=4,
    )
    assert tail_rate[0] == 0.0
    assert torch.isclose(fractions[0], torch.tensor(0.0))
    assert tail_rate[1] > 0.5
    assert torch.isclose(fractions[1], torch.tensor(0.2))
    assert torch.isclose(fractions[2], torch.tensor(0.0))


def test_relative_mad_adaptive_trim_preserves_stable_low_cosine_landmarks():
    from localization_training.ulf_initializer import (
        adaptive_cosine_histogram_trim_schedule,
    )

    # The first landmark is consistently only moderately aligned with its
    # mean prototype. An absolute 0.75 gate would trim it, whereas a
    # landmark-relative tail test must preserve it. The second contains a
    # clearly separated low-cosine mode and should receive the capped trim.
    histogram = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    fractions, tail_rate, thresholds, median, mad = (
        adaptive_cosine_histogram_trim_schedule(
            histogram,
            tail_cosine=0.75,
            min_fraction=0.0,
            max_fraction=0.2,
            min_observations=4,
            mode="relative_mad",
            mad_scale=2.0,
        )
    )
    assert torch.allclose(fractions, torch.tensor([0.0, 0.2]))
    assert torch.allclose(tail_rate, torch.tensor([0.0, 0.2]))
    assert median is not None and mad is not None
    assert thresholds[0] < median[0]
    assert thresholds[1] > -0.5


def test_streaming_weighted_cosine_medoid_chooses_central_observation():
    from localization_training.ulf_initializer import update_weighted_cosine_medoid_state

    # The first descriptor has the largest weighted cosine support from the
    # other two observations; this is the exact cosine medoid once the
    # geometry-weighted prototype has been formed.
    descriptors = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.95, 0.31], [-1.0, 0.0]]), dim=-1
    )
    weights = torch.tensor([3.0, 2.0, 0.25])
    prototype = torch.nn.functional.normalize(
        (descriptors * weights[:, None]).sum(dim=0, keepdim=True), dim=-1
    )
    best_scores = torch.full((1,), -torch.inf)
    best_features = torch.zeros((1, 2))
    # Per-view updates are unique; stream the three observations as they would
    # arrive from separate support views.
    for descriptor in descriptors:
        update_weighted_cosine_medoid_state(
            best_scores,
            best_features,
            torch.tensor([0]),
            descriptor[None],
            prototype,
        )
    exhaustive_scores = (descriptors @ descriptors.T) @ weights
    expected = descriptors[int(exhaustive_scores.argmax().item())]
    assert torch.allclose(best_features[0], expected, atol=1e-6)


def test_lafgs_parser_accepts_explicit_robust_ulf_modes():
    from train_lafgs_map import build_parser

    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "--output_dir", "/tmp/lafgs-robust-test",
            "--scaffold_mode", "ulf_robust_consensus",
            "--initialization_mode", "ulf_robust_geometry",
            "--ulf_consensus_min_visible_views", "4",
            "--ulf_consensus_min_rate", "0.2",
            "--ulf_consensus_view_bins", "4",
            "--ulf_consensus_min_distinct_view_bins", "2",
            "--ulf_fusion_descriptor_trim_fraction", "0.1",
            "--ulf_fusion_reference_mode", "weighted_cosine_medoid",
        ]
    )
    assert args.scaffold_mode == "ulf_robust_consensus"
    assert args.initialization_mode == "ulf_robust_geometry"
    assert args.ulf_consensus_min_distinct_view_bins == 2
    assert args.ulf_fusion_descriptor_trim_fraction == 0.1
    assert args.ulf_fusion_reference_mode == "weighted_cosine_medoid"


def test_lafgs_parser_accepts_adaptive_robust_gwff_modes():
    from train_lafgs_map import build_parser

    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "--output_dir", "/tmp/lafgs-adaptive-gwff-test",
            "--scaffold_mode", "ulf_robust_consensus",
            "--initialization_mode", "ulf_robust_geometry",
            "--ulf_fusion_descriptor_trim_fraction", "0",
            "--ulf_fusion_adaptive_trim",
            "--ulf_fusion_adaptive_trim_min_fraction", "0.0",
            "--ulf_fusion_adaptive_trim_max_fraction", "0.2",
            "--ulf_fusion_adaptive_trim_tail_cosine", "0.75",
            "--ulf_fusion_adaptive_trim_min_observations", "4",
            "--ulf_fusion_adaptive_trim_mode", "relative_mad",
            "--ulf_fusion_adaptive_trim_mad_scale", "2.5",
        ]
    )
    assert args.ulf_fusion_adaptive_trim
    assert args.ulf_fusion_descriptor_trim_fraction == 0.0
    assert args.ulf_fusion_adaptive_trim_max_fraction == 0.2
    assert args.ulf_fusion_adaptive_trim_mode == "relative_mad"
    assert args.ulf_fusion_adaptive_trim_mad_scale == 2.5


def test_formal_lafgs_runners_default_to_stratified_temporal_holdout():
    root = Path(__file__).resolve().parents[1]
    expected = "stratified_temporal_block"
    for script in (
        "run_lafgs_v2_ulfparity_alternating.sh",
        "run_lafgs_v2_robust_initializer_ablation.sh",
        "run_lafgs_v2_factor_matrix.sh",
        "run_lafgs_v2_widebank_distill_refresh.sh",
    ):
        text = (root / "scripts" / script).read_text(encoding="utf-8")
        assert expected in text

    robust = (root / "scripts" / "run_lafgs_v2_robust_initializer_ablation.sh").read_text(
        encoding="utf-8"
    )
    assert 'LAFGS_ROBUST_CAMERA_LOADER_WORKERS:-0' in robust
    assert 'export STDLOC_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS"' in robust
    assert "select_residual) select_residual" in robust
    assert "validation_only_performance_v1" in robust
    assert (
        'ROBUST_PROTOCOL_VERSION="v4_exact_fusion_bins_split${SPLIT_MODE}_seed${SPLIT_SEED}_fullres_native_uncapped"'
        in robust
    )
    assert "LAFGS_ROBUST_INDEPENDENT_BIN_SCORING:-1" in robust
    assert '--ulf_fusion_view_bins "$FUSION_VIEW_BINS"' in robust
    assert "verify_state_protocol" in robust
    assert "verify_eval_protocol" in robust
    assert "verify_eval_config_binding" in robust
    assert 'verify_eval_config_binding "$cfg" "$state"' in robust
    assert "s/^Result are saved in //p" in robust
    assert "s/^Results are saved in //p" in robust
    assert 'LAFGS_ROBUST_MIN_RATE:-0.01' in robust
    assert 'LAFGS_ROBUST_FUSION_REFERENCE_MODE:-mean' in robust
    assert '--ulf_fusion_reference_mode "$FUSION_REFERENCE_MODE"' in robust
    assert 'LAFGS_ROBUST_ADAPTIVE_TRIM:-0' in robust
    assert 'adaptive GWFF requires LAFGS_ROBUST_TRIM_FRACTION=0' in robust
    assert '--ulf_fusion_adaptive_trim' in robust
    assert 'LAFGS_ROBUST_ADAPTIVE_TRIM_MODE:-relative_mad' in robust
    assert 'LAFGS_ROBUST_LANDMARK_SOURCE_PATH:-' in robust
    assert 'verify_bootstrap_landmark_source' in robust
    assert 'LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_WEIGHT:-0.0' in robust
    assert 'RESIDUAL_PROFILE_TAG=' in robust
    assert '--native_global_attractor_weight "$NATIVE_GLOBAL_ATTRACTOR_WEIGHT"' in robust
    assert 'residual profile mismatch for {name}' in robust

    canonical = (root / "scripts" / "run_lafgs_v2_canonical_native_mainline.sh").read_text(
        encoding="utf-8"
    )
    assert 'LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_WEIGHT=0.25' in canonical
    assert 'false-attractor-aware pure-native 5K residual' in canonical

    widebank = (root / "scripts" / "run_lafgs_v2_widebank_distill_refresh.sh").read_text(
        encoding="utf-8"
    )
    assert 'LAFGS_WIDEBANK_CAMERA_LOADER_WORKERS:-0' in widebank
    assert "Wide-bank source split does not match this study" in widebank
    assert "Set LAFGS_WIDEBANK_SOURCE_RUN_ROOT" in widebank
    assert 'WIDEBANK_PROTOCOL_VERSION="v7_split${SPLIT_MODE}_fullres_native_uncapped_qcore${HARD_CORE_TAG}_qres${QUALITY_RESERVOIR_TAG}_qscore${QUALITY_RESERVOIR_SCORE}_qz${QUALITY_RESERVOIR_Z_TAG}"' in widebank
    assert "verify_stage_state_protocol" in widebank
    assert "verify_validation_protocol" in widebank
    assert "results/residual_selection_safety.json" in widebank
    assert "formal_deployment_protocol" in widebank

    factor = (root / "scripts" / "run_lafgs_v2_factor_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert 'LAFGS_FACTOR_CAMERA_LOADER_WORKERS:-0' in factor
    assert 'LAFGS_ULF_SPLIT_MODE="$SPLIT_MODE"' in factor
    assert "strong_control_provenance" in factor
    assert 'FACTOR_PROTOCOL_VERSION="v2_split${SPLIT_MODE}_fullres_native_uncapped"' in factor
    assert "verify_factor_state_protocol" in factor
    assert "verify_factor_validation_protocol" in factor
    assert "LAFGS_ULF_PARITY_KCS_MASK_POLICY=rgb_only" in factor
    assert "--ulf_parity_kcs_mask_policy rgb_only" in factor


def test_bootstrap_matrix_wrapper_keeps_attachment_u0_to_u4_factors_explicit():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_lafgs_v2_bootstrap_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "u0) budget=16000; support_views=128; support_sampling=uniform" in text
    assert "u1) budget=20000; support_views=128; support_sampling=uniform" in text
    assert "u2) budget=20000; support_views=256; support_sampling=pose_diverse" in text
    assert "u3) budget=20000; support_views=0; support_sampling=uniform" in text
    assert "u4) budget=32000; support_views=0; support_sampling=uniform" in text
    assert "LAFGS_ULF_PARITY_KCS_MASK_POLICY=rgb_only" in text
    assert "bootstrap_gate_passed" in text
    assert "Bootstrap-matrix row" in text
    assert "bootstrap_validate" in text
    assert "test_evaluation_forbidden" in text


def test_robust_ba_refresh_runner_is_validation_only_and_fails_closed():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_lafgs_v2_robust_ba_refresh.sh").read_text(
        encoding="utf-8"
    )
    assert "selection_uses_control" in text
    assert "Skipping refresh because BA did not beat the selected residual" in text
    assert "predeclared_terminal_ba_coupling" in text
    assert "select_terminal_refresh" in text
    assert "final_selection_control\": \"original_selected_residual" in text
    assert "LAFGS_ROBUST_BA_BOOTSTRAP_DIR" in text
    assert "bootstrap_source_dir" in text
    assert "manifest_visibility_cache" in text
    assert "descriptor_end_step -1" in text
    assert "--evaluation_camera_subset candidate_validation" in text
    assert "test_evaluation_forbidden" in text
    assert "--evaluation_camera_subset test" not in text
