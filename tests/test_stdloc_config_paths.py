import unittest


class STDLocConfigPathTest(unittest.TestCase):
    def test_ulfloc_native_frontend_requires_the_native_direct_contract(self):
        from stdloc import validate_sparse_frontend_config

        valid = {
            "sparse_frontend": "ulfloc_native",
            "query_feature_contract": "native_resized_input",
            "use_landmark_prior": False,
        }
        self.assertEqual(validate_sparse_frontend_config(valid), "ulfloc_native")

        with self.assertRaisesRegex(ValueError, "native_resized_input"):
            validate_sparse_frontend_config(
                {**valid, "query_feature_contract": "legacy_full_then_resized_map"}
            )
        with self.assertRaisesRegex(ValueError, "use_pair_measurement"):
            validate_sparse_frontend_config({**valid, "use_pair_measurement": True})

    def test_native_matchability_is_solver_only_and_cannot_change_candidates(self):
        from stdloc import validate_sparse_frontend_config

        config = {
            "sparse_frontend": "ulfloc_native",
            "query_feature_contract": "native_resized_input",
            "use_landmark_prior": False,
            "use_native_matchability": True,
            "native_matchability_state_path": "/tmp/native_matchability.pt",
            "topk": 1,
            "threshold": 0.0,
        }
        self.assertEqual(validate_sparse_frontend_config(config), "ulfloc_native")
        self.assertEqual(
            validate_sparse_frontend_config(
                {**config, "use_two_stage_pose_refinement": True}
            ),
            "ulfloc_native",
        )
        with self.assertRaisesRegex(ValueError, "topk must be 1"):
            validate_sparse_frontend_config({**config, "topk": 2})
        with self.assertRaisesRegex(ValueError, "max_matches_per_landmark"):
            validate_sparse_frontend_config(
                {**config, "max_matches_per_landmark": 1}
            )

    def test_pose_sufficient_selector_requires_unchanged_top1_candidates(self):
        from stdloc import validate_sparse_frontend_config

        config = {
            "sparse_frontend": "ulfloc_native_metric",
            "query_feature_contract": "native_resized_input",
            "use_landmark_prior": False,
            "metric_state_path": "/tmp/metric.pt",
            "use_pose_sufficient_selector": True,
            "pose_sufficient_selector_state_path": "/tmp/selector.pt",
            "pose_sufficient_budget": 512,
            "topk": 1,
            "threshold": 0.0,
        }
        self.assertEqual(
            validate_sparse_frontend_config(config), "ulfloc_native_metric"
        )
        with self.assertRaisesRegex(ValueError, "threshold must be 0"):
            validate_sparse_frontend_config({**config, "threshold": 0.1})
        with self.assertRaisesRegex(ValueError, "geometry_balance"):
            validate_sparse_frontend_config(
                {
                    **config,
                    "geometry_balance": {"enabled": True},
                }
            )

    def test_raw_tensor_sha256_matches_artifact_byte_contract(self):
        import hashlib

        import torch

        from stdloc import raw_tensor_sha256

        value = torch.tensor([3, 7, 11], dtype=torch.int64)
        expected = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        self.assertEqual(raw_tensor_sha256(value), expected)

    def test_candidate_teacher_geometry_requires_alignment_and_is_finite(self):
        import torch

        from stdloc import load_candidate_teacher_landmark_geometry

        state = {
            "landmark_indices": torch.tensor([2, 5]),
            "landmark_xyz": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        }
        xyz = load_candidate_teacher_landmark_geometry(state, torch.tensor([2, 5]))
        self.assertEqual(tuple(xyz.shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            load_candidate_teacher_landmark_geometry(state, torch.tensor([5, 2]))
        invalid = dict(state)
        invalid["landmark_xyz"] = torch.tensor(
            [[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_candidate_teacher_landmark_geometry(
                invalid, torch.tensor([2, 5])
            )

    def test_direct_candidate_validation_matches_training_holdout(self):
        from stdloc import select_candidate_validation_cameras
        from train_detector import partition_candidate_teacher_cameras

        cameras = list(range(20))
        _, expected, _ = partition_candidate_teacher_cameras(
            cameras,
            validation_ratio=0.2,
            split_mode="temporal_block",
            split_seed=2026,
        )
        actual = select_candidate_validation_cameras(
            cameras,
            validation_ratio=0.2,
            split_mode="temporal_block",
            split_seed=2026,
            direct_holdout=True,
        )

        self.assertEqual(actual, expected)

    def test_stratified_direct_candidate_validation_matches_training_holdout(self):
        from types import SimpleNamespace

        from stdloc import select_candidate_validation_cameras
        from train_detector import partition_candidate_teacher_cameras

        cameras = [
            SimpleNamespace(image_name=f"seq0/frame{index:05d}.png")
            for index in range(8)
        ] + [
            SimpleNamespace(image_name=f"seq1/frame{index:05d}.png")
            for index in range(6)
        ] + [
            SimpleNamespace(image_name=f"seq2/frame{index:05d}.png")
            for index in range(5)
        ]
        unordered_cameras = list(reversed(cameras))
        _, expected, _ = partition_candidate_teacher_cameras(
            cameras,
            validation_ratio=0.25,
            split_mode="stratified_temporal_block",
            split_seed=2026,
        )
        actual = select_candidate_validation_cameras(
            unordered_cameras,
            validation_ratio=0.25,
            split_mode="stratified_temporal_block",
            split_seed=2026,
            direct_holdout=True,
        )

        self.assertEqual(
            [camera.image_name for camera in actual],
            [camera.image_name for camera in expected],
        )

    def test_explicit_evaluation_camera_list_preserves_requested_order(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from stdloc import load_evaluation_camera_list

        cameras = [
            SimpleNamespace(image_name="seq1/frame00002.png"),
            SimpleNamespace(image_name="seq1/frame00001.png"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text(json.dumps({"image_names": ["seq1/frame00001.png"]}))
            selected = load_evaluation_camera_list(cameras, path)

        self.assertEqual([camera.image_name for camera in selected], ["seq1/frame00001.png"])

    def test_direct_holdout_requires_matching_training_partition(self):
        from stdloc import candidate_direct_holdout_mismatches
        from stdloc import validate_candidate_direct_holdout_compatibility

        state_config = {
            "validation_ratio": 0.2,
            "split_mode": "temporal_block",
            "split_seed": 2026,
        }
        self.assertEqual(
            candidate_direct_holdout_mismatches(
                state_config,
                validation_ratio=0.2,
                split_mode="temporal_block",
                split_seed=2026,
            ),
            [],
        )
        self.assertEqual(
            validate_candidate_direct_holdout_compatibility(
                state_config,
                validation_ratio=0.2,
                split_mode="temporal_block",
                split_seed=2026,
            ),
            [],
        )
        with self.assertRaisesRegex(ValueError, "validation_ratio.*0.0.*0.2"):
            validate_candidate_direct_holdout_compatibility(
                {**state_config, "validation_ratio": 0.0},
                validation_ratio=0.2,
                split_mode="temporal_block",
                split_seed=2026,
            )

    def test_sparse_artifact_overrides_are_explicit_and_independent(self):
        from stdloc import apply_sparse_artifact_overrides

        config = {
            "sparse": {
                "detector_path": "baseline_detector.pth",
                "landmark_feature_override_path": "baseline_state.pt",
            }
        }
        apply_sparse_artifact_overrides(
            config,
            detector_path="/tmp/checkpoint_detector.pth",
            landmark_feature_override_path="/tmp/checkpoint_state.pt",
        )

        self.assertEqual(
            config["sparse"]["detector_path"],
            "/tmp/checkpoint_detector.pth",
        )
        self.assertEqual(
            config["sparse"]["landmark_feature_override_path"],
            "/tmp/checkpoint_state.pt",
        )

    def test_candidate_frontend_mismatch_can_fail_strictly(self):
        from stdloc import candidate_frontend_mismatches
        from stdloc import validate_candidate_frontend_compatibility

        trained = {
            "detect_num": 4096,
            "nms_radius": 2,
            "match_mode": "topk",
            "match_topk": 1,
            "match_threshold": 0.0,
            "dual_softmax": False,
            "dual_softmax_temperature": 0.1,
            "pair_context_topk": 8,
            "map_max_matches_per_landmark": 2,
        }
        evaluated = {
            "detect_num": 8192,
            "nms": 2,
            "mnn_match": False,
            "topk": 1,
            "threshold": 0.0,
            "dual_softmax": False,
            "dual_softmax_temp": 0.1,
            "pair_context_topk": 8,
            "max_matches_per_landmark": 2,
            "candidate_frontend_match_policy": "error",
        }

        self.assertEqual(
            candidate_frontend_mismatches(trained, evaluated),
            [("detect_num", 4096, 8192)],
        )
        with self.assertRaisesRegex(ValueError, "detect_num.*4096.*8192"):
            validate_candidate_frontend_compatibility(trained, evaluated)

        evaluated["detect_num"] = 4096
        self.assertEqual(validate_candidate_frontend_compatibility(trained, evaluated), [])
        evaluated["max_matches_per_landmark"] = 1
        self.assertEqual(
            candidate_frontend_mismatches(trained, evaluated),
            [("map_max_matches_per_landmark", 2, 1)],
        )

    def test_detector_query_feature_contract_mismatch_can_fail_strictly(self):
        from stdloc import validate_detector_query_feature_contract

        matching = {
            "detector_training_query_feature_contract": "native_resized_input",
            "query_feature_contract": "native_resized_input",
            "candidate_frontend_match_policy": "error",
        }
        self.assertIsNone(validate_detector_query_feature_contract(matching))
        mismatched = dict(matching)
        mismatched["query_feature_contract"] = "legacy_full_then_resized_map"
        with self.assertRaisesRegex(ValueError, "query feature contract mismatch"):
            validate_detector_query_feature_contract(mismatched)

    def test_native_reject_threshold_must_match_native_deployment(self):
        from stdloc import validate_native_reject_threshold_contract

        state_config = {
            "native_reject_contract": {
                "enabled": True,
                "deployment_match_threshold": 0.5,
                "source": "current_native_residual",
            }
        }
        matching = validate_native_reject_threshold_contract(
            state_config, {"threshold": 0.5}
        )
        self.assertTrue(matching["matches"])
        with self.assertRaisesRegex(ValueError, "does not match deployment"):
            validate_native_reject_threshold_contract(
                state_config, {"threshold": 0.0}
            )
        with self.assertWarnsRegex(RuntimeWarning, "does not match deployment"):
            override = validate_native_reject_threshold_contract(
                state_config,
                {
                    "threshold": 0.0,
                    "allow_native_reject_threshold_mismatch": True,
                },
            )
        self.assertTrue(override["override"])

    def test_candidate_teacher_features_require_exact_landmark_alignment(self):
        import tempfile
        from pathlib import Path

        import torch

        from stdloc import load_candidate_teacher_landmark_features

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.pt"
            torch.save(
                {
                    "landmark_indices": torch.tensor([2, 5]),
                    "landmark_features": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                },
                path,
            )
            features, _ = load_candidate_teacher_landmark_features(
                path,
                torch.tensor([2, 5]),
                expected_feature_dim=2,
            )
            self.assertTrue(torch.allclose(torch.linalg.norm(features, dim=1), torch.ones(2)))

            with self.assertRaisesRegex(ValueError, "not aligned"):
                load_candidate_teacher_landmark_features(path, torch.tensor([5, 2]))

            malformed_path = Path(tmp) / "malformed_candidate.pt"
            torch.save(
                {
                    "landmark_indices": torch.tensor([2, 5]),
                    "landmark_features": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
                },
                malformed_path,
            )
            with self.assertRaisesRegex(ValueError, "feature count"):
                load_candidate_teacher_landmark_features(
                    malformed_path,
                    torch.tensor([2, 5]),
                    expected_feature_dim=2,
                )

    def test_artifact_hash_and_feature_delta_are_deterministic(self):
        import tempfile
        from pathlib import Path

        import torch

        from stdloc import file_sha256, landmark_feature_delta, tensor_sha256

        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        changed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(tensor_sha256(features), tensor_sha256(features.clone()))
        self.assertNotEqual(tensor_sha256(features), tensor_sha256(changed))
        delta = landmark_feature_delta(features, changed)
        self.assertAlmostEqual(delta["l2_mean"], 2 ** 0.5 / 2.0)
        self.assertAlmostEqual(delta["cosine_mean"], 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"artifact")
            self.assertEqual(file_sha256(path), file_sha256(path))
            self.assertIsNone(file_sha256(Path(tmp) / "missing"))

    def test_evaluation_protocol_hash_captures_resolution_and_image_content(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from stdloc import build_evaluation_protocol

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "processed" / "seq1"
            image_root.mkdir(parents=True)
            image_path = image_root / "frame00001.png"
            image_path.write_bytes(b"first image")
            dataset = SimpleNamespace(
                source_path=str(root),
                images="processed",
                resolution=1,
                longest_edge=640,
                feature_type="sp",
                gaussian_type="2dgs",
            )
            args = SimpleNamespace(
                evaluation_camera_subset="test",
                candidate_query_ratio=0.2,
                candidate_validation_ratio=0.25,
                candidate_split_mode="temporal_block",
                candidate_split_seed=2026,
                candidate_direct_validation_holdout=False,
            )
            camera = SimpleNamespace(
                image_name="seq1/frame00001.png",
                original_image=SimpleNamespace(shape=(3, 1080, 1920)),
            )

            first = build_evaluation_protocol(dataset, args, [camera])
            dataset.resolution = -1
            resized = build_evaluation_protocol(dataset, args, [camera])
            self.assertNotEqual(
                first["protocol_sha256"], resized["protocol_sha256"]
            )
            dataset.resolution = 1
            image_path.write_bytes(b"changed image")
            changed = build_evaluation_protocol(dataset, args, [camera])
            self.assertNotEqual(
                first["protocol_sha256"], changed["protocol_sha256"]
            )

    def test_evaluation_protocol_does_not_consume_scene_dataloader(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from stdloc import build_evaluation_protocol

        class NeverIterateLoader:
            def __init__(self, scene):
                self.dataset = SimpleNamespace(scene=scene, split="test")

            def __iter__(self):
                raise AssertionError("protocol construction consumed the loader")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "processed" / "seq1"
            image_root.mkdir(parents=True)
            (image_root / "frame00001.png").write_bytes(b"image")
            camera_info = SimpleNamespace(
                image_name="seq1/frame00001.png",
                width=1920,
                height=1080,
            )
            scene = SimpleNamespace(
                scene_info=SimpleNamespace(
                    train_cameras=[],
                    test_cameras=[camera_info],
                )
            )
            dataset = SimpleNamespace(
                source_path=str(root),
                images="processed",
                resolution=-1,
                longest_edge=640,
                feature_type="sp",
                gaussian_type="2dgs",
            )
            args = SimpleNamespace(
                evaluation_camera_subset="test",
                candidate_query_ratio=0.2,
                candidate_validation_ratio=0.25,
                candidate_split_mode="temporal_block",
                candidate_split_seed=2026,
                candidate_direct_validation_holdout=False,
            )

            protocol = build_evaluation_protocol(
                dataset,
                args,
                NeverIterateLoader(scene),
            )

            self.assertEqual(
                protocol["loaded_image_shapes"],
                [{"height": 900, "width": 1600, "count": 1}],
            )

    def test_scene_disables_forked_camera_workers_for_cuda_images(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        from scene import Scene

        scene = Scene.__new__(Scene)
        scene.args = SimpleNamespace(data_device="cuda")
        self.assertEqual(scene._camera_loader_num_workers(), 0)
        scene.args.data_device = "cuda:1"
        self.assertEqual(scene._camera_loader_num_workers(), 0)
        scene.args.data_device = "cpu"
        self.assertEqual(scene._camera_loader_num_workers(), 4)
        with patch.dict(os.environ, {"STDLOC_CAMERA_LOADER_WORKERS": "0"}):
            self.assertEqual(scene._camera_loader_num_workers(), 0)
        with patch.dict(os.environ, {"STDLOC_CAMERA_LOADER_WORKERS": "3"}):
            self.assertEqual(scene._camera_loader_num_workers(), 3)
        with patch.dict(os.environ, {"STDLOC_CAMERA_LOADER_WORKERS": "-1"}):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                scene._camera_loader_num_workers()

    def test_topk_match_preserves_keypoint_ids_for_multiple_matches_per_row(self):
        import torch

        from stdloc import topk_match

        correlation = torch.tensor([[[0.9, 0.8, 0.1], [0.7, 0.6, 0.5]]])
        image_idx, landmark_idx, values = topk_match(correlation, topk=2, thr=0.0)

        self.assertEqual(image_idx.tolist(), [0, 0, 1, 1])
        self.assertEqual(landmark_idx.tolist(), [0, 1, 0, 1])
        self.assertTrue(torch.allclose(values, torch.tensor([0.9, 0.8, 0.7, 0.6])))

    def test_resolve_artifact_path_supports_external_model_roots_and_absolute_paths(self):
        from stdloc import resolve_artifact_path

        self.assertEqual(
            resolve_artifact_path("/models/la", "detector/sampled_idx.pkl"),
            "/models/la/detector/sampled_idx.pkl",
        )
        self.assertEqual(
            resolve_artifact_path("/models/la", "detector/sampled_idx.pkl", "/models/base"),
            "/models/base/detector/sampled_idx.pkl",
        )
        self.assertEqual(
            resolve_artifact_path("/models/la", "/tmp/detector.pth", "/models/base"),
            "/tmp/detector.pth",
        )

    def test_validate_sampled_indices_rejects_out_of_range_landmarks(self):
        import torch

        from stdloc import validate_sampled_indices

        with self.assertRaisesRegex(
            ValueError,
            "out of bounds.*point_count=3.*max=3",
        ):
            validate_sampled_indices(torch.tensor([0, 2, 3], device="cpu"), 3)

    def test_resize_sparse_valid_mask_to_feature_grid_uses_area_fraction(self):
        import torch

        from stdloc import resize_sparse_valid_mask_to_feature_grid

        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[:4, :4] = True

        resized = resize_sparse_valid_mask_to_feature_grid(mask, 2, 2, min_fraction=0.5)

        self.assertEqual(resized.tolist(), [[True, False], [False, False]])

    def test_filter_sparse_keypoints_by_valid_mask_counts_removed_points(self):
        import torch

        from stdloc import filter_sparse_keypoints_by_valid_mask

        kp_ids = torch.tensor([0, 1, 5, 10])
        valid = torch.tensor(
            [
                [True, False, True, True],
                [True, True, False, True],
                [True, True, False, True],
            ]
        )

        filtered, diagnostics = filter_sparse_keypoints_by_valid_mask(kp_ids, valid, height=3, width=4)

        self.assertEqual(filtered.tolist(), [0, 5])
        self.assertEqual(diagnostics["detected_keypoints_raw"], 4)
        self.assertEqual(diagnostics["detected_keypoints"], 2)
        self.assertEqual(diagnostics["sparse_valid_mask_filtered_keypoints"], 2)

    def test_select_sparse_keypoints_by_valid_mask_never_refills_invalid_cells(self):
        import torch

        from stdloc import select_sparse_keypoints_by_valid_mask

        kp_ids = torch.tensor([0, 1, 2, 3, 4, 5])
        valid = torch.tensor([[True, False, True], [False, True, False]])

        selected, diagnostics = select_sparse_keypoints_by_valid_mask(
            kp_ids,
            valid,
            height=2,
            width=3,
            target_count=4,
            refill_invalid=True,
        )

        self.assertEqual(selected.tolist(), [0, 2, 4])
        self.assertEqual(diagnostics["detected_keypoints_raw"], 6)
        self.assertEqual(diagnostics["detected_keypoints"], 3)
        self.assertEqual(diagnostics["sparse_valid_mask_selected_valid_keypoints"], 3)
        self.assertEqual(diagnostics["sparse_valid_mask_refill_keypoints"], 0)

    def test_sparse_correspondence_diagnostics_reports_geometry_and_gt_precision(self):
        import numpy as np

        from stdloc import sparse_correspondence_diagnostics

        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        p3d = np.array(
            [
                [-0.2, -0.1, 2.0],
                [0.1, -0.1, 2.2],
                [0.2, 0.1, 2.5],
                [-0.1, 0.2, 3.0],
                [0.3, -0.2, 3.2],
                [-0.3, 0.3, 3.5],
            ],
            dtype=np.float64,
        )
        projected = np.stack(
            [
                K[0, 0] * p3d[:, 0] / p3d[:, 2] + K[0, 2],
                K[1, 1] * p3d[:, 1] / p3d[:, 2] + K[1, 2],
            ],
            axis=1,
        )
        p2d = projected - 0.5

        diagnostics = sparse_correspondence_diagnostics(
            p2d,
            p3d,
            K,
            pose,
            np.arange(6),
            width=100,
            height=80,
            gt_pose_w2c=pose,
            grid_rows=2,
            grid_cols=2,
            voxel_size=0.25,
        )

        self.assertEqual(diagnostics["sparse_diag_match_count"], 6)
        self.assertEqual(diagnostics["sparse_diag_inlier_count"], 6)
        self.assertAlmostEqual(diagnostics["sparse_diag_all_gt_precision_2px"], 1.0)
        self.assertGreater(diagnostics["sparse_diag_inlier_2d_occupied_cells"], 1)
        self.assertGreater(diagnostics["sparse_diag_inlier_depth_range"], 0.0)
        self.assertIn("sparse_diag_inlier_pose_info_condition", diagnostics)
        self.assertIn(
            "sparse_diag_inlier_pose_info_translation_logdet",
            diagnostics,
        )
        self.assertGreater(
            diagnostics["sparse_diag_inlier_pose_info_translation_min_eig"],
            0.0,
        )
        self.assertGreater(
            diagnostics["sparse_diag_inlier_pose_info_translation_worst_std_m"],
            0.0,
        )
        self.assertEqual(
            diagnostics["sparse_diag_inlier_pose_info_effective_count"],
            6.0,
        )
        self.assertEqual(diagnostics["sparse_diag_gt_clean4_count"], 6)
        self.assertEqual(diagnostics["sparse_diag_inlier_gt_clean4_count"], 6)
        self.assertAlmostEqual(
            diagnostics["sparse_diag_inlier_gt_clean4_ratio"],
            1.0,
        )
        self.assertIn(
            "sparse_diag_gt_clean4_pose_info_translation_logdet",
            diagnostics,
        )
        self.assertIn(
            "sparse_diag_inlier_gt_clean4_pose_info_translation_logdet",
            diagnostics,
        )
        self.assertAlmostEqual(
            diagnostics["sparse_diag_all_gt_pose_bias_translation_norm_m"],
            0.0,
            places=8,
        )
        self.assertEqual(
            diagnostics["sparse_diag_all_gt_pose_bias_effective_count"],
            6.0,
        )

    def test_sparse_correspondence_diagnostics_uses_explicit_task_scale(self):
        import numpy as np

        from stdloc import sparse_correspondence_diagnostics

        K = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        p3d = np.array(
            [
                [-1.0, -1.0, 5.0],
                [0.0, -1.0, 5.0],
                [1.0, -1.0, 5.0],
                [-1.0, 1.0, 6.0],
                [0.0, 1.0, 6.0],
                [1.0, 1.0, 6.0],
            ]
        )
        p2d = np.stack(
            [
                100.0 * p3d[:, 0] / p3d[:, 2] + 50.0,
                100.0 * p3d[:, 1] / p3d[:, 2] + 50.0,
            ],
            axis=1,
        ) - 0.5
        diagnostics = sparse_correspondence_diagnostics(
            p2d,
            p3d,
            K,
            np.eye(4),
            np.arange(6),
            100,
            100,
            translation_task_scale_m=0.125,
            rotation_task_scale_degrees=3.0,
        )

        self.assertEqual(
            diagnostics["sparse_diag_inlier_pose_info_translation_task_scale_m"],
            0.125,
        )
        self.assertEqual(
            diagnostics[
                "sparse_diag_inlier_pose_info_rotation_task_scale_degrees"
            ],
            3.0,
        )

    def test_sparse_correspondence_diagnostics_excludes_points_behind_camera(self):
        import numpy as np

        from stdloc import sparse_correspondence_diagnostics

        K = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        p3d = np.array([[0.0, 0.0, 2.0], [1.0, 1.0, -1.0]])
        p2d = np.array([[49.5, 39.5], [10.0, 10.0]])
        diagnostics = sparse_correspondence_diagnostics(
            p2d,
            p3d,
            K,
            np.eye(4),
            np.array([0]),
            100,
            80,
            gt_pose_w2c=np.eye(4),
        )

        self.assertAlmostEqual(
            diagnostics["sparse_diag_all_gt_projected_ratio"], 0.5
        )
        self.assertLess(
            diagnostics["sparse_diag_all_gt_reproj_px_max"], 1e-6
        )
        self.assertAlmostEqual(
            diagnostics["sparse_diag_all_gt_precision_2px"], 0.5
        )

    def test_pose_bias_diagnostic_detects_systematic_reprojection_shift(self):
        import numpy as np

        from stdloc import _pose_bias_stats

        K = np.array(
            [[140.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]]
        )
        pose = np.eye(4, dtype=np.float64)
        points = np.array(
            [
                [-0.8, -0.5, 2.0],
                [0.6, -0.4, 2.4],
                [0.9, 0.7, 3.0],
                [-0.7, 0.8, 3.6],
                [0.2, -0.9, 4.2],
                [-0.3, 0.4, 5.0],
                [1.0, 0.2, 5.8],
            ],
            dtype=np.float64,
        )
        projected = np.stack(
            [
                K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
                K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
            ],
            axis=1,
        )
        shifted_observations = projected - 0.5 + np.array([1.0, 0.0])

        diagnostics = _pose_bias_stats(
            "shifted",
            shifted_observations,
            points,
            K,
            pose,
        )

        self.assertGreater(
            diagnostics["shifted_pose_bias_translation_norm_m"],
            0.0,
        )
        self.assertGreater(
            diagnostics["shifted_pose_bias_soft_inlier_count"],
            6.0,
        )

    def test_eval_translation_pose_information_matches_training_implementation(self):
        import math

        import numpy as np
        import torch

        from localization_training.pose_information import compute_pose_information
        from stdloc import _pose_information_stats

        K = np.array([[140.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.1, -0.05, 0.2]
        points = np.array(
            [
                [-0.8, -0.5, 2.0],
                [0.6, -0.4, 2.4],
                [0.9, 0.7, 3.0],
                [-0.7, 0.8, 3.6],
                [0.2, -0.9, 4.2],
                [-0.3, 0.4, 5.0],
                [1.0, 0.2, 5.8],
            ],
            dtype=np.float64,
        )

        diagnostics = _pose_information_stats(
            "eval",
            points,
            K,
            pose,
            regularization=1e-6,
            translation_task_scale_m=0.02,
            rotation_task_scale_degrees=2.0,
        )
        expected = compute_pose_information(
            torch.from_numpy(points),
            torch.from_numpy(K),
            torch.from_numpy(pose),
            damping=1e-6,
            translation_scale=0.02,
            rotation_scale=math.radians(2.0),
        )

        self.assertAlmostEqual(
            diagnostics["eval_pose_info_translation_logdet"],
            expected.translation_logdet.item(),
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["eval_pose_info_translation_condition"],
            expected.translation_condition_number.item(),
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["eval_pose_info_translation_worst_std_task"],
            expected.translation_worst_std.item(),
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["eval_pose_info_full_delete_gain_mean"],
            expected.scores.mean().item(),
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["eval_pose_info_translation_delete_gain_mean"],
            expected.translation_scores.mean().item(),
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["eval_pose_info_full_set_leverage_mean"],
            expected.full_set_leverage_scores.mean().item(),
            places=5,
        )

    def test_sparse_matchability_diagnostics_separates_coverage_and_recall(self):
        import numpy as np

        from stdloc import sparse_matchability_diagnostics

        K = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        xyz = np.array(
            [
                [0.0, 0.0, 5.0],
                [0.5, 0.0, 5.0],
                [-0.5, 0.0, 5.0],
            ]
        )
        keypoints = np.array([[49.5, 49.5], [10.0, 10.0]])
        topk = np.array([[1, 0, 2], [1, 2, 0]])
        diagnostics = sparse_matchability_diagnostics(
            keypoints,
            topk,
            xyz,
            K,
            np.eye(4),
            100,
            100,
            grid_rows=2,
            grid_cols=2,
        )
        self.assertAlmostEqual(
            diagnostics["sparse_diag_matchable_rate_2px"], 0.5
        )
        self.assertAlmostEqual(
            diagnostics[
                "sparse_diag_conditional_recall_at_1_given_matchable_2px"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            diagnostics[
                "sparse_diag_conditional_recall_at_4_given_matchable_2px"
            ],
            1.0,
        )
        self.assertEqual(
            diagnostics["sparse_diag_unmatchable_count_2px"], 1
        )


if __name__ == "__main__":
    unittest.main()
