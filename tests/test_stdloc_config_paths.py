import unittest


class STDLocConfigPathTest(unittest.TestCase):
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

    def test_select_sparse_keypoints_by_valid_mask_refills_to_target_count(self):
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

        self.assertEqual(selected.tolist(), [0, 2, 4, 1])
        self.assertEqual(diagnostics["detected_keypoints_raw"], 6)
        self.assertEqual(diagnostics["detected_keypoints"], 4)
        self.assertEqual(diagnostics["sparse_valid_mask_selected_valid_keypoints"], 3)
        self.assertEqual(diagnostics["sparse_valid_mask_refill_keypoints"], 1)

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


if __name__ == "__main__":
    unittest.main()
