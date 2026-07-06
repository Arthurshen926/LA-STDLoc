import unittest


class STDLocConfigPathTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
