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


if __name__ == "__main__":
    unittest.main()
