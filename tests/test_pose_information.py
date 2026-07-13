import unittest

import torch


class PoseInformationTest(unittest.TestCase):
    def test_information_scores_are_positive_and_report_condition(self):
        from localization_training.pose_information import (
            compute_pose_information,
            pose_jacobian_analytic,
            pose_jacobian_numeric,
        )

        dtype = torch.float64
        points = torch.tensor(
            [
                [-1.0, -0.5, 5.0],
                [1.0, -0.5, 5.0],
                [-1.0, 0.5, 5.0],
                [1.0, 0.5, 5.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=dtype,
        )
        K = torch.tensor(
            [[180.0, 0.0, 80.0], [0.0, 180.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        pose = torch.eye(4, dtype=dtype)

        J = pose_jacobian_numeric(points, K, pose)
        analytic = pose_jacobian_analytic(points, K, pose)
        self.assertEqual(J.shape, (5, 2, 6))
        self.assertTrue(torch.isfinite(J).all())
        self.assertTrue(torch.allclose(analytic, J, rtol=2e-3, atol=2e-3))

        info = compute_pose_information(points, K, pose, weights=torch.ones(5, dtype=dtype))
        self.assertEqual(info.scores.shape, (5,))
        self.assertTrue((info.scores > 0).all())
        self.assertTrue(torch.isfinite(info.condition_number))
        self.assertGreater(info.logdet.item(), 0.0)
        self.assertEqual(info.translation_scores.shape, (5,))
        self.assertTrue((info.translation_scores >= 0).all())
        self.assertTrue(torch.isfinite(info.translation_worst_std))

    def test_analytic_jacobian_matches_numeric_for_nontrivial_poses(self):
        from localization_training.pose_information import (
            pose_jacobian_analytic,
            pose_jacobian_numeric,
        )
        from localization_training.pose_refiner import se3_exp

        dtype = torch.float64
        points = torch.tensor(
            [
                [-1.2, -0.4, 3.5],
                [0.2, 0.7, 6.0],
                [2.0, -1.0, 12.0],
                [-0.3, 1.4, 25.0],
            ],
            dtype=dtype,
        )
        intrinsics = [
            torch.tensor(
                [[180.0, 0.0, 80.0], [0.0, 220.0, 60.0], [0.0, 0.0, 1.0]],
                dtype=dtype,
            ),
            torch.tensor(
                [[820.0, 0.0, 320.0], [0.0, 760.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=dtype,
            ),
        ]
        poses = [
            torch.eye(4, dtype=dtype),
            se3_exp(torch.tensor([0.2, -0.1, 0.3, 0.03, -0.04, 0.02], dtype=dtype)),
        ]

        for K in intrinsics:
            for pose in poses:
                analytic = pose_jacobian_analytic(points, K, pose)
                numeric = pose_jacobian_numeric(points, K, pose, eps=1e-5)
                maximum_error = (analytic - numeric).abs().max()
                relative_error = torch.linalg.norm(analytic - numeric) / torch.linalg.norm(numeric)
                self.assertLess(maximum_error.item(), 2e-4)
                self.assertLess(relative_error.item(), 2e-6)

    def test_conditional_add_and_delete_match_brute_force_logdet(self):
        from localization_training.pose_information import (
            conditional_add_gain,
            conditional_delete_loss,
            fisher_contributions,
            translation_schur_complement,
        )

        torch.manual_seed(7)
        dtype = torch.float64
        jacobian = torch.randn(8, 2, 6, dtype=dtype)
        weights = torch.linspace(0.2, 1.0, 8, dtype=dtype)
        covariance = torch.stack(
            [torch.diag(torch.tensor([1.0 + i * 0.1, 1.5 + i * 0.2], dtype=dtype)) for i in range(8)]
        )
        contributions = fisher_contributions(
            jacobian,
            weights=weights,
            measurement_covariance=covariance,
        )
        prior = torch.eye(6, dtype=dtype) * 0.25
        full = prior + contributions.sum(dim=0)

        for index in range(contributions.shape[0]):
            contribution = contributions[index]
            without = full - contribution
            expected_full = torch.linalg.slogdet(full).logabsdet - torch.linalg.slogdet(without).logabsdet
            add_gain = conditional_add_gain(without, contribution, objective="full")
            delete_loss = conditional_delete_loss(full, contribution, objective="full")
            self.assertTrue(torch.allclose(add_gain, expected_full, atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(delete_loss, expected_full, atol=1e-10, rtol=1e-10))

            full_translation = translation_schur_complement(full, eps=1e-12)
            without_translation = translation_schur_complement(without, eps=1e-12)
            expected_translation = (
                torch.linalg.slogdet(full_translation).logabsdet
                - torch.linalg.slogdet(without_translation).logabsdet
            )
            add_translation = conditional_add_gain(
                without, contribution, objective="translation"
            )
            delete_translation = conditional_delete_loss(
                full, contribution, objective="translation"
            )
            self.assertTrue(
                torch.allclose(add_translation, expected_translation, atol=1e-10, rtol=1e-10)
            )
            self.assertTrue(
                torch.allclose(delete_translation, expected_translation, atol=1e-10, rtol=1e-10)
            )

    def test_compute_pose_information_reports_exact_leave_one_out_scores(self):
        from localization_training.pose_information import compute_pose_information

        dtype = torch.float64
        points = torch.tensor(
            [
                [-1.0, -0.5, 4.0],
                [1.0, -0.5, 4.5],
                [-1.2, 0.8, 7.0],
                [1.5, 1.0, 9.0],
                [0.0, 0.0, 5.5],
                [0.4, -1.2, 12.0],
            ],
            dtype=dtype,
        )
        K = torch.tensor(
            [[400.0, 0.0, 160.0], [0.0, 420.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        info = compute_pose_information(
            points,
            K,
            torch.eye(4, dtype=dtype),
            weights=torch.tensor([1.0, 0.8, 0.7, 0.6, 0.9, 0.5], dtype=dtype),
            measurement_covariance=torch.full((6,), 2.25, dtype=dtype),
            damping=0.1,
            translation_scale=0.02,
            rotation_scale=torch.deg2rad(torch.tensor(2.0, dtype=dtype)).item(),
        )

        expected = []
        for contribution in info.contributions:
            without = info.matrix - contribution
            expected.append(
                torch.linalg.slogdet(info.matrix).logabsdet
                - torch.linalg.slogdet(without).logabsdet
            )
        expected = torch.stack(expected)
        self.assertTrue(torch.allclose(info.scores, expected, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(info.effective_count.item(), 4.5 ** 2 / 3.55, places=10)


if __name__ == "__main__":
    unittest.main()
