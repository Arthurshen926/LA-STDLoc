import unittest

import numpy as np
import torch

from scripts.eval_discrete_decision_oracles import (
    _pose_term_metrics,
    nearest_gt_targets,
    oracle_assignment_candidates,
    oracle_topk_candidates,
    pair_is_correct,
    provenance_gt_targets,
    select_candidates,
)
from scripts.eval_family_aware_ransac import (
    _cell_balanced_rows,
    _distinct_family_rows,
    _proposal_rows,
)


class DiscreteDecisionOracleTest(unittest.TestCase):
    def test_pose_bias_uses_explicit_translation_task_scale(self):
        metrics = _pose_term_metrics(
            np.eye(6),
            np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            translation_scale_m=0.125,
        )
        self.assertAlmostEqual(metrics["bias_m"], 0.125)

    def test_selector_applies_landmark_quota_before_threshold(self):
        selected = select_candidates(
            [0, 1, 2],
            [5, 5, 6],
            [0.9, 0.8, 0.7],
            threshold=0.75,
            max_matches_per_landmark=1,
        )
        np.testing.assert_array_equal(selected.keypoint_idx, [0])
        np.testing.assert_array_equal(selected.landmark_idx, [5])
        np.testing.assert_array_equal(selected.source_idx, [0])

    def test_gt_hardcap_prioritizes_correct_pair(self):
        normal = select_candidates(
            [0, 1, 2],
            [5, 5, 6],
            [0.9, 0.8, 0.7],
            threshold=0.0,
            max_matches_per_landmark=1,
        )
        oracle = select_candidates(
            [0, 1, 2],
            [5, 5, 6],
            [0.9, 0.8, 0.7],
            threshold=0.0,
            max_matches_per_landmark=1,
            correctness_priority=[False, True, True],
        )
        np.testing.assert_array_equal(normal.keypoint_idx, [0, 2])
        np.testing.assert_array_equal(oracle.keypoint_idx, [1, 2])
        np.testing.assert_array_equal(oracle.source_idx, [1, 2])

    def test_nearest_visible_target_and_pair_labels(self):
        projected = np.asarray(
            [[10.0, 10.0], [12.0, 10.0], [30.0, 30.0]]
        )
        valid = np.asarray([True, False, True])
        keypoints = np.asarray(
            [[10.5, 10.0], [12.0, 10.0], [29.0, 30.0]]
        )
        targets, distance = nearest_gt_targets(
            keypoints, projected, valid, 2.0
        )
        np.testing.assert_array_equal(targets, [0, 0, 2])
        np.testing.assert_allclose(distance, [0.5, 2.0, 1.0])

        labels, pair_distance = pair_is_correct(
            keypoints, [0, 1, 2], projected, valid, 2.0
        )
        np.testing.assert_array_equal(labels, [True, False, True])
        np.testing.assert_allclose(pair_distance, [0.5, 0.0, 1.0])

    def test_oracle_assignment_keeps_only_matchable_native_keypoints(self):
        candidates = oracle_assignment_candidates(
            raw_rows=[0, 1, 2, 3],
            targets=[8, -1, 3, -1],
            scores=[0.2, 0.1, 0.4, 0.3],
        )
        np.testing.assert_array_equal(candidates.keypoint_idx, [0, 2])
        np.testing.assert_array_equal(candidates.landmark_idx, [8, 3])
        np.testing.assert_array_equal(candidates.source_idx, [0, 2])

    def test_one_of_k_oracle_emits_one_candidate_or_null_per_keypoint(self):
        candidates = oracle_topk_candidates(
            topk_landmark_idx=[[4, 8, 9], [2, 3, 5], [7, 1, 6]],
            topk_scores=[
                [0.9, 0.8, 0.7],
                [0.6, 0.5, 0.4],
                [0.3, 0.2, 0.1],
            ],
            candidate_correct=[
                [False, True, True],
                [False, False, False],
                [True, False, False],
            ],
        )
        np.testing.assert_array_equal(candidates.keypoint_idx, [0, 2])
        np.testing.assert_array_equal(candidates.landmark_idx, [8, 7])
        np.testing.assert_allclose(candidates.scores, [0.8, 0.3])

    def test_provenance_targets_require_source_and_reprojection(self):
        keypoints = np.asarray([[10.0, 10.0], [20.0, 20.0]])
        projected = np.asarray(
            [[10.5, 10.0], [20.0, 20.5], [10.1, 10.0]]
        )
        target, distance = provenance_gt_targets(
            keypoints,
            projected,
            np.ones(3, dtype=bool),
            np.asarray([[7, 9], [8, 9]]),
            np.asarray([[0.8, 0.2], [1.0, 0.0]]),
            np.asarray([True, True]),
            {7: [0], 8: [1], 9: [2]},
            2.0,
        )
        np.testing.assert_array_equal(target, [2, 1])
        np.testing.assert_allclose(distance, [0.1, 0.5])

    def test_provenance_targets_respect_anchor_class_mask(self):
        target, _ = provenance_gt_targets(
            np.asarray([[10.0, 10.0]]),
            np.asarray([[10.5, 10.0], [10.1, 10.0]]),
            np.ones(2, dtype=bool),
            np.asarray([[7]]),
            np.asarray([[1.0]]),
            np.asarray([True]),
            {7: [0, 1]},
            2.0,
            allowed_landmarks=np.asarray([True, False]),
        )
        np.testing.assert_array_equal(target, [0])

    def test_family_proposal_keeps_best_row_per_source(self):
        rows = _distinct_family_rows(
            np.asarray([5, 5, 7, 8]),
            np.asarray([0.1, 0.9, 0.8, 0.7]),
        )
        np.testing.assert_array_equal(rows, [1, 2, 3])

    def test_cell_balanced_proposal_remains_family_unique(self):
        p2d = np.asarray(
            [[5, 5], [6, 6], [95, 5], [5, 95], [95, 95]],
            dtype=float,
        )
        families = np.asarray([1, 1, 2, 3, 4])
        rows = _cell_balanced_rows(
            p2d,
            families,
            np.asarray([0.9, 0.1, 0.8, 0.7, 0.6]),
            100,
            100,
            4,
        )
        self.assertEqual(len(np.unique(families[rows])), len(rows))
        self.assertEqual(set(rows.tolist()), {0, 2, 3, 4})

    def test_dependency_proposal_uses_track_identity_not_gaussian_source(self):
        rows = _proposal_rows(
            "dependency",
            np.zeros((4, 2)),
            np.asarray([5, 5, 5, 5]),
            np.asarray([10, 10, 11, 12]),
            np.asarray([0.1, 0.9, 0.8, 0.7]),
            100,
            100,
            4,
        )
        np.testing.assert_array_equal(rows, [1, 2, 3])

    def test_first_mask_position_reports_first_and_missing(self):
        from scripts.train_lafgs_v6_conflict_features import (
            _first_mask_position,
        )

        mask = torch.tensor(
            [
                [False, True, True],
                [False, False, False],
                [True, False, True],
            ]
        )
        position, valid = _first_mask_position(mask)
        self.assertEqual(position.tolist(), [1, 3, 0])
        self.assertEqual(valid.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
