import numpy as np

from localization_training.seed_graph_context import (
    apply_bounded_context,
    assignment_metrics,
    build_positive_pmi_graph,
    camera_center_and_direction,
    confusion_candidate_scores,
    graph_candidate_scores,
)


def test_camera_center_and_direction_from_world_to_camera_pose():
    pose = np.eye(4)
    pose[:3, 3] = [-2.0, 0.0, 0.0]
    center, direction = camera_center_and_direction(pose)
    np.testing.assert_allclose(center, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0])


def test_positive_pmi_does_not_reward_only_popularity():
    # Landmark 0 is popular, while 1 and 2 are specifically coupled.
    incidence = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    graph = build_positive_pmi_graph(incidence, minimum_cohits=2)
    scores = graph_candidate_scores(
        graph, np.array([[0, 2]]), np.array([1])
    )
    assert scores[0, 1] > scores[0, 0]


def test_context_protects_seed_rows_and_reports_swap_quality():
    logits = np.array([[2.0, 1.0], [2.0, 1.9], [2.0, 1.9]])
    context = np.array([[0.0, 10.0], [0.0, 10.0], [10.0, 0.0]])
    selected, _ = apply_bounded_context(
        logits, context, delta_max=1.0, protected_rows=[True, False, False]
    )
    assert selected.tolist() == [0, 1, 0]
    metrics = assignment_metrics(
        np.array([[True, False], [False, True], [False, True]]),
        baseline_selected=np.array([0, 0, 0]),
        selected=selected,
        matchable_rows=np.array([True, True, True]),
    )
    assert metrics["clean_top1_retention"] == 1.0
    assert metrics["beneficial_swaps"] == 1
    assert metrics["harmful_swaps"] == 0
    assert metrics["conditional_recall_at_1_given_matchable"] == 2.0 / 3.0


def test_confusion_pmi_penalizes_seed_specific_false_attractor():
    correct = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int32
    )
    false = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int32
    )
    scores = confusion_candidate_scores(
        correct,
        false,
        candidate_ids=np.array([[1, 2]]),
        seed_ids=np.array([0]),
    )
    assert scores[0, 1] > scores[0, 0]
