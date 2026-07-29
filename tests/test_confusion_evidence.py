import torch
import torch.nn.functional as F

from localization_training.confusion_evidence import (
    ConfusionGraphConfig,
    ConfusionViewPlanningConfig,
    ContrastiveEvidenceConfig,
    build_anchor_family_confusion_graph,
    build_contrastive_synthetic_record,
    pack_contrastive_synthetic_evidence,
    plan_confusion_conditioned_views,
    synthetic_separability_oracle,
)
from localization_training.synthetic_evidence import SyntheticEvidenceConfig


class _IdentityMetric(torch.nn.Module):
    def forward(self, value):
        return value, value.new_zeros(())


def _family(anchor_count):
    return {
        "landmark_indices": torch.arange(anchor_count),
        "prototype_features": torch.empty((0, 2)),
        "prototype_anchor_indices": torch.empty(0, dtype=torch.long),
        "prototype_bias": torch.empty(0),
        "prototype_temperature": torch.empty(0),
        "families": [],
    }


def test_confusion_graph_records_directed_wrong_family_assignment():
    state = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        "anchor_features": F.normalize(
            torch.tensor([[0.8, 0.6], [1.0, 0.0]]), dim=1
        ),
        "source_primitive_ids": torch.tensor([10, 20]),
        "dependency_group_ids": torch.tensor([100, 200]),
    }
    dynamic = {
        "anchor_count": 2,
        "query_names": ["seq1/frame00001.png"],
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "top1_anchor_indices": torch.tensor([1]),
                "top1_scores": torch.tensor([1.0]),
                "gt_reprojection_errors_px": torch.tensor([15.0]),
                "ransac_inlier_mask": torch.tensor([True]),
                "te_cm": 20.0,
            }
        ],
    }
    positives = {
        "anchor_count": 2,
        "query_names": dynamic["query_names"],
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([0]),
            }
        ],
    }
    cache = {
        "seq1/frame00001.png": {
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_keypoints": torch.tensor([[5.0, 5.0]]),
            "native_input_hw": [20, 30],
        }
    }
    graph = build_anchor_family_confusion_graph(
        state=state,
        metric=_IdentityMetric(),
        family=_family(2),
        dynamic=dynamic,
        positives=positives,
        cache=cache,
        query_bins={"seq1/frame00001.png": 3},
        config=ConfusionGraphConfig(minimum_occurrences=1),
        device=torch.device("cpu"),
    )
    assert graph["summary"]["retained_directed_edge_count"] == 1
    assert graph["edges"][0]["correct_anchor"] == 0
    assert graph["edges"][0]["confusing_anchor"] == 1
    assert graph["edges"][0]["harmful_ransac_survivors"] == 1
    assert graph["events"][0]["image_cell"] >= 0


def test_confusion_graph_does_not_mine_gt_clean_teacher_misses():
    state = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        "anchor_features": F.normalize(
            torch.tensor([[0.8, 0.6], [1.0, 0.0]]), dim=1
        ),
        "source_primitive_ids": torch.tensor([10, 20]),
        "dependency_group_ids": torch.tensor([100, 200]),
    }
    dynamic = {
        "anchor_count": 2,
        "query_names": ["seq1/frame00001.png"],
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "top1_anchor_indices": torch.tensor([1]),
                "top1_scores": torch.tensor([1.0]),
                "gt_reprojection_errors_px": torch.tensor([2.0]),
                "ransac_inlier_mask": torch.tensor([True]),
                "te_cm": 3.0,
            }
        ],
    }
    positives = {
        "anchor_count": 2,
        "query_names": dynamic["query_names"],
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([0]),
            }
        ],
    }
    graph = build_anchor_family_confusion_graph(
        state=state,
        metric=_IdentityMetric(),
        family=_family(2),
        dynamic=dynamic,
        positives=positives,
        cache={
            "seq1/frame00001.png": {
                "native_descriptors": torch.tensor([[1.0, 0.0]]),
                "native_keypoints": torch.tensor([[5.0, 5.0]]),
                "native_input_hw": [20, 30],
            }
        },
        query_bins={"seq1/frame00001.png": 3},
        config=ConfusionGraphConfig(minimum_occurrences=1),
        device=torch.device("cpu"),
    )
    assert graph["summary"]["retained_directed_edge_count"] == 0


def test_contrastive_render_labels_strong_ambiguous_and_graph_negative():
    state = {
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 2.0], [0.06, 0.0, 2.0], [0.2, 0.0, 2.0]]
        ),
        "anchor_features": F.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.1]]), dim=1
        ),
        "source_primitive_ids": torch.tensor([10, 11, 20]),
        "dependency_group_ids": torch.tensor([100, 101, 200]),
    }
    record = {
        "query_name": "synthetic:0",
        "source_query": "seq1/frame00001.png",
        "accepted": True,
        "pose_w2c": torch.eye(4),
        "native_K": torch.tensor(
            [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]
        ),
        "native_input_hw": [20, 30],
        "native_keypoints": torch.tensor([[5.0, 5.0]]),
        "native_descriptors": torch.tensor([[1.0, 0.0]]),
        "native_scores": torch.tensor([1.0]),
        "query_rows": torch.tensor([0]),
        "positive_offsets": torch.tensor([0, 1]),
        "positive_indices": torch.tensor([0]),
        "positive_pair_count": 1,
        "config": {
            **SyntheticEvidenceConfig().__dict__,
            "absolute_depth_tolerance": 0.1,
            "relative_depth_tolerance": 0.0,
            "require_support_mask": True,
        },
    }
    graph = {
        "anchor_count": 3,
        "edges": [
            {
                "correct_anchor": 0,
                "confusing_anchor": 2,
                "occurrences": 5,
                "weight": 10.0,
            }
        ],
    }
    output = build_contrastive_synthetic_record(
        record=record,
        state=state,
        metric=_IdentityMetric(),
        family=_family(3),
        confusion_graph=graph,
        rendered_depth=torch.full((1, 20, 30), 2.0),
        alpha=torch.ones(1, 20, 30),
        visibility_config=SyntheticEvidenceConfig(
            absolute_depth_tolerance=0.1,
            relative_depth_tolerance=0.0,
        ),
        config=ContrastiveEvidenceConfig(
            minimum_hard_negative_pairs=1,
            minimum_edge_occurrences=1,
        ),
        device=torch.device("cpu"),
    )
    assert output["positive_indices"].tolist() == [0]
    assert output["ambiguous_indices"].tolist() == [1]
    assert output["hard_negative_indices"].tolist() == [2]
    assert output["hard_negative_positive_indices"].tolist() == [0]
    assert output["contrastive_accepted"]


def test_confusion_view_planner_targets_edges_inside_pose_envelope():
    state = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    }
    names = [
        "seq1/frame00001.png",
        "seq1/frame00002.png",
        "seq2/frame00001.png",
    ]

    def pose(center_x):
        value = torch.eye(4)
        value[0, 3] = -float(center_x)
        return value

    cache = {
        name: {
            "pose_w2c": pose(center),
            "native_input_hw": [120, 200],
            "native_K": torch.tensor(
                [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
            ),
        }
        for name, center in zip(names, [0.0, 0.2, 0.4])
    }
    graph = {
        "anchor_count": 2,
        "query_names": names,
        "edges": [
            {
                "edge_index": 0,
                "correct_anchor": 0,
                "confusing_anchor": 1,
                "occurrences": 8,
                "trajectory_count": 2,
                "weight": 20.0,
            }
        ],
        "events": [
            {
                "edge_index": 0,
                "query_name": names[0],
                "query_row": 0,
                "image_cell": 1,
                "pose_blame": 2.0,
                "score_margin": 0.1,
            }
        ],
    }
    planned = plan_confusion_conditioned_views(
        confusion_graph=graph,
        state=state,
        cache=cache,
        query_bins={names[0]: 0, names[1]: 0, names[2]: 1},
        config=ConfusionViewPlanningConfig(
            maximum_planned_views=2,
            maximum_pose_neighbors=2,
            maximum_views_per_edge=2,
            maximum_views_per_source=2,
            minimum_edge_occurrences=1,
            interpolation_alphas=(0.5,),
        ),
    )
    assert planned
    assert all(record["edge_index"] == 0 for record in planned)
    assert all(record["correct_family_visible"] for record in planned)
    assert any(record["cross_trajectory"] for record in planned)


def test_contrastive_render_respects_planned_confusion_edge():
    state = {
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.4, 0.0, 2.0]]
        ),
        "anchor_features": F.normalize(
            torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.98, 0.2]]), dim=1
        ),
        "source_primitive_ids": torch.tensor([10, 20, 30]),
        "dependency_group_ids": torch.tensor([100, 200, 300]),
    }
    record = {
        "query_name": "synthetic:targeted",
        "source_query": "seq1/frame00001.png",
        "accepted": True,
        "pose_w2c": torch.eye(4),
        "native_K": torch.tensor(
            [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]
        ),
        "native_input_hw": [20, 30],
        "native_keypoints": torch.tensor([[5.0, 5.0]]),
        "native_descriptors": torch.tensor([[1.0, 0.0]]),
        "native_scores": torch.tensor([1.0]),
        "query_rows": torch.tensor([0]),
        "positive_offsets": torch.tensor([0, 1]),
        "positive_indices": torch.tensor([0]),
        "positive_pair_count": 1,
        "config": {
            **SyntheticEvidenceConfig().__dict__,
            "absolute_depth_tolerance": 0.1,
            "relative_depth_tolerance": 0.0,
            "require_support_mask": True,
        },
        "active_evidence_target": {
            "edge_index": 0,
            "correct_anchor": 0,
            "confusing_anchor": 1,
        },
    }
    graph = {
        "anchor_count": 3,
        "edges": [
            {
                "edge_index": 0,
                "correct_anchor": 0,
                "confusing_anchor": 1,
                "occurrences": 5,
                "weight": 10.0,
            },
            {
                "edge_index": 1,
                "correct_anchor": 0,
                "confusing_anchor": 2,
                "occurrences": 10,
                "weight": 20.0,
            },
        ],
    }
    output = build_contrastive_synthetic_record(
        record=record,
        state=state,
        metric=_IdentityMetric(),
        family=_family(3),
        confusion_graph=graph,
        rendered_depth=torch.full((1, 20, 30), 2.0),
        alpha=torch.ones(1, 20, 30),
        visibility_config=SyntheticEvidenceConfig(
            absolute_depth_tolerance=0.1,
            relative_depth_tolerance=0.0,
        ),
        config=ContrastiveEvidenceConfig(
            minimum_hard_negative_pairs=1,
            minimum_edge_occurrences=1,
        ),
        device=torch.device("cpu"),
    )
    assert output["hard_negative_indices"].tolist() == [1]


def test_contrastive_pack_can_preserve_positive_only_r3_records():
    base = {
        "query_name": "synthetic:positive-only",
        "accepted": True,
        "strong_positive_pair_count": 2,
        "ambiguous_pair_count": 1,
        "hard_negative_pair_count": 0,
        "hard_negative_positive_indices": torch.empty(0, dtype=torch.long),
        "hard_negative_indices": torch.empty(0, dtype=torch.long),
        "contrastive_accepted": False,
    }
    graph = {"anchor_count": 3, "summary": {}}
    packed = pack_contrastive_synthetic_evidence(
        [base],
        source={},
        confusion_graph=graph,
        include_positive_only_records=True,
    )
    assert len(packed["records"]) == 1
    assert packed["summary"]["positive_view_count"] == 1
    assert packed["summary"]["contrastive_view_count"] == 0


def test_synthetic_separability_oracle_reports_best_pair_margin():
    state = {
        "anchor_features": F.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
        )
    }
    record = {
        "native_descriptors": torch.tensor([[1.0, 0.0]]),
        "hard_negative_pair_count": 1,
        "hard_negative_offsets": torch.tensor([0, 1]),
        "hard_negative_positive_indices": torch.tensor([0]),
        "hard_negative_indices": torch.tensor([1]),
    }
    output = synthetic_separability_oracle(
        state=state,
        metric=_IdentityMetric(),
        family=_family(2),
        evidence={
            "schema": "lafgs_confusion_contrastive_synthetic_evidence",
            "records": [record],
        },
        device=torch.device("cpu"),
    )
    assert output["edge_count"] == 1
    assert output["separable_edge_fraction"] == 1.0
    assert output["edges"][0]["maximum_margin"] > 0.9
