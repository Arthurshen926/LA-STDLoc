from types import SimpleNamespace

import torch

from common.v6_contracts import (
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
)
from evidence.v6_observer_probes import (
    SCHEMA,
    _select_diverse_candidates,
    build_fixed_map_observer_probe_plan,
)


class _Observations:
    names = ("q0", "q1")

    def __len__(self):
        return 2

    @staticmethod
    def build_view(index):
        pose = torch.eye(4)
        pose[0, 3] = -0.2 * index
        return SimpleNamespace(
            pose_w2c=pose,
            intrinsics=torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            image_hw=(100, 100),
        )


def test_fixed_map_probe_plan_is_observer_only_and_ambiguity_targeted() -> None:
    state = {
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 2.0]]
        ),
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "query_names": ["q0", "q1"],
        "records": [
            {
                "confusion_pairs": torch.tensor([[0, 1, 2]]),
                "certified_pose_valid_alternative_pairs": torch.empty((0, 2)),
                "harmful_inlier_anchor_ids": torch.tensor([1]),
                "ambiguous_inlier_anchor_ids": torch.tensor([2]),
            },
            {},
        ],
    }
    plan = build_fixed_map_observer_probe_plan(
        state,
        _Observations(),
        feedback,
        map_sha256="a" * 64,
        observation_cache_sha256="b" * 64,
        feedback_sha256="c" * 64,
        selected_pose_budget=4,
        maximum_candidates=32,
        anchor_projection_stride=1,
        sensor_variants_per_pose=4,
    )
    assert plan["schema"] == SCHEMA
    assert plan["ambiguity_anchor_count"] == 2
    assert 0 < plan["selected_pose_count"] <= 4
    assert plan["virtual_probes_added_to_map"] is False
    assert plan["virtual_probes_added_to_anchor_observations"] is False
    assert plan["virtual_probes_increase_track_view_count"] is False
    assert plan["render_acceptance_contract"][
        "zbuffer_certification_required_before_observer_evaluation"
    ] is True
    assert all(
        record["sensor_variants"][0] == "clean"
        and len(record["sensor_variants"]) == 4
        and len(set(record["sensor_variants"])) == 4
        and record["render_status"] == "planned_not_yet_zbuffer_certified"
        for record in plan["selected_probes"]
    )


def test_probe_selection_covers_excitation_kinds_before_utility_fill() -> None:
    selected = _select_diverse_candidates(
        utility=[1.0, 0.9, 0.8, 0.7, 0.6],
        kinds=["interpolation", "interpolation", "rotation", "boundary", "reverse"],
        pose_families=torch.arange(5),
        budget=4,
    )
    assert {"interpolation", "rotation", "boundary", "reverse"} == {
        ["interpolation", "interpolation", "rotation", "boundary", "reverse"][index]
        for index in selected
    }


def test_probe_selection_repeats_rare_excitation_kinds_before_greedy_fill() -> None:
    kinds = ["interpolation"] * 6 + ["deficit"] * 3 + ["reverse"] * 3
    selected = _select_diverse_candidates(
        utility=[1.0 - 0.01 * index for index in range(len(kinds))],
        kinds=kinds,
        pose_families=torch.arange(len(kinds)),
        budget=9,
    )
    selected_kinds = [kinds[index] for index in selected]
    assert selected_kinds.count("deficit") == 3
    assert selected_kinds.count("reverse") == 3
    assert selected_kinds.count("interpolation") == 3
