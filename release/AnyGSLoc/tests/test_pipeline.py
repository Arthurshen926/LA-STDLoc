import pytest
import torch

from anygsloc.pipeline.build import construct, observation_payload
from anygsloc.pipeline.report import aggregate, metrics

torch.set_num_threads(1)


def synthetic(mode="pure_gs", count=8):
    K = torch.tensor([[700.0, 0, 320.0], [0, 710.0, 240.0], [0, 0, 1.0]])
    xyz = torch.stack(
        [
            torch.tensor([-0.603 + 0.21 * (i % 4), -0.173 + 0.25 * (i // 4), 5.123])
            for i in range(count)
        ]
    )
    features = torch.eye(256)[:count]
    records = {}
    for i, center in enumerate(
        ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0])
    ):
        pose = torch.eye(4)
        pose[:3, 3] = -torch.tensor(center)
        camera = xyz + pose[:3, 3]
        h = camera @ K.T
        r = {
            "native_keypoints": h[:, :2] / h[:, 2:3] - 0.5,
            "native_descriptors": features.clone(),
            "native_scores": torch.linspace(0.95, 0.8, count),
            "native_K": K.clone(),
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([480, 640]),
            "row_source": torch.full(
                (count,), 0 if mode == "raw_only" else 1, dtype=torch.long
            ),
            "source_row": torch.arange(count),
        }
        if mode != "raw_only":
            r.update(
                v2_valid=torch.ones(count, dtype=torch.bool),
                native_valid_keypoint_mask=torch.ones(count, dtype=torch.bool),
                native_depth_at_keypoints=camera[:, 2].clone(),
                native_alpha_at_keypoints=torch.ones(count),
            )
        records["mapping_" + str(i) + ".png"] = r
    scene = {"mapping_names": list(records), "test_names": ["test.png"]}
    return observation_payload(
        mode, records, scene["mapping_names"], scene["test_names"]
    ), xyz


@pytest.mark.parametrize("mode", ["pure_gs", "raw_only"])
@pytest.mark.parametrize("filtering", [True, False])
def test_construct_calibrated_scene(mode, filtering):
    data, truth = synthetic(mode)
    state, evidence = construct(data, filtering=filtering, device="cpu", workers=1)
    assert len(state["anchor_xyz"]) == len(truth)
    assert torch.allclose(state["anchor_xyz"], truth, atol=1e-5)
    assert state["projective_anchor_observations"]["observation_offsets"][
        -1
    ] == 4 * len(truth)
    assert evidence["v2_enabled"] is (filtering and mode != "raw_only")


def test_test_views_rejected():
    data, _ = synthetic()
    data["test_names"] = [data["mapping_names"][0]]
    with pytest.raises(ValueError, match="test camera"):
        construct(data)


def test_failure_denominator_and_scene_equal_average():
    good = {"success": True, "translation_error_cm": 1.0, "rotation_error_deg": 1.0}
    failure = {
        "success": False,
        "translation_error_cm": None,
        "rotation_error_deg": None,
    }
    a = metrics([good, failure])
    b = metrics([good] * 100)
    assert a["R5_pct"] == 50
    assert aggregate({"small": a, "large": b})["R5_pct"] == 75
    assert aggregate({"small": a, "large": b})["queries"] == 102


def test_assisted_without_eligible_proposals_preserves_real_map():
    from anygsloc.pipeline.assisted import construct as assist

    raw, _ = synthetic("raw_only")
    state, evidence = construct(raw)
    auxiliary = {
        "mapping_names": raw["mapping_names"],
        "uses_test_queries": False,
        "queries": {},
    }
    for name, record in raw["queries"].items():
        n = len(record["native_keypoints"])
        auxiliary["queries"][name] = {
            **{
                k: record[k].clone()
                for k in ["native_keypoints", "native_K", "pose_w2c", "native_input_hw"]
            },
            "depth": torch.ones(n),
            "alpha": torch.zeros(n),
            "certificate_valid": torch.zeros(n, dtype=torch.bool),
        }
    result, detail = assist(
        raw, auxiliary, evidence["association"], state, state, "test", "cpu"
    )
    assert detail["completion"] is None
    assert torch.equal(result["anchor_xyz"], state["anchor_xyz"])
    assert torch.equal(result["anchor_features"], state["anchor_features"])


def test_geometry_uses_all_observations_and_lower_coordinate_median():
    import numpy as np

    from anygsloc.pipeline.geometry import coordinate_group

    data, truth = synthetic()
    records = list(data["queries"].values())
    uv = np.stack([r["native_keypoints"][0].numpy() + 0.5 for r in records])
    K = np.stack([r["native_K"].numpy() for r in records])
    poses = np.stack([r["pose_w2c"].numpy() for r in records])
    proposals = np.array([[1, 4, 2], [2, 3, 1], [3, 2, 4], [4, 1, 3]], dtype=np.float32)
    valid, coords = coordinate_group(proposals, proposals * 2, uv, K, poses)
    assert valid.all()
    assert np.allclose(coords["ray"], truth[0].numpy(), atol=1e-5)
    assert np.array_equal(coords["center"], [2, 2, 2])
    assert np.array_equal(coords["depth"], [4, 4, 4])


def test_cleanup_preserves_unique_view_support_and_pose_additions():
    from anygsloc.mapping.post_pose_cleanup import prune_redundant_coverage

    state = {
        "provenance": {"uses_test_queries": False},
        "anchor_matchability": torch.tensor([1.0, 0.1, 0.2, 0.3]),
        "v6_mapping_query_names": ["reference.png"],
        "projective_anchor_observations": {
            "observation_offsets": torch.arange(5),
            "query_indices": torch.zeros(4, dtype=torch.long),
            "keypoint_indices": torch.tensor([0, 0, 1, 2]),
        },
    }
    reasons = [
        "precision",
        "matching_completion",
        "matching_completion",
        "observability_completion",
    ]
    trace = {
        "trace": [
            {"candidate_universe_id": i, "primary_reason": v}
            for i, v in enumerate(reasons)
        ]
    }
    rows, _ = prune_redundant_coverage(state, torch.arange(4), trace, support_target=3)
    assert rows.tolist() == [0, 2, 3]


def test_timing_subset_matches_accepted_query_schedule():
    import json
    from pathlib import Path

    from anygsloc.pipeline.benchmark import selection
    from reproduction.aggregate import load_record

    catalog = json.loads(
        (Path(__file__).parents[1] / "reproduction/catalog.json").read_text()
    )
    cell = next(
        c
        for c in catalog["localization"]
        if c["group"] == "Cambridge/full" and c["scene"] == "ShopFacade"
    )
    record = next(
        r
        for r in catalog["timing"]
        if r["case"] == "AnyGSLoc_ShopFacade_full" and r["round"] == 0
    )
    accepted = load_record(record["record"])
    names = selection("ShopFacade", cell["query_names"])
    assert names["measured_query_order"] == [r["name"] for r in accepted["queries"]]
    assert names["warmup_query_order"] == accepted["warmup_names"]
