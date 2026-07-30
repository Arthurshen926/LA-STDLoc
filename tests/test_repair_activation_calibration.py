import torch
from torch import nn

from localization_training.counterfactual_repair_routing import (
    ROUTE_FAMILY,
    ROUTE_PRIMARY,
    ROUTE_REJECT,
)
from localization_training.repair_activation_calibration import (
    RepairActivationCalibrationConfig,
    calibrate_repair_route_activations,
)


class IdentityMetric(nn.Module):
    def forward(self, value):
        return value, torch.zeros_like(value)


def _teacher_record(name, positive_target):
    return {
        "query_name": name,
        "query_rows": torch.tensor([0, 1]),
        "positive_offsets": torch.tensor([0, 1, 1]),
        "positive_indices": torch.tensor([positive_target]),
        "ambiguous_offsets": torch.tensor([0, 0, 0]),
        "ambiguous_indices": torch.empty(0, dtype=torch.long),
    }


def test_loto_activation_gate_keeps_precise_mode_and_rejects_attractor():
    names = ["seq1/a.png", "seq2/b.png"]
    records = []
    for query_index, name in enumerate(names):
        records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": torch.tensor([0, 1]),
                "target_anchor_indices": torch.tensor([0, 1]),
                "route": torch.tensor(
                    [ROUTE_PRIMARY, ROUTE_FAMILY], dtype=torch.int8
                ),
                "target_representation": torch.tensor([0, 4]),
            }
        )
    audit = {
        "query_names": names,
        "records": records,
        "summary": {},
    }
    teacher = {
        "query_names": names,
        "records": [
            _teacher_record(name, 0) for name in names
        ],
        "diagnostics": {},
    }
    cache = {
        name: {
            "native_descriptors": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            )
        }
        for name in names
    }
    dynamic = {
        "query_names": names,
        "anchor_count": 2,
        "records": [
            {
                "query_name": name,
                "query_rows": torch.tensor([0, 1]),
                "top1_scores": torch.tensor([0.8, 0.8]),
            }
            for name in names
        ],
    }
    positive = {
        "query_names": names,
        "records": [
            _teacher_record(name, 0) for name in names
        ],
    }
    family = {
        "prototype_features": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        "prototype_anchor_indices": torch.tensor([0, 0, 1]),
        "prototype_bias": torch.tensor([0.0, 0.0, 0.0]),
        "prototype_temperature": torch.ones(3),
    }
    calibrated_audit, _, calibrated_family, report = (
        calibrate_repair_route_activations(
            routed_audit=audit,
            routed_teacher=teacher,
            routed_family=family,
            base_family_count=1,
            positive_teacher=positive,
            query_cache=cache,
            dynamic_outcomes=dynamic,
            metric=IdentityMetric(),
            anchor_count=2,
            config=RepairActivationCalibrationConfig(),
            device=torch.device("cpu"),
        )
    )
    routes = [
        torch.as_tensor(record["route"]).tolist()
        for record in calibrated_audit["records"]
    ]
    assert routes == [
        [ROUTE_PRIMARY, ROUTE_REJECT],
        [ROUTE_PRIMARY, ROUTE_REJECT],
    ]
    assert len(calibrated_family["prototype_features"]) == 1
    assert report["accepted_mode_count"] == 1
    assert report["rejected_mode_count"] == 1
