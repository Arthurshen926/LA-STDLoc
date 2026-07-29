#!/usr/bin/env python3
"""Audit rendered descriptor separability for targeted confusion pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.confusion_evidence import (
    synthetic_separability_oracle,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--contrastive-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    evidence = torch.load(
        args.contrastive_evidence, map_location="cpu", weights_only=False
    )
    output = synthetic_separability_oracle(
        state=state,
        metric=metric,
        family=family,
        evidence=evidence,
        device=torch.device(args.device),
    )
    output["provenance"] = {
        key: str(Path(value).resolve())
        for key, value in vars(args).items()
        if key not in {"output", "device"}
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "edges"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
