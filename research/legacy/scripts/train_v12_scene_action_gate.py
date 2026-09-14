#!/usr/bin/env python3
"""Train a tiny query-only gate from exact paired detector action gains."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from features.scene_action_gate import FEATURE_NAMES, SceneActionGate, feature_tensor


def _atomic_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=120260828)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    payloads = [json.loads(path.read_text()) for path in args.paired_shards]
    if any(
        row.get("split") != "train"
        or row.get("loo_used") is not False
        or row.get("uses_test_queries") is not False
        for row in payloads
    ):
        raise ValueError("V12 action gate requires non-test train paired shards")
    identities = [row["input"] for row in payloads]
    identity_keys = ("anchor_map_sha256", "checkpoint_sha256", "dataset_manifest_sha256")
    if any(
        any(identity[key] != identities[0][key] for key in identity_keys)
        for identity in identities[1:]
    ):
        raise ValueError("V12 action gate paired shard lineages differ")
    rows = [record for payload in payloads for record in payload["paired_rows"]]
    if len({int(row["query_index"]) for row in rows}) != len(rows):
        raise ValueError("V12 action gate training queries overlap")
    x = torch.stack([feature_tensor(row) for row in rows])
    gain = torch.tensor([
        float(row["native"]["task_error"] - row["strength_1.0000"]["task_error"])
        for row in rows
    ])
    baseline_success = torch.tensor([
        row["native"]["translation_error_cm"] < 5
        and row["native"]["rotation_error_deg"] < 5 for row in rows
    ])
    proposal_success = torch.tensor([
        row["strength_1.0000"]["translation_error_cm"] < 5
        and row["strength_1.0000"]["rotation_error_deg"] < 5 for row in rows
    ])
    label = (gain >= 0.001) & (~baseline_success | proposal_success)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    model = SceneActionGate(mean, std)
    positives = label.float().sum()
    negatives = label.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    history = []
    for epoch in range(args.epochs):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(
            logits, label.float(), pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 100 == 0:
            prediction = logits >= 0
            history.append({
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "accuracy": float((prediction == label).float().mean()),
                "activation_fraction": float(prediction.float().mean()),
            })
    checkpoint = {
        "schema": "lafgs_v12_scene_action_gate",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean,
        "feature_std": std,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "map_sha256": identities[0]["anchor_map_sha256"],
        "detector_sha256": identities[0]["checkpoint_sha256"],
        "dataset_manifest_sha256": identities[0]["dataset_manifest_sha256"],
        "training_query_count": len(rows),
        "positive_query_count": int(label.sum()),
        "target": "paired_task_gain_ge_0.001_without_success_to_failure_flip",
        "decision_threshold_logit": 0.0,
        "history": history,
        "input_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.paired_shards
        ],
    }
    _atomic_save(checkpoint, args.output)
    report = {key: value for key, value in checkpoint.items() if key not in {"state_dict", "feature_mean", "feature_std"}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
