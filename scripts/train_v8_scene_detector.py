#!/usr/bin/env python3
"""Train only the lightweight V8 detector head on Gaussian renders."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from features.scene_specific_detector import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_VERSION,
    SceneSpecificDetector,
    detector_metrics,
    tri_state_detector_loss,
)
from features.superpoint import SuperPoint


def _record_image(record: dict) -> torch.Tensor:
    if "rgb_u8" in record:
        return torch.as_tensor(record["rgb_u8"]).float() / 255
    source = torch.load(record["source_record"], map_location="cpu", weights_only=False)
    return torch.as_tensor(source["rgb_float16"]).float()


def _load_records(root: Path, split: str) -> list[Path]:
    records = sorted(root.glob(f"{split}_*.pt"))
    if not records:
        raise FileNotFoundError(f"no {split} detector records under {root}")
    return records


def _augment(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    def uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand((), generator=generator))
    value = image
    value = value * uniform(0.65, 1.45)
    balance = torch.tensor(
        [uniform(0.85, 1.15), uniform(0.85, 1.15), uniform(0.85, 1.15)],
        device=value.device,
    )[:, None, None]
    value = (value * balance).clamp(0, 1).pow(uniform(0.75, 1.35))
    if float(torch.rand((), generator=generator)) < 0.5:
        value = F.avg_pool2d(value[None], 3, stride=1, padding=1)[0]
    noise = torch.randn(value.shape, generator=generator, device="cpu").to(value.device)
    return (value + noise * uniform(0.0, 0.025)).clamp(0, 1)


@torch.inference_mode()
def _evaluate(
    encoder: SuperPoint, head: SceneSpecificDetector, paths: list[Path], device
) -> dict[str, float]:
    metrics = []
    losses = []
    head.eval()
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        image = _record_image(record).to(device=device, dtype=torch.float32)[None]
        labels = record["labels"].to(device)
        contribution = record.get("contribution_weights")
        contribution = None if contribution is None else contribution.to(device)
        dense, _ = encoder._dense_outputs(image)
        logits = head(dense)[0]
        losses.append(float(tri_state_detector_loss(logits, labels, sample_weight=contribution)))
        metrics.append(detector_metrics(logits, labels))
    def finite_mean(key: str) -> float:
        values = [row[key] for row in metrics if math.isfinite(row[key])]
        return sum(values) / len(values) if values else float("nan")
    return {
        "loss": sum(losses) / len(losses),
        "positive_mean": finite_mean("positive_mean"),
        "negative_mean": finite_mean("negative_mean"),
        "separation": finite_mean("separation"),
        "fully_supervised_view_count": sum(math.isfinite(row["separation"]) for row in metrics),
    }


def _atomic_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    device = torch.device(args.device)
    train_paths = _load_records(args.dataset_root, "train")
    validation_paths = _load_records(args.dataset_root, "validation")
    manifest_paths = sorted(args.dataset_root.glob("manifest_*.json"))
    if (args.dataset_root / "manifest.json").exists():
        manifest_paths.append(args.dataset_root / "manifest.json")
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    if not manifests:
        raise FileNotFoundError("scene-detector dataset manifest is missing")
    map_hashes = {row["anchor_map_sha256"] for row in manifests}
    if len(map_hashes) != 1 or any(row.get("uses_test_rgb") is not False for row in manifests):
        raise ValueError("detector data lineage is inconsistent or test-contaminated")

    encoder = SuperPoint().to(device).eval()
    encoder.requires_grad_(False)
    head = SceneSpecificDetector(hidden_dim=args.hidden_dim).to(device)
    initialization = None
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("schema") != CHECKPOINT_SCHEMA or initial.get("lineage", {}).get("uses_test_rgb") is not False:
            raise ValueError("detector initialization violates synthetic-only lineage")
        if int(initial["model"]["hidden_dim"]) != args.hidden_dim:
            raise ValueError("detector initialization architecture differs")
        head.load_state_dict(initial["state_dict"], strict=True)
        initialization = str(args.initial_checkpoint.resolve())
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_paths), generator=generator).tolist()
        head.train()
        losses = []
        for index in order:
            record = torch.load(train_paths[index], map_location="cpu", weights_only=False)
            image = _record_image(record).to(device=device, dtype=torch.float32)
            image = _augment(image, generator)[None]
            labels = record["labels"].to(device)
            contribution = record.get("contribution_weights")
            contribution = None if contribution is None else contribution.to(device)
            with torch.no_grad():
                dense, _ = encoder._dense_outputs(image)
            logits = head(dense)[0]
            loss = tri_state_detector_loss(logits, labels, sample_weight=contribution)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate(encoder, head, validation_paths, device)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **validation}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or validation["loss"] < best["loss"]:
            best = {**validation, "epoch": epoch + 1, "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}}
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "version": CHECKPOINT_VERSION,
        "model": {"feature_dim": 256, "hidden_dim": args.hidden_dim},
        "state_dict": best.pop("state_dict"), "best_validation": best,
        "history": history,
        "lineage": {
            "anchor_map_sha256": next(iter(map_hashes)),
            "uses_source_mapping_rgb": False, "uses_real_training_rgb": False,
            "uses_test_rgb": False, "descriptor_adapter": False,
            "map_conditioned_second_pass": False,
            "feedback_match_supervision": any(
                row.get("schema") in {
                    "lafgs_v10_feedback_scene_detector_dataset_manifest",
                    "lafgs_v11_pose_contribution_detector_dataset_manifest",
                }
                for row in manifests
            ),
            "pose_contribution_weighting": any(
                row.get("schema") == "lafgs_v11_pose_contribution_detector_dataset_manifest"
                for row in manifests
            ),
            "initial_checkpoint": initialization,
        },
    }
    _atomic_save(checkpoint, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({k: v for k, v in checkpoint.items() if k not in {"state_dict"}}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
