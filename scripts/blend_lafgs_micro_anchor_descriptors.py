#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-map", required=True)
    parser.add_argument("--trained-map", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    initial_path = Path(args.initial_map).resolve()
    trained_path = Path(args.trained_map).resolve()
    output_path = Path(args.output).resolve()
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    for state, name in ((initial, "initial"), (trained, "trained")):
        if state.get("schema") != "lafgs_materialized_anchor_map":
            raise ValueError(f"{name} input is not a materialized anchor map")
    for key in (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
    ):
        if not torch.equal(torch.as_tensor(initial[key]), torch.as_tensor(trained[key])):
            raise ValueError(f"anchor alignment mismatch for {key}")
    base_count = int(initial["base_anchor_count"])
    initial_features = F.normalize(initial["anchor_features"].float(), dim=1)
    trained_features = F.normalize(trained["anchor_features"].float(), dim=1)
    if not torch.allclose(
        initial_features[:base_count],
        trained_features[:base_count],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("protected training changed a frozen base descriptor")
    output_features = initial_features.clone()
    output_features[base_count:] = F.normalize(
        (1.0 - args.alpha) * initial_features[base_count:]
        + args.alpha * trained_features[base_count:],
        dim=1,
    )
    output = dict(initial)
    output["anchor_features"] = output_features
    output["descriptor_training"] = {
        "mode": "protected_add_only_bounded_interpolation_v1",
        "alpha": float(args.alpha),
        "initial_map_path": str(initial_path),
        "initial_map_sha256": _sha256(initial_path),
        "trained_map_path": str(trained_path),
        "trained_map_sha256": _sha256(trained_path),
        "old_anchor_descriptors_frozen": True,
        "all_anchor_geometry_frozen": True,
        "new_descriptor_cosine_to_initial_mean": float(
            (
                output_features[base_count:]
                * initial_features[base_count:]
            )
            .sum(dim=1)
            .mean()
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
