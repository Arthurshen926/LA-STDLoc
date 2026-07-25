#!/usr/bin/env python
"""Create a normalized trust-region interpolation of two LaFGS map states."""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    reference = torch.load(args.reference, map_location="cpu")
    candidate = torch.load(args.candidate, map_location="cpu")
    if not torch.equal(
        reference["landmark_indices"], candidate["landmark_indices"]
    ):
        raise ValueError("state landmark IDs differ")
    if reference["landmark_features"].shape != candidate["landmark_features"].shape:
        raise ValueError("state descriptor shapes differ")
    if not torch.allclose(
        reference["landmark_xyz"], candidate["landmark_xyz"], atol=1e-7, rtol=0.0
    ):
        raise ValueError("trust-region interpolation requires fixed geometry")

    output = copy.deepcopy(candidate)
    reference_features = F.normalize(reference["landmark_features"].float(), dim=1)
    candidate_features = F.normalize(candidate["landmark_features"].float(), dim=1)
    output["landmark_features"] = F.normalize(
        (1.0 - args.alpha) * reference_features
        + args.alpha * candidate_features,
        dim=1,
    )
    output["iteration"] = int(round(1000.0 * args.alpha))
    output["config"] = copy.deepcopy(candidate.get("config", {}))
    output["config"]["descriptor_trust_region"] = {
        "schema_version": 1,
        "reference": str(Path(args.reference).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "alpha": float(args.alpha),
        "geometry_fixed": True,
        "interpolation": "normalized_linear",
    }
    output["diagnostics"] = copy.deepcopy(candidate.get("diagnostics", {}))
    output["diagnostics"]["descriptor_trust_region_alpha"] = float(args.alpha)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    print(path)


if __name__ == "__main__":
    main()
