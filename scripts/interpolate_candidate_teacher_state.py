#!/usr/bin/env python3
"""Create a normalized descriptor interpolation between two candidate-teacher states."""

import argparse
import copy
import os

import torch
import torch.nn.functional as F


def _load(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Candidate-teacher state must be a dictionary: {path}")
    for key in ("landmark_indices", "landmark_features"):
        if key not in state:
            raise ValueError(f"Missing {key!r} in {path}")
    return state


def interpolate_states(source_a, source_b, weight_b):
    state_a = _load(source_a)
    state_b = _load(source_b)
    ids_a = torch.as_tensor(state_a["landmark_indices"]).long().reshape(-1)
    ids_b = torch.as_tensor(state_b["landmark_indices"]).long().reshape(-1)
    if not torch.equal(ids_a, ids_b):
        raise ValueError("Interpolation requires identical landmark raw-ID order")
    features_a = torch.as_tensor(state_a["landmark_features"]).float()
    features_b = torch.as_tensor(state_b["landmark_features"]).float()
    if features_a.shape != features_b.shape:
        raise ValueError("Interpolation requires matching descriptor shapes")
    weight_b = float(weight_b)
    features = F.normalize(
        (1.0 - weight_b) * features_a + weight_b * features_b,
        dim=1,
    )
    output = copy.deepcopy(state_b)
    output["landmark_indices"] = ids_a.cpu()
    output["landmark_features"] = features.cpu()
    output["interpolation"] = {
        "source_a": os.path.abspath(source_a),
        "source_b": os.path.abspath(source_b),
        "weight_b": weight_b,
        "normalization": "l2_after_linear_interpolation",
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_a", required=True)
    parser.add_argument("--source_b", required=True)
    parser.add_argument("--weight_b", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = interpolate_states(args.source_a, args.source_b, args.weight_b)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state, args.output)
    print(
        f"saved {args.output}: K={state['landmark_indices'].numel()} "
        f"weight_b={state['interpolation']['weight_b']:.6g}"
    )


if __name__ == "__main__":
    main()
