#!/usr/bin/env python3
"""Export a candidate-teacher state with a chosen adaptive-trust floor.

The training checkpoint retains the unmasked descriptor residual and the
deployment-edge evidence counts.  This tool turns that state into the exact
normalized feature bank used by inference for a specified trust floor, without
changing landmark IDs, geometry, detector, or evidence.
"""

import argparse
import copy
import os

import torch
import torch.nn.functional as F


def _load_state(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Candidate-teacher state must be a dictionary: {path}")
    for key in ("landmark_indices", "landmark_features", "adaptive_trust_state"):
        if key not in state:
            raise ValueError(f"Missing {key!r} in {path}")
    return state


def _trust_alpha(visible_count, correct_count, alpha_min, view_prior):
    visible = torch.as_tensor(visible_count, dtype=torch.float32).reshape(-1)
    correct = torch.as_tensor(correct_count, dtype=torch.float32).reshape(-1)
    if visible.shape != correct.shape:
        raise ValueError("visible and correct evidence counts must have matching shapes")
    if bool((visible < 0).any()) or bool((correct < 0).any()):
        raise ValueError("adaptive-trust evidence counts must be non-negative")
    correct = torch.minimum(correct, visible)
    reliability = correct / visible.clamp_min(1.0)
    view_support = visible / (visible + float(view_prior)).clamp_min(1e-6)
    return (reliability * view_support).clamp(min=float(alpha_min), max=1.0)


def materialize_state(source, alpha_min, view_prior=None):
    """Return an inference-ready state while retaining the resumable raw state."""
    state = _load_state(source)
    if not 0.0 <= float(alpha_min) <= 1.0:
        raise ValueError("alpha_min must be in [0, 1]")
    trust = state["adaptive_trust_state"]
    required = (
        "initial_features",
        "raw_features",
        "visible_count",
        "correct_count",
    )
    missing = [key for key in required if key not in trust]
    if missing:
        raise ValueError("Missing adaptive-trust fields: " + ", ".join(missing))
    if view_prior is None:
        view_prior = state.get("config", {}).get("trust_view_prior", 3.0)
    if float(view_prior) < 0.0:
        raise ValueError("view_prior must be non-negative")

    initial = F.normalize(
        torch.as_tensor(trust["initial_features"], dtype=torch.float32), dim=1
    )
    raw = torch.as_tensor(trust["raw_features"], dtype=torch.float32)
    if initial.shape != raw.shape:
        raise ValueError("initial and raw descriptor tensors must have matching shapes")
    alpha = _trust_alpha(
        trust["visible_count"], trust["correct_count"], alpha_min, view_prior
    )
    if alpha.numel() != initial.shape[0]:
        raise ValueError("adaptive-trust evidence count does not match landmark count")
    features = F.normalize(initial + alpha[:, None] * (raw - initial), dim=1)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("materialized landmark features contain non-finite values")

    output = copy.deepcopy(state)
    output["landmark_features"] = features.cpu()
    output["trust_materialization"] = {
        "source": os.path.abspath(source),
        "alpha_min": float(alpha_min),
        "view_prior": float(view_prior),
        "alpha_min_observed": float(alpha.min()),
        "alpha_median": float(alpha.median()),
        "alpha_max_observed": float(alpha.max()),
        "normalization": "l2_after_trust_weighted_residual",
    }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Materialize a candidate-teacher adaptive-trust checkpoint."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha_min", required=True, type=float)
    parser.add_argument("--view_prior", type=float, default=None)
    args = parser.parse_args()
    state = materialize_state(args.source, args.alpha_min, args.view_prior)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state, args.output)
    metadata = state["trust_materialization"]
    print(
        f"saved {args.output}: K={state['landmark_indices'].numel()} "
        f"alpha=[{metadata['alpha_min_observed']:.6g}, "
        f"{metadata['alpha_median']:.6g}, {metadata['alpha_max_observed']:.6g}]"
    )


if __name__ == "__main__":
    main()
