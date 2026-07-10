import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_candidate_states(base_state, tuned_state, alpha):
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    required = ("landmark_indices", "landmark_features")
    for name in required:
        if name not in base_state or name not in tuned_state:
            raise KeyError(f"both states must contain {name!r}")
    if not torch.equal(base_state["landmark_indices"], tuned_state["landmark_indices"]):
        raise ValueError("candidate states use different landmark indices")
    base_features = base_state["landmark_features"].detach().float().cpu()
    tuned_features = tuned_state["landmark_features"].detach().float().cpu()
    if base_features.shape != tuned_features.shape:
        raise ValueError(
            "candidate states use different feature shapes: "
            f"{tuple(base_features.shape)} vs {tuple(tuned_features.shape)}"
        )

    base_features = F.normalize(base_features, dim=1)
    tuned_features = F.normalize(tuned_features, dim=1)
    interpolated = F.normalize(
        (1.0 - alpha) * base_features + alpha * tuned_features,
        dim=1,
    )
    delta = torch.linalg.norm(interpolated - base_features, dim=1)
    tuned_delta = torch.linalg.norm(tuned_features - base_features, dim=1)
    output = copy.deepcopy(base_state)
    output["landmark_features"] = interpolated
    diagnostics = dict(output.get("diagnostics", {}))
    diagnostics["feature_interpolation"] = {
        "alpha": alpha,
        "drift_from_base_mean": float(delta.mean().item()),
        "drift_from_base_p95": float(torch.quantile(delta, 0.95).item()),
        "drift_from_base_max": float(delta.max().item()),
        "tuned_drift_from_base_mean": float(tuned_delta.mean().item()),
        "tuned_drift_from_base_p95": float(torch.quantile(tuned_delta, 0.95).item()),
        "tuned_drift_from_base_max": float(tuned_delta.max().item()),
    }
    output["diagnostics"] = diagnostics
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate candidate-map descriptors while preserving scorer state."
    )
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--tuned-state", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_state).resolve()
    tuned_path = Path(args.tuned_state).resolve()
    output_path = Path(args.output).resolve()
    base_state = torch.load(base_path, map_location="cpu")
    tuned_state = torch.load(tuned_path, map_location="cpu")
    output_state = interpolate_candidate_states(base_state, tuned_state, args.alpha)
    provenance = output_state["diagnostics"]["feature_interpolation"]
    provenance.update(
        {
            "base_state": str(base_path),
            "base_sha256": file_sha256(base_path),
            "tuned_state": str(tuned_path),
            "tuned_sha256": file_sha256(tuned_path),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_state, temporary_path)
    os.replace(temporary_path, output_path)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
