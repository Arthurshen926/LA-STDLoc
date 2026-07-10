#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def _to_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _as_xyz(value):
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    tensor = tensor.float()
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"Expected xyz tensor with shape [N, 3], got {tuple(tensor.shape)}")
    return tensor


def summarize_distances(distances):
    distances = _to_tensor(distances)
    if distances is None:
        distances = torch.empty(0)
    distances = distances.float().reshape(-1)
    distances = distances[torch.isfinite(distances)]
    if distances.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(distances.numel()),
        "mean": float(distances.mean().item()),
        "median": float(torch.quantile(distances, 0.50).item()),
        "p95": float(torch.quantile(distances, 0.95).item()),
        "p99": float(torch.quantile(distances, 0.99).item()),
        "max": float(distances.max().item()),
    }


def _distance_between(a, b):
    return torch.linalg.norm(a.float() - b.float(), dim=-1)


def _get_first_state_tensor(loc_state, names):
    if not isinstance(loc_state, dict):
        return None
    for name in names:
        value = loc_state.get(name)
        if value is not None:
            return _to_tensor(value)
    return None


def _index_stats(source_index):
    source_index = _to_tensor(source_index)
    if source_index is None or source_index.numel() == 0:
        return None
    source_index = source_index.long().reshape(-1)
    valid = source_index >= 0
    if not bool(valid.any()):
        return {
            "count": int(source_index.numel()),
            "valid_count": 0,
            "unique_source_count": 0,
            "max_children_per_source": 0,
            "mean_children_per_source": None,
        }
    unique, counts = torch.unique(source_index[valid], return_counts=True)
    return {
        "count": int(source_index.numel()),
        "valid_count": int(valid.sum().item()),
        "unique_source_count": int(unique.numel()),
        "max_children_per_source": int(counts.max().item()),
        "mean_children_per_source": float(counts.float().mean().item()),
    }


def _birth_iteration_stats(birth_iteration, reference_iteration):
    birth_iteration = _to_tensor(birth_iteration)
    if birth_iteration is None or birth_iteration.numel() == 0:
        return None
    birth_iteration = birth_iteration.long().reshape(-1)
    return {
        "count": int(birth_iteration.numel()),
        "min": int(birth_iteration.min().item()),
        "max": int(birth_iteration.max().item()),
        "born_after_reference_count": int((birth_iteration > int(reference_iteration)).sum().item()),
    }


def compute_geometry_drift_summary(current_xyz, loc_state=None, reference_xyz=None, iteration=None, reference_iteration=0):
    current_xyz = _as_xyz(current_xyz)
    reference_xyz = _as_xyz(reference_xyz)
    loc_state = {} if loc_state is None else loc_state

    summary = {
        "iteration": None if iteration is None else int(iteration),
        "point_count": int(current_xyz.shape[0]),
        "bbox_min": [float(v) for v in current_xyz.min(dim=0).values.tolist()],
        "bbox_max": [float(v) for v in current_xyz.max(dim=0).values.tolist()],
    }

    if reference_xyz is not None:
        count = min(int(current_xyz.shape[0]), int(reference_xyz.shape[0]))
        summary["row_index_drift"] = summarize_distances(_distance_between(current_xyz[:count], reference_xyz[:count]))
        summary["row_index_drift"]["compared_count"] = count
        summary["row_index_drift"]["current_count"] = int(current_xyz.shape[0])
        summary["row_index_drift"]["reference_count"] = int(reference_xyz.shape[0])

    source_xyz = _get_first_state_tensor(loc_state, ["loc_source_xyz", "source_xyz"])
    if source_xyz is not None:
        source_xyz = _as_xyz(source_xyz)
        count = min(int(current_xyz.shape[0]), int(source_xyz.shape[0]))
        summary["source_xyz_drift"] = summarize_distances(_distance_between(current_xyz[:count], source_xyz[:count]))
        summary["source_xyz_drift"]["compared_count"] = count
        summary["source_xyz_drift"]["source_xyz_count"] = int(source_xyz.shape[0])

    source_index = _get_first_state_tensor(loc_state, ["loc_source_index", "source_index", "loc_source_indices"])
    if source_index is not None:
        summary["source_index"] = _index_stats(source_index)
    if source_index is not None and reference_xyz is not None:
        source_index = source_index.long().reshape(-1)
        count = min(int(current_xyz.shape[0]), int(source_index.shape[0]))
        source_index = source_index[:count]
        valid = (source_index >= 0) & (source_index < int(reference_xyz.shape[0]))
        summary["reference_source_index_drift"] = summarize_distances(
            _distance_between(current_xyz[:count][valid], reference_xyz[source_index[valid]])
        )
        summary["reference_source_index_drift"]["valid_source_count"] = int(valid.sum().item())
        summary["reference_source_index_drift"]["compared_count"] = count

    birth_iteration = _get_first_state_tensor(loc_state, ["loc_birth_iteration", "birth_iteration"])
    birth_stats = _birth_iteration_stats(birth_iteration, reference_iteration)
    if birth_stats is not None:
        summary["birth_iteration"] = birth_stats

    return summary


def _load_ply_xyz(path):
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return torch.stack(
        [
            torch.as_tensor(vertex["x"]),
            torch.as_tensor(vertex["y"]),
            torch.as_tensor(vertex["z"]),
        ],
        dim=1,
    ).float()


def _load_loc_state(path):
    if not path.is_file():
        return {}
    return torch.load(str(path), map_location="cpu")


def _iteration_dirs(model_path):
    point_cloud = Path(model_path) / "point_cloud"
    iterations = []
    for child in point_cloud.glob("iteration_*"):
        if child.is_dir():
            try:
                iterations.append(int(child.name.split("_")[-1]))
            except ValueError:
                pass
    return sorted(iterations)


def diagnose_model(model_path, iterations=None, reference_iteration=None):
    model_path = Path(model_path)
    iterations = _iteration_dirs(model_path) if iterations is None else [int(v) for v in iterations]
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* directories found under {model_path}")
    reference_iteration = int(reference_iteration if reference_iteration is not None else iterations[0])
    reference_ply = model_path / "point_cloud" / f"iteration_{reference_iteration}" / "point_cloud.ply"
    reference_xyz = _load_ply_xyz(reference_ply) if reference_ply.is_file() else None
    summaries = []
    for iteration in iterations:
        iteration_dir = model_path / "point_cloud" / f"iteration_{iteration}"
        ply_path = iteration_dir / "point_cloud.ply"
        if not ply_path.is_file():
            raise FileNotFoundError(str(ply_path))
        current_xyz = _load_ply_xyz(ply_path)
        loc_state = _load_loc_state(iteration_dir / "loc_state.pt")
        summaries.append(
            compute_geometry_drift_summary(
                current_xyz=current_xyz,
                loc_state=loc_state,
                reference_xyz=reference_xyz,
                iteration=iteration,
                reference_iteration=reference_iteration,
            )
        )
    return {
        "model_path": str(model_path),
        "reference_iteration": reference_iteration,
        "iterations": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose LaFGS geometry drift with source-aware topology metadata.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, default=None)
    parser.add_argument("--reference_iteration", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    result = diagnose_model(args.model_path, iterations=args.iterations, reference_iteration=args.reference_iteration)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
