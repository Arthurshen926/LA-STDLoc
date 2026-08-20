"""Exact fresh-process sharding for CPU Track triangulation.

The scientific implementation remains :func:`robust_triangulate_associations`.
This module only packs its immutable inputs once, assigns contiguous landmark
ranges to fresh Python processes, and concatenates their outputs in that same
order.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from evidence.triangulation import robust_triangulate_associations


_TENSOR_NAMES = (
    "landmark_index",
    "query_index",
    "uv",
    "confidence",
    "camera_K",
    "pose_w2c",
    "query_bin",
    "rendered_depth",
)


def _canonical_tensor(name: str, value) -> torch.Tensor | None:
    if value is None:
        return None
    dtype = (
        torch.long
        if name in {"landmark_index", "query_index", "query_bin"}
        else torch.float64
    )
    return torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()


def _pack_inputs(path: Path, values: Mapping[str, object]) -> dict:
    metadata: dict[str, dict | None] = {}
    offset = 0
    with path.open("wb") as stream:
        for name in _TENSOR_NAMES:
            tensor = _canonical_tensor(name, values.get(name))
            if tensor is None:
                metadata[name] = None
                continue
            array = tensor.numpy()
            raw = memoryview(array).cast("B")
            stream.write(raw)
            metadata[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": raw.nbytes,
            }
            offset += raw.nbytes
    return {"byte_count": offset, "tensors": metadata}


def _load_packed(path: Path, metadata: Mapping) -> dict[str, torch.Tensor | None]:
    if path.stat().st_size != int(metadata["byte_count"]):
        raise ValueError("Packed triangulation input size changed")
    loaded: dict[str, torch.Tensor | None] = {}
    for name, record in metadata["tensors"].items():
        if record is None:
            loaded[name] = None
            continue
        array = np.memmap(
            path,
            mode="c",
            dtype=np.dtype(record["dtype"]),
            offset=int(record["offset"]),
            shape=tuple(record["shape"]),
        )
        # COW exposes writable tensors to PyTorch without permitting a worker
        # to change the single packed input shared by every process.
        loaded[name] = torch.from_numpy(array)
    return loaded


def _default_command(job: Path, shard_index: int) -> Sequence[str]:
    return (
        sys.executable,
        "-m",
        "evidence.parallel_triangulation",
        "--job",
        str(job),
        "--shard-index",
        str(shard_index),
    )


def _fixed_ranges(count: int, workers: int) -> list[tuple[int, int]]:
    workers = min(max(int(workers), 1), max(int(count), 1))
    return [
        (count * index // workers, count * (index + 1) // workers)
        for index in range(workers)
    ]


def _run_fresh_processes(
    job: Path,
    shard_count: int,
    command_builder: Callable[[Path, int], Sequence[str]] = _default_command,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    worker_environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root),
    }
    processes: list[subprocess.Popen] = []
    try:
        for shard in range(shard_count):
            processes.append(
                subprocess.Popen(
                    command_builder(job, shard),
                    cwd=repository_root,
                    env=worker_environment,
                )
            )
        failures = []
        for shard, process in enumerate(processes):
            code = process.wait()
            if code:
                failures.append((shard, code))
        if failures:
            raise RuntimeError(f"Fresh triangulation worker failure: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                process.wait()


def robust_triangulate_associations_fresh_cpu(
    *,
    worker_count: int = 2,
    _command_builder: Callable[[Path, int], Sequence[str]] = _default_command,
    **arguments,
) -> dict[str, torch.Tensor]:
    """Run exact triangulation in fresh CPU processes over fixed Track ranges."""
    landmark_count = int(arguments["landmark_count"])
    if landmark_count <= 0:
        raise ValueError("landmark_count must be positive")
    ranges = _fixed_ranges(landmark_count, worker_count)
    tensor_values = {name: arguments.get(name) for name in _TENSOR_NAMES}
    parameters = {
        name: value
        for name, value in arguments.items()
        if name not in _TENSOR_NAMES and name != "landmark_count"
    }
    with tempfile.TemporaryDirectory(prefix="lafgs-triangulation-") as directory:
        root = Path(directory)
        packed = root / "inputs.bin"
        job = root / "job.json"
        record = {
            "schema": "lafgs.fresh_cpu_triangulation",
            "version": 1,
            "landmark_count": landmark_count,
            "ranges": ranges,
            "parameters": parameters,
            "packed": _pack_inputs(packed, tensor_values),
        }
        job.write_text(json.dumps(record, sort_keys=True))
        _run_fresh_processes(job, len(ranges), _command_builder)
        shards = [
            torch.load(root / f"shard_{index:04d}.pt", map_location="cpu")
            for index in range(len(ranges))
        ]
        fields = tuple(shards[0])
        if any(tuple(shard) != fields for shard in shards[1:]):
            raise ValueError("Fresh triangulation shard fields differ")
        return {
            field: torch.cat([shard[field] for shard in shards], dim=0)
            for field in fields
        }


def _run_worker(job_path: Path, shard_index: int) -> None:
    record = json.loads(job_path.read_text())
    if record.get("schema") != "lafgs.fresh_cpu_triangulation":
        raise ValueError("Unexpected fresh triangulation job schema")
    ranges = record["ranges"]
    begin, end = (int(value) for value in ranges[int(shard_index)])
    tensors = _load_packed(job_path.parent / "inputs.bin", record["packed"])
    landmark = tensors["landmark_index"]
    assert landmark is not None
    selected = (landmark >= begin) & (landmark < end)
    for name in ("landmark_index", "query_index", "uv", "confidence", "rendered_depth"):
        if tensors[name] is not None:
            tensors[name] = tensors[name][selected]
    tensors["landmark_index"] = tensors["landmark_index"] - begin
    result = robust_triangulate_associations(
        landmark_count=end - begin,
        **tensors,
        **record["parameters"],
    )
    output = job_path.parent / f"shard_{int(shard_index):04d}.pt"
    temporary = output.with_suffix(".tmp")
    torch.save(result, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    _run_worker(args.job, args.shard_index)


if __name__ == "__main__":
    main()
