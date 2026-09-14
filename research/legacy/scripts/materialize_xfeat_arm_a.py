#!/usr/bin/env python3
"""Materialize a locked CPU-only XFeat detector-repeatability (Arm-A) probe.

The command is offline, mapping-only, descriptor-free, and pair-free.  It
fails closed unless every native input has identity XFeat resizing, which
machine-proves the frozen round/floor mask contract for the Stairs protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Sequence
import uuid

import torch

from map_learning.frontend_upper_bound import file_sha256
from map_learning.xfeat_arm_a import materialize_xfeat_arm_a
from map_learning.xfeat_arm_b import XFeatArtifactSpec, validate_xfeat_artifact


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _local_input(path: str | Path, *, label: str) -> Path:
    text = str(path)
    if "://" in text:
        raise ValueError(f"{label} must be a local file")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _expected_sha256(value: str, *, label: str) -> str:
    digest = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} must be a 64-character SHA256")
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--xfeat-worktree", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--expected-xfeat-tree", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-interpolator-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing probe output",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.device != "cpu":
        raise ValueError("XFeat Arm A is CPU-only")
    query_cache_path = _local_input(args.query_cache, label="query cache")
    teacher_path = _local_input(args.teacher, label="teacher")
    weights_path = _local_input(args.weights, label="XFeat weights")
    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    worktree = Path(args.xfeat_worktree).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output in {query_cache_path, teacher_path, weights_path}:
        raise ValueError("output must not overwrite a source artifact")
    if _is_relative_to(output, worktree):
        raise ValueError("output must not modify the external XFeat worktree")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite explicitly: {output}"
        )

    expected_query_sha = _expected_sha256(
        args.expected_query_cache_sha256,
        label="expected query-cache SHA256",
    )
    expected_teacher_sha = _expected_sha256(
        args.expected_teacher_sha256,
        label="expected teacher SHA256",
    )
    query_sha = file_sha256(query_cache_path)
    teacher_sha = file_sha256(teacher_path)
    if query_sha != expected_query_sha:
        raise ValueError(
            f"query-cache SHA256 mismatch: {query_sha} != {expected_query_sha}"
        )
    if teacher_sha != expected_teacher_sha:
        raise ValueError(
            f"teacher SHA256 mismatch: {teacher_sha} != {expected_teacher_sha}"
        )
    query_cache = _torch_load(query_cache_path)
    teacher = _torch_load(teacher_path)
    artifact_spec = XFeatArtifactSpec(
        worktree=worktree,
        weights=weights_path,
        expected_weights_sha256=args.expected_weights_sha256,
        expected_parent_commit=args.expected_parent_commit,
        expected_xfeat_tree=args.expected_xfeat_tree,
        expected_model_sha256=args.expected_model_sha256,
        expected_interpolator_sha256=args.expected_interpolator_sha256,
        expected_wrapper_sha256=args.expected_wrapper_sha256,
    )
    probe = materialize_xfeat_arm_a(
        query_cache=query_cache,
        teacher=teacher,
        query_cache_path=query_cache_path,
        teacher_path=teacher_path,
        dataset_root=dataset_root,
        artifact_spec=artifact_spec,
    )
    if probe["reference"]["query_cache_sha256"] != query_sha:
        raise RuntimeError("query cache changed while materializing Arm A")
    if probe["reference"]["teacher_sha256"] != teacher_sha:
        raise RuntimeError("teacher changed while materializing Arm A")
    if file_sha256(query_cache_path) != query_sha:
        raise RuntimeError("query cache changed before Arm-A output commit")
    if file_sha256(teacher_path) != teacher_sha:
        raise RuntimeError("teacher changed before Arm-A output commit")
    validate_xfeat_artifact(artifact_spec)
    for name, record in probe["queries"].items():
        image = record["image_lineage"]
        if file_sha256(image["source_image_logical_path"]) != image[
            "source_image_sha256"
        ]:
            raise RuntimeError(f"mapping image changed before output: {name}")
    probe["producer"]["cli"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(probe, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "schema": probe["schema"],
        "arm": probe["producer"]["arm"],
        "mapping_only": probe["mapping_only"],
        "uses_test_queries": probe["uses_test_queries"],
        "device": probe["producer"]["device"],
        "query_count": probe["producer"]["query_count"],
        "detected_count_before_mask": probe["producer"][
            "detected_count_before_mask"
        ],
        "detected_count_after_mask": probe["producer"][
            "detected_count_after_mask"
        ],
        "round_floor_mask_equivalent": probe["producer"][
            "all_queries_round_floor_mask_equivalent"
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
