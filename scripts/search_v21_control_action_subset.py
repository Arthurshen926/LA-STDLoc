#!/usr/bin/env python3
"""Select a quarantined V21 action subset on tuning-control with exact PoseLib."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v21_control_subset_search import (
    ControlSubsetSearchStopped,
    build_control_selected_candidate,
    validate_control_selected_candidate,
    validate_control_subset_search_audit,
)
from map_learning.v21_pose_feedback_transductive import (
    METADATA_FIELD,
    atomic_torch_save_fresh,
    source_record,
    verify_source_record,
)


def _threshold(value: str) -> float | None:
    if value.lower() in {"disabled", "none"}:
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--expected-parent-candidate-sha256", required=True)
    parser.add_argument("--control-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-control-cache-sha256", action="append", required=True
    )
    parser.add_argument(
        "--activation-threshold",
        action="append",
        type=_threshold,
        default=None,
        help="Pre-registered menu item: disabled or an absolute cosine threshold.",
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--maximum-beam-depth", type=int, default=8)
    parser.add_argument("--maximum-greedy-depth", type=int)
    parser.add_argument("--maximum-backward-steps", type=int)
    parser.add_argument("--minimum-paired-r5-gain", type=int, default=1)
    parser.add_argument("--matcher-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(source: dict) -> dict:
    return torch.load(source["path"], map_location="cpu", weights_only=False)


def main() -> None:
    args = _parse_args()
    if len(args.control_cache) != len(args.expected_control_cache_sha256):
        raise ValueError("each control cache needs one expected SHA256")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    stable_source = source_record(
        args.stable_map,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_stable_map_sha256,
    )
    parent_source = source_record(
        args.parent_candidate,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_parent_candidate_sha256,
    )
    control_sources = [
        source_record(path, sha256_file_fn=sha256_file, expected_sha256=digest)
        for path, digest in zip(
            args.control_cache, args.expected_control_cache_sha256
        )
    ]
    stable = _load(stable_source)
    parent = _load(parent_source)
    controls = [_load(source) for source in control_sources]
    if args.audit_output.expanduser().resolve() == args.output.expanduser().resolve():
        raise ValueError("search audit and selected candidate outputs must differ")
    try:
        selected = build_control_selected_candidate(
            stable_map=stable,
            parent_candidate=parent,
            control_cache_payloads=controls,
            stable_map_source=stable_source,
            parent_candidate_source=parent_source,
            control_cache_sources=control_sources,
            activation_threshold_menu=(
                tuple(args.activation_threshold)
                if args.activation_threshold is not None
                else (None, 0.8, 0.85, 0.9, 0.95)
            ),
            beam_width=args.beam_width,
            maximum_beam_depth=args.maximum_beam_depth,
            maximum_greedy_depth=args.maximum_greedy_depth,
            maximum_backward_steps=args.maximum_backward_steps,
            minimum_paired_r5_gain=args.minimum_paired_r5_gain,
            matcher_chunk_size=args.matcher_chunk_size,
            device=args.device,
        )
    except ControlSubsetSearchStopped as stopped:
        for source in [stable_source, parent_source, *control_sources]:
            verify_source_record(source, sha256_file_fn=sha256_file)
        audit_output = atomic_torch_save_fresh(
            stopped.audit,
            args.audit_output,
            validator=validate_control_subset_search_audit,
        )
        print(f"{audit_output} decision=STOP_NO_ACTION candidate_not_written=true")
        return
    for source in [stable_source, parent_source, *control_sources]:
        verify_source_record(source, sha256_file_fn=sha256_file)
    audit_output = atomic_torch_save_fresh(
        selected[METADATA_FIELD]["search_audit"],
        args.audit_output,
        validator=validate_control_subset_search_audit,
    )
    output = atomic_torch_save_fresh(
        selected,
        args.output,
        validator=lambda value: validate_control_selected_candidate(
            value, stable_map=stable, parent_candidate=parent
        ),
    )
    print(f"{audit_output} decision=GO_SELECTED_ACTION")
    print(output)


if __name__ == "__main__":
    main()
