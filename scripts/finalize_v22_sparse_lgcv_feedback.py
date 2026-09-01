#!/usr/bin/env python3
"""Finalize V22 sparse LGCV feedback using continuous pose metrics first."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning import v22_sparse_lgcv_feedback as feedback


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("adaptation", "control", "confirmation"):
        parser.add_argument(f"--{role}-evaluation", type=Path, required=True)
        parser.add_argument(f"--expected-{role}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, expected: str) -> tuple[dict, dict]:
    resolved = path.expanduser().resolve()
    digest = sha256_file(resolved)
    if digest != expected:
        raise ValueError(f"V22 finalizer source SHA differs: {resolved}")
    value = torch.load(resolved, map_location="cpu", weights_only=False)
    feedback.validate_evaluation(value)
    return value, {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def main() -> None:
    args = _args()
    loaded = [
        _load(args.adaptation_evaluation, args.expected_adaptation_sha256),
        _load(args.control_evaluation, args.expected_control_sha256),
        _load(args.confirmation_evaluation, args.expected_confirmation_sha256),
    ]
    payload = feedback.finalize_evaluations(
        [value[0] for value in loaded], [value[1] for value in loaded]
    )
    for source in [value[1] for value in loaded]:
        if sha256_file(source["path"]) != source["sha256"]:
            raise RuntimeError(f"V22 finalizer source changed: {source['path']}")
    output = feedback.atomic_torch_save_final_fresh(payload, args.output)
    print(output)
    print(payload["decision"])
    print(payload["phase_gates"])


if __name__ == "__main__":
    main()
