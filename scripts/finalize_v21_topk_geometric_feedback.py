#!/usr/bin/env python3
"""Finalize the V21 Top-K geometric arm after one confirmation replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

import torch

from common.hashing import sha256_file
from map_learning.v21_topk_geometric_feedback import (
    FINAL_SCHEMA,
    finalize_evaluations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("adaptation", "control", "confirmation"):
        parser.add_argument(f"--{role}-evaluation", type=Path, required=True)
        parser.add_argument(f"--expected-{role}-evaluation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = []
    sources = []
    for role in ("adaptation", "control", "confirmation"):
        path = getattr(args, f"{role}_evaluation").expanduser().resolve()
        digest = sha256_file(path)
        if digest != getattr(args, f"expected_{role}_evaluation_sha256"):
            raise ValueError(f"{role} evaluation SHA differs")
        payloads.append(torch.load(path, map_location="cpu", weights_only=False))
        sources.append(
            {"path": str(path), "sha256": digest, "size_bytes": int(path.stat().st_size)}
        )
    result = finalize_evaluations(payloads, sources)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 Top-K final decision exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(result, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != FINAL_SCHEMA or reloaded != result:
            raise ValueError("V21 Top-K final decision reload differs")
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
    print(result)


if __name__ == "__main__":
    main()
