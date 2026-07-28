#!/usr/bin/env python3
"""Register or verify a LaFGS content-addressed artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.artifact_contract import (
    build_contract,
    verify_contract,
)


def _pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"duplicate parent name: {name}")
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--artifact", required=True)
    register.add_argument("--manifest", required=True)
    register.add_argument("--kind", required=True)
    register.add_argument("--run-type", required=True)
    register.add_argument("--repo-root", default=".")
    register.add_argument("--parent", action="append", default=[])
    register.add_argument("--config-json", default="{}")
    register.add_argument("--query-registry-from", default="")
    register.add_argument("--anchor-registry-from", default="")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    if args.command == "verify":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        verify_contract(payload)
        print(json.dumps({"verified": str(manifest)}, indent=2))
        return

    query_payload = (
        torch.load(
            args.query_registry_from,
            map_location="cpu",
            weights_only=False,
        )
        if args.query_registry_from
        else None
    )
    anchor_payload = (
        torch.load(
            args.anchor_registry_from,
            map_location="cpu",
            weights_only=False,
        )
        if args.anchor_registry_from
        else None
    )
    payload = build_contract(
        artifact=args.artifact,
        kind=args.kind,
        run_type=args.run_type,
        repo_root=args.repo_root,
        parents=_pairs(args.parent),
        resolved_config=json.loads(args.config_json),
        query_registry_payload=query_payload,
        anchor_registry_payload=anchor_payload,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
