"""Materialize mapping-only view-conditioned Anchor descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v27_view_conditioned_anchor_descriptor import (
    make_artifact,
    validate_artifact,
)
from scripts.fixed_pair_matcher_ceiling_common import atomic_torch_save_fresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--mapping-observation-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--minimum-mode-observations", type=int, default=2)
    parser.add_argument("--minimum-mapping-families", type=int, default=2)
    parser.add_argument("--minimum-owner-margin", type=float, default=0.0)
    parser.add_argument("--authorization-device", default="cpu")
    parser.add_argument("--authorization-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    map_path = args.map.resolve()
    cache_path = args.mapping_observation_cache.resolve()
    if sha256_file(map_path) != args.expected_map_sha256:
        raise ValueError("stable F0 map SHA256 differs")
    if sha256_file(cache_path) != args.expected_cache_sha256:
        raise ValueError("mapping observation cache SHA256 differs")
    map_state = torch.load(map_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload = make_artifact(
        map_path=map_path,
        observation_cache_path=cache_path,
        map_state=map_state,
        observation_cache=cache,
        minimum_mode_observations=args.minimum_mode_observations,
        minimum_mapping_families=args.minimum_mapping_families,
        minimum_owner_margin=args.minimum_owner_margin,
        authorization_device=args.authorization_device,
        authorization_chunk_size=args.authorization_chunk_size,
    )
    if (
        sha256_file(map_path) != args.expected_map_sha256
        or sha256_file(cache_path) != args.expected_cache_sha256
    ):
        raise RuntimeError("V27 input changed during materialization")
    output = atomic_torch_save_fresh(
        payload,
        args.output.resolve(),
        validator=lambda value: validate_artifact(value, map_state=map_state),
    )
    print(
        f"wrote {output} sha256={sha256_file(output)} "
        f"valid_modes={int(payload['mode_valid'].sum())} "
        f"authorized_modes={int(payload['mode_authorized'].sum())}"
    )


if __name__ == "__main__":
    main()
