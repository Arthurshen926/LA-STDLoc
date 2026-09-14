#!/usr/bin/env python3
"""Reproducible brute-force versus incremental P8 closure microbenchmark."""

from __future__ import annotations

import argparse
import time

import torch

from evidence.cycle_verified_fisher import (
    _complete_verified_triangles_bruteforce,
    _complete_verified_triangles_incremental,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangles", type=int, default=3000)
    parser.add_argument("--minimum-speedup", type=float, default=100.0)
    args = parser.parse_args()
    if args.triangles < 1:
        raise ValueError("benchmark triangle count must be positive")
    # Disjoint triangles are the registered loop's adversarial bounded case:
    # every iteration adds exactly three edges but all remaining rows are still
    # rescanned.  T=3000 is small enough for a local benchmark and exposes the
    # quadratic scan independently of geometry/probe construction.
    edge_count = 3 * args.triangles
    edges = torch.arange(edge_count, dtype=torch.long).reshape(-1, 3)
    utility = torch.tensor(
        [float(args.triangles - index) for index in range(args.triangles)],
        dtype=torch.float64,
    )

    started = time.perf_counter()
    expected = _complete_verified_triangles_bruteforce(
        triangle_edges=edges,
        triangle_utility=utility,
        selected=set(),
        pair_budget=edge_count,
    )
    brute_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = _complete_verified_triangles_incremental(
        triangle_edges=edges,
        triangle_utility=utility,
        selected=set(),
        pair_budget=edge_count,
        edge_count=edge_count,
    )
    incremental_seconds = time.perf_counter() - started
    if actual != expected:
        raise AssertionError("benchmark implementations selected different edges")
    speedup = brute_seconds / max(incremental_seconds, 1e-12)
    print(
        {
            "edges": edge_count,
            "triangles": args.triangles,
            "budget": edge_count,
            "brute_seconds": brute_seconds,
            "incremental_seconds": incremental_seconds,
            "speedup": speedup,
            "selected_count": len(actual),
        }
    )
    if speedup < args.minimum_speedup:
        raise RuntimeError(
            f"incremental closure speedup {speedup:.2f}x is below "
            f"the required {args.minimum_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
