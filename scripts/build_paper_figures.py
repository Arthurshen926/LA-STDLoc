#!/usr/bin/env python3
"""Build the unified Cambridge qualitative figure set from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualization.paper_figures import FigureArtifacts, PaperFigurePipeline


def main() -> None:
    defaults = FigureArtifacts.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shop-root", type=Path, default=defaults.shop_root)
    parser.add_argument("--shop-dataset", type=Path, default=defaults.shop_dataset)
    parser.add_argument(
        "--prior-experiment-root",
        type=Path,
        default=defaults.prior_experiment_root,
    )
    parser.add_argument(
        "--old-hospital-dataset",
        type=Path,
        default=defaults.old_hospital_dataset,
    )
    parser.add_argument(
        "--old-hospital-eval-dataset",
        type=Path,
        default=defaults.old_hospital_eval_dataset,
    )
    parser.add_argument("--config", type=Path, default=defaults.config)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/pool/sqy/lafgs_paper_figures_20260806"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("1", "2", "3", "4", "5", "all"),
        default=("all",),
    )
    args = parser.parse_args()
    artifacts = FigureArtifacts(
        shop_root=args.shop_root,
        shop_dataset=args.shop_dataset,
        prior_experiment_root=args.prior_experiment_root,
        old_hospital_dataset=args.old_hospital_dataset,
        old_hospital_eval_dataset=args.old_hospital_eval_dataset,
        config=args.config,
    )
    pipeline = PaperFigurePipeline(
        artifacts, args.output, device=args.device, seed=args.seed
    )
    requested = set(args.figures)
    if "all" in requested:
        report = pipeline.build_all()
    else:
        builders = {
            "1": pipeline.build_method_overview,
            "2": pipeline.build_primitive_identity,
            "3": pipeline.build_topology_distillation,
            "4": pipeline.build_match_comparison,
            "5": pipeline.build_prior_flexibility,
        }
        report = {
            "schema": "lafgs_publication_figure_subset",
            "version": 1,
            "figures": {name: builders[name]() for name in sorted(requested)},
        }
        (args.output / "figure_subset_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
