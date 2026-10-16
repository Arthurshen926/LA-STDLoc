"""Run paper stages explicitly; planning never starts a GPU job."""

import argparse
import json
from pathlib import Path

from anygsloc.pipeline.io import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "plan",
            "extract",
            "build",
            "assisted",
            "select",
            "evaluate",
            "geometry",
            "psnr",
            "benchmark",
            "report",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--scene", help="One configured scene; omitted means all scenes, serially"
    )
    parser.add_argument(
        "--input",
        choices=["pure_gs", "raw_only", "hybrid", "real_gs_assisted"],
        default="pure_gs",
    )
    parser.add_argument("--filter", choices=["on", "off"], default="on")
    parser.add_argument(
        "--map",
        default="full",
        choices=[
            "full",
            "single",
            "compact",
            "compact_single",
            "quality_stage",
            "coverage_stage",
            "spatial",
            "quality_topk",
            "tracklength_topk",
            "center",
            "depth",
            "ray",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--round", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config, args.asset_root, args.output)
    scenes = [args.scene] if args.scene else list(config["scenes"])
    if not set(scenes) <= config["scenes"].keys():
        parser.error("unknown scene")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.stage == "plan":
        print(
            json.dumps(
                {"config": config, "scene_order": scenes, "gpu_started": False},
                indent=2,
            )
        )
        return
    import torch

    torch.set_num_threads(4)
    if args.stage == "report":
        from anygsloc.pipeline.report import report

        report(config, scenes)
        return
    for scene in scenes:
        from anygsloc.pipeline import build

        spec = config["scenes"][scene]
        if config.get("frontend") == "alpha_gated" and args.stage in (
            "extract",
            "build",
            "select",
        ):
            from anygsloc.pipeline.indoor import build_indoor

            if args.stage != "build":
                parser.error(
                    "the indoor alpha-gated profile uses the combined build stage"
                )
            build_indoor(config, scene, args.device)
            continue
        root = Path(config["output"]) / scene
        cache = root / "observations" / f"{args.input}.pt"
        cell = root / args.input / ("filtered" if args.filter == "on" else "unfiltered")
        selection = args.map not in ("full", "single", "center", "depth", "ray")
        map_path = (
            cell / "selection" / f"{args.map}.pt"
            if selection
            else cell / f"{args.map}.pt"
        )
        if args.map in ("center", "depth", "ray"):
            map_path = cell / "geometry" / args.map / "deployment_map.pt"
        if args.stage == "extract":
            build.extract(config, scene, args.device)
        elif args.stage == "build":
            build.build(
                config,
                scene,
                args.input,
                args.filter == "on",
                args.device,
                args.workers,
            )
        elif args.stage == "assisted":
            with torch.cuda.device(args.device):
                build.build_assisted(root, args.device)
        elif args.stage == "select":
            build.select(cell, cache)
        elif args.stage == "geometry":
            from anygsloc.pipeline.geometry import build_geometry

            build_geometry(
                cell / "selector_state.pt", cache, spec["prior"], cell / "geometry"
            )
        elif args.stage == "evaluate":
            from anygsloc.evaluation.runner import main as evaluate

            target = cell / "evaluations" / args.map
            if (
                not map_path.exists()
                and args.input == "pure_gs"
                and args.filter == "on"
                and spec.get(f"{args.map}_map")
            ):
                map_path = Path(spec[f"{args.map}_map"])
            options = [
                "--dataset",
                spec["dataset"],
                "--images",
                spec.get("images", "processed"),
                "--map",
                str(map_path),
                "--output",
                str(target),
                "--image-domain",
                "rectified",
                "--mask-policy",
                spec.get("query_mask", "none"),
                "--seed",
                "2026",
                "--device",
                args.device,
            ]
            if spec.get("rectification_contract"):
                options += ["--rectification-contract", spec["rectification_contract"]]
            evaluate(options)
        elif args.stage == "psnr":
            from anygsloc.pipeline.quality import evaluate_psnr

            evaluate_psnr(spec, root / "psnr", args.device)
        elif args.stage == "benchmark":
            from anygsloc.pipeline.benchmark import run

            if (
                not map_path.exists()
                and args.input == "pure_gs"
                and args.filter == "on"
                and spec.get(f"{args.map}_map")
            ):
                map_path = Path(spec[f"{args.map}_map"])
            run(
                spec,
                scene,
                map_path,
                cell / "benchmark" / args.map / f"round_{args.round}",
                args.device,
                args.round,
            )


if __name__ == "__main__":
    main()
