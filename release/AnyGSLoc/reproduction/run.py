"""Print or execute a sequential experiment plan, reusing verified completed stages."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from anygsloc.pipeline.io import load_config, require_complete


def plan(family, indoor=False):
    if indoor:
        if family != "main":
            raise ValueError("indoor paper tables use the main profile only")
        return [
            ("build", "pure_gs", "on", "full"),
            ("evaluate", "pure_gs", "on", "compact"),
        ]
    jobs = [("extract", "pure_gs", "on", "full")]
    if family == "psnr":
        return [("psnr", "pure_gs", "on", "full")]
    modes = (
        ["raw_only", "hybrid", "pure_gs"]
        if family in ("inputs", "assisted")
        else ["pure_gs"]
    )
    for mode in modes:
        jobs.append(("build", mode, "on", "full"))
        jobs.append(("select", mode, "on", "compact"))
        if family == "geometry":
            jobs.append(("geometry", mode, "on", "full"))
            maps = ["center", "depth", "ray"]
        elif family == "selection":
            maps = [
                "full",
                "quality_stage",
                "coverage_stage",
                "compact",
                "spatial",
                "quality_topk",
                "tracklength_topk",
            ]
        elif family == "filter-fusion":
            maps = ["full", "single", "compact", "compact_single"]
        else:
            maps = ["full", "compact"]
        jobs += [("evaluate", mode, "on", m) for m in maps]
    if family == "filter-fusion":
        jobs += [
            ("build", "pure_gs", "off", "full"),
            ("select", "pure_gs", "off", "compact"),
        ]
        jobs += [
            ("evaluate", "pure_gs", "off", m)
            for m in ["full", "single", "compact", "compact_single"]
        ]
    if family == "assisted":
        jobs += [("assisted", "real_gs_assisted", "on", "full")]
        jobs += [("evaluate", "real_gs_assisted", "on", m) for m in ["full", "compact"]]
    return jobs


def finished(root, stage, mode, filtering, map_name, indoor):
    cell = root / mode / ("filtered" if filtering == "on" else "unfiltered")
    if indoor and stage == "build":
        if not (cell / "complete.json").exists():
            return False
        require_complete(cell)
        return True
    if stage == "evaluate":
        target = cell / "evaluations" / map_name
        if not (target / "summary.json").exists():
            return False
        rows = json.loads((target / "results.json").read_text())
        manifest = json.loads((target / "manifest.json").read_text())
        if [r["image_name"] for r in rows] != manifest["queries"]:
            raise ValueError("partial evaluation")
        return True
    target = {
        "extract": root / "observations",
        "build": cell,
        "select": cell / "selection",
        "geometry": cell / "geometry",
        "psnr": root / "psnr",
        "assisted": cell,
    }[stage]
    if not (target / "complete.json").exists():
        return False
    require_complete(target)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "family",
        choices=[
            "main",
            "geometry",
            "filter-fusion",
            "selection",
            "inputs",
            "assisted",
            "psnr",
        ],
    )
    p.add_argument("--config", required=True)
    p.add_argument("--asset-root")
    p.add_argument("--output")
    p.add_argument("--scene")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    config = load_config(args.config, args.asset_root, args.output)
    scenes = [args.scene] if args.scene else list(config["scenes"])
    if not set(scenes) <= config["scenes"].keys():
        p.error("unknown scene")
    if args.family in ("inputs", "assisted") and set(config["input_modes"]) != {
        "raw_only",
        "hybrid",
        "pure_gs",
    }:
        p.error("use configs/cambridge_inputs.json for the input ablations")
    indoor = config.get("frontend") == "alpha_gated"
    if args.execute:
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        binding = {
            "config_sha256": config["config_sha256"],
            "asset_root": config["asset_root"],
        }
        receipt = output / "run_config.json"
        if receipt.exists() and json.loads(receipt.read_text()) != binding:
            raise ValueError(
                "output is bound to another configuration; choose a new --output"
            )
        if not receipt.exists():
            receipt.write_text(json.dumps(binding, indent=2) + "\n")
    for scene in scenes:
        root = Path(config["output"]) / scene
        for stage, mode, filtering, map_name in plan(args.family, indoor):
            command = [
                sys.executable,
                "-m",
                "anygsloc",
                "reproduce",
                stage,
                "--config",
                args.config,
                "--output",
                config["output"],
                "--asset-root",
                config["asset_root"],
                "--scene",
                scene,
                "--input",
                mode,
                "--filter",
                filtering,
                "--map",
                map_name,
                "--device",
                args.device,
                "--workers",
                str(args.workers),
            ]
            print(
                json.dumps({"scene": scene, "stage": stage, "command": command}),
                flush=True,
            )
            if args.execute:
                if finished(root, stage, mode, filtering, map_name, indoor):
                    print("Verified completed stage; reusing.", flush=True)
                else:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
