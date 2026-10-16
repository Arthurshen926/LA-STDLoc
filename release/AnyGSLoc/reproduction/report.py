"""Summarize fresh PSNR, native accuracy, or matched-timing runs."""

import argparse
from pathlib import Path

import numpy as np

from anygsloc.pipeline.benchmark import SCENES, selection
from anygsloc.pipeline.io import load_config, read_json, require_complete, write_json
from anygsloc.pipeline.report import aggregate, metrics, safe
from reproduction.aggregate import timing


def quality_report(root, config, scenes):
    values = {}
    for scene in scenes:
        folder = root / scene / "psnr"
        require_complete(folder)
        rows = read_json(folder / "records.json")
        names = [row["image_name"] for row in rows]
        if (
            len(names) != len(set(names))
            or len(rows) != config["scenes"][scene]["mapping_count"]
        ):
            raise ValueError(f"incomplete mapping views: {scene}")
        values[scene] = {
            "images": len(rows),
            "PSNR_dB": float(np.mean([row["psnr_db"] for row in rows])),
        }
    return {
        "per_scene": values,
        "images": sum(v["images"] for v in values.values()),
        "PSNR_dB": float(np.mean([v["PSNR_dB"] for v in values.values()])),
    }


def native_report(root, config, scenes):
    groups = {"initial": {}, "refined": {}}
    identities = set()
    for scene in scenes:
        folder = root / scene
        manifest = read_json(folder / "manifest.json")
        receipt = read_json(folder / "receipt.json")
        if manifest["stage"] != "accuracy" or receipt["status"] != "complete":
            raise ValueError("a complete accuracy run is required")
        names = manifest["query_names"]
        rows = read_json(folder / manifest["results"])
        if (
            len(names) != len(set(names))
            or len(rows) != len(names)
            or len(rows) != config["scenes"][scene]["test_count"]
        ):
            raise ValueError(f"incomplete native queries: {scene}")
        if manifest["scene"] != scene:
            raise ValueError("native scene identity changed")
        identities.add((manifest["method"], manifest["mapping_semantics"]))
        for stage, prefix in [("initial", "sparse"), ("refined", "dense")]:
            value = metrics(
                [
                    {
                        "translation_error_cm": row[prefix + "_TE"],
                        "rotation_error_deg": row[prefix + "_AE"],
                    }
                    for row in rows
                ]
            )
            value["bytes"] = manifest["storage_bytes"]
            groups[stage][scene] = value
    if len(identities) != 1:
        raise ValueError("mixed native method or mapping-semantic condition")
    method, semantics = identities.pop()
    return {
        "method": method,
        "mapping_semantics": semantics,
        "stages": {
            key: {"per_scene": values, "scene_equal": aggregate(values)}
            for key, values in groups.items()
        },
    }


def timing_report(root, partial=False):
    cells, seen, populations = [], set(), {}
    for path in sorted(root.rglob("round.json")):
        value = read_json(path)
        scene, method = value["scene"], value["method"]
        group = (
            method
            + "_"
            + (
                value["map_name"]
                if method == "AnyGSLoc"
                else value["mapping_semantics"]
            )
        )
        identity = (group, scene, value["round"])
        if identity in seen or scene not in SCENES or value["round"] not in (0, 1, 2):
            raise ValueError("duplicate or invalid timing process")
        seen.add(identity)
        populations.setdefault(group, set()).add((scene, value["round"]))
        expected = selection(scene, value["selection"]["full_query_order"])
        for key, rows in [
            ("measured_query_order", value["queries"]),
            ("warmup_query_order", value["warmup_queries"]),
        ]:
            if [row["name"] for row in rows] != expected[key]:
                raise ValueError("timing query subset/order differs from the paper")
        case = (
            f"AnyGSLoc_{scene}_{value['map_name']}"
            if method == "AnyGSLoc"
            else f"{method}_{scene}_mapping_semantics_{value['mapping_semantics']}_eval"
        )
        cells.append({"case": case, "record": value})
    if not cells:
        raise ValueError("no completed timing rounds")
    expected = {(scene, round_id) for scene in SCENES for round_id in range(3)}
    complete = all(p == expected for p in populations.values())
    if not complete and not partial:
        raise ValueError(
            "requires five scenes and three distinct rounds per configuration; --partial is for smoke checks"
        )
    return {
        "complete_configured_dataset": complete,
        "processes": len(cells),
        "timing": timing(
            {"timing": cells}, loader=lambda value: value, complete=not partial
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["psnr", "native", "timing"])
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Fresh run root; see docs/reproduction.md",
    )
    parser.add_argument("--config", default="configs/cambridge.json")
    parser.add_argument("--scene", help="Explicit partial-scene check for PSNR/native")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Allow incomplete timing campaign for smoke checks",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON output file"
    )
    args = parser.parse_args()
    if args.kind == "timing":
        result = timing_report(args.root, args.partial)
    else:
        config = load_config(args.config)
        scenes = [args.scene] if args.scene else list(config["scenes"])
        if not set(scenes) <= config["scenes"].keys():
            parser.error("unknown scene")
        operation = quality_report if args.kind == "psnr" else native_report
        result = operation(args.root, config, scenes)
        result["complete_configured_dataset"] = set(scenes) == set(config["scenes"])
    write_json(args.output, safe(result))
    print(args.output)


if __name__ == "__main__":
    main()
