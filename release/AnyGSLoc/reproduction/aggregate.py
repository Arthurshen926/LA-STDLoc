"""Recompute paper metrics from the included per-query and per-frame records."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from anygsloc.pipeline.report import aggregate, metrics, safe

ROOT = Path(__file__).resolve().parent


def load_record(ref, lines=False):
    path = ROOT / ref["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError(f"changed record: {path}")
    decoded = gzip.decompress(raw)
    if hashlib.sha256(decoded).hexdigest() != ref.get(
        "payload_sha256", ref["source_sha256"]
    ):
        raise ValueError(f"changed uncompressed record: {path}")
    return (
        [json.loads(line) for line in decoded.splitlines()]
        if lines
        else json.loads(decoded)
    )


def localization(catalog):
    groups = {}
    for cell in catalog["localization"]:
        rows = load_record(cell["record"])
        names = cell["query_names"]
        if len(rows) != cell["query_count"] or len(names) != len(set(names)):
            raise ValueError("incomplete or duplicated query set")
        if "native_stage" in cell:
            prefix = "sparse" if cell["native_stage"] == "initial" else "dense"
            rows = [
                {
                    "image_name": name,
                    "translation_error_cm": r[prefix + "_TE"],
                    "rotation_error_deg": r[prefix + "_AE"],
                }
                for name, r in zip(names, rows)
            ]
        elif [r["image_name"] for r in rows] != names:
            raise ValueError("query order differs from the accepted manifest")
        result = metrics(rows)
        if "points" in cell:
            result.update(points=cell["points"], bytes=cell["map_bytes"])
        groups.setdefault(cell["group"], {})[cell["scene"]] = result
    for name, scenes in groups.items():
        expected = (
            7
            if name.startswith("7Scenes/")
            else 12
            if name.startswith("12Scenes/")
            else 5
        )
        if len(scenes) != expected:
            raise ValueError(f"incomplete scene group: {name}")
    return {k: {"per_scene": v, "scene_equal": aggregate(v)} for k, v in groups.items()}


def quality(catalog):
    groups = {}
    for cell in catalog["psnr"]:
        rows = load_record(cell["records"], lines=True)
        if len({r["name"] for r in rows}) != len(rows):
            raise ValueError("duplicate mapping image")
        # Average per-image PSNR; do not convert the pooled MSE to PSNR.
        groups.setdefault(cell["prior"], {})[cell["scene"]] = {
            "images": len(rows),
            "PSNR_dB": float(np.mean([r["psnr_db"] for r in rows])),
        }
    for scenes in groups.values():
        if len(scenes) != 5 or sum(s["images"] for s in scenes.values()) != 5364:
            raise ValueError("quality evaluation must cover all 5364 mapping views")
    return {
        k: {
            "per_scene": v,
            "images": sum(s["images"] for s in v.values()),
            "PSNR_dB": float(np.mean([s["PSNR_dB"] for s in v.values()])),
        }
        for k, v in groups.items()
    }


def timing(catalog, loader=load_record, complete=True):
    groups = {}
    for cell in catalog["timing"]:
        value = loader(cell["record"])
        case = cell["case"]
        if case.startswith("AnyGSLoc"):
            # Case names are AnyGSLoc_<scene>_<full|compact>.
            fields = case.split("_")
            group = "AnyGSLoc_" + fields[-1]
            scene = "_".join(fields[1:-1])
        else:
            method, rest = case.split("_", 1)
            scene, semantic = rest.split("_mapping_semantics_")
            group = method + "_" + semantic.removesuffix("_eval")
        if len(value["queries"]) != 100 or len(value["warmup_queries"]) != 50:
            raise ValueError("incomplete matched timing process")
        groups.setdefault(group, {}).setdefault(scene, []).append(value)
    output = {}
    for group, scenes in groups.items():
        if complete and (len(scenes) != 5 or any(len(v) != 3 for v in scenes.values())):
            raise ValueError(f"incomplete timing group: {group}")
        rounds = [r for rs in scenes.values() for r in rs]
        rows = [q for r in rounds for q in r["queries"]]
        values = {}
        for stage in ["core_ms", "frame_ms", "coarse_prefix_ms", "dense_increment_ms"]:
            numbers = [
                q["stages"][stage] for q in rows if q["stages"].get(stage) is not None
            ]
            if numbers:
                if len(numbers) != len(rows):
                    raise ValueError("partial stage timing")
                values[stage] = {
                    "mean": float(np.mean(numbers)),
                    **{
                        p: float(np.percentile(numbers, n))
                        for p, n in [("p50", 50), ("p90", 90), ("p95", 95)]
                    },
                }
        values["initialization_s"] = (
            float(np.mean([r["initialization"]["total_ms"] for r in rounds])) / 1000
        )
        native_sparse = [
            q["native_stage_timing"]["sparse_only"]
            for q in rows
            if q.get("native_stage_timing")
            and "sparse_only" in q["native_stage_timing"]
        ]
        if native_sparse:
            if len(native_sparse) != len(rows):
                raise ValueError("partial native sparse-only timing")
            values["native_initial_segment_ms"] = float(np.mean(native_sparse))
        elif group.startswith("STDLoc"):
            values["native_initial_segment_ms"] = values["coarse_prefix_ms"]["mean"]
        values["allocated_GB"] = (
            max(r["measured_torch_peak"]["allocated_bytes"] for r in rounds) / 1e9
        )
        values["reserved_GB"] = (
            max(r["measured_torch_peak"]["reserved_bytes"] for r in rounds) / 1e9
        )
        values["storage_bytes"] = sum(
            rs[0].get("storage_bytes", 0) for rs in scenes.values()
        )
        output[group] = values
    return output


def compute():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    return {
        "revision": catalog["paper_revision"],
        "localization": localization(catalog),
        "psnr": quality(catalog),
        "timing": timing(catalog),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/reference_report"))
    p.add_argument(
        "--check",
        action="store_true",
        help="Compare every numeric metric to the accepted snapshot",
    )
    args = p.parse_args()
    result = safe(compute())
    if args.check:
        expected = json.loads((ROOT / "expected_metrics.json").read_text())
        if result != expected:
            raise ValueError(
                "reference metric snapshot differs; inspect source or aggregator changes"
            )
        print("All accepted per-scene, aggregate, PSNR, and timing metrics match.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "localization.csv").open("w") as stream:
        keys = [
            "group",
            "queries",
            "median_TE_cm",
            "median_RE_deg",
            "mean_TE_cm",
            "P90_TE_cm",
            "P95_TE_cm",
            "R5_pct",
            "R15_pct",
            "points",
            "bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for name, value in result["localization"].items():
            writer.writerow(dict(group=name, **value["scene_equal"]))
    print(args.output / "metrics.json")


if __name__ == "__main__":
    main()
