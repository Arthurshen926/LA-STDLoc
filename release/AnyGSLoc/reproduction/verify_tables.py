"""Check the paper's locally measured table cells against recomputed records."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
TE = "median_TE_cm"
RE = "median_RE_deg"
TAIL = [TE, RE, "mean_TE_cm", "P90_TE_cm", "P95_TE_cm", "R5_pct", "R15_pct"]
PRIORS = {
    "Vanilla 3DGS": "vanilla_3dgs",
    "Vanilla 2DGS": "vanilla_2dgs",
    "MAtCha": "matcha",
    "i3DGS raw": "i3dgs_raw",
    "i3DGS +1k": "i3dgs_1k",
    "Long-LRM raw": "long_lrm_raw",
    "Long-LRM +1k": "long_lrm_1k",
}


def rows(table):
    result = []
    for line in (ROOT / "paper_tables" / f"{table}.tex").read_text().splitlines():
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = re.split(r"(?<!\\)&", line)
        last = re.sub(r"\\(?:textbf|underline)\{([^{}]+)\}", r"\1", cells[-1]).strip()
        if not re.match(r"[0-9]", last):
            continue
        result.append(cells)
    return result


def check(table, cells, start, expected, checks):
    actual = []
    for cell in cells[start:]:
        cell = re.sub(r"\\multirow\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}", r"\1", cell)
        actual += re.findall(r"\d[\d,]*(?:\.\d+)?", cell)
    if len(actual) != len(expected):
        raise ValueError(f"{table}: cell count differs: {actual} vs {expected}")
    for token, value in zip(actual, expected):
        digits = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5000001 * 10 ** (-digits)
        if abs(float(token.replace(",", "")) - value) > tolerance:
            raise ValueError(f"{table}: printed {token}, recomputed {value}")
    checks[table] = checks.get(table, 0) + len(actual)


def verify(data):
    checks = {}
    loc = data["localization"]
    tim = data["timing"]

    def metric(group, keys):
        return [loc[group]["scene_equal"][k] for k in keys]

    def prior_group(prior, mode):
        return (
            f"Cambridge/{mode}" if prior == "vanilla_3dgs" else f"priors/{prior}/{mode}"
        )

    def input_group(mode, branch):
        return f"Cambridge/{branch}" if mode == "pure_gs" else f"inputs/{mode}/{branch}"

    def ff(mode, branch):
        return (
            f"Cambridge/{branch}"
            if mode == "filter_on_fused"
            else f"filter_fusion/{branch}/{mode}"
        )

    def baseline(method, semantic, stage):
        return (
            "Cambridge/" + stage
            if method == "AnyGSLoc"
            else f"native/{method}/{semantic}/{stage}"
        )

    for table, group, order in [
        (
            "T1",
            "Cambridge/full",
            [
                "GreatCourt",
                "KingsCollege",
                "OldHospital",
                "ShopFacade",
                "StMarysChurch",
            ],
        ),
        (
            "T2",
            "7Scenes/compact",
            ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
        ),
    ]:
        row = next(r for r in rows(table) if r[0].strip() == "Ours")
        g = loc[group]
        expected = [g["per_scene"][s][k] for s in order for k in [TE, RE]] + [
            g["scene_equal"][k] for k in [TE, RE]
        ]
        check(table, row, 2, expected, checks)
    group = loc["12Scenes/compact"]
    rs = [r for r in rows("T3") if r[0].strip() == "Ours"]
    names = [
        "apt1_kitchen",
        "apt1_living",
        "apt2_bed",
        "apt2_kitchen",
        "apt2_living",
        "apt2_luke",
        "office1_gates362",
        "office1_gates381",
        "office1_lounge",
        "office1_manolis",
        "office2_5a",
        "office2_5b",
    ]
    for i, row in enumerate(rs):
        expected = [
            group["per_scene"][s][k] for s in names[i * 6 : i * 6 + 6] for k in [TE, RE]
        ]
        if i == 1:
            expected += [group["scene_equal"][k] for k in [TE, RE]]
        check("T3", row, 2, expected, checks)
    for i, row in enumerate(rows("T4")):
        method = ["STDLoc", "STDLoc", "ULFLoc", "ULFLoc", "AnyGSLoc", "AnyGSLoc"][i]
        stage = ["initial", "refined", "initial", "refined", "full", "compact"][i]
        t = tim[method + "_on" if i < 4 else "AnyGSLoc_" + stage]
        expected = metric(
            baseline(method, "on", stage), [TE, RE, "mean_TE_cm", "R5_pct"]
        )
        if row[7].strip():
            expected.append(t["storage_bytes"] / 1e6)
        expected.append(
            t["coarse_prefix_ms" if stage == "initial" else "core_ms"]["mean"]
        )
        check("T4", row, 4, expected, checks)
    for row, (prior, mode) in zip(
        rows("T5"),
        [
            (p, m)
            for p in ["vanilla_3dgs", "matcha"]
            for m in ["center", "depth", "ray"]
        ],
    ):
        check(
            "T5", row, 1, metric(f"geometry/{prior}/{mode}", [TE, RE, "R5_pct"]), checks
        )
    modes = [
        "filter_off_single",
        "filter_on_single",
        "filter_off_fused",
        "filter_on_fused",
    ]
    for table, branch in [("T6", "compact"), ("S_t6_full", "full")]:
        for row, mode in zip(rows(table), modes):
            check(
                table,
                row,
                2,
                metric(
                    ff(mode, branch),
                    ["points", TE, RE, "mean_TE_cm", "P95_TE_cm", "R5_pct"],
                ),
                checks,
            )
    selection = [
        "Cambridge/full",
        "selection/quality_stage",
        "selection/coverage_stage",
        "Cambridge/compact",
        "selection/spatial",
        "selection/quality_topk",
        "selection/tracklength_topk",
    ]
    for row, group in zip(rows("T7"), selection):
        keys = (["points"] if row[1].strip() else []) + [TE, RE, "P95_TE_cm", "R15_pct"]
        check("T7", row, 1, metric(group, keys), checks)
    for table, branch in [("T8", "compact"), ("S_prior_full", "full")]:
        for row in rows(table):
            prior = PRIORS[row[0].strip()]
            g = loc[prior_group(prior, branch)]["scene_equal"]
            check(
                table,
                row,
                1,
                [g[k] for k in [TE, RE, "R5_pct", "points"]]
                + [g["bytes"] / 1e6, data["psnr"][prior]["PSNR_dB"]],
                checks,
            )
    for row, mode in zip(rows("T9"), ["pure_gs", "raw_only", "hybrid"]):
        expected = []
        for branch in ["full", "compact"]:
            g = loc[input_group(mode, branch)]["scene_equal"]
            expected += [g[k] for k in [TE, RE, "R5_pct"]] + [g["bytes"] / 1e6]
        check("T9", row, 1, expected, checks)
    for table in ["S_baselines", "S_baseline_tails"]:
        for row in rows(table):
            method = row[0].strip().replace("ULF-Loc", "ULFLoc")
            semantic = row[1].strip()
            stage = row[2].strip().lower().replace("final", "refined")
            group = baseline(method, semantic, stage)
            expected = metric(
                group,
                [TE, RE, "mean_TE_cm", "R5_pct", "R15_pct"]
                if table == "S_baselines"
                else ["mean_TE_cm", "P90_TE_cm", "P95_TE_cm"],
            )
            if table == "S_baselines":
                expected.append(
                    tim[
                        method + "_" + semantic
                        if method != "AnyGSLoc"
                        else "AnyGSLoc_" + stage
                    ]["storage_bytes"]
                    / 1e6
                )
            check(table, row, 3, expected, checks)
    for table in [
        "S_filter_fusion",
        "S_geometry",
        "S_prior_tails",
        "S_input_tails",
        "S_selection",
    ]:
        for row in rows(table):
            label = row[0].strip().replace("\\_", "_")
            keys = TAIL
            if table == "S_filter_fusion":
                branch, mode = label.split(" / ")
                group = ff(mode, branch.lower())
            elif table == "S_geometry":
                prior, mode = label.split(" / ")
                group = f"geometry/{'vanilla_3dgs' if prior == 'vanilla3DGS' else 'matcha'}/{mode}"
            elif table == "S_prior_tails":
                prior, branch = label.split(" / ")
                group = prior_group(PRIORS[prior], branch.lower())
            elif table == "S_input_tails":
                mode, branch = label.split(" / ")
                group = input_group(
                    {
                        "Reference images": "raw_only",
                        "Gaussian renderings": "pure_gs",
                        "Both": "hybrid",
                    }[mode],
                    branch.lower(),
                )
                if "--" in row[4]:
                    keys = [k for k in keys if k != "P95_TE_cm"]
            else:
                group = {
                    "full": "Cambridge/full",
                    "native_compact": "Cambridge/compact",
                }.get(label, "selection/" + label)
            check(table, row, 1, metric(group, keys), checks)
    for row in rows("S_gs_assisted"):
        group = "inputs/real_gs_assisted/" + row[0].strip().lower()
        g = loc[group]["scene_equal"]
        check(
            "S_gs_assisted",
            row,
            1,
            [
                g[k]
                for k in [
                    TE,
                    RE,
                    "mean_TE_cm",
                    "P90_TE_cm",
                    "R5_pct",
                    "R15_pct",
                    "points",
                ]
            ]
            + [g["bytes"] / 1e6],
            checks,
        )
    for row in rows("S_timing"):
        value = tim[row[0].strip().replace("\\_", "_")]
        check(
            "S_timing",
            row,
            1,
            [
                value[stage][stat]
                for stage in ["core_ms", "frame_ms"]
                for stat in ["mean", "p50", "p90", "p95"]
            ],
            checks,
        )
    for row in rows("S_resources"):
        value = tim[row[0].strip().replace("\\_", "_")]
        check(
            "S_resources",
            row,
            1,
            [value[k] for k in ["initialization_s", "allocated_GB", "reserved_GB"]],
            checks,
        )
    for row in rows("S_native_stages"):
        value = tim[row[0].strip().replace("ULF-Loc", "ULFLoc") + "_" + row[1].strip()]
        check(
            "S_native_stages",
            row,
            2,
            [
                value["native_initial_segment_ms"],
                value["coarse_prefix_ms"]["mean"],
                value["dense_increment_ms"]["mean"],
            ],
            checks,
        )
    if len(checks) != 22:
        raise ValueError("missing table verification")
    return {
        "tables": len(checks),
        "numeric_cells": sum(checks.values()),
        "cells_per_table": checks,
        "literature_rows": "preserved cited values; not local reruns",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(json.loads(args.metrics.read_text())), indent=2))


if __name__ == "__main__":
    main()
