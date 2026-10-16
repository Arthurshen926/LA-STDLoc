"""Regenerate deployment and geometry plots from verified records."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from .style import (
    BASE_DARK,
    GRID,
    INK,
    INK2,
    OURS,
    OURS_TEXT,
    matplotlib_style,
    save_figure,
)


def deployment(metrics, output):
    points = []
    for method in ["STDLoc", "ULFLoc"]:
        time = metrics["timing"][method + "_on"]
        for stage, key in [("initial", "coarse_prefix_ms"), ("refined", "core_ms")]:
            value = metrics["localization"][f"native/{method}/on/{stage}"][
                "scene_equal"
            ]
            points.append(
                dict(
                    method="ULF-Loc" if method == "ULFLoc" else method,
                    variant="Native",
                    stage=stage,
                    R5=value["R5_pct"],
                    ms=time[key]["mean"],
                    MB=time["storage_bytes"] / 1e6,
                )
            )
    for variant in ["full", "compact"]:
        value = metrics["localization"]["Cambridge/" + variant]["scene_equal"]
        time = metrics["timing"]["AnyGSLoc_" + variant]
        points.append(
            dict(
                method="AnyGSLoc",
                variant=variant.title(),
                stage="single",
                R5=value["R5_pct"],
                ms=time["core_ms"]["mean"],
                MB=value["bytes"] / 1e6,
            )
        )
    matplotlib_style()
    fig, axs = plt.subplots(
        1, 2, figsize=(180 / 25.4, 88 / 25.4), gridspec_kw={"width_ratios": [1.18, 1]}
    )
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.205, top=0.88, wspace=0.30)
    for ax, title, xlabel, lim, ticks, ylim, yticks in zip(
        axs,
        ["(a) Localization latency", "(b) Deployment storage"],
        ["Cumulative query-core latency (ms, log)", "Five-scene storage (MB, log)"],
        [(50, 2600), (90, 3600)],
        [[50, 100, 250, 500, 1000, 2000], [100, 250, 500, 1000, 2500]],
        [(18, 51), (35, 51)],
        [[20, 25, 30, 35, 40, 45, 50], [35, 40, 45, 50]],
    ):
        ax.set(
            xscale="log",
            xlim=lim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel="Recall @ 5 cm / 5° (%)",
        )
        ax.set_title(title, loc="left", fontsize=8.5, pad=10)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.grid(color=GRID, lw=0.45)
        ax.set_axisbelow(True)
    colors = [BASE_DARK, BASE_DARK, "#7D8C99", "#7D8C99", OURS, OURS]
    offsets = [(-8, 0), (8, 0), (8, -3), (-8, 8), (8, 7), (8, -7)]
    for i, p in enumerate(points):
        ax = axs[0]
        col = colors[i]
        initial = i in (0, 2)
        ours = i >= 4
        marker = "D" if i == 4 else "*" if i == 5 else "o" if i < 2 else "s"
        ax.scatter(
            p["ms"],
            p["R5"],
            s=75 if i == 5 else 34,
            marker=marker,
            facecolors="white" if initial or i == 4 else col,
            edgecolors=col,
            lw=0.9,
            zorder=4,
        )
        dx, dy = offsets[i]
        label = (p["variant"] if ours else p["method"]) + f"\n{p['ms']:.1f} ms"
        ax.annotate(
            label,
            (p["ms"], p["R5"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=7,
            color=OURS_TEXT if ours else INK,
            linespacing=1.3,
        )
    for j in [0, 2]:
        a, b = points[j : j + 2]
        col = colors[j]
        axs[0].annotate(
            "",
            xy=(b["ms"], b["R5"]),
            xytext=(a["ms"], a["R5"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=1.1,
                shrinkA=5,
                shrinkB=5,
                mutation_scale=8,
            ),
            zorder=2,
        )
        x = (a["ms"] * b["ms"]) ** 0.5
        y = (a["R5"] + b["R5"]) / 2
        axs[0].annotate(
            f"Refine\n+{b['ms'] - a['ms']:.1f} ms",
            (x, y),
            xytext=(-20, 0) if j == 0 else (22, 0),
            textcoords="offset points",
            ha="right" if j == 0 else "left",
            va="center",
            fontsize=7,
            color=INK2,
        )
    for i, off in zip([1, 3, 4, 5], [(-8, 0), (0, -16), (8, 3), (8, 0)]):
        p = points[i]
        col = colors[i]
        marker = "D" if i == 4 else "*" if i == 5 else "o" if i == 1 else "s"
        axs[1].scatter(
            p["MB"],
            p["R5"],
            s=75 if i == 5 else 34,
            marker=marker,
            facecolors="white" if i == 4 else col,
            edgecolors=col,
            lw=0.9,
            zorder=4,
        )
        dx, dy = off
        axs[1].annotate(
            (p["variant"] if i >= 4 else p["method"]) + f"\n{p['MB']:.1f} MB",
            (p["MB"], p["R5"]),
            xytext=off,
            textcoords="offset points",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="center",
            fontsize=7,
            color=OURS_TEXT if i >= 4 else INK,
            linespacing=1.3,
        )
    handles = [
        Line2D(
            [], [], marker="o", ls="", mfc="white", mec=BASE_DARK, ms=5, label="Initial"
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc=BASE_DARK,
            mec=BASE_DARK,
            ms=5,
            label="Refined final",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc="white",
            mec=OURS,
            ms=5,
            label="AnyGSLoc Full",
        ),
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=OURS,
            mec=OURS,
            ms=8,
            label="AnyGSLoc Compact",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.018),
        ncol=4,
        fontsize=7,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    save_figure(fig, output / "E5_deployment")
    plt.close(fig)


def geometry(output, source=None):
    curves = {}
    stats = {}
    if source is None:
        data = np.load(Path(__file__).parent / "geometry_residuals.npz")
        arrays = {k: data[k] for k in ["center", "depth", "ray"]}
    else:
        arrays = {
            k: np.load(source / k / "mapping_residuals.npy")
            for k in ["center", "depth", "ray"]
        }
    n = len(arrays["ray"])
    for k, v in arrays.items():
        if len(v) != n or np.isnan(v).any():
            raise ValueError("inconsistent paired residual population")
        curves[k] = np.sort(v)
        stats[k] = {"N": n, "median_px": float(np.median(v))}
    matplotlib_style()
    fig, ax = plt.subplots(figsize=(88 / 25.4, 65 / 25.4))
    fig.subplots_adjust(left=0.16, right=0.954, bottom=0.18, top=0.98)
    grid = np.geomspace(0.01, 2000, 3001)
    for k, label, col, ls in zip(
        curves,
        ["Gaussian centers", "Rendered depth", "Multiview triangulation"],
        ["#7D8C99", BASE_DARK, OURS],
        ["--", "-.", "-"],
    ):
        ax.plot(
            grid,
            np.searchsorted(curves[k], grid, side="right") / n * 100,
            color=col,
            ls=ls,
            lw=1.7 if k == "ray" else 1.2,
            label=f"{label} ({stats[k]['median_px']:.2f} px)",
        )
        ax.plot(
            stats[k]["median_px"],
            50,
            "o",
            ms=3.5,
            mfc="white",
            mec=col,
            mew=1,
            zorder=4,
        )
    ax.axhline(50, color="#AEB8BF", lw=0.5, ls=":", zorder=0)
    ax.set(
        xscale="log",
        xlim=(0.01, 2000),
        ylim=(0, 100),
        xticks=[0.1, 1, 10, 100, 1000],
        yticks=[0, 25, 50, 75, 100],
    )
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.set_xlabel(
        "Mapping-view reprojection residual (px, log)", fontsize=7.5, labelpad=3
    )
    ax.set_ylabel("Observations within residual (%)", fontsize=7.5, labelpad=3)
    ax.legend(
        loc="lower right",
        fontsize=7,
        handlelength=2.2,
        borderaxespad=0.5,
        labelspacing=0.6,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
    )
    save_figure(fig, output / "E2_geometry_source_evidence")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        type=Path,
        help="New geometry stage folder; omit for accepted St Mary residuals",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deployment(json.loads(args.metrics.read_text()), args.output)
    geometry(args.output, args.geometry)


if __name__ == "__main__":
    main()
