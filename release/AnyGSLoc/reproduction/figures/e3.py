"""Recompose the accepted qualitative figure from its supplied display assets."""

import hashlib
import json
import resource
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from .style import (
    ESTIMATE,
    GT,
    INK,
    INK2,
    MUTED,
    OURS_TEXT,
    matplotlib_style,
)

GRAY = MUTED
EST = ESTIMATE
GREEN = GT
MAGENTA = ESTIMATE
SCENES = ["OldHospital", "KingsCollege"]
NAMES = {"OldHospital": "Old Hospital", "KingsCollege": "King's College"}
STEM = "E3_compact_map"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def panel(ax, data, kind, crop, zoom):
    uv, rgb = data["uv"], data["rgb"]
    mask = data["visible"].copy()
    x0, y0, x1, y1 = crop
    mask &= (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    ax.set_facecolor("#f6f8f9")
    if kind == "compact":
        mask &= data["keep"]
    ax.scatter(
        uv[mask, 0],
        uv[mask, 1],
        s=1.2 if zoom else 0.65,
        alpha=0.85,
        c=rgb[mask],
        lw=0,
        rasterized=True,
    )
    ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#c5ced5")
        sp.set_linewidth(0.5)


def draw(p, OUT):
    A = OUT / "assets"
    matplotlib_style()
    height = 5.55
    fig = plt.figure(figsize=(180 / 25.4, height))
    xs = [0.045, 0.545]
    width = 0.410
    for x, label in zip(xs, ["Full map", "Compact map (ours)"]):
        fig.text(
            x + width / 2,
            0.983,
            label,
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
            color=INK if x == xs[0] else OURS_TEXT,
        )
    fig.text(
        0.25,
        0.948,
        "Descriptor PCA (shared basis)",
        ha="center",
        va="top",
        fontsize=7,
        color=INK2,
    )
    fig.text(
        0.75, 0.948, "Retained landmarks", ha="center", va="top", fontsize=7, color=INK2
    )
    for row, s in enumerate(SCENES):
        m = p["displayed_scenes"][s]
        a = np.load(A / (s + "_display.npz"))
        n, k = m["full_count"], m["compact_count"]
        top = [5.10, 2.43][row] / height
        fig.text(
            0.045,
            top,
            f"({'ab'[row]}) {NAMES[s]}",
            va="top",
            fontsize=8.5,
            weight="bold",
        )
        fig.text(
            0.955,
            top,
            f"Complete map: {n:,} → {k:,}  (−{100 * (1 - k / n):.1f}%)",
            ha="right",
            va="top",
            fontsize=7.5,
            color=OURS_TEXT,
        )
        for col, kind in enumerate(["full", "compact"]):
            bottom, high = [(3.83, 1.08), (1.40, 0.89)][row]
            ax = fig.add_axes([xs[col], bottom / height, width, high / height])
            panel(ax, a, kind, m["display_crop"], False)
            x0, y0, x1, y1 = m["roi_box"]
            ax.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=0.75)
            )
            main = ax
            bottom, high = [(2.67, 1.11), (0.34, 1.00)][row]
            ax = fig.add_axes([xs[col], bottom / height, width, high / height])
            panel(ax, a, kind, m["roi_box"], True)
            for sp in ax.spines.values():
                sp.set_color(INK)
                sp.set_linewidth(0.75)
            for aa, bb in [((x0, y1), (x0, y0)), ((x1, y1), (x1, y0))]:
                fig.add_artist(
                    ConnectionPatch(
                        aa,
                        bb,
                        "data",
                        "data",
                        axesA=main,
                        axesB=ax,
                        color="#8f9ba4",
                        lw=0.45,
                        ls=(0, (2, 1.5)),
                    )
                )
            count = m["roi_full_count"] if kind == "full" else m["roi_compact_count"]
            ax.text(
                0.012,
                0.955,
                f"Detail: {count:,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color=INK if kind == "full" else OURS_TEXT,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9),
            )
    fig.text(
        0.5,
        0.038,
        "All five scenes:  560,996 → 121,379 landmarks  (−78.36%)",
        ha="center",
        va="center",
        fontsize=8,
        color=OURS_TEXT,
        weight="bold",
    )

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    outside = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(ren)
            if (
                b.x0 < -0.5
                or b.y0 < -0.5
                or b.x1 > fig.bbox.width + 0.5
                or b.y1 > fig.bbox.height + 0.5
            ):
                outside.append(t.get_text())
    assert not outside, outside
    for ext in ["pdf", "svg", "png"]:
        args = (
            {"metadata": {"CreationDate": None, "ModDate": None}}
            if ext == "pdf"
            else {"metadata": {"Date": None}}
            if ext == "svg"
            else {}
        )
        fig.savefig(OUT / (STEM + "." + ext), dpi=400 if ext != "png" else 200, **args)
    plt.close(fig)
    p["outputs"] = {ext: sha(OUT / (STEM + "." + ext)) for ext in ["pdf", "svg", "png"]}
    p["script_sha256"] = sha(__file__)
    p["max_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    p["render_validation"] = {
        "text_outside_canvas": outside,
        "minimum_font_pt": 7,
        "visual_inspection": "pending",
    }
    (OUT / (STEM + ".provenance.json")).write_text(json.dumps(p, indent=2) + "\n")
    caption = "Current Full and Compact maps with matched camera projections and architectural detail crops. In Compact, removed landmarks are gray and retained landmarks retain the shared descriptor-PCA colors. PCA is fitted once on all 157,758 current Full descriptors in the two displayed scenes; components and Full-population 1st/99th percentile color bounds are reused for Compact. Each pair shares camera, crop, point size, and a positive-depth image-frustum filter; no rendered-depth or occlusion test is applied. Complete-map counts include every stored landmark, while detail counts include only projections inside the outlined regions. Across all five scenes, selection reduces 560,996 landmarks to 121,379 (78.36% by total count, not a mean of scene percentages). This is a projection/structure illustration, not a denoising or visibility guarantee."
    (OUT / (STEM + ".caption.txt")).write_text(caption + "\n")
    print(
        json.dumps(
            {
                "outputs": p["outputs"],
                "totals": p["totals"],
                "scenes": {
                    s: {k: v for k, v in m.items() if k.endswith("_count")}
                    for s, m in p["displayed_scenes"].items()
                },
                "max_rss_MB": p["max_rss_MB"],
            }
        )
    )
