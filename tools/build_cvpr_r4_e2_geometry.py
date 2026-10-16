"""R4 readability revision of the archived, paired mapping-residual ECDF.

Only presentation changes. Residuals are copied byte-for-byte from the audited
R3 source asset; no projection, matching, geometry, or localization is rerun.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "paper/experiments_cvpr_r3_20261003"
OUT = ROOT / "paper/experiments_cvpr_r4_20261003"
ASSETS = OUT / "assets/e2"
STEM = "E2_geometry_source_evidence"
KINDS = ["center", "depth", "ray"]
COLORS = {"center": "#C28A37", "depth": "#427AB0", "ray": "#087F72"}
LABELS = {"center": "Gaussian-center aggregate", "depth": "Rendered-depth aggregate", "ray": "Multiview rays"}
INK, MUTED = "#22354A", "#667587"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    source_provenance = PREV / f"{STEM}.provenance.json"
    p = json.loads(source_provenance.read_text())
    source_asset = Path(p["residual_asset"]["path"])
    expected = p["residual_asset"]["sha256"]
    assert sha(source_asset) == expected
    asset = ASSETS / source_asset.name
    shutil.copy2(source_asset, asset)
    assert sha(asset) == expected
    data = np.load(asset)
    assert all(len(data[k]) == p["population_observations"] for k in KINDS)
    assert np.unique(data["query_indices"]).size == p["contributing_mapping_views"] == 1465
    assert p["archive_mapping_views"] == 1487
    assert np.unique(data["anchor_rows"]).size == p["population_landmarks"] == 55817

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(3.45, 2.4))
    fig.text(.53, .956, "St. Mary's Church", ha="center", va="top", fontsize=8.2)
    ax = fig.add_axes([.16, .205, .80, .65])
    grid = np.geomspace(.05, 1000, 2501)
    curve_values = {}
    for key, linestyle in zip(KINDS, ["--", "-.", "-"]):
        values = np.sort(data[key])
        assert not np.isnan(values).any()
        assert float(np.median(values)) == p["statistics"][key]["median_px"]
        pct = np.searchsorted(values, grid, side="right") / len(values) * 100
        curve_values[key] = pct
        ax.plot(grid, pct, color=COLORS[key], lw=1.55, ls=linestyle, label=LABELS[key])
    ax.set(xscale="log", xlim=(.05, 1000), ylim=(0, 100),
           xticks=[.1, 1, 10, 100, 1000], yticks=[0, 25, 50, 75, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.grid(axis="both", lw=.45, color="#E1E6EC")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=7)
    ax.set_xlabel("Reprojection error (px)", fontsize=8, labelpad=2)
    ax.set_ylabel("Cumulative observations (%)", fontsize=8, labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.2,
              labelspacing=.5, borderpad=.15)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            outside.append(text.get_text())
    assert not outside, outside
    outputs = {}
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{STEM}.{ext}"
        fig.savefig(path, dpi=360)
        outputs[ext] = {"path": str(path), "sha256": sha(path)}
    plt.close(fig)
    outputs["text_outside_canvas"] = outside
    outputs["minimum_declared_font_pt"] = min(t.get_fontsize() for t in texts)
    np.savez_compressed(ASSETS / "displayed_ecdf.npz", grid_px=grid, **curve_values)
    p["outputs"] = {STEM: outputs}
    p["revision"] = 4
    p["previous_figure_provenance"] = {"path": str(source_provenance), "sha256": sha(source_provenance)}
    p["residual_asset"] = {"path": str(asset), "sha256": sha(asset)}
    p["source_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    p["r4_changes"] = [
        "Shortened and lightened the scene title; observation count remains in the caption.",
        "All residual values, evaluated observations, ECDF grid, denominator, axes, colors, and line styles are unchanged.",
    ]
    p["r4_data_unchanged"] = True
    p["r4_residual_asset_byte_identical_to_r3"] = True
    p["visual_check"] = "Labels stay within the original single-column canvas; the shortened title leaves the original axes and curves unchanged."
    caption = (PREV / f"{STEM}.caption.txt").read_text().replace("in StMarysChurch.", "in St. Mary's Church.")
    (OUT / f"{STEM}.caption.txt").write_text(caption)
    (OUT / f"{STEM}.provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps({"figure": STEM, "data_unchanged": True, "canvas_inches": [3.45, 2.4], "minimum_font_pt": outputs["minimum_declared_font_pt"]}))


if __name__ == "__main__":
    main()
