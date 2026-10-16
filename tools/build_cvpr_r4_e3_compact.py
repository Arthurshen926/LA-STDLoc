"""R4 compact-map readability revision using unchanged, cached display arrays.

No source map, visibility mask, ROI, projection, or PCA is recomputed. Both
columns retain the original point size, camera, crop, visibility and colors.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "paper/experiments_cvpr_r3_20261003"
OUT = ROOT / "paper/experiments_cvpr_r4_20261003"
ASSETS = OUT / "assets/e3"
STEM = "E3_compact_map"
SCENES = ["OldHospital", "KingsCollege"]
NAMES = {"OldHospital": "Old Hospital", "KingsCollege": "King's College"}
INK, MUTED, TEAL = "#22354A", "#657486", "#087F72"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def panel(ax, data, kind, crop, zoom):
    uv, rgb = data["uv"], data["rgb"]
    mask = data["visible"].copy()
    mask &= (uv[:, 0] >= crop[0]) & (uv[:, 0] < crop[2]) & (uv[:, 1] >= crop[1]) & (uv[:, 1] < crop[3])
    if kind == "compact":
        mask &= data["keep"]
    ax.scatter(uv[mask, 0], uv[mask, 1], s=1.05 if zoom else .50,
               c=rgb[mask], lw=0, rasterized=True)
    ax.set(xlim=(crop[0], crop[2]), ylim=(crop[3], crop[1]), aspect="equal")
    ax.axis("off")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    source_provenance = PREV / f"{STEM}.provenance.json"
    p = json.loads(source_provenance.read_text())
    original_assets = {scene: dict(p["display_assets"][scene]) for scene in SCENES}
    for scene in SCENES:
        entry = original_assets[scene]
        assert sha(entry["path"]) == entry["sha256"]
        target = ASSETS / Path(entry["path"]).name
        shutil.copy2(entry["path"], target)
        assert sha(target) == entry["sha256"]
        p["display_assets"][scene] = {"path": str(target), "sha256": sha(target)}
    basis_source = Path(p["pca"]["basis_path"])
    assert sha(basis_source) == p["pca"]["basis_sha256"]
    basis_target = ASSETS / basis_source.name
    shutil.copy2(basis_source, basis_target)
    assert sha(basis_target) == p["pca"]["basis_sha256"]
    p["pca"]["basis_path"] = str(basis_target)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.1, 5.25))
    xs, width = [.045, .545], .410
    for x, label in zip(xs, ["Full landmark map", "Compact landmark map"]):
        fig.text(x + width / 2, .982, label, ha="center", va="top", fontsize=9, weight="bold")
    fig.text(.5, .952, "Shared descriptor PCA", ha="center", va="top", fontsize=7.2, color=MUTED)
    for row, scene in enumerate(SCENES):
        meta = p["displayed_scenes"][scene]
        data = np.load(p["display_assets"][scene]["path"])
        stats = p["all_scene_results"][scene]
        n, k = stats["full"]["landmarks"], stats["compact"]["landmarks"]
        assert int(data["keep"].sum()) == k
        assert len(data["keep"]) == n
        assert int(data["roi"].sum()) == meta["roi_full_count"]
        assert int((data["roi"] & data["keep"]).sum()) == meta["roi_compact_count"]
        top = .924 - row * .435
        fig.text(.045, top, NAMES[scene], va="top", fontsize=8.5, weight="bold")
        fig.text(.955, top, f"Whole map: {n:,} → {k:,}  (−{100*(1-k/n):.1f}%)",
                 va="top", ha="right", fontsize=8, color=TEAL)
        for col, kind in enumerate(["full", "compact"]):
            ax = fig.add_axes([xs[col], top - .235, width, .20])
            panel(ax, data, kind, meta["display_crop"], False)
            x0, y0, x1, y1 = meta["roi_box"]
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=INK, lw=.70))
            ax = fig.add_axes([xs[col], top - .387, width, .135])
            panel(ax, data, kind, meta["roi_box"], True)
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="#C2CCD5", lw=.60))
            label = f"Detail: {meta['roi_full_count']:,} landmarks" if col == 0 else f"Detail: {meta['roi_compact_count']:,} landmarks · {100*meta['roi_retention']:.1f}% retained"
            fig.text(xs[col] + width/2, top - .395, label, ha="center", va="top", fontsize=7.5, color=MUTED)

    total = p["totals_all_five_scenes"]
    fn, cn = total["full"]["landmarks"], total["compact"]["landmarks"]
    prefix = TextArea(f"All five scenes:  {fn:,} → {cn:,} landmarks  (", textprops={"fontsize": 9, "color": INK})
    emphasis = TextArea(f"−{100*(1-cn/fn):.1f}%", textprops={"fontsize": 9, "color": TEAL, "weight": "bold"})
    suffix = TextArea(")", textprops={"fontsize": 9, "color": INK})
    line = HPacker(children=[prefix, emphasis, suffix], align="baseline", pad=0, sep=0)
    footer = AnchoredOffsetbox(loc="center", child=line, frameon=False, pad=0,
                               bbox_to_anchor=(.5, .034), bbox_transform=fig.transFigure, borderpad=0)
    fig.add_artist(footer)

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
        fig.savefig(path, dpi=400)
        outputs[ext] = {"path": str(path), "sha256": sha(path)}
    plt.close(fig)
    outputs["render_validation"] = {"text_outside_canvas": outside, "minimum_declared_font_pt": min(t.get_fontsize() for t in texts)}
    p["outputs"] = {STEM: outputs}
    p["revision"] = 4
    p.pop("unchanged_from_revision_2", None)
    p["previous_figure_provenance"] = {"path": str(source_provenance), "sha256": sha(source_provenance)}
    p["source_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    p["r4_data_unchanged"] = True
    p["r4_display_assets_byte_identical_to_r3"] = True
    p["r4_pca_basis_byte_identical_to_r3"] = True
    p["r2_figure_changes"] = p.pop("new_figure_changes", [])
    p["new_figure_changes"] = ["Added a concise shared-descriptor-PCA explanation.",
                                "Used publication scene names and reduced footer emphasis to the reduction percentage.",
                                "Connected the visual comparison to the existing map-selection accuracy trade-off in the caption only."]
    p["existing_ablation_references"] = {
        "macro": {"path": str(ROOT / "paper/AnyGSLoc_TRO_v6/tables/selection.tex"), "label": "tab:selection"},
        "per_scene": {"path": str(ROOT / "paper/AnyGSLoc_TRO_v6/tables/selection_scenes.tex"), "label": "tab:selection-scenes", "scope": "archive"},
        "caption_reference_policy": "Natural-language reference to map-selection ablation; no potentially unresolved LaTeX reference is added to the standalone figure caption.",
    }
    p["visual_check"] = "Original matched map panels and detail strips remain unchanged; new scene/PCA labels and lighter footer stay within the canvas."
    caption = (PREV / f"{STEM}.caption.txt").read_text()
    transition = "The visualization supports sparsity and structure comparison, not a claim of denoising or uniform localization improvement. "
    assert transition in caption
    kc, oh = p["all_scene_results"]["KingsCollege"], p["all_scene_results"]["OldHospital"]
    tradeoff = (f"The existing map-selection ablation quantifies the localization trade-off: R5 changes from {kc['full']['R5_pct']:.2f}% to {kc['compact']['R5_pct']:.2f}% on King's College and "
                f"from {oh['full']['R5_pct']:.2f}% to {oh['compact']['R5_pct']:.2f}% on Old Hospital (Full to Compact). ")
    caption = caption.replace(transition, transition + tradeoff)
    (OUT / f"{STEM}.caption.txt").write_text(caption)
    (OUT / f"{STEM}.provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps({"figure": STEM, "data_unchanged": True, "canvas_inches": [7.1, 5.25], "minimum_font_pt": outputs["render_validation"]["minimum_declared_font_pt"]}))


if __name__ == "__main__":
    main()
