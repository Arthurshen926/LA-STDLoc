"""R5 compact-map structure comparison using unchanged, cached display arrays.

No source map, visibility mask, projection, or PCA is recomputed. Architectural
ROIs replace the old density-maximizing ROIs. Every display rule is shared by
Full and Compact, including the revised point sizes and fixed ROI rectangles.
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
PREV = ROOT / "paper/experiments_cvpr_r4_20261003"
OUT = ROOT / "paper/experiments_cvpr_r5_20261003"
ASSETS = OUT / "assets/e3"
STEM = "E3_compact_map"
SCENES = ["OldHospital", "KingsCollege"]
NAMES = {"OldHospital": "Old Hospital", "KingsCollege": "King's College"}
INK, MUTED, TEAL = "#22354A", "#657486", "#087F72"
ROIS = {"OldHospital": [840, 290, 1600, 580], "KingsCollege": [1000, 420, 1900, 780]}


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
    ax.scatter(uv[mask, 0], uv[mask, 1], s=.65 if zoom else .60,
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
    figure_height = 5.5
    fig = plt.figure(figsize=(7.1, figure_height))
    xs, width = [.045, .545], .410
    for x, label in zip(xs, ["Full landmark map", "Compact landmark map"]):
        fig.text(x + width / 2, 5.44/figure_height, label, ha="center", va="top", fontsize=9, weight="bold")
    fig.text(.5, 5.27/figure_height, "Shared descriptor PCA", ha="center", va="top", fontsize=7.2, color=MUTED)
    for row, scene in enumerate(SCENES):
        meta = p["displayed_scenes"][scene]
        data = np.load(p["display_assets"][scene]["path"])
        stats = p["all_scene_results"][scene]
        n, k = stats["full"]["landmarks"], stats["compact"]["landmarks"]
        assert int(data["keep"].sum()) == k
        assert len(data["keep"]) == n
        assert int(data["roi"].sum()) == meta["roi_full_count"]
        assert int((data["roi"] & data["keep"]).sum()) == meta["roi_compact_count"]
        old_roi = meta["roi_box"]
        meta["previous_density_roi"] = old_roi
        meta["roi_box"] = ROIS[scene]
        uv = data["uv"]
        x0, y0, x1, y1 = meta["roi_box"]
        roi = data["visible"] & (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        meta["roi_full_count"] = int(roi.sum())
        meta["roi_compact_count"] = int((roi & data["keep"]).sum())
        meta["roi_retention"] = meta["roi_compact_count"] / meta["roi_full_count"]
        np.savez_compressed(ASSETS / f"{scene}_r5_roi.npz", roi=roi, box=np.array(meta["roi_box"]))
        top = [5.10, 2.43][row] / figure_height
        fig.text(.045, top, NAMES[scene], va="top", fontsize=8.5, weight="bold")
        fig.text(.955, top, f"Whole map: {n:,} → {k:,}  (−{100*(1-k/n):.1f}%)",
                 va="top", ha="right", fontsize=8, color=TEAL)
        for col, kind in enumerate(["full", "compact"]):
            main_bottom, main_height = [(3.83, 1.08), (1.40, .89)][row]
            ax = fig.add_axes([xs[col], main_bottom/figure_height, width, main_height/figure_height])
            panel(ax, data, kind, meta["display_crop"], False)
            x0, y0, x1, y1 = meta["roi_box"]
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=INK, lw=.70))
            zoom_bottom, zoom_height = [(2.67, 1.11), (.34, 1.00)][row]
            ax = fig.add_axes([xs[col], zoom_bottom/figure_height, width, zoom_height/figure_height])
            panel(ax, data, kind, meta["roi_box"], True)
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="#C2CCD5", lw=.60))
        fig.text(.5, [2.62, .29][row]/figure_height, f"Detail: {meta['roi_full_count']:,} → {meta['roi_compact_count']:,} landmarks",
                 ha="center", va="top", fontsize=7.5, color=MUTED)

    total = p["totals_all_five_scenes"]
    fn, cn = total["full"]["landmarks"], total["compact"]["landmarks"]
    prefix = TextArea(f"Five scenes:  {fn:,} → {cn:,}  (", textprops={"fontsize": 9, "color": INK})
    emphasis = TextArea(f"−{100*(1-cn/fn):.1f}%", textprops={"fontsize": 9, "color": TEAL, "weight": "bold"})
    suffix = TextArea(")", textprops={"fontsize": 9, "color": INK})
    line = HPacker(children=[prefix, emphasis, suffix], align="baseline", pad=0, sep=0)
    footer = AnchoredOffsetbox(loc="center", child=line, frameon=False, pad=0,
                               bbox_to_anchor=(.5, .085/figure_height), bbox_transform=fig.transFigure, borderpad=0)
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
    p["revision"] = 5
    p.pop("unchanged_from_revision_2", None)
    p["previous_figure_provenance"] = {"path": str(source_provenance), "sha256": sha(source_provenance)}
    p["source_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    p["r5_map_data_unchanged"] = True
    p["r5_source_display_assets_byte_identical_to_r4"] = True
    p["r5_pca_basis_byte_identical_to_r4"] = True
    p["r5_roi_assets"] = {
        scene: {"path": str(ASSETS / f"{scene}_r5_roi.npz"), "sha256": sha(ASSETS / f"{scene}_r5_roi.npz")}
        for scene in SCENES
    }
    p["r5_roi_candidate_review"] = {
        "contact_sheet": str(ASSETS / "roi_candidates.png"),
        "candidate_boxes": str(ASSETS / "roi_candidates.json"),
        "selection": {"OldHospital": "Candidate C, lower blank/ground strip trimmed; complete circular arch contours retained.",
                      "KingsCollege": "Candidate B, window and cornice structure."},
    }
    p["r4_figure_changes"] = p.pop("new_figure_changes", [])
    p["figure_inches"] = [7.1, figure_height]
    p["main_point_area_pt2"] = .60
    p["zoom_point_area_pt2"] = .65
    p["roi_rule"] = "Fixed architectural window/arch rectangles selected by visual inspection of four Full/Compact ROI candidates from the already frozen display arrays. Old Hospital uses candidate C with the lower blank/ground band trimmed while preserving the complete circular arch contours; King's College uses candidate B. Identical crop, visibility, point size, and color rules are applied to both representations. Selection changes only visualization, not any map or localization result."
    p["new_figure_changes"] = ["Replaced densest-point strips with fixed architectural windows and arches to make preserved structure easier to recognize.",
                                "Expanded detail-panel height, used equal Full/Compact point sizes, and kept original visibility and descriptor colors.",
                                "Reported each ROI pair as one count transition without redundant retention percentages; shortened the five-scene footer.",
                                "Linked the visual evidence to the existing overall R5 trade-off and acknowledged scene-specific changes in the caption."]
    p["existing_ablation_references"] = {
        "macro": {"path": str(ROOT / "paper/AnyGSLoc_TRO_v6/tables/selection.tex"), "label": "tab:selection"},
        "per_scene": {"path": str(ROOT / "paper/AnyGSLoc_TRO_v6/tables/selection_scenes.tex"), "label": "tab:selection-scenes", "scope": "archive"},
        "caption_reference_policy": "Natural-language reference to map-selection ablation; no potentially unresolved LaTeX reference is added to the standalone figure caption.",
    }
    p["visual_check"] = "Architectural ROI candidates were reviewed side-by-side; matched window and arch crops replace the old density ROI. Final figure inspected at publication scale."
    full_r5 = float(np.mean([v['full']['R5_pct'] for v in p['all_scene_results'].values()]))
    compact_r5 = float(np.mean([v['compact']['R5_pct'] for v in p['all_scene_results'].values()]))
    p["caption_macro_R5_from_existing_scene_results"] = {"full": full_r5, "compact": compact_r5, "aggregation": "scene-equal mean"}
    caption = ("Full and compact maps retain recognizable window and facade structure at substantially different landmark densities. "
               "Both columns use the same camera, crop, point size, opacity, cached visibility mask, and shared descriptor-PCA colors. "
               "The detail views magnify fixed architectural regions; their count transitions include only landmarks in the displayed common surface band. "
               "Scene labels and the five-scene total count every stored landmark. "
               f"The existing map-selection ablation reports an overall R5 change from {full_r5:.2f}% to {compact_r5:.2f}% as the landmark count falls by 67.1%, with scene-specific accuracy trade-offs. "
               "These views illustrate sparsity and recognizable structure, not denoising or lossless compression. "
               "Only display crops and styling changed; maps, selection, PCA, and localization results are unchanged.")
    caption += "\n\nDisplay protocol and limitations:\n" + "\n".join("- " + s for s in p["limitations"]) + "\n"
    (OUT / f"{STEM}.caption.txt").write_text(caption)
    (OUT / f"{STEM}.provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps({"figure": STEM, "map_data_unchanged": True, "canvas_inches": [7.1, figure_height], "minimum_font_pt": outputs["render_validation"]["minimum_declared_font_pt"]}))


if __name__ == "__main__":
    main()
