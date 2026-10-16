"""Recompose the accepted qualitative figure from its supplied display assets."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .composition import diagonal, source_edges, source_legend
from .style import (
    ESTIMATE,
    GT,
    INK,
    INK2,
    MUTED,
    OURS,
    matplotlib_style,
    save_figure,
)

GRAY = MUTED
EST = ESTIMATE
GREEN = GT
MAGENTA = ESTIMATE
KEYS = ["vanilla_3dgs", "matcha", "vanilla_2dgs", "i3dgs_raw", "i3dgs_1k"]
LABELS = ["3DGS", "MAtCha", "2DGS", "Raw", "+1k updates"]
SCENE_LABELS = {"StMarysChurch": "St. Mary's Church", "ShopFacade": "Shop Facade"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def draw(OUT):
    A = OUT / "assets"
    P = OUT / "E4_gaussian_prior.provenance.json"
    import matplotlib

    matplotlib.use("Agg")
    matplotlib_style()
    d = read(P)
    psnr = {
        "scenes": {
            scene: {"PSNR_dB": values} for scene, values in d["current_PSNR"].items()
        }
    }
    from .style import RULE, chip

    fw = 180 / 25.4
    margin, qwidth, gap = 0.02, 0.98, 0.06
    pw = (fw - 2 * margin - qwidth - 0.10 - 4 * gap) / 5
    xs = [margin] + [margin + qwidth + 0.10 + i * (pw + gap) for i in range(5)]
    widths = [qwidth] + [pw] * 5
    ih = pw * 9 / 16
    mh = ih * 0.9
    ah = pw * 150 / 240
    head = 0.36
    block = 0.20 + ih + 0.05 + mh + 0.05 + ah + 0.20
    legend = 0.30
    fh = head + 2 * block + 0.10 + legend
    fig = plt.figure(figsize=(fw, fh))

    def text_in(x, y, t, **kw):
        return fig.text(x / fw, 1 - y / fh, t, **kw)

    def panel(x, top, w, h):
        return fig.add_axes([x / fw, 1 - (top + h) / fh, w / fw, h / fh])

    # Column headers; the i3DGS pair shares one group header.
    for j, label in enumerate(["Real query"] + LABELS):
        sub = j >= 4
        text_in(
            xs[j] + widths[j] / 2,
            0.30,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5 if sub else 8,
            weight="normal" if sub else "bold",
            color=INK2 if sub else INK,
        )
    text_in(
        (xs[4] + xs[5] + pw) / 2,
        0.11,
        "i3DGS",
        fontsize=8,
        weight="bold",
        ha="center",
        va="bottom",
    )
    fig.add_artist(
        plt.Line2D(
            [(xs[4] + 0.02) / fw, (xs[5] + pw - 0.02) / fw],
            [1 - 0.15 / fh] * 2,
            transform=fig.transFigure,
            color=RULE,
            lw=0.5,
        )
    )
    composition_checks = {}
    for ci, cid in enumerate(d["selected"]):
        item = next(v for v in d["candidates"] if v["id"] == cid)
        scene = item["scene"]
        top = head + ci * (block + 0.10)
        if ci:
            fig.add_artist(
                plt.Line2D(
                    [margin / fw, (fw - margin) / fw],
                    [1 - (top - 0.06) / fh] * 2,
                    transform=fig.transFigure,
                    color=RULE,
                    lw=0.5,
                )
            )
        text_in(
            margin,
            top + 0.14,
            f"({chr(97 + ci)}) {SCENE_LABELS[scene]}",
            fontsize=8.5,
            weight="bold",
            va="bottom",
        )
        tops = [top + 0.20, top + 0.20 + ih + 0.05, top + 0.20 + ih + 0.05 + mh + 0.05]
        ax = panel(xs[0], tops[0] + (ih - qwidth * 9 / 16) / 2, qwidth, qwidth * 9 / 16)
        ax.imshow(Image.open(A / (cid + "_real_query.png")))
        ax.axis("off")
        roi = item["alignment"]["crop_native_px"]
        ax.add_patch(
            Rectangle(
                roi[:2],
                roi[2] - roi[0],
                roi[3] - roi[1],
                fill=False,
                ec="white",
                lw=0.7,
            )
        )
        # Row labels live in the query column, aligned to their rows.
        text_in(
            xs[0] + qwidth,
            tops[1] + mh * 0.38,
            "Compact map",
            ha="right",
            va="center",
            fontsize=7.5,
            weight="bold",
        )
        gt = np.asarray(Image.open(A / (cid + "_gt_reference_crop.png")).convert("RGB"))
        ax = panel(
            xs[0], tops[2] + (ah - qwidth * 150 / 240) / 2, qwidth, qwidth * 150 / 240
        )
        ax.imshow(gt)
        ax.axis("off")
        ax.plot([0, gt.shape[1] - 1], [0, 0], color=GREEN, lw=0.8, clip_on=False)
        chip(
            ax,
            0.03,
            0.06,
            "GT-pose render",
            ax.transAxes,
            color=INK,
            ha="left",
            va="bottom",
        )
        composition_checks[cid] = {}
        best = min(KEYS, key=lambda k: item["results"][k]["translation_error_cm"])
        for j, key in enumerate(KEYS, 1):
            src = d["scenes"][scene]["priors"][key]
            x = xs[j]
            ax = panel(x, tops[0], pw, ih)
            ax.imshow(Image.open(A / (cid + "_" + key + "_prior.png")))
            ax.axis("off")
            chip(
                ax,
                0.03,
                0.06,
                f"{psnr['scenes'][scene]['PSNR_dB'][key]:.2f} dB",
                ax.transAxes,
                color=INK,
                ha="left",
                va="bottom",
            )
            m = np.load(A / (cid + "_" + key + "_map.npz"))
            order = np.argsort(m["depth"])[::-1]
            ax = panel(x, tops[1], pw, mh)
            ax.scatter(
                m["uv"][order, 0],
                m["uv"][order, 1],
                c=m["colors"][order],
                s=0.42,
                lw=0,
                rasterized=True,
            )
            ax.set(xlim=(0, item["width"]), ylim=(item["height"], 0), aspect="equal")
            ax.axis("off")
            chip(
                ax,
                0.97,
                0.06,
                f"N = {src['landmark_count'] / 1000:.1f}k",
                ax.transAxes,
                color=INK2,
                ha="right",
                va="bottom",
            )
            x0, y0, x1, y1 = item["alignment"]["crop_render_px"]
            pred = np.asarray(
                Image.open(A / (cid + "_" + key + "_reference_full_2880.png")).convert(
                    "RGB"
                )
            )[y0:y1, x0:x1]
            mixed, mask = diagonal(gt, pred)
            Image.fromarray(mixed).save(A / (cid + "_" + key + "_diagonal.png"))
            ax = panel(x, tops[2], pw, ah)
            ax.imshow(mixed, interpolation="nearest")
            ax.axis("off")
            source_edges(ax, mixed.shape[1], mixed.shape[0], GREEN, MAGENTA)
            r = item["results"][key]
            is_best = key == best
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        ec=OURS,
                        lw=1.6,
                        clip_on=False,
                        zorder=10,
                    )
                )
            text_in(
                x + pw / 2,
                tops[2] + ah + 0.04,
                f"{r['translation_error_cm']:.2f} cm / {r['rotation_error_deg']:.3f}°",
                ha="center",
                va="top",
                fontsize=7.5,
                weight="bold" if is_best else "normal",
                color=INK,
            )
            composition_checks[cid][key] = {
                "exact_GT_triangle": bool(np.array_equal(mixed[~mask], gt[~mask])),
                "exact_estimate_triangle": bool(
                    np.array_equal(mixed[mask], pred[mask])
                ),
            }
    # Exactly the same physical-size legend and splice grammar as E1.
    source_legend(fig)
    save_figure(fig, OUT / "E4_gaussian_prior")
    plt.close(fig)
    caption = "AnyGSLoc with different Gaussian scene priors. Top: each input prior rendered at the same calibrated ground-truth camera. Reference-view PSNR (dB, higher is better) is the mean over all registered reference views of the corresponding scene (1,487 for St. Mary's Church; 231 for Shop Facade), rather than the single displayed image. All priors share the evaluation views, 1920 x 1080 resolution and full-RGB pixel protocol. Middle: the resulting compact landmark maps; N denotes the complete scene-map size. Bottom: ground-truth and estimated poses rendered with the same Gaussian model in each scene. The GT render occupies the upper triangle (blue upper edge), and the estimated-pose render the lower triangle (orange lower edge); the lowest translation error per scene is outlined. All priors share the crop and the 0.40 pt neutral diagonal boundary; the splice legend is identical to E1. Labels report translation / rotation error. The i3DGS columns compare the raw prior with the prior after 1,000 optimization updates; these are upstream scene-construction variants. Quantitative results across scenes are reported in the prior comparison table."
    (OUT / "E4_gaussian_prior.caption.txt").write_text(caption + "\n")
    d["revision"] = "20261014-current-priors"
    d["diagonal_composition_checks"] = composition_checks
    d["reference_view_PSNR"] = {
        "sources": d["current_PSNR_sources"],
        "scene_means": d["current_PSNR"],
        "split": "registered reference views",
        "figure_integration_done": True,
    }
    d["caption"] = caption
    d["outputs"] = {
        e: {
            "path": str(OUT / ("E4_gaussian_prior." + e)),
            "sha256": digest(OUT / ("E4_gaussian_prior." + e)),
        }
        for e in ["pdf", "svg", "png"]
    }
    d["revision_layout"] = {
        "size_inches": [fw, fh],
        "ordinary_font_pt": 7.5,
        "map_point_area_pt2": 0.30,
        "map_point_rule_shared_across_all_priors": True,
        "input_priors_and_crops_unchanged_v2_maps_and_poses_updated": True,
        "font": "Nimbus Sans",
        "composition": "Exact diagonal; GT upper-right, estimate lower-left; 0.40 pt neutral vector seam",
        "diagonal_line_width_pt": 0.40,
        "PSNR": "Verified reference-view scene mean below each input-prior render",
    }
    P.write_text(json.dumps(d, indent=2))
