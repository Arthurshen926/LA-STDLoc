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
    METHOD_COLORS,
    MUTED,
    OURS,
    matplotlib_style,
    save_figure,
)

GRAY = MUTED
EST = ESTIMATE
GREEN = GT
MAGENTA = ESTIMATE
METHODS = [
    "STDLoc_coarse",
    "STDLoc",
    "ULFLoc_coarse",
    "ULFLoc",
    "AnyGSLoc",
    "AnyGSLoc_compact",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(p, output):
    (output / "E1_final_localization.provenance.json").write_text(
        json.dumps(p, indent=2)
    )


def structural_guide(ax, box):
    """Four short transparent corner brackets; structure remains unobscured."""
    import matplotlib.patheffects as pe

    x0, y0, x1, y1 = box
    arm = 6
    artists = []
    for x, y, dx, dy in [
        (x0, y0, 1, 1),
        (x1, y0, -1, 1),
        (x0, y1, 1, -1),
        (x1, y1, -1, -1),
    ]:
        (line,) = ax.plot(
            [x + dx * arm, x, x],
            [y, y, y + dy * arm],
            color="white",
            linewidth=0.48,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            zorder=5,
            path_effects=[
                pe.Stroke(linewidth=0.95, foreground="#25313c", alpha=0.72),
                pe.Normal(),
            ],
        )
        artists.append(line)
    return artists


def draw(OUT):
    ASSETS = OUT / "assets"
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    p = read(OUT / "source_provenance.json")
    p["revision"] = "20261014-current-3dgs"
    p["scope"] = (
        "Diagonal GT/estimate composition, unified typography and explicit render-source legend; fixed crops; AnyGSLoc poses and errors synchronized to compact-v2."
    )
    for row in p["rows"]:
        for item in row["renders"].values():
            item["path"] = str(ASSETS / Path(item["path"]).name)
            assert digest(item["path"]) == item["sha256"]
        row["local_query_image"] = str(ASSETS / Path(row["local_query_image"]).name)
    matplotlib_style()
    figw, figh = 180 / 25.4, 90 / 25.4
    fig = plt.figure(figsize=(figw, figh))
    left, gap, groupgap = 0.017, 0.011, 0.021
    # A little more space between methods than between each method's stages.
    w = (1 - 2 * left - 3 * gap - 3 * groupgap) / 7
    xs = [left]
    for j in range(6):
        xs.append(xs[-1] + w + (gap if j in [1, 3, 5] else groupgap))
    for a, b, title in [
        (0, 0, "Real query"),
        (1, 2, "STDLoc"),
        (3, 4, "ULF-Loc"),
        (5, 6, "AnyGSLoc"),
    ]:
        fig.text(
            (xs[a] + xs[b] + w) / 2,
            0.982,
            title,
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
            color=METHOD_COLORS["AnyGSLoc"] if title == "AnyGSLoc" else INK,
        )
        if b > a:
            fig.add_artist(
                Line2D(
                    [xs[a], xs[b] + w],
                    [0.941, 0.941],
                    transform=fig.transFigure,
                    color="#bac5cc",
                    lw=0.45,
                )
            )
    for j, title in enumerate(
        [
            "",
            "Initial",
            "Refined (final)",
            "Initial",
            "Refined (final)",
            "Full",
            "Compact",
        ]
    ):
        fig.text(
            xs[j] + w / 2, 0.925, title, ha="center", va="top", fontsize=7.5, color=GRAY
        )
    from matplotlib.patches import FancyBboxPatch

    fig.add_artist(
        FancyBboxPatch(
            (xs[5] - 0.006, 0.075),
            2 * w + gap + 0.012,
            0.915,
            boxstyle="round,pad=0,rounding_size=.006",
            transform=fig.transFigure,
            fc="#eaf3f1",
            ec="none",
            zorder=-1,
        )
    )
    source_legend(fig)
    validations = []
    for i, row in enumerate(p["rows"]):
        row.pop("query_SHA256_matches_local_copy", None)
        row["local_query_sha256"] = digest(row["local_query_image"])
        row["local_query_pixels_match_original"] = bool(
            np.array_equal(
                np.asarray(Image.open(row["query_path"]).convert("RGB")),
                np.asarray(Image.open(row["local_query_image"]).convert("RGB")),
            )
        )
        assert row["local_query_pixels_match_original"]
        archived_stage_checks = []
        for base in ["STDLoc", "ULFLoc"]:
            source = row["result_sources"][base]
            assert digest(source) == p["source_sha256"][source]["sha256"]
            native = read(source)[row["methods"][base]["native_result_index"]]
            for method, stage, tekey, rekey in [
                (base + "_coarse", native["sparse"], "sparse_TE", "sparse_AE"),
                (base, native["dense"][-1], "dense_TE", "dense_AE"),
            ]:
                record = row["methods"][method]
                assert np.array_equal(
                    np.array(record["pose_w2c"]), np.array(stage["pose_w2c"])
                )
                assert (
                    record["TE_cm"] == native[tekey]
                    and record["RE_deg"] == native[rekey]
                )
                archived_stage_checks.append(method)
        ims = {
            m: np.array(Image.open(row["renders"][m]["path"]).convert("RGB"))
            for m in ["GT"] + METHODS
        }
        x0, y0, x1, y1 = row["crop"]
        ch, cw = y1 - y0, x1 - x0
        gt = ims["GT"][y0:y1, x0:x1]
        top = 0.873 - i * 0.275
        h = w * figw / figh * ch / cw
        bottom = top - h
        ax = fig.add_axes([xs[0], bottom, w, h])
        ax.imshow(Image.open(row["local_query_image"]))
        ax.axis("off")
        native = (
            np.array(row["crop"])
            * row["camera"]["width"]
            / row["render_metadata"]["width"]
        )
        ax.add_patch(
            Rectangle(
                native[:2],
                native[2] - native[0],
                native[3] - native[1],
                fill=False,
                ec="white",
                lw=0.8,
            )
        )
        fig.text(
            xs[0] + w / 2,
            bottom - 0.014,
            f"({chr(97 + i)}) "
            + row["display_name"].replace("St. Mary's Church", "St. Mary's\nChurch"),
            ha="center",
            va="top",
            fontsize=7.5,
            weight="bold",
        )
        exact, te_errors, re_errors = [], [], []
        best = min(METHODS, key=lambda m: row["methods"][m]["TE_cm"])
        for j, method in enumerate(METHODS, 1):
            est = ims[method][y0:y1, x0:x1]
            composed, mask = diagonal(gt, est)
            band_path = ASSETS / f"{row['stem']}_{method}_diagonal.png"
            Image.fromarray(composed).save(band_path)

            ax = fig.add_axes([xs[j], bottom, w, h])
            ax.imshow(composed)
            ax.set_xlim(0, cw)
            ax.set_ylim(ch, 0)
            ax.axis("off")
            source_edges(ax, cw, ch, GT, EST)
            exact.append(
                np.array_equal(composed[~mask], gt[~mask])
                and np.array_equal(composed[mask], est[mask])
            )
            metric = row["methods"][method]
            # Compact per-example errors remain readable at the two-column print width.
            from .style import chip

            is_best = method == best
            chip(
                ax,
                0.5,
                -0.12,
                f"{metric['TE_cm']:.2f} cm · {metric['RE_deg']:.3f}°",
                ax.transAxes,
                color=INK,
                weight="bold" if is_best else "normal",
                va="top",
                size=7,
            )
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
            g, e = np.array(row["camera"]["gt_pose_w2c"]), np.array(metric["pose_w2c"])
            te = np.linalg.norm(np.linalg.inv(e)[:3, 3] - np.linalg.inv(g)[:3, 3]) * 100
            re = np.rad2deg(
                np.arccos(np.clip((np.trace(e[:3, :3] @ g[:3, :3].T) - 1) / 2, -1, 1))
            )
            te_errors.append(abs(te - metric["TE_cm"]))
            re_errors.append(abs(re - metric["RE_deg"]))
        assert all(exact)
        assert max(te_errors) < 0.02
        assert (
            max(re_errors) < 0.005
        )  # Archived float32 rotation matrices permit small arccos differences.
        validations.append(
            {
                "scene": row["scene"],
                "exact_unwarped_source_pixels_both_triangles": all(exact),
                "diagonal_mask_same_for_every_method": True,
                "shared_structural_guide": row["structural_guide"],
                "recomputed_TE_max_error_cm": max(te_errors),
                "recomputed_RE_max_error_deg": max(re_errors),
                "native_results_hash_pose_and_TE_RE_exact_match": archived_stage_checks,
            }
        )
    save_figure(fig, OUT / "E1_final_localization")
    plt.close(fig)
    caption = "Qualitative localization comparison on Cambridge Landmarks. Errors are overlaid on each panel; the lowest translation error per scene is outlined. STDLoc and ULF-Loc show initial and refined (final) estimates; AnyGSLoc produces a single-stage estimate. The real query marks the common crop. For each scene, the same Gaussian model is rendered at the ground-truth and estimated poses. Each diagonal splice uses the GT render above the diagonal (green upper edge) and the estimated-pose render below it (magenta lower edge). A thin neutral line marks the splice. All methods share the crop, seam direction and line width; structural continuity across the seam indicates pose alignment. Numbers report translation error (cm) / rotation error (degrees)."
    limitations = p.get("limitations", [])
    p.update(validation=validations, caption=caption, limitations=limitations)
    p["display"] = {
        "figure_inches": [figw, figh],
        "minimum_ordinary_font_pt": 7.5,
        "native_crop_pixels": [320, 240],
        "same_queries_and_ROIs": True,
        "layout": "3 aligned rows × 6 columns; grouped headers and a single shared footer legend",
        "composition": "GT above the top-left to bottom-right diagonal; estimate below; exact source pixels; 0.40 pt neutral vector seam",
        "source_schematic": "Shared E1/E4 physical-size legend: GT upper-right / estimate lower-left triangles.",
        "diagonal_line_width_pt": 0.40,
        "ROIs_unchanged_v2_pose_renders_updated": True,
        "font": "Nimbus Sans",
        "palette": METHOD_COLORS,
        "new_vector_overlay": "Thin outer source edges and a 0.40 pt neutral diagonal; mixed raster source pixels remain unchanged.",
        "scene_display_names": {r["scene"]: r["display_name"] for r in p["rows"]},
    }
    p["source_script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": digest(__file__),
    }
    p["outputs"] = {
        ext: {
            "path": str(OUT / f"E1_final_localization.{ext}"),
            "sha256": digest(OUT / f"E1_final_localization.{ext}"),
        }
        for ext in ["pdf", "svg", "png"]
    }
    (OUT / "E1_final_localization.caption.txt").write_text(caption + "\n")
    save(p, OUT)
    print(
        "Built E1 diagonal revision; source pixels, poses and metrics verified.",
        flush=True,
    )
