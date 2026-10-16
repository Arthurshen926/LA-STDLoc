"""AnyGSLoc figure design system v2 (2026-10-11).

One visual language for all nine figures. Sizes in points unless marked otherwise.

Color roles (validated with dataviz validate_palette.js, light surface, --pairs all):
  OURS   #0A9A8A  AnyGSLoc / track-centric / final compact output
  BLUE   #4A6FC2  mechanism accent: reference views, cameras, coverage support
  ORANGE #DB7D2A  mechanism accent: observations, rays, pose-aware additions
  -> all pass lightness, chroma, CVD (worst 11.5) and normal-vision (worst 17.4) checks.
  ORANGE is below 3:1 on white, so it is never used for text and always ships with
  a direct label (relief rule). Text uses the *_TEXT steps (>= 4.7:1 on white).
Baselines and alternative constructions are neutral (BASE_DARK / BASE_LIGHT) and are
separated by line style or marker shape plus a direct label, never by hue alone.
"""

from pathlib import Path

FONT = "Nimbus Sans"
FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")

# Ink
INK = "#1F2A33"  # primary text, 14.6:1
INK2 = "#55616C"  # secondary text, 6.3:1
MUTED = "#7A8691"  # tertiary text / ticks, 3.7:1 (never below 7 pt)
RULE = "#D5DCE1"  # hairlines, dividers
GRID = "#E8ECEF"  # chart grid
PANEL = "#F4F6F7"  # panel / card surface
PANEL_OURS = "#E8F4F2"  # tinted surface behind our method

# Roles
OURS = "#0A9A8A"
OURS_TEXT = "#0B7468"
BLUE = "#4A6FC2"
BLUE_TEXT = "#3F5FA8"
ORANGE = "#DB7D2A"
ORANGE_TEXT = "#A85A17"
BASE_DARK = "#55616C"  # first baseline / alternative
BASE_LIGHT = "#A3ADB5"  # second baseline / alternative
GHOST = "#D9DFE3"  # removed / context points

# Back-compat names used by the shared composition helpers.
TEAL = OURS
GT = BLUE
ESTIMATE = ORANGE
FAINT = RULE
PURPLE = ORANGE
METHOD_COLORS = {"STDLoc": BASE_DARK, "ULFLoc": BASE_LIGHT, "AnyGSLoc": OURS}

WIDTH_MM = 180
COLUMN_MM = 88
DPI = 400
# Type scale (pt): figure title is never drawn in-figure; captions carry it.
PANEL_PT = 8.5  # panel labels "(a) ..."
LABEL_PT = 7.5  # axis labels, column headers
SECONDARY_PT = 7  # annotations, chips, legends (floor)
TITLE_PT = PANEL_PT
LINE_PT = 1.4  # data lines
HAIR_PT = 0.5  # frames, rules


def matplotlib_style():
    import matplotlib as mpl
    from matplotlib import font_manager

    for path in FONT_DIR.glob("NimbusSans-*.otf"):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "font.size": LABEL_PT,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": "#9AA5AE",
            "axes.linewidth": HAIR_PT,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": HAIR_PT,
            "ytick.major.width": HAIR_PT,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.labelsize": SECONDARY_PT,
            "ytick.labelsize": SECONDARY_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": PANEL_PT,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": SECONDARY_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def chip(
    ax_or_fig,
    x,
    y,
    text,
    transform,
    color=INK,
    fc="white",
    alpha=0.92,
    size=SECONDARY_PT,
    weight="normal",
    ha="center",
    va="center",
    pad=0.22,
):
    """A small rounded label used to overlay values on images."""
    return ax_or_fig.text(
        x,
        y,
        text,
        transform=transform,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        weight=weight,
        bbox=dict(
            boxstyle=f"round,pad={pad},rounding_size=.35", fc=fc, ec="none", alpha=alpha
        ),
        zorder=20,
    )


def save_figure(fig, output):
    for ext in ("pdf", "svg"):
        fig.savefig(Path(output).with_suffix("." + ext), dpi=DPI, pad_inches=0)
    fig.savefig(Path(output).with_suffix(".png"), dpi=DPI)
