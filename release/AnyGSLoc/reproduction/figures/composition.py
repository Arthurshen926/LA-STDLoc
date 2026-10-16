"""Exact diagonal source composition for pose-alignment figures."""

import numpy as np


def diagonal(gt, estimate):
    assert gt.shape == estimate.shape and gt.ndim == 3
    h, w = gt.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    use_estimate = yy / max(h - 1, 1) > xx / max(w - 1, 1)
    result = gt.copy()
    result[use_estimate] = estimate[use_estimate]
    assert np.array_equal(result[~use_estimate], gt[~use_estimate])
    assert np.array_equal(result[use_estimate], estimate[use_estimate])
    return result, use_estimate


def source_edges(ax, width, height, gt_color, estimate_color):
    # Vector-only annotation: exact source pixels remain in the saved mixed image.
    ax.plot([0, width - 1], [0, 0], color=gt_color, lw=0.55, clip_on=False)
    ax.plot(
        [0, width - 1],
        [height - 1, height - 1],
        color=estimate_color,
        lw=0.55,
        clip_on=False,
    )
    ax.plot(
        [0, width - 1],
        [0, height - 1],
        color="#e5e9ec",
        lw=0.40,
        solid_capstyle="butt",
        zorder=6,
    )


def source_legend(fig, y_bottom_mm=4.2):
    """One identical physical-size diagonal legend for E1 and E4."""
    from matplotlib.patches import Polygon

    from .style import ESTIMATE, GT, MUTED

    fw, fh = fig.get_size_inches() * 25.4
    ax = fig.add_axes([3.2 / fw, (y_bottom_mm - 1.7) / fh, 5.8 / fw, 3.4 / fh])
    ax.set(xlim=(0, 1), ylim=(1, 0))
    ax.axis("off")
    ax.add_patch(
        Polygon([[0, 0], [1, 0], [1, 1]], facecolor=GT, alpha=0.22, edgecolor="none")
    )
    ax.add_patch(
        Polygon(
            [[0, 0], [0, 1], [1, 1]], facecolor=ESTIMATE, alpha=0.22, edgecolor="none"
        )
    )
    ax.plot([0, 1], [0, 1], color="#8a959e", lw=0.40)
    ax.plot([0, 1], [0, 0], color=GT, lw=0.55)
    ax.plot([0, 1], [1, 1], color=ESTIMATE, lw=0.55)
    for x, text, color in [
        (11.5, "Rendered at GT pose", GT),
        (61.5, "Rendered at estimated pose", ESTIMATE),
        (127, "Translation / rotation error", MUTED),
    ]:
        fig.text(x / fw, y_bottom_mm / fh, text, fontsize=7.5, color=color, va="center")
