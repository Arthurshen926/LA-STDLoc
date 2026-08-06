"""Shared publication styling and deterministic figure export."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-safe, stable semantics across every paper figure.
COLORS = {
    "ink": "#17202A",
    "muted": "#667085",
    "line": "#CBD5E1",
    "panel": "#F8FAFC",
    "primitive": "#8B95A5",
    "track": "#0072B2",
    "anchor": "#D55E00",
    "candidate": "#7B61A8",
    "core": "#009E73",
    "reserve": "#E69F00",
    "clean": "#009E73",
    "false": "#D62728",
    "inlier": "#00A6D6",
    "a0": "#7B61A8",
    "a1": "#009E73",
}


@dataclass(frozen=True)
class PCAProjection:
    center: np.ndarray
    axes: np.ndarray

    @classmethod
    def fit(cls, xyz: np.ndarray, *, maximum_rows: int = 50000) -> "PCAProjection":
        points = np.asarray(xyz, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 3:
            raise ValueError("PCA projection requires at least three finite points")
        if points.shape[0] > maximum_rows:
            rows = np.linspace(0, points.shape[0] - 1, maximum_rows).astype(int)
            points = points[rows]
        center = np.median(points, axis=0)
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        basis = axes[:2].T
        # Make the orientation deterministic rather than accepting SVD sign flips.
        for column in range(2):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1
        return cls(center=center, axes=basis)

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float64) - self.center) @ self.axes


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply a compact two-column-paper style without mutating global defaults."""
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 7.8,
        "axes.titleweight": "semibold",
        "axes.edgecolor": COLORS["line"],
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 1.0,
    }
    with mpl.rc_context(settings):
        yield


def panel_label(axis, label: str, *, x: float = -0.035, y: float = 1.035) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def clean_image_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def robust_limits(xy: np.ndarray, *, percent: float = 0.8, padding: float = 0.04):
    values = np.asarray(xy, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    low = np.percentile(values, percent, axis=0)
    high = np.percentile(values, 100.0 - percent, axis=0)
    margin = np.maximum(high - low, 1e-6) * float(padding)
    return low - margin, high + margin


def save_figure(
    figure: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = 360,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for suffix in ("pdf", "svg", "png"):
        path = output / f"{stem}.{suffix}"
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        figure.savefig(path, **kwargs)
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: str | Path, *, hash_file: bool = True) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, object] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if hash_file:
        record["sha256"] = sha256(resolved)
    return record
