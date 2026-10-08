"""Shared publication style for the TLE observable-state use case."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


FIGURE_DIR = Path(__file__).resolve().parent


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(figure, stem: str) -> None:
    """Save a vector PDF/SVG plus a high-resolution PNG preview."""
    for suffix, options in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
    ):
        figure.savefig(FIGURE_DIR / f"{stem}.{suffix}", facecolor="white", **options)

