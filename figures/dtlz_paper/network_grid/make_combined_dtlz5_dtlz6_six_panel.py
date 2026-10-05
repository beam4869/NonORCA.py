"""Assemble selected DTLZ5 and D6 networks for final-size manuscript use.

The network panels are the previously rendered NetworkX/Matplotlib outputs, so
their matrices, groups, node styling, and adaptive edge-label positions remain
unchanged. This script only selects and arranges the requested cases.

Figure contract
---------------
Core conclusion: fixed-K ORCA recovers the prescribed block-and-singleton
structure in representative DTLZ5 and D6 positive controls.
Archetype: quantitative grid.
Backend: Python and Matplotlib only.
Output: editable PDF and SVG plus high-resolution PNG and TIFF.
Reviewer risk: the landscape strip reduces edge-label size at manuscript width.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / ".matplotlib-cache"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE_DIR = (
    HERE
    / "networkx_original_code_adjusted_ab_scaled_adaptive_labels"
    / "individual"
)
OUTPUT_DIR = (
    HERE
    / "networkx_original_code_adjusted_ab_scaled_adaptive_labels"
    / "six_panel_dtlz5_dtlz6"
)

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 270.0
FIGURE_HEIGHT_MM = 90.0

# x, y, width, height in millimetres. Each family uses two compact panels
# beside one larger high-dimensional panel, matching the original landscape
# arrangement.
PANELS = (
    ("a", "DTLZ5", 2, 4, (0.0, 45.0, 45.0, 45.0)),
    ("b", "DTLZ5", 4, 7, (0.0, 0.0, 45.0, 45.0)),
    ("c", "DTLZ5", 7, 12, (43.0, 0.0, 90.0, 90.0)),
    ("d", "D6", 2, 4, (135.0, 45.0, 45.0, 45.0)),
    ("e", "D6", 4, 7, (135.0, 0.0, 45.0, 45.0)),
    ("f", "D6", 7, 12, (178.0, 0.0, 90.0, 90.0)),
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def add_panel(
    fig: plt.Figure,
    panel_letter: str,
    family: str,
    intrinsic_dimension: int,
    num_objectives: int,
    rectangle_mm: tuple[float, float, float, float],
) -> None:
    """Place one unchanged rendered network and add its panel heading."""

    x_mm, y_mm, width_mm, height_mm = rectangle_mm
    ax = fig.add_axes(
        [
            x_mm / FIGURE_WIDTH_MM,
            y_mm / FIGURE_HEIGHT_MM,
            width_mm / FIGURE_WIDTH_MM,
            height_mm / FIGURE_HEIGHT_MM,
        ]
    )
    source_family = "dtlz6" if family == "D6" else family.lower()
    stem = (
        f"{source_family}_i{intrinsic_dimension}_m{num_objectives}"
        "_original_code.png"
    )
    ax.imshow(mpimg.imread(SOURCE_DIR / stem), interpolation="lanczos")
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        f"{panel_letter}  {family}({intrinsic_dimension},{num_objectives})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    for panel_letter, family, intrinsic_dimension, num_objectives, rectangle in PANELS:
        add_panel(
            fig,
            panel_letter,
            family,
            intrinsic_dimension,
            num_objectives,
            rectangle,
        )

    base = OUTPUT_DIR / "figure_dtlz5_d6_six_panel_landscape"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
