"""Created on Aug 02 20:17:31 2026"""

import matplotlib as plt

from .constants import (LABEL_FONT_SIZE, LEGEND_FONT_SIZE, LEGEND_TITLE_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE,
                        ANNOTATION_FONT_SIZE, FIGURE_DPI, SAVE_DPI, DEFAULT_FIGURE_SIZE, LINE_WIDTH, MARKER_SIZE,
                        AXES_LINE_WIDTH, TICK_MAJOR_SIZE, TICK_MAJOR_WIDTH, TICK_MINOR_SIZE, TICK_MINOR_WIDTH,
                        TICK_DIRECTION, GRID_ALPHA, GRID_LINESTYLE)


def update_style():
    """Update style for publication-ready figures.

    All values are driven by constants in grb_constants — a single source of truth.
    Changing a constant there automatically updates every figure.
    """
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": LABEL_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
            # Figure
            "figure.dpi": FIGURE_DPI,
            "figure.figsize": DEFAULT_FIGURE_SIZE,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            # Lines / markers
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.linewidth": AXES_LINE_WIDTH,
            # 'scatter.size': MARKER_SIZE,
            # Ticks
            "xtick.major.size": TICK_MAJOR_SIZE,
            "xtick.major.width": TICK_MAJOR_WIDTH,
            "xtick.minor.size": TICK_MINOR_SIZE,
            "xtick.minor.width": TICK_MINOR_WIDTH,
            "xtick.minor.visible": True,
            "xtick.direction": TICK_DIRECTION,
            "ytick.major.size": TICK_MAJOR_SIZE,
            "ytick.major.width": TICK_MAJOR_WIDTH,
            "ytick.minor.size": TICK_MINOR_SIZE,
            "ytick.minor.width": TICK_MINOR_WIDTH,
            "ytick.minor.visible": True,
            "ytick.direction": TICK_DIRECTION,
            # Grid
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LINESTYLE,
            "axes.axisbelow": True,
        }
    )
