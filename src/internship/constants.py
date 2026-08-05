"""Created on Jan 07 14:42:35 2026

Copied over and modified from GRBResearch repository
"""

__all__ = [
    # Font sizes
    "LABEL_FONT_SIZE",
    "LEGEND_TITLE_FONT_SIZE",
    "LEGEND_FONT_SIZE",
    "TICK_FONT_SIZE",
    "TITLE_FONT_SIZE",
    "ANNOTATION_FONT_SIZE",
    # Figure / line / marker geometry
    "FIGURE_DPI",
    "SAVE_DPI",
    "DEFAULT_FIGURE_SIZE",
    "LINE_WIDTH",
    "MARKER_SIZE",
    "CAP_SIZE",
    "AXES_LINE_WIDTH",
    # Tick geometry
    "TICK_MAJOR_SIZE",
    "TICK_MAJOR_WIDTH",
    "TICK_MINOR_SIZE",
    "TICK_MINOR_WIDTH",
    "TICK_DIRECTION",
    # Grid
    "GRID_ALPHA",
    "GRID_LINESTYLE",
]


LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
LEGEND_TITLE_FONT_SIZE = LEGEND_FONT_SIZE
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 10

# Figure / line / marker geometry
FIGURE_DPI = 150
SAVE_DPI = 600
DEFAULT_FIGURE_SIZE = (8, 6)
LINE_WIDTH = 1.0
MARKER_SIZE = 6
CAP_SIZE = 5
AXES_LINE_WIDTH = 0.8

# Tick geometry
TICK_MAJOR_SIZE = 5
TICK_MAJOR_WIDTH = 0.8
TICK_MINOR_SIZE = 3
TICK_MINOR_WIDTH = 0.6
TICK_DIRECTION = "in"  # astronomy convention

# Grid
GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"
