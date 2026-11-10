"""Figure helpers shared by notebooks/figures.ipynb: palette, style and small plot utilities."""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# ====================================================================
# PALETTE
# ====================================================================
PALETTE = {
    # Low-sat fills (large blocks)
    "fill_blue":     "#BEE8FF",
    "fill_orange":   "#FFD37F",
    "fill_salmon":   "#FF7F7F",
    "fill_gray":     "#B9B9B9",
    # High-sat strokes (small marks)
    "stroke_teal":   "#036868",
    "stroke_navy":   "#224686",
    "stroke_clay":   "#A46A4D",
    # Mid-sat accents
    "accent_teal":   "#29BD9A",
    "accent_sky":    "#63C0EF",
    "accent_purple": "#9B77CB",
    # Nuclear-DC paper reserved mappings
    "case0":         "#B9B9B9",   # grid-only DC
    "case1":         "#86DED4",   # BWRX no heat recovery (mint teal)
    "case2":         "#036868",   # BWRX cogen (hero)
    "case3":         "#224686",   # BWRX + absorption
    "case4":         "#A46A4D",   # NGCC on-site
    "premium_pos":   "#29BD9A",
    "premium_neg":   "#FF7F7F",
    "ref":           "#63C0EF",
    # Stacked cost components
    "comp_capital":  "#BEE8FF",
    "comp_om":       "#FFD37F",
    "comp_fuel":     "#FF7F7F",
    "comp_grid_buy": "#A46A4D",
    "comp_grid_sell":"#62AE9E",
}

MAIN_FILLS = [
    PALETTE["fill_blue"],
    PALETTE["fill_salmon"],
    PALETTE["fill_orange"],
    PALETTE["fill_gray"],
]

MAIN_STROKES = [
    PALETTE["stroke_navy"],
    PALETTE["stroke_clay"],
    PALETTE["stroke_teal"],
    "#555555",
]

BLOCK_EDGE = "#FFFFFF"
BLOCK_EDGE_LW = 0.5

JOURNAL_FONT_TARGETS = {
    # Target sizes are finished sizes after manuscript / LaTeX scaling.
    # Sources checked 2026-06-01:
    # Elsevier artwork sizing; IEEE Author Center; Nature Research figure guide.
    "elsevier": {
        "target_pt": 7.0,
        "min_pt": 6.0,
        "max_pt": 10.0,
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
    },
    "applied_energy": {
        "target_pt": 7.0,
        "min_pt": 6.0,
        "max_pt": 10.0,
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
    },
    "ieee": {
        "target_pt": 9.5,
        "min_pt": 9.0,
        "max_pt": 10.0,
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
    },
    "nature": {
        "target_pt": 6.0,
        "min_pt": 5.0,
        "max_pt": 7.0,
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
    },
}


def apply_sci_style(width: str = "ae_single") -> None:
    """Apply the shared rcParams for journal figures.

    width : layout target.
        "ae_single"  - 7pt body for AE/Energy 89mm single-column (default)
        "ae_double"  - 7pt body for 183mm double-column (same fonts)
        "poster"     - 18pt body for slide / poster use
    """
    base = {
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize":    0,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.axisbelow":    True,
        "grid.linestyle":    "--",
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        "savefig.dpi":       600,
        "svg.fonttype":      "none",
        "pdf.fonttype":      42,
        "figure.dpi":        150,
    }
    if width == "poster":
        base.update({
            "font.size":       18,
            "axes.labelsize":  20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        })
    else:
        base.update({
            "font.size":       7,
            "axes.labelsize":  7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
        })
    plt.rcParams.update(base)


def apply_journal_style(
    journal: str = "elsevier",
    width: str = "ae_single",
    final_scale: float = 1.0,
) -> None:
    """Apply sci style with a journal-aware finished-font target.

    Parameters
    ----------
    journal:
        One of ``elsevier``, ``applied_energy``, ``ieee``, or ``nature``.
    width:
        Passed to :func:`apply_sci_style`.
    final_scale:
        Expected manuscript scaling factor, defined as
        ``displayed_width / saved_vector_width``. For example, if LaTeX
        inserts a 7.5 in saved PDF at 7.0 in text width, use 7.0 / 7.5.

    The helper sets source font sizes so the *finished* figure text lands near
    the journal target. Keep any manual ``fontsize=`` overrides at or above the
    resulting ``font.size`` unless a final-PDF check proves compliance.
    """
    key = journal.lower().replace("-", "_").replace(" ", "_")
    if key not in JOURNAL_FONT_TARGETS:
        known = ", ".join(sorted(JOURNAL_FONT_TARGETS))
        raise ValueError(f"Unknown journal profile {journal!r}; choose one of: {known}")
    if final_scale <= 0:
        raise ValueError("final_scale must be positive")

    apply_sci_style(width)
    target = JOURNAL_FONT_TARGETS[key]
    source_pt = target["target_pt"] / final_scale
    plt.rcParams.update({
        "font.sans-serif": target["fonts"],
        "font.size": source_pt,
        "axes.labelsize": source_pt,
        "xtick.labelsize": source_pt,
        "ytick.labelsize": source_pt,
        "legend.fontsize": source_pt,
    })


def save_pdf(fig: plt.Figure, name: str, outdir: str = "figs") -> None:
    """Export a submission-ready vector PDF with a tight bounding box."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight")


def _rc_size(name: str, fallback: float) -> float:
    """Return a numeric rcParams size, tolerating named matplotlib sizes."""
    value = plt.rcParams.get(name, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def add_panel_label(
    ax,
    letter: str,
    x: float = -0.15,
    y: float = 1.05,
    fontsize: float | None = None,
) -> None:
    """Add a bold panel label (a, b, c, ...) at standard offset.

    Place above the upper-left of each panel; use axes-relative coordinates
    so position is consistent across panels regardless of data range.
    """
    if fontsize is None:
        fontsize = _rc_size("font.size", 8.0)
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left")


# ====================================================================
# Helper plot functions for new chart types
# Additional chart helpers
# ====================================================================

def tornado(ax, factors: Sequence[tuple], base: float = 0.0,
            pos_color: str | None = None, neg_color: str | None = None) -> None:
    """Horizontal tornado / sensitivity bar plot."""
    pos_color = pos_color or PALETTE["fill_blue"]
    neg_color = neg_color or PALETTE["fill_salmon"]
    y_pos = np.arange(len(factors))
    for i, (lab, lo, hi) in enumerate(factors):
        if lo < 0:
            ax.barh(i, lo, color=neg_color, edgecolor=BLOCK_EDGE,
                    linewidth=BLOCK_EDGE_LW, alpha=0.85)
        if hi > 0:
            ax.barh(i, hi, color=pos_color, edgecolor=BLOCK_EDGE,
                    linewidth=BLOCK_EDGE_LW, alpha=0.85)
    ax.axvline(base, color="#000000", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[0] for f in factors])
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3, linestyle="--")


def radar(ax, axes_labels: Sequence[str], series: dict,
          colors: Sequence[str] | None = None, ylim: tuple = (0, 1)) -> None:
    """Radar / spider plot.

    ax must be created with `subplot_kw=dict(projection="polar")`.
    series : dict of {label: list_of_values}, one value per axis.
    """
    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    default_colors = [PALETTE["stroke_teal"], PALETTE["stroke_navy"],
                      PALETTE["stroke_clay"], PALETTE["accent_teal"],
                      PALETTE["accent_purple"], PALETTE["accent_sky"]]
    colors = list(colors) if colors is not None else default_colors
    for i, (label, vals) in enumerate(series.items()):
        c = colors[i % len(colors)]
        vals_closed = list(vals) + [vals[0]]
        ax.plot(angles, vals_closed, color=c, linewidth=1.5, label=label)
        ax.fill(angles, vals_closed, color=c, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        axes_labels,
        fontsize=_rc_size("xtick.labelsize", _rc_size("font.size", 6.5)),
    )
    ax.set_ylim(*ylim)


def duration_curve(ax, series: dict, log_y: bool = True) -> None:
    """Duration curve from one or more 1D arrays.

    series : dict of {label: data_array} OR {label: (data_array, color)}.
    """
    auto_colors = [PALETTE["stroke_teal"], PALETTE["stroke_navy"], PALETTE["stroke_clay"],
                   PALETTE["accent_teal"], PALETTE["accent_purple"]]
    for i, (label, val) in enumerate(series.items()):
        if isinstance(val, tuple):
            data, color = val
        else:
            data, color = val, None
        color = color or auto_colors[i % len(auto_colors)]
        sorted_desc = np.sort(np.asarray(data))[::-1]
        pct = np.arange(1, len(sorted_desc) + 1) / len(sorted_desc) * 100
        ax.plot(pct, sorted_desc, color=color, linewidth=1.5, label=label)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("% of hours (sorted descending)")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, linestyle="--", which="both")


def bivariate_heatmap(ax, data: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str],
                      vlow: float = -30, vhigh: float = 30, vcenter: float = 0,
                      fmt: str = "+.0f"):
    """Diverging 2D heatmap centered on `vcenter`.

    Uses fill_salmon → white → fill_blue as the diverging cmap for manuscript palette consistency.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "sf_div", [PALETTE["fill_salmon"], "#FFFFFF", PALETTE["fill_blue"]], N=256
    )
    norm = TwoSlopeNorm(vmin=vlow, vcenter=vcenter, vmax=vhigh)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    threshold = 0.7 * max(abs(vlow), abs(vhigh))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=_rc_size("font.size", 6.5),
                    color="white" if abs(v) > threshold else "#000000")
    return im


def violin_grouped(ax, data_by_group: dict, log_y: bool = False,
                   colors: Sequence[str] | None = None) -> None:
    """Violin plot with low-saturation fills and light separators."""
    labels = list(data_by_group.keys())
    data = list(data_by_group.values())
    default_palette = [PALETTE["stroke_clay"], PALETTE["stroke_teal"],
                       PALETTE["stroke_navy"], PALETTE["accent_purple"],
                       PALETTE["accent_teal"]]
    palette_cycle = list(colors) if colors is not None else default_palette
    pos = np.arange(len(labels))
    parts = ax.violinplot(data, positions=pos, widths=0.7, showmeans=False, showmedians=True)
    for body, c in zip(parts["bodies"], palette_cycle * (len(labels) // len(palette_cycle) + 1)):
        body.set_facecolor(c)
        body.set_edgecolor(BLOCK_EDGE)
        body.set_alpha(0.6)
        body.set_linewidth(BLOCK_EDGE_LW)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#000000")
        parts["cmedians"].set_linewidth(0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    if log_y:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")


def multi_stream_dispatch(ax, hours: np.ndarray, positive_streams: dict,
                          negative_streams: dict | None = None,
                          capacity_line: float | None = None) -> None:
    """Stacked dispatch area - positive streams above zero, negative below.

    positive_streams : dict of {label: 1D array of length len(hours)}.
    negative_streams : optional dict (drawn below zero).
    capacity_line    : optional horizontal dashed line at this y-value.
    """
    pos_colors = [PALETTE["fill_blue"], PALETTE["fill_orange"],
                  PALETTE["accent_teal"], PALETTE["accent_purple"]]
    ax.stackplot(hours, *positive_streams.values(),
                 labels=list(positive_streams.keys()),
                 colors=pos_colors[:len(positive_streams)],
                 alpha=0.85, edgecolor="white", linewidth=0.4)
    if negative_streams:
        cum_neg = np.zeros_like(hours, dtype=float)
        for label, arr in negative_streams.items():
            ax.fill_between(hours, cum_neg, cum_neg - arr,
                            color=PALETTE["stroke_clay"], alpha=0.5,
                            edgecolor="white", linewidth=0.4, label=label)
            cum_neg = cum_neg - arr
        ax.axhline(0, color="#000000", linewidth=0.6)
    if capacity_line is not None:
        ax.plot(hours, np.full_like(hours, capacity_line, dtype=float),
                color=PALETTE["stroke_teal"], linestyle="--", linewidth=1.0,
                label=f"Capacity = {capacity_line}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
