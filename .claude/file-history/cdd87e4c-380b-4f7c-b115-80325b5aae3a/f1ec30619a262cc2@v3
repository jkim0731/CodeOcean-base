"""Bokeh figures for the QC app: linked F+dFF traces, ROI image, metric histograms."""

from __future__ import annotations

import numpy as np
from bokeh.models import ColumnDataSource, LinearColorMapper, Span
from bokeh.plotting import figure


# Colors for each baseline method, kept consistent across F and dFF plots.
BASELINE_COLORS = {
    "short": "#1f77b4",
    "long":  "#2ca02c",
    "F0trend": "#ff7f0e",
    "F0":    "#d62728",
}
# dFFs derived from rolling-window baselines keep the same colors as the F plot;
# the dFFs derived from the global IRLS / IRLS+LOWESS baselines are drawn black,
# distinguished by dash pattern.
DFF_COLORS = {"short": "#1f77b4", "long": "#2ca02c"}
DFF_DASHED = {
    "F0trend": ([6, 4], "F0trend"),
    "F0":      ([2, 4], "F0"),
}


# ------------------------------ trace plots ------------------------------

def _make_trace_figure(title: str, y_label: str, height: int, x_range=None) -> figure:
    # Canvas backend: WebGL falls back to a slow path the moment any line uses
    # ``line_dash`` (we have dashed F0/F0trend dFFs), which dominates the cost
    # of pan/zoom. Pure-canvas is fast for ~10k samples × a handful of lines
    # and renders dashes natively.
    kwargs = dict(
        title=title,
        height=height,
        sizing_mode="stretch_width",
        x_axis_label="time (s)",
        y_axis_label=y_label,
        tools="xpan,xbox_zoom,xwheel_zoom,reset,save",
        active_drag="xpan",
        active_scroll="xwheel_zoom",
        output_backend="canvas",
        lod_threshold=2000,   # turn on level-of-detail downsampling during pan/zoom
        lod_factor=20,
        lod_interval=200,
        lod_timeout=400,
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    fig = figure(**kwargs)
    fig.toolbar.logo = None
    return fig


def make_f_figure(timestamps: np.ndarray, F: np.ndarray,
                  baselines: dict[str, np.ndarray]) -> tuple[figure, dict]:
    """F + all baselines. Returns (figure, sources_dict)."""
    fig = _make_trace_figure("Corrected F + baselines", "F", height=320)
    sources: dict = {}
    f_src = ColumnDataSource({"x": timestamps, "y": F})
    fig.line("x", "y", source=f_src, line_color="#222", line_width=1.0,
             legend_label="F", name="F")
    sources["F"] = f_src
    for name, trace in baselines.items():
        src = ColumnDataSource({"x": timestamps, "y": trace})
        fig.line("x", "y", source=src, line_color=BASELINE_COLORS[name],
                 line_width=1.6, legend_label=name, name=name)
        sources[name] = src
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "9pt"
    fig.legend.spacing = 0
    fig.legend.padding = 4
    return fig, sources


def make_dff_figure(timestamps: np.ndarray, dffs: dict[str, np.ndarray],
                    shared_x_range) -> tuple[figure, dict]:
    """dFF panel. ``dffs`` may include any of: short, long, F0, F0trend.

    short/long use the F-plot colors; F0/F0trend are drawn as black dashed
    lines with distinct dash patterns.
    """
    fig = _make_trace_figure("dFF", "dFF", height=240, x_range=shared_x_range)
    sources: dict = {}
    for name, trace in dffs.items():
        src = ColumnDataSource({"x": timestamps, "y": trace})
        if name in DFF_COLORS:
            fig.line("x", "y", source=src, line_color=DFF_COLORS[name],
                     line_width=1.0, legend_label=f"dFF ({name})", name=name)
        elif name in DFF_DASHED:
            dash, label = DFF_DASHED[name]
            fig.line("x", "y", source=src, line_color="#000",
                     line_width=1.2, line_dash=dash,
                     legend_label=f"dFF ({label})", name=name)
        else:
            fig.line("x", "y", source=src, line_width=1.0,
                     legend_label=f"dFF ({name})", name=name)
        sources[name] = src
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "9pt"
    return fig, sources


def update_trace_sources(sources: dict, timestamps: np.ndarray,
                         F: np.ndarray | None = None,
                         baselines: dict | None = None,
                         dffs: dict | None = None) -> None:
    if F is not None:
        sources["F"].data = {"x": timestamps, "y": np.asarray(F)}
    if baselines is not None:
        for name, trace in baselines.items():
            if name in sources:
                sources[name].data = {"x": timestamps, "y": np.asarray(trace)}
    if dffs is not None:
        for name, trace in dffs.items():
            if name in sources:
                sources[name].data = {"x": timestamps, "y": np.asarray(trace)}


# ------------------------------ image plots ------------------------------

def make_image_figure(size: int = 320):
    """Image figure with a LinearColorMapper for the FOV (so contrast is cheap
    to tweak — only the mapper updates, not the image data) and a separate
    RGBA overlay source for the ROI mask. Returns
    (fig, fov_src, mask_src, color_mapper, mask_renderer).
    """
    fig = figure(
        title="ROI",
        height=size, width=size,
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        toolbar_location="right",
        x_axis_location=None, y_axis_location=None,
    )
    fig.grid.visible = False
    fig.toolbar.logo = None
    color_mapper = LinearColorMapper(palette="Greys256", low=0.0, high=1.0)
    fov_src = ColumnDataSource({"image": [np.zeros((2, 2), dtype=np.float32)],
                                "x": [0], "y": [0], "dw": [2], "dh": [2]})
    mask_src = ColumnDataSource({"image": [np.zeros((2, 2), dtype=np.uint32)],
                                 "x": [0], "y": [0], "dw": [2], "dh": [2]})
    fig.image(image="image", x="x", y="y", dw="dw", dh="dh",
              source=fov_src, color_mapper=color_mapper)
    mask_renderer = fig.image_rgba(image="image", x="x", y="y", dw="dw", dh="dh",
                                   source=mask_src)
    return fig, fov_src, mask_src, color_mapper, mask_renderer


def mask_to_rgba(mask: np.ndarray, color=(255, 32, 32), alpha: int = 110) -> np.ndarray:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return rgba.view(np.uint32).reshape(h, w)


def push_image(fov_src: ColumnDataSource, mask_src: ColumnDataSource,
               fov: np.ndarray, mask: np.ndarray, x0: int = 0, y0: int = 0) -> None:
    """Send raw FOV pixels and the corresponding mask overlay to the image figure."""
    h, w = fov.shape
    fov_disp = np.flipud(fov).astype(np.float32, copy=False)
    mask_disp = np.flipud(mask)
    fov_src.data = {"image": [fov_disp],
                    "x": [x0], "y": [-(y0 + h)], "dw": [w], "dh": [h]}
    mask_src.data = {"image": [mask_to_rgba(mask_disp)],
                     "x": [x0], "y": [-(y0 + h)], "dw": [w], "dh": [h]}


# ------------------------------ histograms ------------------------------

def make_metric_histogram(values: np.ndarray, title: str, n_bins: int = 60,
                          height: int = 110, width: int | None = None) -> tuple[figure, Span]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        finite = np.array([0.0, 1.0])
    counts, edges = np.histogram(finite, bins=n_bins)
    kwargs = dict(title=title, height=height, sizing_mode="stretch_width",
                  tools="", toolbar_location=None, y_axis_label=None,
                  min_border=8)
    if width is not None:
        kwargs["width"] = width
        kwargs["sizing_mode"] = "fixed"
    fig = figure(**kwargs)
    fig.toolbar.logo = None
    fig.quad(top=counts, bottom=0, left=edges[:-1], right=edges[1:],
             fill_color="#888", line_color=None, alpha=0.7)
    span = Span(location=float(np.nan_to_num(finite[0])), dimension="height",
                line_color="red", line_width=2)
    fig.add_layout(span)
    fig.title.text_font_size = "10pt"
    fig.yaxis.major_label_text_font_size = "0pt"
    fig.yaxis.major_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    return fig, span
