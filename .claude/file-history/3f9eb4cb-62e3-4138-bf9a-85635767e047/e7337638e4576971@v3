"""Baseline-fitting QC app.

Inline (Jupyter): see ``launch_in_notebook.ipynb``.
Standalone:
    panel serve /root/capsule/code/qc_app/app.py --port 5006 \
        --address 0.0.0.0 --allow-websocket-origin='*'
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``qc_app`` importable when launched via ``panel serve qc_app/app.py``
# (which only puts the script's own directory on sys.path).
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


class _NullCtx:
    """No-op context manager for code paths without an active Bokeh document."""
    def __enter__(self): return self
    def __exit__(self, *a): return False

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import column as bk_column

from qc_app import curation as curation_mod
from qc_app import data as data_mod
from qc_app import plots as plots_mod
from qc_app import rois as rois_mod

# No ``design=`` here: the Material design adds a CSS layer that slows down
# layout reflows during plot interactions in some setups.
pn.extension(notifications=True, sizing_mode="stretch_width")

BASELINE_ORDER = ["short", "long", "F0trend", "F0"]
DFF_ORDER = ["short", "long", "F0trend", "F0"]
METRIC_ORDER = ["noise", "snr", "bleaching", "sustained", "skewness"]


def build_app(parent_dir: Path = data_mod.PARENT_DIR,
              data_dir: Path = data_mod.DATA_DIR,
              curation_path: Path = curation_mod.DEFAULT_CURATION_PATH) -> pn.viewable.Viewable:
    state = _AppState(parent_dir=parent_dir, data_dir=data_dir,
                      curation_path=curation_path)
    return state.layout()


class _AppState:
    def __init__(self, parent_dir: Path, data_dir: Path, curation_path: Path):
        self.parent_dir = Path(parent_dir)
        self.data_dir = Path(data_dir)
        self.curation_path = Path(curation_path)

        self.sessions = data_mod.list_sessions(parent_dir)
        if not self.sessions:
            raise RuntimeError(f"No sessions found under {parent_dir}")
        self.session_keys = [s.name for s in self.sessions]

        self.curation_df = curation_mod.load_curation(self.curation_path)
        self.metrics_all = data_mod.aggregate_metrics(str(parent_dir))

        self.session_data: data_mod.SessionData | None = None
        self.roi_index = 0
        # Track current FOV/plane state so we can skip redundant updates.
        self._loaded_plane_id: str | None = None
        self._loaded_plane: rois_mod.PlaneAssets | None = None
        self._displayed_roi_key: tuple | None = None  # (session_key, roi_index)

        self._build_widgets()
        self._build_plots()
        self._wire_callbacks()
        self._select_session(self.session_keys[0])

    # ---------------------------- widgets ----------------------------

    def _build_widgets(self):
        self.session_select = pn.widgets.Select(
            name="Session", options=self.session_keys,
            value=self.session_keys[0], width=240)
        self.roi_input = pn.widgets.IntInput(
            name="ROI index", value=0, width=110, start=0, end=0)
        self.prev_btn = pn.widgets.Button(name="◀ Prev (J)", width=100)
        self.next_btn = pn.widgets.Button(name="Next (K) ▶", width=100)
        self.roi_info = pn.pane.Markdown("", height=24, margin=(0, 10))
        self.progress_info = pn.pane.Markdown("", height=24, margin=(0, 10))

        self.baseline_checks = pn.widgets.CheckBoxGroup(
            name="Good baselines", options=BASELINE_ORDER, value=[], inline=True)
        self.undecided_toggle = pn.widgets.Toggle(name="Undecided", value=False, width=110)
        self.save_btn = pn.widgets.Button(name="Save (S)", button_type="primary", width=100)
        self.next_save_btn = pn.widgets.Button(
            name="Save + Next (Space)", button_type="success", width=170)
        self.curation_status = pn.pane.Markdown("", height=24, margin=(0, 10))

        self.image_mode = pn.widgets.RadioButtonGroup(
            name="View", options=["Zoom", "Full FOV"], value="Zoom", width=180)
        self.zoom_pad = pn.widgets.IntSlider(
            name="Zoom pad (px)", value=25, start=5, end=120, width=180)
        self.mask_toggle = pn.widgets.Toggle(
            name="Mask", value=True, width=80, button_type="default")

        # Contrast sliders. Defaults are populated from mask pixels each ROI;
        # the user can then drag them. Range is [-1, 1] in normalized fluorescence
        # (we percentile-stretch the *plane FOV* once on load to a stable scale).
        self.contrast_low = pn.widgets.FloatSlider(
            name="contrast low", start=0.0, end=1.0, value=0.0, step=0.005, width=180)
        self.contrast_high = pn.widgets.FloatSlider(
            name="contrast high", start=0.0, end=1.0, value=1.0, step=0.005, width=180)
        self.contrast_reset = pn.widgets.Button(name="Auto", width=70)

    def _build_plots(self):
        self.f_fig, self.f_sources = plots_mod.make_f_figure(
            timestamps=np.array([0.0, 1.0]),
            F=np.array([0.0, 0.0]),
            baselines={k: np.array([0.0, 0.0]) for k in BASELINE_ORDER},
        )
        self.dff_fig, self.dff_sources = plots_mod.make_dff_figure(
            timestamps=np.array([0.0, 1.0]),
            dffs={k: np.array([0.0, 0.0]) for k in DFF_ORDER},
            shared_x_range=self.f_fig.x_range,
        )
        (self.image_fig, self.fov_src, self.mask_src,
         self.color_mapper, self.mask_renderer) = plots_mod.make_image_figure(size=320)

        self.metric_figs: dict = {}
        self.metric_spans: dict = {}
        for name in METRIC_ORDER:
            vals = (self.metrics_all[name].to_numpy()
                    if name in self.metrics_all.columns else np.zeros(1))
            fig, span = plots_mod.make_metric_histogram(vals, title=name, height=110)
            self.metric_figs[name] = fig
            self.metric_spans[name] = span

    def _batch_updates(self):
        """Group multiple model mutations into a single Bokeh document patch."""
        doc = self.f_fig.document
        if doc is None:
            return _NullCtx()
        try:
            return doc.hold("combine")
        except Exception:
            return _NullCtx()

    def _wire_callbacks(self):
        self.session_select.param.watch(lambda evt: self._select_session(evt.new), "value")
        self.roi_input.param.watch(lambda evt: self._select_roi(int(evt.new)), "value")
        self.prev_btn.on_click(lambda _e: self._select_roi(max(0, self.roi_index - 1)))
        self.next_btn.on_click(lambda _e: self._select_roi(self._clamp_roi(self.roi_index + 1)))
        self.save_btn.on_click(lambda _e: self._save_decision(advance=False))
        self.next_save_btn.on_click(lambda _e: self._save_decision(advance=True))
        self.image_mode.param.watch(lambda _e: self._refresh_image(force=True), "value")
        self.zoom_pad.param.watch(lambda _e: self._refresh_image(force=True), "value")
        self.mask_toggle.param.watch(self._on_mask_toggle, "value")
        self.contrast_low.param.watch(self._on_contrast_change, "value")
        self.contrast_high.param.watch(self._on_contrast_change, "value")
        self.contrast_reset.on_click(lambda _e: self._auto_contrast())

    # ---------------------------- selection ----------------------------

    def _clamp_roi(self, i: int) -> int:
        if self.session_data is None:
            return 0
        return int(np.clip(i, 0, self.session_data.n_rois - 1))

    def _select_session(self, session_key: str):
        sess_path = self.parent_dir / session_key
        self.session_data = data_mod.load_session(str(sess_path))
        n = self.session_data.n_rois
        self.roi_input.end = n - 1
        # Force a refresh even if value didn't change (e.g. both 0 across sessions)
        if self.roi_input.value == 0:
            self.roi_index = 0
            self._refresh_roi()
        else:
            self.roi_input.value = 0  # triggers _select_roi(0)
        # Plane-cache invalidation handled lazily by plane_id mismatch.

    def _select_roi(self, idx: int):
        idx = self._clamp_roi(idx)
        self.roi_index = idx
        if self.roi_input.value != idx:
            self.roi_input.value = idx  # propagate to widget; watcher re-enters
            return
        self._refresh_roi()

    # ---------------------------- refresh ----------------------------

    def _refresh_roi(self):
        sd = self.session_data
        if sd is None:
            return
        i = self.roi_index
        # Skip if we're already showing this ROI (avoids duplicate refreshes
        # caused by widget watchers re-firing).
        if self._displayed_roi_key == (sd.session_key, i):
            return
        self._displayed_roi_key = (sd.session_key, i)

        # Batch all trace-source updates so they ride a single websocket patch
        # instead of one per source.
        with self._batch_updates():
            plots_mod.update_trace_sources(
                self.f_sources,
                timestamps=sd.timestamps,
                F=sd.F[i],
                baselines={k: sd.baselines[k][i] for k in BASELINE_ORDER},
            )
            plots_mod.update_trace_sources(
                self.dff_sources,
                timestamps=sd.timestamps,
                dffs={k: sd.dffs[k][i] for k in DFF_ORDER},
            )
            self.f_fig.x_range.start = float(sd.timestamps[0])
            self.f_fig.x_range.end = float(sd.timestamps[-1])

        roi_row = sd.rois.iloc[i]
        plane_id = roi_row.plane_id
        cell_roi_id = int(roi_row.cell_roi_id)
        depth = int(roi_row.intended_depth) if pd.notna(roi_row.intended_depth) else None
        self.roi_info.object = (
            f"**{sd.session_key}** | ROI **{i}/{sd.n_rois - 1}** | "
            f"plane `{plane_id}` (depth {depth} µm) | cell_roi_id `{cell_roi_id}`"
        )
        self.f_fig.title.text = (
            f"Corrected F + baselines  —  {sd.session_key} / {plane_id} / cell {cell_roi_id}"
        )

        for name, span in self.metric_spans.items():
            if name in sd.metrics.columns:
                span.location = float(sd.metrics[name].iloc[i])

        self._refresh_image(force=False)
        self._load_decision_into_widgets()
        n_saved = self.curation_df[self.curation_df["session_key"] == sd.session_key].shape[0]
        self.progress_info.object = f"curated {n_saved}/{sd.n_rois} in this session"

    # ----- image -----

    def _ensure_plane_loaded(self, plane_id: str) -> rois_mod.PlaneAssets | None:
        sd = self.session_data
        if sd is None:
            return None
        if self._loaded_plane is not None and self._loaded_plane_id == plane_id:
            return self._loaded_plane
        processed_dir = data_mod.find_processed_dir(sd.session_key, str(self.data_dir))
        if processed_dir is None:
            return None
        plane_path = processed_dir / plane_id
        try:
            plane = rois_mod.load_plane_assets(str(plane_path))
        except (FileNotFoundError, OSError):
            return None
        # Pre-normalize FOV to a stable [0,1] scale. Contrast sliders then
        # operate on this normalized FOV via the LinearColorMapper, so contrast
        # adjustment never re-sends image data.
        plane_norm = type(plane)(
            plane_id=plane.plane_id,
            fov=rois_mod.normalize_for_display(plane.fov),
            masks=plane.masks,
            cell_roi_ids=plane.cell_roi_ids,
        )
        self._loaded_plane = plane_norm
        self._loaded_plane_id = plane_id
        return plane_norm

    def _refresh_image(self, *, force: bool):
        sd = self.session_data
        if sd is None:
            return
        roi_row = sd.rois.iloc[self.roi_index]
        plane_id = roi_row.plane_id
        cell_roi_id = int(roi_row.cell_roi_id)
        plane = self._ensure_plane_loaded(plane_id)
        if plane is None:
            self.image_fig.title.text = f"(image unavailable)"
            return
        try:
            mask = rois_mod.get_roi_mask(plane, cell_roi_id)
        except IndexError:
            self.image_fig.title.text = f"(roi {cell_roi_id} not in plane masks)"
            return
        self._current_mask = mask
        self._current_fov = plane.fov

        if self.image_mode.value == "Zoom":
            fov_c, mask_c, (y0, x0) = rois_mod.crop_around_mask(
                plane.fov, mask, pad=int(self.zoom_pad.value))
            plots_mod.push_image(self.fov_src, self.mask_src, fov_c, mask_c, x0=x0, y0=y0)
        else:
            plots_mod.push_image(self.fov_src, self.mask_src, plane.fov, mask)
        self.mask_renderer.visible = bool(self.mask_toggle.value)
        self.image_fig.title.text = f"{plane_id} | cell_roi_id {cell_roi_id}"
        # Default contrast to the mask-pixel percentiles for this ROI.
        self._auto_contrast()

    def _on_mask_toggle(self, evt):
        self.mask_renderer.visible = bool(evt.new)

    def _on_contrast_change(self, _evt):
        lo = float(self.contrast_low.value)
        hi = float(self.contrast_high.value)
        if hi <= lo:
            hi = lo + 1e-3
        self.color_mapper.low = lo
        self.color_mapper.high = hi

    def _auto_contrast(self):
        """Set contrast from FOV pixel values within the current mask."""
        if not hasattr(self, "_current_mask") or self._current_mask is None:
            return
        fov = self._current_fov
        mask = self._current_mask
        pix = fov[mask] if mask.any() else fov.ravel()
        if pix.size == 0:
            return
        lo = float(np.percentile(pix, 2))
        hi = float(np.percentile(pix, 99))
        # Pad a bit so the brightest mask pixel isn't fully saturated.
        span = max(hi - lo, 1e-3)
        lo = max(0.0, lo - 0.05 * span)
        hi = min(1.0, hi + 0.05 * span)
        # Update sliders without re-firing twice.
        with pn.io.unlocked():
            self.contrast_low.value = lo
            self.contrast_high.value = hi
        self.color_mapper.low = lo
        self.color_mapper.high = hi

    # ---------------------------- curation ----------------------------

    def _load_decision_into_widgets(self):
        sd = self.session_data
        decision = curation_mod.lookup_decision(self.curation_df, sd.session_key, self.roi_index)
        if decision is None:
            self.baseline_checks.value = []
            self.undecided_toggle.value = False
            self.curation_status.object = "_no decision yet_"
        else:
            self.baseline_checks.value = decision["selected_list"]
            self.undecided_toggle.value = bool(decision.get("undecided", False))
            self.curation_status.object = (
                f"saved as **{decision['category']}** at {decision['timestamp']}"
            )

    def _save_decision(self, advance: bool):
        sd = self.session_data
        if sd is None:
            return
        roi_row = sd.rois.iloc[self.roi_index]
        self.curation_df = curation_mod.save_decision(
            session_key=sd.session_key,
            roi_index=self.roi_index,
            plane_id=roi_row.plane_id,
            cell_roi_id=int(roi_row.cell_roi_id),
            selected=list(self.baseline_checks.value),
            undecided=bool(self.undecided_toggle.value),
            path=self.curation_path,
        )
        category = curation_mod.derive_category(
            list(self.baseline_checks.value), bool(self.undecided_toggle.value))
        self.curation_status.object = f"saved as **{category}**"
        n_saved = self.curation_df[self.curation_df["session_key"] == sd.session_key].shape[0]
        self.progress_info.object = f"curated {n_saved}/{sd.n_rois} in this session"
        if advance:
            self._select_roi(self._clamp_roi(self.roi_index + 1))

    # ---------------------------- layout ----------------------------

    def layout(self) -> pn.viewable.Viewable:
        nav = pn.Row(
            self.session_select, self.roi_input,
            self.prev_btn, self.next_btn,
            self.roi_info, self.progress_info,
        )

        # Combine F + dFF in a single Bokeh column so Panel only manages one
        # Bokeh model boundary for the trace area — fewer layout reflows.
        traces_bk = bk_column(self.f_fig, self.dff_fig, sizing_mode="stretch_width")
        traces = pn.pane.Bokeh(traces_bk, sizing_mode="stretch_width")

        image_controls = pn.Column(
            pn.Row(self.image_mode, self.mask_toggle),
            self.zoom_pad,
            pn.Row(self.contrast_low, self.contrast_reset),
            self.contrast_high,
            sizing_mode="fixed", width=360,
        )
        right_col = pn.Column(
            image_controls,
            pn.pane.Bokeh(self.image_fig),
            pn.layout.Divider(),
            *[pn.pane.Bokeh(self.metric_figs[name]) for name in METRIC_ORDER],
            sizing_mode="fixed", width=360,
        )

        body = pn.Row(
            traces,
            right_col,
            sizing_mode="stretch_width",
        )

        curation_row = pn.Row(
            pn.pane.Markdown("**Good baselines:**", margin=(8, 6, 0, 6)),
            self.baseline_checks,
            self.undecided_toggle, self.save_btn, self.next_save_btn,
            self.curation_status,
        )

        keyboard = pn.pane.HTML(
            """
            <script>
            (function() {
              if (window.__qc_keys_bound) return;
              window.__qc_keys_bound = true;
              const click = (startsWith) => {
                const btns = Array.from(document.querySelectorAll('button'));
                const b = btns.find(x => (x.innerText || '').trim().startsWith(startsWith));
                if (b) b.click();
              };
              document.addEventListener('keydown', function(e) {
                if (e.target && ['INPUT','TEXTAREA','SELECT'].indexOf(e.target.tagName) >= 0) return;
                if (e.key === 'j' || e.key === 'J') click('◀');
                else if (e.key === 'k' || e.key === 'K') click('Next');
                else if (e.key === 's' || e.key === 'S') click('Save (S)');
                else if (e.key === ' ') { e.preventDefault(); click('Save + Next'); }
              });
            })();
            </script>
            """,
            height=0, margin=0,
        )

        return pn.Column(nav, body, curation_row, keyboard, sizing_mode="stretch_width")


# Make the module servable via ``panel serve``.
if pn.state.curdoc is not None or __name__ == "__main__":
    build_app().servable(title="Baseline QC")
