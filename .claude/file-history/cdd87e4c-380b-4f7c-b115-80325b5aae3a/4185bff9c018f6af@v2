# Session 01 — Baseline QC app

## Goal
Build an interactive QC app for inspecting per-ROI baseline-fitting results
produced by `long_vs_short_baselin_window.ipynb`. Curate which baselines look
reasonable so we can later use the labels to tune protocols / parameters.

Inputs (per session, under `/root/capsule/scratch/first_try/<subject>_<date>/`):

- `F_all_array.npy` — corrected F, shape (N_rois, T)
- `baseline_short_window_all_array.npy`, `baseline_long_window_all_array.npy`
- `F0_all.npy`, `F0trend_all.npy`
- `dff_short_window_all_array.npy`, `dff_long_window_all_array.npy`
- `timestamps.npy`
- Per-ROI metrics: `F_noise.npy`, `F_snr.npy`, `bleaching_metric.npy`,
  `sustained_metric.npy`, `F_skewness.npy`
- `sczdrift_df_all.csv` — has `plane_id`, `cell_roi_id` aligned with the ROI
  axis of all arrays.

ROI/FOV imagery comes from
`/root/capsule/data/multiplane-ophys_<subject>_<date>_*_processed_*/<plane_id>/extraction/<plane_id>_extraction.h5`
(`maxImg` for the FOV; `rois/{coords,data,shape}` for the sparse pixel masks).

## Tool choice
**Bokeh + Panel.** PyQt5 was the only GUI lib pre-installed but won't
work headlessly in the capsule. Bokeh has shared-range linked plots, native
mouse-wheel + box zoom, and image overlays; Panel adds widgets and runs in
both Jupyter and standalone server. Installed `panel==1.8.10`, `bokeh==3.9.0`.

## Module layout (`/root/capsule/code/qc_app/`)
- `data.py` — session listing, lazy-loaded arrays (`mmap_mode='r'`),
  per-session `SessionData` dataclass, `aggregate_metrics()` for distributions,
  `find_processed_dir()` to map session_key → processed asset dir.
- `rois.py` — `load_plane_assets()` reads FOV (`maxImg`) and the sparse
  pixel-masks; `crop_around_mask()` returns a padded zoom; `mask_to_rgba()`
  builds a translucent overlay rendered via `image_rgba`.
- `plots.py` — `make_f_figure`, `make_dff_figure` (sharing `x_range` for
  linked pan/zoom), `make_image_figure`, `make_metric_histogram`. Tools:
  `xpan,xbox_zoom,xwheel_zoom,reset,save`. WebGL backend on traces for speed.
- `curation.py` — append-only CSV at
  `/root/capsule/scratch/first_try/curation.csv`. Re-saving an
  `(session_key, roi_index)` overwrites the prior decision. Category
  `single`/`multiple`/`none`/`undecided` derived from the checkbox count.
- `app.py` — `_AppState` ties widgets, plots, and callbacks together.
  Exposes `build_app()` and is also `panel serve`–able.

## Key design decisions
- `mmap_mode='r'` on the big `.npy` arrays so switching ROIs is essentially
  free (only a single row touched at a time).
- Linked X axis between F and dFF panels via a shared `Range1d` (Bokeh native).
- Image overlay built once per ROI, with a `Zoom` ↔ `Full FOV` toggle and a
  zoom-pad slider.
- Metric histograms aggregate across **all** sessions in
  `/root/capsule/scratch/first_try`, with a red `Span` that updates to mark
  the current ROI's value.
- Keyboard shortcuts (`J`/`K`/`S`/Space) injected via a tiny JS snippet so
  curation is fast — clicks the visible buttons by name to stay in sync with
  Panel's state.
- `mmap`/`functools.lru_cache` reduces repeat I/O when the user revisits
  recently viewed sessions or planes.

## Pitfalls hit
- Bokeh `figure(x_range=None)` errors → only pass `x_range` when given.
- `panel serve` puts the script's directory (not its parent) on `sys.path`,
  so `from qc_app import …` failed. Fixed by prepending the package parent
  to `sys.path` at the top of `app.py`.
- `curation.load_curation` choked on `str` paths → coerced to `Path` inside.

## How to run
- Inline (Jupyter): open `code/qc_app/launch_in_notebook.ipynb`.
- Standalone web app:
  ```
  panel serve /root/capsule/code/qc_app/app.py \\
      --port 5006 --address 0.0.0.0 --allow-websocket-origin='*'
  ```
  Then visit the capsule's port-5006 URL.

## Verified
- `build_app()` constructs cleanly; `Document` serialization yields 2 roots /
  351 models with no errors.
- Stepping ROIs / changing sessions / saving decisions / save+next all work
  via simulated widget value changes (`/tmp/qc_test_curation.csv`).
- `panel serve` returns HTTP 200 on `/app` and `/app/autoload.js` with no
  server-side handler errors after the `sys.path` fix.

## Outputs
- Curation file: `/root/capsule/scratch/first_try/curation.csv`
- Columns: `session_key, roi_index, plane_id, cell_roi_id, selected,
  category, undecided, timestamp`.
