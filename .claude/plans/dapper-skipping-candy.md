# Baseline-fitting parameter sweep + comparison workflow

## Context

**Why this is needed.** The dFF baseline-fitting protocol in `code/baseline_fitting.py` has many parameters and no agreed QC metric. Today, every parameter set is encoded ad-hoc in a notebook (e.g. `long_vs_short_baseline_window.ipynb` → `scratch/first_try/`, `scratch/trend_only_default/`), so it is hard to: (a) re-run a config exactly, (b) compare multiple configs side-by-side in the existing PyQt5 QC app, (c) record the *protocol* that derived a parameter (e.g. `b_init = mean(F − baseline_long)`) rather than the resolved scalar.

**Goal of this plan.** Decide on a workflow for (i) declaring a parameter set as a serializable "recipe", (ii) running it on **one session at a time** while iterating (the runner accepts `--sessions <key>[,<key>...]` so a later sweep across all 28 is one flag away), (iii) extending `code/dff_baseline_search_qc_app/` so the user can pick N runs by recipe field and overlay their `F0trend` (and/or `F0`) traces on the existing per-ROI panel.

**Existing constraints discovered during exploration.**

- `fit_baseline` is the atomic unit. Notebook call (`long_vs_short_baseline_window.ipynb` cell `4cdf2e68` and `code/dff.py` does *not* wrap it):
  - `x0 = [F.mean(), b_init, b_init, b_init, t_max/2, 60, t_max/2]`
  - `b_init = mean(F − baseline_long_window)` (per-ROI)
  - `bounds = [(0,None)]*4 + [(300, t_max*5), (1, 300), (300, t_max*5)]`
  - `M = AsymmetricTukeyBiweight(c_pos=2, c_neg=3)`
  - `fixed_sigma = noise_std(F_all_array, 'mad')` (per-ROI; from `aind_ophys_utils.signal_utils`)
  - `model` = 7-param custom `b_inf + b_slow·E_slow + b_fast·E_fast − b_bright·E_bright`
  - `backend='jax', dtype=jnp.float32`
  - `fit_baseline` is called per ROI under `joblib.Parallel(n_jobs=-1, backend='loky')`.
- Existing GUI (`code/dff_baseline_search_qc_app/qc_app/`) is **PyQt5 + pyqtgraph**, desktop-only. Loads from `parent_dir/<session>/` (default `scratch/first_try`) using a fixed `_BASELINE_FILES` dict in `qc_app/data.py:52-57` mapping {`short`, `long`, `F0trend`, `F0`} → npy filenames. Each ROI panel toggles up to 4 baselines via numeric keys 1–4 — perfect mounting point for "compare N runs".
- Pre-installed: `pydantic`, `optuna`, `click`, `PyYAML`, `joblib`, `jax`. **NOT installed**: MLflow, Hydra, OmegaConf, W&B.
- Disk is the bottleneck: ~794 MB per session × 28 sessions = **22 GB per fully-duplicated run**. Per-run output (F0trend + F0 + res + loss for ~1 k ROIs × ~30 k frames in float32) is ~250 MB/session, ~7 GB/run. Inputs (F, baselines, metrics) should be referenced from a single canonical location, not copied.
- CodeOcean conventions: `/scratch/` for intermediates, `/results/` for capsule outputs, `code/run` is the entry point.

---

## Three candidate workflows

### Workflow A — JSON recipe + numbered run folders + index CSV (recommended)

Each parameter set is a `recipe.json` validated by a Pydantic model. Each component (`x0`, `sigma`, `bounds`, `M`, `model`, `fluctuations`) is a discriminated union: a `kind` string names a function in a small registry plus its kwargs. Resolution at run time turns the recipe into the concrete `(x0_array, sigma_array, bounds, M_instance, model_callable, fit_baseline_kwargs)` tuple.

**Recipe shape.** Two stages: `trend` (parametric IRLS → `F0trend`) and `fluctuations` (LOWESS or percentile → `F0`). Each component is a discriminated union on a `kind` / `method` field so Pydantic enforces one valid shape at a time. Calibration recipe `first_try.json` (must reproduce `scratch/first_try/<session>/F0trend_all.npy` and `F0_all.npy`):

```json
{
  "schema_version": 1,
  "description": "first_try replication: noise_std(mad), b_init from mean(F - long_baseline), t_high=t_max*5, lowess fluctuations",

  "model":  {"kind": "biexp_bright_v1"},
  "x0":     {"kind": "biexp_bright_default",
             "b_init_from": "mean_F_minus_long_baseline",
             "t_fast_init": 60.0,
             "t_slow_init_from": "t_max/2",
             "t_bright_init_from": "t_max/2"},
  "sigma":  {"kind": "noise_std", "method": "mad"},
  "bounds": {"kind": "biexp_bright_default", "t_high_factor": 5, "t_fast_max": 300},
  "M":      {"kind": "AsymmetricTukeyBiweight", "c_pos": 2, "c_neg": 3},

  "fluctuations": {
    "method": "lowess",
    "mode":   "ratio",
    "frac":   0.1,
    "M":      {"kind": "same_as_trend"},
    "maxiter": 5,
    "tol":     1e-3
  },

  "fit": {"backend": "jax", "dtype": "float32",
          "maxiter": 5, "tol": 1e-3,
          "optimizer_options": {"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}}
}
```

The `fluctuations` block is a Pydantic discriminated union on `method`. The percentile branch (e.g. `recipes/lowess_to_percentile.json`) replaces it with:

```json
"fluctuations": {
  "method":     "percentile",
  "mode":       "ratio",
  "frac":       0.1,
  "percentile": null
}
```

`percentile: null` keeps the existing auto-estimate (from `fit_baseline_fluctuations` lines 721–726: weighted-mean rank clipped to `[5, 50]`). Set a number to override.

Field-by-field validation rules the Pydantic model enforces:

| Field | Type | Notes |
|---|---|---|
| `model.kind` | `Literal["biexp_bright_v1"]` | Open enum — add `kind`s as new models are registered |
| `x0.kind` | `Literal["biexp_bright_default", ...]` | Each `kind` has its own typed sub-fields |
| `x0.b_init_from` | `Literal["mean_F_minus_long_baseline", "mean_F_minus_short_baseline", "zero", "scalar"]` | If `"scalar"`, requires `b_init_value: float` |
| `sigma.kind` | `Literal["noise_std", "fixed_value", "mad_residual"]` | `noise_std` requires `method ∈ {"mad","fft","welch"}`; `fixed_value` requires `value: float`; `mad_residual` computes MAD inside `fit_baseline` instead of fixing it |
| `bounds.kind` | `Literal["biexp_bright_default", "biexp_bright_unbounded", ...]` | `t_high_factor`, `t_fast_max` configurable |
| `M.kind` | `Literal["AsymmetricTukeyBiweight", "OneSidedTukeyBiweight", "TukeyBiweight"]` | Each has its own `c` / `c_pos` / `c_neg` requirements |
| `fluctuations.method` | `Literal["lowess", "percentile"]` | Discriminator — only the matching branch's fields are required/allowed |
| `fluctuations.mode` | `Literal["ratio", "subtract"]` | Shared by both branches |
| `fluctuations.frac` | `float ∈ (0, 1]` | Shared |
| `fluctuations.M` (lowess only) | `MSpec ∪ {"kind": "same_as_trend"}` | Allows reuse of trend-stage M or explicit override |
| `fluctuations.maxiter`, `tol` (lowess only) | `int`, `float` | Forbidden in percentile branch |
| `fluctuations.percentile` (percentile only) | `float ∈ [0,100]` or `null` | `null` → auto-estimate; forbidden in lowess branch |

**Outputs saved per `(run, session)`** are now both stages, matching what the GUI consumes today:

```
scratch/runs/0001_<slug>/<session>/
  F0trend_all.npy        # (N,T) — trend stage
  F0_all.npy             # (N,T) — full baseline (trend × fluctuation, or trend + fluctuation)
  res_all.npy            # final OptimizeResult.x for the trend per ROI: (N, n_params)
  loss_all.npy           # M-estimator loss per ROI: (N,)
  info.json              # per-ROI diagnostics: {"trend": {sigma, weights_summary, nit, success},
                         #                       "fluctuations": {lowess_sigma | percentile, size}}
```

**Disk layout.** Numbered + slug, parallel to existing `first_try/` and `trend_only_default/`:

```
scratch/runs/
  index.parquet                      # one row per run, recipe fields flattened (parquet for fast filtering; CSV mirror written too)
  0001_first_try_replication/
    recipe.json
    metadata.json                    # {created_at, host, git_rev, code_version, sessions, n_rois, runtime_s}
    755252_2024-11-12/
      F0trend_all.npy                # (N, T) float32 — only file the GUI strictly needs
      F0_all.npy
      res_all.npy
      loss_all.npy
      info.json                      # per-ROI: sigma, weights summary, lowess_sigma, fit nit, success flag
    755252_2024-11-19/
      ...
  0002_<next>/
    ...
```

Inputs (`F_all_array.npy`, `timestamps.npy`, `baseline_long_window_all_array.npy`, the metric arrays, `sczdrift_df_all.csv`, plane images) stay under the canonical `scratch/first_try/<session>/` and are referenced by path from `metadata.json` → `inputs_dir`. **No duplication.**

**Code components.**

- `code/baseline_search/recipe.py` — Pydantic models (`Recipe`, `X0Spec`, `SigmaSpec`, `BoundsSpec`, `MSpec`, `ModelSpec`, `FluctuationsSpec`, `FitSpec`) with discriminated unions on `kind`. `Recipe.model_validate_json` / `model_dump_json` for I/O. JSON Schema is auto-derived for free.
- `code/baseline_search/registry.py` — module-level dicts `X0_FNS`, `SIGMA_FNS`, `BOUNDS_FNS`, `MODEL_FNS`, `M_FNS`. Each entry is a callable. To add a new variant later: define the function, register a `kind` string. Pydantic gates which kinds are valid.
- `code/baseline_search/resolve.py` — `resolve(recipe, F_array, timestamps, inputs_dir) -> ResolvedFit` returning everything `fit_baseline` needs. Existing notebook protocol becomes the *first* registered set: `noise_std/mad`, `biexp_bright_default` x0/bounds, `AsymmetricTukeyBiweight(2,3)`.
- `code/baseline_search/run.py` — CLI (Pydantic `BaseSettings` like `dff.py:34`) that: (1) loads inputs from `--inputs-dir`, (2) resolves the recipe, (3) parallelizes per-ROI `fit_baseline` via `joblib.Parallel`, (4) writes outputs + metadata + appends a row to `runs/index.parquet`. Run as `python -m baseline_search.run --recipe path/to/recipe.json --inputs-dir scratch/first_try --out scratch/runs --slug first_try_replication --sessions 755252_2024-11-12,...`.
- `code/baseline_search/recipes/` — checked-in JSON files (one per parameter set you want to keep). The "first_try replication" recipe is the calibration target: re-running it must reproduce `scratch/first_try/<session>/F0trend_all.npy` byte-for-byte.

**GUI extension.** Smallest possible change to `qc_app/`:

- New module `qc_app/runs.py`: `discover_runs(runs_dir) → DataFrame` (loads `index.parquet`); `load_run_arrays(run_dir, session_key, kinds=("F0trend","F0")) → dict`.
- `qc_app/data.py`: extend `load_session` to accept an optional list of `(run_id, kind, label)` tuples — `kind` ∈ `{"F0trend","F0"}` — and append them as additional entries in the `baselines` dict. The GUI's existing 4-baseline toggle machinery (`app.py:479-521`) keeps working — slots 1–4 just become user-selectable `(run_id, kind)` pairs instead of the hardcoded short/long/F0trend/F0.
- New "Compare runs" dialog (PyQt `QDialog`): a `QTableView` over the runs index with column filters (e.g. `sigma.method == "mad"`, `fluctuations.method == "percentile"`); the user picks up to 4 `(run, F0trend|F0)` pairs; the dialog calls `load_session` with the selection and the existing TracePanel re-renders. Three natural pick patterns the dialog should make easy: (a) one run, both `F0trend` and `F0` shown — like the legacy view but for an arbitrary recipe; (b) up to 4 runs' `F0trend` — compare trend-stage decisions; (c) up to 4 runs' `F0` — compare full-baseline decisions, including LOWESS-vs-percentile fluctuations.
- A "Run mode" radio in the toolbar switches between **legacy** (current 4 hardcoded baselines from `first_try/`, default) and **runs** (compare-N from `runs/`). The existing curation behavior is undisturbed for legacy mode.

**Pros.** Plain files, easy to git, easy to diff. Recipes are one human-readable JSON. Pydantic gives validation, schema generation, and IDE completion for free. The numbered-folder layout is what the user already does informally — this just formalizes it. GUI change is additive (≈2 new files + ~50 lines in `app.py`, `data.py`).

**Cons.** No fancy dashboard. No automatic hyperparameter search. The registry is a small piece of Python you have to maintain (adding a new `x0` variant = ~10 lines + a `kind` string).

**Effort.** Recipe + registry + runner: ~1–2 days. GUI extension: ~half a day.

---

### Workflow B — MLflow tracking server, file:// backend

Each run becomes an MLflow run inside an experiment. Recipe fields → `mlflow.log_param`. F0trend arrays → `mlflow.log_artifact`. The MLflow UI handles the parameter search; the user picks N runs and the GUI reads the artifacts.

**Disk.** `mlruns/` directory under `scratch/`. MLflow auto-creates per-run subfolders with hashes (not numeric, but it can be fronted by a name).

**Code components.**

- `pip install mlflow` (~150 MB with deps — adds to image size).
- `baseline_search/run.py` wraps the per-run loop in `with mlflow.start_run(run_name=...)`; logs all recipe leaves as params. Per-session F0trend npy → artifact. Per-ROI fit metadata → CSV artifact.
- For the GUI, `mlflow.search_runs(filter_string=...)` returns a DataFrame; `mlflow.artifacts.download_artifacts(run_id, "F0trend_all.npy")` materializes a path. Same TracePanel wiring as Workflow A.

**Pros.** Battle-tested experiment tracking with a ready-made UI for filtering by parameter. REST API if you ever want a notebook to ask "what runs had `sigma.method=mad` and `c_pos<3`?".

**Cons (significant for this project).**

- New service to install + run. The MLflow UI is a separate web app; the desktop PyQt5 GUI still needs custom plotting, so MLflow's UI is **redundant** — it doesn't visualize fluorescence traces.
- Artifacts are blobs in nested hash dirs. Looking up "where is run X's F0trend for session Y" is harder than `runs/0003_*/755252_2024-11-12/F0trend_all.npy`.
- 28 sessions × ~1 k ROIs × ~30 k frames per run = ~3 GB of artifacts per run, slow to upload to MLflow's local store.
- Recipes-as-flat-params loses structure. MLflow params are `Dict[str, str]`, max 250 chars per value, and hierarchical recipes (`x0.b_init_from`) become flattened keys you re-assemble manually.
- `model` callable, `M` instance — same registry-by-name problem as Workflow A. MLflow doesn't help here.
- Adds a dependency CodeOcean has to build into the image.

**When this becomes the right call.** If you eventually run the sweep across many machines, share results with collaborators, and want a hosted UI you don't maintain.

---

### Workflow C — Hydra/OmegaConf structured configs + multirun

Recipes live as YAML with composition (`defaults: [model: biexp_bright, sigma: mad, bounds: t_high_5x]`). Hydra's `--multirun` runs the cartesian product of overrides automatically. Each run gets a timestamped output dir.

**Code components.**

- `pip install hydra-core omegaconf` (not installed today).
- `code/baseline_search/conf/` tree of YAML fragments.
- `run.py` decorated with `@hydra.main(config_path="conf", config_name="recipe")`.
- Hydra writes to its own `outputs/YYYY-MM-DD/HH-MM-SS/` layout, or `multirun/...`. To match the user's "numbered folders" requirement, override `hydra.run.dir` and `hydra.sweep.dir`.

**Pros.** Best-in-class config composition. CLI sweep is one line: `python run.py --multirun sigma.method=mad,welch x0.kind=biexp_bright_default,zero_init`. Resolved config saved alongside output automatically.

**Cons.**

- New dep. Hydra is opinionated about its output dir layout; making it produce `runs/0001_<slug>/` cleanly takes some wrestling.
- YAML composition is great when recipes share large skeletons, but you only have one model and one M-estimator class to compose — composition mostly overkill for this scope.
- The Pydantic discriminated-union approach in Workflow A captures the same structure with no new dependency, *plus* JSON-schema validation, *plus* it survives a notebook (`Recipe.model_validate_json(open("recipe.json").read())` works in any Python REPL without Hydra's `@main` decorator).

**When this becomes the right call.** Once you have ≥3 independent components each with multiple variants and want to sweep cartesian products without writing the loop yourself.

---

### Optional D — Optuna (only if you add an objective)

Optuna *is* installed. But Optuna optimizes against an objective; without a QC metric there is nothing to maximize, so Optuna degenerates into a glorified grid runner. Reach for it once you have a QC metric — then `objective(trial)` calls `fit_baseline` with `trial.suggest_*` parameters and returns the metric. The recipe-and-registry layer from Workflow A is still needed to materialize trials into `fit_baseline` kwargs.

---

## Recommendation

**Workflow A.** It matches what the user already does informally (named folders under `scratch/`), uses only pre-installed deps, keeps recipes git-trackable, and the GUI extension is mostly additive on top of an existing 4-baseline toggle UI. MLflow's UI is redundant given the PyQt5 app already does the comparison plotting; Hydra's composition is overkill at this stage. Both can be added later (recipes become MLflow params; recipes become Hydra configs) without rewriting the runner — the registry stays.

---

## Critical files to add or modify

To add (Workflow A):

- `code/baseline_search/__init__.py`
- `code/baseline_search/recipe.py` — Pydantic models for the recipe
- `code/baseline_search/registry.py` — name→callable maps for x0 / sigma / bounds / M / model
- `code/baseline_search/resolve.py` — recipe → concrete fit args
- `code/baseline_search/run.py` — CLI + parallel per-ROI loop + writer
- `code/baseline_search/recipes/first_try.json` — calibration target (must reproduce existing `first_try/` outputs)
- `code/dff_baseline_search_qc_app/qc_app/runs.py` — discover/load run outputs

To modify:

- `code/dff_baseline_search_qc_app/qc_app/data.py` (`load_session`, `_BASELINE_FILES`) — accept additional run-derived baselines
- `code/dff_baseline_search_qc_app/qc_app/app.py` — add "Compare runs" dialog, run-mode toggle in toolbar
- `code/dff_baseline_search_qc_app/qc_app/curation.py` — extend `selected` schema so the "best fit" can name a `run_id`, not just `short/long/F0trend/F0`

To reuse without modification:

- `code/baseline_fitting.py:739` `fit_baseline` — atomic fit unit
- `aind_ophys_utils.signal_utils.noise_std` — sigma estimator
- `aind_ophys_utils.dff` — already used by `dff.py`
- `joblib.Parallel` — same as the existing notebook (`Parallel(n_jobs=-1, backend='loky')`)

---

## Verification

1. **Calibration test (one session, single source of truth).** Pick one session — recommend `755252_2024-11-19` since it is already loaded in `irls_stepwise_roi29.ipynb` and we know its expected fit behavior. Run:
   ```
   python -m baseline_search.run \
     --recipe code/baseline_search/recipes/first_try.json \
     --inputs-dir scratch/first_try \
     --out scratch/runs \
     --slug first_try_replication \
     --sessions 755252_2024-11-19
   ```
   Assert `np.allclose(scratch/runs/0001_first_try_replication/755252_2024-11-19/F0trend_all.npy, scratch/first_try/755252_2024-11-19/F0trend_all.npy, atol=1e-5)` and the same for `F0_all.npy`. If it does not match byte-for-byte (modulo float tolerance), the recipe-resolution code has drifted from the notebook. **Do not generalize to other sessions until this test passes** — iterate the recipe / registry on this single session first. Once green, the same recipe + `--sessions all` produces all 28 runs.
2. **Round-trip recipe.** `Recipe.model_validate_json(open(p).read()).model_dump_json()` must equal the on-disk JSON modulo whitespace. JSON-Schema (`Recipe.model_json_schema()`) printed once and saved as `code/baseline_search/recipe.schema.json` for editor autocomplete.
3. **Index integrity.** After two runs, `pd.read_parquet("scratch/runs/index.parquet")` has 2 rows with distinct `run_id`s and parseable `recipe_path`s; flattened recipe fields (`sigma_method`, `M_c_pos`, ...) are queryable: `df.query("sigma_method == 'mad' and M_c_pos == 2")` returns row 1.
4. **GUI compare.** Launch `dff-qc --parent-dir scratch/first_try --runs-dir scratch/runs`. Open the new "Compare runs" dialog, filter by `sigma.method == "mad"`, pick 2 runs, confirm both `F0trend` traces overlay on the existing F + raw trace panel and toggle on/off via keys 1–2. Step through 5 ROIs without errors; the curation CSV records the selected run_id.
5. **No regression on legacy mode.** With `--runs-dir` omitted, the GUI behaves identically to today (same 4 hardcoded baseline files, same keyboard shortcuts). Curation CSVs from the legacy mode remain readable.

---

## Open decisions to surface at exit

- **Numbering scheme.** `0001_<slug>/` (zero-padded + slug for human readability) vs. pure `0001/` with the slug only in `metadata.json`. Recommendation: zero-padded + slug.
- **GUI layout.** Use the existing 4-baseline toggle for runs (max 4 simultaneously) vs. add an unlimited "compare mode" with a stacked legend. Recommendation: reuse the existing 4-toggle for now — least disruption, matches keys 1–4.
- **Where the "first_try" baseline files (`baseline_short`, `baseline_long`) belong.** They are *inputs* to recipes (used to derive `b_init`), not outputs of recipes. They stay in `scratch/first_try/<session>/` and run folders reference them via `metadata.json:inputs_dir`.
- **Slot assignment when both `F0trend` and `F0` exist per run.** With 4 toggle slots, three reasonable defaults: (a) one run filling 2 slots (`F0trend` + `F0`) — closest to the existing per-recipe view; (b) 4 runs × `F0trend`; (c) 4 runs × `F0`. The Compare-runs dialog should expose all three as one-click presets and let the user mix in the table view. Recommendation: ship (a) as the default when one run is picked; auto-switch to (b) when 2+ runs are picked; user can override.
