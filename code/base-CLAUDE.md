# base-CLAUDE.md — interacting with Code Ocean (search · attach · run · capture)

Drop this in (or `@`-include it from) a session's `CLAUDE.md` when the task is **Code Ocean data
orchestration**: finding data assets, attaching them, running a capsule/pipeline via the all-users
pipeline monitor, and capturing the result as a schema-compliant derived data asset.

It distills hard-won, working patterns for the AIND Code Ocean deployment
(`https://codeocean.allenneuraldynamics.org`). Read the **CAUTIONS** section before writing any
code — several of these cost real debugging time.

---

## 0. Environment & auth

- **Token**: `os.getenv("API_SECRET")` (set as a CO secret; the all-users monitor needs it too).
  **Never print/echo it** — transcripts are published; secrets must be scrubbed (see the global
  secret-redaction rule). The domain is `https://codeocean.allenneuraldynamics.org/`.
- **Client** (two equivalent entry points):
  ```python
  from codeocean import CodeOcean
  co = CodeOcean(domain="https://codeocean.allenneuraldynamics.org/", token=os.getenv("API_SECRET"))
  # or, simplest:
  from lamf_analysis.code_ocean import code_ocean_utils as cou
  co = cou.get_co_client()
  ```
- **Install** (env): `lamf-analysis` (gives `code_ocean_utils`, `zstack_utils`, `json_utils`),
  `codeocean==0.14.0`, `aind-codeocean-pipeline-monitor`, and — only if you write schema json —
  `aind-data-schema==1.2.0` (see CAUTIONS). `lamf-analysis` pulls a heavy dep tree (torch/dask/etc.).
- **Storage**: scratch/temp under `/scratch/tmp`, never `/tmp` or `/` (overlay fills + locks the env).

---

## 1. The capsule registry: `CO_capsule_infos.xlsx`

`/root/capsule/code/CO_capsule_infos.xlsx` is the **source of truth** for capsule ids, URLs, and the
result-capture **suffix/tags** each capsule should use. Always look up here rather than guessing.

```python
import pandas as pd                       # needs openpyxl: pip install --no-cache-dir openpyxl
df = pd.ExcelFile("/root/capsule/code/CO_capsule_infos.xlsx").parse("processing")
row = df[df["capsule name"].str.contains("autocoreg", case=False, na=False)]
# columns: Type, capsule name, Git link, capsule url, capsule id, suffix, result tags,
#          required data type, pre-attached data asset name/id, Note
```
- **`capsule id`** → use in `RunParams(capsule_id=...)` and as a capsule's `code_url`
  (`capsule url` = `…/capsule/<id>/tree`).
- **`suffix`** → the capture `process_name_suffix` AND the derived-asset process name. **Match it
  exactly** — downstream capsules search results *by this suffix*; drift breaks discovery.
- **`required data type`** tells you what to attach (e.g. `HCR processed (R1); HCR-ROI-label;
  cortical-zstack-registration; cortical-zstack-segmentation`).
- Sheets: `processing` (the pipeline capsules), `base and regular uses`, `current dev`.
- The shared **all-users pipeline monitor** is `aind-co-pipeline-monitor-capsule-all-users`,
  id `567b5b98-8d41-413b-9375-9ca610ca2fd3`.

---

## 2. Searching for data assets

```python
from codeocean.components import SearchFilter
from codeocean.data_asset import DataAssetSearchParams

params = DataAssetSearchParams(
    offset=0, limit=1000, archived=False, favorite=False,
    sort_order="desc", sort_field="name",      # or "created"
    filters=[SearchFilter(key="tags", value="HCR"),
             SearchFilter(key="name", value=f"HCR_{sid}")],   # name filter is a SUBSTRING
)
results = co.data_assets.search_data_assets(params).results   # list; each has .id .name .created .tags
```

**Derived-asset helpers** (`lamf_analysis.code_ocean.code_ocean_utils`):
```python
assets = cou.get_derived_assets(sid, "cortical-zstack-registration")      # list of asset objects
df     = cou.get_derived_assets_df(sid, "cortical-zstack-registration",   # tidy DataFrame
                                   add_s3_location=True)  # adds 's3_path' (else NO s3_path column!)
# df cols: derived_asset_id, derived_asset_name, raw_asset_name, session_name,
#          derived_date, derived_time [, s3_path]
```
- **Provenance / lineage**: `co.data_assets.get_data_asset(aid).provenance.data_assets` → list of
  input asset ids that produced `aid`. Walk it to find the raw/processed source of a derived asset.
- **Pick latest**: sort by `derived_asset_name` (acquisition dt sorts first) or `created`, take the
  last; break ties by `created`.

---

## 3. Attach + run + capture via the all-users pipeline monitor

A reproducible capsule **cannot attach assets to itself mid-run**. So a *monitor* capsule builds a
`PipelineMonitorSettings`, hands it to the **all-users monitor** capsule, which: attaches the assets
→ runs the target capsule → captures `/results` as a derived data asset.

```python
from aind_codeocean_pipeline_monitor.models import CaptureSettings, PipelineMonitorSettings
from codeocean.computation import RunParams, DataAssetsRunParam, NamedRunParam, ComputationState

ALL_USERS_MONITOR = "567b5b98-8d41-413b-9375-9ca610ca2fd3"
TARGET_CAPSULE_ID = "<from the excel sheet>"
SUFFIX            = "<from the excel 'suffix' column>"

settings = PipelineMonitorSettings(
    run_params=RunParams(
        capsule_id=TARGET_CAPSULE_ID,
        data_assets=[                                  # MOUNT NAME MUST MATCH the target's globs!
            DataAssetsRunParam(id=hcr_id,  mount=hcr_name),     # e.g. capsule globs HCR_<sid>_*_processed_*
            DataAssetsRunParam(id=reg_id,  mount=reg_name),
            # ...
        ],
        named_parameters=[NamedRunParam(param_name="subject_id", value=sid)],  # capsule app-panel params
    ),
    capture_settings=CaptureSettings(
        process_name_suffix=SUFFIX,                    # -> derived name <input>_<SUFFIX>_<dt>
        tags=["derived", "HCR", SUFFIX, sid],
        custom_metadata={"data level": "derived",
                         "experiment type": "HCR",     # CONTROLLED VOCAB — see CAUTIONS
                         "subject id": sid},
    ),
)
# hand the settings to the all-users monitor as its single JSON parameter:
comp = co.computations.run_capsule(RunParams(
    capsule_id=ALL_USERS_MONITOR,
    parameters=[settings.model_dump_json(exclude_none=True)]))
# poll: co.computations.get_computation(comp.id).state in {ComputationState.Completed, .Failed}
```

**How the captured asset is named** (`aind_codeocean_pipeline_monitor`): it reads
`/results/data_description.json`; if `data_level == "derived"` **and** the `name` matches
`DataRegex.DERIVED`, it uses that name; otherwise it falls back to
`<input_data_name>_<process_name_suffix>_<dt>`. So either ship a valid derived
`data_description.json` (see §5) or rely on `process_name_suffix`.

**Always provide a dry-run** (e.g. `--test 1`) that searches + prints the resolved assets but does
NOT trigger, and validate before a real run.

---

## 4. Interactive (cloud-workstation) capture

When a result is produced interactively (Ubuntu workstation), `/results` is ephemeral — capture a
persistent **`/scratch`** folder instead:
```python
from codeocean.data_asset import DataAssetParams, Source, CloudWorkstationSource
params = DataAssetParams(
    name=f"{sid}_{SUFFIX}_{ts}", mount=f"{sid}_{SUFFIX}_{ts}",
    tags=["derived", "HCR", SUFFIX, sid],
    custom_metadata={"data level": "derived", "experiment type": "HCR", "subject id": sid},
    source=Source(cloud_workstation=CloudWorkstationSource(
        id=os.getenv("CO_COMPUTATION_ID"), path="/root/capsule/scratch/<folder>")))
asset = co.data_assets.create_data_asset(params)
```
Prefer **one asset per subject** (a per-subject folder) so per-subject metadata is unambiguous.

---

## 5. docDB schema compliance (`data_description.json` + `processing.json`)

For a derived asset to land in docDB, write both files into the captured folder via
`lamf_analysis.code_ocean.json_utils.process_json_files(source_asset_name, capture_name,
start_dt, run_parameters, INPUT_PROCESSING_DICT, process_name=SUFFIX, process_level)`:
- `source_asset_name` = the **processed source asset base** = `asset.name.split("_processed_")[0]`
  (HCR/mFISH → `HCR_<sid>_<acq-dt>`; ophys → `multiplane-ophys_<sid>_<acq-dt>`). Must keep the
  acquisition datetime (the DERIVED-name regex requires it). `subject_id = source.split("_")[1]`.
- `capture_name` = `source_asset_name` (the lib appends `_<SUFFIX>_<creation-dt>`).
- `process_level="subject"` writes only the two files; any other value also copies
  subject/procedures/rig|instrument.json from the source.
- `INPUT_PROCESSING_DICT = {"name": "Other", "software_version": "...", "code_url": "<capsule url>",
  "notes": "..."}` (`name` must be a valid aind ProcessName; `"Other"` is always safe).
- Reproducible capsule → writes to `/root/capsule/results`. Interactive → set
  `json_utils.RESULTS_PATH = Path(<the /scratch folder being captured>)` first.

---

## 6. CAUTIONS (read before coding)

1. **`experiment type` is a controlled vocabulary.** `"coregistration"` is **rejected** on capture.
   Anything HCR/mFISH-related (incl. CZ↔HCR coregistration) must be `"experiment type": "HCR"`.
   Anything ophys related (incl. behavior videos) must be `"experiment type": "multiplane-ophys"`.
   `"data level"` is `"derived"`/`"raw data"`. (Free-form `tags` are unconstrained.)
2. **`_processed_` is a substring trap.** Searching name `HCR_{sid}` + filtering `_processed_` ALSO
   matches the derived `HCR_{sid}_..._processed_..._HCR-ROI-label_<dt>` asset. To get the PLAIN
   processed asset, anchor the match to the processing timestamp:
   `re.match(r"^HCR_.*_processed_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name)`.
3. **Never `rglob` over `/data`.** Processed assets contain segmentation **zarr** (10k+ files);
   recursive globs walk it and stall for minutes. Use one-level `glob("*<asset>*/file.json")` or
   check `dir / "file.json"` directly. (This bit `json_utils` — its upstream `rglob` was the cause.)
4. **`aind-data-schema` version pin.** `json_utils` hard-asserts `__version__ == "1.2.0"`; the base
   env ships 2.x → pin `aind-data-schema==1.2.0` (no numpy pin; needs `pydantic>=2.7`, fine with
   `pydantic<2.11`). If the capsule installs lamf-analysis (whose requirements.txt leaves
   aind-data-schema UNPINNED → 2.x), re-pin 1.2.0 **last** in postInstall. For lean capsules that
   only need the helper, **vendor `json_utils.py`** (it's self-contained: aind_data_schema + stdlib,
   empty `__init__`s) rather than dragging in lamf-analysis's heavy tree.
5. **Mount names must match the target capsule's data globs.** `DataAssetsRunParam(mount=...)` becomes
   `/data/<mount>/`; set `mount = asset_name` so e.g. `HCR_{sid}_*_processed_*` resolves.
6. **`.git` of deployed capsules under `/` is often root-owned** (pushed as root) → `claude-user`
   cannot commit/push (and cannot complete a root-started rebase: `.git/rebase-merge/*` is
   root-owned). Resolve the working tree + `git add`, then have the user run the commit / `git
   rebase --continue` as **root**.
7. **Capture name vs source name.** The captured derived name needs the source's acquisition
   datetime (DERIVED regex). `name.split("_processed_")[0]` preserves it; don't strip further.
8. **Env rebuild needed** after changing a capsule's env (Dockerfile/postInstall): bump the
   `# cache bust` token so Code Ocean re-clones/re-installs.
9. **Dry-run first**, log the resolved (asset id, name) tuples, and STOP with a clear message when a
    required upstream asset isn't found yet (don't silently trigger a partial run).

---

## 7. Quick reference

| Need | Call |
|------|------|
| client | `cou.get_co_client()` |
| search | `co.data_assets.search_data_assets(DataAssetSearchParams(filters=[SearchFilter(key=…, value=…)], …)).results` |
| derived list / df | `cou.get_derived_assets(sid, proc)` / `cou.get_derived_assets_df(sid, proc, add_s3_location=True)` |
| lineage | `co.data_assets.get_data_asset(aid).provenance.data_assets` |
| run a capsule | `co.computations.run_capsule(RunParams(capsule_id, data_assets=[…], named_parameters=[…]))` |
| attach+run+capture | hand `PipelineMonitorSettings(...).model_dump_json()` to capsule `567b5b98-…` |
| poll | `co.computations.get_computation(id).state` → `ComputationState.{Completed,Failed}` |
| interactive capture | `co.data_assets.create_data_asset(DataAssetParams(source=Source(cloud_workstation=CloudWorkstationSource(id=os.getenv("CO_COMPUTATION_ID"), path=…))))` |
| schema json | `json_utils.process_json_files(source_name, source_name, start_dt, params, INPUT_PROCESSING_DICT, SUFFIX, "subject")` |
