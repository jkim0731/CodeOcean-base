# Attach + run + capture via the all-users pipeline monitor

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
`<raw_data_name>_<process_name_suffix>_<dt>` (raw_data_name consists of modality, subject id, date, and time, connected with `_` - having anything after that, i.e., more than 3 `_`s mean that it is derived, so strip them). So either ship a valid derived
`data_description.json` (see §5) or rely on `process_name_suffix`.

**Always provide a dry-run** (e.g. `--test 1`) that searches + prints the resolved assets but does
NOT trigger, and validate before a real run.