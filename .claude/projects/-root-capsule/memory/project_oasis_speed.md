---
name: project-oasis-speed
description: OASIS deconvolution status — complete for 0001/0002, partial for 0003, pending for 0004–0008
metadata: 
  node_type: memory
  type: project
  originSessionId: f2b8ba44-1a76-46fe-ad80-8d81c2bba375
---

OASIS deconvolution of dFF arrays (~500 ROIs × ~9000 frames).

**Completion status (in /scratch/runs/ as of 2026-05-14):**
- `0000_first_try`: 42/42 complete (events_short.h5 + events_long.h5)
- `0001_binit0_c23`: 42/42 complete
- `0002_binit0_c24`: 42/42 complete
- `0003_binit0_c25`: ~12–13/42 complete (partial)
- `0004_binit0_c33` through `0008_binit0_c45`: 0/42 each

**To run remaining (~479 files × ~7 GB):** Use `run_oasis_all.py` which reads from `RUNS_DIR = /results/runs`. Must be run from a proper capsule Run (not interactive) because /results is root-owned in interactive mode.

**Why deferred:** In interactive mode /results/ and /output/ are root-owned; /code has ~9.2 GB free (barely enough but risky). The right time to run is during a capsule Run where /results/ is writable.

**Why serial:** `run_oasis.py` spawns an internal Pool per call. `run_oasis_all.py` already runs serially (tqdm loop). No outer parallelism needed — internal Pool handles concurrency. The ~170s/file estimate was from a context with Pool conflict; serial should be faster.

**How to apply:** When /results/ is writable, `cd /code && python run_oasis_all.py`. It skips already-done files automatically.
