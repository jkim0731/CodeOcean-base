---
name: project-oasis-speed
description: OASIS deconvolution is very slow (~170s/file) due to internal multiprocessing; outer parallelization would conflict
metadata: 
  node_type: memory
  type: project
  originSessionId: f2b8ba44-1a76-46fe-ad80-8d81c2bba375
---

OASIS deconvolution of dFF arrays (~500 ROIs × ~9000 frames) takes ~170 s/file serially.

**Why:** `run_oasis.py` spawns a `multiprocessing.Pool` internally (one worker per ROI). Adding an outer parallel loop over the 756 dFF files would create competing process pools and likely slow things down further or cause resource contention.

**How to apply:** When revisiting deconvolution, benchmark a single-file run and profile whether the bottleneck is CPU-bound (pool workers) or I/O-bound (loading large .npy). Consider: (1) flattening all ROIs across all files into one pool call, (2) using joblib with a single backend and pre-loading traces in batches, or (3) running OASIS per-session (all 18 dFF files for one session in one pool call) rather than per-file.
