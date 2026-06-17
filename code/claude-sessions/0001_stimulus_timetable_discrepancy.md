# 0001 — Stimulus Timetable Discrepancy: comb vs. aind-metadata-mapper

**Date:** 2026-06-04  
**Versions compared:**  
- [`aind-metadata-mapper` v0.29.3](https://github.com/AllenNeuralDynamics/aind-metadata-mapper/blob/v0.29.3/src/aind_metadata_mapper/stimulus/camstim.py)  
- [`comb` branch `for_gcamp_validation`](https://github.com/AllenNeuralDynamics/comb/blob/for_gcamp_validation/src/comb/behavior_session_dataset.py)

---

## Summary

On a subset of sessions, stimulus presentation timestamps produced by **aind-metadata-mapper** are shifted **+1 second** relative to those from **comb**. The comb result was verified as correct against neuronal responses. The root cause is a divergence in how the two codebases compute the **monitor delay** from the photodiode signal — specifically, an extra large-rise fallback in aind-metadata-mapper that detects a wrong `ptd_start` when the normal pattern-matching fails.

---

## Pipeline Overview

### comb (`behavior_session_dataset.py`)

```
get_synchronized_frame_times()
  └─ SyncDataset.get_edges("rising", STIMULUS_KEYS)   ← RISING edges of vsync_stim
  └─ trim_discontiguous_times(threshold=100)

get_stimulus_presentations(monitor_delay=0.03613)      ← hardcoded by default
  └─ StimulusTimestamps(timestamps=rising_vsync, monitor_delay=delay)
     └─ final timestamps = rising_vsync + delay
```

When `calculate_monitor_delay=True` is exposed:
```
get_monitor_delay_stage_1()
  └─ calculate_monitor_delay_visual_coding(photodiode_rise, vsync_stim_fall)
     └─ returns (delay, delay_std)
     └─ if ptd_start/ptd_end not found → returns ASSUMED_DELAY = 0.0351 s, std = 0
```

### aind-metadata-mapper (`camstim.py`, `build_behavior_table`)

```
get_ophys_stimulus_timestamps()
  └─ get_clipped_stim_timestamps()
     └─ get_behavior_stim_timestamps() → FALLING edges of vsync_stim
     └─ clips to pkl data length
     └─ removes first frame if rising[1]-rising[0] > 0.2 s (DAQ spike)

extract_frame_times_with_delay()
  └─ calculate_frame_mean_time() → finds ptd_start, ptd_end
  └─ computes delay = mean(photodiode_rise[ptd_start+i] - vsync_fall[i*120+60])
  └─ returns scalar delay

final timestamps = falling_vsync + delay
```

---

## Differences Between the Two Codebases

### 1. Vsync edge direction

| | comb | aind-metadata-mapper |
|---|---|---|
| Edge used | **Rising** | **Falling** |

Rising and falling edges of the vsync pulse differ by the pulse width. At 60 Hz this is a small consistent offset (≪1 ms for a brief TTL pulse), not a session-specific effect.

### 2. Monitor delay: hardcoded vs. dynamic

| | comb (default) | aind-metadata-mapper |
|---|---|---|
| Method | Fixed `0.03613 s` | Dynamic from photodiode per session |
| Fallback when detection fails | `ASSUMED_DELAY = 0.0351 s` | `ASSUMED_DELAY = 0.0356 s` |

The difference between comb's hardcoded value and comb's dynamic result is only **~1 ms** across all tested sessions, confirming that this is not the source of the 1-second discrepancy.

### 3. Frame count clipping

| | comb | aind-metadata-mapper |
|---|---|---|
| Strategy | `trim_discontiguous_times` — trims after first gap > 100× median interval | Clips timestamp array to match PKL data length |
| Initial spike | None | Removes first frame if `rising[1]−rising[0] > 0.2 s` |

Different arrays may result for sessions with trailing extra sync pulses or an initial DAQ spike.

### 4. ptd_start detection — **root cause of the 1-second error**

Both codebases use identical medium-rise pattern matching (short: 0.1–0.3 s, medium: 0.5–1.5 s intervals). However, aind-metadata-mapper has an **extra fallback** when fewer than 3 medium rises are found:

**comb `calculate_monitor_delay_visual_coding`:**
```python
for medium_rise_index in medium_rise_indexes:
    if set(range(medium_rise_index - 2, medium_rise_index)) <= short_set:
        ptd_start = medium_rise_index + 1
    elif set(range(medium_rise_index + 1, medium_rise_index + 3)) <= short_set:
        ptd_end = medium_rise_index
# if < 3 medium rises: ptd_start or ptd_end stays None → ASSUMED_DELAY returned
```

**aind-metadata-mapper `calculate_frame_mean_time`:**
```python
if len(medium_rise_indexes) < 3:           # ← extra branch absent in comb
    large_rise_indexes = np.where(
        (photodiode_rise_diff > 1.9) & (photodiode_rise_diff < 2.1)
    )[0]
    for large_rise_index in large_rise_indexes:
        if set(range(large_rise_index - 2, large_rise_index)) <= short_set:
            ptd_start = large_rise_index + 1
        elif set(range(large_rise_index + 1, large_rise_index + 3)) <= short_set:
            ptd_end = large_rise_index
else:
    # same medium-rise loop as comb
```

When `len(medium_rise_indexes) < 3`, comb gracefully falls back to `ASSUMED_DELAY ≈ 0.035 s`. aind-metadata-mapper activates large-rise detection (1.9–2.1 s intervals), which on these rig sessions finds a `ptd_start` that is **one photodiode cycle off** from the correct position.

### 5. Incomplete "one second flip" correction

aind-metadata-mapper has an explicit correction for the 1-second ptd_start error:

```python
delay = np.mean(delay_rise[:-1])
delay_std = np.std(delay_rise[:-1])

if delay_std > DELAY_THRESHOLD or np.isnan(delay):   # DELAY_THRESHOLD = 0.002
    if np.abs((delay - 1) - ASSUMED_DELAY) < DELAY_THRESHOLD:
        logger.info("One second flip required")
        return delay - 1                              # subtract 1 second
return delay                                          # ← returns ~1.036 s if std is low
```

The correction is gated on `delay_std > 0.002 s`. When `ptd_start` is wrong by exactly one cycle, all `delay_rise[i]` values are **consistently** ~1.036 s (low std). The gate never opens and the uncorrected ~1.036 s delay is returned.

### 6. Stale `photodiode_rise_diff` in error-correction loop

In `extract_frame_times_with_delay`, the error-correction loop modifies `photodiode_rise` by deleting events but does **not** recompute `photodiode_rise_diff`:

```python
# aind-metadata-mapper — diff goes stale after first deletion
while any(photodiode_rise_diff[ptd_start:ptd_end] < 1.8):
    photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
    ptd_end -= 1
    # photodiode_rise_diff NOT updated ← bug
```

In comb, the diff is recomputed each iteration:
```python
while any(photodiode_rise_diff[ptd_start:ptd_end] < 1.8):
    photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
    ptd_end -= 1
    photodiode_rise_diff = np.ediff1d(photodiode_rise)  # ← recalculated
```

This stale-diff bug can cause incorrect deletions if there are multiple photodiode errors, but is secondary to the large-rise fallback as the primary 1-second cause.

---

## Observed Evidence

From `260604_test_monitor_delay_comb.ipynb` (comb dynamic delay vs. hardcoded):

| Session | Dynamic delay | Std | Δ vs hardcoded 0.03613 |
|---|---|---|---|
| 767018_2025-02-17 | 0.0351 s | **0** | −1.03 ms |
| 800995_2025-09-17 | 0.0351 s | **0** | −1.03 ms |
| 804670_2025-10-02 | 0.0351 s | **0** | −1.03 ms |
| 755252_2025-01-08 | 0.03658 s | 3×10⁻⁵ | +0.45 ms |
| 823049_2025-12-09 | 0.01779 s | 4×10⁻⁵ | −18 ms |

`std = 0` exactly means `ptd_start` or `ptd_end` was `None` — comb's pattern detection failed and returned `ASSUMED_DELAY = 0.0351`. These are exactly the **3 sessions expected to show the 1-second discrepancy** with aind-metadata-mapper (large-rise fallback activates, wrong `ptd_start`, ~1.036 s delay returned uncorrected).

The 2 sessions with `std > 0` (755252, 823049) have `len(medium_rise_indexes) >= 3`; both codebases use the same algorithm and agree.

---

## Failure Chain (Affected Sessions)

```
len(medium_rise_indexes) < 3
        │
        ├── comb:          no fallback → ptd_start = None → ASSUMED_DELAY 0.0351 s
        │                  final timestamps ≈ correct
        │
        └── aind-mm:       large-rise fallback activates
                           ptd_start off by 1 cycle (≈1 s at 60 Hz)
                           delay_rise[i] ≈ 1.036 s (consistent, low std)
                           delay_std < 0.002 → "one second flip" gate stays closed
                           returns 1.036 s as monitor delay
                           final timestamps = vsync_fall + 1.036 s ≈ 1 second too late
```

---

## Verification Script

Run on one of the 3 affected sessions to confirm:

```python
import h5py, numpy as np
from aind_metadata_mapper.open_ephys.utils.stim_utils import calculate_frame_mean_time, FRAME_KEYS
from aind_metadata_mapper.open_ephys.utils import sync_utils as sync

with h5py.File("path/to/sync.h5", "r") as f:
    photodiode_rise = np.array(sync.get_rising_edges(f, "stim_photodiode"), dtype=float) / 100000.0
    diff = np.ediff1d(photodiode_rise)
    medium = np.where((diff > 0.5) & (diff < 1.5))[0]
    large  = np.where((diff > 1.9) & (diff < 2.1))[0]
    print(f"medium_rise_indexes count: {len(medium)}")   # expect < 3
    print(f"large_rise_indexes count:  {len(large)}")

    ptd_start, ptd_end = calculate_frame_mean_time(f, FRAME_KEYS)
    print(f"ptd_start={ptd_start}, ptd_end={ptd_end}")

    if ptd_start is not None:
        stim_vsync_fall = sync.get_edges(f, "falling", FRAME_KEYS)
        delay0 = photodiode_rise[ptd_start] - stim_vsync_fall[60]
        print(f"delay_rise[0] = {delay0:.4f} s")   # ~1.036 confirms 1-cycle offset
```

---

## Recommended Fix

The safest fix for aind-metadata-mapper is to **not apply the large-rise fallback** and instead return `ASSUMED_DELAY` when medium rises are insufficient — matching comb's behavior:

```python
# In calculate_frame_mean_time, replace the large-rise fallback with:
if len(medium_rise_indexes) < 3:
    return None, None   # caller returns ASSUMED_DELAY, consistent with comb
```

Alternatively, if the large-rise fallback is intentionally retained, the "one second flip" correction should be applied unconditionally when `delay > 0.5 s`, not gated on `delay_std`:

```python
# In extract_frame_times_with_delay:
if delay > 0.5:   # unambiguously wrong; no real monitor delay exceeds ~100 ms
    if np.abs((delay - 1) - ASSUMED_DELAY) < 0.05:
        return delay - 1
```
