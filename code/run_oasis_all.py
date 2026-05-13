#!/usr/bin/env python3
"""Deconvolve all dFF arrays with OASIS and save events + denoised to h5.

Output files (in the same folder as each dff .npy):
  0000_first_try/<session>/events_short.h5
  0000_first_try/<session>/events_long.h5
  000X_binit0_cXX/<session>/events_F0trend.h5
  000X_binit0_cXX/<session>/events_F0.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_oasis import run_oasis  # noqa: E402

RUNS_DIR   = Path("/results/runs")
INPUTS_DIR = RUNS_DIR / "0000_first_try"

# (dff filename, output h5 filename)
INPUT_SPECS = [
    ("dff_short_window_all_array.npy", "events_short.h5"),
    ("dff_long_window_all_array.npy",  "events_long.h5"),
]
COMBO_SPECS = [
    ("dff_F0trend_all.npy", "events_F0trend.h5"),
    ("dff_F0_all.npy",      "events_F0.h5"),
]

_fr_cache: dict[str, float] = {}


def _frame_rate(sess_key: str) -> float:
    if sess_key not in _fr_cache:
        ts = np.load(INPUTS_DIR / sess_key / "timestamps.npy")
        _fr_cache[sess_key] = float(1.0 / np.median(np.diff(ts)))
    return _fr_cache[sess_key]


def collect_jobs() -> list[tuple[Path, Path, str]]:
    jobs: list[tuple[Path, Path, str]] = []

    # short/long from inputs folder
    for sess_dir in sorted(INPUTS_DIR.iterdir()):
        if not sess_dir.is_dir():
            continue
        for dff_name, h5_name in INPUT_SPECS:
            dff = sess_dir / dff_name
            if dff.exists():
                jobs.append((dff, sess_dir / h5_name, sess_dir.name))

    # combo F0trend/F0 from binit0 run folders
    for run_dir in sorted(RUNS_DIR.glob("0[0-9][0-9][0-9]_binit0_*")):
        for sess_dir in sorted(run_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            for dff_name, h5_name in COMBO_SPECS:
                dff = sess_dir / dff_name
                if dff.exists():
                    jobs.append((dff, sess_dir / h5_name, sess_dir.name))

    return jobs


def main() -> None:
    jobs = collect_jobs()
    todo    = [(d, o, s) for d, o, s in jobs if not o.exists()]
    skipped = len(jobs) - len(todo)

    print(f"Jobs total : {len(jobs)}")
    print(f"To run     : {len(todo)}")
    print(f"Already done (skipped): {skipped}")
    print()

    errors = 0
    bar = tqdm(todo, unit="file", dynamic_ncols=True)
    for dff_file, out_file, sess_key in bar:
        label = f"{dff_file.parent.parent.name}/{sess_key}/{dff_file.name}"
        bar.set_description(label[-60:])
        try:
            traces = np.load(dff_file).astype(np.float32)
            fr     = _frame_rate(sess_key)
            run_oasis(traces, fr, str(out_file))
        except Exception as e:
            tqdm.write(f"  ERROR {label}: {e}")
            errors += 1

    print()
    print(f"Finished. Errors: {errors}")


if __name__ == "__main__":
    main()
