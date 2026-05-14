#!/usr/bin/env python3
"""Compute per-ROI noise-criterion winners from saved med_neg_residuals_F0.npy
and write a filtered CSV for inspection in the binit0 QC app.

Usage:
    python generate_winner_list.py [--winners c33,c45] [--out /results/runs/roi_list.csv]

Outputs one row per ROI that matches the requested winner set, sorted by
winner then by how close the winning bar is to the target (best matches first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR   = Path("/scratch/runs")
INPUTS_DIR = RUNS_DIR / "0000_first_try"
TARGET_COEF = 0.674

COMBOS    = [(2,3),(2,4),(2,5),(3,3),(3,4),(3,5),(4,4),(4,5)]
COMBO_KEY = {c: f"c{''.join(map(str,c))}" for c in COMBOS}
COMBO_KEYS = [COMBO_KEY[c] for c in COMBOS]


def find_run_dirs() -> dict[tuple, Path]:
    run_dirs: dict[tuple, Path] = {}
    for run_dir in sorted(RUNS_DIR.glob("0[0-9][0-9][0-9]_binit0_*")):
        rp = run_dir / "recipe.json"
        if not rp.exists():
            continue
        recipe = json.loads(rp.read_text())
        combo = (int(recipe["M"]["c_pos"]), int(recipe["M"]["c_neg"]))
        run_dirs[combo] = run_dir
    return run_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winners", default="c33,c45",
                    help="Comma-separated combo keys to filter (default: c33,c45)")
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/runs/roi_list.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    target_winners = [w.strip() for w in args.winners.split(",")]
    run_dirs = find_run_dirs()

    sessions = sorted(p.name for p in INPUTS_DIR.iterdir() if p.is_dir())
    print(f"Sessions: {len(sessions)}  |  Combos found: {len(run_dirs)}")

    rows = []
    for sess_key in sessions:
        noise_path = INPUTS_DIR / sess_key / "F_noise.npy"
        if not noise_path.exists():
            continue
        noise = np.load(noise_path)
        N = len(noise)

        # load med_neg_residuals_F0trend (IRLS) for each available combo
        med_neg: dict[str, np.ndarray] = {}
        for combo in COMBOS:
            key = COMBO_KEY[combo]
            rd  = run_dirs.get(combo)
            if rd is None:
                continue
            p = rd / sess_key / "med_neg_residuals_F0trend.npy"
            if p.exists():
                med_neg[key] = np.load(p)

        if not med_neg:
            continue

        for roi_idx in range(N):
            target = TARGET_COEF * float(noise[roi_idx])
            dists  = {
                k: abs(float(v[roi_idx]) - target)
                for k, v in med_neg.items()
                if np.isfinite(v[roi_idx])
            }
            if not dists:
                continue

            ranked   = sorted(dists.items(), key=lambda x: x[1])
            winner   = ranked[0][0]
            runnerup = ranked[1][0] if len(ranked) > 1 else ""

            rows.append({
                "session_key":   sess_key,
                "roi_index":     roi_idx,
                "winner":        winner,
                "runner_up":     runnerup,
                "winner_dist":   round(ranked[0][1], 4),
                "runner_up_dist": round(ranked[1][1], 4) if len(ranked) > 1 else None,
            })

    df = pd.DataFrame(rows)
    print("\nWinner distribution (all ROIs):")
    print(df["winner"].value_counts().to_string())

    filtered = (df[df["winner"].isin(target_winners)]
                .sort_values(["winner", "winner_dist"])
                .reset_index(drop=True))
    print(f"\nFiltered ({', '.join(target_winners)}): {len(filtered)} ROIs")
    print(filtered["winner"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
