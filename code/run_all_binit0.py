"""Run all 8 binit0 (c_pos, c_neg) combinations across all available sessions.

Existing run folders (0017-0019) are extended to the full session list.
New run folders (0020-0024) are created for the missing combinations.

Usage:
    python run_all_binit0.py [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "dff_baseline_search_qc_app"))
from baseline_search.recipe import Recipe

RUNS_DIR    = Path("/results/runs")
INPUTS_755  = Path("/results/runs/0000_first_try")
INPUTS_804  = Path("/results/runs/804670_inputs")
RECIPES_DIR = Path(__file__).resolve().parent / "dff_baseline_search_qc_app/baseline_search/recipes/binit0"
T_START     = 5.0
N_JOBS      = 16

# (c_pos, c_neg) → existing run_id or None (to be allocated)
COMBOS = {
    (2, 3): 17,
    (2, 4): 18,
    (2, 5): None,
    (3, 3): 19,
    (3, 4): None,
    (3, 5): None,
    (4, 4): None,
    (4, 5): None,
}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def rebuild_index():
    rows = []
    for rd in sorted(RUNS_DIR.glob("0[0-9][0-9][0-9]_*")):
        if not rd.is_dir():
            continue
        mp = rd / "metadata.json"
        rp = rd / "recipe.json"
        if not mp.exists() or not rp.exists():
            continue
        m   = json.loads(mp.read_text())
        rec = Recipe.from_json(rp.read_text())
        fid = int(rd.name[:4])
        fl  = _flatten(rec.model_dump())
        rows.append({
            "run_id":      f"{fid:04d}",
            "slug":        rd.name[5:],
            "run_dir":     str(rd),
            "created_at":  m["created_at"],
            "n_sessions":  len(m["sessions"]),
            "description": m.get("description", ""),
            **{f"recipe_{k}": v for k, v in fl.items()},
        })
    df = pd.DataFrame(rows)
    df.to_csv(RUNS_DIR / "index.csv", index=False)
    return df


def list_sessions(inputs_dir: Path) -> list[str]:
    return sorted(p.name for p in inputs_dir.iterdir()
                  if p.is_dir() and (p / "F_all_array.npy").exists())


def run_refit(run_dir: Path, sessions: list[str], inputs_dir: Path, dry: bool) -> None:
    if not sessions:
        return
    cmd = [
        sys.executable, "-m", "baseline_search.refit_from5s",
        "--session",    ",".join(sessions),
        "--runs_dir",   str(RUNS_DIR),
        "--inputs_dir", str(inputs_dir),
        "--t_start",    str(T_START),
        "--n_jobs",     str(N_JOBS),
        "--verbose",    "0",
        "--run_dirs",   run_dir.name,
    ]
    print(f"  $ {' '.join(cmd[-8:])}")
    if not dry:
        subprocess.run(cmd, check=True,
                       cwd=str(Path(__file__).resolve().parent / "dff_baseline_search_qc_app"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Determine sessions available for each mouse
    sess_755 = list_sessions(INPUTS_755)
    sess_804 = list_sessions(INPUTS_804) if INPUTS_804.exists() else []
    print(f"755252 sessions: {len(sess_755)}")
    print(f"804670 sessions: {len(sess_804)}")

    # Allocate new run IDs for missing combos
    # Resolve run IDs: prefer existing folder with matching slug over allocating new ones
    slug_to_id = {
        p.name[5:]: int(p.name[:4])
        for p in RUNS_DIR.glob("0[0-9][0-9][0-9]_*") if p.is_dir()
    }
    existing_ids = sorted(slug_to_id.values())
    next_id = max(existing_ids, default=0) + 1
    for combo, rid in sorted(COMBOS.items()):
        if rid is None:
            slug = f"cpos{combo[0]}_cneg{combo[1]}_lowess_binit0"
            if slug in slug_to_id:
                COMBOS[combo] = slug_to_id[slug]
            else:
                COMBOS[combo] = next_id
                next_id += 1

    print("\nPlan:")
    for (cp, cn), rid in sorted(COMBOS.items(), key=lambda x: x[1]):
        slug = f"cpos{cp}_cneg{cn}_lowess_binit0"
        print(f"  {rid:04d}_{slug}")

    if args.dry_run:
        print("\n[dry run — no fitting]")
        return

    for (cp, cn), run_id in sorted(COMBOS.items(), key=lambda x: x[1]):
        slug        = f"cpos{cp}_cneg{cn}_lowess_binit0"
        recipe_path = RECIPES_DIR / f"{slug}.json"
        run_dir     = RUNS_DIR / f"{run_id:04d}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

        recipe = Recipe.from_json(recipe_path.read_text())
        (run_dir / "recipe.json").write_text(recipe.to_json())

        # Which sessions still need to be fitted?
        already_done = {p.name for p in run_dir.iterdir()
                        if p.is_dir() and (p / "F0trend_all.npy").exists()}

        to_do_755 = [s for s in sess_755 if s not in already_done]
        to_do_804 = [s for s in sess_804 if s not in already_done]

        print(f"\n=== {run_dir.name} ===")
        print(f"  755252: {len(to_do_755)} sessions to fit")
        print(f"  804670: {len(to_do_804)} sessions to fit")

        if to_do_755:
            run_refit(run_dir, to_do_755, INPUTS_755, args.dry_run)
        if to_do_804:
            run_refit(run_dir, to_do_804, INPUTS_804, args.dry_run)

        # Update metadata
        all_done = {p.name for p in run_dir.iterdir()
                    if p.is_dir() and (p / "F0trend_all.npy").exists()}
        sess_summaries = [{"session": s} for s in sorted(all_done)]
        metadata = {
            "run_id":      run_id,
            "slug":        slug,
            "description": recipe.description,
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "host":        socket.gethostname(),
            "user":        os.environ.get("USER", "unknown"),
            "git_rev":     None,
            "inputs_dir":  str(INPUTS_755),
            "recipe_path": str(recipe_path),
            "sessions":    sess_summaries,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    df = rebuild_index()
    print(f"\nindex.csv rebuilt: {len(df)} rows")
    print(df[["run_id", "slug", "n_sessions"]].to_string(index=False))


if __name__ == "__main__":
    main()
