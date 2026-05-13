#!/usr/bin/env bash
# Run the full binit0 pipeline for subjects 755252 and 804670.
#
# Usage (from /root/capsule/code/):
#   bash run_binit0.sh
#
# Outputs:
#   /root/capsule/results/runs/0000_first_try/<session>/   -- inputs + images
#   /root/capsule/results/runs/0001_binit0_c23/<session>/  -- combo fits
#   ...
#   /root/capsule/results/runs/index.csv                   -- run index
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR=/root/capsule/results/runs/0000_first_try
OUT_DIR=/root/capsule/results/runs
RECIPES_DIR="$SCRIPT_DIR/baseline_search/recipes/binit0"

# ── Step 1: prepare session inputs ───────────────────────────────────────────
echo "=== [1/2] Preparing sessions ==="
cd "$SCRIPT_DIR"
python prepare_sessions.py --subject_id 755252 --save_dir "$INPUTS_DIR"
python prepare_sessions.py --subject_id 804670 --save_dir "$INPUTS_DIR"

# ── Step 2: collect session keys ─────────────────────────────────────────────
SESSIONS=$(ls "$INPUTS_DIR" | paste -sd ',')
echo "Sessions: $SESSIONS"

# ── Step 3: run all 8 binit0 combos ─────────────────────────────────────────
echo "=== [2/2] Running binit0 combo fits ==="
COMBOS=(c23 c24 c25 c33 c34 c35 c44 c45)
for combo in "${COMBOS[@]}"; do
    echo ""
    echo "--- binit0_$combo ---"
    python -m baseline_search.run \
        --recipe    "$RECIPES_DIR/binit0_${combo}.json" \
        --inputs_dir "$INPUTS_DIR" \
        --out        "$OUT_DIR" \
        --slug       "binit0_${combo}" \
        --sessions   "$SESSIONS" \
        --n_jobs     -1 \
        --verbose    5
done

echo ""
echo "=== Done. Results in $OUT_DIR ==="
