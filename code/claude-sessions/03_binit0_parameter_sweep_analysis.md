# Session 03 — binit0 parameter sweep & noise-criterion QC app

## Work done

### 1. All 8 binit0 (c_pos, c_neg) runs across both mice
- `run_all_binit0.py` orchestrates 8 combos × 42 sessions (26×755252 + 16×804670)
- Runs 0017–0024; idempotent (skips done sessions, resolves existing slugs)
- Fixed duplicate-ID bug (0025–0029 deleted)

### 2. Analysis notebook: `04_binit0_optimal_cposcneg_analysis.ipynb`
- For each ROI: find combo where |median(neg residuals)| is closest to 0.674 × noise_std(F,'mad')
- Results across 20,501 ROIs / 42 sessions:
  - (2,3) wins 37.3%, (3,5) 21.2%, (2,5) 17.3%, (2,4) 15.5%, (3,4) 7.5%, (4,5) 1.0%
  - (4,4) never wins
- Spearman correlations:
  - best c_neg ← skewness ρ=−0.20: high-skewness neurons prefer c_neg=3 (strict)
  - best c_pos ← SNR ρ=−0.36, sustained ρ=−0.42: low-SNR/quiet neurons prefer larger c_pos
- IRLS convergence: 94% converge at iteration 1, 6% at iteration 2; none hit maxiter=5

### 3. binit0 noise-criterion QC app (`/root/capsule/code/binit0_qc_app/`)
New standalone PyQt5+pyqtgraph app to visually verify the noise-criterion combo selection.

**Files:**
- `data.py` — session loading, combo-run mapping, `compute_noise_bar()`
- `curation.py` — curation CSV with noise_winner, visual_best, verdict, notes
- `app.py` — TracePanel (9 traces), NoiseCriterionPlot (bar chart), ImagePanel, MetricHistograms, CurationPanel, MainWindow
- `main.py` — CLI entry point

**Launch:**
```bash
cd /root/capsule/code
python -m binit0_qc_app.main
```

**Features:**
- 9 traces: short (blue), long (green), F0trend for (2,3)–(4,5) (7 combo colors)
- Noise bar plot: |median(neg residuals)| per combo vs 0.674·σ_noise target line
  - Winner bar: full opacity + gold border + ★ ratio label
  - Non-winners: semi-transparent
- Curation bottom bar: noise winner (auto), visual best (dropdown), verdict (agree/disagree/unsure), notes
- Keys 1–9 toggle traces, J/K navigate ROIs, S save, Space save+next, Z/A/M image controls

## Key paths
- Runs: `/results/runs/0017–0024_*_lowess_binit0`
- Input sessions: `/results/runs/0000_first_try/`, `/results/runs/804670_inputs/`
- Curation output: `/results/binit0_qc_curation.csv`
