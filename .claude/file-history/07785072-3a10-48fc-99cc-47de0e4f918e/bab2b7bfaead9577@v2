# Session 02 — F0trend flipping bug investigation
**Date:** 2026-05-06

## Context
User reports that `F0trend` (output of `nonlinear_fit` / `fit_baseline` in `baseline_fitting.py`) sometimes produces **huge negative values** in sessions saved to `/root/capsule/scratch/first_try`. The computation was run in `long_vs_short_baseline_window.ipynb` with `fixed_sigma=F_noise`.

## Data
- Subject 755252, 26 sessions (2024-11-12 to 2025-01-14)
- 24 of 26 sessions have at least one ROI with negative F0trend
- Worst case: -428,063 (session 755252_2025-01-07)
- Typical range: 3–20 ROIs per session with F0trend < 0

## Root Cause

**`fixed_sigma=F_noise` (measurement noise) is the wrong sigma for the IRLS outer loop.**

### What `F_noise` is vs. what IRLS sigma needs
- `F_noise` = shot noise of fluorescence (fast frame-to-frame variability), computed via `noise_std(F, 'mad')`. Typical value: 50–160 AU.
- IRLS sigma = scale of residuals **from the current trend estimate**, which includes shot noise + normal baseline fluctuations. Typical value: 3–3.5× larger than `F_noise`.

### Effect of the mismatch
With `M = AsymmetricTukeyBiweight(c_pos=2, c_neg=3)` and `sigma = F_noise`:
- Threshold for downweighting: `c_pos * sigma = 2 * F_noise` (e.g., 2 × 53 = 106 AU)
- But `F.std()` for a typical active ROI is 200–800 AU
- Result: even baseline-level fluctuations exceed the threshold → **only 45–52% of frames contribute gradient** (vs 82–84% with auto-sigma)

### The flipping cascade
1. OLS pre-pass gives a reasonable starting solution
2. IRLS iteration 1: robust objective with tiny fixed sigma → half the frames get zero gradient
3. Optimizer drifts toward a local minimum: `b_bright` grows large (making F0trend negative)
4. Now **all** frames have residuals >> `c * sigma` → **gradient = 0 everywhere**
5. Optimizer is stuck; final F0trend is deeply negative, e.g., −3000 to −700 when F ∈ [196, 2369]

### Model identifiability issue (contributing factor)
The custom model in the notebook:
```python
b_inf + b_slow * exp(-t/t_slow) + b_fast * exp(-t/t_fast) - b_bright * exp(-t/t_bright)
```
When `t_slow ≈ t_bright` (both bounded in [300, 5 × t_max]), the bleaching term `+b_slow * exp` and brightening term `-b_bright * exp` can **partially cancel**, creating degenerate solutions. A bleaching trace can be mis-fit as: `b_inf=0, b_bright=huge` (model starts hugely negative, "brightens" toward 0), rather than `b_inf=floor, b_slow=range` (starts high, bleaches). The tiny-sigma IRLS is what triggers the flip.

## Verified Examples (session 755252_2024-11-13)

| ROI | F range | F_noise | fixed_sigma | F0trend (original) | F0trend (fix: no fixed_sigma) | long_window baseline |
|-----|---------|---------|-------------|---------------------|-------------------------------|----------------------|
| 130 | [196, 2369] | 53 | 53 | [-3242, -872] | [372, 1307] ✓ | [383, 586] |
| 422 | [840, 6126] | 158 | 158 | [-2718, -704] | [1507, 3180] ✓ | [1414, 2261] |

## Fix

**Remove `fixed_sigma=F_noise` from the nonlinear trend fit.** Let the IRLS auto-compute sigma per iteration from `MAD(residuals)`.

### In `long_vs_short_baseline_window.ipynb` (notebook cell with `_fit_one`):
```python
# BEFORE (broken):
F0, F0trend, res, info = fit_baseline(
    F, timestamps, model, x0, bounds,
    M=M,
    fixed_sigma=noise,   # <-- THIS is the problem
    backend="jax", dtype=jnp.float32)

# AFTER (fixed):
F0, F0trend, res, info = fit_baseline(
    F, timestamps, model, x0, bounds,
    M=M,
    fixed_sigma=None,    # auto-compute sigma per IRLS iteration
    backend="jax", dtype=jnp.float32)
```

`fixed_sigma` is designed to pass `res.sigma` **between two fitting steps** (trend → fluctuation), not to inject measurement noise into the IRLS loop.

### Optional secondary safeguard: bound b_bright
Add an explicit upper bound on `b_bright` to prevent unbounded negative offsets:
```python
bounds=[
    (0, None),    # b_inf
    (0, None),    # b_slow
    (0, None),    # b_fast
    (0, np.percentile(F, 99)),  # b_bright: upper-bounded by signal range
    (300, t_high_bound),  # t_slow
    (1, 300),             # t_fast
    (300, t_high_bound),  # t_bright
]
```
In practice, Fix 1 alone is sufficient — both fixes gave identical results in testing.

## Next Steps
- Re-run `_fit_one` in the notebook without `fixed_sigma`, save new results alongside old ones for comparison
- Investigate whether F0trend (without LOWESS) is actually better than F0 for inhibitory neurons
- Design QC metric to detect bad F0trend fits (see if `min(F0trend) < 0` is a clean flag)
