# Session 04 — Dynamic regularization: critique and analysis
**Date:** 2026-05-14

## Goal
Evaluate the proposed dynamic regularization: when proportion of frames < F0trend < 5%, tighten bounds on b_slow and b_bright to 50% of the unregularized fitted values.

## Context
User is settled on F0trend as the final baseline, using binit0 recipes (b_init=0) under the current `biexp_bright_default` bounds. The (3,3), (4,4), (4,5) combos survive the `frac_below > 5%` filter while (2,3), (2,4), (2,5), (3,4), (3,5) fail for problematic ROIs. The specific example is 755252_2024-12-13, VISp_4, cell_roi_id=32 (roi_idx=257).

## Data analysis

### ROI 257 parameters
- F range: 1179–7560, mean=3171, std=989, ptp=6381
- noise_std (MAD): 160.89
- t_max: 4507s

| Combo | b_inf | b_slow | b_bright | t_slow | t_bright | F0trend(0) | frac_below | ratio |
|-------|-------|--------|----------|--------|----------|------------|------------|-------|
| (2,3) | 0 | 3883 | 5724 | 1329 | 2622 | -1841 | 0.000 | n/a |
| (3,3) | 0 | 4281 | 2742 | 6391 | 745  | +1539 | 0.320 | 2.52 |
| (4,4) | 0 | 4336 | 2749 | 6641 | 829  | +1587 | 0.334 | 2.64 |
| (4,5) | 0 | 6087 | 4422 | 4610 | 1479 | +1665 | 0.254 | 2.05 |

### Population-level blown stats (frac_below < 5%)
- (2,3): 453/20501 (2.2%) | blown b_bright p95 = 208,668
- (2,4): 494/20501 (2.4%) | blown b_bright p95 = 375,612
- (2,5): 391/20501 (1.9%) | blown b_bright p95 = 792,599
- (3,4): 31/20501 (0.2%)
- (3,5): 97/20501 (0.5%)
- (3,3), (4,4): 0 blown
- (4,5): 2/20501 (0.01%)

## Critique of the proposed approach

### What works
1. **Trigger (frac_below < 5%)** — correctly identifies blown fits. ROI 257 blown combos all have frac_below=0.
2. **Focusing on b_bright** — b_bright IS the driver of negative F0trend.

### Fatal flaw: b_slow should NOT be regularized
For ROI 257:
- Blown b_slow: 3883–4654
- Correct solution (3,3): b_slow = **4281**
- 50% cap on (2,3) b_slow: 1941 → **blocks the correct solution (4281)**

Across all blown fits: b_slow median (blown) ≈ 1816–4412 vs b_slow median (good) ≈ 1534. The blown fits have b_slow in the same range as good fits. b_slow is not the problem.

**Conclusion: capping b_slow to 50% of blown value would actively prevent the optimizer from finding the correct solution.**

### Unreliable: 50% of blown b_bright
- Blown b_bright has p95 of 208k–792k across combos. 50% of that is still 104k–396k → far too loose.
- For ROI 257 it "works by chance": 50% × 5724 = 2862, correct is 2742. But this is coincidental.
- The blown values span orders of magnitude depending on severity of failure.

**Better alternative**: Use `b_bright_ub = F.ptp()` or `b_bright_ub = F.mean()` as a data-driven physical upper bound for the second pass.

### b_inf, b_fast: no regularization needed
- b_inf is 0 in both good and bad fits for this ROI; rarely a problem.
- b_fast is 0 across almost all ROIs. Not contributing.

### t_bright deserves attention (overlooked)
Blown fits have t_bright/t_slow median = 1.4–1.9 (negative term decays slower than positive). For ROI 257 (2,3): t_bright/t_slow = 2622/1329 = **1.97** vs good fit (3,3): t_bright/t_slow = 745/6391 = **0.12**.

When t_bright > t_slow, the negative term outlasts the positive: even if F0trend(0) starts positive, it will decrease over time and may go negative mid-session.

**Proposed additional constraint for second pass**: `t_bright_ub = min(t_max, fitted_t_slow)` (t_bright can't be longer than t_slow).

**Caution**: The t_max upper bound alone (`t_bright_ub = t_max`) won't help ROI 257 (blown t_bright=2622 < t_max=4507). Need to tie it to the t_slow from the failed fit.

### Root cause: fixed_sigma = noise_std (Session 02 bug is still present)
The binit0 recipes use `sigma: {kind: noise_std}`. With c_pos=2 and σ=noise_std=161, the IRLS threshold is 2×161=322. If F0trend is very wrong, residuals >> 322 → all residuals exceed threshold → near-zero gradient → optimizer stuck. This is exactly the bug documented in Session 02. (3,3) survives because c_pos=3 → threshold=483, somewhat better; (4,x) with c_pos=4 → threshold=644, even better.

The proposed dynamic regularization is a patch on top of a root-cause issue. It may work for the specific ROI but won't generalize cleanly.

## Proposed improved approach
1. **Don't regularize b_slow** (ever, in this scheme)
2. **For b_bright second pass**: use `b_bright_ub = F.ptp()` (data-driven, not based on blown values)
3. **For t_bright second pass**: add `t_bright_ub = fitted_t_slow` from the failed fit (ensures brightening decays before bleaching ends)
4. **Longer term**: switch sigma to `mad_residual` for the binit0 trend fits (fix the root cause)

## Notebook
`05_dynamic_regularization_critique.ipynb`
