---
name: project-dynamic-regularization
description: Analysis of dynamic regularization for blown F0trend fits (frac_below < 5%) — key findings from Session 04
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d3a6d1e-25cd-455f-a6f2-ff355d74710c
---

User wants to add dynamic regularization: if `frac_below < 5%` (F always above F0trend), refit with tightened bounds based on unregularized fitted values.

**Why:** F0trend too low (making no frames < F0trend) due to b_bright blowing up beyond b_slow in biexp_bright_v1 model. Root cause: fixed_sigma = noise_std in IRLS with small c_pos creates zero-gradient trap (same as Session 02 bug).

**Session 04 findings (ROI 257 = 755252_2024-12-13 VISp_4 cell_roi_id 32, roi_idx=257):**
- Blown combos (2,3), (2,4), (2,5), (3,4), (3,5): b_bright > b_slow + b_inf → F0trend(0) < 0; t_bright > t_slow → trend worsens over time
- Winning combos (3,3), (4,4), (4,5): b_bright/b_slow ≈ 0.63–0.73; t_bright/t_slow ≈ 0.12–0.32
- Population: 993/20501 ROIs (4.8%) have at least one blown combo

**Critical issues with proposed b_slow cap (50% of blown):**
- Blown b_slow for (2,3): 3883 → cap = 1941. Correct answer needs b_slow = 4281. BLOCKS correct solution.
- Confirmed across all blown ROIs in (2,3): most cases would be blocked.

**b_bright cap (50% of blown):**
- Works for ROI 257 by coincidence (blown b_bright=5724 → cap=2862, correct=2742)
- Blown b_bright p95 = 208k–792k; 50% still far too loose for extreme cases

**Better alternatives identified:**
1. Don't regularize b_slow
2. b_bright_ub = F.ptp() or F.mean() (data-driven, not from blown fit)
3. Add t_bright_ub = t_slow_from_blown_fit (enforces t_bright < t_slow; key discriminator)
4. Root cause fix: switch sigma.kind to mad_residual in binit0 recipes

**How to apply:** See notebook `05_dynamic_regularization_critique.ipynb` and session log `04_dynamic_regularization_critique.md`.
