# Fluorescence Baseline Fitting

A composable pipeline for fitting fluorescence baselines from calcium imaging
or similar 1-D time-series data. Built on NumPy/JAX, scipy, and statsmodels.

The pipeline decomposes the baseline into two components:

1. **Slow trend** — a parametric bleaching model fit via nonlinear least squares
   (OLS or robust IRLS)
2. **Local fluctuations** — a smooth baseline estimated from the detrended
   signal via LOWESS or a sliding percentile filter

## Dependencies

- `numpy`
- `jax` (with `jax_enable_x64=True`)
- `scipy`
- `statsmodels`
- `aind_ophys_utils`

## Quick Start

```python
from baseline_fitting import fit_baseline, single_exp, AsymmetricTukeyBiweight
from aind_ophys_utils.signal_utils import noise_std
import numpy as np

# Estimate noise scale (robust to transients and trend)
fixed_sigma = noise_std(trace)

F0, F0trend, res, info = fit_baseline(
    trace, t,
    model=single_exp,
    x0=[trace[-1], trace[0] - trace[-1], 100.0],
    M=AsymmetricTukeyBiweight(c_pos=3, c_neg=4),
    mode="ratio",
    frac=0.1,
    fixed_sigma=fixed_sigma,
)
```

## API Overview

### Trend models

| Function | Description |
|---|---|
| `single_exp` | Asymptotic + single exponential decay: `b_inf + b·exp(-t/τ)` |
| `double_exp` | Biphasic decay: slow and fast exponential components |
| `bright` | Triphasic decay with a saturating brightening component |

Each model accepts a `return_jac=True` flag (NumPy backend) or an `xp`
argument for JAX tracing. Custom models can be passed to `nonlinear_fit` /
`fit_baseline` as long as they follow the `model(params, t) -> array`
signature.

### M-estimator norms

| Class | Description |
|---|---|
| `TukeyBiweight` | Symmetric Tukey biweight; downweights outliers on both sides |
| `OneSidedTukeyBiweight` | Quadratic for negative residuals, Tukey for positive; natural choice for fluorescence (transients are upward) |
| `AsymmetricTukeyBiweight` | General asymmetric biweight with independent `c_pos` / `c_neg` tuning constants |

All norms implement the statsmodels `RobustNorm` interface and support a
`.with_xp(jnp)` method for seamless JAX compatibility.

### Fitting functions

| Function | Description |
|---|---|
| `nonlinear_fit` | Fit any parametric model via OLS or robust IRLS. NumPy backend uses analytic Jacobians; JAX backend uses autodiff + JIT. Returns fitted values and an `OptimizeResult` augmented with `res.sigma` and `res.weights`. |
| `robust_lowess` | Wraps statsmodels LOWESS in an outer IRLS loop using any M-estimator norm. Supports warm-start weights and a fixed scale. |
| `fit_baseline_fluctuations` | Detrends a trace by a slow trend (ratio or subtract), estimates local fluctuations via LOWESS or percentile filter, then retrends to recover the full baseline. |
| `fit_baseline` | Top-level convenience wrapper: runs `nonlinear_fit` for the slow trend, then `fit_baseline_fluctuations` for local fluctuations. Returns `(F0, F0trend, res, info)`. |

## Design Notes

- **Detrending modes**: `mode="ratio"` divides `trace` by the trend (use when
  fluorescence is multiplicatively modulated, e.g. bleaching scales the whole
  signal); `mode="subtract"` removes the trend additively.
- **Unit consistency**: `fixed_sigma` is always supplied in absolute
  fluorescence units. When `mode="ratio"`, it is rescaled internally before
  being passed to the smoother — the caller never needs to handle this.
- **Scale estimation**: `fixed_sigma` is best estimated using
  `aind_ophys_utils.signal_utils.noise_std`, which supports several methods:
  `'fft'` and `'welch'` estimate the noise from the high-frequency tail of the
  power spectral density; `'mad'` uses a robust MAD on the residual after
  rolling-median subtraction. All three are more reliable than the
  per-iteration MAD inside the IRLS loop, which can be biased by large
  transients or a slowly varying trend.
- **Backend choice**: `backend="numpy"` uses analytic Jacobians when the model
  provides them and is the fastest choice for simple models (`single_exp`,
  `double_exp`). `backend="jax"` becomes faster for complex models with many
  parameters (e.g. `bright` with 9 parameters), where the cost of numerical
  gradient estimation dominates — including in parallel batch processing.
