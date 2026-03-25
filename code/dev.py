import inspect
from typing import Callable, Literal
import jax
import jax.numpy as jnp
import numpy as np
from aind_ophys_utils.signal_utils import percentile_filter
from scipy.optimize import minimize, OptimizeResult
from statsmodels.nonparametric._smoothers_lowess import lowess as _sm_lowess
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm

jax.config.update("jax_enable_x64", True)


# -----------------------------
#  Baselines with Jacobians
# -----------------------------
def single_exp(
    params: np.ndarray, t: np.ndarray, return_jac: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [b_inf, b, tau].
    t : np.ndarray
        Timestamps
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction

    Returns
    -------
    y : np.ndarray
        Model prediction
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True)
    """
    b_inf, b, tau = params
    E = np.exp(-t / tau)
    A = 1 + b * E
    y = b_inf * A
    if not return_jac:
        return y

    bE = b_inf * E

    J = np.empty((t.size, 3))
    J[:, 0] = A  # d/d b_inf
    J[:, 1] = bE  # d/d b
    J[:, 2] = b / tau**2 * (bE * t)  # d/d tau

    return y, J


def double_exp(
    params: np.ndarray, t: np.ndarray, return_jac: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with biphasic exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, t_slow, t_fast]
    t : np.ndarray
        Timestamps
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction

    Returns
    -------
    y : np.ndarray
        Model prediction
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True)
    """
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    E_slow = np.exp(-t / t_slow)
    E_fast = np.exp(-t / t_fast)
    A = 1 + b_slow * E_slow + b_fast * E_fast
    y = b_inf * A
    if not return_jac:
        return y

    bE_s = b_inf * E_slow
    bE_f = b_inf * E_fast

    J = np.empty((t.size, 5))
    J[:, 0] = A  # d/d b_inf
    J[:, 1] = bE_s  # d/d b_slow
    J[:, 2] = bE_f  # d/d b_fast
    J[:, 3] = b_slow / t_slow**2 * (bE_s * t)  # d/d t_slow
    J[:, 4] = b_fast / t_fast**2 * (bE_f * t)  # d/d t_fast

    return y, J


def bright(
    params: np.ndarray, t: np.ndarray, return_jac: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Bright baseline with triphasic decay and saturating brightening.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, b_rapid, b_bright,
                           t_slow, t_fast, t_rapid, t_bright]
    t : np.ndarray
        Timestamps
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction

    Returns
    -------
    y : np.ndarray
        Model prediction
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True)
    """
    b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params
    # exponentials
    E_slow = np.exp(-t / t_slow)
    E_fast = np.exp(-t / t_fast)
    E_rapid = np.exp(-t / t_rapid)
    E_bright = np.exp(-t / t_bright)
    # helper terms
    A = 1 + b_slow * E_slow + b_fast * E_fast + b_rapid * E_rapid
    B = 1 - b_bright * E_bright

    y = b_inf * A * B
    if not return_jac:
        return y

    # precompute repeated products
    bE_B_slow = b_inf * E_slow * B
    bE_B_fast = b_inf * E_fast * B
    bE_B_rapid = b_inf * E_rapid * B
    A_Eb = A * E_bright
    b_inf_A_Eb = b_inf * A_Eb
    # initialize Jacobian
    J = np.empty((t.size, 9))
    # amplitudes
    J[:, 0] = A * B  # d/d b_inf
    J[:, 1] = bE_B_slow  # d/d b_slow
    J[:, 2] = bE_B_fast  # d/d b_fast
    J[:, 3] = bE_B_rapid  # d/d b_rapid
    J[:, 4] = -b_inf_A_Eb  # d/d b_bright
    # time constants
    J[:, 5] = b_slow / t_slow**2 * bE_B_slow * t  # d/d t_slow
    J[:, 6] = b_fast / t_fast**2 * bE_B_fast * t  # d/d t_fast
    J[:, 7] = b_rapid / t_rapid**2 * bE_B_rapid * t  # d/d t_rapid
    J[:, 8] = -b_bright / t_bright**2 * b_inf_A_Eb * t  # d/d t_bright

    return y, J


# -----------------------------
#  Baselines using JAX
# -----------------------------
def single_exp_jax(params, t):
    b_inf, b, tau = params
    return b_inf * (1 + b * jnp.exp(-t / tau))


def double_exp_jax(params, t):
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    return b_inf * (1 + b_slow * jnp.exp(-t / t_slow) + b_fast * jnp.exp(-t / t_fast))


def bright_jax(params, t):
    b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params
    A = (
        1
        + b_slow * jnp.exp(-t / t_slow)
        + b_fast * jnp.exp(-t / t_fast)
        + b_rapid * jnp.exp(-t / t_rapid)
    )
    B = 1 - b_bright * jnp.exp(-t / t_bright)
    return b_inf * A * B


# -----------------------------
#  M-estimators
# -----------------------------
class AsymmetricTukeyBiweight(RobustNorm):
    """
    Asymmetric Tukey Biweight norm for robust regression.

    Allows different tuning constants for positive and negative residuals,
    providing more flexibility in handling asymmetric outliers.

    Parameters
    ----------
    c_pos : float, optional
        Tuning constant for positive residuals, default is 4.685
    c_neg : float, optional
        Tuning constant for negative residuals, default is 4.685
    """

    def __init__(self, c_pos=4.685, c_neg=4.685):
        if c_pos <= 0 or c_neg <= 0:
            raise ValueError("Tuning constants must be positive")
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.factor_pos = c_pos**2 / 6
        self.factor_neg = c_neg**2 / 6

    @staticmethod
    def _rho_half(z, c, factor):
        if np.isinf(c):
            return 0.5 * z**2
        t2 = (z / c) ** 2
        return np.where(t2 <= 1, factor * (1 - (1 - t2) ** 3), factor)

    def rho(self, z):
        z = np.asarray(z)
        return np.where(
            z > 0,
            self._rho_half(z, self.c_pos, self.factor_pos),
            self._rho_half(z, self.c_neg, self.factor_neg),
        )

    def psi(self, z):
        z = np.asarray(z)
        c = np.where(z > 0, self.c_pos, self.c_neg)
        return np.where(np.abs(z) <= c, z * (1 - (z / c) ** 2) ** 2, 0.0)
    
    def weights(self, z):
        z = np.asarray(z)
        c = np.where(z > 0, self.c_pos, self.c_neg)
        return np.where(np.abs(z) <= c, (1 - (z / c) ** 2) ** 2, 0.0)
    
    def psi_deriv(self, z):
        z = np.asarray(z)
        c = np.where(z > 0, self.c_pos, self.c_neg)
        t2 = (z / c) ** 2
        return np.where(np.abs(z) <= c, (1 - t2) ** 2 - 4 * t2 * (1 - t2), 0.0)

        
class OneSidedTukeyBiweight(AsymmetricTukeyBiweight):
    """
    A one-sided Tukey Biweight norm that applies quadratic loss to negative
    residuals and Tukey biweight loss to positive residuals.

    This is implemented as a special case of AsymmetricTukeyBiweight
    with c_neg=np.inf, which simplifies to quadratic loss for negative values.
    """

    def __init__(self, c=4.685):
        super().__init__(c_pos=c, c_neg=np.inf)


class AsymmetricTukeyBiweight_jax:
    """Asymmetric Tukey biweight M-estimator (JAX-compatible)."""

    def __init__(self, c_pos=4.685, c_neg=4.685):
        if c_pos <= 0 or c_neg <= 0:
            raise ValueError("Tuning constants must be positive")
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.factor_pos = c_pos**2 / 6
        self.factor_neg = c_neg**2 / 6

    @staticmethod
    def _rho_half(z, c, factor):
        if np.isinf(c):                          # Python-level scalar check, safe for JAX
            return 0.5 * z**2
        t2 = (z / c) ** 2
        return jnp.where(t2 <= 1, factor * (1 - (1 - t2) ** 3), factor)

    def rho(self, z):
        return jnp.where(
            z > 0,
            self._rho_half(z, self.c_pos, self.factor_pos),
            self._rho_half(z, self.c_neg, self.factor_neg),
        )

    def psi(self, z):
        c = jnp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return jnp.where(jnp.abs(z) <= c, z * (1 - (z / c) ** 2) ** 2, 0.0)

    def weights(self, z):
        c = jnp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return jnp.where(jnp.abs(z) <= c, (1 - (z / c) ** 2) ** 2, 0.0)

    def psi_deriv(self, z):
        c = jnp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        t2 = (z / c) ** 2
        return jnp.where(jnp.abs(z) <= c, (1 - t2) ** 2 - 4 * t2 * (1 - t2), 0.0)


class OneSidedTukeyBiweight_jax(AsymmetricTukeyBiweight_jax):
    def __init__(self, c=4.685):
        super().__init__(c_pos=c, c_neg=np.inf)


class TukeyBiweight_jax(AsymmetricTukeyBiweight_jax):
    def __init__(self, c=4.685):
        super().__init__(c_pos=c, c_neg=c)

    def rho(self, z):
        return self._rho_half(z, self.c_pos, self.factor_pos)


# -----------------------------
#  Fitting Functions
# -----------------------------
def nonlinear_fit(
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    M: RobustNorm | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
    dtype=jnp.float64,
) -> tuple[np.ndarray, OptimizeResult]:
    """
    Fit a nonlinear model to a 1-D trace using OLS or robust IRLS.

    Supports two backends:

    - ``"numpy"``: uses analytic Jacobian if ``model`` accepts
      ``return_jac=True``, otherwise falls back to the optimizer's numerical
      gradient estimate.
    - ``"jax"``: differentiates ``model`` automatically via
      ``jax.value_and_grad``; the ``model`` need not support ``return_jac``.

    Parameters
    ----------
    trace : np.ndarray
        Observed signal, shape ``(N,)``.
    t : np.ndarray
        Time vector passed to ``model``, shape ``(N,)``.
    model : callable
        ``model(params, t) -> np.ndarray``.
        For the numpy backend, may optionally support
        ``model(params, t, return_jac=True) -> (np.ndarray, np.ndarray)``,
        returning ``(prediction, J)`` where ``J`` has shape ``(N, n_params)``.
        For the JAX backend, must use ``jnp`` operations throughout.
    init_params : np.ndarray
        Initial parameter vector, shape ``(n_params,)``.
    bounds : sequence of (min, max) pairs or None
        Parameter bounds passed to ``scipy.optimize.minimize``.
    M : RobustNorm or None
        M-estimator norm (e.g. ``TukeyBiweight``).
        ``None`` → ordinary least squares (OLS).
        Otherwise → iteratively re-weighted least squares (IRLS) using
        ``M.rho`` for the loss and ``M.psi`` for the gradient.
        For the JAX backend, must be a JAX-compatible norm (e.g.
        ``TukeyBiweight_jax``).
    optimizer : str
        Solver passed to ``scipy.optimize.minimize``, default ``"L-BFGS-B"``.
    maxiter : int
        Maximum number of IRLS outer iterations. Ignored when ``M=None``.
    tol : float
        IRLS convergence tolerance on the relative parameter change
        ``‖x_new − x‖ / (‖x‖ + ε)``.
    optimizer_options : dict or None
        Options forwarded to ``scipy.optimize.minimize``.
        Defaults to ``{"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}``.
    backend : {"numpy", "jax"}
        Numerical backend. ``"jax"`` enables automatic differentiation and
        JIT compilation; ``"numpy"`` uses model-bundled Jacobians when
        available.
    dtype : jax dtype
        Floating-point precision for the JAX backend, default
        ``jnp.float64``. Requires ``jax_enable_x64=True``.

    Returns
    -------
    fitted : np.ndarray
        Model prediction at the converged parameters, shape ``(N,)``.
    res : OptimizeResult
        Result from the final ``scipy.optimize.minimize`` call, with 
        additional attributes ``res.sigma`` and ``res.weights``
        (robust scale estimate and weights, set only when ``M`` is not ``None``).
    """
    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}

    use_jax = backend == "jax"

    # check once whether model supports return_jac
    has_return_jac = not use_jax and "return_jac" in inspect.signature(model).parameters

    if use_jax:
        t_ = jnp.asarray(t, dtype=dtype)
        y_ = jnp.asarray(trace, dtype=dtype)
        x = jnp.asarray(init_params, dtype=dtype)

        def _make_obj(loss_and_grad_fn):
            cache = {}

            def fun(theta):
                val, grad = loss_and_grad_fn(jnp.asarray(theta, dtype=dtype))
                cache["g"] = np.array(grad)
                return float(val)

            return fun, lambda theta: cache["g"]

        def _ols_loss(theta):
            return jnp.sum((y_ - model(theta, t_)) ** 2)

        ols_val_grad = jax.jit(jax.value_and_grad(_ols_loss))

        if M is not None:

            def _robust_loss(theta, sigma):
                return jnp.sum(M.rho((y_ - model(theta, t_)) / sigma))

            robust_val_grad = jax.jit(jax.value_and_grad(_robust_loss))
    else:
        t_ = np.asarray(t)
        y_ = np.asarray(trace)
        x = np.asarray(init_params, dtype=float).copy()

    # ----------------------------
    # objective factories
    # ----------------------------
    def make_objective_numpy(sigma=None):
        if sigma is None:

            def obj(theta):
                if has_return_jac:
                    y_pred, J = model(theta, t_, return_jac=True)
                    r = y_ - y_pred
                    return np.sum(r**2), -2.0 * J.T @ r
                r = y_ - model(theta, t_)
                return np.sum(r**2)

        else:

            def obj(theta):
                if has_return_jac:
                    y_pred, J = model(theta, t_, return_jac=True)
                    r = y_ - y_pred
                    u = r / sigma
                    return np.sum(M.rho(u)), -(J.T @ M.psi(u)) / sigma
                r = y_ - model(theta, t_)
                u = r / sigma
                return np.sum(M.rho(u))

        return obj

    def make_objective_jax(sigma=None):
        if sigma is None:
            return _make_obj(ols_val_grad)
        else:
            return _make_obj(lambda theta: robust_val_grad(theta, sigma))

    make_objective = make_objective_jax if use_jax else make_objective_numpy
    provides_grad = use_jax or has_return_jac

    # ----------------------------
    # OLS — always runs; also serves as IRLS pre-pass for cold starts
    # ----------------------------
    fun_or_pair = make_objective()
    fun, jac_ = fun_or_pair if use_jax else (fun_or_pair, provides_grad)
    res = minimize(
        fun,
        x,
        bounds=bounds,
        method=optimizer,
        jac=jac_,
        options=optimizer_options,
    )
    x = jnp.asarray(res.x, dtype=dtype) if use_jax else res.x

    # ----------------------------
    # IRLS
    # ----------------------------
    for _ in range(max(1, maxiter) if M is not None else 0):
        resid = y_ - model(x, t_)

        if use_jax:
            sigma = jnp.median(jnp.abs(resid)) * 1.4826
            sigma = jnp.where(sigma == 0, jnp.std(resid), sigma)
        else:
            sigma = scale.mad(resid, center=0)
            if sigma == 0:
                sigma = np.std(resid)

        fun_or_pair = make_objective(sigma)
        fun, jac_ = fun_or_pair if use_jax else (fun_or_pair, provides_grad)
        res = minimize(
            fun,
            x,
            bounds=bounds,
            method=optimizer,
            jac=jac_,
            options=optimizer_options,
        )

        x_new = jnp.asarray(res.x, dtype=dtype) if use_jax else res.x
        norm = jnp.linalg.norm if use_jax else np.linalg.norm
        if norm(x_new - x) / (norm(x) + 1e-12) < tol:
            x = x_new
            break
        x = x_new

    fitted = np.array(model(x, t_))
    if M is not None:
        res.sigma = float(sigma)
        u = (np.array(y_) - fitted) / float(sigma)
        res.weights = np.array(M.weights(u))
    return fitted, res


def _robust_lowess(
    y: np.ndarray,
    t: np.ndarray,
    frac: float = 0.1,
    weights: np.ndarray | None = None,
    M: RobustNorm | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Robust LOWESS smoother with optional outer IRLS loop.

    Parameters
    ----------
    y : np.ndarray
        Raw signal.
    t : np.ndarray
        Timestamps (must be sorted).
    frac : float
        LOWESS bandwidth as fraction of data length.
    weights : np.ndarray or None
        Initial point weights, e.g. res.weights from trend fit.
        Warm-starts the IRLS loop. Defaults to uniform weights.
    M : RobustNorm or None
        If None, single-pass LOWESS. Otherwise, outer IRLS using M.weights.
        Any M-estimator with a .weights(z) method works, including
        AsymmetricTukeyBiweight and OneSidedTukeyBiweight.
    maxiter : int
        Maximum IRLS iterations. Ignored when M is None.
    tol : float
        Convergence tolerance on max weight change between iterations.

    Returns
    -------
    fluctuation : np.ndarray
        Smoothed signal.
    """
    y = y.astype(np.float64)
    t = t.astype(np.float64)
    w_current = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones(len(y), dtype=np.float64)
    )
    delta = 0.01 * (t[-1] - t[0])

    for _ in range(max(1, maxiter) if M is not None else 1):
        fluctuation = _sm_lowess(
            y, t, t,
            resid_weights=w_current,
            frac=frac,
            it=0,
            delta=delta,
        )[0][:, 1]

        if M is None:
            break

        resid = y - fluctuation
        sigma = np.median(np.abs(resid)) * 1.4826
        if sigma == 0:
            sigma = np.std(resid)

        w_new = M.weights(resid / sigma)
        if np.max(np.abs(w_new - w_current)) < tol:
            break
        w_current = w_new

    return fluctuation


def fit_baseline_fluctuations(
    trace: np.ndarray,
    t: np.ndarray,
    trend: np.ndarray | None = None,
    mode: Literal["ratio", "subtract"] = "ratio",
    frac: float = 0.1,
    weights: np.ndarray | None = None,
    method: Literal["lowess", "percentile"] = "lowess",
    M: RobustNorm | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate baseline fluctuations from a fluorescence trace.

    Optionally detrends by a slow trend (e.g. from :func:`nonlinear_fit`),
    estimates fluctuations via LOWESS or a percentile filter, then retrend
    to recover the full baseline.

    Parameters
    ----------
    trace : np.ndarray
        Raw fluorescence signal, shape ``(N,)``.
    t : np.ndarray
        Timestamps, shape ``(N,)``. Must be sorted in ascending order.
    trend : np.ndarray or None
        Slow trend component (e.g. bleaching fit), shape ``(N,)``.
        If ``None``, no detrending is applied and ``baseline == fluctuation``.
    mode : {"ratio", "subtract"}
        How to detrend. ``"ratio"`` divides ``trace`` by ``trend``
        (use when fluorescence is multiplicatively modulated);
        ``"subtract"`` subtracts ``trend`` additively.
    frac : float
        Bandwidth as a fraction of ``N``. Controls the smoothing window
        for both ``"lowess"`` (LOWESS bandwidth) and ``"percentile"``
        (filter half-width).
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``. For ``"lowess"``, warm-starts the
        IRLS loop (e.g. pass ``res.weights`` from :func:`nonlinear_fit`).
        For ``"percentile"``, used to estimate the baseline percentile when
        ``percentile=None``.
    method : {"lowess", "percentile"}
        Smoothing method.
        ``"lowess"`` — robust locally weighted regression via
        :func:`_robust_lowess`.
        ``"percentile"`` — sliding percentile filter via
        :func:`percentile_filter`.
    M : RobustNorm or None
        *LOWESS only.* M-estimator with a ``.weights(z)`` method
        (e.g. :class:`OneSidedTukeyBiweight`). If ``None``, single-pass
        LOWESS without outer IRLS. Ignored when ``method="percentile"``.
    maxiter : int
        *LOWESS only.* Maximum number of IRLS outer iterations.
        Ignored when ``M=None`` or ``method="percentile"``.
    tol : float
        *LOWESS only.* IRLS convergence tolerance on the maximum absolute
        weight change between iterations.
        Ignored when ``M=None`` or ``method="percentile"``.
    percentile : float or None
        *Percentile only.* Percentile to track (0–100). If ``None``,
        estimated from ``weights``: the percentile rank of the
        weighted mean of ``y`` among its own samples, clipped to ``[5, 50]``.
        Ignored when ``method="lowess"``.

    Returns
    -------
    baseline : np.ndarray
        Full baseline in the original signal space, shape ``(N,)``.
        Equal to ``fluctuation`` when ``trend=None``.
    fluctuation : np.ndarray
        Detrended baseline estimate, shape ``(N,)``.
        Ratio relative to ``trend`` when ``mode="ratio"``;
        additive residual when ``mode="subtract"``.
    """
    # detrend — shared
    if trend is None:
        y = trace.astype(np.float64)
    elif mode == "ratio":
        y = (trace / np.where(trend != 0, trend, np.nan)).astype(np.float64)
    else:
        y = (trace - trend).astype(np.float64)

    # dispatch — method-specific, receives y, returns fluctuation
    if method == "lowess":
        fluctuation = _robust_lowess(y, t, frac, weights, M, maxiter, tol)
    elif method == "percentile":
        size = max(1, round(frac * len(trace)))
        if percentile is None:
            # estimate from weights if available
            mu_w = np.average(y, weights=weights) if weights is not None else np.median(y)
            percentile = np.clip(np.mean(y <= mu_w) * 100, 5, 50)
        fluctuation = percentile_filter(y, percentile, size)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # retrend — shared
    if trend is None:
        return fluctuation, fluctuation
    baseline = trend * fluctuation if mode == "ratio" else trend + fluctuation
    return baseline, fluctuation
