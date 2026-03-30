import inspect
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from aind_ophys_utils.signal_utils import percentile_filter
from scipy.optimize import OptimizeResult, minimize
from statsmodels.nonparametric._smoothers_lowess import lowess as _sm_lowess
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm

jax.config.update("jax_enable_x64", True)


# -----------------------------
#  Baselines with Jacobians or JAX tracing for autodiff
# -----------------------------
def single_exp(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [b_inf, b, tau].
        ``b_inf`` — asymptotic (t→∞) baseline level.
        ``b``   — bleaching amplitude (same units as signal).
        ``tau`` — bleaching time constant.
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b, tau = params
    E = xp.exp(-t / tau)
    y = b_inf + b * E
    if not return_jac:
        return y

    J = np.empty((t.size, 3))
    J[:, 0] = 1.0  # d/d b_inf
    J[:, 1] = E  # d/d b
    J[:, 2] = b / tau**2 * (E * t)  # d/d tau
    return y, J


def double_exp(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with biphasic exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, t_slow, t_fast].
        ``b_inf``  — asymptotic (t→∞) baseline level.
        ``b_slow``, ``b_fast`` — bleaching amplitudes (same units as signal).
        ``t_slow``, ``t_fast`` — bleaching time constants.
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    E_slow = xp.exp(-t / t_slow)
    E_fast = xp.exp(-t / t_fast)
    y = b_inf + b_slow * E_slow + b_fast * E_fast
    if not return_jac:
        return y

    J = np.empty((t.size, 5))
    J[:, 0] = 1.0  # d/d b_inf
    J[:, 1] = E_slow  # d/d b_slow
    J[:, 2] = E_fast  # d/d b_fast
    J[:, 3] = b_slow / t_slow**2 * (E_slow * t)  # d/d t_slow
    J[:, 4] = b_fast / t_fast**2 * (E_fast * t)  # d/d t_fast
    return y, J


def bright(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Bright baseline with triphasic decay and saturating brightening.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, b_rapid, b_bright,
                           t_slow, t_fast, t_rapid, t_bright].
        ``b_inf``            — asymptotic (t→∞) baseline level.
        ``b_slow``, ``b_fast``, ``b_rapid`` — bleaching amplitudes (> 0).
        ``b_bright``         — brightening amplitude (> 0); signal starts
                               suppressed by ``b_bright`` and recovers.
        ``t_slow``, ``t_fast``, ``t_rapid``, ``t_bright`` — time constants.
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params
    E_slow = xp.exp(-t / t_slow)
    E_fast = xp.exp(-t / t_fast)
    E_rapid = xp.exp(-t / t_rapid)
    E_bright = xp.exp(-t / t_bright)
    y = (
        b_inf
        + b_slow * E_slow
        + b_fast * E_fast
        + b_rapid * E_rapid
        - b_bright * E_bright
    )
    if not return_jac:
        return y

    J = np.empty((t.size, 9))
    J[:, 0] = 1.0  # d/d b_inf
    J[:, 1] = E_slow  # d/d b_slow
    J[:, 2] = E_fast  # d/d b_fast
    J[:, 3] = E_rapid  # d/d b_rapid
    J[:, 4] = -E_bright  # d/d b_bright
    J[:, 5] = b_slow / t_slow**2 * (E_slow * t)  # d/d t_slow
    J[:, 6] = b_fast / t_fast**2 * (E_fast * t)  # d/d t_fast
    J[:, 7] = b_rapid / t_rapid**2 * (E_rapid * t)  # d/d t_rapid
    J[:, 8] = -b_bright / t_bright**2 * (E_bright * t)  # d/d t_bright
    return y, J


# Old parametrization using multiplicative b_inf. Might be less well conditioned for optimization,
# there's a multiplicative ridge in parameter space where you can trade off b_inf against b


def single_exp_old(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [b_inf, b, tau].
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b, tau = params
    E = xp.exp(-t / tau)
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


def double_exp_old(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Baseline with biphasic exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, t_slow, t_fast].
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    E_slow = xp.exp(-t / t_slow)
    E_fast = xp.exp(-t / t_fast)
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


def bright_old(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Bright baseline with triphasic decay and saturating brightening.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector: [b_inf, b_slow, b_fast, b_rapid, b_bright,
                           t_slow, t_fast, t_rapid, t_bright].
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    return_jac : bool, optional
        If True, return Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to parameters (returned if return_jac=True).
    """
    b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params
    E_slow = xp.exp(-t / t_slow)
    E_fast = xp.exp(-t / t_fast)
    E_rapid = xp.exp(-t / t_rapid)
    E_bright = xp.exp(-t / t_bright)
    A = 1 + b_slow * E_slow + b_fast * E_fast + b_rapid * E_rapid
    B = 1 - b_bright * E_bright
    y = b_inf * A * B
    if not return_jac:
        return y

    bE_B_slow = b_inf * E_slow * B
    bE_B_fast = b_inf * E_fast * B
    bE_B_rapid = b_inf * E_rapid * B
    b_inf_A_Eb = b_inf * A * E_bright
    J = np.empty((t.size, 9))
    J[:, 0] = A * B  # d/d b_inf
    J[:, 1] = bE_B_slow  # d/d b_slow
    J[:, 2] = bE_B_fast  # d/d b_fast
    J[:, 3] = bE_B_rapid  # d/d b_rapid
    J[:, 4] = -b_inf_A_Eb  # d/d b_bright
    J[:, 5] = b_slow / t_slow**2 * bE_B_slow * t  # d/d t_slow
    J[:, 6] = b_fast / t_fast**2 * bE_B_fast * t  # d/d t_fast
    J[:, 7] = b_rapid / t_rapid**2 * bE_B_rapid * t  # d/d t_rapid
    J[:, 8] = -b_bright / t_bright**2 * b_inf_A_Eb * t  # d/d t_bright
    return y, J


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
        Tuning constant for positive residuals, default is 4.685.
    c_neg : float, optional
        Tuning constant for negative residuals, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c_pos: float = 4.685, c_neg: float = 4.685, xp=np):
        if c_pos <= 0 or c_neg <= 0:
            raise ValueError("Tuning constants must be positive")
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.factor_pos = c_pos**2 / 6
        self.factor_neg = c_neg**2 / 6
        self.xp = xp

    def _rho_half(self, z, c: float, factor: float):
        if np.isinf(c):  # Python-level scalar check, safe inside JAX traces
            return 0.5 * z**2
        t2 = (z / c) ** 2
        return self.xp.where(t2 <= 1, factor * (1 - (1 - t2) ** 3), factor)

    def rho(self, z):
        z = self.xp.asarray(z)
        return self.xp.where(
            z > 0,
            self._rho_half(z, self.c_pos, self.factor_pos),
            self._rho_half(z, self.c_neg, self.factor_neg),
        )

    def psi(self, z):
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return self.xp.where(self.xp.abs(z) <= c, z * (1 - (z / c) ** 2) ** 2, 0.0)

    def weights(self, z):
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return self.xp.where(self.xp.abs(z) <= c, (1 - (z / c) ** 2) ** 2, 0.0)

    def psi_deriv(self, z):
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        t2 = (z / c) ** 2
        return self.xp.where(
            self.xp.abs(z) <= c, (1 - t2) ** 2 - 4 * t2 * (1 - t2), 0.0
        )

    def with_xp(self, xp):
        return AsymmetricTukeyBiweight(c_pos=self.c_pos, c_neg=self.c_neg, xp=xp)


class OneSidedTukeyBiweight(AsymmetricTukeyBiweight):
    """
    One-sided Tukey Biweight norm: quadratic loss for negative residuals,
    Tukey biweight loss for positive residuals.

    Implemented as :class:`AsymmetricTukeyBiweight` with ``c_neg=np.inf``.

    Parameters
    ----------
    c : float, optional
        Tuning constant for positive residuals, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c: float = 4.685, xp=np):
        super().__init__(c_pos=c, c_neg=np.inf, xp=xp)


class TukeyBiweight(AsymmetricTukeyBiweight):
    """
    Symmetric Tukey Biweight norm.

    Parameters
    ----------
    c : float, optional
        Tuning constant, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c: float = 4.685, xp=np):
        super().__init__(c_pos=c, c_neg=c, xp=xp)

    def rho(self, z):
        z = self.xp.asarray(z)
        return self._rho_half(z, self.c_pos, self.factor_pos)


# -----------------------------
#  Fitting Functions
# -----------------------------
def nonlinear_fit(
    # --- data / model ---
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    # --- optimizer ---
    optimizer: str = "L-BFGS-B",
    optimizer_options: dict | None = None,
    # --- backend ---
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
        ``model(params, t) -> np.ndarray | jax.Array``.
        For the numpy backend, may optionally support
        ``model(params, t, return_jac=True) -> (np.ndarray, np.ndarray)``,
        returning ``(prediction, J)`` where ``J`` has shape ``(N, n_params)``.
        For the JAX backend, it's wrapped automatically if the model has an
        xp parameter.
    x0 : np.ndarray
        Initial parameter vector, shape ``(n_params,)``.
    bounds : sequence of (min, max) pairs or None
        Parameter bounds passed to ``scipy.optimize.minimize``.
    M : RobustNorm or None
        M-estimator norm (e.g. ``TukeyBiweight``).
        ``None`` → ordinary least squares (OLS).
        Otherwise → iteratively re-weighted least squares (IRLS) using
        ``M.rho`` for the loss and ``M.psi`` for the gradient.
        For the JAX backend, it's converted automatically via ``with_xp(jnp)``.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``, multiplied into the OLS loss only;
        does not affect the robust IRLS objective. When ``M=None``, this
        performs weighted OLS. When ``M`` is set, it warm-starts the OLS
        pre-pass from a prior fit's ``res.weights``. ``None`` → uniform weights.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
    maxiter : int
        Maximum number of IRLS outer iterations. Ignored when ``M=None``.
    tol : float
        IRLS convergence tolerance on the relative parameter change
        ``‖x_new − x‖ / (‖x‖ + ε)``.
    optimizer : str
        Solver passed to ``scipy.optimize.minimize``, default ``"L-BFGS-B"``.
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
        additional attributes ``res.sigma`` (float, robust scale estimate)
        and ``res.weights`` (np.ndarray, shape ``(N,)``, M-estimator weights),
        set only when ``M`` is not ``None``.
    """
    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}

    use_jax = backend == "jax"
    if use_jax:
        if M is not None:
            M = M.with_xp(jnp)
        if "xp" in inspect.signature(model).parameters:
            model = partial(model, xp=jnp)

    # check once whether model supports return_jac
    has_return_jac = not use_jax and "return_jac" in inspect.signature(model).parameters

    if use_jax:
        t_ = jnp.asarray(t, dtype=dtype)
        y_ = jnp.asarray(trace, dtype=dtype)
        x = jnp.asarray(x0, dtype=dtype)
        w_ = jnp.asarray(weights, dtype=dtype) if weights is not None else None

        def _make_obj(loss_and_grad_fn):
            cache = {}

            def fun(theta):
                val, grad = loss_and_grad_fn(jnp.asarray(theta, dtype=dtype))
                cache["g"] = np.array(grad)
                return float(val)

            return fun, lambda theta: cache["g"]

        def _ols_loss(theta):
            r = y_ - model(theta, t_)
            return jnp.sum(w_ * r**2) if w_ is not None else jnp.sum(r**2)

        ols_val_grad = jax.jit(jax.value_and_grad(_ols_loss))

        if M is not None:

            def _robust_loss(theta, sigma):
                return jnp.sum(M.rho((y_ - model(theta, t_)) / sigma))

            robust_val_grad = jax.jit(jax.value_and_grad(_robust_loss))
    else:
        t_ = np.asarray(t)
        y_ = np.asarray(trace)
        x = np.asarray(x0, dtype=float).copy()
        w_ = np.asarray(weights) if weights is not None else None

    # ----------------------------
    # objective factories
    # ----------------------------
    def make_objective_numpy(sigma=None):
        if sigma is None:

            def obj(theta):
                if has_return_jac:
                    y_pred, J = model(theta, t_, return_jac=True)
                    r = y_ - y_pred
                    if w_ is not None:
                        return np.sum(w_ * r**2), -2.0 * (J.T @ (w_ * r))
                    return np.sum(r**2), -2.0 * J.T @ r
                r = y_ - model(theta, t_)
                return np.sum(w_ * r**2) if w_ is not None else np.sum(r**2)

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

        if fixed_sigma is not None:  # use fixed sigma if provided
            _sigma = fixed_sigma
        elif use_jax:
            _sigma = jnp.median(jnp.abs(resid)) * 1.4826
            _sigma = jnp.where(_sigma == 0, jnp.std(resid), _sigma)
        else:
            _sigma = scale.mad(resid, center=0)
            if _sigma == 0:
                _sigma = np.std(resid)

        fun_or_pair = make_objective(_sigma)
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
        res.sigma = float(_sigma)
        u = (np.array(y_) - fitted) / float(_sigma)
        res.weights = np.array(M.weights(u))
    return fitted, res


def robust_lowess(
    # --- data ---
    y: np.ndarray,
    t: np.ndarray,
    # --- smoother ---
    frac: float = 0.1,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float | None]:
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
    M : RobustNorm or None
        If None, single-pass LOWESS. Otherwise, outer IRLS using M.weights.
        Any M-estimator with a .weights(z) method works, including
        AsymmetricTukeyBiweight and OneSidedTukeyBiweight.
    weights : np.ndarray or None
        Initial point weights, e.g. res.weights from trend fit.
        Warm-starts the IRLS loop. Defaults to uniform weights.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
    maxiter : int
        Maximum IRLS iterations. Ignored when M is None.
    tol : float
        Convergence tolerance on max weight change between iterations.

    Returns
    -------
    fluctuation : np.ndarray
        Smoothed signal.
    w_current : np.ndarray
        Final IRLS point weights, shape ``(N,)``.
        Uniform (all ones) when ``M=None``; otherwise the converged
        M-estimator weights from the last iteration.
    sigma : float | None
        Robust scale estimate.
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
            y,
            t,
            t,
            resid_weights=w_current,
            frac=frac,
            it=0,
            delta=delta,
        )[0][:, 1]

        if M is None:
            sigma = None
            break

        resid = y - fluctuation
        if fixed_sigma is not None:
            sigma = fixed_sigma
        else:
            sigma = np.median(np.abs(resid)) * 1.4826
            if sigma == 0:
                sigma = np.std(resid)

        w_new = M.weights(resid / sigma)
        if np.max(np.abs(w_new - w_current)) < tol:
            break
        w_current = w_new

    return fluctuation, w_current, sigma


def fit_baseline_fluctuations(
    # --- data ---
    trace: np.ndarray,
    t: np.ndarray,
    trend: np.ndarray | None = None,
    # --- smoother ---
    mode: Literal["ratio", "subtract"] = "ratio",
    frac: float = 0.1,
    method: Literal["lowess", "percentile"] = "lowess",
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    # --- percentile ---
    percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
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
    method : {"lowess", "percentile"}
        Smoothing method.
        ``"lowess"`` — robust locally weighted regression via
        :func:`robust_lowess`.
        ``"percentile"`` — sliding percentile filter via
        :func:`percentile_filter`.
    M : RobustNorm or None
        *LOWESS only.* M-estimator with a ``.weights(z)`` method
        (e.g. :class:`OneSidedTukeyBiweight`). If ``None``, single-pass
        LOWESS without outer IRLS. Ignored when ``method="percentile"``.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``. For ``"lowess"``, warm-starts the
        IRLS loop (e.g. pass ``res.weights`` from :func:`nonlinear_fit`).
        For ``"percentile"``, used to estimate the baseline percentile when
        ``percentile=None``.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
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
    info : dict
        Diagnostics from the fluctuation fit.
        ``{"lowess_weights": w, "lowess_sigma": sigma}`` when ``method="lowess"``;
        ``{"percentile": p, "size": s}`` when ``method="percentile"``.
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
        fluctuation, w, sigma = robust_lowess(y, t, frac, M, weights, fixed_sigma, maxiter, tol)
        info = {"lowess_weights": w, "lowess_sigma": sigma}
    elif method == "percentile":
        size = max(1, round(frac * len(trace)))
        if percentile is None:
            # estimate from weights if available
            mu_w = (
                np.average(y, weights=weights) if weights is not None else np.median(y)
            )
            percentile = np.clip(np.mean(y <= mu_w) * 100, 5, 50)
        fluctuation = percentile_filter(y, percentile, size)
        info = {"percentile": percentile, "size": size}
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # retrend — shared
    if trend is None:
        return fluctuation, fluctuation, info
    baseline = trend * fluctuation if mode == "ratio" else trend + fluctuation
    return baseline, fluctuation, info


def fit_baseline(
    # --- data / model ---
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    M_fluctuations: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    # --- smoother ---
    mode: Literal["ratio", "subtract"] = "ratio",
    frac: float = 0.1,
    method: Literal["lowess", "percentile"] = "lowess",
    # --- percentile ---
    percentile: float | None = None,
    # --- optimizer ---
    optimizer: str = "L-BFGS-B",
    optimizer_options: dict | None = None,
    # --- backend ---
    backend: Literal["numpy", "jax"] = "numpy",
    dtype=jnp.float64,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult, dict]:
    """
    Fit a full fluorescence baseline: slow trend then local fluctuations.

    Combines :func:`nonlinear_fit` (parametric trend) with
    :func:`fit_baseline_fluctuations` (LOWESS or percentile smoothing) into a
    single call.

    Parameters
    ----------
    trace : np.ndarray
        Raw fluorescence signal, shape ``(N,)``.
    t : np.ndarray
        Timestamps, shape ``(N,)``. Must be sorted in ascending order.
    model : callable
        Parametric trend model, e.g. :func:`single_exp` or :func:`double_exp`.
        See :func:`nonlinear_fit` for calling conventions.
    x0 : np.ndarray
        Initial parameter vector for ``model``, shape ``(n_params,)``.
    bounds : sequence of (min, max) pairs or None
        Parameter bounds passed to ``scipy.optimize.minimize``.
    M : RobustNorm or None
        M-estimator for the trend fit (:func:`nonlinear_fit`).
        ``None`` → OLS.  For ``backend="jax"``, it's converted automatically
        via ``with_xp(jnp)``.
        When ``M_fluctuations`` is ``None``, also used for the fluctuation fit.
    M_fluctuations : RobustNorm or None
        M-estimator for the fluctuation fit (:func:`fit_baseline_fluctuations`).
        Falls back to ``M.with_xp(np)`` when ``None``, ensuring NumPy arrays
        are used in LOWESS regardless of backend.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``, used to warm-start the OLS
        pre-pass of the trend fit (passed to :func:`nonlinear_fit`).
        Typically ``res.weights`` from a previous call. ``None`` → uniform
        weights. The fluctuation fit always uses the weights produced by
        the trend fit, not this argument.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
    maxiter : int
        Maximum IRLS iterations for both the trend and fluctuation fits.
    tol : float
        Convergence tolerance for both IRLS loops.
    mode : {"ratio", "subtract"}
        How to detrend before estimating fluctuations. ``"ratio"`` divides
        ``trace`` by the trend (multiplicative); ``"subtract"`` removes it
        additively.
    frac : float
        Smoothing bandwidth as a fraction of ``N``, passed to
        :func:`fit_baseline_fluctuations`.
    method : {"lowess", "percentile"}
        Fluctuation estimation method.
    percentile : float or None
        *Percentile method only.* Percentile to track (0–100). Auto-estimated
        from ``res.weights`` when ``None``.
    optimizer : str
        Solver passed to ``scipy.optimize.minimize``, default ``"L-BFGS-B"``.
    optimizer_options : dict or None
        Options forwarded to ``scipy.optimize.minimize``.
        Defaults to ``{"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}``.
    backend : {"numpy", "jax"}
        Backend for the trend fit. ``"jax"`` enables autodiff and JIT
        compilation; ``"numpy"`` uses analytic Jacobians when available.
    dtype : jax dtype
        Floating-point precision for the JAX backend, default
        ``jnp.float64``. Requires ``jax_enable_x64=True``.

    Returns
    -------
    F0 : np.ndarray
        Full baseline in the original signal space, shape ``(N,)``.
    F0trend : np.ndarray
        Parametric trend component (output of :func:`nonlinear_fit`),
        shape ``(N,)``.
    res : OptimizeResult
        Result from the final trend optimisation, with additional attributes
        ``res.sigma`` and ``res.weights`` when ``M`` is not ``None``.
    info : dict
        Diagnostics from the fluctuation fit.
        ``{"lowess_weights": w, "lowess_sigma": sigma}`` when ``method="lowess"``;
        ``{"percentile": p, "size": s}`` when ``method="percentile"``.
    """
    if M_fluctuations is None:
        M_fluctuations = M.with_xp(np) if M is not None else None
    F0trend, res = nonlinear_fit(
        trace,
        t,
        model,
        x0,
        bounds,
        M,
        weights,
        fixed_sigma,
        maxiter,
        tol,
        optimizer,
        optimizer_options=optimizer_options,
        backend=backend,
        dtype=dtype,
    )
    weights = getattr(res, "weights", None)
    F0, _, info = fit_baseline_fluctuations(
        trace,
        t,
        F0trend,
        mode,
        frac,
        method,
        M_fluctuations,
        weights,
        fixed_sigma,
        maxiter,
        tol,
        percentile,
    )
    return F0, F0trend, res, info
