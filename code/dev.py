import inspect
import logging
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm, TukeyBiweight

jax.config.update("jax_enable_x64", True)


# -----------------------------
#  Baselines
# -----------------------------
def single_exp(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Baseline with exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [b_inf, b, tau]
    t : np.ndarray
        Timestamps

    Returns
    -------
    np.ndarray
        Baseline signal
    """
    b_inf, b, tau = params
    return b_inf * (1 + b * np.exp(-t / tau))


def single_exp_jax(params, t):
    b_inf, b, tau = params
    return b_inf * (1 + b * jnp.exp(-t / tau))


def double_exp(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Baseline with biphasic exponential decay (bleaching).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [b_inf, b_slow, b_fast, t_slow, t_fast]
    t : np.ndarray
        Timestamps

    Returns
    -------
    np.ndarray
        Baseline signal
    """
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    return b_inf * (1 + b_slow * np.exp(-t / t_slow) + b_fast * np.exp(-t / t_fast))


def double_exp_jax(params, t):
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    return b_inf * (1 + b_slow * jnp.exp(-t / t_slow) + b_fast * jnp.exp(-t / t_fast))


def bright(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Baseline with triphasic exponential decay (bleaching) multiplied by
    increasing saturating exponential (brightening).

    Parameters
    ----------
    params : np.ndarray
        Parameter vector
        [b_inf, b_slow, b_fast, b_rapid, b_bright,
         t_slow, t_fast, t_rapid, t_bright]
    t : np.ndarray
        Timestamps

    Returns
    -------
    np.ndarray
        Baseline signal
    """
    b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params

    A = (
        1
        + b_slow * np.exp(-t / t_slow)
        + b_fast * np.exp(-t / t_fast)
        + b_rapid * np.exp(-t / t_rapid)
    )
    B = 1 - b_bright * np.exp(-t / t_bright)

    return b_inf * A * B


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
#  analytic Jacobians
# -----------------------------
def single_exp_jac(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Jacobian of single exponential baseline.

    Parameters
    ----------
    params : np.ndarray
        [b_inf, b, tau]
    t : np.ndarray
        Timestamps

    Returns
    -------
    J : np.ndarray, shape (len(t), 3)
        Jacobian matrix d(model)/d(params)
    """
    b_inf, b, tau = params
    E = np.exp(-t / tau)
    bE = b_inf * E

    J = np.empty((t.size, 3))
    J[:, 0] = 1 + b * E  # d/d b_inf
    J[:, 1] = bE  # d/d b
    J[:, 2] = b / tau**2 * (bE * t)  # d/d tau

    return J


def double_exp_jac(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Jacobian of double exponential baseline.

    Parameters
    ----------
    params : np.ndarray
        [b_inf, b_slow, b_fast, t_slow, t_fast]
    t : np.ndarray
        Timestamps

    Returns
    -------
    J : np.ndarray, shape (len(t), 5)
        Jacobian matrix d(model)/d(params)
    """
    b_inf, b_slow, b_fast, t_slow, t_fast = params
    E_slow = np.exp(-t / t_slow)
    E_fast = np.exp(-t / t_fast)
    bE_s = b_inf * E_slow
    bE_f = b_inf * E_fast

    J = np.empty((t.size, 5))
    J[:, 0] = 1 + b_slow * E_slow + b_fast * E_fast  # d/d b_inf
    J[:, 1] = bE_s  # d/d b_slow
    J[:, 2] = bE_f  # d/d b_fast
    J[:, 3] = b_slow / t_slow**2 * (bE_s * t)  # d/d t_slow
    J[:, 4] = b_fast / t_fast**2 * (bE_f * t)  # d/d t_fast

    return J


def bright_jac(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Jacobian of triphasic exponential baseline with brightening.
    Uses temporary arrays to avoid recomputing repeated terms.

    Parameters
    ----------
    params : np.ndarray
        [b_inf, b_slow, b_fast, b_rapid, b_bright,
         t_slow, t_fast, t_rapid, t_bright]
    t : np.ndarray
        Timestamps

    Returns
    -------
    J : np.ndarray, shape (len(t), 9)
        Jacobian matrix d(model)/d(params)
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

    return J


# -----------------------------
#  Baselines with Jacobians
# -----------------------------
def single_exp_with_jac(
    params: np.ndarray, t: np.ndarray, return_jac: bool = False
) -> np.ndarray:
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


def double_exp_with_jac(params: np.ndarray, t: np.ndarray, return_jac: bool = False):
    """
    Baseline with biphasic exponential decay (bleaching).

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


def bright_with_jac(params: np.ndarray, t: np.ndarray, return_jac: bool = False):
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

    def rho(self, z):
        z = np.asarray(z)
        res = np.empty_like(z)
        # Handle positive side
        pos_mask = z > 0
        if np.isinf(self.c_pos):
            res[pos_mask] = 0.5 * z[pos_mask] ** 2
        else:
            pos_inside = pos_mask & (z <= self.c_pos)
            pos_outside = z > self.c_pos
            res[pos_inside] = self.factor_pos * (
                1 - (1 - (z[pos_inside] / self.c_pos) ** 2) ** 3
            )
            res[pos_outside] = self.factor_pos
        # Handle negative side
        neg_mask = z <= 0
        if np.isinf(self.c_neg):
            res[neg_mask] = 0.5 * z[neg_mask] ** 2
        else:
            neg_inside = neg_mask & (z >= -self.c_neg)
            neg_outside = z < -self.c_neg
            res[neg_inside] = self.factor_neg * (
                1 - (1 - (z[neg_inside] / self.c_neg) ** 2) ** 3
            )
            res[neg_outside] = self.factor_neg

        return res

    def psi(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        res[pos_inside] = z[pos_inside] * (1 - (z[pos_inside] / self.c_pos) ** 2) ** 2
        res[neg_inside] = z[neg_inside] * (1 - (z[neg_inside] / self.c_neg) ** 2) ** 2
        return res

    def weights(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        res[pos_inside] = (1 - (z[pos_inside] / self.c_pos) ** 2) ** 2
        res[neg_inside] = (1 - (z[neg_inside] / self.c_neg) ** 2) ** 2
        return res

    def psi_deriv(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        t_pos = z[pos_inside] / self.c_pos
        t_pos_sq = t_pos**2
        res[pos_inside] = (1 - t_pos_sq) ** 2 - 4 * t_pos_sq * (
            1 - t_pos_sq
        ) / self.c_pos**2
        t_neg = z[neg_inside] / self.c_neg
        t_neg_sq = t_neg**2
        res[neg_inside] = (1 - t_neg_sq) ** 2 - 4 * t_neg_sq * (
            1 - t_neg_sq
        ) / self.c_neg**2
        return res


class OneSidedTukeyBiweight(AsymmetricTukeyBiweight):
    """
    A one-sided Tukey Biweight norm that applies quadratic loss to negative
    residuals and Tukey biweight loss to positive residuals.

    This is implemented as a special case of AsymmetricTukeyBiweight
    with c_neg=np.inf, which simplifies to quadratic loss for negative values.
    """

    def __init__(self, c=4.685):
        super().__init__(c_pos=c, c_neg=np.inf)


class TukeyBiweight_jax:
    """Tukey biweight M-estimator"""

    def __init__(self, c=4.685):
        self.c = c

    def rho(self, u):
        c = jnp.asarray(self.c, dtype=u.dtype)
        abs_u = jnp.abs(u)
        mask = abs_u <= c
        return jnp.where(mask, (c**2 / 6) * (1 - (1 - (u / c) ** 2) ** 3), c**2 / 6)

    def psi(self, u):
        c = jnp.asarray(self.c, dtype=u.dtype)
        mask = jnp.abs(u) <= c
        return jnp.where(mask, u * (1 - (u / c) ** 2) ** 2, 0.0)


# -----------------------------
#  Fitting Functions
# -----------------------------
def tc_brightfit(
    trace: np.ndarray,
    t: np.ndarray,
    rss_thresh: tuple[float, float] = (0.98, 0.997),
    M: RobustNorm = TukeyBiweight(3),
    maxiter: int = 5,
    tol: float = 1e-3,
    update_scale: bool = True,
    skewness_factor: float = 1.0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray, "OptimizeResult"]:
    """Fit trace with baseline (bleaching x brightening) using OLS or IRLS."""

    CEND = "\33[0m"
    CBOLD = "\33[1m"
    CRED = "\33[31m"
    CGREEN = "\33[32m"

    T = len(trace)
    Tds = T // 10
    if rss_thresh == "BIC":
        rss_thresh = [Tds ** (-2 / Tds)] * 2
    elif rss_thresh == "AIC":
        rss_thresh = [np.exp(-4 / Tds)] * 2

    def optimize(trace, x0, ds=1, maxiter=20000, weights=1, plot=plot):
        """Optimize parameters for a given model structure."""
        trace_ds = trace[: T // ds * ds].reshape(-1, ds).mean(1)
        optimize_param = ~np.isnan(x0)

        params = np.array([0] * 5 + [np.inf] * 4)

        def objective(params_to_optimize):
            params[optimize_param] = params_to_optimize
            resid = trace_ds - bright(params, t[ds // 2 :: ds][: len(trace_ds)])
            return np.sum(weights * resid**2)

        bounds = np.array(
            [(0, np.inf)] * 5 + [(300, np.inf), (1, 1200), (1, 180), (60, np.inf)]
        )

        res = minimize(
            objective,
            np.array(x0)[optimize_param],
            bounds=bounds[optimize_param],
            method="Nelder-Mead",
            options={"maxiter": maxiter},
        )

        params[optimize_param] = res.x
        res.params_full = params.copy()

        logging.info(
            f"Cost: {res.fun:.3f}  "
            f"Success: {CGREEN if res.success else CRED} {res.success} {CEND}  "
            f"{res.message}"
        )

        if plot:
            plot_fit(params, trace, t)

        return params, res

    # ---- double exponential fit ----

    x0 = np.array(
        [trace[-1000:].mean(), 0.35, 0.2, np.nan, np.nan, 3600, 240, np.nan, np.nan]
    )

    logging.info(f"{CBOLD}Fit of 10x decimated trace with double-exp{CEND}")
    x2, res2 = optimize(trace, x0, 10)
    cost2 = res2.fun

    if x2[6] > x2[5]:
        x2[[1, 2, 5, 6]] = x2[[2, 1, 6, 5]]

    # ---- brightening fit ----

    x0[~np.isnan(x0)] = x2[~np.isnan(x0)]
    x0[[4, 8]] = 0.1, 2000

    logging.info(f"{CBOLD}Fit of 10x decimated trace with brightening{CEND}")
    xB, resB = optimize(trace, x0, 10, 3000)
    costB = resB.fun

    cost_ratio = costB / cost2
    include_bright = cost_ratio < rss_thresh[0]

    logging.info(
        f"Cost reduction by including brightening is {(cost_ratio-1)*100:.3f}%, "
        f"thus {CBOLD}{'including' if include_bright else 'skipping'}{CEND} brightening term."
    )

    if include_bright:
        x0[~np.isnan(x0)] = xB[~np.isnan(x0)]
    else:
        x0[[4, 8]] = np.nan

    # ---- third exponential ----

    x0[[3, 7]] = 0.1, 50

    logging.info(f"{CBOLD}Fit of 10x decimated trace with triple-exp{CEND}")
    x3, res3 = optimize(trace, x0, 10, 3000)
    cost3 = res3.fun

    order = np.argsort(x3[5:8])[::-1]
    x3[5:8] = x3[5 + order]
    x3[1:4] = x3[1 + order]

    cost_ratio = cost3 / (costB if include_bright else cost2)
    include_3rd = cost_ratio < rss_thresh[1]

    logging.info(
        f"Cost reduction by including 3rd exponential is {(cost_ratio-1)*100:.3f}%, "
        f"thus {CBOLD}{'including' if include_3rd else 'skipping'}{CEND} 3rd exponential term."
    )

    if include_3rd:
        x0[~np.isnan(x0)] = x3[~np.isnan(x0)]
    else:
        x0[[3, 7]] = np.nan

    params = np.array([0] * 5 + [np.inf] * 4)
    params[~np.isnan(x0)] = x0[~np.isnan(x0)]

    logging.info(
        f"Cost on original trace with params obtained on decimated trace is "
        f"{np.sum((trace - bright(params, t)) ** 2):.3f}"
    )

    logging.info(
        f"{CBOLD}Fit of original trace with "
        f"{'triple-exp' if include_3rd else 'double-exp'} "
        f"and {'' if include_bright else 'no '}brightening{CEND}"
    )

    x, res_final = optimize(trace, x0)

    # ---- IRLS stage ----

    if maxiter > 0 and M is not None and res_final.fun > 0:

        f0 = bright(x, t)
        resid = trace - f0
        scl = scale.mad(resid if skewness_factor == 0 else resid[resid < 0], center=0)

        deviance = M(resid / scl).sum()
        iteration = 0
        converged = False

        while not converged:

            iteration += 1

            weights = M.weights(resid / scl)

            x[np.isnan(x0)] = np.nan
            x, res_final = optimize(trace, x, weights=weights, plot=False)

            f0 = bright(x, t)
            resid = trace - f0

            if update_scale:
                scl = scale.mad(
                    resid if skewness_factor == 0 else resid[resid < 0], center=0
                )

            dev_pre = deviance
            deviance = M(resid / scl).sum()

            converged = iteration >= maxiter or np.abs(deviance / dev_pre - 1) < tol

        res_final.irls_iterations = iteration
        res_final.sigma = scl
        res_final.fun = deviance

    baseline = bright(x, t)

    return baseline, res_final


def fit_baseline_trend(
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    jac: Callable | str | None = None,
    M: RobustNorm | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = None,
    dtype=jnp.float64,
):
    """
    General nonlinear model fitting.

    Parameters
    ----------
    trace : array
        Observed signal
    t : array
        Time vector (passed to model)
    model : callable
        model(params, t) -> model prediction
    init_params : array
        Initial parameter vector
    bounds : array
        Parameter bounds
    jac : callable, optional
        Analytical Jacobian  jac(params, t) -> array_like (len(t), n_params)
        If "jax", automatic differentiation using JAX
        If None, gradient is computed numerically by SciPy optimizer
    M : None or RobustNorm object with .rho(u)
        If None -> OLS
        If provided -> IRLS with M-estimator
    optimizer : str
        Method for scipy.optimize.minimize
    maxiter : int
        Maximum IRLS iterations (ignored if M=None)
    tol : float
        Convergence tolerance
    optimizer_options : dict
        Options passed to scipy.optimize.minimize

    Returns
    -------
    fitted : np.ndarray
        Fitted model
    res : OptimizeResult
        Optimizer Result
    """

    if jac == "jax":
        return nonlinear_fit_jax(
            trace,
            t,
            model,
            init_params,
            bounds,
            M,
            optimizer,
            maxiter,
            tol,
            optimizer_options,
            dtype,
        )
    else:
        return nonlinear_fit_numpy(
            trace,
            t,
            model,
            init_params,
            bounds,
            jac,
            M,
            optimizer,
            maxiter,
            tol,
            optimizer_options,
        )


def nonlinear_fit_numpy(
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    jac: Callable | None = None,
    M: RobustNorm | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = None,
):
    """
    General nonlinear model fitting.

    Parameters
    ----------
    trace : array
        Observed signal
    t : array
        Time vector (passed to model)
    model : callable
        model(params, t) -> model prediction
    init_params : array
        Initial parameter vector
    bounds : array
        Parameter bounds
    jac : callable or "jax", optional
        Analytical Jacobian  jac(params, t) -> array_like (len(t), n_params)
        If None, gradient is computed numerically by optimizer.
    M : None or RobustNorm object with .rho(u)
        If None -> OLS
        If provided -> IRLS with M-estimator
    optimizer : str
        Method for scipy.optimize.minimize
    maxiter : int
        Maximum IRLS iterations (ignored if M=None)
    tol : float
        Convergence tolerance
    optimizer_options : dict
        Options passed to scipy.optimize.minimize

    Returns
    -------
    fitted : np.ndarray
        Fitted model
    res : OptimizeResult
        Optimizer Result
    """

    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000}

    x = np.asarray(init_params).copy()

    # ----------------------------
    # OLS case
    # ----------------------------
    if M is None:

        def objective(p):
            resid = trace - model(p, t)
            loss = np.sum(resid**2)
            # loss = np.mean(resid ** 2)
            if jac is None:
                return loss
            J = jac(p, t)
            grad = -2 * J.T @ resid
            # grad = -2/resid.size * J.T @ resid
            return loss, grad

        res = minimize(
            objective,
            x,
            bounds=bounds,
            method=optimizer,
            jac=(jac is not None),
            options=optimizer_options,
        )

        return model(res.x, t), res

    # ----------------------------
    # Robust IRLS case
    # ----------------------------
    for iteration in range(maxiter):

        resid = trace - model(x, t)

        # estimate scale
        sigma = scale.mad(resid, center=0)
        if sigma == 0:
            sigma = np.std(resid)

        def objective(p):
            resid = trace - model(p, t)
            u = resid / sigma
            loss = np.sum(M.rho(u))
            # loss = np.mean(M.rho(u))
            if jac is None:
                return loss
            J = jac(p, t)
            psi = M.psi(u)
            grad = -(J.T @ psi) / sigma
            # grad = -(J.T @ psi) / (sigma * resid.size)
            return loss, grad

        res = minimize(
            objective,
            x,
            bounds=bounds,
            method=optimizer,
            jac=(jac is not None),
            options=optimizer_options,
        )

        if np.linalg.norm(res.x - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = res.x
            break
        x = res.x

    res.sigma = sigma

    return model(x, t), res


def fit_baseline_trend2(
    trace: np.ndarray,
    t: np.ndarray,
    model: callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    M: RobustNorm | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = None,
):
    """
    Fit a nonlinear baseline to a single trace.
    Supports OLS (M=None) or IRLS (M=M-estimator).
    Model can optionally return a Jacobian as `model(params, t, return_jac=True)`.

    Parameters
    ----------
    trace : np.ndarray
        Observed signal
    t : np.ndarray
        Time vector
    model : callable
        model(params, t) -> prediction
        optionally supports `return_jac=True`
    init_params : np.ndarray
        Initial parameter vector
    bounds : tuple[tuple[float, float], ...] | None
        Parameter bounds
    M : None or RobustNorm
        If None -> OLS, otherwise IRLS with M-estimator
    optimizer : str
        Optimization method for scipy.minimize
    maxiter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    optimizer_options : dict | None
        Options for scipy.minimize

    Returns
    -------
    fitted : np.ndarray
        Fitted model
    res : OptimizeResult
        Optimizer Result
    """
    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000}

    x = np.asarray(init_params).copy()

    # ----------------------------
    # check if model supports return_jac argument
    sig = inspect.signature(model)
    has_return_jac = "return_jac" in sig.parameters

    # ----------------------------
    # OLS case
    # ----------------------------
    if M is None:

        def objective(p):
            if has_return_jac:
                y_pred, J = model(p, t, return_jac=True)
                resid = trace - y_pred
                grad = -2 * J.T @ resid
                return np.sum(resid**2), grad
            else:
                resid = trace - model(p, t)
                return np.sum(resid**2)

        res = minimize(
            objective,
            x,
            bounds=bounds,
            method=optimizer,
            jac=has_return_jac,
            options=optimizer_options,
        )
        return model(res.x, t), res

    # ----------------------------
    # Robust IRLS case
    # ----------------------------
    for iteration in range(maxiter):

        resid = trace - model(x, t)

        # estimate scale
        sigma = scale.mad(resid, center=0)
        if sigma == 0:
            sigma = np.std(resid)

        def objective(p):
            r = trace - model(p, t)
            u = r / sigma
            loss = np.sum(M.rho(u))
            if has_return_jac:
                _, J = model(p, t, return_jac=True)
                psi = M.psi(u)
                grad = -(J.T @ psi) / sigma
                return loss, grad
            else:
                return loss

        res = minimize(
            objective,
            x,
            bounds=bounds,
            method=optimizer,
            jac=has_return_jac,
            options=optimizer_options,
        )

        # check convergence
        if np.linalg.norm(res.x - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = res.x
            break
        x = res.x

    res.sigma = sigma

    return model(x, t), res


def nonlinear_fit_jax(
    trace: np.ndarray,
    t: np.ndarray,
    model: callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    M: TukeyBiweight_jax | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = dict(maxiter=20000, ftol=1e-12, gtol=1e-10),
    dtype=jnp.float64,
):
    """
    JAX-based nonlinear baseline fitting with optional IRLS robust regression.
    """

    # convert data to JAX arrays
    t = jnp.asarray(t, dtype=dtype)
    y = jnp.asarray(trace, dtype=dtype)
    x = jnp.asarray(init_params, dtype=dtype)

    # -------------------------
    # Helper: create fun/jac closure
    # -------------------------
    def make_fun_and_jac(loss_grad_fn):
        grad_cache = {}

        def fun(p):
            val, grad = loss_grad_fn(jnp.asarray(p, dtype=dtype))
            grad_cache["grad"] = np.array(grad)
            return float(val)

        def jac(p):
            return grad_cache["grad"]

        return fun, jac

    # -------------------------
    # Pre-jit loss functions
    # -------------------------
    def ols_loss(params):
        resid = y - model(params, t)
        return jnp.sum(resid**2)

    ols_loss_grad = jax.jit(jax.value_and_grad(ols_loss))

    if M is not None:

        def robust_loss(params, sigma):
            resid = y - model(params, t)
            u = resid / sigma
            return jnp.sum(M.rho(u))

        robust_loss_grad = jax.jit(jax.value_and_grad(robust_loss))

    # -------------------------
    # OLS case
    # -------------------------
    if M is None:
        fun, jac = make_fun_and_jac(ols_loss_grad)
        res = minimize(
            fun, x, jac=jac, bounds=bounds, method=optimizer, options=optimizer_options
        )
        params = jnp.array(res.x, dtype=dtype)
        fitted = model(params, t)
        return jnp.array(fitted), res

    # -------------------------
    # IRLS loop for robust regression
    # -------------------------
    for iteration in range(maxiter):
        # compute MAD scale
        resid = y - model(x, t)
        sigma = jnp.median(jnp.abs(resid)) * dtype(1.4826)
        sigma = jnp.where(sigma == 0, jnp.std(resid), sigma)

        # make fun/jac for this sigma
        fun, jac = make_fun_and_jac(
            lambda p: robust_loss_grad(jnp.asarray(p, dtype=dtype), sigma)
        )

        # optimize step
        res = minimize(
            fun, x, jac=jac, bounds=bounds, method=optimizer, options=optimizer_options
        )
        x_new = jnp.asarray(res.x, dtype=dtype)

        # update sigma with new params
        resid_new = y - model(x_new, t)
        sigma = jnp.median(jnp.abs(resid_new)) * dtype(1.4826)
        sigma = jnp.where(sigma == 0, jnp.std(resid_new), sigma)

        # check convergence
        eps = dtype(1e-12)
        if jnp.linalg.norm(x_new - x) / (jnp.linalg.norm(x) + eps) < tol:
            x = x_new
            break

        x = x_new

    res.sigma = sigma
    fitted = model(x, t)
    return np.array(fitted), res


def nonlinear_fit(
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    init_params: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    jac: Callable | None = None,
    M: RobustNorm | None = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 5,
    tol: float = 1e-3,
    optimizer_options: dict | None = None,
    backend: str = "numpy",  # "numpy" or "jax"
    dtype=jnp.float64,
):
    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}

    use_jax = backend == "jax"

    if use_jax:
        t_ = jnp.asarray(t, dtype=dtype)
        y_ = jnp.asarray(trace, dtype=dtype)
        x = jnp.asarray(init_params, dtype=dtype)

        def _make_obj(loss_and_grad_fn):
            cache = {}

            def fun(p):
                val, grad = loss_and_grad_fn(jnp.asarray(p, dtype=dtype))
                cache["g"] = np.array(grad)
                return float(val)

            return fun, lambda p: cache["g"]

        def _ols_loss(p):
            return jnp.sum((y_ - model(p, t_)) ** 2)

        ols_val_grad = jax.jit(jax.value_and_grad(_ols_loss))

        if M is not None:

            def _robust_loss(p, sigma):
                return jnp.sum(M.rho((y_ - model(p, t_)) / sigma))

            robust_val_grad = jax.jit(jax.value_and_grad(_robust_loss))
    else:
        t_ = np.asarray(t)
        y_ = np.asarray(trace)
        x = np.asarray(init_params, dtype=float).copy()

    # ----------------------------
    # Build objective for one step
    # ----------------------------
    def make_objective_numpy(sigma=None):
        if sigma is None:  # OLS

            def obj(p):
                r = y_ - model(p, t_)
                loss = np.sum(r**2)
                if jac is None:
                    return loss
                return loss, -2.0 * jac(p, t_).T @ r

        else:  # robust

            def obj(p):
                r = y_ - model(p, t_)
                u = r / sigma
                loss = np.sum(M.rho(u))
                if jac is None:
                    return loss
                return loss, -(jac(p, t_).T @ M.psi(u)) / sigma

        return obj

    def make_objective_jax(sigma=None):
        if sigma is None:
            return _make_obj(ols_val_grad)
        else:
            return _make_obj(lambda p: robust_val_grad(p, sigma))

    make_objective = make_objective_jax if use_jax else make_objective_numpy

    # ----------------------------
    # OLS case
    # ----------------------------
    if M is None:
        fun_or_pair = make_objective()
        if use_jax:
            fun, jac_ = fun_or_pair
        else:
            fun, jac_ = fun_or_pair, (jac is not None)

        res = minimize(
            fun,
            x,
            bounds=bounds,
            method=optimizer,
            jac=jac_ if use_jax else (jac is not None),
            options=optimizer_options,
        )
        fitted = model(jnp.asarray(res.x, dtype=dtype) if use_jax else res.x, t_)
        return np.array(fitted), res

    # ----------------------------
    # IRLS loop
    # ----------------------------
    for _ in range(maxiter):
        resid = y_ - model(x, t_)

        if use_jax:
            sigma = jnp.median(jnp.abs(resid)) * 1.4826
            sigma = jnp.where(sigma == 0, jnp.std(resid), sigma)
        else:
            sigma = scale.mad(resid, center=0)
            if sigma == 0:
                sigma = np.std(resid)

        fun_or_pair = make_objective(sigma)
        if use_jax:
            fun, jac_ = fun_or_pair
        else:
            fun, jac_ = fun_or_pair, (jac is not None)

        res = minimize(
            fun,
            x,
            bounds=bounds,
            method=optimizer,
            jac=jac_ if use_jax else (jac is not None),
            options=optimizer_options,
        )

        x_new = jnp.asarray(res.x, dtype=dtype) if use_jax else res.x
        norm_x = jnp.linalg.norm if use_jax else np.linalg.norm
        if norm_x(x_new - x) / (norm_x(x) + 1e-12) < tol:
            x = x_new
            break
        x = x_new

    res.sigma = float(sigma)
    fitted = model(x, t_)
    return np.array(fitted), res
