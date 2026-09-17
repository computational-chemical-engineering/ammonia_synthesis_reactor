"""
Numerical solvers for the membrane reactor model.

Contains generic Newton-Raphson solver with Armijo line search,
and continuation methods for robust nonlinear solving.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array
import scipy.sparse.linalg as sla

from .numerical_safety import RecoverableNumericalError

# Module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# Solver configuration
# =============================================================================


@dataclass
class NewtonConfig:
    """Configuration for Newton solver."""

    max_iterations: int = 10
    rtol: float = 1e-6
    atol: float = 0.0
    ord_norm: int = 2
    armijo_coeff: float = 1e-4
    min_line_search_alpha: float = 1e-3
    use_line_search: bool = True
    verbose: int = 0


@dataclass
class NewtonResult:
    """Result of Newton solver."""

    converged: bool
    num_iterations: int
    residual_norm: float
    residual_norm_initial: float
    convergence_factor: float

    @property
    def relative_reduction(self) -> float:
        """Relative residual reduction."""
        if self.residual_norm_initial > 0:
            return self.residual_norm / self.residual_norm_initial
        return 0.0


# =============================================================================
# Line search
# =============================================================================


def armijo_line_search(
    x: NDArray,
    dx: NDArray,
    g_norm: float,
    residual_fn: Callable[[NDArray], NDArray],
    norm_fn: Callable[[NDArray], float],
    armijo_coeff: float = 1e-4,
    min_alpha: float = 1e-3,
    max_backtracks: int = 10,
) -> Tuple[NDArray, float, float, bool]:
    """Perform Armijo backtracking line search.

    Finds step size alpha such that:
        ||g(x + alpha*dx)|| <= (1 - armijo_coeff*alpha) * ||g(x)||

    Args:
        x: Current solution vector
        dx: Newton step direction
        g_norm: Current residual norm ||g(x)||
        residual_fn: Function computing residual g(x)
        norm_fn: Function computing norm of residual
        armijo_coeff: Sufficient decrease parameter (typically 1e-4)
        min_alpha: Minimum step size before declaring failure
        max_backtracks: Maximum number of backtracking steps

    Returns:
        Tuple of (x_new, g_new_norm, alpha, success)
    """
    alpha = 1.0
    x_new = x.copy()
    best_norm, best_x, best_alpha = np.inf, None, 0.0

    def evaluate(x_trial):
        """Residual norm of a trial point; overflow/invalid counts as inf.

        A trial point far outside the physical region can make the guarded
        kinetics raise. That must mean "backtrack", not "abort the solve".
        """
        try:
            g_trial = residual_fn(x_trial)
        except RecoverableNumericalError:
            return np.inf
        n = norm_fn(g_trial)
        return n if np.isfinite(n) else np.inf

    for _ in range(max_backtracks):
        x_new[:] = x + alpha * dx
        g_new_norm = evaluate(x_new)

        # Armijo sufficient decrease condition
        if g_new_norm <= (1.0 - armijo_coeff * alpha) * g_norm:
            return x_new, g_new_norm, alpha, True

        if g_new_norm < best_norm:
            best_norm, best_x, best_alpha = g_new_norm, x_new.copy(), alpha
        alpha *= 0.5
        if alpha < min_alpha:
            break

    # No sufficient decrease: return the best finite trial seen. If every
    # trial overflowed, return x unchanged with alpha=0 so the caller can
    # abort this Newton solve gracefully instead of stepping into overflow.
    if best_x is not None and np.isfinite(best_norm):
        return best_x, best_norm, best_alpha, False
    return x.copy(), g_norm, 0.0, False


# =============================================================================
# Newton solver
# =============================================================================


def newton_solve(
    x0: NDArray,
    residual_fn: Callable[[NDArray], NDArray],
    jacobian_fn: Callable[[NDArray], csc_array],
    config: Optional[NewtonConfig] = None,
    norm_fn: Optional[Callable[[NDArray], float]] = None,
    callback: Optional[Callable[[int, NDArray, float], None]] = None,
) -> Tuple[NDArray, NewtonResult]:
    """Generic Newton-Raphson solver with optional line search.

    Solves g(x) = 0 using Newton's method:
        x_{k+1} = x_k - J(x_k)^{-1} g(x_k)

    Args:
        x0: Initial guess (will be modified in-place)
        residual_fn: Function computing residual g(x)
        jacobian_fn: Function computing Jacobian J(x)
        config: Solver configuration (uses defaults if None)
        norm_fn: Custom norm function (defaults to L2 norm)
        callback: Optional callback(iteration, x, residual_norm)

    Returns:
        Tuple of (solution, NewtonResult)
    """
    if config is None:
        config = NewtonConfig()

    if norm_fn is None:

        def default_norm(g: NDArray) -> float:
            """Return the ord-norm of the flattened residual."""
            return np.linalg.norm(g.ravel(), ord=config.ord_norm)

        norm_fn = default_norm

    x = x0.ravel()
    x_shape = x0.shape

    g_norm_init = None
    g_norm = None
    converged = False
    num_iters = 0

    for k in range(config.max_iterations):
        # Compute residual and Jacobian
        g = residual_fn(x.reshape(x_shape))
        jac = jacobian_fn(x.reshape(x_shape))

        g_norm = norm_fn(g)
        if k == 0:
            g_norm_init = g_norm

        if callback is not None:
            callback(k, x.reshape(x_shape), g_norm)

        # Check convergence
        if g_norm < max(config.rtol * g_norm_init, config.atol):
            converged = True
            num_iters = k + 1
            break

        # Compute Newton step
        dx = -sla.spsolve(jac, g.ravel())

        # Apply step (with optional line search)
        if config.use_line_search:

            def resid_fn(xv):
                """Flattened-residual wrapper for the line search."""
                return residual_fn(xv.reshape(x_shape)).ravel()

            x_new, g_norm_new, alpha, ls_success = armijo_line_search(
                x,
                dx,
                g_norm,
                resid_fn,
                norm_fn,
                armijo_coeff=config.armijo_coeff,
                min_alpha=config.min_line_search_alpha,
            )
            x[:] = x_new

            if not ls_success and config.verbose > 0:
                warnings.warn(
                    f"Line search failed at iteration {k}: "
                    f"residual {g_norm_new:.2e} > {g_norm:.2e}",
                    RuntimeWarning,
                )
        else:
            x[:] = x + dx

        num_iters = k + 1

    # Final residual evaluation
    g = residual_fn(x.reshape(x_shape))
    g_norm = norm_fn(g)

    conv_factor = g_norm / g_norm_init if g_norm_init > 0 else 0.0

    result = NewtonResult(
        converged=converged,
        num_iterations=num_iters,
        residual_norm=g_norm,
        residual_norm_initial=g_norm_init,
        convergence_factor=conv_factor,
    )

    return x.reshape(x_shape), result


# =============================================================================
# Continuation methods
# =============================================================================


@dataclass
class ContinuationConfig:
    """Configuration for continuation solver."""

    parameter_max: float = 1.0
    step_init: float = 1e-2
    step_min: float = 1e-4
    step_increase: float = 1.5
    step_decrease: float = 0.5
    convergence_rate_threshold: float = 3.0
    verbose: int = 0


@dataclass
class ContinuationState:
    """State for continuation solver with predictor history."""

    parameter: float = 0.0
    step_size: float = 1e-2

    # Solution history for predictor
    x_prev: Optional[NDArray] = None
    x_prev_prev: Optional[NDArray] = None
    param_prev: Optional[float] = None
    param_prev_prev: Optional[float] = None

    def update_history(self, x: NDArray, param: float):
        """Update solution history after successful step."""
        self.x_prev_prev = self.x_prev
        self.param_prev_prev = self.param_prev
        self.x_prev = x.copy()
        self.param_prev = param

    def predict(self, x_current: NDArray, new_param: float) -> NDArray:
        """Compute predictor estimate for new parameter value.

        Uses secant (first-order) predictor if history available,
        otherwise uses zero-order predictor (previous solution).
        """
        if self.x_prev_prev is not None and self.param_prev_prev is not None:
            d_param = self.param_prev - self.param_prev_prev
            if abs(d_param) > 1e-9:
                step_ratio = (new_param - self.param_prev) / d_param
                x_pred = self.x_prev + (self.x_prev - self.x_prev_prev) * step_ratio
                return np.maximum(x_pred, 0)  # Ensure non-negative

        if self.x_prev is not None:
            return self.x_prev.copy()

        return x_current.copy()


def continuation_solve(
    x0: NDArray,
    solve_fn: Callable[[NDArray, float], Tuple[NDArray, NewtonResult]],
    config: Optional[ContinuationConfig] = None,
    state: Optional[ContinuationState] = None,
) -> Tuple[NDArray, bool, ContinuationState]:
    """Adaptive continuation solver for parameter-dependent problems.

    Solves a sequence of problems g(x; p) = 0 where p is a continuation
    parameter that is increased from 0 to parameter_max.

    Uses predictor-corrector scheme:
    - Predictor: Secant extrapolation from previous solutions
    - Corrector: Newton solve at new parameter value

    Step size is adapted based on Newton convergence:
    - Increase after successful convergence
    - Decrease after failure, then retry

    Args:
        x0: Initial solution guess
        solve_fn: Function solve_fn(x, param) -> (x_new, NewtonResult)
                  that solves g(x; param) = 0
        config: Continuation configuration
        state: Optional continuation state (for resuming)

    Returns:
        Tuple of (solution, converged, final_state)
    """
    if config is None:
        config = ContinuationConfig()

    if state is None:
        state = ContinuationState(
            parameter=0.0,
            step_size=min(config.step_init, config.parameter_max),
        )

    x = x0.copy()
    is_first_step = state.x_prev is None

    while True:
        # Apply predictor if we have history
        if not is_first_step:
            x = state.predict(x, state.parameter)

        # Corrector: Newton solve
        if config.verbose > 1:
            logger.info(
                "Attempting parameter = %.4f (step = %.4f)...",
                state.parameter,
                state.step_size,
            )

        x_new, result = solve_fn(x, state.parameter)

        is_converging = (
            result.convergence_factor < config.convergence_rate_threshold
            or result.converged
        )

        if result.converged and state.parameter >= config.parameter_max:
            # Fully converged at target parameter
            x = x_new
            break

        if is_converging:
            if config.verbose > 1:
                logger.info(
                    "  Converged in %d iterations, residual = %.2e",
                    result.num_iterations,
                    result.residual_norm,
                )

            # Update history and advance parameter
            if not is_first_step:
                state.update_history(x_new, state.parameter)
                if result.converged:
                    state.step_size *= config.step_increase

            x = x_new
            state.parameter = min(
                state.parameter + state.step_size, config.parameter_max
            )
            is_first_step = False

        else:
            if config.verbose > 1:
                logger.warning(
                    "  Failed after %d iterations. Reducing step size.",
                    result.num_iterations,
                )

            # Restore and reduce step
            if state.x_prev is not None:
                x = state.x_prev.copy()
                state.parameter = state.param_prev
            elif is_first_step:
                state.parameter = 0.0
                is_first_step = False

            state.step_size *= config.step_decrease

            if state.step_size < config.step_min:
                if config.verbose > 0:
                    warnings.warn(
                        f"Continuation failed: step size {state.step_size:.2e} "
                        f"< minimum {config.step_min:.2e}",
                        RuntimeWarning,
                    )
                return x, False, state

    if config.verbose > 1:
        logger.info("Continuation completed at parameter = %.4f", state.parameter)

    return x, True, state
