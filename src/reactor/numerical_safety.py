"""Guards that convert low-level numerical failures into retryable solver errors."""

import warnings

import numpy as np

class RecoverableNumericalError(RuntimeError):
    """Numerical failure that the outer solver can treat as a retryable step rejection."""

    def __init__(self, operation, original_exception, context=None):
        """Store the failed operation, original exception, and context dict, and build the combined message."""
        self.operation = operation
        self.original_exception = original_exception
        self.context = context or {}
        message = f"{operation}: {original_exception}"
        if self.context:
            context_str = ", ".join(f"{key}={value}" for key, value in self.context.items())
            message = f"{message} ({context_str})"
        super().__init__(message)


def finite_minmax_context(name, values, scale=1.0):
    """Return a small min/max context payload for diagnostics."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {f"{name}_state": "nonfinite"}
    return {
        f"{name}_min": float(np.min(finite) * scale),
        f"{name}_max": float(np.max(finite) * scale),
    }


def require_positive(name, values, *, context=None):
    """Raise a retryable error if a field became non-finite or non-positive."""
    array = np.asarray(values, dtype=float)
    payload = dict(context or {})
    payload.update(finite_minmax_context(name, array))
    if not np.all(np.isfinite(array)):
        raise RecoverableNumericalError(
            f"validating {name}",
            ValueError(f"{name} contains non-finite values"),
            context=payload,
        )
    if np.any(array <= 0.0):
        raise RecoverableNumericalError(
            f"validating {name}",
            ValueError(f"{name} must stay positive"),
            context=payload,
        )


def guarded_compute(operation, compute_fn, **context):
    """Run fragile numerical code with local error-state handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                return compute_fn()
    except (FloatingPointError, RuntimeWarning, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise RecoverableNumericalError(operation, exc, context=context) from exc
