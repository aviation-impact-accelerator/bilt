"""Thin SciPy adapters; scaling, acceptance and budget enforcement live outside."""

import importlib
import typing

import numpy as np

from bilt._backend._errors import BiltBackendError
from bilt._backend._nonlinear_solve import NonlinearResult

if typing.TYPE_CHECKING:
    import types
    from collections.abc import Callable

    from bilt._backend._nonlinear_solve import NonlinearControls


def _scipy() -> types.ModuleType:
    try:
        return importlib.import_module("scipy.optimize")
    except ImportError as error:
        message = "SciPy nonlinear solvers require the bilt[nonlinear] extra."
        raise BiltBackendError(message) from error


class ScipyHybrid:
    """Powell hybrid root finding with a finite-difference Jacobian."""

    name = "scipy-hybr"
    backend = "python"

    def solve(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
        *,
        controls: NonlinearControls,
    ) -> NonlinearResult[np.ndarray]:
        result = _scipy().root(
            residual,
            initial_guess,
            method="hybr",
            jac=None,
            options={"xtol": controls.step_tolerance, "maxfev": controls.max_evaluations},
        )
        return NonlinearResult(result.x, bool(result.success), str(result.message))


class ScipyLeastSquares:
    """Trust-region reflective least squares with a finite-difference Jacobian.

    A least-squares minimum need not be a root. Bilt still requires every
    scaled residual to satisfy the common root acceptance criterion.
    """

    name = "scipy-trf"
    backend = "python"

    def solve(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
        *,
        controls: NonlinearControls,
    ) -> NonlinearResult[np.ndarray]:
        if controls.step_tolerance <= np.finfo(float).eps:
            message = "SciPy trust-region least squares requires step_tolerance above float64 machine epsilon."
            raise BiltBackendError(message)
        result = _scipy().least_squares(
            residual,
            initial_guess,
            method="trf",
            jac="2-point",
            ftol=None,
            xtol=controls.step_tolerance,
            gtol=controls.step_tolerance,
            max_nfev=controls.max_evaluations,
        )
        return NonlinearResult(result.x, bool(result.success), str(result.message))
