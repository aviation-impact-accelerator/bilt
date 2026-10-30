"""Python scalar nonlinear execution safeguards, independent of the algorithm."""

import typing

import numpy as np

from bilt._backend._errors import BiltBackendError, BiltNonlinearSolveError
from bilt._backend._nonlinear_solve import NonlinearControls, NonlinearSolver
from bilt._backend._python._nonlinear_solvers import ScipyHybrid

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def _vector(value: object, name: str, size: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        message = f"{name} must be a real scalar vector"
        raise BiltNonlinearSolveError(message) from error
    if array.ndim != 1 or not array.size or (size is not None and array.shape != (size,)):
        msg = f"{name} must be a nonempty scalar vector of length {size}"
        raise BiltNonlinearSolveError(msg)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        msg = f"{name} must contain finite real values"
        raise BiltNonlinearSolveError(msg)
    try:
        with np.errstate(over="raise", invalid="raise"):
            return array.astype(np.float64)
    except ArithmeticError as error:
        message = f"{name} cannot be represented as finite float64 values"
        raise BiltNonlinearSolveError(message) from error


def solve_nonlinear(  # noqa: PLR0913 - explicit solver, scales and numerical controls
    residual: Callable[[tuple[float, ...]], object],
    initial_guess: tuple[float, ...],
    *,
    variable_scales: tuple[float, ...],
    residual_scales: tuple[float, ...],
    controls: NonlinearControls,
    solver: NonlinearSolver | None = None,
) -> tuple[float, ...]:
    """Apply common scaling, budgets and acceptance checks around one adapter."""
    selected = ScipyHybrid() if solver is None else solver
    if not isinstance(selected, NonlinearSolver) or selected.backend != "python":
        message = "The Python backend requires a nonlinear solver adapter for backend='python'."
        raise BiltBackendError(message)
    algorithm = typing.cast("NonlinearSolver[np.ndarray]", selected)
    guess = _vector(initial_guess, "initial guess")
    size = len(guess)
    x_scale = _vector(variable_scales, "variable scales", size)
    f_scale = _vector(residual_scales, "residual scales", size)
    if (x_scale <= 0).any() or (f_scale <= 0).any():
        msg = "variable and residual scales must be positive"
        raise BiltNonlinearSolveError(msg)
    evaluations = 0

    def scaled_residual(z: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        if evaluations >= controls.max_evaluations:
            msg = "evaluation budget exhausted"
            raise BiltNonlinearSolveError(msg, evaluations=evaluations)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                x = _vector(z * x_scale, "candidate unknowns", size)
                evaluations += 1
                value = residual(tuple(map(float, x)))
                return _vector(_vector(value, "residual", size) / f_scale, "scaled residual", size)
        except BiltNonlinearSolveError as error:
            raise BiltNonlinearSolveError(error.problem, evaluations=evaluations) from error
        except (ArithmeticError, ValueError) as error:
            msg = f"residual evaluation failed: {error}"
            raise BiltNonlinearSolveError(msg, evaluations=evaluations) from error

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled_guess = _vector(guess / x_scale, "scaled initial guess", size)
    except ArithmeticError as error:
        msg = "non-finite scaled initial guess"
        raise BiltNonlinearSolveError(msg, evaluations=evaluations) from error
    result = algorithm.solve(scaled_residual, scaled_guess, controls=controls)
    try:
        candidate = _vector(result.solution, "solver candidate", size)
    except BiltNonlinearSolveError as error:
        raise BiltNonlinearSolveError(error.problem, evaluations=evaluations) from error
    final_residual = scaled_residual(candidate)
    norm = float(np.max(np.abs(final_residual)))
    if not result.converged or norm > controls.residual_tolerance:
        msg = f"root finding failed with {algorithm.name}: {result.message}; scaled residual infinity norm {norm:g}"
        raise BiltNonlinearSolveError(
            msg,
            evaluations=evaluations,
            residual_norm=norm,
        )
    return tuple(map(float, candidate * x_scale))
