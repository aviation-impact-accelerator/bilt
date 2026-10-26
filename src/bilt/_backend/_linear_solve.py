"""Scalar linear-solve intrinsic and runtime validation shared by backends."""

import numpy as np

from bilt._backend._errors import BiltLinearSolveError

LINEAR_SOLVE_INTRINSIC_NAME = "__bilt_solve_linear__"
_MATRIX_RANK = 2


def validate_linear_shapes(matrix_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> None:
    """Require one square system with scalar entries, rather than batched systems."""
    if len(matrix_shape) != _MATRIX_RANK or matrix_shape[0] != matrix_shape[1] or rhs_shape != (matrix_shape[0],):
        problem = "expected a square matrix and vector of scalar coefficients"
        raise BiltLinearSolveError(problem)


def validate_linear_inputs(matrix: object, rhs: object) -> None:
    """Reject non-finite data before reporting a numerical solve failure."""
    if not np.isfinite(np.asarray(matrix)).all() or not np.isfinite(np.asarray(rhs)).all():
        problem = "coefficients and right-hand side must be finite"
        raise BiltLinearSolveError(problem)


def validate_linear_solution(matrix: object, rhs: object, solution: object) -> None:
    """Reject invalid inputs and failed solves, including JAX's singular NaNs."""
    validate_linear_inputs(matrix, rhs)
    if not np.isfinite(np.asarray(solution)).all():
        problem = "the matrix is singular or the solution is non-finite"
        raise BiltLinearSolveError(problem)


def numpy_solve_linear(matrix: object, rhs: object) -> np.ndarray:
    """Solve once with NumPy; do not form an explicit matrix inverse."""
    coefficients = np.asarray(matrix)
    values = np.asarray(rhs)
    validate_linear_shapes(coefficients.shape, values.shape)
    validate_linear_inputs(coefficients, values)
    dtype = np.result_type(coefficients, values, np.float64)
    try:
        solution = np.linalg.solve(coefficients.astype(dtype), values.astype(dtype))
    except np.linalg.LinAlgError as error:
        problem = "the matrix is singular"
        raise BiltLinearSolveError(problem) from error
    validate_linear_solution(coefficients, values, solution)
    return solution
