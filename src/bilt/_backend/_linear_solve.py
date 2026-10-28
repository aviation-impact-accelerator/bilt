"""Linear-solve intrinsic and runtime validation shared by backends."""

import numpy as np

from bilt._backend._errors import BiltLinearSolveError
from bilt._backend._linear_assembly import LinearLayout, assemble_joint, assemble_pointwise, project_joint

LINEAR_SOLVE_INTRINSIC_NAME = "__bilt_solve_linear__"
LINEAR_BASIS_INTRINSIC_NAME = "__bilt_linear_basis__"
LINEAR_EXPAND_INTRINSIC_NAME = "__bilt_linear_expand__"
LINEAR_ZERO_INTRINSIC_NAME = "__bilt_linear_zero__"
_MATRIX_RANK = 2


def validate_linear_shapes(matrix_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> None:
    """Require square trailing matrix axes and matching batched vector axes."""
    if len(matrix_shape) < _MATRIX_RANK or matrix_shape[-2] != matrix_shape[-1] or rhs_shape != matrix_shape[:-1]:
        problem = "expected square trailing matrix axes and matching right-hand-side vectors"
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


def numpy_solve_linear(
    matrix: object,
    rhs: object,
    *,
    batch_rank: int | None = None,
    layout: LinearLayout | None = None,
    values: tuple[object, ...] = (),
) -> np.ndarray | tuple[object, ...]:
    """Solve once with NumPy; do not form an explicit matrix inverse."""
    if layout is not None:
        matrix, rhs = assemble_joint(matrix, rhs, values, layout, xp=np)
    elif batch_rank is not None:
        matrix, rhs = assemble_pointwise(matrix, rhs, batch_rank, xp=np)
    coefficients = np.asarray(matrix)
    rhs_values = np.asarray(rhs)
    validate_linear_shapes(coefficients.shape, rhs_values.shape)
    validate_linear_inputs(coefficients, rhs_values)
    dtype = np.result_type(coefficients, rhs_values, np.float64)
    try:
        solution = np.linalg.solve(coefficients.astype(dtype), rhs_values.astype(dtype)[..., None])[..., 0]
    except np.linalg.LinAlgError as error:
        problem = "the matrix is singular"
        raise BiltLinearSolveError(problem) from error
    validate_linear_solution(coefficients, rhs_values, solution)
    return solution if layout is None else project_joint(solution, values, layout, xp=np)
