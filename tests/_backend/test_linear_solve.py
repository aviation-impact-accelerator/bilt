import ast
import typing

import pytest

from bilt._backend import BackendLinearSystem
from bilt._backend._python import PYTHON_BACKEND


def parse_to_ast(src: str) -> ast.FunctionDef:
    return typing.cast("ast.FunctionDef", ast.parse(src).body[0])


def constant(name: str, value: float):
    return PYTHON_BACKEND.transpile(
        parse_to_ast(
            f"def {name}() -> float:\n"
            f"    return {value}"
        )
    )


def test_linear_system_requires_at_least_one_unknown() -> None:
    with pytest.raises(
        ValueError,
        match="must contain at least one unknown",
    ):
        BackendLinearSystem(
            parameters=(),
            unknowns=(),
            coefficients=(),
            rhs=(),
        )


def test_linear_system_coefficient_matrix_must_have_correct_number_of_rows() -> None:
    with pytest.raises(
        ValueError,
        match="coefficient matrix must be square",
    ):
        BackendLinearSystem(
            parameters=(),
            unknowns=("x", "y"),
            coefficients=((constant("a", 1.0), constant("b", 0.0)),),
            rhs=(
                constant("p", 1.0),
                constant("q", 2.0),
            ),
        )


def test_linear_system_coefficient_matrix_must_have_correct_number_of_columns() -> None:
    with pytest.raises(
        ValueError,
        match="coefficient matrix must be square",
    ):
        BackendLinearSystem(
            parameters=(),
            unknowns=("x", "y"),
            coefficients=(
                (constant("a", 1.0),),
                (constant("b", 0.0), constant("c", 1.0)),
            ),
            rhs=(
                constant("p", 1.0),
                constant("q", 2.0),
            ),
        )


def test_linear_system_rhs_must_match_number_of_unknowns() -> None:
    with pytest.raises(
        ValueError,
        match="RHS must match the number of unknowns",
    ):
        BackendLinearSystem(
            parameters=(),
            unknowns=("x", "y"),
            coefficients=(
                (constant("a", 1.0), constant("b", 0.0)),
                (constant("c", 0.0), constant("d", 1.0)),
            ),
            rhs=(constant("p", 1.0),),
        )