import operator
import typing

import numpy as np
import pytest

from bilt import Tensor, dim, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.core._to_ast import tree_to_functiondef
from bilt.errors import BiltTensorDomainError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import Domain, domain
from bilt.symbolic._expressions import Expr, TensorExpr, tensor_symbol_expr

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend

TIME = dim("time")
BRANCH = dim("branch")


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


def _evaluate(expression: Expr, values: dict[str, Tensor], backend: Backend) -> tuple[np.ndarray, Domain]:
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        lowered = lower_expression(
            expression,
            symbol_domains={name: value.domain for name, value in values.items()},
            canonical_order=evaluation.canonical_order,
        )
        compiled = backend.transpile(
            tree_to_functiondef(lowered.tree, name="result", parameter_order=tuple(sorted(values)))
        )
        raw_inputs: dict[str, object] = dict(values)
        inputs = backend.apply_alignment(lower_inputs(raw_inputs), generate_alignment_plan(raw_inputs))
        return np.asarray(compiled.evaluate(inputs)), lowered.domain


@pytest.mark.parametrize(
    "build",
    (
        lambda value: value + 1,
        lambda value: 1 + value,
        lambda value: value - 1,
        lambda value: 1 - value,
        lambda value: value * 2,
        lambda value: 2 * value,
        lambda value: value / 2,
        lambda value: 2 / value,
        lambda value: value**2,
        lambda value: 2**value,
        lambda value: -value,
        lambda value: +value,
    ),
    ids=("add", "radd", "sub", "rsub", "mul", "rmul", "div", "rdiv", "pow", "rpow", "neg", "pos"),
)
def test_arithmetic_results_can_be_reduced(
    backend: Backend, build: collections.abc.Callable[[typing.Any], typing.Any]
) -> None:
    values = np.arange(1.0, 7.0).reshape(2, 3)
    expression = build(tensor_symbol_expr("value")).sum(BRANCH)
    result, result_domain = _evaluate(expression, {"value": tensor(values, over=(BRANCH, TIME))}, backend)
    np.testing.assert_allclose(result, np.sum(build(values), axis=0, keepdims=True), rtol=1e-5)
    assert result_domain == domain(TIME)


@pytest.mark.parametrize("reduction", ("sum", "mean", "min", "max", "prod"))
def test_reductions_can_be_chained_after_arithmetic(backend: Backend, reduction: str) -> None:
    values = np.arange(1.0, 7.0).reshape(2, 3)
    expression = getattr(tensor_symbol_expr("value") + 1, reduction)(BRANCH).mean(TIME)
    result, result_domain = _evaluate(expression, {"value": tensor(values, over=(BRANCH, TIME))}, backend)
    expected = getattr(np, reduction)(values + 1, axis=0, keepdims=True).mean(axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result_domain == Domain.scalar()


@pytest.mark.parametrize("selection", ("isel", "sel"))
def test_composed_selections_can_be_reduced(backend: Backend, selection: str) -> None:
    values = np.arange(1.0, 7.0).reshape(2, 3)
    expression = getattr(tensor_symbol_expr("value") + 1, selection)(
        BRANCH, 1 if selection == "isel" else "reference"
    ).sum(TIME)
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(BRANCH, ("first", "reference")))
        result, result_domain = _evaluate(expression, {"value": tensor(values, over=(BRANCH, TIME))}, backend)
    np.testing.assert_allclose(result, np.sum(values[1] + 1), rtol=1e-5)
    assert result_domain == Domain.scalar()


def test_arithmetic_after_reduction_can_be_selected(backend: Backend) -> None:
    values = np.arange(1.0, 7.0).reshape(2, 3)
    offset = np.array([10.0, 20.0, 30.0])
    expression = (tensor_symbol_expr("value").sum(BRANCH) + Expr.symbol("offset")).isel(TIME, 1)
    result, result_domain = _evaluate(
        expression,
        {"value": tensor(values, over=(BRANCH, TIME)), "offset": tensor(offset, over=TIME)},
        backend,
    )
    np.testing.assert_allclose(result, np.sum(values, axis=0)[1] + offset[1], rtol=1e-5)
    assert result_domain == Domain.scalar()


@pytest.mark.parametrize(
    "build",
    (
        lambda value: value.sum(BRANCH).mean(BRANCH),
        lambda value: value.sum(BRANCH).isel(BRANCH, 0),
        lambda value: (value.sum(BRANCH) + 1).sum(BRANCH),
    ),
    ids=("reduce-again", "select-removed-axis", "arithmetic-after-reduction"),
)
def test_removed_dimensions_are_rejected_during_lowering(
    build: collections.abc.Callable[[TensorExpr], TensorExpr],
) -> None:
    expression = build(tensor_symbol_expr("value"))
    with pytest.raises(BiltTensorDomainError, match="not present"):
        lower_expression(
            expression, symbol_domains={"value": domain(BRANCH, TIME)}, canonical_order=domain(BRANCH, TIME)
        )


def test_scalar_reduction_is_rejected_during_lowering() -> None:
    expression = tensor_symbol_expr("value").sum(TIME)
    with pytest.raises(BiltTensorDomainError, match="not present"):
        lower_expression(expression, symbol_domains={"value": Domain.scalar()}, canonical_order=domain(TIME))


@pytest.mark.parametrize("operation", (operator.add, operator.sub, operator.mul, operator.truediv, operator.pow))
@pytest.mark.parametrize("plain_first", (False, True), ids=("tensor-left", "tensor-right"))
def test_mixed_plain_and_tensor_arithmetic_keeps_tensor_operations(
    backend: Backend,
    operation: collections.abc.Callable[[typing.Any, typing.Any], typing.Any],
    plain_first: bool,
) -> None:
    plain = Expr.integer(2)
    value = tensor_symbol_expr("value")
    data = np.arange(1.0, 7.0).reshape(2, 3)
    expression = operation(plain, value) if plain_first else operation(value, plain)
    assert isinstance(expression, TensorExpr)
    result, result_domain = _evaluate(expression.sum(BRANCH), {"value": tensor(data, over=(BRANCH, TIME))}, backend)
    expected = operation(2, data) if plain_first else operation(data, 2)
    np.testing.assert_allclose(result, np.sum(expected, axis=0, keepdims=True), rtol=1e-5)
    assert result_domain == domain(TIME)


@pytest.mark.parametrize("method", ("sum", "mean", "min", "max", "prod", "isel", "sel"))
def test_plain_expressions_do_not_expose_tensor_operations(method: str) -> None:
    expression = Expr.symbol("value") + 2
    assert type(expression) is Expr
    assert not hasattr(expression, method)
