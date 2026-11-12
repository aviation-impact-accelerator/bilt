"""Tensor conditionals use masked selection internally."""

# ruff: noqa: PLR2004, SIM108

lazy from contextlib import AbstractContextManager

import numpy as np
import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
)
from bilt.symbolic._atoms import MASK_AND, MASK_NOT, MASK_OR, WHERE
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr, conditional_expr


@system
class Masked(metaclass=Interface):
    x: Tensor = field()
    output: Tensor = field()


@equation
def absolute_value(x: float, output: float) -> bool:
    if x > 0:
        selected = x
    else:
        selected = -x
    return output == selected


@equation
def masked_range(x: float, output: float) -> bool:
    if (x > 0) & (x < 5):
        selected = x
    else:
        selected = 0.0
    return output == selected


@system
class Choice(metaclass=Interface):
    select: float = field()
    a: Tensor = field()
    b: Tensor = field()
    output: Tensor = field()


@equation
def select_tensor(select: float, a: float, b: float, output: float) -> bool:
    if select > 0:
        selected = a
    else:
        selected = b
    return output == selected


@system
class Broadcast(metaclass=Interface):
    condition: Tensor = field()
    a: Tensor = field()
    b: Tensor = field()
    output: Tensor = field()


@equation
def broadcast_mask(condition: float, a: float, b: float, output: float) -> bool:
    if condition > 0:
        selected = a
    else:
        selected = b
    return output == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_elementwise_conditional_on_both_backends(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Masked, [absolute_value])
    with backend, implementation:
        result = Masked(x=tensor([-2.0, 3.0, -4.0], over=dim("location"))).output
    assert isinstance(result, Tensor)
    assert result.domain == domain(dim("location"))
    np.testing.assert_allclose(result.values, [2.0, 3.0, 4.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_compound_mask_predicate(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Masked, [masked_range])
    with backend, implementation:
        result = Masked(x=tensor([-2.0, 3.0, 6.0], over=dim("location"))).output
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [0.0, 3.0, 0.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_scalar_predicate_selects_complete_tensor(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Choice, [select_tensor])
    location = dim("location")
    with backend, implementation:
        result = Choice(
            select=1.0,
            a=tensor([1.0, 2.0], over=location),
            b=tensor([5.0, 6.0], over=location),
        ).output
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [1.0, 2.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_mask_broadcasts_by_named_domain(backend: AbstractContextManager[object]) -> None:
    location = dim("location")
    time = dim("time")
    implementation = Implementation(Broadcast, [broadcast_mask])
    with backend, implementation:
        result = Broadcast(
            condition=tensor([1.0, -1.0], over=time),
            a=tensor([10.0, 20.0], over=time),
            b=tensor([1.0, 2.0, 3.0], over=location),
        ).output
    assert isinstance(result, Tensor)
    assert set(result.domain.dims) == {time, location}
    expected = np.where(
        np.array([1.0, -1.0])[:, None] > 0,
        np.array([10.0, 20.0])[:, None],
        np.array([1.0, 2.0, 3.0])[None, :],
    )
    if result.domain.dims == (location, time):
        expected = expected.T
    np.testing.assert_allclose(result.values, expected)


def test_mask_composition_is_symbolic() -> None:
    x = Expr.symbol("x")
    positive = x > 0
    small = x < 5
    combined = (positive & small) | ~positive
    assert combined.head is MASK_OR
    assert combined.args[0].head is MASK_AND
    assert combined.args[1].head is MASK_NOT
    assert conditional_expr(combined, x, -x).tree.args[0] is combined


def test_where_is_an_internal_lowered_operation() -> None:
    x = Expr.symbol("x")
    lowered = conditional_expr(x > 0, x, -x)
    assert lowered.tree.head is not WHERE
