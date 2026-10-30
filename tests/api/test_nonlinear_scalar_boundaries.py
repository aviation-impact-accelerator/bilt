"""Linear routing and scalar representations at nonlinear execution boundaries."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.errors import BiltAnalysisError, BiltNonlinearBlockError
from bilt.numerics import EvaluationOptions, RootOptions, VariableOptions


@system
class Pair(metaclass=Interface):
    x: float = field()
    y: float = field()
    p: float = field()
    q: float = field()
    mean: float = field()
    other: float = field()
    samples: Tensor = field()


@equation
def total(x: float, y: float, q: float) -> bool:
    return x + y == q


@equation
def product(x: float, y: float, p: float) -> bool:
    return x * y == p


def _options() -> EvaluationOptions:
    return EvaluationOptions(
        nonlinear=RootOptions(
            variables={"x": VariableOptions(initial=1.0, scale=1.0), "y": VariableOptions(initial=4.0, scale=1.0)},
            residual_scales={product: 6.0, total: 5.0},
        )
    )


@pytest.mark.parametrize("configured", (False, True))
def test_empty_root_options_preserve_expanded_linear_child_routes(*, configured: bool) -> None:
    @equation
    def child_total(x: float, y: float, p: float) -> bool:
        return x + y == p

    @equation
    def feedback(inner: Pair, q: float) -> bool:
        return inner.x - inner.y == q

    @equation
    def supply_p(p: float, inner: Pair) -> bool:
        return inner.p == p

    @equation
    def supply_q(q: float, inner: Pair) -> bool:
        return inner.q == q

    @equation
    def expose(x: float, inner: Pair) -> bool:
        return x == inner.x

    child = Implementation(
        Pair, [supply_p, expose, feedback], internal_implementations={"inner": Implementation(Pair, [child_total])}
    )
    if configured:
        child = child.with_options(EvaluationOptions(nonlinear=RootOptions(variables={}, residual_scales={})))
    parent = Implementation(Pair, [supply_p, supply_q, expose], internal_implementations={"inner": child})
    with parent:
        assert Pair(p=10.0, q=2.0).x == pytest.approx(6.0)


@pytest.mark.parametrize("via_linear", (False, True))
def test_scalar_reduction_and_aligned_linear_results_can_feed_nonlinear_blocks(*, via_linear: bool) -> None:
    axis = dim("samples")

    @equation
    def average(mean: float, samples: Tensor) -> bool:
        return mean == samples.mean(axis)

    @equation
    def supply(p: float, mean: float) -> bool:
        return p == mean

    @equation
    def linear_total(p: float, other: float, mean: float) -> bool:
        return p + other == mean

    @equation
    def linear_difference(p: float, other: float) -> bool:
        return p - other == 0

    operations = [average, product, total, *([linear_total, linear_difference] if via_linear else [supply])]
    model = Implementation(Pair, operations).with_options(_options())
    value = 12.0 if via_linear else 6.0
    with model:
        instance = Pair(samples=tensor([value, value], over=axis), q=5.0)
        assert instance.x == pytest.approx(2.0)
        assert instance.y == pytest.approx(3.0)


def test_a_one_element_tensor_domain_is_still_rejected() -> None:
    with (
        Implementation(Pair, [product, total]).with_options(_options()),
        pytest.raises(BiltAnalysisError, match="scalar input domains"),
    ):
        _ = Pair(p=typing.cast("float", tensor([6.0], over=dim("scenario"))), q=5.0).x


@pytest.mark.parametrize("shape", ((), (1,), (1, 1)))
def test_scalar_array_representations_are_normalized(shape: tuple[int, ...]) -> None:
    with Implementation(Pair, [product, total]).with_options(_options()):
        instance = Pair(p=typing.cast("float", np.full(shape, 6.0)), q=5.0)
        assert instance.x == pytest.approx(2.0)
        assert instance.y == pytest.approx(3.0)


@pytest.mark.parametrize(
    "value", (np.array([np.nan]), np.array([True]), np.array([1j]), np.array([6.0, 6.0]), np.array([]))
)
def test_invalid_scalar_array_values_are_still_rejected(value: np.ndarray) -> None:
    with (
        Implementation(Pair, [product, total]).with_options(_options()),
        pytest.raises(BiltNonlinearBlockError, match=r"scalar|one value"),
    ):
        _ = Pair(p=typing.cast("float", value), q=5.0).x
