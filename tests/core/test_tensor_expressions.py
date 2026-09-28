from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt import Implementation, Interface, field, system
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.math import mean
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, domain
from bilt.symbolic._expressions import Expr, TensorExpr, tensor_symbol_expr

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


def test_tensor_parameter_uses_tensor_expression() -> None:
    @equation
    def relation(
        value: Tensor,
        result: Tensor,
    ) -> bool:
        return result == value

    assert isinstance(
        relation.__bilt_equation_symbols__["value"],
        TensorExpr,
    )
    assert isinstance(
        relation.__bilt_equation_symbols__["result"],
        TensorExpr,
    )


def test_non_tensor_parameter_uses_plain_expression() -> None:
    @equation
    def relation(
        value: float,
        result: float,
    ) -> bool:
        return result == value

    assert type(relation.__bilt_equation_symbols__["value"]) is Expr
    assert type(relation.__bilt_equation_symbols__["result"]) is Expr


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_mean_reduces_named_dimension(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        averaged: Tensor = field()

    @equation
    def average(values: Tensor, averaged: Tensor) -> bool:
        return averaged == mean(values, time)

    implementation = Implementation(Demo, [average])
    demo = Demo(
        values=tensor(
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ],
            over=(location, time),
        ),
    )

    with backend, implementation:
        result = demo.averaged

    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        np.array([3.0, 4.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_mean_over_only_dimension_returns_scalar(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        averaged: float = field()

    @equation
    def average(values: Tensor, averaged: float) -> bool:
        return averaged == mean(values, time)

    implementation = Implementation(Demo, [average])
    demo = Demo(
        values=tensor(
            [1.0, 2.0, 3.0],
            over=time,
        ),
    )

    with backend, implementation:
        result = demo.averaged

    assert result == pytest.approx(2.0)


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_reduced_domain_propagates_to_later_equation(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        averaged: Tensor = field()
        offset: float = field()
        result: Tensor = field()

    @equation
    def average(values: Tensor, averaged: Tensor) -> bool:
        return averaged == mean(values, time)

    @equation
    def add_offset(
        averaged: Tensor,
        offset: float,
        result: Tensor,
    ) -> bool:
        return result == averaged + offset

    implementation = Implementation(
        Demo,
        [average, add_offset],
    )
    demo = Demo(
        values=tensor(
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ],
            over=(location, time),
        ),
        offset=10.0,
    )

    with backend, implementation:
        result = demo.result

    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        np.array([13.0, 14.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_mean_uses_aligned_canonical_axis(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        reference: Tensor = field()
        averaged: Tensor = field()

    @equation
    def average(
        values: Tensor,
        reference: Tensor,
        averaged: Tensor,
    ) -> bool:
        return averaged == mean(values, time) + 0 * reference

    implementation = Implementation(Demo, [average])
    demo = Demo(
        values=tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            over=(time, location),
        ),
        reference=tensor(
            [0.0, 0.0],
            over=location,
        ),
    )

    with backend, implementation:
        result = demo.averaged

    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        np.array([3.0, 4.0]),
    )


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_evaluation_cache_distinguishes_input_domains(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    location = Dim("location")
    time = Dim("time")
    canonical_order = domain(location, time)

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        offset: Tensor = field()
        result: Tensor = field()

    @equation
    def average_with_offset(
        values: Tensor,
        offset: Tensor,
        result: Tensor,
    ) -> bool:
        return result == mean(values, time) + offset

    implementation = Implementation(Demo, [average_with_offset])

    cases: list[tuple[dict[str, object], Domain, np.ndarray]] = [
        (
            {
                "values": tensor(
                    [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                    over=(location, time),
                ),
                "offset": tensor(10.0, over=()),
            },
            domain(location),
            np.array([13.0, 14.0]),
        ),
        (
            {
                "values": tensor([1.0, 3.0, 5.0], over=time),
                "offset": tensor(
                    [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
                    over=(location, time),
                ),
            },
            domain(location, time),
            np.array([[13.0, 13.0, 13.0], [23.0, 23.0, 23.0]]),
        ),
    ]

    for inputs, expected_domain, expected_values in cases:
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_all(inputs.values())
            with monkeypatch.context() as patch:
                patch.setattr(evaluation, "canonical_order", canonical_order)

                result = implementation.evaluate(inputs, "result", backend)

        assert isinstance(result, Tensor)
        assert result.domain == expected_domain
        np.testing.assert_allclose(result.values, expected_values)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_evaluation_cache_distinguishes_canonical_order(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        averaged: Tensor = field()

    @equation
    def average(values: Tensor, averaged: Tensor) -> bool:
        return averaged == mean(values, time)

    implementation = Implementation(Demo, [average])
    inputs: dict[str, object] = {
        "values": tensor(
            [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
            over=(location, time),
        ),
    }

    for canonical_order in (
        domain(location, time),
        domain(time, location),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_all(inputs.values())
            with monkeypatch.context() as patch:
                patch.setattr(evaluation, "canonical_order", canonical_order)
                assert evaluation.canonical_order == canonical_order

                result = implementation.evaluate(
                    inputs,
                    "averaged",
                    backend,
                )

        assert isinstance(result, Tensor)
        assert result.domain == domain(location)
        np.testing.assert_allclose(result.values, [3.0, 4.0])


def test_tensor_expression_arithmetic_returns_plain_expression() -> None:
    left = tensor_symbol_expr("left")
    right = tensor_symbol_expr("right")

    result = left + right

    assert type(result) is Expr
