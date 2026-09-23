from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt import Implementation, Interface, field, system
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.math import mean
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


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
