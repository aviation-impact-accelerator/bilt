from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import Dim, coordinate_axis
from bilt.symbolic._domains import Domain, domain

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.mark.parametrize(
    ("method_name", "expected"),
    (
        ("mean", np.array([3.0, 4.0])),
        ("sum", np.array([9.0, 12.0])),
        ("min", np.array([1.0, 2.0])),
        ("max", np.array([5.0, 6.0])),
        ("prod", np.array([15.0, 48.0])),
    ),
)
@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_reduction_reduces_named_dimension(
    backend: Backend,
    method_name: str,
    expected: np.ndarray,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        reduced: Tensor = field()

    @equation
    def reduce(values: Tensor, reduced: Tensor) -> bool:
        reduction = getattr(values, method_name)
        return reduced == reduction(time)

    implementation = Implementation(Demo, [reduce])
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
        result = demo.reduced

    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        expected,
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
        return averaged == values.mean(time)

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
        return averaged == values.mean(time)

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
        return averaged == values.mean(time) + 0 * reference

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
        return result == values.mean(time) + offset

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
        return averaged == values.mean(time)

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


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_transform_mean_reduces_named_dimension(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        averaged: Tensor = field()

    @transform
    def averaged(values: Tensor) -> Tensor:
        return values.mean(time)

    implementation = Implementation(Demo, [averaged])
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
def test_transform_mean_inside_arithmetic(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        offset: Tensor = field()
        result: Tensor = field()

    @transform
    def result(
        values: Tensor,
        offset: Tensor,
    ) -> Tensor:
        return values.mean(time) + offset

    implementation = Implementation(Demo, [result])
    demo = Demo(
        values=tensor(
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ],
            over=(location, time),
        ),
        offset=tensor(10.0, over=()),
    )

    with backend, implementation:
        evaluated = demo.result

    assert evaluated.domain == domain(location)
    np.testing.assert_allclose(
        evaluated.values,
        np.array([13.0, 14.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_sel_over_only_dimension_returns_scalar(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        selected: float = field()

    @equation
    def select(values: Tensor, selected: float) -> bool:
        return selected == values.sel(time, 2025)

    implementation = Implementation(Demo, [select])
    inputs: dict[str, object] = {
        "values": tensor(
            [10.0, 20.0, 30.0],
            over=time,
        ),
    }

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(
            coordinate_axis(
                time,
                (2024, 2025, 2026),
            )
        )

        result = implementation.evaluate(
            inputs,
            "selected",
            backend,
        )

    assert result == pytest.approx(20.0)


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_sel_cache_respects_coordinate_order(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        selected: float = field()

    @equation
    def select(values: Tensor, selected: float) -> bool:
        return selected == values.sel(time, 2025)

    implementation = Implementation(Demo, [select])
    inputs: dict[str, object] = {
        "values": tensor([10.0, 20.0, 30.0], over=time),
    }

    # Reuse one implementation so both evaluations share its compilation cache.
    for coordinates, expected in (
        ((2024, 2025, 2026), 20.0),
        ((2025, 2024, 2026), 10.0),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind(coordinate_axis(time, coordinates))

            result = implementation.evaluate(
                inputs,
                "selected",
                backend,
            )

        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_isel_over_only_dimension_returns_scalar(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        selected: float = field()

    @equation
    def select(values: Tensor, selected: float) -> bool:
        return selected == values.isel(time, 1)

    implementation = Implementation(Demo, [select])
    demo = Demo(
        values=tensor(
            [1.0, 3.0, 5.0],
            over=time,
        ),
    )

    with backend, implementation:
        result = demo.selected

    assert result == pytest.approx(3.0)


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_isel_uses_aligned_canonical_axis(
    backend: Backend,
) -> None:
    location = Dim("location")
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        reference: Tensor = field()
        selected: Tensor = field()

    @equation
    def select(
        values: Tensor,
        reference: Tensor,
        selected: Tensor,
    ) -> bool:
        return selected == values.isel(time, 1) + 0 * reference

    implementation = Implementation(Demo, [select])
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
        result = demo.selected

    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        np.array([3.0, 4.0]),
    )
