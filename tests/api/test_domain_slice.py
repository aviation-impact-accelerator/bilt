"""End-to-end regressions for symbolic positional tensor slicing."""

from typing import TYPE_CHECKING, Annotated

import numpy as np
import pytest

from bilt import Implementation, Interface, QuantityTensor, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.errors import BiltAnalysisError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import tensor_symbol_expr
from bilt.units import kilogram
from bilt.units.dimensions import MASS

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


type MassTensor = Annotated[QuantityTensor, MASS]


@pytest.fixture(autouse=True)
def _select_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
@pytest.mark.parametrize(
    ("selector", "expected"),
    (
        (slice(1, 4), [1.0, 2.0, 3.0]),
        (slice(None, None, -1), [4.0, 3.0, 2.0, 1.0, 0.0]),
        (slice(0, None, 2), [0.0, 2.0, 4.0]),
        (slice(-3, None), [2.0, 3.0, 4.0]),
        (slice(20, None), []),
    ),
)
def test_isel_slicing_matches_python_semantics(backend: Backend, selector: slice, expected: list[float]) -> None:
    time = dim("slice_source")
    subset = dim("slice_output")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def calculate(values: Tensor, result: Tensor) -> bool:
        return result == values.isel(time, selector, into=subset)

    implementation = Implementation(Model, [calculate])
    with backend, implementation:
        result = Model(values=tensor([0.0, 1.0, 2.0, 3.0, 4.0], over=time)).result

    assert result.domain == domain(subset)
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_preserves_other_dimensions_and_can_feed_reduction(backend: Backend) -> None:
    location, time, subset = dim("location"), dim("time"), dim("selected_time")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        selected: Tensor = field()
        total: Tensor = field()

    @equation
    def select(values: Tensor, selected: Tensor) -> bool:
        return selected == values.isel(time, slice(1, None, 2), into=subset)

    @equation
    def aggregate(selected: Tensor, total: Tensor) -> bool:
        return total == selected.sum(subset)

    original = tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], over=(location, time))
    implementation = Implementation(Model, [select, aggregate])
    with backend, implementation:
        model = Model(values=original)
        selected = model.selected
        total = model.total

    assert selected.domain == domain(location, subset)
    np.testing.assert_allclose(selected.values, [[20.0, 40.0], [2.0, 4.0]])
    assert total.domain == domain(location)
    np.testing.assert_allclose(total.values, [60.0, 6.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_propagates_coordinates_in_reverse_order(backend: Backend) -> None:
    time, subset = dim("year"), dim("reversed_year")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        selected: Tensor = field()

    @equation
    def select(values: Tensor, selected: Tensor) -> bool:
        return selected == values.isel(time, slice(None, None, -1), into=subset)

    implementation = Implementation(Model, [select])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(time, (2023, 2024, 2025)))
        with backend, implementation:
            selected = Model(values=tensor([1.0, 2.0, 3.0], over=time)).selected
        assert evaluation.domains.axis(subset).coordinates == (2025, 2024, 2023)

    np.testing.assert_allclose(selected.values, [3.0, 2.0, 1.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_length_inferred_again_for_each_evaluation(backend: Backend) -> None:
    time, subset = dim("dynamic_time"), dim("dynamic_subset")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        selected: Tensor = field()

    @equation
    def select(values: Tensor, selected: Tensor) -> bool:
        return selected == values.isel(time, slice(1, None), into=subset)

    implementation = Implementation(Model, [select])
    with backend, implementation:
        first = Model(values=tensor([1.0, 2.0, 3.0], over=time)).selected
        second = Model(values=tensor([1.0, 2.0, 3.0, 4.0, 5.0], over=time)).selected

    assert first.shape == (2,)
    assert second.shape == (4,)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_preserves_quantity_units(backend: Backend) -> None:
    time, subset = dim("mass_time"), dim("mass_subset")

    @system
    class Model(metaclass=Interface):
        values: MassTensor = field()
        selected: MassTensor = field()

    @equation
    def select(values: MassTensor, selected: MassTensor) -> bool:
        return selected == values.isel(time, slice(1, 3), into=subset)

    implementation = Implementation(Model, [select])
    with backend, implementation:
        result = Model(values=tensor([1.0, 2.0, 3.0, 4.0], over=time, unit=kilogram)).selected

    assert result.unit == kilogram
    assert result.domain == domain(subset)
    np.testing.assert_allclose(result.values, [2.0, 3.0])


def test_slice_validates_public_symbolic_api() -> None:
    source, target = dim("source"), dim("target")
    expression = tensor_symbol_expr("values")

    with pytest.raises(TypeError, match="explicit 'into'"):
        expression.isel(source, slice(0, 3))
    with pytest.raises(ValueError, match="distinct"):
        expression.isel(source, slice(0, 3), into=source)
    with pytest.raises(TypeError, match="literal integer"):
        expression.isel(source, slice(0.5, 3), into=target)
    with pytest.raises(ValueError, match="zero"):
        expression.isel(source, slice(0, 3, 0), into=target)
    with pytest.raises(ValueError, match="only valid"):
        expression.isel(source, 1, into=target)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_rejects_destination_already_in_receiver_domain(backend: Backend) -> None:
    time, scenario = dim("time"), dim("scenario")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def select(values: Tensor, result: Tensor) -> bool:
        return result == values.isel(time, slice(1, 3), into=scenario)

    implementation = Implementation(Model, [select])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="already exists"):
        _ = Model(values=tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], over=(scenario, time))).result
