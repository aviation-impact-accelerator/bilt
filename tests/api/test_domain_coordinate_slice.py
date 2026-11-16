"""End-to-end coordinate slicing tests on both execution backends."""

import datetime as dt
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
    ("selected", "expected", "labels"),
    (
        (slice(2024, 2026), (2.0, 3.0, 4.0), (2024, 2025, 2026)),
        (slice(2025, 2023, -1), (3.0, 2.0, 1.0), (2025, 2024, 2023)),
        (slice(None, 2024), (1.0, 2.0), (2023, 2024)),
        (slice(2025, None), (3.0, 4.0), (2025, 2026)),
        (slice(None, None, -2), (4.0, 2.0), (2026, 2024)),
        (slice(2026, 2023, 2), (), ()),
    ),
)
def test_inclusive_coordinate_slices(
    backend: Backend,
    selected: slice,
    expected: tuple[float, ...],
    labels: tuple[int, ...],
) -> None:
    time, subset = dim("year"), dim("selected_year")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def select(values: Tensor, result: Tensor) -> bool:
        return result == values.sel(time, selected, into=subset)

    implementation = Implementation(Model, [select])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(time, (2023, 2024, 2025, 2026)))
        with backend, implementation:
            result = Model(values=tensor([1.0, 2.0, 3.0, 4.0], over=time)).result
        assert evaluation.domains.axis(subset).coordinates == labels

    assert result.domain == domain(subset)
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_coordinate_dates_and_reduction_over_sliced_axis(backend: Backend) -> None:
    day, period = dim("day"), dim("period")
    dates = (dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4))

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        selected: Tensor = field()
        total: float = field()

    @equation
    def select(values: Tensor, selected: Tensor) -> bool:
        return selected == values.sel(day, slice(dates[1], dates[2]), into=period)

    @equation
    def total(selected: Tensor, total: float) -> bool:
        return total == selected.sum(period)

    implementation = Implementation(Model, [select, total])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(day, dates))
        with backend, implementation:
            model = Model(values=tensor([2.0, 3.0, 5.0, 7.0], over=day))
            sliced = model.selected
            summed = model.total
        assert evaluation.domains.axis(period).coordinates == dates[1:3]

    np.testing.assert_array_equal(sliced.values, [3, 5])
    assert summed == pytest.approx(8.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_coordinate_slice_preserves_other_axes_and_units(backend: Backend) -> None:
    region, year, selected = dim("region"), dim("year"), dim("selected")
    values = tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], over=(region, year), unit=kilogram)

    @system
    class Model(metaclass=Interface):
        source: MassTensor = field()
        target: MassTensor = field()

    @equation
    def select(source: MassTensor, target: MassTensor) -> bool:
        return target == source.sel(year, slice(2024, 2025), into=selected)

    implementation = Implementation(Model, [select])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(year, (2023, 2024, 2025)))
        with backend, implementation:
            result = Model(source=values).target
    assert result.unit == kilogram
    assert result.domain == domain(region, selected)
    np.testing.assert_array_equal(result.values, [[2, 3], [20, 30]])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_scalar_sel_still_removes_dimension(backend: Backend) -> None:
    year = dim("year")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: float = field()

    @equation
    def choose(values: Tensor, result: float) -> bool:
        return result == values.sel(year, 2024)

    implementation = Implementation(Model, [choose])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(year, (2023, 2024, 2025)))
        with backend, implementation:
            result = Model(values=tensor([4.0, 6.0, 8.0], over=year)).result
    assert result == pytest.approx(6.0)


def test_coordinate_slice_rejects_invalid_api() -> None:
    year, subset = dim("year"), dim("subset")
    x = tensor_symbol_expr("x")
    with pytest.raises(TypeError, match="explicit 'into'"):
        x.sel(year, slice(2024, 2026))
    with pytest.raises(ValueError, match="distinct"):
        x.sel(year, slice(2024, 2026), into=year)
    with pytest.raises(ValueError, match="zero"):
        x.sel(year, slice(None, None, 0), into=subset)
    with pytest.raises(TypeError, match="integer"):
        x.sel(year, slice(None, None, 0.5), into=subset)
    with pytest.raises(TypeError, match="literal coordinate"):
        x.sel(year, slice(object(), None), into=subset)
    with pytest.raises(ValueError, match="only valid"):
        x.sel(year, 2024, into=subset)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_slice_missing_endpoint_fails_in_analysis(backend: Backend) -> None:
    year, subset = dim("year"), dim("subset")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def choose(values: Tensor, result: Tensor) -> bool:
        return result == values.sel(year, slice(2022, 2025), into=subset)

    implementation = Implementation(Model, [choose])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(year, (2023, 2024, 2025)))
        with backend, implementation, pytest.raises(BiltAnalysisError, match="not present"):
            _ = Model(values=tensor([4.0, 6.0, 8.0], over=year)).result


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_unlabelled_source_fails_in_analysis(backend: Backend) -> None:
    year, subset = dim("year"), dim("subset")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def choose(values: Tensor, result: Tensor) -> bool:
        return result == values.sel(year, slice(2024, 2025), into=subset)

    implementation = Implementation(Model, [choose])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="no coordinate axis"):
        _ = Model(values=tensor([4.0, 6.0, 8.0], over=year)).result


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_changed_coordinate_order_is_resolved_each_evaluation(backend: Backend) -> None:
    year, subset = dim("year"), dim("subset")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        selected: Tensor = field()

    @equation
    def choose(values: Tensor, selected: Tensor) -> bool:
        return selected == values.sel(year, slice(2024, 2025), into=subset)

    implementation = Implementation(Model, [choose])
    outcomes = []
    for labels in ((2023, 2024, 2025, 2026), (2025, 2024, 2023, 2026)):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_axis(coordinate_axis(year, labels))
            with backend, implementation:
                result = Model(values=tensor([1.0, 2.0, 3.0, 4.0], over=year)).selected
            outcomes.append(tuple(result.values.tolist()))

    assert outcomes == [(2.0, 3.0), ()]
