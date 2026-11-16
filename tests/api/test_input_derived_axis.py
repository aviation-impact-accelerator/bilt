"""Model input-dependent named axes with static backend specialisation."""

from typing import TYPE_CHECKING, Annotated

import numpy as np
import pytest

from bilt import Implementation, Interface, QuantityTensor, Tensor, dim, equation, field, system
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.errors import BiltAnalysisError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import domain
from bilt.units import Quantity, absolute_year
from bilt.units.dimensions import ABSOLUTE, TIME

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


type AbsoluteTimeTensor = Annotated[QuantityTensor, TIME, ABSOLUTE]
type AbsoluteTime = Annotated[Quantity, TIME, ABSOLUTE]


@pytest.fixture(autouse=True)
def _select_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_calendar_years_from_scalar_inputs(backend: Backend) -> None:
    modeling_date = dim("modeling_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=modeling_date, inclusive=True)

    implementation = Implementation(Model, [years])
    with backend, implementation:
        actual = Model(start=2025.0, end=2030.0).years

    assert actual.domain == domain(modeling_date)
    np.testing.assert_array_equal(actual.values, [2025, 2026, 2027, 2028, 2029, 2030])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_dynamic_axis_resolution_and_labelled_selection(backend: Backend) -> None:
    date = dim("date")
    selected = dim("selected_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()
        chosen: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date, inclusive=True)

    @equation
    def choose(years: Tensor, chosen: Tensor) -> bool:
        return chosen == years.sel(date, slice(2025, 2027), into=selected)

    implementation = Implementation(Model, [years, choose])
    with EvaluationContext() as evaluation:
        with backend, implementation:
            result = Model(start=2023.0, end=2028.0).chosen
        assert evaluation.domains.axis(date).coordinates == (2023, 2024, 2025, 2026, 2027, 2028)
        assert evaluation.domains.axis(selected).coordinates == (2025, 2026, 2027)

    assert result.domain == domain(selected)
    np.testing.assert_allclose(result.values, [2025, 2026, 2027])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_output_size_changes_on_repeated_evaluations(backend: Backend) -> None:
    date = dim("range_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date, inclusive=True)

    implementation = Implementation(Model, [years])
    with backend, implementation:
        short = Model(start=2025.0, end=2026.0).years
        long = Model(start=2025.0, end=2030.0).years

    assert short.shape == (2,)
    assert long.shape == (6,)
    np.testing.assert_allclose(long.values, np.arange(2025, 2031))


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_absolute_year_unit_preserved(backend: Backend) -> None:
    date = dim("absolute_date")

    @system
    class Model(metaclass=Interface):
        start: AbsoluteTime = field()
        end: AbsoluteTime = field()
        years: AbsoluteTimeTensor = field()

    @equation
    def years(start: AbsoluteTime, end: AbsoluteTime, years: AbsoluteTimeTensor) -> bool:
        return years == Tensor.arange(start, end, over=date, inclusive=True, unit=absolute_year)

    implementation = Implementation(Model, [years])
    with backend, implementation:
        result = Model(
            start=Quantity(2020.0, absolute_year),
            end=Quantity(2022.0, absolute_year),
        ).years

    assert result.domain == domain(date)
    assert isinstance(result, QuantityTensor)
    np.testing.assert_allclose(result.convert_to(absolute_year).values, [2020, 2021, 2022], rtol=1e-6)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_descending_and_stepped_range(backend: Backend) -> None:
    date = dim("descending_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date, step=-2, inclusive=True)

    implementation = Implementation(Model, [years])
    with backend, implementation:
        result = Model(start=2030.0, end=2024.0).years

    np.testing.assert_allclose(result.values, [2030, 2028, 2026, 2024])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_unresolved_intermediate_range_boundary_is_rejected(backend: Backend) -> None:
    date = dim("generated_date")

    @system
    class Model(metaclass=Interface):
        origin: float = field()
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def calculate_start(origin: float, start: float) -> bool:
        return start == origin + 1

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date)

    implementation = Implementation(Model, [calculate_start, years])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="directly supplied scalar"):
        _ = Model(origin=2024.0, end=2027.0).years


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
@pytest.mark.parametrize(("start", "end"), ((2025.5, 2030.0), (float("nan"), 2030.0)))
def test_invalid_boundary_values_rejected(backend: Backend, start: float, end: float) -> None:
    date = dim("invalid_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date)

    implementation = Implementation(Model, [years])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="integer"):
        _ = Model(start=start, end=end).years


def test_invalid_step_and_dimension_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Tensor.arange(2020, 2030, over=dim("date"), step=0)
    with pytest.raises(TypeError, match="named"):
        Tensor.arange(2020, 2030, over="date")  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_conflicting_axis_binding_rejected(backend: Backend) -> None:
    date = dim("conflicting_date")

    @system
    class Model(metaclass=Interface):
        start: float = field()
        end: float = field()
        years: Tensor = field()

    @equation
    def years(start: float, end: float, years: Tensor) -> bool:
        return years == Tensor.arange(start, end, over=date)

    implementation = Implementation(Model, [years])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(date, (2000, 2001)))
        with backend, implementation, pytest.raises(BiltAnalysisError, match="conflicting coordinate"):
            _ = Model(start=2020.0, end=2024.0).years
