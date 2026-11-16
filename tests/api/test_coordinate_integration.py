"""Coordinate-aware trapezoidal integration through the public Bilt API."""

from typing import TYPE_CHECKING, Annotated

import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, QuantityTensor, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._domain_registry import DomainRegistry
from bilt.core._evaluation_context import EvaluationContext
from bilt.errors import BiltAnalysisError, BiltDomainBindingError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import tensor_symbol_expr
from bilt.units import kilogram, meter, second, teragram, watt, year
from bilt.units.dimensions import MASS

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


type MassTensor = Annotated[QuantityTensor, MASS]
type ForceResponse = Annotated[QuantityTensor, (watt / meter**2 / teragram).dimension]
type IntegratedResponse = Annotated[Quantity, ((watt / meter**2) * year / teragram).dimension]


@pytest.fixture(autouse=True)
def _select_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_nonuniform_trapezoid_matches_analytic_result(backend: Backend) -> None:
    elapsed = dim("elapsed")

    @system
    class Model(metaclass=Interface):
        response: Tensor = field()
        area: float = field()

    @equation
    def integrate(response: Tensor, area: float) -> bool:
        return area == response.integrate(elapsed)

    implementation = Implementation(Model, [integrate])
    coordinates = (0.0, 2.0, 5.0, 10.0)
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(elapsed, coordinates))
        with backend, implementation:
            actual = Model(response=tensor([0.0, 4.0, 10.0, 20.0], over=elapsed)).area

    # Integral of 2x over [0, 10].
    assert actual == pytest.approx(100.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_descending_coordinates_give_signed_integral(backend: Backend) -> None:
    elapsed = dim("elapsed")

    @system
    class Model(metaclass=Interface):
        response: Tensor = field()
        area: float = field()

    @equation
    def integrate(response: Tensor, area: float) -> bool:
        return area == response.integrate(elapsed)

    implementation = Implementation(Model, [integrate])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(elapsed, (10.0, 6.0, 3.0, 0.0)))
        with backend, implementation:
            result = Model(response=tensor([1.0, 1.0, 1.0, 1.0], over=elapsed)).area

    assert result == pytest.approx(-10.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_higher_rank_integration_preserves_other_named_dimensions(backend: Backend) -> None:
    region, elapsed = dim("region"), dim("elapsed")

    @system
    class Model(metaclass=Interface):
        response: Tensor = field()
        area: Tensor = field()

    @equation
    def integrate(response: Tensor, area: Tensor) -> bool:
        return area == response.integrate(elapsed)

    implementation = Implementation(Model, [integrate])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(elapsed, (0.0, 2.0, 5.0)))
        with backend, implementation:
            result = Model(
                response=tensor([[0.0, 2.0, 5.0], [0.0, 4.0, 10.0]], over=(region, elapsed))
            ).area

    assert result.domain == domain(region)
    np.testing.assert_allclose(result.values, [12.5, 25.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_cyof_gwp20_integrates_response_with_time_units(backend: Backend) -> None:
    elapsed = dim("carbon_dioxide_emissions_period")
    window = dim("gwp_window")
    forcing_unit = (watt / meter**2) / teragram
    integrated_unit = forcing_unit * year

    @system
    class Model(metaclass=Interface):
        response: ForceResponse = field()
        gwp: IntegratedResponse = field()

    @equation
    def integrate(response: ForceResponse, gwp: IntegratedResponse) -> bool:
        return gwp == response.sel(elapsed, slice(0, 20), into=window).integrate(
            window, coordinate_unit=year
        )

    implementation = Implementation(Model, [integrate])
    dates = tuple(range(0, 101))
    forcing = np.exp(-np.asarray(dates) / 36.54)
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(elapsed, dates))
        with backend, implementation:
            result = Model(response=tensor(forcing, over=elapsed, unit=forcing_unit)).gwp

    expected = np.trapezoid(forcing[:21], x=np.arange(21))
    assert isinstance(result, Quantity)
    assert result.convert_to(integrated_unit).value == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_integral_can_feed_further_equations(backend: Backend) -> None:
    elapsed = dim("elapsed")

    @system
    class Model(metaclass=Interface):
        signal: Tensor = field()
        area: float = field()
        double_area: float = field()

    @equation
    def integrate(signal: Tensor, area: float) -> bool:
        return area == signal.integrate(elapsed)

    @equation
    def double(area: float, double_area: float) -> bool:
        return double_area == 2 * area

    implementation = Implementation(Model, [integrate, double])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(elapsed, (0.0, 1.0, 3.0)))
        with backend, implementation:
            result = Model(signal=tensor([2.0, 2.0, 2.0], over=elapsed)).double_area

    assert result == pytest.approx(12.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_integral_resolves_new_weights_per_evaluation(backend: Backend) -> None:
    elapsed = dim("elapsed")

    @system
    class Model(metaclass=Interface):
        signal: Tensor = field()
        area: float = field()

    @equation
    def integrate(signal: Tensor, area: float) -> bool:
        return area == signal.integrate(elapsed)

    implementation = Implementation(Model, [integrate])
    results: list[float] = []
    for coordinates in ((0.0, 1.0, 2.0), (0.0, 3.0, 6.0)):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_axis(coordinate_axis(elapsed, coordinates))
            with backend, implementation:
                results.append(Model(signal=tensor([1.0, 1.0, 1.0], over=elapsed)).area)
    assert results == pytest.approx([2.0, 6.0])


@pytest.mark.parametrize(
    ("coordinates", "reason"),
    [
        ((0.0,), "at least two"),
        ((0.0, 0.0, 1.0), "strictly monotonic"),
        ((0.0, 2.0, 1.0), "strictly monotonic"),
        ((0.0, float("nan")), "finite numerical"),
        (("a", "b"), "finite numerical"),
    ],
)
def test_registry_rejects_invalid_coordinate_axes(coordinates: tuple[object, ...], reason: str) -> None:
    elapsed = dim("elapsed")
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(elapsed, coordinates))
    with pytest.raises(BiltDomainBindingError, match=reason):
        registry.integration_weights(elapsed, coordinate_unit=None)


def test_unlabelled_axis_cannot_be_integrated() -> None:
    elapsed = dim("elapsed")
    registry = DomainRegistry()
    registry.bind(tensor([1.0, 2.0], over=elapsed))
    with pytest.raises(BiltDomainBindingError, match="no coordinate axis"):
        registry.integration_weights(elapsed, coordinate_unit=None)


def test_coordinate_unit_is_relative_and_required_for_physical_time() -> None:
    from bilt.units import absolute_year

    elapsed = dim("elapsed")
    expression = tensor_symbol_expr("x")
    with pytest.raises(TypeError, match="must be a Unit"):
        expression.integrate(elapsed, coordinate_unit="year")
    with pytest.raises(ValueError, match="relative"):
        expression.integrate(elapsed, coordinate_unit=absolute_year)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_integration_rejects_missing_named_axis(backend: Backend) -> None:
    elapsed, region = dim("elapsed"), dim("region")

    @system
    class Model(metaclass=Interface):
        values: Tensor = field()
        area: float = field()

    @equation
    def integrate(values: Tensor, area: float) -> bool:
        return area == values.integrate(elapsed)

    implementation = Implementation(Model, [integrate])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="dimension"):
        _ = Model(values=tensor([1.0, 2.0], over=region)).area


def test_registry_converts_year_spacing_to_coherent_seconds() -> None:
    elapsed = dim("elapsed")
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(elapsed, (0.0, 2.0, 5.0)))
    weights = registry.integration_weights(elapsed, coordinate_unit=year)
    seconds_per_year = 365.25 * 24 * 60 * 60
    np.testing.assert_allclose(weights, [seconds_per_year, 2.5 * seconds_per_year, 1.5 * seconds_per_year])
