"""A declared tensor port reduces and indexes in an equation, and the reduction is checked at decoration."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, Tensor, dim, equation, field, system, tensor
from bilt.core._evaluation_context import EvaluationContext
from bilt.errors import BiltEquationDimensionalityError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.units import fraction, kelvin, kilogram
from bilt.units.dimensions import ABSOLUTE, DIMENSIONLESS, FORCE, MASS, TEMPERATURE

Mass = typing.Annotated[Quantity, MASS]
MassOverADomain = typing.Annotated[Tensor, MASS]
Force = typing.Annotated[Quantity, FORCE]
Ratio = typing.Annotated[Quantity, DIMENSIONLESS]
RatioOverADomain = typing.Annotated[Tensor, DIMENSIONLESS]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
AbsoluteTemperatureOverADomain = typing.Annotated[Tensor, TEMPERATURE, ABSOLUTE]
time = dim("time")


@system
class Scale(metaclass=Interface):
    m: MassOverADomain = field()
    reduced: Mass = field()


@system
class Thermometer(metaclass=Interface):
    t: AbsoluteTemperatureOverADomain = field()
    reduced: AbsoluteTemperature = field()


@pytest.mark.parametrize(
    ("reduction", "expected"),
    (("mean", 1.5), ("sum", 3.0), ("min", 1.0), ("max", 2.0)),
)
def test_a_reduction_of_a_declared_tensor_port_solves_in_the_coherent_unit(reduction: str, expected: float) -> None:
    @equation
    def reduce(reduced: Mass, m: MassOverADomain) -> bool:
        return reduced == getattr(m, reduction)(time)

    with Implementation(Scale, [reduce]):
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).reduced

    assert result == expected * kilogram


def test_a_positional_selection_of_a_declared_tensor_port_solves() -> None:
    @equation
    def first(reduced: Mass, m: MassOverADomain) -> bool:
        return reduced == m.isel(time, 0)

    with Implementation(Scale, [first]):
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).reduced

    assert result == 1.0 * kilogram


def test_a_coordinate_selection_of_a_declared_tensor_port_solves() -> None:
    @equation
    def at_2025(reduced: Mass, m: MassOverADomain) -> bool:
        return reduced == m.sel(time, 2025)

    with Implementation(Scale, [at_2025]), EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(time, (2024, 2025)))
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).reduced

    assert result == 2.0 * kilogram


def test_a_reduction_relating_unlike_dimensionalities_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="'total'"):

        @equation
        def total(reduced: Force, m: MassOverADomain) -> bool:
            return reduced == m.sum(time)


@pytest.mark.parametrize("selection", ("isel", "sel"))
def test_a_selection_relating_unlike_dimensionalities_raises(selection: str) -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="'selected'"):

        @equation
        def selected(reduced: Force, m: MassOverADomain) -> bool:
            return reduced == (m.isel(time, 0) if selection == "isel" else m.sel(time, 2025))


def test_a_product_along_an_axis_of_a_dimensional_port_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="along an axis"):

        @equation
        def product(reduced: Mass, m: MassOverADomain) -> bool:
            return reduced == m.prod(time)


def test_a_product_along_an_axis_of_a_dimensionless_port_is_dimensionless() -> None:
    @system
    class Ratios(metaclass=Interface):
        ratios: RatioOverADomain = field()
        combined: Ratio = field()

    @equation
    def combining(combined: Ratio, ratios: RatioOverADomain) -> bool:
        return combined == ratios.prod(time)

    with Implementation(Ratios, [combining]):
        result = Ratios(ratios=tensor([2.0, 3.0], over=time, unit=fraction)).combined

    assert result == Quantity(6.0, fraction)


def test_a_product_along_an_axis_of_a_dimensionless_port_against_a_dimensional_one_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="'combining'"):

        @equation
        def combining(mass: Mass, ratios: RatioOverADomain) -> bool:
            return mass == ratios.prod(time)


def test_a_sum_along_an_axis_of_an_absolute_port_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="adds absolute quantities along an axis"):

        @equation
        def total(reduced: AbsoluteTemperature, t: AbsoluteTemperatureOverADomain) -> bool:
            return reduced == t.sum(time)


def test_a_selection_of_an_absolute_port_keeps_its_absoluteness() -> None:
    @equation
    def first(reduced: AbsoluteTemperature, t: AbsoluteTemperatureOverADomain) -> bool:
        return reduced == t.isel(time, 0)

    with Implementation(Thermometer, [first]):
        result = Thermometer(t=tensor([273.15, 373.15], over=time, unit=abs(kelvin))).reduced

    assert result == Quantity(273.15, abs(kelvin))


def test_a_selection_of_an_absolute_port_related_to_a_relative_one_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="relates an absolute quantity to a relative one"):

        @equation
        def first(reduced: typing.Annotated[Quantity, TEMPERATURE], t: AbsoluteTemperatureOverADomain) -> bool:
            return reduced == t.isel(time, 0)


def test_a_mean_of_a_declared_tensor_port_in_a_sub_expression_solves() -> None:
    @equation
    def doubled_mean(reduced: Mass, m: MassOverADomain) -> bool:
        return reduced == m.mean(time) * 2.0

    with Implementation(Scale, [doubled_mean]):
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).reduced

    np.testing.assert_allclose(result.value, 3.0)
