import typing

import pytest

from bilt import transform
from bilt.errors import BiltDimensionalityError, BiltTransformDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import AREA, DIMENSIONLESS, ENERGY, FORCE, LENGTH, MASS, PRESSURE, TIME, VOLTAGE

Area = typing.Annotated[Quantity, AREA]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Energy = typing.Annotated[Quantity, ENERGY]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Pressure = typing.Annotated[Quantity, PRESSURE]
Time = typing.Annotated[Quantity, TIME]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Acceleration = typing.Annotated[Quantity, LENGTH / TIME**2]


def test_product_matching_the_return_annotation_decorates() -> None:
    @transform
    def work(f: Force, d: Length) -> Energy:
        return f * d


def test_quotient_matching_the_return_annotation_decorates() -> None:
    @transform
    def stress(f: Force, a: Area) -> Pressure:
        return f / a


def test_quotient_cancelling_to_dimensionless_decorates() -> None:
    @transform
    def ratio(a: Mass, b: Mass) -> Dimensionless:
        return a / b


def test_product_contradicting_the_return_annotation_raises() -> None:
    with pytest.raises(BiltTransformDimensionalityError, match="work") as raised:

        @transform
        def work(f: Force, d: Length) -> Pressure:
            return f * d

    message = str(raised.value)
    assert str(ENERGY) in message
    assert str(PRESSURE) in message


def test_quotient_contradicting_the_return_annotation_raises() -> None:
    with pytest.raises(BiltTransformDimensionalityError, match="speed") as raised:

        @transform
        def speed(d: Length, t: Time) -> Acceleration:
            return d / t

    message = str(raised.value)
    assert str(LENGTH / TIME) in message
    assert str(LENGTH / TIME**2) in message


def test_inhomogeneous_body_raises_chained_from_the_units_error() -> None:
    with pytest.raises(BiltTransformDimensionalityError) as raised:

        @transform
        def combine(a: Mass, b: Length) -> Mass:
            return a + b

    assert isinstance(raised.value, BiltDimensionalityError)
    assert isinstance(raised.value.__cause__, BiltDimensionalityError)


def test_inhomogeneous_body_raises_without_a_return_annotation() -> None:
    with pytest.raises(BiltTransformDimensionalityError, match="combine"):

        @transform
        def combine(a: Mass, b: Length):  # noqa: ANN202
            return a + b


def test_body_failing_on_the_probe_falls_back_to_its_annotation() -> None:
    @transform
    def singular(m: Mass) -> Voltage:
        return m / (m.value - 1.0)


def test_non_quantity_result_falls_back_to_its_annotation() -> None:
    @transform
    def magnitude(m: Mass) -> Voltage:
        return typing.cast("Quantity", m.value)


def test_parameter_without_a_dimensionality_is_not_probed() -> None:
    @transform
    def combine(a: Mass, b: Quantity) -> Mass:
        return a + b
