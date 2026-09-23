"""Dimensional analysis for `@transform`, performed by probing the body at decoration."""

import typing

import pytest

from bilt import transform
from bilt.errors import BiltUnitsError
from bilt.units import Quantity
from bilt.units.dimensions import FORCE, LENGTH, MASS, TIME, VOLTAGE

Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Acceleration = typing.Annotated[Quantity, LENGTH / TIME**2]


def test_consistent_transform_decorates() -> None:
    @transform
    def force(m: Mass, a: Acceleration) -> Force:
        return m * a


def test_partially_annotated_transform_is_not_probed() -> None:
    @transform
    def half_annotated(m: Mass, a: float) -> Force:
        return m + m * a


def test_unprobeable_body_decorates() -> None:
    @transform
    def clamp(m: Mass) -> Mass:
        if m > Quantity(1.0, m.unit):
            return m
        return m * 2.0


@pytest.mark.xfail(reason="the transform body is not probed at decoration yet")
def test_probe_contradicting_the_return_annotation_raises() -> None:
    with pytest.raises(BiltUnitsError, match="mislabelled") as raised:

        @transform
        def mislabelled(m: Mass, a: Acceleration) -> Voltage:
            return m * a

    message = str(raised.value)
    assert "force" in message
    assert "voltage" in message


@pytest.mark.xfail(reason="an inhomogeneous transform body is not rejected at decoration yet")
def test_inhomogeneous_transform_body_raises() -> None:
    with pytest.raises(BiltUnitsError, match="combine") as raised:

        @transform
        def combine(a: Mass, b: Length) -> Mass:
            return a + b

    message = str(raised.value)
    assert "mass" in message
    assert "length" in message
