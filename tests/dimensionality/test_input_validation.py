"""The values a system instance is given are checked against the quantity each port declares."""

import typing

import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.errors import BiltInputDimensionalityError
from bilt.units import UNITLESS, Quantity
from bilt.units.dimensions import CURRENT, DIMENSIONLESS, LENGTH, MASS, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, kilometer, meter, ohm

Current = typing.Annotated[Quantity, CURRENT]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@system
class Resistor(metaclass=Interface):
    v: Voltage = field()
    i: Current = field()
    r: Resistance = field()


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


@system
class Amplifier(metaclass=Interface):
    gain: Dimensionless = field()
    x: Length = field()
    amplified: Length = field()


@equation
def amplifying(amplified: Length, gain: Dimensionless, x: Length) -> bool:
    return amplified == gain * x


def test_a_bare_float_at_an_annotated_port_raises() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'i'") as raised:
        Resistor(i=2.0, r=3.0 * ohm)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    assert str(CURRENT) in str(raised.value)


def test_a_quantity_of_the_wrong_dimensionality_raises() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'i'") as raised:
        Resistor(i=2.0 * meter, r=3.0 * ohm)

    message = str(raised.value)
    assert str(CURRENT) in message
    assert str(LENGTH) in message


def test_a_quantity_of_the_declared_dimensionality_evaluates() -> None:
    with Implementation(Resistor, [ohms_law]):
        result = Resistor(i=2.0 * ampere, r=3.0 * ohm).v

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(6.0)
    assert result.unit.dimension == VOLTAGE


def test_a_dimensionless_port_rejects_a_bare_float() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'gain'"):
        Amplifier(gain=2.0, x=3.0 * meter)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_a_dimensionless_port_accepts_a_dimensionless_quantity() -> None:
    with Implementation(Amplifier, [amplifying]):
        result = Amplifier(gain=Quantity(2.0, UNITLESS), x=3.0 * meter).amplified

    assert result == 6.0 * meter


def test_a_declared_port_accepts_a_quantity_in_any_unit_of_its_dimensionality() -> None:
    with Implementation(Amplifier, [amplifying]):
        result = Amplifier(gain=Quantity(2.0, UNITLESS), x=3.0 * kilometer).amplified

    assert result == 6000.0 * meter


def test_a_tensor_at_an_annotated_port_is_not_checked() -> None:
    @system
    class Scale(metaclass=Interface):
        m: Mass = field()
        doubled: Mass = field()

    @equation
    def doubling(doubled: Mass, m: Mass) -> bool:
        return doubled == m * 2.0

    masses = tensor([1.0, 2.0], over=dim("time"))
    with Implementation(Scale, [doubling]):
        result = Scale(m=masses).doubled  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    assert isinstance(result, Tensor)
    assert result.values.tolist() == [2.0, 4.0]


def test_a_quantity_of_the_wrong_dimensionality_at_a_port_only_a_constituent_reads_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        scaled: Voltage = field()

    @equation
    def doubling(scaled: Voltage, v: Voltage) -> bool:
        return scaled == v * 2.0

    with Implementation(Rig, [doubling]), pytest.raises(BiltInputDimensionalityError, match="'scaled'") as raised:
        _ = Rig(scaled=4.0 * meter).v

    message = str(raised.value)
    assert str(VOLTAGE) in message
    assert str(LENGTH) in message
