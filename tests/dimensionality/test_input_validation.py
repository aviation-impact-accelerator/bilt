"""The values a system instance is given are checked against the quantity each port declares."""

import typing

import pytest

from bilt import PYTHON_BACKEND

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.errors import BiltInputDimensionalityError
from bilt.units import UNITLESS, Quantity
from bilt.units.dimensions import ABSOLUTE, CURRENT, DIMENSIONLESS, LENGTH, MASS, RESISTANCE, TEMPERATURE, VOLTAGE
from bilt.units.units import absolute_celsius, ampere, kelvin, kilogram, kilometer, meter, ohm

Current = typing.Annotated[Quantity, CURRENT]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Temperature = typing.Annotated[Quantity, TEMPERATURE]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
MassOverADomain = typing.Annotated[Tensor, MASS]
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


@system
class MassScale(metaclass=Interface):
    m: Mass = field()
    doubled: Mass = field()


@system
class TensorScale(metaclass=Interface):
    m: MassOverADomain = field()
    doubled: Mass = field()


@system
class PlainScale(metaclass=Interface):
    x: Tensor = field()
    doubled: Tensor = field()


@system
class Thermometer(metaclass=Interface):
    t: Temperature = field()
    same: Temperature = field()


@system
class AbsoluteThermometer(metaclass=Interface):
    t: AbsoluteTemperature = field()
    same: AbsoluteTemperature = field()


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


def test_a_tensor_with_no_unit_at_an_annotated_port_raises_and_names_the_label() -> None:
    masses = tensor([1.0, 2.0], over=dim("time"))
    with pytest.raises(BiltInputDimensionalityError, match="'m'") as raised:
        MassScale(m=masses)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    message = str(raised.value)
    assert "tensor(..., unit=...)" in message
    assert "[time]" in message
    assert "1.0" not in message


def test_a_tensor_is_checked_for_dimensionality() -> None:
    lengths = tensor([1.0, 2.0], over=dim("time"), unit=meter)
    with pytest.raises(BiltInputDimensionalityError, match="'m'") as raised:
        MassScale(m=lengths)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    message = str(raised.value)
    assert str(MASS) in message
    assert "meter" in message


def test_an_absolute_tensor_at_a_relative_port_raises() -> None:
    temperatures = tensor([1.0, 2.0], over=dim("time"), unit=absolute_celsius)
    with pytest.raises(BiltInputDimensionalityError, match="'t'"):
        Thermometer(t=temperatures)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_a_relative_tensor_at_an_absolute_port_raises() -> None:
    temperatures = tensor([1.0, 2.0], over=dim("time"), unit=kelvin)
    with pytest.raises(BiltInputDimensionalityError, match="'t'"):
        AbsoluteThermometer(t=temperatures)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_a_unit_carrying_tensor_at_a_plain_port_raises() -> None:
    masses = tensor([1.0, 2.0], over=dim("time"), unit=kilogram)
    with pytest.raises(BiltInputDimensionalityError, match="'x'") as raised:
        PlainScale(x=masses)

    message = str(raised.value)
    assert "plain port" in message
    assert "Tensor" in message


def test_a_scalar_quantity_at_a_tensor_annotated_port_raises() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'m'") as raised:
        TensorScale(m=3.0 * kilogram)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    message = str(raised.value)
    assert "varies over a domain" in message
    assert str(MASS) in message


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

def test_direct_evaluate_rejects_wrong_input_dimension() -> None:
    model = Implementation(Resistor, [ohms_law])

    with pytest.raises(BiltInputDimensionalityError, match="'i'"):
        model.evaluate(
            {"i": 2.0 * meter, "r": 3.0 * ohm},
            "v",
            PYTHON_BACKEND,
        )
