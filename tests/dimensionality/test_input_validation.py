"""The values a system instance is given are checked against the quantity each port declares."""

import typing

import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.errors import BiltInputDimensionalityError
from bilt.units import Dimension, Quantity, Unit
from bilt.units.dimensions import CURRENT, LENGTH, MASS, RESISTANCE, TIME, VOLTAGE
from bilt.units.units import ampere, meter, ohm, second

Current = typing.Annotated[Quantity, CURRENT]
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
class QuantityRig(metaclass=Interface):
    q: Quantity = field()
    doubled: Quantity = field()


@equation
def doubling_quantity(doubled: Quantity, q: Quantity) -> bool:
    return doubled == q * 2.0


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


def test_a_unit_bearing_port_refuses_a_bare_float() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'q'"):
        QuantityRig(q=2.0)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("unit", "dimensionality"), ((meter, LENGTH), (second, TIME)))
def test_a_unit_bearing_port_accepts_a_quantity_of_any_dimensionality(unit: Unit, dimensionality: Dimension) -> None:
    with Implementation(QuantityRig, [doubling_quantity]):
        result = QuantityRig(q=2.0 * unit).doubled

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(4.0)
    assert result.unit.dimension == dimensionality


def test_an_unconstrained_port_accepts_a_quantity() -> None:
    @system
    class FloatRig(metaclass=Interface):
        x: float = field()
        doubled: float = field()

    @equation
    def doubling_float(doubled: float, x: float) -> bool:
        return doubled == x * 2.0

    with Implementation(FloatRig, [doubling_float]):
        result = FloatRig(x=2.0 * meter).doubled  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(4.0)
    assert result.unit.dimension == LENGTH


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


def test_a_quantity_contradicting_an_inferred_dimensionality_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        scaled: Quantity = field()

    @equation
    def doubling(scaled: Quantity, v: Voltage) -> bool:
        return scaled == v * 2.0

    with Implementation(Rig, [doubling]), pytest.raises(BiltInputDimensionalityError, match="'scaled'") as raised:
        _ = Rig(scaled=4.0 * meter).v

    message = str(raised.value)
    assert str(VOLTAGE) in message
    assert str(LENGTH) in message
    assert "doubling" in message


def test_a_float_port_accepts_a_bare_float_whatever_dimensionality_the_implementation_gives_it() -> None:
    @system
    class Rig(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def doubling(y: float, x: Voltage) -> bool:
        return y == x * 2.0

    with Implementation(Rig, [doubling]):
        result = Rig(x=2.0).y

    assert result == pytest.approx(4.0)
