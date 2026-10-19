"""Direct evaluation enforces the same input contract as system construction."""

import typing

import pytest

from bilt import PYTHON_BACKEND, Implementation, Interface, Tensor, equation, field, system, transform
from bilt.errors import BiltInputDimensionalityError
from bilt.units import Quantity, absolute_celsius, ampere, kelvin, meter, ohm, second
from bilt.units.dimensions import ABSOLUTE, CURRENT, LENGTH, RESISTANCE, TEMPERATURE, VOLTAGE

Current = typing.Annotated[Quantity, CURRENT]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Length = typing.Annotated[Quantity, LENGTH]
LengthSeries = typing.Annotated[Tensor, LENGTH]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]


@system
class Resistor(metaclass=Interface):
    i: Current = field()
    r: Resistance = field()
    v: Voltage = field()


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


@pytest.mark.parametrize("current", (2.0, 2.0 * meter))
def test_direct_evaluate_rejects_invalid_current(current: object) -> None:
    model = Implementation(Resistor, [ohms_law])

    with pytest.raises(BiltInputDimensionalityError, match="'i'"):
        model.evaluate({"i": current, "r": 3.0 * ohm}, "v", PYTHON_BACKEND)


def test_direct_evaluate_accepts_valid_inputs() -> None:
    model = Implementation(Resistor, [ohms_law])

    result = model.evaluate({"i": 2.0 * ampere, "r": 3.0 * ohm}, "v", PYTHON_BACKEND)

    assert isinstance(result, Quantity)
    assert result.unit.dimension == VOLTAGE
    assert result.value == pytest.approx(6.0)


def test_direct_evaluate_validates_before_returning_an_existing_output() -> None:
    model = Implementation(Resistor, [ohms_law])

    with pytest.raises(BiltInputDimensionalityError, match="'i'"):
        model.evaluate({"i": 2.0 * meter}, "i", PYTHON_BACKEND)


@system
class TemperatureProbe(metaclass=Interface):
    reading: AbsoluteTemperature = field()


def test_direct_evaluate_rejects_a_relative_temperature_at_an_absolute_port() -> None:
    model = Implementation(TemperatureProbe, [])

    with pytest.raises(BiltInputDimensionalityError, match="'reading'"):
        model.evaluate({"reading": 20.0 * kelvin}, "reading", PYTHON_BACKEND)


def test_direct_evaluate_accepts_an_absolute_temperature() -> None:
    model = Implementation(TemperatureProbe, [])
    reading = Quantity(20.0, absolute_celsius)

    assert model.evaluate({"reading": reading}, "reading", PYTHON_BACKEND) is reading


@system
class Plain(metaclass=Interface):
    value: float = field()


def test_direct_evaluate_rejects_a_quantity_at_a_plain_port() -> None:
    model = Implementation(Plain, [])

    with pytest.raises(BiltInputDimensionalityError, match="'value'"):
        model.evaluate({"value": 2.0 * meter}, "value", PYTHON_BACKEND)


@system
class Stretch(metaclass=Interface):
    x: Length = field()
    y: Length = field()


@transform
def intermediate(x: Length) -> Length:
    return x * 2.0


@transform
def y(intermediate: Length) -> Length:
    return intermediate * 3.0


def test_direct_evaluate_checks_an_intermediate_input_declaration() -> None:
    model = Implementation(Stretch, [intermediate, y])

    with pytest.raises(BiltInputDimensionalityError, match="'intermediate'"):
        model.evaluate({"intermediate": 2.0 * second}, "y", PYTHON_BACKEND)


def test_direct_evaluate_accepts_a_declared_intermediate_input() -> None:
    model = Implementation(Stretch, [intermediate, y])

    assert model.evaluate({"intermediate": 2.0 * meter}, "y", PYTHON_BACKEND) == 6.0 * meter


def test_direct_evaluate_lifts_a_declared_intermediate_output() -> None:
    model = Implementation(Stretch, [intermediate, y])

    assert model.evaluate({"x": 2.0 * meter}, "intermediate", PYTHON_BACKEND) == 4.0 * meter


@system
class TensorStretch(metaclass=Interface):
    x: LengthSeries = field()
    y: LengthSeries = field()


@equation
def stretching(y: Length, x: Length) -> bool:
    return y == x * 2.0


def test_direct_evaluate_keeps_the_interfaces_tensor_requirement() -> None:
    model = Implementation(TensorStretch, [stretching])

    with pytest.raises(BiltInputDimensionalityError, match="'x'"):
        model.evaluate({"x": 2.0 * meter}, "y", PYTHON_BACKEND)
