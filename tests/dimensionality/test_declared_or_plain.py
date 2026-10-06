"""A port either declares its dimensionality or stands outside the unit system, and nothing is inferred between them."""

import typing

import pytest

from bilt import Implementation, Interface, Tensor, equation, field, system, transform
from bilt.errors import (
    BiltEquationDimensionalityError,
    BiltImplementationDimensionalityError,
)
from bilt.units import UNITLESS, Quantity
from bilt.units.dimensions import DIMENSIONLESS, LENGTH, MASS, VOLTAGE
from bilt.units.units import volt

Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Voltage = typing.Annotated[Quantity, VOLTAGE]


def test_a_system_interface_may_mix_declared_and_plain_ports() -> None:
    @system
    class Rig(metaclass=Interface):
        m: Mass = field()
        count: float = field()

    @equation
    def weighing(m: Mass) -> bool:
        return m == 0

    @equation
    def counting(count: float) -> bool:
        return count == 0

    Implementation(Rig, [weighing, counting])


def test_a_port_the_interface_declares_and_a_constituent_leaves_plain_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        x: Length = field()
        y: Length = field()

    @equation
    def doubling(y: float, x: float) -> bool:
        return y == x * 2.0

    with pytest.raises(BiltImplementationDimensionalityError, match="in only some places") as raised:
        Implementation(Rig, [doubling])

    message = str(raised.value)
    assert f"{LENGTH}: declared by Rig" in message
    assert "left plain by doubling" in message


def test_a_port_the_interface_leaves_plain_and_a_constituent_declares_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def doubling(y: Length, x: Length) -> bool:
        return y == x * 2.0

    with pytest.raises(BiltImplementationDimensionalityError, match="in only some places") as raised:
        Implementation(Rig, [doubling])

    message = str(raised.value)
    assert f"{LENGTH}: declared by doubling" in message
    assert "left plain by Rig" in message


def test_a_transform_without_a_return_annotation_does_not_declare_its_output() -> None:
    @system
    class Rig(metaclass=Interface):
        x: float = field()
        half: Mass = field()

    @transform
    def half(x: float):  # noqa: ANN202
        return x / 2.0

    with pytest.raises(BiltImplementationDimensionalityError, match="'half' a dimensionality in only some places"):
        Implementation(Rig, [half])


def test_an_intermediate_port_declared_by_one_constituent_and_plain_in_another_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        x: Mass = field()

    @equation
    def declared_link(x: Mass, middle: Mass) -> bool:
        return x == middle

    @equation
    def plain_link(middle: float, other: float) -> bool:
        return middle == other

    with pytest.raises(
        BiltImplementationDimensionalityError, match="'middle' a dimensionality in only some places"
    ) as raised:
        Implementation(Rig, [declared_link, plain_link])

    message = str(raised.value)
    assert f"{MASS}: declared by declared_link" in message
    assert "left plain by plain_link" in message


def test_a_plain_port_answers_with_a_bare_number_however_many_declared_ports_exist_elsewhere() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        x: float = field()
        scaled: float = field()

    @equation
    def doubling(scaled: float, x: float) -> bool:
        return scaled == x * 2.0

    @equation
    def resting(v: Voltage) -> bool:
        return v == 0

    with Implementation(Rig, [doubling, resting]):
        result = Rig(v=1.0 * volt, x=2.0).scaled

    assert result == pytest.approx(4.0)
    assert not isinstance(result, Quantity)


def test_a_tensor_port_meeting_a_declared_port_in_one_equation_raises() -> None:
    def scaling(m: Mass, scales: Tensor) -> bool:
        return m == m * scales

    with pytest.raises(BiltEquationDimensionalityError, match=r"plain 'scales' with declared 'm'"):
        equation(scaling)


def test_a_transform_with_a_plain_flag_beside_declared_ports_evaluates_by_the_flag() -> None:
    @system
    class Combiner(metaclass=Interface):
        a: Dimensionless = field()
        b: Dimensionless = field()
        add: bool = field()
        combined: Dimensionless = field()

    @transform
    def combined(a: Dimensionless, b: Dimensionless, add: bool) -> Dimensionless:
        return a + b if add else a * b

    two, three = Quantity(2.0, UNITLESS), Quantity(3.0, UNITLESS)
    with Implementation(Combiner, [combined]):
        added = Combiner(a=two, b=three, add=True).combined
        multiplied = Combiner(a=two, b=three, add=False).combined

    assert added == Quantity(5.0, UNITLESS)
    assert multiplied == Quantity(6.0, UNITLESS)


def test_an_equation_with_a_plain_exponent_beside_declared_ports_evaluates() -> None:
    @system
    class Power(metaclass=Interface):
        x: Dimensionless = field()
        n: float = field()
        y: Dimensionless = field()

    @equation
    def powered(y: Dimensionless, x: Dimensionless, n: float) -> bool:
        return y == x**n

    with Implementation(Power, [powered]):
        result = Power(x=Quantity(3.0, UNITLESS), n=2.0).y

    assert result == Quantity(9.0, UNITLESS)


def test_a_plain_transform_port_that_another_owner_declares_raises() -> None:
    @system
    class Rig(metaclass=Interface):
        a: Dimensionless = field()
        scale: Dimensionless = field()
        scaled: Dimensionless = field()

    @transform
    def scaled(a: Dimensionless, scale: float) -> Dimensionless:
        return a * scale

    with pytest.raises(BiltImplementationDimensionalityError, match="'scale' a dimensionality in only some places"):
        Implementation(Rig, [scaled])


@system
class Amplifier(metaclass=Interface):
    gain: Dimensionless = field()
    x: Length = field()
    amplified: Length = field()


@equation
def amplifying(amplified: Length, gain: Dimensionless, x: Length) -> bool:
    return amplified == gain * x
