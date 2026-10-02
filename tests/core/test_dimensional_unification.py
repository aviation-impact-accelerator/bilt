import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system._dimensional_unification import check_dimensional_consistency
from bilt.errors import BiltImplementationDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import AREA, CURRENT, FORCE, LENGTH, MASS, POWER, RESISTANCE, VOLTAGE

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent

Area = typing.Annotated[Quantity, AREA]
Current = typing.Annotated[Quantity, CURRENT]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@system
class Rig(metaclass=Interface):
    v: Quantity = field()
    i: Quantity = field()
    r: Quantity = field()
    scaled: Quantity = field()
    halved: Quantity = field()
    power: Quantity = field()
    m: Mass = field()


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


@equation
def doubling(scaled: Quantity, v: Quantity) -> bool:
    return scaled == v * 2.0


@equation
def halving(halved: Quantity, scaled: Quantity) -> bool:
    return halved == scaled / 2.0


@equation
def weighing(scaled: Quantity, m: Quantity) -> bool:
    return scaled == m


@equation
def metering(power: Quantity, m: Quantity) -> bool:
    return power == m


@equation
def squared(a: Quantity, scaled: Quantity) -> bool:
    return a == scaled**2


def _inconsistency(system: Interface, constituents: collections.abc.Iterable[BiltModelConstituent]) -> str:
    """Return the message of the error that checking `constituents` against `system` raises."""
    with pytest.raises(BiltImplementationDimensionalityError) as raised:
        check_dimensional_consistency(system, constituents)
    return str(raised.value)


def test_consistent_constituents_pass() -> None:
    check_dimensional_consistency(Rig, [ohms_law, doubling, halving])


def test_an_equation_infers_an_unannotated_port_from_its_neighbours() -> None:
    message = _inconsistency(Rig, [ohms_law, doubling, weighing])
    assert "makes equation 'weighing' dimensionally inconsistent" in message
    assert f"'scaled': {VOLTAGE}: inferred by doubling from 'v'" in message
    assert f"'m': {MASS}: declared by Rig" in message


def test_inference_chains_across_constituents() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Quantity = field()
        i: Quantity = field()
        r: Quantity = field()
        scaled: Quantity = field()
        halved: Mass = field()

    message = _inconsistency(Declared, [halving, doubling, ohms_law])
    assert "makes equation 'halving' dimensionally inconsistent" in message
    assert f"'scaled': {VOLTAGE}: inferred by doubling from 'v'" in message
    assert f"'halved': {MASS}: declared by Declared" in message


def test_a_port_under_an_integer_power_is_inferred() -> None:
    @system
    class Declared(metaclass=Interface):
        a: Area = field()
        scaled: Quantity = field()
        m: Mass = field()

    message = _inconsistency(Declared, [squared, weighing])
    assert f"'scaled': {LENGTH}: inferred by squared from 'a'" in message


def test_a_power_no_dimensionality_can_meet_is_inconsistent() -> None:
    @system
    class Declared(metaclass=Interface):
        a: Length = field()
        scaled: Quantity = field()

    message = _inconsistency(Declared, [squared])
    assert "makes equation 'squared' dimensionally inconsistent" in message
    assert f"'a': {LENGTH}: declared by Declared" in message


def test_a_port_on_both_sides_is_inferred() -> None:
    @equation
    def accumulating(scaled: Quantity, v: Quantity) -> bool:
        return scaled + v == 2.0 * scaled

    message = _inconsistency(Rig, [ohms_law, accumulating, weighing])
    assert f"'scaled': {VOLTAGE}: inferred by accumulating from 'v'" in message


def test_an_inequality_infers_like_an_equality() -> None:
    @equation
    def bounding(scaled: Quantity, v: Quantity) -> bool:
        return scaled < v

    message = _inconsistency(Rig, [ohms_law, bounding, weighing])
    assert f"'scaled': {VOLTAGE}: inferred by bounding from 'v'" in message


def test_an_inference_from_no_port_names_only_its_equation() -> None:
    @equation
    def unity(scaled: Quantity, i: Quantity) -> bool:  # noqa: ARG001
        return scaled == 1

    message = _inconsistency(Rig, [ohms_law, unity, weighing])
    assert "makes equation 'weighing' dimensionally inconsistent" in message
    assert "inferred by unity\n" in message


def test_a_transform_output_takes_its_return_annotation() -> None:
    @transform
    def power(v: Quantity, i: Quantity) -> typing.Annotated[Quantity, POWER]:
        return v * i

    message = _inconsistency(Rig, [ohms_law, power, metering])
    assert "makes equation 'metering' dimensionally inconsistent" in message
    assert f"'power': {POWER}: declared by power" in message


def test_an_unannotated_transform_output_is_not_inferred_from_its_body() -> None:
    @transform
    def power(v: Quantity, i: Quantity) -> Quantity:
        return v * i

    check_dimensional_consistency(Rig, [ohms_law, power, metering])


def test_ports_nothing_declares_take_whatever_an_equation_implies() -> None:
    check_dimensional_consistency(Rig, [doubling, halving, weighing])


def test_the_interface_declares_ports() -> None:
    @system
    class Declared(metaclass=Interface):
        scaled: Force = field()
        v: Quantity = field()
        i: Quantity = field()
        r: Quantity = field()

    message = _inconsistency(Declared, [ohms_law, doubling])
    assert "makes equation 'doubling' dimensionally inconsistent" in message
    assert f"'scaled': {FORCE}: declared by Declared" in message


def test_a_declared_conflict_names_both_sources() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Voltage = field()
        i: Quantity = field()
        r: Quantity = field()

    @equation
    def resting(v: Mass) -> bool:
        return v == 0

    message = _inconsistency(Declared, [ohms_law, resting])
    assert "'v'" in message
    assert f"{VOLTAGE}: declared by Declared" in message
    assert f"{MASS}: declared by resting" in message


def test_a_conflict_reads_the_same_whatever_the_constituent_order() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Quantity = field()
        i: Quantity = field()
        r: Quantity = field()
        scaled: Force = field()
        halved: Quantity = field()

    messages = set()
    for constituents in ([ohms_law, doubling, halving], [halving, doubling, ohms_law]):
        with pytest.raises(BiltImplementationDimensionalityError) as raised:
            check_dimensional_consistency(Declared, constituents)
        messages.add(str(raised.value))

    assert len(messages) == 1


def test_an_equation_contradicting_declared_ports_is_inconsistent() -> None:
    @system
    class Declared(metaclass=Interface):
        a: Mass = field()
        b: Length = field()

    @equation
    def squares(a: Quantity, b: Quantity) -> bool:
        return a**2 == b**2

    message = _inconsistency(Declared, [squares])
    assert "squares" in message
    assert "declared by Declared" in message


def test_a_sub_system_declares_its_ports_under_their_dotted_names() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: Voltage = field()
        i: Quantity = field()
        r: Quantity = field()

    @system
    class Parent(metaclass=Interface):
        out: Mass = field()

    resistor = SubSystemConstituent("resistor", Implementation(Resistor, []))

    @equation
    def wiring(out: Quantity, resistor: Resistor) -> bool:
        return out == resistor.v

    message = _inconsistency(Parent, [resistor, wiring])
    assert "makes equation 'wiring' dimensionally inconsistent" in message
    assert f"'resistor.v': {VOLTAGE}: declared by resistor" in message
