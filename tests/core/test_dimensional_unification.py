import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system._dimensional_unification import unify_dimensionalities
from bilt.errors import BiltImplementationDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, FORCE, LENGTH, MASS, POWER, RESISTANCE, VOLTAGE

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


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


@equation
def doubling(scaled: Quantity, v: Quantity) -> bool:
    return scaled == v * 2.0


@equation
def halving(halved: Quantity, scaled: Quantity) -> bool:
    return halved == scaled / 2.0


def test_declared_dimensionalities_resolve() -> None:
    resolved = unify_dimensionalities(Rig, [ohms_law])

    assert resolved == {"v": VOLTAGE, "i": CURRENT, "r": RESISTANCE}


def test_an_equation_infers_an_unannotated_port_from_its_neighbours() -> None:
    assert unify_dimensionalities(Rig, [ohms_law, doubling])["scaled"] is VOLTAGE


def test_inference_chains_across_constituents() -> None:
    assert unify_dimensionalities(Rig, [halving, doubling, ohms_law])["halved"] is VOLTAGE


def test_a_transform_output_takes_its_return_annotation() -> None:
    @transform
    def power(v: Quantity, i: Quantity) -> typing.Annotated[Quantity, POWER]:
        return v * i

    assert unify_dimensionalities(Rig, [ohms_law, power])["power"] is POWER


def test_an_unannotated_transform_output_is_not_inferred_from_its_body() -> None:
    @transform
    def power(v: Quantity, i: Quantity) -> Quantity:
        return v * i

    assert "power" not in unify_dimensionalities(Rig, [ohms_law, power])


def test_ports_nothing_constrains_stay_unresolved() -> None:
    resolved = unify_dimensionalities(Rig, [doubling, halving])

    assert resolved == {}


def test_the_interface_declares_ports() -> None:
    @system
    class Declared(metaclass=Interface):
        scaled: Force = field()
        v: Quantity = field()

    assert unify_dimensionalities(Declared, [doubling])["v"] is FORCE


def test_a_declared_conflict_names_both_sources() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Voltage = field()
        i: Quantity = field()
        r: Quantity = field()

    @equation
    def weighing(v: Mass) -> bool:
        return v == 0

    with pytest.raises(BiltImplementationDimensionalityError, match="'v'") as raised:
        unify_dimensionalities(Declared, [ohms_law, weighing])

    message = str(raised.value)
    assert "voltage" in message
    assert "mass" in message
    assert "declared by Declared" in message
    assert "declared by weighing" in message


def test_an_inferred_conflict_names_the_port_it_was_inferred_from() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Quantity = field()
        i: Quantity = field()
        r: Quantity = field()
        scaled: Quantity = field()
        m: Mass = field()

    @equation
    def weighing(scaled: Quantity, m: Quantity) -> bool:
        return scaled == m * 2.0

    with pytest.raises(BiltImplementationDimensionalityError, match="'scaled'") as raised:
        unify_dimensionalities(Declared, [ohms_law, doubling, weighing])

    message = str(raised.value)
    assert "inferred by doubling from 'v'" in message
    assert "inferred by weighing from 'm'" in message


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
            unify_dimensionalities(Declared, constituents)
        messages.add(str(raised.value))

    assert len(messages) == 1


def test_an_equation_inconsistent_with_inferred_ports_raises() -> None:
    @system
    class Declared(metaclass=Interface):
        v: Quantity = field()
        i: Quantity = field()
        r: Quantity = field()
        scaled: Quantity = field()
        m: Mass = field()

    @equation
    def squaring(scaled: Quantity, m: Quantity) -> bool:
        return scaled**2 == m**2

    with pytest.raises(
        BiltImplementationDimensionalityError, match="makes equation 'squaring' dimensionally inconsistent"
    ) as raised:
        unify_dimensionalities(Declared, [ohms_law, doubling, squaring])

    assert "inferred by doubling from 'v'" in str(raised.value)


def test_an_equation_no_port_can_be_isolated_from_is_still_checked() -> None:
    @system
    class Declared(metaclass=Interface):
        a: Mass = field()
        b: Length = field()

    @equation
    def squares(a: Quantity, b: Quantity) -> bool:
        return a**2 == b**2

    with pytest.raises(BiltImplementationDimensionalityError, match="squares") as raised:
        unify_dimensionalities(Declared, [squares])

    message = str(raised.value)
    assert "declared by Declared" in message


def test_a_sub_system_declares_its_ports_under_their_dotted_names() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: Voltage = field()
        i: Quantity = field()
        r: Quantity = field()

    @system
    class Parent(metaclass=Interface):
        out: Quantity = field()

    resistor = SubSystemConstituent("resistor", Implementation(Resistor, []))

    assert unify_dimensionalities(Parent, [resistor]) == {"resistor.v": VOLTAGE}
