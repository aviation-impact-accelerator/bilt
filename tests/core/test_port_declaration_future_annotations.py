from __future__ import annotations

import typing

from bilt import Interface, equation, field, system, transform
from bilt.units import Quantity
from bilt.units.dimensions import MASS

Mass = typing.Annotated[Quantity, MASS]


@system
class Child(metaclass=Interface):
    v: Mass = field()


def test_system_resolves_annotated_fields() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: Mass = field()
        limit: typing.ClassVar[float] = 1.0

    assert Rig.__bilt_port_declarations__ == {"mass": MASS}


def test_equation_resolves_annotated_and_sub_system_parameters() -> None:
    @equation
    def link(child: Child, m: Mass) -> bool:
        return child.v == m

    assert link.__bilt_port_declarations__ == {"child.v": MASS, "m": MASS}


def test_transform_resolves_annotated_parameters_and_return() -> None:
    @transform
    def doubled(m: Mass) -> Mass:
        return m * 2.0

    assert doubled.__bilt_port_declarations__ == {"m": MASS, "doubled": MASS}
