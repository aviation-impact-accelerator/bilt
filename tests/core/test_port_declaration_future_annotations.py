from __future__ import annotations

import typing

from bilt import Interface, equation, field, system, transform
from bilt.core._port_declaration import PortDeclaration
from bilt.units import Quantity  # noqa: TC001
from bilt.units.dimensions import MASS

MASS_DECLARATION = PortDeclaration(carries_quantity=True, dimensionality=MASS)


@system
class Child(metaclass=Interface):
    v: typing.Annotated[Quantity, MASS] = field()


def test_system_resolves_annotated_fields() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: typing.Annotated[Quantity, MASS] = field()
        limit: typing.ClassVar[float] = 1.0

    assert Rig.__bilt_port_declarations__ == {"mass": MASS_DECLARATION}


def test_equation_resolves_annotated_and_sub_system_parameters() -> None:
    @equation
    def link(child: Child, m: typing.Annotated[Quantity, MASS]) -> bool:
        return child.v == m

    assert link.__bilt_port_declarations__ == {"child.v": MASS_DECLARATION, "m": MASS_DECLARATION}


def test_transform_resolves_annotated_parameters_and_return() -> None:
    @transform
    def doubled(m: typing.Annotated[Quantity, MASS]) -> typing.Annotated[Quantity, MASS]:
        return m * 2.0

    assert doubled.__bilt_port_declarations__ == {"m": MASS_DECLARATION, "doubled": MASS_DECLARATION}
