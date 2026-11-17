"""Equations shared by every two-terminal component.

Sign convention (passive): every two-terminal component has a positive and a negative
terminal. `current` flows from the positive terminal, through the component, to the
negative terminal. `voltage` is the positive terminal potential minus the negative
terminal potential.

Equations bind by port name, so any component whose interface uses these port names
can include these equations in its implementation.
"""

from typing import Annotated

from bilt import Quantity, equation
from bilt.units.dimensions import VOLTAGE


@equation
def potential_difference(
    positive_potential: Annotated[Quantity, VOLTAGE],
    negative_potential: Annotated[Quantity, VOLTAGE],
    voltage: Annotated[Quantity, VOLTAGE],
) -> bool:
    """Voltage across a two-terminal component."""
    return voltage == positive_potential - negative_potential
