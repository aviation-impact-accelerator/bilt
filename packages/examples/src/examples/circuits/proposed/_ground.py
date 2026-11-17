"""Ground: fixes the potential of the junction it is connected to.

Its pin current is left free, which absorbs the one Kirchhoff current law that is
redundant in any circuit, so a circuit never has to leave one out by hand.
"""

from typing import Annotated

from bilt_proposals import field

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, system
from bilt.units.dimensions import VOLTAGE
from examples.circuits.proposed._pin import Pin


@system
class Ground(metaclass=Interface):
    """A single pin held at a reference potential."""

    terminal: Pin = field()
    potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()


@equation
def ground_reference(terminal: Pin, potential: Annotated[Quantity, VOLTAGE]) -> bool:
    """The pin is at the reference potential."""
    return terminal.potential == potential


ground = Implementation(Ground, [ground_reference])
