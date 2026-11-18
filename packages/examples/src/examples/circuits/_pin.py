"""Electrical pin: the group of ports a component shares with whatever it is connected to."""

from typing import Annotated

from bilt_proposals import field

from bilt import Interface, Quantity, QuantityTensor, system
from bilt.units.dimensions import CURRENT, VOLTAGE


@system
class Pin(metaclass=Interface):
    """Potential is shared at a junction; current, positive into the component, sums to zero."""

    potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field(flow=True)
