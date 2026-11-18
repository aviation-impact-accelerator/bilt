"""Point in the plane: the group of ports a primitive shares with whatever it is connected to."""

from typing import Annotated

from bilt import Interface, Quantity, QuantityTensor, field, system
from bilt.units.dimensions import LENGTH


@system
class Point(metaclass=Interface):
    """Cartesian coordinates, both shared at a junction (neither is a flow)."""

    x: Annotated[Quantity | QuantityTensor, LENGTH] = field()
    y: Annotated[Quantity | QuantityTensor, LENGTH] = field()
