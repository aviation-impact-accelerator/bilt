"""Point in the plane."""

from typing import Annotated

from bilt import Interface, Quantity, QuantityTensor, field, system
from bilt.units.dimensions import LENGTH


@system
class Point(metaclass=Interface):
    """A point at Cartesian coordinates `(x, y)`."""

    x: Annotated[Quantity | QuantityTensor, LENGTH] = field()
    y: Annotated[Quantity | QuantityTensor, LENGTH] = field()
