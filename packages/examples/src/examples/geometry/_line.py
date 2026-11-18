"""Straight line in the plane, and the relation of a point lying on one."""

from typing import Annotated

from bilt_proposals import equation, field

from bilt import Interface, Quantity, QuantityTensor, system
from bilt.units.dimensions import DIMENSIONLESS, LENGTH
from examples.geometry._point import Point


@system
class Line(metaclass=Interface):
    """The line `y = gradient * x + intercept`.

    `intercept` is where the line crosses the y-axis. Vertical lines have no gradient,
    so cannot be described this way. A line has no relations of its own, so, like
    `Point`, it has no implementation.
    """

    gradient: Annotated[Quantity | QuantityTensor, DIMENSIONLESS] = field()
    intercept: Annotated[Quantity | QuantityTensor, LENGTH] = field()


@equation
def on_line(point: Point, line: Line) -> bool:
    """`point` lies on `line`."""
    return point.y == line.gradient * point.x + line.intercept
