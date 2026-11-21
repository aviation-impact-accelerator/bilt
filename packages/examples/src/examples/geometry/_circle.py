"""Circle in the plane, and the relation of a point lying on one."""

from typing import Annotated

from bilt_proposals import equation, field

from bilt import Implementation, Interface, Quantity, QuantityTensor, system
from bilt.units import meter
from bilt.units.dimensions import LENGTH
from examples.geometry._point import Point


@system
class Circle(metaclass=Interface):
    """The circle of `radius` about `centre`."""

    centre: Point = field()
    radius: Annotated[Quantity | QuantityTensor, LENGTH] = field()


@equation
def radius_is_nonnegative(radius: Annotated[Quantity, LENGTH]) -> bool:
    """A radius is never negative."""
    return radius >= 0.0 * meter


@equation
def point_is_on_circle(point: Point, circle: Circle) -> bool:
    """`point` lies on `circle`."""
    dx = point.x - circle.centre.x
    dy = point.y - circle.centre.y
    # A port read through a group is typed `Quantity | QuantityTensor`, as declared on
    # its interface, and `QuantityTensor` has no typed `**`.
    return dx**2 + dy**2 == circle.radius**2  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]


euclidean_circle = Implementation(Circle, [radius_is_nonnegative])
