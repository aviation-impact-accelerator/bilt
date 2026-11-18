"""Circle-line intersection built from `Circle`, `Line` and `Point`.

An intersection is by definition a circle, a line and a point on both, so every
implementation of it has these as fields, and a request reads and writes them directly.
The implementation is the two facts about the point. There is one point, not one per
shape, so nothing needs connecting.
"""

from bilt_proposals import field

from bilt import Implementation, Interface, system
from examples.geometry._circle import Circle, circle, on_circle
from examples.geometry._line import Line, on_line
from examples.geometry._point import Point


@system
class CircleLineIntersection(metaclass=Interface):
    """A `point` on both `circle` and `line`."""

    circle: Circle = field()
    line: Line = field()
    point: Point = field()


circle_line_intersection = Implementation(
    CircleLineIntersection,
    [on_circle, on_line],
    internal_implementations={"circle": circle},
)
