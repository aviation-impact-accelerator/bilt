"""Geometric primitives in the plane and the systems built from them.

Written against proposed APIs (`bilt_proposals`): importing this package raises
`ProposalNotImplementedError` until groups of ports and inequalities exist in Bilt.
"""

from examples.geometry._circle import Circle, circle
from examples.geometry._circle_line_intersection import CircleLineIntersection, circle_line_intersection
from examples.geometry._line import Line
from examples.geometry._point import Point

__all__ = [
    "Circle",
    "CircleLineIntersection",
    "Line",
    "Point",
    "circle",
    "circle_line_intersection",
]
