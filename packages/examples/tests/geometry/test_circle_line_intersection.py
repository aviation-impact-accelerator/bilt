"""Circle-line intersection: choosing between the two points where a line crosses a circle.

Expected values are exact. The circle about (1, 2) with radius 5 and the line
y = -2x + 14 meet where (x - 1)^2 + (12 - 2x)^2 = 25, that is 5x^2 - 50x + 120 = 0, or
(x - 4)(x - 6) = 0. So the intersections are (4, 6) and (6, 2).

Through the point (4, 6) with gradient -2 passes the line with intercept 14, and about
(1, 2) the circle with radius 5 (or -5, which `radius_is_nonnegative` excludes).

`examples.geometry` uses proposed APIs, so importing it raises
`ProposalNotImplementedError`. Each test imports it locally, so that the error marks
that test as an expected failure instead of stopping collection.
"""

import pytest
from bilt_proposals import ContinuousUnknown, ProposalNotImplementedError

from bilt.units import UNITLESS, meter


@pytest.mark.xfail(raises=ProposalNotImplementedError, reason="groups of ports and inequalities are proposed")
def test_model_bound_selects_the_radius() -> None:
    from examples.geometry import Circle, CircleLineIntersection, Line, Point, euclidean_circle_line_intersection  # noqa: PLC0415

    intersection = CircleLineIntersection(
        circle=Circle(centre=Point(x=1.0 * meter, y=2.0 * meter)),
        line=Line(gradient=-2.0 * UNITLESS),
        point=Point(x=4.0 * meter, y=6.0 * meter),
    )

    with euclidean_circle_line_intersection:
        assert intersection.circle.radius.convert_to(meter).value == pytest.approx(5.0)
        assert intersection.line.intercept.convert_to(meter).value == pytest.approx(14.0)


@pytest.mark.xfail(raises=ProposalNotImplementedError, reason="groups of ports and ContinuousUnknown are proposed")
def test_bound_selects_an_intersection() -> None:
    from examples.geometry import Circle, CircleLineIntersection, Line, Point, euclidean_circle_line_intersection  # noqa: PLC0415

    intersection = CircleLineIntersection(
        circle=Circle(centre=Point(x=1.0 * meter, y=2.0 * meter), radius=5.0 * meter),
        line=Line(gradient=-2.0 * UNITLESS, intercept=14.0 * meter),
        point=Point(x=ContinuousUnknown(lower=5.0 * meter)),
    )

    with euclidean_circle_line_intersection:
        assert intersection.point.x.convert_to(meter).value == pytest.approx(6.0)
        assert intersection.point.y.convert_to(meter).value == pytest.approx(2.0)


@pytest.mark.xfail(raises=ProposalNotImplementedError, reason="groups of ports and ContinuousUnknown are proposed")
def test_guess_selects_an_intersection() -> None:
    from examples.geometry import Circle, CircleLineIntersection, Line, Point, euclidean_circle_line_intersection  # noqa: PLC0415

    intersection = CircleLineIntersection(
        circle=Circle(centre=Point(x=1.0 * meter, y=2.0 * meter), radius=5.0 * meter),
        line=Line(gradient=-2.0 * UNITLESS, intercept=14.0 * meter),
        point=Point(x=ContinuousUnknown(guess=3.0 * meter), y=ContinuousUnknown(guess=8.0 * meter)),
    )

    with euclidean_circle_line_intersection:
        assert intersection.point.x.convert_to(meter).value == pytest.approx(4.0)
        assert intersection.point.y.convert_to(meter).value == pytest.approx(6.0)
