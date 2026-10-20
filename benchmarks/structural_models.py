"""Scalar fixtures imported by structural-search workers before timing starts."""

# ruff: noqa: INP001 - standalone benchmark companion, not an installed package

__all__ = ["build_model", "expected_value"]

import typing

from bilt import Implementation, Interface, equation, field, system

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@system
class Ports(metaclass=Interface):
    """Common interface for every level and both sibling instances."""

    a: float = field()
    b: float = field()
    x: float = field()
    y: float = field()


@equation
def total(a: float, x: float, y: float) -> bool:
    """Constrain the sum in the coupled core."""
    return x + y == a


@equation
def difference(b: float, x: float, y: float) -> bool:
    """Constrain the difference in the coupled core."""
    return x - y == b


@equation
def explicit_x(a: float, b: float, x: float) -> bool:
    """Compute the acyclic core output."""
    return x == a + b


@equation
def explicit_y(x: float, y: float) -> bool:
    """Retain an additional determined core attribute."""
    return y == x + 1


@equation
def free_branch(x: float, u: float, v: float) -> bool:
    """Attach an underdetermined branch to the relevant scope."""
    return u + v == x


@equation
def a_link(a: float, inner: Ports) -> bool:
    """Supply the child through an identity boundary equation."""
    return inner.a == a


@equation
def b_link(b: float, inner: Ports) -> bool:
    """Supply the child's second boundary input."""
    return inner.b == b


@equation
def x_link(x: float, inner: Ports) -> bool:
    """Expose the child output unchanged."""
    return x == inner.x


@equation
def arithmetic_a(a: float, b: float, inner: Ports) -> bool:
    """Compute the child input in the parent."""
    return inner.a == 2 * a + b


@equation
def arithmetic_b(b: float, inner: Ports) -> bool:
    """Offset the child's second input in the parent."""
    return inner.b == b + 1


@equation
def arithmetic_x(x: float, inner: Ports) -> bool:
    """Combine a child result with a parent-produced boundary value."""
    return x == inner.x + inner.b


@equation
def left_a(a: float, b: float, left: Ports) -> bool:
    """Compute the left sibling's first input."""
    return left.a == a + b


@equation
def left_b(b: float, left: Ports) -> bool:
    """Supply the left sibling's second input."""
    return left.b == b


@equation
def right_a(a: float, left: Ports, right: Ports) -> bool:
    """Make the right sibling depend on the left sibling's result."""
    return right.a == left.x + a


@equation
def right_b(b: float, right: Ports) -> bool:
    """Offset the right sibling's second input."""
    return right.b == b + 1


@equation
def sibling_x(x: float, left: Ports, right: Ports) -> bool:
    """Join both sibling results in the parent."""
    return x == left.x + 2 * right.x


LAYOUTS = {
    "links": ((a_link, b_link, x_link), ("inner",)),
    "arithmetic": ((arithmetic_a, arithmetic_b, arithmetic_x), ("inner",)),
    "siblings": ((left_a, left_b, right_a, right_b, sibling_x), ("left", "right")),
}


def build_model(*, coupled: bool, depth: int, rectangular: bool = False, layout: str = "links") -> Implementation:
    """Build a nested chain or binary tree with original primitive row identities."""
    core: list[BiltModelConstituent] = [total, difference] if coupled else [explicit_x, explicit_y]
    if rectangular:
        core.append(free_branch)
    implementation = Implementation(Ports, core)
    operations, children = LAYOUTS[layout]
    for _ in range(depth):
        # Siblings deliberately share one implementation object. Expansion must
        # still preserve every instance path and its separate constraints.
        implementation = Implementation(
            Ports,
            operations,
            internal_implementations=dict.fromkeys(children, implementation),
        )
    return implementation


def expected_value(*, coupled: bool, depth: int, layout: str, a: float = 4.0, b: float = 2.0) -> float:
    """Calculate the analytic output independently of either planning solver."""
    if depth == 0:
        return (a + b) / 2 if coupled else a + b

    def recur(aa: float, bb: float) -> float:
        return expected_value(coupled=coupled, depth=depth - 1, layout=layout, a=aa, b=bb)

    if layout == "links":
        return recur(a, b)
    if layout == "arithmetic":
        return recur(2 * a + b, b + 1) + b + 1
    left = recur(a + b, b)
    right = recur(left + a, b + 1)
    return left + 2 * right
